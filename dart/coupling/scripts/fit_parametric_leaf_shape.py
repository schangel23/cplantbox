"""Offline fitter — MF3D scans → maize parametric-leaf-shape distribution.

S0 of `PLAN_PARAMETRIC_LEAF_SHAPE_2026-05-09_REV1.md`.

Pipeline (see plan §S0):

    MF3D world-frame CPs
        ─→ canonical_library.to_local_frame(tip_canonical_rotate=False)
        ─→ symmetric extraction: midrib (droop, along), normalised half-width
        ─→ degree-4 B-spline interpolation on the N_U=11 u-stations (n_cp = 11,
           exact interpolation at machine epsilon)

Frame choice (NPZ-compat, ``tip_canonical_rotate=False``)
---------------------------------------------------------

The maize ``surface_cp`` grids in ``maize_calibrated.xml`` were baked
in the **NPZ-compat** frame (the ``canonical_leaf_library.npz`` build
that the existing lofter consumes). Verified empirically: donor mean
midline droop at u=1 with ``tcr=False`` matches XML L0 droop to ≈1.2×
(real MF3D-vs-Vidal size drift), whereas ``tcr=True`` would give ≈2×
because tip-canonicalisation rotates lateral drift into the droop axis.

Concretely, with ``tcr=False``:
- midline ``+y_local`` (droop) is the *signed* droop component along
  the leaf's natural azimuth — averages toward zero under random
  azimuthal sign across donors only via gate (c)'s cancellation test;
- midline ``+x_local`` is the per-leaf lateral drift (per-leaf large,
  population-mean ≈ 0 by random-sign cancellation) — discarded under D9.

The XML side runs the same extractor + spline on each `<leaf>`'s
`surface_cp` median grid, producing one **per-rank intercept** (15 in
total, ranks L0..L14).

Symmetric vs asymmetric content
-------------------------------

D9 of the plan keeps only **symmetric**, frame-invariant components in
the parametric distribution:

    midrib(u)   = (0, m_y(u), m_z(u))
    halfwidth(u) ∈ [0, 1]     (lateral spread, normalised to peak = 1)

The XML grids carry real per-side asymmetry, midline lateral drift, and
off-midline OOP curl that the symmetric model cannot represent. Per the
2026-05-09 user decision (`asym_residual` option), each per-rank intercept
also carries a frozen `(N_U, N_V, 3)` **asymmetric residual grid**:

    asym_residual[r] = XML_grid[r] - sym_reconstruction(intercept[r])

At runtime, `evaluate(intercept[r], scale=0)` produces XML rank `r`'s
grid bit-for-bit (the symmetric splines exactly interpolate the symmetric
components at the 11 u-stations, and the asymmetric residual is added
on top frozen). `evaluate(intercept[r] + scale * L @ z)` perturbs only
the symmetric splines; the asymmetric residual stays frozen. This is
what allows S6's gate G8 (byte-identical mean draw) to pass without
violating D9's symmetric-only deviation contract.

Deviation pool: per-donor symmetric coefficients (33-dim) minus the
XML intercept of the corresponding rank → pooled covariance for the
shared (across-rank) plant-to-plant variation.

Outputs
-------
- `dart/coupling/data/maize_leaf_shape_distribution.json` (runtime input)
- `dart/coupling/data/maize_leaf_shape_fit_quality.json`  (calibration log)
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.interpolate import make_interp_spline

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dart.coupling.geometry.canonical_cp_grid import N_U, N_V  # noqa: E402
from dart.coupling.geometry.canonical_library import to_local_frame  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SPLINE_DEGREE = 4   # plan §S0 step 3
N_CP = N_U          # exact interpolation at u-stations (= 11)
N_BASIS_TOTAL = 3 * N_CP  # droop + along + width = 33

U_VALS = np.linspace(0.0, 1.0, N_U)
V_VALS = np.linspace(0.0, 1.0, N_V)

DEFAULT_MF3D_JSON = (
    "/home/lukas/PHD/Resources/MaizeField3d/maizefield3d_canonical_cps.json"
)
DEFAULT_XML = REPO_ROOT / "dart/coupling/data/maize_calibrated.xml"
DEFAULT_DIST_OUT = REPO_ROOT / "dart/coupling/data/maize_leaf_shape_distribution.json"
DEFAULT_QUALITY_OUT = (
    REPO_ROOT / "dart/coupling/data/maize_leaf_shape_fit_quality.json"
)

N_RANKS = 15        # XML L0..L14
N_POSITIONS = 14    # MF3D positions 0..13 (D12 mapping)


# ---------------------------------------------------------------------------
# Symmetric extraction (D9 §1)
# ---------------------------------------------------------------------------
@dataclass
class SymmetricComponents:
    droop: np.ndarray       # (N_U,)  midline +y_local (cm)
    along: np.ndarray       # (N_U,)  midline +z_local (cm)
    halfwidth_norm: np.ndarray  # (N_U,)  in [0, 1] — w(u)/max_w
    max_w: float            # cm — peak half-width across u (per-leaf or per-rank)
    lmax_self: float        # cm — midrib polyline arc length (collar→tip in (y,z))


def midrib_arc_length(droop: np.ndarray, along: np.ndarray) -> float:
    """Polyline arc length of the midrib in the (y_local, z_local) plane.

    The midrib's natural length — equal to ``along[-1]`` for a straight leaf,
    and approximately the leaf's true length for a drooping one (always
    >= euclidean distance from collar to tip). Used as the per-leaf size
    scale for normalising droop + along into dimensionless shape coefficients
    (PLAN_PARAMETRIC_LEAF_SHAPE_2026-05-09_REV1 fix 2b — "Refit dimensionless").
    """
    dy = np.diff(droop)
    dz = np.diff(along)
    return float(np.sum(np.sqrt(dy ** 2 + dz ** 2)))


def extract_symmetric(cps_local: np.ndarray) -> SymmetricComponents:
    """Project a leaf-local CP grid onto the symmetric parametric model.

    Args:
        cps_local: ``(N_U, N_V, 3)`` control points already in the
            canonical-library leaf-local frame (collar at origin,
            ``+x_local`` lateral, ``+y_local`` OOP/droop, ``+z_local``
            along-midrib).

    Returns:
        :class:`SymmetricComponents` carrying the midline (droop, along)
        components, the normalised half-width function, and the per-leaf
        ``max_w`` scale used for the normalisation. Discards midline
        ``+x_local`` drift, per-side asymmetry and off-midline ``+y_local``
        curl per D9 §1.
    """
    if cps_local.shape != (N_U, N_V, 3):
        raise ValueError(
            f"expected ({N_U}, {N_V}, 3) local-frame CP grid, got {cps_local.shape}"
        )
    mid = N_V // 2
    droop = cps_local[:, mid, 1].copy()
    along = cps_local[:, mid, 2].copy()
    half_w_raw = (cps_local[:, :, 0].max(axis=1) - cps_local[:, :, 0].min(axis=1)) / 2.0
    max_w = float(half_w_raw.max())
    if max_w < 1e-9:
        raise ValueError("degenerate leaf: max half-width is zero")
    halfwidth_norm = half_w_raw / max_w
    lmax_self = midrib_arc_length(droop, along)
    if lmax_self < 1e-9:
        raise ValueError("degenerate leaf: midrib arc length is zero")
    return SymmetricComponents(
        droop=droop, along=along, halfwidth_norm=halfwidth_norm, max_w=max_w,
        lmax_self=lmax_self,
    )


def fit_intercept(sym: SymmetricComponents) -> np.ndarray:
    """Fit degree-4 splines on dimensionless (droop/lmax, along/lmax, halfwidth/max_w).

    Returns a (3 * N_CP,) flat coefficient vector ``[droop_norm | along_norm |
    width_norm]``, all dimensionless. Per-leaf size enters at evaluation time
    via the lmax / max_w multipliers (fix 2b: shape decoupled from size).

    With ``n_cp = N_U`` and ``make_interp_spline(k=4)``, the fit is exact
    interpolation at machine epsilon. Multiplying the reconstructed splines
    by ``sym.lmax_self`` and ``sym.max_w`` reproduces the input grid bit-for-bit.
    """
    droop_norm = sym.droop / sym.lmax_self
    along_norm = sym.along / sym.lmax_self
    droop_sp = make_interp_spline(U_VALS, droop_norm, k=SPLINE_DEGREE)
    along_sp = make_interp_spline(U_VALS, along_norm, k=SPLINE_DEGREE)
    width_sp = make_interp_spline(U_VALS, sym.halfwidth_norm, k=SPLINE_DEGREE)
    coeffs = np.concatenate([droop_sp.c, along_sp.c, width_sp.c])
    if coeffs.shape != (N_BASIS_TOTAL,):
        raise RuntimeError(f"unexpected coeff shape {coeffs.shape}")
    return coeffs


def evaluate_symmetric(
    coeffs: np.ndarray,
    max_w: float,
    lmax: float,
    u_grid: np.ndarray | None = None,
    v_grid: np.ndarray | None = None,
) -> np.ndarray:
    """Reconstruct the symmetric-flat surface from a coefficient vector.

    Args:
        coeffs: ``(3 * N_CP,)`` flat coefficient vector from
            :func:`fit_intercept` or a sampled ``intercept + L @ z``.
            All three blocks are dimensionless (fix 2b).
        max_w: per-rank peak half-width in cm — multiplies the
            normalised half-width back to physical lateral units.
        lmax: per-rank midrib arc length in cm — multiplies the
            normalised droop + along splines back to physical units.
        u_grid: optional ``(M_u,)`` u-stations (default U_VALS).
        v_grid: optional ``(M_v,)`` v-stations (default V_VALS).

    Returns:
        ``(M_u, M_v, 3)`` array in absolute cm. Lateral component is
        identically zero on the midline (``v = 0.5``); the model is
        symmetric in v.
    """
    if coeffs.shape != (N_BASIS_TOTAL,):
        raise ValueError(f"expected ({N_BASIS_TOTAL},) coeffs, got {coeffs.shape}")
    droop_c = coeffs[0:N_CP]
    along_c = coeffs[N_CP:2 * N_CP]
    width_c = coeffs[2 * N_CP:3 * N_CP]
    knots = make_interp_spline(U_VALS, np.zeros(N_U), k=SPLINE_DEGREE).t
    droop_sp = type(make_interp_spline(U_VALS, np.zeros(N_U), k=SPLINE_DEGREE))(
        knots, droop_c, SPLINE_DEGREE,
    )
    along_sp = type(droop_sp)(knots, along_c, SPLINE_DEGREE)
    width_sp = type(droop_sp)(knots, width_c, SPLINE_DEGREE)

    u_arr = U_VALS if u_grid is None else np.asarray(u_grid, dtype=np.float64)
    v_arr = V_VALS if v_grid is None else np.asarray(v_grid, dtype=np.float64)
    out = np.zeros((u_arr.size, v_arr.size, 3), dtype=np.float64)
    for iu, u in enumerate(u_arr):
        m_y = float(droop_sp(u)) * lmax
        m_z = float(along_sp(u)) * lmax
        w = float(width_sp(u))
        for iv, v in enumerate(v_arr):
            out[iu, iv, 0] = (v - 0.5) * w * max_w
            out[iu, iv, 1] = m_y
            out[iu, iv, 2] = m_z
    return out


def compute_asym_residual(xml_grid: np.ndarray, intercept: np.ndarray,
                          max_w: float, lmax: float) -> np.ndarray:
    """Per-rank frozen asymmetric residual grid (in absolute cm).

    ``asym_residual[r] = XML_grid[r] - evaluate_symmetric(intercept[r], max_w, lmax)``

    At runtime, ``evaluate(intercept[r], scale=0) = sym_recon + asym_residual[r]``
    reproduces ``XML_grid[r]`` bit-for-bit when called with the same
    ``max_w`` and ``lmax`` baked at fit time. Sampling deviations only
    perturbs ``sym_recon``; the residual stays frozen.
    """
    sym_recon = evaluate_symmetric(intercept, max_w, lmax)
    return xml_grid - sym_recon


# ---------------------------------------------------------------------------
# Discarded-content (D9 §4) probe
# ---------------------------------------------------------------------------
@dataclass
class DiscardedContentMetrics:
    """Pose-coupled / asymmetric content rejected by D9 §1.

    Per-u arrays are recorded in two forms:

    * ``*_signed_per_u`` — the raw signed value at each u-station. Across
      donors with random tip-azimuth, the signed midline drift and per-side
      asymmetry should cancel (population mean ≈ 0). Used in gate (c).
    * ``*_abs_per_u`` — magnitude. Used in per-donor outlier flagging
      (3σ-of-population threshold) and recorded in the per-donor log.

    Off-midline OOP curl RMS is only ever non-negative; only the
    magnitude form is meaningful here.
    """

    midline_lateral_drift_signed_per_u: list[float]   # C[u, mid, 0]
    midline_lateral_drift_abs_per_u: list[float]      # |C[u, mid, 0]|
    per_side_asymmetry_signed_per_u: list[float]      # (max_v + min_v)/2
    per_side_asymmetry_abs_per_u: list[float]         # |...|
    off_midline_oop_rms_per_u: list[float]            # RMS_v(C[u, v, 1] - mid_y[u])

    def as_dict(self) -> dict:
        return {
            "midline_lateral_drift_signed_per_u": self.midline_lateral_drift_signed_per_u,
            "midline_lateral_drift_abs_per_u": self.midline_lateral_drift_abs_per_u,
            "per_side_asymmetry_signed_per_u": self.per_side_asymmetry_signed_per_u,
            "per_side_asymmetry_abs_per_u": self.per_side_asymmetry_abs_per_u,
            "off_midline_oop_rms_per_u": self.off_midline_oop_rms_per_u,
        }


def discarded_content_metrics(cps_local: np.ndarray) -> DiscardedContentMetrics:
    mid = N_V // 2
    midline_x_signed = cps_local[:, mid, 0]
    asym_x_signed = (cps_local[:, :, 0].max(axis=1) + cps_local[:, :, 0].min(axis=1)) / 2.0
    off_y = cps_local[:, :, 1] - cps_local[:, mid:mid + 1, 1]
    off_y_rms = np.sqrt((off_y ** 2).mean(axis=1))
    return DiscardedContentMetrics(
        midline_lateral_drift_signed_per_u=[float(v) for v in midline_x_signed],
        midline_lateral_drift_abs_per_u=[float(v) for v in np.abs(midline_x_signed)],
        per_side_asymmetry_signed_per_u=[float(v) for v in asym_x_signed],
        per_side_asymmetry_abs_per_u=[float(v) for v in np.abs(asym_x_signed)],
        off_midline_oop_rms_per_u=[float(v) for v in off_y_rms],
    )


# ---------------------------------------------------------------------------
# XML loader
# ---------------------------------------------------------------------------
def load_xml_leaf_grids(xml_path: Path) -> tuple[list[np.ndarray], list[float], list[str]]:
    """Read all 15 maize leaf RPs and return their surface_cp grids.

    Returns
    -------
    grids : list of (N_U, N_V, 3) arrays, indexed by rank 0..14
    lmax_xml : list of float (rank's lmax in cm, for residual normalisation)
    names : list of str ("maize_leaf_L0", ...)
    """
    import xml.etree.ElementTree as ET

    tree = ET.parse(xml_path)
    root = tree.getroot()
    leaves = root.findall(".//leaf")
    if len(leaves) != N_RANKS:
        raise ValueError(
            f"expected {N_RANKS} <leaf> elements in {xml_path}, found {len(leaves)}"
        )
    leaves_by_subtype: dict[int, ET.Element] = {
        int(lf.attrib["subType"]): lf for lf in leaves
    }
    grids: list[np.ndarray] = []
    lmax: list[float] = []
    names: list[str] = []
    for r in range(N_RANKS):
        subtype = r + 2  # subType 2 = L0, ..., subType 16 = L14
        lf = leaves_by_subtype.get(subtype)
        if lf is None:
            raise ValueError(f"missing <leaf subType=\"{subtype}\"> for rank {r}")
        cps = np.full((N_U, N_V, 3), np.nan, dtype=np.float64)
        for cp in lf.findall("parameter[@name='surface_cp']"):
            u = float(cp.attrib["u"])
            v = float(cp.attrib["v"])
            x = float(cp.attrib["x"])
            y = float(cp.attrib["y"])
            z = float(cp.attrib["z"])
            i = int(round(u * (N_U - 1)))
            j = int(round(v * (N_V - 1)))
            cps[i, j] = (x, y, z)
        if np.isnan(cps).any():
            raise ValueError(f"L{r} (subType={subtype}) has missing surface_cp entries")
        grids.append(cps)
        lmax_param = lf.find("parameter[@name='lmax']")
        lmax.append(float(lmax_param.attrib["value"]) if lmax_param is not None else float("nan"))
        names.append(lf.attrib.get("name", f"L{r}"))
    return grids, lmax, names


# ---------------------------------------------------------------------------
# MF3D loader
# ---------------------------------------------------------------------------
def load_mf3d_leaves(mf3d_json: Path) -> dict:
    with open(mf3d_json) as f:
        data = json.load(f)
    if data.get("n_u") != N_U or data.get("n_v") != N_V:
        raise ValueError(
            f"MF3D JSON has n_u={data.get('n_u')}, n_v={data.get('n_v')}; expected {N_U}, {N_V}"
        )
    return data


# ---------------------------------------------------------------------------
# Per-donor processing
# ---------------------------------------------------------------------------
@dataclass
class DonorRecord:
    plant_id: str
    position: int
    coeffs: np.ndarray      # (N_BASIS_TOTAL,)
    max_w: float            # cm
    discarded: DiscardedContentMetrics
    sym_fit_rms: float       # spline-fit RMS at u-stations on (droop, along, halfwidth*max_w); ≈ 0 under exact interpolation
    sym_lossy_rms: float     # full reconstruction residual including asymmetric content that the model deliberately discards (informational)


def process_donor_leaf(
    plant_id: str,
    position: int,
    cps_world: np.ndarray,
) -> DonorRecord | None:
    """Project + extract + fit one MF3D donor leaf. Returns None on failure."""
    if cps_world.shape != (N_U, N_V, 3):
        return None
    if not np.isfinite(cps_world).all():
        return None
    try:
        cps_local, _, _ = to_local_frame(
            cps_world, normalize_arc=False, tip_canonical_rotate=False,
        )
    except ValueError:
        return None
    try:
        sym = extract_symmetric(cps_local)
    except ValueError:
        return None
    coeffs = fit_intercept(sym)
    discarded = discarded_content_metrics(cps_local)

    # Gate (d) — symmetric spline-fit RMS at u-stations only (compare
    # extracted donor sym components against the spline re-evaluation,
    # in physical cm). With make_interp_spline(k=4) and n_cp = N_U, this is
    # exact interpolation at machine epsilon. Coefs are dimensionless
    # post-fix-2b; multiply droop/along splines by lmax_self to compare
    # against extracted (cm) values.
    droop_c = coeffs[0:N_CP]
    along_c = coeffs[N_CP:2 * N_CP]
    width_c = coeffs[2 * N_CP:3 * N_CP]
    knots = make_interp_spline(U_VALS, np.zeros(N_U), k=SPLINE_DEGREE).t
    SP = type(make_interp_spline(U_VALS, np.zeros(N_U), k=SPLINE_DEGREE))
    droop_re = SP(knots, droop_c, SPLINE_DEGREE)(U_VALS) * sym.lmax_self
    along_re = SP(knots, along_c, SPLINE_DEGREE)(U_VALS) * sym.lmax_self
    width_re = SP(knots, width_c, SPLINE_DEGREE)(U_VALS)
    fit_residual = np.concatenate([
        droop_re - sym.droop,
        along_re - sym.along,
        (width_re - sym.halfwidth_norm) * sym.max_w,  # bring width to cm units
    ])
    sym_fit_rms = float(np.sqrt(np.mean(fit_residual ** 2)))

    # Informational: full grid lossy reconstruction (carries discarded
    # asymmetric content; not the gate-(d) metric).
    sym_recon = evaluate_symmetric(coeffs, sym.max_w, sym.lmax_self)
    sym_lossy_rms = float(np.sqrt(np.mean((cps_local - sym_recon) ** 2)))

    return DonorRecord(
        plant_id=plant_id,
        position=position,
        coeffs=coeffs,
        max_w=sym.max_w,
        discarded=discarded,
        sym_fit_rms=sym_fit_rms,
        sym_lossy_rms=sym_lossy_rms,
    )


# ---------------------------------------------------------------------------
# Acceptance gates (S0 §S0 acceptance gate)
# ---------------------------------------------------------------------------
def gate_a_per_rank_anchor(
    intercepts: list[np.ndarray],
    asym_residuals: list[np.ndarray],
    max_w_xml: list[float],
    lmax_intercept: list[float],
    xml_grids: list[np.ndarray],
) -> tuple[bool, list[dict]]:
    """For each rank r, evaluate(intercept[r], scale=0) + asym_residual[r] reproduces
    XML rank r's surface_cps grid to <= 1e-9 cm element-wise.
    """
    results = []
    all_pass = True
    for r in range(N_RANKS):
        sym_recon = evaluate_symmetric(intercepts[r], max_w_xml[r], lmax_intercept[r])
        full_recon = sym_recon + asym_residuals[r]
        diff = full_recon - xml_grids[r]
        max_abs = float(np.max(np.abs(diff)))
        passed = max_abs <= 1e-9
        all_pass = all_pass and passed
        results.append({"rank": r, "max_abs_cm": max_abs, "pass": passed})
    return all_pass, results


def gate_b_pose_invariance(
    intercepts: list[np.ndarray],
    max_w_xml: list[float],
    lmax_intercept: list[float],
    rng_seed: int = 1234,
    n_synthetic: int = 1000,
    cholesky_factor: np.ndarray | None = None,
) -> tuple[bool, dict]:
    """Midline lateral component (`evaluate(...)[u, mid, 0]`) is exactly 0
    for all 15 intercepts AND for n_synthetic random samples.

    Note: gate (b) only inspects the *symmetric* reconstruction. The
    asym_residual carries midline lateral drift from XML (~0.5-1 cm); the
    parametric model itself has no midline-x axis, so deviations cannot
    reintroduce drift, but the frozen residual does.
    """
    mid = N_V // 2
    max_abs_intercepts = 0.0
    for r in range(N_RANKS):
        sym = evaluate_symmetric(intercepts[r], max_w_xml[r], lmax_intercept[r])
        max_abs_intercepts = max(max_abs_intercepts, float(np.max(np.abs(sym[:, mid, 0]))))
    rng = np.random.default_rng(rng_seed)
    max_abs_synth = 0.0
    if cholesky_factor is not None:
        for _ in range(n_synthetic):
            z = rng.standard_normal(N_BASIS_TOTAL)
            sample = intercepts[0] + cholesky_factor @ z
            sym = evaluate_symmetric(sample, max_w_xml[0], lmax_intercept[0])
            max_abs_synth = max(max_abs_synth, float(np.max(np.abs(sym[:, mid, 0]))))
    pass_intercepts = max_abs_intercepts <= 1e-12
    pass_synth = max_abs_synth <= 1e-12
    return (pass_intercepts and pass_synth), {
        "max_abs_midline_lateral_intercepts_cm": max_abs_intercepts,
        "max_abs_midline_lateral_synthetic_cm": max_abs_synth,
        "n_synthetic_samples": n_synthetic if cholesky_factor is not None else 0,
        "pass_intercepts": pass_intercepts,
        "pass_synthetic": pass_synth,
    }


def gate_c_discarded_content_sanity(
    donor_records: list[DonorRecord],
    max_w_pop_mean: float,
) -> tuple[bool, dict]:
    """Discarded pose-coupled content cancels under random-sign averaging.

    The plan §D9 §4 expects population-mean midline drift and per-side
    asymmetry to be ≈ 0 across donors. Magnitudes (``|...|``) cannot
    average to 0; the SIGNED population means (over donors, averaged
    over u-stations) should — because tip-canonical rotation distributes
    azimuthal sign randomly across the population.

    Threshold: ``|signed_mean| <= 0.10 * max_w_population_mean`` (per-leaf
    drift can be ~``max_w``, so the population mean shrinks toward 0 with
    sample size; on the full dataset this is well-satisfied).

    The per-donor magnitude statistics are reported alongside (mean of
    abs values) so a non-zero magnitude is visible without conflating
    cancellation failure with per-leaf signal presence.
    """
    n = len(donor_records)
    midline_signed_donor = np.array(
        [np.mean(rec.discarded.midline_lateral_drift_signed_per_u) for rec in donor_records]
    ) if n else np.zeros(0)
    midline_abs_donor = np.array(
        [np.mean(rec.discarded.midline_lateral_drift_abs_per_u) for rec in donor_records]
    ) if n else np.zeros(0)
    asym_signed_donor = np.array(
        [np.mean(rec.discarded.per_side_asymmetry_signed_per_u) for rec in donor_records]
    ) if n else np.zeros(0)
    asym_abs_donor = np.array(
        [np.mean(rec.discarded.per_side_asymmetry_abs_per_u) for rec in donor_records]
    ) if n else np.zeros(0)
    off_y_rms_donor = np.array(
        [np.mean(rec.discarded.off_midline_oop_rms_per_u) for rec in donor_records]
    ) if n else np.zeros(0)

    mid_signed_pop = float(midline_signed_donor.mean()) if n else 0.0
    asym_signed_pop = float(asym_signed_donor.mean()) if n else 0.0
    threshold = 0.10 * max_w_pop_mean
    pass_mid = abs(mid_signed_pop) <= threshold
    pass_asym = abs(asym_signed_pop) <= threshold
    return (pass_mid and pass_asym), {
        "max_w_population_mean_cm": max_w_pop_mean,
        "threshold_cm_(0.10*max_w)": threshold,
        "midline_lateral_drift_signed_population_mean_cm": mid_signed_pop,
        "midline_lateral_drift_abs_population_mean_cm": float(midline_abs_donor.mean()) if n else 0.0,
        "per_side_asymmetry_signed_population_mean_cm": asym_signed_pop,
        "per_side_asymmetry_abs_population_mean_cm": float(asym_abs_donor.mean()) if n else 0.0,
        "off_midline_oop_rms_population_mean_cm": float(off_y_rms_donor.mean()) if n else 0.0,
        "pass_midline_drift_cancellation": pass_mid,
        "pass_asymmetry_cancellation": pass_asym,
    }


def gate_d_per_donor_reconstruction(
    donor_records: list[DonorRecord],
    lmax_pop_mean_proxy_cm: float,
) -> tuple[bool, dict]:
    """Median per-donor *symmetric spline-fit* RMS <= 5 % of lmax.

    Per the plan: "are the splines flexible enough to fit the symmetric
    component well". With ``n_cp = N_U`` and ``make_interp_spline(k=4)``
    this is exact interpolation at machine epsilon — gate passes with
    massive headroom. The lossy-reconstruction RMS (which includes the
    deliberately-discarded asymmetric content) is reported separately
    as an informational diagnostic.
    """
    fit_rms = np.array([rec.sym_fit_rms for rec in donor_records])
    lossy_rms = np.array([rec.sym_lossy_rms for rec in donor_records])
    median_fit = float(np.median(fit_rms))
    median_lossy = float(np.median(lossy_rms))
    threshold = 0.05 * lmax_pop_mean_proxy_cm
    n_outliers = int((fit_rms > 0.10 * lmax_pop_mean_proxy_cm).sum())
    return (median_fit <= threshold), {
        "median_sym_fit_rms_cm": median_fit,
        "median_sym_lossy_rms_cm_informational": median_lossy,
        "threshold_cm_(0.05*lmax_proxy)": threshold,
        "lmax_proxy_cm": lmax_pop_mean_proxy_cm,
        "n_donors_fit_above_10pct": n_outliers,
        "n_donors_total": len(donor_records),
    }


def _build_per_donor_records(records: list["DonorRecord"]) -> list[dict]:
    """Compact per-donor log: scalars for all donors; per-u arrays only for
    3σ outliers on any discarded-content axis. Trims ~11 MB → ~1 MB.
    """
    if not records:
        return []
    midline_means = np.array(
        [np.mean(r.discarded.midline_lateral_drift_abs_per_u) for r in records]
    )
    asym_means = np.array(
        [np.mean(r.discarded.per_side_asymmetry_abs_per_u) for r in records]
    )
    off_y_means = np.array(
        [np.mean(r.discarded.off_midline_oop_rms_per_u) for r in records]
    )
    sigmas = (midline_means.std(), asym_means.std(), off_y_means.std())
    means = (midline_means.mean(), asym_means.mean(), off_y_means.mean())
    out = []
    for i, rec in enumerate(records):
        is_outlier = (
            abs(midline_means[i] - means[0]) > 3 * sigmas[0]
            or abs(asym_means[i] - means[1]) > 3 * sigmas[1]
            or abs(off_y_means[i] - means[2]) > 3 * sigmas[2]
        )
        record = {
            "plant_id": rec.plant_id,
            "position": rec.position,
            "max_w_cm": rec.max_w,
            "sym_fit_rms_cm": rec.sym_fit_rms,
            "sym_lossy_rms_cm": rec.sym_lossy_rms,
            "midline_drift_abs_mean_cm": float(midline_means[i]),
            "asymmetry_abs_mean_cm": float(asym_means[i]),
            "off_midline_oop_rms_mean_cm": float(off_y_means[i]),
            "is_3sigma_outlier": bool(is_outlier),
        }
        if is_outlier:
            record["discarded_per_u_cm"] = rec.discarded.as_dict()
        out.append(record)
    return out


def gate_e_mf3d_xml_consistency(
    intercepts: list[np.ndarray],
    donor_records: list[DonorRecord],
    tol: float = 0.05,
) -> tuple[bool, list[dict]]:
    """For each rank r, ||mean_donor[r] - intercept[r]|| / ||intercept[r]|| <= 5%."""
    results = []
    all_pass = True
    for r in range(N_POSITIONS):  # 0..13 only
        donors_at_r = [rec for rec in donor_records if rec.position == r]
        if not donors_at_r:
            results.append({"rank": r, "n_donors": 0, "pass": True, "skipped": True})
            continue
        mean_donor = np.mean([rec.coeffs for rec in donors_at_r], axis=0)
        l2_diff = float(np.linalg.norm(mean_donor - intercepts[r]))
        l2_intercept = float(np.linalg.norm(intercepts[r]))
        ratio = l2_diff / max(l2_intercept, 1e-12)
        passed = ratio <= tol
        all_pass = all_pass and passed
        results.append({
            "rank": r,
            "n_donors": len(donors_at_r),
            "l2_donor_minus_intercept": l2_diff,
            "l2_intercept": l2_intercept,
            "ratio": ratio,
            "tol": tol,
            "pass": passed,
        })
    return all_pass, results


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mf3d", type=Path, default=Path(DEFAULT_MF3D_JSON))
    parser.add_argument("--xml", type=Path, default=Path(DEFAULT_XML))
    parser.add_argument("--out-dist", type=Path, default=DEFAULT_DIST_OUT)
    parser.add_argument("--out-quality", type=Path, default=DEFAULT_QUALITY_OUT)
    parser.add_argument("--max-donors", type=int, default=0,
                        help="cap donor count for fast smoke (0 = all)")
    parser.add_argument("--cov-eps-rel", type=float, default=1e-6,
                        help="relative regularisation: eps = cov_eps_rel * trace(Σ)/dim")
    parser.add_argument("--pca-K", type=int, default=8,
                        help="PCA truncation: keep top K eigenmodes of Σ. "
                             "0 = no truncation (use full Cholesky); "
                             "K > 0 stores `pca_components` and `pca_eigenvalues` "
                             "in the JSON for the C++ loader (fix path α). "
                             "Plan recommends K ≈ 5-10 (knocks out the noise "
                             "modes that produce non-monotonic along, oscillating "
                             "droop, negative halfwidth at scale ≥ 0.3).")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    def _silent(*args_, **kwargs_):  # pyright: ignore[reportUnusedVariable]
        del args_, kwargs_
    log = _silent if args.quiet else print

    # ── 1. XML side: per-rank intercepts + asym residuals ───────────────────
    log(f"Reading XML: {args.xml}")
    xml_grids, lmax_xml_param, leaf_names = load_xml_leaf_grids(args.xml)
    intercepts: list[np.ndarray] = []
    asym_residuals: list[np.ndarray] = []
    max_w_xml: list[float] = []
    lmax_intercept: list[float] = []  # arc-length midrib lmax per rank (fix 2b)
    for r, grid in enumerate(xml_grids):
        sym = extract_symmetric(grid)
        intercept = fit_intercept(sym)
        asym = compute_asym_residual(grid, intercept, sym.max_w, sym.lmax_self)
        intercepts.append(intercept)
        asym_residuals.append(asym)
        max_w_xml.append(sym.max_w)
        lmax_intercept.append(sym.lmax_self)
        log(f"  rank {r:2d} ({leaf_names[r]:>16}): max_w_xml={sym.max_w:6.3f} cm,"
            f" lmax_arc={sym.lmax_self:6.2f} cm,"
            f" lmax_param={lmax_xml_param[r]:6.2f} cm,"
            f" |asym|_max={float(np.max(np.abs(asym))):.4f} cm")

    # ── 2. MF3D donors: project + extract + fit ─────────────────────────────
    log(f"Reading MF3D: {args.mf3d}")
    mf3d = load_mf3d_leaves(args.mf3d)
    plants = mf3d["plants"]
    donor_records: list[DonorRecord] = []
    n_seen = 0
    n_skipped = 0
    n_per_pos = [0] * N_POSITIONS
    for plant in plants:
        plant_id = str(plant.get("plant_id", "?"))
        for leaf in plant.get("leaves", []):
            n_seen += 1
            if args.max_donors and len(donor_records) >= args.max_donors:
                break
            position = int(leaf.get("position", -1))
            if not (0 <= position < N_POSITIONS):
                n_skipped += 1
                continue
            cps_world = np.asarray(leaf.get("cps_cm"), dtype=np.float64)
            rec = process_donor_leaf(plant_id, position, cps_world)
            if rec is None:
                n_skipped += 1
                continue
            donor_records.append(rec)
            n_per_pos[position] += 1
        if args.max_donors and len(donor_records) >= args.max_donors:
            break
    log(f"Donors processed: {len(donor_records)} (seen={n_seen}, skipped={n_skipped})")
    log("Per-position donor counts: " + ", ".join(
        f"p{p}:{n}" for p, n in enumerate(n_per_pos)
    ))

    # ── 3. Pooled covariance over (donor, position) deviations ─────────────
    deviations = np.stack([
        rec.coeffs - intercepts[rec.position] for rec in donor_records
    ], axis=0)  # (N_donors, N_BASIS_TOTAL)
    cov = np.cov(deviations, rowvar=False)  # (N_BASIS_TOTAL, N_BASIS_TOTAL)
    if cov.shape != (N_BASIS_TOTAL, N_BASIS_TOTAL):
        raise RuntimeError(f"unexpected cov shape {cov.shape}")
    eps = args.cov_eps_rel * float(np.trace(cov)) / N_BASIS_TOTAL
    cov_reg = cov + eps * np.eye(N_BASIS_TOTAL)
    L = np.linalg.cholesky(cov_reg)
    log(f"Σ shape: {cov.shape}, trace={float(np.trace(cov)):.4e},"
        f" eps={eps:.4e} (rel={args.cov_eps_rel})")

    # ── 3a. PCA truncation (fix path α) ────────────────────────────────────
    # Eigendecompose Σ and keep top K dims. The full Cholesky is retained in
    # the JSON for backwards compat / fallback; the C++ loader prefers the
    # PCA block when present (~30 LOC change in
    # ``LeafShapeDistribution::makeShape``: draw ``z_K ~ N(0, I_K)`` and
    # compute ``coeffs = intercept + scale · U_K · √Λ_K · z_K``).
    #
    # Knocks out the noise modes responsible for root cause #2 of the post-
    # S8 visual review: 33-dim Gaussian without geometric constraints can
    # sample regions producing non-monotonic along, oscillating droop, or
    # negative halfwidth. Top K = 5–10 dominant modes are physically
    # interpretable (overall droop magnitude, lance-vs-ovate, mid-curvature)
    # and match the few shape modes that actually vary across cultivars in
    # FSPM literature.
    eigvals_asc, eigvecs_asc = np.linalg.eigh(cov_reg)
    eigvals_full = eigvals_asc[::-1]            # descending
    eigvecs_full = eigvecs_asc[:, ::-1]         # columns are eigenvectors
    eigvals_full = np.maximum(eigvals_full, 0.0)
    total_variance = float(np.sum(eigvals_full))
    K = max(0, min(args.pca_K, N_BASIS_TOTAL))
    if K > 0:
        eigvecs_K = eigvecs_full[:, :K]         # (N_BASIS_TOTAL, K)
        eigvals_K = eigvals_full[:K]            # (K,)
        retained = float(np.sum(eigvals_K)) / max(total_variance, 1e-30)
        log(f"PCA: K={K}/{N_BASIS_TOTAL}, retained variance="
            f"{retained:.4f} of {total_variance:.4e}")
        # Top-3 scree contribution for quick eyeball
        for k in range(min(K, 5)):
            log(f"  λ_{k+1}={eigvals_K[k]:.4e} "
                f"(cumulative={float(np.sum(eigvals_K[:k+1]))/max(total_variance,1e-30):.4f})")
        # Storage convention: pca_components is K rows × N_BASIS_TOTAL cols
        # (each row = one eigenvector), so the C++ loader can iterate
        # row-major-style ``components[k][i]`` per mode.
        pca_components = eigvecs_K.T.tolist()
        pca_eigenvalues = eigvals_K.tolist()
        pca_block = {
            "K": K,
            "n_components": K,
            "pca_components": pca_components,
            "pca_eigenvalues": pca_eigenvalues,
            "retained_variance_fraction": retained,
            "total_variance": total_variance,
            "all_eigenvalues_descending": eigvals_full.tolist(),
            "regularisation_eps": eps,
        }
    else:
        pca_block = None
        log("PCA: K=0, no truncation (use full Cholesky)")

    # ── 4. Run acceptance gates ─────────────────────────────────────────────
    log("\n── Acceptance gates ──")
    gate_a_pass, gate_a_detail = gate_a_per_rank_anchor(
        intercepts, asym_residuals, max_w_xml, lmax_intercept, xml_grids,
    )
    log(f"  (a) per-rank anchor (FP precision): {'PASS' if gate_a_pass else 'FAIL'}")
    for d in gate_a_detail:
        marker = "✓" if d["pass"] else "✗"
        log(f"      {marker} rank {d['rank']:2d}: max|err|={d['max_abs_cm']:.3e} cm")

    gate_b_pass, gate_b_detail = gate_b_pose_invariance(
        intercepts, max_w_xml, lmax_intercept, cholesky_factor=L,
    )
    log(f"  (b) midline lateral = 0 by construction: {'PASS' if gate_b_pass else 'FAIL'}"
        f" (intercepts max={gate_b_detail['max_abs_midline_lateral_intercepts_cm']:.3e},"
        f" synth max={gate_b_detail['max_abs_midline_lateral_synthetic_cm']:.3e})")

    max_w_pop_mean = float(np.mean([rec.max_w for rec in donor_records]))
    gate_c_pass, gate_c_detail = gate_c_discarded_content_sanity(
        donor_records, max_w_pop_mean,
    )
    log(f"  (c) discarded-content cancellation: {'PASS' if gate_c_pass else 'FAIL'}"
        f" (signed mid_drift={gate_c_detail['midline_lateral_drift_signed_population_mean_cm']:+.3f} cm,"
        f" signed asym={gate_c_detail['per_side_asymmetry_signed_population_mean_cm']:+.3f} cm,"
        f" thresh=±{gate_c_detail['threshold_cm_(0.10*max_w)']:.3f} cm;"
        f" |mid_drift|={gate_c_detail['midline_lateral_drift_abs_population_mean_cm']:.3f} cm,"
        f" |asym|={gate_c_detail['per_side_asymmetry_abs_population_mean_cm']:.3f} cm)")

    lmax_proxy = float(np.mean(lmax_xml_param))
    gate_d_pass, gate_d_detail = gate_d_per_donor_reconstruction(
        donor_records, lmax_proxy,
    )
    log(f"  (d) per-donor symmetric-spline fit: {'PASS' if gate_d_pass else 'FAIL'}"
        f" (median fit_rms={gate_d_detail['median_sym_fit_rms_cm']:.3e} cm,"
        f" thresh={gate_d_detail['threshold_cm_(0.05*lmax_proxy)']:.3f} cm;"
        f" median lossy_rms={gate_d_detail['median_sym_lossy_rms_cm_informational']:.3f} cm informational)")

    # Gate (e) is advisory per plan-doc text "worth knowing before any
    # C++ work" — measures MF3D-vs-XML calibration drift (independent of
    # fitter correctness). DC offset is absorbed by np.cov (mean
    # subtraction), so runtime sampling stays centred on XML intercept.
    # Failure here flags a real-data divergence to triage at S6 bake.
    gate_e_pass, gate_e_detail = gate_e_mf3d_xml_consistency(intercepts, donor_records)
    log(f"  (e) MF3D vs XML intercept consistency [advisory]:"
        f" {'PASS' if gate_e_pass else 'FAIL'}")
    for d in gate_e_detail:
        if d.get("skipped"):
            log(f"      ∅ rank {d['rank']:2d}: no donors (skipped)")
            continue
        marker = "✓" if d["pass"] else "✗"
        log(f"      {marker} rank {d['rank']:2d}: ratio={d['ratio']:.3f}"
            f" (n={d['n_donors']})")

    blocker_pass = gate_a_pass and gate_b_pass and gate_c_pass and gate_d_pass
    all_gates = blocker_pass and gate_e_pass
    log(f"\nBLOCKER GATES (a..d): {'PASS' if blocker_pass else 'FAIL'}")
    log(f"ADVISORY GATE  (e):    {'PASS' if gate_e_pass else 'FAIL'}")

    # ── 5. Emit distribution JSON ───────────────────────────────────────────
    knot_vector = make_interp_spline(U_VALS, np.zeros(N_U), k=SPLINE_DEGREE).t.tolist()
    distribution = {
        "schema_version": "1.0",
        "frame_convention": (
            "canonical_library NPZ-compat local frame:"
            " +x_local lateral (tangent × UP), +y_local OOP/droop,"
            " +z_local along-midrib;"
            " donors projected via to_local_frame(tip_canonical_rotate=False)"
            " to match the XML surface_cp bake convention"
        ),
        "n_components": N_BASIS_TOTAL,
        "n_cp_per_axis": N_CP,
        "spline_degree": SPLINE_DEGREE,
        "spline_knots_u": knot_vector,
        "u_stations": U_VALS.tolist(),
        "v_stations": V_VALS.tolist(),
        "n_u": N_U,
        "n_v": N_V,
        "n_ranks": N_RANKS,
        "intercepts": {str(r): intercepts[r].tolist() for r in range(N_RANKS)},
        "max_w_xml_cm": {str(r): max_w_xml[r] for r in range(N_RANKS)},
        "lmax_intercept_cm": {str(r): lmax_intercept[r] for r in range(N_RANKS)},
        "lmax_xml_cm": {str(r): lmax_xml_param[r] for r in range(N_RANKS)},
        "leaf_names": {str(r): leaf_names[r] for r in range(N_RANKS)},
        "asym_residual_grids_cm": {
            str(r): asym_residuals[r].tolist() for r in range(N_RANKS)
        },
        "covariance": cov.tolist(),
        "covariance_regularisation_eps": eps,
        "cholesky_factor": L.tolist(),
        "donor_count": len(donor_records),
        "donors_per_position": {str(p): n_per_pos[p] for p in range(N_POSITIONS)},
        "position_to_rank_mapping": {str(p): p for p in range(N_POSITIONS)},
        "pca_truncation": pca_block,
        "fit_residual_summary": {
            "median_sym_fit_rms_cm": gate_d_detail["median_sym_fit_rms_cm"],
            "median_sym_lossy_rms_cm_informational": gate_d_detail[
                "median_sym_lossy_rms_cm_informational"
            ],
            "lmax_proxy_cm": lmax_proxy,
            "max_w_population_mean_cm": max_w_pop_mean,
        },
        "coeffs_block_layout": {
            "droop": [0, N_CP],
            "along": [N_CP, 2 * N_CP],
            "halfwidth_norm": [2 * N_CP, 3 * N_CP],
        },
        "gates": {
            "a_per_rank_anchor": {"pass": bool(gate_a_pass), "details": gate_a_detail},
            "b_pose_invariance": {"pass": bool(gate_b_pass), "details": gate_b_detail},
            "c_discarded_content": {"pass": bool(gate_c_pass), "details": gate_c_detail},
            "d_per_donor_reconstruction": {"pass": bool(gate_d_pass), "details": gate_d_detail},
            "e_mf3d_xml_consistency_advisory": {
                "pass": bool(gate_e_pass), "details": gate_e_detail,
                "note": "Advisory: measures real MF3D-vs-XML calibration drift; mean offset is absorbed by np.cov before sampling.",
            },
            "blocker_pass": bool(blocker_pass),
            "all_pass": bool(all_gates),
        },
    }
    args.out_dist.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_dist, "w") as f:
        json.dump(distribution, f, indent=2)
    log(f"\nWrote distribution JSON: {args.out_dist}")

    # ── 6. Emit fit-quality JSON ────────────────────────────────────────────
    quality = {
        "schema_version": "1.0",
        "donor_count": len(donor_records),
        "donors_per_position": {str(p): n_per_pos[p] for p in range(N_POSITIONS)},
        "max_w_population_stats_cm": {
            "mean": max_w_pop_mean,
            "std": float(np.std([rec.max_w for rec in donor_records])),
            "min": float(min((rec.max_w for rec in donor_records), default=0.0)),
            "max": float(max((rec.max_w for rec in donor_records), default=0.0)),
        },
        "discarded_content_population_stats_cm": {
            "midline_lateral_drift_signed_mean": gate_c_detail[
                "midline_lateral_drift_signed_population_mean_cm"
            ],
            "midline_lateral_drift_abs_mean": gate_c_detail[
                "midline_lateral_drift_abs_population_mean_cm"
            ],
            "per_side_asymmetry_signed_mean": gate_c_detail[
                "per_side_asymmetry_signed_population_mean_cm"
            ],
            "per_side_asymmetry_abs_mean": gate_c_detail[
                "per_side_asymmetry_abs_population_mean_cm"
            ],
            "off_midline_oop_rms_mean": gate_c_detail[
                "off_midline_oop_rms_population_mean_cm"
            ],
        },
        "per_donor_records": _build_per_donor_records(donor_records),
        "gates_summary": {
            "a_per_rank_anchor": gate_a_pass,
            "b_pose_invariance": gate_b_pass,
            "c_discarded_content": gate_c_pass,
            "d_per_donor_reconstruction": gate_d_pass,
            "e_mf3d_xml_consistency_advisory": gate_e_pass,
            "blocker_pass": blocker_pass,
            "all_pass": all_gates,
        },
    }
    args.out_quality.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_quality, "w") as f:
        json.dump(quality, f, indent=2)
    log(f"Wrote fit-quality JSON: {args.out_quality}")

    return 0 if blocker_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
