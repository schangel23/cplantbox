"""S3 acceptance gates — `ParametricLeafShape::evaluate` reconstruction.

Plan: Literature/Chapter 1/Concepts/CPBOXBALENOCOUPLING/PLAN_PARAMETRIC_LEAF_SHAPE_2026-05-09_REV1.md.

S3 is the spline + asym-residual reconstruction at the C++ side. These tests
exercise it through the minimal pybind exposure added in the same commit:

1. Analytic reconstruction (gate S3.1) — feed straight-droop / parabolic-width
   spline coefficients with zero asymmetric residual; ``evaluate(u, v)`` must
   reproduce the analytic surface to ≤ 1e-9 cm.

2. Per-rank XML anchor (gate S3.2 — re-test of S0 gate (a) through the C++
   evaluator). For each of the 15 maize ranks, construct
   ``ParametricLeafShape(intercept[r], asym_residual[r])`` and sample on the
   canonical (N_U=11, N_V=5) grid; the result must reproduce the XML's
   ``surface_cp`` median grid to ≤ 1e-9 cm element-wise.

3. Donor round-trip (gate S3.3) — fit a single MF3D donor leaf with the S0
   fitter, reconstruct via C++ ``ParametricLeafShape``; RMS deviation from
   the raw donor local-frame CPs (with off-midline ``+y_local`` content
   excluded since the symmetric model deliberately discards it) ≤ 5 % of
   ``lmax``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import plantbox as pb
from scipy.interpolate import make_interp_spline

REPO_ROOT = Path(__file__).resolve().parents[3]
DIST_JSON = REPO_ROOT / "dart/coupling/data/maize_leaf_shape_distribution.json"
XML_PATH = REPO_ROOT / "dart/coupling/data/maize_calibrated.xml"
MF3D_JSON = Path("/home/lukas/PHD/Resources/MaizeField3d/maizefield3d_canonical_cps.json")


@pytest.fixture(scope="module")
def distribution() -> dict:
    if not DIST_JSON.exists():
        pytest.skip(f"S0 distribution missing: {DIST_JSON}")
    with open(DIST_JSON) as f:
        return json.load(f)


def _build_parametric(distribution: dict, rank: int) -> pb.ParametricLeafShape:
    """Construct a C++ ParametricLeafShape from the S0 distribution JSON."""
    n_u = distribution["n_u"]
    n_v = distribution["n_v"]
    n_cp = distribution["n_cp_per_axis"]
    degree = distribution["spline_degree"]
    knots = distribution["spline_knots_u"]
    intercept = np.asarray(distribution["intercepts"][str(rank)], dtype=np.float64)
    droop_c = intercept[0:n_cp].tolist()
    along_c = intercept[n_cp:2 * n_cp].tolist()
    width_c = intercept[2 * n_cp:3 * n_cp].tolist()
    residual = np.asarray(
        distribution["asym_residual_grids_cm"][str(rank)], dtype=np.float64
    )  # (n_u, n_v, 3)
    residual_flat = [pb.Vector3d(*residual[iu, iv]) for iu in range(n_u) for iv in range(n_v)]
    # S6 max_w bake: ParametricLeafShape baked with per-rank max_w_xml_cm
    # so evaluate() reproduces XML at FP precision regardless of evaluate's
    # runtime max_w arg (which is now ignored on the parametric path).
    max_w_intercept = float(distribution["max_w_xml_cm"][str(rank)])
    lmax_intercept = float(distribution["lmax_intercept_cm"][str(rank)])
    return pb.ParametricLeafShape(
        rank=rank,
        spline_knots_u=knots,
        spline_degree=degree,
        midrib_droop_coeffs=droop_c,
        midrib_along_coeffs=along_c,
        halfwidth_coeffs=width_c,
        asym_residual_grid=residual_flat,
        n_u=n_u,
        n_v=n_v,
        max_w_intercept=max_w_intercept,
        lmax_intercept=lmax_intercept,
    )


# ---------------------------------------------------------------------------
# Gate S3.1 — analytic
# ---------------------------------------------------------------------------
def test_s3_analytic_straight_droop_parabolic_width():
    """Straight-line droop, constant along, parabolic half-width, zero residual.

    With m_y(u) = a*u, m_z(u) = b, w(u) = 4*u*(1-u) (peak = 1 at u = 0.5),
    evaluate(u, v) at any (u, v) in [0,1]^2 must reproduce
        ((v - 0.5) * 4*u*(1-u) * max_w, a*u, b)
    to within FP precision (≤ 1e-9 cm). The B-spline interpolation is exact
    on its own coefficient vector at the u-stations; in between, scipy's
    BSpline (the reference implementation) and our De Boor evaluator agree.
    """
    n_u = 11
    n_v = 5
    degree = 4
    u_vals = np.linspace(0.0, 1.0, n_u)

    a = 3.7
    b = -1.4

    droop_data = a * u_vals
    along_data = np.full_like(u_vals, b)
    width_data = 4.0 * u_vals * (1.0 - u_vals)  # parabola, peak 1 at u=0.5

    droop_sp = make_interp_spline(u_vals, droop_data, k=degree)
    along_sp = make_interp_spline(u_vals, along_data, k=degree)
    width_sp = make_interp_spline(u_vals, width_data, k=degree)
    knots = droop_sp.t.tolist()

    residual = [pb.Vector3d(0.0, 0.0, 0.0) for _ in range(n_u * n_v)]
    max_w = 2.7  # cm; baked at construction (S6 max_w bake — evaluate's
                 # runtime max_w arg is ignored on the parametric path)
    shape = pb.ParametricLeafShape(
        rank=0,
        spline_knots_u=knots,
        spline_degree=degree,
        midrib_droop_coeffs=droop_sp.c.tolist(),
        midrib_along_coeffs=along_sp.c.tolist(),
        halfwidth_coeffs=width_sp.c.tolist(),
        asym_residual_grid=residual,
        n_u=n_u,
        n_v=n_v,
        max_w_intercept=max_w,
        lmax_intercept=1.0,  # coefs are already in absolute cm — identity
    )

    rng = np.random.default_rng(123)
    test_pts = rng.uniform(size=(200, 2))
    test_pts = np.vstack([test_pts, [[0.0, 0.0], [1.0, 1.0], [0.5, 0.5]]])

    max_abs = 0.0
    for (u, v) in test_pts:
        out = shape.evaluate(u=float(u), v=float(v), lmax=50.0, max_w=999.0)
        # Reference uses scipy BSpline at the same (u) — equivalent by construction
        m_y = float(droop_sp(u))
        m_z = float(along_sp(u))
        w = float(width_sp(u))
        ref = np.array([(v - 0.5) * w * max_w, m_y, m_z])
        diff = np.array([out.x - ref[0], out.y - ref[1], out.z - ref[2]])
        max_abs = max(max_abs, float(np.max(np.abs(diff))))
    assert max_abs <= 1e-9, f"analytic S3.1: max |err| = {max_abs:.3e} cm > 1e-9"


# ---------------------------------------------------------------------------
# Gate S3.2 — per-rank XML anchor through C++ evaluator
# ---------------------------------------------------------------------------
def _load_xml_grids() -> tuple[list[np.ndarray], list[float]]:
    """Read 15 maize leaf surface_cp grids + per-rank Width_blade values."""
    import xml.etree.ElementTree as ET

    tree = ET.parse(XML_PATH)
    root = tree.getroot()
    leaves = {int(lf.attrib["subType"]): lf for lf in root.findall(".//leaf")}
    n_u, n_v = 11, 5
    grids = []
    width_blade = []
    for r in range(15):
        lf = leaves[r + 2]
        cps = np.zeros((n_u, n_v, 3), dtype=np.float64)
        for cp in lf.findall("parameter[@name='surface_cp']"):
            i = int(round(float(cp.attrib["u"]) * (n_u - 1)))
            j = int(round(float(cp.attrib["v"]) * (n_v - 1)))
            cps[i, j] = (
                float(cp.attrib["x"]),
                float(cp.attrib["y"]),
                float(cp.attrib["z"]),
            )
        grids.append(cps)
        wb = lf.find("parameter[@name='Width_blade']")
        width_blade.append(float(wb.attrib["value"]))
    return grids, width_blade


@pytest.mark.parametrize("rank", list(range(15)))
def test_s3_per_rank_xml_anchor(distribution: dict, rank: int):
    """For rank r, evaluate(intercept[r] + residual[r]) on the canonical grid
    reproduces XML rank r's surface_cps to ≤ 1e-9 cm element-wise.

    This is the C++-side re-test of S0's gate (a) — the Python fitter passed
    that gate using its own scipy-based evaluator; this test pins that the
    C++ De Boor evaluator + bilinear residual interp produce the same result.
    """
    if not XML_PATH.exists():
        pytest.skip(f"XML missing: {XML_PATH}")
    xml_grids, _ = _load_xml_grids()
    shape = _build_parametric(distribution, rank)
    max_w_xml = float(distribution["max_w_xml_cm"][str(rank)])

    n_u = distribution["n_u"]
    n_v = distribution["n_v"]
    out = np.zeros((n_u, n_v, 3), dtype=np.float64)
    u_vals = np.linspace(0.0, 1.0, n_u)
    v_vals = np.linspace(0.0, 1.0, n_v)
    for iu, u in enumerate(u_vals):
        for iv, v in enumerate(v_vals):
            p = shape.evaluate(u=float(u), v=float(v), lmax=50.0, max_w=max_w_xml)
            out[iu, iv] = (p.x, p.y, p.z)

    diff = out - xml_grids[rank]
    max_abs = float(np.max(np.abs(diff)))
    assert max_abs <= 1e-9, (
        f"S3.2 rank {rank}: max |diff| = {max_abs:.3e} cm > 1e-9"
    )


# ---------------------------------------------------------------------------
# Gate S3.3 — donor round-trip
# ---------------------------------------------------------------------------
def test_s3_donor_round_trip(distribution: dict):
    """One MF3D donor: fit donor coefficients via the S0 fitter, reconstruct
    through the C++ evaluator with **zero asym residual** (this gate measures
    the symmetric-spline fit fidelity, not rank-specific asymmetry — that's
    S3.2's job). RMS deviation from raw donor local-frame CPs (with off-midline
    +y_local excluded) ≤ 5 % of lmax.

    Off-midline +y_local exclusion matches the plan: the symmetric parametric
    model writes m_y(u) into every v-row by design (flat cross-section, Phase
    1), so off-midline curl is residual content the model does not attempt to
    fit. We compare on the (x, z) full grid plus the (y) midline only.
    """
    if not MF3D_JSON.exists():
        pytest.skip(f"MF3D scans missing: {MF3D_JSON}")
    import sys
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from dart.coupling.geometry.canonical_library import to_local_frame
    from dart.coupling.scripts.fit_parametric_leaf_shape import (
        N_CP,
        SPLINE_DEGREE,
        U_VALS,
        extract_symmetric,
        fit_intercept,
    )

    with open(MF3D_JSON) as f:
        mf3d = json.load(f)
    # Pick first usable donor at position 5 (mid-canopy, well-populated)
    chosen = None
    for plant in mf3d["plants"]:
        for leaf in plant.get("leaves", []):
            if int(leaf.get("position", -1)) != 5:
                continue
            cps_world = np.asarray(leaf.get("cps_cm"), dtype=np.float64)
            if cps_world.shape != (11, 5, 3) or not np.isfinite(cps_world).all():
                continue
            try:
                cps_local, _, _ = to_local_frame(
                    cps_world, normalize_arc=False, tip_canonical_rotate=False,
                )
            except Exception:
                continue
            try:
                sym = extract_symmetric(cps_local)
            except Exception:
                continue
            chosen = (cps_local, sym)
            break
        if chosen is not None:
            break
    if chosen is None:
        pytest.skip("no usable MF3D donor at position 5")
    cps_local, sym = chosen

    coeffs_donor = fit_intercept(sym)
    n_u = distribution["n_u"]
    n_v = distribution["n_v"]
    knots = distribution["spline_knots_u"]
    # Zero asym residual — gate S3.3 measures symmetric-spline fit fidelity.
    residual_flat = [pb.Vector3d(0.0, 0.0, 0.0) for _ in range(n_u * n_v)]
    droop_c = coeffs_donor[0:N_CP].tolist()
    along_c = coeffs_donor[N_CP:2 * N_CP].tolist()
    width_c = coeffs_donor[2 * N_CP:3 * N_CP].tolist()
    # S6 max_w bake: donor's own peak half-width baked at construction
    # (replaces the previous evaluate-time max_w arg, now ignored).
    # Fix 2b: donor's own midrib arc length baked at construction —
    # coeffs_donor are dimensionless post-fit, multiplied through evaluate
    # by lmax_intercept to recover absolute cm.
    max_w_donor = sym.max_w
    lmax_donor = sym.lmax_self
    shape = pb.ParametricLeafShape(
        rank=5,
        spline_knots_u=knots,
        spline_degree=SPLINE_DEGREE,
        midrib_droop_coeffs=droop_c,
        midrib_along_coeffs=along_c,
        halfwidth_coeffs=width_c,
        asym_residual_grid=residual_flat,
        n_u=n_u,
        n_v=n_v,
        max_w_intercept=max_w_donor,
        lmax_intercept=lmax_donor,
    )

    out = np.zeros((n_u, n_v, 3), dtype=np.float64)
    for iu, u in enumerate(U_VALS):
        for iv in range(n_v):
            v = iv / (n_v - 1)
            p = shape.evaluate(u=float(u), v=float(v), lmax=50.0, max_w=999.0)
            out[iu, iv] = (p.x, p.y, p.z)

    # Compare to raw donor local-frame CPs with off-midline +y_local excluded:
    #   - x, z components: compare full grid
    #   - y component: compare midline (v_idx = n_v//2) only
    mid = n_v // 2
    diff_x = (out[:, :, 0] - cps_local[:, :, 0]).ravel()
    diff_z = (out[:, :, 2] - cps_local[:, :, 2]).ravel()
    diff_y_mid = out[:, mid, 1] - cps_local[:, mid, 1]
    residuals = np.concatenate([diff_x, diff_z, diff_y_mid])
    rms = float(np.sqrt(np.mean(residuals ** 2)))

    # Use the donor's apparent lmax (along-axis range) as the normaliser
    along_range = float(cps_local[:, mid, 2].max() - cps_local[:, mid, 2].min())
    threshold = 0.05 * along_range
    assert rms <= threshold, (
        f"S3.3: donor round-trip RMS = {rms:.3f} cm vs threshold "
        f"{threshold:.3f} cm (5 %% of lmax_proxy = {along_range:.3f} cm)"
    )
