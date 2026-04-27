"""Unit tests for the Pheno4D PLY → NURBS LSQ fitter (Phase 2b).

Covers:
  1. Output shape and dtype.
  2. LSQ residual on a synthetic cylindrical leaf patch (known geometry).
  3. Orientation idempotence after ``enforce_orientation``.
  4. Collar-end orientation: ``u=0`` end of the fitted midrib is closer to
     the specified ``stem_base`` than ``u=1``.
  5. (Optional) real Pheno4D `.txt` smoke test — skipped if fixtures are
     not on disk.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Bring the Pheno4D fitter onto sys.path.
_VAULT = Path(__file__).resolve().parents[4]
_PHENO4D_DIR = _VAULT / "Resources" / "Pheno4D"
if str(_PHENO4D_DIR) not in sys.path:
    sys.path.insert(0, str(_PHENO4D_DIR))

from ply_to_nurbs import (  # noqa: E402
    fit_nurbs_to_pheno4d_leaf,
    parameterise_leaf,
    fit_nurbs_to_pheno4d_txt,
)


# ---------------------------------------------------------------------------
# Synthetic leaf generator
# ---------------------------------------------------------------------------
def _synthetic_curved_leaf(
    n_points: int = 6000,
    length_cm: float = 60.0,
    half_width_cm: float = 4.0,
    sag_cm: float = 6.0,
    noise_cm: float = 0.2,
    rng_seed: int = 0,
) -> np.ndarray:
    """Sample points on a parabolic-sagged leaf patch.

    Coordinate system:
      - +X axis is leaf-long, origin is the collar.
      - Y spans the leaf width at each x.
      - Z dips downward (sag) with a parabolic profile along x.
    """
    rng = np.random.default_rng(rng_seed)
    u = rng.uniform(0.0, 1.0, size=n_points)
    v = rng.uniform(-1.0, 1.0, size=n_points)
    # Width narrows near tip (tapered): half_width * (1 - u^2).
    w = half_width_cm * (1.0 - 0.6 * u * u)
    # Parabolic vertical sag: 0 at collar, minimum at mid-leaf, returns
    # near 0 at tip — characteristic drooping leaf shape.
    sag = -sag_cm * 4.0 * u * (1.0 - u)
    pts = np.column_stack([
        u * length_cm,
        v * w,
        sag + 0.3 * v * v,  # slight gutter
    ])
    if noise_cm > 0:
        pts = pts + rng.normal(0.0, noise_cm, size=pts.shape)
    return pts


# ---------------------------------------------------------------------------
# 1. Output contract
# ---------------------------------------------------------------------------
def test_output_shape_and_dtype():
    pts = _synthetic_curved_leaf()
    cps = fit_nurbs_to_pheno4d_leaf(pts)
    assert isinstance(cps, np.ndarray)
    assert cps.shape == (11, 5, 3)
    assert cps.dtype == np.float64


# ---------------------------------------------------------------------------
# 2. Fit residual
# ---------------------------------------------------------------------------
def test_fit_residual_beats_noise_floor():
    """On a smooth synthetic leaf the LSQ residual should track noise."""
    pts_clean = _synthetic_curved_leaf(noise_cm=0.0)
    _, resid_clean = fit_nurbs_to_pheno4d_leaf(pts_clean, return_residual=True)
    assert resid_clean["rmse_cm"] < 0.2, resid_clean

    pts_noisy = _synthetic_curved_leaf(noise_cm=0.2)
    _, resid_noisy = fit_nurbs_to_pheno4d_leaf(pts_noisy, return_residual=True)
    # Noise floor is ~0.2 cm per axis → expect rmse ~0.3-0.4 cm, well under
    # the 0.5 cm plan-document acceptance threshold.
    assert resid_noisy["rmse_cm"] < 0.5, resid_noisy
    assert resid_noisy["n_points"] == pts_noisy.shape[0]


# ---------------------------------------------------------------------------
# 3. Orientation
# ---------------------------------------------------------------------------
def test_orientation_idempotent():
    from dart.coupling.geometry.canonical_cp_grid import enforce_orientation
    pts = _synthetic_curved_leaf()
    cps = fit_nurbs_to_pheno4d_leaf(pts, apply_orientation=True)
    cps_again = enforce_orientation(cps)
    assert np.array_equal(cps, cps_again)


# ---------------------------------------------------------------------------
# 4. Collar orientation
# ---------------------------------------------------------------------------
def test_collar_end_is_closer_to_stem_base():
    pts = _synthetic_curved_leaf()
    stem_base = np.array([0.0, 0.0, 0.0])  # collar sits at origin above
    cps = fit_nurbs_to_pheno4d_leaf(
        pts, stem_base=stem_base, apply_orientation=False
    )
    midrib = cps[:, cps.shape[1] // 2, :]  # v=0.5 row ≈ midrib
    d_u0 = float(np.linalg.norm(midrib[0] - stem_base))
    d_u1 = float(np.linalg.norm(midrib[-1] - stem_base))
    assert d_u0 < d_u1, (
        f"u=0 should be the collar (closer to stem base); "
        f"got d(u=0)={d_u0:.2f} >= d(u=1)={d_u1:.2f}"
    )


def test_collar_orientation_flips_when_data_reversed():
    """Reversing the synthetic cloud along +X (so 'tip' is near the origin)
    should cause the fitter to flip u so u=0 still matches the stem base."""
    pts = _synthetic_curved_leaf()
    # Reflect across origin along +X, then shift so the tip ends up at origin.
    pts_rev = pts.copy()
    pts_rev[:, 0] = -pts_rev[:, 0]
    # Now pts_rev has the collar (originally at x=0) at x=0, and the tip at
    # x=-60. To simulate "tip at origin, collar far away", shift so tip=0:
    pts_rev[:, 0] += 60.0  # tip was at x=-60 → now 0; collar was at 0 → 60

    cps = fit_nurbs_to_pheno4d_leaf(
        pts_rev, stem_base=np.zeros(3), apply_orientation=False
    )
    midrib = cps[:, cps.shape[1] // 2, :]
    d_u0 = float(np.linalg.norm(midrib[0] - np.zeros(3)))
    d_u1 = float(np.linalg.norm(midrib[-1] - np.zeros(3)))
    assert d_u0 < d_u1, (
        f"collar flip failed: d(u=0)={d_u0:.2f} vs d(u=1)={d_u1:.2f}"
    )


# ---------------------------------------------------------------------------
# 5. Parameterisation range
# ---------------------------------------------------------------------------
def test_parameterisation_spans_unit_square():
    pts = _synthetic_curved_leaf()
    u, v, _ = parameterise_leaf(pts)
    assert u.min() == pytest.approx(0.0, abs=1e-9)
    assert u.max() == pytest.approx(1.0, abs=1e-9)
    assert v.min() == pytest.approx(0.0, abs=1e-9)
    assert v.max() == pytest.approx(1.0, abs=1e-9)
    assert u.shape == (pts.shape[0],)
    assert v.shape == (pts.shape[0],)


# ---------------------------------------------------------------------------
# 6. Real Pheno4D smoke test
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "txt_rel,leaf_label",
    [
        ("Maize01/M01_0317_a.txt", 2),
        ("Maize01/M01_0325_a.txt", 3),
    ],
)
def test_real_pheno4d_leaf_fits(txt_rel: str, leaf_label: int):
    """Smoke test on real annotated scans — skipped if files absent."""
    txt_path = _PHENO4D_DIR / txt_rel
    if not txt_path.exists():
        pytest.skip(f"fixture missing: {txt_path}")

    result = fit_nurbs_to_pheno4d_txt(str(txt_path), min_points=200)
    if leaf_label not in result or result[leaf_label].get("status") != "ok":
        pytest.skip(f"leaf {leaf_label} not fittable in {txt_rel}: {result.get(leaf_label)}")

    payload = result[leaf_label]
    cps = np.array(payload["cps_cm"])
    assert cps.shape == (11, 5, 3)
    resid = payload["fit_residual"]
    # Pheno4D scan noise ~2-5 mm; expect RMSE well under 1 cm.
    assert resid["rmse_cm"] < 1.0, resid
    # Leaf extent sanity: at least a few cm end-to-end.
    assert resid["u_extent_cm"] > 2.0


# ---------------------------------------------------------------------------
# 7. Boundary-CP magnitude gate (regression — pre-fix fits produced
#    |CP| ∈ thousands-to-millions of cm on several Pheno4D plants).
# ---------------------------------------------------------------------------
def test_boundary_cp_magnitude_on_sparse_cloud():
    """Cloud supported only on u ∈ [0.2, 0.8] leaves the u=0 / u=N_U-1 CP
    rows under-determined. With weighted-Tikhonov boundary regularisation
    (``regularise_boundary=1e-1`` default), those rows stay bounded; with
    the old isotropic ``1e-6`` they blew up by 3-6 orders of magnitude.
    """
    pts = _synthetic_curved_leaf(n_points=8000, noise_cm=0.1)
    # Keep only the interior 60% along the principal axis — x ∈ [12, 48] cm
    # on a 60-cm synthetic leaf corresponds to u ∈ [0.2, 0.8].
    length_cm = 60.0
    x = pts[:, 0]
    keep = (x >= 0.2 * length_cm) & (x <= 0.8 * length_cm)
    pts_sparse = pts[keep]
    assert pts_sparse.shape[0] > 200, "too few points after clipping"

    cps = fit_nurbs_to_pheno4d_leaf(pts_sparse, apply_orientation=False)
    assert cps.shape == (11, 5, 3)

    # "Bounded" here means within a few leaf-lengths of the origin, not
    # thousands-to-millions as in the pre-fix fits. For a 60 cm synthetic
    # leaf, 150 cm ≈ 2.5× length is a reasonable "not-blown-up" gate;
    # catches the pre-fix O(10³)–O(10⁶) cm failures with 1-4 orders of
    # magnitude of margin. Real Pheno4D leaves span ~1–2 cm natively, so
    # the ``fit_all_ply_to_nurbs`` acceptance gate of 20 cm on real data
    # is far tighter than this synthetic stress test.
    length_cm = 60.0
    gate_cm = 2.5 * length_cm  # 150 cm
    max_collar = float(np.abs(cps[0]).max())
    max_tip = float(np.abs(cps[-1]).max())
    assert max_collar < gate_cm, (
        f"collar row (u=0) max |CP| = {max_collar:.2f} cm "
        f"— boundary regularisation not holding (gate {gate_cm:.0f} cm)"
    )
    assert max_tip < gate_cm, (
        f"tip row (u=N_U-1) max |CP| = {max_tip:.2f} cm "
        f"— boundary regularisation not holding (gate {gate_cm:.0f} cm)"
    )

    # Interior rows should not be distorted by the boundary pull.
    max_interior = float(np.abs(cps[1:-1]).max())
    assert max_interior < gate_cm, (
        f"interior CPs max |CP| = {max_interior:.2f} cm"
    )


def test_boundary_regularization_preserves_interior_fit():
    """Adding the boundary-α diagonal must not degrade the interior fit
    below the existing 0.5 cm RMSE gate. Catches future regressions where
    someone raises ``regularise_boundary`` enough to distort non-boundary
    CPs.
    """
    pts = _synthetic_curved_leaf(noise_cm=0.2)
    cps, resid = fit_nurbs_to_pheno4d_leaf(
        pts,
        return_residual=True,
        # Explicitly pass the new defaults — pins the contract in the test.
        regularise=1e-6,
        regularise_boundary=1e-1,
    )
    assert resid["rmse_cm"] < 0.5, resid
    # Also assert all CPs are bounded for the full-coverage fit.
    assert float(np.abs(cps).max()) < 100.0
