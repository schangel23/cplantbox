"""Young-stage Pheno4D blending into the native surface_cps pipeline.

Covers the Pheno4D young-library → mature-MF3D cross-fade added in
``canonical_library.build_young_library_from_pheno4d`` +
``blend_young_mature_cps``:

* the built npz has the expected shape and per-bucket midrib arc = 1.0
* ``blend_young_mature_cps`` endpoints: maturity ≤ fade_start → pure
  young (scaled by mature_length); maturity ≥ fade_end → pure mature;
  intermediate maturity produces a grid whose midrib arc is close to
  mature_length (no catastrophic shrinkage)
* the cross-fade is monotonic-ish in maturity (L2 distance to mature
  is non-increasing in [fade_start, fade_end])
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dart.coupling.geometry.canonical_library import (
    N_U,
    N_V,
    _midrib_arc,
    _planarise_pheno4d_fit,
    blend_young_mature_cps,
    load_young_library,
    select_young_cps,
)


YOUNG_LIB_PATH = Path(
    "/home/lukas/PHD/CPlantBox/dart/coupling/data/pheno4d_young_library.npz"
)


@pytest.fixture(scope="module")
def young_lib():
    if not YOUNG_LIB_PATH.exists():
        pytest.skip(f"young library missing: {YOUNG_LIB_PATH}")
    return load_young_library(YOUNG_LIB_PATH)


def test_library_shape_and_arcs(young_lib):
    cps = young_lib["cps_normalised"]
    centers = young_lib["bucket_centers"]
    counts = young_lib["counts"]

    assert cps.ndim == 4
    assert cps.shape[1:] == (11, 5, 3)
    assert cps.shape[0] == centers.shape[0] == counts.shape[0]
    assert cps.shape[0] >= 1

    for i in range(cps.shape[0]):
        arc = _midrib_arc(cps[i])
        assert arc == pytest.approx(1.0, abs=1e-3), (
            f"bucket {i} (center={centers[i]:.2f}) arc={arc:.4f} "
            "(expected 1.0 after post-aggregation renormalisation)"
        )


def test_blend_endpoints(young_lib):
    mature_len = 60.0
    mature = np.zeros((11, 5, 3), dtype=np.float64)
    mature[:, :, 2] = np.linspace(0.0, mature_len, 11)[:, None]
    mature[:, :, 0] = np.linspace(-0.5, 0.5, 5)[None, :] * 8.0

    young = select_young_cps(young_lib, 0.25)

    pure_young = blend_young_mature_cps(
        young, mature, maturity=0.1, mature_length=mature_len,
    )
    expected_young_scaled = young * mature_len
    assert np.allclose(pure_young, expected_young_scaled)

    pure_mature = blend_young_mature_cps(
        young, mature, maturity=0.95, mature_length=mature_len,
    )
    assert np.allclose(pure_mature, mature)


def test_blend_preserves_midrib_arc(young_lib):
    mature_len = 72.0
    mature = np.zeros((11, 5, 3), dtype=np.float64)
    u = np.linspace(0.0, 1.0, 11)
    mature[:, :, 2] = (u * mature_len)[:, None]
    young = select_young_cps(young_lib, 0.55)

    for maturity in (0.0, 0.3, 0.6, 1.0):
        blended = blend_young_mature_cps(
            young, mature, maturity=maturity, mature_length=mature_len,
        )
        arc = _midrib_arc(blended)
        assert arc == pytest.approx(mature_len, rel=0.05), (
            f"maturity={maturity} blended arc={arc:.2f} vs {mature_len}"
        )


def test_blend_monotonic_toward_mature(young_lib):
    mature_len = 50.0
    mature = np.zeros((11, 5, 3), dtype=np.float64)
    mature[:, :, 2] = np.linspace(0.0, mature_len, 11)[:, None]
    mature[:, :, 0] = np.linspace(-3.0, 3.0, 5)[None, :]

    young = select_young_cps(young_lib, 0.35)

    maturities = np.linspace(0.0, 1.0, 11)
    dists = []
    for m in maturities:
        blended = blend_young_mature_cps(
            young, mature, maturity=float(m), mature_length=mature_len,
        )
        dists.append(float(np.linalg.norm(blended - mature)))

    for a, b in zip(dists, dists[1:]):
        assert b <= a + 1e-9, f"distance to mature increased: {dists}"


def test_select_young_nearest(young_lib):
    centers = np.asarray(young_lib["bucket_centers"])
    for target in (0.0, 0.5, 0.95):
        cps = select_young_cps(young_lib, target)
        idx = int(np.argmin(np.abs(centers - target)))
        assert np.allclose(cps, young_lib["cps_normalised"][idx])


# ---------------------------------------------------------------------------
# Planariser unit tests
# ---------------------------------------------------------------------------
def _synth_planar_blade(
    length: float = 1.0,
    max_half_width: float = 0.05,
    midrib_curve_x: float = 0.0,
) -> np.ndarray:
    """Build a synthetic flat blade in the xz-plane.

    Midrib runs from (0,0,0) to (midrib_curve_x, 0, length) along a
    quadratic in z. Half-widths taper linearly toward the tip.
    """
    u = np.linspace(0.0, 1.0, N_U)
    # Midrib in xz-plane: small sideways bow controlled by midrib_curve_x.
    mid_x = midrib_curve_x * (4.0 * u * (1.0 - u))  # 0 at ends, peak mid
    mid_z = u * length
    # Half-width taper: 0 at base, peak at u≈0.2, 0 at tip.
    taper = np.where(u < 0.2, u / 0.2, (1.0 - u) / 0.8) * max_half_width
    cps = np.zeros((N_U, N_V, 3), dtype=np.float64)
    v_offsets = np.linspace(-1.0, 1.0, N_V)
    for i in range(N_U):
        for j in range(N_V):
            cps[i, j, 0] = mid_x[i] + v_offsets[j] * taper[i]
            cps[i, j, 1] = 0.0
            cps[i, j, 2] = mid_z[i]
    return cps


def _apply_droop(cps: np.ndarray, droop_amp: float) -> np.ndarray:
    """Bend midrib forward (into +y) by a quadratic droop term."""
    cps = cps.copy()
    u = np.linspace(0.0, 1.0, cps.shape[0])
    dy = droop_amp * u * u  # zero at base, max at tip
    cps[..., 1] += dy[:, None]
    return cps


def _apply_twist(cps: np.ndarray, twist_total_rad: float) -> np.ndarray:
    """Rotate each u-row around its midrib by an angle linear in u."""
    cps = cps.copy()
    mid_j = cps.shape[1] // 2
    u = np.linspace(0.0, 1.0, cps.shape[0])
    angles = twist_total_rad * u
    for i in range(cps.shape[0]):
        M = cps[i, mid_j].copy()
        c = np.cos(angles[i])
        s = np.sin(angles[i])
        # Rotate around local midrib-tangent axis (approximate: use +z).
        rel = cps[i] - M
        rotated = np.stack([
            c * rel[:, 0] - s * rel[:, 1],
            s * rel[:, 0] + c * rel[:, 1],
            rel[:, 2],
        ], axis=1)
        cps[i] = M + rotated
    return cps


def test_planarise_clean_blade_passes():
    blade = _synth_planar_blade(length=1.0, max_half_width=0.05)
    cleaned = _planarise_pheno4d_fit(blade)
    assert cleaned is not None, "clean planar blade should survive QA"
    mid_j = cleaned.shape[1] // 2
    midrib = cleaned[:, mid_j, :]
    arc = _midrib_arc(cleaned)
    assert arc > 0
    # After planarisation the midrib should point almost straight +z.
    tip_z_frac = (midrib[-1, 2] - midrib[0, 2]) / arc
    assert tip_z_frac > 0.95


def test_planarise_removes_droop():
    blade = _synth_planar_blade(length=1.0, max_half_width=0.05)
    drooped = _apply_droop(blade, droop_amp=0.15)  # 15% of arc tipward droop
    cleaned = _planarise_pheno4d_fit(drooped)
    assert cleaned is not None, "mild droop should be planarised, not rejected"
    mid_j = cleaned.shape[1] // 2
    midrib = cleaned[:, mid_j, :]
    arc = _midrib_arc(cleaned)
    y_range = (midrib[:, 1].max() - midrib[:, 1].min()) / arc
    assert y_range < 0.05, (
        f"y-range after planarisation should collapse to ~0; got {y_range:.3f}"
    )


def test_planarise_rejects_whorl_wrap():
    # Build a spiral midrib winding ~180° around +z to trip the whorl-wrap gate.
    u = np.linspace(0.0, 1.0, N_U)
    radius = 0.08 + 0.04 * u
    theta = np.pi * u  # 180° total
    cps = np.zeros((N_U, N_V, 3), dtype=np.float64)
    v_offsets = np.linspace(-1.0, 1.0, N_V)
    width = 0.05
    for i in range(N_U):
        cx = radius[i] * np.cos(theta[i])
        cy = radius[i] * np.sin(theta[i])
        for j in range(N_V):
            cps[i, j, 0] = cx + v_offsets[j] * width
            cps[i, j, 1] = cy
            cps[i, j, 2] = u[i]
    cleaned = _planarise_pheno4d_fit(cps, max_wind_deg=60.0)
    assert cleaned is None, "180° whorl-wrap should be rejected by the gate"


def test_planarise_symmetrises_twist():
    blade = _synth_planar_blade(length=1.0, max_half_width=0.05)
    twisted = _apply_twist(blade, twist_total_rad=0.3)  # ~17° total twist
    cleaned = _planarise_pheno4d_fit(twisted)
    assert cleaned is not None
    # After step 3 the blade should be reflection-symmetric across v = mid_j.
    mid_j = cleaned.shape[1] // 2
    for i in range(cleaned.shape[0]):
        M = cleaned[i, mid_j]
        for j in range(mid_j):
            j_m = cleaned.shape[1] - 1 - j
            p_j = cleaned[i, j] - M
            p_m = cleaned[i, j_m] - M
            # Symmetric = p_m is the reflection of p_j across the symmetry plane.
            # Parallel component (midrib tangent, plane_normal) equal, across-
            # blade component flipped. Simplest check: sum has zero projection
            # on the across-blade direction after symmetrisation.
            assert np.linalg.norm(p_j + p_m - _parallel_component(p_j + p_m, cleaned, i)) \
                == pytest.approx(0.0, abs=5e-2)


def _parallel_component(vec: np.ndarray, cps: np.ndarray, i: int) -> np.ndarray:
    """Project ``vec`` onto the symmetry plane (tangent, plane_normal) at u_i.

    Used by the symmetrisation test to isolate the across-blade residual.
    """
    mid_j = cps.shape[1] // 2
    if i + 1 < cps.shape[0]:
        t = cps[i + 1, mid_j] - cps[i, mid_j]
    else:
        t = cps[i, mid_j] - cps[i - 1, mid_j]
    t_len = float(np.linalg.norm(t))
    if t_len < 1e-9:
        return np.zeros(3)
    t = t / t_len
    # Plane normal = +y (by planariser convention).
    n = np.array([0.0, 1.0, 0.0])
    out = np.zeros(3)
    out += float(vec @ t) * t
    out += float(vec @ n) * n
    return out
