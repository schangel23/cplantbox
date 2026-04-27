"""Unit tests for the canonical CP grid module (Phase 0).

Covers:
  1. Knot-vector construction (clamped, uniform, correct length).
  2. Round-trip (synthetic grid → PlantGL NurbsPatch → evaluate corners).
  3. geomdl vs PlantGL agreement (ensures the transpose convention in
     cp_grid_to_geomdl_surface is correct).
  4. Orientation convention: mirrored input gets flipped back.
"""

from __future__ import annotations

import numpy as np
import pytest

from dart.coupling.geometry.canonical_cp_grid import (
    N_U, N_V, DEG_U, DEG_V,
    build_uniform_knotvector, canonical_knots,
    cp_grid_to_plantgl_patch, cp_grid_to_geomdl_surface,
    eval_grid, enforce_orientation,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _synthetic_flat_cps() -> np.ndarray:
    """11×5 CP grid describing a flat rectangle:
      u (axis 0): 0 cm → 10 cm along +x
      v (axis 1): −1 cm → +1 cm along +y
      z = 0 everywhere
    """
    us = np.linspace(0.0, 10.0, N_U)
    vs = np.linspace(-1.0, 1.0, N_V)
    cps = np.zeros((N_U, N_V, 3), dtype=np.float64)
    for i, u in enumerate(us):
        for j, v in enumerate(vs):
            cps[i, j] = (u, v, 0.0)
    return cps


# ---------------------------------------------------------------------------
# 1. Knot vectors
# ---------------------------------------------------------------------------
def test_knot_vector_shapes_and_clamping():
    uk = build_uniform_knotvector(N_U, DEG_U)
    vk = build_uniform_knotvector(N_V, DEG_V)
    assert len(uk) == N_U + DEG_U + 1 == 15
    assert len(vk) == N_V + DEG_V + 1 == 8
    # Clamped: first deg+1 are 0, last deg+1 are 1
    assert np.allclose(uk[: DEG_U + 1], 0.0)
    assert np.allclose(uk[-(DEG_U + 1):], 1.0)
    assert np.allclose(vk[: DEG_V + 1], 0.0)
    assert np.allclose(vk[-(DEG_V + 1):], 1.0)
    # Interior knots strictly increasing in (0, 1)
    interior_u = uk[DEG_U + 1: -(DEG_U + 1)]
    assert np.all(interior_u > 0) and np.all(interior_u < 1)
    assert np.all(np.diff(interior_u) > 0)


def test_canonical_knots_matches_builder():
    uk1, vk1 = canonical_knots()
    uk2 = build_uniform_knotvector(N_U, DEG_U)
    vk2 = build_uniform_knotvector(N_V, DEG_V)
    assert np.array_equal(uk1, uk2)
    assert np.array_equal(vk1, vk2)


def test_knot_builder_rejects_insufficient_ctrlpts():
    with pytest.raises(ValueError):
        build_uniform_knotvector(3, 3)  # n_ctrl must be > degree


# ---------------------------------------------------------------------------
# 2. PlantGL round-trip
# ---------------------------------------------------------------------------
def test_plantgl_patch_corner_evaluation():
    cps = _synthetic_flat_cps()
    patch = cp_grid_to_plantgl_patch(cps)

    # Clamped knots → corners of parametric square hit corner CPs.
    p00 = patch.getPointAt(0.0, 0.0)
    p10 = patch.getPointAt(1.0, 0.0)
    p01 = patch.getPointAt(0.0, 1.0)
    p11 = patch.getPointAt(1.0, 1.0)

    assert np.allclose([p00.x, p00.y, p00.z], cps[0, 0])
    assert np.allclose([p10.x, p10.y, p10.z], cps[-1, 0])
    assert np.allclose([p01.x, p01.y, p01.z], cps[0, -1])
    assert np.allclose([p11.x, p11.y, p11.z], cps[-1, -1])


def test_plantgl_patch_midrib_on_centerline():
    cps = _synthetic_flat_cps()
    patch = cp_grid_to_plantgl_patch(cps)
    # A flat rectangular CP grid → midrib at v=0.5 should sit on the y=0
    # centerline for every u.
    for u in np.linspace(0, 1, 7):
        p = patch.getPointAt(float(u), 0.5)
        assert abs(p.y) < 1e-9
        assert abs(p.z) < 1e-9


def test_eval_grid_shape_and_normals():
    cps = _synthetic_flat_cps()
    patch = cp_grid_to_plantgl_patch(cps)
    verts, norms = eval_grid(patch, n_u=4, n_v=3)
    assert verts.shape == (12, 3)
    assert norms.shape == (12, 3)
    # Flat XY-plane input → all normals should be ±z
    assert np.allclose(np.abs(norms[:, 2]), 1.0, atol=1e-6)
    assert np.allclose(norms[:, :2], 0.0, atol=1e-6)


def test_cp_grid_wrong_shape_raises():
    bad = np.zeros((10, 5, 3))
    with pytest.raises(ValueError):
        cp_grid_to_plantgl_patch(bad)
    with pytest.raises(ValueError):
        cp_grid_to_geomdl_surface(bad)


# ---------------------------------------------------------------------------
# 3. geomdl vs PlantGL agreement (catches transpose bugs)
# ---------------------------------------------------------------------------
def test_geomdl_plantgl_agree_at_random_points():
    rng = np.random.RandomState(42)
    # A non-trivial CP grid (flat grid can hide axis bugs by symmetry).
    cps = _synthetic_flat_cps()
    # Perturb each CP's z so u and v directions are distinguishable.
    for i in range(N_U):
        for j in range(N_V):
            # z depends asymmetrically on both u and v indices
            cps[i, j, 2] = 0.1 * i + 0.3 * j + 0.05 * i * j

    patch = cp_grid_to_plantgl_patch(cps)
    surf = cp_grid_to_geomdl_surface(cps)

    for _ in range(10):
        u = float(rng.uniform(0, 1))
        v = float(rng.uniform(0, 1))
        p_pgl = patch.getPointAt(u, v)
        p_geo = surf.evaluate_single((u, v))
        pgl = np.array([p_pgl.x, p_pgl.y, p_pgl.z])
        geo = np.array(p_geo)
        assert np.allclose(pgl, geo, atol=1e-6), (
            f"PlantGL vs geomdl disagree at (u={u}, v={v}): "
            f"pgl={pgl} geo={geo}"
        )


# ---------------------------------------------------------------------------
# 4. Orientation convention
# ---------------------------------------------------------------------------
def test_enforce_orientation_is_idempotent_on_canonical_input():
    cps = _synthetic_flat_cps()
    out = enforce_orientation(cps)
    assert np.array_equal(out, cps)


def test_enforce_orientation_flips_mirrored_input():
    cps = _synthetic_flat_cps()
    mirrored = cps[:, ::-1, :].copy()
    fixed = enforce_orientation(mirrored)
    assert np.allclose(fixed, cps)


def test_enforce_orientation_preserves_valid_rotation():
    """Rotating the leaf around the +z axis should not trigger an erroneous
    flip — the convention is "v=0 on +x_local side", where x_local comes
    from SVD of the leaf itself.
    """
    cps = _synthetic_flat_cps()
    # Rotate 30° around +z. SVD frame follows; orientation should hold.
    theta = np.deg2rad(30)
    R = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0, 0, 1],
    ])
    rotated = (cps.reshape(-1, 3) @ R.T).reshape(cps.shape)
    out = enforce_orientation(rotated)
    # Expect no flip — result should equal rotated, not rotated[:, ::-1, :]
    assert np.allclose(out, rotated)
