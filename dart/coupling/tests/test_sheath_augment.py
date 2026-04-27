"""Phase E: procedural compound sheath+blade CP grid.

Validates :func:`augment_with_sheath` — the positioning-choice approach
described in plan section E.2. A blade-only library CP grid is extended in
the -u direction with ``n_sheath_rows`` control points whose x/y coordinates
wrap the parent-stem axis. No change to grid topology is needed on the
blade side; the NURBS surface naturally interpolates the wrap.
"""

from __future__ import annotations

import numpy as np

from dart.coupling.geometry.canonical_library import augment_with_sheath
from dart.coupling.geometry.canonical_cp_grid import N_U, N_V


def _fake_blade(n_u: int = N_U, n_v: int = N_V, length: float = 60.0,
                 width: float = 8.0) -> np.ndarray:
    """Produce a smooth synthetic blade CP grid in leaf-local frame."""
    u = np.linspace(0.0, length, n_u)
    v_half = width * 0.5
    v = np.linspace(-v_half, v_half, n_v)
    cps = np.zeros((n_u, n_v, 3), dtype=np.float64)
    for i_u, z in enumerate(u):
        frac = float(i_u) / max(n_u - 1, 1)
        # Taper the tip for realism.
        w = v * (1.0 - 0.8 * frac ** 2)
        cps[i_u, :, 0] = w
        cps[i_u, :, 1] = 0.1 * frac  # subtle droop
        cps[i_u, :, 2] = z
    return cps


def test_augment_shape_and_frame():
    blade = _fake_blade()
    stem_r = 0.5
    sheath_len = 12.0
    n_sheath = 3
    compound = augment_with_sheath(
        blade,
        stem_radius_cm=stem_r,
        sheath_length_cm=sheath_len,
        wrap_deg=272.0,
        n_sheath_rows=n_sheath,
    )
    assert compound.shape == (N_U + n_sheath, N_V, 3)
    # Blade rows unchanged.
    np.testing.assert_allclose(compound[n_sheath:], blade, atol=1e-12)


def test_sheath_wraps_stem_axis():
    """Sheath CPs must lie on a cylinder of radius ``stem_radius_cm`` around
    the +z axis, with z < 0 (below the collar)."""
    blade = _fake_blade()
    stem_r = 0.42
    sheath_len = 10.0
    n_sheath = 4
    compound = augment_with_sheath(
        blade,
        stem_radius_cm=stem_r,
        sheath_length_cm=sheath_len,
        n_sheath_rows=n_sheath,
    )
    sheath = compound[:n_sheath]

    # All sheath CPs at radius stem_r from z-axis.
    radii = np.linalg.norm(sheath[:, :, :2], axis=-1)
    np.testing.assert_allclose(radii, stem_r, rtol=1e-6)

    # All sheath CPs strictly below collar.
    assert float(sheath[:, :, 2].max()) < 0.0
    # Tip of sheath at -sheath_length.
    assert abs(float(sheath[0, :, 2].min()) + sheath_len) < 1e-6


def test_wrap_arc_spans_requested_degrees():
    blade = _fake_blade()
    compound = augment_with_sheath(
        blade,
        stem_radius_cm=0.5,
        sheath_length_cm=10.0,
        wrap_deg=272.0,
        n_sheath_rows=3,
    )
    sheath = compound[:3]
    # v=0 and v=N_V-1 sit at the seam (half_wrap degrees either side).
    seam_a = sheath[0, 0, :2]
    seam_b = sheath[0, -1, :2]
    # Angle measured from +y (blade-facing) around the axis.
    def _angle(xy):
        return float(np.degrees(np.arctan2(xy[0], xy[1])))
    a = _angle(seam_a)
    b = _angle(seam_b)
    assert abs((b - a) - 272.0) < 1e-3, (
        f"wrap span {b - a:.2f}°, expected 272.0°"
    )


def _build_nurbs_patch(cps: np.ndarray, deg_u: int, deg_v: int):
    """Build a PlantGL ``NurbsPatch`` for arbitrary CP grid shape (not just
    the canonical 11×5). The canonical helper hardcodes ``(N_U, N_V, 3)``."""
    from dart.coupling.geometry.canonical_cp_grid import (
        ensure_plantgl_loaded, build_uniform_knotvector,
    )
    ensure_plantgl_loaded()
    from openalea.plantgl.scenegraph import NurbsPatch, Point4Matrix, RealArray
    n_u, n_v, _ = cps.shape
    rows = [
        [(float(cps[i, j, 0]), float(cps[i, j, 1]), float(cps[i, j, 2]), 1.0)
         for j in range(n_v)]
        for i in range(n_u)
    ]
    ku = build_uniform_knotvector(n_u, deg_u)
    kv = build_uniform_knotvector(n_v, deg_v)
    return NurbsPatch(Point4Matrix(rows), deg_u, deg_v,
                      RealArray(ku.tolist()), RealArray(kv.tolist()))


def test_nurbs_evaluation_produces_finite_mesh():
    """Feed the compound CPs through PlantGL NURBS evaluation. The surface
    must be continuous (no NaN, no huge jumps at the collar row)."""
    from dart.coupling.geometry.canonical_cp_grid import DEG_U, DEG_V

    blade = _fake_blade(length=50.0, width=10.0)
    compound = augment_with_sheath(
        blade,
        stem_radius_cm=0.4,
        sheath_length_cm=8.0,
        wrap_deg=272.0,
        n_sheath_rows=3,
    )
    patch = _build_nurbs_patch(compound, DEG_U, DEG_V)

    n_u_eval, n_v_eval = 48, 16
    verts = np.empty((n_u_eval * n_v_eval, 3), dtype=np.float64)
    us = np.linspace(0.0, 1.0, n_u_eval)
    vs = np.linspace(0.0, 1.0, n_v_eval)
    for i, u in enumerate(us):
        for j, v in enumerate(vs):
            p = patch.getPointAt(float(u), float(v))
            verts[i * n_v_eval + j] = (p.x, p.y, p.z)

    assert np.isfinite(verts).all(), "NURBS evaluation produced NaN/Inf"
    assert float(verts[:, 2].min()) < -1.0, "sheath not rendered below collar"
    assert float(verts[:, 2].max()) > 30.0, "blade tip not reached"


if __name__ == "__main__":
    test_augment_shape_and_frame()
    print("shape + frame: OK")
    test_sheath_wraps_stem_axis()
    print("sheath wraps axis: OK")
    test_wrap_arc_spans_requested_degrees()
    print("wrap arc span: OK")
    test_nurbs_evaluation_produces_finite_mesh()
    print("NURBS eval produces finite mesh: OK")
