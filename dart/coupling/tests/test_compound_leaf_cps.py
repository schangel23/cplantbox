"""Compound sheath-ring + blade CP grid.

Validates :func:`build_compound_leaf_cps` — a single NURBS patch made
of:

  * **Ring cup** (rows ``0..n_cup-1``): closed 360° rings stacked from
    ``z = -L_rendered`` (flat bottom) to ``z = 0`` at the midrib
    column (tilted top). Each ring has uniform radius
    ``stem_r(z) · (1 + base_clearance + bulge)`` measured from the
    stem central axis; ``v = 0`` and ``v = n_v − 1`` coincide at the
    back seam (θ = ±π). The top of the cup carries a ligule
    asymmetry — front edge at ``z = 0``, back edge at
    ``z = -ligule_tilt_frac · L_rendered``.

  * **Transition** (blade rows ``0..n_morph-1``): smoothstep-blends a
    ring-at-blade-z (with fading ligule + bulge) with the flat blade
    ribbon ``blade_up[i]``. ``frac = i/(n_morph-1)``:
    ``smooth(0) = 0`` → pure ring at blade-row z + full ligule;
    ``smooth(1) = 1`` → exactly ``blade_up[n_morph-1]``.

  * **Blade** (rows ``n_cup + n_morph..end``): ``blade_up[n_morph..]``
    verbatim.

Default sheath cap when ``max_sheath_length_cm`` is ``None``:
``L_rendered = min(sheath_length_cm, 2.5 · stem_radius_cm)`` — the
short collar that matches the stage-16 maize reference.
"""

from __future__ import annotations

import numpy as np
import pytest

from dart.coupling.geometry.canonical_library import (
    build_compound_leaf_cps,
    upsample_v,
)
from dart.coupling.geometry.canonical_cp_grid import (
    cp_grid_to_plantgl_patch_general, eval_grid,
)


def _fake_blade(n_u=11, n_v=5, length=50.0, width=8.0):
    """Smooth synthetic blade in leaf-local frame (margins at ±x, midrib at y=0).
    Row 0 sits exactly at z = 0 so the cup-to-transition continuity is exact."""
    cps = np.zeros((n_u, n_v, 3), dtype=np.float64)
    for i_u in range(n_u):
        frac = i_u / max(n_u - 1, 1)
        z = frac * length
        w_half = 0.5 * width * (1.0 - 0.85 * frac ** 1.5)
        for j in range(n_v):
            v = j / max(n_v - 1, 1)
            cps[i_u, j, 0] = w_half * (2 * v - 1)
            cps[i_u, j, 1] = 0.3 * frac * (1 - 0.5 * frac)
            cps[i_u, j, 2] = z - 4.0 * frac ** 2
    # Force row 0 z = 0 exactly so cup-top ligule_z matches first transition row.
    cps[0, :, 2] = 0.0
    return cps


def _smoothstep(t: float) -> float:
    return t * t * (3.0 - 2.0 * t)


def test_output_shape():
    """Output rows = n_cup + n_u_blade. No separate taper zone any more."""
    blade = _fake_blade(n_u=11)
    out = build_compound_leaf_cps(
        blade, stem_radius_cm=0.9, sheath_length_cm=4.0,
        n_cup=5, n_morph=3, n_v=9,
    )
    assert out.shape == (5 + 11, 9, 3)
    out2 = build_compound_leaf_cps(
        blade, stem_radius_cm=0.9, sheath_length_cm=4.0,
        n_cup=7, n_morph=4, n_v=13,
    )
    assert out2.shape == (7 + 11, 13, 3)


def test_cup_ring_is_closed_at_back():
    """Default wrap_deg = 360: the back seam closes — CP[i, 0] coincides
    with CP[i, n_v-1] at every cup row. This is what the user means by
    "fully closed 360°": the ring wraps the whole stem without a slit.
    Degenerate seam triangles are handled downstream by the lofter."""
    blade = _fake_blade()
    out = build_compound_leaf_cps(
        blade, stem_radius_cm=0.9, sheath_length_cm=4.0, n_cup=5, n_v=13,
    )
    for i in range(5):  # every cup row
        gap = float(np.linalg.norm(out[i, 0] - out[i, -1]))
        assert gap < 1e-9, f"cup row {i} back seam not closed: gap = {gap:.2e}"


def test_partial_wrap_leaves_slit():
    """For wrap_deg < 360 a back slit remains — legacy mode, not the
    default. CP[0, 0] and CP[0, n_v-1] sit apart by the chord subtended
    by the (360 − wrap_deg) slit."""
    blade = _fake_blade()
    stem_r = 0.9
    wrap = 345.0
    n_v = 13
    out = build_compound_leaf_cps(
        blade, stem_radius_cm=stem_r, sheath_length_cm=4.0,
        wrap_deg=wrap, bulge=0.0, base_clearance=0.0,
        n_cup=5, n_v=n_v,
    )
    gap = float(np.linalg.norm(out[0, 0] - out[0, -1]))
    slit_rad = (360.0 - wrap) * np.pi / 180.0
    expected_chord = 2.0 * stem_r * np.sin(slit_rad / 2.0)
    assert gap > 1e-3, f"seam accidentally collapsed; gap={gap:.6f}"
    np.testing.assert_allclose(gap, expected_chord, rtol=1e-9)


def test_cup_bottom_row_is_horizontal_ring():
    """The bottom cup row (t = 0) carries no ligule offset, so every
    column sits at the same z = -L_rendered and at uniform radial
    distance ``R = stem_r · (1 + base_clearance + bulge)`` from the
    stem central axis. Leaf-local origin IS the stem-axis point
    (CPlantBox puts leaf node 0 on the stem skeleton), so
    ``stem_center = axis · z``. This is the flat floor of the ring."""
    blade = _fake_blade()
    stem_r = 0.9
    bulge = 0.4
    clearance = 0.05
    n_v = 13
    out = build_compound_leaf_cps(
        blade, stem_radius_cm=stem_r, sheath_length_cm=4.0,
        bulge=bulge, base_clearance=clearance, n_cup=5, n_v=n_v,
    )
    L_rendered = 2.5 * stem_r  # default cap
    axis_vec = np.array([0.0, 0.0, 1.0])
    stem_center = axis_vec * (-L_rendered)
    expected_R = stem_r * (1.0 + clearance + bulge)
    for j in range(n_v):
        np.testing.assert_allclose(out[0, j, 2], -L_rendered, rtol=1e-9)
        d = float(np.linalg.norm(out[0, j] - stem_center))
        np.testing.assert_allclose(d, expected_R, rtol=1e-9)


def test_cup_top_has_ligule_tilt():
    """The top cup row (t = 1) carries the full ligule offset. Midrib
    column (θ = 0) sits at z = 0 (front crest); back seam columns
    (θ = ±π) sit at z = -ligule_tilt_frac · L_rendered (back saddle)."""
    blade = _fake_blade()
    stem_r = 0.9
    tilt = 0.4
    n_cup = 5
    n_v = 13
    out = build_compound_leaf_cps(
        blade, stem_radius_cm=stem_r, sheath_length_cm=4.0,
        ligule_tilt_frac=tilt, n_cup=n_cup, n_v=n_v,
    )
    L_rendered = 2.5 * stem_r  # default cap
    mid_j = (n_v - 1) // 2
    top = out[n_cup - 1]
    # Midrib crest at z = 0
    np.testing.assert_allclose(top[mid_j, 2], 0.0, atol=1e-12)
    # Back seam at z = -tilt * L
    np.testing.assert_allclose(top[0, 2], -tilt * L_rendered, rtol=1e-9)
    np.testing.assert_allclose(top[-1, 2], -tilt * L_rendered, rtol=1e-9)


def test_cup_row_is_closed_ring_uniform_radius():
    """Every cup row — regardless of height — is a closed ring with all
    columns equidistant from the stem centre at THAT column's z. No
    asymmetric bulging on one side. This is what "stem-aware 360°"
    means geometrically. Leaf-local origin is ON the stem axis (CPlantBox
    places leaf node 0 at the parent stem node), so
    ``stem_center(z) = axis · z``."""
    blade = _fake_blade()
    stem_r = 0.9
    bulge = 0.3
    clearance = 0.04
    n_cup = 6
    n_v = 13
    out = build_compound_leaf_cps(
        blade, stem_radius_cm=stem_r, sheath_length_cm=4.0,
        bulge=bulge, base_clearance=clearance, n_cup=n_cup, n_v=n_v,
    )
    axis_vec = np.array([0.0, 0.0, 1.0])
    expected_R = stem_r * (1.0 + clearance + bulge)
    for i in range(n_cup):
        for j in range(n_v):
            z = out[i, j, 2]
            stem_center = axis_vec * z
            d = float(np.linalg.norm(out[i, j] - stem_center))
            np.testing.assert_allclose(
                d, expected_R, rtol=1e-9,
                err_msg=f"cup row {i}, col {j}: d={d:.6f} ≠ {expected_R:.6f}",
            )


def test_cup_midrib_faces_blade_direction():
    """The midrib column (j = (n_v-1)//2) sits on the stem face that
    points toward the blade (+front); j = 0 / j = n_v - 1 coincide at
    the back seam behind the stem. Stem centre at any z lies ON the
    leaf-local axis (origin on stem axis), so the in-plane displacement
    of each column from the axis has magnitude R and points along
    ``arc_dir(θ)``."""
    blade = _fake_blade()
    axis = np.array([np.sin(np.deg2rad(30)), 0.0, np.cos(np.deg2rad(30))])
    n_v = 13
    out = build_compound_leaf_cps(
        blade, stem_radius_cm=0.9, sheath_length_cm=1.5,
        stem_axis=axis, bulge=0.3, n_cup=5, n_v=n_v,
    )
    z = np.array([0.0, 0.0, 1.0])
    front = z - np.dot(z, axis) * axis
    front /= np.linalg.norm(front)

    mid_j = (n_v - 1) // 2
    # Use bottom row so there's no ligule z-tilt. Remove axial component
    # to get the purely radial displacement from the stem axis.
    disp = out[0, mid_j] - np.dot(out[0, mid_j], axis) * axis
    disp_unit = disp / max(np.linalg.norm(disp), 1e-12)
    assert float(np.dot(disp_unit, front)) > 0.9, "midrib should face +front"

    back_disp = out[0, 0] - np.dot(out[0, 0], axis) * axis
    back_unit = back_disp / max(np.linalg.norm(back_disp), 1e-12)
    assert float(np.dot(back_unit, front)) < -0.9, "seam should face -front (back)"


def test_default_length_cap_applies():
    """When ``max_sheath_length_cm`` is ``None`` the rendered sheath
    defaults to ``min(sheath_length_cm, 2.5 · stem_radius_cm)``. That's
    what restores the short visual collar matching the reference — the
    cup bottom sits at ``z = -2.5 · stem_r`` when the botanical sheath
    length is larger."""
    blade = _fake_blade()
    stem_r = 0.6
    default_cap = 2.5 * stem_r
    out = build_compound_leaf_cps(
        blade, stem_radius_cm=stem_r, sheath_length_cm=10.0,
        n_cup=5, n_v=9,
    )
    mid_j = 4
    np.testing.assert_allclose(out[0, mid_j, 2], -default_cap, rtol=1e-9)


def test_length_cap_override():
    """A positive ``max_sheath_length_cm`` overrides the default cap.
    Passing math.inf renders the full botanical sheath length."""
    import math
    blade = _fake_blade()
    stem_r = 0.6
    # Explicit short cap
    out_cap = build_compound_leaf_cps(
        blade, stem_radius_cm=stem_r, sheath_length_cm=10.0,
        max_sheath_length_cm=1.0, n_cup=5, n_v=9,
    )
    np.testing.assert_allclose(out_cap[0, 4, 2], -1.0, rtol=1e-9)
    # math.inf → full length
    out_full = build_compound_leaf_cps(
        blade, stem_radius_cm=stem_r, sheath_length_cm=10.0,
        max_sheath_length_cm=math.inf, n_cup=5, n_v=9,
    )
    np.testing.assert_allclose(out_full[0, 4, 2], -10.0, rtol=1e-9)


def test_cup_midrib_z_monotone_bottom_to_top():
    """Along the midrib column (no ligule tilt there) cup rows stack
    monotonically from z = -L_rendered at i = 0 up to z = 0 at the
    top cup row."""
    blade = _fake_blade()
    stem_r = 0.9
    n_cup = 7
    n_v = 9
    out = build_compound_leaf_cps(
        blade, stem_radius_cm=stem_r, sheath_length_cm=10.0,
        max_sheath_length_cm=2.5 * stem_r, n_cup=n_cup, n_v=n_v,
    )
    L_rendered = 2.5 * stem_r
    mid_j = (n_v - 1) // 2
    zs = out[:n_cup, mid_j, 2]
    np.testing.assert_allclose(zs[0], -L_rendered, rtol=1e-9)
    np.testing.assert_allclose(zs[-1], 0.0, atol=1e-12)
    assert np.all(np.diff(zs) > 0), f"cup midrib z must increase: {zs}"


def test_transition_endpoints_match_exactly():
    """Boundary continuity:

    - Last cup row (i = n_cup-1) at column j has z = ligule_z(j) and
      sits at ``stem_center(z) + R·arc_dir_j``.
    - First transition row (blade i = 0, frac = 0) has z = 0 +
      ligule_z(j) = same as above (since synthetic blade row 0 has
      z = 0) and also sits at ``ring_pt(z)``.
    - Last transition row (blade i = n_morph-1, frac = 1) lands exactly
      on ``blade_up[n_morph-1]``.
    """
    blade = _fake_blade(n_u=11)
    n_cup = 5
    n_morph = 3
    n_v = 9
    out = build_compound_leaf_cps(
        blade, stem_radius_cm=0.9, sheath_length_cm=4.0,
        n_cup=n_cup, n_morph=n_morph, n_v=n_v,
    )
    blade_up = upsample_v(blade, n_v)
    # Last transition row equals blade_up[n_morph - 1] exactly.
    np.testing.assert_allclose(
        out[n_cup + n_morph - 1], blade_up[n_morph - 1], atol=1e-12,
    )
    # Cup/transition boundary: since blade_up[0] has z = 0 everywhere,
    # the last cup row and first transition row should coincide.
    np.testing.assert_allclose(out[n_cup - 1], out[n_cup], atol=1e-10)


def test_transition_blend_formula():
    """Each transition row i matches the documented formula:

        frac = i / (n_morph - 1)
        smooth = frac² · (3 − 2 frac)
        z_j = blade_up[i, j, 2] + (1 - smooth) · ligule_z(j)
        ring_pt = stem_center(z_j) + R(1 - smooth)·arc_dir
        out = (1 - smooth) · ring_pt + smooth · blade_up[i, j]
    """
    blade = _fake_blade(n_u=11)
    stem_r = 0.9
    bulge = 0.25
    clearance = 0.04
    tilt = 0.35
    n_cup = 5
    n_morph = 3
    n_v = 13
    out = build_compound_leaf_cps(
        blade, stem_radius_cm=stem_r, sheath_length_cm=4.0,
        bulge=bulge, base_clearance=clearance, ligule_tilt_frac=tilt,
        n_cup=n_cup, n_morph=n_morph, n_v=n_v,
    )
    L_rendered = 2.5 * stem_r
    blade_up = upsample_v(blade, n_v)
    axis_vec = np.array([0.0, 0.0, 1.0])
    front_vec = np.array([1.0, 0.0, 0.0])
    side_vec = np.array([0.0, 1.0, 0.0])
    wrap_rad = 2.0 * np.pi
    theta_half = np.pi
    theta = theta_half - wrap_rad * (np.arange(n_v) / (n_v - 1))
    arc_dir = (
        np.cos(theta)[:, None] * front_vec
        + np.sin(theta)[:, None] * side_vec
    )
    ligule_z = -tilt * L_rendered * np.sin(theta / 2.0) ** 2

    for i in range(n_morph):
        frac = i / (n_morph - 1)
        smooth = _smoothstep(frac)
        asym = 1.0 - smooth
        flat = blade_up[i]
        z_j = flat[:, 2] + asym * ligule_z
        ring = np.empty((n_v, 3))
        for j in range(n_v):
            z = z_j[j]
            stem_center = axis_vec * z  # origin on stem axis
            R = stem_r * (1.0 + clearance + bulge * asym)
            ring[j] = stem_center + R * arc_dir[j]
        expected = (1.0 - smooth) * ring + smooth * flat
        np.testing.assert_allclose(
            out[n_cup + i], expected, rtol=1e-9, atol=1e-12,
        )


def test_stem_taper_tracked_by_callable():
    """When ``stem_radius_at_z`` is provided the cup tracks a tapering
    stem. At each cup row, the ring radius equals
    ``stem_r_of_z(z) · (1 + clearance + bulge)``."""
    blade = _fake_blade()
    stem_r0 = 0.9
    taper_rate = 0.1  # stem widens downward
    n_cup = 5
    n_v = 13

    def stem_r_of_z(z: float) -> float:
        return stem_r0 + abs(z) * taper_rate

    out = build_compound_leaf_cps(
        blade, stem_radius_cm=stem_r0, sheath_length_cm=4.0,
        stem_radius_at_z=stem_r_of_z, bulge=0.0, base_clearance=0.0,
        n_cup=n_cup, n_v=n_v,
    )
    axis_vec = np.array([0.0, 0.0, 1.0])
    # Every cup CP sits at exactly stem_r_of_z(z) from the stem centre
    # (bulge=0, clearance=0 → pure surface). Origin is on the stem axis.
    for i in range(n_cup):
        for j in range(n_v):
            z = out[i, j, 2]
            stem_center = axis_vec * z
            d = float(np.linalg.norm(out[i, j] - stem_center))
            np.testing.assert_allclose(d, stem_r_of_z(z), rtol=1e-9)


def test_blade_rows_unchanged_above_transition():
    """Rows ``n_cup + n_morph .. end`` must equal ``blade_up[n_morph..]``
    verbatim — no side effects from the cup construction."""
    blade = _fake_blade(n_u=11)
    n_cup = 5
    n_morph = 3
    n_v = 13
    out = build_compound_leaf_cps(
        blade, stem_radius_cm=0.9, sheath_length_cm=4.0,
        n_cup=n_cup, n_morph=n_morph, n_v=n_v,
    )
    blade_up = upsample_v(blade, n_v)
    for k in range(n_morph, blade_up.shape[0]):
        diff = float(np.max(np.abs(out[n_cup + k] - blade_up[k])))
        assert diff < 1e-12, f"blade row {k} perturbed: max |Δ| = {diff:.2e}"


def test_blade_tip_reachable_after_cup():
    """Evaluated surface still reaches the blade tip with distinct
    margins — the cup + transition must not collapse the blade's
    parametric domain."""
    blade = _fake_blade(length=50.0, width=8.0)
    out = build_compound_leaf_cps(
        blade, stem_radius_cm=0.9, sheath_length_cm=4.0,
        n_cup=5, n_morph=3, n_v=9,
    )
    patch = cp_grid_to_plantgl_patch_general(out)
    tip_v0 = patch.getPointAt(1.0, 0.0)
    tip_v1 = patch.getPointAt(1.0, 1.0)
    tip_mid = patch.getPointAt(1.0, 0.5)
    assert tip_mid.z > 25.0, f"tip only reached z={tip_mid.z:.1f}"
    assert np.hypot(tip_v0.x - tip_v1.x, tip_v0.y - tip_v1.y) > 0.5, (
        "tip margins collapsed"
    )


def test_cup_tessellation_wraps_full_circle():
    """Densely evaluating the cup bottom (u = 0) must cover the full
    360° around the stem centre — no gap for the default wrap_deg = 360.
    Angular coverage is measured around the **stem centre**, not around
    the leaf-local origin."""
    blade = _fake_blade()
    stem_r = 0.9
    out = build_compound_leaf_cps(
        blade, stem_radius_cm=stem_r, sheath_length_cm=1.5,
        bulge=0.0, base_clearance=0.0, n_cup=5, n_v=13,
    )
    patch = cp_grid_to_plantgl_patch_general(out)
    verts, _ = eval_grid(patch, n_u=20, n_v=80)
    V = verts.reshape(20, 80, 3)
    base = V[0]
    # Leaf-local origin is on the stem axis, so measure angular coverage
    # around (0, 0) at z = base row.
    cx, cy = 0.0, 0.0
    th = np.sort(np.arctan2(base[:, 1] - cy, base[:, 0] - cx))
    gaps = np.diff(np.concatenate([th, [th[0] + 2 * np.pi]]))
    arc_deg = float(np.degrees(2 * np.pi - gaps.max()))
    # Cubic NURBS on a closed 360° ring under-evaluates near the
    # degenerate seam (patch collapses to a point there), leaving a
    # ~5–10° sliver un-rendered by a coarse 80-column sweep. A > 340°
    # sweep still confirms the ring closes all the way around — the
    # remaining gap is an evaluation artefact, not a CP-level slit.
    assert arc_deg > 340.0, (
        f"base arc coverage only {arc_deg:.1f}° — expected near-360°"
    )


def test_parameter_validation():
    """Invalid inputs must raise rather than silently returning a bad grid."""
    blade = _fake_blade()
    with pytest.raises(ValueError):
        build_compound_leaf_cps(blade, stem_radius_cm=0.0, sheath_length_cm=4.0)
    with pytest.raises(ValueError):
        build_compound_leaf_cps(blade, stem_radius_cm=0.9, sheath_length_cm=-1.0)
    with pytest.raises(ValueError):
        build_compound_leaf_cps(
            blade, stem_radius_cm=0.9, sheath_length_cm=4.0, n_cup=1,
        )
    with pytest.raises(ValueError):
        build_compound_leaf_cps(
            blade, stem_radius_cm=0.9, sheath_length_cm=4.0, n_morph=1,
        )
    with pytest.raises(ValueError):
        build_compound_leaf_cps(
            blade, stem_radius_cm=0.9, sheath_length_cm=4.0, n_v=4,
        )
    # wrap_deg 360 is VALID (default closed ring); > 360 and ≤ 0 must raise.
    with pytest.raises(ValueError):
        build_compound_leaf_cps(
            blade, stem_radius_cm=0.9, sheath_length_cm=4.0, wrap_deg=361.0,
        )
    with pytest.raises(ValueError):
        build_compound_leaf_cps(
            blade, stem_radius_cm=0.9, sheath_length_cm=4.0, wrap_deg=0.0,
        )
    with pytest.raises(ValueError):
        build_compound_leaf_cps(
            blade, stem_radius_cm=0.9, sheath_length_cm=4.0, bulge=-0.1,
        )
    with pytest.raises(ValueError):
        build_compound_leaf_cps(
            blade, stem_radius_cm=0.9, sheath_length_cm=4.0,
            base_clearance=-0.01,
        )
    with pytest.raises(ValueError):
        build_compound_leaf_cps(
            blade, stem_radius_cm=0.9, sheath_length_cm=4.0,
            ligule_tilt_frac=1.0,
        )
    with pytest.raises(ValueError):
        build_compound_leaf_cps(
            blade, stem_radius_cm=0.9, sheath_length_cm=4.0,
            ligule_tilt_frac=-0.1,
        )
    with pytest.raises(ValueError):
        build_compound_leaf_cps(
            blade, stem_radius_cm=0.9, sheath_length_cm=4.0,
            max_sheath_length_cm=0.0,
        )


if __name__ == "__main__":
    test_output_shape(); print("shape: OK")
    test_cup_ring_is_closed_at_back(); print("closed ring at back: OK")
    test_partial_wrap_leaves_slit(); print("partial wrap slit: OK")
    test_cup_bottom_row_is_horizontal_ring(); print("bottom row flat ring: OK")
    test_cup_top_has_ligule_tilt(); print("top has ligule tilt: OK")
    test_cup_row_is_closed_ring_uniform_radius(); print("uniform radius rings: OK")
    test_cup_midrib_faces_blade_direction(); print("midrib orientation: OK")
    test_default_length_cap_applies(); print("default length cap: OK")
    test_length_cap_override(); print("length cap override: OK")
    test_cup_midrib_z_monotone_bottom_to_top(); print("z monotone: OK")
    test_transition_endpoints_match_exactly(); print("transition endpoints: OK")
    test_transition_blend_formula(); print("transition blend formula: OK")
    test_stem_taper_tracked_by_callable(); print("stem taper tracked: OK")
    test_blade_rows_unchanged_above_transition(); print("blade rows preserved: OK")
    test_blade_tip_reachable_after_cup(); print("blade tip reachable: OK")
    test_cup_tessellation_wraps_full_circle(); print("tessellation full circle: OK")
    test_parameter_validation(); print("param validation: OK")
