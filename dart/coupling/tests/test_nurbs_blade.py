"""Unit tests for the canonical NurbsPatch leaf lofter (Phase 1).

Covers:
  1. Output shape contract (9-tuple: 8-tuple mesh data + canonical CP grid).
  2. Vertex / triangle / quad counts are deterministic from tessellation.
  3. Per-triangle ``segment_ids`` map back into the original skeleton range.
  4. Organ IDs propagate correctly.
  5. ``use_nurbs_backend`` dispatch via ``loft_organs`` returns a valid
     ``G3Mesh`` and is selectable per-organ.
  6. Flat-ribbon sanity: a flat skeleton + constant width produces a mesh
     that is within the width bounds and sits on the horizontal plane.
  7. Canonical CP exposure: ``(N_U=11, N_V=5, 3)`` grid is returned and
     round-trips through ``cp_grid_to_plantgl_patch`` + ``eval_grid`` to the
     same tessellated vertices.
  8. ``G3Mesh.organ_cps`` round-trips the per-organ CP grid through
     ``loft_organs``.
"""
from __future__ import annotations

import numpy as np

from dart.coupling.geometry.canonical_cp_grid import (
    N_U, N_V, cp_grid_to_plantgl_patch, eval_grid,
)
from dart.coupling.geometry.g1_to_g3 import loft_organs
from dart.coupling.geometry.nurbs_blade import loft_leaf_nurbs


def _synthetic_leaf(organ_id: int = 7, n_skel: int = 15,
                    deformed: bool = False) -> dict:
    t = np.linspace(0.0, 1.0, n_skel)
    skel = np.column_stack([
        25.0 * t,
        0.4 * np.sin(np.pi * t),
        30.0 - 8.0 * t + 2.0 * t * t,
    ])
    widths = np.maximum(3.0 * (1 - np.abs(t - 0.5) * 1.2) * (1 - 0.2 * t), 0.2)
    organ = {
        "type": "leaf",
        "organ_id": organ_id,
        "skeleton": skel,
        "widths": widths,
        "_orig_segment_map": np.arange(n_skel - 1, dtype=np.int32),
    }
    if deformed:
        organ.update(
            wave_normal_amp=0.4,
            wave_normal_freq=3.0,
            twist_max=0.3,
            curl_amp=0.25,
            gutter_depths=np.full(n_skel, 0.15),
        )
    return organ


# ---------------------------------------------------------------------------
# 1. Output contract
# ---------------------------------------------------------------------------
def test_output_shape_and_dtype():
    organ = _synthetic_leaf()
    result = loft_leaf_nurbs(organ)
    assert len(result) == 10
    verts, idxs, norms, uvs, oids, sids, qidxs, qoids, cps, is_midrib = result
    assert verts.shape == (30 * 7, 3) and verts.dtype == np.float64
    assert idxs.shape == (2 * 29 * 6, 3) and idxs.dtype == np.int32
    assert norms.shape == (30 * 7, 3)
    assert uvs.shape == (30 * 7, 2)
    assert oids.shape == (2 * 29 * 6,) and oids.dtype == np.int32
    assert sids.shape == (2 * 29 * 6,) and sids.dtype == np.int32
    assert qidxs is not None
    assert qidxs.shape == (29 * 6, 4) and qidxs.dtype == np.int32
    assert qoids is not None
    assert qoids.shape == (29 * 6,) and qoids.dtype == np.int32
    assert cps.shape == (N_U, N_V, 3) and cps.dtype == np.float64


def test_tessellation_resolution_is_configurable():
    organ = _synthetic_leaf()
    result = loft_leaf_nurbs(organ, n_u_eval=20, n_v_eval=5)
    verts, idxs = result[0], result[1]
    qidxs = result[6]
    cps = result[8]
    assert verts.shape == (20 * 5, 3)
    assert idxs.shape == (2 * 19 * 4, 3)
    assert qidxs is not None and qidxs.shape == (19 * 4, 4)
    # CP grid shape is fixed by canonical_cp_grid regardless of tessellation.
    assert cps.shape == (N_U, N_V, 3)


# ---------------------------------------------------------------------------
# 2. Segment ID tracking
# ---------------------------------------------------------------------------
def test_segment_ids_span_skeleton_range():
    organ = _synthetic_leaf(n_skel=15)
    sids = loft_leaf_nurbs(organ)[5]
    unique = np.unique(sids)
    # The segment map covers 14 original segments (0..13). Every triangle must
    # point to a valid original segment.
    assert unique.min() >= 0
    assert unique.max() < 14
    # With 30 tessellation rows mapping into 14 segments, most segments should
    # receive at least one triangle.
    assert len(unique) >= 10


# ---------------------------------------------------------------------------
# 3. Organ IDs
# ---------------------------------------------------------------------------
def test_organ_ids_propagate():
    organ = _synthetic_leaf(organ_id=99)
    result = loft_leaf_nurbs(organ)
    oids, qoids = result[4], result[7]
    assert np.all(oids == 99)
    assert qoids is not None and np.all(qoids == 99)


# ---------------------------------------------------------------------------
# 4. Deformations move vertices
# ---------------------------------------------------------------------------
def test_deformations_change_geometry():
    plain = _synthetic_leaf(deformed=False)
    curly = _synthetic_leaf(deformed=True)
    v_plain = loft_leaf_nurbs(plain)[0]
    v_curly = loft_leaf_nurbs(curly)[0]
    # Same CP count and corner anchors, so corners should roughly match, but
    # interior vertices must differ substantially when deformations are on.
    diff = np.linalg.norm(v_plain - v_curly, axis=1)
    assert diff.max() > 0.2, (
        f"deformations produced no visible change (max diff {diff.max():.3f} cm)"
    )


# ---------------------------------------------------------------------------
# 5. Dispatch through loft_organs
# ---------------------------------------------------------------------------
def test_loft_organs_dispatches_nurbs_backend():
    organs = [_synthetic_leaf(organ_id=1), _synthetic_leaf(organ_id=2)]

    mesh_legacy = loft_organs(organs, use_nurbs_backend=False, smooth=False)
    mesh_nurbs = loft_organs(organs, use_nurbs_backend=True, smooth=False)

    assert mesh_legacy.n_triangles > 0
    assert mesh_nurbs.n_triangles > 0
    # Both backends produce geometry of comparable bbox extent.
    legacy_span = np.ptp(mesh_legacy.vertices, axis=0)
    nurbs_span = np.ptp(mesh_nurbs.vertices, axis=0)
    assert np.abs(legacy_span - nurbs_span).max() < 10.0


def test_per_organ_override():
    organs = [
        dict(_synthetic_leaf(organ_id=1), use_nurbs_backend=True),
        _synthetic_leaf(organ_id=2),
    ]
    # Global flag OFF, but organ 1 opts in.
    mesh = loft_organs(organs, use_nurbs_backend=False, smooth=False)
    n_org1 = int(np.sum(mesh.organ_ids == 1))
    n_org2 = int(np.sum(mesh.organ_ids == 2))
    assert n_org1 > 0 and n_org2 > 0
    # Organ 1 produces exactly the NURBS triangle count (2 * (n_u-1) * (n_v-1)).
    # Defaults: nurbs_n_u_eval=30, nurbs_n_v_eval=21 → 2 * 29 * 20 = 1160.
    assert n_org1 == 2 * 29 * 20


# ---------------------------------------------------------------------------
# 6. Flat-ribbon sanity
# ---------------------------------------------------------------------------
def test_flat_horizontal_ribbon():
    """A straight horizontal skeleton with constant width should produce a
    mesh whose z range is tiny (< 0.05 cm — just floating-point noise) and
    whose y range is within ±half-width."""
    n = 15
    t = np.linspace(0, 1, n)
    skel = np.column_stack([20.0 * t, np.zeros(n), np.full(n, 30.0)])
    widths = np.full(n, 2.0)
    organ = {
        "type": "leaf", "organ_id": 5,
        "skeleton": skel, "widths": widths,
        "_orig_segment_map": np.arange(n - 1, dtype=np.int32),
    }
    verts, *_ = loft_leaf_nurbs(organ)

    # z should be flat (within floating-point tolerance) — flat ribbon on the
    # z=30 plane.
    assert abs(verts[:, 2].max() - verts[:, 2].min()) < 0.05

    # y extent: full-width 2 cm → ±1 cm about y=0
    assert verts[:, 1].min() >= -1.01
    assert verts[:, 1].max() <= 1.01

    # x extent: ~20 cm along the leaf
    assert 19.0 < (verts[:, 0].max() - verts[:, 0].min()) < 21.0


# ---------------------------------------------------------------------------
# 7. Canonical CP exposure + round-trip
# ---------------------------------------------------------------------------
def test_cps_round_trip_through_patch():
    """The returned CP grid, re-fed through the canonical PlantGL patch
    builder and ``eval_grid``, must reproduce the tessellated vertices we
    got directly from ``loft_leaf_nurbs``. This is the load-bearing
    guarantee the fitter relies on: ``cps`` is the ground truth that gave
    those vertices, not some stale pre-deformation copy.
    """
    organ = _synthetic_leaf(deformed=True)
    verts, *_, cps, _is_midrib = loft_leaf_nurbs(organ, n_u_eval=30, n_v_eval=7)
    # Shape & finiteness.
    assert cps.shape == (N_U, N_V, 3)
    assert np.all(np.isfinite(cps))
    # Round-trip: rebuild the patch and re-evaluate at the same grid.
    patch = cp_grid_to_plantgl_patch(cps)
    verts_rt, _ = eval_grid(patch, n_u=30, n_v=7)
    np.testing.assert_allclose(verts_rt, verts, atol=1e-6, rtol=0)


def test_loft_organs_exposes_organ_cps():
    """``loft_organs(..., use_nurbs_backend=True)`` must populate the
    ``G3Mesh.organ_cps`` dict keyed by each organ's id, and the stored
    array must match the array returned directly from ``loft_leaf_nurbs``
    for the same organ (modulo float copies)."""
    organ = _synthetic_leaf(organ_id=42, deformed=True)
    # Direct call to the lofter (without subdivision / smoothing).
    direct_cps = loft_leaf_nurbs(organ)[8]
    # Same organ through loft_organs with NURBS backend + smoothing off.
    mesh = loft_organs([organ], use_nurbs_backend=True,
                       subdivide=False, smooth=False)
    assert 42 in mesh.organ_cps
    stored = mesh.organ_cps[42]
    assert stored.shape == (N_U, N_V, 3)
    np.testing.assert_allclose(stored, direct_cps, atol=1e-12, rtol=0)


def test_loft_organs_quad_ribbon_skips_organ_cps():
    """Quad-ribbon leaves (``use_nurbs_backend=False``) must NOT appear in
    ``organ_cps``. The fitter relies on ``.get(oid)`` returning ``None`` to
    fall back to the Chamfer path."""
    organ = _synthetic_leaf(organ_id=7)
    mesh = loft_organs([organ], use_nurbs_backend=False, smooth=False)
    assert 7 not in mesh.organ_cps
    assert mesh.organ_cps == {}
