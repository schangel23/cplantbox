#!/usr/bin/env python3
"""
Step 5 — Sheath Mesher Tests

Tests for sheath mesh generation (geometry/sheath_mesher.py) and
pipeline integration (sheath organs flowing through cplantbox_adapter
and g1_to_g3).

Usage:
  cd /home/lukas/PHD/CPlantBox
  cpbenv/bin/python3 -m pytest dart/coupling/tests/test_sheath_mesher.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Add coupling package to path
COUPLING_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(COUPLING_DIR.parent.parent))

from dart.coupling.geometry.sheath_mesher import (
    mesh_sheath,
    _build_arc_cross_section,
    _triangulate_sweep,
)
from dart.coupling.geometry.g1_to_g3 import G3Mesh, loft_organs


# ─── Fixtures ───────────────────────────────────────────────────────────

def _make_straight_sheath(n_pts=20, base_radius=0.8, collar_radius=0.4,
                          length=8.0):
    """Create a simple straight vertical sheath for testing."""
    skeleton = np.zeros((n_pts, 3))
    skeleton[:, 2] = np.linspace(5.0, 5.0 + length, n_pts)
    radii = np.linspace(base_radius, collar_radius, n_pts)
    return skeleton, radii


def _make_curved_sheath(n_pts=30, base_radius=0.7, collar_radius=0.3,
                        length=10.0):
    """Create a curved sheath (slight arc) for testing."""
    t = np.linspace(0, 1, n_pts)
    skeleton = np.zeros((n_pts, 3))
    skeleton[:, 0] = 0.5 * np.sin(np.pi * t * 0.3)  # slight lateral curve
    skeleton[:, 2] = 5.0 + t * length
    radii = np.linspace(base_radius, collar_radius, n_pts)
    return skeleton, radii


def _make_stem_skeleton():
    """Create a simple stem skeleton for frame alignment tests."""
    n = 50
    skeleton = np.zeros((n, 3))
    skeleton[:, 2] = np.linspace(0.0, 30.0, n)
    return skeleton


def _make_sheath_organ(organ_id=0, **kwargs):
    """Create a sheath organ dict suitable for loft_organs()."""
    skeleton, radii = _make_straight_sheath(**kwargs)
    return {
        "type": "sheath",
        "part_type": "sheath",
        "skeleton": skeleton,
        "widths": radii * 2.0,
        "radii": radii,
        "wrap_angle": np.radians(330),
        "overlap_angle": np.radians(30),
        "sheath_thickness": 0.04,
        "stem_skeleton": _make_stem_skeleton(),
        "organ_id": organ_id,
        "name": f"sheath_{organ_id}",
        "node_ids": list(range(len(skeleton))),
    }


# ─── TestSheathGeometry ────────────────────────────────────────────────

class TestSheathGeometry:
    """Unit tests for mesh_sheath() output geometry."""

    def test_valid_mesh(self):
        """Vertices are (N,3), indices are (K,3), no NaN/Inf."""
        skeleton, radii = _make_straight_sheath()
        verts, idxs, norms, uvs, oids, sids = mesh_sheath(skeleton, radii)

        assert verts.shape[1] == 3
        assert idxs.shape[1] == 3
        assert norms.shape[1] == 3
        assert uvs.shape[1] == 2
        assert not np.any(np.isnan(verts))
        assert not np.any(np.isinf(verts))
        assert not np.any(np.isnan(norms))

    def test_vertex_count(self):
        """Vertex count matches n_axis * n_cross."""
        skeleton, radii = _make_straight_sheath(n_pts=15)
        n_arc = 24
        verts, *_ = mesh_sheath(skeleton, radii, n_arc=n_arc)

        # n_overlap = max(3, int(24 * 0.52 / 5.76)) = max(3, 2) = 3
        n_overlap = 3
        n_cross = n_arc + n_overlap
        expected_verts = 15 * n_cross
        assert len(verts) == expected_verts

    def test_triangle_count(self):
        """Triangle count is 2 * (n_axis-1) * (n_cross-1)."""
        n_pts = 15
        n_arc = 24
        skeleton, radii = _make_straight_sheath(n_pts=n_pts)
        _, idxs, *_ = mesh_sheath(skeleton, radii, n_arc=n_arc)

        n_overlap = 3
        n_cross = n_arc + n_overlap
        expected_tris = 2 * (n_pts - 1) * (n_cross - 1)
        assert len(idxs) == expected_tris

    def test_normals_outward(self):
        """Normals should point outward (dot(normal, center_to_vertex) > 0)."""
        skeleton, radii = _make_straight_sheath()
        verts, idxs, norms, *_ = mesh_sheath(skeleton, radii)

        # Check a sample of vertices: normal should roughly align with
        # direction from skeleton center to vertex
        n_cross = len(verts) // len(skeleton)
        for i in range(0, len(skeleton), 5):
            center = skeleton[i]
            for j in range(0, n_cross, 6):
                idx = i * n_cross + j
                if idx >= len(verts):
                    break
                to_vertex = verts[idx] - center
                dot = np.dot(norms[idx], to_vertex)
                assert dot > -1e-6, (
                    f"Normal at vertex {idx} points inward: dot={dot:.4f}"
                )

    def test_wrap_angle_span(self):
        """Vertices in each ring span approximately the wrap angle."""
        skeleton, radii = _make_straight_sheath(n_pts=10)
        wrap_deg = 330
        verts, *_ = mesh_sheath(
            skeleton, radii,
            wrap_angle=np.radians(wrap_deg),
            overlap_angle=np.radians(0),  # no overlap for clean test
            n_arc=36,
        )

        n_cross = len(verts) // 10
        # Check middle ring
        ring_start = 5 * n_cross
        ring_verts = verts[ring_start:ring_start + n_cross]
        center = skeleton[5]

        # Compute angles from center to each vertex
        dirs = ring_verts - center
        # Project onto plane perpendicular to tangent (roughly XY at skeleton[5])
        angles = np.arctan2(dirs[:, 1], dirs[:, 0])
        angles = np.unwrap(angles)
        angular_span = abs(angles[-1] - angles[0])

        # Should be close to 330 degrees = 5.76 rad
        assert abs(angular_span - np.radians(wrap_deg)) < 0.5, (
            f"Angular span {np.degrees(angular_span):.1f}deg, expected ~{wrap_deg}deg"
        )

    def test_taper(self):
        """Base ring should be wider than collar ring."""
        skeleton, radii = _make_straight_sheath(
            base_radius=1.0, collar_radius=0.3
        )
        verts, *_ = mesh_sheath(skeleton, radii)

        n_cross = len(verts) // len(skeleton)
        # Base ring: first n_cross vertices
        base_ring = verts[:n_cross]
        base_spread = np.max(np.linalg.norm(
            base_ring - skeleton[0], axis=1
        ))

        # Collar ring: last n_cross vertices
        collar_ring = verts[-n_cross:]
        collar_spread = np.max(np.linalg.norm(
            collar_ring - skeleton[-1], axis=1
        ))

        assert base_spread > collar_spread * 1.5, (
            f"Base spread {base_spread:.3f} should be much wider than "
            f"collar spread {collar_spread:.3f}"
        )

    def test_overlap_offset(self):
        """Overlap vertices should be at radius + thickness."""
        skeleton, radii = _make_straight_sheath(n_pts=10)
        thickness = 0.1  # exaggerated for test
        n_arc = 12
        verts, *_ = mesh_sheath(
            skeleton, radii,
            thickness=thickness,
            n_arc=n_arc,
        )

        # n_overlap = max(3, int(12 * 0.52/5.76)) = 3
        n_cross = n_arc + 3

        # Check middle ring: overlap vertices should be farther from center
        ring_idx = 5
        center = skeleton[ring_idx]
        expected_radius = radii[ring_idx]

        main_dists = []
        overlap_dists = []
        for j in range(n_cross):
            idx = ring_idx * n_cross + j
            d = np.linalg.norm(verts[idx] - center)
            if j < n_arc:
                main_dists.append(d)
            else:
                overlap_dists.append(d)

        mean_main = np.mean(main_dists)
        mean_overlap = np.mean(overlap_dists)

        assert abs(mean_main - expected_radius) < 0.05, (
            f"Main arc radius {mean_main:.3f}, expected {expected_radius:.3f}"
        )
        assert mean_overlap > mean_main, (
            f"Overlap radius {mean_overlap:.3f} should exceed main {mean_main:.3f}"
        )
        assert abs(mean_overlap - mean_main - thickness) < 0.05, (
            f"Overlap offset {mean_overlap - mean_main:.3f}, expected {thickness}"
        )

    def test_uv_range(self):
        """UV coordinates should be in [0, 1]."""
        skeleton, radii = _make_straight_sheath()
        _, _, _, uvs, *_ = mesh_sheath(skeleton, radii)

        assert np.all(uvs >= -1e-6), f"UV min: {uvs.min():.6f}"
        assert np.all(uvs <= 1.0 + 1e-6), f"UV max: {uvs.max():.6f}"


class TestSheathEdgeCases:
    """Edge cases and degenerate inputs."""

    def test_short_sheath_skipped(self):
        """Sheaths shorter than 0.5 cm produce empty mesh."""
        skeleton = np.array([[0, 0, 0], [0, 0, 0.3]])  # 0.3 cm
        radii = np.array([0.5, 0.5])
        verts, idxs, *_ = mesh_sheath(skeleton, radii)
        assert len(verts) == 0
        assert len(idxs) == 0

    def test_single_point_sheath(self):
        """Single-point skeleton produces empty mesh."""
        skeleton = np.array([[0, 0, 5.0]])
        radii = np.array([0.5])
        verts, idxs, *_ = mesh_sheath(skeleton, radii)
        assert len(verts) == 0
        assert len(idxs) == 0

    def test_with_stem_skeleton(self):
        """Stem skeleton alignment doesn't crash."""
        skeleton, radii = _make_straight_sheath()
        stem_skel = _make_stem_skeleton()
        verts, idxs, *_ = mesh_sheath(
            skeleton, radii, stem_skeleton=stem_skel
        )
        assert len(verts) > 0
        assert len(idxs) > 0

    def test_curved_sheath(self):
        """Curved sheath produces valid mesh without NaN."""
        skeleton, radii = _make_curved_sheath()
        verts, idxs, norms, *_ = mesh_sheath(skeleton, radii)
        assert len(verts) > 0
        assert not np.any(np.isnan(verts))
        assert not np.any(np.isnan(norms))


# ─── TestSheathInPipeline ──────────────────────────────────────────────

class TestSheathInPipeline:
    """Integration tests: sheath organs flowing through loft_organs()."""

    def test_sheath_lofted_in_pipeline(self):
        """loft_organs() handles sheath type without error."""
        organs = [_make_sheath_organ(organ_id=0)]
        mesh = loft_organs(organs, subdivide=False, smooth=False)

        assert mesh.n_vertices > 0
        assert mesh.n_triangles > 0

    def test_mixed_organs(self):
        """Pipeline handles stem + sheath + leaf organs together."""
        stem_skel = _make_stem_skeleton()
        organs = [
            {
                "type": "stem",
                "part_type": "stem",
                "skeleton": stem_skel,
                "widths": np.full(len(stem_skel), 1.5),
                "organ_id": 0,
                "name": "stem_0",
                "node_ids": list(range(len(stem_skel))),
            },
            _make_sheath_organ(organ_id=1),
            {
                "type": "leaf",
                "part_type": "blade",
                "skeleton": np.column_stack([
                    np.linspace(0, 15, 30),
                    np.zeros(30),
                    np.full(30, 10.0),
                ]),
                "widths": np.full(30, 3.0),
                "organ_id": 2,
                "name": "leaf_2",
                "node_ids": list(range(30)),
            },
        ]
        mesh = loft_organs(organs, subdivide=False, smooth=False)

        # All three organ types present
        unique_oids = np.unique(mesh.organ_ids)
        assert len(unique_oids) == 3

    def test_part_type_in_meta(self):
        """organ_meta has correct part_type per organ."""
        stem_skel = _make_stem_skeleton()
        organs = [
            {
                "type": "stem",
                "part_type": "stem",
                "skeleton": stem_skel,
                "widths": np.full(len(stem_skel), 1.5),
                "organ_id": 0,
                "name": "stem_0",
                "node_ids": list(range(len(stem_skel))),
            },
            _make_sheath_organ(organ_id=1),
        ]
        mesh = loft_organs(organs, subdivide=False, smooth=False)

        meta_types = {m["organ_id"]: m.get("part_type") for m in mesh.organ_meta}
        assert meta_types[0] == "stem"
        assert meta_types[1] == "sheath"

    def test_backward_compat_no_sheath(self):
        """When no sheaths are present, pipeline behaves identically."""
        stem_skel = _make_stem_skeleton()
        leaf_skel = np.column_stack([
            np.linspace(0, 15, 30),
            np.zeros(30),
            np.full(30, 10.0),
        ])
        organs = [
            {
                "type": "stem",
                "skeleton": stem_skel,
                "widths": np.full(len(stem_skel), 1.5),
                "organ_id": 0,
                "name": "stem_0",
                "node_ids": list(range(len(stem_skel))),
            },
            {
                "type": "leaf",
                "skeleton": leaf_skel,
                "widths": np.full(30, 3.0),
                "organ_id": 1,
                "name": "leaf_1",
                "node_ids": list(range(30)),
            },
        ]
        mesh = loft_organs(organs, subdivide=False, smooth=False)

        assert mesh.n_vertices > 0
        assert mesh.n_triangles > 0
        # part_type defaults to type when not specified
        for meta in mesh.organ_meta:
            assert "part_type" in meta
            assert meta["part_type"] == meta["type"]

    def test_obj_write_materials(self, tmp_path):
        """to_obj with write_materials emits usemtl lines."""
        organs = [_make_sheath_organ(organ_id=0)]
        mesh = loft_organs(organs, subdivide=False, smooth=False)

        obj_path = tmp_path / "test_sheath.obj"
        mesh.to_obj(obj_path, write_materials=True)

        content = obj_path.read_text()
        assert "usemtl sheath" in content


# ─── TestBuildArcCrossSection ──────────────────────────────────────────

class TestBuildArcCrossSection:
    """Unit tests for _build_arc_cross_section helper."""

    def test_cross_section_shape(self):
        """Cross-section has correct shape."""
        cs = _build_arc_cross_section(
            wrap_angle=np.radians(330),
            overlap_angle=np.radians(30),
            n_arc=24,
        )
        # n_overlap = max(3, int(24 * 0.52/5.76)) = 3
        assert cs.shape == (27, 2)

    def test_main_arc_not_flagged(self):
        """Main arc points have overlap flag = 0."""
        cs = _build_arc_cross_section(np.radians(330), np.radians(30), 24)
        assert np.all(cs[:24, 1] == 0.0)

    def test_overlap_flagged(self):
        """Overlap points have flag = 1."""
        cs = _build_arc_cross_section(np.radians(330), np.radians(30), 24)
        assert np.all(cs[24:, 1] == 1.0)

    def test_angles_monotonic(self):
        """Angles increase monotonically."""
        cs = _build_arc_cross_section(np.radians(330), np.radians(30), 24)
        angles = cs[:, 0]
        assert np.all(np.diff(angles) > 0)


# ─── TestTriangulateSweep ──────────────────────────────────────────────

class TestTriangulateSweep:
    """Unit tests for _triangulate_sweep helper."""

    def test_index_bounds(self):
        """All triangle indices are within vertex bounds."""
        n_axis, n_cross = 10, 27
        idxs, oids, sids = _triangulate_sweep(n_axis, n_cross, organ_id=5)

        max_idx = n_axis * n_cross - 1
        assert np.all(idxs >= 0)
        assert np.all(idxs <= max_idx)

    def test_organ_ids(self):
        """All triangles have the correct organ_id."""
        _, oids, _ = _triangulate_sweep(10, 27, organ_id=42)
        assert np.all(oids == 42)

    def test_segment_ids_range(self):
        """Segment IDs are in [0, n_axis-2]."""
        n_axis = 15
        _, _, sids = _triangulate_sweep(n_axis, 27, organ_id=0)
        assert sids.min() >= 0
        assert sids.max() <= n_axis - 2
