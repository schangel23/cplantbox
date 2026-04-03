"""
G1-to-G3 geometry lofting engine.

Converts G1 skeleton representations (ordered 3D points with widths) into
G3 triangle meshes suitable for DART radiative transfer coupling.

Leaf lofting uses Baker's quad-base approach (flat ribbon from skeleton midrib).
Stem lofting uses cylindrical rings connected by triangle strips.
"""

import numpy as np
from collections import Counter
from scipy.interpolate import CubicSpline
from pathlib import Path


class G3Mesh:
    """Triangle mesh produced by G1-to-G3 lofting.

    Attributes:
        vertices:    (M, 3) float64 vertex positions
        indices:     (K, 3) int32 triangle vertex indices
        normals:     (M, 3) float64 per-vertex normals
        uvs:         (M, 2) float64 UV coordinates
        organ_ids:   (K,)   int32 organ ID per triangle
        segment_ids: (K,)   int32 original skeleton segment index per triangle
        organ_meta:  list of dicts with organ metadata for mapping export
    """

    def __init__(self, vertices, indices, normals, uvs, organ_ids,
                 segment_ids=None, organ_meta=None):
        self.vertices = np.asarray(vertices, dtype=np.float64)
        self.indices = np.asarray(indices, dtype=np.int32)
        self.normals = np.asarray(normals, dtype=np.float64)
        self.uvs = np.asarray(uvs, dtype=np.float64)
        self.organ_ids = np.asarray(organ_ids, dtype=np.int32)
        if segment_ids is not None:
            self.segment_ids = np.asarray(segment_ids, dtype=np.int32)
        else:
            self.segment_ids = np.full(len(self.indices), -1, dtype=np.int32)
        self.organ_meta = organ_meta or []

    @property
    def n_vertices(self):
        return len(self.vertices)

    @property
    def n_triangles(self):
        return len(self.indices)

    def to_obj(self, filepath, group_by_organ=True, group_prefix=""):
        """Export mesh to Wavefront OBJ format.

        Args:
            filepath: Output .obj file path.
            group_by_organ: If True, write 'g organ_<id>' groups.
            group_prefix: Optional prefix for group names (e.g. "p0_" for
                multi-plant exports → "p0_organ_0", "p0_organ_1", ...).
        """
        filepath = Path(filepath)
        with open(filepath, "w") as f:
            f.write("# G1-to-G3 lofted mesh\n")
            for v in self.vertices:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            for n in self.normals:
                f.write(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n")
            for uv in self.uvs:
                f.write(f"vt {uv[0]:.6f} {uv[1]:.6f}\n")

            if group_by_organ:
                unique_ids = np.unique(self.organ_ids)
                for oid in unique_ids:
                    gname = f"{group_prefix}organ_{oid}"
                    f.write(f"g {gname}\n")
                    mask = self.organ_ids == oid
                    for tri in self.indices[mask]:
                        # OBJ is 1-indexed
                        a, b, c = tri + 1
                        f.write(f"f {a}/{a}/{a} {b}/{b}/{b} {c}/{c}/{c}\n")
            else:
                for tri in self.indices:
                    a, b, c = tri + 1
                    f.write(f"f {a}/{a}/{a} {b}/{b}/{b} {c}/{c}/{c}\n")

    def to_mapping_json(self, filepath):
        """Export triangle-to-segment mapping as JSON for DART→CPlantBox feedback.

        For each organ, lists every original skeleton segment with:
        - node_ids: source node IDs (CPlantBox global IDs or Pheno4D indices)
        - uv_range: [u_start, u_end] normalized arc-length range
        - triangle_indices: global triangle indices in this mesh
        - triangle_count: number of triangles covering this segment

        Args:
            filepath: Output .json file path.
        """
        import json

        # Precompute all triangle areas
        v0 = self.vertices[self.indices[:, 0]]
        v1 = self.vertices[self.indices[:, 1]]
        v2 = self.vertices[self.indices[:, 2]]
        tri_areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)

        # Build per-organ, per-segment triangle lists
        organ_segments = {}  # organ_id -> {seg_idx -> [tri_indices]}
        for tri_idx in range(len(self.indices)):
            oid = int(self.organ_ids[tri_idx])
            sid = int(self.segment_ids[tri_idx])
            if oid not in organ_segments:
                organ_segments[oid] = {}
            if sid not in organ_segments[oid]:
                organ_segments[oid][sid] = []
            organ_segments[oid][sid].append(tri_idx)

        # Build output structure
        mapping = {"organs": [], "n_triangles": int(len(self.indices))}

        for meta in self.organ_meta:
            oid = meta["organ_id"]
            node_ids = meta.get("node_ids", [])
            arc_lengths = meta.get("arc_lengths", [])
            n_orig_segs = max(len(node_ids) - 1, 0)
            seg_dict = organ_segments.get(oid, {})

            segments = []
            for seg_idx in range(n_orig_segs):
                tri_list = seg_dict.get(seg_idx, [])

                # UV range from original arc-lengths
                if len(arc_lengths) > seg_idx + 1:
                    uv_start = float(arc_lengths[seg_idx])
                    uv_end = float(arc_lengths[seg_idx + 1])
                else:
                    uv_start = seg_idx / max(n_orig_segs, 1)
                    uv_end = (seg_idx + 1) / max(n_orig_segs, 1)

                seg_area = float(tri_areas[tri_list].sum()) if tri_list else 0.0
                seg_entry = {
                    "segment_idx": seg_idx,
                    "node_ids": [int(node_ids[seg_idx]), int(node_ids[seg_idx + 1])],
                    "uv_range": [round(uv_start, 6), round(uv_end, 6)],
                    "triangle_indices": tri_list,
                    "triangle_count": len(tri_list),
                    "total_area_cm2": round(seg_area, 6),
                }
                segments.append(seg_entry)

            organ_entry = {
                "organ_id": oid,
                "name": meta.get("name", f"organ_{oid}"),
                "type": meta.get("type", "unknown"),
                "n_segments": n_orig_segs,
                "n_node_ids": len(node_ids),
                "segments": segments,
            }
            if 'plant_id' in meta:
                organ_entry['plant_id'] = meta['plant_id']
            mapping["organs"].append(organ_entry)

        with open(filepath, "w") as f:
            json.dump(mapping, f, indent=2)

    def to_vtk_polydata(self):
        """Convert to vtkPolyData. VTK is imported lazily."""
        import vtk
        from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray

        points = vtk.vtkPoints()
        pts_vtk = numpy_to_vtk(self.vertices, deep=True)
        points.SetData(pts_vtk)

        # Build cell array from triangle indices
        n_tri = self.n_triangles
        cells_np = np.empty((n_tri, 4), dtype=np.int64)
        cells_np[:, 0] = 3
        cells_np[:, 1:] = self.indices
        cells_vtk = numpy_to_vtkIdTypeArray(cells_np.ravel(), deep=True)
        cell_array = vtk.vtkCellArray()
        cell_array.SetCells(n_tri, cells_vtk)

        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points)
        polydata.SetPolys(cell_array)

        # Normals
        normals_vtk = numpy_to_vtk(self.normals, deep=True)
        normals_vtk.SetName("Normals")
        polydata.GetPointData().SetNormals(normals_vtk)

        # UVs
        uvs_vtk = numpy_to_vtk(self.uvs, deep=True)
        uvs_vtk.SetName("UV")
        polydata.GetPointData().SetTCoords(uvs_vtk)

        # Organ IDs (cell data)
        ids_vtk = numpy_to_vtk(self.organ_ids.astype(np.int32), deep=True)
        ids_vtk.SetName("OrganID")
        polydata.GetCellData().AddArray(ids_vtk)

        # Segment IDs (cell data)
        seg_vtk = numpy_to_vtk(self.segment_ids.astype(np.int32), deep=True)
        seg_vtk.SetName("SegmentID")
        polydata.GetCellData().AddArray(seg_vtk)

        return polydata


def _compute_tangents(skeleton, smooth_window=0):
    """Compute unit tangent vectors along a skeleton using central differences.

    Args:
        skeleton: (N, 3) array of ordered 3D points.
        smooth_window: If > 0, apply a running-average smooth over this many
            neighbors (on each side) to reduce noise in high-density skeletons.

    Returns:
        (N, 3) array of unit tangent vectors.
    """
    n = len(skeleton)
    tangents = np.empty_like(skeleton)

    if n < 2:
        tangents[:] = [0, 0, 1]
        return tangents

    # Central differences (more stable than forward-only)
    tangents[0] = skeleton[1] - skeleton[0]
    tangents[-1] = skeleton[-1] - skeleton[-2]
    if n > 2:
        tangents[1:-1] = skeleton[2:] - skeleton[:-2]

    # Optional: smooth tangent field to reduce noise on dense skeletons
    if smooth_window > 0 and n > 2 * smooth_window + 1:
        kernel = 2 * smooth_window + 1
        from scipy.ndimage import uniform_filter1d
        for dim in range(3):
            tangents[1:-1, dim] = uniform_filter1d(
                tangents[1:-1, dim], size=kernel, mode='nearest'
            )

    # Normalize
    lengths = np.linalg.norm(tangents, axis=1, keepdims=True)
    lengths = np.maximum(lengths, 1e-12)
    tangents /= lengths

    return tangents


def _compute_arc_lengths(skeleton):
    """Compute cumulative arc-length parameter (0 at base, 1 at tip).

    Args:
        skeleton: (N, 3) array.

    Returns:
        (N,) array of normalized arc-length values.
    """
    diffs = np.diff(skeleton, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    total = cumulative[-1]
    if total < 1e-12:
        return np.linspace(0, 1, len(skeleton))
    return cumulative / total


def _subdivide_skeleton(skeleton, widths, target_spacing=0.5):
    """Upsample a coarse skeleton using cubic spline interpolation.

    Args:
        skeleton: (N, 3) array of ordered 3D points.
        widths: (N,) array of widths at each skeleton point.
        target_spacing: Desired spacing between points in cm.

    Returns:
        Tuple of (new_skeleton, new_widths, orig_segment_map) where
        orig_segment_map is an (M-1,) int array mapping each new segment
        to its original skeleton segment index.
    """
    skeleton = np.asarray(skeleton, dtype=np.float64)
    widths = np.asarray(widths, dtype=np.float64)
    n = len(skeleton)
    if n < 3:
        # Identity mapping: segment i maps to original segment i
        return skeleton, widths, np.arange(max(n - 1, 0), dtype=np.int32)

    # Compute arc-length parameterization
    diffs = np.diff(skeleton, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)
    total_length = seg_lengths.sum()
    if total_length < 1e-12:
        return skeleton, widths, np.arange(max(n - 1, 0), dtype=np.int32)

    cumulative = np.concatenate([[0.0], np.cumsum(seg_lengths)])

    avg_spacing = total_length / (n - 1)
    if avg_spacing <= target_spacing:
        return skeleton, widths, np.arange(n - 1, dtype=np.int32)

    # Number of output points
    m = max(n, int(np.ceil(total_length / target_spacing)) + 1)

    # Cubic spline interpolation for each coordinate and widths
    cs_x = CubicSpline(cumulative, skeleton[:, 0])
    cs_y = CubicSpline(cumulative, skeleton[:, 1])
    cs_z = CubicSpline(cumulative, skeleton[:, 2])
    cs_w = CubicSpline(cumulative, widths)

    new_arc = np.linspace(0, total_length, m)
    new_skeleton = np.column_stack([cs_x(new_arc), cs_y(new_arc), cs_z(new_arc)])
    new_widths = cs_w(new_arc)
    # Clamp widths: cubic spline can overshoot to negative or near-zero
    # values.  Enforce the same minimum as cplantbox_adapter (0.15 cm)
    # to prevent degenerate slivers downstream.
    new_widths = np.maximum(new_widths, 0.15)

    # Map each new segment (between new points i and i+1) back to the
    # original skeleton segment it falls within.  Use the midpoint of
    # each new segment and find which original arc-length interval it
    # belongs to.
    new_seg_midpoints = (new_arc[:-1] + new_arc[1:]) / 2.0
    # np.searchsorted gives the index of the first cumulative value > midpoint
    orig_segment_map = np.searchsorted(cumulative, new_seg_midpoints, side="right") - 1
    orig_segment_map = np.clip(orig_segment_map, 0, n - 2).astype(np.int32)

    return new_skeleton, new_widths, orig_segment_map


def _loft_leaf(organ):
    """Loft a leaf organ into ribbon geometry with optional cross-sectional curvature.

    Baker's approach extended: at each skeleton point, place N_CROSS vertices
    across the width. If gutter_depths are provided, the cross-section is
    curved (concave like a real maize leaf midrib channel).

    Orientation modes (in priority order):
    1. per_point_normals: array of normals at each skeleton point (captures twist)
    2. plane_normal: single global normal (flat leaf, no twist)
    3. parallel transport: follows skeleton 3D trajectory

    Returns:
        (vertices, indices, normals, uvs, organ_ids, segment_ids)
    """
    skeleton = np.asarray(organ["skeleton"], dtype=np.float64)
    widths = np.asarray(organ["widths"], dtype=np.float64)
    organ_id = organ["organ_id"]
    plane_normal = organ.get("plane_normal")
    per_point_normals = organ.get("per_point_normals")
    gutter_depths = organ.get("gutter_depths")

    # Uniformize skeleton spacing.  CPlantBox skeletons can have variable
    # segment lengths (tropism curves, growth steps).  Uniform spacing
    # produces regular triangles and avoids thin-strip slivers.
    #
    # IMPORTANT: n_new must be >= len(skeleton) to guarantee every original
    # CPlantBox segment maps to at least one mesh panel.  Previously n_new
    # could be smaller (e.g. 151 vs 300), causing ~50% of original segments
    # to get zero triangles and therefore zero aPAR in the DART pipeline.
    min_seg_len = 0.2  # cm — target uniform spacing
    total_skel_len = np.sum(np.linalg.norm(np.diff(skeleton, axis=0), axis=1))
    avg_spacing = total_skel_len / max(len(skeleton) - 1, 1)
    if avg_spacing < min_seg_len and len(skeleton) > 3 and total_skel_len > min_seg_len * 2:
        # Resample at uniform spacing, keeping at least as many points as
        # the original skeleton so no segments are lost in the mapping.
        cum = np.concatenate([[0.0], np.cumsum(np.linalg.norm(
            np.diff(skeleton, axis=0), axis=1))])
        n_new = max(len(skeleton), int(np.ceil(total_skel_len / min_seg_len)) + 1)
        new_arc = np.linspace(0, total_skel_len, n_new)
        new_skeleton = np.column_stack([
            np.interp(new_arc, cum, skeleton[:, d]) for d in range(3)
        ])
        new_widths = np.interp(new_arc, cum, widths)
        new_widths = np.maximum(new_widths, 0.15)

        # Rebuild per_point_normals if present
        if per_point_normals is not None:
            ppn = np.asarray(per_point_normals, dtype=np.float64)
            new_ppn = np.column_stack([
                np.interp(new_arc, cum, ppn[:, d]) for d in range(3)
            ])
            norms = np.linalg.norm(new_ppn, axis=1, keepdims=True)
            new_ppn /= np.maximum(norms, 1e-8)
            per_point_normals = new_ppn

        if gutter_depths is not None:
            gd = np.asarray(gutter_depths, dtype=np.float64)
            gutter_depths = np.interp(new_arc, cum, gd)

        # Rebuild orig_segment_map: map each new segment to the original
        # segment whose arc-length interval contains its midpoint.
        orig_seg_map = organ.get("_orig_segment_map")
        if orig_seg_map is not None:
            n_orig = len(cum) - 1  # number of original segments
            new_mid = (new_arc[:-1] + new_arc[1:]) / 2.0
            new_map = np.searchsorted(cum, new_mid, side="right") - 1
            new_map = np.clip(new_map, 0, n_orig - 1)

            # Safety: ensure every original segment is represented.
            # With n_new >= len(skeleton), gaps are rare (only with highly
            # non-uniform original spacing) but we handle them by stealing
            # new segments from originals that have multiple representatives.
            for _pass in range(10):
                counts = Counter(int(x) for x in new_map)
                missing = [k for k in range(n_orig) if k not in counts]
                if not missing:
                    break
                for orig_idx in missing:
                    orig_mid = (cum[orig_idx] + cum[orig_idx + 1]) / 2.0
                    dists = np.abs(new_mid - orig_mid)
                    order = np.argsort(dists)
                    for candidate in order:
                        if counts[int(new_map[candidate])] > 1:
                            counts[int(new_map[candidate])] -= 1
                            new_map[candidate] = orig_idx
                            counts[orig_idx] = 1
                            break

            new_seg_map = orig_seg_map[new_map]
            organ = dict(organ, _orig_segment_map=new_seg_map)

        skeleton = new_skeleton
        widths = new_widths

    n = len(skeleton)

    # Number of vertices across the width.
    # Need >=7 for visible edge ruffling and internal blade variation.
    # With gutter: 7 for smooth curved cross-section.
    # Without gutter but with wave effects: 7 for edge deformation.
    # Plain flat ribbon: 2 is fine.
    has_blade_effects = any(organ.get(k, 0) != 0 for k in (
        "wave_normal_amp", "curl_amp", "edge_ruffle_amp", "twist_max"))
    n_cross = 7 if (gutter_depths is not None or has_blade_effects) else 2

    # Use smoothed tangents for the binormal frame to prevent zigzag artifacts
    # on dense, high-curvature skeletons (e.g. drooping maize leaves with dx=0.1)
    smooth_win = max(1, n // 30) if n > 20 else 0
    tangents = _compute_tangents(skeleton, smooth_window=smooth_win)
    arc = _compute_arc_lengths(skeleton)

    # Curvature-adaptive width capping with forward-propagating smooth taper.
    # At high-curvature points (drooping tips), the blade half-width must
    # not exceed the radius of curvature to prevent self-intersection.
    # Instead of hard per-point clamping (which creates abrupt pinches),
    # compute the curvature limit at each point, then propagate the
    # tightest constraint forward with gradual tapering.
    if n >= 3:
        curv_max_w = np.full(n, 1e6)  # unconstrained by default
        for i in range(1, n - 1):
            t_prev = tangents[i - 1]
            t_next = tangents[i]
            dt = t_next - t_prev
            seg_len = np.linalg.norm(skeleton[i + 1] - skeleton[i - 1]) / 2.0
            if seg_len > 1e-8:
                kappa = np.linalg.norm(dt) / seg_len
                if kappa > 1e-6:
                    curv_max_w[i] = 2.0 / kappa  # full width ≤ 2*R

        # Forward-propagate: once curvature forces a reduction, taper back
        # up gradually (max 1% width increase per node) to avoid sudden jumps.
        # 1% per node spreads the transition over ~100 nodes for very smooth taper.
        for i in range(1, n):
            curv_max_w[i] = min(curv_max_w[i], curv_max_w[i - 1] * 1.01)
        # Backward-propagate: approach the constrained zone gradually too
        for i in range(n - 2, -1, -1):
            curv_max_w[i] = min(curv_max_w[i], curv_max_w[i + 1] * 1.01)

        for i in range(n):
            if widths[i] > curv_max_w[i]:
                widths[i] = curv_max_w[i]

        # Re-clamp after curvature capping: very high curvature can push
        # widths below the minimum, creating degenerate slivers that cause
        # Baleno Newton divergence.  0.15 cm matches the adapter minimum.
        widths = np.maximum(widths, 0.15)

    # Resolve per-point normals: interpolate to match skeleton length if needed
    use_per_point = per_point_normals is not None
    if use_per_point:
        per_point_normals = np.asarray(per_point_normals, dtype=np.float64)
        if len(per_point_normals) != n:
            # Interpolate normals to match subdivided skeleton length
            old_t = np.linspace(0, 1, len(per_point_normals))
            new_t = np.linspace(0, 1, n)
            interp_normals = np.empty((n, 3))
            for dim in range(3):
                interp_normals[:, dim] = np.interp(new_t, old_t, per_point_normals[:, dim])
            # Re-normalize
            norms = np.linalg.norm(interp_normals, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-8)
            per_point_normals = interp_normals / norms

    use_plane = not use_per_point and plane_normal is not None
    if use_plane:
        plane_normal = np.asarray(plane_normal, dtype=np.float64)
        pn_len = np.linalg.norm(plane_normal)
        if pn_len < 1e-6:
            use_plane = False
        else:
            plane_normal /= pn_len

    up = np.array([0.0, 0.0, 1.0])
    fallback = np.array([1.0, 0.0, 0.0])

    # Gravity-referenced mode: pre-compute a smooth binormal field so the
    # leaf blade stays approximately horizontal.  This avoids degenerate
    # frames when the tangent is nearly vertical and prevents blades from
    # appearing edge-on.
    use_gravity = not use_per_point and not use_plane

    # Pre-compute binormal field for gravity-referenced mode
    if use_gravity:
        binormal_field = np.empty((n, 3))

        # Strategy: fit best-fit plane to skeleton via SVD.  Maize leaves
        # curve essentially in a plane, so the plane normal gives a stable
        # reference that never degenerates (unlike the gravity-up cross
        # product which fails when the tangent is vertical).
        centroid = skeleton.mean(axis=0)
        centered = skeleton - centroid
        _, svals, vh = np.linalg.svd(centered, full_matrices=False)

        # The plane is well-defined when the smallest singular value is
        # much smaller than the second (leaf lies in a plane).
        plane_ok = (svals[2] < svals[1] * 0.5) if svals[1] > 1e-8 else False

        # Compute SVD plane normal as a stable fallback for near-vertical
        # tangent points where the gravity cross-product degenerates.
        svd_pn = None
        if plane_ok:
            svd_pn = vh[2].copy()

        for i in range(n):
            t = tangents[i]
            # Primary: gravity-referenced binormal (keeps blade horizontal)
            face_normal = up - np.dot(up, t) * t
            fn_len = np.linalg.norm(face_normal)

            if fn_len > 0.3:
                # Tangent is sufficiently non-vertical — gravity works
                face_normal /= fn_len
                bn = np.cross(t, face_normal)
            elif svd_pn is not None:
                # Near-vertical tangent — use SVD plane normal as stable
                # binormal reference (project perpendicular to tangent)
                bn = svd_pn - np.dot(svd_pn, t) * t
            else:
                # No SVD plane available — use raw gravity with fallback
                if fn_len > 1e-6:
                    face_normal /= fn_len
                    bn = np.cross(t, face_normal)
                else:
                    bn = np.cross(t, fallback)

            bn_len = np.linalg.norm(bn)
            if bn_len < 1e-6:
                bn = np.cross(t, fallback)
                bn_len = np.linalg.norm(bn)
            binormal_field[i] = bn / bn_len

        # Fix sign consistency: ensure all binormals point the same way
        for i in range(1, n):
            if np.dot(binormal_field[i], binormal_field[i - 1]) < 0:
                binormal_field[i] = -binormal_field[i]

        # Smooth binormal field to prevent abrupt changes at high-curvature points
        if n > 5:
            kernel = max(3, n // 10) | 1  # odd, proportional to length
            from scipy.ndimage import uniform_filter1d
            for dim in range(3):
                binormal_field[:, dim] = uniform_filter1d(
                    binormal_field[:, dim], size=kernel, mode='nearest'
                )
            # Re-normalize after smoothing
            bn_lengths = np.linalg.norm(binormal_field, axis=1, keepdims=True)
            bn_lengths = np.maximum(bn_lengths, 1e-12)
            binormal_field /= bn_lengths

    if use_per_point:
        binormal = np.cross(per_point_normals[0], tangents[0])
        bn_len = np.linalg.norm(binormal)
        if bn_len < 1e-6:
            binormal = np.cross(tangents[0], up)
            bn_len = np.linalg.norm(binormal)
    elif use_plane:
        binormal = np.cross(plane_normal, tangents[0])
        bn_len = np.linalg.norm(binormal)
        if bn_len < 1e-6:
            use_plane = False

    if use_gravity:
        binormal = binormal_field[0].copy()
    elif not use_per_point and not use_plane:
        binormal = np.cross(tangents[0], up)
        bn_len = np.linalg.norm(binormal)
        if bn_len < 1e-6:
            binormal = np.cross(tangents[0], fallback)
            bn_len = np.linalg.norm(binormal)
        binormal /= np.linalg.norm(binormal)

    # Cross-section positions: fractions across width [-0.5, ..., 0.5]
    cross_fracs = np.linspace(-0.5, 0.5, n_cross)
    # Gutter profile: parabolic dip (deepest at center, 0 at edges)
    # depth(f) = gutter_depth * (1 - (2*f)^2), where f in [-0.5, 0.5]
    cross_gutter = 1.0 - (2.0 * cross_fracs) ** 2  # 0 at edges, 1 at center

    # --- Leaf blade waviness, twist, curl & edge ruffling ---
    # Real maize leaves have dramatic internal bending, edge ruffling,
    # twist, and asymmetric curl. Effects:
    #   1. Vertical undulation (up-down waves perpendicular to blade midrib)
    #   2. Lateral sway (side-to-side displacement along binormal)
    #   3. Twist (blade rotates around the midrib axis along its length)
    #   4. Asymmetric curl (one edge up, other down — oscillates along leaf)
    #   5. Edge ruffling (high-frequency edge-only undulation — the dramatic
    #      wavy margin seen on real maize leaves)
    #   6. Internal fold (cross-section curvature variation along the leaf —
    #      the blade bends/buckles at interior points)
    wave_normal_amp = organ.get("wave_normal_amp", 0.0)
    wave_normal_freq = organ.get("wave_normal_freq", 3.5)
    wave_normal_phase = organ.get("wave_normal_phase", 0.0)
    wave_lateral_amp = organ.get("wave_lateral_amp", 0.0)
    wave_lateral_freq = organ.get("wave_lateral_freq", 2.0)
    wave_lateral_phase = organ.get("wave_lateral_phase", 0.0)
    twist_max = organ.get("twist_max", 0.0)
    curl_amp = organ.get("curl_amp", 0.0)
    curl_freq = organ.get("curl_freq", 2.0)
    curl_phase = organ.get("curl_phase", 0.0)
    curl_onset = organ.get("curl_onset", None)  # separate onset for curl ramp
    # Edge ruffling: high-frequency undulation at leaf margins only.
    # Creates the characteristic ruffled/wavy edge of real maize leaves.
    edge_ruffle_amp = organ.get("edge_ruffle_amp", 0.0)  # cm displacement at edge
    edge_ruffle_freq = organ.get("edge_ruffle_freq", 7.0)  # waves per leaf (high freq)
    edge_ruffle_phase = organ.get("edge_ruffle_phase", 0.0)
    # Internal fold: the blade bends/buckles at points between midrib and edge.
    # This is a low-frequency cross-sectional curvature change.
    fold_amp = organ.get("fold_amp", 0.0)  # cm displacement at quarter-width points
    fold_freq = organ.get("fold_freq", 2.5)  # folds per leaf length
    fold_phase = organ.get("fold_phase", 0.0)
    # Ramp onset: fraction of leaf length where deformation effects begin.
    # MaizeField3D measured ~0.05 (5%); hand-tuned default was 0.15.
    ramp_onset = organ.get("ramp_onset", 0.15)

    total_arc = arc[-1] if arc[-1] > 0 else 1.0
    has_waves = (wave_normal_amp > 0 or wave_lateral_amp > 0
                 or abs(twist_max) > 0 or curl_amp > 0
                 or edge_ruffle_amp > 0 or fold_amp > 0)

    # --- Spline-based geometry features ---
    # Source priority: fitted_extended_cps (per-plant optimized) > XML splines (population defaults)
    # fitted_extended_cps: dict of feature_name -> list of CP values (evenly spaced in [0,1])
    # XML splines: dict with 'phi' (knot positions) and 'values' (per-knot data)
    fitted_ext = organ.get("fitted_extended_cps", {})

    def _make_spline_from_fitted(cp_values):
        """Convert evenly-spaced CP list to spline dict."""
        k = len(cp_values)
        return {'phi': np.linspace(0, 1, k), 'values': np.array(cp_values)}

    oop_curv_spline = (_make_spline_from_fitted(fitted_ext['out_of_plane_curv'])
                       if 'out_of_plane_curv' in fitted_ext
                       else organ.get("oop_curv_spline"))
    asymmetry_spline = (_make_spline_from_fitted(fitted_ext['asymmetry'])
                        if 'asymmetry' in fitted_ext
                        else organ.get("asymmetry_spline"))
    edge_curl_spline = (_make_spline_from_fitted(fitted_ext['edge_curl'])
                        if 'edge_curl' in fitted_ext
                        else organ.get("edge_curl_spline"))
    cross_section_spline = (_make_spline_from_fitted(fitted_ext['cross_section_profile'])
                            if 'cross_section_profile' in fitted_ext
                            else organ.get("cross_section_spline"))

    has_spline_features = any(s is not None for s in [
        oop_curv_spline, asymmetry_spline, edge_curl_spline, cross_section_spline])

    # Bump n_cross to 7 if spline features need cross-section detail
    if has_spline_features and n_cross < 7:
        n_cross = 7
        cross_fracs = np.linspace(-0.5, 0.5, n_cross)
        cross_gutter = 1.0 - (2.0 * cross_fracs) ** 2

    # Pre-compute per-skeleton-point spline-interpolated values
    arc_fracs = arc / total_arc if total_arc > 0 else np.linspace(0, 1, n)

    def _interp_spline(spline_dict):
        """Interpolate spline control points at arc fraction positions."""
        return np.interp(arc_fracs, spline_dict['phi'], spline_dict['values'])

    # Linear ramp for spline features (same as deformations: zero at base, full at tip)
    spline_ramp = np.maximum(0.0, (arc_fracs - ramp_onset) / max(1.0 - ramp_onset, 1e-12))

    oop_curv_values = None
    if oop_curv_spline is not None:
        raw = _interp_spline(oop_curv_spline)
        # Quadratic ramp for skeleton-modifying features (like twist)
        oop_curv_values = raw * spline_ramp * spline_ramp

    asymmetry_values = None
    if asymmetry_spline is not None:
        asymmetry_values = _interp_spline(asymmetry_spline) * spline_ramp

    edge_curl_values = None
    if edge_curl_spline is not None:
        edge_curl_values = _interp_spline(edge_curl_spline) * spline_ramp

    cross_section_values = None
    if cross_section_spline is not None:
        cross_section_values = _interp_spline(cross_section_spline) * spline_ramp

    # Pre-compute per-skeleton-point offsets (midrib-level effects)
    wave_normal_offsets = np.zeros(n)
    wave_lateral_offsets = np.zeros(n)
    twist_angles = np.zeros(n)
    curl_factors = np.zeros(n)
    edge_ruffle_base = np.zeros(n)  # base ruffle value per skeleton point
    fold_factors = np.zeros(n)      # fold strength per skeleton point

    # --- Fitted spline CPs override sinusoidal deformations ---
    # When present, these come from gradient-fitted per-plant optimization
    # (diff_lofter pipeline) and replace the random sinusoidal model entirely.
    fitted_cps = organ.get("fitted_deform_cps")
    if fitted_cps is not None:
        has_waves = True
        if n_cross < 7:
            n_cross = 7
            cross_fracs = np.linspace(-0.5, 0.5, n_cross)
            cross_gutter = 1.0 - (2.0 * cross_fracs) ** 2

        def _interp_cp(cp_values):
            """Interpolate evenly-spaced CPs at arc fraction positions."""
            k = len(cp_values)
            cp_phi = np.linspace(0, 1, k)
            return np.interp(arc_fracs, cp_phi, cp_values)

        for cp_name, target_arr, use_sq in [
            ('wave_normal', wave_normal_offsets, False),
            ('wave_lateral', wave_lateral_offsets, False),
            ('twist', twist_angles, True),
            ('curl', curl_factors, False),
            ('edge_ruffle', edge_ruffle_base, False),
            ('fold', fold_factors, False),
        ]:
            if cp_name in fitted_cps:
                interp = _interp_cp(fitted_cps[cp_name])
                ramp_use = spline_ramp * spline_ramp if use_sq else spline_ramp
                target_arr[:] = interp * ramp_use

    elif has_waves:
        for i in range(n):
            t_frac = arc[i] / total_arc  # 0 at base, 1 at tip
            # Linear ramp: effects start at ramp_onset along the leaf.
            # MaizeField3D data shows onset ~5%, hand-tuned default was 15%.
            ramp = max(0.0, (t_frac - ramp_onset) / (1.0 - ramp_onset))
            ramp_sq = ramp * ramp  # quadratic for twist only

            wave_normal_offsets[i] = wave_normal_amp * ramp * np.sin(
                2 * np.pi * wave_normal_freq * t_frac + wave_normal_phase)
            wave_lateral_offsets[i] = wave_lateral_amp * ramp * np.sin(
                2 * np.pi * wave_lateral_freq * t_frac + wave_lateral_phase)
            # Twist ramps quadratically (gentle near base, strong at tip)
            twist_angles[i] = twist_max * ramp_sq
            # Curl: low-freq asymmetric edge displacement
            # Uses its own onset if provided (curl starts later than ruffle)
            if curl_onset is not None:
                curl_ramp = max(0.0, (t_frac - curl_onset) / (1.0 - curl_onset))
            else:
                curl_ramp = ramp
            curl_factors[i] = curl_amp * curl_ramp * np.sin(
                2 * np.pi * curl_freq * t_frac + curl_phase)
            # Edge ruffle: high-freq, computed per skeleton point.
            # Per-vertex amplitude depends on distance from midrib (applied in vertex loop).
            edge_ruffle_base[i] = edge_ruffle_amp * ramp * np.sin(
                2 * np.pi * edge_ruffle_freq * t_frac + edge_ruffle_phase)
            # Internal fold: cross-sectional curvature variation
            fold_factors[i] = fold_amp * ramp * np.sin(
                2 * np.pi * fold_freq * t_frac + fold_phase)

    # --- Apply out-of-plane curvature: modify skeleton before vertex sweep ---
    # Integrates curvature along binormal to displace the skeleton perpendicular
    # to the growth plane (e.g. modeling droop/lift not captured by tropism).
    if oop_curv_values is not None and use_gravity:
        diffs_oop = np.diff(skeleton, axis=0)
        seg_lens_oop = np.linalg.norm(diffs_oop, axis=1)
        d_angle = oop_curv_values[:-1] * seg_lens_oop
        cum_angle = np.concatenate([[0.0], np.cumsum(d_angle)])
        oop_disp = np.concatenate([[0.0], np.cumsum(
            np.sin(cum_angle[:-1]) * seg_lens_oop)])
        skeleton = skeleton + oop_disp[:, np.newaxis] * binormal_field

    vertices = np.empty((n_cross * n, 3))
    normals_arr = np.empty((n_cross * n, 3))
    uvs = np.empty((n_cross * n, 2))

    # Max width for proportional deformation scaling
    max_w = float(np.max(widths)) if np.max(widths) > 0.01 else 1.0

    for i in range(n):
        t = tangents[i]
        w = widths[i]

        if i > 0:
            if use_per_point:
                # Derive binormal from per-point normal × tangent
                binormal = np.cross(per_point_normals[i], t)
                bn_len = np.linalg.norm(binormal)
                if bn_len < 1e-6:
                    binormal = binormal_prev - np.dot(binormal_prev, t) * t
                    bn_len = np.linalg.norm(binormal)
                binormal /= max(bn_len, 1e-12)
            elif use_plane:
                binormal = np.cross(plane_normal, t)
                bn_len = np.linalg.norm(binormal)
                if bn_len < 1e-6:
                    binormal = binormal_prev - np.dot(binormal_prev, t) * t
                    bn_len = np.linalg.norm(binormal)
                binormal /= max(bn_len, 1e-12)
            elif use_gravity:
                # Use pre-computed smooth gravity-referenced binormal
                binormal = binormal_field[i].copy()
            else:
                # Parallel transport: project previous binormal onto the
                # plane perpendicular to the current (smoothed) tangent.
                new_bn = binormal_prev - np.dot(binormal_prev, t) * t
                bn_len = np.linalg.norm(new_bn)
                if bn_len < 1e-6:
                    new_bn = np.cross(t, up)
                    bn_len = np.linalg.norm(new_bn)
                    if bn_len < 1e-6:
                        new_bn = np.cross(t, fallback)
                        bn_len = np.linalg.norm(new_bn)
                new_bn /= bn_len
                binormal = new_bn

            # Prevent binormal flip (not needed for gravity mode, but kept for others)
            if not use_gravity and np.dot(binormal, binormal_prev) < 0:
                binormal = -binormal

        binormal_prev = binormal.copy()

        normal = np.cross(t, binormal)
        normal_len = np.linalg.norm(normal)
        if normal_len > 1e-12:
            normal /= normal_len

        # Apply twist: rotate binormal and normal around tangent
        if has_waves and abs(twist_angles[i]) > 1e-6:
            ca = np.cos(twist_angles[i])
            sa = np.sin(twist_angles[i])
            bn_twisted = ca * binormal + sa * normal
            nm_twisted = -sa * binormal + ca * normal
            binormal = bn_twisted / max(np.linalg.norm(bn_twisted), 1e-12)
            normal = nm_twisted / max(np.linalg.norm(nm_twisted), 1e-12)

        center = skeleton[i].copy()
        # Apply blade waviness: vertical + lateral displacement
        if has_waves:
            center += wave_normal_offsets[i] * normal
            center += wave_lateral_offsets[i] * binormal
        gd = 0.0
        if gutter_depths is not None:
            gd_idx = min(i, len(gutter_depths) - 1)
            gd = gutter_depths[gd_idx] if gd_idx >= 0 else 0.0

        for j in range(n_cross):
            frac = cross_fracs[j]  # -0.5 to 0.5
            lateral = frac * w * binormal
            # Gutter: dip center downward (negative normal direction)
            gutter_offset = -gd * cross_gutter[j] * normal
            # Asymmetric curl: edges displaced in opposite normal directions
            curl_offset = (2.0 * frac) * curl_factors[i] * normal
            # Edge ruffling: high-frequency undulation at leaf margins.
            # Amplitude is proportional to |frac|^2 (zero at midrib, max at edges).
            # Left and right edges are out of phase (+ frac*pi offset).
            edge_frac = (2.0 * abs(frac)) ** 2  # 0 at center, 1 at edge
            ruffle_offset = edge_frac * edge_ruffle_base[i] * normal
            # Add opposite-phase component so left/right edges ruffle differently
            if frac < 0:
                ruffle_offset *= -1.0
            # Internal fold: blade buckles at quarter-width points.
            # Uses sin(pi*|2*frac|) which peaks at the quarter-width positions
            # (frac=±0.25) and is zero at midrib (frac=0) and edges (frac=±0.5).
            fold_profile = np.sin(np.pi * abs(2.0 * frac))
            fold_offset = fold_factors[i] * fold_profile * normal

            # Scale cross-section deformation proportionally to width.
            # Curl/ruffle/fold are physical effects of a wide blade — they
            # must diminish as the blade tapers.  Without this, curl at the
            # tip creates a "fish tail" split because the displacement
            # dominates over the narrowing width.
            w_fade = w / max_w
            curl_offset *= w_fade
            ruffle_offset *= w_fade
            fold_offset *= w_fade
            gutter_offset *= w_fade

            # --- Spline-based geometry features ---
            # Asymmetry: shift cross-section center along binormal
            asym_offset = np.zeros(3)
            if asymmetry_values is not None:
                asym_offset = asymmetry_values[i] * binormal * w_fade

            # Edge curl: margin-only deflection angle (cubic edge profile)
            edge_curl_offset = np.zeros(3)
            if edge_curl_values is not None:
                edge_influence = (2.0 * abs(frac)) ** 3  # concentrated at margins
                edge_curl_offset = (edge_influence * np.tan(edge_curl_values[i])
                                    * w * 0.5 * normal * w_fade)

            # Cross-section profile: parabolic transverse curvature
            cs_offset = np.zeros(3)
            if cross_section_values is not None:
                cs_profile = (2.0 * frac) ** 2  # 0 at center, 1 at edges
                cs_offset = cross_section_values[i] * cs_profile * normal * w_fade

            v_idx = n_cross * i + j
            vertices[v_idx] = (center + lateral + gutter_offset
                               + curl_offset + ruffle_offset + fold_offset
                               + asym_offset + edge_curl_offset + cs_offset)

            # Per-vertex normal: for curved cross-section, tilt normals outward
            if n_cross > 2 and gd > 0:
                # Approximate normal from cross-section tangent
                tilt = 2.0 * gd * (2.0 * frac) * binormal / max(w, 0.01)
                v_normal = normal + tilt
                v_normal /= max(np.linalg.norm(v_normal), 1e-12)
            else:
                v_normal = normal
            normals_arr[v_idx] = v_normal

            uvs[v_idx] = [arc[i], frac + 0.5]  # v in [0, 1]

    # Build triangle strips: (n_cross-1) quads per segment, 2 tris per quad
    n_segs = n - 1
    n_tris_per_seg = 2 * (n_cross - 1)
    indices = np.empty((n_segs * n_tris_per_seg, 3), dtype=np.int32)

    for i in range(n_segs):
        for j in range(n_cross - 1):
            bl = n_cross * i + j
            br = n_cross * i + j + 1
            tl = n_cross * (i + 1) + j
            tr = n_cross * (i + 1) + j + 1
            tri_base = i * n_tris_per_seg + 2 * j
            indices[tri_base] = [bl, tl, br]
            indices[tri_base + 1] = [br, tl, tr]

    organ_ids = np.full(len(indices), organ_id, dtype=np.int32)

    orig_seg_map = organ.get("_orig_segment_map")
    segment_ids = np.empty(len(indices), dtype=np.int32)
    for i in range(n_segs):
        if orig_seg_map is not None:
            sid = int(orig_seg_map[i])
        else:
            sid = i
        for k in range(n_tris_per_seg):
            segment_ids[i * n_tris_per_seg + k] = sid

    return vertices, indices, normals_arr, uvs, organ_ids, segment_ids


def _apply_internode_modulation(skeleton, widths, node_heights_z,
                                 node_bulge=0.12, node_band_cm=1.0,
                                 groove_depth=0.04, groove_width_cm=0.8,
                                 internode_barrel=0.02):
    """Modulate stem widths to create visible internode structure.

    Real maize stems have swollen nodes (wider bands at leaf attachment
    points) with smooth cylindrical internodes between them.  Each node
    is flanked by shallow grooves that enhance the visual contrast.

    Args:
        skeleton: (N, 3) stem skeleton points.
        widths: (N,) current widths (diameter).
        node_heights_z: List of Z heights where leaves attach.
        node_bulge: Fractional radius increase at nodes (0.35 = 35% wider).
        node_band_cm: Half-width of the raised-cosine node band in cm.
        groove_depth: Fractional constriction in groove flanking each node.
        groove_width_cm: Half-width of groove bands in cm.
        internode_barrel: Slight barrel shape of internodes.

    Returns:
        (N,) modulated widths.
    """
    if not node_heights_z or len(node_heights_z) < 2:
        return widths

    z = skeleton[:, 2]
    modulated = widths.copy()
    nodes_z = sorted(node_heights_z)

    for i in range(len(z)):
        zi = z[i]

        # Node bulge: raised-cosine bump centered at each leaf attachment
        # height.  Each bump is smooth (C1 continuous) and zero outside
        # ±node_band_cm of the node.  Take the max contribution from all
        # nearby nodes (overlapping bands don't stack).
        node_bump = 0.0
        for nz in nodes_z:
            dist = abs(zi - nz)
            if dist < node_band_cm:
                bump = 0.5 * (1.0 + np.cos(np.pi * dist / node_band_cm))
                node_bump = max(node_bump, bump)

        # Groove: shallow constriction just above and below each node.
        # Creates the visible "ring" effect by darkening the shadow line.
        # Groove bands sit at ±(node_band_cm + groove_width_cm/2) from node.
        groove = 0.0
        for nz in nodes_z:
            dist = abs(zi - nz)
            groove_center = node_band_cm + groove_width_cm * 0.5
            groove_dist = abs(dist - groove_center)
            if groove_dist < groove_width_cm * 0.5:
                g = 0.5 * (1.0 + np.cos(np.pi * groove_dist / (groove_width_cm * 0.5)))
                groove = max(groove, g)

        # Internode barrel: very subtle outward bow at internode midpoints.
        barrel = 0.0
        if node_bump < 0.05 and groove < 0.05:
            seg_idx = np.searchsorted(nodes_z, zi, side='right') - 1
            if 0 <= seg_idx < len(nodes_z) - 1:
                z_lo = nodes_z[seg_idx]
                z_hi = nodes_z[seg_idx + 1]
                dz = z_hi - z_lo
                if dz > 3.0:
                    t = (zi - z_lo) / dz
                    barrel = np.sin(np.pi * t)

        scale = (1.0 + node_bulge * node_bump
                 - groove_depth * groove
                 + internode_barrel * barrel)
        modulated[i] = widths[i] * scale

    return modulated


def _loft_stem(organ, n_sides=8):
    """Loft a stem organ into cylindrical tube geometry with end caps.

    Creates rings of vertices at each skeleton point, connected by
    triangle strips.  Uses parallel transport for the frame to avoid
    discontinuous twisting between rings.  Adds disc caps at both ends.

    Internode modulation: if 'node_heights_z' is provided in the organ
    dict, modulates the radius to create visible internode segments.

    Returns:
        (vertices, indices, normals, uvs, organ_ids)
    """
    skeleton = np.asarray(organ["skeleton"], dtype=np.float64)
    widths = np.asarray(organ["widths"], dtype=np.float64)
    organ_id = organ["organ_id"]
    n = len(skeleton)

    # Apply internode modulation if leaf attachment heights are provided
    node_heights_z = organ.get("node_heights_z")
    if node_heights_z:
        widths = _apply_internode_modulation(skeleton, widths, node_heights_z)

    tangents = _compute_tangents(skeleton)
    arc = _compute_arc_lengths(skeleton)

    up = np.array([0.0, 0.0, 1.0])
    fallback = np.array([1.0, 0.0, 0.0])

    # Angles around the tube
    angles = np.linspace(0, 2 * np.pi, n_sides, endpoint=False)

    # Allocate ring vertices + 2 cap center vertices
    n_ring_verts = n * n_sides
    n_total_verts = n_ring_verts + 2  # +2 for bottom and top cap centers
    vertices = np.empty((n_total_verts, 3))
    normals_arr = np.empty((n_total_verts, 3))
    uvs = np.empty((n_total_verts, 2))

    # Initial frame at first point
    binormal = np.cross(tangents[0], up)
    bn_len = np.linalg.norm(binormal)
    if bn_len < 1e-6:
        binormal = np.cross(tangents[0], fallback)
        bn_len = np.linalg.norm(binormal)
    binormal /= bn_len

    for i in range(n):
        t = tangents[i]
        radius = widths[i] / 2.0

        if i > 0:
            # Parallel transport: project previous binormal onto plane
            # perpendicular to current tangent
            binormal = binormal - np.dot(binormal, t) * t
            bn_len = np.linalg.norm(binormal)
            if bn_len < 1e-6:
                binormal = np.cross(t, up)
                bn_len = np.linalg.norm(binormal)
                if bn_len < 1e-6:
                    binormal = np.cross(t, fallback)
                    bn_len = np.linalg.norm(binormal)
            binormal /= bn_len

        normal = np.cross(t, binormal)
        normal_len = np.linalg.norm(normal)
        if normal_len > 1e-12:
            normal /= normal_len

        center = skeleton[i]
        for j in range(n_sides):
            a = angles[j]
            direction = np.cos(a) * binormal + np.sin(a) * normal
            idx = i * n_sides + j
            vertices[idx] = center + radius * direction
            normals_arr[idx] = direction  # outward-pointing
            uvs[idx] = [arc[i], a / (2.0 * np.pi)]

    # Cap center vertices
    bot_center_idx = n_ring_verts
    top_center_idx = n_ring_verts + 1
    vertices[bot_center_idx] = skeleton[0]
    vertices[top_center_idx] = skeleton[-1]
    # Cap normals point along the tangent (inward for bottom, outward for top)
    normals_arr[bot_center_idx] = -tangents[0]
    normals_arr[top_center_idx] = tangents[-1]
    uvs[bot_center_idx] = [0.0, 0.5]
    uvs[top_center_idx] = [1.0, 0.5]

    # Build triangle strips between consecutive rings
    # Winding order: counter-clockwise when viewed from outside
    n_segments = n - 1
    n_wall_tris = 2 * n_segments * n_sides
    n_cap_tris = 2 * n_sides  # bottom + top cap fans
    n_total_tris = n_wall_tris + n_cap_tris

    indices = np.empty((n_total_tris, 3), dtype=np.int32)
    segment_ids = np.empty(n_total_tris, dtype=np.int32)
    orig_seg_map = organ.get("_orig_segment_map")
    tri_idx = 0

    # Wall triangles
    for i in range(n_segments):
        if orig_seg_map is not None:
            sid = int(orig_seg_map[i])
        else:
            sid = i
        for j in range(n_sides):
            j_next = (j + 1) % n_sides
            # Current ring
            c0 = i * n_sides + j
            c1 = i * n_sides + j_next
            # Next ring
            n0 = (i + 1) * n_sides + j
            n1 = (i + 1) * n_sides + j_next
            indices[tri_idx] = [c0, c1, n0]
            indices[tri_idx + 1] = [c1, n1, n0]
            segment_ids[tri_idx] = sid
            segment_ids[tri_idx + 1] = sid
            tri_idx += 2

    # Bottom cap: fan from center to first ring (winding faces downward)
    for j in range(n_sides):
        j_next = (j + 1) % n_sides
        indices[tri_idx] = [bot_center_idx, j_next, j]
        segment_ids[tri_idx] = 0
        tri_idx += 1

    # Top cap: fan from center to last ring (winding faces upward)
    last_ring_start = (n - 1) * n_sides
    last_sid = (n - 2) if orig_seg_map is None else int(orig_seg_map[-1])
    for j in range(n_sides):
        j_next = (j + 1) % n_sides
        indices[tri_idx] = [top_center_idx, last_ring_start + j,
                            last_ring_start + j_next]
        segment_ids[tri_idx] = last_sid
        tri_idx += 1

    organ_ids = np.full(len(indices), organ_id, dtype=np.int32)

    return vertices, indices, normals_arr, uvs, organ_ids, segment_ids


def loft_organs(organs, stem_sides=8, subdivide=True, target_spacing=0.5,
                smooth=True, smooth_iterations=3):
    """Loft all organs into a single G3Mesh.

    Args:
        organs: List of organ dicts with keys:
            - type: "leaf" or "stem"
            - skeleton: (N, 3) array of ordered 3D points
            - widths: (N,) array of full-width at each skeleton point
            - organ_id: int for DART segment tracking
            - name: optional label
        stem_sides: Number of sides for stem cylinder cross-sections.
        subdivide: If True, upsample coarse skeletons via cubic spline.
        target_spacing: Target spacing in cm for skeleton subdivision.
        smooth: If True, apply Laplacian smoothing to the final mesh.
        smooth_iterations: Number of Laplacian smoothing passes.

    Returns:
        G3Mesh with all organs combined.
    """
    all_verts = []
    all_indices = []
    all_normals = []
    all_uvs = []
    all_organ_ids = []
    all_segment_ids = []
    organ_meta = []
    vertex_offset = 0

    for organ in organs:
        # Compute original arc-lengths before subdivision (for mapping JSON)
        orig_skeleton = np.asarray(organ["skeleton"], dtype=np.float64)
        orig_diffs = np.diff(orig_skeleton, axis=0)
        orig_seg_lens = np.linalg.norm(orig_diffs, axis=1)
        orig_cumul = np.concatenate([[0.0], np.cumsum(orig_seg_lens)])
        orig_total = orig_cumul[-1]
        if orig_total > 1e-12:
            orig_arc_norm = (orig_cumul / orig_total).tolist()
        else:
            orig_arc_norm = np.linspace(0, 1, len(orig_skeleton)).tolist()

        # Store organ metadata for mapping export
        node_ids = organ.get("node_ids", list(range(len(orig_skeleton))))
        organ_meta.append({
            "organ_id": organ["organ_id"],
            "name": organ.get("name", f"organ_{organ['organ_id']}"),
            "type": organ["type"],
            "node_ids": node_ids,
            "arc_lengths": orig_arc_norm,
        })

        # Optionally subdivide coarse skeletons before lofting
        if subdivide:
            skel, wid, orig_seg_map = _subdivide_skeleton(
                organ["skeleton"], organ["widths"], target_spacing=target_spacing
            )
            organ = dict(organ, skeleton=skel, widths=wid,
                         _orig_segment_map=orig_seg_map)
        else:
            organ = dict(organ)

        otype = organ["type"]
        if otype == "leaf":
            verts, idxs, norms, uvs, oids, sids = _loft_leaf(organ)
        elif otype in ("stem", "root"):
            verts, idxs, norms, uvs, oids, sids = _loft_stem(organ, n_sides=stem_sides)
        else:
            raise ValueError(f"Unknown organ type: {otype!r}")

        all_verts.append(verts)
        all_indices.append(idxs + vertex_offset)
        all_normals.append(norms)
        all_uvs.append(uvs)
        all_organ_ids.append(oids)
        all_segment_ids.append(sids)
        vertex_offset += len(verts)

    if not all_verts:
        return G3Mesh(
            np.empty((0, 3)), np.empty((0, 3), dtype=np.int32),
            np.empty((0, 3)), np.empty((0, 2)), np.empty(0, dtype=np.int32),
        )

    mesh = G3Mesh(
        vertices=np.concatenate(all_verts),
        indices=np.concatenate(all_indices),
        normals=np.concatenate(all_normals),
        uvs=np.concatenate(all_uvs),
        organ_ids=np.concatenate(all_organ_ids),
        segment_ids=np.concatenate(all_segment_ids),
        organ_meta=organ_meta,
    )

    if smooth:
        mesh = _laplacian_smooth(mesh, iterations=smooth_iterations)

    mesh = _remove_degenerate_triangles(mesh)

    return mesh


def _remove_degenerate_triangles(mesh, min_area_cm2=0.001):
    """Remove triangles below a minimum area threshold.

    Degenerate slivers at leaf tips (width → 0) cause Baleno's Newton
    solver to diverge, producing non-physical temperatures (80–150 °C).
    Removing them at the geometry stage prevents the issue entirely.

    Mesh vertices are in cm, so areas from cross products are in cm².
    Day-10 normal triangles are ~0.01–0.04 cm²; mature plants ~2–3 cm²;
    degenerate tip slivers are ~0.0001 cm².  Default threshold 0.001 cm²
    removes only the true slivers across all growth stages.

    Args:
        mesh: G3Mesh instance (vertices in cm).
        min_area_cm2: Minimum triangle area in cm².  Default 0.001 cm².

    Returns:
        New G3Mesh with degenerate triangles removed.
    """
    verts = mesh.vertices

    # Compute triangle areas via cross product
    v0 = verts[mesh.indices[:, 0]]
    v1 = verts[mesh.indices[:, 1]]
    v2 = verts[mesh.indices[:, 2]]
    areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)

    keep = areas >= min_area_cm2
    n_removed = int(np.sum(~keep))

    if n_removed > 0:
        print(f"  Removed {n_removed}/{len(mesh.indices)} degenerate triangles "
              f"(area < {min_area_cm2} cm²)")

    return G3Mesh(
        vertices=mesh.vertices,
        indices=mesh.indices[keep],
        normals=mesh.normals,
        uvs=mesh.uvs,
        organ_ids=mesh.organ_ids[keep],
        segment_ids=mesh.segment_ids[keep],
        organ_meta=mesh.organ_meta,
    )


def render_views(mesh, output_dir, prefix="g3"):
    """Render side/front/top/angle views of a G3Mesh to PNG files.

    Uses offscreen VTK rendering. Saves four images:
    {prefix}_side.png, {prefix}_front.png, {prefix}_top.png, {prefix}_angle.png.

    Args:
        mesh: G3Mesh instance.
        output_dir: Directory to write PNG files.
        prefix: Filename prefix.
    """
    import vtk

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    polydata = mesh.to_vtk_polydata()

    # Recompute normals for consistent shading
    normals_filter = vtk.vtkPolyDataNormals()
    normals_filter.SetInputData(polydata)
    normals_filter.ComputePointNormalsOn()
    normals_filter.ComputeCellNormalsOn()
    normals_filter.AutoOrientNormalsOn()
    normals_filter.ConsistencyOn()
    normals_filter.SplittingOff()
    normals_filter.Update()

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(normals_filter.GetOutput())

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(0.2, 0.7, 0.2)  # green
    actor.GetProperty().BackfaceCullingOff()
    actor.GetProperty().SetAmbient(0.3)
    actor.GetProperty().SetDiffuse(0.7)

    renderer = vtk.vtkRenderer()
    renderer.AddActor(actor)
    renderer.SetBackground(1, 1, 1)

    render_window = vtk.vtkRenderWindow()
    render_window.SetSize(1200, 900)
    render_window.SetOffScreenRendering(1)
    render_window.AddRenderer(renderer)

    # Compute mesh center and bounds for camera positioning
    bounds = polydata.GetBounds()
    cx = (bounds[0] + bounds[1]) / 2
    cy = (bounds[2] + bounds[3]) / 2
    cz = (bounds[4] + bounds[5]) / 2
    dx = bounds[1] - bounds[0]
    dy = bounds[3] - bounds[2]
    dz = bounds[5] - bounds[4]
    max_dim = max(dx, dy, dz, 1e-6)
    dist = max_dim * 2.5

    views = {
        "side":  (cx + dist, cy, cz),
        "front": (cx, cy - dist, cz),
        "top":   (cx, cy, cz + dist),
        "angle": (cx + dist * 0.6, cy - dist * 0.6, cz + dist * 0.6),
    }
    up_vectors = {
        "side":  (0, 0, 1),
        "front": (0, 0, 1),
        "top":   (0, 1, 0),
        "angle": (0, 0, 1),
    }

    w2i = vtk.vtkWindowToImageFilter()
    w2i.SetInput(render_window)

    writer = vtk.vtkPNGWriter()

    for view_name, cam_pos in views.items():
        camera = renderer.GetActiveCamera()
        camera.SetPosition(*cam_pos)
        camera.SetFocalPoint(cx, cy, cz)
        camera.SetViewUp(*up_vectors[view_name])
        renderer.ResetCameraClippingRange()

        render_window.Render()
        w2i.Modified()
        w2i.Update()

        filepath = output_dir / f"{prefix}_{view_name}.png"
        writer.SetFileName(str(filepath))
        writer.SetInputConnection(w2i.GetOutputPort())
        writer.Write()


def _laplacian_smooth(mesh, iterations=3, lambda_factor=0.5):
    """Apply Laplacian smoothing to a G3Mesh, preserving boundary vertices.

    Args:
        mesh: G3Mesh instance.
        iterations: Number of smoothing passes.
        lambda_factor: Step size toward neighbor average (0-1).

    Returns:
        New G3Mesh with smoothed vertex positions and recalculated normals.
    """
    vertices = mesh.vertices.copy()
    indices = mesh.indices
    n_verts = len(vertices)

    # Build adjacency: for each vertex, the set of neighboring vertex indices
    adjacency = [set() for _ in range(n_verts)]
    for tri in indices:
        a, b, c = tri
        adjacency[a].update([b, c])
        adjacency[b].update([a, c])
        adjacency[c].update([a, b])

    # Find boundary vertices: vertices on edges that belong to only 1 triangle
    edge_count = {}
    for tri in indices:
        a, b, c = tri
        for e in [(min(a, b), max(a, b)),
                   (min(b, c), max(b, c)),
                   (min(a, c), max(a, c))]:
            edge_count[e] = edge_count.get(e, 0) + 1

    boundary = set()
    for (v0, v1), count in edge_count.items():
        if count == 1:
            boundary.add(v0)
            boundary.add(v1)

    # Build vertex-to-triangle adjacency for quality checking
    vert_tris = [[] for _ in range(n_verts)]
    for ti, tri in enumerate(indices):
        for v in tri:
            vert_tris[v].append(ti)

    # Minimum triangle area to preserve during smoothing (cm²).
    # Prevents Laplacian smoothing from collapsing narrow triangles
    # at leaf tips into degenerate slivers.
    min_area_smooth = 0.002  # cm²

    # Iterative Laplacian smoothing with quality guard
    for _ in range(iterations):
        new_verts = vertices.copy()
        for i in range(n_verts):
            if i in boundary or not adjacency[i]:
                continue
            neighbors = list(adjacency[i])
            avg = vertices[neighbors].mean(axis=0)
            candidate = vertices[i] + lambda_factor * (avg - vertices[i])

            # Check that moving this vertex wouldn't create any
            # triangle with area below the minimum threshold.
            ok = True
            for ti in vert_tris[i]:
                a, b, c = indices[ti]
                va = candidate if a == i else new_verts[a]
                vb = candidate if b == i else new_verts[b]
                vc = candidate if c == i else new_verts[c]
                area = 0.5 * np.linalg.norm(np.cross(vb - va, vc - va))
                if area < min_area_smooth:
                    ok = False
                    break
            if ok:
                new_verts[i] = candidate
        vertices = new_verts

    # Recalculate per-vertex normals as average of adjacent face normals
    normals = np.zeros_like(vertices)
    for tri in indices:
        a, b, c = tri
        e1 = vertices[b] - vertices[a]
        e2 = vertices[c] - vertices[a]
        fn = np.cross(e1, e2)
        normals[a] += fn
        normals[b] += fn
        normals[c] += fn

    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    lengths = np.maximum(lengths, 1e-12)
    normals /= lengths

    return G3Mesh(vertices, mesh.indices.copy(), normals, mesh.uvs.copy(),
                  mesh.organ_ids.copy(), mesh.segment_ids.copy(),
                  mesh.organ_meta)
