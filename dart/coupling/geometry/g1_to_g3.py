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


COMPACT_OBJ_KWARGS = {"write_normals": False, "write_uvs": False, "precision": 4}


class G3Mesh:
    """Triangle mesh produced by G1-to-G3 lofting.

    Attributes:
        vertices:     (M, 3) float64 vertex positions
        indices:      (K, 3) int32 triangle vertex indices
        normals:      (M, 3) float64 per-vertex normals
        uvs:          (M, 2) float64 UV coordinates
        organ_ids:    (K,)   int32 organ ID per triangle
        segment_ids:  (K,)   int32 original skeleton segment index per triangle
        organ_meta:   list of dicts with organ metadata for mapping export
        quad_indices: (Q, 4) int32 quad vertex indices (optional, for OBJ export)
        quad_organ_ids: (Q,) int32 organ ID per quad (optional)
        organ_cps:    dict mapping ``organ_id -> (N_U, N_V, 3) float64`` canonical
                      NURBS control-point grid. Populated only for leaves lofted
                      with the NURBS backend; absent for quad-ribbon leaves,
                      stems, roots, and sheaths. Consumers (CP-space fitters)
                      must use ``organ_cps.get(oid)`` and fall back when None.
    """

    def __init__(self, vertices, indices, normals, uvs, organ_ids,
                 segment_ids=None, organ_meta=None,
                 quad_indices=None, quad_organ_ids=None,
                 organ_cps=None, is_midrib=None):
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
        self.quad_indices = (np.asarray(quad_indices, dtype=np.int32)
                            if quad_indices is not None else None)
        self.quad_organ_ids = (np.asarray(quad_organ_ids, dtype=np.int32)
                              if quad_organ_ids is not None else None)
        self.organ_cps = dict(organ_cps) if organ_cps else {}
        # Per-triangle midrib flag: True for tris belonging to the central rib
        # band of a leaf. Routed to a separate OBJ sub-group + DART optical
        # property so the midrib gets its own PROSPECT params.
        self.is_midrib = (np.asarray(is_midrib, dtype=bool)
                          if is_midrib is not None
                          else np.zeros(len(self.indices), dtype=bool))

    @property
    def n_vertices(self):
        return len(self.vertices)

    @property
    def n_triangles(self):
        return len(self.indices)

    @property
    def n_quads(self):
        return len(self.quad_indices) if self.quad_indices is not None else 0

    def to_obj(self, filepath, group_by_organ=True, group_prefix="",
               prefer_quads=False, write_materials=False,
               write_normals=True, write_uvs=True, precision=6):
        """Export mesh to Wavefront OBJ format.

        Args:
            filepath: Output .obj file path.
            group_by_organ: If True, write 'g organ_<id>' groups.
            group_prefix: Optional prefix for group names.
            prefer_quads: If True and quad_indices exists, write quads
                instead of triangles for organs that have them.
            write_materials: If True, emit 'usemtl <part_type>' lines
                before each organ group (reads part_type from organ_meta).
            write_normals: If False, skip ``vn`` lines and the ``/n`` face
                index. Normals are recoverable from triangles, so this is a
                lossless size reduction for downstream tools that recompute
                them (DART, Blender, MeshLab).
            write_uvs: If False, skip ``vt`` lines and the ``/uv`` face
                index. Drop only when no per-vertex property mapping is
                consumed downstream.
            precision: Decimal places for ``v``/``vn``/``vt`` floats.
                Default 6 (=10 nm in cm). 4 (=1 µm) is well below any
                DART/RT-relevant scale and ~25 % smaller.
        """
        filepath = Path(filepath)
        use_quads = prefer_quads and self.quad_indices is not None

        # Build organ_id -> part_type lookup from organ_meta
        part_type_map = {}
        if write_materials and self.organ_meta:
            for meta in self.organ_meta:
                part_type_map[meta["organ_id"]] = meta.get("part_type", meta.get("type", "unknown"))

        # Sidecar .mtl with default material colours so the midrib group
        # picks up a contrasting Kd in MeshLab / Paraview without any manual
        # material setup. Only written when write_materials=True.
        mtl_path = None
        if write_materials:
            mtl_path = filepath.with_suffix(".mtl")
            with open(mtl_path, "w") as fmtl:
                fmtl.write("# G1-to-G3 default materials\n")
                # blade: leaf green
                fmtl.write("newmtl blade\nKa 0.05 0.10 0.05\n"
                           "Kd 0.30 0.65 0.20\nKs 0.10 0.10 0.10\nNs 12\n\n")
                fmtl.write("newmtl blade_senescent\nKa 0.10 0.07 0.02\n"
                           "Kd 0.55 0.40 0.10\nKs 0.05 0.05 0.05\nNs 8\n\n")
                # midrib: pale yellow-green ridge — visibly distinct from blade
                fmtl.write("newmtl midrib\nKa 0.10 0.10 0.05\n"
                           "Kd 0.85 0.90 0.45\nKs 0.20 0.20 0.10\nNs 24\n\n")
                fmtl.write("newmtl stem\nKa 0.08 0.10 0.03\n"
                           "Kd 0.45 0.60 0.20\nKs 0.05 0.05 0.05\nNs 8\n\n")
                fmtl.write("newmtl tassel\nKa 0.10 0.07 0.02\n"
                           "Kd 0.70 0.55 0.20\nKs 0.10 0.10 0.05\nNs 16\n")

        p = int(precision)
        v_fmt = f"v {{0:.{p}f}} {{1:.{p}f}} {{2:.{p}f}}\n"
        vn_fmt = f"vn {{0:.{p}f}} {{1:.{p}f}} {{2:.{p}f}}\n"
        vt_fmt = f"vt {{0:.{p}f}} {{1:.{p}f}}\n"
        if write_uvs and write_normals:
            _idx_fmt = "{0}/{0}/{0}"
        elif write_normals:
            _idx_fmt = "{0}//{0}"
        elif write_uvs:
            _idx_fmt = "{0}/{0}"
        else:
            _idx_fmt = "{0}"

        with open(filepath, "w") as f:
            f.write("# G1-to-G3 lofted mesh\n")
            if mtl_path is not None:
                f.write(f"mtllib {mtl_path.name}\n")
            for v in self.vertices:
                f.write(v_fmt.format(v[0], v[1], v[2]))
            if write_normals:
                for n in self.normals:
                    f.write(vn_fmt.format(n[0], n[1], n[2]))
            if write_uvs:
                for uv in self.uvs:
                    f.write(vt_fmt.format(uv[0], uv[1]))

            def _write_face(face):
                parts = " ".join(_idx_fmt.format(v + 1) for v in face)
                f.write(f"f {parts}\n")

            if group_by_organ:
                unique_ids = np.unique(self.organ_ids)
                meta_by_id = {m["organ_id"]: m for m in self.organ_meta
                              if "organ_id" in m}
                for oid in unique_ids:
                    meta = meta_by_id.get(int(oid))
                    meta_name = meta.get("name", "") if meta else ""
                    # Senescent leaves flow through as ``senescent_leaf_N`` so
                    # downstream DART routing can register a withered optical
                    # property — mirrors the tassel prefix flow.
                    if meta_name.startswith((
                        "tassel_spike_", "tassel_branch_", "senescent_leaf_",
                    )):
                        gname = f"{group_prefix}{meta_name}"
                    else:
                        gname = f"{group_prefix}organ_{oid}"
                    if use_quads and self.quad_organ_ids is not None:
                        # Quad-mode mesh has no midrib mask (midrib only on
                        # tris). Write the whole organ as one quad group.
                        f.write(f"g {gname}\n")
                        if write_materials and oid in part_type_map:
                            f.write(f"usemtl {part_type_map[oid]}\n")
                        qmask = self.quad_organ_ids == oid
                        for quad in self.quad_indices[qmask]:
                            _write_face(quad)
                    else:
                        mask = self.organ_ids == oid
                        # Split into blade tris vs. midrib tris (if any).
                        # Midrib gets its own group + ``usemtl midrib`` so
                        # DART / viewers can dispatch a separate property.
                        midrib_mask = mask & self.is_midrib
                        blade_mask = mask & ~self.is_midrib
                        if blade_mask.any():
                            f.write(f"g {gname}\n")
                            if write_materials and oid in part_type_map:
                                f.write(f"usemtl {part_type_map[oid]}\n")
                            for tri in self.indices[blade_mask]:
                                _write_face(tri)
                        if midrib_mask.any():
                            f.write(f"g {gname}_midrib\n")
                            if write_materials:
                                f.write(f"usemtl midrib\n")
                            for tri in self.indices[midrib_mask]:
                                _write_face(tri)
            else:
                if use_quads:
                    for quad in self.quad_indices:
                        _write_face(quad)
                else:
                    for tri in self.indices:
                        _write_face(tri)

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
                "part_type": meta.get("part_type", meta.get("type", "unknown")),
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
    # Raised central midrib: narrow Gaussian bump along frac=0. Combined
    # with `gutter_depths`, gives the maize cross-section: V-channel with
    # a small ridge running along the centerline (real maize).
    # `midrib_amps_cm` is per-skeleton-point amplitude (cm); empty/None disables.
    # `midrib_half_width` is the Gaussian sigma in normalized [-0.5, 0.5] units.
    midrib_amps = organ.get("midrib_amps_cm")
    midrib_half_width = float(organ.get("midrib_half_width", 0.10))

    # H_top analytical invariant (see maize_growth.py:124-132 / learnings
    # §4.1). Would have caught the day-55/day-88 tassel-gap family before
    # the user reported it. Active when the caller supplies H_ins, theta,
    # and lmax; runs once per leaf as a non-fatal warning so it never
    # breaks a production run but flags drift as it happens. Disabled via
    # `organ["check_h_top_invariant"] = False`; tolerance defaults to
    # 5 % of lmax as written in the learnings doc but can be overridden
    # with `organ["h_top_tolerance_cm"]`.
    _h_ins = organ.get("insertion_height_cm")
    _theta = organ.get("theta_rad")
    _lmax = organ.get("lmax_cm")
    if (organ.get("check_h_top_invariant", True) and _h_ins is not None
            and _theta is not None and _lmax is not None
            and _lmax > 0.1 and len(skeleton) > 0):
        _h_measured = float(skeleton[-1, 2])
        _h_predicted = float(_h_ins) + float(np.sin(_theta)) * float(_lmax)
        _tol = float(organ.get("h_top_tolerance_cm", 0.05 * _lmax))
        if abs(_h_measured - _h_predicted) > _tol:
            print(
                f"  [H_top invariant] "
                f"{organ.get('name', f'leaf_{organ_id}')}: "
                f"tip z={_h_measured:.2f} cm vs predicted "
                f"{_h_predicted:.2f} cm (H_ins={_h_ins:.2f}, "
                f"theta={float(np.degrees(_theta)):.1f}°, lmax={_lmax:.1f} cm); "
                f"drift {_h_measured - _h_predicted:+.2f} cm > ±{_tol:.2f} cm"
            )

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
    target_n_verts = organ.get("target_n_verts")
    # Force resampling when target_n_verts is set (vertex count must match OBJ)
    needs_resample = (avg_spacing < min_seg_len and len(skeleton) > 3
                      and total_skel_len > min_seg_len * 2)
    if target_n_verts is not None or organ.get("arc_positions") is not None:
        needs_resample = True
    if needs_resample:
        # Resample at uniform spacing, keeping at least as many points as
        # the original skeleton so no segments are lost in the mapping.
        cum = np.concatenate([[0.0], np.cumsum(np.linalg.norm(
            np.diff(skeleton, axis=0), axis=1))])
        if target_n_verts is not None:
            # n_verts = n_skel * n_cross → n_skel = target_n_verts / n_cross
            # Use explicit target_n_cross if provided, else infer from effects
            _nc = organ.get("target_n_cross")
            if _nc is None:
                _nc = 7 if (organ.get("per_node_displacements") or
                            organ.get("gutter_depths") or
                            organ.get("wave_normal_amp", 0) > 0 or
                            organ.get("curl_amp", 0) > 0) else 2
            n_new = max(3, target_n_verts // _nc)
        else:
            n_new = max(len(skeleton), int(np.ceil(total_skel_len / min_seg_len)) + 1)
        # Non-uniform arc positions: if provided, place skeleton points
        # at specific arc-length fractions (extracted from OBJ grid).
        custom_arc = organ.get("arc_positions")
        if custom_arc is not None:
            custom_arc = np.asarray(custom_arc, dtype=np.float64)
            n_new = len(custom_arc)
            new_arc = custom_arc * total_skel_len  # fractions [0,1] → cm
        else:
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

        if midrib_amps is not None:
            ma = np.asarray(midrib_amps, dtype=np.float64)
            midrib_amps = np.interp(new_arc, cum, ma)

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
    # When target_n_cross is explicit (OBJ fitting), use it directly.
    # Otherwise: 7 for curved cross-sections / deformations, 2 for flat ribbons.
    explicit_n_cross = organ.get("target_n_cross")
    if explicit_n_cross is not None:
        n_cross = explicit_n_cross
    else:
        has_blade_effects = any(organ.get(k, 0) != 0 for k in (
            "wave_normal_amp", "curl_amp", "edge_ruffle_amp", "twist_max"))
        has_midrib = (midrib_amps is not None
                      and float(np.max(np.abs(np.asarray(midrib_amps)))) > 1e-6)
        n_cross = 7 if (gutter_depths is not None
                        or has_blade_effects
                        or has_midrib) else 2

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
    # Non-uniform cross fractions: if provided, use per-row custom fractions
    # (extracted from OBJ grid). Shape: (n_skel, n_cross) or (n_cross,).
    custom_cross = organ.get("cross_fractions")
    if custom_cross is not None:
        custom_cross = np.asarray(custom_cross, dtype=np.float64)
        if custom_cross.ndim == 1:
            cross_fracs = custom_cross
        else:
            cross_fracs = custom_cross[0]  # first row; per-row handled in loop
    else:
        cross_fracs = np.linspace(-0.5, 0.5, n_cross)
    # Gutter profile: parabolic dip (deepest at center, 0 at edges)
    # depth(f) = gutter_depth * (1 - (2*f)^2), where f in [-0.5, 0.5]
    cross_gutter = 1.0 - (2.0 * cross_fracs) ** 2  # 0 at edges, 1 at center

    # --- Per-node displacement field (direct vertex control) ---
    # When present, applies per-skeleton-point displacement vectors that
    # override the parameterized deformation model. Each array has shape
    # (n_skeleton_points,) and is interpolated to the resampled skeleton.
    # Used by fit_lofter_params.py to achieve 1:1 vertex matching.
    per_node_disp = organ.get("per_node_displacements")
    if per_node_disp is not None:
        def _interp_pnd(arr):
            k = len(arr)
            if k == n:
                return np.array(arr, dtype=np.float64)
            src_t = np.linspace(0, 1, k)
            dst_t = np.linspace(0, 1, n)
            return np.interp(dst_t, src_t, arr)

        pnd_normal = _interp_pnd(per_node_disp.get("normal", np.zeros(n)))
        pnd_binormal = _interp_pnd(per_node_disp.get("binormal", np.zeros(n)))
        pnd_tangent = _interp_pnd(per_node_disp.get("tangent", np.zeros(n)))
        # Per-node gutter depth (varies along leaf, not global)
        pnd_gutter = _interp_pnd(per_node_disp.get("gutter", np.zeros(n)))
        # Per-node cross-section V-angle (radians, 0=flat, >0=V-shaped)
        pnd_cs_angle = _interp_pnd(per_node_disp.get("cs_angle", np.zeros(n)))
        # Per-node twist (radians)
        pnd_twist = _interp_pnd(per_node_disp.get("twist", np.zeros(n)))
        # Per-node width multiplier (1.0 = no change)
        pnd_width_mult = _interp_pnd(
            per_node_disp.get("width_mult", np.ones(n)))
        has_per_node = True
        # Bump to 7-cross for surface detail (unless caller locked n_cross)
        if explicit_n_cross is None and n_cross < 7:
            n_cross = 7
            cross_fracs = np.linspace(-0.5, 0.5, n_cross)
            cross_gutter = 1.0 - (2.0 * cross_fracs) ** 2
    else:
        has_per_node = False

    # --- Sheath wrapping parameters ---
    # When present, the leaf base wraps around the stem in a circular arc.
    # sheath_frac: (n,) array, 1.0 at base → 0.0 at blade transition
    # sheath_center: (2,) XY of stem axis
    # sheath_radius: float, arc radius (cm)
    # sheath_wrap_angle: float, total angular extent (radians)
    sheath_params = organ.get("sheath")
    if sheath_params is not None:
        sheath_frac_raw = np.asarray(sheath_params.get("fraction", []), dtype=np.float64)
        if len(sheath_frac_raw) != n:
            src_t = np.linspace(0, 1, len(sheath_frac_raw))
            dst_t = np.linspace(0, 1, n)
            sheath_frac = np.interp(dst_t, src_t, sheath_frac_raw)
        else:
            sheath_frac = sheath_frac_raw
        sheath_center_xy = np.asarray(sheath_params["center_xy"], dtype=np.float64)
        sheath_r = float(sheath_params["radius"])
        sheath_wrap_angle = float(sheath_params.get("wrap_angle", np.radians(345)))
        # Base angle at each skeleton point: direction from stem center to skeleton
        sheath_base_angle = np.arctan2(
            skeleton[:, 1] - sheath_center_xy[1],
            skeleton[:, 0] - sheath_center_xy[0])
    else:
        sheath_frac = None
        sheath_center_xy = None
        sheath_r = 0.0
        sheath_wrap_angle = 0.0
        sheath_base_angle = None

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
    # bltree.lsy-style independent ZLeft / ZRight phases. Default both to the
    # shared phase so legacy organ dicts (no _L/_R keys) stay bit-identical.
    wave_normal_phase_L = organ.get("wave_normal_phase_L", wave_normal_phase)
    wave_normal_phase_R = organ.get("wave_normal_phase_R", wave_normal_phase)
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
    if has_spline_features and explicit_n_cross is None and n_cross < 7:
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
    wave_normal_offsets_L = np.zeros(n)
    wave_normal_offsets_R = np.zeros(n)
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
        if explicit_n_cross is None and n_cross < 7:
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
        # Fitted-CP path has a single `wave_normal` series — mirror it onto
        # the L/R edges so the per-vertex interpolation below stays equivalent
        # to the old rigid-center shift for this code path.
        if 'wave_normal' in fitted_cps:
            wave_normal_offsets_L[:] = wave_normal_offsets
            wave_normal_offsets_R[:] = wave_normal_offsets

    elif has_waves:
        for i in range(n):
            t_frac = arc[i] / total_arc  # 0 at base, 1 at tip
            # Linear ramp: effects start at ramp_onset along the leaf.
            # MaizeField3D data shows onset ~5%, hand-tuned default was 15%.
            ramp = max(0.0, (t_frac - ramp_onset) / (1.0 - ramp_onset))
            ramp_sq = ramp * ramp  # quadratic for twist only

            # ZLeft / ZRight independent phases (bltree.lsy:9-10). The mean
            # is kept in `wave_normal_offsets[i]` so the old path — legacy
            # callers that still read that array — behaves as before when
            # phase_L == phase_R.
            _zl = wave_normal_amp * ramp * np.sin(
                2 * np.pi * wave_normal_freq * t_frac + wave_normal_phase_L)
            _zr = wave_normal_amp * ramp * np.sin(
                2 * np.pi * wave_normal_freq * t_frac + wave_normal_phase_R)
            wave_normal_offsets_L[i] = _zl
            wave_normal_offsets_R[i] = _zr
            wave_normal_offsets[i] = 0.5 * (_zl + _zr)
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

        if has_per_node:
            # Per-node displacement: direct control over each skeleton point
            center += pnd_normal[i] * normal
            center += pnd_binormal[i] * binormal
            center += pnd_tangent[i] * tangents[i]
            gd = pnd_gutter[i]
            w = w * pnd_width_mult[i]
            # Apply per-node twist
            if abs(pnd_twist[i]) > 1e-6:
                ca = np.cos(pnd_twist[i])
                sa = np.sin(pnd_twist[i])
                bn_tw = ca * binormal + sa * normal
                nm_tw = -sa * binormal + ca * normal
                binormal = bn_tw / max(np.linalg.norm(bn_tw), 1e-12)
                normal = nm_tw / max(np.linalg.norm(nm_tw), 1e-12)
        else:
            # Apply blade waviness: lateral stays at center (symmetric), but
            # the vertical undulation is now applied per-vertex below so
            # ZLeft/ZRight can differ independently (bltree.lsy:9-10). When
            # phase_L == phase_R (legacy fallback) the per-vertex value equals
            # the old rigid center shift at every cross-section point.
            if has_waves:
                center += wave_lateral_offsets[i] * binormal
            gd = 0.0
            if gutter_depths is not None:
                gd_idx = min(i, len(gutter_depths) - 1)
                gd = gutter_depths[gd_idx] if gd_idx >= 0 else 0.0

        # Per-row cross fractions when custom_cross is 2D
        if custom_cross is not None and custom_cross.ndim == 2 and i < len(custom_cross):
            row_fracs = custom_cross[i]
        else:
            row_fracs = cross_fracs

        # Sheath wrapping: at the leaf base, cross-section follows a
        # circular arc around the stem instead of a straight line.
        sheath_f = sheath_frac[i] if sheath_frac is not None else 0.0

        for j in range(n_cross):
            frac = row_fracs[j]  # -0.5 to 0.5

            if sheath_f > 0.01:
                # Sheath mode: place vertex on circular arc around stem
                angle = sheath_base_angle[i] + frac * sheath_wrap_angle * sheath_f
                sheath_pos = np.array([
                    sheath_center_xy[0] + sheath_r * np.cos(angle),
                    sheath_center_xy[1] + sheath_r * np.sin(angle),
                    skeleton[i, 2]
                ])
                flat_pos = center + frac * w * binormal
                # Smooth blend between sheath arc and flat blade
                lateral_pos = sheath_f * sheath_pos + (1.0 - sheath_f) * flat_pos
                lateral = lateral_pos - center
            else:
                lateral = frac * w * binormal

            if has_per_node:
                # Per-node gutter + cross-section V-angle
                gutter_sign = -1.0 if normal[2] >= 0 else 1.0
                gutter_offset = gutter_sign * gd * cross_gutter[j] * normal
                # Cross-section V-angle: parabolic transverse curvature
                cs_offset = np.zeros(3)
                if abs(pnd_cs_angle[i]) > 0.01:
                    cs_profile = (2.0 * frac) ** 2
                    cs_offset = pnd_cs_angle[i] * cs_profile * normal
                # Geometric midrib ridge: removed. The painted stripe is
                # carried by the midrib material tag on the original surface,
                # so the cross-section keeps a clean gutter U with no
                # centerline relief.
                midrib_offset = np.zeros(3)
                v_idx = n_cross * i + j
                vertices[v_idx] = (center + lateral + gutter_offset
                                   + cs_offset + midrib_offset)
            else:
                # Original parametric deformation model
                gutter_sign = -1.0 if normal[2] >= 0 else 1.0
                gutter_offset = gutter_sign * gd * cross_gutter[j] * normal
                curl_offset = (2.0 * frac) * curl_factors[i] * normal
                edge_frac = (2.0 * abs(frac)) ** 2
                ruffle_offset = edge_frac * edge_ruffle_base[i] * normal
                if frac < 0:
                    ruffle_offset *= -1.0
                fold_profile = np.sin(np.pi * abs(2.0 * frac))
                fold_offset = fold_factors[i] * fold_profile * normal

                # Per-vertex vertical-undulation offset interpolating L/R
                # phases across the blade width. frac in [-0.5, +0.5] maps
                # to [L, R]; midrib (frac=0) gets the mean. When the two
                # phases match (legacy fallback), this reduces to the old
                # rigid center shift.
                wave_frac = 0.5 + frac
                wave_normal_vertex = (
                    (1.0 - wave_frac) * wave_normal_offsets_L[i]
                    + wave_frac * wave_normal_offsets_R[i]
                )
                wave_normal_offset = wave_normal_vertex * normal

                w_fade = w / max_w
                curl_offset *= w_fade
                ruffle_offset *= w_fade
                fold_offset *= w_fade
                gutter_offset *= w_fade

                asym_offset = np.zeros(3)
                if asymmetry_values is not None:
                    asym_offset = asymmetry_values[i] * binormal * w_fade

                edge_curl_offset = np.zeros(3)
                if edge_curl_values is not None:
                    edge_influence = (2.0 * abs(frac)) ** 3
                    edge_curl_offset = (edge_influence * np.tan(edge_curl_values[i])
                                        * w * 0.5 * normal * w_fade)

                cs_offset = np.zeros(3)
                if cross_section_values is not None:
                    cs_profile = (2.0 * frac) ** 2
                    cs_offset = cross_section_values[i] * cs_profile * normal * w_fade

                # Geometric midrib ridge: removed. The material tag carries
                # the stripe on the original surface.
                midrib_offset = np.zeros(3)

                v_idx = n_cross * i + j
                vertices[v_idx] = (center + lateral + gutter_offset
                                   + curl_offset + ruffle_offset + fold_offset
                                   + asym_offset + edge_curl_offset + cs_offset
                                   + wave_normal_offset + midrib_offset)

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
    # Midrib triangle mask: True where the strip falls within the central
    # band of the cross-section (Stage B optical routing). Uses a v-band
    # half-width in normalised [-0.5, 0.5] frac coordinates.
    _band_input = organ.get("midrib_band_v_frac", midrib_half_width)
    if np.isscalar(_band_input):
        midrib_band_v = float(_band_input)
    else:
        # Per-skeleton-point band — quad-ribbon currently uses a single
        # scalar test (cross_fracs is constant per cross-section), so use
        # the array's mean. NURBS path uses the full per-u taper.
        midrib_band_v = float(np.mean(np.asarray(_band_input)))
    is_midrib_tri = np.zeros(n_segs * n_tris_per_seg, dtype=bool)
    has_midrib_active = (midrib_amps is not None
                         and float(np.max(np.abs(np.asarray(midrib_amps)))) > 1e-6)
    # Per-row amplitude gate: the adapter's basal ramp drives midrib_amps
    # to ~0 near the collar, but the original tagging was a single
    # organ-level boolean and therefore painted the optical stripe across
    # the full arc — including the basal ramp region that has no
    # geometric ridge. Skip rows whose local amplitude is below 1 % of
    # the organ max so the painted stripe matches the geometric ridge.
    if has_midrib_active:
        _amp_arr = np.asarray(midrib_amps, dtype=np.float64)
        _amp_threshold = 0.01 * float(np.max(np.abs(_amp_arr)))
    else:
        _amp_arr = None
        _amp_threshold = 0.0

    for i in range(n_segs):
        if has_midrib_active:
            _row_amp = 0.5 * (abs(_amp_arr[i]) + abs(_amp_arr[i + 1]))
            _row_active = _row_amp > _amp_threshold
        else:
            _row_active = False
        for j in range(n_cross - 1):
            bl = n_cross * i + j
            br = n_cross * i + j + 1
            tl = n_cross * (i + 1) + j
            tr = n_cross * (i + 1) + j + 1
            tri_base = i * n_tris_per_seg + 2 * j
            indices[tri_base] = [bl, tl, br]
            indices[tri_base + 1] = [br, tl, tr]
            if _row_active:
                # Strip j-(j+1) is "midrib" when both rails sit inside the
                # band. Use the average |frac| of the two rails as the test.
                mean_abs_frac = 0.5 * (abs(cross_fracs[j])
                                       + abs(cross_fracs[j + 1]))
                if mean_abs_frac <= midrib_band_v:
                    is_midrib_tri[tri_base] = True
                    is_midrib_tri[tri_base + 1] = True

    # Build quad indices (same grid, one quad per cell instead of 2 tris)
    emit_quads = organ.get("emit_quads", False)
    if emit_quads:
        n_quads_per_seg = n_cross - 1
        quad_indices = np.empty((n_segs * n_quads_per_seg, 4), dtype=np.int32)
        for i in range(n_segs):
            for j in range(n_cross - 1):
                bl = n_cross * i + j
                br = n_cross * i + j + 1
                tl = n_cross * (i + 1) + j
                tr = n_cross * (i + 1) + j + 1
                quad_indices[i * n_quads_per_seg + j] = [bl, br, tr, tl]
    else:
        quad_indices = None

    organ_ids = np.full(len(indices), organ_id, dtype=np.int32)
    quad_organ_ids = (np.full(len(quad_indices), organ_id, dtype=np.int32)
                      if quad_indices is not None else None)

    orig_seg_map = organ.get("_orig_segment_map")
    segment_ids = np.empty(len(indices), dtype=np.int32)
    for i in range(n_segs):
        if orig_seg_map is not None:
            sid = int(orig_seg_map[i])
        else:
            sid = i
        for k in range(n_tris_per_seg):
            segment_ids[i * n_tris_per_seg + k] = sid

    # Trim to exact vertex count (for prime OBJ vertex counts where
    # n_skel * n_cross overshoots by 1-2 vertices).  Remove the excess
    # tip vertices and any triangles that reference them.
    trim_n = organ.get("trim_to_n_verts")
    if trim_n is not None and len(vertices) > trim_n:
        keep_mask = np.all(indices < trim_n, axis=1)
        indices = indices[keep_mask]
        organ_ids = organ_ids[keep_mask]
        segment_ids = segment_ids[keep_mask]
        is_midrib_tri = is_midrib_tri[keep_mask]
        if quad_indices is not None:
            q_keep = np.all(quad_indices < trim_n, axis=1)
            quad_indices = quad_indices[q_keep]
            quad_organ_ids = quad_organ_ids[q_keep]
        vertices = vertices[:trim_n]
        normals_arr = normals_arr[:trim_n]
        uvs = uvs[:trim_n]

    # Per-vertex displacement: direct (M, 3) offsets for 1:1 OBJ matching.
    # Applied after trim so the displacement array size matches final vertices.
    per_vert_disp = organ.get("per_vertex_displacements")
    if per_vert_disp is not None:
        pvd = np.asarray(per_vert_disp, dtype=np.float64)
        if pvd.shape == vertices.shape:
            vertices += pvd

    # Reference faces: use OBJ face connectivity instead of lofter grid.
    # Stored as pipeline data during fitting; replayed at generation time.
    # Supports mixed tris (3-vert) and quads (4-vert).
    ref_faces = organ.get("reference_faces")
    if ref_faces is not None:
        ref_quads = [f for f in ref_faces if len(f) == 4]
        ref_tris = [f for f in ref_faces if len(f) == 3]
        if ref_quads:
            quad_indices = np.array(ref_quads, dtype=np.int32)
            quad_organ_ids = np.full(len(ref_quads), organ_id, dtype=np.int32)
        # Replace tri indices with reference tris + triangulated quads
        all_tris = list(ref_tris)
        for q in ref_quads:
            all_tris.append([q[0], q[1], q[2]])
            all_tris.append([q[0], q[2], q[3]])
        if all_tris:
            indices = np.array(all_tris, dtype=np.int32)
            organ_ids = np.full(len(indices), organ_id, dtype=np.int32)
            segment_ids = np.full(len(indices), 0, dtype=np.int32)
            # Reference-faces path bypasses the lofter grid, so we can't
            # tag midrib tris from cross_fracs. Default to no midrib here.
            is_midrib_tri = np.zeros(len(indices), dtype=bool)

    return (vertices, indices, normals_arr, uvs, organ_ids, segment_ids,
            quad_indices, quad_organ_ids, is_midrib_tri)


def _stem_internode_width_scale_at_z(zi, node_heights_z,
                                     node_bulge=0.12, node_band_cm=1.0,
                                     groove_depth=0.04, groove_width_cm=0.8,
                                     internode_barrel=0.02):
    """Return the rendered stem width scale from internode modulation at z."""
    if not node_heights_z or len(node_heights_z) < 2:
        return 1.0

    nodes_z = sorted(node_heights_z)

    # Node bulge: raised-cosine bump centered at each leaf attachment
    # height. Each bump is smooth and zero outside ±node_band_cm.
    node_bump = 0.0
    for nz in nodes_z:
        dist = abs(float(zi) - nz)
        if dist < node_band_cm:
            bump = 0.5 * (1.0 + np.cos(np.pi * dist / node_band_cm))
            node_bump = max(node_bump, bump)

    # Groove: shallow constriction just above and below each node.
    groove = 0.0
    for nz in nodes_z:
        dist = abs(float(zi) - nz)
        groove_center = node_band_cm + groove_width_cm * 0.5
        groove_dist = abs(dist - groove_center)
        if groove_dist < groove_width_cm * 0.5:
            g = 0.5 * (
                1.0 + np.cos(np.pi * groove_dist / (groove_width_cm * 0.5))
            )
            groove = max(groove, g)

    # Internode barrel: very subtle outward bow at internode midpoints.
    barrel = 0.0
    if node_bump < 0.05 and groove < 0.05:
        seg_idx = np.searchsorted(nodes_z, float(zi), side='right') - 1
        if 0 <= seg_idx < len(nodes_z) - 1:
            z_lo = nodes_z[seg_idx]
            z_hi = nodes_z[seg_idx + 1]
            dz = z_hi - z_lo
            if dz > 3.0:
                t = (float(zi) - z_lo) / dz
                barrel = np.sin(np.pi * t)

    return (
        1.0 + node_bulge * node_bump
        - groove_depth * groove
        + internode_barrel * barrel
    )


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
    for i in range(len(z)):
        zi = z[i]
        scale = _stem_internode_width_scale_at_z(
            zi, node_heights_z,
            node_bulge=node_bulge,
            node_band_cm=node_band_cm,
            groove_depth=groove_depth,
            groove_width_cm=groove_width_cm,
            internode_barrel=internode_barrel,
        )
        modulated[i] = widths[i] * scale

    return modulated


def _clip_stem_above_top_leaf(organ, pad=0.0, min_stub=1.0):
    """Trim transient young-plant stem stub above the topmost leaf.

    CPlantBox elongates the stem past the last emerged leaf while the
    next phytomer waits on its phyllochron delay — rendering a bare
    stub for young plants. Real maize has the pseudostem and leaf
    sheaths wrapping the apex; there is no visible bare stem above the
    last leaf. This helper truncates the skeleton at
    ``max(node_heights_z) + pad`` so the sheath mesh at the top leaf
    becomes the visual end of the shoot.

    Returns the (possibly modified) organ dict. Skipped when no leaves
    are attached, the stub is below ``min_stub``, or the skeleton is
    too short.
    """
    # Tassel organs are positioned above the canopy on purpose — their
    # skeleton is the tassel itself, not a bare stem stub. Do not clip.
    if organ.get("name", "").startswith(("tassel_spike_", "tassel_branch_")):
        return organ

    node_heights_z = organ.get("node_heights_z")
    if not node_heights_z:
        return organ

    skeleton = np.asarray(organ["skeleton"], dtype=np.float64)
    widths = np.asarray(organ["widths"], dtype=np.float64)
    if len(skeleton) < 2:
        return organ

    z_apex = float(skeleton[-1, 2])
    z_top_leaf = float(max(node_heights_z))
    if z_apex - z_top_leaf < min_stub:
        return organ

    z_clip = z_top_leaf + pad
    # Honor an opt-in minimum clip height set by the adapter when a tassel
    # is attached — prevents amputating the mainstem below the tassel base.
    z_floor = organ.get("no_clip_above_z")
    if z_floor is not None:
        z_clip = max(z_clip, float(z_floor))
    if z_clip >= z_apex:
        return organ

    z_skel = skeleton[:, 2]
    below = np.where(z_skel <= z_clip)[0]
    if len(below) == 0:
        return organ
    last_idx = int(below[-1])
    if last_idx >= len(skeleton) - 1:
        return organ

    z1 = z_skel[last_idx]
    z2 = z_skel[last_idx + 1]
    if z2 > z1 + 1e-9:
        t = (z_clip - z1) / (z2 - z1)
        clip_pos = skeleton[last_idx] + t * (skeleton[last_idx + 1] - skeleton[last_idx])
        clip_w = widths[last_idx] + t * (widths[last_idx + 1] - widths[last_idx])
    else:
        clip_pos = skeleton[last_idx].copy()
        clip_w = widths[last_idx]

    new_skel = np.vstack([skeleton[: last_idx + 1], clip_pos[None, :]])
    new_widths = np.concatenate([widths[: last_idx + 1], [clip_w]])

    new_organ = dict(organ, skeleton=new_skel, widths=new_widths)
    # Preserve per-segment mapping: new clipped skeleton has (last_idx + 1)
    # segments; the last one still originates from subdivided segment
    # ``last_idx`` (now cut short).
    orig_map = organ.get("_orig_segment_map")
    if orig_map is not None:
        new_organ["_orig_segment_map"] = np.asarray(
            orig_map[: last_idx + 1], dtype=np.int32,
        )
    return new_organ


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

    # Floor tube width so wall triangles at tapered tips survive the
    # downstream degenerate-triangle filter (min area 0.001 cm²).  Without
    # this, tassel branches with 0.02 cm diameter tips get amputated 0.5–
    # 1.8 cm short of their skeleton tip, while the anther billboards (which
    # use the full skeleton) extend past the remaining tube.
    widths = np.maximum(widths, 0.08)

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
                smooth=True, smooth_iterations=3, use_nurbs_backend=False,
                nurbs_n_u_eval=30, nurbs_n_v_eval=21,
                with_tassel_billboards=True, tassel_billboard_seed=42):
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
        use_nurbs_backend: If True, loft leaves with the canonical 11x5
            PlantGL ``NurbsPatch`` backend (`nurbs_blade.loft_leaf_nurbs`)
            instead of the legacy quad-ribbon lofter. Per-organ override:
            set ``organ["use_nurbs_backend"] = True/False`` to opt in/out.
        nurbs_n_u_eval, nurbs_n_v_eval: Tessellation resolution for the
            NURBS backend (30x7 default).
        with_tassel_billboards: If True and any organs have names starting
            with ``tassel_spike_`` or ``tassel_branch_``, append anther
            cross-billboards to the mesh. No-op when no tassel organs are
            present. Runs after smoothing and before degenerate-triangle
            removal so over-thin anthers are culled too.
        tassel_billboard_seed: RNG seed for billboard jitter determinism.

    Returns:
        G3Mesh with all organs combined.
    """
    all_verts = []
    all_indices = []
    all_normals = []
    all_uvs = []
    all_organ_ids = []
    all_segment_ids = []
    all_quad_indices = []
    all_quad_organ_ids = []
    all_midrib = []  # per-organ per-tri bool masks; concat into mesh.is_midrib
    organ_cps: dict = {}
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
            "part_type": organ.get("part_type", organ["type"]),
            "node_ids": node_ids,
            "arc_lengths": orig_arc_norm,
        })

        # Optionally subdivide coarse skeletons before lofting.
        # Leaves carrying a pre-fitted surface_cps_local grid bypass
        # subdivision entirely — the NURBS backend consumes the library CPs
        # directly and uses the original skeleton only for per-triangle
        # segment-ID mapping (identity map).
        if "surface_cps_local" in organ:
            organ = dict(organ)
            orig_skel_n = len(np.asarray(organ["skeleton"], dtype=np.float64))
            organ["_orig_segment_map"] = np.arange(
                max(0, orig_skel_n - 1), dtype=np.int32,
            )
        elif subdivide:
            skel, wid, orig_seg_map = _subdivide_skeleton(
                organ["skeleton"], organ["widths"], target_spacing=target_spacing
            )
            organ = dict(organ, skeleton=skel, widths=wid,
                         _orig_segment_map=orig_seg_map)
        else:
            organ = dict(organ)

        otype = organ["type"]
        qidxs, qoids = None, None
        midrib_tags = None  # per-tri bool array; only set by leaf lofters
        if otype == "leaf":
            # Per-organ opt-in overrides the global flag.
            use_nurbs = organ.get("use_nurbs_backend", use_nurbs_backend)
            if use_nurbs:
                from .nurbs_blade import loft_leaf_nurbs
                result = loft_leaf_nurbs(
                    organ, n_u_eval=nurbs_n_u_eval, n_v_eval=nurbs_n_v_eval,
                )
                # NURBS backend returns a tuple with the canonical CP grid
                # as element 8 and the midrib mask as element 9 (when set).
                organ_cps[int(organ["organ_id"])] = np.asarray(
                    result[8], dtype=np.float64
                )
                if len(result) > 9:
                    midrib_tags = np.asarray(result[9], dtype=bool)
            else:
                result = _loft_leaf(organ)
                if len(result) > 8:
                    midrib_tags = np.asarray(result[8], dtype=bool)
            verts, idxs, norms, uvs, oids, sids = result[:6]
            qidxs, qoids = result[6], result[7]
        elif otype == "sheath":
            from .sheath_mesher import mesh_sheath
            verts, idxs, norms, uvs, oids, sids = mesh_sheath(
                skeleton=organ["skeleton"],
                radii=organ.get("radii", organ["widths"] / 2.0),
                wrap_angle=organ.get("wrap_angle", np.radians(330)),
                overlap_angle=organ.get("overlap_angle", np.radians(30)),
                thickness=organ.get("sheath_thickness", 0.04),
                stem_skeleton=organ.get("stem_skeleton"),
                organ_id=organ["organ_id"],
            )
        elif otype in ("stem", "root"):
            # Clip transient young-plant stem stub so the topmost sheath
            # geometry (already wrapping the stem just below) is the visual
            # end of the shoot — real maize has no bare stem poking above
            # the last leaf. No cap/cone is added; the top disc is covered
            # by the sheath mesh at the top leaf.
            if otype == "stem":
                organ = _clip_stem_above_top_leaf(organ, pad=0.0)
            verts, idxs, norms, uvs, oids, sids = _loft_stem(
                organ, n_sides=stem_sides,
            )
        else:
            raise ValueError(f"Unknown organ type: {otype!r}")

        all_verts.append(verts)
        all_indices.append(idxs + vertex_offset)
        all_normals.append(norms)
        all_uvs.append(uvs)
        all_organ_ids.append(oids)
        all_segment_ids.append(sids)
        # Per-tri midrib mask: leaves emit a real bool array (some True if
        # the organ has midrib_amps_cm > 0); other organs are all-False.
        if midrib_tags is not None and len(midrib_tags) == len(idxs):
            all_midrib.append(midrib_tags)
        else:
            all_midrib.append(np.zeros(len(idxs), dtype=bool))
        if qidxs is not None:
            all_quad_indices.append(qidxs + vertex_offset)
            all_quad_organ_ids.append(qoids)
        vertex_offset += len(verts)

    if not all_verts:
        return G3Mesh(
            np.empty((0, 3)), np.empty((0, 3), dtype=np.int32),
            np.empty((0, 3)), np.empty((0, 2)), np.empty(0, dtype=np.int32),
            organ_cps=organ_cps,
        )

    quad_idx = (np.concatenate(all_quad_indices)
                if all_quad_indices else None)
    quad_oid = (np.concatenate(all_quad_organ_ids)
                if all_quad_organ_ids else None)

    mesh = G3Mesh(
        vertices=np.concatenate(all_verts),
        indices=np.concatenate(all_indices),
        normals=np.concatenate(all_normals),
        uvs=np.concatenate(all_uvs),
        organ_ids=np.concatenate(all_organ_ids),
        segment_ids=np.concatenate(all_segment_ids),
        organ_meta=organ_meta,
        quad_indices=quad_idx,
        quad_organ_ids=quad_oid,
        organ_cps=organ_cps,
        is_midrib=np.concatenate(all_midrib) if all_midrib else None,
    )

    if smooth:
        mesh = _laplacian_smooth(mesh, iterations=smooth_iterations)

    if with_tassel_billboards:
        has_tassel = any(
            o.get("name", "").startswith(("tassel_spike_", "tassel_branch_"))
            for o in organs
        )
        if has_tassel:
            from .tassel_billboards import append_tassel_billboards
            append_tassel_billboards(mesh, organs,
                                     seed=tassel_billboard_seed,
                                     verbose=True)

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
        quad_indices=mesh.quad_indices,
        quad_organ_ids=mesh.quad_organ_ids,
        organ_cps=mesh.organ_cps,
        is_midrib=mesh.is_midrib[keep],
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

    fixed_vertices = set(boundary)
    fixed_organ_ids = {
        int(meta["organ_id"])
        for meta in mesh.organ_meta
        if meta.get("type") in ("stem", "root")
    }
    if fixed_organ_ids:
        fixed_tri_mask = np.isin(mesh.organ_ids, list(fixed_organ_ids))
        if np.any(fixed_tri_mask):
            fixed_vertices.update(mesh.indices[fixed_tri_mask].ravel().tolist())

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
            if i in fixed_vertices or not adjacency[i]:
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
                  mesh.organ_meta,
                  quad_indices=mesh.quad_indices,
                  quad_organ_ids=mesh.quad_organ_ids,
                  organ_cps=mesh.organ_cps,
                  is_midrib=mesh.is_midrib.copy())
