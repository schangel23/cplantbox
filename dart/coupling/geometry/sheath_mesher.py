"""Sheath mesh generator: partial-arc cylindrical sweep for maize leaf sheaths.

Sheaths wrap ~330 degrees around the stem as a cylindrical arc with an
overlap strip where the edges pass over each other.  Uses the same
parallel-transport ring-sweep approach as _loft_stem() in g1_to_g3.py,
but with a partial-arc cross-section instead of a full circle.

Pure numpy implementation (no PlantGL dependency).
"""

import numpy as np


def mesh_sheath(
    skeleton,
    radii,
    wrap_angle=5.76,
    overlap_angle=0.52,
    thickness=0.04,
    n_arc=24,
    stem_skeleton=None,
    organ_id=0,
):
    """Generate a triangle mesh for a leaf sheath organ.

    Creates a partial-cylinder mesh by sweeping a partial-arc cross-section
    along the sheath axis (skeleton), with an overlap strip offset outward
    by `thickness` to model the natural scroll where sheath edges pass
    over each other.

    Args:
        skeleton: (N, 3) array of sheath axis points (base to collar).
        radii: (N,) array of radius at each axis point (taper: wider at
            base, narrower at collar).
        wrap_angle: Main arc extent in radians (~330 deg = 5.76 rad).
        overlap_angle: Overlap strip extent in radians (~30 deg = 0.52 rad).
        thickness: Radial offset for overlap strip in cm.
        n_arc: Number of vertices along the main arc cross-section.
        stem_skeleton: (M, 3) optional stem axis for frame alignment.
            If provided, the initial binormal is aligned radially away
            from the stem.
        organ_id: Integer organ ID for DART triangle tagging.

    Returns:
        Tuple of (vertices, indices, normals, uvs, organ_ids, segment_ids):
            vertices:    (V, 3) float64
            indices:     (T, 3) int32
            normals:     (V, 3) float64
            uvs:         (V, 2) float64
            organ_ids:   (T,)   int32
            segment_ids: (T,)   int32
    """
    skeleton = np.asarray(skeleton, dtype=np.float64)
    radii = np.asarray(radii, dtype=np.float64)
    n_axis = len(skeleton)

    if n_axis < 2:
        return _empty_mesh()

    # Skip sheaths with negligible arc length
    arc_len = np.sum(np.linalg.norm(np.diff(skeleton, axis=0), axis=1))
    if arc_len < 0.5:
        return _empty_mesh()

    # Build cross-section profile
    cross_section = _build_arc_cross_section(
        wrap_angle, overlap_angle, n_arc
    )
    n_cross = len(cross_section)

    # Sweep cross-section along skeleton
    vertices, normals, uvs = _sweep_cross_section(
        skeleton, radii, cross_section, stem_skeleton, thickness
    )

    # Triangulate between consecutive rings
    indices, organ_ids, segment_ids = _triangulate_sweep(
        n_axis, n_cross, organ_id
    )

    return vertices, indices, normals, uvs, organ_ids, segment_ids


def _empty_mesh():
    """Return empty arrays for skipped sheaths."""
    return (
        np.empty((0, 3), dtype=np.float64),
        np.empty((0, 3), dtype=np.int32),
        np.empty((0, 3), dtype=np.float64),
        np.empty((0, 2), dtype=np.float64),
        np.empty(0, dtype=np.int32),
        np.empty(0, dtype=np.int32),
    )


def _build_arc_cross_section(wrap_angle, overlap_angle, n_arc):
    """Build 2D cross-section points for the main arc + overlap strip.

    The main arc spans [0, wrap_angle] with n_arc points.
    The overlap strip continues beyond wrap_angle for overlap_angle.
    Thickness offset is applied during the sweep step, not here.

    Args:
        wrap_angle: Main arc extent in radians.
        overlap_angle: Overlap strip extent in radians.
        n_arc: Number of points along the main arc.

    Returns:
        cross_section: (n_total, 2) array where column 0 = angle,
            column 1 = 0.0 for main arc / 1.0 for overlap strip.
    """
    # Main arc points
    main_angles = np.linspace(0.0, wrap_angle, n_arc)

    # Overlap strip: 3-4 points past the main arc, offset outward
    n_overlap = max(3, int(n_arc * overlap_angle / wrap_angle))
    overlap_angles = np.linspace(
        wrap_angle, wrap_angle + overlap_angle, n_overlap + 1
    )[1:]  # skip the boundary (duplicated from main arc end)

    all_angles = np.concatenate([main_angles, overlap_angles])
    n_total = len(all_angles)

    # Store as (angle, is_overlap) pairs encoded in a 2D array:
    # column 0 = angle, column 1 = 0 for main, 1 for overlap
    cross_section = np.zeros((n_total, 2))
    cross_section[:, 0] = all_angles
    cross_section[n_arc:, 1] = 1.0  # flag overlap points

    return cross_section


def _sweep_cross_section(skeleton, radii, cross_section, stem_skeleton,
                         thickness=0.04):
    """Sweep cross-section along skeleton to produce vertices, normals, UVs.

    Uses parallel transport for stable frame propagation (same approach
    as _loft_stem in g1_to_g3.py).

    Args:
        skeleton: (N, 3) axis points.
        radii: (N,) radius at each axis point.
        cross_section: (n_cross, 2) from _build_arc_cross_section.
        stem_skeleton: Optional (M, 3) stem axis for initial frame.
        thickness: Radial offset for overlap strip in cm.

    Returns:
        vertices: (N * n_cross, 3)
        normals:  (N * n_cross, 3)
        uvs:      (N * n_cross, 2)
    """
    n_axis = len(skeleton)
    n_cross = len(cross_section)
    angles = cross_section[:, 0]
    is_overlap = cross_section[:, 1] > 0.5

    # Compute tangents via central differences
    tangents = np.empty_like(skeleton)
    tangents[0] = skeleton[1] - skeleton[0]
    tangents[-1] = skeleton[-1] - skeleton[-2]
    if n_axis > 2:
        tangents[1:-1] = skeleton[2:] - skeleton[:-2]
    lengths = np.linalg.norm(tangents, axis=1, keepdims=True)
    lengths = np.maximum(lengths, 1e-12)
    tangents /= lengths

    # Arc-length parameter for UV mapping
    diffs = np.diff(skeleton, axis=0)
    seg_lens = np.linalg.norm(diffs, axis=1)
    cum_arc = np.concatenate([[0.0], np.cumsum(seg_lens)])
    total_arc = cum_arc[-1]
    if total_arc < 1e-12:
        u_param = np.linspace(0.0, 1.0, n_axis)
    else:
        u_param = cum_arc / total_arc

    # V parameter: fraction around arc (0 = start of arc, 1 = end including overlap)
    total_angle = angles[-1] - angles[0]
    if total_angle < 1e-12:
        v_param = np.linspace(0.0, 1.0, n_cross)
    else:
        v_param = (angles - angles[0]) / total_angle

    # Initial binormal: prefer radial direction from stem
    up = np.array([0.0, 0.0, 1.0])
    fallback = np.array([1.0, 0.0, 0.0])

    if stem_skeleton is not None and len(stem_skeleton) >= 2:
        # Find closest stem point to sheath base
        stem_pts = np.asarray(stem_skeleton, dtype=np.float64)
        dists = np.linalg.norm(stem_pts - skeleton[0], axis=1)
        closest_idx = np.argmin(dists)
        stem_center = stem_pts[closest_idx]
        radial = skeleton[0] - stem_center
        radial[2] = 0.0  # project to horizontal for consistent orientation
        radial_len = np.linalg.norm(radial)
        if radial_len > 1e-6:
            binormal = radial / radial_len
            # Ensure perpendicular to first tangent
            binormal = binormal - np.dot(binormal, tangents[0]) * tangents[0]
            bn_len = np.linalg.norm(binormal)
            if bn_len > 1e-6:
                binormal /= bn_len
            else:
                binormal = np.cross(tangents[0], up)
                binormal /= np.linalg.norm(binormal)
        else:
            binormal = np.cross(tangents[0], up)
            bn_len = np.linalg.norm(binormal)
            if bn_len < 1e-6:
                binormal = np.cross(tangents[0], fallback)
            binormal /= np.linalg.norm(binormal)
    else:
        binormal = np.cross(tangents[0], up)
        bn_len = np.linalg.norm(binormal)
        if bn_len < 1e-6:
            binormal = np.cross(tangents[0], fallback)
        binormal /= np.linalg.norm(binormal)

    # Allocate output arrays
    vertices = np.empty((n_axis * n_cross, 3))
    normals_arr = np.empty((n_axis * n_cross, 3))
    uvs = np.empty((n_axis * n_cross, 2))

    for i in range(n_axis):
        t = tangents[i]
        radius = radii[i]

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
        for j in range(n_cross):
            a = angles[j]
            direction = np.cos(a) * binormal + np.sin(a) * normal

            # Overlap points offset outward by thickness
            r = radius + thickness if is_overlap[j] else radius

            idx = i * n_cross + j
            vertices[idx] = center + r * direction
            normals_arr[idx] = direction  # outward-pointing
            uvs[idx] = [u_param[i], v_param[j]]

    return vertices, normals_arr, uvs


def _triangulate_sweep(n_axis, n_cross, organ_id):
    """Build triangle indices between consecutive rings.

    Same strip pattern as _loft_stem: for each pair of consecutive
    axis positions and consecutive cross-section positions, create
    two triangles forming a quad.

    Note: unlike _loft_stem, the arc is NOT closed — vertex j and
    vertex 0 are NOT connected (partial arc, not full circle).

    Args:
        n_axis: Number of skeleton points.
        n_cross: Number of cross-section points per ring.
        organ_id: Organ ID for all triangles.

    Returns:
        indices:     (T, 3) int32
        organ_ids:   (T,)   int32
        segment_ids: (T,)   int32
    """
    n_segments = n_axis - 1
    n_strip = n_cross - 1  # partial arc: no wrap-around
    n_tris = 2 * n_segments * n_strip

    indices = np.empty((n_tris, 3), dtype=np.int32)
    segment_ids = np.empty(n_tris, dtype=np.int32)

    tri_idx = 0
    for i in range(n_segments):
        for j in range(n_strip):
            j_next = j + 1  # no modulo — open arc
            # Current ring
            c0 = i * n_cross + j
            c1 = i * n_cross + j_next
            # Next ring
            n0 = (i + 1) * n_cross + j
            n1 = (i + 1) * n_cross + j_next
            indices[tri_idx] = [c0, c1, n0]
            indices[tri_idx + 1] = [c1, n1, n0]
            segment_ids[tri_idx] = i
            segment_ids[tri_idx + 1] = i
            tri_idx += 2

    organ_ids = np.full(n_tris, organ_id, dtype=np.int32)

    return indices, organ_ids, segment_ids
