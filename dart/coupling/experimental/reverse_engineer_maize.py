#!/usr/bin/env python3
"""Reverse-engineer maize OBJ growth stages into CPlantBox G1 representations.

Extracts ground-truth skeletons from 16 growth-stage OBJ models, then tests
each geometric property against CPlantBox's growth equations and the G1→G3
lofter to identify gaps and provide concrete solutions.

Usage:
    python3 reverse_engineer_maize.py /path/to/export/ --output output/reverse_engineer/
    python3 reverse_engineer_maize.py /path/to/export/ --skip-cplantbox --skip-lofter  # G1 only
"""

import argparse
import json
import os
import sys
import warnings
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from scipy.spatial import KDTree as cKDTree

# Optional imports — graceful fallback
try:
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.model_selection import cross_val_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

try:
    import torch
    HAS_TORCH = True
    HAS_CUDA = torch.cuda.is_available()
except ImportError:
    HAS_TORCH = False
    HAS_CUDA = False

# Server config
MAX_WORKERS = int(os.environ.get("RE_MAX_WORKERS", "64"))
DIFF_LOFTER_STEPS = int(os.environ.get("RE_DIFF_LOFTER_STEPS", "100"))
DIFF_LOFTER_LR = float(os.environ.get("RE_DIFF_LOFTER_LR", "0.01"))


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class LeafG1:
    """Extracted G1 representation of a single leaf."""
    leaf_id: int                  # consistent across stages (vertex-tracked)
    position: int                 # insertion order (1=lowest)
    skeleton: list                # (N, 3) cm
    widths: list                  # (N,) cm
    per_point_normals: list       # (N, 3)
    curvature_profile: list       # (N-2,) 1/cm — in-plane curvature
    oop_curvature_profile: list   # (N-2,) 1/cm — out-of-plane curvature
    twist_profile: list           # (N-1,) radians — cumulative normal rotation
    cross_section_angles: list    # (N,) radians — V-angle of cross section
    length: float                 # arc length cm
    max_width: float              # cm
    insertion_angle: float        # radians from vertical
    insertion_height: float       # cm (absolute Z, positive up)
    base_point: list              # (3,) cm
    tip_point: list               # (3,) cm
    width_profile_normalized: list  # (10,) normalized widths at 10 evenly-spaced positions
    asymmetry_profile: list       # (N,) left-right width difference cm
    area: float                   # cm² (integrated width × arc-length)
    # Sheath parameters (for lofter sheath wrapping)
    sheath: Optional[dict] = None  # {"fraction": (N,), "center_xy": (2,),
                                   #  "radius": float, "wrap_angle": float (rad)}


@dataclass
class StemG1:
    """Extracted G1 representation of the stem."""
    skeleton: list      # (N, 3) cm
    radii: list         # (N,) cm
    height: float       # cm
    internode_lengths: list  # cm — distances between leaf insertions
    diameter_profile: list  # (10,) normalized


@dataclass
class StageData:
    """All extracted data for one growth stage."""
    stage: int
    file: str
    leaves: list        # list of LeafG1
    stem: object        # StemG1 or None
    vstage: int         # V-stage (number of developed leaves)
    day_estimate: float # estimated calendar day


@dataclass
class Gap:
    """A detected gap between reference geometry and model capability."""
    parameter: str
    leaf_positions: list      # which leaves affected
    stages: list              # which stages affected
    severity: float           # Chamfer contribution cm
    status: str               # EXISTS_UNUSED | EXISTS_INSUFFICIENT | MISSING
    current_model: str        # what CPlantBox/lofter currently does
    needed: str               # what the reference geometry requires
    solution: str             # concrete fix
    extracted_values: dict    # values extracted from OBJ for the fix
    code_change: Optional[str] = None  # C++ change if needed
    category: str = "cplantbox"  # cplantbox | lofter


# ===========================================================================
# STEP 1: OBJ Parser + Leaf Tracking
# ===========================================================================

def parse_obj(path):
    """Parse Wavefront OBJ file, return vertices (cm), face groups, and per-vertex normals.

    Returns:
        verts: (N, 3) ndarray in cm
        groups: dict of group_name -> list of faces (each face = list of vertex indices)
        vert_normals: (N, 3) ndarray of averaged per-vertex normals (unit vectors),
                      or None if the OBJ has no vn lines.
    """
    verts = []
    vnormals = []
    groups = {}
    current_group = "default"

    # Accumulate normal indices per vertex for averaging
    vert_normal_accum = defaultdict(list)

    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            if parts[0] == "v" and len(parts) >= 4:
                verts.append([float(parts[1]) * 100,
                              float(parts[2]) * 100,
                              float(parts[3]) * 100])  # m → cm
            elif parts[0] == "vn" and len(parts) >= 4:
                vnormals.append([float(parts[1]),
                                 float(parts[2]),
                                 float(parts[3])])
            elif parts[0] == "g" and len(parts) > 1:
                current_group = parts[1]
                if current_group not in groups:
                    groups[current_group] = []
            elif parts[0] == "f":
                face_verts = []
                for p in parts[1:]:
                    components = p.split("/")
                    vi = int(components[0]) - 1
                    face_verts.append(vi)
                    # Collect normal index if present (format: v/vt/vn or v//vn)
                    if len(components) >= 3 and components[2]:
                        ni = int(components[2]) - 1
                        vert_normal_accum[vi].append(ni)
                if current_group not in groups:
                    groups[current_group] = []
                groups[current_group].append(face_verts)

    verts_arr = np.array(verts, dtype=np.float64)

    # Build per-vertex averaged normals
    vert_normals = None
    if vnormals:
        vnormals_arr = np.array(vnormals, dtype=np.float64)
        vert_normals = np.zeros((len(verts), 3), dtype=np.float64)
        for vi, ni_list in vert_normal_accum.items():
            if ni_list:
                avg = vnormals_arr[ni_list].mean(axis=0)
                nl = np.linalg.norm(avg)
                if nl > 1e-10:
                    vert_normals[vi] = avg / nl
                else:
                    vert_normals[vi] = [0, 0, 1]

    return verts_arr, groups, vert_normals


def find_connected_components(faces):
    """Union-Find connected components from face list."""
    parent = {}

    def find(x):
        while parent.get(x, x) != x:
            parent[x] = parent.get(parent[x], parent[x])
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for face in faces:
        for i in range(1, len(face)):
            union(face[0], face[i])

    all_verts = set()
    for f in faces:
        all_verts.update(f)

    components = defaultdict(set)
    for v in all_verts:
        components[find(v)].add(v)
    return list(components.values())


def track_leaves_across_stages(all_stage_components):
    """Match leaf components across stages by vertex ID overlap.

    Returns dict mapping canonical_leaf_id -> list of (stage_idx, component).
    """
    if not all_stage_components:
        return {}

    # Use first stage with full leaf count as reference
    ref_idx = 0
    ref_comps = all_stage_components[ref_idx]

    # Assign canonical IDs sorted by mean Z (insertion height, most negative = highest)
    ref_sorted = sorted(ref_comps, key=lambda c: np.mean(list(c)))
    canonical = {}
    for leaf_id, comp in enumerate(ref_sorted):
        canonical[leaf_id] = {ref_idx: comp}

    # Match subsequent stages
    for stage_idx in range(len(all_stage_components)):
        if stage_idx == ref_idx:
            continue
        comps = all_stage_components[stage_idx]
        for comp in comps:
            best_id, best_overlap = -1, 0
            for lid, stage_map in canonical.items():
                ref_comp = stage_map[ref_idx]
                overlap = len(comp & ref_comp)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_id = lid
            if best_id >= 0 and best_overlap > 0.5 * len(comp):
                canonical[best_id][stage_idx] = comp
    return canonical


def count_developed_leaves(leaves_g1, threshold_cm=5.0):
    """Count leaves with arc length > threshold → V-stage."""
    return sum(1 for leaf in leaves_g1 if leaf.length > threshold_cm)


def vstage_to_day(vstage):
    """Map V-stage to approximate calendar day (maize phenology).

    Based on typical maize development: ~3.5 days per leaf (phyllochron).
    V1 ≈ day 10 (emergence + first leaf collar visible).
    """
    if vstage <= 0:
        return 5.0
    return 10.0 + (vstage - 1) * 3.5


# ===========================================================================
# STEP 2: Mesh → G1 Skeleton Extraction
# ===========================================================================

def _find_boundary_edges(faces, vert_ids):
    """Find boundary edges (shared by exactly 1 face) within a component."""
    edge_count = defaultdict(int)
    comp_faces = [f for f in faces if all(v in vert_ids for v in f)]
    for face in comp_faces:
        n = len(face)
        for i in range(n):
            e = tuple(sorted([face[i], face[(i + 1) % n]]))
            edge_count[e] += 1
    boundary = [e for e, c in edge_count.items() if c == 1]
    return boundary, comp_faces


def _walk_boundary_loop(boundary_edges):
    """Walk boundary edges into an ordered loop of vertex IDs."""
    adj = defaultdict(list)
    for a, b in boundary_edges:
        adj[a].append(b)
        adj[b].append(a)

    start = boundary_edges[0][0]
    loop = [start]
    visited = {start}
    while True:
        nexts = [n for n in adj[loop[-1]] if n not in visited]
        if not nexts:
            break
        loop.append(nexts[0])
        visited.add(nexts[0])
    return loop


def _split_boundary(loop, verts):
    """Split boundary loop into left and right chains at base/tip.

    Base = vertex closest to stem (highest Z, i.e., least negative).
    Tip = vertex farthest from base along boundary.
    """
    loop_pos = verts[loop]  # (M, 3)

    # Base = highest Z (closest to ground/stem insertion)
    base_idx = np.argmax(loop_pos[:, 2])

    # Tip = farthest from base by geodesic (boundary arc length)
    n = len(loop)
    # Compute cumulative arc lengths from base in both directions
    arc = np.zeros(n)
    for i in range(1, n):
        arc[i] = arc[i - 1] + np.linalg.norm(loop_pos[i % n] - loop_pos[(i - 1) % n])
    total_arc = arc[-1] + np.linalg.norm(loop_pos[0] - loop_pos[-1])

    # Reorder so base is at index 0
    loop = loop[base_idx:] + loop[:base_idx]
    loop_pos = verts[loop]

    # Recompute arc from base
    arc = np.zeros(n)
    for i in range(1, n):
        arc[i] = arc[i - 1] + np.linalg.norm(loop_pos[i] - loop_pos[i - 1])
    total_arc = arc[-1] + np.linalg.norm(loop_pos[0] - loop_pos[-1])

    # Tip = point at ~half total arc length (farthest along boundary)
    tip_idx = np.argmin(np.abs(arc - total_arc / 2))

    # Split into two chains: base→tip (chain A) and tip→base (chain B, reversed)
    chain_a = loop[:tip_idx + 1]          # base to tip
    chain_b = loop[tip_idx:] + [loop[0]]  # tip back to base
    chain_b = list(reversed(chain_b))     # reverse so it goes base to tip

    return chain_a, chain_b


def _correspond_chains(chain_a, chain_b, verts, n_samples=20):
    """Arc-length parametrize two chains and sample at uniform positions.

    Returns skeleton (midpoints), widths (distances), left/right points.
    """
    def arc_sample(chain, n):
        pts = verts[chain]
        diffs = np.diff(pts, axis=0)
        seg_lens = np.linalg.norm(diffs, axis=1)
        cum = np.concatenate([[0], np.cumsum(seg_lens)])
        total = cum[-1]
        if total < 1e-6:
            return np.tile(pts[0], (n, 1))
        t_uniform = np.linspace(0, total, n)
        sampled = np.zeros((n, 3))
        for dim in range(3):
            sampled[:, dim] = np.interp(t_uniform, cum, pts[:, dim])
        return sampled

    pts_a = arc_sample(chain_a, n_samples)
    pts_b = arc_sample(chain_b, n_samples)

    skeleton = (pts_a + pts_b) / 2.0
    widths = np.linalg.norm(pts_a - pts_b, axis=1)
    return skeleton, widths, pts_a, pts_b


def _refine_skeleton(skeleton, widths, pts_a, pts_b, verts, vert_ids):
    """Refine skeleton using cross-section centroid projection.

    The initial boundary-midpoint skeleton uses only the two boundary chains.
    This projects ALL mesh interior vertices onto the skeleton, bins them into
    cross-sections, and replaces each skeleton point with the bin centroid.
    Reduces chamfer from ~6cm to ~2cm in a single pass (verified).

    Args:
        skeleton: (N, 3) initial boundary-midpoint skeleton
        widths: (N,) initial widths
        pts_a: (N, 3) left boundary points
        pts_b: (N, 3) right boundary points
        verts: Full vertex array (global indices)
        vert_ids: Set of vertex indices belonging to this leaf

    Returns:
        (skeleton, widths, pts_a, pts_b) — all refined in-place shapes
    """
    n_skel = len(skeleton)
    all_pts = verts[sorted(vert_ids)]

    # Skip refinement for tiny leaves (fewer vertices than skeleton samples)
    if len(all_pts) < n_skel:
        return skeleton, widths, pts_a, pts_b

    # Tangent frame at each skeleton point
    tangents = np.zeros((n_skel, 3))
    tangents[0] = skeleton[1] - skeleton[0]
    tangents[-1] = skeleton[-1] - skeleton[-2]
    for i in range(1, n_skel - 1):
        tangents[i] = skeleton[i + 1] - skeleton[i - 1]
    tn = np.linalg.norm(tangents, axis=1, keepdims=True)
    tangents /= np.maximum(tn, 1e-8)

    # Assign each mesh vertex to nearest skeleton point
    skel_tree = cKDTree(skeleton)
    _, skel_idx = skel_tree.query(all_pts)

    # Per-bin centroid and max perpendicular extent
    new_skel = skeleton.copy()
    new_widths = widths.copy()
    for i in range(n_skel):
        mask = skel_idx == i
        if mask.sum() < 2:
            continue
        bin_pts = all_pts[mask]
        centroid = bin_pts.mean(axis=0)
        new_skel[i] = centroid

        # Width = 2 × max perpendicular distance from centroid
        offsets = bin_pts - centroid
        perp = offsets - np.outer(np.dot(offsets, tangents[i]), tangents[i])
        new_widths[i] = np.linalg.norm(perp, axis=1).max() * 2

    # Update pts_a/pts_b: shift by the same delta as the skeleton
    delta = new_skel - skeleton
    new_pts_a = pts_a + delta
    new_pts_b = pts_b + delta

    return new_skel, new_widths, new_pts_a, new_pts_b


def _compute_curvature(skeleton):
    """Compute in-plane and out-of-plane curvature from skeleton points.

    Returns (curvature, oop_curvature) arrays of length N-2.
    """
    if len(skeleton) < 3:
        return np.array([]), np.array([])

    tangents = np.diff(skeleton, axis=0)
    seg_lens = np.linalg.norm(tangents, axis=1)
    seg_lens = np.maximum(seg_lens, 1e-8)
    tangents = tangents / seg_lens[:, None]

    # Curvature = |dT/ds| at interior points
    curvature = np.zeros(len(skeleton) - 2)
    oop_curvature = np.zeros(len(skeleton) - 2)

    for i in range(len(curvature)):
        dt = tangents[i + 1] - tangents[i]
        ds = (seg_lens[i] + seg_lens[i + 1]) / 2.0
        kappa_vec = dt / ds
        curvature[i] = np.linalg.norm(kappa_vec)

        # OOP component: project curvature onto gravity direction
        # In-plane = curvature in the plane of the skeleton
        # OOP = component perpendicular to the local osculating plane
        if i > 0 and curvature[i] > 1e-8:
            # Normal of osculating plane from consecutive tangents
            binormal = np.cross(tangents[i], tangents[i + 1])
            bn = np.linalg.norm(binormal)
            if bn > 1e-8:
                binormal /= bn
                # OOP curvature = component of kappa along binormal
                oop_curvature[i] = abs(np.dot(kappa_vec, binormal))

    return curvature, oop_curvature


def _compute_twist(skeleton, per_point_normals):
    """Compute total twist (rotation of normal around tangent) in radians.

    Measures the net rotation from base to tip, not cumulative noise.
    Uses signed angle via cross product to avoid accumulating unsigned noise.
    """
    if len(skeleton) < 3 or len(per_point_normals) < 3:
        return np.array([0.0])

    tangents = np.diff(skeleton, axis=0)
    seg_lens = np.linalg.norm(tangents, axis=1)
    seg_lens = np.maximum(seg_lens, 1e-8)
    tangents = tangents / seg_lens[:, None]

    # Smooth normals first (median filter over 3 neighbors) to reduce noise
    normals_smooth = per_point_normals.copy()
    for i in range(1, len(normals_smooth) - 1):
        normals_smooth[i] = np.mean(per_point_normals[max(0, i-1):i+2], axis=0)
        nl = np.linalg.norm(normals_smooth[i])
        if nl > 1e-8:
            normals_smooth[i] /= nl

    # Compute signed twist angle between base and tip
    # Project normals onto plane perpendicular to local tangent
    twist = np.zeros(len(tangents))
    for i in range(1, len(tangents)):
        t = tangents[min(i, len(tangents) - 1)]
        n0 = normals_smooth[i - 1]
        n1 = normals_smooth[i]
        # Project onto plane perpendicular to tangent
        n0p = n0 - np.dot(n0, t) * t
        n1p = n1 - np.dot(n1, t) * t
        l0, l1 = np.linalg.norm(n0p), np.linalg.norm(n1p)
        if l0 > 1e-8 and l1 > 1e-8:
            n0p /= l0
            n1p /= l1
            # Signed angle via cross product
            cross = np.cross(n0p, n1p)
            sign = np.sign(np.dot(cross, t))
            cos_angle = np.clip(np.dot(n0p, n1p), -1, 1)
            twist[i] = twist[i - 1] + sign * np.arccos(cos_angle)
        else:
            twist[i] = twist[i - 1]

    return twist


def _compute_per_point_normals(skeleton, comp_verts, comp_faces, vert_ids):
    """Compute per-skeleton-point normals from face normals via nearest vertex."""
    if len(comp_faces) == 0 or len(skeleton) == 0:
        # Fallback: use gravity-referenced normals
        return _gravity_normals(skeleton)

    # Compute face normals
    face_normals = []
    face_centers = []
    all_verts = comp_verts
    for face in comp_faces:
        vs = all_verts[face]
        if len(vs) >= 3:
            n = np.cross(vs[1] - vs[0], vs[2] - vs[0])
            nl = np.linalg.norm(n)
            if nl > 1e-10:
                n /= nl
            face_normals.append(n)
            face_centers.append(vs.mean(axis=0))

    if not face_normals:
        return _gravity_normals(skeleton)

    face_centers = np.array(face_centers)
    face_normals = np.array(face_normals)

    # For each skeleton point, average normals of nearest faces
    tree = cKDTree(face_centers)
    normals = np.zeros_like(skeleton)
    for i, pt in enumerate(skeleton):
        _, idxs = tree.query(pt, k=min(5, len(face_centers)))
        if isinstance(idxs, (int, np.integer)):
            idxs = [idxs]
        avg_n = face_normals[idxs].mean(axis=0)
        nl = np.linalg.norm(avg_n)
        if nl > 1e-10:
            normals[i] = avg_n / nl
        else:
            normals[i] = [0, 0, 1]
    return normals


def _gravity_normals(skeleton):
    """Fallback: normals perpendicular to tangent, biased upward."""
    if len(skeleton) < 2:
        return np.array([[0, 0, 1]] * len(skeleton))
    tangents = np.diff(skeleton, axis=0)
    tangents = np.vstack([tangents, tangents[-1:]])
    norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    tangents = tangents / norms
    up = np.array([0, 0, 1.0])
    normals = np.cross(tangents, np.cross(up, tangents))
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    return normals / norms


def _compute_cross_section_angle(left_pts, right_pts, skeleton, normals):
    """Estimate V-angle of leaf cross-section at each skeleton point.

    V-angle = angle between the two half-planes (left and right) of the leaf
    relative to the skeleton normal. 180° = perfectly flat.
    """
    angles = np.full(len(skeleton), np.pi)  # default flat
    for i in range(len(skeleton)):
        to_left = left_pts[i] - skeleton[i]
        to_right = right_pts[i] - skeleton[i]
        ll = np.linalg.norm(to_left)
        lr = np.linalg.norm(to_right)
        if ll > 0.01 and lr > 0.01:
            cos_a = np.clip(np.dot(to_left / ll, to_right / lr), -1, 1)
            angles[i] = np.arccos(cos_a)
    return angles


def _detect_sheath(skeleton, verts, vert_ids, stem_center_xy, stem_radius):
    """Detect sheath wrapping at the leaf base and compute lofter-compatible params.

    Z-slices the FULL leaf vertex set to find where angular extent around the
    stem exceeds 90°. Maps the sheath Z-range back to skeleton nodes via Z.

    Args:
        skeleton: (N, 3) skeleton points (base at index 0)
        verts: Full vertex array
        vert_ids: Set/list of vertex indices belonging to this leaf
        stem_center_xy: (2,) XY position of stem axis
        stem_radius: float, stem radius in cm

    Returns:
        dict with keys: fraction (N,), center_xy (2,), radius, wrap_angle (rad),
              sheath_height (float, cm)
        or None if no sheath detected.
    """
    if stem_center_xy is None or len(skeleton) < 3:
        return None

    leaf_pts = verts[sorted(vert_ids)]
    n_skel = len(skeleton)

    z_range = leaf_pts[:, 2].max() - leaf_pts[:, 2].min()
    if z_range < 2.0:
        return None

    # Step 1: Z-slice full leaf to find angular wrap at each height
    # Use fixed 0.5cm spacing (not proportional) so sheaths aren't missed on tall leaves
    z_min, z_max = leaf_pts[:, 2].min(), leaf_pts[:, 2].max()
    n_slices = max(20, int(z_range / 0.5))
    z_edges = np.linspace(z_min, z_max, n_slices)
    dz = 1.0  # fixed 1cm band for robust sheath detection

    wrap_at_z = np.zeros(n_slices)
    radius_at_z = np.zeros(n_slices)

    for si in range(n_slices):
        mask = np.abs(leaf_pts[:, 2] - z_edges[si]) < dz
        if mask.sum() < 3:
            continue
        pts = leaf_pts[mask]
        angles = np.arctan2(pts[:, 1] - stem_center_xy[1],
                           pts[:, 0] - stem_center_xy[0])
        angles_sorted = np.sort(angles)
        gaps = np.diff(angles_sorted)
        wrap_gap = 2 * np.pi + angles_sorted[0] - angles_sorted[-1]
        gaps = np.append(gaps, wrap_gap)
        wrap_at_z[si] = 2 * np.pi - gaps.max()
        radius_at_z[si] = np.linalg.norm(
            pts[:, :2] - stem_center_xy, axis=1).mean()

    # Step 2: Find sheath Z-range (contiguous from base where wrap > 90°)
    sheath_threshold = np.pi / 2
    sheath_z_top = z_min
    for si in range(n_slices):
        if wrap_at_z[si] > sheath_threshold:
            sheath_z_top = z_edges[si]
        elif z_edges[si] > sheath_z_top + 3.0:
            # Allow 3cm gap then stop
            break

    sheath_height = sheath_z_top - z_min
    if sheath_height < 1.0:
        return None

    # Representative wrap angle and radius
    sheath_mask = (wrap_at_z > sheath_threshold) & (z_edges <= sheath_z_top)
    if sheath_mask.sum() == 0:
        return None
    wrap_angle = float(np.median(wrap_at_z[sheath_mask]))
    sheath_r = float(np.median(radius_at_z[sheath_mask]))
    sheath_r = max(sheath_r, stem_radius)

    # Step 3: Map to skeleton fraction via Z position
    transition = min(2.0, sheath_height * 0.3)
    fraction = np.zeros(n_skel)
    for i in range(n_skel):
        z_i = skeleton[i, 2]
        if z_i <= sheath_z_top - transition:
            fraction[i] = 1.0
        elif z_i <= sheath_z_top:
            fraction[i] = (sheath_z_top - z_i) / max(transition, 0.01)

    if (fraction > 0.1).sum() < 2:
        return None

    return {
        "fraction": fraction.tolist(),
        "center_xy": stem_center_xy.tolist(),
        "radius": sheath_r,
        "wrap_angle": wrap_angle,
        "sheath_height": sheath_height,
    }


# ===========================================================================
# STEP 2b: BFS-Based Skeleton Extraction (sheath + blade continuous)
# ===========================================================================

# Region labels for BFS layers
_BLADE = 0
_COLLAR = 1
_SHEATH = 2


def _build_adjacency(faces, vert_ids):
    """Build vertex adjacency graph (edge-only) for a leaf component.

    Returns:
        adj: dict of vertex_id -> set of neighbor vertex_ids
        boundary_verts: set of boundary vertex ids
        comp_faces: list of faces belonging to this component
    """
    comp_faces = [f for f in faces if all(v in vert_ids for v in f)]

    adj = defaultdict(set)
    edge_face_count = defaultdict(int)
    for face in comp_faces:
        n = len(face)
        for i in range(n):
            a, b = face[i], face[(i + 1) % n]
            adj[a].add(b)
            adj[b].add(a)
            e = (min(a, b), max(a, b))
            edge_face_count[e] += 1

    boundary_edges = {e for e, c in edge_face_count.items() if c == 1}
    boundary_verts = set()
    for a, b in boundary_edges:
        boundary_verts.add(a)
        boundary_verts.add(b)

    return adj, boundary_verts, comp_faces


def _find_tip_vertex(vert_ids, verts, adj, boundary_verts, stem_center_xy=None):
    """Find the tip vertex for BFS start.

    The tip is the boundary vertex farthest from the stem center in 3D.
    For horizontal/drooping leaves, this correctly picks the distal tip
    rather than a random low-Z vertex.

    Fallback for closed meshes (no boundary): farthest vertex from centroid.
    """
    candidates = boundary_verts if boundary_verts else vert_ids

    if stem_center_xy is not None and len(candidates) > 2:
        # Farthest from stem center in XY + Z distance from stem base
        # Use full 3D distance from stem axis (projected to nearest Z)
        stem_z = verts[sorted(vert_ids), 2].max()  # stem insertion ≈ max Z
        stem_3d = np.array([stem_center_xy[0], stem_center_xy[1], stem_z])
        best = max(candidates,
                   key=lambda v: np.linalg.norm(verts[v] - stem_3d))
    else:
        # Fallback: farthest from centroid
        centroid = verts[sorted(vert_ids)].mean(axis=0)
        best = max(candidates,
                   key=lambda v: np.linalg.norm(verts[v] - centroid))

    return best


def _bfs_layers(tip, adj, vert_ids):
    """Breadth-first search from tip. Returns ordered list of layers.

    Each layer is a list of vertex indices forming a cross-section.
    """
    visited = {tip}
    layers = [[tip]]
    while True:
        next_layer = []
        for v in layers[-1]:
            for nb in adj[v]:
                if nb not in visited and nb in vert_ids:
                    visited.add(nb)
                    next_layer.append(nb)
        if not next_layer:
            break
        layers.append(next_layer)
    return layers


def _compute_arc_span(layer_pts_xy, center_xy):
    """Compute angular arc span of points around a center.

    Returns arc span in radians.
    """
    if len(layer_pts_xy) < 2:
        return 0.0
    dxy = layer_pts_xy - center_xy
    angles = np.arctan2(dxy[:, 1], dxy[:, 0])
    angles_sorted = np.sort(angles)
    if len(angles_sorted) < 2:
        return 0.0
    gaps = np.diff(angles_sorted)
    wrap_gap = 2 * np.pi + angles_sorted[0] - angles_sorted[-1]
    all_gaps = np.append(gaps, wrap_gap)
    return float(2 * np.pi - all_gaps.max())


def _classify_layers(layers, verts, stem_center_xy):
    """Classify each BFS layer as BLADE, COLLAR, or SHEATH.

    Returns:
        labels: list of _BLADE/_COLLAR/_SHEATH per layer
        arc_spans: list of arc span in radians per layer
    """
    labels = []
    arc_spans = []

    for layer in layers:
        pts = verts[layer]
        if len(layer) < 3 or stem_center_xy is None:
            labels.append(_BLADE)
            arc_spans.append(0.0)
            continue

        span = _compute_arc_span(pts[:, :2], stem_center_xy)
        arc_spans.append(span)

        if span >= np.deg2rad(150):
            labels.append(_SHEATH)
        elif span >= np.deg2rad(90):
            labels.append(_COLLAR)
        else:
            labels.append(_BLADE)

    return labels, arc_spans


def _angular_midpoint(layer, verts, stem_center_xy):
    """Compute the angular midpoint of a layer on the leaf surface.

    Places the skeleton point at the midpoint of the covered arc,
    at the mean radius from the stem center.

    Returns (x, y, z) position.
    """
    pts = verts[layer]
    dxy = pts[:, :2] - stem_center_xy
    angles = np.arctan2(dxy[:, 1], dxy[:, 0])
    r = np.linalg.norm(dxy, axis=1).mean()
    z = pts[:, 2].mean()

    # Find the largest angular gap (where the leaf doesn't wrap)
    angles_sorted = np.sort(angles)
    gaps = np.diff(angles_sorted)
    wrap_gap = 2 * np.pi + angles_sorted[0] - angles_sorted[-1]
    all_gaps = np.append(gaps, wrap_gap)
    gap_idx = np.argmax(all_gaps)

    if gap_idx < len(gaps):
        # Gap is between sorted angles[gap_idx] and sorted angles[gap_idx+1]
        arc_start = angles_sorted[gap_idx + 1]
        arc_end = angles_sorted[gap_idx] + 2 * np.pi
    else:
        # Gap is the wrap-around gap
        arc_start = angles_sorted[0]
        arc_end = angles_sorted[-1]

    mid_angle = (arc_start + arc_end) / 2.0

    x = stem_center_xy[0] + r * np.cos(mid_angle)
    y = stem_center_xy[1] + r * np.sin(mid_angle)
    return np.array([x, y, z])


def _layer_skeleton_point(layer, verts, region, stem_center_xy, arc_span):
    """Compute skeleton point for one BFS layer.

    BLADE: vertex centroid
    SHEATH: angular midpoint at mean radius from stem center
    COLLAR: blend between centroid and angular midpoint
    """
    pts = verts[layer]
    centroid = pts.mean(axis=0)

    if region == _BLADE or stem_center_xy is None or len(layer) < 3:
        return centroid

    if region == _SHEATH:
        return _angular_midpoint(layer, verts, stem_center_xy)

    # COLLAR: blend based on arc span (90° → 100% centroid, 150° → 100% angular)
    t = np.clip((arc_span - np.deg2rad(90)) / np.deg2rad(60), 0, 1)
    angular_pt = _angular_midpoint(layer, verts, stem_center_xy)
    return (1 - t) * centroid + t * angular_pt


def _layer_width(layer, verts, region, stem_center_xy, arc_span,
                 tangent=None):
    """Compute width for one BFS layer.

    BLADE: max perpendicular extent relative to skeleton tangent.
           Falls back to max pairwise distance if no tangent.
    SHEATH: arc length = r * arc_span
    COLLAR: linear span (transition)
    """
    pts = verts[layer]
    if len(layer) < 2:
        return 0.0

    if region == _SHEATH and stem_center_xy is not None:
        dxy = pts[:, :2] - stem_center_xy
        r = np.linalg.norm(dxy, axis=1).mean()
        return float(r * arc_span)

    # BLADE or COLLAR: max perpendicular extent from centroid
    centroid = pts.mean(axis=0)
    offsets = pts - centroid

    if tangent is not None:
        tn = np.linalg.norm(tangent)
        if tn > 1e-8:
            t_unit = tangent / tn
            # Remove component along tangent
            perp = offsets - np.outer(np.dot(offsets, t_unit), t_unit)
            return float(np.linalg.norm(perp, axis=1).max() * 2)

    # Fallback: max pairwise distance
    from scipy.spatial.distance import pdist
    return float(pdist(pts).max())


def _layer_normal(layer, vert_normals, verts, stem_center_xy=None):
    """Average per-vertex normals for a BFS layer.

    If vert_normals is None, fall back to gravity-referenced normals.
    For sheath layers, ensure normal points outward from stem.
    """
    if vert_normals is None:
        return np.array([0.0, 0.0, 1.0])

    avg = vert_normals[layer].mean(axis=0)
    nl = np.linalg.norm(avg)
    if nl < 1e-10:
        return np.array([0.0, 0.0, 1.0])
    avg /= nl

    # For sheath: ensure normal points away from stem center
    if stem_center_xy is not None and len(layer) >= 3:
        pts = verts[layer]
        center_xy = pts[:, :2].mean(axis=0)
        outward = center_xy - stem_center_xy
        outward_3d = np.array([outward[0], outward[1], 0.0])
        if np.dot(avg, outward_3d) < 0:
            avg = -avg

    return avg


def extract_leaf_g1_bfs(comp, verts, faces, leaf_id, position, n_samples=20,
                        stem_center_xy=None, stem_radius=1.0,
                        vert_normals=None):
    """Extract G1 representation using BFS cross-sectional layers.

    Produces a continuous skeleton from tip → blade → collar → sheath → base.
    The sheath region is annotated on the skeleton, not separated.
    """
    vert_ids = comp

    # Build adjacency
    adj, boundary_verts, comp_faces = _build_adjacency(faces, vert_ids)

    if len(vert_ids) < 4 or len(comp_faces) < 2:
        # Degenerate leaf
        center = verts[list(vert_ids)].mean(axis=0)
        return LeafG1(
            leaf_id=leaf_id, position=position,
            skeleton=[center.tolist()], widths=[0.0],
            per_point_normals=[[0, 0, 1]], curvature_profile=[],
            oop_curvature_profile=[], twist_profile=[],
            cross_section_angles=[np.pi], length=0.0, max_width=0.0,
            insertion_angle=0.0, insertion_height=center[2],
            base_point=center.tolist(), tip_point=center.tolist(),
            width_profile_normalized=[0] * 10, asymmetry_profile=[0.0],
            area=0.0,
        )

    # Find tip and run BFS
    tip = _find_tip_vertex(vert_ids, verts, adj, boundary_verts, stem_center_xy)
    layers = _bfs_layers(tip, adj, vert_ids)

    if len(layers) < 3:
        center = verts[list(vert_ids)].mean(axis=0)
        return LeafG1(
            leaf_id=leaf_id, position=position,
            skeleton=[center.tolist()], widths=[0.0],
            per_point_normals=[[0, 0, 1]], curvature_profile=[],
            oop_curvature_profile=[], twist_profile=[],
            cross_section_angles=[np.pi], length=0.0, max_width=0.0,
            insertion_angle=0.0, insertion_height=center[2],
            base_point=center.tolist(), tip_point=center.tolist(),
            width_profile_normalized=[0] * 10, asymmetry_profile=[0.0],
            area=0.0,
        )

    # Classify layers
    labels, arc_spans = _classify_layers(layers, verts, stem_center_xy)

    # ---- Build skeleton: blade centroids + CIRCUMFERENTIAL sheath path ----
    # The skeleton traces the actual leaf path:
    #   blade: one centroid per BFS layer (flat cross-sections)
    #   sheath: angular-binned path wrapping around the stem
    # The mesh is one continuous quad strip — same technique in both regions.

    # Find first collar/sheath layer
    first_sheath = None
    for i, l in enumerate(labels):
        if l >= _COLLAR:
            first_sheath = i
            break

    # BLADE: one centroid per BFS layer
    blade_skeleton = []
    blade_normals = []
    blade_end = first_sheath if first_sheath is not None else len(layers)
    for i in range(blade_end):
        pt = _layer_skeleton_point(layers[i], verts, labels[i], stem_center_xy,
                                   arc_spans[i])
        n = _layer_normal(layers[i], vert_normals, verts, stem_center_xy)
        blade_skeleton.append(pt)
        blade_normals.append(n)

    # Blade tangents for width
    blade_tangents = np.zeros((blade_end, 3))
    if blade_end >= 2:
        ba = np.array(blade_skeleton)
        blade_tangents[0] = ba[1] - ba[0]
        blade_tangents[-1] = ba[-1] - ba[-2]
        for i in range(1, blade_end - 1):
            blade_tangents[i] = ba[i + 1] - ba[i - 1]

    blade_widths = []
    for i in range(blade_end):
        w = _layer_width(layers[i], verts, labels[i], stem_center_xy,
                         arc_spans[i], tangent=blade_tangents[i])
        blade_widths.append(w)

    # SHEATH: circumferential path around stem (angular-binned)
    sheath_skeleton = []
    sheath_normals = []
    sheath_widths = []

    if first_sheath is not None and stem_center_xy is not None:
        sheath_vids = []
        for i in range(first_sheath, len(layers)):
            sheath_vids.extend(layers[i])

        if sheath_vids:
            sheath_pts = verts[sheath_vids]
            dxy = sheath_pts[:, :2] - stem_center_xy
            angles = np.arctan2(dxy[:, 1], dxy[:, 0])

            # Entry angle from last blade point
            if blade_skeleton:
                entry_angle = np.arctan2(
                    blade_skeleton[-1][1] - stem_center_xy[1],
                    blade_skeleton[-1][0] - stem_center_xy[0])
            else:
                entry_angle = 0.0

            # Unwrap angles relative to entry
            rel_angles = (angles - entry_angle + np.pi) % (2 * np.pi) - np.pi
            # Pick dominant wrapping direction
            if (rel_angles < 0).sum() > (rel_angles > 0).sum():
                rel_angles[rel_angles > 0] -= 2 * np.pi
            else:
                rel_angles[rel_angles < 0] += 2 * np.pi

            order = np.argsort(rel_angles)
            angle_min, angle_max = rel_angles[order[0]], rel_angles[order[-1]]
            angle_range = abs(angle_max - angle_min)

            # ~1 skeleton point per 20° of wrap
            n_sh = max(3, int(np.degrees(angle_range) / 20))
            angle_bins = np.linspace(angle_min, angle_max, n_sh + 1)

            for bi in range(n_sh):
                lo, hi = angle_bins[bi], angle_bins[bi + 1]
                mask = (rel_angles >= lo) & (rel_angles < hi)
                if bi == n_sh - 1:
                    mask |= (rel_angles == hi)
                if mask.sum() == 0:
                    continue

                bin_pts = sheath_pts[mask]
                skel_pt = bin_pts.mean(axis=0)
                sheath_skeleton.append(skel_pt)

                # Width = Z extent of the angular bin
                z_span = bin_pts[:, 2].max() - bin_pts[:, 2].min()
                sheath_widths.append(max(z_span, 0.5))

                # Normal: radially outward from stem
                out_xy = skel_pt[:2] - stem_center_xy
                out_n = np.linalg.norm(out_xy)
                if out_n > 1e-8:
                    sheath_normals.append(np.array([out_xy[0]/out_n, out_xy[1]/out_n, 0.0]))
                else:
                    sheath_normals.append(np.array([0.0, 0.0, 1.0]))

    # Combine blade + sheath
    raw_skeleton = np.array(blade_skeleton + sheath_skeleton)
    raw_widths = np.array(blade_widths + sheath_widths)
    raw_normals = np.array(blade_normals + sheath_normals)
    raw_is_sheath = np.array([False] * len(blade_skeleton) + [True] * len(sheath_skeleton))

    # Gaussian-smooth skeleton (sigma=1 layer) to remove zigzag
    if len(raw_skeleton) >= 5:
        from scipy.ndimage import gaussian_filter1d
        # Preserve endpoints
        smoothed = gaussian_filter1d(raw_skeleton, sigma=1.0, axis=0)
        smoothed[0] = raw_skeleton[0]
        smoothed[-1] = raw_skeleton[-1]
        raw_skeleton = smoothed

    # Resample to n_samples points via arc-length parameterization
    diffs = np.diff(raw_skeleton, axis=0)
    seg_lens = np.linalg.norm(diffs, axis=1)
    cum_arc = np.concatenate([[0], np.cumsum(seg_lens)])
    total_arc = cum_arc[-1]

    if total_arc < 0.1:
        center = raw_skeleton.mean(axis=0)
        return LeafG1(
            leaf_id=leaf_id, position=position,
            skeleton=[center.tolist()], widths=[0.0],
            per_point_normals=[[0, 0, 1]], curvature_profile=[],
            oop_curvature_profile=[], twist_profile=[],
            cross_section_angles=[np.pi], length=0.0, max_width=0.0,
            insertion_angle=0.0, insertion_height=center[2],
            base_point=center.tolist(), tip_point=center.tolist(),
            width_profile_normalized=[0] * 10, asymmetry_profile=[0.0],
            area=0.0,
        )

    t_uniform = np.linspace(0, total_arc, n_samples)
    skeleton = np.zeros((n_samples, 3))
    for dim in range(3):
        skeleton[:, dim] = np.interp(t_uniform, cum_arc, raw_skeleton[:, dim])

    # Interpolate raw widths to resampled skeleton (smooth, no gaps)
    widths = np.interp(t_uniform, cum_arc, raw_widths)

    normals = np.zeros((n_samples, 3))
    for dim in range(3):
        normals[:, dim] = np.interp(t_uniform, cum_arc, raw_normals[:, dim])
    # Re-normalize normals
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.maximum(norms, 1e-10)

    # Interpolate sheath flag to resampled skeleton
    raw_sheath_float = raw_is_sheath.astype(float)
    resampled_sheath = np.interp(t_uniform, cum_arc, raw_sheath_float)

    # Curvature via existing helper
    curvature, oop_curvature = _compute_curvature(skeleton)

    # Twist via existing helper
    twist = _compute_twist(skeleton, normals)

    # Cross-section angles — approximate from widths (no left/right chains in BFS)
    cs_angles = np.full(n_samples, np.pi)  # default flat

    # Arc length
    length = float(total_arc)

    # Insertion angle
    if n_samples >= 2:
        first_seg = skeleton[1] - skeleton[0]
        fl = np.linalg.norm(first_seg)
        if fl > 0.01:
            insertion_angle = float(np.arccos(np.clip(abs(first_seg[2]) / fl, 0, 1)))
        else:
            insertion_angle = 0.0
    else:
        insertion_angle = 0.0

    # Width profile normalized to 10 positions
    resampled_arc = np.linspace(0, total_arc, n_samples)
    t_10 = np.linspace(0, total_arc, 10)
    wp = np.interp(t_10, resampled_arc, widths)
    max_w = widths.max()
    if max_w > 0.01:
        wp_norm = (wp / max_w).tolist()
    else:
        wp_norm = [0.0] * 10

    # Asymmetry — not available from BFS (no left/right chains), set to zero
    asymmetry = [0.0] * n_samples

    # Area
    area = float(getattr(np, 'trapezoid', np.trapz)(widths, resampled_arc))

    # Build sheath dict from circumferential skeleton
    sheath = None
    has_sheath = raw_is_sheath.any()
    if has_sheath and stem_center_xy is not None:
        # Fraction: interpolated from raw sheath flag
        fraction = np.interp(t_uniform, cum_arc, raw_is_sheath.astype(float))

        # Wrap angle and radius from sheath skeleton points
        sheath_skel_pts = raw_skeleton[raw_is_sheath]
        if len(sheath_skel_pts) >= 2:
            dxy = sheath_skel_pts[:, :2] - stem_center_xy
            sheath_r = float(np.linalg.norm(dxy, axis=1).mean())
            sh_angles = np.arctan2(dxy[:, 1], dxy[:, 0])
            ang_diffs = np.diff(sh_angles)
            ang_diffs = (ang_diffs + np.pi) % (2 * np.pi) - np.pi
            wrap_angle = float(abs(ang_diffs.sum()))
        else:
            sheath_r = stem_radius
            wrap_angle = np.pi

        sheath_z = sheath_skel_pts[:, 2]
        sheath_height = float(sheath_z.max() - sheath_z.min()) if len(sheath_z) > 1 else 0.0

        nonzero = np.where(fraction > 0.01)[0]
        if len(nonzero) >= 1:
            sheath = {
                "fraction": fraction.tolist(),
                "center_xy": stem_center_xy.tolist(),
                "radius": sheath_r,
                "wrap_angle": wrap_angle,
                "sheath_height": float(sheath_height),
                "sheath_start_idx": int(nonzero[0]),
                "sheath_end_idx": int(nonzero[-1]),
            }

    # ---- CPlantBox compatibility: reverse to BASE→TIP, clamp widths ----
    # CPlantBox convention: skeleton[0] = base (stem attachment), skeleton[-1] = tip
    # Our BFS goes tip→sheath(base), so reverse everything
    skeleton = skeleton[::-1].copy()
    widths = widths[::-1].copy()
    normals = normals[::-1].copy()
    if len(curvature) > 0:
        curvature = curvature[::-1].copy()
        oop_curvature = oop_curvature[::-1].copy()
    if len(twist) > 0:
        twist = twist[::-1].copy()
    cs_angles = cs_angles[::-1].copy()
    asymmetry = asymmetry[::-1]
    if sheath is not None:
        sheath["fraction"] = sheath["fraction"][::-1]
        old_start = sheath["sheath_start_idx"]
        old_end = sheath["sheath_end_idx"]
        sheath["sheath_start_idx"] = n_samples - 1 - old_end
        sheath["sheath_end_idx"] = n_samples - 1 - old_start

    # Recompute normalized width profile after reversal
    resampled_arc = np.linspace(0, total_arc, n_samples)
    t_10 = np.linspace(0, total_arc, 10)
    wp = np.interp(t_10, resampled_arc, widths)
    max_w = widths.max()
    wp_norm = (wp / max_w).tolist() if max_w > 0.01 else [0.0] * 10

    # Clamp minimum width to 0.15 cm (CPlantBox convention, prevents degenerate tris)
    widths = np.maximum(widths, 0.15)

    # Tip taper: linear from 70% to 100% of length (CPlantBox convention)
    if n_samples >= 4 and total_arc > 1.0:
        cum_len = np.linspace(0, total_arc, n_samples)
        for i in range(n_samples):
            frac = cum_len[i] / total_arc
            if frac > 0.70:
                t = (frac - 0.70) / 0.30
                widths[i] = max(widths[i] * (1.0 - t), 0.15)

    # Insertion angle: angle of first skeleton segment from vertical (at base)
    if n_samples >= 2:
        first_seg = skeleton[1] - skeleton[0]
        fl = np.linalg.norm(first_seg)
        if fl > 0.01:
            insertion_angle = float(np.arccos(np.clip(abs(first_seg[2]) / fl, 0, 1)))

    # Area after width adjustments
    area = float(getattr(np, 'trapezoid', np.trapz)(widths, resampled_arc))

    return LeafG1(
        leaf_id=leaf_id, position=position,
        skeleton=skeleton.tolist(), widths=widths.tolist(),
        per_point_normals=normals.tolist(),
        curvature_profile=curvature.tolist(),
        oop_curvature_profile=oop_curvature.tolist(),
        twist_profile=twist.tolist(),
        cross_section_angles=cs_angles.tolist(),
        length=length,
        max_width=float(widths.max()) if len(widths) > 0 else 0.0,
        insertion_angle=insertion_angle,
        insertion_height=float(skeleton[0, 2]),  # base = index 0 after reversal
        base_point=skeleton[0].tolist(),   # base = index 0 (stem attachment)
        tip_point=skeleton[-1].tolist(),  # tip = last index (leaf end)
        width_profile_normalized=wp_norm,
        asymmetry_profile=asymmetry,
        area=area,
        sheath=sheath,
    )


def extract_leaf_g1_boundary(comp, verts, faces, leaf_id, position, n_samples=20,
                             stem_center_xy=None, stem_radius=1.0,
                             vert_normals=None):
    """Extract G1 using boundary-edge walking (legacy method)."""
    vert_ids = comp
    boundary_edges, comp_faces = _find_boundary_edges(faces, vert_ids)

    if len(boundary_edges) < 4:
        # Degenerate leaf (collapsed at early stages)
        center = verts[list(vert_ids)].mean(axis=0)
        return LeafG1(
            leaf_id=leaf_id, position=position,
            skeleton=[center.tolist()], widths=[0.0],
            per_point_normals=[[0, 0, 1]], curvature_profile=[],
            oop_curvature_profile=[], twist_profile=[],
            cross_section_angles=[np.pi], length=0.0, max_width=0.0,
            insertion_angle=0.0, insertion_height=center[2],
            base_point=center.tolist(), tip_point=center.tolist(),
            width_profile_normalized=[0] * 10, asymmetry_profile=[0.0],
            area=0.0,
        )

    loop = _walk_boundary_loop(boundary_edges)
    chain_a, chain_b = _split_boundary(loop, verts)
    skeleton, widths, pts_a, pts_b = _correspond_chains(
        chain_a, chain_b, verts, n_samples)

    # Refine skeleton using cross-section centroid projection
    skeleton, widths, pts_a, pts_b = _refine_skeleton(
        skeleton, widths, pts_a, pts_b, verts, vert_ids)

    # Per-point normals
    comp_verts_arr = verts  # full vertex array (face indices are global)
    normals = _compute_per_point_normals(skeleton, comp_verts_arr, comp_faces, vert_ids)

    # Curvature
    curvature, oop_curvature = _compute_curvature(skeleton)

    # Twist
    twist = _compute_twist(skeleton, normals)

    # Cross-section V-angle
    cs_angles = _compute_cross_section_angle(pts_a, pts_b, skeleton, normals)

    # Arc length
    diffs = np.diff(skeleton, axis=0)
    seg_lens = np.linalg.norm(diffs, axis=1)
    length = float(seg_lens.sum())

    # Insertion angle (angle of first skeleton segment from vertical)
    if len(skeleton) >= 2:
        first_seg = skeleton[1] - skeleton[0]
        fl = np.linalg.norm(first_seg)
        if fl > 0.01:
            insertion_angle = float(np.arccos(np.clip(
                abs(first_seg[2]) / fl, 0, 1)))
        else:
            insertion_angle = 0.0
    else:
        insertion_angle = 0.0

    # Width profile normalized to 10 positions
    if length > 0.1:
        cum_arc = np.concatenate([[0], np.cumsum(seg_lens)])
        t_10 = np.linspace(0, cum_arc[-1], 10)
        wp = np.interp(t_10, cum_arc, widths)
        max_w = widths.max()
        if max_w > 0.01:
            wp_norm = (wp / max_w).tolist()
        else:
            wp_norm = [0.0] * 10
    else:
        wp_norm = [0.0] * 10

    # Asymmetry (left - right half-width)
    dist_a = np.linalg.norm(pts_a - skeleton, axis=1)
    dist_b = np.linalg.norm(pts_b - skeleton, axis=1)
    asymmetry = (dist_a - dist_b).tolist()

    # Area (trapezoidal integration of width along arc length)
    if length > 0.1:
        cum_arc = np.concatenate([[0], np.cumsum(seg_lens)])
        area = float(getattr(np, 'trapezoid', np.trapz)(widths, cum_arc))
    else:
        area = 0.0

    # Sheath detection
    sheath = _detect_sheath(skeleton, verts, vert_ids,
                            stem_center_xy, stem_radius)

    return LeafG1(
        leaf_id=leaf_id, position=position,
        skeleton=skeleton.tolist(), widths=widths.tolist(),
        per_point_normals=normals.tolist(),
        curvature_profile=curvature.tolist(),
        oop_curvature_profile=oop_curvature.tolist(),
        twist_profile=twist.tolist(),
        cross_section_angles=cs_angles.tolist(),
        length=length, max_width=float(widths.max()) if len(widths) > 0 else 0.0,
        insertion_angle=insertion_angle,
        insertion_height=float(skeleton[0, 2]) if len(skeleton) > 0 else 0.0,
        base_point=skeleton[0].tolist() if len(skeleton) > 0 else [0, 0, 0],
        tip_point=skeleton[-1].tolist() if len(skeleton) > 0 else [0, 0, 0],
        width_profile_normalized=wp_norm,
        asymmetry_profile=asymmetry,
        area=area,
        sheath=sheath,
    )


def extract_leaf_g1(comp, verts, faces, leaf_id, position, n_samples=20,
                    stem_center_xy=None, stem_radius=1.0,
                    vert_normals=None, method="bfs"):
    """Extract G1 representation from a leaf mesh component.

    Args:
        method: "bfs" (default, BFS cross-sectional layers) or
                "boundary" (legacy boundary-edge walking)
    """
    if method == "boundary":
        return extract_leaf_g1_boundary(comp, verts, faces, leaf_id, position,
                                        n_samples, stem_center_xy, stem_radius,
                                        vert_normals)
    return extract_leaf_g1_bfs(comp, verts, faces, leaf_id, position,
                               n_samples, stem_center_xy, stem_radius,
                               vert_normals)


def extract_stem_g1(verts, faces, leaf_bases):
    """Extract stem G1 from stem mesh group.

    leaf_bases: list of (z, position) for each leaf insertion → internode lengths.
    """
    if not faces:
        return None

    # Stem vertices
    stem_verts_ids = set()
    for f in faces:
        stem_verts_ids.update(f)
    if not stem_verts_ids:
        return None

    sv = verts[sorted(stem_verts_ids)]

    # Centerline: bin by Z, take mean XY per bin
    z_min, z_max = sv[:, 2].min(), sv[:, 2].max()
    height = z_max - z_min
    if height < 0.1:
        return StemG1(skeleton=[], radii=[], height=0.0,
                      internode_lengths=[], diameter_profile=[0] * 10)

    n_bins = max(10, int(height / 1.0))  # ~1cm bins
    z_edges = np.linspace(z_min, z_max, n_bins + 1)
    skeleton = []
    radii = []
    for i in range(n_bins):
        mask = (sv[:, 2] >= z_edges[i]) & (sv[:, 2] < z_edges[i + 1])
        if i == n_bins - 1:  # include upper bound
            mask |= (sv[:, 2] == z_edges[i + 1])
        if mask.sum() > 0:
            bin_pts = sv[mask]
            center = bin_pts.mean(axis=0)
            skeleton.append(center)
            # Radius = mean distance from center in XY
            dxy = np.linalg.norm(bin_pts[:, :2] - center[:2], axis=1)
            radii.append(float(dxy.mean()) if len(dxy) > 0 else 0.0)

    skeleton = np.array(skeleton)
    radii = np.array(radii)

    # Internode lengths from leaf insertion heights
    if leaf_bases:
        sorted_bases = sorted(leaf_bases, key=lambda x: x[0])  # sort by Z
        internode_lengths = []
        for i in range(1, len(sorted_bases)):
            internode_lengths.append(abs(sorted_bases[i][0] - sorted_bases[i - 1][0]))
    else:
        internode_lengths = []

    # Normalized diameter profile (10 points along height)
    if len(radii) > 1 and radii.max() > 0:
        dp = np.interp(np.linspace(0, 1, 10),
                        np.linspace(0, 1, len(radii)), radii)
        dp_norm = (dp / radii.max()).tolist()
    else:
        dp_norm = [1.0] * 10

    return StemG1(
        skeleton=skeleton.tolist(), radii=radii.tolist(),
        height=float(height), internode_lengths=internode_lengths,
        diameter_profile=dp_norm,
    )


def _extract_stage_g1(args):
    """Worker function for parallel G1 extraction of a single stage.

    Args: tuple of (idx, stage_num, fpath_str, verts, groups, vert_normals,
          canonical, position_map, n_samples)
    Returns: StageData for this stage.
    """
    idx, stage_num, fpath_str, verts, groups, vert_normals, canonical_serialized, position_map, n_samples = args

    # Gather all faces by group
    leaf_faces = []
    stem_faces = []
    for gname, gfaces in groups.items():
        if "leaf" in gname.lower():
            leaf_faces.extend(gfaces)
        elif "stem" in gname.lower():
            stem_faces.extend(gfaces)

    # Extract stem center and radius FIRST (needed for sheath detection)
    stem_center_xy = None
    stem_radius = 1.0
    if stem_faces:
        stem_vids = set()
        for f in stem_faces:
            stem_vids.update(f)
        if stem_vids:
            sv = verts[sorted(stem_vids)]
            stem_center_xy = np.array(sv[:, :2].mean(axis=0))
            stem_radius = float(np.linalg.norm(
                sv[:, :2] - stem_center_xy, axis=1).mean())

    # Extract each tracked leaf
    leaves = []
    for lid_str, stage_map in canonical_serialized.items():
        lid = int(lid_str)
        if str(idx) in stage_map:
            comp = set(stage_map[str(idx)])
            g1 = extract_leaf_g1(comp, verts, leaf_faces, lid,
                                 position_map[lid_str], n_samples,
                                 stem_center_xy=stem_center_xy,
                                 stem_radius=stem_radius,
                                 vert_normals=vert_normals)
            leaves.append(g1)

    leaves.sort(key=lambda l: l.position)
    leaf_bases = [(l.insertion_height, l.position)
                  for l in leaves if l.length > 1.0]
    stem = extract_stem_g1(verts, stem_faces, leaf_bases)

    vstage = count_developed_leaves(leaves, threshold_cm=5.0)
    day = vstage_to_day(vstage)

    return StageData(
        stage=stage_num, file=fpath_str,
        leaves=leaves, stem=stem,
        vstage=vstage, day_estimate=day,
    )


def process_all_stages(export_dir, stages_range=None, n_samples=20, verbose=False,
                       n_workers=None):
    """Parse all OBJ files and extract G1 for every stage.

    Uses multiprocessing for parallel G1 extraction when n_workers > 1.
    Returns list of StageData.
    """
    export_dir = Path(export_dir)
    manifest_path = export_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        files = [(s["stage"], export_dir / s["file"]) for s in manifest["stages"]]
    else:
        files = sorted(export_dir.glob("maize_stage_*.obj"))
        files = [(i + 1, f) for i, f in enumerate(files)]

    if stages_range:
        lo, hi = stages_range
        files = [(s, f) for s, f in files if lo <= s <= hi]

    # First pass: parse all, find components, track across stages
    all_verts = []
    all_groups = []
    all_comps = []
    all_vert_normals = []
    for stage_num, fpath in files:
        if verbose:
            print(f"  Parsing stage {stage_num}: {fpath.name}")
        verts, groups, vert_normals = parse_obj(fpath)
        all_verts.append(verts)
        all_groups.append(groups)
        all_vert_normals.append(vert_normals)
        # Find leaf components
        leaf_faces = []
        for gname, gfaces in groups.items():
            if "leaf" in gname.lower():
                leaf_faces.extend(gfaces)
        comps = find_connected_components(leaf_faces)
        all_comps.append(comps)

    # Track leaves across stages
    canonical = track_leaves_across_stages(all_comps)

    # Sort canonical by mean Z of first stage where leaf appears
    def leaf_sort_key(lid):
        for sidx in range(len(all_verts)):
            if sidx in canonical[lid]:
                comp = canonical[lid][sidx]
                return -np.mean(all_verts[sidx][list(comp)][:, 2])
        return 0
    sorted_lids = sorted(canonical.keys(), key=leaf_sort_key)
    position_map = {lid: pos + 1 for pos, lid in enumerate(sorted_lids)}

    # Serialize canonical for multiprocessing (convert sets to lists, keys to strings)
    canonical_ser = {}
    for lid, stage_map in canonical.items():
        canonical_ser[str(lid)] = {str(sidx): list(comp)
                                    for sidx, comp in stage_map.items()}
    position_map_ser = {str(lid): pos for lid, pos in position_map.items()}

    # Second pass: extract G1 per stage (parallel if n_workers > 1)
    effective_workers = min(n_workers or MAX_WORKERS, len(files))

    if effective_workers > 1 and len(files) > 1:
        if verbose:
            print(f"  Extracting G1 in parallel ({effective_workers} workers, "
                  f"{len(files)} stages)...")
        worker_args = [
            (idx, stage_num, fpath.name, all_verts[idx], all_groups[idx],
             all_vert_normals[idx], canonical_ser, position_map_ser, n_samples)
            for idx, (stage_num, fpath) in enumerate(files)
        ]
        stages = []
        with ProcessPoolExecutor(max_workers=effective_workers) as executor:
            futures = {executor.submit(_extract_stage_g1, a): a[1]
                       for a in worker_args}
            for future in as_completed(futures):
                stage_num = futures[future]
                try:
                    result = future.result()
                    stages.append(result)
                except Exception as e:
                    print(f"  WARNING: Stage {stage_num} extraction failed: {e}",
                          file=sys.stderr)
        stages.sort(key=lambda s: s.stage)
    else:
        # Sequential fallback
        stages = []
        for idx, (stage_num, fpath) in enumerate(files):
            if verbose:
                print(f"  Extracting G1 for stage {stage_num}...")
            verts = all_verts[idx]
            groups = all_groups[idx]
            vnorms = all_vert_normals[idx]

            leaf_faces = []
            for gname, gfaces in groups.items():
                if "leaf" in gname.lower():
                    leaf_faces.extend(gfaces)

            # Extract stem center for sheath detection
            stem_faces_seq = []
            for gname, gfaces in groups.items():
                if "stem" in gname.lower():
                    stem_faces_seq.extend(gfaces)
            stem_center_xy_seq = None
            stem_radius_seq = 1.0
            if stem_faces_seq:
                sv_ids = set()
                for f in stem_faces_seq:
                    sv_ids.update(f)
                if sv_ids:
                    sv = verts[sorted(sv_ids)]
                    stem_center_xy_seq = np.array(sv[:, :2].mean(axis=0))
                    stem_radius_seq = float(np.linalg.norm(
                        sv[:, :2] - stem_center_xy_seq, axis=1).mean())

            leaves = []
            for lid, stage_map in canonical.items():
                if idx in stage_map:
                    comp = stage_map[idx]
                    g1 = extract_leaf_g1(comp, verts, leaf_faces, lid,
                                         position_map[lid], n_samples,
                                         stem_center_xy=stem_center_xy_seq,
                                         stem_radius=stem_radius_seq,
                                         vert_normals=vnorms)
                    leaves.append(g1)

            leaves.sort(key=lambda l: l.position)

            stem_faces = []
            for gname, gfaces in groups.items():
                if "stem" in gname.lower():
                    stem_faces.extend(gfaces)
            leaf_bases = [(l.insertion_height, l.position)
                          for l in leaves if l.length > 1.0]
            stem = extract_stem_g1(verts, stem_faces, leaf_bases)

            vstage = count_developed_leaves(leaves, threshold_cm=5.0)
            day = vstage_to_day(vstage)

            stages.append(StageData(
                stage=stage_num, file=fpath.name,
                leaves=leaves, stem=stem,
                vstage=vstage, day_estimate=day,
            ))

    return stages


# ===========================================================================
# STEP 3: CPlantBox Capability Analysis
# ===========================================================================

def _fit_exponential_growth(lengths_per_stage, days_per_stage):
    """Fit L(t) = K * (1 - exp(-r/K * t)) to length trajectory.

    Returns (K, r, r_squared, residuals).
    """
    lengths = np.array(lengths_per_stage)
    days = np.array(days_per_stage)
    # Filter to stages where leaf exists
    mask = lengths > 0.1
    if mask.sum() < 3:
        return lengths.max(), 1.0, 0.0, lengths

    l, d = lengths[mask], days[mask]
    K_init = l.max() * 1.05

    def exp_growth(t, r, K):
        return K * (1 - np.exp(-(r / max(K, 0.1)) * t))

    try:
        popt, _ = curve_fit(exp_growth, d, l, p0=[2.0, K_init],
                            bounds=([0.01, l.max() * 0.8],
                                    [20.0, l.max() * 2.0]),
                            maxfev=5000)
        r_fit, K_fit = popt
        predicted = exp_growth(d, r_fit, K_fit)
        ss_res = np.sum((l - predicted) ** 2)
        ss_tot = np.sum((l - l.mean()) ** 2)
        r2 = 1 - ss_res / max(ss_tot, 1e-10)
        return float(K_fit), float(r_fit), float(r2), (l - predicted).tolist()
    except (RuntimeError, ValueError):
        return float(l.max()), 1.0, 0.0, l.tolist()


def _fit_linear_growth(lengths, days):
    """Fit L(t) = min(K, r*t)."""
    mask = np.array(lengths) > 0.1
    if mask.sum() < 2:
        return max(lengths), 1.0, 0.0
    l, d = np.array(lengths)[mask], np.array(days)[mask]
    K = l.max()

    def lin_growth(t, r):
        return np.minimum(K, r * t)

    try:
        popt, _ = curve_fit(lin_growth, d, l, p0=[2.0],
                            bounds=([0.01], [20.0]), maxfev=2000)
        r_fit = popt[0]
        predicted = lin_growth(d, r_fit)
        ss_res = np.sum((l - predicted) ** 2)
        ss_tot = np.sum((l - l.mean()) ** 2)
        r2 = 1 - ss_res / max(ss_tot, 1e-10)
        return float(K), float(r_fit), float(r2)
    except (RuntimeError, ValueError):
        return float(K), 1.0, 0.0


def _analyze_tropism_fit(curvature_profile, length):
    """Test if curvature is representable by CPlantBox's tropism model.

    CPlantBox tropism: random walk with constant sigma → expected curvature
    is roughly constant along leaf (or monotonically increasing with tropismExponent).

    Returns: (can_represent, best_sigma, best_tropism_age_frac, residual_rms,
              curvature_spline_needed).
    """
    if len(curvature_profile) < 3:
        return True, 0.0, 1.0, 0.0, None

    kappa = np.array(curvature_profile)
    n = len(kappa)
    frac = np.linspace(0, 1, n)

    # Model 1: Constant sigma (uniform curvature)
    mean_kappa = kappa.mean()
    resid_const = np.sqrt(np.mean((kappa - mean_kappa) ** 2))

    # Model 2: TropismAge (zero curvature for frac < f, then constant)
    best_f, best_resid = 0, resid_const
    for f_try in np.linspace(0, 0.95, 20):
        model = np.where(frac < f_try, 0, mean_kappa)
        r = np.sqrt(np.mean((kappa - model) ** 2))
        if r < best_resid:
            best_f = f_try
            best_resid = r

    # Model 3: TropismExponent (sigma increases as frac^exponent)
    best_exp_resid = resid_const
    for exp_try in [0.5, 1.0, 1.5, 2.0, 3.0]:
        model = mean_kappa * (frac ** exp_try)
        # Scale to match total curvature
        scale = kappa.sum() / max(model.sum(), 1e-10)
        model *= scale
        r = np.sqrt(np.mean((kappa - model) ** 2))
        if r < best_exp_resid:
            best_exp_resid = r

    # If best model residual is > 30% of mean curvature, need spline
    threshold = max(0.3 * mean_kappa, 0.005)  # 0.005 1/cm minimum
    best_model_resid = min(best_resid, best_exp_resid)
    can_represent = best_model_resid < threshold

    # Extract spline knots if needed
    spline_needed = None
    if not can_represent and n >= 5:
        # Sample 5 control points
        indices = np.linspace(0, n - 1, 5, dtype=int)
        spline_needed = {
            "phi": frac[indices].tolist(),
            "kappa": kappa[indices].tolist(),
        }

    return can_represent, float(mean_kappa), float(best_f), float(best_model_resid), spline_needed


def _analyze_width_evolution(width_profiles_across_stages):
    """Check if width profile shape is constant (just scales) or changes.

    CPlantBox leafGeometry is a static normalized profile — if the actual shape
    changes with growth, that's a gap.
    """
    valid = [(s, wp) for s, wp in width_profiles_across_stages if max(wp) > 0.01]
    if len(valid) < 3:
        return True, 0.0, None

    # Normalize each profile to peak=1
    normalized = []
    for s, wp in valid:
        mx = max(wp)
        if mx > 0.01:
            normalized.append(np.array(wp) / mx)

    # Compute pairwise cosine similarity
    sims = []
    for i in range(len(normalized)):
        for j in range(i + 1, len(normalized)):
            dot = np.dot(normalized[i], normalized[j])
            n1 = np.linalg.norm(normalized[i])
            n2 = np.linalg.norm(normalized[j])
            if n1 > 0 and n2 > 0:
                sims.append(dot / (n1 * n2))

    mean_sim = np.mean(sims) if sims else 1.0
    shape_changes = mean_sim < 0.95  # threshold for "constant shape"

    # If changing, extract profiles at 3 stages (early, mid, late)
    extracted = None
    if shape_changes and len(valid) >= 3:
        idx_early = 0
        idx_mid = len(valid) // 2
        idx_late = len(valid) - 1
        extracted = {
            "early_stage": valid[idx_early][0],
            "early_profile": normalized[idx_early].tolist(),
            "mid_stage": valid[idx_mid][0],
            "mid_profile": normalized[idx_mid].tolist(),
            "late_stage": valid[idx_late][0],
            "late_profile": normalized[idx_late].tolist(),
        }

    return not shape_changes, float(1.0 - mean_sim), extracted


def _analyze_asymmetry(asymmetry_profiles_across_stages):
    """Check if leaves have significant left/right asymmetry."""
    all_asym = []
    for s, ap in asymmetry_profiles_across_stages:
        all_asym.extend([abs(a) for a in ap])
    if not all_asym:
        return False, 0.0, None
    mean_asym = np.mean(all_asym)
    significant = mean_asym > 0.3  # > 3mm average asymmetry

    extracted = None
    if significant:
        # Use last (most mature) stage
        last_s, last_ap = asymmetry_profiles_across_stages[-1]
        n = len(last_ap)
        if n >= 5:
            indices = np.linspace(0, n - 1, 5, dtype=int)
            extracted = {
                "phi": np.linspace(0, 1, 5).tolist(),
                "offset": [last_ap[i] for i in indices],
            }
    return significant, float(mean_asym), extracted


def _analyze_twist(twist_profiles, lengths_per_stage=None):
    """Check if leaves have significant net twist (base to tip).

    Only considers stages where leaf is well-developed (length > 20cm)
    to avoid noise from small-leaf normal estimation.
    """
    net_twists = []
    for i, (s, tp) in enumerate(twist_profiles):
        if len(tp) > 1:
            # Only consider well-developed leaves
            if lengths_per_stage and i < len(lengths_per_stage):
                if lengths_per_stage[i] < 20:
                    continue
            net_twists.append(abs(tp[-1]))
    if not net_twists:
        return False, 0.0
    # Use median (robust to outliers from noisy normals)
    median_twist = float(np.median(net_twists))
    significant = median_twist > np.radians(15)
    return significant, float(np.degrees(median_twist))


def _analyze_cross_section(cs_angles_across_stages):
    """Check if cross-section V-angle varies significantly along leaf."""
    deviations = []
    for s, angles in cs_angles_across_stages:
        a = np.array(angles)
        if len(a) > 3:
            # Deviation from flat (π)
            dev = np.abs(a - np.pi)
            deviations.append(dev.mean())

    if not deviations:
        return False, 0.0, None

    mean_dev = np.mean(deviations)
    significant = mean_dev > np.radians(10)  # > 10° from flat

    extracted = None
    if significant and cs_angles_across_stages:
        last_s, last_a = cs_angles_across_stages[-1]
        n = len(last_a)
        if n >= 5:
            indices = np.linspace(0, n - 1, 5, dtype=int)
            extracted = {
                "phi": np.linspace(0, 1, 5).tolist(),
                "angle_deg": [float(np.degrees(last_a[i])) for i in indices],
            }
    return significant, float(np.degrees(mean_dev)), extracted


def _analyze_internode_pattern(internode_lengths):
    """Check if internode pattern fits CPlantBox's lnf models."""
    if len(internode_lengths) < 3:
        return "constant", internode_lengths, 1.0

    ln = np.array(internode_lengths)
    n = len(ln)
    x = np.arange(n)

    # Test constant
    const_resid = np.sqrt(np.mean((ln - ln.mean()) ** 2))

    # Test linear increasing (lnf=1)
    try:
        p = np.polyfit(x, ln, 1)
        lin_pred = np.polyval(p, x)
        lin_resid = np.sqrt(np.mean((ln - lin_pred) ** 2))
    except:
        lin_resid = const_resid

    # Test exponential (lnf=3)
    try:
        def exp_fn(x, a, b):
            return a * np.exp(b * x)
        popt, _ = curve_fit(exp_fn, x, ln, p0=[ln[0], 0.1], maxfev=2000)
        exp_pred = exp_fn(x, *popt)
        exp_resid = np.sqrt(np.mean((ln - exp_pred) ** 2))
    except:
        exp_resid = const_resid

    best = min([(const_resid, "constant"), (lin_resid, "linear"),
                (exp_resid, "exponential")])
    return best[1], ln.tolist(), float(best[0])


def _analyze_phyllotaxis(leaves_g1):
    """Measure azimuthal angle between consecutive leaf insertions."""
    if len(leaves_g1) < 2:
        return 180.0, []

    angles = []
    for i in range(1, len(leaves_g1)):
        if leaves_g1[i].length < 2 or leaves_g1[i - 1].length < 2:
            continue
        # Direction from stem to leaf base (XY plane)
        b0 = np.array(leaves_g1[i - 1].base_point[:2])
        b1 = np.array(leaves_g1[i].base_point[:2])
        # Stem center at that height (approximate as origin in XY)
        d0 = b0  # relative to stem center
        d1 = b1
        l0, l1 = np.linalg.norm(d0), np.linalg.norm(d1)
        if l0 > 0.1 and l1 > 0.1:
            cos_a = np.clip(np.dot(d0, d1) / (l0 * l1), -1, 1)
            angles.append(float(np.degrees(np.arccos(cos_a))))

    mean_angle = np.mean(angles) if angles else 180.0
    return mean_angle, angles


def analyze_cplantbox_gaps(stages, verbose=False):
    """Run all CPlantBox capability tests. Returns list of Gap objects."""
    gaps = []

    # Collect per-leaf trajectories across stages
    n_leaves = max(len(s.leaves) for s in stages) if stages else 0
    leaf_positions = set()
    for s in stages:
        for l in s.leaves:
            leaf_positions.add(l.position)

    days = [s.day_estimate for s in stages]

    for pos in sorted(leaf_positions):
        # Gather this leaf's data across stages
        lengths = []
        widths_max = []
        width_profiles = []
        curvature_profiles = []
        oop_profiles = []
        twist_profiles = []
        cs_profiles = []
        asymmetry_profiles = []
        insertion_angles = []

        for s in stages:
            leaf = next((l for l in s.leaves if l.position == pos), None)
            if leaf:
                lengths.append(leaf.length)
                widths_max.append(leaf.max_width)
                width_profiles.append((s.stage, leaf.width_profile_normalized))
                curvature_profiles.append((s.stage, leaf.curvature_profile))
                oop_profiles.append((s.stage, leaf.oop_curvature_profile))
                twist_profiles.append((s.stage, leaf.twist_profile))
                cs_profiles.append((s.stage, leaf.cross_section_angles))
                asymmetry_profiles.append((s.stage, leaf.asymmetry_profile))
                insertion_angles.append(leaf.insertion_angle)
            else:
                lengths.append(0)
                widths_max.append(0)

        if verbose:
            print(f"  Analyzing leaf position {pos}...")

        # --- Growth trajectory fit ---
        K_exp, r_exp, r2_exp, resid_exp = _fit_exponential_growth(lengths, days)
        K_lin, r_lin, r2_lin = _fit_linear_growth(lengths, days)

        if r2_exp < 0.85 and r2_lin < 0.85:
            gaps.append(Gap(
                parameter="growth_function",
                leaf_positions=[pos], stages=list(range(1, len(stages) + 1)),
                severity=float(np.sqrt(np.mean(np.array(resid_exp) ** 2))) if len(resid_exp) > 0 else 0,
                status="EXISTS_INSUFFICIENT",
                current_model=f"Exponential (R²={r2_exp:.2f}) and Linear (R²={r2_lin:.2f})",
                needed="Neither exponential nor linear growth fits well (R² < 0.85)",
                solution=f"Consider sigmoid/Gompertz growth function. Extracted: K={K_exp:.1f}cm, r={r_exp:.2f}cm/d",
                extracted_values={"K": K_exp, "r_exp": r_exp, "r2_exp": r2_exp,
                                  "r_lin": r_lin, "r2_lin": r2_lin,
                                  "lengths": lengths, "days": days},
                code_change="Add gf=3 (Gompertz) to growth.h GrowthFunction enum",
            ))

        # --- Tropism / curvature ---
        # Use most mature stage's curvature
        mature_curv = [cp for s, cp in curvature_profiles if len(cp) > 3]
        if mature_curv:
            curv = mature_curv[-1]
            can_repr, sigma, trop_age_frac, resid, spline = _analyze_tropism_fit(
                curv, max(lengths))
            if not can_repr:
                gaps.append(Gap(
                    parameter="leafCurvaturePhi/Kappa",
                    leaf_positions=[pos],
                    stages=[s for s, cp in curvature_profiles if len(cp) > 3],
                    severity=float(resid * max(lengths) * 0.5),  # approximate Chamfer
                    status="EXISTS_UNUSED",
                    current_model=f"Tropism random walk (sigma={sigma:.4f}), tropismAge frac={trop_age_frac:.2f}",
                    needed=f"Deterministic curvature profile not representable by constant sigma",
                    solution=f"Set leafCurvaturePhi/Kappa for subType {pos + 1}: {json.dumps(spline)}",
                    extracted_values={"spline": spline, "sigma_fit": sigma,
                                      "tropismAge_frac": trop_age_frac},
                ))

        # --- OOP curvature ---
        oop_data = [(s, op) for s, op in oop_profiles if len(op) > 2]
        if oop_data:
            max_oop = max(max(op) for _, op in oop_data if len(op) > 0)
            if max_oop > 0.01:  # > 0.01 1/cm
                last_oop = oop_data[-1][1]
                n = len(last_oop)
                indices = np.linspace(0, n - 1, 5, dtype=int) if n >= 5 else range(n)
                gaps.append(Gap(
                    parameter="leafOOPCurvPhi/Kappa",
                    leaf_positions=[pos],
                    stages=[s for s, _ in oop_data],
                    severity=float(max_oop * max(lengths) * 0.3),
                    status="EXISTS_UNUSED",
                    current_model="No OOP curvature set (defaults to 0)",
                    needed=f"OOP curvature up to {max_oop:.4f} 1/cm",
                    solution=f"Set leafOOPCurvPhi/Kappa for subType {pos + 1}",
                    extracted_values={
                        "phi": np.linspace(0, 1, len(indices)).tolist(),
                        "kappa": [float(last_oop[i]) for i in indices],
                    },
                ))

        # --- Width profile evolution ---
        shape_const, shape_change_mag, profiles = _analyze_width_evolution(width_profiles)
        if not shape_const and profiles:
            gaps.append(Gap(
                parameter="leafGeometry",
                leaf_positions=[pos],
                stages=list(range(1, len(stages) + 1)),
                severity=float(shape_change_mag * max(widths_max) * 5),
                status="EXISTS_INSUFFICIENT",
                current_model="Static normalized leafGeometry (scales with area only)",
                needed="Width profile shape changes during growth (not just scaling)",
                solution="Make leafGeometry age/maturity-dependent in Leaf.cpp getLeafVisX_()",
                extracted_values=profiles,
                code_change="Leaf::getLeafVisX_() should interpolate between early/mature leafGeometry based on length/lmax",
            ))

        # --- Asymmetry ---
        sig_asym, mean_asym, asym_vals = _analyze_asymmetry(asymmetry_profiles)
        if sig_asym and asym_vals:
            gaps.append(Gap(
                parameter="leafAsymmetryPhi/Offset",
                leaf_positions=[pos],
                stages=[s for s, _ in asymmetry_profiles],
                severity=float(mean_asym * 2),
                status="EXISTS_UNUSED",
                current_model="No asymmetry (symmetric width assumed)",
                needed=f"Mean asymmetry {mean_asym:.1f}cm left-right",
                solution=f"Set leafAsymmetryPhi/Offset for subType {pos + 1}",
                extracted_values=asym_vals,
            ))

        # --- Twist ---
        sig_twist, median_twist_deg = _analyze_twist(twist_profiles, lengths)
        if sig_twist:
            gaps.append(Gap(
                parameter="leafTwist",
                leaf_positions=[pos],
                stages=[s for s, _ in twist_profiles],
                severity=float(median_twist_deg * 0.005 * max(widths_max)),
                status="MISSING",
                current_model="CPlantBox has no leaf twist parameter (lofter-only)",
                needed=f"Median twist {median_twist_deg:.0f}° base-to-tip",
                solution="Add leafTwistPhi/Kappa to LeafRandomParameter, or handle in lofter only",
                extracted_values={"median_twist_deg": median_twist_deg},
                code_change="Add leafTwistPhi[], leafTwistAngle[] to leafparameter.h (same pattern as leafCurvature)",
            ))

        # --- Cross-section ---
        sig_cs, cs_dev_deg, cs_vals = _analyze_cross_section(cs_profiles)
        if sig_cs and cs_vals:
            gaps.append(Gap(
                parameter="leafCrossSectionPhi/Curv",
                leaf_positions=[pos],
                stages=[s for s, _ in cs_profiles],
                severity=float(cs_dev_deg * 0.02 * max(widths_max)),
                status="EXISTS_UNUSED",
                current_model="Flat cross-section (no V-angle)",
                needed=f"Cross-section deviates {cs_dev_deg:.0f}° from flat",
                solution=f"Set leafCrossSectionPhi/Curv for subType {pos + 1}",
                extracted_values=cs_vals,
            ))

    # --- Global gaps ---

    # Stem internode pattern
    for s in stages:
        if s.stem and s.stem.internode_lengths:
            pattern, ln_vals, resid = _analyze_internode_pattern(s.stem.internode_lengths)
            if resid > 1.0:  # > 1cm RMS residual
                gaps.append(Gap(
                    parameter="ln_vector",
                    leaf_positions=list(range(1, n_leaves + 1)),
                    stages=[s.stage],
                    severity=float(resid),
                    status="EXISTS_INSUFFICIENT",
                    current_model=f"Best fit: {pattern} (RMS={resid:.1f}cm)",
                    needed="Irregular internode pattern",
                    solution=f"Use explicit ln vector instead of lnf pattern: {ln_vals}",
                    extracted_values={"pattern": pattern, "ln": ln_vals},
                ))
            break  # Only check most mature stage

    # Phyllotaxis
    mature_stage = stages[-1] if stages else None
    if mature_stage:
        mean_angle, angles = _analyze_phyllotaxis(mature_stage.leaves)
        if abs(mean_angle - 180) > 15:  # Not distichous
            gaps.append(Gap(
                parameter="rotBeta",
                leaf_positions=list(range(1, n_leaves + 1)),
                stages=[mature_stage.stage],
                severity=float(abs(mean_angle - 180) * 0.05),
                status="EXISTS_INSUFFICIENT",
                current_model=f"Distichous (180° rotBeta)",
                needed=f"Mean inter-leaf angle: {mean_angle:.0f}°",
                solution=f"Set rotBeta to {np.radians(mean_angle):.3f} rad ({mean_angle:.0f}°)",
                extracted_values={"mean_angle_deg": mean_angle,
                                  "per_leaf_angles": angles},
            ))

    # Emergence timing
    vstages = [(s.stage, s.vstage) for s in stages]
    if len(vstages) > 3:
        vs = [v for _, v in vstages]
        diffs = [vs[i + 1] - vs[i] for i in range(len(vs) - 1)]
        phyllochron_stages = np.mean([1.0 / d if d > 0 else 0 for d in diffs]) if any(
            d > 0 for d in diffs) else 0
        gaps.append(Gap(
            parameter="ldelay/phyllochron",
            leaf_positions=list(range(1, n_leaves + 1)),
            stages=[s for s, _ in vstages],
            severity=0.5,  # moderate — timing affects all downstream
            status="EXISTS_UNUSED" if any(d != diffs[0] for d in diffs if d > 0) else "EXISTS_UNUSED",
            current_model="Constant phyllochron via ldelay",
            needed=f"V-stage progression: {vstages}",
            solution=f"Set phyllochron ≈ {3.5:.1f} days. Per-leaf ldelay values extracted.",
            extracted_values={"vstages": vstages, "days": days,
                              "emergence_stage": {pos: next(
                                  (s.stage for s in stages
                                   if any(l.position == pos and l.length > 5 for l in s.leaves)),
                                  None) for pos in sorted(leaf_positions)}},
        ))

    return gaps


# ===========================================================================
# STEP 4: Lofter Capability Analysis
# ===========================================================================

def _chamfer_distance(pts1, pts2):
    """Chamfer distance — GPU (torch) if available, else CPU (scipy KDTree)."""
    if len(pts1) == 0 or len(pts2) == 0:
        return 999.0
    if HAS_TORCH and HAS_CUDA and len(pts1) > 100:
        device = torch.device("cuda")
        t1 = torch.as_tensor(pts1, dtype=torch.float32, device=device)
        t2 = torch.as_tensor(pts2, dtype=torch.float32, device=device)
        dists = torch.cdist(t1.unsqueeze(0), t2.unsqueeze(0)).squeeze(0)
        d1 = dists.min(dim=1).values.mean()
        d2 = dists.min(dim=0).values.mean()
        return float((d1 + d2) / 2.0)
    else:
        tree1 = cKDTree(pts1)
        tree2 = cKDTree(pts2)
        d1, _ = tree2.query(pts1)
        d2, _ = tree1.query(pts2)
        return float((d1.mean() + d2.mean()) / 2.0)


def _mesh_to_points(verts, faces, n_samples=2000):
    """Sample points uniformly on mesh surface."""
    if len(faces) == 0:
        return verts[:n_samples] if len(verts) > 0 else np.zeros((0, 3))

    # Compute face areas for weighted sampling
    areas = []
    for f in faces:
        vs = verts[f]
        if len(vs) >= 3:
            e1 = vs[1] - vs[0]
            e2 = vs[2] - vs[0]
            areas.append(0.5 * np.linalg.norm(np.cross(e1, e2)))
        else:
            areas.append(0)
    areas = np.array(areas)
    total = areas.sum()
    if total < 1e-10:
        return verts[:n_samples]

    # Sample faces proportional to area
    probs = areas / total
    sampled_faces = np.random.choice(len(faces), size=n_samples, p=probs)

    points = []
    for fi in sampled_faces:
        vs = verts[faces[fi]]
        if len(vs) >= 3:
            r1, r2 = np.random.random(), np.random.random()
            if r1 + r2 > 1:
                r1, r2 = 1 - r1, 1 - r2
            pt = vs[0] + r1 * (vs[1] - vs[0]) + r2 * (vs[2] - vs[0])
            points.append(pt)
    return np.array(points) if points else verts[:n_samples]


def analyze_lofter_gaps(stages, export_dir, verbose=False):
    """Run lofter round-trip analysis. Returns list of Gap objects."""
    gaps = []

    # Try importing lofter
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from geometry.g1_to_g3 import loft_organs
        from geometry.cplantbox_adapter import _leaf_wave_params
        HAS_LOFTER = True
    except ImportError as e:
        if verbose:
            print(f"  WARNING: Cannot import lofter: {e}")
        return [Gap(
            parameter="lofter_import",
            leaf_positions=[], stages=[],
            severity=0, status="MISSING",
            current_model="N/A", needed="Lofter import failed",
            solution=f"Fix import: {e}",
            extracted_values={}, category="lofter",
        )]

    export_dir = Path(export_dir)

    # For each mature stage, do round-trip test
    test_stages = [stages[-1]]  # most mature stage
    if len(stages) > 4:
        test_stages.append(stages[len(stages) // 2])  # mid stage

    for stage_data in test_stages:
        if verbose:
            print(f"  Lofter round-trip for stage {stage_data.stage}...")

        # Load original mesh
        verts_orig, groups_orig, _ = parse_obj(export_dir / stage_data.file)
        leaf_faces_orig = []
        for gname, gfaces in groups_orig.items():
            if "leaf" in gname.lower():
                leaf_faces_orig.extend(gfaces)
        orig_points = _mesh_to_points(verts_orig, leaf_faces_orig, n_samples=3000)

        # Build organ dicts from extracted G1
        organs_bare = []
        organs_deformed = []
        for leaf in stage_data.leaves:
            if leaf.length < 2:
                continue
            skel = np.array(leaf.skeleton)
            w = np.array(leaf.widths)
            if len(skel) < 3 or w.max() < 0.1:
                continue

            organ = {
                "type": "leaf",
                "skeleton": skel,
                "widths": w,
                "organ_id": leaf.leaf_id,
                "name": f"leaf_{leaf.position}",
                "node_ids": list(range(len(skel))),
            }
            organs_bare.append(dict(organ))

            # With default deformations
            organ_def = dict(organ)
            try:
                rng = np.random.RandomState(42 + leaf.position)
                wave_params = _leaf_wave_params(leaf.length, rng,
                                                position=leaf.position)
                organ_def.update(wave_params)
            except Exception:
                pass
            organs_deformed.append(organ_def)

        if not organs_bare:
            continue

        # Tier 1: Bare skeleton+widths
        try:
            mesh_bare = loft_organs(organs_bare, subdivide=True, smooth=True)
            bare_points = mesh_bare.vertices
            chamfer_bare = _chamfer_distance(bare_points, orig_points)
        except Exception as e:
            if verbose:
                print(f"    Bare lofter failed: {e}")
            chamfer_bare = 999.0

        # Tier 2: With default deformations
        try:
            mesh_def = loft_organs(organs_deformed, subdivide=True, smooth=True)
            def_points = mesh_def.vertices
            chamfer_def = _chamfer_distance(def_points, orig_points)
        except Exception as e:
            if verbose:
                print(f"    Deformed lofter failed: {e}")
            chamfer_def = 999.0

        if verbose:
            print(f"    Stage {stage_data.stage}: Chamfer bare={chamfer_bare:.2f}cm, "
                  f"default_deform={chamfer_def:.2f}cm")

        # Record lofter gaps based on Chamfer
        if chamfer_bare > 1.0:
            gaps.append(Gap(
                parameter="lofter_skeleton_fidelity",
                leaf_positions=list(range(1, len(stage_data.leaves) + 1)),
                stages=[stage_data.stage],
                severity=float(chamfer_bare),
                status="EXISTS_INSUFFICIENT",
                current_model=f"Bare skeleton+widths Chamfer = {chamfer_bare:.2f}cm",
                needed="Chamfer < 1.0cm from skeleton+widths alone",
                solution="Skeleton extraction or lofter cross-section model needs improvement",
                extracted_values={"chamfer_bare": chamfer_bare,
                                  "chamfer_default_deform": chamfer_def},
                category="lofter",
            ))

        if chamfer_def > chamfer_bare * 0.5 and chamfer_def > 0.5:
            gaps.append(Gap(
                parameter="lofter_deformations",
                leaf_positions=list(range(1, len(stage_data.leaves) + 1)),
                stages=[stage_data.stage],
                severity=float(chamfer_def),
                status="EXISTS_INSUFFICIENT",
                current_model=f"Default deformations Chamfer = {chamfer_def:.2f}cm "
                              f"(bare = {chamfer_bare:.2f}cm)",
                needed="Deformations should reduce Chamfer by > 50%",
                solution="Optimize deformation parameters per-leaf via diff_lofter",
                extracted_values={"chamfer_bare": chamfer_bare,
                                  "chamfer_default_deform": chamfer_def,
                                  "improvement_pct": float(
                                      (1 - chamfer_def / max(chamfer_bare, 0.01)) * 100)},
                category="lofter",
            ))

    # Tier 3: Diff-lofter optimization (if torch available)
    if HAS_TORCH and stages:
        if verbose:
            print("  Attempting diff_lofter optimization...")
        try:
            sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
            from experimental.diff_lofter.lofter import loft_leaf as diff_loft_leaf
            from experimental.diff_lofter.lofter import resample_skeleton, compute_arc_fracs
            from experimental.diff_lofter.frames import compute_tangents, compute_binormal_field
            from experimental.diff_lofter.deformations import compute_deformations

            device = torch.device("cuda" if HAS_CUDA else "cpu")
            if verbose:
                print(f"    Diff-lofter on {device}")

            stage_data = test_stages[-1]  # most mature
            verts_orig, groups_orig, _ = parse_obj(export_dir / stage_data.file)
            leaf_faces_orig = []
            for gname, gfaces in groups_orig.items():
                if "leaf" in gname.lower():
                    leaf_faces_orig.extend(gfaces)

            # Build per-leaf target point clouds (NOT whole plant)
            leaf_comps_orig = find_connected_components(leaf_faces_orig)
            # Match to tracked leaves by vertex overlap
            leaf_target_map = {}  # leaf_id -> target points
            for leaf in stage_data.leaves:
                if leaf.length < 5:
                    continue
                leaf_vid = set()
                # Find component matching this leaf's vertex IDs
                for comp in leaf_comps_orig:
                    # Check if skeleton base point is near any component vertex
                    comp_verts = verts_orig[sorted(comp)]
                    base = np.array(leaf.base_point)
                    dists_to_base = np.linalg.norm(comp_verts - base, axis=1)
                    if dists_to_base.min() < 2.0:  # within 2cm
                        leaf_vid = comp
                        break
                if leaf_vid:
                    comp_faces = [f for f in leaf_faces_orig
                                  if all(v in leaf_vid for v in f)]
                    pts = _mesh_to_points(verts_orig, comp_faces, n_samples=500)
                    leaf_target_map[leaf.leaf_id] = pts

            # Optimize deformations for each leaf against its OWN target
            best_chamfers = []
            for leaf in stage_data.leaves:
                if leaf.length < 5:
                    continue
                if leaf.leaf_id not in leaf_target_map:
                    continue
                skel = np.array(leaf.skeleton)
                w = np.array(leaf.widths)
                if len(skel) < 3 or w.max() < 0.1:
                    continue

                # Per-leaf target (NOT whole plant)
                leaf_target = leaf_target_map[leaf.leaf_id]
                target_t = torch.as_tensor(leaf_target, dtype=torch.float32,
                                           device=device)

                skel_t = torch.as_tensor(skel, dtype=torch.float32, device=device)
                w_t = torch.as_tensor(w, dtype=torch.float32, device=device)
                skel_r, w_r = resample_skeleton(skel_t, w_t)
                tangents = compute_tangents(skel_r)
                binormals = compute_binormal_field(skel_r, tangents)
                arc_fracs = compute_arc_fracs(skel_r)

                # Optimizable deformation parameters
                params = {
                    "wave_normal_amp": torch.tensor(0.5, device=device, requires_grad=True),
                    "wave_lateral_amp": torch.tensor(0.3, device=device, requires_grad=True),
                    "twist_max": torch.tensor(0.3, device=device, requires_grad=True),
                    "curl_amp": torch.tensor(0.5, device=device, requires_grad=True),
                    "edge_ruffle_amp": torch.tensor(0.3, device=device, requires_grad=True),
                    "fold_amp": torch.tensor(0.2, device=device, requires_grad=True),
                }
                optimizer = torch.optim.Adam(list(params.values()), lr=DIFF_LOFTER_LR)

                best_loss = float("inf")
                for step in range(DIFF_LOFTER_STEPS):
                    optimizer.zero_grad()
                    deforms = compute_deformations(
                        arc_fracs,
                        params["wave_normal_amp"], 2.5, 0.0,
                        params["wave_lateral_amp"], 1.5, 0.0,
                        params["twist_max"],
                        params["curl_amp"], 1.0, 0.0,
                        params["edge_ruffle_amp"], 4.0, 0.0,
                        params["fold_amp"], 2.0, 0.0,
                    )
                    verts = diff_loft_leaf(skel_r, w_r, deforms, tangents, binormals)
                    dists = torch.cdist(verts.unsqueeze(0), target_t.unsqueeze(0)).squeeze(0)
                    d1 = dists.min(dim=1).values.mean()
                    d2 = dists.min(dim=0).values.mean()
                    loss = (d1 + d2) / 2.0
                    loss.backward()
                    optimizer.step()
                    if loss.item() < best_loss:
                        best_loss = loss.item()

                best_chamfers.append(best_loss)
                if verbose:
                    print(f"      Leaf {leaf.position}: optimized Chamfer = {best_loss:.2f}cm")

            if best_chamfers:
                chamfer_optimized = float(np.mean(best_chamfers))
                if verbose:
                    print(f"    Optimized Chamfer: {chamfer_optimized:.2f}cm "
                          f"(per-leaf mean, {len(best_chamfers)} leaves)")
                gaps.append(Gap(
                    parameter="lofter_optimized_deformations",
                    leaf_positions=list(range(1, len(stage_data.leaves) + 1)),
                    stages=[stage_data.stage],
                    severity=float(chamfer_optimized),
                    status="EXISTS_INSUFFICIENT" if chamfer_optimized > 0.5 else "EXISTS_UNUSED",
                    current_model=f"Diff-lofter optimized Chamfer = {chamfer_optimized:.2f}cm",
                    needed="Chamfer < 0.5cm with optimized deformations",
                    solution="Residual after optimization = fundamental lofter limitation",
                    extracted_values={
                        "chamfer_optimized": chamfer_optimized,
                        "per_leaf_chamfers": best_chamfers,
                        "n_steps": DIFF_LOFTER_STEPS,
                    },
                    category="lofter",
                ))

        except ImportError as e:
            if verbose:
                print(f"    Diff-lofter not available: {e}")
        except Exception as e:
            if verbose:
                print(f"    Diff-lofter optimization failed: {e}")

    return gaps


# ===========================================================================
# STEP 5: ML Impact Ranking + Report
# ===========================================================================

def rank_gaps(gaps):
    """Rank gaps by severity × breadth (how many leaves × stages affected)."""
    for g in gaps:
        breadth = len(g.leaf_positions) * max(len(g.stages), 1)
        g.severity = g.severity * (1 + 0.1 * breadth)  # scale by breadth
    return sorted(gaps, key=lambda g: -g.severity)


def ml_feature_importance(stages, gaps):
    """Use gradient boosting to identify which G1 traits predict Chamfer error.

    Returns feature importance dict or None if sklearn unavailable.
    """
    if not HAS_SKLEARN:
        return None

    # Build feature matrix from leaf traits
    features = []
    targets = []  # Use gap severity as proxy for error
    feature_names = ["length", "max_width", "insertion_angle",
                     "mean_curvature", "max_oop_curvature", "max_twist",
                     "mean_cs_deviation", "asymmetry", "area",
                     "position", "stage"]

    for stage_data in stages:
        for leaf in stage_data.leaves:
            if leaf.length < 2:
                continue
            curv = np.array(leaf.curvature_profile)
            oop = np.array(leaf.oop_curvature_profile)
            twist = np.array(leaf.twist_profile)
            cs = np.array(leaf.cross_section_angles)
            asym = np.array(leaf.asymmetry_profile)

            row = [
                leaf.length,
                leaf.max_width,
                leaf.insertion_angle,
                float(curv.mean()) if len(curv) > 0 else 0,
                float(oop.max()) if len(oop) > 0 else 0,
                float(twist.max()) if len(twist) > 0 else 0,
                float(np.abs(cs - np.pi).mean()) if len(cs) > 0 else 0,
                float(np.abs(asym).mean()) if len(asym) > 0 else 0,
                leaf.area,
                leaf.position,
                stage_data.stage,
            ]
            features.append(row)

            # Target: aggregate severity of gaps affecting this leaf position
            leaf_gap_severity = sum(
                g.severity for g in gaps
                if leaf.position in g.leaf_positions
            )
            targets.append(leaf_gap_severity)

    if len(features) < 10:
        return None

    X = np.array(features)
    y = np.array(targets)

    model = GradientBoostingRegressor(n_estimators=100, max_depth=3,
                                       random_state=42)
    # Cross-validated R²
    scores = cross_val_score(model, X, y, cv=min(5, len(X)), scoring="r2")
    model.fit(X, y)

    importance = dict(zip(feature_names, model.feature_importances_.tolist()))
    importance["cross_val_r2"] = float(scores.mean())

    return importance


def generate_report(stages, gaps_cpb, gaps_lofter, ml_importance, output_dir):
    """Generate JSON outputs and HTML report."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- JSON outputs ---

    # Extracted traits
    traits = {}
    for s in stages:
        stage_traits = {
            "stage": s.stage, "vstage": s.vstage,
            "day_estimate": s.day_estimate,
            "leaves": [asdict(l) for l in s.leaves],
        }
        if s.stem:
            stage_traits["stem"] = asdict(s.stem)
        traits[f"stage_{s.stage:02d}"] = stage_traits

    (output_dir / "extracted_traits.json").write_text(
        json.dumps(traits, indent=2, default=str))

    # CPlantBox gaps
    cpb_gaps_json = [asdict(g) for g in gaps_cpb]
    (output_dir / "gaps_cplantbox.json").write_text(
        json.dumps(cpb_gaps_json, indent=2, default=str))

    # Lofter gaps
    lofter_gaps_json = [asdict(g) for g in gaps_lofter]
    (output_dir / "gaps_lofter.json").write_text(
        json.dumps(lofter_gaps_json, indent=2, default=str))

    # Ranked solutions
    all_gaps = rank_gaps(gaps_cpb + gaps_lofter)
    solutions = []
    for i, g in enumerate(all_gaps):
        solutions.append({
            "rank": i + 1,
            "parameter": g.parameter,
            "category": g.category,
            "severity": g.severity,
            "status": g.status,
            "solution": g.solution,
            "leaf_positions": g.leaf_positions,
            "extracted_values": g.extracted_values,
            "code_change": g.code_change,
        })
    (output_dir / "solutions_ranked.json").write_text(
        json.dumps(solutions, indent=2, default=str))

    # ML importance
    if ml_importance:
        (output_dir / "ml_feature_importance.json").write_text(
            json.dumps(ml_importance, indent=2))

    # --- V-stage summary ---
    vstage_summary = []
    for s in stages:
        vstage_summary.append({
            "stage": s.stage,
            "vstage": s.vstage,
            "day": s.day_estimate,
            "n_developed": len([l for l in s.leaves if l.length > 5]),
            "max_leaf_length": max((l.length for l in s.leaves), default=0),
            "stem_height": s.stem.height if s.stem else 0,
        })
    (output_dir / "vstage_summary.json").write_text(
        json.dumps(vstage_summary, indent=2))

    # --- Matplotlib figures ---
    if HAS_MPL:
        _generate_figures(stages, gaps_cpb, gaps_lofter, ml_importance, output_dir)

    # --- HTML report ---
    _generate_html_report(stages, all_gaps, ml_importance, vstage_summary, output_dir)

    # --- Console summary ---
    print("\n" + "=" * 70)
    print("REVERSE-ENGINEERING ANALYSIS COMPLETE")
    print("=" * 70)

    print(f"\nStages analyzed: {len(stages)}")
    print(f"V-stage range: V{stages[0].vstage} → V{stages[-1].vstage}")
    print(f"Day range: {stages[0].day_estimate:.0f} → {stages[-1].day_estimate:.0f}")

    print(f"\n--- TOP GAPS (CPlantBox Growth) [{len(gaps_cpb)} found] ---")
    for g in sorted(gaps_cpb, key=lambda x: -x.severity)[:5]:
        leaves = ",".join(str(p) for p in g.leaf_positions[:5])
        print(f"  [{g.status}] severity={g.severity:.2f}cm  {g.parameter} (leaves {leaves})")
        print(f"    Current: {g.current_model[:80]}")
        print(f"    Solution: {g.solution[:100]}")
        if g.code_change:
            print(f"    C++ change: {g.code_change[:100]}")

    print(f"\n--- TOP GAPS (Lofter) [{len(gaps_lofter)} found] ---")
    for g in sorted(gaps_lofter, key=lambda x: -x.severity)[:5]:
        print(f"  [{g.status}] severity={g.severity:.2f}cm  {g.parameter}")
        print(f"    Solution: {g.solution[:100]}")

    if ml_importance:
        print(f"\n--- ML Feature Importance (R²={ml_importance.get('cross_val_r2', 0):.2f}) ---")
        sorted_imp = sorted(
            [(k, v) for k, v in ml_importance.items() if k != "cross_val_r2"],
            key=lambda x: -x[1])
        for name, imp in sorted_imp[:5]:
            print(f"  {name:25s} {imp:.3f}")

    print(f"\nOutputs saved to: {output_dir}")
    print(f"  extracted_traits.json    — full G1 data (16 stages × 16 leaves)")
    print(f"  gaps_cplantbox.json      — CPlantBox gaps with solutions")
    print(f"  gaps_lofter.json         — Lofter gaps with solutions")
    print(f"  solutions_ranked.json    — All solutions ranked by impact")


def _generate_figures(stages, gaps_cpb, gaps_lofter, ml_importance, output_dir):
    """Generate matplotlib figures."""
    # Fig 1: Growth trajectories per leaf
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Collect per-leaf trajectories
    positions = sorted(set(l.position for s in stages for l in s.leaves))
    stage_nums = [s.stage for s in stages]

    # 1a: Leaf lengths
    ax = axes[0, 0]
    for pos in positions:
        lengths = []
        for s in stages:
            leaf = next((l for l in s.leaves if l.position == pos), None)
            lengths.append(leaf.length if leaf else 0)
        ax.plot(stage_nums, lengths, "-o", markersize=3, label=f"Leaf {pos}")
    ax.set_xlabel("Growth Stage")
    ax.set_ylabel("Leaf Length (cm)")
    ax.set_title("Leaf Length Trajectories")
    ax.legend(fontsize=6, ncol=3)

    # 1b: Max widths
    ax = axes[0, 1]
    for pos in positions:
        widths = []
        for s in stages:
            leaf = next((l for l in s.leaves if l.position == pos), None)
            widths.append(leaf.max_width if leaf else 0)
        ax.plot(stage_nums, widths, "-o", markersize=3, label=f"Leaf {pos}")
    ax.set_xlabel("Growth Stage")
    ax.set_ylabel("Max Width (cm)")
    ax.set_title("Leaf Width Trajectories")

    # 1c: Insertion angles
    ax = axes[1, 0]
    for pos in positions:
        angles = []
        for s in stages:
            leaf = next((l for l in s.leaves if l.position == pos), None)
            angles.append(np.degrees(leaf.insertion_angle) if leaf and leaf.length > 2 else np.nan)
        ax.plot(stage_nums, angles, "-o", markersize=3, label=f"Leaf {pos}")
    ax.set_xlabel("Growth Stage")
    ax.set_ylabel("Insertion Angle (°)")
    ax.set_title("Insertion Angle Trajectories")

    # 1d: V-stage + stem height
    ax = axes[1, 1]
    vstages = [s.vstage for s in stages]
    heights = [s.stem.height if s.stem else 0 for s in stages]
    ax.plot(stage_nums, vstages, "b-o", label="V-stage", markersize=4)
    ax2 = ax.twinx()
    ax2.plot(stage_nums, heights, "r-s", label="Stem height", markersize=4)
    ax.set_xlabel("Growth Stage")
    ax.set_ylabel("V-stage", color="b")
    ax2.set_ylabel("Stem Height (cm)", color="r")
    ax.set_title("V-stage and Stem Height")

    plt.tight_layout()
    fig.savefig(output_dir / "growth_trajectories.png", dpi=150)
    plt.close(fig)

    # Fig 2: Gap severity bar chart
    all_gaps = sorted(gaps_cpb + gaps_lofter, key=lambda g: -g.severity)[:15]
    if all_gaps:
        fig, ax = plt.subplots(figsize=(12, 6))
        labels = [f"{g.parameter}\n({g.category})" for g in all_gaps]
        severities = [g.severity for g in all_gaps]
        colors = ["#e74c3c" if g.category == "cplantbox" else "#3498db" for g in all_gaps]
        ax.barh(range(len(labels)), severities, color=colors)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("Severity (cm)")
        ax.set_title("Top Gaps by Severity (red=CPlantBox, blue=Lofter)")
        ax.invert_yaxis()
        plt.tight_layout()
        fig.savefig(output_dir / "gap_severity.png", dpi=150)
        plt.close(fig)

    # Fig 3: Feature importance
    if ml_importance:
        imp = {k: v for k, v in ml_importance.items() if k != "cross_val_r2"}
        if imp:
            fig, ax = plt.subplots(figsize=(8, 5))
            sorted_items = sorted(imp.items(), key=lambda x: x[1])
            ax.barh([x[0] for x in sorted_items],
                    [x[1] for x in sorted_items], color="#2ecc71")
            ax.set_xlabel("Feature Importance")
            ax.set_title(f"ML Feature Importance (CV R²={ml_importance.get('cross_val_r2', 0):.2f})")
            plt.tight_layout()
            fig.savefig(output_dir / "feature_importance.png", dpi=150)
            plt.close(fig)


def _generate_html_report(stages, ranked_gaps, ml_importance, vstage_summary, output_dir):
    """Generate standalone HTML report."""
    html = ["""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>Maize Reverse-Engineering Report</title>
<style>
body { font-family: -apple-system, sans-serif; max-width: 1100px; margin: 0 auto; padding: 20px; }
table { border-collapse: collapse; width: 100%; margin: 15px 0; }
th, td { border: 1px solid #ddd; padding: 8px; text-align: left; font-size: 13px; }
th { background: #f5f5f5; }
.severity-high { color: #e74c3c; font-weight: bold; }
.severity-med { color: #f39c12; }
.severity-low { color: #27ae60; }
.gap-cpb { background: #fdecea; }
.gap-lofter { background: #eaf2f8; }
code { background: #f0f0f0; padding: 2px 5px; border-radius: 3px; font-size: 12px; }
pre { background: #f8f8f8; padding: 10px; border-radius: 5px; overflow-x: auto; font-size: 12px; }
h2 { border-bottom: 2px solid #333; padding-bottom: 5px; }
img { max-width: 100%; border: 1px solid #ddd; margin: 10px 0; }
</style></head><body>
<h1>Maize Growth Stage Reverse-Engineering Report</h1>
"""]

    # V-stage table
    html.append("<h2>V-Stage Mapping</h2><table><tr>"
                "<th>Stage</th><th>V-stage</th><th>Est. Day</th>"
                "<th>Developed Leaves</th><th>Max Leaf Length (cm)</th>"
                "<th>Stem Height (cm)</th></tr>")
    for v in vstage_summary:
        html.append(f"<tr><td>{v['stage']}</td><td>V{v['vstage']}</td>"
                     f"<td>{v['day']:.0f}</td><td>{v['n_developed']}</td>"
                     f"<td>{v['max_leaf_length']:.1f}</td>"
                     f"<td>{v['stem_height']:.1f}</td></tr>")
    html.append("</table>")

    # Figures
    for fig_name, title in [("growth_trajectories.png", "Growth Trajectories"),
                             ("gap_severity.png", "Gap Severity"),
                             ("feature_importance.png", "ML Feature Importance")]:
        if (output_dir / fig_name).exists():
            html.append(f"<h2>{title}</h2><img src='{fig_name}'>")

    # Ranked solutions table
    html.append("<h2>Ranked Solutions</h2><table><tr>"
                "<th>#</th><th>Parameter</th><th>Category</th>"
                "<th>Severity</th><th>Status</th><th>Solution</th>"
                "<th>C++ Change</th></tr>")
    for g in ranked_gaps[:20]:
        sev_class = "severity-high" if g.severity > 2 else \
                    "severity-med" if g.severity > 0.5 else "severity-low"
        row_class = "gap-cpb" if g.category == "cplantbox" else "gap-lofter"
        code = g.code_change or "—"
        html.append(f"<tr class='{row_class}'>"
                     f"<td>{ranked_gaps.index(g) + 1}</td>"
                     f"<td><code>{g.parameter}</code></td>"
                     f"<td>{g.category}</td>"
                     f"<td class='{sev_class}'>{g.severity:.2f}</td>"
                     f"<td>{g.status}</td>"
                     f"<td>{g.solution}</td>"
                     f"<td>{code}</td></tr>")
    html.append("</table>")

    # ML importance table
    if ml_importance:
        html.append(f"<h2>ML Feature Importance (CV R²={ml_importance.get('cross_val_r2', 0):.2f})</h2>")
        html.append("<table><tr><th>Feature</th><th>Importance</th></tr>")
        sorted_imp = sorted(
            [(k, v) for k, v in ml_importance.items() if k != "cross_val_r2"],
            key=lambda x: -x[1])
        for name, imp in sorted_imp:
            html.append(f"<tr><td>{name}</td><td>{imp:.4f}</td></tr>")
        html.append("</table>")

    html.append("</body></html>")
    (output_dir / "report.html").write_text("\n".join(html))


# ===========================================================================
# STEP 0: Export skeletons as OBJ for visual verification
# ===========================================================================

def export_skeletons_obj(stages, output_dir):
    """Export extracted skeletons as OBJ polylines for visual inspection."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for stage_data in stages:
        lines = [f"# Extracted skeletons for stage {stage_data.stage}"]
        vert_offset = 1
        for leaf in stage_data.leaves:
            skel = np.array(leaf.skeleton)
            if len(skel) < 2:
                continue
            lines.append(f"o leaf_{leaf.position}")
            for pt in skel:
                lines.append(f"v {pt[0]:.4f} {pt[1]:.4f} {pt[2]:.4f}")
            # Polyline
            indices = " ".join(str(vert_offset + i) for i in range(len(skel)))
            lines.append(f"l {indices}")
            vert_offset += len(skel)

        # Stem
        if stage_data.stem and stage_data.stem.skeleton:
            skel = np.array(stage_data.stem.skeleton)
            lines.append("o stem")
            for pt in skel:
                lines.append(f"v {pt[0]:.4f} {pt[1]:.4f} {pt[2]:.4f}")
            indices = " ".join(str(vert_offset + i) for i in range(len(skel)))
            lines.append(f"l {indices}")

        out_path = output_dir / f"skeleton_stage_{stage_data.stage:02d}.obj"
        out_path.write_text("\n".join(lines))


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Reverse-engineer maize OBJ growth stages → CPlantBox gap analysis")
    parser.add_argument("export_dir", help="Directory with maize_stage_*.obj files")
    parser.add_argument("--output", "-o", default="output/reverse_engineer",
                        help="Output directory")
    parser.add_argument("--skip-cplantbox", action="store_true",
                        help="Skip CPlantBox capability analysis")
    parser.add_argument("--skip-lofter", action="store_true",
                        help="Skip lofter round-trip analysis")
    parser.add_argument("--skip-difflofter", action="store_true",
                        help="Skip diff_lofter GPU optimization")
    parser.add_argument("--stages", default="1-16",
                        help="Stage range to process (e.g., '1-16' or '5-10')")
    parser.add_argument("--n-samples", type=int, default=20,
                        help="Skeleton resampling points per leaf")
    parser.add_argument("--workers", "-w", type=int, default=None,
                        help=f"Max parallel workers (default: {MAX_WORKERS}, env RE_MAX_WORKERS)")
    parser.add_argument("--diff-lofter-steps", type=int, default=None,
                        help=f"Gradient optimization steps (default: {DIFF_LOFTER_STEPS}, env RE_DIFF_LOFTER_STEPS)")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    # Parse stage range
    parts = args.stages.split("-")
    stages_range = (int(parts[0]), int(parts[1])) if len(parts) == 2 else (1, 16)

    print("=" * 70)
    print("MAIZE REVERSE-ENGINEERING PIPELINE")
    print("=" * 70)
    print(f"Export dir: {args.export_dir}")
    diff_steps = args.diff_lofter_steps if args.diff_lofter_steps is not None else DIFF_LOFTER_STEPS

    print(f"Output dir: {args.output}")
    print(f"Stages: {stages_range[0]}-{stages_range[1]}")
    n_workers = args.workers or MAX_WORKERS
    print(f"Workers: {n_workers} | GPU: {'AX4000' if HAS_CUDA else 'none'} | "
          f"Torch: {'yes' if HAS_TORCH else 'no'}")

    # Step 1-2: Parse + Extract
    print("\n[Step 1-2] Parsing OBJ files and extracting G1 skeletons...")
    stages = process_all_stages(args.export_dir, stages_range,
                                args.n_samples, args.verbose,
                                n_workers=n_workers)
    print(f"  Extracted {len(stages)} stages, "
          f"{sum(len(s.leaves) for s in stages)} total leaf G1s")

    # Export skeletons for visual check
    skel_dir = Path(args.output) / "skeletons"
    export_skeletons_obj(stages, skel_dir)
    print(f"  Skeleton OBJs exported to {skel_dir}/")

    # Step 3: CPlantBox analysis
    gaps_cpb = []
    if not args.skip_cplantbox:
        print("\n[Step 3] Analyzing CPlantBox capability gaps...")
        gaps_cpb = analyze_cplantbox_gaps(stages, args.verbose)
        print(f"  Found {len(gaps_cpb)} CPlantBox gaps")
    else:
        print("\n[Step 3] Skipped (--skip-cplantbox)")

    # Step 4: Lofter analysis
    gaps_lofter = []
    if not args.skip_lofter:
        print("\n[Step 4] Analyzing lofter capability gaps...")
        gaps_lofter = analyze_lofter_gaps(stages, args.export_dir, args.verbose)
        print(f"  Found {len(gaps_lofter)} lofter gaps")
    else:
        print("\n[Step 4] Skipped (--skip-lofter)")

    # Step 5: ML + Report
    print("\n[Step 5] Running ML analysis and generating report...")
    ml_importance = ml_feature_importance(stages, gaps_cpb + gaps_lofter)
    if ml_importance:
        print(f"  ML model CV R²: {ml_importance.get('cross_val_r2', 0):.2f}")
    elif not HAS_SKLEARN:
        print("  ML analysis skipped (sklearn not available)")
    else:
        print("  ML analysis skipped (insufficient data)")

    generate_report(stages, gaps_cpb, gaps_lofter, ml_importance, args.output)


if __name__ == "__main__":
    main()
