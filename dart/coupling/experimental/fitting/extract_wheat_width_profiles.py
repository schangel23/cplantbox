"""Extract per-leaf width profiles from MVS-Pheno wheat scan data.

Reads the paper's extraction outputs (PLY skeletons, PCD point clouds,
stem attachment points) and measures half-width at each skeleton node
by projecting nearby PCD points onto the local cross-section plane.

Outputs a JSON artifact with per-leaf skeleton + width arrays, decoupling
the open3d-dependent extraction from the torch-dependent reconstruction.

Usage (in darteb_venv):
    python -m dart.coupling.experimental.fitting.extract_wheat_width_profiles \
        /path/to/parameterextraction/wheat1/ -o wheat1_profiles.json
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def parse_ply_skeleton(ply_path: str) -> np.ndarray:
    """Parse ASCII PLY line-set file and trace the ordered skeleton path.

    The PLY files from parameterextraction contain unordered vertices with
    an edge list defining connectivity. We build an adjacency graph, find
    the two degree-1 endpoints, and walk from one to the other.

    Args:
        ply_path: Path to leaf_in_N.ply file.

    Returns:
        (N, 3) ordered skeleton points from one endpoint to the other.
    """
    with open(ply_path, 'r') as f:
        lines = f.readlines()

    # Parse header
    n_vertices = 0
    n_edges = 0
    header_end = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith('element vertex'):
            n_vertices = int(stripped.split()[-1])
        elif stripped.startswith('element edge'):
            n_edges = int(stripped.split()[-1])
        elif stripped == 'end_header':
            header_end = i + 1
            break

    # Parse vertices
    vertices = np.zeros((n_vertices, 3))
    for i in range(n_vertices):
        parts = lines[header_end + i].strip().split()
        vertices[i] = [float(parts[0]), float(parts[1]), float(parts[2])]

    # Parse edges
    edges = []
    edge_start = header_end + n_vertices
    for i in range(n_edges):
        parts = lines[edge_start + i].strip().split()
        edges.append((int(parts[0]), int(parts[1])))

    # Build adjacency
    adj = defaultdict(list)
    for v1, v2 in edges:
        adj[v1].append(v2)
        adj[v2].append(v1)

    # Find degree-1 endpoints
    endpoints = [v for v in range(n_vertices) if len(adj[v]) == 1]
    if len(endpoints) != 2:
        # Fallback: if graph is weird, just return vertices in order
        return vertices

    # Walk from first endpoint to second
    path = [endpoints[0]]
    visited = {endpoints[0]}
    current = endpoints[0]
    while len(path) < n_vertices:
        found = False
        for neighbor in adj[current]:
            if neighbor not in visited:
                path.append(neighbor)
                visited.add(neighbor)
                current = neighbor
                found = True
                break
        if not found:
            break

    return vertices[path]


def orient_skeleton_to_base(
    skeleton: np.ndarray,
    stem_attachment_point: np.ndarray,
) -> np.ndarray:
    """Orient skeleton so index 0 is the base (closest to stem attachment).

    Args:
        skeleton: (N, 3) ordered skeleton points.
        stem_attachment_point: (3,) stem attachment coordinate.

    Returns:
        (N, 3) skeleton with base at index 0.
    """
    d_start = np.linalg.norm(skeleton[0] - stem_attachment_point)
    d_end = np.linalg.norm(skeleton[-1] - stem_attachment_point)
    if d_end < d_start:
        return skeleton[::-1].copy()
    return skeleton


def bridge_skeleton(
    skeleton: np.ndarray,
    stem_attachment_point: np.ndarray,
    min_gap: float = 0.3,
) -> np.ndarray:
    """Prepend stem attachment point if there is a gap to skeleton start.

    Args:
        skeleton: (N, 3) oriented skeleton (base at index 0).
        stem_attachment_point: (3,) stem attachment coordinate.
        min_gap: Minimum gap in cm to warrant bridging.

    Returns:
        (N+1, 3) or (N, 3) skeleton with attachment point prepended if needed.
    """
    gap = np.linalg.norm(skeleton[0] - stem_attachment_point)
    if gap > min_gap:
        return np.vstack([stem_attachment_point.reshape(1, 3), skeleton])
    return skeleton


def _compute_tangents_np(skeleton: np.ndarray) -> np.ndarray:
    """Central-difference tangent vectors (numpy version).

    Args:
        skeleton: (N, 3) ordered points.

    Returns:
        (N, 3) unit tangent vectors.
    """
    n = skeleton.shape[0]
    tangents = np.zeros_like(skeleton)
    if n < 2:
        tangents[:] = [0, 0, 1]
        return tangents

    tangents[0] = skeleton[1] - skeleton[0]
    tangents[-1] = skeleton[-1] - skeleton[-2]
    if n > 2:
        tangents[1:-1] = skeleton[2:] - skeleton[:-2]

    norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return tangents / norms


def extract_width_profile(
    skeleton: np.ndarray,
    leaf_pcd_points: np.ndarray,
    width_cap: float = 2.0,
    min_width: float = 0.15,
    percentile: float = 90.0,
) -> np.ndarray:
    """Measure half-width at each skeleton node via cross-section projection.

    At each node: select nearby PCD points, project onto the cross-section
    plane (perpendicular to tangent), measure the 90th percentile of
    perpendicular distances from the skeleton.

    Args:
        skeleton: (N, 3) ordered skeleton points.
        leaf_pcd_points: (M, 3) leaf point cloud.
        width_cap: Maximum half-width in cm (values above are curl artifacts).
        min_width: Minimum half-width in cm.
        percentile: Percentile of distances to use (robust to outliers).

    Returns:
        (N,) half-widths in cm.
    """
    n = skeleton.shape[0]
    tangents = _compute_tangents_np(skeleton)

    # Compute inter-node distances for adaptive search radius
    seg_dists = np.linalg.norm(np.diff(skeleton, axis=0), axis=1)

    # Global fallback width from PCD bounding box
    if leaf_pcd_points.shape[0] >= 3:
        centered = leaf_pcd_points - leaf_pcd_points.mean(axis=0)
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        proj2 = centered @ vh[1]
        global_half_width = (proj2.max() - proj2.min()) / 2.0
        global_half_width = np.clip(global_half_width, min_width, width_cap)
    else:
        global_half_width = 0.5

    half_widths = np.zeros(n)

    for i in range(n):
        # Adaptive search radius: half of sum of adjacent segment lengths
        if i == 0:
            radius = seg_dists[0] if len(seg_dists) > 0 else 1.0
        elif i == n - 1:
            radius = seg_dists[-1] if len(seg_dists) > 0 else 1.0
        else:
            radius = 0.5 * (seg_dists[i - 1] + seg_dists[i])
        radius = np.clip(radius, 0.3, 2.0)

        # Select points within radius
        dists_to_node = np.linalg.norm(leaf_pcd_points - skeleton[i], axis=1)
        mask = dists_to_node <= radius
        nearby = leaf_pcd_points[mask]

        if nearby.shape[0] < 3:
            half_widths[i] = global_half_width
            continue

        # Project onto cross-section plane (remove tangent component)
        t = tangents[i]
        vecs = nearby - skeleton[i]
        along = (vecs @ t).reshape(-1, 1) * t.reshape(1, 3)
        perp = vecs - along
        perp_dists = np.linalg.norm(perp, axis=1)

        half_widths[i] = np.percentile(perp_dists, percentile)

    # Clamp
    half_widths = np.clip(half_widths, min_width, width_cap)
    return half_widths


def extract_all_profiles(
    scan_dir: str,
    output_path: str | None = None,
    width_cap: float = 2.0,
) -> dict:
    """Extract width profiles for all leaves in a scan directory.

    Reads the paper's outputs: PLY skeletons, PCD point clouds,
    stem attachment points, and result.xlsx for stem assignments.

    Args:
        scan_dir: Path to parameterextraction/wheatN/ directory.
        output_path: If provided, save JSON artifact here.
        width_cap: Maximum half-width cap in cm.

    Returns:
        Dict with scan metadata and per-leaf profiles.
    """
    import open3d as o3d

    scan_dir = Path(scan_dir)  # type: ignore[assignment]
    leaf_dir = scan_dir / 'leaf'
    scan_name = scan_dir.name

    # Load stem attachment points
    attach_pcd = o3d.io.read_point_cloud(str(scan_dir / 'leaf_stem_close.pcd'))
    attach_pts = np.asarray(attach_pcd.points)
    n_attach = attach_pts.shape[0]

    # Load result.xlsx for stem assignments
    stem_ids = {}
    try:
        import pandas as pd
        df = pd.read_excel(str(scan_dir / 'result.xlsx'))
        leaf_of_stem = df['leaf_of_stem'].dropna().values
        for i, sid in enumerate(leaf_of_stem):
            stem_ids[i] = int(sid)
    except Exception:
        pass

    # Discover leaf count from PLY files
    leaf_ids = sorted([
        int(p.stem.split('_')[-1])
        for p in leaf_dir.glob('leaf_in_*.ply')
    ])

    leaves = []
    for lid in leaf_ids:
        ply_path = leaf_dir / f'leaf_in_{lid}.ply'
        pcd_path = leaf_dir / f'leaf{lid}.pcd'

        if not ply_path.exists() or not pcd_path.exists():
            print(f"  Skipping leaf {lid}: missing files", file=sys.stderr)
            continue

        # Parse skeleton
        skeleton = parse_ply_skeleton(str(ply_path))

        # Get attachment point (1:1 mapping if available)
        if lid < n_attach:
            attach_pt = attach_pts[lid]
        else:
            # Fallback: nearest attachment point to skeleton midpoint
            mid = skeleton[len(skeleton) // 2]
            dists = np.linalg.norm(attach_pts - mid, axis=1)
            attach_pt = attach_pts[dists.argmin()]

        # Orient and bridge
        skeleton = orient_skeleton_to_base(skeleton, attach_pt)
        skeleton = bridge_skeleton(skeleton, attach_pt)

        # Load leaf PCD
        leaf_pcd = o3d.io.read_point_cloud(str(pcd_path))
        leaf_pts = np.asarray(leaf_pcd.points)

        # Extract width profile
        half_widths = extract_width_profile(
            skeleton, leaf_pts, width_cap=width_cap,
        )

        # Arc length
        arc_length = float(np.sum(np.linalg.norm(np.diff(skeleton, axis=0), axis=1)))

        leaves.append({
            'leaf_id': lid,
            'skeleton': skeleton.tolist(),
            'half_widths_cm': half_widths.tolist(),
            'arc_length_cm': round(arc_length, 3),
            'n_pcd_points': int(leaf_pts.shape[0]),
            'n_skeleton_nodes': int(skeleton.shape[0]),
            'stem_id': stem_ids.get(lid, -1),
            'stem_attachment': attach_pt.tolist(),
        })

        max_w = half_widths.max()
        print(f"  Leaf {lid:2d}: {skeleton.shape[0]} nodes, "
              f"arc={arc_length:.1f} cm, max_hw={max_w:.2f} cm, "
              f"{leaf_pts.shape[0]} pts", file=sys.stderr)

    result = {
        'scan': scan_name,
        'n_leaves': len(leaves),
        'leaves': leaves,
    }

    if output_path:
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\nSaved: {output_path} ({len(leaves)} leaves)", file=sys.stderr)

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Extract width profiles from wheat scan data")
    parser.add_argument("scan_dir",
                        help="Path to parameterextraction/wheatN/ directory")
    parser.add_argument("-o", "--output", default=None,
                        help="Output JSON path (default: <scan_name>_profiles.json)")
    parser.add_argument("--width-cap", type=float, default=2.0,
                        help="Maximum half-width cap in cm (default: 2.0)")
    args = parser.parse_args()

    scan_dir = Path(args.scan_dir)
    output = args.output or f"{scan_dir.name}_profiles.json"

    extract_all_profiles(str(scan_dir), output, width_cap=args.width_cap)


if __name__ == '__main__':
    main()
