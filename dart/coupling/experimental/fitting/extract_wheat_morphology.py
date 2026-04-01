"""Extract per-leaf morphology from segmented wheat point cloud scans.

Reads the parameterextraction/*.txt files (8-col: X Y Z R G B sem inst)
and computes per-leaf length, width, insertion height, angle.

Outputs wheat_stats.json compatible with the fitting pipeline.

Usage:
    python -m dart.coupling.experimental.fitting.extract_wheat_morphology \
        /path/to/parameterextraction/ --output dart/coupling/data/wheat_stats.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def load_segmented_wheat(path):
    """Load 8-column wheat scan: X Y Z R G B semantic instance."""
    raw = np.loadtxt(path, dtype=np.float64)
    header = raw[0].astype(int)
    data = raw[1:]
    xyz = data[:, :3]
    sem = data[:, 6].astype(int)
    inst = data[:, 7].astype(int)
    return xyz, sem, inst, header


def extract_leaf_stats(xyz, sem, inst):
    """Extract per-leaf morphology from segmented point cloud.

    Returns list of dicts with: length, width, base_z, tip_z, angle_rad, n_points.
    """
    leaf_mask = sem == 100  # 100 = leaf in this dataset
    leaf_ids = np.unique(inst[leaf_mask])

    # Get stem centroid for angle computation
    stem_mask = sem == 200
    if stem_mask.any():
        stem_center_xy = xyz[stem_mask, :2].mean(axis=0)
    else:
        stem_center_xy = xyz[leaf_mask, :2].mean(axis=0)

    leaves = []
    for lid in sorted(leaf_ids):
        mask = (sem == 100) & (inst == lid)
        pts = xyz[mask]
        n = pts.shape[0]
        if n < 10:
            continue

        # PCA for length/width
        centered = pts - pts.mean(axis=0)
        _, s, vh = np.linalg.svd(centered, full_matrices=False)
        # Length along principal axis
        proj = centered @ vh[0]
        length = proj.max() - proj.min()
        # Width along second axis
        proj2 = centered @ vh[1]
        width = proj2.max() - proj2.min()

        # Base = point closest to stem center at lowest Z
        # Sort by Z, take bottom 10% as base candidates
        z_sorted = np.argsort(pts[:, 2])
        n_base = max(3, n // 10)
        base_candidates = pts[z_sorted[-n_base:]]  # highest Z = closest to stem
        # Actually: base is where leaf attaches to stem (highest Z for drooping leaves,
        # or closest to stem center). Use closest-to-stem-center approach.
        dists_to_stem = np.sqrt(((pts[:, :2] - stem_center_xy) ** 2).sum(axis=1))
        base_idx = dists_to_stem.argmin()
        base_pt = pts[base_idx]
        # Tip = farthest from base
        dists_to_base = np.sqrt(((pts - base_pt) ** 2).sum(axis=1))
        tip_idx = dists_to_base.argmax()
        tip_pt = pts[tip_idx]

        # Insertion angle: angle between leaf direction and vertical
        leaf_vec = tip_pt - base_pt
        leaf_vec_norm = leaf_vec / (np.linalg.norm(leaf_vec) + 1e-8)
        vertical = np.array([0, 0, 1])
        cos_angle = np.clip(np.dot(leaf_vec_norm, vertical), -1, 1)
        angle_rad = np.arccos(abs(cos_angle))  # angle from vertical

        leaves.append({
            'inst_id': int(lid),
            'n_points': n,
            'length': float(length),
            'width': float(width),
            'base_z': float(base_pt[2]),
            'tip_z': float(tip_pt[2]),
            'angle_rad': float(angle_rad),
            'base_xy': base_pt[:2].tolist(),
        })

    return leaves


def count_tillers(xyz, sem, inst):
    """Estimate tiller count from stem segmentation."""
    stem_mask = sem == 200
    if not stem_mask.any():
        return 1
    stem_pts = xyz[stem_mask]
    # Slice at 1/3 height and cluster XY
    z_range = stem_pts[:, 2].max() - stem_pts[:, 2].min()
    z_low = stem_pts[:, 2].min() + z_range * 0.2
    z_high = stem_pts[:, 2].min() + z_range * 0.4
    slice_mask = (stem_pts[:, 2] > z_low) & (stem_pts[:, 2] < z_high)
    if slice_mask.sum() < 10:
        return 1
    slice_xy = stem_pts[slice_mask, :2]
    # DBSCAN clustering
    from scipy.cluster.hierarchy import fcluster, linkage
    Z = linkage(slice_xy, method='single')
    labels = fcluster(Z, t=1.5, criterion='distance')  # 1.5 cm threshold
    return int(labels.max())


def assign_positions(leaves, n_positions=8):
    """Assign leaves to positions by insertion height (base_z).

    Groups leaves into n_positions bins by base_z.
    Returns list of (position, leaf_stats) tuples.
    """
    if not leaves:
        return []
    # Sort by base_z (highest = lowest on plant since Z may be inverted)
    sorted_leaves = sorted(leaves, key=lambda l: l['base_z'])
    # Bin into positions
    n = len(sorted_leaves)
    bin_size = max(1, n // n_positions)
    positioned = []
    for i, leaf in enumerate(sorted_leaves):
        pos = min(i // bin_size, n_positions - 1)
        positioned.append((pos, leaf))
    return positioned


def main():
    parser = argparse.ArgumentParser(description="Extract wheat morphology from segmented scans")
    parser.add_argument("scan_dir", help="Directory with wheat1.txt ... wheat8.txt")
    parser.add_argument("--output", default="wheat_stats.json", help="Output JSON path")
    parser.add_argument("--n-positions", type=int, default=8, help="Number of leaf positions")
    args = parser.parse_args()

    scan_dir = Path(args.scan_dir)
    all_positioned = {i: [] for i in range(args.n_positions)}
    tiller_counts = []

    for scan_idx in range(1, 9):
        path = scan_dir / f"wheat{scan_idx}.txt"
        if not path.exists():
            print(f"  Skipping {path} (not found)", file=sys.stderr)
            continue

        print(f"Processing {path.name}...", file=sys.stderr)
        xyz, sem, inst, header = load_segmented_wheat(path)

        # Normalize Z: ground at 0
        xyz[:, 2] -= xyz[:, 2].min()

        n_tillers = count_tillers(xyz, sem, inst)
        tiller_counts.append(n_tillers)

        leaves = extract_leaf_stats(xyz, sem, inst)
        print(f"  {len(leaves)} leaves, ~{n_tillers} tillers", file=sys.stderr)

        positioned = assign_positions(leaves, args.n_positions)
        for pos, leaf in positioned:
            all_positioned[pos].append(leaf)

    # Compute per-position statistics
    per_position = []
    for pos in range(args.n_positions):
        leaves = all_positioned[pos]
        if not leaves:
            # Fallback
            per_position.append({
                "position": pos,
                "lmax": 15.0,
                "lmax_std": 5.0,
                "Width_blade": 1.0,
                "Width_petiole": 0.3,
                "areaMax": 22.0,
                "theta": 0.3,
                "r": 2.0,
                "tropismS": 0.015,
                "tropismAge": 8.0,
                "n_samples": 0,
                "median_length_cm": 15.0,
                "median_max_width_cm": 1.0,
            })
            continue

        lengths = [l['length'] for l in leaves]
        widths = [l['width'] for l in leaves]
        angles = [l['angle_rad'] for l in leaves]

        med_length = float(np.median(lengths))
        med_width = float(np.median(widths))
        med_theta = float(np.median(angles))

        # CPlantBox Width_blade is half-width (one side)
        width_blade = med_width / 2.0
        area_max = med_length * med_width * 0.73

        per_position.append({
            "position": pos,
            "lmax": float(np.percentile(lengths, 75)),  # generous to allow variation
            "lmax_std": float(np.std(lengths)),
            "Width_blade": width_blade,
            "Width_petiole": width_blade * 0.35,
            "areaMax": area_max,
            "theta": med_theta,
            "r": 2.0,
            "tropismS": 0.015,
            "tropismAge": 5.0 + pos,
            "n_samples": len(leaves),
            "median_length_cm": med_length,
            "median_max_width_cm": med_width,
        })

    median_tillers = int(np.median(tiller_counts)) if tiller_counts else 3
    # Estimate stem params from tallest leaf base_z
    all_base_z = [l['base_z'] for leaves in all_positioned.values() for l in leaves]
    stem_height = float(np.percentile(all_base_z, 95)) if all_base_z else 63.0

    stats = {
        "source": "wheat_parameterextraction_scans",
        "n_plants": len(tiller_counts),
        "n_positions": args.n_positions,
        "median_leaf_count": sum(len(v) for v in all_positioned.values()) // max(len(tiller_counts), 1),
        "median_tillers": median_tillers,
        "stem": {
            "lmax": stem_height,
            "ln": stem_height / args.n_positions,
            "lb": 2.0,
            "n_leaves": args.n_positions,
        },
        "per_position": per_position,
    }

    with open(args.output, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"\nSaved: {args.output}", file=sys.stderr)
    print(f"Tillers per scan: {tiller_counts} (median={median_tillers})", file=sys.stderr)
    print(f"Stem height: {stem_height:.1f} cm", file=sys.stderr)
    for p in per_position:
        print(f"  L{p['position']}: lmax={p['lmax']:.1f} width={p['median_max_width_cm']:.1f} "
              f"theta={p['theta']:.2f} n={p['n_samples']}", file=sys.stderr)


if __name__ == '__main__':
    main()
