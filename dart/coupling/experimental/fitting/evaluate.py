"""Evaluation and comparison utilities for fitted plants."""

import json
from pathlib import Path

import numpy as np


def compute_metrics(gen_points: np.ndarray, target_points: np.ndarray) -> dict:
    """Compute geometric comparison metrics between generated and target point clouds."""
    from scipy.spatial import cKDTree

    tree_gen = cKDTree(gen_points)
    tree_tgt = cKDTree(target_points)

    d_gen_to_tgt, _ = tree_tgt.query(gen_points)
    d_tgt_to_gen, _ = tree_gen.query(target_points)

    return {
        'chamfer': float(np.mean(d_gen_to_tgt) + np.mean(d_tgt_to_gen)) / 2.0,
        'hausdorff': float(max(np.max(d_gen_to_tgt), np.max(d_tgt_to_gen))),
        'mean_gen_to_tgt': float(np.mean(d_gen_to_tgt)),
        'mean_tgt_to_gen': float(np.mean(d_tgt_to_gen)),
        'median_gen_to_tgt': float(np.median(d_gen_to_tgt)),
        'p95_gen_to_tgt': float(np.percentile(d_gen_to_tgt, 95)),
    }


def export_comparison_obj(
    gen_points: np.ndarray,
    target_points: np.ndarray,
    output_path: str,
):
    """Export both point clouds as OBJ for visual comparison in Blender.

    Generated points are written as vertices; target points offset by 100cm in X
    for side-by-side viewing.
    """
    with open(output_path, 'w') as f:
        f.write("# Generated plant\n")
        for p in gen_points:
            f.write(f"v {p[0]:.4f} {p[1]:.4f} {p[2]:.4f}\n")
        f.write(f"\n# Target plant (offset +100 cm in X)\n")
        for p in target_points:
            f.write(f"v {p[0] + 100:.4f} {p[1]:.4f} {p[2]:.4f}\n")


def summarize_results(results_dir: str) -> dict:
    """Aggregate per-plant results into a summary."""
    results_path = Path(results_dir)
    files = sorted(results_path.glob('*_result.json'))

    chamfer_values = []
    for f in files:
        with open(f) as fp:
            r = json.load(fp)
        chamfer_values.append(r.get('final_loss', r.get('best_loss', float('inf'))))

    if not chamfer_values:
        return {'n_plants': 0}

    arr = np.array(chamfer_values)
    return {
        'n_plants': len(arr),
        'chamfer_mean': float(arr.mean()),
        'chamfer_median': float(np.median(arr)),
        'chamfer_std': float(arr.std()),
        'chamfer_min': float(arr.min()),
        'chamfer_max': float(arr.max()),
        'chamfer_p25': float(np.percentile(arr, 25)),
        'chamfer_p75': float(np.percentile(arr, 75)),
    }
