"""Infer stem curvature from leaf insertion positions.

Two approaches:
1. MaizeField3D NURBS: extract full 3D leaf base positions from .dat files
2. Point cloud fallback: estimate stem from height-sliced centroids

The inferred stem curve is converted to a CPlantBox stem tropismS value.
"""

import numpy as np
from pathlib import Path
from scipy.interpolate import CubicSpline


def extract_leaf_bases_from_nurbs(dat_dir: str, n_eval: int = 20) -> dict:
    """Parse MaizeField3D NURBS .dat files, extract 3D leaf base positions.

    Args:
        dat_dir: directory with .dat NURBS files (one per plant)
        n_eval: evaluation points per parametric dimension

    Returns:
        dict mapping plant_id (str) -> (n_leaves, 3) array of base XYZ in cm
    """
    from scipy.interpolate import BSpline

    dat_path = Path(dat_dir)
    files = sorted(dat_path.glob("*.dat"))
    if not files:
        raise FileNotFoundError(f"No .dat files in {dat_dir}")

    all_bases = {}

    for f in files:
        plant_id = f.stem
        bases = []

        with open(f) as fp:
            content = fp.read()

        # Each leaf section starts with "Leaf<N>" or similar header
        # Parse NURBS control points (3x6 grid per leaf in meters)
        sections = content.split("Leaf")
        for sec in sections[1:]:  # skip header
            lines = [l.strip() for l in sec.strip().split('\n') if l.strip()]
            # Find control point lines (contain 4 floats: x y z w)
            cp_lines = []
            for line in lines[1:]:  # skip leaf header line
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        vals = [float(x) for x in parts[:3]]
                        cp_lines.append(vals)
                    except ValueError:
                        continue

            if len(cp_lines) >= 3:
                # First control point row = leaf base
                base_xyz = np.array(cp_lines[0]) * 100.0  # m -> cm
                bases.append(base_xyz)

        if bases:
            all_bases[plant_id] = np.array(bases)

    return all_bases


def infer_stem_curve(
    leaf_bases: dict,
    n_positions: int = 11,
) -> tuple[np.ndarray, float]:
    """From per-plant leaf base positions, fit a median stem centerline.

    Args:
        leaf_bases: dict from extract_leaf_bases_from_nurbs()
        n_positions: expected number of leaf positions

    Returns:
        (stem_curve, curvature)
        stem_curve: (n_positions, 3) median stem centerline in cm
        curvature: mean curvature in 1/cm (suitable for tropismS)
    """
    # Collect per-position bases across all plants
    per_pos = [[] for _ in range(n_positions)]

    for plant_id, bases in leaf_bases.items():
        n = min(len(bases), n_positions)
        for i in range(n):
            per_pos[i].append(bases[i])

    # Compute median XYZ per position
    median_bases = []
    for i in range(n_positions):
        if per_pos[i]:
            median_bases.append(np.median(per_pos[i], axis=0))
        else:
            # Interpolate missing positions
            median_bases.append(np.zeros(3))

    stem_curve = np.array(median_bases)

    # Center XY at base
    stem_curve[:, 0] -= stem_curve[0, 0]
    stem_curve[:, 1] -= stem_curve[0, 1]
    stem_curve[:, 2] -= stem_curve[0, 2]

    # Compute mean curvature
    if len(stem_curve) >= 3:
        diffs = np.diff(stem_curve, axis=0)
        seg_lens = np.linalg.norm(diffs, axis=1)
        tangents = diffs / seg_lens[:, None].clip(min=1e-8)
        dt = np.diff(tangents, axis=0)
        ds = (seg_lens[:-1] + seg_lens[1:]) / 2.0
        curvatures = np.linalg.norm(dt, axis=1) / ds.clip(min=1e-8)
        mean_curvature = float(np.mean(curvatures))
    else:
        mean_curvature = 0.0

    return stem_curve, mean_curvature


def infer_stem_from_pointcloud(
    points: np.ndarray,
    n_slices: int = 20,
) -> tuple[np.ndarray, float]:
    """Estimate stem curve from an unsegmented point cloud.

    Slices the point cloud by height, computes XY centroid per slice.
    The centroid trajectory traces the stem (including lean).

    Args:
        points: (N, 3) point cloud in cm, centered
        n_slices: number of height slices

    Returns:
        (stem_curve, curvature)
        stem_curve: (n_slices, 3) estimated centerline
        curvature: mean curvature in 1/cm
    """
    z_min, z_max = points[:, 2].min(), points[:, 2].max()
    z_edges = np.linspace(z_min, z_max, n_slices + 1)

    centroids = []
    for i in range(n_slices):
        mask = (points[:, 2] >= z_edges[i]) & (points[:, 2] < z_edges[i + 1])
        if mask.sum() > 5:
            centroid = points[mask].mean(axis=0)
        else:
            centroid = np.array([0, 0, (z_edges[i] + z_edges[i + 1]) / 2])
        centroids.append(centroid)

    stem_curve = np.array(centroids)

    # Center XY at base
    stem_curve[:, 0] -= stem_curve[0, 0]
    stem_curve[:, 1] -= stem_curve[0, 1]

    # Compute curvature
    if len(stem_curve) >= 3:
        diffs = np.diff(stem_curve, axis=0)
        seg_lens = np.linalg.norm(diffs, axis=1)
        tangents = diffs / seg_lens[:, None].clip(min=1e-8)
        dt = np.diff(tangents, axis=0)
        ds = (seg_lens[:-1] + seg_lens[1:]) / 2.0
        curvatures = np.linalg.norm(dt, axis=1) / ds.clip(min=1e-8)
        mean_curvature = float(np.mean(curvatures))
    else:
        mean_curvature = 0.0

    return stem_curve, mean_curvature


def stem_curve_to_tropism(curvature: float) -> float:
    """Convert a curvature value to CPlantBox stem tropismS.

    CPlantBox tropismS is roughly the expected angular change per cm.
    Curvature (1/cm) maps directly.

    Args:
        curvature: mean curvature in 1/cm

    Returns:
        tropismS value for stem organ
    """
    # Clamp to reasonable range for maize
    return float(np.clip(curvature, 0.0, 0.02))
