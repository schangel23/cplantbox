"""Load point clouds from various formats (.stl, .ply, .xyz, .pcd)."""

from pathlib import Path

import numpy as np
import trimesh


def align_rotation_z(target_points: np.ndarray, reference_points: np.ndarray,
                     n_angles: int = 36) -> tuple[np.ndarray, float]:
    """Brute-force Z-axis rotation alignment via Chamfer distance.

    Tries n_angles evenly spaced rotations around Z, returns the rotation
    that minimizes Chamfer distance to reference_points.

    Args:
        target_points: (N, 3) point cloud to rotate
        reference_points: (M, 3) point cloud to align against
        n_angles: number of angles to try (default 36 = every 10 degrees)

    Returns:
        (rotated_points, best_angle_degrees)
    """
    from scipy.spatial import cKDTree

    tree_ref = cKDTree(reference_points)
    best_angle = 0
    best_chamfer = float('inf')
    best_rotated = target_points

    for deg in np.linspace(0, 360, n_angles, endpoint=False):
        rad = np.radians(deg)
        c, s = np.cos(rad), np.sin(rad)
        rotated = target_points.copy()
        rotated[:, 0] = target_points[:, 0] * c - target_points[:, 1] * s
        rotated[:, 1] = target_points[:, 0] * s + target_points[:, 1] * c

        d1, _ = tree_ref.query(rotated)
        chamfer = float(np.mean(d1))

        if chamfer < best_chamfer:
            best_chamfer = chamfer
            best_angle = deg
            best_rotated = rotated

    return best_rotated, best_angle


def load_ply(path: str | Path) -> tuple[np.ndarray, np.ndarray | None]:
    """Load a PLY file, returning (points (N,3), colors (N,3) or None).

    Colors are returned as float32 in [0, 1].
    """
    cloud = trimesh.load(path)

    if isinstance(cloud, trimesh.Trimesh):
        points = np.asarray(cloud.vertices, dtype=np.float32)
        if cloud.visual and hasattr(cloud.visual, "vertex_colors"):
            vc = np.asarray(cloud.visual.vertex_colors, dtype=np.float32)
            colors = vc[:, :3] / 255.0 if vc.max() > 1.0 else vc[:, :3]
        else:
            colors = None
    elif isinstance(cloud, trimesh.PointCloud):
        points = np.asarray(cloud.vertices, dtype=np.float32)
        if cloud.colors is not None and len(cloud.colors) > 0:
            vc = np.asarray(cloud.colors, dtype=np.float32)
            colors = vc[:, :3] / 255.0 if vc.max() > 1.0 else vc[:, :3]
        else:
            colors = None
    else:
        raise ValueError(f"Unexpected trimesh type: {type(cloud)}")

    return points, colors


def load_pointcloud(
    path: str | Path, n_points: int = 10000, units: str = 'auto',
) -> tuple[np.ndarray, np.ndarray | None]:
    """Auto-detect format and load point cloud. Subsample if needed.

    Supported: .stl, .ply, .xyz, .txt, .pcd
    Args:
        path: file path
        n_points: subsample target
        units: 'auto', 'mm', 'cm', or 'm'. Auto uses heuristics.
    Returns (points (n_points, 3), colors (n_points, 3) or None).
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".stl":
        from .stl_loader import load_stl_as_pointcloud

        return load_stl_as_pointcloud(path, n_points), None

    if suffix == ".ply":
        points, colors = load_ply(path)
    elif suffix == ".xyz":
        data = np.loadtxt(path, dtype=np.float32)
        if data.shape[1] >= 6:
            points, colors = data[:, :3], data[:, 3:6] / 255.0
        else:
            points, colors = data[:, :3], None
    elif suffix == ".txt":
        data = np.loadtxt(path, dtype=np.float32)
        points = data[:, :3]
        colors = None
        if data.shape[1] >= 6 and data[:, 3:6].max() > 1.0:
            # XYZ RGB format (0-255 range) — filter plant by brightness
            rgb = data[:, 3:6]
            colors = rgb / 255.0
            brightness = rgb.mean(axis=1)
            plant_mask = brightness < 170
            if 0 < plant_mask.sum() < len(points):
                points = points[plant_mask]
                colors = colors[plant_mask]
        elif data.shape[1] >= 5:
            # Pheno4D format: column 4 = organ label (0=background)
            organ_label = data[:, 4]
            plant_mask = organ_label > 0
            if 0 < plant_mask.sum() < len(points):
                points = points[plant_mask]
    elif suffix == ".pcd":
        cloud = trimesh.load(path)
        points = np.asarray(cloud.vertices, dtype=np.float32)
        colors = None
    else:
        raise ValueError(f"Unsupported format: {suffix}")

    # Unit conversion
    if units == 'mm':
        points /= 10.0
    elif units == 'm':
        points *= 100.0
    elif units == 'auto':
        extents = points.max(axis=0) - points.min(axis=0)
        z_extent = extents[2] if len(extents) > 2 else extents.max()
        if z_extent < 5.0:
            points *= 100.0  # meters → cm
        elif z_extent > 500.0:
            points /= 10.0   # mm → cm

    # Center XY at origin, Z starts at ground (min Z)
    points[:, 0] -= points[:, 0].mean()
    points[:, 1] -= points[:, 1].mean()
    points[:, 2] -= points[:, 2].min()

    # Subsample if needed
    n = len(points)
    if n > n_points:
        idx = np.random.default_rng(42).choice(n, n_points, replace=False)
        points = points[idx]
        if colors is not None:
            colors = colors[idx]
    elif n < n_points:
        # Upsample with replacement
        idx = np.random.default_rng(42).choice(n, n_points, replace=True)
        points = points[idx]
        if colors is not None:
            colors = colors[idx]

    return points, colors


def extract_leaf_labels_from_rgb(colors: np.ndarray) -> np.ndarray:
    """Cluster unique RGB colors to get integer leaf labels per point.

    For MaizeField3D segmented PLYs where each leaf has a unique RGB color.
    Returns (N,) int array of leaf labels.
    """
    # Quantize colors to avoid floating-point noise
    quantized = np.round(colors * 255).astype(np.int32)
    # Pack RGB into a single int for fast unique-finding
    packed = quantized[:, 0] * 65536 + quantized[:, 1] * 256 + quantized[:, 2]
    unique_colors, labels = np.unique(packed, return_inverse=True)
    return labels.astype(np.int32)
