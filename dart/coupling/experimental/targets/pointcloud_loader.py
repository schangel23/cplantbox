"""Load point clouds from various formats (.stl, .ply, .xyz, .pcd)."""

from pathlib import Path

import numpy as np
import trimesh


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
    path: str | Path, n_points: int = 10000
) -> tuple[np.ndarray, np.ndarray | None]:
    """Auto-detect format and load point cloud. Subsample if needed.

    Supported: .stl, .ply, .xyz, .pcd
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
    elif suffix == ".pcd":
        cloud = trimesh.load(path)
        points = np.asarray(cloud.vertices, dtype=np.float32)
        colors = None
    else:
        raise ValueError(f"Unsupported format: {suffix}")

    # Auto-detect units: if extents < 10 in all dims, assume meters → convert to cm
    extents = points.max(axis=0) - points.min(axis=0)
    if extents.max() < 10.0:
        points *= 100.0

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
