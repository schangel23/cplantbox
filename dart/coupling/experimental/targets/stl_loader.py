"""Load STL meshes as point clouds for target geometry."""

from pathlib import Path

import numpy as np
import trimesh


def load_stl_as_pointcloud(path: str | Path, n_points: int = 10000) -> np.ndarray:
    """Load an STL file and uniformly sample points from its surface.

    Returns (n_points, 3) array in cm.
    """
    mesh = trimesh.load(path, force="mesh")
    points = mesh.sample(n_points)
    points = np.asarray(points, dtype=np.float32)

    # Auto-detect units: if extents < 10 in all dims, assume meters → convert to cm
    extents = points.max(axis=0) - points.min(axis=0)
    if extents.max() < 10.0:
        points *= 100.0

    # Center XY at origin, Z starts at ground (min Z)
    points[:, 0] -= points[:, 0].mean()
    points[:, 1] -= points[:, 1].mean()
    points[:, 2] -= points[:, 2].min()

    return points


def load_stl_batch(
    stl_dir: str | Path, n_points: int = 10000, max_files: int | None = None
) -> list[np.ndarray]:
    """Load multiple STL files from a directory.

    Returns list of (n_points, 3) arrays.
    """
    stl_dir = Path(stl_dir)
    paths = sorted(stl_dir.glob("*.stl"))
    if max_files is not None:
        paths = paths[:max_files]
    return [load_stl_as_pointcloud(p, n_points) for p in paths]
