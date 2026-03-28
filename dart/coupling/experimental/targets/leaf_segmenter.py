"""Leaf segmentation for target point clouds."""

import numpy as np


def segment_by_height(points: np.ndarray, n_leaves: int = 11) -> np.ndarray:
    """Simple height-based leaf segmentation for unsegmented point clouds.

    Clusters points by Z-coordinate into n_leaves groups using equal-frequency
    binning. Returns (N,) int labels.
    """
    z = points[:, 2]
    # Equal-frequency bins so each leaf gets roughly the same number of points
    percentiles = np.linspace(0, 100, n_leaves + 1)
    bin_edges = np.percentile(z, percentiles)
    # digitize returns 1-based, clip to valid range
    labels = np.digitize(z, bin_edges[1:-1], right=False)
    return labels.astype(np.int32)


def segment_leaves(
    points: np.ndarray, colors: np.ndarray | None = None, n_leaves: int = 11
) -> np.ndarray:
    """Segment point cloud into leaf labels.

    If colors are available, uses RGB clustering (for MaizeField3D segmented PLYs).
    Otherwise falls back to height-based segmentation.
    Returns (N,) int labels.
    """
    if colors is not None:
        from .pointcloud_loader import extract_leaf_labels_from_rgb

        labels = extract_leaf_labels_from_rgb(colors)
        n_unique = len(np.unique(labels))
        # If too many segments (fine-grained RGB), fall back to height-based
        if n_unique > n_leaves * 3:
            return segment_by_height(points, n_leaves)
        return labels

    return segment_by_height(points, n_leaves)
