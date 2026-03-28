"""Chamfer distance loss for point cloud comparison."""

import torch


def chamfer_distance(pc1: torch.Tensor, pc2: torch.Tensor) -> torch.Tensor:
    """GPU Chamfer distance between two point clouds.

    Args:
        pc1: (P1, 3) float tensor
        pc2: (P2, 3) float tensor

    Returns:
        Scalar mean Chamfer distance.
    """
    # (P1, P2) pairwise squared distances
    dists = torch.cdist(pc1.unsqueeze(0), pc2.unsqueeze(0)).squeeze(0)  # (P1, P2)
    d1 = dists.min(dim=1).values  # (P1,) — nearest in pc2 for each pc1 point
    d2 = dists.min(dim=0).values  # (P2,) — nearest in pc1 for each pc2 point
    return (d1.mean() + d2.mean()) / 2.0


def chamfer_distance_batch(pc1: torch.Tensor, pc2: torch.Tensor) -> torch.Tensor:
    """Batched Chamfer distance.

    Args:
        pc1: (B, P1, 3) float tensor
        pc2: (B, P2, 3) float tensor

    Returns:
        (B,) tensor of per-sample Chamfer distances.
    """
    dists = torch.cdist(pc1, pc2)  # (B, P1, P2)
    d1 = dists.min(dim=2).values  # (B, P1)
    d2 = dists.min(dim=1).values  # (B, P2)
    return (d1.mean(dim=1) + d2.mean(dim=1)) / 2.0
