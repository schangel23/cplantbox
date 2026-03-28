"""Per-leaf Chamfer distance with height-based leaf matching."""

import torch

from .chamfer import chamfer_distance


def per_leaf_chamfer(
    gen_points: torch.Tensor,
    gen_labels: torch.Tensor,
    target_points: torch.Tensor,
    target_labels: torch.Tensor,
) -> torch.Tensor:
    """Compute per-leaf Chamfer distance with automatic leaf matching.

    Leaves are matched between generated and target by median Z height.
    Returns weighted sum of per-leaf Chamfer distances, weighted by
    target leaf point count.

    Args:
        gen_points: (N1, 3) generated point cloud
        gen_labels: (N1,) int leaf labels
        target_points: (N2, 3) target point cloud
        target_labels: (N2,) int leaf labels
    """
    gen_ids = gen_labels.unique()
    target_ids = target_labels.unique()

    # Compute median Z for each leaf in both sets
    gen_median_z = torch.stack(
        [gen_points[gen_labels == i, 2].median() for i in gen_ids]
    )
    target_median_z = torch.stack(
        [target_points[target_labels == i, 2].median() for i in target_ids]
    )

    # Match by nearest median Z (greedy)
    # (n_gen, n_target) distance matrix
    z_dists = (gen_median_z.unsqueeze(1) - target_median_z.unsqueeze(0)).abs()

    total_loss = gen_points.new_tensor(0.0)
    total_weight = 0.0
    matched_target = set()

    # Sort gen leaves by Z for stable matching
    gen_order = gen_median_z.argsort()

    for gi in gen_order:
        gen_id = gen_ids[gi]
        # Find nearest unmatched target leaf
        dists_row = z_dists[gi].clone()
        for mt in matched_target:
            dists_row[mt] = float("inf")
        if (dists_row == float("inf")).all():
            continue
        ti = dists_row.argmin().item()
        matched_target.add(ti)
        target_id = target_ids[ti]

        gen_leaf = gen_points[gen_labels == gen_id]
        target_leaf = target_points[target_labels == target_id]

        weight = float(len(target_leaf))
        total_loss = total_loss + chamfer_distance(gen_leaf, target_leaf) * weight
        total_weight += weight

    if total_weight > 0:
        total_loss = total_loss / total_weight

    return total_loss
