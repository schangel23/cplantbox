"""End-to-end gradient-based fitting of CPlantBox params to target point clouds."""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ..losses.chamfer import chamfer_distance
from ..losses.per_leaf import per_leaf_chamfer
from ..losses.regularization import prior_loss
from ..diff_lofter.lofter import loft_leaf
from ..diff_lofter.frames import compute_tangents, compute_binormal_field
from ..diff_lofter.deformations import compute_deformations
from ..surrogate.model import SkeletonSurrogate
from ..surrogate.analytic_baseline import analytic_skeleton


# Indices into the flat param vector for each leaf position
N_STRUCTURAL = 7  # lmax, Width_blade, theta, tropismS, tropismAge, r, areaMax
N_DEFORM = 5      # wave_normal_amp, twist_max, curl_amp, edge_ruffle_amp, fold_amp
N_PER_LEAF = N_STRUCTURAL + N_DEFORM  # 12
N_POSITIONS = 11
N_STEM = 1  # ln
N_PARAMS = N_PER_LEAF * N_POSITIONS + N_STEM  # 133

STRUCTURAL_NAMES = ['lmax', 'Width_blade', 'theta', 'tropismS', 'tropismAge', 'r', 'areaMax']
DEFORM_NAMES = ['wave_normal_amp', 'twist_max', 'curl_amp', 'edge_ruffle_amp', 'fold_amp']


def _load_prior(stats_path: str, device: str = 'cuda') -> tuple[torch.Tensor, torch.Tensor]:
    """Load MaizeField3D medians as prior means and stds."""
    with open(stats_path) as f:
        stats = json.load(f)

    means = []
    stds = []
    for pos in range(N_POSITIONS):
        s = stats[str(pos)] if str(pos) in stats else stats[pos]
        for name in STRUCTURAL_NAMES:
            val = float(s.get(name, 1.0))
            means.append(val)
            stds.append(max(abs(val) * 0.3, 0.1))  # 30% relative std
        for name in DEFORM_NAMES:
            means.append(0.0)  # deformations centered at 0
            stds.append(1.0)

    # Stem ln
    means.append(14.5)
    stds.append(3.0)

    return (
        torch.tensor(means, dtype=torch.float32, device=device),
        torch.tensor(stds, dtype=torch.float32, device=device),
    )


def _params_to_leaves(params: torch.Tensor) -> list[dict]:
    """Unpack flat param vector into per-leaf param dicts."""
    leaves = []
    for pos in range(N_POSITIONS):
        offset = pos * N_PER_LEAF
        leaf = {
            'lmax': params[offset + 0],
            'Width_blade': params[offset + 1],
            'theta': params[offset + 2],
            'tropismS': params[offset + 3],
            'tropismAge': params[offset + 4],
            'r': params[offset + 5],
            'areaMax': params[offset + 6],
            'wave_normal_amp': params[offset + 7],
            'twist_max': params[offset + 8],
            'curl_amp': params[offset + 9],
            'edge_ruffle_amp': params[offset + 10],
            'fold_amp': params[offset + 11],
            'position': pos,
        }
        leaves.append(leaf)
    return leaves


def _generate_plant_pointcloud(
    params: torch.Tensor,
    surrogate: SkeletonSurrogate,
    n_cross: int = 7,
    device: str = 'cuda',
) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward pass: params → surrogate → diff lofter → point cloud.

    Returns:
        points: (V, 3) vertex positions
        labels: (V,) leaf index per vertex
    """
    leaves = _params_to_leaves(params)
    all_points = []
    all_labels = []

    for leaf in leaves:
        pos = leaf['position']
        # Build surrogate input: 12 leaf params + position index
        leaf_params = torch.stack([
            leaf['lmax'], leaf['Width_blade'], leaf['theta'],
            leaf['tropismS'], leaf['tropismAge'], leaf['r'], leaf['areaMax'],
            leaf['wave_normal_amp'], leaf['twist_max'], leaf['curl_amp'],
            leaf['edge_ruffle_amp'], leaf['fold_amp'],
        ])
        pos_tensor = torch.tensor([float(pos)], device=device)
        surrogate_input = torch.cat([leaf_params, pos_tensor]).unsqueeze(0)  # (1, 13)

        # Predict skeleton via surrogate
        baseline = analytic_skeleton(
            leaf['lmax'].unsqueeze(0), leaf['theta'].unsqueeze(0),
            leaf['tropismS'].unsqueeze(0), leaf['Width_blade'].unsqueeze(0),
        )  # (1, 64, 4)
        residual = surrogate(surrogate_input)  # (1, 64, 4)
        skeleton_full = baseline + residual  # (1, 64, 4)

        skeleton = skeleton_full[0, :, :3]  # (64, 3)
        widths = skeleton_full[0, :, 3].clamp(min=0.15)  # (64,)

        # Trim to valid nodes (non-zero positions)
        valid = (skeleton.abs().sum(dim=1) > 0.01)
        if valid.sum() < 3:
            continue
        skeleton = skeleton[valid]
        widths = widths[valid]

        # Compute frames
        tangents = compute_tangents(skeleton)
        binormals = compute_binormal_field(skeleton, tangents)

        # Compute arc fractions
        seg_lens = torch.norm(torch.diff(skeleton, dim=0), dim=1)
        arc = torch.cat([torch.zeros(1, device=device), seg_lens.cumsum(0)])
        arc_fracs = arc / arc[-1].clamp(min=1e-6)

        # Compute deformations
        deforms = compute_deformations(
            arc_fracs,
            wave_normal_amp=leaf['wave_normal_amp'],
            wave_normal_freq=3.5,
            wave_normal_phase=0.0,
            wave_lateral_amp=leaf['wave_normal_amp'] * 0.3,  # proportional
            wave_lateral_freq=2.0,
            wave_lateral_phase=0.0,
            twist_max=leaf['twist_max'],
            curl_amp=leaf['curl_amp'],
            curl_freq=2.0,
            curl_phase=0.0,
            edge_ruffle_amp=leaf['edge_ruffle_amp'],
            edge_ruffle_freq=7.0,
            edge_ruffle_phase=0.0,
            fold_amp=leaf['fold_amp'],
            fold_freq=2.5,
            fold_phase=0.0,
        )

        # Loft leaf
        verts = loft_leaf(skeleton, widths, deforms, tangents, binormals, n_cross=n_cross)
        all_points.append(verts)
        all_labels.append(torch.full((len(verts),), pos, device=device, dtype=torch.long))

    if not all_points:
        return torch.zeros((1, 3), device=device), torch.zeros(1, device=device, dtype=torch.long)

    return torch.cat(all_points, dim=0), torch.cat(all_labels, dim=0)


def fit_plant(
    target_points: np.ndarray,
    target_labels: np.ndarray | None,
    surrogate: SkeletonSurrogate,
    stats_path: str,
    n_steps: int = 2000,
    phase1_steps: int = 500,
    lr_phase1: float = 1e-3,
    lr_phase2: float = 1e-4,
    lambda_reg: float = 0.01,
    device: str = 'cuda',
) -> dict:
    """Fit CPlantBox parameters to a target point cloud.

    Args:
        target_points: (N, 3) target point cloud (numpy)
        target_labels: (N,) leaf labels (numpy, optional)
        surrogate: pre-trained SkeletonSurrogate model (frozen)
        stats_path: path to maizefield3d_stats.json
        n_steps: total optimization steps
        phase1_steps: steps for structural-only optimization
        lr_phase1/lr_phase2: learning rates for two phases
        lambda_reg: regularization weight
        device: cuda or cpu

    Returns:
        dict with 'params' (optimized), 'loss_history', 'initial_loss', 'final_loss'
    """
    surrogate = surrogate.to(device).eval()
    for p in surrogate.parameters():
        p.requires_grad_(False)

    # Load priors
    prior_means, prior_stds = _load_prior(stats_path, device)

    # Initialize params at prior means
    params = prior_means.clone().detach().requires_grad_(True)

    # Target to GPU
    target_t = torch.tensor(target_points, dtype=torch.float32, device=device)
    has_labels = target_labels is not None
    target_labels_t = None
    if has_labels:
        target_labels_t = torch.tensor(target_labels, dtype=torch.long, device=device)

    loss_history = []

    # Phase 1: structural only
    # Create mask: 1 for structural params, 0 for deformation params
    structural_mask = torch.zeros(N_PARAMS, device=device)
    for pos in range(N_POSITIONS):
        offset = pos * N_PER_LEAF
        structural_mask[offset:offset + N_STRUCTURAL] = 1.0
    structural_mask[-1] = 1.0  # stem ln

    optimizer = torch.optim.Adam([params], lr=lr_phase1)

    for step in range(n_steps):
        if step == phase1_steps:
            # Phase 2: unfreeze deformations, reduce LR
            optimizer = torch.optim.Adam([params], lr=lr_phase2)

        optimizer.zero_grad()

        gen_points, gen_labels = _generate_plant_pointcloud(params, surrogate, device=device)

        # Compute loss
        if has_labels and step >= phase1_steps:
            loss = per_leaf_chamfer(gen_points, gen_labels, target_t, target_labels_t)
        else:
            loss = chamfer_distance(gen_points, target_t)

        reg = prior_loss(params, prior_means, prior_stds)
        total_loss = loss + lambda_reg * reg

        total_loss.backward()

        # Phase 1: zero gradients on deformation params
        if step < phase1_steps and params.grad is not None:
            params.grad *= structural_mask

        optimizer.step()

        loss_val = total_loss.item()
        loss_history.append(loss_val)

        if step % 100 == 0 or step == n_steps - 1:
            print(f"  step {step:4d}/{n_steps}: loss={loss_val:.6f} "
                  f"(chamfer={loss.item():.6f}, reg={reg.item():.6f})")

    return {
        'params': params.detach().cpu().numpy(),
        'loss_history': loss_history,
        'initial_loss': loss_history[0],
        'final_loss': loss_history[-1],
        'param_names': STRUCTURAL_NAMES + DEFORM_NAMES,
    }


def fit_dataset(
    target_dir: str,
    surrogate_path: str,
    stats_path: str,
    output_dir: str,
    max_plants: int | None = None,
    device: str = 'cuda',
    **fit_kwargs,
):
    """Fit all plants in a directory of STL/PLY files.

    Saves per-plant results as JSON + aggregate summary.
    """
    from ..targets.pointcloud_loader import load_pointcloud
    from ..targets.leaf_segmenter import segment_leaves

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load surrogate
    surrogate = SkeletonSurrogate()
    checkpoint = torch.load(surrogate_path, map_location=device, weights_only=True)
    surrogate.load_state_dict(checkpoint['model_state_dict'])
    surrogate = surrogate.to(device).eval()

    # Find target files
    target_path = Path(target_dir)
    files = sorted(target_path.glob('*.stl')) + sorted(target_path.glob('*.ply'))
    if max_plants:
        files = files[:max_plants]

    results = []
    for i, f in enumerate(files):
        print(f"[{i+1}/{len(files)}] Fitting {f.name}...")
        points, colors = load_pointcloud(str(f))
        labels = segment_leaves(points, colors)

        result = fit_plant(
            points, labels, surrogate, stats_path, device=device, **fit_kwargs
        )
        result['file'] = f.name
        results.append(result)

        # Save per-plant
        plant_out = {k: v for k, v in result.items() if k != 'loss_history'}
        plant_out['params'] = result['params'].tolist()
        with open(output_path / f"{f.stem}_result.json", 'w') as fp:
            json.dump(plant_out, fp, indent=2)

    # Aggregate summary
    chamfer_values = [r['final_loss'] for r in results]
    summary = {
        'n_plants': len(results),
        'chamfer_mean': float(np.mean(chamfer_values)),
        'chamfer_median': float(np.median(chamfer_values)),
        'chamfer_std': float(np.std(chamfer_values)),
        'chamfer_max': float(np.max(chamfer_values)),
    }
    with open(output_path / 'summary.json', 'w') as fp:
        json.dump(summary, fp, indent=2)

    print(f"\nDone. Mean Chamfer: {summary['chamfer_mean']:.4f}, "
          f"Median: {summary['chamfer_median']:.4f}")
    return summary
