"""Optuna objective function for geometry feature search.

Each trial:
1. Optuna picks which features are active (categorical on/off)
2. Optuna picks structural XML params (same as existing optimizer)
3. CPlantBox grows plant → leaf organs
4. GPU gradient descent optimizes baseline deformations + active extended features
5. Returns mean Chamfer distance across reference plants
"""

import numpy as np
import torch

from ..diff_lofter.deformations import (
    compute_deformations_spline,
    compute_extended_deformations,
    make_spline_control_points,
    make_extended_control_points,
    _interp_linear,
    SPLINE_DEFORM_NAMES,
    DEFAULT_N_CP,
)
from ..diff_lofter.frames import compute_binormal_field, compute_tangents
from ..diff_lofter.lofter import compute_arc_fracs, loft_leaf
from ..losses.chamfer import chamfer_distance
from ..fitting.optuna_optimizer import (
    _grow_and_extract,
    N_POSITIONS,
)
from .catalog import FEATURE_CATALOG, SEARCH_FEATURE_NAMES


# Gradient descent hyperparams
N_WIDTH_CP = 5
DEFORM_CP_CLAMP = 1.0
EXT_CP_CLAMP = 2.0
WIDTH_PROFILE_MIN = 0.3
WIDTH_PROFILE_MAX = 1.8
REG_WEIGHT = 0.005
DEFAULT_DEFORM_STEPS = 60
DEFAULT_DEFORM_LR = 0.05


def suggest_structural_params(trial, per_pos):
    """Suggest structural XML params from Optuna trial.

    Same as existing optimizer but extracted for reuse.
    """
    params = {}
    for pos in range(N_POSITIONS):
        s = per_pos[pos]
        prefix = f'l{pos}_'

        lmax = float(s.get('lmax', 60.0))
        width = float(s.get('Width_blade', 4.0))
        r = float(s.get('r', 3.0))
        tage = float(s.get('tropismAge', 5.0))

        params[f'leaf_{pos}'] = {
            'lmax': trial.suggest_float(prefix + 'lmax', max(lmax * 0.5, 20), lmax * 1.8),
            'Width_blade': trial.suggest_float(prefix + 'Wbl', max(width * 0.3, 1), width * 2.5),
            'theta': trial.suggest_float(prefix + 'theta', 0.15, 1.4),
            'tropismS': trial.suggest_float(prefix + 'tropS', 0.001, 0.1, log=True),
            'tropismAge': trial.suggest_float(prefix + 'tropAge', 1.0, max(tage * 2, 15)),
            'r': trial.suggest_float(prefix + 'r', max(r * 0.3, 0.5), r * 3.0),
            'collarLength': trial.suggest_float(prefix + 'collar', 0.0, 25.0),
            'initBeta': trial.suggest_float(prefix + 'iBeta', -3.14, 3.14),
        }

    params['stem_ln'] = trial.suggest_float('stem_ln', 8.0, 22.0)
    params['stem_tropismS'] = trial.suggest_float('stem_tropS', 0.0, 0.015)
    return params


def suggest_active_features(trial) -> set[str]:
    """Suggest which extended features are active for this trial."""
    active = set()
    for name in SEARCH_FEATURE_NAMES:
        if trial.suggest_categorical(f'feat_{name}', [True, False]):
            active.add(name)
    return active


def _optimize_with_features(
    leaf_organs,
    target_pc,
    active_features,
    device='cuda',
    n_steps=DEFAULT_DEFORM_STEPS,
    lr=DEFAULT_DEFORM_LR,
):
    """GPU gradient descent over baseline deformations + active extended features.

    Args:
        leaf_organs: List of organ dicts from _grow_and_extract.
        target_pc: (M, 3) GPU tensor of reference points.
        active_features: Set of feature names from catalog to optimize.
        device: Torch device.
        n_steps: Adam iterations.
        lr: Learning rate.

    Returns:
        (best_chamfer, best_params_dict)
    """
    if not leaf_organs:
        return 1e6, {}

    leaf_data = []
    grad_params = []

    for organ in leaf_organs:
        skeleton = torch.tensor(organ['skeleton'], dtype=torch.float32, device=device)
        widths_base = torch.tensor(organ['widths'], dtype=torch.float32, device=device)
        if skeleton.shape[0] < 3:
            continue

        tangents = compute_tangents(skeleton)
        binormals = compute_binormal_field(skeleton, tangents)
        arc_fracs = compute_arc_fracs(skeleton)

        # Baseline deformation CPs (always active)
        cp = make_spline_control_points(n_cp=DEFAULT_N_CP, device=device, requires_grad=True)
        for v in cp.values():
            grad_params.append(v)

        # Width profile (always active)
        width_profile = torch.ones(N_WIDTH_CP, device=device, dtype=torch.float32,
                                   requires_grad=True)
        grad_params.append(width_profile)

        # Extended feature CPs (only for active features)
        ext_cp = make_extended_control_points(
            active_features, FEATURE_CATALOG, device=device, requires_grad=True
        )
        for v in ext_cp.values():
            grad_params.append(v)

        leaf_data.append({
            'skeleton': skeleton,
            'widths_base': widths_base,
            'tangents': tangents,
            'binormals': binormals,
            'arc_fracs': arc_fracs,
            'cp': cp,
            'width_profile': width_profile,
            'ext_cp': ext_cp,
        })

    if not leaf_data or not grad_params:
        return 1e6, {}

    optimizer = torch.optim.Adam(grad_params, lr=lr)
    best_loss = float('inf')
    best_params = {}

    for step in range(n_steps):
        optimizer.zero_grad()
        all_verts = []
        reg = torch.tensor(0.0, device=device)

        for ld in leaf_data:
            # Baseline deformations
            deforms = compute_deformations_spline(ld['arc_fracs'], ld['cp'])

            # Width profile
            w_mult = _interp_linear(ld['arc_fracs'], ld['width_profile'])
            widths = ld['widths_base'] * w_mult

            # Extended deformations (only active ones)
            ext_deforms = None
            if ld['ext_cp']:
                ext_deforms = compute_extended_deformations(
                    ld['arc_fracs'], ld['ext_cp']
                )

            # Loft with extended features
            verts = loft_leaf(
                ld['skeleton'], widths, deforms,
                ld['tangents'], ld['binormals'],
                n_cross=7,
                extended_deformations=ext_deforms,
            )
            all_verts.append(verts)

            # Regularization on all CPs
            for t in ld['cp'].values():
                reg = reg + REG_WEIGHT * (t ** 2).sum()
            for t in ld['ext_cp'].values():
                reg = reg + REG_WEIGHT * (t ** 2).sum()

        gen_pc = torch.cat(all_verts, dim=0)
        loss = chamfer_distance(gen_pc, target_pc) + reg
        loss.backward()
        optimizer.step()

        # Clamp parameters
        with torch.no_grad():
            for ld in leaf_data:
                for t in ld['cp'].values():
                    t.clamp_(-DEFORM_CP_CLAMP, DEFORM_CP_CLAMP)
                ld['width_profile'].clamp_(WIDTH_PROFILE_MIN, WIDTH_PROFILE_MAX)
                for name, t in ld['ext_cp'].items():
                    spec = FEATURE_CATALOG[name]
                    t.clamp_(spec['bounds'][0], spec['bounds'][1])

        # Track best (Chamfer only, not regularized)
        with torch.no_grad():
            chamfer_only = chamfer_distance(gen_pc.detach(), target_pc).item()
        if chamfer_only < best_loss:
            best_loss = chamfer_only
            best_params = {
                i: {
                    'baseline_cp': {
                        name: ld['cp'][name].detach().cpu().tolist()
                        for name in SPLINE_DEFORM_NAMES
                    },
                    'width_profile': ld['width_profile'].detach().cpu().tolist(),
                    'extended_cp': {
                        name: ld['ext_cp'][name].detach().cpu().tolist()
                        for name in ld['ext_cp']
                    },
                }
                for i, ld in enumerate(leaf_data)
            }

    return best_loss, best_params


def make_objective(
    reference_plants,
    per_pos_stats,
    day=60,
    device='cuda',
    template_xml=None,
    deform_steps=DEFAULT_DEFORM_STEPS,
    deform_lr=DEFAULT_DEFORM_LR,
):
    """Create an Optuna objective function for feature search.

    Args:
        reference_plants: List of dicts with 'points' (np.ndarray (N,3))
            and 'name' (str) for each reference plant.
        per_pos_stats: Per-position stats list (from maizefield3d_stats.json).
        day: Simulation day.
        device: Torch device.
        template_xml: Calibrated XML path.
        deform_steps: GPU gradient descent steps per trial.
        deform_lr: Adam learning rate.

    Returns:
        Callable objective(trial) -> float (mean Chamfer).
    """
    # Pre-upload reference point clouds to GPU
    ref_gpu = []
    for rp in reference_plants:
        pts = rp['points']
        if len(pts) > 5000:
            idx = np.random.RandomState(42).choice(len(pts), 5000, replace=False)
            pts = pts[idx]
        ref_gpu.append({
            'name': rp['name'],
            'points': torch.tensor(pts, dtype=torch.float32, device=device),
        })

    def objective(trial):
        # 1. Feature selection
        active_features = suggest_active_features(trial)

        # 2. Structural params
        params_dict = suggest_structural_params(trial, per_pos_stats)

        # 3. Grow plant
        organs = _grow_and_extract(params_dict, day=day, template_xml=template_xml)
        if organs is None:
            return 1e6

        # 4. Evaluate across reference plants with pruning
        chamfers = []
        for i, ref in enumerate(ref_gpu):
            chamfer, _ = _optimize_with_features(
                organs, ref['points'], active_features,
                device=device, n_steps=deform_steps, lr=deform_lr,
            )
            chamfers.append(chamfer)

            # Optuna pruning: report intermediate value
            trial.report(np.mean(chamfers), i)
            if trial.should_prune():
                raise __import__('optuna').TrialPruned()

        return float(np.mean(chamfers))

    return objective


def make_cpu_objective(
    reference_plants,
    per_pos_stats,
    day=60,
    template_xml=None,
    deform_steps=DEFAULT_DEFORM_STEPS,
    deform_lr=DEFAULT_DEFORM_LR,
):
    """Create a CPU-only objective (no GPU Chamfer).

    For massively parallel workers (128+) where GPU contention is a problem.
    Uses scipy KD-tree for Chamfer computation instead of torch.cdist.
    Deformation optimization still uses PyTorch but on CPU.

    Args:
        Same as make_objective but without device.

    Returns:
        Callable objective(trial) -> float.
    """
    from scipy.spatial import cKDTree

    def chamfer_cpu(gen_pts_np, ref_pts_np):
        """CPU Chamfer distance using KD-trees."""
        tree_ref = cKDTree(ref_pts_np)
        tree_gen = cKDTree(gen_pts_np)
        d1, _ = tree_ref.query(gen_pts_np)
        d2, _ = tree_gen.query(ref_pts_np)
        return (d1.mean() + d2.mean()) / 2.0

    # Subsample reference plants
    ref_data = []
    for rp in reference_plants:
        pts = rp['points']
        if len(pts) > 5000:
            idx = np.random.RandomState(42).choice(len(pts), 5000, replace=False)
            pts = pts[idx]
        ref_data.append({'name': rp['name'], 'points': pts})

    def objective(trial):
        active_features = suggest_active_features(trial)
        params_dict = suggest_structural_params(trial, per_pos_stats)

        organs = _grow_and_extract(params_dict, day=day, template_xml=template_xml)
        if organs is None:
            return 1e6

        chamfers = []
        for i, ref in enumerate(ref_data):
            # CPU gradient descent
            target_cpu = torch.tensor(ref['points'], dtype=torch.float32, device='cpu')
            chamfer, _ = _optimize_with_features(
                organs, target_cpu, active_features,
                device='cpu', n_steps=deform_steps, lr=deform_lr,
            )
            chamfers.append(chamfer)

            trial.report(np.mean(chamfers), i)
            if trial.should_prune():
                raise __import__('optuna').TrialPruned()

        return float(np.mean(chamfers))

    return objective
