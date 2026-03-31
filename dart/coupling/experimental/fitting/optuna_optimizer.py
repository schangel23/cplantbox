"""Optuna TPE optimizer for CPlantBox structural fitting.

Replaces CMA-ES with Optuna's Tree-structured Parzen Estimator.
Better sample efficiency for high-dim mixed spaces, native pruning
of bad candidates, and handles discrete params (lnf) properly.

Uses the same _grow_and_extract + _optimize_deformations pipeline
as the hybrid optimizer, but with Optuna driving the structural search.
"""

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

from ..diff_lofter.deformations import (
    compute_deformations_spline,
    make_spline_control_points,
    _interp_linear,
    SPLINE_DEFORM_NAMES,
    DEFAULT_N_CP,
)
from ..diff_lofter.frames import compute_binormal_field, compute_tangents
from ..diff_lofter.lofter import compute_arc_fracs, loft_leaf
from ..losses.chamfer import chamfer_distance


N_POSITIONS = 11
N_WIDTH_CP = 5
DEFORM_CP_CLAMP = 1.0
WIDTH_PROFILE_MIN = 0.3
WIDTH_PROFILE_MAX = 1.8
REG_WEIGHT = 0.005


def _grow_and_extract(params_dict, day=60, template_xml=None):
    """Grow CPlantBox plant from param dict, return leaf organs."""
    import xml.etree.ElementTree as ET

    if template_xml is None:
        from dart.coupling.config import DATA_DIR
        template_xml = str(DATA_DIR / "maize_calibrated.xml")

    tmp_path = None
    try:
        tree = ET.parse(template_xml)
        root = tree.getroot()

        for organ in root.iter('organ'):
            if organ.get('type') == 'leaf':
                sub = int(organ.get('subType', '0'))
                pos = sub - 2
                if 0 <= pos < N_POSITIONS:
                    p = params_dict[f'leaf_{pos}']
                    xml_map = {
                        'lmax': p['lmax'],
                        'Width_blade': p['Width_blade'],
                        'theta': p['theta'],
                        'tropismS': p['tropismS'],
                        'tropismAge': p['tropismAge'],
                        'r': p['r'],
                        'areaMax': p['lmax'] * p['Width_blade'] * 2.0 * 0.73,
                        'collarLength': p['collarLength'],
                        'InitBeta': p['initBeta'],
                    }
                    for elem in organ:
                        name = elem.get('name', '')
                        if name in xml_map:
                            elem.set('value', str(xml_map[name]))

            elif organ.get('type') == 'stem':
                for elem in organ:
                    name = elem.get('name', '')
                    if name == 'ln':
                        elem.set('value', str(params_dict['stem_ln']))
                    elif name == 'tropismS':
                        elem.set('value', str(params_dict['stem_tropismS']))

        tmp = tempfile.NamedTemporaryFile(suffix='.xml', delete=False)
        tree.write(tmp.name)
        tmp_path = tmp.name
        tmp.close()

        old_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        try:
            from dart.coupling.growth.grow import grow_plant
            from dart.coupling.geometry.cplantbox_adapter import extract_organs_for_lofter
            plant = grow_plant(tmp_path, simulation_time=day)
            organs = extract_organs_for_lofter(plant, skip_roots=True)
        finally:
            sys.stdout.close()
            sys.stdout = old_stdout

        return [o for o in organs if o['type'] == 'leaf']

    except Exception as e:
        print(f"  CPlantBox failed: {e}", file=sys.stderr)
        return None
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


def _optimize_deformations(leaf_organs, target_pc, device='cuda', n_steps=80, lr=0.05):
    """GPU: optimize spline deformations + width profiles with clamping."""
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

        cp = make_spline_control_points(n_cp=DEFAULT_N_CP, device=device, requires_grad=True)
        for v in cp.values():
            grad_params.append(v)

        width_profile = torch.ones(N_WIDTH_CP, device=device, dtype=torch.float32,
                                   requires_grad=True)
        grad_params.append(width_profile)

        leaf_data.append({
            'skeleton': skeleton, 'widths_base': widths_base,
            'tangents': tangents, 'binormals': binormals,
            'arc_fracs': arc_fracs, 'cp': cp, 'width_profile': width_profile,
        })

    if not leaf_data or not grad_params:
        return 1e6, {}

    optimizer = torch.optim.Adam(grad_params, lr=lr)
    best_loss = float('inf')
    best_params = {}

    for _ in range(n_steps):
        optimizer.zero_grad()
        all_verts = []
        reg = torch.tensor(0.0, device=device)

        for ld in leaf_data:
            deforms = compute_deformations_spline(ld['arc_fracs'], ld['cp'])
            w_mult = _interp_linear(ld['arc_fracs'], ld['width_profile'])
            widths = ld['widths_base'] * w_mult
            verts = loft_leaf(ld['skeleton'], widths, deforms,
                              ld['tangents'], ld['binormals'], n_cross=7)
            all_verts.append(verts)
            for t in ld['cp'].values():
                reg = reg + REG_WEIGHT * (t ** 2).sum()

        gen_pc = torch.cat(all_verts, dim=0)
        loss = chamfer_distance(gen_pc, target_pc) + reg
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            for ld in leaf_data:
                for t in ld['cp'].values():
                    t.clamp_(-DEFORM_CP_CLAMP, DEFORM_CP_CLAMP)
                ld['width_profile'].clamp_(WIDTH_PROFILE_MIN, WIDTH_PROFILE_MAX)

        with torch.no_grad():
            chamfer_only = chamfer_distance(gen_pc.detach(), target_pc).item()
        if chamfer_only < best_loss:
            best_loss = chamfer_only
            best_params = {
                i: {
                    'control_points': {
                        name: ld['cp'][name].detach().cpu().tolist()
                        for name in SPLINE_DEFORM_NAMES
                    },
                    'width_profile': ld['width_profile'].detach().cpu().tolist(),
                }
                for i, ld in enumerate(leaf_data)
            }

    return best_loss, best_params


def fit_plant_optuna(
    target_points,
    stats_path,
    max_evals=2000,
    deform_steps=80,
    deform_lr=0.05,
    day=60,
    device='cuda',
    seed=42,
    template_xml=None,
):
    """Fit CPlantBox params using Optuna TPE + GPU gradient descent.

    Args:
        target_points: (N, 3) numpy array
        stats_path: path to maizefield3d_stats.json
        max_evals: Optuna trials
        deform_steps: gradient steps per trial
        deform_lr: Adam LR for deformations
        day: simulation day
        device: torch device
        seed: random seed
        template_xml: calibrated XML path

    Returns:
        dict with best params, loss, etc.
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    with open(stats_path) as f:
        stats = json.load(f)

    per_pos = stats.get('per_position', stats) if isinstance(stats, dict) else stats
    if not isinstance(per_pos, list):
        per_pos = [per_pos[str(i)] for i in range(N_POSITIONS)]

    # Subsample + align target
    if len(target_points) > 5000:
        idx = np.random.RandomState(seed).choice(len(target_points), 5000, replace=False)
        target_sub = target_points[idx]
    else:
        target_sub = target_points

    # Grow default plant for rotation alignment
    default_params = _build_default_params(per_pos)
    init_organs = _grow_and_extract(default_params, day=day, template_xml=template_xml)
    if init_organs:
        from dart.coupling.geometry.g1_to_g3 import loft_organs
        ref_mesh = loft_organs(init_organs)
        ref_pts = np.array(ref_mesh.vertices)
        if len(ref_pts) > 5000:
            ref_pts = ref_pts[np.random.RandomState(seed).choice(len(ref_pts), 5000, replace=False)]
        from ..targets.pointcloud_loader import align_rotation_z
        target_sub, best_angle = align_rotation_z(target_sub, ref_pts, n_angles=72)
        print(f"Rotation alignment: {best_angle:.0f} deg", file=sys.stderr)

    target_gpu = torch.tensor(target_sub, dtype=torch.float32, device=device)

    best_deforms = {}

    def objective(trial):
        nonlocal best_deforms

        params_dict = _suggest_params(trial, per_pos)
        organs = _grow_and_extract(params_dict, day=day, template_xml=template_xml)
        if organs is None:
            return 1e6

        chamfer, deforms = _optimize_deformations(
            organs, target_gpu, device, deform_steps, deform_lr
        )

        try:
            is_best = chamfer <= trial.study.best_value
        except ValueError:
            is_best = True  # first trial
        if is_best:
            best_deforms = deforms

        return chamfer

    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=seed, multivariate=True),
    )

    print(f"Optuna TPE: {max_evals} trials, {N_POSITIONS} leaves", file=sys.stderr)

    # Enqueue default params as first trial
    study.enqueue_trial(_params_to_enqueue(per_pos))

    study.optimize(objective, n_trials=max_evals, show_progress_bar=True)

    best = study.best_trial
    print(f"Best Chamfer: {best.value:.2f} cm after {len(study.trials)} trials",
          file=sys.stderr)

    return {
        'optuna_params': best.params,
        'deform_params': best_deforms,
        'best_loss': float(best.value),
        'n_trials': len(study.trials),
    }


def _build_default_params(per_pos):
    """Build default param dict from stats."""
    params = {}
    for pos in range(N_POSITIONS):
        s = per_pos[pos]
        params[f'leaf_{pos}'] = {
            'lmax': float(s.get('lmax', 60.0)),
            'Width_blade': float(s.get('Width_blade', 4.0)),
            'theta': float(s.get('theta', 0.7)),
            'tropismS': float(s.get('tropismS', 0.03)),
            'tropismAge': float(s.get('tropismAge', 5.0)),
            'r': float(s.get('r', 3.0)),
            'collarLength': 10.0,
            'initBeta': 0.2,
        }
    params['stem_ln'] = 14.5
    params['stem_tropismS'] = 0.002
    return params


def _suggest_params(trial, per_pos):
    """Suggest structural params for one Optuna trial."""
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


def _params_to_enqueue(per_pos):
    """Convert default params to Optuna enqueue format."""
    enqueue = {}
    for pos in range(N_POSITIONS):
        s = per_pos[pos]
        prefix = f'l{pos}_'
        enqueue[prefix + 'lmax'] = float(s.get('lmax', 60.0))
        enqueue[prefix + 'Wbl'] = float(s.get('Width_blade', 4.0))
        enqueue[prefix + 'theta'] = float(s.get('theta', 0.7))
        enqueue[prefix + 'tropS'] = float(s.get('tropismS', 0.03))
        enqueue[prefix + 'tropAge'] = float(s.get('tropismAge', 5.0))
        enqueue[prefix + 'r'] = float(s.get('r', 3.0))
        enqueue[prefix + 'collar'] = 10.0
        enqueue[prefix + 'iBeta'] = 0.2
    enqueue['stem_ln'] = 14.5
    enqueue['stem_tropS'] = 0.002
    return enqueue


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python -m dart.coupling.experimental.fitting.optuna_optimizer "
              "<target.stl> <stats.json> [max_evals]")
        sys.exit(1)

    from ..targets.pointcloud_loader import load_pointcloud

    target_path = sys.argv[1]
    stats_path = sys.argv[2]
    max_evals = int(sys.argv[3]) if len(sys.argv) > 3 else 2000

    target_pts, _ = load_pointcloud(target_path, n_points=10000)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    result = fit_plant_optuna(
        target_pts, stats_path, max_evals=max_evals, device=device,
    )

    out_base = target_path.rsplit('.', 1)[0]
    with open(f'{out_base}_optuna_fit.json', 'w') as f:
        json.dump(result, f, indent=2, default=str)
    print(f"Saved: {out_base}_optuna_fit.json")
