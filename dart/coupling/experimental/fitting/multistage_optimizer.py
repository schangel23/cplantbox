"""Multi-stage sequential optimizer: fit one set of XML params across multiple growth stages.

Same structural params evaluated at multiple (day, target) pairs.
Loss = sum of Chamfer distances across all stages.

Uses sequential per-leaf sep-CMA-ES with fast skeleton-only evaluation,
then a single GPU deformation pass at the end per stage.
"""

import json
import multiprocessing as mp
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

from ..diff_lofter.deformations import (
    compute_deformations_spline,
    make_spline_control_points,
    SPLINE_DEFORM_NAMES,
    DEFAULT_N_CP,
)
from ..diff_lofter.frames import compute_binormal_field, compute_tangents
from ..diff_lofter.lofter import compute_arc_fracs, loft_leaf, loft_stem
from ..losses.chamfer import chamfer_distance

N_POSITIONS = 11

LEAF_PARAMS = [
    'lmax', 'Width_blade', 'theta', 'tropismS', 'tropismAge',
    'r', 'collarLength', 'initBeta',
    'kappa_base', 'kappa_mid', 'kappa_tip',
]

CURVATURE_PHI = [0.0, 0.5, 1.0]


def _grow_single(stem_params, leaf_params_list, day=60, template_xml=None):
    """Grow CPlantBox plant, return leaf organs."""
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
                if 0 <= pos < N_POSITIONS and pos < len(leaf_params_list):
                    p = leaf_params_list[pos]
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

                    to_remove = []
                    for elem in organ:
                        name = elem.get('name', '')
                        if name in xml_map:
                            elem.set('value', str(xml_map[name]))
                        elif name == 'leafCurvature':
                            to_remove.append(elem)
                    for elem in to_remove:
                        organ.remove(elem)

                    curv = ET.SubElement(organ, 'parameter')
                    curv.set('name', 'leafCurvature')
                    curv.set('phi', ' '.join(str(v) for v in CURVATURE_PHI))
                    curv.set('kappa', f"{p['kappa_base']} {p['kappa_mid']} {p['kappa_tip']}")

            elif organ.get('type') == 'stem':
                for elem in organ:
                    name = elem.get('name', '')
                    if name == 'ln':
                        elem.set('value', str(stem_params['ln']))
                    elif name == 'tropismS':
                        elem.set('value', str(stem_params['tropismS']))
                    elif name == 'lnf':
                        elem.set('value', str(int(round(stem_params.get('lnf', 0)))))

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

        return organs

    except Exception as e:
        print(f"  CPlantBox failed: {e}", file=sys.stderr)
        return None
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


def _fast_chamfer(organs, target_pc, device='cuda'):
    """Fast Chamfer: loft with zero deformations, no gradient optimization."""
    all_verts = []
    for organ in organs:
        skeleton = torch.tensor(organ['skeleton'], dtype=torch.float32, device=device)
        widths = torch.tensor(organ['widths'], dtype=torch.float32, device=device)
        if skeleton.shape[0] < 3:
            continue
        tangents = compute_tangents(skeleton)
        binormals = compute_binormal_field(skeleton, tangents)

        if organ.get('type') in ('stem', 'root'):
            verts = loft_stem(skeleton, widths, tangents, binormals, n_sides=8)
        else:
            arc_fracs = compute_arc_fracs(skeleton)
            cp = make_spline_control_points(n_cp=DEFAULT_N_CP, device=device, requires_grad=False)
            deforms = compute_deformations_spline(arc_fracs, cp)
            verts = loft_leaf(skeleton, widths, deforms, tangents, binormals, n_cross=7)
        all_verts.append(verts)

    if not all_verts:
        return 1e6
    gen_pc = torch.cat(all_verts, dim=0)
    with torch.no_grad():
        return chamfer_distance(gen_pc, target_pc).item()


def _multistage_chamfer(leaf_organs_per_stage, targets_gpu, device='cuda'):
    """Sum Chamfer across all growth stages."""
    total = 0.0
    for organs, target_pc in zip(leaf_organs_per_stage, targets_gpu):
        if organs is None:
            total += 1e6
        else:
            total += _fast_chamfer(organs, target_pc, device)
    return total


def _grow_worker_multi(args):
    """CPU worker: grow plant at multiple days, return list of organ lists."""
    stem_params, leaf_params_list, days, template_xml = args
    results = []
    for day in days:
        organs = _grow_single(stem_params, leaf_params_list, day=day, template_xml=template_xml)
        results.append(organs)
    return results


def _default_leaf_params(stats_pos):
    return {
        'lmax': float(stats_pos.get('lmax', 60.0)),
        'Width_blade': float(stats_pos.get('Width_blade', 4.0)),
        'theta': float(stats_pos.get('theta', 0.7)),
        'tropismS': float(stats_pos.get('tropismS', 0.03)),
        'tropismAge': float(stats_pos.get('tropismAge', 5.0)),
        'r': float(stats_pos.get('r', 3.0)),
        'collarLength': 10.0,
        'initBeta': 0.2,
        'kappa_base': 0.0,
        'kappa_mid': 0.0,
        'kappa_tip': 0.0,
    }


def _leaf_params_from_vec(vec):
    return {k: v for k, v in zip(LEAF_PARAMS, vec)}


def _leaf_params_to_vec(params):
    return [params[k] for k in LEAF_PARAMS]


def _leaf_bounds(stats_pos):
    default = _default_leaf_params(stats_pos)
    x0 = _leaf_params_to_vec(default)
    lmax = default['lmax']
    width = default['Width_blade']
    r = default['r']
    tage = default['tropismAge']

    lo = [max(lmax*0.5, 20), max(width*0.3, 1), 0.2, 0.001, 1.0,
          max(r*0.3, 0.5), 0.0, -3.14, 0.0, 0.0, 0.0]
    hi = [lmax*1.8, width*2.5, 0.85, 0.1, max(tage*2, 15),
          r*3.0, 30.0, 3.14, 0.05, 0.15, 0.25]
    return np.array(x0), np.array(lo), np.array(hi)


def _optimize_deformations_single(organs, target_pc, device='cuda', n_steps=100, lr=0.05):
    """GPU deformation optimization for a single stage. No width profile.

    Leaves get spline deformations. Stems are lofted as cylinders (no deformations).
    """
    if not organs:
        return 1e6, {}

    leaf_data = []
    stem_data = []
    grad_params = []

    for organ in organs:
        skeleton = torch.tensor(organ['skeleton'], dtype=torch.float32, device=device)
        widths = torch.tensor(organ['widths'], dtype=torch.float32, device=device)
        if skeleton.shape[0] < 3:
            continue
        tangents = compute_tangents(skeleton)
        binormals = compute_binormal_field(skeleton, tangents)

        if organ.get('type') in ('stem', 'root'):
            stem_data.append({'skeleton': skeleton, 'widths': widths,
                              'tangents': tangents, 'binormals': binormals})
        else:
            arc_fracs = compute_arc_fracs(skeleton)
            cp = make_spline_control_points(n_cp=DEFAULT_N_CP, device=device, requires_grad=True)
            for v in cp.values():
                grad_params.append(v)
            leaf_data.append({'skeleton': skeleton, 'widths': widths, 'tangents': tangents,
                              'binormals': binormals, 'arc_fracs': arc_fracs, 'cp': cp})

    if not leaf_data and not stem_data:
        return 1e6, {}

    # Stem vertices (fixed, no optimization)
    stem_verts = []
    for sd in stem_data:
        stem_verts.append(loft_stem(sd['skeleton'], sd['widths'], sd['tangents'], sd['binormals']))

    if not grad_params:
        # No leaves to optimize — just return stem Chamfer
        if stem_verts:
            gen_pc = torch.cat(stem_verts, dim=0)
            with torch.no_grad():
                return chamfer_distance(gen_pc, target_pc).item(), {}
        return 1e6, {}

    optimizer = torch.optim.Adam(grad_params, lr=lr)
    best_loss = float('inf')
    best_params = {}

    for _ in range(n_steps):
        optimizer.zero_grad()
        all_verts = list(stem_verts)  # stems are fixed
        for ld in leaf_data:
            deforms = compute_deformations_spline(ld['arc_fracs'], ld['cp'])
            all_verts.append(loft_leaf(ld['skeleton'], ld['widths'], deforms,
                                       ld['tangents'], ld['binormals'], n_cross=7))
        loss = chamfer_distance(torch.cat(all_verts, dim=0), target_pc)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            for ld in leaf_data:
                for t in ld['cp'].values():
                    t.clamp_(-1.5, 1.5)
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_params = {
                i: {'control_points': {name: ld['cp'][name].detach().cpu().tolist()
                                       for name in SPLINE_DEFORM_NAMES}}
                for i, ld in enumerate(leaf_data)
            }

    return best_loss, best_params


def fit_plant_multistage(
    stages,
    stats_path,
    stem_evals=200,
    leaf_evals=500,
    device='cuda',
    seed=42,
    template_xml=None,
    n_workers=64,
):
    """Fit one set of XML params across multiple growth stages.

    Args:
        stages: list of (day, target_points) tuples.
            Each target_points is (N, 3) numpy array in cm.
        stats_path: path to maizefield3d_stats.json
        stem_evals: CMA-ES evals for stem (3D)
        leaf_evals: CMA-ES evals per leaf (11D)
        device: torch device
        seed: random seed
        template_xml: calibrated XML path
        n_workers: CPU workers

    Returns:
        dict with fitted params, per-stage losses, etc.
    """
    import cma

    n_workers = min(n_workers, mp.cpu_count())
    days = [s[0] for s in stages]
    n_stages = len(stages)

    with open(stats_path) as f:
        stats = json.load(f)
    per_pos = stats.get('per_position', stats) if isinstance(stats, dict) else stats
    if not isinstance(per_pos, list):
        per_pos = [per_pos[str(i)] for i in range(N_POSITIONS)]

    # Subsample targets
    targets_np = []
    for day, pts in stages:
        if len(pts) > 5000:
            idx = np.random.RandomState(seed).choice(len(pts), 5000, replace=False)
            targets_np.append(pts[idx])
        else:
            targets_np.append(pts)

    # Rotation alignment using mature stage (last one)
    leaf_params_list = [_default_leaf_params(per_pos[i]) for i in range(N_POSITIONS)]
    stem_params = {'ln': 14.5, 'tropismS': 0.002, 'lnf': 0.0}

    mature_organs = _grow_single(stem_params, leaf_params_list, day=days[-1], template_xml=template_xml)
    if mature_organs:
        from dart.coupling.geometry.g1_to_g3 import loft_organs
        ref_pts = np.array(loft_organs(mature_organs).vertices)
        if len(ref_pts) > 5000:
            ref_pts = ref_pts[np.random.RandomState(seed).choice(len(ref_pts), 5000, replace=False)]
        from ..targets.pointcloud_loader import align_rotation_z
        # Align all targets using the same rotation
        for i in range(n_stages):
            targets_np[i], angle = align_rotation_z(targets_np[i], ref_pts, n_angles=72)
        print(f"Rotation alignment: {angle:.0f} deg", file=sys.stderr)

    targets_gpu = [torch.tensor(t, dtype=torch.float32, device=device) for t in targets_np]

    print(f"\nMulti-stage fitting: {n_stages} stages, days={days}", file=sys.stderr)
    print(f"  Target sizes: {[len(t) for t in targets_np]}", file=sys.stderr)

    # ========== PHASE 1: Fit stem (3D) — sum Chamfer across all stages ==========
    print(f"\n=== PHASE 1: Stem fitting (3D, {stem_evals} evals, {n_stages} stages, "
          f"{n_workers} workers) ===", file=sys.stderr)

    opts = cma.CMAOptions()
    opts['maxfevals'] = stem_evals
    opts['seed'] = seed
    opts['bounds'] = [[8.0, 0.0, 0.0], [22.0, 0.015, 5.0]]
    opts['verbose'] = -1
    opts['CMA_diagonal'] = True

    es = cma.CMAEvolutionStrategy([14.5, 0.002, 0.0], 0.3, opts)
    while not es.stop():
        solutions = es.ask()

        # Parallel: grow each candidate at all stages
        grow_args = []
        for x in solutions:
            sp = {'ln': x[0], 'tropismS': x[1], 'lnf': x[2]}
            grow_args.append((sp, leaf_params_list, days, template_xml))

        with mp.Pool(min(n_workers, len(solutions))) as pool:
            all_results = pool.map(_grow_worker_multi, grow_args)

        fitnesses = []
        for organs_per_stage in all_results:
            total = 0.0
            for organs, tgt in zip(organs_per_stage, targets_gpu):
                if organs is None:
                    total += 1e6
                else:
                    total += _fast_chamfer(organs, tgt, device)
            fitnesses.append(total / n_stages)

        es.tell(solutions, fitnesses)

    best_stem = es.result.xbest
    stem_params = {'ln': best_stem[0], 'tropismS': best_stem[1], 'lnf': best_stem[2]}
    print(f"  Stem: ln={stem_params['ln']:.1f}, tropismS={stem_params['tropismS']:.4f}, "
          f"avg_chamfer={es.result.fbest:.2f}", file=sys.stderr)

    # ========== PHASE 2: Fit each leaf (11D) — sum Chamfer across all stages ==========
    print(f"\n=== PHASE 2: Per-leaf fitting (11D x {N_POSITIONS}, {leaf_evals} evals each, "
          f"{n_stages} stages, {n_workers} workers) ===", file=sys.stderr)

    per_leaf_losses = []

    for pos in range(N_POSITIONS):
        x0, lo, hi = _leaf_bounds(per_pos[pos])

        opts = cma.CMAOptions()
        opts['maxfevals'] = leaf_evals
        opts['seed'] = seed + pos
        opts['bounds'] = [lo.tolist(), hi.tolist()]
        opts['verbose'] = -1
        opts['CMA_diagonal'] = True
        opts['popsize'] = 64

        x0_clipped = np.clip(x0, lo * 1.01, hi * 0.99)
        es = cma.CMAEvolutionStrategy(x0_clipped, 0.3, opts)

        while not es.stop():
            solutions = es.ask()

            grow_args = []
            for x in solutions:
                lp = list(leaf_params_list)
                lp[pos] = _leaf_params_from_vec(x)
                grow_args.append((stem_params, lp, days, template_xml))

            with mp.Pool(min(n_workers, len(solutions))) as pool:
                all_results = pool.map(_grow_worker_multi, grow_args)

            fitnesses = []
            for organs_per_stage in all_results:
                total = 0.0
                for organs, tgt in zip(organs_per_stage, targets_gpu):
                    if organs is None:
                        total += 1e6
                    else:
                        total += _fast_chamfer(organs, tgt, device)
                fitnesses.append(total / n_stages)

            es.tell(solutions, fitnesses)

        best_x = es.result.xbest
        leaf_params_list[pos] = _leaf_params_from_vec(best_x)
        per_leaf_losses.append(float(es.result.fbest))

        print(f"  Leaf {pos} (st={pos+2}): avg_chamfer={es.result.fbest:.2f} cm  "
              f"lmax={best_x[0]:.1f} Wbl={best_x[1]:.1f} theta={best_x[2]:.2f} "
              f"r={best_x[5]:.1f} tropAge={best_x[4]:.1f}",
              file=sys.stderr)

    # ========== PHASE 3: Final evaluation + deformations per stage ==========
    print(f"\n=== PHASE 3: Final evaluation + deformations ===", file=sys.stderr)

    stage_results = []
    for i, (day, _) in enumerate(stages):
        organs = _grow_single(stem_params, leaf_params_list, day=day, template_xml=template_xml)
        if organs is None:
            stage_results.append({'day': day, 'chamfer': 1e6, 'deform_params': {}})
            continue

        skel_chamfer = _fast_chamfer(organs, targets_gpu[i], device)
        final_chamfer, deforms = _optimize_deformations_single(
            organs, targets_gpu[i], device, n_steps=100, lr=0.05,
        )
        stage_results.append({
            'day': day,
            'skeleton_chamfer': float(skel_chamfer),
            'final_chamfer': float(final_chamfer),
            'n_leaves': len(organs),
            'deform_params': deforms,
        })
        print(f"  Day {day}: skeleton={skel_chamfer:.2f} → final={final_chamfer:.2f} cm "
              f"({len(organs)} leaves)", file=sys.stderr)

    return {
        'stem_params': stem_params,
        'leaf_params': [_leaf_params_to_vec(lp) for lp in leaf_params_list],
        'leaf_param_names': LEAF_PARAMS,
        'per_leaf_losses': per_leaf_losses,
        'stage_results': stage_results,
        'days': days,
    }


if __name__ == '__main__':
    print("Usage: import and call fit_plant_multistage() with a list of (day, target_points) tuples.")
