"""Hybrid CMA-ES (structural, CPU) + gradient descent (deformations, GPU).

CMA-ES searches structural XML params + per-leaf InitBeta + lnf through
real CPlantBox. For each candidate, gradient descent optimizes spline-based
deformation control points + per-leaf width profiles through the
differentiable PyTorch lofter + GPU Chamfer distance.

CPU workers run CPlantBox in parallel (no CUDA).
Main thread runs GPU deformation optimization sequentially (no fork issues).
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
    compute_deformations,
    compute_deformations_spline,
    make_spline_control_points,
    SPLINE_DEFORM_NAMES,
    DEFAULT_N_CP,
)
from ..diff_lofter.frames import compute_binormal_field, compute_tangents
from ..diff_lofter.lofter import compute_arc_fracs, loft_leaf
from ..losses.chamfer import chamfer_distance

N_POSITIONS = 11
XML_PARAMS = [
    'lmax', 'Width_blade', 'theta', 'tropismS', 'tropismAge', 'r',
    'width_taper', 'collarLength', 'tropismExponent', 'initBeta',
]
N_XML_PER_LEAF = len(XML_PARAMS)  # 10
N_GLOBAL_PARAMS = 3  # stem_ln + stem_tropismS + lnf
N_XML_TOTAL = N_XML_PER_LEAF * N_POSITIONS + N_GLOBAL_PARAMS

# Legacy sinusoidal names (kept for backward compat)
DEFORM_AMP_NAMES = ['wave_normal_amp', 'twist_max', 'curl_amp', 'edge_ruffle_amp', 'fold_amp']

# Width profile: 5 leafGeometry x-values per leaf (gradient-optimized)
N_WIDTH_CP = 5


def _grow_and_extract(xml_params, day=60, template_xml=None):
    """CPU-only: run CPlantBox with given structural params, return leaf organ dicts.

    Args:
        xml_params: (78,) array — 7 XML params × 11 positions + stem_ln
        day: simulation days
        template_xml: path to calibrated XML template

    Returns:
        list of leaf organ dicts (numpy), or None on failure
    """
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
                    offset = pos * N_XML_PER_LEAF
                    lmax_val = xml_params[offset + 0]
                    width_val = xml_params[offset + 1]
                    r_val = xml_params[offset + 5]
                    width_taper = xml_params[offset + 6]  # 0=sharp taper, 1=full width

                    collar_len = xml_params[offset + 7]
                    trop_exp = xml_params[offset + 8]
                    init_beta = xml_params[offset + 9]

                    xml_map = {
                        'lmax': lmax_val,
                        'Width_blade': width_val,
                        'theta': xml_params[offset + 2],
                        'tropismS': xml_params[offset + 3],
                        'tropismAge': xml_params[offset + 4],
                        'r': r_val,
                        'areaMax': lmax_val * width_val * 2.0 * 0.73,
                        'collarLength': collar_len,
                        'tropismExponent': trop_exp,
                        'InitBeta': init_beta,
                    }

                    for p in organ:
                        name = p.get('name', '')
                        if name in xml_map:
                            p.set('value', str(xml_map[name]))
                        elif name == 'leafGeometry':
                            geom_str = p.get('value', '')
                            if geom_str:
                                try:
                                    pairs = [x.strip().split() for x in geom_str.split(',')]
                                    new_pairs = []
                                    for phi_s, x_s in pairs:
                                        phi = float(phi_s)
                                        x = float(x_s)
                                        x_new = x + width_taper * (1.0 - x) * 0.5
                                        new_pairs.append(f"{phi} {x_new:.4f}")
                                    p.set('value', ', '.join(new_pairs))
                                except (ValueError, IndexError):
                                    pass
            elif organ.get('type') == 'stem':
                stem_ln = xml_params[-3]
                stem_tropismS = xml_params[-2]
                stem_lnf = int(round(xml_params[-1]))
                for p in organ:
                    name = p.get('name', '')
                    if name == 'ln':
                        p.set('value', str(stem_ln))
                    elif name == 'tropismS':
                        p.set('value', str(stem_tropismS))
                    elif name == 'lnf':
                        p.set('value', str(stem_lnf))

        tmp = tempfile.NamedTemporaryFile(suffix='.xml', delete=False)
        tree.write(tmp.name)
        tmp_path = tmp.name
        tmp.close()

        # Suppress CPlantBox stdout
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


def _eval_worker(args):
    """CPU worker for multiprocessing. No CUDA here."""
    xml_params, day, template_xml = args
    return _grow_and_extract(xml_params, day=day, template_xml=template_xml)


def _optimize_deformations(
    leaf_organs,
    target_pc,
    device='cuda',
    n_steps=100,
    lr=0.05,
    n_cp=DEFAULT_N_CP,
):
    """GPU: optimize spline deformation control points + width profiles.

    Args:
        leaf_organs: list of numpy organ dicts from CPlantBox adapter
        target_pc: (M, 3) torch tensor on GPU
        device: cuda device
        n_steps: Adam steps
        lr: learning rate
        n_cp: number of control points per deformation type

    Returns:
        (best_chamfer_float, best_params_dict)
    """
    from ..diff_lofter.deformations import _interp_linear

    if not leaf_organs:
        return 1e6, {}

    leaf_data = []
    grad_params = []  # all requires_grad tensors

    for organ in leaf_organs:
        skeleton = torch.tensor(organ['skeleton'], dtype=torch.float32, device=device)
        widths_base = torch.tensor(organ['widths'], dtype=torch.float32, device=device)

        if skeleton.shape[0] < 3:
            continue

        tangents = compute_tangents(skeleton)
        binormals = compute_binormal_field(skeleton, tangents)
        arc_fracs = compute_arc_fracs(skeleton)

        # Spline control points for deformations (learnable, init 0)
        cp = make_spline_control_points(n_cp=n_cp, device=device, requires_grad=True)
        for v in cp.values():
            grad_params.append(v)

        # Width profile: N_WIDTH_CP multipliers in [0.5, 1.5], init 1.0
        width_profile = torch.ones(N_WIDTH_CP, device=device, dtype=torch.float32,
                                   requires_grad=True)
        grad_params.append(width_profile)

        leaf_data.append({
            'skeleton': skeleton,
            'widths_base': widths_base,
            'tangents': tangents,
            'binormals': binormals,
            'arc_fracs': arc_fracs,
            'cp': cp,
            'width_profile': width_profile,
            'ramp_onset': organ.get('ramp_onset', 0.15),
        })

    if not leaf_data or not grad_params:
        return 1e6, {}

    optimizer = torch.optim.Adam(grad_params, lr=lr)
    best_loss = float('inf')
    best_params = {}

    for _step in range(n_steps):
        optimizer.zero_grad()

        all_verts = []
        for ld in leaf_data:
            deforms = compute_deformations_spline(
                ld['arc_fracs'], ld['cp'], ramp_onset=ld['ramp_onset'],
            )
            # Apply width profile: interpolate multipliers along arc length
            w_mult = _interp_linear(ld['arc_fracs'], ld['width_profile'])
            widths = ld['widths_base'] * w_mult

            verts = loft_leaf(
                ld['skeleton'], widths, deforms,
                ld['tangents'], ld['binormals'], n_cross=7,
            )
            all_verts.append(verts)

        gen_pc = torch.cat(all_verts, dim=0)
        loss = chamfer_distance(gen_pc, target_pc)
        loss.backward()
        optimizer.step()

        loss_val = loss.item()
        if loss_val < best_loss:
            best_loss = loss_val
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


def fit_plant_hybrid(
    target_points,
    stats_path,
    max_evals=30000,
    n_workers=64,
    deform_steps=100,
    deform_lr=0.05,
    sigma0=0.3,
    day=60,
    device='cuda',
    seed=42,
    template_xml=None,
):
    """Fit CPlantBox params to target using hybrid CMA-ES + GPU gradient descent.

    Args:
        target_points: (N, 3) numpy array, target point cloud (cm, centered)
        stats_path: path to maizefield3d_stats.json
        max_evals: max CMA-ES evaluations
        n_workers: CPU workers for parallel CPlantBox growth
        deform_steps: gradient descent steps per candidate
        deform_lr: Adam learning rate for deformations
        sigma0: CMA-ES initial step size
        day: simulation day
        device: CUDA device
        seed: random seed
        template_xml: path to calibrated XML (None = default)

    Returns:
        dict with structural params, deformation params, chamfer scores
    """
    import cma

    with open(stats_path) as f:
        stats = json.load(f)

    per_pos = stats.get('per_position', stats) if isinstance(stats, dict) else stats
    if not isinstance(per_pos, list):
        per_pos = [per_pos[str(i)] for i in range(N_POSITIONS)]

    # Build x0 and bounds for 78 structural params
    x0 = []
    bounds_lo = []
    bounds_hi = []

    for pos in range(N_POSITIONS):
        s = per_pos[pos]
        for name in XML_PARAMS:
            if name == 'width_taper':
                val = 0.0  # default: no taper modification
            elif name == 'collarLength':
                val = 10.0  # default: 10cm straight base
            elif name == 'tropismExponent':
                val = 1.0  # default: uniform curvature
            elif name == 'initBeta':
                val = 0.2  # default initial radial rotation
            else:
                val = float(s.get(name, 1.0))
            x0.append(val)

            if name in ('lmax', 'Width_blade'):
                bounds_lo.append(max(val * 0.3, 1.0))
                bounds_hi.append(val * 2.5)
            elif name == 'theta':
                bounds_lo.append(0.01)
                bounds_hi.append(1.5)
            elif name == 'tropismS':
                bounds_lo.append(0.0005)
                bounds_hi.append(0.1)
            elif name == 'tropismAge':
                bounds_lo.append(1.0)
                bounds_hi.append(max(val * 2.0, 15.0))
            elif name == 'r':
                bounds_lo.append(max(val * 0.3, 0.5))
                bounds_hi.append(val * 3.0)
            elif name == 'width_taper':
                bounds_lo.append(-0.5)
                bounds_hi.append(1.0)
            elif name == 'collarLength':
                bounds_lo.append(0.0)   # no collar (default)
                bounds_hi.append(30.0)  # up to 30cm straight base
            elif name == 'tropismExponent':
                bounds_lo.append(0.5)   # more uniform than default
                bounds_hi.append(5.0)   # very tip-concentrated
            elif name == 'initBeta':
                bounds_lo.append(-3.14)  # full azimuthal range
                bounds_hi.append(3.14)

    # Global params
    x0.append(14.5)  # stem ln
    bounds_lo.append(8.0)
    bounds_hi.append(22.0)

    x0.append(0.002)  # stem tropismS (slight lean)
    bounds_lo.append(0.0)
    bounds_hi.append(0.015)

    x0.append(0.0)  # lnf: internode length function type (0=homogeneous)
    bounds_lo.append(0.0)
    bounds_hi.append(5.0)  # integer types 0-5

    x0 = np.array(x0)
    bounds_lo = np.array(bounds_lo)
    bounds_hi = np.array(bounds_hi)
    x0 = np.clip(x0, bounds_lo * 1.01, bounds_hi * 0.99)

    # Subsample target
    if len(target_points) > 5000:
        idx = np.random.RandomState(seed).choice(len(target_points), 5000, replace=False)
        target_sub = target_points[idx]
    else:
        target_sub = target_points

    # Align target rotation to default CPlantBox plant
    print("Aligning target rotation...", file=sys.stderr)
    init_organs = _grow_and_extract(x0, day=day, template_xml=template_xml)
    if init_organs:
        from dart.coupling.geometry.g1_to_g3 import loft_organs
        ref_mesh = loft_organs(init_organs)
        ref_pts = ref_mesh.vertices
        if len(ref_pts) > 5000:
            ref_pts = ref_pts[np.random.RandomState(seed).choice(len(ref_pts), 5000, replace=False)]
        from ..targets.pointcloud_loader import align_rotation_z
        target_sub, best_angle = align_rotation_z(target_sub, ref_pts, n_angles=72)
        print(f"  Best rotation: {best_angle:.0f} deg", file=sys.stderr)

    target_gpu = torch.tensor(target_sub, dtype=torch.float32, device=device)

    # Initial evaluation
    init_loss, _ = _optimize_deformations(init_organs, target_gpu, device, deform_steps, deform_lr)
    print(f"Initial Chamfer: {init_loss:.2f}", file=sys.stderr)

    # CMA-ES setup
    opts = cma.CMAOptions()
    opts['maxfevals'] = max_evals
    opts['seed'] = seed
    opts['bounds'] = [bounds_lo.tolist(), bounds_hi.tolist()]
    opts['verbose'] = -1
    opts['tolfun'] = 0.1

    es = cma.CMAEvolutionStrategy(x0, sigma0, opts)

    n_workers = min(n_workers, mp.cpu_count())
    n_grad_dims = (len(SPLINE_DEFORM_NAMES) * DEFAULT_N_CP + N_WIDTH_CP) * N_POSITIONS
    print(f"Hybrid: {N_XML_TOTAL} CMA-ES dims + {n_grad_dims} grad dims, "
          f"{max_evals} max evals, {n_workers} CPU workers", file=sys.stderr)

    counter = 0
    best_deforms = {}

    with mp.Pool(n_workers) as pool:
        while not es.stop():
            solutions = es.ask()

            # CPU parallel: grow all candidates
            args = [(x, day, template_xml) for x in solutions]
            all_organs = pool.map(_eval_worker, args)

            # GPU sequential: optimize deformations for each
            fitnesses = []
            for organs in all_organs:
                if organs is None:
                    fitnesses.append(1e6)
                    continue
                chamfer, deforms = _optimize_deformations(
                    organs, target_gpu, device, deform_steps, deform_lr
                )
                fitnesses.append(chamfer)
                if chamfer <= es.result.fbest if es.result.fbest is not None else True:
                    best_deforms = deforms

            es.tell(solutions, fitnesses)
            counter += len(solutions)

            if counter % 100 < len(solutions):
                print(f"  eval {counter}: best={es.result.fbest:.2f}", file=sys.stderr)

    res = es.result
    print(f"Final Chamfer: {res.fbest:.2f} after {counter} evals", file=sys.stderr)

    return {
        'xml_params': res.xbest.tolist(),
        'xml_param_names': XML_PARAMS + ['stem_ln', 'stem_tropismS', 'lnf'],
        'deform_params': best_deforms,
        'deform_param_names': list(SPLINE_DEFORM_NAMES) + ['width_profile'],
        'best_loss': float(res.fbest),
        'initial_loss': float(init_loss),
        'n_evals': counter,
    }
