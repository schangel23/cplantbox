"""Sequential per-leaf sep-CMA-ES optimizer.

Fits stem first (3D), then each leaf independently (11D).
CMA-ES at 11D converges in ~500 evals — fast and reliable.

Uses diagonal CMA-ES (CMA_diagonal=True) since leaf params are mostly
independent. No clamping or regularization on deformations.
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
    _interp_linear,
    SPLINE_DEFORM_NAMES,
    DEFAULT_N_CP,
)
from ..diff_lofter.frames import compute_binormal_field, compute_tangents
from ..diff_lofter.lofter import compute_arc_fracs, loft_leaf, loft_stem
from ..losses.chamfer import chamfer_distance

from .species_config import SpeciesConfig, LEAF_PARAMS, MAIZE

# Curvature spline knot positions
CURVATURE_PHI = [0.0, 0.5, 1.0]


def _grow_single(stem_params, leaf_params_list, day=60, template_xml=None, species=None):
    """Grow CPlantBox with given stem + per-leaf params. Return leaf organs."""
    import xml.etree.ElementTree as ET

    if species is None:
        species = MAIZE

    if template_xml is None:
        if species.template_xml:
            template_xml = species.template_xml
        else:
            from dart.coupling.config import DATA_DIR
            template_xml = str(DATA_DIR / "maize_calibrated.xml")

    tmp_path = None
    try:
        tree = ET.parse(template_xml)
        root = tree.getroot()

        # Iterate leaf elements (tag='leaf', not 'organ')
        for leaf_el in root.iter('leaf'):
            sub = int(leaf_el.get('subType', '0'))
            pos = sub - species.subtype_offset
            if 0 <= pos < species.n_positions and pos < len(leaf_params_list):
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

                # Remove old leafCurvature, set params
                to_remove = []
                for elem in leaf_el:
                    name = elem.get('name', '')
                    if name in xml_map:
                        elem.set('value', str(xml_map[name]))
                    elif name == 'leafCurvature':
                        to_remove.append(elem)
                for elem in to_remove:
                    leaf_el.remove(elem)

                # Add curvature spline
                curv = ET.SubElement(leaf_el, 'parameter')
                curv.set('name', 'leafCurvature')
                curv.set('phi', ' '.join(str(v) for v in CURVATURE_PHI))
                curv.set('kappa', f"{p['kappa_base']} {p['kappa_mid']} {p['kappa_tip']}")

        # Iterate stem elements
        for stem_el in root.iter('stem'):
            for elem in stem_el:
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


def _optimize_deformations(organs, target_pc, device='cuda', n_steps=100, lr=0.05):
    """GPU: optimize leaf spline deformations. Stems lofted as fixed cylinders."""
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
            leaf_data.append({
                'skeleton': skeleton, 'widths': widths,
                'tangents': tangents, 'binormals': binormals,
                'arc_fracs': arc_fracs, 'cp': cp,
            })

    if not leaf_data and not stem_data:
        return 1e6, {}

    # Stem vertices (fixed, no optimization)
    stem_verts = []
    for sd in stem_data:
        stem_verts.append(loft_stem(sd['skeleton'], sd['widths'], sd['tangents'], sd['binormals']))

    if not grad_params:
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
            verts = loft_leaf(ld['skeleton'], ld['widths'], deforms,
                              ld['tangents'], ld['binormals'], n_cross=7)
            all_verts.append(verts)

        gen_pc = torch.cat(all_verts, dim=0)
        loss = chamfer_distance(gen_pc, target_pc)
        loss.backward()
        optimizer.step()

        # Clamp to prevent unrealistic deformations
        with torch.no_grad():
            for ld in leaf_data:
                for t in ld['cp'].values():
                    t.clamp_(-1.5, 1.5)

        loss_val = loss.item()
        if loss_val < best_loss:
            best_loss = loss_val
            best_params = {
                i: {
                    'control_points': {
                        name: ld['cp'][name].detach().cpu().tolist()
                        for name in SPLINE_DEFORM_NAMES
                    },
                }
                for i, ld in enumerate(leaf_data)
            }

    return best_loss, best_params


def _default_leaf_params(stats_pos, species=None, position=0):
    """Build default params for one leaf from stats."""
    if species is None:
        species = MAIZE
    return species.default_leaf_params(stats_pos, position=position)


def _leaf_params_from_vec(vec):
    """Convert CMA-ES vector to leaf param dict."""
    return {k: v for k, v in zip(LEAF_PARAMS, vec)}


def _leaf_params_to_vec(params):
    """Convert leaf param dict to CMA-ES vector."""
    return [params[k] for k in LEAF_PARAMS]


def _leaf_bounds(stats_pos, species=None):
    """Return (x0, lo, hi) for one leaf's CMA-ES."""
    if species is None:
        species = MAIZE
    return species.leaf_bounds(stats_pos)


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


def _grow_worker(args):
    """CPU worker: grow plant and return leaf organs. No CUDA."""
    if len(args) == 5:
        stem_params, leaf_params_list, day, template_xml, species = args
    else:
        stem_params, leaf_params_list, day, template_xml = args
        species = MAIZE
    return _grow_single(stem_params, leaf_params_list, day=day,
                        template_xml=template_xml, species=species)


def _eval_batch(solutions, stem_params, leaf_params_list, pos, per_pos,
                target_gpu, device, deform_steps, deform_lr, day, template_xml,
                n_workers, species=None):
    """Evaluate a batch: parallel CPU growth + fast skeleton-only Chamfer.

    Args:
        solutions: list of CMA-ES candidate vectors
        pos: which leaf position is being optimized (-1 for stem)
        species: SpeciesConfig
        Other args: context for growth and evaluation

    Returns:
        list of fitness values
    """
    if species is None:
        species = MAIZE

    # Build per-candidate param lists (CPU)
    grow_args = []
    for x in solutions:
        if pos < 0:
            # Stem fitting
            sp = {'ln': x[0], 'tropismS': x[1], 'lnf': x[2]}
            grow_args.append((sp, leaf_params_list, day, template_xml, species))
        else:
            # Leaf fitting
            lp = list(leaf_params_list)
            lp[pos] = _leaf_params_from_vec(x)
            grow_args.append((stem_params, lp, day, template_xml, species))

    # Parallel CPU growth
    with mp.Pool(min(n_workers, len(solutions))) as pool:
        all_organs = pool.map(_grow_worker, grow_args)

    # Fast skeleton-only Chamfer (no deformation optimization during search)
    fitnesses = []
    for organs in all_organs:
        if organs is None:
            fitnesses.append(1e6)
            continue
        chamfer = _fast_chamfer(organs, target_gpu, device)
        fitnesses.append(chamfer)

    return fitnesses


def fit_plant_sequential(
    target_points,
    stats_path,
    stem_evals=200,
    leaf_evals=500,
    deform_steps=100,
    deform_lr=0.05,
    day=60,
    device='cuda',
    seed=42,
    template_xml=None,
    n_workers=16,
    species=None,
):
    """Sequential fitting: stem → each leaf independently.

    CPlantBox growth is parallelized across CPU workers.
    GPU deformation optimization runs sequentially (CUDA not fork-safe).

    Args:
        target_points: (N, 3) numpy array
        stats_path: path to maizefield3d_stats.json
        stem_evals: CMA-ES evals for stem (3D)
        leaf_evals: CMA-ES evals per leaf (11D)
        deform_steps: gradient steps for deformations
        deform_lr: Adam LR
        day: simulation day
        device: torch device
        seed: random seed
        template_xml: calibrated XML path
        n_workers: CPU workers for parallel CPlantBox growth

    Returns:
        dict with fitted params, per-leaf losses, total loss
    """
    import cma

    if species is None:
        species = MAIZE

    n_workers = min(n_workers, mp.cpu_count())

    with open(stats_path) as f:
        stats = json.load(f)

    per_pos = stats.get('per_position', stats) if isinstance(stats, dict) else stats
    if not isinstance(per_pos, list):
        per_pos = [per_pos[str(i)] for i in range(species.n_positions)]

    # Subsample target
    if len(target_points) > 5000:
        idx = np.random.RandomState(seed).choice(len(target_points), 5000, replace=False)
        target_sub = target_points[idx]
    else:
        target_sub = target_points

    # Default params
    leaf_params_list = [_default_leaf_params(per_pos[i], species=species, position=i)
                        for i in range(species.n_positions)]
    stem_params = {'ln': species.stem_ln, 'tropismS': species.stem_tropismS, 'lnf': species.stem_lnf}

    # Rotation alignment
    init_organs = _grow_single(stem_params, leaf_params_list, day=day,
                               template_xml=template_xml, species=species)
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

    # ========== PHASE 1: Fit stem (3D) ==========
    print(f"\n=== PHASE 1: Stem fitting (3D, {stem_evals} evals, {n_workers} workers) ===",
          file=sys.stderr)

    stem_x0 = [species.stem_ln, species.stem_tropismS, species.stem_lnf]
    stem_lo = [species.stem_ln_bounds[0], species.stem_tropismS_bounds[0], species.stem_lnf_bounds[0]]
    stem_hi = [species.stem_ln_bounds[1], species.stem_tropismS_bounds[1], species.stem_lnf_bounds[1]]

    opts = cma.CMAOptions()
    opts['maxfevals'] = stem_evals
    opts['seed'] = seed
    opts['bounds'] = [stem_lo, stem_hi]
    opts['verbose'] = -1
    opts['CMA_diagonal'] = True

    es = cma.CMAEvolutionStrategy(stem_x0, 0.3, opts)
    while not es.stop():
        solutions = es.ask()
        fitnesses = _eval_batch(
            solutions, stem_params, leaf_params_list, pos=-1, per_pos=per_pos,
            target_gpu=target_gpu, device=device,
            deform_steps=deform_steps, deform_lr=deform_lr,
            day=day, template_xml=template_xml, n_workers=n_workers,
            species=species,
        )
        es.tell(solutions, fitnesses)

    best_stem = es.result.xbest
    stem_params = {'ln': best_stem[0], 'tropismS': best_stem[1], 'lnf': best_stem[2]}
    print(f"  Stem: ln={stem_params['ln']:.1f}, tropismS={stem_params['tropismS']:.4f}, "
          f"lnf={stem_params['lnf']:.1f}, chamfer={es.result.fbest:.2f}", file=sys.stderr)

    # ========== PHASE 2: Fit each leaf (11D each) ==========
    print(f"\n=== PHASE 2: Per-leaf fitting (11D x {species.n_positions}, {leaf_evals} evals each, "
          f"{n_workers} workers) ===", file=sys.stderr)

    best_deforms = {}
    per_leaf_losses = []

    for pos in range(species.n_positions):
        x0, lo, hi = _leaf_bounds(per_pos[pos], species=species)

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
            fitnesses = _eval_batch(
                solutions, stem_params, leaf_params_list, pos=pos, per_pos=per_pos,
                target_gpu=target_gpu, device=device,
                deform_steps=deform_steps, deform_lr=deform_lr,
                day=day, template_xml=template_xml, n_workers=n_workers,
                species=species,
            )
            es.tell(solutions, fitnesses)

        best_x = es.result.xbest
        leaf_params_list[pos] = _leaf_params_from_vec(best_x)
        per_leaf_losses.append(float(es.result.fbest))

        print(f"  Leaf {pos} (st={pos+species.subtype_offset}): chamfer={es.result.fbest:.2f} cm  "
              f"lmax={best_x[0]:.1f} Wbl={best_x[1]:.1f} theta={best_x[2]:.2f}",
              file=sys.stderr)

    # ========== PHASE 3: Final evaluation with all best params ==========
    print(f"\n=== PHASE 3: Final evaluation ===", file=sys.stderr)
    final_organs = _grow_single(stem_params, leaf_params_list, day=day,
                                template_xml=template_xml, species=species)
    if final_organs:
        final_chamfer, final_deforms = _optimize_deformations(
            final_organs, target_gpu, device, deform_steps, deform_lr
        )
        best_deforms = final_deforms
        print(f"  Final Chamfer: {final_chamfer:.2f} cm", file=sys.stderr)
    else:
        final_chamfer = 1e6

    return {
        'species': species.name,
        'stem_params': stem_params,
        'leaf_params': [_leaf_params_to_vec(lp) for lp in leaf_params_list],
        'leaf_param_names': LEAF_PARAMS,
        'deform_params': best_deforms,
        'deform_param_names': list(SPLINE_DEFORM_NAMES),
        'per_leaf_losses': per_leaf_losses,
        'final_loss': float(final_chamfer),
        'stem_evals': stem_evals,
        'leaf_evals': leaf_evals,
    }


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python -m dart.coupling.experimental.fitting.sequential_optimizer "
              "<target.stl> <stats.json> [leaf_evals]")
        sys.exit(1)

    from ..targets.pointcloud_loader import load_pointcloud

    target_path = sys.argv[1]
    stats_path = sys.argv[2]
    leaf_evals = int(sys.argv[3]) if len(sys.argv) > 3 else 500

    target_pts, _ = load_pointcloud(target_path, n_points=10000)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    result = fit_plant_sequential(
        target_pts, stats_path, leaf_evals=leaf_evals, device=device,
    )

    out_base = target_path.rsplit('.', 1)[0]
    out_path = f'{out_base}_sequential_fit.json'
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"Saved: {out_path}")
    print(f"Final: {result['final_loss']:.2f} cm")
