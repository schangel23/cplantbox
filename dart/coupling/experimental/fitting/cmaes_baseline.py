"""CMA-ES black-box fitting of CPlantBox params to target point clouds.

Optimizes the real CPlantBox + numpy lofter directly (no surrogate).
Reduced parameter space: 4 key visual params per leaf position
(lmax, Width_blade, theta, tropismS) + stem ln = 45 dimensions.
"""

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

N_POSITIONS = 11
# Only optimize params that most affect visual shape
CMAES_PARAMS = ['lmax', 'Width_blade', 'theta', 'tropismS']
N_CMAES_PER_LEAF = len(CMAES_PARAMS)  # 4
N_CMAES_TOTAL = N_CMAES_PER_LEAF * N_POSITIONS + 1  # 45


def _evaluate_cplantbox(param_vector, target_pc, day=60, _counter=None):
    """Run CPlantBox with given params, loft mesh, compute Chamfer distance."""
    import xml.etree.ElementTree as ET
    from scipy.spatial import cKDTree

    tmp_path = None
    try:
        from dart.coupling.config import DATA_DIR
        template = DATA_DIR / "maize_calibrated.xml"
        tree = ET.parse(template)
        root = tree.getroot()

        for organ in root.iter('organ'):
            if organ.get('type') == 'leaf':
                sub = int(organ.get('subType', '0'))
                pos = sub - 2
                if 0 <= pos < N_POSITIONS:
                    offset = pos * N_CMAES_PER_LEAF
                    param_map = {
                        'lmax': param_vector[offset + 0],
                        'Width_blade': param_vector[offset + 1],
                        'theta': param_vector[offset + 2],
                        'tropismS': param_vector[offset + 3],
                    }
                    # Also update areaMax to be consistent with new width
                    lmax_val = param_vector[offset + 0]
                    width_val = param_vector[offset + 1]
                    param_map['areaMax'] = lmax_val * width_val * 2.0 * 0.73

                    for p in organ:
                        name = p.get('name', '')
                        if name in param_map:
                            p.set('value', str(param_map[name]))
            elif organ.get('type') == 'stem':
                for p in organ:
                    if p.get('name') == 'ln':
                        p.set('value', str(param_vector[-1]))

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
            from dart.coupling.geometry.g1_to_g3 import loft_organs

            plant = grow_plant(tmp_path, simulation_time=day)
            organs = extract_organs_for_lofter(plant, skip_roots=True)
        finally:
            sys.stdout.close()
            sys.stdout = old_stdout

        leaf_organs = [o for o in organs if o['type'] == 'leaf']
        if not leaf_organs:
            return 1e6

        mesh = loft_organs(leaf_organs)
        verts = mesh.vertices

        if len(verts) > 10000:
            idx = np.random.choice(len(verts), 10000, replace=False)
            verts = verts[idx]

        tree_gen = cKDTree(verts)
        tree_tgt = cKDTree(target_pc)
        d1, _ = tree_tgt.query(verts)
        d2, _ = tree_gen.query(target_pc)
        chamfer = float(np.mean(d1) + np.mean(d2)) / 2.0

        if _counter is not None:
            _counter[0] += 1
            if _counter[0] % 20 == 0:
                print(f"  eval {_counter[0]}: chamfer={chamfer:.2f}", file=sys.stderr)

        return chamfer

    except Exception as e:
        print(f"  CPlantBox eval failed: {e}", file=sys.stderr)
        return 1e6
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


def fit_plant_cmaes(
    target_points: np.ndarray,
    stats_path: str,
    max_evals: int = 2000,
    sigma0: float = 0.3,
    day: int = 60,
    seed: int = 42,
) -> dict:
    """Fit CPlantBox params to a target using CMA-ES.

    Optimizes 45 dimensions: 4 key params per leaf position + stem ln.

    Args:
        target_points: (N, 3) target point cloud (cm, centered)
        stats_path: path to maizefield3d_stats.json
        max_evals: maximum function evaluations
        sigma0: initial step size
        day: simulation day
        seed: random seed

    Returns:
        dict with 'params', 'best_loss', 'n_evals', 'param_names'
    """
    import cma

    with open(stats_path) as f:
        stats = json.load(f)

    x0 = []
    bounds_lo = []
    bounds_hi = []

    per_pos = stats.get('per_position', stats) if isinstance(stats, dict) else stats
    if not isinstance(per_pos, list):
        per_pos = [per_pos[str(i)] for i in range(N_POSITIONS)]

    for pos in range(N_POSITIONS):
        s = per_pos[pos]
        for name in CMAES_PARAMS:
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

    # Stem ln
    x0.append(14.5)
    bounds_lo.append(8.0)
    bounds_hi.append(22.0)

    x0 = np.array(x0)
    bounds_lo = np.array(bounds_lo)
    bounds_hi = np.array(bounds_hi)

    # Clamp x0 to be strictly within bounds
    x0 = np.clip(x0, bounds_lo * 1.01, bounds_hi * 0.99)

    # Subsample target
    if len(target_points) > 5000:
        idx = np.random.RandomState(seed).choice(len(target_points), 5000, replace=False)
        target_sub = target_points[idx]
    else:
        target_sub = target_points

    counter = [0]

    def objective(x):
        return _evaluate_cplantbox(x, target_sub, day=day, _counter=counter)

    # Initial evaluation
    init_loss = _evaluate_cplantbox(x0, target_sub, day=day)
    print(f"Initial Chamfer: {init_loss:.2f}", file=sys.stderr)

    opts = cma.CMAOptions()
    opts['maxfevals'] = max_evals
    opts['seed'] = seed
    opts['bounds'] = [bounds_lo, bounds_hi]
    opts['verbose'] = -1
    opts['tolfun'] = 0.1  # stop if loss change < 0.1

    es = cma.CMAEvolutionStrategy(x0, sigma0, opts)
    es.optimize(objective)

    res = es.result
    print(f"Final Chamfer: {res.fbest:.2f} after {counter[0]} evals", file=sys.stderr)

    return {
        'params': res.xbest.tolist(),
        'param_names': CMAES_PARAMS + ['stem_ln'],
        'best_loss': float(res.fbest),
        'initial_loss': float(init_loss),
        'n_evals': counter[0],
    }
