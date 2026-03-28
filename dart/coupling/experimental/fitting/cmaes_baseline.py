"""CMA-ES black-box baseline for parameter fitting (CPU, uses real CPlantBox)."""

import json
import sys
from pathlib import Path

import numpy as np

from .optimize import N_PARAMS, N_PER_LEAF, N_POSITIONS, N_STRUCTURAL, STRUCTURAL_NAMES, DEFORM_NAMES


def _evaluate_cplantbox(param_vector, target_pc, day=60):
    """Run CPlantBox with given params, loft mesh, compute Chamfer distance.

    This uses the real CPlantBox + numpy lofter (not the differentiable surrogate).
    Returns Chamfer distance as a float.
    """
    import tempfile
    import xml.etree.ElementTree as ET
    from dart.coupling.growth.grow import grow_plant
    from dart.coupling.geometry.cplantbox_adapter import extract_organs_for_lofter
    from dart.coupling.geometry.g1_to_g3 import loft_organs
    from dart.coupling.config import DATA_DIR

    tmp_path = None
    try:
        # Create modified XML from param vector
        template = DATA_DIR / "maize_calibrated.xml"
        tree = ET.parse(template)
        root = tree.getroot()

        # Modify leaf subtypes (subType 2..12 = positions 0..10)
        for organ in root.iter('organ'):
            if organ.get('type') == 'leaf':
                sub = int(organ.get('subType', '0'))
                pos = sub - 2
                if 0 <= pos < N_POSITIONS:
                    offset = pos * N_PER_LEAF
                    for param in organ:
                        name = param.get('name', '')
                        if name == 'lmax':
                            param.set('value', str(param_vector[offset + 0]))
                        elif name == 'Width_blade':
                            param.set('value', str(param_vector[offset + 1]))
                        elif name == 'theta':
                            param.set('value', str(param_vector[offset + 2]))
                        elif name == 'tropismS':
                            param.set('value', str(param_vector[offset + 3]))
                        elif name == 'tropismAge':
                            param.set('value', str(param_vector[offset + 4]))
                        elif name == 'r':
                            param.set('value', str(param_vector[offset + 5]))
                        elif name == 'areaMax':
                            param.set('value', str(param_vector[offset + 6]))
            elif organ.get('type') == 'stem':
                for param in organ:
                    if param.get('name') == 'ln':
                        param.set('value', str(param_vector[-1]))

        tmp = tempfile.NamedTemporaryFile(suffix='.xml', delete=False)
        tree.write(tmp.name)
        tmp_path = tmp.name
        tmp.close()

        # Grow plant using modified XML
        plant = grow_plant(tmp_path, simulation_time=day)
        organs = extract_organs_for_lofter(plant, skip_roots=True)

        # Filter to leaves only
        leaf_organs = [o for o in organs if o['type'] == 'leaf']
        if not leaf_organs:
            return 1e6

        mesh = loft_organs(leaf_organs)
        verts = mesh.vertices

        # Subsample to 10k points for fair comparison
        if len(verts) > 10000:
            idx = np.random.choice(len(verts), 10000, replace=False)
            verts = verts[idx]

        # Chamfer distance (numpy, CPU)
        from scipy.spatial import cKDTree
        tree1 = cKDTree(verts)
        tree2 = cKDTree(target_pc)
        d1, _ = tree2.query(verts)
        d2, _ = tree1.query(target_pc)
        return float(np.mean(d1) + np.mean(d2)) / 2.0

    except Exception as e:
        print(f"  CPlantBox eval failed: {e}", file=sys.stderr)
        return 1e6
    finally:
        import os
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


def fit_plant_cmaes(
    target_points: np.ndarray,
    stats_path: str,
    max_evals: int = 1000,
    sigma0: float = 0.3,
    day: int = 60,
    seed: int = 42,
) -> dict:
    """Fit CPlantBox params to a target using CMA-ES (black-box, CPU).

    Args:
        target_points: (N, 3) target point cloud
        stats_path: path to maizefield3d_stats.json
        max_evals: maximum function evaluations
        sigma0: initial step size (relative to param range)
        day: simulation day
        seed: random seed

    Returns:
        dict with 'params', 'best_loss', 'n_evals'
    """
    import cma

    # Load priors as initial guess
    with open(stats_path) as f:
        stats = json.load(f)

    x0 = []
    bounds_lo = []
    bounds_hi = []
    for pos in range(N_POSITIONS):
        s = stats[str(pos)] if str(pos) in stats else stats[pos]
        for name in STRUCTURAL_NAMES:
            val = float(s.get(name, 1.0))
            x0.append(val)
            bounds_lo.append(val * 0.5)
            bounds_hi.append(val * 1.5)
        for name in DEFORM_NAMES:
            x0.append(0.0)
            bounds_lo.append(-2.0)
            bounds_hi.append(2.0)

    # Stem ln
    x0.append(14.5)
    bounds_lo.append(10.0)
    bounds_hi.append(20.0)

    x0 = np.array(x0)

    # Subsample target for speed
    if len(target_points) > 5000:
        idx = np.random.RandomState(seed).choice(len(target_points), 5000, replace=False)
        target_sub = target_points[idx]
    else:
        target_sub = target_points

    n_evals = [0]

    def objective(x):
        n_evals[0] += 1
        return _evaluate_cplantbox(x, target_sub, day=day)

    opts = cma.CMAOptions()
    opts['maxfevals'] = max_evals
    opts['seed'] = seed
    opts['bounds'] = [bounds_lo, bounds_hi]
    opts['verbose'] = -1  # quiet

    es = cma.CMAEvolutionStrategy(x0, sigma0, opts)
    es.optimize(objective)

    result = es.result
    return {
        'params': result.xbest.tolist(),
        'best_loss': float(result.fbest),
        'n_evals': n_evals[0],
    }
