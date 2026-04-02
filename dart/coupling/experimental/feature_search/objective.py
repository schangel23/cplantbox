"""Optuna objective function for multi-species geometry feature search.

Each trial:
1. Optuna picks which features are active (categorical on/off) — SHARED across species
2. A random reference plant is picked (could be maize or wheat)
3. Species-appropriate structural params are suggested + plant is grown
4. GPU/CPU gradient descent optimizes deformations + active extended features
5. Returns Chamfer distance

The feature selection is species-agnostic — features that help BOTH species
are universal and worth implementing in C++.
"""

import json
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
from ..fitting.species_config import SpeciesConfig, get_species
from .catalog import FEATURE_CATALOG, SEARCH_FEATURE_NAMES


# Gradient descent hyperparams
N_WIDTH_CP = 5
DEFORM_CP_CLAMP = 1.0
WIDTH_PROFILE_MIN = 0.3
WIDTH_PROFILE_MAX = 1.8
REG_WEIGHT = 0.005
DEFAULT_DEFORM_STEPS = 15
DEFAULT_DEFORM_LR = 0.05


def suggest_active_features(trial) -> set[str]:
    """Suggest which extended features are active for this trial.

    This is SHARED across species — same feature selection regardless of
    whether the reference is maize or wheat.
    """
    active = set()
    for name in SEARCH_FEATURE_NAMES:
        if trial.suggest_categorical(f'feat_{name}', [True, False]):
            active.add(name)
    return active


def suggest_species_params(trial, species_cfg: SpeciesConfig, per_pos_stats: list) -> dict:
    """Suggest structural params using species-specific bounds.

    Args:
        trial: Optuna trial.
        species_cfg: SpeciesConfig for the target species.
        per_pos_stats: Per-position stats list for this species.

    Returns:
        Dict suitable for _grow_and_extract_species().
    """
    params = {}
    n_pos = species_cfg.n_positions

    for pos in range(n_pos):
        s = per_pos_stats[pos] if pos < len(per_pos_stats) else {}
        prefix = f'l{pos}_'

        _, lo, hi = species_cfg.leaf_bounds(s)

        # Suggest each param within species-appropriate bounds
        params[f'leaf_{pos}'] = {
            'lmax': trial.suggest_float(prefix + 'lmax', float(lo[0]), float(hi[0])),
            'Width_blade': trial.suggest_float(prefix + 'Wbl', float(lo[1]), float(hi[1])),
            'theta': trial.suggest_float(prefix + 'theta', float(lo[2]), float(hi[2])),
            'tropismS': trial.suggest_float(prefix + 'tropS', float(lo[3]), float(hi[3]), log=True),
            'tropismAge': trial.suggest_float(prefix + 'tropAge', float(lo[4]), float(hi[4])),
            'r': trial.suggest_float(prefix + 'r', float(lo[5]), float(hi[5])),
            'collarLength': trial.suggest_float(prefix + 'collar', float(lo[6]), float(hi[6])),
            'initBeta': trial.suggest_float(prefix + 'iBeta', float(lo[7]), float(hi[7])),
        }

    params['stem_ln'] = trial.suggest_float(
        'stem_ln', species_cfg.stem_ln_bounds[0], species_cfg.stem_ln_bounds[1]
    )
    params['stem_tropismS'] = trial.suggest_float(
        'stem_tropS', species_cfg.stem_tropismS_bounds[0], species_cfg.stem_tropismS_bounds[1]
    )
    return params


def _grow_and_extract_species(params_dict, species_cfg: SpeciesConfig, day=60):
    """Grow a plant using species-appropriate XML template.

    Adapts _grow_and_extract to work with any species config.
    """
    import os
    import sys
    import tempfile
    import xml.etree.ElementTree as ET

    template_xml = species_cfg.template_xml
    if template_xml is None:
        from dart.coupling.config import DATA_DIR
        template_xml = str(DATA_DIR / f"{species_cfg.name}_calibrated.xml")

    n_pos = species_cfg.n_positions
    sub_offset = species_cfg.subtype_offset
    tmp_path = None

    try:
        tree = ET.parse(template_xml)
        root = tree.getroot()

        for organ in root.iter('organ'):
            if organ.get('type') == 'leaf':
                sub = int(organ.get('subType', '0'))
                pos = sub - sub_offset
                if 0 <= pos < n_pos:
                    p = params_dict.get(f'leaf_{pos}')
                    if p is None:
                        continue
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
        print(f"  CPlantBox failed ({species_cfg.name}): {e}", file=sys.stderr)
        return None
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


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
        leaf_organs: List of organ dicts from growth.
        target_pc: (M, 3) tensor of reference points.
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

    for _ in range(n_steps):
        optimizer.zero_grad()
        all_verts = []
        reg = torch.tensor(0.0, device=device)

        for ld in leaf_data:
            deforms = compute_deformations_spline(ld['arc_fracs'], ld['cp'])
            w_mult = _interp_linear(ld['arc_fracs'], ld['width_profile'])
            widths = ld['widths_base'] * w_mult

            ext_deforms = None
            if ld['ext_cp']:
                ext_deforms = compute_extended_deformations(
                    ld['arc_fracs'], ld['ext_cp']
                )

            verts = loft_leaf(
                ld['skeleton'], widths, deforms,
                ld['tangents'], ld['binormals'],
                n_cross=7,
                extended_deformations=ext_deforms,
            )
            all_verts.append(verts)

            for t in ld['cp'].values():
                reg = reg + REG_WEIGHT * (t ** 2).sum()
            for t in ld['ext_cp'].values():
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
                for name, t in ld['ext_cp'].items():
                    spec = FEATURE_CATALOG[name]
                    t.clamp_(spec['bounds'][0], spec['bounds'][1])

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


# --- Species data registry ---

class SpeciesData:
    """Holds per-species stats + config for the objective function."""

    def __init__(self, species_name: str, stats_path: str, day: int = 60):
        self.cfg = get_species(species_name)
        self.day = day
        with open(stats_path) as f:
            raw = json.load(f)
        per_pos = raw.get('per_position', raw) if isinstance(raw, dict) else raw
        if not isinstance(per_pos, list):
            per_pos = [per_pos[str(i)] for i in range(self.cfg.n_positions)]
        self.per_pos_stats = per_pos


# --- Objective factories ---

def make_objective(
    reference_plants,
    species_data: dict[str, SpeciesData],
    device='cuda',
    deform_steps=DEFAULT_DEFORM_STEPS,
    deform_lr=DEFAULT_DEFORM_LR,
    refs_per_trial=1,
    max_points=2000,
):
    """Create a species-aware Optuna objective for feature search.

    Feature selection is SHARED across species. Structural params adapt to
    whichever species the randomly sampled reference belongs to.

    Args:
        reference_plants: List of dicts with 'points' (np.ndarray),
            'name' (str), and 'species' (str, e.g., 'maize' or 'wheat').
        species_data: Dict mapping species name to SpeciesData.
        device: Torch device.
        deform_steps: Gradient descent steps per trial.
        deform_lr: Adam learning rate.
        refs_per_trial: References per trial (default 1).
        max_points: Max points per reference for Chamfer.

    Returns:
        Callable objective(trial) -> float.
    """
    ref_gpu = []
    for rp in reference_plants:
        pts = rp['points']
        if len(pts) > max_points:
            idx = np.random.RandomState(42).choice(len(pts), max_points, replace=False)
            pts = pts[idx]
        ref_gpu.append({
            'name': rp['name'],
            'species': rp.get('species', 'maize'),
            'points': torch.tensor(pts, dtype=torch.float32, device=device),
        })

    rng = np.random.RandomState(42)

    def objective(trial):
        # 1. Feature selection — SHARED across species
        active_features = suggest_active_features(trial)

        # 2. Pick random reference(s)
        indices = rng.choice(len(ref_gpu), size=min(refs_per_trial, len(ref_gpu)), replace=False)

        chamfers = []
        for idx in indices:
            ref = ref_gpu[idx]
            species = ref['species']
            sd = species_data[species]

            # 3. Species-specific structural params
            params_dict = suggest_species_params(trial, sd.cfg, sd.per_pos_stats)

            # 4. Grow with species-appropriate XML
            organs = _grow_and_extract_species(params_dict, sd.cfg, day=sd.day)
            if organs is None:
                chamfers.append(1e6)
                continue

            # 5. Optimize deformations + features against this reference
            chamfer, _ = _optimize_with_features(
                organs, ref['points'], active_features,
                device=device, n_steps=deform_steps, lr=deform_lr,
            )
            chamfers.append(chamfer)

        return float(np.mean(chamfers))

    return objective


def make_cpu_objective(
    reference_plants,
    species_data: dict[str, SpeciesData],
    deform_steps=DEFAULT_DEFORM_STEPS,
    deform_lr=DEFAULT_DEFORM_LR,
    refs_per_trial=1,
    max_points=2000,
):
    """CPU-only species-aware objective for 128+ parallel workers.

    Same as make_objective but uses CPU tensors (no GPU contention).
    """
    ref_data = []
    for rp in reference_plants:
        pts = rp['points']
        if len(pts) > max_points:
            idx = np.random.RandomState(42).choice(len(pts), max_points, replace=False)
            pts = pts[idx]
        ref_data.append({
            'name': rp['name'],
            'species': rp.get('species', 'maize'),
            'points': pts,
        })

    rng = np.random.RandomState(42)

    def objective(trial):
        active_features = suggest_active_features(trial)

        indices = rng.choice(len(ref_data), size=min(refs_per_trial, len(ref_data)), replace=False)

        chamfers = []
        for idx in indices:
            ref = ref_data[idx]
            species = ref['species']
            sd = species_data[species]

            params_dict = suggest_species_params(trial, sd.cfg, sd.per_pos_stats)
            organs = _grow_and_extract_species(params_dict, sd.cfg, day=sd.day)
            if organs is None:
                chamfers.append(1e6)
                continue

            target_cpu = torch.tensor(ref['points'], dtype=torch.float32, device='cpu')
            chamfer, _ = _optimize_with_features(
                organs, target_cpu, active_features,
                device='cpu', n_steps=deform_steps, lr=deform_lr,
            )
            chamfers.append(chamfer)

        return float(np.mean(chamfers))

    return objective
