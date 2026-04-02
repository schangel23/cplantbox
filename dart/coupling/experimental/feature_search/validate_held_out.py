"""Validate geometry features on held-out MaizeField3D plants.

For each held-out plant:
  1. Fit structural params only (Optuna/CMA-ES) → Chamfer_struct
  2. Fit structural + default geometry features (from XML) → Chamfer_defaults
  3. Fit structural + per-plant gradient-descended features → Chamfer_fitted

Targets:
  - Chamfer_defaults < 8 cm  (geometry features improve over struct-only)
  - Chamfer_fitted   < 3 cm  (per-plant fitting pushes to high accuracy)

Usage (on server):
    source /media/data/Lukas/CPlantBox/cpbenv/bin/activate
    cd /media/data/Lukas/CPlantBox
    python -m dart.coupling.experimental.feature_search.validate_held_out \
        --stl-dir /path/to/MaizeField3d/stl/ \
        --stats /path/to/maizefield3d_stats.json \
        --output /path/to/validation_results.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from .objective import (
    _grow_and_extract_species,
    _optimize_with_features,
    SpeciesData,
    suggest_species_params,
    DEFAULT_DEFORM_STEPS,
    DEFAULT_DEFORM_LR,
)

# Reference set — EXCLUDE these from validation
REFERENCE_IDS = {1, 10, 50, 100, 150, 200, 250, 300, 400, 500}

# Held-out validation set — evenly spaced, not in reference set
HELD_OUT_IDS = [25, 75, 125, 175, 225, 275, 350, 425, 475, 510]

# Features we're validating
TARGET_FEATURES = {'out_of_plane_curv', 'asymmetry', 'edge_curl', 'cross_section_profile'}


def load_scan_pointcloud(stl_path, n_points=5000):
    """Load STL as point cloud, convert to cm, center at base."""
    import trimesh
    mesh = trimesh.load(stl_path)
    pts = np.array(mesh.vertices)
    if pts.max() < 10:  # meters → cm
        pts *= 100.0
    pts[:, 2] -= pts[:, 2].min()
    if len(pts) > n_points:
        idx = np.random.RandomState(42).choice(len(pts), n_points, replace=False)
        pts = pts[idx]
    return pts


def fit_structural_only(target_pc, species_data, device='cpu', n_trials=200):
    """Fit structural params with NO geometry features. Returns best Chamfer."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    sd = species_data
    target_t = torch.tensor(target_pc, dtype=torch.float32, device=device)

    def objective(trial):
        params = suggest_species_params(trial, sd.cfg, sd.per_pos_stats)
        organs = _grow_and_extract_species(params, sd.cfg, day=sd.day)
        if organs is None:
            return 1e6
        chamfer, _ = _optimize_with_features(
            organs, target_t, set(),  # no extended features
            device=device, n_steps=DEFAULT_DEFORM_STEPS, lr=DEFAULT_DEFORM_LR,
        )
        return chamfer

    study = optuna.create_study(direction='minimize',
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_value, study.best_params


def fit_with_defaults(target_pc, best_struct_params, species_data, device='cpu'):
    """Use best structural params + default geometry features (from XML). Returns Chamfer."""
    sd = species_data
    organs = _grow_and_extract_species(best_struct_params, sd.cfg, day=sd.day)
    if organs is None:
        return 1e6

    target_t = torch.tensor(target_pc, dtype=torch.float32, device=device)
    # Use all 4 target features as active
    chamfer, _ = _optimize_with_features(
        organs, target_t, TARGET_FEATURES,
        device=device, n_steps=DEFAULT_DEFORM_STEPS, lr=DEFAULT_DEFORM_LR,
    )
    return chamfer


def fit_with_per_plant(target_pc, species_data, device='cpu', n_trials=200):
    """Full per-plant fitting: structural + geometry features. Returns best Chamfer."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    sd = species_data
    target_t = torch.tensor(target_pc, dtype=torch.float32, device=device)

    def objective(trial):
        params = suggest_species_params(trial, sd.cfg, sd.per_pos_stats)
        organs = _grow_and_extract_species(params, sd.cfg, day=sd.day)
        if organs is None:
            return 1e6
        chamfer, _ = _optimize_with_features(
            organs, target_t, TARGET_FEATURES,
            device=device, n_steps=30, lr=DEFAULT_DEFORM_LR,  # more steps for quality
        )
        return chamfer

    study = optuna.create_study(direction='minimize',
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_value


def main():
    parser = argparse.ArgumentParser(description='Validate geometry features on held-out plants')
    parser.add_argument('--stl-dir', required=True, help='Directory with MaizeField3D STL files')
    parser.add_argument('--stats', required=True, help='Path to maizefield3d_stats.json')
    parser.add_argument('--output', default='validation_results.json', help='Output JSON path')
    parser.add_argument('--device', default='auto', help='cuda or cpu')
    parser.add_argument('--n-trials', type=int, default=200, help='Optuna trials per plant')
    parser.add_argument('--plants', type=str, default=None,
                        help='Comma-separated plant IDs (default: all held-out)')
    args = parser.parse_args()

    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    plant_ids = HELD_OUT_IDS
    if args.plants:
        plant_ids = [int(x) for x in args.plants.split(',')]

    sd = SpeciesData('maize', args.stats, day=60)

    results = {}
    print(f'{"Plant":>6s}  {"Struct-only":>12s}  {"With defaults":>14s}  {"Per-plant":>10s}')
    print('-' * 48)

    for pid in plant_ids:
        stl_path = f'{args.stl_dir}/{pid:04d}.stl'
        if not Path(stl_path).exists():
            print(f'{pid:>6d}  MISSING', file=sys.stderr)
            continue

        scan = load_scan_pointcloud(stl_path)

        # 1. Structural-only
        cd_struct, best_params = fit_structural_only(
            scan, sd, device=device, n_trials=args.n_trials)

        # 2. With default features
        cd_defaults = fit_with_defaults(scan, best_params, sd, device=device)

        # 3. Full per-plant fitting
        cd_fitted = fit_with_per_plant(scan, sd, device=device, n_trials=args.n_trials)

        results[pid] = {
            'struct_only': cd_struct,
            'with_defaults': cd_defaults,
            'per_plant_fitted': cd_fitted,
        }
        print(f'{pid:>6d}  {cd_struct:>12.2f}  {cd_defaults:>14.2f}  {cd_fitted:>10.2f}')

    # Summary
    if results:
        vals = list(results.values())
        mean_struct = np.mean([v['struct_only'] for v in vals])
        mean_defaults = np.mean([v['with_defaults'] for v in vals])
        mean_fitted = np.mean([v['per_plant_fitted'] for v in vals])
        print('-' * 48)
        print(f'{"Mean":>6s}  {mean_struct:>12.2f}  {mean_defaults:>14.2f}  {mean_fitted:>10.2f}')
        print(f'\nTargets: defaults < 8 cm ({mean_defaults:.2f}), '
              f'fitted < 3 cm ({mean_fitted:.2f})')

        results['summary'] = {
            'mean_struct_only': mean_struct,
            'mean_with_defaults': mean_defaults,
            'mean_per_plant_fitted': mean_fitted,
            'n_plants': len(vals),
            'held_out_ids': plant_ids,
        }

    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2, default=float)
    print(f'\nResults saved to {args.output}')


if __name__ == '__main__':
    main()
