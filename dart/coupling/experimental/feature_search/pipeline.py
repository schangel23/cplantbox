"""Main Optuna pipeline for geometry feature discovery.

Usage:
    python -m dart.coupling.experimental.feature_search.pipeline \
        --n-trials 10000 --workers 128 \
        --refs-dir /path/to/references \
        --stats /path/to/maizefield3d_stats.json

Designed for fire-and-forget execution on a multi-core server.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np


def load_reference_plants(refs_config):
    """Load reference plants from a config file or directory.

    Config file format (JSON):
    [
        {"path": "/path/to/plant.stl", "name": "mf3d_0001", "species": "maize"},
        {"path": "/path/to/scan.ply", "name": "pheno4d_m01_0317", "species": "maize"},
        ...
    ]

    Or: pass a directory and all .stl/.ply/.pcd files will be loaded.

    Returns:
        List of dicts with 'points' (np.ndarray) and 'name' (str).
    """
    from ..targets.pointcloud_loader import load_pointcloud

    if os.path.isdir(refs_config):
        paths = sorted(
            p for p in Path(refs_config).glob('*')
            if p.suffix.lower() in ('.stl', '.ply', '.pcd', '.xyz', '.txt')
        )
        plants = []
        for p in paths:
            pts, _ = load_pointcloud(str(p), n_points=10000)
            plants.append({'points': pts, 'name': p.stem})
        return plants

    with open(refs_config) as f:
        config = json.load(f)

    plants = []
    for entry in config:
        pts, _ = load_pointcloud(entry['path'], n_points=10000)
        plants.append({
            'points': pts,
            'name': entry.get('name', Path(entry['path']).stem),
            'species': entry.get('species', 'maize'),
        })
    return plants


def create_study(storage_path, study_name='feature_search', seed=42):
    """Create or load an Optuna study with appropriate backend.

    Uses JournalStorage for multi-worker safety (up to 128+ concurrent writers).
    """
    import optuna
    from optuna.storages import JournalStorage, JournalFileStorage

    journal_path = str(storage_path)
    storage = JournalStorage(JournalFileStorage(journal_path))

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction='minimize',
        sampler=optuna.samplers.TPESampler(
            seed=seed,
            multivariate=True,
            n_startup_trials=50,  # random exploration before TPE kicks in
        ),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=20,
            n_warmup_steps=3,  # prune after evaluating 3 reference plants
        ),
        load_if_exists=True,
    )
    return study


def analyze_results(study, output_dir):
    """Analyze and export feature search results.

    Produces:
    - feature_importance.json: ranked feature importance
    - best_config.json: best trial's feature configuration
    - convergence.png: optimization history plot
    """
    import optuna

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Best trial info
    best = study.best_trial
    best_config = {
        'best_chamfer': best.value,
        'n_trials': len(study.trials),
        'best_trial_number': best.number,
        'active_features': {
            name: best.params.get(f'feat_{name}', False)
            for name in _get_feature_names_from_study(study)
        },
        'structural_params': {
            k: v for k, v in best.params.items()
            if not k.startswith('feat_')
        },
    }
    with open(output_dir / 'best_config.json', 'w') as f:
        json.dump(best_config, f, indent=2)

    # Feature importance via Optuna
    try:
        importances = optuna.importance.get_param_importances(study)
        # Filter to feature activation params
        feature_importances = {
            k.replace('feat_', ''): v
            for k, v in importances.items()
            if k.startswith('feat_')
        }
        # Sort by importance
        ranked = dict(sorted(feature_importances.items(), key=lambda x: -x[1]))
        with open(output_dir / 'feature_importance.json', 'w') as f:
            json.dump(ranked, f, indent=2)
        print("\n=== Feature Importance Ranking ===")
        for name, imp in ranked.items():
            active_in_best = best_config['active_features'].get(name, False)
            marker = " *" if active_in_best else ""
            print(f"  {name:30s}  importance={imp:.4f}{marker}")
    except Exception as e:
        print(f"Feature importance analysis failed: {e}", file=sys.stderr)

    # Feature activation frequency in top-N trials
    completed = [t for t in study.trials
                 if t.state == optuna.trial.TrialState.COMPLETE]
    if completed:
        completed.sort(key=lambda t: t.value)
        top_n = completed[:min(100, len(completed))]
        freq = {}
        for name in _get_feature_names_from_study(study):
            count = sum(1 for t in top_n if t.params.get(f'feat_{name}', False))
            freq[name] = count / len(top_n)
        freq_ranked = dict(sorted(freq.items(), key=lambda x: -x[1]))
        with open(output_dir / 'feature_frequency_top100.json', 'w') as f:
            json.dump(freq_ranked, f, indent=2)
        print("\n=== Feature Frequency in Top-100 Trials ===")
        for name, f_val in freq_ranked.items():
            print(f"  {name:30s}  {f_val:.0%} active")

    # Convergence plot
    try:
        fig = optuna.visualization.plot_optimization_history(study)
        fig.write_image(str(output_dir / 'convergence.png'))
        print(f"\nConvergence plot: {output_dir / 'convergence.png'}")
    except Exception:
        pass  # plotly may not be available

    # Summary
    print(f"\nBest Chamfer: {best.value:.3f} cm")
    print(f"Total trials: {len(study.trials)}")
    print(f"Results saved to: {output_dir}")


def _get_feature_names_from_study(study):
    """Extract feature names from study params."""
    from .catalog import SEARCH_FEATURE_NAMES
    return SEARCH_FEATURE_NAMES


def main():
    parser = argparse.ArgumentParser(
        description='Optuna feature discovery pipeline for CPlantBox geometry'
    )
    parser.add_argument('--n-trials', type=int, default=10000,
                        help='Number of Optuna trials')
    parser.add_argument('--workers', type=int, default=1,
                        help='Number of parallel workers (use 1 for testing)')
    parser.add_argument('--refs', required=True,
                        help='Path to references config JSON or directory of STL/PLY files')
    parser.add_argument('--stats', required=True, nargs='+',
                        help='Species stats: "maize:/path/to/stats.json" "wheat:/path/to/stats.json"')
    parser.add_argument('--output', default='feature_search_results',
                        help='Output directory for results')
    parser.add_argument('--storage', default='feature_search_journal.log',
                        help='Optuna JournalStorage file path')
    parser.add_argument('--study-name', default='feature_search',
                        help='Optuna study name')
    parser.add_argument('--day', type=int, default=60,
                        help='CPlantBox simulation day')
    parser.add_argument('--device', default='auto',
                        help='Torch device: cuda, cpu, or auto')
    parser.add_argument('--deform-steps', type=int, default=15,
                        help='Gradient descent steps per trial (default 15, enough for signal)')
    parser.add_argument('--deform-lr', type=float, default=0.05,
                        help='Adam learning rate for deformations')
    parser.add_argument('--refs-per-trial', type=int, default=1,
                        help='Reference plants per trial (default 1 for speed, more for precision)')
    parser.add_argument('--max-points', type=int, default=2000,
                        help='Max points per reference for Chamfer (default 2000)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cpu-chamfer', action='store_true',
                        help='Use CPU KD-tree Chamfer (for 128+ workers without GPU)')
    parser.add_argument('--analyze-only', action='store_true',
                        help='Only analyze existing study, no new trials')
    args = parser.parse_args()

    # Device selection
    if args.device == 'auto':
        import torch
        device = 'cuda' if torch.cuda.is_available() and not args.cpu_chamfer else 'cpu'
    else:
        device = args.device

    # Load reference plants
    print(f"Loading reference plants from: {args.refs}")
    reference_plants = load_reference_plants(args.refs)
    print(f"  Loaded {len(reference_plants)} reference plants")

    # Count species
    species_in_refs = set(rp.get('species', 'maize') for rp in reference_plants)
    print(f"  Species: {species_in_refs}")

    # Load per-species stats
    # Format: "maize:/path/to/stats.json" "wheat:/path/to/stats.json"
    from .objective import SpeciesData
    species_data = {}
    for spec in args.stats:
        if ':' in spec:
            species_name, stats_path = spec.split(':', 1)
        else:
            # Default: assume maize if no species prefix
            species_name = 'maize'
            stats_path = spec
        species_data[species_name] = SpeciesData(species_name, stats_path, day=args.day)
        print(f"  {species_name}: {stats_path} ({species_data[species_name].cfg.n_positions} leaf positions)")

    # Verify all referenced species have stats
    for sp in species_in_refs:
        if sp not in species_data:
            print(f"WARNING: species '{sp}' in references but no --stats provided. "
                  f"Those references will cause failures.", file=sys.stderr)

    # Create/load study
    study = create_study(args.storage, args.study_name, args.seed)

    if args.analyze_only:
        analyze_results(study, args.output)
        return

    # Build objective
    if args.cpu_chamfer or device == 'cpu':
        from .objective import make_cpu_objective
        objective = make_cpu_objective(
            reference_plants, species_data,
            deform_steps=args.deform_steps,
            deform_lr=args.deform_lr,
            refs_per_trial=args.refs_per_trial,
            max_points=args.max_points,
        )
        print(f"Using CPU Chamfer (KD-tree) — good for {args.workers}+ workers")
    else:
        from .objective import make_objective
        objective = make_objective(
            reference_plants, species_data,
            device=device,
            deform_steps=args.deform_steps,
            deform_lr=args.deform_lr,
            refs_per_trial=args.refs_per_trial,
            max_points=args.max_points,
        )
        print(f"Using GPU Chamfer on {device}")

    # Print search space summary
    from .catalog import describe_catalog
    print(f"\n{describe_catalog()}")
    print(f"\nConfiguration:")
    print(f"  Trials:  {args.n_trials}")
    print(f"  Workers: {args.workers}")
    print(f"  Device:  {device}")
    print(f"  Refs:    {len(reference_plants)} plants ({', '.join(species_in_refs)})")
    print(f"  Steps:   {args.deform_steps} gradient steps/trial")
    print(f"  Storage: {args.storage}")

    # Run optimization
    start = time.time()

    if args.workers <= 1:
        # Single worker — simple
        study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)
    else:
        # Multi-worker: each worker runs n_trials/workers trials
        # In production, launch N separate processes pointing at same storage file.
        # This single-process path is for testing.
        print(f"\nFor {args.workers} workers, launch {args.workers} processes:")
        print(f"  python -m dart.coupling.experimental.feature_search.pipeline \\")
        print(f"    --refs {args.refs} --stats {args.stats} \\")
        print(f"    --n-trials {args.n_trials // args.workers} \\")
        print(f"    --workers 1 --storage {args.storage} \\")
        print(f"    --study-name {args.study_name} --cpu-chamfer")
        print(f"\nRunning single-process with {args.n_trials} trials...")
        study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)

    elapsed = time.time() - start
    print(f"\nCompleted in {elapsed/3600:.1f} hours")

    # Analyze
    analyze_results(study, args.output)


if __name__ == '__main__':
    main()
