#!/usr/bin/env python3
"""
Test Phase 10: Iterative Tuzet-Baleno gs Coupling on Day 55.

Reuses existing Phase 1 (APAR) + Phase 2 (Baleno) setup to test convergence
of the iterative gs feedback loop.

Usage:
  cd /home/lukas/PHD
  source CPlantBox/cpbenv/bin/activate
  python -m coupling.test_iterative_coupling
"""

import sys
import json
import time
import numpy as np
from pathlib import Path

# Add coupling package to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from coupling.config import DEFAULT_XML, OUTPUT_DIR, PHOTO_PATH
from coupling.growth import grow_plant
from coupling.photosynthesis.coupled import (
    load_phase_csvs, verify_segment_alignment,
    build_apar_array, build_tleaf_array,
)
from coupling.photosynthesis.iterative import (
    run_iterative_coupling,
    segment_gs_to_triangle_gs,
    write_triangle_gs_csv,
    RHO_OVER_M,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SIMULATION_DAYS = 55
MAPPING_JSON = str(OUTPUT_DIR / 'maize_day55_dart_mapping.json')
REINDEX_JSON = str(OUTPUT_DIR / 'maize_day55_reindex.json')
GRID_INFO = str(OUTPUT_DIR / 'grid_info.json')
PHASE1_CSV = OUTPUT_DIR / 'maize_day55_segment_apar.csv'
PHASE2_CSV = OUTPUT_DIR / 'maize_day55_baleno_segments.csv'

BALENO_SIM_DIR = '/home/lukas/DART/bin/python_script/dart-eb-main/user_data/simulations/cpb_maize_eb'
BALENO_SIMU_NAME = 'cpb_maize_eb'

# Conditions
TARGET_PAR_UMOL = 1000.0
TAIR_C = 25.0
RH = 0.7
SOIL_PSI_CM = -500.0


def main():
    t_start = time.time()

    print("=" * 70)
    print("Phase 10 Test: Iterative Tuzet-Baleno gs Coupling (Day 55)")
    print("=" * 70)

    # --- Step 1: Grow plant ---
    print("\n--- Step 1: Grow plant ---")
    plant = grow_plant(
        str(DEFAULT_XML), simulation_time=SIMULATION_DAYS,
        enable_photosynthesis=True, seed=42,
    )

    # --- Step 2: Load Phase 1/2 data ---
    print("\n--- Step 2: Load Phase 1/2 data ---")
    csv_data = load_phase_csvs(PHASE1_CSV, PHASE2_CSV)
    verify_segment_alignment(plant, csv_data)
    apar_umol = build_apar_array(csv_data, TARGET_PAR_UMOL)
    tleaf_initial = build_tleaf_array(csv_data)

    print(f"\n  Phase 2 (Ball-Berry) Tleaf: mean={np.mean(tleaf_initial):.2f} C, "
          f"range=[{tleaf_initial.min():.2f}, {tleaf_initial.max():.2f}]")

    # --- Step 3: Quick plumbing test ---
    print("\n--- Step 3: Plumbing test (reverse mapping) ---")
    # Do a single CPlantBox solve to get gs, then test the mapping
    from coupling.photosynthesis.iterative import _extract_gs_from_solve
    gs_test = _extract_gs_from_solve(
        plant, SIMULATION_DAYS, apar_umol, tleaf_initial, RH, SOIL_PSI_CM)
    if gs_test is None:
        print("ERROR: gs extraction failed!")
        return

    active_gs = gs_test[gs_test > 0]
    print(f"  gs extracted: {len(active_gs)}/{len(gs_test)} active segments")
    print(f"  gs range: [{active_gs.min():.6f}, {active_gs.max():.6f}] mol CO2/m²/s")
    print(f"  gs mean:  {active_gs.mean():.6f} mol CO2/m²/s")

    # Test reverse mapping
    tri_result = segment_gs_to_triangle_gs(gs_test, MAPPING_JSON, REINDEX_JSON)
    print(f"  Triangle mapping: {tri_result['coverage']*100:.1f}% coverage, "
          f"{tri_result['n_triangles_total']} total triangles")

    # Test CSV write
    test_csv = OUTPUT_DIR / 'test_external_gs.csv'
    write_triangle_gs_csv(tri_result['rcw_per_triangle'], test_csv)
    n_lines = sum(1 for _ in open(test_csv))
    print(f"  CSV written: {test_csv.name} ({n_lines} leaf triangle entries)")

    # Unit conversion round-trip check
    gs_h2o = active_gs.mean() * 1.6
    rcw = RHO_OVER_M / gs_h2o
    gs_back = RHO_OVER_M / rcw / 1.6
    print(f"  Round-trip: gs={active_gs.mean():.6f} -> rcw={rcw:.2f} s/m -> "
          f"gs={gs_back:.6f} (err={abs(gs_back - active_gs.mean()):.2e})")

    # --- Step 4: Run iterative coupling ---
    print("\n--- Step 4: Iterative Tuzet-Baleno coupling ---")
    print(f"  Baleno sim: {BALENO_SIMU_NAME}")
    print(f"  Max iterations: 6, tolerance: 5%, damping: 0.6")
    print(f"  Soil psi: {SOIL_PSI_CM} cm (well-watered)")

    t_iter = time.time()
    result = run_iterative_coupling(
        plant, SIMULATION_DAYS,
        par_umol=apar_umol,
        mapping_json_path=MAPPING_JSON,
        reindex_json_path=REINDEX_JSON,
        baleno_sim_dir=BALENO_SIM_DIR,
        baleno_simu_name=BALENO_SIMU_NAME,
        grid_info_path=GRID_INFO,
        center_plant_idx=4,
        max_iterations=6,
        gs_tolerance=0.05,
        damping_alpha=0.6,
        soil_psi_cm=SOIL_PSI_CM,
        tair_c=TAIR_C,
        rh=RH,
        initial_tleaf=tleaf_initial,
    )
    iter_time = time.time() - t_iter

    # --- Step 5: Report results ---
    print(f"\n{'=' * 70}")
    print(f"RESULTS")
    print(f"{'=' * 70}")

    if result is None:
        print("FAILED: No result returned!")
        return

    print(f"\n  Converged:    {result['converged']}")
    print(f"  Iterations:   {result['iterations']}")
    print(f"  Time:         {iter_time:.1f}s "
          f"({iter_time/max(result['iterations'],1):.1f}s/iter)")

    print(f"\n  Final Tleaf:  mean={np.mean(result['tleaf_per_segment']):.2f} C, "
          f"range=[{result['tleaf_per_segment'].min():.2f}, "
          f"{result['tleaf_per_segment'].max():.2f}]")
    print(f"  Initial Tleaf (Ball-Berry): mean={np.mean(tleaf_initial):.2f} C")
    tleaf_shift = np.mean(result['tleaf_per_segment']) - np.mean(tleaf_initial)
    print(f"  Tleaf shift:  {tleaf_shift:+.2f} C")

    gs_final = result['gs_per_segment']
    active_final = gs_final[gs_final > 0]
    print(f"\n  Final gs:     mean={active_final.mean():.6f} mol CO2/m²/s")
    print(f"  Final An:     {result['an_total_mmol']:.3f} mmol CO2/d")

    # Compare with Phase 4 (single-pass Ball-Berry)
    phase4_An = 2250.75  # from coupled_results.json (informed_3d)
    an_change = (result['an_total_mmol'] - phase4_An) / phase4_An * 100
    print(f"\n  Phase 4 An (Ball-Berry Tleaf): {phase4_An:.2f} mmol CO2/d")
    print(f"  Phase 10 An (Tuzet-iterated):  {result['an_total_mmol']:.2f} mmol CO2/d")
    print(f"  An change: {an_change:+.1f}%")

    # Convergence history
    print(f"\n  Convergence history:")
    print(f"  {'Iter':>4} {'gs_mean':>12} {'Tleaf_mean':>12} {'An_mmol':>12} "
          f"{'mean_rel_Δgs':>14}")
    for h in result['gs_history']:
        rel = f"{h['mean_rel_change']:.4f}" if h['mean_rel_change'] is not None else "—"
        print(f"  {h['iteration']:>4} {h['gs_mean']:>12.6f} "
              f"{h['tleaf_mean']:>12.2f} {h['an_total_mmol']:>12.2f} "
              f"{rel:>14}")

    # Save results
    out_path = OUTPUT_DIR / 'phase10_iterative_results.json'
    save_data = {
        'converged': result['converged'],
        'iterations': result['iterations'],
        'time_seconds': iter_time,
        'tleaf_mean_initial_C': float(np.mean(tleaf_initial)),
        'tleaf_mean_final_C': float(np.mean(result['tleaf_per_segment'])),
        'tleaf_shift_C': float(tleaf_shift),
        'gs_mean_final': float(active_final.mean()),
        'an_total_mmol': result['an_total_mmol'],
        'an_phase4_mmol': phase4_An,
        'an_change_pct': an_change,
        'soil_psi_cm': SOIL_PSI_CM,
        'gs_history': result['gs_history'],
    }
    with open(out_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\n  Saved: {out_path}")

    total_time = time.time() - t_start
    print(f"\n  Total time: {total_time:.1f}s")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
