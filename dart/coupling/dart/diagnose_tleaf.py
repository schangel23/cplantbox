#!/usr/bin/env python3
"""Diagnose extreme Tleaf values from Baleno energy balance output.

Run from CPlantBox root:
    python3 -m dart.coupling.dart.diagnose_tleaf [baleno_sim_dir]

If no argument given, uses default BALENO_USER_DATA / simulations / SIMU_NAME_EB.
"""

import sys
import numpy as np
from pathlib import Path

def diagnose(baleno_sim_dir=None):
    if baleno_sim_dir is None:
        from ..config import DART_HOME
        DART_EB_DIR = DART_HOME / "bin" / "python_script" / "dart-eb-main"
        # Find most recent Baleno sim
        user_data = DART_EB_DIR / 'user_data' / 'simulations'
        if not user_data.exists():
            print(f"No simulations dir at {user_data}")
            return
        sims = sorted(user_data.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
        if not sims:
            print("No simulations found")
            return
        baleno_sim_dir = sims[0]
        print(f"Using most recent simulation: {baleno_sim_dir.name}")

    baleno_sim_dir = Path(baleno_sim_dir)
    output_base = baleno_sim_dir / 'output'
    results_dir = output_base / 'final_results'

    # --- Read scene ---
    scene_file = None
    for candidate in [output_base / 'scene', results_dir / 'scene.csv']:
        if candidate.exists():
            scene_file = candidate
            break
    if scene_file is None:
        print(f"ERROR: No scene file in {output_base}")
        return

    with open(scene_file) as f:
        header_line = f.readline().strip()
    delim = ';' if ';' in header_line else ','
    scene_header = [h.strip() for h in header_line.split(delim)]
    scene_str = np.genfromtxt(str(scene_file), skip_header=1, delimiter=delim, dtype=str)

    print(f"\nScene file: {scene_file}")
    print(f"  Columns: {scene_header}")
    print(f"  Rows: {scene_str.shape[0]}")

    col_type_id = scene_header.index('TYPE_ID') if 'TYPE_ID' in scene_header else 0
    col_dart_name = scene_header.index('DART_NAME') if 'DART_NAME' in scene_header else 1
    col_index_obj = scene_header.index('INDEX_OBJECT') if 'INDEX_OBJECT' in scene_header else 2
    col_surface = scene_header.index('SURFACE') if 'SURFACE' in scene_header else 5

    type_ids = scene_str[:, col_type_id].astype(float).astype(int)
    dart_names = scene_str[:, col_dart_name]
    surfaces = scene_str[:, col_surface].astype(float)

    # --- Read energy balance ---
    eb_file = results_dir / 'energy_balance_3D.csv'
    if not eb_file.exists():
        print(f"ERROR: {eb_file} not found")
        return

    with open(eb_file) as f:
        eb_header_line = f.readline().strip()
    eb_header = [h.strip() for h in eb_header_line.split(delim)]
    eb_data = np.genfromtxt(str(eb_file), skip_header=1, delimiter=delim,
                            dtype=float, filling_values=np.nan)

    print(f"\nEnergy balance: {eb_file.name}")
    print(f"  Columns: {eb_header}")
    print(f"  Rows: {eb_data.shape[0]}")

    # Find columns
    col_temp = -1
    col_eb_err = -1
    for i, h in enumerate(eb_header):
        if 'temperature' in h.lower():
            col_temp = i
        if 'error' in h.lower():
            col_eb_err = i

    if col_temp < 0:
        print("ERROR: No TEMPERATURE column found")
        return

    temps_k = eb_data[:, col_temp]
    temps_c = temps_k - 273.15

    # --- Read radiation for ABSORPTION_PAR ---
    rad_file = results_dir / 'radiation_3D.csv'
    rad_data = None
    col_apar = -1
    col_sunlit = -1
    if rad_file.exists():
        with open(rad_file) as f:
            rad_header = [h.strip() for h in f.readline().strip().split(delim)]
        rad_data = np.genfromtxt(str(rad_file), skip_header=1, delimiter=delim,
                                 dtype=float, filling_values=np.nan)
        for i, h in enumerate(rad_header):
            if 'absorption_par' in h.lower():
                col_apar = i
            if 'sunlit' in h.lower():
                col_sunlit = i
        print(f"\nRadiation: {rad_file.name}")
        print(f"  Columns: {rad_header}")

    # --- Read heat fluxes ---
    flux_file = results_dir / 'heat_fluxes_3D.csv'
    flux_data = None
    col_le = col_h = col_rs = -1
    if flux_file.exists():
        with open(flux_file) as f:
            flux_header = [h.strip() for h in f.readline().strip().split(delim)]
        flux_data = np.genfromtxt(str(flux_file), skip_header=1, delimiter=delim,
                                  dtype=float, filling_values=np.nan)
        for i, h in enumerate(flux_header):
            if 'latent' in h.lower():
                col_le = i
            if 'sensible' in h.lower():
                col_h = i
            if 'surface_resistance' in h.lower() or 'stomatal' in h.lower():
                col_rs = i
        print(f"\nHeat fluxes: {flux_file.name}")
        print(f"  Columns: {flux_header}")

    # --- Analysis ---
    print("\n" + "=" * 70)
    print("TEMPERATURE DISTRIBUTION (ALL TRIANGLES)")
    print("=" * 70)

    valid = ~np.isnan(temps_c)
    print(f"  Total triangles: {len(temps_c)}")
    print(f"  Valid temperatures: {np.sum(valid)}")
    print(f"  NaN temperatures: {np.sum(~valid)}")

    if np.sum(valid) == 0:
        print("  No valid temperatures!")
        return

    tc = temps_c[valid]
    print(f"\n  Mean:   {np.mean(tc):.2f} °C")
    print(f"  Median: {np.median(tc):.2f} °C")
    print(f"  Std:    {np.std(tc):.2f} °C")
    print(f"  Min:    {np.min(tc):.2f} °C")
    print(f"  Max:    {np.max(tc):.2f} °C")

    # Percentiles
    for p in [1, 5, 25, 50, 75, 95, 99]:
        print(f"  P{p:02d}:    {np.percentile(tc, p):.2f} °C")

    # --- Breakdown by TYPE_ID ---
    print("\n" + "=" * 70)
    print("TEMPERATURE BY TYPE_ID")
    print("=" * 70)

    unique_types = np.unique(type_ids)
    for tid in unique_types:
        mask = (type_ids == tid) & valid
        n = np.sum(mask)
        if n == 0:
            continue
        t = temps_c[mask]
        areas = surfaces[mask]
        print(f"\n  TYPE_ID={tid} ({n} triangles, area={np.sum(areas):.4f} m²)")
        print(f"    Tleaf: mean={np.mean(t):.2f}, min={np.min(t):.2f}, max={np.max(t):.2f}")
        if col_eb_err >= 0:
            errs = np.abs(eb_data[mask, col_eb_err])
            errs = errs[~np.isnan(errs)]
            if len(errs) > 0:
                print(f"    EB_ERROR: mean={np.mean(errs):.4f}, max={np.max(errs):.4f}")
        if rad_data is not None and col_apar >= 0:
            apars = rad_data[mask, col_apar]
            apars = apars[~np.isnan(apars)]
            if len(apars) > 0:
                print(f"    aPAR: mean={np.mean(apars):.4f}, min={np.min(apars):.4f}, max={np.max(apars):.4f}")

    # --- Identify extreme triangles ---
    print("\n" + "=" * 70)
    print("EXTREME TEMPERATURE TRIANGLES (> 45°C or < 5°C)")
    print("=" * 70)

    extreme_mask = valid & ((temps_c > 45.0) | (temps_c < 5.0))
    n_extreme = np.sum(extreme_mask)
    print(f"  Extreme triangles: {n_extreme} / {np.sum(valid)}")

    if n_extreme > 0:
        extreme_idx = np.where(extreme_mask)[0]
        # Show up to 20
        for idx in extreme_idx[:20]:
            info = f"  Row {idx}: Tleaf={temps_c[idx]:.2f}°C"
            info += f", TYPE_ID={type_ids[idx]}"
            info += f", DART_NAME={dart_names[idx]}"
            info += f", area={surfaces[idx]:.6f} m²"
            if col_eb_err >= 0:
                info += f", EB_ERR={eb_data[idx, col_eb_err]:.4f}"
            if rad_data is not None and col_apar >= 0:
                info += f", aPAR={rad_data[idx, col_apar]:.4f}"
            if rad_data is not None and col_sunlit >= 0:
                info += f", SUNLIT={rad_data[idx, col_sunlit]:.0f}"
            if flux_data is not None and col_le >= 0:
                info += f", LE={flux_data[idx, col_le]:.2f}"
            if flux_data is not None and col_h >= 0:
                info += f", H={flux_data[idx, col_h]:.2f}"
            if flux_data is not None and col_rs >= 0:
                info += f", Rs={flux_data[idx, col_rs]:.2f}"
            print(info)

        if n_extreme > 20:
            print(f"  ... and {n_extreme - 20} more")

        # Correlations
        extreme_areas = surfaces[extreme_mask]
        normal_areas = surfaces[valid & ~extreme_mask]
        print(f"\n  Area comparison:")
        print(f"    Extreme triangles: mean area = {np.mean(extreme_areas):.6f} m²")
        print(f"    Normal triangles:  mean area = {np.mean(normal_areas):.6f} m²")

        if col_eb_err >= 0:
            extreme_errs = np.abs(eb_data[extreme_mask, col_eb_err])
            normal_errs = np.abs(eb_data[valid & ~extreme_mask, col_eb_err])
            extreme_errs = extreme_errs[~np.isnan(extreme_errs)]
            normal_errs = normal_errs[~np.isnan(normal_errs)]
            if len(extreme_errs) > 0 and len(normal_errs) > 0:
                print(f"\n  EB_ERROR comparison:")
                print(f"    Extreme: mean={np.mean(extreme_errs):.4f}, max={np.max(extreme_errs):.4f}")
                print(f"    Normal:  mean={np.mean(normal_errs):.4f}, max={np.max(normal_errs):.4f}")

        if rad_data is not None and col_apar >= 0:
            extreme_apars = rad_data[extreme_mask, col_apar]
            normal_apars = rad_data[valid & ~extreme_mask, col_apar]
            extreme_apars = extreme_apars[~np.isnan(extreme_apars)]
            normal_apars = normal_apars[~np.isnan(normal_apars)]
            if len(extreme_apars) > 0 and len(normal_apars) > 0:
                print(f"\n  aPAR comparison:")
                print(f"    Extreme: mean={np.mean(extreme_apars):.4f}")
                print(f"    Normal:  mean={np.mean(normal_apars):.4f}")

    # --- Leaf-only analysis (pipeline's actual filter) ---
    print("\n" + "=" * 70)
    print("LEAF-ONLY ANALYSIS (type_ids >= 100 or == 5)")
    print("=" * 70)

    leaf_mask = (type_ids >= 100) | (type_ids == 5)
    leaf_temps = temps_c[leaf_mask & valid]
    print(f"  Leaf triangles: {np.sum(leaf_mask)}")
    print(f"  With valid temp: {len(leaf_temps)}")

    if len(leaf_temps) > 0:
        print(f"  Tleaf: mean={np.mean(leaf_temps):.2f}, "
              f"min={np.min(leaf_temps):.2f}, max={np.max(leaf_temps):.2f}")

        # How many are > 45°C?
        hot = leaf_temps > 45.0
        print(f"  > 45°C: {np.sum(hot)} ({100*np.sum(hot)/len(leaf_temps):.1f}%)")
        cold = leaf_temps < 5.0
        print(f"  < 5°C:  {np.sum(cold)} ({100*np.sum(cold)/len(leaf_temps):.1f}%)")

        # Check if type_id=5 specifically is the problem
        type5_mask = (type_ids == 5) & valid
        type100plus_mask = (type_ids >= 100) & valid
        if np.sum(type5_mask) > 0:
            t5 = temps_c[type5_mask]
            print(f"\n  TYPE_ID=5 specifically:")
            print(f"    Count: {np.sum(type5_mask)}")
            print(f"    Tleaf: mean={np.mean(t5):.2f}, min={np.min(t5):.2f}, max={np.max(t5):.2f}")
            print(f"    > 45°C: {np.sum(t5 > 45)}")
            # What DART_NAMEs do type_id=5 triangles have?
            t5_names = np.unique(dart_names[type_ids == 5])
            print(f"    DART_NAMEs: {t5_names[:10]}")

        if np.sum(type100plus_mask) > 0:
            t100 = temps_c[type100plus_mask]
            print(f"\n  TYPE_ID>=100:")
            print(f"    Count: {np.sum(type100plus_mask)}")
            print(f"    Tleaf: mean={np.mean(t100):.2f}, min={np.min(t100):.2f}, max={np.max(t100):.2f}")
            print(f"    > 45°C: {np.sum(t100 > 45)}")

    # --- Check for ground triangles leaking in ---
    print("\n" + "=" * 70)
    print("GROUND / SOIL TRIANGLES")
    print("=" * 70)

    ground_types = [0, 1, 2, 3, 4]  # common ground type_ids
    for gt in ground_types:
        gt_mask = (type_ids == gt) & valid
        if np.sum(gt_mask) > 0:
            gt_temps = temps_c[gt_mask]
            print(f"  TYPE_ID={gt}: {np.sum(gt_mask)} tris, "
                  f"Tleaf: mean={np.mean(gt_temps):.2f}, "
                  f"range=[{np.min(gt_temps):.2f}, {np.max(gt_temps):.2f}]")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == '__main__':
    sim_dir = sys.argv[1] if len(sys.argv) > 1 else None
    diagnose(sim_dir)
