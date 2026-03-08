#!/usr/bin/env python3
"""
Phase 9: Time-Series Diurnal Coupling Loop (Multi-Plant).

Runs the full CPlantBox-DART-Baleno-photosynthesis coupling chain at multiple
sun angles through a day to compute diurnally-integrated carbon gain.

Uses 9 unique plant realizations (seeds 42-50) for realistic field-level
statistics, following the multifield.py approach.

Two modes:
  Mode A: Single-day diurnal cycle
    python run_diurnal.py --days 55 --timestep-min 30

  Mode B: Multi-day growth series
    python run_diurnal.py --growth-days 20,30,40,50,55 --timestep-min 30

Per-timestep loop (within one day):
  1. pvlib -> sun_zen, sun_azi (skip if below horizon)
  2. DART: update sun angles, re-run direction+phase+dart (skip maket)
  3. Read radiative budget -> per-plant per-segment aPAR (9 arrays)
  4. Baleno: update atmosphere + _I sun angles, re-run energy balance
  5. Read Baleno -> per-segment Tleaf (center plant or uniform)
  6. CPlantBox photosynthesis: 9 per-plant solves
  7. Store hourly results (field mean +/- std)
  integrate(An, dt) -> daily carbon gain per plant + field statistics

Key optimization: geometry is static within a day -> maket runs once.

Usage:
  cd /home/lukas/PHD
  source CPlantBox/cpbenv/bin/activate
  python CPlantBox/dart/coupling/run_diurnal.py --days 55 --timestep-min 60
"""

import json
import shutil
import argparse
import time
import numpy as np
from pathlib import Path

from ..config import DEFAULT_XML, OUTPUT_DIR
from ..growth.grow import grow_plant
from ..geometry import (
    convert_obj_to_dart, convert_mapping_json_groups,
    loft_organs, extract_organs_for_lofter,
)
from ..prospect_params import get_prospect_params, log_consistency

from ..utils.solar_position import get_solar_positions, sim_day_to_date, get_clearsky_par
from ..utils.met_forcing import diurnal_met_profile

# Re-use functions from existing coupling scripts
from ..dart.simulation import (
    create_dart_simulation_multi,
    run_dart_full,
    update_sun_and_rerun,
    update_datetime_and_rerun,
    read_ori_reindex_multi,
    read_and_aggregate_apar_multi,
    PAR_BANDS,
)
from ..dart.baleno import (
    setup_baleno_full,
    setup_baleno_full_multi,
    update_baleno_atmosphere,
    update_baleno_sun_and_rerun_I,
    update_baleno_datetime_and_rerun_I,
    run_baleno_subprocess,
    read_baleno_tleaf,
    read_baleno_tleaf_multi,
    restore_config_files,
)
from .coupled import run_photosynthesis_solve

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
XML_PATH = str(DEFAULT_XML)

# Location: Juelich, Germany
LAT = 50.92
LON = 6.36
SOWING_DATE = '2025-05-01'

# Scene geometry (same as Phase 1)
SCENE_SIZE = [4, 4]
GRID_NX, GRID_NY = 3, 3
GRID_SPACING_X = 0.75
GRID_SPACING_Y = 0.25
N_PLANTS = GRID_NX * GRID_NY
CENTER_PLANT_IDX = N_PLANTS // 2
FIELD_FILENAME = 'plant_field.txt'
FIELD_SEED = 42


def _real_met_kwargs(sim_day):
    """Get diurnal_met_profile kwargs from real daily met data for sim_day.

    Returns empty dict if no data available (uses default sinusoidal).
    """
    from ..carbon.dvs_partitioning import get_daily_met
    day_met = get_daily_met(sim_day)
    if day_met is None:
        return {}
    kwargs = {
        'T_min': day_met['T_min_C'],
        'T_max': day_met['T_max_C'],
    }
    if 'RH_min' in day_met:
        kwargs['RH_min'] = day_met['RH_min']
        kwargs['RH_max'] = day_met['RH_max']
    if 'wind_mean_ms' in day_met:
        kwargs['wind_min'] = day_met['wind_mean_ms'] * 0.5
        kwargs['wind_max'] = day_met['wind_max_ms'] / 3.0
    return kwargs


# ============================================================================
# Plant setup: grow 9 unique plants + export meshes
# ============================================================================
def setup_plants_and_meshes(sim_day, output_subdir, plants=None):
    """Grow or re-loft 9 plants, export G3 meshes + DART OBJs + mapping JSONs.

    Args:
        sim_day: Simulation day (days since sowing).
        output_subdir: Directory for output files.
        plants: Optional list of 9 pre-existing pb.MappedPlant instances.
            When provided, skip grow_plant() and use these for mesh extraction.

    Returns:
        dict with keys: plants, meshes, mappings, dart_obj_paths,
        dart_mapping_paths, grid_info, grid_path
    """
    out = Path(output_subdir)
    out.mkdir(parents=True, exist_ok=True)

    plant_list = []
    meshes = []
    mappings = []
    dart_obj_paths = []
    dart_mapping_paths = []

    for i in range(N_PLANTS):
        seed = FIELD_SEED + i
        prefix = f"p{i}_"
        print(f"\n  --- Plant {i} (seed={seed}) ---")

        if plants is not None:
            plant = plants[i]
        else:
            plant = grow_plant(XML_PATH, simulation_time=sim_day,
                               min_stem_nodes=50, min_leaf_nodes=20, seed=seed,
                               enable_photosynthesis=True)

        # Extract organs with plant prefix for unique group names
        organ_dicts = extract_organs_for_lofter(
            plant, min_stem_nodes=50, min_leaf_nodes=20,
            name_prefix=prefix,
        )
        for od in organ_dicts:
            od['plant_id'] = i

        mesh = loft_organs(organ_dicts, stem_sides=16)

        # Export OBJ with plant-prefixed group names
        obj_path = out / f'maize_day{sim_day}_p{i}.obj'
        mesh.to_obj(str(obj_path), group_by_organ=True, group_prefix=prefix)

        # Export mapping JSON
        json_path = out / f'maize_day{sim_day}_p{i}_mapping.json'
        mesh.to_mapping_json(str(json_path))

        # Convert to DART coordinates
        dart_obj = out / f'maize_day{sim_day}_p{i}_dart.obj'
        convert_obj_to_dart(obj_path, dart_obj, scale=0.01,
                            zero_pad_groups=True)

        dart_mapping = out / f'maize_day{sim_day}_p{i}_dart_mapping.json'
        shutil.copy(json_path, dart_mapping)
        convert_mapping_json_groups(str(dart_mapping))

        with open(dart_mapping) as f:
            mapping = json.load(f)

        plant_list.append(plant)
        meshes.append(mesh)
        mappings.append(mapping)
        dart_obj_paths.append(dart_obj)
        dart_mapping_paths.append(dart_mapping)

        n_leaf = sum(1 for o in mapping['organs'] if o['type'] == 'leaf')
        print(f"    {mapping['n_triangles']} tris, {n_leaf} leaf organs")

    # Grid info
    positions = []
    for iy in range(GRID_NY):
        for ix in range(GRID_NX):
            x = 2.0 + (ix - (GRID_NX - 1) / 2) * GRID_SPACING_X
            y = 2.0 + (iy - (GRID_NY - 1) / 2) * GRID_SPACING_Y
            positions.append((x, y))

    grid_info = {
        'grid_nx': GRID_NX, 'grid_ny': GRID_NY,
        'spacing_x_m': GRID_SPACING_X, 'spacing_y_m': GRID_SPACING_Y,
        'n_plants': N_PLANTS,
        'center_plant_idx': CENTER_PLANT_IDX,
        'positions_m': positions,
        'field_filename': FIELD_FILENAME,
        'unique_models': True,
    }
    grid_path = out / 'grid_info.json'
    with open(grid_path, 'w') as f:
        json.dump(grid_info, f, indent=2)

    total_tris = sum(m['n_triangles'] for m in mappings)
    print(f"\n  Total: {N_PLANTS} plants, {total_tris} triangles")

    return {
        'plants': plant_list,
        'meshes': meshes,
        'mappings': mappings,
        'dart_obj_paths': dart_obj_paths,
        'dart_mapping_paths': dart_mapping_paths,
        'grid_info': grid_info,
        'grid_path': grid_path,
    }


# ============================================================================
# Single-day diurnal loop (Mode A)
# ============================================================================
def run_single_day(sim_day, timestep_min=30, enable_baleno=True,
                   met_csv=None, skip_photosynthesis=False,
                   iterate_gs=False, gs_max_iterations=6,
                   gs_tolerance=0.05, gs_damping_alpha=0.6):
    """Run full diurnal coupling for a single day with 9 unique plants.

    Args:
        sim_day: CPlantBox simulation day (days since sowing).
        timestep_min: Timestep in minutes (default 30).
        enable_baleno: If True, run Baleno energy balance per timestep.
        met_csv: Optional path to CSV with met forcing.
        skip_photosynthesis: If True, only compute aPAR (for testing).
        iterate_gs: If True, use iterative Tuzet-Baleno gs coupling
            (Phase 10) instead of single-pass Ball-Berry.
        gs_max_iterations: Max iterations for gs convergence.
        gs_tolerance: Relative convergence threshold (0.05 = 5%).
        gs_damping_alpha: Under-relaxation factor for gs.

    Returns:
        dict with 'hourly' list, 'daily_An_mol_per_plant', 'daily_An_mol_field_mean'.
    """
    calendar_date = sim_day_to_date(sim_day, SOWING_DATE)
    print(f"\n{'=' * 70}")
    print(f"DIURNAL LOOP: Day {sim_day} ({calendar_date})")
    print(f"  Timestep: {timestep_min} min, Baleno: {enable_baleno}, "
          f"iterate_gs: {iterate_gs}")
    print(f"  Plants: {N_PLANTS} unique realizations (seeds {FIELD_SEED}-{FIELD_SEED + N_PLANTS - 1})")
    print(f"{'=' * 70}")

    # --- Solar positions ---
    solar_df = get_solar_positions(
        calendar_date, LAT, LON, freq=f'{timestep_min}min')
    n_daylight = len(solar_df)
    print(f"\n  Solar positions: {n_daylight} daylight timesteps")
    if n_daylight == 0:
        print("  ERROR: No daylight hours!")
        return {'hourly': [], 'daily_An_mol_field_mean': 0.0,
                'daily_An_mol_per_plant': [0.0] * N_PLANTS}

    sunrise = solar_df.index[0]
    sunset = solar_df.index[-1]
    print(f"  Sunrise: {sunrise.strftime('%H:%M')} UTC, "
          f"Sunset: {sunset.strftime('%H:%M')} UTC")

    # --- Met forcing ---
    if met_csv:
        from utils.met_forcing import load_met_csv
        met = load_met_csv(met_csv)
    else:
        met = diurnal_met_profile(calendar_date, LAT, LON,
                                   freq=f'{timestep_min}min',
                                   **_real_met_kwargs(sim_day))
    print(f"  Met: T={met['T_air_C'].min():.1f}-{met['T_air_C'].max():.1f} C, "
          f"RH={met['RH'].min():.0%}-{met['RH'].max():.0%}")

    # --- Setup output directory ---
    day_dir = OUTPUT_DIR / 'diurnal' / f'day{sim_day}'
    day_dir.mkdir(parents=True, exist_ok=True)

    # --- Grow 9 unique plants + export meshes ---
    prospect_params = get_prospect_params(sim_day)
    log_consistency(sim_day)

    setup = setup_plants_and_meshes(sim_day, day_dir)
    dart_obj_paths = setup['dart_obj_paths']
    mappings = setup['mappings']
    grid_info = setup['grid_info']
    grid_path = setup['grid_path']

    # --- Create multi-plant DART simulation (once -- geometry static) ---
    simu_name = f'cpb_diurnal_day{sim_day}'
    first_zen = solar_df.iloc[0]['apparent_zenith']
    first_azi = solar_df.iloc[0]['azimuth']
    first_ts_time = solar_df.index[0]

    print(f"\n  Creating multi-plant DART simulation: {simu_name}")
    print(f"  Initial sun: zenith={first_zen:.1f}, azimuth={first_azi:.1f}")
    print(f"  Using DART exactDate=1 (date={calendar_date}, "
          f"UTC={first_ts_time.hour:02d}:{first_ts_time.minute:02d})")

    simu = create_dart_simulation_multi(
        obj_paths=dart_obj_paths,
        mapping_json_paths=setup['dart_mapping_paths'],
        simu_name=simu_name,
        prospect_params=prospect_params,
        scene_size=SCENE_SIZE,
        grid_info=grid_info,
        par_bands=PAR_BANDS,
        field_filename=FIELD_FILENAME,
        calendar_date=calendar_date,
        hour_utc=first_ts_time.hour,
        minute_utc=first_ts_time.minute,
        lat=LAT, lon=LON,
        use_exact_date=True,
    )

    # --- Run full DART (first time -- includes maket) ---
    print(f"  Running full DART (with maket, {N_PLANTS} models)...")
    t0 = time.time()
    run_dart_full(simu, timeout=1200)
    print(f"  DART full run: {time.time() - t0:.1f}s")

    # --- Read .ori reindex for all plants (once) ---
    reindex_infos = read_ori_reindex_multi(simu, dart_obj_paths)
    if reindex_infos is None:
        print("  ERROR: No .ori reindex tables!")
        return {'hourly': [], 'daily_An_mol_field_mean': 0.0,
                'daily_An_mol_per_plant': [0.0] * N_PLANTS}

    # Save per-plant reindex JSONs
    per_plant_reindex_paths = []
    for pi in range(N_PLANTS):
        ri = reindex_infos[pi]
        ri_path = day_dir / f'maize_day{sim_day}_p{pi}_reindex.json'
        ri_save = {
            'dart_to_obj': {str(k): v.tolist()
                            for k, v in ri['dart_to_obj'].items()},
            'group_names': ri['groups_sorted'],
            'group_offsets': {g: ri['group_offsets'][g]
                              for g in ri['groups_sorted']},
        }
        with open(ri_path, 'w') as f:
            json.dump(ri_save, f, indent=2)
        per_plant_reindex_paths.append(ri_path)

    # Backward compat: center reindex path alias
    reindex_path = per_plant_reindex_paths[CENTER_PLANT_IDX]

    # --- Setup Baleno (once -- geometry static, all plants) ---
    baleno_setup = None
    center_dart_obj = dart_obj_paths[CENTER_PLANT_IDX]
    center_dart_mapping = setup['dart_mapping_paths'][CENTER_PLANT_IDX]
    if enable_baleno:
        print(f"\n  Setting up Baleno (all {N_PLANTS} plants)...")
        try:
            baleno_setup = setup_baleno_full_multi(
                obj_paths=[str(p) for p in dart_obj_paths],
                mapping_json_paths=[str(p) for p in setup['dart_mapping_paths']],
                reindex_json_paths=[str(p) for p in per_plant_reindex_paths],
                grid_info_path=str(grid_path),
                prospect_params=prospect_params,
                scene_size=SCENE_SIZE,
                dart_simu_name=f'cpb_diurnal_day{sim_day}_eb',
                baleno_simu_name=f'cpb_diurnal_day{sim_day}_eb',
                field_filename=FIELD_FILENAME,
                calendar_date=calendar_date,
                hour_utc=first_ts_time.hour,
                minute_utc=first_ts_time.minute,
                lat=LAT, lon=LON,
                use_exact_date=True,
            )
            # Run initial full Baleno (with maket on _I)
            print(f"  Running initial Baleno _I full DART...")
            t0 = time.time()
            baleno_setup['simu_I'].run.full(timeout=1800)
            print(f"  Baleno _I full: {time.time() - t0:.1f}s")
        except Exception as e:
            import traceback
            print(f"\n  {'!' * 60}")
            print(f"  BALENO SETUP FAILED — all Tleaf will default to Tair!")
            print(f"  Error: {e}")
            traceback.print_exc()
            print(f"  {'!' * 60}\n")
            baleno_setup = None

    # --- Diurnal loop ---
    print(f"\n{'=' * 70}")
    print(f"DIURNAL LOOP: {n_daylight} timesteps x {N_PLANTS} plants")
    print(f"{'=' * 70}")

    hourly_results = []
    for step_i, (ts_time, ts_row) in enumerate(solar_df.iterrows()):
        sun_zen = ts_row['apparent_zenith']
        sun_azi = ts_row['azimuth']
        ts_label = ts_time.strftime('%H:%M')

        print(f"\n  [{step_i+1}/{n_daylight}] {ts_label} UTC -- "
              f"zen={sun_zen:.1f}, azi={sun_azi:.1f}")

        t_step = time.time()

        # --- Get met conditions for this timestep ---
        if ts_time in met.index:
            met_row = met.loc[ts_time]
        else:
            idx = met.index.get_indexer([ts_time], method='nearest')[0]
            met_row = met.iloc[idx]

        T_air_C = float(met_row['T_air_C'])
        T_air_K = float(met_row['T_air_K'])
        ea_hPa = float(met_row['ea_hPa'])
        wind_ms = float(met_row['wind_ms'])
        rh = float(met_row['RH'])

        # --- Update DART sun + re-run RT ---
        if step_i == 0:
            pass  # First timestep already ran full DART
        else:
            t0 = time.time()
            update_datetime_and_rerun(
                simu, calendar_date, ts_time.hour, ts_time.minute)
            print(f"    DART RT: {time.time() - t0:.1f}s")

        # --- Read per-plant per-segment aPAR ---
        all_plant_apar = read_and_aggregate_apar_multi(
            simu, mappings, reindex_infos,
        )
        if all_plant_apar is None:
            print(f"    WARNING: aPAR read failed, skipping timestep")
            continue

        # Field-level aPAR statistics
        plant_mean_apars = [float(np.mean(a)) for a in all_plant_apar]
        field_mean_apar = float(np.mean(plant_mean_apars))
        field_std_apar = float(np.std(plant_mean_apars))
        print(f"    aPAR: field mean={field_mean_apar:.4f} "
              f"(+/-{field_std_apar:.4f})")

        # --- Baleno energy balance (optional, all plants) ---
        # Skip Baleno at very low PAR (dawn/dusk): EB solver is unreliable
        # and Tleaf ≈ Tair anyway when shortwave radiation is negligible.
        MIN_PAR_FOR_BALENO = 50.0  # W/m²
        all_tleaf_baleno = None  # list of per-plant Tleaf arrays
        baleno_ok_flag = False
        clearsky_par_wm2_early = get_clearsky_par(ts_time, LAT, LON)
        if baleno_setup is not None and clearsky_par_wm2_early < MIN_PAR_FOR_BALENO:
            print(f"    PAR={clearsky_par_wm2_early:.1f} W/m² < {MIN_PAR_FOR_BALENO} "
                  f"-> skip Baleno, Tleaf=Tair")
        elif baleno_setup is not None:
            try:
                t0 = time.time()
                update_baleno_datetime_and_rerun_I(
                    baleno_setup['simu_I'], calendar_date,
                    ts_time.hour, ts_time.minute)
                update_baleno_atmosphere(
                    baleno_setup['baleno_sim_dir'],
                    T_air_K=T_air_K, ea_hPa=ea_hPa, wind_ms=wind_ms,
                )
                from ..dart.baleno import BALENO_DIR
                baleno_config_path = BALENO_DIR / 'resources' / 'config.ini'
                import textwrap
                baleno_config_path.write_text(textwrap.dedent(f"""\
                    [simulation]
                    user_data_path =
                    name = {baleno_setup['baleno_simu_name']}
                """))
                # Adaptive timeout: shorter at low PAR where Baleno struggles
                baleno_timeout = 600 if clearsky_par_wm2_early < 100.0 else 1800
                ok = run_baleno_subprocess(timeout=baleno_timeout)
                if ok:
                    all_tleaf_baleno = read_baleno_tleaf_multi(
                        baleno_setup['baleno_sim_dir'],
                        [str(p) for p in setup['dart_mapping_paths']],
                        [str(p) for p in per_plant_reindex_paths],
                        N_PLANTS, tair_c=T_air_C,
                    )
                baleno_ok_flag = all_tleaf_baleno is not None
                baleno_time = time.time() - t0
                print(f"    Baleno: {baleno_time:.1f}s, "
                      f"Tleaf={'OK' if baleno_ok_flag else 'FAILED'}")
                if baleno_ok_flag:
                    from ..dart.baleno import log_baleno_diagnostics
                    log_baleno_diagnostics(
                        baleno_setup['baleno_sim_dir'],
                        all_tleaf_baleno[CENTER_PLANT_IDX], T_air_C)
                else:
                    print(f"    WARNING: Baleno ran but Tleaf extraction failed "
                          f"(subprocess_ok={ok}). Using Tair={T_air_C:.1f}C")
            except Exception as e:
                import traceback
                print(f"    Baleno error: {e}")
                traceback.print_exc()

        # --- Scale DART aPAR to absolute umol/m2/s ---
        n_par_bands = len(PAR_BANDS)
        clearsky_par_wm2 = get_clearsky_par(ts_time, LAT, LON)
        actual_par_per_band = clearsky_par_wm2 / n_par_bands

        # Convert per-plant aPAR arrays to physical umol
        all_par_umol = []
        for pi in range(N_PLANTS):
            apar_abs_wm2 = all_plant_apar[pi] * actual_par_per_band
            par_umol = np.clip(apar_abs_wm2 * 4.57, 0.0, 3000.0)
            all_par_umol.append(par_umol)

        mean_par_umol = float(np.mean([np.mean(p) for p in all_par_umol]))
        print(f"    Clearsky PAR: {clearsky_par_wm2:.1f} W/m2, "
              f"absolute aPAR: field mean={mean_par_umol:.1f} umol/m2/s")

        # --- Per-plant Tleaf arrays ---
        all_tleaf = []
        for pi in range(N_PLANTS):
            n_segs = len(all_par_umol[pi])
            if all_tleaf_baleno is not None and pi < len(all_tleaf_baleno):
                t = all_tleaf_baleno[pi]
                if len(t) == n_segs:
                    all_tleaf.append(t.copy())
                else:
                    all_tleaf.append(np.full(n_segs, T_air_C))
            else:
                all_tleaf.append(np.full(n_segs, T_air_C))

        # --- CPlantBox photosynthesis (per-plant) ---
        per_plant_An = [0.0] * N_PLANTS
        if not skip_photosynthesis:
            if iterate_gs and baleno_setup is not None:
                # Iterative Tuzet-Baleno coupling for ALL plants
                from .iterative import run_iterative_coupling_multi

                # Grow fresh plants for photosynthesis solve
                iter_plants = []
                for pi in range(N_PLANTS):
                    seed = FIELD_SEED + pi
                    plant_ts = grow_plant(
                        XML_PATH, simulation_time=sim_day,
                        enable_photosynthesis=True, seed=seed,
                    )
                    iter_plants.append(plant_ts)

                iter_results = run_iterative_coupling_multi(
                    iter_plants, sim_day,
                    par_umol_per_plant=all_par_umol,
                    mapping_json_paths=[str(p) for p in setup['dart_mapping_paths']],
                    reindex_json_paths=[str(p) for p in per_plant_reindex_paths],
                    baleno_sim_dir=str(baleno_setup['baleno_sim_dir']),
                    baleno_simu_name=baleno_setup['baleno_simu_name'],
                    n_plants=N_PLANTS,
                    max_iterations=gs_max_iterations,
                    gs_tolerance=gs_tolerance,
                    damping_alpha=gs_damping_alpha,
                    soil_psi_cm=-500.0,
                    tair_c=T_air_C, rh=rh,
                    initial_tleaf=all_tleaf,
                )
                if iter_results is not None:
                    for pi in range(N_PLANTS):
                        per_plant_An[pi] = iter_results[pi]['an_total_mmol']
                        all_tleaf[pi] = iter_results[pi]['tleaf_per_segment']
            else:
                for pi in range(N_PLANTS):
                    seed = FIELD_SEED + pi
                    plant_ts = grow_plant(
                        XML_PATH, simulation_time=sim_day,
                        enable_photosynthesis=True, seed=seed,
                    )

                    result = run_photosynthesis_solve(
                        plant_ts, sim_day,
                        par=all_par_umol[pi], tleaf=all_tleaf[pi],
                        label=f"ts_{ts_label}_p{pi}",
                        rh=rh, soil_psi_cm=-500.0,
                    )
                    if result is not None:
                        per_plant_An[pi] = result['An_total_mmol']

            An_field_mean = float(np.mean(per_plant_An))
            An_field_std = float(np.std(per_plant_An))
            print(f"    An: field mean={An_field_mean:.3f} "
                  f"+/-{An_field_std:.3f} mmol CO2/d")

        # Compute mean_tleaf AFTER iterative coupling has updated all_tleaf
        mean_tleaf = float(np.mean([np.mean(t) for t in all_tleaf]))

        # --- Per-timestep DVS carbon tracking ---
        An_field_mean = float(np.mean(per_plant_An))
        dvs_carbon = None
        if not skip_photosynthesis and An_field_mean > 0:
            try:
                from ..carbon.dvs_partitioning import partition_carbon_dvs
                dvs_carbon = partition_carbon_dvs(An_field_mean, sim_day, Tair_C=T_air_C)
            except Exception as e:
                print(f"    DVS carbon tracking error: {e}")

        step_time = time.time() - t_step
        print(f"    Step time: {step_time:.1f}s")

        hourly_row = {
            'time_utc': ts_label,
            'zenith': float(sun_zen),
            'azimuth': float(sun_azi),
            'T_air_C': T_air_C,
            'RH': rh,
            'wind_ms': wind_ms,
            'clearsky_par_Wm2': clearsky_par_wm2,
            'mean_apar_umol': mean_par_umol,
            'dart_mean_apar': field_mean_apar,
            'mean_tleaf_C': mean_tleaf,
            'baleno_ok': baleno_ok_flag,
            'An_field_mean_mmol_d': An_field_mean,
            'An_field_std_mmol_d': float(np.std(per_plant_An)),
        }
        if dvs_carbon is not None:
            hourly_row['Rm_dvs_mmol'] = dvs_carbon['Rm_total_mmol']
            hourly_row['Rg_dvs_mmol'] = dvs_carbon['Rg_total_mmol']
            hourly_row['FR_leaf_dvs'] = dvs_carbon['FR_leaf']
            hourly_row['FR_root_dvs'] = dvs_carbon['FR_root']
        # Per-plant An values
        for pi in range(N_PLANTS):
            hourly_row[f'An_p{pi}'] = per_plant_An[pi]
        hourly_results.append(hourly_row)

    # --- Cleanup Baleno configs ---
    if baleno_setup is not None:
        try:
            restore_config_files(baleno_setup['backups'])
        except Exception:
            pass

    # --- Integrate daily carbon (per-plant) ---
    daily_An_per_plant = _integrate_daily_per_plant(
        hourly_results, timestep_min)
    daily_An_field_mean = float(np.mean(daily_An_per_plant))
    daily_An_field_std = float(np.std(daily_An_per_plant))

    # --- Save results ---
    _save_diurnal_results(day_dir, sim_day, calendar_date, hourly_results,
                           daily_An_per_plant, daily_An_field_mean,
                           daily_An_field_std, timestep_min)

    print(f"\n{'=' * 70}")
    print(f"DAY {sim_day} COMPLETE")
    print(f"  Timesteps: {len(hourly_results)}")
    print(f"  Daily An per plant: {[f'{a:.6f}' for a in daily_An_per_plant]}")
    print(f"  Daily An field mean: {daily_An_field_mean:.6f} mol CO2/plant/day")
    print(f"  Daily An field std:  {daily_An_field_std:.6f} mol CO2/plant/day")
    if daily_An_field_mean > 0:
        cv_pct = daily_An_field_std / daily_An_field_mean * 100
        print(f"  Field CV: {cv_pct:.1f}%")
    print(f"  Output: {day_dir}")
    print(f"{'=' * 70}")

    return {
        'hourly': hourly_results,
        'daily_An_mol_per_plant': daily_An_per_plant,
        'daily_An_mol_field_mean': daily_An_field_mean,
        'daily_An_mol_field_std': daily_An_field_std,
        # Backward compat
        'daily_An_mol': daily_An_field_mean,
    }


def run_single_day_with_carbon(sim_day, timestep_min=30, enable_baleno=True,
                               met_csv=None, skip_photosynthesis=False,
                               iterate_gs=False, gs_max_iterations=6,
                               gs_tolerance=0.05, gs_damping_alpha=0.6,
                               carbon_method='auto', warm_start=None,
                               gdd_accumulated=None):
    """Run diurnal loop + daily carbon partitioning and AgroC export.

    Wraps run_single_day() and appends:
      1. Grows one representative plant (center seed) with roots
      2. Runs photosynthesis at peak-hour conditions
      3. Scales per-segment An to match diurnal-integrated daily total
      4. Solves carbon partitioning (phloem or DVS)
      5. Exports AgroC coupling timestep

    Args:
        sim_day: simulation day (days since sowing).
        timestep_min: diurnal timestep [min].
        enable_baleno: run Baleno energy balance.
        met_csv: optional met forcing CSV.
        skip_photosynthesis: skip photosynthesis (aPAR only).
        iterate_gs: iterative Tuzet-Baleno coupling.
        gs_max_iterations, gs_tolerance, gs_damping_alpha: gs params.
        carbon_method: 'auto', 'phloem', or 'dvs'.
        gdd_accumulated: Accumulated GDD from sowing (°C·day). If provided,
            DVS is computed from thermal time instead of calendar days.

    Returns:
        dict with all run_single_day keys plus 'daily_carbon' and 'daily_agroc_ts'.
    """
    # 1. Run diurnal photosynthesis loop
    result = run_single_day(
        sim_day, timestep_min=timestep_min,
        enable_baleno=enable_baleno, met_csv=met_csv,
        skip_photosynthesis=skip_photosynthesis,
        iterate_gs=iterate_gs,
        gs_max_iterations=gs_max_iterations,
        gs_tolerance=gs_tolerance,
        gs_damping_alpha=gs_damping_alpha,
    )

    daily_An_mol = result.get('daily_An_mol_field_mean', 0.0)
    daily_An_mmol = daily_An_mol * 1000.0

    if daily_An_mmol <= 0 or skip_photosynthesis:
        result['daily_carbon'] = None
        result['daily_agroc_ts'] = None
        return result

    # 2. Grow center plant with roots for carbon partitioning
    from ..growth.grow import grow_plant, run_photosynthesis, extract_lai_profile
    from ..carbon import solve_carbon_partitioning
    from ..agroc import export_agroc_timestep

    center_seed = FIELD_SEED + CENTER_PLANT_IDX
    plant = grow_plant(XML_PATH, simulation_time=sim_day,
                       min_stem_nodes=50, min_leaf_nodes=20,
                       seed=center_seed, enable_photosynthesis=True)

    # 3. Run photosynthesis at peak conditions to get per-segment An shape
    day_dir = OUTPUT_DIR / 'diurnal' / f'day{sim_day}'
    day_dir.mkdir(parents=True, exist_ok=True)
    prefix = str(day_dir / f'carbon_photo_day{sim_day}')
    hm = run_photosynthesis(plant, sim_time=sim_day, output_prefix=prefix,
                            par_umol=1000.0, tair_c=25.0)

    if hm is None:
        result['daily_carbon'] = None
        result['daily_agroc_ts'] = None
        return result

    # 4. Scale per-segment An to match diurnal-integrated daily total
    An_leaf = np.array(hm.get_net_assimilation())  # mol CO2/d per seg
    An_peak_total = float(np.sum(An_leaf))
    if An_peak_total > 0:
        scale = (daily_An_mol) / An_peak_total
        An_leaf_scaled = An_leaf * scale
    else:
        An_leaf_scaled = An_leaf

    # 5. Carbon partitioning
    try:
        carbon = solve_carbon_partitioning(
            plant, An_leaf_scaled, Tair_C=25.0,
            method=carbon_method, day=sim_day,
            gdd_accumulated=gdd_accumulated,
        )
    except Exception as e:
        print(f"  Carbon partitioning error: {e}")
        carbon = None

    # 6. LAI + AgroC export
    lai = extract_lai_profile(plant, n_bins=10)
    agroc_ts = None
    if carbon is not None:
        try:
            agroc_ts = export_agroc_timestep(
                plant, hm, carbon, lai,
                day=sim_day, par_umol=1000.0, tair_c=25.0,
            )
        except Exception as e:
            print(f"  AgroC export error: {e}")

    result['daily_carbon'] = carbon
    result['daily_agroc_ts'] = agroc_ts
    return result


def _integrate_daily_per_plant(hourly_results, timestep_min):
    """Integrate per-timestep An to daily total for each plant.

    Returns:
        List of daily An in mol CO2/plant/day (one per plant).
    """
    if not hourly_results:
        return [0.0] * N_PLANTS

    dt_day = timestep_min / (24 * 60)
    n = len(hourly_results)

    daily_per_plant = []
    for pi in range(N_PLANTS):
        key = f'An_p{pi}'
        An_values = [r.get(key, 0.0) for r in hourly_results]

        if n == 1:
            daily_mmol = An_values[0] * dt_day
        else:
            daily_mmol = 0.0
            for i in range(n - 1):
                daily_mmol += (An_values[i] + An_values[i + 1]) / 2.0 * dt_day

        daily_per_plant.append(daily_mmol / 1000.0)  # mmol -> mol

    return daily_per_plant


def _save_diurnal_results(day_dir, sim_day, calendar_date, hourly_results,
                            daily_An_per_plant, daily_An_field_mean,
                            daily_An_field_std, timestep_min):
    """Save hourly CSV, daily summary JSON, and diurnal curve plot."""
    import csv

    # --- Hourly CSV ---
    csv_path = day_dir / 'hourly_results.csv'
    fieldnames = [
        'time_utc', 'zenith', 'azimuth', 'T_air_C', 'RH', 'wind_ms',
        'clearsky_par_Wm2', 'mean_apar_umol', 'dart_mean_apar',
        'mean_tleaf_C', 'baleno_ok',
        'An_field_mean_mmol_d', 'An_field_std_mmol_d',
        'Rm_dvs_mmol', 'Rg_dvs_mmol', 'FR_leaf_dvs', 'FR_root_dvs',
    ]
    for pi in range(N_PLANTS):
        fieldnames.append(f'An_p{pi}')

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in hourly_results:
            writer.writerow({k: r.get(k, '') for k in fieldnames})
    print(f"  CSV: {csv_path}")

    # --- Daily summary JSON ---
    summary = {
        'sim_day': sim_day,
        'calendar_date': str(calendar_date),
        'timestep_min': timestep_min,
        'n_timesteps': len(hourly_results),
        'n_plants': N_PLANTS,
        'seeds': list(range(FIELD_SEED, FIELD_SEED + N_PLANTS)),
        'daily_An_mol_per_plant': daily_An_per_plant,
        'daily_An_mol_field_mean': daily_An_field_mean,
        'daily_An_mol_field_std': daily_An_field_std,
    }
    if daily_An_field_mean > 0:
        summary['field_CV_pct'] = daily_An_field_std / daily_An_field_mean * 100

    if hourly_results:
        An_means = [r['An_field_mean_mmol_d'] for r in hourly_results]
        peak_idx = An_means.index(max(An_means))
        summary['peak_An_mmol_d'] = max(An_means)
        summary['peak_time_utc'] = hourly_results[peak_idx]['time_utc']
        summary['sunrise_utc'] = hourly_results[0]['time_utc']
        summary['sunset_utc'] = hourly_results[-1]['time_utc']

    json_path = day_dir / 'daily_summary.json'
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  JSON: {json_path}")

    # --- Diurnal curve plot ---
    try:
        _plot_diurnal_curve(day_dir, hourly_results, sim_day, calendar_date,
                             daily_An_field_mean, daily_An_field_std)
    except Exception as e:
        print(f"  WARNING: Plot failed: {e}")


def _plot_diurnal_curve(day_dir, hourly_results, sim_day, calendar_date,
                          daily_An_mean, daily_An_std):
    """Create 2x2 diurnal curve plot with field mean +/- std shading."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    if not hourly_results:
        return

    times = [r['time_utc'] for r in hourly_results]
    hours = []
    for t in times:
        h, m = t.split(':')
        hours.append(int(h) + int(m) / 60.0)
    hours = np.array(hours)

    An_mean = np.array([r['An_field_mean_mmol_d'] for r in hourly_results])
    An_std = np.array([r['An_field_std_mmol_d'] for r in hourly_results])
    apar = np.array([r['mean_apar_umol'] for r in hourly_results])
    tleaf = np.array([r['mean_tleaf_C'] for r in hourly_results])
    zen = np.array([r['zenith'] for r in hourly_results])

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor='white')
    fig.suptitle(
        f"Diurnal Coupling: Day {sim_day} ({calendar_date}) - "
        f"{N_PLANTS} plants\n"
        f"Daily integrated An = {daily_An_mean:.4f} +/- "
        f"{daily_An_std:.4f} mol CO$_2$/plant/day",
        fontsize=13, fontweight='bold', y=0.98,
    )

    # Panel 1: An diurnal curve with field mean +/- std
    ax = axes[0, 0]
    ax.plot(hours, An_mean, 'o-', color='forestgreen', lw=2, ms=4,
            label='Field mean')
    ax.fill_between(hours, An_mean - An_std, An_mean + An_std,
                    color='forestgreen', alpha=0.2, label='+/- 1 std')
    ax.set_xlabel('Hour (UTC)')
    ax.set_ylabel('An (mmol CO$_2$ d$^{-1}$)')
    ax.set_title('Net Assimilation (field)')
    ax.axhline(0, color='k', lw=0.5)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 2: aPAR diurnal curve
    ax = axes[0, 1]
    ax.plot(hours, apar, 'o-', color='goldenrod', lw=2, ms=4)
    ax.set_xlabel('Hour (UTC)')
    ax.set_ylabel('Mean aPAR (umol m$^{-2}$ s$^{-1}$)')
    ax.set_title('Absorbed PAR (field mean)')
    ax.grid(True, alpha=0.3)

    # Panel 3: Tleaf diurnal curve
    ax = axes[1, 0]
    T_air = np.array([r['T_air_C'] for r in hourly_results])
    ax.plot(hours, tleaf, 'o-', color='firebrick', lw=2, ms=4, label='Tleaf')
    ax.plot(hours, T_air, '--', color='steelblue', lw=1.5, label='Tair')
    ax.set_xlabel('Hour (UTC)')
    ax.set_ylabel('Temperature (C)')
    ax.set_title('Leaf vs Air Temperature')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 4: Solar zenith
    ax = axes[1, 1]
    ax.plot(hours, zen, 'o-', color='orange', lw=2, ms=4)
    ax.set_xlabel('Hour (UTC)')
    ax.set_ylabel('Solar Zenith (deg)')
    ax.set_title('Solar Zenith Angle')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plot_path = day_dir / 'diurnal_curve.png'
    fig.savefig(str(plot_path), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Plot: {plot_path}")


# ============================================================================
# Growth series (Mode B)
# ============================================================================
def run_growth_series(growth_days, timestep_min=30, enable_baleno=True):
    """Run Mode A at multiple growth stages.

    Args:
        growth_days: List of simulation days.
        timestep_min: Timestep in minutes.
        enable_baleno: Whether to run Baleno.

    Returns:
        dict mapping day -> daily result.
    """
    print(f"\n{'=' * 70}")
    print(f"GROWTH SERIES: {growth_days}")
    print(f"{'=' * 70}")

    series_results = {}
    for day in growth_days:
        result = run_single_day(
            day, timestep_min=timestep_min,
            enable_baleno=enable_baleno,
        )
        series_results[day] = result

    # --- Save series summary ---
    series_dir = OUTPUT_DIR / 'diurnal' / 'growth_series'
    series_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        'growth_days': growth_days,
        'timestep_min': timestep_min,
        'n_plants': N_PLANTS,
        'daily_An_mol_field_mean': {
            str(d): r['daily_An_mol_field_mean']
            for d, r in series_results.items()
        },
        'daily_An_mol_field_std': {
            str(d): r['daily_An_mol_field_std']
            for d, r in series_results.items()
        },
    }
    json_path = series_dir / 'growth_series_results.json'
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)

    # --- Growth series plot ---
    try:
        _plot_growth_series(series_dir, growth_days, series_results)
    except Exception as e:
        print(f"  WARNING: Growth series plot failed: {e}")

    print(f"\n{'=' * 70}")
    print(f"GROWTH SERIES COMPLETE")
    for d in growth_days:
        An_mean = series_results[d]['daily_An_mol_field_mean']
        An_std = series_results[d]['daily_An_mol_field_std']
        print(f"  Day {d:>3}: {An_mean:.6f} +/- {An_std:.6f} mol CO2/plant/day")
    print(f"{'=' * 70}")

    return series_results


def run_production_series(growth_days, timestep_min=60, enable_baleno=True,
                          iterate_gs=True, gs_max_iterations=6,
                          gs_tolerance=0.05, gs_damping_alpha=0.6,
                          carbon_method='auto', run_agroc_fortran=False,
                          resume=False):
    """Run full production diurnal campaign: DART + Baleno + gs + carbon + AgroC.

    Like run_growth_series() but calls run_single_day_with_carbon() per day,
    supports checkpointing/resume for multi-day runs, and optionally runs
    AgroC Fortran after all days complete.

    GDD is accumulated across the growth series using daily mean temperatures
    from the met forcing profile, providing proper thermal-time-based DVS
    for carbon partitioning.

    Args:
        growth_days: List of simulation days.
        timestep_min: Timestep in minutes.
        enable_baleno: Run Baleno energy balance per timestep.
        iterate_gs: Use iterative Tuzet-Baleno gs coupling.
        gs_max_iterations: Max iterations for gs convergence.
        gs_tolerance: Relative convergence threshold.
        gs_damping_alpha: Under-relaxation factor for gs.
        carbon_method: 'auto', 'phloem', or 'dvs'.
        run_agroc_fortran: If True, run AgroC Fortran after all days.
        resume: If True, skip already-completed days from checkpoint.

    Returns:
        dict mapping day -> daily result.
    """
    from ..carbon.dvs_partitioning import gdd_at_day, dvs_from_gdd, _dvs_from_day

    print(f"\n{'=' * 70}")
    print(f"PRODUCTION SERIES: {growth_days}")
    print(f"  Carbon: {carbon_method}, AgroC Fortran: {run_agroc_fortran}, "
          f"Resume: {resume}")
    print(f"{'=' * 70}")

    # --- Checkpoint logic ---
    checkpoint_dir = OUTPUT_DIR / 'diurnal'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / 'production_checkpoint.json'

    completed_days = []
    daily_summaries = {}
    if resume and checkpoint_path.exists():
        with open(checkpoint_path) as f:
            ckpt = json.load(f)
        completed_days = ckpt.get('completed_days', [])
        daily_summaries = ckpt.get('daily_summaries', {})
        print(f"  Resumed from checkpoint: {len(completed_days)} days completed "
              f"({completed_days})")

    # --- Run each day ---
    series_results = {}
    all_agroc_timesteps = []

    for day in growth_days:
        if day in completed_days:
            print(f"\n  [SKIP] Day {day} already completed (checkpoint)")
            continue

        # Compute GDD from real daily met data (falls back to None → calendar DVS)
        gdd_accumulated = gdd_at_day(day)
        if gdd_accumulated is not None:
            dvs_gdd = dvs_from_gdd(gdd_accumulated)
            dvs_cal = _dvs_from_day(day)
            print(f"\n  GDD={gdd_accumulated:.1f} -> DVS={dvs_gdd:.3f} "
                  f"(calendar-day DVS would be {dvs_cal:.3f})")
        else:
            print(f"\n  No daily met data -> using calendar-day DVS")

        # Extract warm-start from previous day's checkpoint
        ws = None
        prev_days = [d for d in completed_days if d < day]
        if prev_days:
            prev_summary = daily_summaries.get(str(prev_days[-1]), {})
            if prev_summary.get('C_ST_mean') is not None:
                ws = {
                    'C_ST_mean': prev_summary['C_ST_mean'],
                    'C_ST_min': prev_summary['C_ST_min'],
                    'C_ST_max': prev_summary['C_ST_max'],
                }

        result = run_single_day_with_carbon(
            day, timestep_min=timestep_min,
            enable_baleno=enable_baleno,
            iterate_gs=iterate_gs,
            gs_max_iterations=gs_max_iterations,
            gs_tolerance=gs_tolerance,
            gs_damping_alpha=gs_damping_alpha,
            carbon_method=carbon_method,
            warm_start=ws,
            gdd_accumulated=gdd_accumulated,
        )
        series_results[day] = result

        # Collect AgroC timestep
        if result.get('daily_agroc_ts') is not None:
            all_agroc_timesteps.append(result['daily_agroc_ts'])

        # Save checkpoint after each day (include C_ST stats for warm-start)
        completed_days.append(day)
        carbon_result = result.get('daily_carbon') or {}
        daily_summaries[str(day)] = {
            'An_field_mean': result.get('daily_An_mol_field_mean', 0.0),
            'carbon_source': carbon_result.get('partitioning_source', 'none'),
            'C_ST_mean': carbon_result.get('C_ST_mean'),
            'C_ST_min': carbon_result.get('C_ST_min'),
            'C_ST_max': carbon_result.get('C_ST_max'),
            'gdd_accumulated': gdd_accumulated,
        }
        ckpt_data = {
            'completed_days': completed_days,
            'growth_days': growth_days,
            'timestep_min': timestep_min,
            'daily_summaries': daily_summaries,
        }
        with open(checkpoint_path, 'w') as f:
            json.dump(ckpt_data, f, indent=2)
        print(f"  Checkpoint saved: {len(completed_days)}/{len(growth_days)} days")

    # --- Write combined coupling CSV ---
    if all_agroc_timesteps:
        from ..agroc import export_coupling_csv
        series_dir = OUTPUT_DIR / 'diurnal' / 'production'
        series_dir.mkdir(parents=True, exist_ok=True)
        csv_path = series_dir / 'coupling.csv'
        n_layers = all_agroc_timesteps[0].get('n_layers', 20)
        export_coupling_csv(all_agroc_timesteps, csv_path, n_layers)
        print(f"\n  Combined coupling CSV: {csv_path}")

        # --- Optional AgroC Fortran run ---
        if run_agroc_fortran:
            try:
                from ..agroc.run import (
                    get_agroc_src, prepare_agroc_workdir, run_agroc,
                    validate_agroc_outputs,
                )
                agroc_src = get_agroc_src()
                agroc_out = series_dir / 'agroc_run'
                agroc_out.mkdir(parents=True, exist_ok=True)
                workdir = prepare_agroc_workdir(agroc_src, agroc_out, csv_path)
                proc = run_agroc(workdir, timeout=600)
                if proc.returncode == 0:
                    validation = validate_agroc_outputs(workdir, csv_path)
                    print(f"  AgroC: {'PASSED' if validation['passed'] else 'WARNINGS'}")
                else:
                    print(f"  AgroC FAILED (exit code {proc.returncode})")
            except Exception as e:
                print(f"  AgroC error: {e}")

    # --- Save production summary ---
    series_dir = OUTPUT_DIR / 'diurnal' / 'production'
    series_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        'growth_days': growth_days,
        'timestep_min': timestep_min,
        'n_plants': N_PLANTS,
        'carbon_method': carbon_method,
        'iterate_gs': iterate_gs,
        'enable_baleno': enable_baleno,
        'daily_summaries': daily_summaries,
        'n_agroc_timesteps': len(all_agroc_timesteps),
    }
    json_path = series_dir / 'production_summary.json'
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary: {json_path}")

    # --- Growth series plot (reuse existing) ---
    # Only plot days that were actually run this session
    if series_results:
        try:
            _plot_growth_series(series_dir, list(series_results.keys()),
                                series_results)
        except Exception as e:
            print(f"  WARNING: Growth series plot failed: {e}")

    print(f"\n{'=' * 70}")
    print(f"PRODUCTION SERIES COMPLETE")
    print(f"  Days completed: {len(completed_days)}/{len(growth_days)}")
    for d in growth_days:
        s = daily_summaries.get(str(d), {})
        An = s.get('An_field_mean', 0.0)
        src = s.get('carbon_source', 'N/A')
        print(f"  Day {d:>3}: An={An:.6f} mol/plant/day  [{src}]")
    if all_agroc_timesteps:
        print(f"  AgroC timesteps: {len(all_agroc_timesteps)}")
    print(f"{'=' * 70}")

    return series_results


def _plot_growth_series(series_dir, growth_days, series_results):
    """Plot daily An vs growth day with error bars."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    days = np.array(growth_days)
    An_mean = np.array([series_results[d]['daily_An_mol_field_mean']
                        for d in growth_days]) * 1000
    An_std = np.array([series_results[d]['daily_An_mol_field_std']
                       for d in growth_days]) * 1000

    fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')
    ax.errorbar(days, An_mean, yerr=An_std, fmt='o-', color='forestgreen',
                lw=2, ms=8, capsize=5, elinewidth=1.5)
    ax.set_xlabel('Growth Day (days since sowing)')
    ax.set_ylabel('Daily An (mmol CO$_2$ plant$^{-1}$ day$^{-1}$)')
    ax.set_title(f'Growth Series: Daily Carbon Assimilation ({N_PLANTS} plants)')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = series_dir / 'growth_series_curve.png'
    fig.savefig(str(plot_path), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Growth series plot: {plot_path}")


# ============================================================================
# Uniform baseline (no DART, no Baleno)
# ============================================================================
def run_single_day_uniform(sim_day, timestep_min=30, met_csv=None):
    """Run diurnal cycle with uniform clearsky PAR and Tleaf=Tair.

    Baseline mode: no DART radiative transfer, no Baleno energy balance.
    All leaf segments receive the same scalar clearsky PAR and Tair.
    Everything else identical to run_single_day().
    """
    calendar_date = sim_day_to_date(sim_day, SOWING_DATE)
    print(f"\n{'=' * 70}")
    print(f"DIURNAL LOOP (UNIFORM BASELINE): Day {sim_day} ({calendar_date})")
    print(f"  Timestep: {timestep_min} min, PAR: clearsky, Tleaf: Tair")
    print(f"  Plants: {N_PLANTS} unique realizations "
          f"(seeds {FIELD_SEED}-{FIELD_SEED + N_PLANTS - 1})")
    print(f"{'=' * 70}")

    # --- Solar positions ---
    solar_df = get_solar_positions(
        calendar_date, LAT, LON, freq=f'{timestep_min}min')
    n_daylight = len(solar_df)
    print(f"\n  Solar positions: {n_daylight} daylight timesteps")
    if n_daylight == 0:
        print("  ERROR: No daylight hours!")
        return {'hourly': [], 'daily_An_mol_field_mean': 0.0,
                'daily_An_mol_per_plant': [0.0] * N_PLANTS}

    sunrise = solar_df.index[0]
    sunset = solar_df.index[-1]
    print(f"  Sunrise: {sunrise.strftime('%H:%M')} UTC, "
          f"Sunset: {sunset.strftime('%H:%M')} UTC")

    # --- Met forcing ---
    if met_csv:
        from ..utils.met_forcing import load_met_csv
        met = load_met_csv(met_csv)
    else:
        met = diurnal_met_profile(calendar_date, LAT, LON,
                                   freq=f'{timestep_min}min',
                                   **_real_met_kwargs(sim_day))
    print(f"  Met: T={met['T_air_C'].min():.1f}-{met['T_air_C'].max():.1f} C, "
          f"RH={met['RH'].min():.0%}-{met['RH'].max():.0%}")

    # --- Setup output directory ---
    day_dir = OUTPUT_DIR / 'diurnal_uniform' / f'day{sim_day}'
    day_dir.mkdir(parents=True, exist_ok=True)

    # --- Grow 9 plants + export meshes (for segment count + comparison) ---
    prospect_params = get_prospect_params(sim_day)
    log_consistency(sim_day)
    setup = setup_plants_and_meshes(sim_day, day_dir)

    # --- Diurnal loop (no DART, no Baleno) ---
    print(f"\n{'=' * 70}")
    print(f"DIURNAL LOOP (UNIFORM): {n_daylight} timesteps x {N_PLANTS} plants")
    print(f"{'=' * 70}")

    hourly_results = []
    for step_i, (ts_time, ts_row) in enumerate(solar_df.iterrows()):
        sun_zen = ts_row['apparent_zenith']
        sun_azi = ts_row['azimuth']
        ts_label = ts_time.strftime('%H:%M')

        print(f"\n  [{step_i+1}/{n_daylight}] {ts_label} UTC -- "
              f"zen={sun_zen:.1f}, azi={sun_azi:.1f}")

        t_step = time.time()

        # --- Met conditions ---
        if ts_time in met.index:
            met_row = met.loc[ts_time]
        else:
            idx = met.index.get_indexer([ts_time], method='nearest')[0]
            met_row = met.iloc[idx]

        T_air_C = float(met_row['T_air_C'])
        rh = float(met_row['RH'])
        wind_ms = float(met_row['wind_ms'])

        # --- Clearsky PAR as scalar for all segments ---
        clearsky_par_wm2 = get_clearsky_par(ts_time, LAT, LON)
        par_umol = clearsky_par_wm2 * 4.57  # W/m2 -> umol/m2/s

        print(f"    Clearsky PAR: {clearsky_par_wm2:.1f} W/m2 = "
              f"{par_umol:.1f} umol/m2/s, Tleaf=Tair={T_air_C:.1f}C")

        # --- Per-plant photosynthesis (scalar PAR + Tair) ---
        per_plant_An = [0.0] * N_PLANTS
        for pi in range(N_PLANTS):
            seed = FIELD_SEED + pi
            plant_ts = grow_plant(
                XML_PATH, simulation_time=sim_day,
                enable_photosynthesis=True, seed=seed,
            )
            ps_result = run_photosynthesis_solve(
                plant_ts, sim_day,
                par=par_umol, tleaf=T_air_C,
                label=f"uniform_ts_{ts_label}_p{pi}",
                rh=rh, soil_psi_cm=-500.0,
            )
            if ps_result is not None:
                per_plant_An[pi] = ps_result['An_total_mmol']

        An_field_mean = float(np.mean(per_plant_An))
        An_field_std = float(np.std(per_plant_An))
        print(f"    An: field mean={An_field_mean:.3f} "
              f"+/-{An_field_std:.3f} mmol CO2/d")

        # --- Per-timestep DVS carbon tracking ---
        dvs_carbon = None
        if An_field_mean > 0:
            try:
                from ..carbon.dvs_partitioning import partition_carbon_dvs
                dvs_carbon = partition_carbon_dvs(
                    An_field_mean, sim_day, Tair_C=T_air_C)
            except Exception as e:
                print(f"    DVS carbon tracking error: {e}")

        step_time = time.time() - t_step
        print(f"    Step time: {step_time:.1f}s")

        hourly_row = {
            'time_utc': ts_label,
            'zenith': float(sun_zen),
            'azimuth': float(sun_azi),
            'T_air_C': T_air_C,
            'RH': rh,
            'wind_ms': wind_ms,
            'clearsky_par_Wm2': clearsky_par_wm2,
            'mean_apar_umol': par_umol,   # uniform = clearsky
            'dart_mean_apar': 'N/A',
            'mean_tleaf_C': T_air_C,      # Tleaf = Tair
            'baleno_ok': False,
            'An_field_mean_mmol_d': An_field_mean,
            'An_field_std_mmol_d': An_field_std,
        }
        if dvs_carbon is not None:
            hourly_row['Rm_dvs_mmol'] = dvs_carbon['Rm_total_mmol']
            hourly_row['Rg_dvs_mmol'] = dvs_carbon['Rg_total_mmol']
            hourly_row['FR_leaf_dvs'] = dvs_carbon['FR_leaf']
            hourly_row['FR_root_dvs'] = dvs_carbon['FR_root']
        for pi in range(N_PLANTS):
            hourly_row[f'An_p{pi}'] = per_plant_An[pi]
        hourly_results.append(hourly_row)

    # --- Integrate daily carbon (per-plant) ---
    daily_An_per_plant = _integrate_daily_per_plant(
        hourly_results, timestep_min)
    daily_An_field_mean = float(np.mean(daily_An_per_plant))
    daily_An_field_std = float(np.std(daily_An_per_plant))

    # --- Save results ---
    _save_diurnal_results(day_dir, sim_day, calendar_date, hourly_results,
                           daily_An_per_plant, daily_An_field_mean,
                           daily_An_field_std, timestep_min)

    print(f"\n{'=' * 70}")
    print(f"DAY {sim_day} COMPLETE (UNIFORM BASELINE)")
    print(f"  Timesteps: {len(hourly_results)}")
    print(f"  Daily An per plant: {[f'{a:.6f}' for a in daily_An_per_plant]}")
    print(f"  Daily An field mean: {daily_An_field_mean:.6f} mol CO2/plant/day")
    print(f"  Daily An field std:  {daily_An_field_std:.6f} mol CO2/plant/day")
    if daily_An_field_mean > 0:
        cv_pct = daily_An_field_std / daily_An_field_mean * 100
        print(f"  Field CV: {cv_pct:.1f}%")
    print(f"  Output: {day_dir}")
    print(f"{'=' * 70}")

    return {
        'hourly': hourly_results,
        'daily_An_mol_per_plant': daily_An_per_plant,
        'daily_An_mol_field_mean': daily_An_field_mean,
        'daily_An_mol_field_std': daily_An_field_std,
        'daily_An_mol': daily_An_field_mean,
    }


def run_single_day_uniform_with_carbon(sim_day, timestep_min=30,
                                        met_csv=None, carbon_method='auto',
                                        warm_start=None,
                                        gdd_accumulated=None):
    """Run uniform diurnal loop + daily carbon partitioning and AgroC export.

    Wraps run_single_day_uniform() the same way run_single_day_with_carbon()
    wraps run_single_day(). Identical carbon partitioning + AgroC export.
    """
    # 1. Run uniform diurnal photosynthesis loop
    result = run_single_day_uniform(
        sim_day, timestep_min=timestep_min, met_csv=met_csv,
    )

    daily_An_mol = result.get('daily_An_mol_field_mean', 0.0)
    daily_An_mmol = daily_An_mol * 1000.0

    if daily_An_mmol <= 0:
        result['daily_carbon'] = None
        result['daily_agroc_ts'] = None
        return result

    # 2. Grow center plant with roots for carbon partitioning
    from ..growth.grow import run_photosynthesis, extract_lai_profile
    from ..carbon import solve_carbon_partitioning
    from ..agroc import export_agroc_timestep

    center_seed = FIELD_SEED + CENTER_PLANT_IDX
    plant = grow_plant(XML_PATH, simulation_time=sim_day,
                       min_stem_nodes=50, min_leaf_nodes=20,
                       seed=center_seed, enable_photosynthesis=True)

    # 3. Run photosynthesis at peak conditions to get per-segment An shape
    day_dir = OUTPUT_DIR / 'diurnal_uniform' / f'day{sim_day}'
    day_dir.mkdir(parents=True, exist_ok=True)
    prefix = str(day_dir / f'carbon_photo_day{sim_day}')
    hm = run_photosynthesis(plant, sim_time=sim_day, output_prefix=prefix,
                            par_umol=1000.0, tair_c=25.0)

    if hm is None:
        result['daily_carbon'] = None
        result['daily_agroc_ts'] = None
        return result

    # 4. Scale per-segment An to match diurnal-integrated daily total
    An_leaf = np.array(hm.get_net_assimilation())
    An_peak_total = float(np.sum(An_leaf))
    if An_peak_total > 0:
        scale = daily_An_mol / An_peak_total
        An_leaf_scaled = An_leaf * scale
    else:
        An_leaf_scaled = An_leaf

    # 5. Carbon partitioning
    try:
        carbon = solve_carbon_partitioning(
            plant, An_leaf_scaled, Tair_C=25.0,
            method=carbon_method, day=sim_day,
            warm_start=warm_start,
            gdd_accumulated=gdd_accumulated,
        )
    except Exception as e:
        print(f"  Carbon partitioning error: {e}")
        carbon = None

    # 6. LAI + AgroC export
    lai = extract_lai_profile(plant, n_bins=10)
    agroc_ts = None
    if carbon is not None:
        try:
            agroc_ts = export_agroc_timestep(
                plant, hm, carbon, lai,
                day=sim_day, par_umol=1000.0, tair_c=25.0,
            )
        except Exception as e:
            print(f"  AgroC export error: {e}")

    result['daily_carbon'] = carbon
    result['daily_agroc_ts'] = agroc_ts
    return result


def run_growth_series_uniform(growth_days, timestep_min=30):
    """Run uniform baseline at multiple growth stages (no carbon).

    Same as run_growth_series() but with uniform clearsky PAR + Tleaf=Tair.
    """
    print(f"\n{'=' * 70}")
    print(f"GROWTH SERIES (UNIFORM BASELINE): {growth_days}")
    print(f"{'=' * 70}")

    series_results = {}
    for day in growth_days:
        result = run_single_day_uniform(day, timestep_min=timestep_min)
        series_results[day] = result

    # --- Save series summary ---
    series_dir = OUTPUT_DIR / 'diurnal_uniform' / 'growth_series'
    series_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        'growth_days': growth_days,
        'timestep_min': timestep_min,
        'n_plants': N_PLANTS,
        'mode': 'uniform_baseline',
        'daily_An_mol_field_mean': {
            str(d): r['daily_An_mol_field_mean']
            for d, r in series_results.items()
        },
        'daily_An_mol_field_std': {
            str(d): r['daily_An_mol_field_std']
            for d, r in series_results.items()
        },
    }
    json_path = series_dir / 'growth_series_results.json'
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)

    try:
        _plot_growth_series(series_dir, growth_days, series_results)
    except Exception as e:
        print(f"  WARNING: Growth series plot failed: {e}")

    print(f"\n{'=' * 70}")
    print(f"GROWTH SERIES COMPLETE (UNIFORM BASELINE)")
    for d in growth_days:
        An_mean = series_results[d]['daily_An_mol_field_mean']
        An_std = series_results[d]['daily_An_mol_field_std']
        print(f"  Day {d:>3}: {An_mean:.6f} +/- {An_std:.6f} mol CO2/plant/day")
    print(f"{'=' * 70}")

    return series_results


def run_production_series_uniform(growth_days, timestep_min=60,
                                   carbon_method='auto',
                                   run_agroc_fortran=False, resume=False):
    """Run uniform baseline production campaign with carbon + AgroC.

    Same as run_production_series() but using uniform clearsky PAR + Tleaf=Tair.
    No DART, no Baleno, no iterative gs. Output to diurnal_uniform/.

    GDD is accumulated across the growth series using daily mean temperatures
    from the met forcing profile.
    """
    from ..carbon.dvs_partitioning import gdd_at_day, dvs_from_gdd, _dvs_from_day

    print(f"\n{'=' * 70}")
    print(f"PRODUCTION SERIES (UNIFORM BASELINE): {growth_days}")
    print(f"  Carbon: {carbon_method}, AgroC Fortran: {run_agroc_fortran}, "
          f"Resume: {resume}")
    print(f"{'=' * 70}")

    # --- Checkpoint logic ---
    checkpoint_dir = OUTPUT_DIR / 'diurnal_uniform'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / 'production_checkpoint.json'

    completed_days = []
    daily_summaries = {}
    if resume and checkpoint_path.exists():
        with open(checkpoint_path) as f:
            ckpt = json.load(f)
        completed_days = ckpt.get('completed_days', [])
        daily_summaries = ckpt.get('daily_summaries', {})
        print(f"  Resumed from checkpoint: {len(completed_days)} days completed "
              f"({completed_days})")

    # --- Run each day ---
    series_results = {}
    all_agroc_timesteps = []

    for day in growth_days:
        if day in completed_days:
            print(f"\n  [SKIP] Day {day} already completed (checkpoint)")
            continue

        # Compute GDD from real daily met data
        gdd_accumulated = gdd_at_day(day)
        if gdd_accumulated is not None:
            dvs_gdd = dvs_from_gdd(gdd_accumulated)
            dvs_cal = _dvs_from_day(day)
            print(f"\n  GDD={gdd_accumulated:.1f} -> DVS={dvs_gdd:.3f} "
                  f"(calendar-day DVS would be {dvs_cal:.3f})")
        else:
            print(f"\n  No daily met data -> using calendar-day DVS")

        # Extract warm-start from previous day's checkpoint
        ws = None
        prev_days = [d for d in completed_days if d < day]
        if prev_days:
            prev_summary = daily_summaries.get(str(prev_days[-1]), {})
            if prev_summary.get('C_ST_mean') is not None:
                ws = {
                    'C_ST_mean': prev_summary['C_ST_mean'],
                    'C_ST_min': prev_summary['C_ST_min'],
                    'C_ST_max': prev_summary['C_ST_max'],
                }

        result = run_single_day_uniform_with_carbon(
            day, timestep_min=timestep_min,
            carbon_method=carbon_method,
            warm_start=ws,
            gdd_accumulated=gdd_accumulated,
        )
        series_results[day] = result

        # Collect AgroC timestep
        if result.get('daily_agroc_ts') is not None:
            all_agroc_timesteps.append(result['daily_agroc_ts'])

        # Save checkpoint after each day (include C_ST stats for warm-start)
        completed_days.append(day)
        carbon_result = result.get('daily_carbon') or {}
        daily_summaries[str(day)] = {
            'An_field_mean': result.get('daily_An_mol_field_mean', 0.0),
            'carbon_source': carbon_result.get('partitioning_source', 'none'),
            'C_ST_mean': carbon_result.get('C_ST_mean'),
            'C_ST_min': carbon_result.get('C_ST_min'),
            'C_ST_max': carbon_result.get('C_ST_max'),
            'gdd_accumulated': gdd_accumulated,
        }
        ckpt_data = {
            'completed_days': completed_days,
            'growth_days': growth_days,
            'timestep_min': timestep_min,
            'mode': 'uniform_baseline',
            'daily_summaries': daily_summaries,
        }
        with open(checkpoint_path, 'w') as f:
            json.dump(ckpt_data, f, indent=2)
        print(f"  Checkpoint saved: {len(completed_days)}/{len(growth_days)} days")

    # --- Write combined coupling CSV ---
    if all_agroc_timesteps:
        from ..agroc import export_coupling_csv
        series_dir = OUTPUT_DIR / 'diurnal_uniform' / 'production'
        series_dir.mkdir(parents=True, exist_ok=True)
        csv_path = series_dir / 'coupling.csv'
        n_layers = all_agroc_timesteps[0].get('n_layers', 20)
        export_coupling_csv(all_agroc_timesteps, csv_path, n_layers)
        print(f"\n  Combined coupling CSV: {csv_path}")

        # --- Optional AgroC Fortran run ---
        if run_agroc_fortran:
            try:
                from ..agroc.run import (
                    get_agroc_src, prepare_agroc_workdir, run_agroc,
                    validate_agroc_outputs,
                )
                agroc_src = get_agroc_src()
                agroc_out = series_dir / 'agroc_run'
                agroc_out.mkdir(parents=True, exist_ok=True)
                workdir = prepare_agroc_workdir(agroc_src, agroc_out, csv_path)
                proc = run_agroc(workdir, timeout=600)
                if proc.returncode == 0:
                    validation = validate_agroc_outputs(workdir, csv_path)
                    print(f"  AgroC: "
                          f"{'PASSED' if validation['passed'] else 'WARNINGS'}")
                else:
                    print(f"  AgroC FAILED (exit code {proc.returncode})")
            except Exception as e:
                print(f"  AgroC error: {e}")

    # --- Save production summary ---
    series_dir = OUTPUT_DIR / 'diurnal_uniform' / 'production'
    series_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        'growth_days': growth_days,
        'timestep_min': timestep_min,
        'n_plants': N_PLANTS,
        'carbon_method': carbon_method,
        'mode': 'uniform_baseline',
        'iterate_gs': False,
        'enable_baleno': False,
        'daily_summaries': daily_summaries,
        'n_agroc_timesteps': len(all_agroc_timesteps),
    }
    json_path = series_dir / 'production_summary.json'
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary: {json_path}")

    # --- Growth series plot ---
    if series_results:
        try:
            _plot_growth_series(series_dir, list(series_results.keys()),
                                series_results)
        except Exception as e:
            print(f"  WARNING: Growth series plot failed: {e}")

    print(f"\n{'=' * 70}")
    print(f"PRODUCTION SERIES COMPLETE (UNIFORM BASELINE)")
    print(f"  Days completed: {len(completed_days)}/{len(growth_days)}")
    for d in growth_days:
        s = daily_summaries.get(str(d), {})
        An = s.get('An_field_mean', 0.0)
        src = s.get('carbon_source', 'N/A')
        print(f"  Day {d:>3}: An={An:.6f} mol/plant/day  [{src}]")
    if all_agroc_timesteps:
        print(f"  AgroC timesteps: {len(all_agroc_timesteps)}")
    print(f"{'=' * 70}")

    return series_results


# ============================================================================
# Carbon-feedback growth mode
# ============================================================================
def compute_daily_an_uniform(plant, sim_day, tair_c=25.0):
    """Compute daily-integrated An for one plant using clearsky PAR.

    Uses peak-hour photosynthesis at clearsky max, scaled by a cosine
    diurnal integration factor to approximate full-day An.

    Args:
        plant: pb.MappedPlant (grown, with photosynthesis enabled).
        sim_day: Simulation day (days since sowing).
        tair_c: Air temperature [C].

    Returns:
        An_leaf: np.array, mol CO2/d per leaf segment (daily integrated).
    """
    from ..growth.grow import run_photosynthesis

    calendar_date = sim_day_to_date(sim_day, SOWING_DATE)

    # Peak clearsky PAR at solar noon
    solar_df = get_solar_positions(calendar_date, LAT, LON, freq='30min')
    if len(solar_df) == 0:
        return np.zeros(0)

    # Find peak PAR across daytime hours
    peak_par_wm2 = 0.0
    for ts_time, _ in solar_df.iterrows():
        par = get_clearsky_par(ts_time, LAT, LON)
        peak_par_wm2 = max(peak_par_wm2, par)

    if peak_par_wm2 <= 0:
        return np.zeros(0)

    peak_par_umol = peak_par_wm2 * 4.57

    # Run photosynthesis at peak PAR
    hm = run_photosynthesis(plant, sim_time=sim_day,
                            output_prefix=None,
                            par_umol=peak_par_umol, tair_c=tair_c)
    if hm is None:
        return np.zeros(0)

    An_peak = np.array(hm.get_net_assimilation())  # mol CO2/d per seg

    # Scale from peak instantaneous to daily integral.
    # Daylength hours * cosine integration factor (2/pi ~ 0.637).
    n_daylight = len(solar_df)
    daylength_hours = n_daylight * 0.5  # 30-min steps
    # Peak An is computed as if PAR persisted for 24h. Scale down:
    # daily_An = An_peak * (daylength_hours / 24) * (2/pi)
    day_fraction = (daylength_hours / 24.0) * (2.0 / np.pi)
    An_daily = An_peak * day_fraction

    return An_daily


def run_production_series_carbon(growth_days, timestep_min=60,
                                  enable_baleno=True, iterate_gs=True,
                                  gs_max_iterations=6, gs_tolerance=0.05,
                                  gs_damping_alpha=0.6,
                                  carbon_method='auto',
                                  run_agroc_fortran=False, resume=False):
    """Production series with carbon-feedback growth.

    Flow:
    1. Bootstrap: grow 9 plants parametrically to growth_days[0]
    2. Switch all plants to CWLimitedGrowth (gf=3)
    3. For each DART observation day D:
       a. Re-loft persistent plants -> DART RT -> Baleno -> diurnal An
       b. Carbon partitioning per plant (DART-informed)
       c. Daily stepping from D to D_next with carbon-limited growth:
          For each inter-day d:
            For each plant:
              - Uniform photosynthesis (clearsky PAR, Tleaf=Tair)
              - Phloem solve -> Rg_node -> CW_Gr -> simulate(1.0)
    4. Save checkpoint after each DART day

    Args:
        growth_days: List of DART observation days (sorted).
        timestep_min: Diurnal timestep [min] for DART days.
        enable_baleno: Run Baleno energy balance on DART days.
        iterate_gs: Iterative Tuzet-Baleno gs coupling on DART days.
        gs_max_iterations: Max gs iterations.
        gs_tolerance: Relative gs convergence threshold.
        gs_damping_alpha: Under-relaxation factor for gs.
        carbon_method: Carbon partitioning method ('auto', 'phloem', 'dvs').
        run_agroc_fortran: If True, run AgroC Fortran after completion.
        resume: If True, skip already-completed days from checkpoint.

    Returns:
        dict mapping day -> daily result.
    """
    from ..growth.grow import init_plant, run_photosynthesis, extract_lai_profile
    from ..growth.carbon_growth import (
        enable_cw_limited_growth, step_plant_carbon,
    )
    from ..carbon import solve_carbon_partitioning

    print(f"\n{'=' * 70}")
    print(f"PRODUCTION SERIES (CARBON FEEDBACK): {growth_days}")
    print(f"  Carbon: {carbon_method}, Baleno: {enable_baleno}, "
          f"iterate_gs: {iterate_gs}")
    print(f"{'=' * 70}")

    from ..carbon.dvs_partitioning import gdd_at_day, dvs_from_gdd

    # --- Checkpoint logic ---
    checkpoint_dir = OUTPUT_DIR / 'diurnal_carbon'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / 'production_checkpoint.json'

    completed_dart_days = []
    daily_summaries = {}
    per_plant_warm_starts = [{} for _ in range(N_PLANTS)]
    if resume and checkpoint_path.exists():
        with open(checkpoint_path) as f:
            ckpt = json.load(f)
        completed_dart_days = ckpt.get('completed_dart_days', [])
        daily_summaries = ckpt.get('daily_summaries', {})
        print(f"  Resumed from checkpoint: {len(completed_dart_days)} "
              f"DART days completed ({completed_dart_days})")

    # --- 1. Bootstrap: grow 9 plants parametrically to first DART day ---
    first_day = growth_days[0]
    print(f"\n  Bootstrapping {N_PLANTS} plants parametrically to day {first_day}...")

    persistent_plants = []
    for i in range(N_PLANTS):
        seed = FIELD_SEED + i
        plant = grow_plant(XML_PATH, simulation_time=first_day,
                           min_stem_nodes=50, min_leaf_nodes=20, seed=seed,
                           enable_photosynthesis=True)
        persistent_plants.append(plant)
        organs = plant.getOrgans()
        n_leaves = sum(1 for o in organs
                       if o.organType() == 4)  # pb.OrganTypes.leaf
        print(f"    Plant {i} (seed={seed}): {n_leaves} leaves, "
              f"{len(plant.getNodes())} nodes")

    # --- 2. Switch all plants to CWLimitedGrowth ---
    print(f"\n  Switching all plants to CWLimitedGrowth (gf=3)...")
    for plant in persistent_plants:
        enable_cw_limited_growth(plant)

    # --- 3. Process each DART observation day ---
    series_results = {}

    for day_idx, dart_day in enumerate(growth_days):
        if dart_day in completed_dart_days:
            print(f"\n  [SKIP] DART day {dart_day} already completed (checkpoint)")
            continue

        # Compute GDD from real daily met data
        gdd_accumulated = gdd_at_day(dart_day)
        if gdd_accumulated is not None:
            dvs_gdd = dvs_from_gdd(gdd_accumulated)
        else:
            dvs_gdd = None

        print(f"\n{'=' * 70}")
        print(f"DART OBSERVATION DAY {dart_day} "
              f"({day_idx+1}/{len(growth_days)})")
        if dvs_gdd is not None:
            print(f"  GDD={gdd_accumulated:.1f} -> DVS={dvs_gdd:.3f}")
        print(f"{'=' * 70}")

        # --- 3a. Re-loft persistent plants, run DART + photosynthesis ---
        day_dir = checkpoint_dir / f'day{dart_day}'
        day_dir.mkdir(parents=True, exist_ok=True)

        # Run the full diurnal loop using persistent plants for mesh export
        # but fresh plants for photosynthesis (CPlantBox photosynthesis needs
        # a clean HydraulicModel). The DART geometry comes from persistent
        # plants, capturing carbon-feedback structural differences.
        prospect_params = get_prospect_params(dart_day)
        log_consistency(dart_day)

        setup = setup_plants_and_meshes(
            dart_day, day_dir, plants=persistent_plants)
        dart_obj_paths = setup['dart_obj_paths']
        mappings = setup['mappings']
        grid_info = setup['grid_info']
        grid_path = setup['grid_path']

        # Run DART simulation on the carbon-feedback geometry
        calendar_date = sim_day_to_date(dart_day, SOWING_DATE)
        solar_df = get_solar_positions(
            calendar_date, LAT, LON, freq=f'{timestep_min}min')
        n_daylight = len(solar_df)

        if n_daylight == 0:
            print(f"  No daylight at day {dart_day}, skipping")
            series_results[dart_day] = {
                'daily_An_mol_per_plant': [0.0] * N_PLANTS,
                'daily_An_mol_field_mean': 0.0,
                'daily_An_mol_field_std': 0.0,
            }
            completed_dart_days.append(dart_day)
            continue

        met = diurnal_met_profile(calendar_date, LAT, LON,
                                   freq=f'{timestep_min}min',
                                   **_real_met_kwargs(dart_day))

        first_ts_time = solar_df.index[0]
        simu_name = f'cpb_carbon_day{dart_day}'

        print(f"\n  Creating multi-plant DART simulation: {simu_name}")
        simu = create_dart_simulation_multi(
            obj_paths=dart_obj_paths,
            mapping_json_paths=setup['dart_mapping_paths'],
            simu_name=simu_name,
            prospect_params=prospect_params,
            scene_size=SCENE_SIZE,
            grid_info=grid_info,
            par_bands=PAR_BANDS,
            field_filename=FIELD_FILENAME,
            calendar_date=calendar_date,
            hour_utc=first_ts_time.hour,
            minute_utc=first_ts_time.minute,
            lat=LAT, lon=LON,
            use_exact_date=True,
        )

        print(f"  Running full DART...")
        t0 = time.time()
        run_dart_full(simu, timeout=1200)
        print(f"  DART full run: {time.time() - t0:.1f}s")

        reindex_infos = read_ori_reindex_multi(simu, dart_obj_paths)
        if reindex_infos is None:
            print(f"  ERROR: No .ori reindex tables!")
            series_results[dart_day] = {
                'daily_An_mol_per_plant': [0.0] * N_PLANTS,
                'daily_An_mol_field_mean': 0.0,
                'daily_An_mol_field_std': 0.0,
            }
            completed_dart_days.append(dart_day)
            continue

        # Save per-plant reindex JSONs
        per_plant_reindex_paths = []
        for pi in range(N_PLANTS):
            ri = reindex_infos[pi]
            ri_path = day_dir / f'maize_day{dart_day}_p{pi}_reindex.json'
            ri_save = {
                'dart_to_obj': {str(k): v.tolist()
                                for k, v in ri['dart_to_obj'].items()},
                'group_names': ri['groups_sorted'],
                'group_offsets': {g: ri['group_offsets'][g]
                                  for g in ri['groups_sorted']},
            }
            with open(ri_path, 'w') as f:
                json.dump(ri_save, f, indent=2)
            per_plant_reindex_paths.append(ri_path)

        # Setup multi-plant Baleno (once per DART day)
        baleno_setup = None
        if enable_baleno:
            print(f"\n  Setting up Baleno (all {N_PLANTS} plants)...")
            try:
                baleno_setup = setup_baleno_full_multi(
                    obj_paths=[str(p) for p in dart_obj_paths],
                    mapping_json_paths=[str(p) for p in setup['dart_mapping_paths']],
                    reindex_json_paths=[str(p) for p in per_plant_reindex_paths],
                    grid_info_path=str(grid_path),
                    prospect_params=prospect_params,
                    scene_size=SCENE_SIZE,
                    dart_simu_name=f'cpb_carbon_day{dart_day}_eb',
                    baleno_simu_name=f'cpb_carbon_day{dart_day}_eb',
                    field_filename=FIELD_FILENAME,
                    calendar_date=calendar_date,
                    hour_utc=first_ts_time.hour,
                    minute_utc=first_ts_time.minute,
                    lat=LAT, lon=LON,
                    use_exact_date=True,
                )
                print(f"  Running initial Baleno _I full DART...")
                t0 = time.time()
                baleno_setup['simu_I'].run.full(timeout=1800)
                print(f"  Baleno _I full: {time.time() - t0:.1f}s")
            except Exception as e:
                import traceback
                print(f"\n  {'!' * 60}")
                print(f"  BALENO SETUP FAILED — Tleaf will default to Tair!")
                print(f"  Error: {e}")
                traceback.print_exc()
                print(f"  {'!' * 60}\n")
                baleno_setup = None

        # Diurnal loop: accumulate per-plant An across timesteps
        daily_An_per_plant_mmol = [0.0] * N_PLANTS
        hourly_results = []
        dt_day = timestep_min / (24 * 60)
        n_steps_done = 0

        for step_i, (ts_time, ts_row) in enumerate(solar_df.iterrows()):
            sun_zen = ts_row['apparent_zenith']
            sun_azi = ts_row.get('azimuth', 0.0)
            ts_label = ts_time.strftime('%H:%M')
            print(f"\n  [{step_i+1}/{n_daylight}] {ts_label} UTC")

            if ts_time in met.index:
                met_row = met.loc[ts_time]
            else:
                idx = met.index.get_indexer([ts_time], method='nearest')[0]
                met_row = met.iloc[idx]

            T_air_C = float(met_row['T_air_C'])
            T_air_K = float(met_row.get('T_air_K', T_air_C + 273.15))
            rh = float(met_row['RH'])
            wind_ms = float(met_row.get('wind_ms', 2.0))
            ea_hPa = float(met_row.get(
                'ea_hPa',
                rh * 6.112 * np.exp(17.67 * T_air_C / (T_air_C + 243.5))))

            # Update DART sun + re-run RT
            if step_i > 0:
                update_datetime_and_rerun(
                    simu, calendar_date, ts_time.hour, ts_time.minute)

            # Read per-plant aPAR
            all_plant_apar = read_and_aggregate_apar_multi(
                simu, mappings, reindex_infos)
            if all_plant_apar is None:
                print(f"    WARNING: aPAR read failed, skipping timestep")
                continue

            # Convert to physical umol
            n_par_bands = len(PAR_BANDS)
            clearsky_par_wm2 = get_clearsky_par(ts_time, LAT, LON)
            actual_par_per_band = clearsky_par_wm2 / n_par_bands

            per_plant_An_ts = [0.0] * N_PLANTS
            mean_par_umol_ts = 0.0
            all_par_umol = []
            all_tleaf_baleno = None
            mean_tleaf_ts = T_air_C
            baleno_ok_ts = False

            for pi in range(N_PLANTS):
                plant_p = persistent_plants[pi]
                n_leaf_segs = len(plant_p.getSegmentIds(4))

                apar_wm2 = all_plant_apar[pi] * actual_par_per_band
                par_umol = np.clip(apar_wm2 * 4.57, 0.0, 3000.0)

                if len(par_umol) != n_leaf_segs:
                    par_umol = float(np.mean(par_umol))

                if not np.isscalar(par_umol):
                    mean_par_umol_ts += float(np.mean(par_umol))
                else:
                    mean_par_umol_ts += par_umol
                all_par_umol.append(par_umol)

            mean_par_umol_ts /= max(N_PLANTS, 1)

            # Baleno + iterative coupling (if enabled)
            MIN_PAR_FOR_BALENO = 50.0
            # Adaptive timeout: shorter at low PAR where Baleno struggles
            baleno_timeout = 600 if clearsky_par_wm2 < 100.0 else 1800
            if (iterate_gs and baleno_setup is not None
                    and clearsky_par_wm2 >= MIN_PAR_FOR_BALENO):
                try:
                    from .iterative import run_iterative_coupling_multi

                    # Update Baleno datetime + atmosphere
                    if step_i > 0:
                        update_baleno_datetime_and_rerun_I(
                            baleno_setup['simu_I'], calendar_date,
                            ts_time.hour, ts_time.minute)
                    update_baleno_atmosphere(
                        baleno_setup['baleno_sim_dir'],
                        T_air_K=T_air_K, ea_hPa=ea_hPa, wind_ms=wind_ms,
                    )
                    from ..dart.baleno import BALENO_DIR
                    baleno_config_path = BALENO_DIR / 'resources' / 'config.ini'
                    import textwrap
                    baleno_config_path.write_text(textwrap.dedent(f"""\
                        [simulation]
                        user_data_path =
                        name = {baleno_setup['baleno_simu_name']}
                    """))

                    # Initial Baleno pass (Ball-Berry) for scene mapping
                    ok = run_baleno_subprocess(timeout=baleno_timeout)
                    if ok:
                        # Get initial Tleaf
                        init_tleaf = read_baleno_tleaf_multi(
                            baleno_setup['baleno_sim_dir'],
                            [str(p) for p in setup['dart_mapping_paths']],
                            [str(p) for p in per_plant_reindex_paths],
                            N_PLANTS, tair_c=T_air_C,
                        )

                        iter_results = run_iterative_coupling_multi(
                            persistent_plants, dart_day,
                            par_umol_per_plant=all_par_umol,
                            mapping_json_paths=[str(p) for p in setup['dart_mapping_paths']],
                            reindex_json_paths=[str(p) for p in per_plant_reindex_paths],
                            baleno_sim_dir=str(baleno_setup['baleno_sim_dir']),
                            baleno_simu_name=baleno_setup['baleno_simu_name'],
                            n_plants=N_PLANTS,
                            max_iterations=gs_max_iterations,
                            gs_tolerance=gs_tolerance,
                            damping_alpha=gs_damping_alpha,
                            soil_psi_cm=-500.0,
                            tair_c=T_air_C, rh=rh,
                            initial_tleaf=init_tleaf,
                        )
                        if iter_results is not None:
                            for pi in range(N_PLANTS):
                                per_plant_An_ts[pi] = iter_results[pi]['an_total_mmol']
                            tleaf_means = [float(np.mean(iter_results[pi]['tleaf_per_segment']))
                                           for pi in range(N_PLANTS)]
                            mean_tleaf_ts = float(np.mean(tleaf_means))
                            baleno_ok_ts = True
                        else:
                            # Fall through to simple photosynthesis
                            baleno_ok_ts = False
                    else:
                        print(f"    Initial Baleno failed, using Tleaf=Tair")
                except Exception as e:
                    import traceback
                    print(f"    Iterative coupling error: {e}")
                    traceback.print_exc()

            elif (baleno_setup is not None
                  and clearsky_par_wm2 >= MIN_PAR_FOR_BALENO):
                # Non-iterative Baleno: single pass for Tleaf
                try:
                    if step_i > 0:
                        update_baleno_datetime_and_rerun_I(
                            baleno_setup['simu_I'], calendar_date,
                            ts_time.hour, ts_time.minute)
                    update_baleno_atmosphere(
                        baleno_setup['baleno_sim_dir'],
                        T_air_K=T_air_K, ea_hPa=ea_hPa, wind_ms=wind_ms,
                    )
                    from ..dart.baleno import BALENO_DIR
                    baleno_config_path = BALENO_DIR / 'resources' / 'config.ini'
                    import textwrap
                    baleno_config_path.write_text(textwrap.dedent(f"""\
                        [simulation]
                        user_data_path =
                        name = {baleno_setup['baleno_simu_name']}
                    """))
                    ok = run_baleno_subprocess(timeout=baleno_timeout)
                    if ok:
                        all_tleaf_baleno = read_baleno_tleaf_multi(
                            baleno_setup['baleno_sim_dir'],
                            [str(p) for p in setup['dart_mapping_paths']],
                            [str(p) for p in per_plant_reindex_paths],
                            N_PLANTS, tair_c=T_air_C,
                        )
                        if all_tleaf_baleno is not None:
                            baleno_ok_ts = True
                            tleaf_means = [float(np.mean(t)) for t in all_tleaf_baleno]
                            mean_tleaf_ts = float(np.mean(tleaf_means))
                except Exception as e:
                    print(f"    Baleno error: {e}")

            # Simple photosynthesis fallback:
            # - iterate_gs=True but coupling failed -> use Tair
            # - iterate_gs=False, baleno_ok -> use Baleno Tleaf
            # - iterate_gs=False, no baleno -> use Tair
            # Skip if iterative coupling already produced results
            if not (iterate_gs and baleno_ok_ts):
                for pi in range(N_PLANTS):
                    plant_p = persistent_plants[pi]
                    tleaf_pi = T_air_C
                    if (baleno_ok_ts
                            and all_tleaf_baleno is not None
                            and pi < len(all_tleaf_baleno)):
                        tleaf_pi = all_tleaf_baleno[pi]

                    result = run_photosynthesis_solve(
                        plant_p, dart_day,
                        par=all_par_umol[pi], tleaf=tleaf_pi,
                        label=f"carbon_ts_{ts_label}_p{pi}",
                        rh=rh, soil_psi_cm=-500.0,
                    )
                    if result is not None:
                        per_plant_An_ts[pi] = result['An_total_mmol']

            # Trapezoidal integration
            for pi in range(N_PLANTS):
                daily_An_per_plant_mmol[pi] += per_plant_An_ts[pi] * dt_day
            n_steps_done += 1

            # Collect hourly data for CSV/plots
            An_field_mean_ts = float(np.mean(per_plant_An_ts))
            An_field_std_ts = float(np.std(per_plant_An_ts))
            field_mean_apar = float(np.mean([
                float(np.mean(all_plant_apar[pi]))
                for pi in range(N_PLANTS)
            ])) if all_plant_apar is not None else 0.0

            hourly_row = {
                'time_utc': ts_label,
                'zenith': float(sun_zen),
                'azimuth': float(sun_azi),
                'T_air_C': T_air_C,
                'RH': rh,
                'wind_ms': wind_ms,
                'clearsky_par_Wm2': clearsky_par_wm2,
                'mean_apar_umol': mean_par_umol_ts,
                'dart_mean_apar': field_mean_apar,
                'mean_tleaf_C': mean_tleaf_ts,
                'baleno_ok': baleno_ok_ts,
                'An_field_mean_mmol_d': An_field_mean_ts,
                'An_field_std_mmol_d': An_field_std_ts,
            }
            for pi in range(N_PLANTS):
                hourly_row[f'An_p{pi}'] = per_plant_An_ts[pi]
            hourly_results.append(hourly_row)

        # Cleanup Baleno after diurnal loop
        if baleno_setup is not None:
            try:
                restore_config_files(baleno_setup['backups'])
            except Exception:
                pass

        # Integrate daily per-plant An
        daily_An_per_plant = _integrate_daily_per_plant(
            hourly_results, timestep_min)
        daily_An_per_plant_mol = daily_An_per_plant
        daily_An_field_mean = float(np.mean(daily_An_per_plant_mol))
        daily_An_field_std = float(np.std(daily_An_per_plant_mol))

        # Compute daily mean temperature from hourly results
        daily_mean_tair = float(np.mean([r['T_air_C'] for r in hourly_results]))

        print(f"\n  Day {dart_day}: An field mean = {daily_An_field_mean:.6f} "
              f"+/- {daily_An_field_std:.6f} mol CO2/plant/day "
              f"({n_steps_done} timesteps, mean Tair={daily_mean_tair:.1f}°C)")

        # Save CSV, JSON summary, and diurnal curve plot
        try:
            _save_diurnal_results(
                day_dir, dart_day, calendar_date, hourly_results,
                daily_An_per_plant, daily_An_field_mean,
                daily_An_field_std, timestep_min)
        except Exception as e:
            print(f"  WARNING: Failed to save diurnal results: {e}")

        # --- 3b. Per-plant carbon partitioning on DART day ---
        per_plant_carbon = []
        for pi in range(N_PLANTS):
            plant = persistent_plants[pi]
            daily_An_mol = daily_An_per_plant_mol[pi]
            if daily_An_mol <= 0:
                per_plant_carbon.append(None)
                continue

            # Run photosynthesis at peak on persistent plant to get
            # per-segment An shape for carbon partitioning
            carbon_prefix = str(day_dir / f'carbon_photo_day{dart_day}_p{pi}')
            hm = run_photosynthesis(
                plant, sim_time=dart_day,
                output_prefix=carbon_prefix,
                par_umol=1000.0, tair_c=daily_mean_tair)

            if hm is None:
                per_plant_carbon.append(None)
                continue

            An_leaf = np.array(hm.get_net_assimilation())
            An_peak_total = float(np.sum(An_leaf))
            if An_peak_total > 0:
                An_leaf_scaled = An_leaf * (daily_An_mol / An_peak_total)
            else:
                An_leaf_scaled = An_leaf

            try:
                carbon = solve_carbon_partitioning(
                    plant, An_leaf_scaled, Tair_C=daily_mean_tair,
                    method=carbon_method, day=dart_day,
                    warm_start=per_plant_warm_starts[pi] or None,
                    gdd_accumulated=gdd_accumulated,
                )
                per_plant_carbon.append(carbon)
                per_plant_warm_starts[pi] = {
                    'C_ST_mean': carbon.get('C_ST_mean'),
                    'C_ST_min': carbon.get('C_ST_min'),
                    'C_ST_max': carbon.get('C_ST_max'),
                }
            except Exception as e:
                print(f"    Plant {pi} carbon partitioning error: {e}")
                per_plant_carbon.append(None)

        # --- 3c. Daily stepping from dart_day to next dart_day ---
        if day_idx + 1 < len(growth_days):
            next_dart_day = growth_days[day_idx + 1]
        else:
            next_dart_day = dart_day + 1  # just one step after last DART day

        n_inter_days = next_dart_day - dart_day - 1
        if n_inter_days > 0:
            print(f"\n  Inter-day stepping: day {dart_day+1} to "
                  f"{next_dart_day-1} ({n_inter_days} days)")

        for inter_day in range(dart_day + 1, next_dart_day):
            inter_gdd = gdd_at_day(inter_day)

            print(f"    Day {inter_day}", end='')
            if inter_gdd is not None:
                print(f" (GDD={inter_gdd:.0f})", end='')
            print(":", end='')
            for pi in range(N_PLANTS):
                plant = persistent_plants[pi]

                # Uniform photosynthesis for this inter-day
                An_daily = compute_daily_an_uniform(
                    plant, inter_day, tair_c=daily_mean_tair)

                if len(An_daily) == 0:
                    print(f" p{pi}:skip", end='')
                    continue

                # Carbon-limited growth step
                try:
                    result = step_plant_carbon(
                        plant, An_daily, sim_day=inter_day,
                        tair_c=daily_mean_tair, dt=1.0,
                        warm_start=per_plant_warm_starts[pi] or None,
                        gdd_accumulated=inter_gdd,
                    )
                    per_plant_warm_starts[pi] = {
                        'C_ST_mean': result.get('C_ST_mean'),
                        'C_ST_min': result.get('C_ST_min'),
                        'C_ST_max': result.get('C_ST_max'),
                    }
                    print(f" p{pi}:ok", end='')
                except Exception as e:
                    print(f" p{pi}:err({e})", end='')

            print()  # newline after all plants

        # --- 3d. Carbon-limited growth step on DART day itself ---
        # Use the DART-informed An for this day's growth step
        print(f"  DART-day growth step (day {dart_day}):")
        for pi in range(N_PLANTS):
            plant = persistent_plants[pi]
            daily_An_mol = daily_An_per_plant_mol[pi]
            if daily_An_mol <= 0:
                continue

            # Get per-segment An shape from persistent plant
            hm = run_photosynthesis(
                plant, sim_time=dart_day,
                output_prefix=None, par_umol=1000.0, tair_c=daily_mean_tair)
            if hm is None:
                continue

            An_leaf = np.array(hm.get_net_assimilation())
            An_peak_total = float(np.sum(An_leaf))
            if An_peak_total > 0:
                An_leaf_scaled = An_leaf * (daily_An_mol / An_peak_total)
            else:
                An_leaf_scaled = An_leaf

            try:
                result = step_plant_carbon(
                    plant, An_leaf_scaled, sim_day=dart_day,
                    tair_c=daily_mean_tair, dt=1.0,
                    warm_start=per_plant_warm_starts[pi] or None,
                    gdd_accumulated=gdd_accumulated,
                )
                per_plant_warm_starts[pi] = {
                    'C_ST_mean': result.get('C_ST_mean'),
                    'C_ST_min': result.get('C_ST_min'),
                    'C_ST_max': result.get('C_ST_max'),
                }
            except Exception as e:
                print(f"    Plant {pi} growth step error: {e}")

        # --- Append carbon partitioning to daily_summary.json ---
        try:
            json_path = day_dir / 'daily_summary.json'
            if json_path.exists():
                with open(json_path) as f:
                    daily_json = json.load(f)
            else:
                daily_json = {}
            carbon_per_plant = []
            for pi, c in enumerate(per_plant_carbon):
                if c is not None:
                    carbon_per_plant.append({
                        'plant': pi,
                        'Rm_total_mmol': c.get('Rm_total_mmol'),
                        'Rg_total_mmol': c.get('Rg_total_mmol'),
                        'FR_leaf': c.get('FR_leaf'),
                        'FR_stem': c.get('FR_stem'),
                        'FR_root': c.get('FR_root'),
                        'FR_storage': c.get('FR_storage'),
                        'growth_mmol_d': c.get('growth_mmol_d'),
                        'carbon_balance_error': c.get('carbon_balance_error'),
                        'DVS': c.get('DVS'),
                        'partitioning_source': c.get('partitioning_source'),
                    })
            daily_json['carbon_partitioning'] = carbon_per_plant
            with open(json_path, 'w') as f:
                json.dump(daily_json, f, indent=2)
            print(f"  Carbon partitioning added to {json_path}")
        except Exception as e:
            print(f"  WARNING: Failed to save carbon partitioning: {e}")

        # --- Save results ---
        series_results[dart_day] = {
            'daily_An_mol_per_plant': daily_An_per_plant_mol,
            'daily_An_mol_field_mean': daily_An_field_mean,
            'daily_An_mol_field_std': daily_An_field_std,
        }

        # Save checkpoint
        completed_dart_days.append(dart_day)
        daily_summaries[str(dart_day)] = {
            'An_field_mean': daily_An_field_mean,
            'An_per_plant': daily_An_per_plant_mol,
            'n_timesteps': n_steps_done,
            'gdd_accumulated': gdd_accumulated,
        }
        ckpt_data = {
            'completed_dart_days': completed_dart_days,
            'growth_days': growth_days,
            'timestep_min': timestep_min,
            'growth_mode': 'carbon',
            'daily_summaries': daily_summaries,
        }
        with open(checkpoint_path, 'w') as f:
            json.dump(ckpt_data, f, indent=2)
        print(f"  Checkpoint saved: {len(completed_dart_days)}/{len(growth_days)}")

    # --- Final summary ---
    summary = {
        'growth_days': growth_days,
        'timestep_min': timestep_min,
        'growth_mode': 'carbon',
        'n_plants': N_PLANTS,
        'carbon_method': carbon_method,
        'enable_baleno': enable_baleno,
        'iterate_gs': iterate_gs,
        'daily_summaries': daily_summaries,
    }
    json_path = checkpoint_dir / 'production_summary.json'
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)

    # Growth series plot
    try:
        _plot_growth_series(checkpoint_dir, growth_days, series_results)
    except Exception as e:
        print(f"  WARNING: Growth series plot failed: {e}")

    print(f"\n{'=' * 70}")
    print(f"PRODUCTION SERIES COMPLETE (CARBON FEEDBACK)")
    print(f"  Days completed: {len(completed_dart_days)}/{len(growth_days)}")
    for d in growth_days:
        s = daily_summaries.get(str(d), {})
        An = s.get('An_field_mean', 0.0)
        print(f"  Day {d:>3}: An={An:.6f} mol/plant/day")
    print(f"{'=' * 70}")

    return series_results


# ============================================================================
# CLI
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description='Phase 9: Diurnal Coupling Loop (Multi-Plant)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Mode A: Single day diurnal cycle (hourly timestep for faster test)
  python run_diurnal.py --days 55 --timestep-min 60

  # Mode A: Single day, 30-min resolution
  python run_diurnal.py --days 55 --timestep-min 30

  # Mode B: Multi-day growth series
  python run_diurnal.py --growth-days 20,35,55 --timestep-min 60

  # Mode A without Baleno (faster, uses Tair for Tleaf)
  python run_diurnal.py --days 55 --timestep-min 60 --no-baleno
        """,
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--days', type=int,
                      help='Single day: simulation day for diurnal cycle')
    mode.add_argument('--growth-days', type=str,
                      help='Growth series: comma-separated simulation days')

    parser.add_argument('--timestep-min', type=int, default=30,
                        help='Timestep in minutes (default: 30)')
    parser.add_argument('--no-baleno', action='store_true',
                        help='Skip Baleno energy balance (use Tair for Tleaf)')
    parser.add_argument('--met-csv', type=str, default=None,
                        help='Path to met forcing CSV')
    parser.add_argument('--skip-photosynthesis', action='store_true',
                        help='Only compute aPAR (skip photosynthesis)')
    parser.add_argument('--iterate-gs', action='store_true',
                        help='Enable iterative Tuzet-Baleno gs coupling (Phase 10)')
    parser.add_argument('--gs-max-iter', type=int, default=6,
                        help='Max gs coupling iterations (default: 6)')
    parser.add_argument('--gs-tolerance', type=float, default=0.05,
                        help='Relative gs convergence threshold (default: 0.05)')
    parser.add_argument('--gs-damping', type=float, default=0.6,
                        help='gs under-relaxation factor (default: 0.6)')

    # Production mode flags (Mode B with carbon + AgroC)
    parser.add_argument('--with-carbon', action='store_true',
                        help='Enable carbon partitioning + AgroC export per day')
    parser.add_argument('--with-agroc', action='store_true',
                        help='Run AgroC Fortran after all days complete')
    parser.add_argument('--carbon-method', type=str, default='auto',
                        choices=['auto', 'phloem', 'dvs'],
                        help='Carbon partitioning method (default: auto)')
    parser.add_argument('--resume', action='store_true',
                        help='Resume interrupted production run from checkpoint')
    parser.add_argument('--uniform', action='store_true',
                        help='Uniform baseline: skip DART/Baleno, '
                             'use clearsky PAR + Tair for all segments')
    parser.add_argument('--growth-mode', type=str, default='parametric',
                        choices=['parametric', 'carbon'],
                        help='Growth mode: parametric (default) or '
                             'carbon-limited feedback')

    args = parser.parse_args()

    print("Phase 9: Time-Series Diurnal Coupling Loop (Multi-Plant)")
    print("=" * 70)

    enable_baleno = not args.no_baleno

    if args.uniform:
        # Uniform baseline: skip DART/Baleno, use clearsky PAR + Tair
        if args.days is not None:
            result = run_single_day_uniform(
                args.days,
                timestep_min=args.timestep_min,
                met_csv=args.met_csv,
            )
        else:
            growth_days = [int(d.strip()) for d in args.growth_days.split(',')]
            if args.with_carbon:
                result = run_production_series_uniform(
                    growth_days,
                    timestep_min=args.timestep_min,
                    carbon_method=args.carbon_method,
                    run_agroc_fortran=args.with_agroc,
                    resume=args.resume,
                )
            else:
                result = run_growth_series_uniform(
                    growth_days,
                    timestep_min=args.timestep_min,
                )
    elif args.days is not None:
        # Mode A: single day
        result = run_single_day(
            args.days,
            timestep_min=args.timestep_min,
            enable_baleno=enable_baleno,
            met_csv=args.met_csv,
            skip_photosynthesis=args.skip_photosynthesis,
            iterate_gs=args.iterate_gs,
            gs_max_iterations=args.gs_max_iter,
            gs_tolerance=args.gs_tolerance,
            gs_damping_alpha=args.gs_damping,
        )
    else:
        # Mode B: growth series
        growth_days = [int(d.strip()) for d in args.growth_days.split(',')]
        if args.growth_mode == 'carbon' and args.with_carbon:
            # Carbon-feedback growth mode
            result = run_production_series_carbon(
                growth_days,
                timestep_min=args.timestep_min,
                enable_baleno=enable_baleno,
                iterate_gs=args.iterate_gs,
                gs_max_iterations=args.gs_max_iter,
                gs_tolerance=args.gs_tolerance,
                gs_damping_alpha=args.gs_damping,
                carbon_method=args.carbon_method,
                run_agroc_fortran=args.with_agroc,
                resume=args.resume,
            )
        elif args.with_carbon:
            # Production mode: full carbon + AgroC pipeline (parametric growth)
            result = run_production_series(
                growth_days,
                timestep_min=args.timestep_min,
                enable_baleno=enable_baleno,
                iterate_gs=args.iterate_gs,
                gs_max_iterations=args.gs_max_iter,
                gs_tolerance=args.gs_tolerance,
                gs_damping_alpha=args.gs_damping,
                carbon_method=args.carbon_method,
                run_agroc_fortran=args.with_agroc,
                resume=args.resume,
            )
        else:
            result = run_growth_series(
                growth_days,
                timestep_min=args.timestep_min,
                enable_baleno=enable_baleno,
            )


if __name__ == '__main__':
    main()
