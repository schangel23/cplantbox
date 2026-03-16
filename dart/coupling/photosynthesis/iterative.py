#!/usr/bin/env python3
"""
Phase 10: Iterative Tuzet-Baleno gs Coupling.

Iterative feedback loop between CPlantBox Tuzet stomatal conductance
and Baleno energy balance:

  1. Initialize Tleaf = Tair (no warm-start — avoids dual-equilibrium trap)
  2. CPlantBox solve with Tleaf -> gs_tuzet, An
  3. Feed damped gs as rcw to Baleno ExternalGS plugin -> new Tleaf
  4. Repeat until gs converges or max_iterations reached

Scene file bootstrap: if no scene file exists (first call per growth day),
runs Baleno once with ExternalGS + empty gs CSV (fallback_rcw) to generate
the scene file for triangle-to-segment mapping.

Data flow:
  CPlantBox hm.gco2 [mol CO2/m²/s] -> gs_h2o = gco2 * 1.6
  -> segment_gs_to_triangle_gs() via mapping JSON
  -> rcw = rho_air/(M_air*1e-3) / gs_h2o  [s/m]
  -> CSV file -> ExternalGS Baleno plugin -> Tleaf per-triangle
  -> read_baleno_tleaf() -> per-segment Tleaf
  -> CPlantBox hm.solve(TairC=tleaf) -> new gco2
  -> check convergence
"""

import json
import numpy as np
from pathlib import Path

from ..config import get_species, get_hydraulics_json, get_photosynthesis_json, get_phloem_json
from ..prospect_params import (get_prospect_params, get_prospect_params_per_position,
                               get_chl_per_segment, vcmax25_from_cab)
from .coupled import run_photosynthesis_solve

# Unit conversion constants
RHO_AIR = 1.225     # kg/m³ (dry air at 20°C, 1 atm)
M_AIR = 28.96e-3    # kg/mol (molar mass of air)
RHO_OVER_M = RHO_AIR / M_AIR  # ~42.3 mol/m³

# Physical limits
RCW_MIN = 10.0      # minimum stomatal resistance [s/m] (very open stomata)
RCW_MAX = 1e6       # maximum stomatal resistance [s/m] (closed stomata)


def segment_gs_to_triangle_gs(gs_per_segment, mapping_json_path,
                              reindex_json_path=None):
    """Reverse-map per-segment gs to per-triangle stomatal resistance (rcw).

    Args:
        gs_per_segment: array of gs [mol CO2 m⁻² s⁻¹] per leaf segment.
            Size = n_leaf_segments (same order as CPlantBox getSegmentIds(4)).
        mapping_json_path: Path to DART mapping JSON with segment->triangle info.
        reindex_json_path: Path to .ori reindex JSON (optional, for total tri count).

    Returns:
        dict with:
          - rcw_per_triangle: array of rcw [s/m] per OBJ triangle (global index).
          - n_triangles_total: total number of triangles in OBJ.
          - coverage: fraction of triangles that received a gs value.
    """
    with open(mapping_json_path) as f:
        mapping = json.load(f)

    # Determine total triangle count
    n_total = mapping.get('n_triangles', 0)
    if reindex_json_path and Path(reindex_json_path).exists():
        with open(reindex_json_path) as f:
            reindex = json.load(f)
        # Total = sum of all .ori group triangle counts
        total_from_reindex = sum(
            len(v) for v in reindex.get('dart_to_obj', {}).values())
        if total_from_reindex > 0:
            n_total = max(n_total, total_from_reindex)

    # Initialize with high resistance (closed stomata = no evaporation)
    rcw = np.full(n_total, RCW_MAX, dtype=np.float64)

    # Walk through leaf organs in mapping JSON
    seg_counter = 0
    n_assigned = 0
    for organ in mapping['organs']:
        if organ['type'] != 'leaf':
            continue
        for seg in organ['segments']:
            if seg_counter >= len(gs_per_segment):
                break

            gco2 = gs_per_segment[seg_counter]
            seg_counter += 1

            # Convert: gco2 [mol CO2/m²/s] -> gs_h2o [mol H2O/m²/s] -> rcw [s/m]
            gs_h2o = gco2 * 1.6
            if gs_h2o > 1e-10:
                seg_rcw = RHO_OVER_M / gs_h2o
            else:
                seg_rcw = RCW_MAX

            seg_rcw = np.clip(seg_rcw, RCW_MIN, RCW_MAX)

            # Assign to all triangles belonging to this segment
            for tri_idx in seg['triangle_indices']:
                if 0 <= tri_idx < n_total:
                    rcw[tri_idx] = seg_rcw
                    n_assigned += 1

    coverage = n_assigned / max(n_total, 1)

    # Diagnostic: warn if rcw values are unexpectedly high
    leaf_rcw = rcw[rcw < RCW_MAX]
    if len(leaf_rcw) > 0:
        median_rcw = float(np.median(leaf_rcw))
        if median_rcw > 800:
            print(f"  WARNING: median rcw={median_rcw:.0f} s/m is very high "
                  f"for well-watered maize (expected 50-500 s/m). "
                  f"Check CPlantBox Tuzet gs parameters.")

    return {
        'rcw_per_triangle': rcw,
        'n_triangles_total': n_total,
        'coverage': coverage,
    }


def write_triangle_gs_csv(rcw_per_triangle, output_path):
    """Write per-triangle rcw to CSV for ExternalGS plugin.

    Format: triangle_index,rcw_s_per_m (no header).
    Only writes triangles with rcw < RCW_MAX (i.e., leaf triangles with valid gs).
    """
    output_path = Path(output_path)
    with open(output_path, 'w') as f:
        for i, rcw in enumerate(rcw_per_triangle):
            if rcw < RCW_MAX:
                f.write(f"{i},{rcw:.6f}\n")


def run_iterative_coupling(
    plant, sim_time, par_umol, mapping_json_path, reindex_json_path,
    baleno_sim_dir, baleno_simu_name,
    grid_info_path=None, center_plant_idx=4,
    max_iterations=6, gs_tolerance=0.05,
    damping_alpha=0.6,
    soil_psi_cm=-500.0,
    tair_c=25.0, rh=0.6,
    initial_tleaf=None,
):
    """Iterative Tuzet-Baleno coupling loop.

    1. Initial Baleno run (Ball-Berry) -> Tleaf_0  (or use initial_tleaf)
    2. CPlantBox solve with Tleaf_0 -> gs_tuzet_0, An_0
    3. Compare gs_tuzet vs gs_prev
    4. If not converged: feed damped gs back to Baleno -> Tleaf_1
    5. Repeat until convergence or max_iterations

    Args:
        plant: pb.MappedPlant (grown, with soil grid).
        sim_time: simulation day.
        par_umol: per-segment PAR array [µmol/m²/s] or scalar.
        mapping_json_path: Path to DART mapping JSON.
        reindex_json_path: Path to .ori reindex JSON.
        baleno_sim_dir: Path to Baleno simulation directory.
        baleno_simu_name: Baleno simulation name.
        grid_info_path: Path to grid_info.json (optional).
        center_plant_idx: Index of center plant in grid.
        max_iterations: Maximum coupling iterations.
        gs_tolerance: Relative convergence threshold for gs (0.05 = 5%).
        damping_alpha: Under-relaxation factor (0 < alpha <= 1).
            gs_next = alpha * gs_tuzet + (1 - alpha) * gs_prev
        soil_psi_cm: Soil water potential [cm].
        tair_c: Air temperature [°C] (used if initial_tleaf is None and
            Baleno is skipped on first pass).
        rh: Relative humidity [0-1].
        initial_tleaf: Optional pre-computed Tleaf array from prior Baleno run.
            If provided, skip initial Ball-Berry Baleno run.

    Returns:
        dict with:
          - tleaf_per_segment: final Tleaf array [°C]
          - gs_per_segment: final gs array [mol CO2/m²/s]
          - an_per_segment: final An array [µmol CO2/m²/s]
          - an_total_mmol: total An [mmol CO2/d]
          - iterations: number of iterations taken
          - gs_history: list of dicts per iteration
          - converged: bool
    """
    from ..dart.baleno import (
        run_baleno_subprocess, read_baleno_tleaf,
        BALENO_DIR,
    )
    from ..dart.parsers import write_json5
    import plantbox as pb

    print(f"\n{'=' * 70}")
    print(f"ITERATIVE TUZET-BALENO COUPLING")
    print(f"  max_iter={max_iterations}, tol={gs_tolerance*100:.1f}%, "
          f"alpha={damping_alpha}")
    print(f"  soil_psi={soil_psi_cm} cm, T_air={tair_c} C, RH={rh*100:.0f}%")
    print(f"{'=' * 70}")

    n_leaf_segs = len(plant.getSegmentIds(4))

    # --- Iteration 0: get initial Tleaf ---
    if initial_tleaf is not None:
        tleaf = np.asarray(initial_tleaf, dtype=np.float64)
        print(f"\n  Using provided initial Tleaf: mean={np.mean(tleaf):.2f} C")
    else:
        tleaf = np.full(n_leaf_segs, tair_c, dtype=np.float64)
        print(f"\n  Using uniform initial Tleaf: {tair_c} C")

    gs_history = []
    gs_prev = None
    converged = False
    final_result = None

    for iteration in range(max_iterations):
        print(f"\n  --- Iteration {iteration + 1}/{max_iterations} ---")

        # --- CPlantBox photosynthesis solve ---
        result = run_photosynthesis_solve(
            plant, sim_time,
            par=par_umol, tleaf=tleaf,
            label=f"iter_{iteration+1}",
            rh=rh, soil_psi_cm=soil_psi_cm,
        )
        if result is None:
            print(f"  ERROR: photosynthesis solve failed at iteration {iteration+1}")
            break

        # Extract gs from CPlantBox hydraulic model
        # hm.gco2 is not directly accessible after run_photosynthesis_solve
        # returns. We need to re-derive gs from An and the diffusion equation.
        # Alternative: extract from the solve itself.
        # For robustness, compute gs from An using the Tuzet inverse:
        #   gs_co2 = An / (Ca - Ci)  where Ci ~ 0.7*Ca for C4
        # But better: re-run with access to hm object.
        gs_tuzet = _extract_gs_from_solve(
            plant, sim_time, par_umol, tleaf, rh, soil_psi_cm)

        if gs_tuzet is None:
            print(f"  ERROR: gs extraction failed at iteration {iteration+1}")
            break

        gs_mean = float(np.mean(gs_tuzet[gs_tuzet > 0]))
        print(f"  gs_tuzet: mean={gs_mean:.6f} mol CO2/m²/s, "
              f"range=[{gs_tuzet.min():.6f}, {gs_tuzet.max():.6f}]")

        # --- Check convergence ---
        if gs_prev is not None:
            mask = gs_prev > 1e-10
            if np.any(mask):
                rel_change = np.abs(gs_tuzet[mask] - gs_prev[mask]) / gs_prev[mask]
                max_rel_change = float(np.max(rel_change))
                mean_rel_change = float(np.mean(rel_change))
            else:
                max_rel_change = 1.0
                mean_rel_change = 1.0

            print(f"  gs convergence: max_rel={max_rel_change:.4f}, "
                  f"mean_rel={mean_rel_change:.4f} (tol={gs_tolerance:.4f})")

            gs_history.append({
                'iteration': iteration + 1,
                'gs_mean': gs_mean,
                'tleaf_mean': float(np.mean(tleaf)),
                'an_total_mmol': result['An_total_mmol'],
                'max_rel_change': max_rel_change,
                'mean_rel_change': mean_rel_change,
            })

            if mean_rel_change < gs_tolerance:
                print(f"  CONVERGED at iteration {iteration + 1}")
                converged = True
                final_result = result
                break
        else:
            gs_history.append({
                'iteration': iteration + 1,
                'gs_mean': gs_mean,
                'tleaf_mean': float(np.mean(tleaf)),
                'an_total_mmol': result['An_total_mmol'],
                'max_rel_change': None,
                'mean_rel_change': None,
            })

        final_result = result

        # --- Damp gs ---
        if gs_prev is not None:
            gs_damped = damping_alpha * gs_tuzet + (1 - damping_alpha) * gs_prev
        else:
            gs_damped = gs_tuzet.copy()

        gs_prev = gs_damped.copy()

        # --- Convert segment gs -> triangle rcw -> CSV ---
        tri_result = segment_gs_to_triangle_gs(
            gs_damped, mapping_json_path, reindex_json_path)

        gs_csv_path = Path(baleno_sim_dir) / 'input' / 'external_gs.csv'
        write_triangle_gs_csv(tri_result['rcw_per_triangle'], gs_csv_path)
        leaf_rcw = tri_result['rcw_per_triangle'][tri_result['rcw_per_triangle'] < RCW_MAX]
        if len(leaf_rcw) > 0:
            print(f"  rcw stats: median={np.median(leaf_rcw):.0f}, "
                  f"mean={np.mean(leaf_rcw):.0f}, "
                  f"range=[{np.min(leaf_rcw):.0f}, {np.max(leaf_rcw):.0f}] s/m")
        print(f"  Wrote triangle gs CSV: {tri_result['coverage']*100:.1f}% coverage")

        # --- Update Baleno config to use ExternalGS plugin ---
        # Use mean Cab/N across per-position LOPS profiles
        leaf_organs = [o for o in plant.getOrgans()
                       if o.organType() == pb.OrganTypes.leaf]
        n_lv = len(leaf_organs)
        per_pos = get_prospect_params_per_position(sim_time, n_lv)
        mean_cab = float(np.mean([p["Cab"] for p in per_pos]))
        mean_n = float(np.mean([p["N"] for p in per_pos]))
        base_p = get_prospect_params(sim_time)
        input_dir = Path(baleno_sim_dir) / 'input'
        write_json5(input_dir / 'vegetation.json5', {
            "Plugin": "ExternalGS",
            "Model": "VegetationExternalGS",
            "PAR_min": 0.400,
            "PAR_max": 0.700,
            "Cab": round(mean_cab, 1), "Cca": 10, "Cs": 0,
            "Cw": base_p["Cw"], "Cdm": base_p["Cm"],
            "N": round(mean_n, 2), "fqe": 0.01,
            "Vcmax25": round(vcmax25_from_cab(mean_cab), 1),
            "BallBerrySlope": 8, "BallBerry0": 0.01,
            "RdPerVcmax25": get_species()["rd_per_vcmax25"],
            "Type": get_species()["photo_type"],
            "rho_thermal": 0.01, "tau_thermal": 0.01, "stress_factor": 1,
        })

        # Write plugin-specific input with gs_file path
        plugins_dir = input_dir / 'plugins'
        plugins_dir.mkdir(parents=True, exist_ok=True)
        write_json5(plugins_dir / 'ExternalGS_input.json5', {
            "gs_file": str(gs_csv_path),
            "fallback_rcw": 100.0,
            "Vcmax25": round(vcmax25_from_cab(mean_cab), 1),
            "RdPerVcmax25": get_species()["rd_per_vcmax25"],
            "Type": get_species()["photo_type"],
            "fqe": 0.01,
        })

        # --- Write Baleno config.ini ---
        baleno_config_path = BALENO_DIR / 'resources' / 'config.ini'
        import textwrap
        baleno_config_path.write_text(textwrap.dedent(f"""\
            [simulation]
            user_data_path =
            name = {baleno_simu_name}
        """))

        # --- Run Baleno with ExternalGS ---
        print(f"  Running Baleno (ExternalGS)...")
        ok = run_baleno_subprocess(timeout=3600)
        if not ok:
            print(f"  ERROR: Baleno subprocess failed at iteration {iteration+1}")
            break

        # --- Read updated Tleaf ---
        tleaf_new = read_baleno_tleaf(
            str(baleno_sim_dir), mapping_json_path, reindex_json_path,
            grid_info_path=grid_info_path,
            center_plant_idx=center_plant_idx,
            tair_c=tair_c,
        )
        if tleaf_new is None:
            print(f"  ERROR: Tleaf read failed at iteration {iteration+1}")
            break

        if len(tleaf_new) != n_leaf_segs:
            print(f"  WARNING: Tleaf size mismatch ({len(tleaf_new)} vs "
                  f"{n_leaf_segs}), padding/truncating")
            if len(tleaf_new) < n_leaf_segs:
                tleaf_new = np.pad(
                    tleaf_new, (0, n_leaf_segs - len(tleaf_new)),
                    constant_values=tair_c)
            else:
                tleaf_new = tleaf_new[:n_leaf_segs]

        # Diagnostic: Tleaf direction relative to Tair
        tleaf_old_mean = float(np.mean(tleaf))
        tleaf = tleaf_new
        tleaf_new_mean = float(np.mean(tleaf))
        direction = "toward" if abs(tleaf_new_mean - tair_c) < abs(tleaf_old_mean - tair_c) else "away from"
        print(f"  Tleaf updated: mean={tleaf_new_mean:.2f}C "
              f"(was {tleaf_old_mean:.2f}C), moved {direction} Tair={tair_c:.1f}C")
        print(f"  Tleaf range=[{tleaf.min():.2f}, {tleaf.max():.2f}]")

        # Log Baleno EB diagnostics for this iteration
        from ..dart.baleno import log_baleno_diagnostics
        log_baleno_diagnostics(baleno_sim_dir, tleaf, tair_c)

    # --- Final results ---
    n_iters = len(gs_history)
    if final_result is None:
        print(f"\n  FAILED: no valid result after {n_iters} iterations")
        return None

    print(f"\n{'=' * 70}")
    print(f"ITERATIVE COUPLING {'CONVERGED' if converged else 'DID NOT CONVERGE'}")
    print(f"  Iterations: {n_iters}")
    print(f"  Final Tleaf: mean={np.mean(tleaf):.2f} C")
    if gs_prev is not None:
        print(f"  Final gs: mean={np.mean(gs_prev[gs_prev > 0]):.6f} mol CO2/m²/s")
    print(f"  Final An: {final_result['An_total_mmol']:.3f} mmol CO2/d")
    print(f"{'=' * 70}")

    return {
        'tleaf_per_segment': tleaf,
        'gs_per_segment': gs_prev if gs_prev is not None else gs_tuzet,
        'an_per_segment': final_result['An_per_umol'],
        'an_total_mmol': final_result['An_total_mmol'],
        'psi_leaf_cm': final_result.get('psi_leaf_cm'),
        'psi_leaf_MPa': final_result.get('psi_leaf_MPa'),
        'iterations': n_iters,
        'gs_history': gs_history,
        'converged': converged,
    }


def _extract_gs_from_solve(plant, sim_time, par_umol, tleaf, rh, soil_psi_cm):
    """Run CPlantBox photosynthesis and extract per-segment gs (gco2).

    Returns array of gs [mol CO2/m²/s] per leaf segment, or None on failure.
    """
    from plantbox.functional.phloem_flux import PhloemFluxPython
    from plantbox.functional.PlantHydraulicParameters import PlantHydraulicParameters
    from ..prospect_params import get_chl_for_photosynthesis

    params = PlantHydraulicParameters()
    params.read_parameters(get_hydraulics_json())

    hm = PhloemFluxPython(plant, params)
    hm.read_photosynthesis_parameters(filename=get_photosynthesis_json())
    hm.read_phloem_parameters(filename=get_phloem_json())

    # Per-segment Chl from LOPS profiles
    chl_per_seg = get_chl_per_segment(sim_time, plant)
    seg_check = plant.getSegmentIds(4)
    if len(chl_per_seg) == len(seg_check):
        hm.Chl = chl_per_seg
    else:
        hm.Chl = [get_chl_for_photosynthesis(sim_time)]

    # Soil water potential
    depth = 100
    p_s = np.linspace(soil_psi_cm, soil_psi_cm - depth, depth)

    # Vapour pressure
    if np.isscalar(tleaf):
        es = hm.get_es(tleaf)
    else:
        es = hm.get_es(float(np.mean(tleaf)))
    ea = es * rh

    # PAR conversion: µmol/m²/s -> mol/cm²/d
    if np.isscalar(par_umol):
        par_mol = par_umol * 1e-6 * 86400 * 1e-4
    else:
        par_mol = np.asarray(par_umol) * 1e-6 * 86400 * 1e-4

    try:
        hm.solve(
            sim_time=sim_time, rsx=p_s, cells=True,
            ea=ea, es=es, PAR=par_mol, TairC=tleaf, verbose=0,
        )
    except Exception as e:
        print(f"  gs extraction solve error: {e}")
        return None

    # Extract gs (gco2) from the solved hydraulic model
    try:
        gco2 = np.array(hm.gco2)
    except AttributeError:
        # Fallback: derive from An
        # gs_co2 = An / (Ca - Ci), but we don't have Ci directly
        # Use approximate inverse: An is available
        An = np.array(hm.get_net_assimilation())  # mol CO2/d per segment
        # Convert to µmol/m²/s for gs estimation
        seg_areas = np.array(hm.get_leafBlade_area())  # cm²
        seg_areas = np.maximum(seg_areas, 1e-6)
        An_umol = An * 1e6 / 86400 * 1e4 / seg_areas  # µmol/m²/s

        # Simple gs estimate: gs = An * 1.6 / (Ca - Ci)
        from ..config import DEFAULT_CO2_PPM
        Ca = DEFAULT_CO2_PPM
        Ci_frac = get_species()["ci_ca_ratio"]
        gs_h2o = An_umol * 1.6 / (Ca * (1 - Ci_frac) + 1e-10)  # mol/m²/s approx
        gco2 = gs_h2o / 1.6
        gco2 = np.maximum(gco2, 0.0)

    return np.asarray(gco2, dtype=np.float64)


# ============================================================================
# Multi-plant iterative coupling
# ============================================================================

def build_scene_row_mapping(baleno_sim_dir, reindex_json_paths, n_plants):
    """Build mapping from (plant_idx, OBJ_face_idx) -> Baleno scene row index.

    Parses the Baleno scene file's DART_NAME column (_mo{pi}_go{gi} format)
    and INDEX_OBJECT column, then uses per-plant .ori reindex to convert
    (group, INDEX_OBJECT) -> absolute OBJ face index.

    Args:
        baleno_sim_dir: Path to Baleno simulation directory.
        reindex_json_paths: List of reindex JSON paths (one per plant).
        n_plants: Number of plants.

    Returns:
        dict with:
          - plant_to_obj_to_scene: {plant_idx: {obj_face_idx: scene_row_idx}}
          - n_scene_rows: total number of rows in scene file
    """
    import re

    output_base = Path(baleno_sim_dir) / 'output'

    # Find scene file
    scene_file = None
    results_dir = output_base / 'final_results'
    candidates = [output_base / 'scene', results_dir / 'scene.csv',
                  results_dir / 'scene']
    for candidate in candidates:
        if candidate.exists() and candidate.stat().st_size > 0:
            scene_file = candidate
            break
    if scene_file is None:
        print(f"  [scene_mapping] No scene file found in {baleno_sim_dir}")
        print(f"    Checked: {[str(c) for c in candidates]}")
        if output_base.exists():
            import os as _os
            all_files = []
            for root, dirs, files in _os.walk(str(output_base)):
                for fn in files:
                    all_files.append(str(Path(root) / fn))
            print(f"    Output dir contents ({len(all_files)} files): "
                  f"{all_files[:10]}")
        else:
            print(f"    Output dir does not exist: {output_base}")
        return None

    from ..dart.parsers import detect_delimiter
    delimiter = detect_delimiter(scene_file)

    scene_str = np.genfromtxt(str(scene_file), skip_header=1,
                               delimiter=delimiter, dtype=str)
    with open(scene_file) as f:
        header_line = f.readline().strip()
    scene_header = [h.strip() for h in header_line.split(delimiter)]

    col_type_id = scene_header.index('TYPE_ID') if 'TYPE_ID' in scene_header else 0
    col_dart_name = scene_header.index('DART_NAME') if 'DART_NAME' in scene_header else 1
    col_index_obj = scene_header.index('INDEX_OBJECT') if 'INDEX_OBJECT' in scene_header else 2

    type_ids = scene_str[:, col_type_id].astype(float).astype(int)
    dart_names = scene_str[:, col_dart_name]
    index_in_object = scene_str[:, col_index_obj].astype(float).astype(int)
    leaf_mask = (type_ids >= 100) | (type_ids == 5)
    n_total = len(type_ids)

    # Load per-plant reindex
    per_plant_dart_to_obj = {}
    for pi in range(n_plants):
        with open(reindex_json_paths[pi]) as f:
            ri = json.load(f)
        dart_to_obj = {}
        for gi_str, obj_indices in ri['dart_to_obj'].items():
            dart_to_obj[int(gi_str)] = np.array(obj_indices, dtype=np.int64)
        per_plant_dart_to_obj[pi] = dart_to_obj

    # Build mapping
    plant_to_obj_to_scene = {pi: {} for pi in range(n_plants)}
    for row_idx in range(n_total):
        if not leaf_mask[row_idx]:
            continue
        dn = dart_names[row_idx]
        mo_m = re.search(r'_mo(\d+)', dn)
        go_m = re.search(r'_go(\d+)', dn)
        if mo_m and go_m:
            instance = int(mo_m.group(1))
            group = int(go_m.group(1))
            if instance < n_plants and group in per_plant_dart_to_obj.get(instance, {}):
                idx = index_in_object[row_idx]
                ori_arr = per_plant_dart_to_obj[instance][group]
                if idx < len(ori_arr):
                    obj_face = int(ori_arr[idx])
                    plant_to_obj_to_scene[instance][obj_face] = row_idx

    return {
        'plant_to_obj_to_scene': plant_to_obj_to_scene,
        'n_scene_rows': n_total,
    }


def segment_gs_to_scene_rcw_multi(gs_per_segment_list, mapping_json_paths,
                                    scene_row_mapping, n_scene_rows):
    """Map per-plant segment gs arrays to a single scene-row-indexed rcw array.

    Args:
        gs_per_segment_list: List of per-plant gs arrays [mol CO2/m²/s].
        mapping_json_paths: List of mapping JSON paths (one per plant).
        scene_row_mapping: dict {plant_idx: {obj_face_idx: scene_row_idx}}
            from build_scene_row_mapping().
        n_scene_rows: Total number of scene rows.

    Returns:
        np.ndarray of rcw [s/m] per scene row (size = n_scene_rows).
    """
    rcw = np.full(n_scene_rows, RCW_MAX, dtype=np.float64)

    for pi, gs_per_segment in enumerate(gs_per_segment_list):
        with open(mapping_json_paths[pi]) as f:
            mapping = json.load(f)

        obj_to_scene = scene_row_mapping.get(pi, {})
        seg_counter = 0
        for organ in mapping['organs']:
            if organ['type'] != 'leaf':
                continue
            for seg in organ['segments']:
                if seg_counter >= len(gs_per_segment):
                    break

                gco2 = gs_per_segment[seg_counter]
                seg_counter += 1

                gs_h2o = gco2 * 1.6
                if gs_h2o > 1e-10:
                    seg_rcw = RHO_OVER_M / gs_h2o
                else:
                    seg_rcw = RCW_MAX
                seg_rcw = np.clip(seg_rcw, RCW_MIN, RCW_MAX)

                for tri_idx in seg['triangle_indices']:
                    if tri_idx in obj_to_scene:
                        scene_row = obj_to_scene[tri_idx]
                        rcw[scene_row] = seg_rcw

    return rcw


def run_iterative_coupling_multi(
    plants, sim_time, par_umol_per_plant,
    mapping_json_paths, reindex_json_paths,
    baleno_sim_dir, baleno_simu_name,
    n_plants=9,
    max_iterations=6, gs_tolerance=0.05,
    damping_alpha=0.6,
    soil_psi_cm=-500.0,
    tair_c=25.0, rh=0.6,
    initial_tleaf=None,
    with_sif=False,
    baleno_timeout=3600,
):
    """Multi-plant iterative Tuzet-Baleno coupling loop.

    All plants get full iterative coupling with their own per-segment Tleaf
    from a single multi-plant Baleno simulation.

    Args:
        plants: List of pb.MappedPlant instances (one per plant).
        sim_time: Simulation day.
        par_umol_per_plant: List of per-segment PAR arrays [µmol/m²/s].
        mapping_json_paths: List of mapping JSON paths (one per plant).
        reindex_json_paths: List of reindex JSON paths (one per plant).
        baleno_sim_dir: Path to Baleno simulation directory.
        baleno_simu_name: Baleno simulation name.
        n_plants: Number of plants.
        max_iterations: Maximum coupling iterations.
        gs_tolerance: Relative convergence threshold for gs.
        damping_alpha: Under-relaxation factor.
        soil_psi_cm: Soil water potential [cm].
        tair_c: Air temperature [°C].
        rh: Relative humidity [0-1].
        initial_tleaf: Optional list of per-plant Tleaf arrays.

    Returns:
        List of n_plants result dicts, each with:
          tleaf_per_segment, gs_per_segment, an_per_segment,
          an_total_mmol, iterations, gs_history, converged.
        Or None on complete failure.
    """
    from ..dart.baleno import (
        run_baleno_subprocess, read_baleno_tleaf_multi,
        BALENO_DIR, log_baleno_diagnostics,
    )
    from ..dart.parsers import write_json5
    import plantbox as pb

    print(f"\n{'=' * 70}")
    print(f"ITERATIVE TUZET-BALENO COUPLING (MULTI-PLANT, {n_plants} plants)")
    print(f"  max_iter={max_iterations}, tol={gs_tolerance*100:.1f}%, "
          f"alpha={damping_alpha}")
    print(f"  soil_psi={soil_psi_cm} cm, T_air={tair_c} C, RH={rh*100:.0f}%")
    print(f"{'=' * 70}")

    # Per-plant leaf segment counts
    n_leaf_segs = [len(plants[pi].getSegmentIds(4)) for pi in range(n_plants)]

    # Initialize per-plant Tleaf — always from Tair (no warm-start).
    # Warm-starting from previous timestep causes a dual-equilibrium trap:
    # cold morning Tleaf propagates forward, suppressing An until noon
    # radiation is strong enough to break out. Starting from Tair ensures
    # the solver always finds the physical (warm) equilibrium.
    tleaf = []
    for pi in range(n_plants):
        if initial_tleaf is not None and pi < len(initial_tleaf):
            tleaf.append(np.asarray(initial_tleaf[pi], dtype=np.float64))
        else:
            tleaf.append(np.full(n_leaf_segs[pi], tair_c, dtype=np.float64))

    # Build scene row mapping (once, from initial Baleno run)
    scene_mapping = build_scene_row_mapping(
        baleno_sim_dir, reindex_json_paths, n_plants)

    if scene_mapping is None:
        # No scene file yet — bootstrap with ExternalGS + empty gs CSV.
        # This replaces the old Ball-Berry bootstrap: any Baleno run
        # produces the scene file, so we run ExternalGS with fallback_rcw
        # for all triangles.  The resulting Tleaf is discarded (we always
        # start from Tair), so the fallback_rcw value doesn't matter.
        print(f"  No scene file — bootstrapping with ExternalGS...")
        # Ensure gs CSV does NOT exist — ExternalGS falls back to
        # fallback_rcw when the file is missing (an empty file crashes
        # np.loadtxt).
        gs_csv_path = Path(baleno_sim_dir) / 'input' / 'external_gs.csv'
        if gs_csv_path.exists():
            gs_csv_path.unlink()

        leaf_organs = [o for o in plants[0].getOrgans()
                       if o.organType() == pb.OrganTypes.leaf]
        n_lv = len(leaf_organs)
        per_pos = get_prospect_params_per_position(sim_time, n_lv)
        mean_cab = float(np.mean([p["Cab"] for p in per_pos]))
        mean_n = float(np.mean([p["N"] for p in per_pos]))
        base_p = get_prospect_params(sim_time)
        input_dir = Path(baleno_sim_dir) / 'input'
        fqe_val = 0.01 if with_sif else 0
        write_json5(input_dir / 'vegetation.json5', {
            "Plugin": "ExternalGS",
            "Model": "VegetationExternalGS",
            "PAR_min": 0.400, "PAR_max": 0.700,
            "Cab": round(mean_cab, 1), "Cca": 10, "Cs": 0,
            "Cw": base_p["Cw"], "Cdm": base_p["Cm"],
            "N": round(mean_n, 2), "fqe": fqe_val,
            "Vcmax25": round(vcmax25_from_cab(mean_cab), 1),
            "BallBerrySlope": 8, "BallBerry0": 0.01,
            "RdPerVcmax25": get_species()["rd_per_vcmax25"],
            "Type": get_species()["photo_type"],
            "rho_thermal": 0.01, "tau_thermal": 0.01, "stress_factor": 1,
        })
        plugins_dir = input_dir / 'plugins'
        plugins_dir.mkdir(parents=True, exist_ok=True)
        write_json5(plugins_dir / 'ExternalGS_input.json5', {
            "gs_file": str(gs_csv_path),
            "fallback_rcw": 100.0,
            "Vcmax25": round(vcmax25_from_cab(mean_cab), 1),
            "RdPerVcmax25": get_species()["rd_per_vcmax25"],
            "Type": get_species()["photo_type"],
            "fqe": fqe_val,
            "Kn0": 5.01, "Knalpha": 1.93, "Knbeta": 10.0,
        })
        import textwrap
        baleno_config_path = BALENO_DIR / 'resources' / 'config.ini'
        baleno_config_path.write_text(textwrap.dedent(f"""\
            [simulation]
            user_data_path =
            name = {baleno_simu_name}
        """))
        print(f"  Running Baleno bootstrap (ExternalGS, {n_plants} plants)...")
        ok = run_baleno_subprocess(timeout=baleno_timeout)
        if not ok:
            print(f"  ERROR: Baleno bootstrap failed")
            return None
        scene_mapping = build_scene_row_mapping(
            baleno_sim_dir, reindex_json_paths, n_plants)
        if scene_mapping is None:
            print(f"  ERROR: Scene file still missing after bootstrap")
            return None
        print(f"  Bootstrap complete — scene mapping built")

    plant_to_obj_to_scene = scene_mapping['plant_to_obj_to_scene']
    n_scene_rows = scene_mapping['n_scene_rows']

    gs_history_per_plant = [[] for _ in range(n_plants)]
    gs_prev = [None] * n_plants
    converged_flags = [False] * n_plants
    final_results = [None] * n_plants

    for iteration in range(max_iterations):
        print(f"\n  --- Iteration {iteration + 1}/{max_iterations} ---")

        # 1. Run photosynthesis for each plant
        all_gs_tuzet = []
        for pi in range(n_plants):
            result = run_photosynthesis_solve(
                plants[pi], sim_time,
                par=par_umol_per_plant[pi], tleaf=tleaf[pi],
                label=f"iter_{iteration+1}_p{pi}",
                rh=rh, soil_psi_cm=soil_psi_cm,
            )
            if result is None:
                print(f"  Plant {pi}: photosynthesis solve FAILED")
                all_gs_tuzet.append(np.zeros(n_leaf_segs[pi]))
                continue

            final_results[pi] = result

            gs = _extract_gs_from_solve(
                plants[pi], sim_time, par_umol_per_plant[pi],
                tleaf[pi], rh, soil_psi_cm)
            if gs is None:
                gs = np.zeros(n_leaf_segs[pi])
            all_gs_tuzet.append(gs)

        # 2. Check convergence per plant
        all_converged = True
        for pi in range(n_plants):
            if converged_flags[pi]:
                continue
            gs_tuzet = all_gs_tuzet[pi]
            gs_mean = float(np.mean(gs_tuzet[gs_tuzet > 0])) if np.any(gs_tuzet > 0) else 0.0

            if gs_prev[pi] is not None:
                mask = gs_prev[pi] > 1e-10
                if np.any(mask):
                    rel_change = np.abs(gs_tuzet[mask] - gs_prev[pi][mask]) / gs_prev[pi][mask]
                    mean_rel_change = float(np.mean(rel_change))
                else:
                    mean_rel_change = 1.0

                gs_history_per_plant[pi].append({
                    'iteration': iteration + 1,
                    'gs_mean': gs_mean,
                    'tleaf_mean': float(np.mean(tleaf[pi])),
                    'an_total_mmol': final_results[pi]['An_total_mmol'] if final_results[pi] else 0.0,
                    'mean_rel_change': mean_rel_change,
                })

                if mean_rel_change < gs_tolerance:
                    converged_flags[pi] = True
                    print(f"  Plant {pi}: CONVERGED (mean_rel={mean_rel_change:.4f})")
                else:
                    all_converged = False
            else:
                gs_history_per_plant[pi].append({
                    'iteration': iteration + 1,
                    'gs_mean': gs_mean,
                    'tleaf_mean': float(np.mean(tleaf[pi])),
                    'an_total_mmol': final_results[pi]['An_total_mmol'] if final_results[pi] else 0.0,
                    'mean_rel_change': None,
                })
                all_converged = False

        if all_converged and iteration > 0:
            print(f"  ALL plants converged at iteration {iteration + 1}")
            break

        # 3. Damp gs
        gs_damped = []
        for pi in range(n_plants):
            if converged_flags[pi]:
                gs_damped.append(gs_prev[pi].copy())
            elif gs_prev[pi] is not None:
                gs_d = damping_alpha * all_gs_tuzet[pi] + (1 - damping_alpha) * gs_prev[pi]
                gs_damped.append(gs_d)
            else:
                gs_damped.append(all_gs_tuzet[pi].copy())
            gs_prev[pi] = gs_damped[pi].copy()

        # 4. Build combined rcw array and write CSV
        rcw = segment_gs_to_scene_rcw_multi(
            gs_damped, mapping_json_paths,
            plant_to_obj_to_scene, n_scene_rows)

        gs_csv_path = Path(baleno_sim_dir) / 'input' / 'external_gs.csv'
        write_triangle_gs_csv(rcw, gs_csv_path)

        leaf_rcw = rcw[rcw < RCW_MAX]
        if len(leaf_rcw) > 0:
            print(f"  rcw stats: median={np.median(leaf_rcw):.0f}, "
                  f"mean={np.mean(leaf_rcw):.0f} s/m, "
                  f"{len(leaf_rcw)} leaf triangles")

        # 5. Update Baleno config to use ExternalGS plugin
        leaf_organs = [o for o in plants[0].getOrgans()
                       if o.organType() == pb.OrganTypes.leaf]
        n_lv = len(leaf_organs)
        per_pos = get_prospect_params_per_position(sim_time, n_lv)
        mean_cab = float(np.mean([p["Cab"] for p in per_pos]))
        mean_n = float(np.mean([p["N"] for p in per_pos]))
        base_p = get_prospect_params(sim_time)
        input_dir = Path(baleno_sim_dir) / 'input'
        fqe_val = 0.01 if with_sif else 0
        veg_json = {
            "Plugin": "ExternalGS",
            "Model": "VegetationExternalGS",
            "PAR_min": 0.400, "PAR_max": 0.700,
            "Cab": round(mean_cab, 1), "Cca": 10, "Cs": 0,
            "Cw": base_p["Cw"], "Cdm": base_p["Cm"],
            "N": round(mean_n, 2), "fqe": fqe_val,
            "Vcmax25": round(vcmax25_from_cab(mean_cab), 1),
            "BallBerrySlope": 8, "BallBerry0": 0.01,
            "RdPerVcmax25": get_species()["rd_per_vcmax25"],
            "Type": get_species()["photo_type"],
            "rho_thermal": 0.01, "tau_thermal": 0.01, "stress_factor": 1,
        }
        if with_sif:
            veg_json["Kn0"] = 5.01
            veg_json["Knalpha"] = 1.93
            veg_json["Knbeta"] = 10.0
        write_json5(input_dir / 'vegetation.json5', veg_json)

        plugins_dir = input_dir / 'plugins'
        plugins_dir.mkdir(parents=True, exist_ok=True)
        write_json5(plugins_dir / 'ExternalGS_input.json5', {
            "gs_file": str(gs_csv_path),
            "fallback_rcw": 100.0,
            "Vcmax25": round(vcmax25_from_cab(mean_cab), 1),
            "RdPerVcmax25": get_species()["rd_per_vcmax25"],
            "Type": get_species()["photo_type"],
            "fqe": fqe_val,
            "Kn0": 5.01,
            "Knalpha": 1.93,
            "Knbeta": 10.0,
        })

        # 6. Write Baleno config.ini + run
        baleno_config_path = BALENO_DIR / 'resources' / 'config.ini'
        import textwrap
        baleno_config_path.write_text(textwrap.dedent(f"""\
            [simulation]
            user_data_path =
            name = {baleno_simu_name}
        """))

        print(f"  Running Baleno (ExternalGS, {n_plants} plants)...")
        ok = run_baleno_subprocess(timeout=baleno_timeout)
        if not ok:
            print(f"  ERROR: Baleno subprocess failed at iteration {iteration+1}")
            break

        # 7. Read per-plant Tleaf + diagnostics
        tleaf_new_list = read_baleno_tleaf_multi(
            str(baleno_sim_dir), mapping_json_paths, reindex_json_paths,
            n_plants, tair_c=tair_c)
        if tleaf_new_list is None:
            print(f"  ERROR: Tleaf read failed at iteration {iteration+1}")
            break

        # Log EB diagnostics (was missing in multi-plant path)
        log_baleno_diagnostics(
            baleno_sim_dir, tleaf_new_list[0], tair_c)

        # Update Tleaf per plant (skip converged)
        for pi in range(n_plants):
            if converged_flags[pi]:
                continue
            tleaf_new = tleaf_new_list[pi]
            if len(tleaf_new) != n_leaf_segs[pi]:
                if len(tleaf_new) < n_leaf_segs[pi]:
                    tleaf_new = np.pad(
                        tleaf_new, (0, n_leaf_segs[pi] - len(tleaf_new)),
                        constant_values=tair_c)
                else:
                    tleaf_new = tleaf_new[:n_leaf_segs[pi]]

            old_mean = float(np.mean(tleaf[pi]))
            tleaf[pi] = tleaf_new
            new_mean = float(np.mean(tleaf_new))
            print(f"  Plant {pi}: Tleaf mean={new_mean:.2f}C (was {old_mean:.2f}C)")

    # Build final results
    n_iters = max(len(h) for h in gs_history_per_plant) if any(gs_history_per_plant) else 0
    n_converged = sum(converged_flags)
    print(f"\n{'=' * 70}")
    print(f"ITERATIVE COUPLING: {n_converged}/{n_plants} converged, "
          f"{n_iters} iterations")
    for pi in range(n_plants):
        if final_results[pi]:
            print(f"  Plant {pi}: An={final_results[pi]['An_total_mmol']:.3f} mmol, "
                  f"Tleaf={np.mean(tleaf[pi]):.2f}C, "
                  f"{'converged' if converged_flags[pi] else 'not converged'}")
    print(f"{'=' * 70}")

    # Read fluorescence (eta) from final Baleno run if with_sif
    per_plant_eta = [None] * n_plants
    per_plant_tri_data = [None] * n_plants
    per_plant_tri_raw = [None] * n_plants
    if with_sif:
        from ..dart.baleno import read_baleno_outputs_multi
        sif_result = read_baleno_outputs_multi(
            str(baleno_sim_dir), mapping_json_paths, reindex_json_paths,
            n_plants, tair_c=tair_c, read_fluorescence=True)
        if sif_result is not None:
            per_plant_eta = sif_result.get('eta', [None] * n_plants)
            per_plant_tri_data = sif_result.get('tri_data', [None] * n_plants)
            per_plant_tri_raw = sif_result.get('tri_data_raw', [None] * n_plants)
            print(f"  SIF: read fluorescence for {n_plants} plants")

    results = []
    for pi in range(n_plants):
        r = final_results[pi]
        rd = {
            'tleaf_per_segment': tleaf[pi],
            'gs_per_segment': gs_prev[pi] if gs_prev[pi] is not None else np.zeros(n_leaf_segs[pi]),
            'an_per_segment': r['An_per_umol'] if r else np.zeros(n_leaf_segs[pi]),
            'an_total_mmol': r['An_total_mmol'] if r else 0.0,
            'psi_leaf_cm': r.get('psi_leaf_cm') if r else None,
            'psi_leaf_MPa': r.get('psi_leaf_MPa') if r else None,
            'iterations': n_iters,
            'gs_history': gs_history_per_plant[pi],
            'converged': converged_flags[pi],
        }
        if with_sif:
            rd['eta_per_segment'] = per_plant_eta[pi] if pi < len(per_plant_eta) else None
            rd['tri_data'] = per_plant_tri_data[pi] if pi < len(per_plant_tri_data) else None
            rd['tri_data_raw'] = per_plant_tri_raw[pi] if pi < len(per_plant_tri_raw) else None
        results.append(rd)

    return results
