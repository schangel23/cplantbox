#!/usr/bin/env python3
"""
Phase 10: Iterative Tuzet-Baleno gs Coupling.

Replaces one-shot Ball-Berry gs in Baleno with CPlantBox's Tuzet stomatal
conductance via an iterative feedback loop:

  1. Initial Baleno run (Ball-Berry) -> Tleaf_0
  2. CPlantBox solve with Tleaf_0 -> gs_tuzet_0, An_0
  3. Feed damped gs back to Baleno -> Tleaf_1
  4. Repeat until gs converges or max_iterations reached

The Tuzet model's fw(psi) water-stress response naturally handles any soil
water potential — no special drought handling is needed.

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

from ..config import PHOTO_PATH
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
        run_baleno_subprocess, read_baleno_tleaf, _write_json5,
        BALENO_DIR,
    )
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
        print(f"  Wrote triangle gs CSV: {tri_result['coverage']*100:.1f}% coverage")

        # --- Update Baleno config to use ExternalGS plugin ---
        input_dir = Path(baleno_sim_dir) / 'input'
        _write_json5(input_dir / 'vegetation.json5', {
            "Plugin": "ExternalGS",
            "Model": "VegetationExternalGS",
            "PAR_min": 0.400,
            "PAR_max": 0.780,
            "Cab": 55, "Cca": 10, "Cs": 0, "Cw": 0.012, "Cdm": 0.01,
            "N": 1.4, "fqe": 0,
            "Vcmax25": 50, "BallBerrySlope": 8, "BallBerry0": 0.01,
            "RdPerVcmax25": 0.015, "Type": "C4",
            "rho_thermal": 0.01, "tau_thermal": 0.01, "stress_factor": 1,
        })

        # Write plugin-specific input with gs_file path
        plugins_dir = input_dir / 'plugins'
        plugins_dir.mkdir(parents=True, exist_ok=True)
        _write_json5(plugins_dir / 'ExternalGS_input.json5', {
            "gs_file": str(gs_csv_path),
            "fallback_rcw": 100.0,
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
        ok = run_baleno_subprocess(timeout=1800)
        if not ok:
            print(f"  ERROR: Baleno subprocess failed at iteration {iteration+1}")
            break

        # --- Read updated Tleaf ---
        tleaf_new = read_baleno_tleaf(
            str(baleno_sim_dir), mapping_json_path, reindex_json_path,
            grid_info_path=grid_info_path,
            center_plant_idx=center_plant_idx,
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

        tleaf = tleaf_new
        print(f"  Tleaf updated: mean={np.mean(tleaf):.2f} C, "
              f"range=[{tleaf.min():.2f}, {tleaf.max():.2f}]")

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
        'iterations': n_iters,
        'gs_history': gs_history,
        'converged': converged,
    }


def _extract_gs_from_solve(plant, sim_time, par_umol, tleaf, rh, soil_psi_cm):
    """Run CPlantBox photosynthesis and extract per-segment gs (gco2).

    Returns array of gs [mol CO2/m²/s] per leaf segment, or None on failure.
    """
    from plantbox.functional.Photosynthesis import PhotosynthesisPython
    from plantbox.functional.PlantHydraulicParameters import PlantHydraulicParameters
    from ..prospect_params import get_chl_for_photosynthesis

    params = PlantHydraulicParameters()
    params.read_parameters(PHOTO_PATH + "maize_couvreur2012_hydraulics")

    hm = PhotosynthesisPython(plant, params)
    hm.read_photosynthesis_parameters(
        filename=PHOTO_PATH + "maize_C4_photosynthesis_parameters")

    chl = get_chl_for_photosynthesis(sim_time)
    hm.Chl = [chl]

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

        # Simple gs estimate for C4: gs = An * 1.6 / (Ca - Ci)
        # Assume Ci/Ca ~ 0.4 for C4 (typical)
        Ca = 400.0  # ppm
        Ci_frac = 0.4
        gs_h2o = An_umol * 1.6 / (Ca * (1 - Ci_frac) + 1e-10)  # mol/m²/s approx
        gco2 = gs_h2o / 1.6
        gco2 = np.maximum(gco2, 0.0)

    return np.asarray(gco2, dtype=np.float64)
