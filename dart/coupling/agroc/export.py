"""AgroC coupling export: orchestration and CSV output."""

import json
import warnings
from pathlib import Path

import numpy as np

from .profiles import (
    compute_root_respiration_profile,
    compute_root_exudation_profile,
    compute_root_dead_carbon_profile,
    compute_root_water_uptake_profile,
    compute_aboveground_fluxes,
)


def _estimate_Rg_root(carbon_result):
    """Estimate root growth respiration from carbon result keys.

    Neither the phloem nor DVS solver exposes Rg_root as a dict key,
    but Rg_total and partitioning fractions are available.  For DVS,
    Rg is proportional to FR; for phloem, this is a reasonable
    approximation (Rg per organ ~ organ fraction of total sink).
    """
    Rg_total = carbon_result.get("Rg_total_mmol", 0.0)
    FR_root = carbon_result.get("FR_root", 0.0)
    FR_sum = (
        carbon_result.get("FR_leaf", 0.0)
        + carbon_result.get("FR_stem", 0.0)
        + FR_root
        + carbon_result.get("FR_storage", 0.0)
    )
    if FR_sum <= 0 or Rg_total <= 0:
        return 0.0
    return Rg_total * FR_root / FR_sum


# ---------------------------------------------------------------------------
# Single-timestep orchestrator
# ---------------------------------------------------------------------------

def export_agroc_timestep(plant, hm, carbon_result, lai_profile,
                          day, par_umol, tair_c,
                          n_layers=20, depth_cm=100.0,
                          row_spacing_cm=75.0, plant_spacing_cm=20.0):
    """Compute all AgroC-facing profiles for one timestep.

    Calls all ``compute_*`` functions, runs conservation checks
    (warns if profile sum deviates >1% from input total), and returns
    a unified dict.

    Args:
        plant: pb.MappedPlant (grown, with roots).
        hm: PhloemFluxPython (after solve) or None.
        carbon_result: dict from ``solve_carbon_partitioning()`` or None.
        lai_profile: dict from ``extract_lai_profile()``.
        day: simulation day.
        par_umol: PAR [umol m-2 s-1].
        tair_c: air temperature [C].
        n_layers, depth_cm, row_spacing_cm, plant_spacing_cm: grid geometry.

    Returns:
        dict with all profiles and scalar fluxes.
    """
    grid_kw = dict(
        n_layers=n_layers, depth_cm=depth_cm,
        row_spacing_cm=row_spacing_cm, plant_spacing_cm=plant_spacing_cm,
    )

    ground_area = row_spacing_cm * plant_spacing_cm

    # --- Extract totals from carbon result ---
    if carbon_result is not None:
        Rm_root = carbon_result.get("Rm_root", 0.0)
        Rg_root = _estimate_Rg_root(carbon_result)
        exud_total = float(np.sum(carbon_result.get("root_exud_mmol_d", [0.0])))
        dead_total = float(np.sum(carbon_result.get("root_dead_mmol_d", [0.0])))
        An_total_mmol = carbon_result.get("An_total_mmol", None)
        if An_total_mmol is None:
            # Reconstruct from Rm + Rg + growth + storage
            An_total_mmol = (
                carbon_result.get("Rm_total_mmol", 0.0)
                + carbon_result.get("Rg_total_mmol", 0.0)
                + carbon_result.get("growth_mmol_d", 0.0)
                + carbon_result.get("stem_storage_mmol", 0.0)
                + carbon_result.get("seed_reserve_mmol", 0.0)
            )
    else:
        Rm_root = Rg_root = exud_total = dead_total = 0.0
        An_total_mmol = 0.0

    # Compute An from hm only when carbon_result doesn't provide it.
    # In the diurnal pipeline, carbon_result has the correct diurnal-scaled An;
    # hm still holds the unscaled reference photosynthesis (PAR=1000, T=25°C).
    if hm is not None and carbon_result is None:
        An_leaf = np.array(hm.get_net_assimilation())
        An_total_mmol = float(np.sum(An_leaf)) * 1000.0

    # --- Root respiration profile ---
    root_resp = compute_root_respiration_profile(
        plant, Rm_root, Rg_root, **grid_kw,
    )

    # --- Root exudation profile ---
    root_exud = compute_root_exudation_profile(
        plant, exud_total, **grid_kw,
    )

    # --- Dead root carbon profile ---
    root_dead = compute_root_dead_carbon_profile(
        plant, dead_total, **grid_kw,
    )

    # --- Root water uptake profile ---
    root_wuptake = compute_root_water_uptake_profile(
        hm, plant, **grid_kw,
    )

    # --- Aboveground fluxes ---
    if carbon_result is not None:
        above = compute_aboveground_fluxes(
            carbon_result, An_total_mmol, ground_area,
        )
    else:
        above = {
            "GPP_mol_co2_per_cm2_d": 0.0,
            "aboveground_resp_mol_co2_per_cm2_d": 0.0,
        }

    # --- Conservation checks ---
    conservation = []

    def _check(name, total_in, profile_sum, unit="mmol"):
        if abs(total_in) < 1e-12:
            return
        rel_err = abs(profile_sum - total_in) / abs(total_in)
        status = "OK" if rel_err <= 0.01 else "WARN"
        msg = (f"  {name}: input={total_in:.4f} {unit}, "
               f"sum={profile_sum:.4f} {unit}, err={rel_err:.4%} [{status}]")
        conservation.append(msg)
        if rel_err > 0.01:
            warnings.warn(
                f"Conservation check failed for {name}: {rel_err:.2%} error"
            )

    _check("root_resp", root_resp["total_input_mmol"],
           root_resp["profile_sum_mmol"])
    # Exudation conservation: compare in kg C
    if exud_total > 0:
        expected_kg_c = (exud_total * 342.3e-6 * 0.467)
        _check("root_exud", expected_kg_c,
               root_exud["profile_sum_kg_c"], unit="kg C")
    if dead_total > 0:
        expected_kg_c = (dead_total * 342.3e-6 * 0.467)
        _check("root_dead", expected_kg_c,
               root_dead["profile_sum_kg_c"], unit="kg C")

    # --- RLD profile (re-use from grow.py pattern) ---
    try:
        import plantbox as pb
        ana = pb.SegmentAnalyser(plant)
        ana.filter("organType", pb.root)
        layer_thickness = depth_cm / n_layers
        rld_raw = np.array(
            ana.distribution("length", 0.0, -depth_cm, n_layers, True)
        )
        layer_vol = layer_thickness * ground_area
        rrd = rld_raw / max(float(np.sum(rld_raw)), 1e-12)
    except Exception:
        rrd = np.zeros(n_layers)

    # --- Assemble output ---
    return {
        "day": int(day),
        "par_umol": float(par_umol),
        "tair_c": float(tair_c),
        "n_layers": n_layers,
        "depth_cm": depth_cm,
        "row_spacing_cm": row_spacing_cm,
        "plant_spacing_cm": plant_spacing_cm,
        "LAI": float(lai_profile.get("LAI", 0.0)),
        "An_total_mmol": float(An_total_mmol),
        # Aboveground
        "GPP_mol_co2_per_cm2_d": above["GPP_mol_co2_per_cm2_d"],
        "aboveground_resp_mol_co2_per_cm2_d": above["aboveground_resp_mol_co2_per_cm2_d"],
        # Profiles (n_layers each)
        "root_resp_mol_co2_per_cm3_d": root_resp["profile_mol_co2_per_cm3_d"],
        "root_exud_kg_c_per_cm3_d": root_exud["profile_kg_c_per_cm3_d"],
        "root_dead_kg_c_per_cm3_d": root_dead["profile_kg_c_per_cm3_d"],
        "root_wuptake_cm3_per_cm3_d": root_wuptake["profile_cm3_per_cm3_d"],
        "rrd": rrd,
        # Conservation
        "conservation": conservation,
        # Carbon result metadata
        "partitioning_source": (
            carbon_result.get("partitioning_source", "none")
            if carbon_result else "none"
        ),
    }


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def export_coupling_csv(timestep_list, output_path, n_layers):
    """Write coupling CSV with one row per timestep and all layer columns.

    Columns: time_d, LAI, GPP_mol_co2_per_cm2_d,
             aboveground_resp_mol_co2_per_cm2_d,
             root_resp_L00..L{N-1}, root_exud_L00..L{N-1},
             root_dead_L00..L{N-1}, root_wuptake_L00..L{N-1},
             rrd_L00..L{N-1}.

    Also writes a companion ``_metadata.json`` with layer depths, volumes,
    units.

    Args:
        timestep_list: list of dicts from ``export_agroc_timestep()``.
        output_path: Path to CSV file.
        n_layers: number of soil layers.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build header
    layer_ids = [f"L{i:02d}" for i in range(n_layers)]

    header_parts = ["time_d", "LAI",
                    "GPP_mol_co2_per_cm2_d",
                    "aboveground_resp_mol_co2_per_cm2_d"]
    for prefix in ["root_resp", "root_exud", "root_dead",
                    "root_wuptake", "rrd"]:
        for lid in layer_ids:
            header_parts.append(f"{prefix}_{lid}")

    header = ",".join(header_parts)

    rows = []
    for ts in timestep_list:
        vals = [
            f"{ts['day']:.1f}",
            f"{ts['LAI']:.4f}",
            f"{ts['GPP_mol_co2_per_cm2_d']:.8e}",
            f"{ts['aboveground_resp_mol_co2_per_cm2_d']:.8e}",
        ]
        resp = ts["root_resp_mol_co2_per_cm3_d"]
        exud = ts["root_exud_kg_c_per_cm3_d"]
        dead = ts["root_dead_kg_c_per_cm3_d"]
        wup = ts["root_wuptake_cm3_per_cm3_d"]
        rrd = ts["rrd"]

        for arr in [resp, exud, dead, wup, rrd]:
            for v in arr:
                vals.append(f"{v:.8e}")

        rows.append(",".join(vals))

    output_path.write_text(header + "\n" + "\n".join(rows) + "\n")
    print(f"  Coupling CSV: {output_path} ({len(rows)} timesteps)")

    # --- Companion metadata JSON ---
    meta_path = output_path.with_suffix("").with_suffix(".metadata.json")
    if len(timestep_list) > 0:
        ts0 = timestep_list[0]
        depth = ts0["depth_cm"]
        thickness = depth / n_layers
        ground_area = ts0["row_spacing_cm"] * ts0["plant_spacing_cm"]
        layer_vol = thickness * ground_area

        depth_top = np.linspace(0, depth - thickness, n_layers).tolist()
        depth_bot = (np.array(depth_top) + thickness).tolist()

        meta = {
            "n_layers": n_layers,
            "depth_cm": depth,
            "layer_thickness_cm": thickness,
            "ground_area_cm2": ground_area,
            "layer_volume_cm3": layer_vol,
            "row_spacing_cm": ts0["row_spacing_cm"],
            "plant_spacing_cm": ts0["plant_spacing_cm"],
            "depth_top_cm": depth_top,
            "depth_bot_cm": depth_bot,
            "units": {
                "root_resp": "mol CO2 / cm3 / d",
                "root_exud": "kg C / cm3 / d",
                "root_dead": "kg C / cm3 / d",
                "root_wuptake": "cm3 H2O / cm3 soil / d",
                "rrd": "relative root density (sum=1)",
                "GPP": "mol CO2 / cm2 ground / d",
                "aboveground_resp": "mol CO2 / cm2 ground / d",
            },
            "fortran_variable_mapping": {
                "root_resp": "rnodert(ri) — plants.f90:1639",
                "root_exud": "rnodexu(ri) — plants.f90:1577",
                "root_dead": "rnodedeadw(ri) — plants.f90:1599",
                "GPP": "GPP",
                "aboveground_resp": "aboveground_respiration",
            },
        }
    else:
        meta = {"n_layers": n_layers, "note": "no timesteps"}

    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Metadata JSON: {meta_path}")

    return output_path


# ---------------------------------------------------------------------------
# Multi-plant averaging
# ---------------------------------------------------------------------------

def average_agroc_timesteps(timesteps):
    """Average multiple per-plant AgroC timesteps into a field-mean timestep.

    Args:
        timesteps: list of dicts from ``export_agroc_timestep()`` (may contain None).

    Returns:
        dict (same structure as a single timestep), or None if no valid entries.
    """
    valid = [ts for ts in timesteps if ts is not None]
    if not valid:
        return None

    n = len(valid)
    ts0 = valid[0]

    # Average scalars
    avg = {}
    for key in ('LAI', 'An_total_mmol',
                'GPP_mol_co2_per_cm2_d', 'aboveground_resp_mol_co2_per_cm2_d'):
        avg[key] = sum(ts[key] for ts in valid) / n

    # Average array profiles element-wise
    for key in ('root_resp_mol_co2_per_cm3_d', 'root_exud_kg_c_per_cm3_d',
                'root_dead_kg_c_per_cm3_d', 'root_wuptake_cm3_per_cm3_d', 'rrd'):
        stacked = np.array([ts[key] for ts in valid])
        avg[key] = stacked.mean(axis=0)

    # Copy metadata from first valid timestep
    for key in ('day', 'par_umol', 'tair_c', 'n_layers', 'depth_cm',
                'row_spacing_cm', 'plant_spacing_cm'):
        avg[key] = ts0[key]

    # Conservation + source: take from first
    avg['conservation'] = ts0.get('conservation', [])
    avg['partitioning_source'] = ts0.get('partitioning_source', 'none')

    return avg


# ---------------------------------------------------------------------------
# CLI entry point (called from __main__.py)
# ---------------------------------------------------------------------------

def main_export(args):
    """CLI handler for the ``agroc-export`` subcommand."""
    import numpy as np
    from ..growth import (
        grow_plant, run_photosynthesis,
        extract_lai_profile, extract_rld_profile, export_rrd_in,
    )
    from ..carbon import solve_carbon_partitioning
    from ..config import DEFAULT_XML, OUTPUT_DIR, get_species_name

    agroc_out = OUTPUT_DIR / "agroc_export"
    agroc_out.mkdir(parents=True, exist_ok=True)
    species = get_species_name()

    grid_kw = dict(
        n_layers=args.layers, depth_cm=args.depth,
        row_spacing_cm=args.row_spacing,
        plant_spacing_cm=args.plant_spacing,
    )

    def _run_agroc_day(day, par, tair, method, out_dir):
        """Run full pipeline for one day, return timestep dict."""
        print(f"\n{'='*60}")
        print(f"AGROC EXPORT — {species} Day {day}")
        print(f"{'='*60}")

        # 1. Grow plant
        plant = grow_plant(
            xml_path=str(DEFAULT_XML),
            simulation_time=day,
            enable_photosynthesis=True,
            seed=42,
        )

        # 2. LAI extraction
        lai = extract_lai_profile(
            plant, n_bins=10,
            row_spacing_cm=args.row_spacing,
            plant_spacing_cm=args.plant_spacing,
        )
        print(f"  LAI: {lai['LAI']:.2f}")

        # 3. Photosynthesis
        prefix = str(out_dir / f"{species}_day{day}_photo")
        hm = run_photosynthesis(
            plant, sim_time=day, output_prefix=prefix,
            par_umol=par, tair_c=tair,
        )
        if hm is None:
            print(f"  WARNING: Photosynthesis failed, using zeros")

        # 4. Carbon partitioning
        carbon_result = None
        if hm is not None:
            An_leaf = np.array(hm.get_net_assimilation())
            try:
                carbon_result = solve_carbon_partitioning(
                    plant, An_leaf, Tair_C=tair,
                    method=method, day=day,
                )
            except Exception as e:
                print(f"  WARNING: Carbon partitioning failed: {e}")

        # 5. RLD + rrd.in export
        rld = extract_rld_profile(plant, **grid_kw)
        export_rrd_in(rld, out_dir / f"{species}_day{day}_rrd.in")

        # 6. AgroC timestep
        ts = export_agroc_timestep(
            plant, hm, carbon_result, lai,
            day=day, par_umol=par, tair_c=tair,
            **grid_kw,
        )

        # Print conservation report
        print(f"\n  Conservation checks:")
        for line in ts["conservation"]:
            print(line)
        if not ts["conservation"]:
            print("    (no non-zero fluxes to check)")

        # Print summary
        print(f"\n  GPP          : {ts['GPP_mol_co2_per_cm2_d']:.6e} "
              f"mol CO2/cm2/d")
        print(f"  Above resp   : "
              f"{ts['aboveground_resp_mol_co2_per_cm2_d']:.6e} "
              f"mol CO2/cm2/d")
        print(f"  Root resp max: "
              f"{np.max(ts['root_resp_mol_co2_per_cm3_d']):.6e} "
              f"mol CO2/cm3/d")
        print(f"  Water uptake : "
              f"{ts.get('root_wuptake_cm3_per_cm3_d', np.zeros(1)).sum():.4f} "
              f"(sum profile)")
        print(f"  Source       : {ts['partitioning_source']}")

        return ts

    if args.multi_day:
        test_days = [20, 35, 55]
        timesteps = []
        for day in test_days:
            ts = _run_agroc_day(
                day, args.par, args.tair, args.method, agroc_out,
            )
            timesteps.append(ts)

        # Write coupling CSV
        csv_path = agroc_out / f"{species}_coupling.csv"
        export_coupling_csv(timesteps, csv_path, args.layers)

        # Conservation report
        report_path = agroc_out / "conservation_report.txt"
        lines = []
        for ts in timesteps:
            lines.append(f"Day {ts['day']}:")
            lines.extend(ts["conservation"])
            lines.append("")
        report_path.write_text("\n".join(lines))
        print(f"\n  Conservation report: {report_path}")
    else:
        ts = _run_agroc_day(
            args.day, args.par, args.tair, args.method, agroc_out,
        )

        # Write single-day CSV
        csv_path = agroc_out / f"{species}_day{args.day}_coupling.csv"
        export_coupling_csv([ts], csv_path, args.layers)

        # Conservation report
        report_path = agroc_out / "conservation_report.txt"
        lines = [f"Day {ts['day']}:"] + ts["conservation"]
        report_path.write_text("\n".join(lines))
        print(f"\n  Conservation report: {report_path}")
