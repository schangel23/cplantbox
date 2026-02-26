"""CLI entry point: python -m coupling <subcommand>."""
import argparse
import os
import sys


def cli():
    parser = argparse.ArgumentParser(
        prog="coupling",
        description="CPlantBox-DART coupling pipeline.",
    )
    parser.add_argument(
        "--species", type=str, default=None,
        help="Plant species (default: maize). Sets COUPLING_SPECIES env var.",
    )
    parser.add_argument(
        "--threads", type=int, default=None,
        help="DART thread count (default: 8). Sets DART_THREADS env var.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # Phase 1 — DART RT simulation
    p_sim = sub.add_parser("simulation", help="Phase 1: DART radiative transfer")
    p_sim.add_argument("--day", type=int, default=55)

    # Phase 2 — Baleno energy balance
    sub.add_parser("baleno", help="Phase 2: Baleno energy balance")

    # Phase 4 — coupled photosynthesis
    sub.add_parser("photosynthesis", help="Phase 4: coupled photosynthesis")

    # Phase 5 — validation
    sub.add_parser("validate", help="Phase 5: coupling validation")

    # Phase 6 — multi-plant unique realizations
    p_mf = sub.add_parser("multifield", help="Phase 6: multi-plant field")
    p_mf.add_argument("--seeds", type=str, default="42-50")

    # Phase 9 — diurnal loop (has its own argparse, delegate)
    sub.add_parser("diurnal", help="Phase 9: diurnal coupling loop")

    # Session 2 — RLD profile extraction
    p_rld = sub.add_parser("rld", help="Session 2: extract RLD profile")
    p_rld.add_argument("--day", type=int, default=55, help="Simulation day")
    p_rld.add_argument("--layers", type=int, default=20, help="Number of depth layers")
    p_rld.add_argument("--depth", type=float, default=100.0, help="Max soil depth [cm]")
    p_rld.add_argument("--row-spacing", type=float, default=75.0,
                        help="Inter-row spacing [cm] (default: 75)")
    p_rld.add_argument("--plant-spacing", type=float, default=20.0,
                        help="Intra-row spacing [cm] (default: 20)")
    p_rld.add_argument("--multi-day", action="store_true",
                        help="Run days 20, 35, 55 and produce growth trajectory plot")

    # Stage 2 Session 4 — carbon partitioning
    p_carbon = sub.add_parser("carbon", help="Stage 2: carbon partitioning")
    p_carbon.add_argument("--day", type=int, default=55, help="Simulation day")
    p_carbon.add_argument("--method", type=str, default="auto",
                          choices=["auto", "phloem", "dvs"],
                          help="Partitioning method (default: auto)")
    p_carbon.add_argument("--par", type=float, default=1000.0,
                          help="PAR [umol m-2 s-1] (default: 1000)")
    p_carbon.add_argument("--tair", type=float, default=25.0,
                          help="Air temperature [C] (default: 25)")

    # Stage 2 Session 5 — LAI + plant summary
    p_summary = sub.add_parser("summary", help="Session 5: LAI + plant summary")
    p_summary.add_argument("--day", type=int, default=55, help="Simulation day")
    p_summary.add_argument("--par", type=float, default=1000.0,
                           help="PAR [umol m-2 s-1] (default: 1000)")
    p_summary.add_argument("--tair", type=float, default=25.0,
                           help="Air temperature [C] (default: 25)")
    p_summary.add_argument("--method", type=str, default="auto",
                           choices=["auto", "phloem", "dvs"],
                           help="Carbon partitioning method (default: auto)")
    p_summary.add_argument("--row-spacing", type=float, default=75.0,
                           help="Inter-row spacing [cm] (default: 75)")
    p_summary.add_argument("--plant-spacing", type=float, default=20.0,
                           help="Intra-row spacing [cm] (default: 20)")
    p_summary.add_argument("--bins", type=int, default=10,
                           help="Number of vertical LAI bins (default: 10)")
    p_summary.add_argument("--multi-day", action="store_true",
                           help="Run days [10, 20, 30, 40, 55] and produce growth trajectory")

    # Stage 2 Session 6 — AgroC coupling export
    p_agroc = sub.add_parser("agroc-export",
                             help="Session 6: AgroC coupling profiles + CSV")
    p_agroc.add_argument("--day", type=int, default=55, help="Simulation day")
    p_agroc.add_argument("--layers", type=int, default=20,
                         help="Number of soil depth layers (default: 20)")
    p_agroc.add_argument("--depth", type=float, default=100.0,
                         help="Max soil depth [cm] (default: 100)")
    p_agroc.add_argument("--row-spacing", type=float, default=75.0,
                         help="Inter-row spacing [cm] (default: 75)")
    p_agroc.add_argument("--plant-spacing", type=float, default=20.0,
                         help="Intra-row spacing [cm] (default: 20)")
    p_agroc.add_argument("--par", type=float, default=1000.0,
                         help="PAR [umol m-2 s-1] (default: 1000)")
    p_agroc.add_argument("--tair", type=float, default=25.0,
                         help="Air temperature [C] (default: 25)")
    p_agroc.add_argument("--method", type=str, default="auto",
                         choices=["auto", "phloem", "dvs"],
                         help="Carbon partitioning method (default: auto)")
    p_agroc.add_argument("--multi-day", action="store_true",
                         help="Run days [20, 35, 55] trajectory")

    # Session 8 — integration test
    p_integ = sub.add_parser("integration-test",
                             help="Session 8: full pipeline integration test")
    p_integ.add_argument("--day", type=int, default=55,
                         help="Simulation day (default: 55)")
    p_integ.add_argument("--skip-dart", action="store_true",
                         help="Skip DART/Baleno (uniform PAR/Tleaf)")
    p_integ.add_argument("--skip-agroc", action="store_true",
                         help="Skip AgroC Fortran test")

    # Growth helpers
    sub.add_parser("grow", help="Grow calibrated plant")
    sub.add_parser("calibrate", help="Calibrate maize XML")

    args, remaining = parser.parse_known_args()

    # Set env vars BEFORE importing subcommands (config.py reads them at import)
    if args.species:
        os.environ["COUPLING_SPECIES"] = args.species.lower()
    if args.threads is not None:
        os.environ["DART_THREADS"] = str(args.threads)

    if args.command == "simulation":
        from .dart.simulation import main
        main()
    elif args.command == "baleno":
        from .dart.baleno import main
        main()
    elif args.command == "photosynthesis":
        from .photosynthesis.coupled import main
        main()
    elif args.command == "validate":
        from .validation.validate import main
        main()
    elif args.command == "multifield":
        from .dart.multifield import main
        main()
    elif args.command == "diurnal":
        # diurnal has its own argparse — pass remaining args via sys.argv
        sys.argv = [sys.argv[0]] + remaining
        from .photosynthesis.diurnal import main
        main()
    elif args.command == "rld":
        from .growth.grow import (
            grow_plant, extract_rld_profile, export_rld_csv,
            export_rrd_in, plot_rld_profile, plot_rld_growth_trajectory,
        )
        from .config import DEFAULT_XML, OUTPUT_DIR

        rld_out = OUTPUT_DIR / "session2"
        rld_out.mkdir(parents=True, exist_ok=True)

        if args.multi_day:
            # Run multiple growth stages
            test_days = [20, 35, 55]
            profiles = {}
            for day in test_days:
                print(f"\n{'='*60}")
                print(f"RLD EXTRACTION — Day {day}")
                print(f"{'='*60}")
                plant = grow_plant(
                    xml_path=str(DEFAULT_XML),
                    simulation_time=day,
                    enable_photosynthesis=True,
                    seed=42,
                )
                prof = extract_rld_profile(
                    plant, n_layers=args.layers, depth_cm=args.depth,
                    row_spacing_cm=args.row_spacing,
                    plant_spacing_cm=args.plant_spacing,
                )
                profiles[day] = prof
                export_rld_csv(prof, rld_out / f"maize_day{day}_rld_profile.csv")
                export_rrd_in(prof, rld_out / f"maize_day{day}_rrd.in")
                plot_rld_profile(prof, rld_out / f"maize_day{day}_rld_profile.png",
                                 day=day)
            plot_rld_growth_trajectory(profiles,
                                       rld_out / "rld_growth_trajectory.png")
        else:
            plant = grow_plant(
                xml_path=str(DEFAULT_XML),
                simulation_time=args.day,
                enable_photosynthesis=True,
                seed=42,
            )
            prof = extract_rld_profile(
                plant, n_layers=args.layers, depth_cm=args.depth,
                row_spacing_cm=args.row_spacing,
                plant_spacing_cm=args.plant_spacing,
            )
            export_rld_csv(prof,
                           rld_out / f"maize_day{args.day}_rld_profile.csv")
            export_rrd_in(prof,
                          rld_out / f"maize_day{args.day}_rrd.in")
            plot_rld_profile(prof,
                             rld_out / f"maize_day{args.day}_rld_profile.png",
                             day=args.day)

            print(f"\n  Total root length: {prof['total_root_length_cm']:.0f} cm")
            print(f"  Max root depth: {prof['max_root_depth_cm']:.1f} cm")
            print(f"  Surface RLD: {prof['RLD_cm_per_cm3'][0]:.4f} cm/cm3")
    elif args.command == "carbon":
        import numpy as np
        from .growth.grow import grow_plant, run_photosynthesis
        from .carbon import solve_carbon_partitioning
        from .config import DEFAULT_XML, OUTPUT_DIR

        carbon_out = OUTPUT_DIR / "carbon"
        carbon_out.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"CARBON PARTITIONING — Day {args.day}, method={args.method}")
        print(f"{'='*60}")

        # 1. Grow plant
        plant = grow_plant(
            xml_path=str(DEFAULT_XML),
            simulation_time=args.day,
            enable_photosynthesis=True,
            seed=42,
        )

        # 2. Run photosynthesis to get An per leaf segment
        prefix = str(carbon_out / f"day{args.day}_photo")
        hm = run_photosynthesis(
            plant, sim_time=args.day, output_prefix=prefix,
            par_umol=args.par, tair_c=args.tair,
        )
        if hm is None:
            print("ERROR: Photosynthesis solve failed.")
            sys.exit(1)

        An_leaf = np.array(hm.get_net_assimilation())  # mol CO2/d per leaf seg
        An_total_mmol = float(np.sum(An_leaf)) * 1000.0

        # 3. Solve carbon partitioning
        result = solve_carbon_partitioning(
            plant, An_leaf, Tair_C=args.tair,
            method=args.method, day=args.day,
        )

        # 4. Print results
        print(f"\n{'='*60}")
        print(f"CARBON PARTITIONING RESULTS ({result['partitioning_source']})")
        print(f"{'='*60}")
        print(f"  An total (input)     : {An_total_mmol:.1f} mmol CO2/d")
        print(f"  Rm total             : {result['Rm_total_mmol']:.1f} mmol/d")
        print(f"    Rm leaf            : {result['Rm_leaf']:.1f}")
        print(f"    Rm stem            : {result['Rm_stem']:.1f}")
        print(f"    Rm root            : {result['Rm_root']:.1f}")
        print(f"    Rm storage         : {result['Rm_storage']:.1f}")
        print(f"  Rg total             : {result['Rg_total_mmol']:.1f} mmol/d")
        print(f"  Growth               : {result['growth_mmol_d']:.1f} mmol/d")
        if 'stem_storage_mmol' in result:
            print(f"  Stem storage         : {result['stem_storage_mmol']:.1f} mmol/d")
        if result.get('seed_reserve_mmol', 0) > 0:
            print(f"  Seed reserve         : {result['seed_reserve_mmol']:.1f} mmol/d")
        print(f"  Partitioning fractions:")
        print(f"    FR_leaf            : {result['FR_leaf']:.3f}")
        print(f"    FR_stem            : {result['FR_stem']:.3f}")
        print(f"    FR_root            : {result['FR_root']:.3f}")
        print(f"    FR_storage         : {result['FR_storage']:.3f}")
        print(f"    Sum                : {result['FR_leaf']+result['FR_stem']+result['FR_root']+result['FR_storage']:.3f}")
        print(f"  Carbon balance error : {result['carbon_balance_error']:.2%}")
        if 'C_ST_mean' in result and not np.isnan(result.get('C_ST_mean', np.nan)):
            print(f"  C_ST mean            : {result['C_ST_mean']:.3f} mmol/cm3")
            print(f"  C_ST range           : [{result.get('C_ST_min', 0):.3f}, {result.get('C_ST_max', 0):.3f}]")
            print(f"  Picard iterations    : {result.get('n_iterations', 'N/A')}")
            print(f"  Converged            : {result.get('converged', 'N/A')}")

        # 5. Save results
        import json
        out_path = carbon_out / f"day{args.day}_{args.method}_carbon.json"
        # Convert numpy types for JSON serialization
        serializable = {}
        for k, v in result.items():
            if isinstance(v, np.ndarray):
                serializable[k] = v.tolist()
            elif isinstance(v, (np.floating, np.integer)):
                serializable[k] = float(v)
            else:
                serializable[k] = v
        with open(out_path, 'w') as f:
            json.dump(serializable, f, indent=2)
        print(f"\n  Results saved: {out_path}")

    elif args.command == "summary":
        import json
        import numpy as np
        from .growth.grow import (
            grow_plant, run_photosynthesis,
            extract_lai_profile, export_lai_csv, plot_lai_profile,
            extract_plant_summary, plot_growth_trajectory,
        )
        from .carbon import solve_carbon_partitioning
        from .config import DEFAULT_XML, OUTPUT_DIR, get_species_name

        summary_out = OUTPUT_DIR / "session5"
        summary_out.mkdir(parents=True, exist_ok=True)
        species = get_species_name()

        def _run_single_day(day, par, tair, method, row_sp, plant_sp, bins,
                            out_dir):
            """Run full pipeline for one day, return summary dict."""
            print(f"\n{'='*60}")
            print(f"SESSION 5: LAI + PLANT SUMMARY — {species} Day {day}")
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
                plant, n_bins=bins,
                row_spacing_cm=row_sp,
                plant_spacing_cm=plant_sp,
            )
            export_lai_csv(lai, out_dir / f"{species}_day{day}_lai_profile.csv")
            plot_lai_profile(lai, out_dir / f"{species}_day{day}_lai_profile.png",
                             day=day)

            print(f"\n  LAI: {lai['LAI']:.2f}")
            print(f"  Total leaf area: {lai['total_leaf_area_cm2']:.0f} cm2 "
                  f"({lai['total_leaf_area_m2']:.4f} m2)")
            print(f"  Plant height: {lai['plant_height_cm']:.1f} cm")
            print(f"  Leaf organs: {lai['n_leaf_organs']}, "
                  f"segments: {lai['n_leaf_segments']}")

            # 3. Photosynthesis
            prefix = str(out_dir / f"{species}_day{day}_photo")
            hm = run_photosynthesis(
                plant, sim_time=day, output_prefix=prefix,
                par_umol=par, tair_c=tair,
            )
            if hm is None:
                print(f"  WARNING: Photosynthesis failed for day {day}, "
                      f"skipping carbon partitioning")

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

            # 5. Assemble summary
            summary = extract_plant_summary(
                plant, hm, carbon_result, lai, day,
                par_umol=par, tair_c=tair,
            )

            # 6. Save JSON
            json_path = out_dir / f"{species}_day{day}_summary.json"
            with open(json_path, 'w') as f:
                json.dump(summary, f, indent=2)
            print(f"\n  Summary JSON: {json_path}")

            return summary

        if args.multi_day:
            test_days = [10, 20, 30, 40, 55]
            summaries = {}
            for day in test_days:
                summaries[day] = _run_single_day(
                    day, args.par, args.tair, args.method,
                    args.row_spacing, args.plant_spacing, args.bins,
                    summary_out,
                )

            # Growth trajectory plot
            plot_growth_trajectory(
                summaries, summary_out / "growth_trajectory.png")

            # Combined results JSON
            combined_path = summary_out / "session5_results.json"
            with open(combined_path, 'w') as f:
                json.dump(
                    {str(d): s for d, s in summaries.items()},
                    f, indent=2,
                )
            print(f"\n  Combined results: {combined_path}")
        else:
            _run_single_day(
                args.day, args.par, args.tair, args.method,
                args.row_spacing, args.plant_spacing, args.bins,
                summary_out,
            )

    elif args.command == "agroc-export":
        import json
        import numpy as np
        from .growth.grow import (
            grow_plant, run_photosynthesis,
            extract_lai_profile, extract_rld_profile, export_rrd_in,
        )
        from .carbon import solve_carbon_partitioning
        from .agroc import export_agroc_timestep, export_coupling_csv
        from .config import DEFAULT_XML, OUTPUT_DIR, get_species_name

        agroc_out = OUTPUT_DIR / "session6"
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
            print(f"SESSION 6: AgroC EXPORT — {species} Day {day}")
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

            # 5. RLD + rrd.in export (re-use session 2)
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

    elif args.command == "integration-test":
        from .tests.test_session8_integration import main as integ_main
        integ_main(
            day=args.day,
            skip_dart=args.skip_dart,
            skip_agroc=args.skip_agroc,
        )
    elif args.command == "grow":
        sys.argv = [sys.argv[0]] + remaining
        from .growth.grow import main
        main()
    elif args.command == "calibrate":
        sys.argv = [sys.argv[0]] + remaining
        from .growth.calibrate import main
        main()


if __name__ == "__main__":
    cli()
