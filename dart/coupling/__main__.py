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
        "--site", type=str, default=None,
        help="Site name (e.g. us-ne1, juelich). Sets COUPLING_SITE env var.",
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

    # Stage 2 Session 7 — AgroC Fortran run
    p_agroc_run = sub.add_parser("agroc-run",
                                 help="Session 7: run AgroC with ExternalPlantMode")
    p_agroc_run.add_argument("--agroc-src", type=str, default=None,
                             help="Path to AgroC source dir (default: AGROC_SRC env var)")
    p_agroc_run.add_argument("--coupling-csv", type=str, required=True,
                             help="Path to coupling.csv from agroc-export")
    p_agroc_run.add_argument("--output-dir", type=str, default=None,
                             help="Output directory (default: coupling/output/agroc_run)")
    p_agroc_run.add_argument("--timeout", type=int, default=300,
                             help="AgroC timeout in seconds (default: 300)")

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

    # Pipeline runner (config-file-driven)
    p_run = sub.add_parser("run", help="Run pipeline from config JSON")
    p_run.add_argument("config_file", help="Path to pipeline config JSON")
    p_run.add_argument("--validate-only", action="store_true",
                       help="Only validate system, do not run")

    # Config generator
    p_config = sub.add_parser("create-config",
                              help="Generate default config JSON")
    p_config.add_argument("output", nargs="?", default="pipeline_config.json",
                          help="Output path (default: pipeline_config.json)")

    # Dashboard
    p_dash = sub.add_parser("dashboard", help="Launch web dashboard")
    p_dash.add_argument("--port", type=int, default=8050)
    p_dash.add_argument("--host", type=str, default="127.0.0.1")
    p_dash.add_argument("--debug", action="store_true")

    args, remaining = parser.parse_known_args()

    # Set env vars BEFORE importing subcommands (config.py reads them at import)
    if args.species:
        os.environ["COUPLING_SPECIES"] = args.species.lower()
    if args.site:
        os.environ["COUPLING_SITE"] = args.site.lower()
    if args.threads is not None:
        os.environ["DART_THREADS"] = str(args.threads)
        # Limit CPU affinity — LuxCore ignores nbThreads XML and uses all cores
        try:
            n = min(args.threads, os.cpu_count() or args.threads)
            os.sched_setaffinity(0, set(range(n)))
        except (OSError, AttributeError):
            pass

    if args.command == "simulation":
        from .dart.simulation import main
        main()
    elif args.command == "baleno":
        from .dart.baleno_standalone import main
        main()
    elif args.command == "photosynthesis":
        from .photosynthesis.coupled import main
        main()
    elif args.command == "validate":
        from .validation.validate import main
        main()
    elif args.command == "diurnal":
        # diurnal has its own argparse — pass remaining args via sys.argv
        sys.argv = [sys.argv[0]] + remaining
        from .photosynthesis.diurnal import main
        main()
    elif args.command == "rld":
        from .growth.profiles import main_rld
        main_rld(args)
    elif args.command == "carbon":
        from .carbon.cli import main_carbon
        main_carbon(args)
    elif args.command == "summary":
        from .growth.profiles import main_summary
        main_summary(args)
    elif args.command == "agroc-export":
        from .agroc.export import main_export
        main_export(args)
    elif args.command == "agroc-run":
        from .agroc.run import main_agroc_run
        main_agroc_run(args)
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
    elif args.command == "run":
        from .pipeline import PipelineConfig, PipelineRunner
        config = PipelineConfig.load(args.config_file)
        if args.species:
            config.species = args.species
        if args.threads is not None:
            config.threads = args.threads
        runner = PipelineRunner(config)
        if args.validate_only:
            v = runner.validate_system()
            for name, info in v.items():
                status = "OK" if info["ok"] else "FAIL"
                detail = f" ({info['error']})" if info.get("error") else ""
                print(f"  {name}: {status}{detail}")
        else:
            runner.run()
    elif args.command == "create-config":
        from .pipeline import PipelineConfig
        config = PipelineConfig()
        if args.species:
            config.species = args.species
        config.save(args.output)
        print(f"Config saved to {args.output}")
    elif args.command == "dashboard":
        from ..dashboard import create_app
        app = create_app()
        app.run(port=args.port, host=args.host, debug=args.debug)


if __name__ == "__main__":
    cli()
