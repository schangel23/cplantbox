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

    # Growth helpers
    sub.add_parser("grow", help="Grow calibrated plant")
    sub.add_parser("calibrate", help="Calibrate maize XML")

    args, remaining = parser.parse_known_args()

    # Set species env var BEFORE importing subcommands (config.py reads it)
    if args.species:
        os.environ["COUPLING_SPECIES"] = args.species.lower()

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
