"""CLI entry point: python -m dart.dashboard [--port 8050] [--host 127.0.0.1] [--debug]."""

import argparse


def main():
    parser = argparse.ArgumentParser(
        prog="dart.dashboard",
        description="CPlantBox-DART coupling web dashboard.",
    )
    parser.add_argument("--port", type=int, default=8050, help="Server port (default: 8050)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Bind address (default: 127.0.0.1)")
    parser.add_argument("--debug", action="store_true", help="Enable Dash debug mode")
    args = parser.parse_args()

    from .app import create_app

    app = create_app()
    app.run(port=args.port, host=args.host, debug=args.debug)


if __name__ == "__main__":
    main()
