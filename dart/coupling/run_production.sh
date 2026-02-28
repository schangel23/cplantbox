#!/usr/bin/env bash
# =============================================================================
# Production Run: Full Multifield Diurnal Simulation
# =============================================================================
#
# Runs the complete CPlantBox-DART-Baleno coupling campaign:
#   9 unique plants x 13 growth days x hourly timesteps
#   DART RT + Baleno EB + iterative gs + carbon partitioning + AgroC export
#
# Unlike run_full_pipeline.sh (which validates each component separately),
# this script runs the full integrated diurnal loop with all features enabled.
#
# USAGE:
#   bash dart/coupling/run_production.sh                    # default 13 days
#   bash dart/coupling/run_production.sh --resume           # resume interrupted
#   bash dart/coupling/run_production.sh --no-baleno        # skip energy balance
#   AGROC_SRC=/path/to/agroc bash dart/coupling/run_production.sh --with-agroc
#
# RUNTIME: ~100-200 hours (13 days x 12h x 9 plants x DART+Baleno+iterative)
#          Recommended: run in tmux on the data server
#
# OUTPUT:
#   dart/coupling/output/diurnal/day<N>/   per-day results (hourly CSV, plots)
#   dart/coupling/output/diurnal/production/  combined coupling CSV + summary
#   dart/coupling/output/diurnal/production_checkpoint.json  resume checkpoint
#
# ENVIRONMENT (auto-detected, override with env vars):
#   DART_HOME       Path to DART installation
#   DARTRC          Path to DART license file
#   BALENO_PYTHON   Path to Baleno Python interpreter
#   AGROC_SRC       Path to AgroC source dir (only for --with-agroc)
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Paths: auto-detect local vs server
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Auto-detect venv (server uses Python 3.12, local uses 3.14)
if [[ -d "$REPO_ROOT/cpbenv" ]]; then
    VENV="$REPO_ROOT/cpbenv"
else
    echo "ERROR: cpbenv not found at $REPO_ROOT/cpbenv"
    exit 1
fi

# Auto-detect DART
if [[ -z "${DART_HOME:-}" ]]; then
    if [[ -d "/media/data/Lukas/DART" ]]; then
        export DART_HOME="/media/data/Lukas/DART"
        export DARTRC="/media/data/Lukas/DART/.dartrc"
        # Baleno needs Python 3.11+ (StrEnum). DART bundles Python 3.8 which is
        # too old. Use cpbenv's Python 3.12 instead.
        export BALENO_PYTHON="$REPO_ROOT/cpbenv/bin/python3"
    elif [[ -d "/home/lukas/DART" ]]; then
        export DART_HOME="/home/lukas/DART"
    fi
fi

# ---------------------------------------------------------------------------
# Activate venv
# ---------------------------------------------------------------------------
# shellcheck disable=SC1090
source "$VENV/bin/activate"
cd "$REPO_ROOT"

echo "============================================================"
echo "PRODUCTION RUN: Full Multifield Diurnal Simulation"
echo "============================================================"
echo "  Repository:    $REPO_ROOT"
echo "  Python:        $(which python3) ($(python3 --version 2>&1))"
echo "  DART_HOME:     ${DART_HOME:-NOT SET}"
echo "  BALENO_PYTHON: ${BALENO_PYTHON:-NOT SET} ($($BALENO_PYTHON --version 2>&1 || echo MISSING))"
echo "  Start:         $(date)"
echo "============================================================"

# ---------------------------------------------------------------------------
# Run the production diurnal campaign
# ---------------------------------------------------------------------------
python3 -m dart.coupling diurnal \
    --growth-days "10,14,18,22,26,30,34,38,42,46,50,54,58" \
    --timestep-min 60 \
    --iterate-gs \
    --with-carbon \
    --carbon-method auto \
    "$@"

echo ""
echo "============================================================"
echo "PRODUCTION RUN COMPLETE: $(date)"
echo "============================================================"
