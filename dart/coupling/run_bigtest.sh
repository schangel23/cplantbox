#!/usr/bin/env bash
# =============================================================================
# Big Test: 25 Growth Days × Full Diurnal × Baleno × Iterative Tuzet gs
# =============================================================================
#
# Runs the complete CPlantBox-DART-Baleno-photosynthesis coupling chain for
# 25 growth stages (day 10 to day 58, every 2 days), with:
#   - 9 unique plant realizations per day
#   - Hourly sun angle resolution (60-min timestep)
#   - Baleno energy balance per timestep
#   - Iterative Tuzet-Baleno gs coupling (Phase 10)
#
# ESTIMATED RUNTIME: ~20-30 hours (server-dependent)
#
# USAGE:
#   # Option 1 — run in tmux (recommended, survives SSH disconnect)
#   tmux new -s bigtest
#   bash dart/coupling/run_bigtest.sh
#   # Detach: Ctrl+B, D — Re-attach: tmux attach -t bigtest
#
#   # Option 2 — run with nohup (background, survives SSH disconnect)
#   nohup bash dart/coupling/run_bigtest.sh &
#   tail -f dart/coupling/output/bigtest_run.log
#
#   # Option 3 — run without iterative gs (faster, ~10-15h)
#   bash dart/coupling/run_bigtest.sh --no-iterate-gs
#
# OUTPUT:
#   dart/coupling/output/diurnal/day{N}/hourly_results.csv     per-day hourly data
#   dart/coupling/output/diurnal/day{N}/daily_summary.json     per-day carbon totals
#   dart/coupling/output/diurnal/day{N}/diurnal_curve.png      per-day plot
#   dart/coupling/output/diurnal/growth_series/                growth curve across all days
#   dart/coupling/output/bigtest_run.log                       full run log
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

VENV="$REPO_ROOT/cpbenv"
LOG_FILE="$SCRIPT_DIR/output/bigtest_run.log"

# 25 growth days: day 10 to day 58, every 2 days
GROWTH_DAYS="10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58"

TIMESTEP_MIN=60   # hourly (faster); use 30 for higher temporal resolution

# Iterative gs coupling flag (default: enabled)
ITERATE_GS="--iterate-gs"
for arg in "$@"; do
  [[ "$arg" == "--no-iterate-gs" ]] && ITERATE_GS=""
done

# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------
if [[ ! -d "$VENV" ]]; then
  echo "ERROR: venv not found at $VENV"
  echo "  Run the setup instructions in dart/SERVER_SETUP.md first."
  exit 1
fi

if [[ -z "${DART_HOME:-}" ]]; then
  echo "WARNING: DART_HOME is not set."
  echo "  Set it with: export DART_HOME=/path/to/DART"
  echo "  Or edit dart/coupling/config.py directly."
fi

if [[ -z "${DARTRC:-}" ]]; then
  echo "WARNING: DARTRC is not set."
  echo "  Set it with: export DARTRC=~/.dartrcv1457 (adjust filename)"
fi

if [[ -z "${BALENO_PYTHON:-}" ]]; then
  echo "WARNING: BALENO_PYTHON is not set."
  echo "  Set it with: export BALENO_PYTHON=/path/to/dart-eb-venv/bin/python3"
fi

mkdir -p "$SCRIPT_DIR/output"

# ---------------------------------------------------------------------------
# Activate venv
# ---------------------------------------------------------------------------
# shellcheck disable=SC1090
source "$VENV/bin/activate"
echo "Python: $(which python3) ($(python3 --version))"
python3 -c "import plantbox; print('CPlantBox: OK')"
python3 -c "import pytools4dart; print('pytools4dart: OK')"

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
echo "============================================================"
echo "BIG TEST: 25 growth days, diurnal + Baleno + iterative gs"
echo "Growth days: $GROWTH_DAYS"
echo "Timestep: ${TIMESTEP_MIN} min"
echo "Iterate gs: ${ITERATE_GS:-disabled}"
echo "Log: $LOG_FILE"
echo "Start: $(date)"
echo "============================================================"

cd "$REPO_ROOT"

python3 -m dart.coupling diurnal \
  --growth-days "$GROWTH_DAYS" \
  --timestep-min "$TIMESTEP_MIN" \
  $ITERATE_GS \
  2>&1 | tee "$LOG_FILE"

echo "============================================================"
echo "DONE: $(date)"
echo "Results: $SCRIPT_DIR/output/diurnal/"
echo "============================================================"
