#!/usr/bin/env bash
# =============================================================================
# Full Pipeline: CPlantBox → DART RT → Baleno EB → Photosynthesis → Carbon
#                → AgroC Export → Integration Test → Diurnal Loop
# =============================================================================
#
# End-to-end validation of the complete Stage 1 + Stage 2 pipeline.
# Runs each component sequentially with verification gates.
#
# MODES:
#   --quick       Skip DART/Baleno, uniform PAR (server without DART install)
#   --full        Full DART + Baleno + iterative gs (requires DART + license)
#   --diurnal     Also run multi-day diurnal loop (adds ~2-8h depending on mode)
#   --days N      Override growth day (default: 55)
#
# ESTIMATED RUNTIME:
#   Quick mode:   ~5 min
#   Full mode:    ~30 min (single day)
#   Full+diurnal: ~8-20h (25 growth days × hourly)
#
# USAGE:
#   # Quick test (no DART needed)
#   bash dart/coupling/run_full_pipeline.sh --quick
#
#   # Full test with DART (single day)
#   bash dart/coupling/run_full_pipeline.sh --full
#
#   # Full test + diurnal sweep (tmux recommended)
#   tmux new -s pipeline
#   bash dart/coupling/run_full_pipeline.sh --full --diurnal
#
#   # Detach: Ctrl+B, D — Re-attach: tmux attach -t pipeline
#
# ENVIRONMENT:
#   AGROC_SRC     Path to AgroC source dir with compiled binary (auto-detected)
#   DART_HOME     Path to DART installation (required for --full)
#   DARTRC        Path to DART license file (required for --full)
#   BALENO_PYTHON Path to Baleno Python interpreter (required for --full)
#
# OUTPUT:
#   dart/coupling/output/pipeline_run/
#     pipeline.log           full log
#     step1_grow/            plant growth + mesh
#     step2_rld/             RLD profiles + rrd.in
#     step3_carbon/          carbon partitioning (phloem + DVS)
#     step4_summary/         LAI + plant summary
#     step5_agroc/           AgroC coupling CSV + conservation
#     step5b_agroc_run/      AgroC Fortran with ExternalPlantMode
#     step6_session8/        integration test results
#     step7_dart/            DART RT (full mode only)
#     step8_baleno/          Baleno EB (full mode only)
#     step9_iterative/       iterative gs (full mode only)
#     step10_diurnal/        multi-day diurnal (if --diurnal)
#     pipeline_summary.json  per-step pass/fail + timings
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
MODE="quick"
RUN_DIURNAL=false
DAY=55
DIURNAL_DAYS="10,14,18,22,26,30,34,38,42,46,50,54,58"
DIURNAL_TIMESTEP=60

for arg in "$@"; do
  case "$arg" in
    --quick)     MODE="quick" ;;
    --full)      MODE="full" ;;
    --diurnal)   RUN_DIURNAL=true ;;
    --days=*)    DAY="${arg#--days=}" ;;
    --days)      shift; DAY="$1" ;; # handled below
    *)           ;;
  esac
done

# Handle --days N (two-arg form)
ARGS=("$@")
for i in "${!ARGS[@]}"; do
  if [[ "${ARGS[$i]}" == "--days" ]] && [[ -n "${ARGS[$((i+1))]:-}" ]]; then
    DAY="${ARGS[$((i+1))]}"
  fi
done

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
VENV="$REPO_ROOT/cpbenv"
OUTDIR="$SCRIPT_DIR/output/pipeline_run"
LOG="$OUTDIR/pipeline.log"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

STEP_RESULTS=()
STEP_TIMES=()
TOTAL_START=$(date +%s)

step_header() {
  local num=$1 name=$2
  echo ""
  echo "============================================================"
  echo -e "${CYAN}STEP $num: $name${NC}"
  echo "============================================================"
  echo ""
}

step_pass() {
  local num=$1 name=$2 elapsed=$3
  echo -e "  ${GREEN}[PASS]${NC} Step $num: $name (${elapsed}s)"
  STEP_RESULTS+=("PASS:$num:$name")
  STEP_TIMES+=("$elapsed")
}

step_fail() {
  local num=$1 name=$2 elapsed=$3
  echo -e "  ${RED}[FAIL]${NC} Step $num: $name (${elapsed}s)"
  STEP_RESULTS+=("FAIL:$num:$name")
  STEP_TIMES+=("$elapsed")
}

step_skip() {
  local num=$1 name=$2
  echo -e "  ${YELLOW}[SKIP]${NC} Step $num: $name"
  STEP_RESULTS+=("SKIP:$num:$name")
  STEP_TIMES+=("0")
}

run_step() {
  # run_step <step_num> <step_name> <command...>
  # Runs command, captures exit code, logs to $LOG, records pass/fail.
  # Does NOT abort on failure (continues to next step).
  local num=$1 name=$2
  shift 2
  local start end elapsed rc
  start=$(date +%s)
  step_header "$num" "$name" | tee -a "$LOG"

  # Run command with output going to both terminal and log.
  # Use pipefail to capture the command's exit code through tee.
  rc=0
  "$@" 2>&1 | tee -a "$LOG" || rc=$?

  end=$(date +%s)
  elapsed=$((end - start))
  if [[ $rc -eq 0 ]]; then
    step_pass "$num" "$name" "$elapsed" | tee -a "$LOG"
  else
    step_fail "$num" "$name" "$elapsed" | tee -a "$LOG"
  fi
}

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
mkdir -p "$OUTDIR"

echo "============================================================" | tee "$LOG"
echo "FULL PIPELINE TEST" | tee -a "$LOG"
echo "============================================================" | tee -a "$LOG"
echo "  Mode:        $MODE" | tee -a "$LOG"
echo "  Day:         $DAY" | tee -a "$LOG"
echo "  Diurnal:     $RUN_DIURNAL" | tee -a "$LOG"
echo "  Repository:  $REPO_ROOT" | tee -a "$LOG"
echo "  Output:      $OUTDIR" | tee -a "$LOG"
echo "  Start:       $(date)" | tee -a "$LOG"
echo "============================================================" | tee -a "$LOG"

# Sanity check: venv
if [[ ! -d "$VENV" ]]; then
  echo -e "${RED}ERROR: venv not found at $VENV${NC}" | tee -a "$LOG"
  echo "  Create it with: python3 -m venv cpbenv && source cpbenv/bin/activate && pip install ." | tee -a "$LOG"
  exit 1
fi

# shellcheck disable=SC1090
source "$VENV/bin/activate"
echo "  Python:      $(which python3) ($(python3 --version 2>&1))" | tee -a "$LOG"

# Verify CPlantBox
python3 -c "import plantbox; print(f'  CPlantBox:   OK ({plantbox.__file__})')" 2>&1 | tee -a "$LOG"

# Verify DART (full mode only)
if [[ "$MODE" == "full" ]]; then
  if [[ -z "${DART_HOME:-}" ]]; then
    echo -e "${RED}ERROR: DART_HOME not set. Required for --full mode.${NC}" | tee -a "$LOG"
    echo "  export DART_HOME=/path/to/DART" | tee -a "$LOG"
    exit 1
  fi
  if [[ -z "${DARTRC:-}" ]]; then
    echo -e "${YELLOW}WARNING: DARTRC not set. DART may fail without license.${NC}" | tee -a "$LOG"
  fi
  if [[ -z "${BALENO_PYTHON:-}" ]]; then
    echo -e "${YELLOW}WARNING: BALENO_PYTHON not set. Baleno steps will fail.${NC}" | tee -a "$LOG"
  fi
  python3 -c "import pytools4dart; print(f'  pytools4dart: OK')" 2>&1 | tee -a "$LOG" || true
fi

cd "$REPO_ROOT"

# ---------------------------------------------------------------------------
# STEP 1: Grow plant (roots + shoots) + G3 mesh export
# ---------------------------------------------------------------------------
STEP1_OUT="$OUTDIR/step1_grow"
mkdir -p "$STEP1_OUT"

run_step 1 "Grow plant (day $DAY, roots+shoots, G3 mesh)" \
  python3 -m dart.coupling grow \
    --xml dart/coupling/data/maize_calibrated.xml \
    --days "$DAY" \
    --output "$STEP1_OUT/maize_day${DAY}" \
    --photosynthesis

# ---------------------------------------------------------------------------
# STEP 2: RLD profile extraction
# ---------------------------------------------------------------------------
STEP2_OUT="$OUTDIR/step2_rld"
mkdir -p "$STEP2_OUT"

run_step 2 "RLD profile + rrd.in (multi-day)" \
  python3 -m dart.coupling rld \
    --multi-day --layers 20 --depth 100

# ---------------------------------------------------------------------------
# STEP 3: Carbon partitioning (phloem + DVS)
# ---------------------------------------------------------------------------
STEP3_OUT="$OUTDIR/step3_carbon"
mkdir -p "$STEP3_OUT"

run_step 3 "Carbon partitioning (phloem, day $DAY)" \
  python3 -m dart.coupling carbon \
    --day "$DAY" --method phloem

# Also run DVS for comparison (non-gated, logged only)
echo "  --- DVS comparison ---" | tee -a "$LOG"
python3 -m dart.coupling carbon \
  --day "$DAY" --method dvs \
  2>&1 | tee -a "$LOG" || true

# ---------------------------------------------------------------------------
# STEP 4: LAI + plant summary (multi-day trajectory)
# ---------------------------------------------------------------------------
STEP4_OUT="$OUTDIR/step4_summary"
mkdir -p "$STEP4_OUT"

run_step 4 "LAI + plant summary (multi-day)" \
  python3 -m dart.coupling summary \
    --multi-day --method auto

# ---------------------------------------------------------------------------
# STEP 5: AgroC coupling export (multi-day)
# ---------------------------------------------------------------------------
STEP5_OUT="$OUTDIR/step5_agroc"
mkdir -p "$STEP5_OUT"

run_step 5 "AgroC coupling CSV (multi-day)" \
  python3 -m dart.coupling agroc-export \
    --multi-day --method auto

# ---------------------------------------------------------------------------
# STEP 5b: Run AgroC with ExternalPlantMode
# ---------------------------------------------------------------------------
AGROC_SRC="${AGROC_SRC:-}"

# Auto-detect AgroC source directory
if [[ -z "$AGROC_SRC" ]]; then
  if [[ -f "/home/lukas/PHD/agroC_20250327_1511/src/agroC" ]]; then
    AGROC_SRC="/home/lukas/PHD/agroC_20250327_1511/src"
  elif [[ -f "/media/data/Lukas/agroC_20250327_1511/src/agroC" ]]; then
    AGROC_SRC="/media/data/Lukas/agroC_20250327_1511/src"
  fi
fi

# Find the coupling CSV produced by Step 5
COUPLING_CSV=""
if [[ -d "$SCRIPT_DIR/output/session6" ]]; then
  COUPLING_CSV=$(find "$SCRIPT_DIR/output/session6" -name "*_coupling.csv" -type f | head -1)
fi

if [[ -n "$AGROC_SRC" ]] && [[ -f "$AGROC_SRC/agroC" ]] && [[ -n "$COUPLING_CSV" ]]; then
  STEP5B_OUT="$OUTDIR/step5b_agroc_run"
  mkdir -p "$STEP5B_OUT"

  run_step 5b "AgroC soil simulation (ExternalPlantMode)" \
    python3 -m dart.coupling agroc-run \
      --agroc-src "$AGROC_SRC" \
      --coupling-csv "$COUPLING_CSV" \
      --output-dir "$STEP5B_OUT"
else
  if [[ -z "$AGROC_SRC" ]] || [[ ! -f "${AGROC_SRC:-/nonexistent}/agroC" ]]; then
    step_skip 5b "AgroC run (binary not found; set AGROC_SRC)"
  else
    step_skip 5b "AgroC run (coupling CSV not found; run Step 5 first)"
  fi
fi

# ---------------------------------------------------------------------------
# STEP 6: Session 8 integration test
# ---------------------------------------------------------------------------
STEP6_OUT="$OUTDIR/step6_session8"
mkdir -p "$STEP6_OUT"

if [[ "$MODE" == "quick" ]]; then
  run_step 6 "Integration test (skip-DART, skip-AgroC)" \
    python3 -m dart.coupling integration-test \
      --day "$DAY" --skip-dart --skip-agroc
else
  run_step 6 "Integration test (full)" \
    python3 -m dart.coupling integration-test \
      --day "$DAY"
fi

# ---------------------------------------------------------------------------
# STEP 7-9: DART RT → Baleno EB → Iterative gs (full mode only)
# ---------------------------------------------------------------------------
if [[ "$MODE" == "full" ]]; then

  # STEP 7: DART RT
  STEP7_OUT="$OUTDIR/step7_dart"
  mkdir -p "$STEP7_OUT"
  run_step 7 "DART radiative transfer (day $DAY)" \
    python3 -m dart.coupling simulation --day "$DAY"

  # STEP 8: Baleno energy balance
  STEP8_OUT="$OUTDIR/step8_baleno"
  mkdir -p "$STEP8_OUT"
  run_step 8 "Baleno energy balance" \
    python3 -m dart.coupling baleno

  # STEP 9: Iterative Tuzet-Baleno gs
  STEP9_OUT="$OUTDIR/step9_iterative"
  mkdir -p "$STEP9_OUT"
  run_step 9 "Iterative Tuzet-Baleno gs coupling" \
    python3 -m dart.coupling photosynthesis

else
  step_skip 7 "DART RT (quick mode)"
  step_skip 8 "Baleno EB (quick mode)"
  step_skip 9 "Iterative gs (quick mode)"
fi

# ---------------------------------------------------------------------------
# STEP 10: Multi-day diurnal loop (optional)
# ---------------------------------------------------------------------------
if [[ "$RUN_DIURNAL" == true ]]; then
  STEP10_OUT="$OUTDIR/step10_diurnal"
  mkdir -p "$STEP10_OUT"

  DIURNAL_FLAGS=""
  if [[ "$MODE" == "quick" ]]; then
    DIURNAL_FLAGS="--no-baleno --skip-photosynthesis"
  else
    DIURNAL_FLAGS="--iterate-gs"
  fi

  run_step 10 "Diurnal loop (${DIURNAL_DAYS}, ${DIURNAL_TIMESTEP}min)" \
    python3 -m dart.coupling diurnal \
      --growth-days "$DIURNAL_DAYS" \
      --timestep-min "$DIURNAL_TIMESTEP" \
      $DIURNAL_FLAGS
else
  step_skip 10 "Diurnal loop (not requested)"
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
TOTAL_END=$(date +%s)
TOTAL_ELAPSED=$((TOTAL_END - TOTAL_START))

echo "" | tee -a "$LOG"
echo "============================================================" | tee -a "$LOG"
echo "PIPELINE SUMMARY" | tee -a "$LOG"
echo "============================================================" | tee -a "$LOG"

N_PASS=0
N_FAIL=0
N_SKIP=0

for i in "${!STEP_RESULTS[@]}"; do
  result="${STEP_RESULTS[$i]}"
  time="${STEP_TIMES[$i]}"
  status="${result%%:*}"
  rest="${result#*:}"
  num="${rest%%:*}"
  name="${rest#*:}"

  case "$status" in
    PASS) echo -e "  ${GREEN}[PASS]${NC} Step $num: $name (${time}s)" | tee -a "$LOG"; ((N_PASS++)) ;;
    FAIL) echo -e "  ${RED}[FAIL]${NC} Step $num: $name (${time}s)" | tee -a "$LOG"; ((N_FAIL++)) ;;
    SKIP) echo -e "  ${YELLOW}[SKIP]${NC} Step $num: $name" | tee -a "$LOG"; ((N_SKIP++)) ;;
  esac
done

echo "" | tee -a "$LOG"
echo "  Total: $N_PASS passed, $N_FAIL failed, $N_SKIP skipped" | tee -a "$LOG"
echo "  Time:  ${TOTAL_ELAPSED}s ($(( TOTAL_ELAPSED / 60 ))m $(( TOTAL_ELAPSED % 60 ))s)" | tee -a "$LOG"
echo "  End:   $(date)" | tee -a "$LOG"
echo "  Log:   $LOG" | tee -a "$LOG"
echo "============================================================" | tee -a "$LOG"

# Write machine-readable summary JSON
SUMMARY_JSON="$OUTDIR/pipeline_summary.json"
{
  echo "{"
  echo "  \"mode\": \"$MODE\","
  echo "  \"day\": $DAY,"
  echo "  \"diurnal\": $([[ "$RUN_DIURNAL" == true ]] && echo 'true' || echo 'false'),"
  echo "  \"total_time_s\": $TOTAL_ELAPSED,"
  echo "  \"passed\": $N_PASS,"
  echo "  \"failed\": $N_FAIL,"
  echo "  \"skipped\": $N_SKIP,"
  echo "  \"all_passed\": $([[ $N_FAIL -eq 0 ]] && echo 'true' || echo 'false'),"
  echo "  \"steps\": ["
  for i in "${!STEP_RESULTS[@]}"; do
    result="${STEP_RESULTS[$i]}"
    time="${STEP_TIMES[$i]}"
    status="${result%%:*}"
    rest="${result#*:}"
    num="${rest%%:*}"
    name="${rest#*:}"
    comma=$([[ $i -lt $(( ${#STEP_RESULTS[@]} - 1 )) ]] && echo "," || echo "")
    echo "    {\"step\": $num, \"name\": \"$name\", \"status\": \"$status\", \"time_s\": $time}$comma"
  done
  echo "  ]"
  echo "}"
} > "$SUMMARY_JSON"
echo "  Summary JSON: $SUMMARY_JSON" | tee -a "$LOG"

# Exit with failure if any step failed
if [[ $N_FAIL -gt 0 ]]; then
  echo -e "\n${RED}PIPELINE FAILED ($N_FAIL steps failed)${NC}" | tee -a "$LOG"
  exit 1
else
  echo -e "\n${GREEN}PIPELINE PASSED${NC}" | tee -a "$LOG"
  exit 0
fi
