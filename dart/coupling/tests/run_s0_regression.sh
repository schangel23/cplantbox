#!/usr/bin/env bash
# S0 regression bundle — ADR §S0.8 ("Add CI step that runs the matrix sweep
# + the non-maize opt-in pytest on every PR").
#
# Two gates, both required green:
#   1. Native-XML matrix sweep (Lock #7 topology + per-node sha256) against
#      tests/baselines/cross_species_baseline_pre_s0.json — every XML in
#      gui/cplantbox/params/*.xml must reproduce its pre-S0 fingerprint.
#   2. test_multi_phase_stem_non_maize.py — wheat XML opted in to
#      MultiPhaseStemGrowth with placeholder per-rank arrays must simulate
#      30 days without crash, NaN, or unhandled-field warning.
#
# Usage (from /home/lukas/PHD/CPlantBox):
#   bash dart/coupling/tests/run_s0_regression.sh
#
# Wire into a fork-side GH Actions step (or pre-push hook) when ready to
# enforce on every PR. Not added to upstream .github/workflows/testing.yml
# because dart/coupling/ does not exist upstream.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

PY="${CPB_PYTHON:-cpbenv/bin/python3}"
if [[ ! -x "$PY" ]]; then
  echo "ERROR: cpbenv interpreter not found at $PY (override with CPB_PYTHON=...)" >&2
  exit 1
fi

echo "==> [1/2] cross-species matrix sweep (--verify)"
"$PY" dart/coupling/tests/baselines/capture_cross_species_baseline.py --verify

echo
echo "==> [2/2] non-maize MultiPhaseStemGrowth opt-in pytest"
"$PY" -m pytest dart/coupling/tests/test_multi_phase_stem_non_maize.py -v

echo
echo "S0 regression bundle: PASSED"
