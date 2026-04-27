#!/usr/bin/env python3
"""Session 6 (D.1) regression — Fournier-Andrieu hard-invariant asserts.

Plan: PLAN_FOURNIER_ANDRIEU_INTERNODE_KINETICS_2026-04-23.md §D.1.

Codifies the Phase-IV asymptote timing bounds from plan §D.1 against the
Layer-A per-rank kinetics shipped in `data/phase_III_per_rank.json`. These
are pure arithmetic over phase_I + phase_II + D_n + phase_IV and do NOT
require running the simulator — if they fail, either `phase_III_per_rank.json`
or `fa_kinetics.FAParams` constants drifted. The simulator-driven calendar
endpoint (topmost leaf z = 150.35 cm, tassel emergence day ~125) is already
asserted by `test_tassel_peduncle_scope.py`.

Usage:
    cpbenv/bin/python3 -m pytest dart/coupling/tests/test_fa_hard_invariants.py -v
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

COUPLING_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(COUPLING_DIR.parent.parent))

from dart.coupling.fa_kinetics import FAParams  # noqa: E402

KINETICS_PATH = COUPLING_DIR / "data" / "phase_III_per_rank.json"
PLASTOCHRON_DEGCD = 19.2   # Zhu 2014 Table 2; plan §B.5 test fixture


@pytest.fixture(scope="module")
def kinetics() -> dict:
    return json.loads(KINETICS_PATH.read_text())


@pytest.fixture(scope="module")
def params(kinetics) -> FAParams:
    c = kinetics["constants_inherited"]
    return FAParams(
        r_I=c["r_I_phase_I_rate_degCd_inv"],
        phase_I_duration=float(c["phase_I_duration_degCd"]),
        phase_II_duration=float(c["phase_II_duration_degCd"]),
        phase_IV_duration=float(c["phase_IV_duration_degCd_operational"]),
        phase_IV_k=c["phase_IV_k_decay_degCd_inv"],
    )


def _total_duration(params: FAParams, D_n: float) -> float:
    """From internode-n initiation to end of Phase IV (IL_final reached operationally)."""
    return params.phase_I_duration + params.phase_II_duration + D_n + params.phase_IV_duration


def _D(kinetics: dict, rank: int) -> float:
    return float(kinetics["D_n_degCd"]["values"][str(rank)])


# -- Phase IV asymptote timing, plan §D.1 ----------------------------------


def test_phyt6_total_time_to_IL_final(kinetics, params):
    """phyt6 total: phase_I+II+D_6+IV = 309+25+32+30 = 396 °Cd."""
    t = _total_duration(params, _D(kinetics, 6))
    assert 320 <= t <= 420, f"phyt6 IL_final time = {t} °Cd outside (320, 420)"


def test_phyt10_total_time_to_IL_final(kinetics, params):
    """phyt10 total: phase_I+II+D_10+IV = 309+25+41+30 = 405 °Cd."""
    t = _total_duration(params, _D(kinetics, 10))
    assert 350 <= t <= 450, f"phyt10 IL_final time = {t} °Cd outside (350, 450)"


def test_phyt15_total_time_to_IL_final(kinetics, params):
    """phyt15 total: phase_I+II+D_15+IV = 309+25+79+30 = 443 °Cd."""
    t = _total_duration(params, _D(kinetics, 15))
    assert 350 <= t <= 500, f"phyt15 IL_final time = {t} °Cd outside (350, 500)"


def test_upper_stem_finish_after_phyt10_init(kinetics, params):
    """Time from phyt10 init to topmost-rank IL_final completion.

    Plan §D.1: `upper_stem_finish_after_phyt10_init < 600`.
    With 16 vegetative phytomers (maize_calibrated.xml) and plastochron 19.2 °Cd,
    phyt16 initiates 6 plastochrons = 115.2 °Cd after phyt10, then completes
    Phase IV at phyt16_init + 309 + 25 + D_16(90) + 30 = 454 °Cd later.
    Total: 115.2 + 454 = 569.2 °Cd < 600. Covers the D_16 linear extrapolation
    from S5.3 (14.5 °Cd/rank); re-evaluate if N_vegetative_phytomers drops below
    14 or D_n calibration tightens via WebPlotDigitizer.
    """
    topmost_rank = 16
    dt_init = (topmost_rank - 10) * PLASTOCHRON_DEGCD
    topmost_full = _total_duration(params, _D(kinetics, topmost_rank))
    t = dt_init + topmost_full
    assert t < 600, (
        f"upper_stem_finish_after_phyt10_init = {t:.1f} °Cd >= 600; "
        f"dt_init={dt_init:.1f}, topmost_full(D_16)={topmost_full:.1f}"
    )


# -- Consistency vs D.2 internal cross-check -------------------------------


def test_phyt9_IL_derivation_matches_fig13(kinetics):
    """Layer A internal consistency check (§1 of phase_III_per_rank.json).

    phyt9: v_n * D_n + IL_end_II + phase_IV_extrapolation ≈ IL_final.
    Derived value: 0.38 * 43 + 4.5 + 2 = 22.84 cm; Fig 13 reports 23 cm.
    If this fails, either v_n/D_n/IL_final drifted inconsistently.
    """
    v = float(kinetics["v_n_cm_per_degCd"]["expt_1B_primary"]["9"])
    D = float(kinetics["D_n_degCd"]["values"]["9"])
    IL_end_II = float(kinetics["constants_inherited"]["IL_at_end_phase_II_cm_mean"])
    IL_final = float(kinetics["IL_final_cross_check_cm"]["values"]["9"])
    # Add ~2 cm of Phase IV contribution (asymptotic exponential; not exact, but
    # captures the ~90% convergence bundled into IL_final).
    derived = v * D + IL_end_II + 2.0
    rel_err = abs(derived - IL_final) / IL_final
    assert rel_err < 0.05, (
        f"phyt9 derived IL {derived:.2f} vs Fig-13 {IL_final:.2f} "
        f"(rel err {rel_err:.1%} >5%)"
    )
