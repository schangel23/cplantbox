"""B.5 coordination unit test (Fournier-Andrieu internode kinetics).

Exit gate for Session 2 of
`Literature/Chapter 1/Concepts/CPBOXBALENOCOUPLING/PLAN_FOURNIER_ANDRIEU_INTERNODE_KINETICS_2026-04-23.md`.

Validates the kinetic algorithm that will be ported to C++ in Session 3 (B.3):
phase-boundary arithmetic on the Andrieu axis, rank-convention dispatch
(internode uses c_n, sheath uses c_{n-1}), collar-emergence consistency, and
the basal-zero + pre-initiation edge cases.

The module-under-test (`dart.coupling.fa_kinetics`) is the Python reference
implementation; the S3 C++ port in `Stem::calcLengthPerPhytomer` must agree
with it numerically on the same inputs.

Usage (from /home/lukas/PHD/CPlantBox):
    cpbenv/bin/python3 -m pytest dart/coupling/tests/test_fa_coordination.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

COUPLING_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(COUPLING_DIR.parent.parent))

from dart.coupling.fa_kinetics import (  # noqa: E402
    FAParams,
    HALF_PLASTOCHRON_LAG_DEGCD,
    IL_AT_END_PHASE_II_CM,
    IL_INIT_CM,
    init_tt_from_primordium,
    internode_collar_trigger_rank,
    internode_length,
    phase_boundaries,
    sheath_collar_trigger_rank,
    synthesize_collar_schedule,
)


# --- Fixtures --------------------------------------------------------------


PLASTOCHRON_DEGCD = 19.2   # Zhu 2014 Table 2 line 88, Déa


def _fa_params_with_fig12() -> FAParams:
    """FAParams populated with FA 2000 Fig 12 per-rank vectors (phyt 6-15).

    Matches `dart/coupling/data/phase_III_per_rank.json`. IL_final values come
    from FA 2000 Fig 13 right-axis (IL_final_cross_check_cm block in the same
    JSON). Only populated for ranks 6-15 — ranks 1-4 are basal-zero; rank 5 has
    no Fig 12 data in the spec.
    """
    v_n = {
        6: 0.13, 7: 0.20, 8: 0.32, 9: 0.38, 10: 0.36, 11: 0.28,
        12: 0.25, 13: 0.24, 14: 0.19, 15: 0.16,
    }
    D_n = {
        6: 32, 7: 52, 8: 40, 9: 43, 10: 41, 11: 50,
        12: 56, 13: 50, 14: 63, 15: 79,
    }
    IL_final = {
        6: 7.0, 7: 14.0, 8: 22.0, 9: 23.0, 10: 22.0, 11: 22.0,
        12: 20.0, 13: 18.0, 14: 19.0, 15: 16.0,
    }
    return FAParams(
        internode_v_n=v_n,
        internode_D_n=D_n,
        internode_IL_final=IL_final,
    )


def _primordium_schedule(first_tt: float = 50.0) -> dict:
    """Synthetic leaf-n primordium initiation schedule at plastochron 19.2 °Cd."""
    return {n: first_tt + (n - 1) * PLASTOCHRON_DEGCD for n in range(1, 17)}


# --- B.5 assertion 1: Phase II fires exactly at collar emergence -----------


def test_phase_II_start_equals_collar_emergence():
    """FA 2000 line 159: Phase II trigger IS same-rank collar emergence (zero lag).

    Under the initiation-anchored scheme of plan §B.3, the Phase I end boundary
    must coincide with the synthesized same-rank collar emergence for every rank
    that has Fig 12 data. Any drift here would mean the spec's ``9.6 °Cd
    initiation lag + 309 °Cd Phase I duration = same-rank collar emergence``
    identity is broken.
    """
    params = _fa_params_with_fig12()
    primordium = _primordium_schedule()
    collar = synthesize_collar_schedule(primordium, params)

    for n in range(5, 16):  # ranks with meaningful kinetics
        init_tt = init_tt_from_primordium(primordium[n])
        boundaries = phase_boundaries(n, init_tt, params)
        assert boundaries["phase_I_end"] == pytest.approx(collar[n]), (
            f"Phase II start (= Phase I end) for rank {n} must coincide with "
            f"same-rank collar emergence; got phase_I_end={boundaries['phase_I_end']}, "
            f"collar={collar[n]}"
        )


# --- B.5 assertion 2: Phase III starts 25 °Cd after Phase II ---------------


def test_phase_III_start_is_25_degCd_after_phase_II():
    """FA 2000 line 157: Phase II duration = 25 °Cd (X-ray expt 1A, phyt 11-15)."""
    params = _fa_params_with_fig12()
    primordium = _primordium_schedule()

    for n in range(5, 16):
        init_tt = init_tt_from_primordium(primordium[n])
        b = phase_boundaries(n, init_tt, params)
        assert b["phase_II_end"] - b["phase_I_end"] == pytest.approx(25.0), (
            f"Phase II duration for rank {n} must be 25 °Cd"
        )


# --- B.5 assertion 3: Phase III ends at phase_II_end + D_n -----------------


def test_phase_III_end_is_per_rank_D_n():
    """FA 2000 Fig 12B: Phase III duration varies by rank (32-79 °Cd)."""
    params = _fa_params_with_fig12()
    primordium = _primordium_schedule()

    for n, D_n_expected in params.internode_D_n.items():
        init_tt = init_tt_from_primordium(primordium[n])
        b = phase_boundaries(n, init_tt, params)
        assert b["phase_III_end"] - b["phase_II_end"] == pytest.approx(D_n_expected), (
            f"Phase III duration for rank {n} must equal D_n={D_n_expected}"
        )


# --- B.5 assertion 4: Phase IV is 30 °Cd (operational) ---------------------


def test_phase_IV_duration_is_30_degCd_operational():
    """FA 2000 line 261: operational Phase IV end = x2+20; end_III ≈ x2-10."""
    params = _fa_params_with_fig12()
    primordium = _primordium_schedule()

    for n in params.internode_D_n.keys():
        init_tt = init_tt_from_primordium(primordium[n])
        b = phase_boundaries(n, init_tt, params)
        assert b["phase_IV_end"] - b["phase_III_end"] == pytest.approx(30.0), (
            f"Phase IV operational duration for rank {n} must be 30 °Cd"
        )


# --- B.5 assertion 5: Rank-convention off-by-one gate ----------------------


def test_internode_uses_same_rank_collar():
    """FA 2000 line 159: internode-n Phase II trigger is c_n (NOT c_{n-1}).

    The rank-dispatch helper is separate from the length math so a future
    refactor that accidentally uses c_{n-1} for internode would fail THIS
    assertion rather than silently shift the whole H(TT) calendar by one
    phyllochron. Zhu 2014 is the attractor for the mistake (they mix c_n for
    internode and c_{n-1} for sheath in one coordination model).
    """
    for n in range(5, 16):
        assert internode_collar_trigger_rank(n) == n, (
            f"internode-{n} must be triggered by c_{n} (same rank), not c_{n - 1}"
        )


def test_sheath_uses_previous_rank_collar():
    """Zhu 2014: sheath-n dispatch uses c_{n-1}."""
    for n in range(5, 16):
        assert sheath_collar_trigger_rank(n) == n - 1, (
            f"sheath-{n} must be triggered by c_{n - 1} (previous rank)"
        )


def test_rank_convention_mutation_is_detectable():
    """Meta-assertion: if someone reverses the convention, at least one of the
    two rank-dispatch tests above must fail for every n ≥ 5.

    If a future refactor folds both dispatchers into a single helper that
    defaults to one convention, THIS test makes the silent drift visible.
    """
    for n in range(5, 16):
        assert internode_collar_trigger_rank(n) != sheath_collar_trigger_rank(n), (
            f"rank {n}: internode and sheath must disagree on collar rank"
        )


# --- B.5 assertion 6: Initiation lag is 9.6 °Cd half-plastochron -----------


def test_initiation_lag_is_half_plastochron():
    """FA 2000 line 207 / Fournier 2005: 9.6 °Cd lag from primordium to internode init."""
    assert HALF_PLASTOCHRON_LAG_DEGCD == pytest.approx(9.6)
    # Cross-check that 9.6 is half of the Déa plastochron (Zhu 2014 line 88 = 19.2)
    assert HALF_PLASTOCHRON_LAG_DEGCD == pytest.approx(PLASTOCHRON_DEGCD / 2.0, abs=0.1)

    primordium_tt = 100.0
    assert init_tt_from_primordium(primordium_tt) == pytest.approx(109.6)


# --- B.5 assertion 7: tau = 0 returns IL_INIT_CM ---------------------------


def test_length_at_initiation_is_IL_INIT_CM():
    """At tau=0, internode-n exists and has length IL_INIT_CM = 0.0025 cm."""
    params = _fa_params_with_fig12()
    for n in params.internode_v_n.keys():
        assert internode_length(0.0, n, params) == pytest.approx(IL_INIT_CM)


# --- B.5 assertion 8: tau < 0 returns 0 (pre-initiation) -------------------


def test_length_before_initiation_is_zero():
    """Pre-initiation (tau < 0) must return 0, distinct from tau=0 which returns IL_INIT_CM."""
    params = _fa_params_with_fig12()
    for n in params.internode_v_n.keys():
        assert internode_length(-1e-9, n, params) == 0.0, (
            f"rank {n} at tau=-epsilon must be 0 (pre-initiation), not IL_INIT_CM"
        )
        assert internode_length(-100.0, n, params) == 0.0


def test_basal_zero_ranks_always_zero():
    """Zhu 2014 line 127 + He 2021: ranks 1-4 are zero-length at all tau."""
    params = _fa_params_with_fig12()
    for n in params.basal_zero_ranks:
        for tau in (-50.0, 0.0, 50.0, 500.0):
            assert internode_length(tau, n, params) == 0.0


# --- Phase continuity and boundary-value sanity ----------------------------


def test_phase_I_end_matches_phase_II_start_length():
    """IL is continuous at the Phase I→II boundary by construction (linear
    interpolation from IL_end_I to IL_AT_END_PHASE_II_CM)."""
    params = _fa_params_with_fig12()
    for n in params.internode_v_n.keys():
        IL_just_below = internode_length(params.phase_I_duration - 1e-6, n, params)
        IL_just_above = internode_length(params.phase_I_duration + 1e-6, n, params)
        assert IL_just_below == pytest.approx(IL_just_above, abs=1e-3), (
            f"rank {n}: Phase I→II boundary is discontinuous"
        )


def test_phase_II_end_length_is_4p5_cm():
    """FA 2000 line 223: IL at end of Phase II = 4.5 cm (uniform across phyt 7-15)."""
    params = _fa_params_with_fig12()
    phase_II_end_tau = params.phase_I_duration + params.phase_II_duration
    for n in params.internode_v_n.keys():
        if n < 7:
            continue  # FA 2000 line 223 scope is phyt 7-15
        IL_end_II = internode_length(phase_II_end_tau, n, params)
        assert IL_end_II == pytest.approx(IL_AT_END_PHASE_II_CM, abs=1e-6)


def test_phase_III_linear_slope_is_v_n():
    """Phase III is strictly linear at v_n; check two interior samples."""
    params = _fa_params_with_fig12()
    phase_II_end_tau = params.phase_I_duration + params.phase_II_duration
    for n, v_n in params.internode_v_n.items():
        D_n = params.internode_D_n[n]
        tau_mid1 = phase_II_end_tau + 0.25 * D_n
        tau_mid2 = phase_II_end_tau + 0.75 * D_n
        IL1 = internode_length(tau_mid1, n, params)
        IL2 = internode_length(tau_mid2, n, params)
        slope = (IL2 - IL1) / (tau_mid2 - tau_mid1)
        assert slope == pytest.approx(v_n, rel=1e-6), (
            f"rank {n}: Phase III slope should be v_n={v_n}, got {slope}"
        )


def test_phase_IV_approaches_IL_final():
    """Phase IV exponential decay: IL(tau→∞) → IL_final at rate k=0.09."""
    params = _fa_params_with_fig12()
    for n, IL_final in params.internode_IL_final.items():
        phase_III_end_tau = (
            params.phase_I_duration + params.phase_II_duration + params.internode_D_n[n]
        )
        # 5 * 1/k ≈ 55 °Cd gets us to 99.3% of asymptote
        IL_late = internode_length(phase_III_end_tau + 5.0 / params.phase_IV_k, n, params)
        assert IL_late == pytest.approx(IL_final, rel=0.01), (
            f"rank {n}: Phase IV should approach IL_final={IL_final}, got {IL_late}"
        )


# --- Hard Invariant #2: tt_emergence cannot absorb H(TT) residuals ---------


def test_collar_schedule_is_derived_not_overridden():
    """Consistency identity check: ``collar_n == primordium_n + 9.6 + 309``.

    If someone adds a `collar_override_schedule` that bypasses this identity,
    they'd be violating Hard Invariant #2 (tt_emergence absorbs leaf-calendar
    drift only, never H(TT) residuals). The `synthesize_collar_schedule` helper
    is the single source of truth.
    """
    params = _fa_params_with_fig12()
    primordium = _primordium_schedule()
    collar = synthesize_collar_schedule(primordium, params)
    for n, p_tt in primordium.items():
        assert collar[n] == pytest.approx(
            p_tt + HALF_PLASTOCHRON_LAG_DEGCD + params.phase_I_duration
        )


# --- Smoke test: phyt-9 IL_final cross-check from Fig-12/Fig-13 ------------


def test_phyt9_matches_fig13_IL_final_cross_check():
    """Self-consistency: phyt-9 kinetic budget ends near Fig-13 IL_final.

    `phase_III_per_rank.json:47-51` gives: 0.38 * 43 + 4.5 = 20.84 cm at
    end-of-Phase-III; Fig-13 IL_final = 23 cm. Phase IV adds ~2 cm over 30 °Cd.
    Verify the reference implementation reproduces this within 1%.
    """
    params = _fa_params_with_fig12()
    # Sample well past Phase IV to confirm asymptote
    tau_late = params.phase_I_duration + params.phase_II_duration + params.internode_D_n[9] + 100.0
    IL = internode_length(tau_late, 9, params)
    assert IL == pytest.approx(23.0, rel=0.01)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
