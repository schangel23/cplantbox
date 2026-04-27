"""Fournier-Andrieu per-phytomer internode kinetics — pure-Python reference.

This module is the S2 deliverable of
`Literature/Chapter 1/Concepts/CPBOXBALENOCOUPLING/PLAN_FOURNIER_ANDRIEU_INTERNODE_KINETICS_2026-04-23.md`.
It mirrors the C++ algorithm that will land in `Stem::calcLengthPerPhytomer`
(plan §B.3) so that the S3 port can be validated bit-for-bit against this
reference.

Scope (plan §B.5):
  * Phase-boundary arithmetic on the Andrieu axis (Tb=9.8 °C) with tau anchored
    at internode initiation.
  * Rank-convention dispatch: internode-n kinetics consume same-rank collar
    emergence (c_n); sheath-n kinetics consume previous-rank collar (c_{n-1}).
  * A consistency check that Phase I end coincides with same-rank collar
    emergence when the schedule is synthesized from plausible leaf-primordium
    initiation and phyllochron timing.

All constants match plan §B.3 literally — when B.3 ports to C++ those numeric
defaults must be copied verbatim from the shipped `StemRandomParameter` fields
(`r_I=0.023`, `phase_I_duration=309`, `phase_II_duration=25`,
`phase_IV_duration=30`, `phase_IV_k=0.09`, `IL_INIT_CM=0.0025`,
`IL_AT_END_PHASE_II_CM=4.5`, half-plastochron lag 9.6 °Cd).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping, Sequence

IL_INIT_CM = 0.0025            # Zhu 2014 initial internode length at tau=0
IL_AT_END_PHASE_II_CM = 4.5    # FA 2000 line 223 (phyt 7-15 mean)
HALF_PLASTOCHRON_LAG_DEGCD = 9.6  # FA 2000 line 207, Fournier 2005 lines 22-23


@dataclass(frozen=True)
class FAParams:
    """Mirror of the per-stem FA subset of StemRandomParameter."""

    r_I: float = 0.023
    phase_I_duration: float = 309.0
    phase_II_duration: float = 25.0
    phase_IV_duration: float = 30.0
    phase_IV_k: float = 0.09
    basal_zero_ranks: Sequence[int] = field(default_factory=lambda: (1, 2, 3, 4))
    internode_v_n: Mapping[int, float] = field(default_factory=dict)
    internode_D_n: Mapping[int, float] = field(default_factory=dict)
    internode_IL_final: Mapping[int, float] = field(default_factory=dict)


def phase_boundaries(n: int, init_tt_n: float, params: FAParams) -> Dict[str, float]:
    """Phase boundaries for internode n on the Andrieu axis, anchored at initiation.

    Given the Andrieu-axis thermal time at which internode n initiated
    (== leaf-n primordium initiation + HALF_PLASTOCHRON_LAG_DEGCD), returns the
    four phase boundaries as absolute Andrieu-axis values. Matches plan §B.3.
    """
    D_n = params.internode_D_n.get(n, 0.0)
    phase_I_end = init_tt_n + params.phase_I_duration
    phase_II_end = phase_I_end + params.phase_II_duration
    phase_III_end = phase_II_end + D_n
    phase_IV_end = phase_III_end + params.phase_IV_duration
    return {
        "init": init_tt_n,
        "phase_I_end": phase_I_end,         # == phase_II_start
        "phase_II_end": phase_II_end,       # == phase_III_start
        "phase_III_end": phase_III_end,     # == phase_IV_start
        "phase_IV_end": phase_IV_end,
    }


def internode_length(tau: float, n: int, params: FAParams) -> float:
    """Internode length in cm given tau = andrieu_tt - init_tt_n (plan §B.3)."""
    if n in params.basal_zero_ranks:
        return 0.0
    if tau < 0.0:
        return 0.0
    if tau < params.phase_I_duration:
        import math
        return IL_INIT_CM * math.exp(params.r_I * tau)

    phase_II_end = params.phase_I_duration + params.phase_II_duration
    import math
    if tau < phase_II_end:
        IL_end_I = IL_INIT_CM * math.exp(params.r_I * params.phase_I_duration)
        frac = (tau - params.phase_I_duration) / params.phase_II_duration
        return IL_end_I + frac * (IL_AT_END_PHASE_II_CM - IL_end_I)

    D_n = params.internode_D_n.get(n)
    v_n = params.internode_v_n.get(n)
    if D_n is None or v_n is None:
        raise KeyError(
            f"internode_v_n / internode_D_n missing for rank {n}; required for Phase III/IV"
        )
    phase_III_end = phase_II_end + D_n
    if tau < phase_III_end:
        return IL_AT_END_PHASE_II_CM + v_n * (tau - phase_II_end)

    IL_end_III = IL_AT_END_PHASE_II_CM + v_n * D_n
    IL_final = params.internode_IL_final.get(n)
    if IL_final is None:
        raise KeyError(f"internode_IL_final missing for rank {n}; required for Phase IV")
    return IL_final - (IL_final - IL_end_III) * math.exp(
        -params.phase_IV_k * (tau - phase_III_end)
    )


def internode_collar_trigger_rank(n: int) -> int:
    """Rank of the collar that gates internode n's Phase I→II transition.

    FA 2000 line 159: Phase II trigger is same-rank sheath collar emergence.
    Distinct from leaf-elongation coordination (Fournier 2005 line 57) which uses
    two-ranks-below collar. This IS the rank-convention off-by-one trap.
    """
    return n


def sheath_collar_trigger_rank(n: int) -> int:
    """Rank of the collar that gates sheath n's own elongation transitions.

    Zhu 2014 mixes c_n for internode and c_{n-1} for sheath in one coordination
    model — reversing these gives a plausible-looking but biologically wrong
    calendar. The B.5 unit test mutates this helper to verify the test detects
    the flip.
    """
    return n - 1


def init_tt_from_primordium(primordium_tt_n: float) -> float:
    """Internode-n initiates 9.6 °Cd after leaf-n primordium initiation.

    Direct from FA 2000 line 207 / Fournier 2005. Scalar (no per-rank variance)
    per the spec's current choice.
    """
    return primordium_tt_n + HALF_PLASTOCHRON_LAG_DEGCD


def synthesize_collar_schedule(
    primordium_schedule: Mapping[int, float],
    params: FAParams,
) -> Dict[int, float]:
    """Given leaf-n primordium init times, return Andrieu-tt collar emergences.

    Uses the FA consistency identity: same-rank collar emergence coincides with
    end of internode-n's Phase I. That is:
        collar_n = primordium_n + 9.6 + phase_I_duration
    This is what Hard Invariant #2 protects — `tt_emergence` only shifts to
    absorb leaf-appearance drift, never H(TT) residuals.
    """
    return {
        n: init_tt_from_primordium(p) + params.phase_I_duration
        for n, p in primordium_schedule.items()
    }
