#!/usr/bin/env python3
"""S2.C — Leaf::simulate dispatches through f_gf->getLength
(ADR_LEAF_KINEMATICS_2026-04-28 §S2 + §M9).

S2.A added MultiPhaseLeafGrowth as dead code; S2.C deletes the
``if (use_fa_kinetics)`` logistic shadow at the old Leaf.cpp:268-286
location and routes every FA-on leaf through ``calcLength`` ->
``f_gf->getLength``.  This test file is the dispatch contract:

  T1  Source contains no leaf-side ``use_fa_kinetics`` consumer
      (grep guard so the shadow does not creep back in).

  T2  Under gf=6, the maize subType=2 leaf grown for several days
      tracks the Andrieu Phase-E exponential ``L = L_min * exp(R1 *
      (TT - T0))`` within a small discretisation tolerance.

  T3  Under gf=1 (the legacy default) the same leaf grows on the
      ExponentialGrowth ``r * dt`` curve — proves S2.C did not
      silently change the default leaf growth function.

  T4  Under gf=6 with R1_n=0 (unconfigured), getLength falls through
      to ExponentialGrowth's ``k * (1 - exp(-r/k * t))`` so a
      half-baked XML still produces visible growth instead of a
      silent freeze (Lock #6 minor finding 10).

T2/T3/T4 mutate the in-tree maize_calibrated.xml LRP for subType=2
in-memory after readParameters; we never write XML or step the
production calibration.  This keeps the tests light (~5 simulated
days) while exercising the full Leaf::simulate dispatch path.
"""
import math
import sys
from pathlib import Path

import pytest

import plantbox as pb


SRC_LEAF_CPP = Path(__file__).resolve().parents[3] / "src" / "structural" / "Leaf.cpp"
MAIZE_XML = Path(__file__).resolve().parents[1] / "data" / "maize_calibrated.xml"


# -------------------------------------------------------------------------
# T1 — source-level guard: the shadow consumer must be gone.
# -------------------------------------------------------------------------
def test_t1_logistic_shadow_consumer_retired():
    """Leaf.cpp must not contain a ``use_fa_kinetics`` *consumer* (an
    expression that branches the dispatch).  References inside
    comments explaining the historical retirement are allowed; the
    grep below catches any code line that reads the field, e.g.
    ``lrp.use_fa_kinetics``, ``.use_fa_kinetics``, or
    ``->use_fa_kinetics``.

    Failure mode this guards: a future cherry-pick or merge
    accidentally re-introducing the logistic-shadow ``if`` branch
    (`if (lrp_fa.use_fa_kinetics ...)`).
    """
    src = SRC_LEAF_CPP.read_text()
    offending = []
    for ln, line in enumerate(src.splitlines(), 1):
        before_comment = line.split("//", 1)[0]
        if "use_fa_kinetics" in before_comment:
            offending.append((ln, line.rstrip()))
    assert not offending, (
        "Leaf.cpp contains an active use_fa_kinetics consumer (the FA "
        "logistic shadow). S2.C deleted this shadow; if it has come "
        "back, restore the f_gf->getLength dispatch:\n"
        + "\n".join(f"  L{ln}: {ln_text}" for ln, ln_text in offending)
    )


# -------------------------------------------------------------------------
# Helpers for T2/T3/T4 — load maize_calibrated.xml, then mutate the
# subType=2 LRP in-memory before initialize().
# -------------------------------------------------------------------------
def _load_maize_with_leaf_overrides(
    *,
    leaf_gf: int,
    R1: float = 0.0,
    R2: float = 0.0,
    lag_exp: float = 0.0,
    D_lin: float = 0.0,
    T0: float = 0.0,
    L_min: float = 0.025,
    leaf_lmax: float = 9999.0,
    leaf_r: float = 2.0,
):
    """Read maize_calibrated.xml, override ``leaf subType=2`` (the first
    blade leaf) so it is born within ~5 days and follows the controlled
    test parameters, return the configured plant.
    """
    plant = pb.MappedPlant(7)
    plant.readParameters(str(MAIZE_XML))
    plant.setSeed(7)
    lrp = plant.getOrganRandomParameter(pb.OrganTypes.leaf, 2)
    lrp.gf = leaf_gf
    # Strip both legacy birth gates so the leaf is born immediately on
    # the first simulate step.  ``ldelay`` is also defaulted away from
    # any TT-axis pin the bake script may have stamped.
    lrp.ldelay = 0.0
    lrp.use_thermal_emergence = 0
    lrp.tt_emergence = -1.0
    # Andrieu primitives — caller controls.
    lrp.R1_n = R1
    lrp.R2_n = R2
    lrp.lag_exp_n = lag_exp
    lrp.D_lin_n = D_lin
    lrp.T0_n = T0
    lrp.L_min = L_min
    # Generous lmax + zero `lb` so the leaf has no basal/branching zone
    # to traverse before it can elongate.  Keep ln/la at their XML
    # defaults (LeafRandomParameter.ln is a scalar mean, not a vector).
    lrp.lmax = leaf_lmax
    lrp.r = leaf_r
    lrp.lb = 0.0
    lrp.la = leaf_lmax
    plant.initialize(False)
    return plant


def _grow_and_probe_subtype_2(plant, sim_days: float, T_air: float = 25.0):
    plant.setAirTemperature(T_air)
    total = 0.0
    while total < sim_days:
        step = min(1.0, sim_days - total)
        plant.simulate(step, False)
        total += step
    leaves = [
        lf for lf in plant.getOrgans(pb.OrganTypes.leaf)
        if int(lf.getParameter("subType")) == 2
    ]
    assert leaves, "maize subType=2 leaf was not spawned within sim horizon"
    return float(leaves[0].getLength(False)), float(plant.getAccumulatedAndrieuTT())


# -------------------------------------------------------------------------
# T2 — gf=6 dispatch produces Andrieu Phase-E exponential growth.
# -------------------------------------------------------------------------
def test_t2_gf6_andrieu_phase_e_curve():
    """With gf=6 + Andrieu primitives configured, a leaf grown past T0
    follows ``L = L_min * exp(R1 * (TT - T0))`` inside Phase E.

    The leaf-side dispatch in Leaf::simulate is now ``calcLength(age_+dt_)
    + epsilonDx``; under gf=6 ``calcLength`` is forwarded to
    ``MultiPhaseLeafGrowth::getLength`` which ignores the calendar age
    and reads ``plant->getAccumulatedAndrieuTT()`` directly (Lock #5).
    The realised leaf length therefore tracks the Andrieu curve at the
    plant's TT, not the calendar age.
    """
    R1 = 0.041
    lag = 200.0  # large lag so we stay inside Phase E for the whole horizon
    D_lin = 120.0
    T0 = 30.0    # tiny T0 so we cross into Phase E within ~3 days at T_air=25
    L_min = 0.025
    plant = _load_maize_with_leaf_overrides(
        leaf_gf=6, R1=R1, R2=0.78, lag_exp=lag, D_lin=D_lin, T0=T0, L_min=L_min,
    )
    sim_days = 7.0
    L, tt = _grow_and_probe_subtype_2(plant, sim_days, T_air=25.0)
    # Sanity: TT must have crossed T0 (Tb=9.8 with no T_opt cap →
    # f_T = 25 - 9.8 ≈ 15.2 °Cd/day → 7 days ≈ 106 °Cd >> T0=30).
    assert tt > T0 + 30.0, (
        f"Plant Andrieu TT={tt:.1f} did not reach Phase E (T0={T0})."
    )
    # Confirm the test horizon is still inside Phase E (lag chosen
    # so this is comfortably true).
    assert tt < T0 + lag, (
        f"Test horizon escaped Phase E (tt={tt:.1f} ≥ T0+lag={T0 + lag})."
    )
    L_expected = L_min * math.exp(R1 * (tt - T0))
    # Realised length lags target by at most one daily step; the
    # `e = target - length; dl = max(e, 0)` clamp absorbs the
    # discretisation. Allow ~50% rel tolerance to give the leaf its
    # bring-up days at L_min before it tracks the curve.
    assert L == pytest.approx(L_expected, rel=0.5), (
        f"gf=6 dispatch did not produce Andrieu Phase-E curve: "
        f"got L={L:.4f}, expected ~{L_expected:.4f} "
        f"(tt={tt:.1f}, T0={T0}, R1={R1})."
    )


# -------------------------------------------------------------------------
# T3 — gf=1 (default) keeps ExponentialGrowth scalar dispatch.
# -------------------------------------------------------------------------
def test_t3_gf1_default_scalar_dispatch():
    """Same fixture, gf=1 (ExponentialGrowth).  The leaf must follow
    ``L = lmax * (1 - exp(-r/lmax * t))`` rather than the Andrieu
    curve — proves S2.C did not silently change the default leaf
    growth function (the dead-code gate from S2.A still holds: gf=6
    must be opted in).
    """
    plant = _load_maize_with_leaf_overrides(leaf_gf=1, leaf_lmax=9999.0, leaf_r=2.0)
    sim_days = 5.0
    L, _tt = _grow_and_probe_subtype_2(plant, sim_days, T_air=25.0)
    # ExponentialGrowth: leaf with r=2, lmax=9999 grows ~r*dt for
    # small t (saturation kicks in at t ~ lmax/r ~ 5000 days).
    # After 5 days expect L ~ 10 cm; allow generous tolerance.
    assert 5.0 < L < 15.0, (
        f"gf=1 default dispatch did not produce r*dt linear curve: L={L:.2f}"
    )


# -------------------------------------------------------------------------
# T4 — gf=6 with R1_n=0 falls back to ExponentialGrowth.
# -------------------------------------------------------------------------
def test_t4_gf6_unconfigured_falls_back_to_exponential():
    """Lock #6 minor finding 10: when MultiPhaseLeafGrowth is minted
    via gf=6 but the Andrieu primitives are unset (R1_n stays at the
    sentinel 0.0), the GF must fall back to the ExponentialGrowth
    formula instead of returning zero — a half-baked XML should still
    produce visible growth.
    """
    plant = _load_maize_with_leaf_overrides(leaf_gf=6, leaf_lmax=9999.0, leaf_r=2.0)
    # Confirm the LRP carries gf=6 + R1_n=0 — i.e. the test fixture
    # is genuinely "minted MPL but unconfigured", which the fallback
    # path is meant to cover.
    lrp = plant.getOrganRandomParameter(pb.OrganTypes.leaf, 2)
    assert isinstance(lrp.f_gf, pb.MultiPhaseLeafGrowth)
    assert lrp.R1_n == 0.0
    sim_days = 5.0
    L, _tt = _grow_and_probe_subtype_2(plant, sim_days, T_air=25.0)
    assert L > 1.0, (
        f"Unconfigured gf=6 leaf produced near-zero length L={L:.4f}; "
        "the empty-array fallback should yield ExponentialGrowth growth."
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
