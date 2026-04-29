#!/usr/bin/env python3
"""S2.D — bake script + Andrieu leaf primitives + gf=6 opt-in.

ADR_LEAF_KINEMATICS_2026-04-28 §S2 sub-step "extend the bake script"
+ §D6 (calibration values live in XML, not Python).  S2.A shipped
``MultiPhaseLeafGrowth`` as dead code; S2.C wired the runtime dispatch
through ``f_gf->getLength``; this commit (S2.D) bakes the per-rank
Andrieu/Hillier/Birch primitives + ``gf=6`` opt-in into
``maize_calibrated.xml`` so the runtime contract is pure XML + C++
end-to-end.

Tests:

  T1  Round-trip: bake → re-load baked XML → diff every leaf-side
      Andrieu field per subType.  Symmetric to S0.7 T1 for the stem
      side.  Must be exact-equal.

  T2  ``maize_calibrated.xml`` (the in-tree calibrated XML, not a
      tmp copy) carries the full S2.D state — at least one leaf
      subType has ``gf=6`` and matches ``phase_III_per_rank_LEAF.json``
      to within float round-trip precision.  This is the "ship gate":
      a fresh checkout produces the same calibrated plant after
      ``readParameters`` + ``initialize`` + ``simulate``.

  T3  Hard gate (D6 closure): grow 25 days starting from the baked
      XML through pure C++ (no Python helpers between
      ``readParameters`` and ``simulate``); confirm the leaves
      labelled ``gf=6`` actually dispatch through
      ``MultiPhaseLeafGrowth`` at runtime AND grow non-zero length.

  T4  Gated juvenile ranks (1-3) stay at ``gf=1`` per ADR §C4 — the
      bake must not silently opt them in.  Their R1_n/R2_n stay 0
      so any accidental gf=6 mint at runtime falls through to
      ExponentialGrowth via the Lock #6 empty-array guard.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

import plantbox as pb

from dart.coupling.config import DEFAULT_XML
from dart.coupling.scripts.bake_calibration_to_xml import (
    LEAF_KINETICS_JSON,
    bake,
    enable_andrieu_on_leaves,
)


HARD_GATE_DAYS = 25
HARD_GATE_SEED = 42


def _leaf_andrieu_snapshot(plant) -> dict:
    """Snapshot per-leaf-LRP Andrieu fields the S2.D bake touches."""
    out = {}
    for lrp in plant.getOrganRandomParameter(pb.OrganTypes.leaf):
        out[int(lrp.subType)] = {
            "gf": int(getattr(lrp, "gf")),
            "R1_n": float(getattr(lrp, "R1_n")),
            "R2_n": float(getattr(lrp, "R2_n")),
            "lag_exp_n": float(getattr(lrp, "lag_exp_n")),
            "D_lin_n": float(getattr(lrp, "D_lin_n")),
            "T0_n": float(getattr(lrp, "T0_n")),
            "L_min": float(getattr(lrp, "L_min")),
        }
    return out


@pytest.fixture
def baked_pair(tmp_path):
    """Bake a copy of the master XML and return both sides plus the
    in-memory post-bake snapshot to diff against the on-disk reload."""
    master_xml = tmp_path / "maize_calibrated.master.xml"
    baked_xml = tmp_path / "maize_calibrated.baked.xml"
    shutil.copy(DEFAULT_XML, master_xml)

    bake(master_xml, baked_xml, verbose=False)

    # Re-do the bake side in memory (importing the helper rather than
    # caching the bake's plant — keeps the test honest about what the
    # bake actually does end-to-end).
    from dart.coupling.growth.grow import (
        enable_fa_on_leaves,
        enable_fa_on_mainstem,
        setup_successor_where,
    )
    plant_pre = pb.MappedPlant()
    plant_pre.readParameters(str(master_xml))
    setup_successor_where(plant_pre)
    enable_fa_on_mainstem(plant_pre, verbose=False)
    enable_fa_on_leaves(plant_pre, verbose=False)
    enable_andrieu_on_leaves(plant_pre, verbose=False)
    pre_snapshot = _leaf_andrieu_snapshot(plant_pre)

    return master_xml, baked_xml, pre_snapshot


def test_T1_andrieu_round_trip_diff_is_empty(baked_pair):
    """T1: bake → load baked → diff every leaf Andrieu field. Empty."""
    _, baked_xml, pre_snapshot = baked_pair

    plant_post = pb.MappedPlant()
    plant_post.readParameters(str(baked_xml))
    post_snapshot = _leaf_andrieu_snapshot(plant_post)

    assert set(pre_snapshot) == set(post_snapshot), (
        f"Leaf subType set diverged: "
        f"pre={sorted(pre_snapshot)} post={sorted(post_snapshot)}"
    )
    for sub, want in pre_snapshot.items():
        got = post_snapshot[sub]
        assert got == want, (
            f"Leaf subType={sub} Andrieu fields did not round-trip: "
            f"pre={want!r} post={got!r}"
        )


def test_T2_shipped_xml_carries_andrieu_calibration():
    """The shipped ``maize_calibrated.xml`` must carry S2.D state — i.e.
    the bake has been run on it after S2.D landed.  Reading the XML
    cold (no helpers) must surface gf=6 + populated Andrieu primitives
    on at least the rank-11 ear leaf and rank-15 flag leaf, which are
    canonical fitted ranks (Andrieu 2006 §4.2).
    """
    plant = pb.MappedPlant()
    plant.readParameters(str(DEFAULT_XML))
    with LEAF_KINETICS_JSON.open() as f:
        kin = json.load(f)
    rank_by_n = {int(row["n"]): row for row in kin["ranks"]}
    subtype_to_rank = {
        int(k): int(v) for k, v in
        kin["_meta"]["leaf_subtype_to_rank_map"].items()
    }

    # Ranks 11 and 15 are fitted (gated=False) with non-zero published R1.
    canary_subtypes = [
        sub for sub, rank in subtype_to_rank.items()
        if rank in (11, 15) and not bool(rank_by_n[rank]["gated"])
    ]
    assert canary_subtypes, "no canary subTypes for ranks 11/15 — fixture bad"

    for sub in canary_subtypes:
        lrp = plant.getOrganRandomParameter(pb.OrganTypes.leaf, sub)
        rank = subtype_to_rank[sub]
        row = rank_by_n[rank]
        assert int(lrp.gf) == 6, (
            f"shipped XML leaf subType={sub} gf={int(lrp.gf)}, expected 6 — "
            "rerun `python -m dart.coupling.scripts.bake_calibration_to_xml` "
            "and commit the resulting maize_calibrated.xml."
        )
        assert lrp.R1_n == pytest.approx(float(row["R1_Cd_inv"]), rel=1e-12)
        assert lrp.R2_n == pytest.approx(
            float(row["R2_cm_per_Cd_rescaled"]), rel=1e-12,
        )
        assert lrp.lag_exp_n == pytest.approx(
            float(row["lag_exp_Cd_published"]), rel=1e-12,
        )
        assert lrp.D_lin_n == pytest.approx(float(row["D_lin_Cd"]), rel=1e-12)
        assert lrp.T0_n == pytest.approx(float(row["T0_Cd"]), rel=1e-12)


def test_T3_pure_xml_grow_dispatches_through_andrieu():
    """T3 (D6 hard gate, leaf side): a 25-day grow starting from the
    shipped baked XML through pure C++ (no Python helpers between
    readParameters and simulate) produces leaves whose ``f_gf`` is
    ``MultiPhaseLeafGrowth`` for the gf=6 subTypes AND grows them to
    nonzero length.

    This is the leaf-side analogue of S0.7 T2: it operationalises D6
    for the kinetic dispatch path that S2.A added and S2.C wired up.
    """
    plant = pb.MappedPlant(HARD_GATE_SEED)
    plant.readParameters(str(DEFAULT_XML))
    plant.setSeed(HARD_GATE_SEED)
    plant.initialize()
    for _ in range(HARD_GATE_DAYS):
        plant.simulate(1.0, verbose=False)

    # At least the rank-4..6 lower leaves should have emerged in 25 days
    # under default met (constant T_air → ~15.2 °Cd/day Andrieu TT →
    # ~380 °Cd in 25 days; T0 for rank 4 is 57.6, so rank-4 leaves are
    # well past Phase E by day 25).
    leaves = list(plant.getOrgans(pb.OrganTypes.leaf))
    assert leaves, "no leaves spawned in 25-day pure-XML grow"

    # The dispatch invariant: every leaf whose LRP has gf=6 must be
    # backed by MultiPhaseLeafGrowth at runtime, AND if it has been
    # alive for any length of time it must carry positive length.
    seen_gf6_with_length = 0
    for lf in leaves:
        sub = int(lf.getParameter("subType"))
        lrp = plant.getOrganRandomParameter(pb.OrganTypes.leaf, sub)
        if int(lrp.gf) != 6:
            continue
        assert isinstance(lrp.f_gf, pb.MultiPhaseLeafGrowth), (
            f"leaf subType={sub} gf=6 but f_gf is {type(lrp.f_gf).__name__}"
        )
        if float(lf.getLength(False)) > 0.025:  # past L_min
            seen_gf6_with_length += 1
    assert seen_gf6_with_length > 0, (
        "no gf=6 leaves grew past L_min in 25 days — Andrieu kinetics "
        "are minted but the dispatch path is not delivering growth."
    )


def test_T4_gated_juveniles_stay_on_exponential_growth():
    """T4: ranks 1-3 are gated juvenile (Andrieu 2006 §3.A no-fit).  The
    bake must not silently opt them into gf=6 — they stay at the
    legacy default ExponentialGrowth.  Their R1_n / R2_n fields must
    be zero so any accidental gf=6 mint downstream falls through to
    ExponentialGrowth via the Lock #6 empty-array guard.
    """
    plant = pb.MappedPlant()
    plant.readParameters(str(DEFAULT_XML))
    with LEAF_KINETICS_JSON.open() as f:
        kin = json.load(f)
    rank_by_n = {int(row["n"]): row for row in kin["ranks"]}
    subtype_to_rank = {
        int(k): int(v) for k, v in
        kin["_meta"]["leaf_subtype_to_rank_map"].items()
    }

    gated_subtypes = [
        sub for sub, rank in subtype_to_rank.items()
        if bool(rank_by_n[rank]["gated"])
    ]
    assert gated_subtypes, (
        "no gated subTypes — JSON metadata regression "
        "(ranks 1-3 should be gated)."
    )

    for sub in gated_subtypes:
        lrp = plant.getOrganRandomParameter(pb.OrganTypes.leaf, sub)
        assert int(lrp.gf) == 1, (
            f"gated subType={sub} unexpectedly opted into gf={int(lrp.gf)}; "
            "ranks 1-3 must stay on ExponentialGrowth."
        )
        assert lrp.R1_n == 0.0, (
            f"gated subType={sub} carries R1_n={lrp.R1_n}, must be 0 "
            "so the Lock #6 empty-array fallback covers any accidental mint."
        )
        assert lrp.R2_n == 0.0
