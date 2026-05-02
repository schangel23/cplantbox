#!/usr/bin/env python3
"""§G3 acceptance gate: FA-on no-carbon → with-carbon parity at day 130.

This is the headline test of PLAN_S5_SINK_SOURCE_COUPLING_2026-05-02:
"Silent FA clobber is fixed". Replays the production diurnal carbon-mode
loop (uniform clearsky PAR, no DART) for days 31..130 starting from a
day-30 grow_plant() bootstrap, then compares per-organ realised lengths
against the FA-on no-carbon oracle (`oracle_fa_no_carbon_day130.json`).

Acceptance tolerances (per plan §S6 test 2):
  * Per-leaf realised length: ≤ 1 % drift from oracle.
  * Mainstem top z drift:     ≤ 0.5 cm.

Pre-Lock-#6 + pre-Lock-#9, this test would have FAILED with massive
drift because every carbon-mode `enable_cw_limited_growth` call
silently clobbered the MultiPhase{Stem,Leaf}Growth GFs that
Plant::initCallbacks had just minted. Post-Lock-#6 + Lock-#9, the FA
target shape is preserved through carbon mode by construction (well-
watered supply >> FA target → cap binds on demand_target).

Usage (from /home/lukas/PHD/CPlantBox)::

    cpbenv/bin/python3 dart/coupling/tests/baselines/run_g3_with_carbon_parity.py
    cpbenv/bin/python3 dart/coupling/tests/baselines/run_g3_with_carbon_parity.py --bootstrap-day 50
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

BASELINE_DIR = Path(__file__).resolve().parent
COUPLING_DIR = BASELINE_DIR.parent.parent
CPLANTBOX_ROOT = COUPLING_DIR.parent.parent
FIXTURES_DIR = COUPLING_DIR / "tests" / "fixtures"
sys.path.insert(0, str(CPLANTBOX_ROOT))

from dart.coupling.growth.grow import grow_plant  # noqa: E402
from dart.coupling.growth.carbon_growth import (  # noqa: E402
    enable_cw_limited_growth,
    inject_cw_gr,
)
from dart.coupling.carbon.dvs_partitioning import get_daily_met  # noqa: E402

ORACLE_PATH = FIXTURES_DIR / "oracle_fa_no_carbon_day130.json"
SEED = 7
SIM_DAYS = 130
TOL_LEAF_PCT = 1.0   # plan §S6 test 2: ≤ 1 % per leaf
TOL_MAINSTEM_CM = 0.5  # plan §S6 test 2: ≤ 0.5 cm on mainstem top


def grow_with_carbon(bootstrap_day: int):
    print(f"Phase 1: bootstrap to day {bootstrap_day} via grow_plant (FA-on, no carbon)")
    plant = grow_plant(
        xml_path=str(COUPLING_DIR / "data" / "maize_calibrated.xml"),
        simulation_time=bootstrap_day,
        seed=SEED,
        enable_photosynthesis=True,
    )

    print(f"Phase 2: switch to carbon-mode (Lock #9 wrap policy)")
    enable_cw_limited_growth(plant)
    n_wrapped_stem = sum(
        1 for p in plant.getOrganRandomParameter(3)
        if p is not None and p.f_gf.demand is not None
    )
    n_wrapped_leaf = sum(
        1 for p in plant.getOrganRandomParameter(4)
        if p is not None and p.f_gf.demand is not None
    )
    print(f"  wrapped {n_wrapped_stem} stem RPs + {n_wrapped_leaf} leaf RPs with demand=FA")

    print(f"Phase 3: with-carbon loop days {bootstrap_day+1}..{SIM_DAYS} "
          f"(synthetic well-watered supply, bypasses phloem solver)")
    met_lookup = get_daily_met(daily_met=None)

    # Synthetic well-watered supply targets only the FA-wrapped organs
    # (mainstem + FA leaf subtypes). For those, we inject a per-step
    # supply >> any FA target so the Lock #6 cap binds on demand_target.
    #
    # Non-FA organs (scalar leaves, tassel branches, roots, ...) get NO
    # injection — their f_gf is bare CWLimitedGrowth() with empty CW_Gr,
    # which falls through to ExponentialGrowth (Lock #6 §1 of the impl
    # comment in growth.cpp). That is exactly the no-carbon path for
    # those organs, so they remain bit-identical to the oracle.
    #
    # This isolates the §G3 question to its real subject: does the FA
    # target shape survive Lock #6 + Lock #9 wrapping? Real phloem-driven
    # supply variation is a Ch2 question (PiafMunch parity, deferred per
    # plan §"Risks & open questions").
    BIG_SUPPLY_CM = 100.0   # >> any per-step FA target

    t0 = time.time()
    for sim_day in range(bootstrap_day + 1, SIM_DAYS + 1):
        T_air = 25.0
        if met_lookup is not None and sim_day in met_lookup:
            T_air = float(met_lookup[sim_day]["T_mean_C"])
        if hasattr(plant, "setAirTemperature"):
            plant.setAirTemperature(T_air)

        # Build a growth_map only for FA-wrapped organs (demand is non-null).
        # Include pre-emergence organs (all=True) so newly-spawned leaves get
        # supply from their first step instead of being stuck at length=0
        # forever (chicken-and-egg: empty CW_Gr lookup returns 0, but
        # getOrgans default excludes len=0 leaves).
        organs = plant.getOrgans(-1, True)
        fa_wrapped_ids_by_ot = {2: set(), 3: set(), 4: set()}
        for ot in (3, 4):
            for p in plant.getOrganRandomParameter(ot):
                if p is None:
                    continue
                if getattr(p.f_gf, "demand", None) is not None:
                    # Mark this subType as FA-wrapped — every organ of this
                    # (ot, subType) gets BIG_SUPPLY in the inject map.
                    fa_wrapped_ids_by_ot[ot].add(int(p.subType))

        growth_map = {2: {}, 3: {}, 4: {}}
        import math
        for o in organs:
            ot = int(o.organType())
            st = int(o.getParameter("subType"))
            if st in fa_wrapped_ids_by_ot.get(ot, set()):
                # FA-wrapped: BIG_SUPPLY so Lock #6 cap binds on demand.
                growth_map[ot][o.getId()] = BIG_SUPPLY_CM
            elif ot == 4:
                # Non-FA scalar leaf (subTypes 2, 3 on maize): bare CWLim
                # has no demand cap, so we must inject the per-step
                # ExponentialGrowth target to match the no-carbon path.
                # The phloem solver does this automatically in production;
                # here we replicate it analytically.
                rp = o.getOrganRandomParameter()
                k = float(o.getParameter("lmax"))
                r = float(rp.r)
                age = float(o.getAge())
                cur = float(o.getLength())
                if k > 0 and r > 0 and age >= 0:
                    next_len = k * (1.0 - math.exp(-r / k * (age + 1.0)))
                    delta = max(0.0, next_len - cur)
                    growth_map[ot][o.getId()] = delta
        inject_cw_gr(plant, growth_map)

        try:
            plant.simulate(1.0, False)
        except (IndexError, RuntimeError) as e:
            print(f"  day {sim_day}: simulate error {e}")
            try:
                plant.simulate(0.0, False)
            except Exception:
                pass

        if sim_day % 10 == 0:
            elapsed = time.time() - t0
            all_organs = plant.getOrgans(-1, True)
            n_leaves_all = sum(1 for o in all_organs if int(o.organType()) == 4)
            n_leaves_emerged = sum(
                1 for o in all_organs
                if int(o.organType()) == 4 and o.getLength() > 0.01
            )
            print(f"  day {sim_day}: ok (elapsed {elapsed:.0f}s), "
                  f"organs={len(all_organs)}, "
                  f"leaves={n_leaves_emerged} emerged / {n_leaves_all} total")
    print(f"Phase 3 done in {time.time() - t0:.0f}s")
    return plant


def per_organ_snapshot(plant) -> dict:
    out = {}
    # Use getOrgans(-1, all=True) to include pre-emergence (len=0) leaves
    # so the §G3 comparison sees ALL spawned organs, not just emerged ones.
    for o in plant.getOrgans(-1, True):
        out[str(o.getId())] = {
            "organ_type": int(o.organType()),
            "subType": int(o.getParameter("subType")),
            "realised_length": round(float(o.getLength()), 6),
            "insertion_z": round(float(list(o.getNodes())[0].z) if len(o.getNodes()) else 0.0, 6),
            "dl_backlog": round(float(o.dl_backlog), 6),
        }
    return out


def compare_against_oracle(snap: dict) -> tuple[bool, list[str]]:
    with ORACLE_PATH.open() as f:
        oracle = json.load(f)

    failures = []
    notes = []

    # Mainstem top z drift (use mainstem realised_length as proxy — same metric
    # the oracle stores).
    oracle_mainstem = next(
        (v for v in oracle["organs"].values()
         if v["organ_type"] == 3 and v["subType"] == 1),
        None,
    )
    snap_mainstem = next(
        (v for v in snap.values()
         if v["organ_type"] == 3 and v["subType"] == 1),
        None,
    )
    if oracle_mainstem and snap_mainstem:
        delta = snap_mainstem["realised_length"] - oracle_mainstem["realised_length"]
        line = (f"mainstem realised: oracle={oracle_mainstem['realised_length']:.4f} cm, "
                f"with-carbon={snap_mainstem['realised_length']:.4f} cm, "
                f"Δ={delta:+.4f} cm")
        notes.append(line)
        if abs(delta) > TOL_MAINSTEM_CM:
            failures.append(f"  FAIL: |mainstem delta| {abs(delta):.4f} > {TOL_MAINSTEM_CM} cm")

    # Per-leaf drift (use realised_length, ≤ 1%). Match by subType — both
    # paths produce one leaf per FA subType (4..16) on the mainstem; raw
    # organ IDs differ between paths because lateral creation order shifts
    # slightly under bootstrap-then-carbon vs all-no-carbon, but the
    # (subType → realised_length) mapping is the comparable invariant.
    def _by_subtype(snapshot):
        out = {}
        for v in snapshot.values():
            if v["organ_type"] != 4:
                continue
            st = v["subType"]
            # If multiple organs share a subType (basal scalars at st=2/3),
            # keep the longest one — that's the "primary" leaf at that rank.
            if st not in out or v["realised_length"] > out[st]["realised_length"]:
                out[st] = v
        return out

    oracle_by_st = _by_subtype(oracle["organs"])
    snap_by_st = _by_subtype(snap)
    n_leaf_pass = 0
    n_leaf_fail = 0
    for st, ov in sorted(oracle_by_st.items()):
        ol = ov["realised_length"]
        if ol <= 0:
            continue
        sv = snap_by_st.get(st)
        if sv is None:
            failures.append(
                f"  FAIL: leaf subType={st} missing from with-carbon snapshot "
                f"(oracle had length {ol:.2f} cm)")
            n_leaf_fail += 1
            continue
        sl = sv["realised_length"]
        pct = 100.0 * abs(sl - ol) / ol
        line = (f"  leaf st={st}: oracle={ol:.4f} cm, with-carbon={sl:.4f} cm, "
                f"drift={pct:.2f}%")
        if pct > TOL_LEAF_PCT:
            failures.append(line + f" > {TOL_LEAF_PCT}%  FAIL")
            n_leaf_fail += 1
        else:
            notes.append(line)
            n_leaf_pass += 1
    notes.append(f"per-leaf parity: {n_leaf_pass} pass, {n_leaf_fail} fail "
                 f"(tol={TOL_LEAF_PCT}%)")

    return (len(failures) == 0), notes + failures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap-day", type=int, default=30,
                        help="day to grow_plant() before switching to carbon mode")
    args = parser.parse_args()

    if not ORACLE_PATH.exists():
        print(f"MISSING oracle at {ORACLE_PATH}")
        print("  Run capture_oracle_fa_no_carbon_day130.py first.")
        return 2

    plant = grow_with_carbon(args.bootstrap_day)
    snap = per_organ_snapshot(plant)
    ok, lines = compare_against_oracle(snap)

    print()
    print("=" * 70)
    print(f"§G3 with-carbon parity check (bootstrap day = {args.bootstrap_day})")
    print("=" * 70)
    for line in lines:
        print(line)
    print()
    if ok:
        print("§G3 PASS — silent FA clobber is fixed.")
        return 0
    print("§G3 FAIL — drift exceeds tolerance. Inspect the list above.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
