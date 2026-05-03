#!/usr/bin/env python3
"""P1 oracle capture: well-watered with-carbon, per-rank stem state, day 130.

PLAN_PER_RANK_CARBON_FA_2026-05-03 §S1 fixture.

Drives the §G3 well-watered carbon path (bootstrap day 1..30 via grow_plant,
then Lock #9 wrap + synthetic BIG_SUPPLY for days 31..130) and captures
per-rank ``length_per_n`` + per-rank ``cessation_age_per_n`` for every
FA-on stem. This is the bit-identical regression target for P1: when the
new per-rank cap dispatch (S3..S5) runs with full supply, it must
reproduce these per-rank lengths exactly.

Schema (per plan §S1)
---------------------
{
  "meta": {...},
  "plants": {
    "<seed>": {
      "<organ_id>": {
        "organ_type": int,
        "subType":    int,
        "realised_length": float,
        "length_per_n":          [float, ...],   # populated only on FA-on stems
        "cessation_age_per_n":   [float, ...],
        "cessation_age_global":  float,
        "node_to_phytomer":      [int, ...]
      }
    }
  }
}

Usage (from /home/lukas/PHD/CPlantBox)
--------------------------------------
    cpbenv/bin/python3 dart/coupling/tests/baselines/capture_p1_per_rank_oracle.py
    cpbenv/bin/python3 dart/coupling/tests/baselines/capture_p1_per_rank_oracle.py --verify
    cpbenv/bin/python3 dart/coupling/tests/baselines/capture_p1_per_rank_oracle.py --bootstrap-day 50
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
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


XML_PATH = COUPLING_DIR / "data" / "maize_calibrated.xml"
OUT_JSON = FIXTURES_DIR / "oracle_p1_per_rank_well_watered_day130.json"

SEED = 7
SIM_DAYS = 130
BOOTSTRAP_DAY_DEFAULT = 30
BIG_SUPPLY_CM = 100.0  # mirrors run_g3_with_carbon_parity.py


def grow_with_carbon(bootstrap_day: int):
    """Bootstrap to ``bootstrap_day`` via grow_plant, then run carbon mode
    with synthetic well-watered supply through day ``SIM_DAYS``.
    """
    print(f"P1 oracle capture: bootstrap to day {bootstrap_day} (FA-on, no carbon)")
    plant = grow_plant(
        xml_path=str(XML_PATH),
        simulation_time=bootstrap_day,
        seed=SEED,
        enable_photosynthesis=True,
    )

    print("  switching to carbon mode (Lock #9 wrap)")
    enable_cw_limited_growth(plant)

    met_lookup = get_daily_met(daily_met=None)

    t0 = time.time()
    for sim_day in range(bootstrap_day + 1, SIM_DAYS + 1):
        T_air = 25.0
        if met_lookup is not None and sim_day in met_lookup:
            T_air = float(met_lookup[sim_day]["T_mean_C"])
        if hasattr(plant, "setAirTemperature"):
            plant.setAirTemperature(T_air)

        organs = plant.getOrgans(-1, True)
        fa_wrapped_subtypes = {2: set(), 3: set(), 4: set()}
        for ot in (3, 4):
            for p in plant.getOrganRandomParameter(ot):
                if p is None:
                    continue
                if getattr(p.f_gf, "demand", None) is not None:
                    fa_wrapped_subtypes[ot].add(int(p.subType))

        growth_map = {2: {}, 3: {}, 4: {}}
        for o in organs:
            ot = int(o.organType())
            st = int(o.getParameter("subType"))
            if st in fa_wrapped_subtypes.get(ot, set()):
                growth_map[ot][o.getId()] = BIG_SUPPLY_CM
            elif ot == 4:
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

        if sim_day % 20 == 0:
            elapsed = time.time() - t0
            all_organs = plant.getOrgans(-1, True)
            print(f"  day {sim_day}: ok (elapsed {elapsed:.0f}s), "
                  f"organs={len(all_organs)}")

    print(f"  done in {time.time() - t0:.0f}s")
    return plant


def capture_plant_snapshot(plant) -> dict:
    """One-plant snapshot, keyed by organ id. Per-rank arrays only on stems."""
    out: dict[str, dict] = {}
    for o in sorted(plant.getOrgans(-1, True), key=lambda x: x.getId()):
        ot = int(o.organType())
        st = int(o.getParameter("subType"))
        oid = str(o.getId())
        record: dict = {
            "organ_type": ot,
            "subType": st,
            "realised_length": round(float(o.getLength()), 6),
        }
        if ot == 3:  # stem
            rp = o.getOrganRandomParameter()
            f_gf = getattr(rp, "f_gf", None)
            demand = getattr(f_gf, "demand", None)
            state_map = getattr(demand, "per_organ_state", None)
            sid = o.getId()
            if state_map is not None and sid in state_map:
                state = state_map[sid]
                record["length_per_n"] = [
                    round(float(x), 6) for x in state.length_per_n
                ]
                record["cessation_age_per_n"] = [
                    round(float(x), 6) for x in state.cessation_age_per_n
                ]
            else:
                record["length_per_n"] = []
                record["cessation_age_per_n"] = []
            cessation_global = getattr(o, "cessation_age_", -1.0)
            record["cessation_age_global"] = round(float(cessation_global), 6)
            ntp = list(getattr(o, "node_to_phytomer", []) or [])
            record["node_to_phytomer"] = [int(x) for x in ntp]
        out[oid] = record
    return out


def fingerprint(plant_records: dict) -> str:
    """Stable sha256 over the per-rank arrays so verify mode can diff cleanly."""
    h = hashlib.sha256()
    for seed_key in sorted(plant_records.keys()):
        organs = plant_records[seed_key]
        for oid in sorted(organs.keys(), key=int):
            v = organs[oid]
            h.update(struct.pack("<iid", v["organ_type"], v["subType"],
                                  v["realised_length"]))
            for x in v.get("length_per_n", []):
                h.update(struct.pack("<d", x))
            for x in v.get("cessation_age_per_n", []):
                h.update(struct.pack("<d", x))
    return h.hexdigest()


def capture_mode(bootstrap_day: int) -> None:
    plant = grow_with_carbon(bootstrap_day)
    snapshot = capture_plant_snapshot(plant)
    plants = {str(SEED): snapshot}
    sha = fingerprint(plants)

    n_stems = sum(1 for v in snapshot.values() if v["organ_type"] == 3)
    n_leaves = sum(1 for v in snapshot.values() if v["organ_type"] == 4)
    fa_stems = [
        v for v in snapshot.values()
        if v["organ_type"] == 3 and v["length_per_n"]
    ]
    print(f"  organs={len(snapshot)} (stems={n_stems}, leaves={n_leaves}); "
          f"FA-on stems with per-rank state = {len(fa_stems)}")
    print(f"  fingerprint sha256: {sha}")

    payload = {
        "meta": {
            "slug": "oracle_p1_per_rank_well_watered_day130",
            "xml": str(XML_PATH.relative_to(CPLANTBOX_ROOT)),
            "seeds": [SEED],
            "sim_days": SIM_DAYS,
            "bootstrap_day": bootstrap_day,
            "carbon_feedback": True,
            "wrap_policy": "Lock #9 (S5)",
            "supply_mode": "synthetic_big_supply",
            "big_supply_cm": BIG_SUPPLY_CM,
            "n_organs": len(snapshot),
            "fingerprint_sha256": sha,
        },
        "plants": plants,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with OUT_JSON.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"  -> {OUT_JSON.relative_to(CPLANTBOX_ROOT)}")


def verify_mode(bootstrap_day: int) -> int:
    if not OUT_JSON.exists():
        print(f"MISSING oracle at {OUT_JSON}")
        return 1
    with OUT_JSON.open() as f:
        baseline = json.load(f)
    plant = grow_with_carbon(bootstrap_day)
    snapshot = capture_plant_snapshot(plant)
    sha = fingerprint({str(SEED): snapshot})
    expected = baseline["meta"]["fingerprint_sha256"]
    if sha != expected:
        print(f"  DIFF: expected {expected}\n        got      {sha}")
        return 1
    print(f"  OK (matches oracle); n_organs={len(snapshot)}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--bootstrap-day", type=int,
                        default=BOOTSTRAP_DAY_DEFAULT,
                        help="day at which to switch from grow_plant to carbon mode")
    args = parser.parse_args()
    if args.verify:
        return verify_mode(args.bootstrap_day)
    capture_mode(args.bootstrap_day)
    return 0


if __name__ == "__main__":
    sys.exit(main())
