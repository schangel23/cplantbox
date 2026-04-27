#!/usr/bin/env python3
"""Session 5.1 — Baseline V-stage calendar under FA-on vs FA-off.

Counts leaf collars per day under Juelich 2024 met, first for the stock
mainstem (FA flag off — our existing Nielsen-calibrated reference) and
then with FA kinetics enabled on subType=1 (the Session 3 configuration).
Reports V1..V6 first-hit days for both variants so we can quantify the
leaf-appearance-calendar drift Session 5 is asked to absorb via
`tt_emergence` refit (Hard Invariant #2, plan §C.1).

Run (from /home/lukas/PHD/CPlantBox):
    cpbenv/bin/python3 dart/coupling/tests/baselines/s5_vstage_fa_baseline.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

BASELINE_DIR = Path(__file__).resolve().parent
COUPLING_DIR = BASELINE_DIR.parent.parent
CPLANTBOX_ROOT = COUPLING_DIR.parent.parent
sys.path.insert(0, str(CPLANTBOX_ROOT))

import plantbox as pb  # noqa: E402

from dart.coupling.carbon.dvs_partitioning import get_daily_met  # noqa: E402
from dart.coupling.growth.grow import setup_successor_where  # noqa: E402

XML_PATH = COUPLING_DIR / "data" / "maize_calibrated.xml"
KINETICS_PATH = COUPLING_DIR / "data" / "phase_III_per_rank.json"
SEED = 7
MAX_DAYS = 70                    # V6 target is day 57 under Nielsen
MAX_RANK = 16
V_TARGETS = (1, 2, 3, 4, 5, 6)
COLLAR_RELEASE = 0.45
COLLAR_THRESHOLD = 0.30
# Nielsen (from project_young_whorl_fix.md): V1=17, V2=25, V3=33, V4=51, V6=57.
NIELSEN = {1: 17, 2: 25, 3: 33, 4: 51, 5: None, 6: 57}


def load_fa_kinetics(n_ranks: int):
    data = json.loads(KINETICS_PATH.read_text())
    v_table = data["v_n_cm_per_degCd"]["expt_1B_primary"]
    d_table = data["D_n_degCd"]["values"]
    il_table = data["IL_final_cross_check_cm"]["values"]
    v_n, D_n, IL = [0.0] * n_ranks, [0.0] * n_ranks, [0.0] * n_ranks
    for n in range(1, n_ranks + 1):
        k = str(n)
        v_n[n - 1] = float(v_table.get(k, v_table.get("15", 0.18)))
        D_n[n - 1] = float(d_table.get(k, d_table.get("15", 79)))
        IL[n - 1] = float(il_table.get(k, il_table.get("15", 16)))
    return v_n, D_n, IL


def enable_fa(plant):
    srp = plant.getOrganRandomParameter(pb.OrganTypes.stem, 1)
    srp.use_fournier_andrieu_kinetics = True
    v_n, D_n, IL = load_fa_kinetics(MAX_RANK)
    srp.internode_v_n = v_n
    srp.internode_D_n = D_n
    srp.internode_IL_final = IL


def count_collars(plant) -> tuple[int, int, int, int]:
    n_total = n_coll = n_emrg = n_whrl = 0
    for organ in plant.getOrgans(pb.leaf):
        nodes = organ.getNodes()
        if len(nodes) < 2:
            continue
        lrp = organ.getLeafRandomParameter()
        lmax = max(float(lrp.lmax), 1e-9)
        cur = 0.0
        pts = [(float(n.x), float(n.y), float(n.z)) for n in nodes]
        for i in range(1, len(pts)):
            dx, dy, dz = (pts[i][j] - pts[i - 1][j] for j in range(3))
            cur += (dx * dx + dy * dy + dz * dz) ** 0.5
        m = min(cur / lmax, 1.0)
        n_total += 1
        if m >= COLLAR_RELEASE:
            n_coll += 1
        elif m >= COLLAR_THRESHOLD:
            n_emrg += 1
        else:
            n_whrl += 1
    return n_total, n_coll, n_emrg, n_whrl


def simulate_days(fa_on: bool, max_days: int = MAX_DAYS):
    plant = pb.MappedPlant(SEED)
    plant.readParameters(str(XML_PATH))
    plant.setSeed(SEED)
    setup_successor_where(plant)
    if fa_on:
        enable_fa(plant)
    plant.initialize()

    met = get_daily_met(daily_met=None)
    targets = list(V_TARGETS)
    hits: dict[int, int] = {}
    per_day: list[dict] = []

    for day in range(1, max_days + 1):
        T = float(met.get(day, {}).get("T_mean_C", 25.0)) if met else 25.0
        plant.setAirTemperature(T)
        plant.simulate(1.0, False)
        tt = plant.getAccumulatedTT() if hasattr(plant, "getAccumulatedTT") else -1.0
        tt_a = plant.getAccumulatedAndrieuTT() if hasattr(plant, "getAccumulatedAndrieuTT") else -1.0
        n_total, n_coll, n_emrg, n_whrl = count_collars(plant)
        per_day.append({
            "day": day, "tt_tb8": tt, "tt_andrieu": tt_a,
            "total": n_total, "coll": n_coll, "emrg": n_emrg, "whrl": n_whrl,
        })
        while targets and n_coll >= targets[0] and (n_emrg + n_whrl) >= 2:
            v = targets.pop(0)
            hits[v] = day
        if not targets:
            break

    return hits, per_day


def fmt_v(day):
    return f"{day:3d}" if day is not None else "  —"


def main():
    print(f"S5.1 V-stage baseline (seed={SEED}, Juelich 2024 met, max {MAX_DAYS} days)")
    print(f"  XML: {XML_PATH.name}")
    print()

    print("Running FA-off (Nielsen reference) ...")
    hits_off, log_off = simulate_days(fa_on=False)
    print("Running FA-on  (Session 3 kinetics) ...")
    hits_on, log_on = simulate_days(fa_on=True)

    print()
    print(f"{'V':>3} {'Nielsen':>9} {'FA-off':>8} {'FA-on':>8} "
          f"{'Δoff':>6} {'Δon':>6} {'Δ(on-off)':>10}")
    for v in V_TARGETS:
        ref = NIELSEN.get(v)
        d_off = hits_off.get(v)
        d_on = hits_on.get(v)
        delta_off = (d_off - ref) if (d_off is not None and ref is not None) else None
        delta_on = (d_on - ref) if (d_on is not None and ref is not None) else None
        delta_fa = (d_on - d_off) if (d_on is not None and d_off is not None) else None
        print(
            f"V{v:<2} {fmt_v(ref):>9} {fmt_v(d_off):>8} {fmt_v(d_on):>8} "
            f"{('' if delta_off is None else f'{delta_off:+d}'):>6} "
            f"{('' if delta_on is None else f'{delta_on:+d}'):>6} "
            f"{('' if delta_fa is None else f'{delta_fa:+d}'):>10}"
        )

    out = BASELINE_DIR / "s5_vstage_fa_baseline.json"
    out.write_text(json.dumps({
        "seed": SEED,
        "xml": XML_PATH.name,
        "max_days": MAX_DAYS,
        "nielsen_targets": NIELSEN,
        "hits_fa_off": hits_off,
        "hits_fa_on": hits_on,
        "daily_fa_off": log_off,
        "daily_fa_on": log_on,
    }, indent=2, default=float))
    print(f"\nSaved full log to {out}")

    # Exit-gate summary per plan D.3: V1..V6 within ±2 days of Nielsen.
    failures = []
    for v in V_TARGETS:
        ref = NIELSEN.get(v)
        if ref is None:
            continue
        d_on = hits_on.get(v)
        if d_on is None:
            failures.append(f"V{v} not reached by day {MAX_DAYS} under FA-on")
            continue
        if abs(d_on - ref) > 2:
            failures.append(f"V{v}: FA-on day {d_on} vs Nielsen {ref} (Δ={d_on - ref:+d}, >2d)")
    print()
    if failures:
        print(f"D.3 exit gate: FAIL ({len(failures)} drift(s) >2 days)")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("D.3 exit gate: PASS (all V1..V6 within ±2 days of Nielsen)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
