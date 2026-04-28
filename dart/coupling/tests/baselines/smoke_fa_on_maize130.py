#!/usr/bin/env python3
"""Session 3 smoke test: maize_calibrated with FA kinetics enabled, 130 days.

Enables `use_fournier_andrieu_kinetics` on the mainstem and populates per-rank
kinetic vectors from `dart/coupling/data/phase_III_per_rank.json`, then runs
130 days under Juelich 2024 met. Reports mainstem top height, plant height,
tassel emergence day, and per-rank internode lengths for cross-checking
against D.1 invariants.

Usage (from /home/lukas/PHD/CPlantBox):
    cpbenv/bin/python3 dart/coupling/tests/baselines/smoke_fa_on_maize130.py
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
SIM_DAYS = 130
MAX_RANK = 16                                        # maize_calibrated mainstem has 16 leaves


def load_fa_kinetics(n_ranks: int):
    with KINETICS_PATH.open() as f:
        data = json.load(f)
    v_table = data["v_n_cm_per_degCd"]["expt_1B_primary"]
    d_table = data["D_n_degCd"]["values"]
    il_table = data["IL_final_cross_check_cm"]["values"]
    # Build 0-indexed arrays of length n_ranks (rank n → index n-1).
    v_n = [0.0] * n_ranks
    D_n = [0.0] * n_ranks
    IL_final = [0.0] * n_ranks
    for n in range(1, n_ranks + 1):
        key = str(n)
        # Default for out-of-table ranks: fall back to nearest (gentle extrapolation
        # for ranks 16-17 that Fig 12B didn't measure).
        v_n[n - 1] = float(v_table.get(key, v_table.get("15", 0.18)))
        D_n[n - 1] = float(d_table.get(key, d_table.get("15", 79)))
        IL_final[n - 1] = float(il_table.get(key, il_table.get("15", 16)))
    return v_n, D_n, IL_final


def enable_fa_on_mainstem(plant):
    srp = plant.getOrganRandomParameter(pb.OrganTypes.stem, 1)
    srp.use_fournier_andrieu_kinetics = True
    v_n, D_n, IL_final = load_fa_kinetics(MAX_RANK)
    srp.internode_v_n = v_n
    srp.internode_D_n = D_n
    srp.internode_IL_final = IL_final
    print(f"FA enabled on mainstem (subType=1):")
    print(f"  v_n (cm/degCd):   {v_n}")
    print(f"  D_n (degCd):      {D_n}")
    print(f"  IL_final (cm):    {IL_final}")


def grow_fa(xml_path, days, seed):
    plant = pb.MappedPlant(seed)
    plant.readParameters(str(xml_path))
    plant.setSeed(seed)
    setup_successor_where(plant)
    enable_fa_on_mainstem(plant)
    plant.initialize()

    met_lookup = get_daily_met(daily_met=None)
    tassel_emerge_day = None
    dt = 1.0
    total = 0.0
    while total < days:
        step = min(dt, days - total)
        sim_day_1b = int(total) + 1
        T_air = float(met_lookup[sim_day_1b]["T_mean_C"]) if (met_lookup and sim_day_1b in met_lookup) else 25.0
        plant.setAirTemperature(T_air)
        try:
            plant.simulate(step, False)
            total += step
        except (IndexError, RuntimeError) as e:
            print(f"  simulate() error at day {total + step:.1f}: {e}")
            break
        # Track tassel emergence
        if tassel_emerge_day is None:
            for o in plant.getOrgans():
                if o.organType() == pb.OrganTypes.stem:
                    st = int(o.getParameter("subType"))
                    if st == 20 and o.getAge() > 0:
                        tassel_emerge_day = total
                        break
    return plant, tassel_emerge_day


def summarize(plant, tassel_emerge_day):
    organs = plant.getOrgans()
    stems = [o for o in organs if o.organType() == pb.OrganTypes.stem]
    mainstems = [s for s in stems if int(s.getParameter("subType")) == 1]
    leaves = [o for o in organs if o.organType() == pb.OrganTypes.leaf]
    tassel_spikes = [s for s in stems if int(s.getParameter("subType")) == 20]

    if len(mainstems) != 1:
        print(f"WARNING: expected 1 mainstem, got {len(mainstems)}")
        return
    mainstem = mainstems[0]
    mainstem_nodes = list(mainstem.getNodes())
    mainstem_top_z = max(float(n.z) for n in mainstem_nodes)
    mainstem_length = mainstem.getLength()

    mainstem_id = mainstem.getId()
    mainstem_leaves = [
        lf for lf in leaves
        if lf.getParent() and lf.getParent().getId() == mainstem_id
    ]
    topmost_leaf_insertion_z = 0.0
    if mainstem_leaves:
        topmost_leaf_insertion_z = max(
            float(lf.getNodes()[0].z) for lf in mainstem_leaves if len(lf.getNodes()) > 0
        )

    print("\n=== Session 3 FA-on smoke test results ===")
    print(f"Sim days:          {SIM_DAYS}")
    print(f"Mainstem nodes:    {len(mainstem_nodes)}")
    print(f"Mainstem length:   {mainstem_length:.2f} cm")
    print(f"Mainstem top z:    {mainstem_top_z:.2f} cm (structural FA output, 187–197 cm per plan D.1 §580)")
    print(f"Topmost leaf z:    {topmost_leaf_insertion_z:.2f} cm (D.1 endpoint invariant, FA-on post-S1-S4: 186.96 ± 1.0 cm)")
    print(f"N leaves:          {len(leaves)}")
    print(f"N tassel spikes:   {len(tassel_spikes)}")
    if tassel_emerge_day is not None:
        print(f"Tassel emerged:    day {tassel_emerge_day:.0f} (invariant target 120-130)")
    else:
        print(f"Tassel emerged:    NOT DETECTED within {SIM_DAYS} days")

    # Per-rank leaf emergence_andrieu_tt sanity
    mainstem_leaves = sorted(
        [lf for lf in leaves if lf.getParent() and lf.getParent().getId() == mainstem.getId()],
        key=lambda o: o.getId(),
    )
    print(f"\nPer-rank leaf emergence_andrieu_tt (sampled from Leaf.emergence_andrieu_tt_):")
    for i, lf in enumerate(mainstem_leaves[:MAX_RANK], 1):
        try:
            etta = lf.getEmergenceAndrieuTT() if hasattr(lf, "getEmergenceAndrieuTT") else None
        except Exception:
            etta = None
        sub = int(lf.getParameter("subType"))
        print(f"  rank {i:2d} (subType {sub:2d}): emergence_andrieu_tt={etta}")

    # Invariant checks (D.1 post-Session-4 observables — see plan §580)
    failures = []
    if abs(topmost_leaf_insertion_z - 186.96) >= 1.0:
        failures.append(
            f"topmost_leaf_insertion_z {topmost_leaf_insertion_z:.2f} not within 186.96±1.0 "
            "(post-S1-S4 peduncle-exuberance fix anchor)"
        )
    if not (187.0 <= mainstem_top_z <= 197.0):
        failures.append(
            f"mainstem_top_z {mainstem_top_z:.2f} outside structural-output bound [187.0, 197.0]"
        )
    if tassel_emerge_day is None or not (120 <= tassel_emerge_day <= 130):
        failures.append(f"tassel_emerge_day={tassel_emerge_day} not in 120..130")
    if failures:
        print(f"\nINVARIANT FAILURES ({len(failures)}):")
        for f in failures:
            print(f"  - {f}")
        return False
    print("\nAll day-130 invariants pass (D.1 partial preview).")
    return True


def main():
    print(f"Running FA-on maize smoke test: {XML_PATH.name}, seed={SEED}, days={SIM_DAYS}")
    plant, tassel_emerge_day = grow_fa(XML_PATH, SIM_DAYS, SEED)
    summarize(plant, tassel_emerge_day)


if __name__ == "__main__":
    main()
