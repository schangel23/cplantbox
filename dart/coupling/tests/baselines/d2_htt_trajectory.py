#!/usr/bin/env python3
"""Session 6 (D.2) — H(TT) trajectory capture for Birch 2002 cross-check.

Plan: PLAN_FOURNIER_ANDRIEU_INTERNODE_KINETICS_2026-04-23.md §D.2.

Records daily (day, tt_tb8, tt_andrieu, mainstem_top_z, mainstem_length)
under FA-on maize_calibrated.xml with seed 7 and Juelich 2024 daily met.
Output is consumed by `test_fa_htt_trajectory.py` which compares the
trajectory against the Python oracle Σ IL_n(tau_n) constructed from
`fa_kinetics.py` + `phase_III_per_rank.json`.

Scope note: under thin B.3.5 (S3-shipped), all mainstem growth is appended
at the apex. Total apex height H(TT) = Σ IL_n(tau_n) is well-defined and
matches the FA target length by construction (the calcLengthPerPhytomer
sum that drives `createSegments`). So D.2's self-consistency check proves
the C++ port reproduces the Python oracle's cumulative curve — it does NOT
prove per-rank internode-length trajectories track Birch Fig 6 without
full per-phytomer bookkeeping (S3b, deferred).

External Birch Déa H(TT) digitization is a Ch-2 item; for Ch-1 D.2 we
validate against the internally-consistent oracle (FA 2000 data that
Birch 2002 reanalyzes).

Run (from /home/lukas/PHD/CPlantBox):
    cpbenv/bin/python3 dart/coupling/tests/baselines/d2_htt_trajectory.py
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
MAX_DAYS = 130
MAX_RANK = 16


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


def extract_mainstem(plant):
    for o in plant.getOrgans():
        if o.organType() == pb.OrganTypes.stem and int(o.getParameter("subType")) == 1:
            return o
    return None


def extract_mainstem_leaf_emergences(plant):
    """Return [(rank, subType, emergence_andrieu_tt), ...] for mainstem leaves.

    Rank = 1-based ordinal among the mainstem's leaf children (acropetal via
    parentNI). Reads Leaf::emergence_andrieu_tt_ directly; -1 means not yet
    emerged.
    """
    mainstem = extract_mainstem(plant)
    if mainstem is None:
        return []
    mainstem_id = mainstem.getId()
    mainstem_leaves = [lf for lf in plant.getOrgans()
                       if lf.organType() == pb.OrganTypes.leaf
                       and lf.getParent() is not None
                       and lf.getParent().getId() == mainstem_id]
    mainstem_leaves.sort(key=lambda lf: lf.parentNI)
    out = []
    for rank, lf in enumerate(mainstem_leaves, start=1):
        em = float(lf.getEmergenceAndrieuTT()) if hasattr(lf, "getEmergenceAndrieuTT") else -1.0
        out.append({
            "rank": rank,
            "subType": int(lf.getParameter("subType")),
            "parentNI": int(lf.parentNI),
            "emergence_andrieu_tt": em,
        })
    return out


def main():
    print(f"D.2 H(TT) trajectory capture (seed={SEED}, Juelich 2024 met, {MAX_DAYS} d)")
    plant = pb.MappedPlant(SEED)
    plant.readParameters(str(XML_PATH))
    plant.setSeed(SEED)
    setup_successor_where(plant)
    enable_fa(plant)
    plant.initialize()

    met = get_daily_met(daily_met=None)
    trajectory = []
    for day in range(1, MAX_DAYS + 1):
        T = float(met.get(day, {}).get("T_mean_C", 25.0)) if met else 25.0
        plant.setAirTemperature(T)
        try:
            plant.simulate(1.0, False)
        except (IndexError, RuntimeError) as e:
            print(f"  simulate() error at day {day}: {e}")
            break
        tt_tb8 = plant.getAccumulatedTT() if hasattr(plant, "getAccumulatedTT") else -1.0
        tt_a = plant.getAccumulatedAndrieuTT() if hasattr(plant, "getAccumulatedAndrieuTT") else -1.0
        mainstem = extract_mainstem(plant)
        if mainstem is None:
            continue
        nodes = list(mainstem.getNodes())
        if not nodes:
            continue
        mainstem_top_z = max(float(n.z) for n in nodes)
        mainstem_length = float(mainstem.getLength())
        trajectory.append({
            "day": day,
            "T_mean_C": T,
            "tt_tb8": tt_tb8,
            "tt_andrieu": tt_a,
            "mainstem_top_z_cm": mainstem_top_z,
            "mainstem_length_cm": mainstem_length,
            "n_mainstem_nodes": len(nodes),
        })
        if day % 10 == 0:
            print(f"  d={day:3d} T={T:5.1f}°C  TT_8={tt_tb8:6.1f}  TT_A={tt_a:6.1f}  "
                  f"top_z={mainstem_top_z:6.2f} cm  L={mainstem_length:6.2f} cm  "
                  f"nodes={len(nodes)}")

    # Final leaf emergence schedule (Andrieu-axis primordium proxies per rank).
    leaf_emergences = extract_mainstem_leaf_emergences(plant)

    out = BASELINE_DIR / "d2_htt_trajectory.json"
    out.write_text(json.dumps({
        "seed": SEED,
        "xml": XML_PATH.name,
        "max_days": MAX_DAYS,
        "n_ranks": MAX_RANK,
        "trajectory": trajectory,
        "leaf_emergences_final": leaf_emergences,
    }, indent=2, default=float))
    print(f"\nSaved {len(trajectory)} daily samples to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
