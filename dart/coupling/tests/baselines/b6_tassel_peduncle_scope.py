#!/usr/bin/env python3
"""Session 4 (B.6) — tassel + peduncle + cessation scope regression.

Plan: PLAN_FOURNIER_ANDRIEU_INTERNODE_KINETICS_2026-04-23.md §B.6.

Three failure-mode probes:
  1. Tassel internode (subType 20/21) cessation latch firing on the wrong
     subtype.
  2. cessation_age_ triggering before all leaves have emerged.
  3. Mainstem subType=1 includes the peduncle (apical `la` zone above the
     topmost leaf), so FA kinetics run on it as if it were vegetative.

Approach: run two day-130 maize sims under Juelich met, seed 7:
  - FA-on  (mainstem `use_fournier_andrieu_kinetics=True`, per-rank vectors
    populated from `data/phase_III_per_rank.json`)
  - FA-off (same XML, flag forced False — bit-identical baseline path)

For each, snapshot:
  - mainstem subType=1 node count, top z, total length
  - per-rank leaf insertion z (rank = leaf order on mainstem)
  - peduncle length = mainstem_top_z - topmost_leaf_insertion_z
  - tassel subType=20 first-node z, emergence day
  - cessation_andrieu_tt latch fired-day (FA-on uses cessation_andrieu_tt_;
    FA-off uses the legacy cessation_age_ → wallclock day proxy)

Compute peduncle_kinetic_error_cm = |peduncle_FA - peduncle_scalar|.

Writes summary to `b6_tassel_peduncle_scope.json` for the pytest to consume.

Usage (from /home/lukas/PHD/CPlantBox):
    cpbenv/bin/python3 dart/coupling/tests/baselines/b6_tassel_peduncle_scope.py
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
MAX_RANK = 16
SUMMARY_PATH = BASELINE_DIR / "b6_tassel_peduncle_scope.json"


def load_fa_kinetics(n_ranks: int):
    with KINETICS_PATH.open() as f:
        data = json.load(f)
    v_table = data["v_n_cm_per_degCd"]["expt_1B_primary"]
    d_table = data["D_n_degCd"]["values"]
    il_table = data["IL_final_cross_check_cm"]["values"]
    v_n = [0.0] * n_ranks
    D_n = [0.0] * n_ranks
    IL_final = [0.0] * n_ranks
    for n in range(1, n_ranks + 1):
        key = str(n)
        v_n[n - 1] = float(v_table.get(key, v_table.get("15", 0.18)))
        D_n[n - 1] = float(d_table.get(key, d_table.get("15", 79)))
        IL_final[n - 1] = float(il_table.get(key, il_table.get("15", 16)))
    return v_n, D_n, IL_final


def configure_mainstem(plant, fa_on: bool):
    srp = plant.getOrganRandomParameter(pb.OrganTypes.stem, 1)
    srp.use_fournier_andrieu_kinetics = bool(fa_on)
    if fa_on:
        v_n, D_n, IL_final = load_fa_kinetics(MAX_RANK)
        srp.internode_v_n = v_n
        srp.internode_D_n = D_n
        srp.internode_IL_final = IL_final


def grow(xml_path: Path, fa_on: bool, days: int, seed: int):
    plant = pb.MappedPlant(seed)
    plant.readParameters(str(xml_path))
    plant.setSeed(seed)
    setup_successor_where(plant)
    configure_mainstem(plant, fa_on)
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
            print(f"  [{('FA-on' if fa_on else 'FA-off')}] simulate() error at day {total + step:.1f}: {e}")
            break
        if tassel_emerge_day is None:
            for o in plant.getOrgans():
                if o.organType() == pb.OrganTypes.stem and int(o.getParameter("subType")) == 20:
                    if o.getAge() > 0:
                        tassel_emerge_day = total
                        break
    return plant, tassel_emerge_day


def snapshot(plant, tassel_emerge_day, fa_on: bool) -> dict:
    organs = plant.getOrgans()
    stems = [o for o in organs if o.organType() == pb.OrganTypes.stem]
    leaves = [o for o in organs if o.organType() == pb.OrganTypes.leaf]
    mainstems = [s for s in stems if int(s.getParameter("subType")) == 1]
    tassel_spikes = [s for s in stems if int(s.getParameter("subType")) == 20]
    tassel_branches = [s for s in stems if int(s.getParameter("subType")) == 21]

    if len(mainstems) != 1:
        raise RuntimeError(f"expected exactly 1 mainstem, got {len(mainstems)}")
    mainstem = mainstems[0]
    mainstem_nodes = list(mainstem.getNodes())
    mainstem_top_z = max(float(n.z) for n in mainstem_nodes)
    mainstem_length = mainstem.getLength()
    mainstem_id = mainstem.getId()

    # Leaves whose parent is the mainstem, ordered by parentNI (acropetal).
    mainstem_leaves = [lf for lf in leaves
                       if lf.getParent() is not None and lf.getParent().getId() == mainstem_id]
    mainstem_leaves.sort(key=lambda lf: lf.parentNI)

    leaf_insertions = []
    for lf in mainstem_leaves:
        pni = int(lf.parentNI)
        if 0 <= pni < len(mainstem_nodes):
            insertion_z = float(mainstem_nodes[pni].z)
        else:
            insertion_z = float("nan")
        leaf_insertions.append({
            "leaf_id": int(lf.getId()),
            "subType": int(lf.getParameter("subType")),
            "parentNI": pni,
            "insertion_z_cm": insertion_z,
            "age_d": float(lf.getAge()),
        })

    # Peduncle = apical zone of mainstem subType=1 above the topmost leaf
    # insertion. Per maize_calibrated.xml: la=22 cm.
    peduncle_length_cm = None
    if leaf_insertions:
        topmost_leaf_z = max(li["insertion_z_cm"] for li in leaf_insertions)
        peduncle_length_cm = mainstem_top_z - topmost_leaf_z

    # Tassel spike first-node z (insertion onto mainstem) and length.
    tassel_spike_summary = []
    for ts in tassel_spikes:
        ts_nodes = list(ts.getNodes())
        ts_first_z = float(ts_nodes[0].z) if ts_nodes else float("nan")
        ts_top_z = max((float(n.z) for n in ts_nodes), default=float("nan"))
        tassel_spike_summary.append({
            "spike_id": int(ts.getId()),
            "first_node_z_cm": ts_first_z,
            "top_node_z_cm": ts_top_z,
            "length_cm": float(ts.getLength()),
            "age_d": float(ts.getAge()),
            "n_nodes": len(ts_nodes),
        })

    # Cessation diagnostics (FA-on samples cessation_andrieu_tt_; both axes
    # exposed on mainstem if they fired).
    cessation = {}
    try:
        cessation["cessation_age_d"] = float(getattr(mainstem, "cessation_age_", -1.0))
    except Exception:
        cessation["cessation_age_d"] = None
    try:
        cessation["cessation_andrieu_tt_degCd"] = float(getattr(mainstem, "cessation_andrieu_tt_", -1.0))
    except Exception:
        cessation["cessation_andrieu_tt_degCd"] = None

    # Mainstem lifecycle flag — load-bearing for PiafMunch (runPM.cpp:697,847
    # gate carbon-water growth on isActive()). Post-S1-S4 + codex-rescue
    # follow-up, FA-on must report False after cessation_age_ latches.
    try:
        mainstem_is_active = bool(mainstem.isActive())
    except Exception:
        mainstem_is_active = None

    return {
        "fa_on": fa_on,
        "sim_days": SIM_DAYS,
        "seed": SEED,
        "n_mainstem_nodes": len(mainstem_nodes),
        "mainstem_length_cm": float(mainstem_length),
        "mainstem_top_z_cm": mainstem_top_z,
        "mainstem_is_active": mainstem_is_active,
        "n_leaves_total": len(leaves),
        "n_mainstem_leaves": len(mainstem_leaves),
        "leaf_insertions": leaf_insertions,
        "peduncle_length_cm": peduncle_length_cm,
        "n_tassel_spikes": len(tassel_spikes),
        "n_tassel_branches": len(tassel_branches),
        "tassel_spikes": tassel_spike_summary,
        "tassel_emerge_day": tassel_emerge_day,
        "cessation": cessation,
    }


def print_snapshot(label: str, snap: dict):
    print(f"\n=== {label} ===")
    print(f"sim_days={snap['sim_days']}  seed={snap['seed']}  fa_on={snap['fa_on']}")
    print(f"mainstem subType=1: nodes={snap['n_mainstem_nodes']}  length={snap['mainstem_length_cm']:.2f} cm  top_z={snap['mainstem_top_z_cm']:.2f} cm")
    print(f"  attached leaves: {snap['n_mainstem_leaves']} (total leaves on plant: {snap['n_leaves_total']})")
    print(f"  topmost leaf insertion z: {max((li['insertion_z_cm'] for li in snap['leaf_insertions']), default=float('nan')):.2f} cm")
    print(f"  peduncle length (mainstem_top - topmost_leaf_z): {snap['peduncle_length_cm']:.2f} cm  [XML la=22]")
    print(f"tassel spikes (subType=20): {snap['n_tassel_spikes']}, branches (subType=21): {snap['n_tassel_branches']}")
    for ts in snap["tassel_spikes"]:
        print(f"  spike#{ts['spike_id']}: first_z={ts['first_node_z_cm']:.2f}  top_z={ts['top_node_z_cm']:.2f}  length={ts['length_cm']:.2f}  age={ts['age_d']:.1f} d  nodes={ts['n_nodes']}")
    print(f"  tassel emerge day: {snap['tassel_emerge_day']}")
    print(f"  cessation: age={snap['cessation']['cessation_age_d']} d  andrieu_tt={snap['cessation']['cessation_andrieu_tt_degCd']} degCd")
    print(f"  mainstem isActive(): {snap['mainstem_is_active']}")


def report_discovery(fa_snap: dict, scalar_snap: dict) -> dict:
    """B.6 discovery: peduncle scope + FA-vs-scalar mismatch."""
    n_vegetative_leaves = MAX_RANK
    fa_mainstem_nodes = fa_snap["n_mainstem_nodes"]
    fa_n_attached = fa_snap["n_mainstem_leaves"]

    # Plan §B.6: "If [mainstem_node_count] is exactly 16: peduncle is subType=20.
    # If it's 17+: peduncle in subType=1, FA runs on it."
    # Note: at thin B.3.5, mainstems carry hundreds of nodes (dx=0.1 resampling).
    #
    # Two independent facts captured separately so a downstream test can't
    # silently confuse them:
    #   * peduncle_in_mainstem_subtype1 — true topological ownership: did we
    #     successfully measure peduncle_length_cm from a subType=1 mainstem?
    #     This is the always-True B.6 precondition (failure = the peduncle
    #     migrated off subType=1, e.g. into subType=20 tassel internodes).
    #   * peduncle_exuberant_in_mainstem — the regression marker: is the
    #     apical leafless zone large enough (>5 cm) to constitute "peduncle
    #     exuberance"? True pre-S1-S4 (~42 cm), False post-S1-S4 (~1 cm).
    peduncle_in_mainstem = fa_snap["peduncle_length_cm"] is not None
    peduncle_exuberant = (fa_snap["peduncle_length_cm"] or 0.0) > 5.0
    tassel_above_mainstem = False
    if fa_snap["tassel_spikes"]:
        ts_first_z = fa_snap["tassel_spikes"][0]["first_node_z_cm"]
        tassel_above_mainstem = ts_first_z >= fa_snap["mainstem_top_z_cm"] - 1.0

    pedun_fa = fa_snap["peduncle_length_cm"] or 0.0
    pedun_scalar = scalar_snap["peduncle_length_cm"] or 0.0
    peduncle_kinetic_error_cm = abs(pedun_fa - pedun_scalar)

    return {
        "n_vegetative_leaves": n_vegetative_leaves,
        "fa_n_mainstem_attached_leaves": fa_n_attached,
        "fa_mainstem_node_count": fa_mainstem_nodes,
        "peduncle_in_mainstem_subtype1": bool(peduncle_in_mainstem),
        "peduncle_exuberant_in_mainstem": bool(peduncle_exuberant),
        "tassel_first_node_at_or_above_mainstem_top": bool(tassel_above_mainstem),
        "peduncle_length_fa_cm": pedun_fa,
        "peduncle_length_scalar_cm": pedun_scalar,
        "peduncle_kinetic_error_cm": peduncle_kinetic_error_cm,
        "mainstem_top_z_fa_cm": fa_snap["mainstem_top_z_cm"],
        "mainstem_top_z_scalar_cm": scalar_snap["mainstem_top_z_cm"],
        "mainstem_top_diff_cm": abs(fa_snap["mainstem_top_z_cm"] - scalar_snap["mainstem_top_z_cm"]),
    }


def main():
    print(f"B.6 scope regression: {XML_PATH.name}, seed={SEED}, days={SIM_DAYS}")

    print("\n[1/2] FA-OFF (scalar) baseline grow")
    plant_off, tassel_off = grow(XML_PATH, fa_on=False, days=SIM_DAYS, seed=SEED)
    snap_off = snapshot(plant_off, tassel_off, fa_on=False)
    del plant_off
    print_snapshot("SCALAR (FA-off)", snap_off)

    print("\n[2/2] FA-ON grow")
    plant_on, tassel_on = grow(XML_PATH, fa_on=True, days=SIM_DAYS, seed=SEED)
    snap_on = snapshot(plant_on, tassel_on, fa_on=True)
    del plant_on
    print_snapshot("FA-on (Fournier-Andrieu mainstem)", snap_on)

    discovery = report_discovery(snap_on, snap_off)
    print("\n=== B.6 discovery ===")
    print(f"  vegetative leaf count (calibrated):                 {discovery['n_vegetative_leaves']}")
    print(f"  FA mainstem attached-leaf count:                    {discovery['fa_n_mainstem_attached_leaves']}")
    print(f"  FA mainstem node count (resampled @ dx):            {discovery['fa_mainstem_node_count']}")
    print(f"  peduncle owned by mainstem subType=1?               {discovery['peduncle_in_mainstem_subtype1']}  (B.6 precondition)")
    print(f"  peduncle exuberant (>5 cm apical zone, FA-on)?      {discovery['peduncle_exuberant_in_mainstem']}  (regression marker)")
    print(f"  tassel spike first-node at/above mainstem top?      {discovery['tassel_first_node_at_or_above_mainstem_top']}")
    print(f"  peduncle length FA / scalar / |diff| (cm):          {discovery['peduncle_length_fa_cm']:.2f}  {discovery['peduncle_length_scalar_cm']:.2f}  {discovery['peduncle_kinetic_error_cm']:.2f}")
    print(f"  mainstem top z FA / scalar / |diff| (cm):           {discovery['mainstem_top_z_fa_cm']:.2f}  {discovery['mainstem_top_z_scalar_cm']:.2f}  {discovery['mainstem_top_diff_cm']:.2f}")

    summary = {
        "plan_section": "B.6",
        "xml": str(XML_PATH.relative_to(CPLANTBOX_ROOT)),
        "seed": SEED,
        "sim_days": SIM_DAYS,
        "fa_off": snap_off,
        "fa_on": snap_on,
        "discovery": discovery,
    }
    with SUMMARY_PATH.open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"\nWrote summary: {SUMMARY_PATH.relative_to(CPLANTBOX_ROOT)}")

    # Invariants per plan §B.6 + §D.1.
    failures = []
    if not snap_on["tassel_emerge_day"] or not (120 <= snap_on["tassel_emerge_day"] <= 130):
        failures.append(f"tassel_emerge_day={snap_on['tassel_emerge_day']} not in [120,130]")
    if not (0 < snap_on["n_tassel_spikes"]):
        failures.append("FA-on produced no tassel spike (subType=20)")
    if not (0 < snap_on["n_mainstem_leaves"] <= MAX_RANK):
        failures.append(f"FA-on mainstem leaf count {snap_on['n_mainstem_leaves']} not in (0,{MAX_RANK}]")

    if failures:
        print(f"\nB.6 INVARIANT FAILURES ({len(failures)}):")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print("\nB.6 invariants pass (tassel window + mainstem-leaf count + tassel spawned).")


if __name__ == "__main__":
    main()
