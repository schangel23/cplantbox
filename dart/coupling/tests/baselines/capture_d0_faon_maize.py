#!/usr/bin/env python3
"""S3b.4 D.0 baseline capture: maize_calibrated with FA kinetics enabled, 130 days.

Plan: PLAN_FULL_B35_PER_PHYTOMER_BOOKKEEPING_2026-04-23.md §C.

Enables `use_fournier_andrieu_kinetics=True` on the mainstem subType=1 and
populates the per-rank kinetic vectors from `data/phase_III_per_rank.json`,
then runs 130 days under Juelich 2024 daily met (seed=7, MappedPlant ctor
deterministic seeding). Captures the same signature shape as the scalar-path
D.0 baselines (`capture_d0_baselines.py`) into
`d0_maize_calibrated_faon_s3b_130d.json` — aggregate counts (stems, leaves,
roots, mainstem nodes, leaf nodes, root nodes, segments, organs) plus SHA256
hash of (mainstem + all-stems + leaves + roots + segcount).

Pairs with `capture_d0_baselines.py` maize_calibrated_flagoff_130d case to give
an FA-on/FA-off diff on the same XML, and with the plan's thin-B.3.5 reference
(~1962 nodes at the stem level — node count shift must stay < 5% per §C step 1).

Usage (from /home/lukas/PHD/CPlantBox):
    cpbenv/bin/python3 dart/coupling/tests/baselines/capture_d0_faon_maize.py
    cpbenv/bin/python3 dart/coupling/tests/baselines/capture_d0_faon_maize.py --verify
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
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
OUT_PATH = BASELINE_DIR / "d0_maize_calibrated_faon_s3b_130d.json"

SEED = 7
SIM_DAYS = 130
MAX_RANK = 16
T_AIR_DEFAULT = 25.0


def _pack_xyz(nodes) -> bytes:
    buf = bytearray()
    for n in nodes:
        buf += struct.pack("<ddd", float(n.x), float(n.y), float(n.z))
    return bytes(buf)


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


def enable_fa_on_mainstem(plant):
    srp = plant.getOrganRandomParameter(pb.OrganTypes.stem, 1)
    srp.use_fournier_andrieu_kinetics = True
    v_n, D_n, IL_final = load_fa_kinetics(MAX_RANK)
    srp.internode_v_n = v_n
    srp.internode_D_n = D_n
    srp.internode_IL_final = IL_final


def grow_fa(xml_path: Path, sim_days: int, seed: int):
    plant = pb.MappedPlant(seed)
    plant.readParameters(str(xml_path))
    plant.setSeed(seed)
    setup_successor_where(plant)
    enable_fa_on_mainstem(plant)
    plant.initialize()

    met_lookup = get_daily_met(daily_met=None)
    dt = 1.0
    total = 0.0
    while total < sim_days:
        step = min(dt, sim_days - total)
        sim_day_1b = int(total) + 1
        if met_lookup is not None and sim_day_1b in met_lookup:
            T_air = float(met_lookup[sim_day_1b]["T_mean_C"])
        else:
            T_air = T_AIR_DEFAULT
        plant.setAirTemperature(T_air)
        try:
            plant.simulate(step, False)
            total += step
        except (IndexError, RuntimeError) as e:
            print(f"  simulate() error at day {total + step:.1f}: {e}")
            try:
                plant.simulate(0.0)
            except Exception:
                pass
            break
    return plant


def capture_signature(plant) -> dict:
    organs = plant.getOrgans()
    stems = [o for o in organs if o.organType() == pb.OrganTypes.stem]
    leaves = [o for o in organs if o.organType() == pb.OrganTypes.leaf]
    roots = [o for o in organs if o.organType() == pb.OrganTypes.root]

    mainstem_bytes = b""
    stem_bytes_all = b""
    leaf_bytes = b""
    root_bytes = b""
    n_mainstem_nodes = 0
    n_leaf_nodes = 0
    n_root_nodes = 0
    n_segments_total = 0
    mainstem_length = 0.0
    mainstem_top_z = 0.0
    topmost_leaf_insertion_z = 0.0
    mainstem_id = -1

    for s in sorted(stems, key=lambda o: o.getId()):
        nodes = list(s.getNodes())
        stem_bytes_all += _pack_xyz(nodes)
        if int(s.getParameter("subType")) == 1 and len(mainstem_bytes) == 0:
            mainstem_bytes = _pack_xyz(nodes)
            n_mainstem_nodes = len(nodes)
            mainstem_length = float(s.getLength())
            mainstem_top_z = max(float(n.z) for n in nodes) if nodes else 0.0
            mainstem_id = s.getId()
    for lf in sorted(leaves, key=lambda o: o.getId()):
        nodes = list(lf.getNodes())
        leaf_bytes += _pack_xyz(nodes)
        n_leaf_nodes += len(nodes)
    for r in sorted(roots, key=lambda o: o.getId()):
        nodes = list(r.getNodes())
        root_bytes += _pack_xyz(nodes)
        n_root_nodes += len(nodes)
    for o in organs:
        n_segments_total += max(0, len(o.getNodes()) - 1)

    if mainstem_id != -1:
        for lf in leaves:
            p = lf.getParent()
            if p is None or p.getId() != mainstem_id:
                continue
            lf_nodes = lf.getNodes()
            if len(lf_nodes) > 0:
                z = float(lf_nodes[0].z)
                if z > topmost_leaf_insertion_z:
                    topmost_leaf_insertion_z = z

    tassel_spikes = [s for s in stems if int(s.getParameter("subType")) == 20]
    n_tassel_spikes = len(tassel_spikes)

    h = hashlib.sha256()
    h.update(b"mainstem:")
    h.update(mainstem_bytes)
    h.update(b"|stems_all:")
    h.update(stem_bytes_all)
    h.update(b"|leaves:")
    h.update(leaf_bytes)
    h.update(b"|roots:")
    h.update(root_bytes)
    h.update(b"|nseg:")
    h.update(struct.pack("<q", n_segments_total))
    return {
        "slug": "maize_calibrated_faon_s3b_130d",
        "xml": str(XML_PATH.relative_to(CPLANTBOX_ROOT)),
        "seed": SEED,
        "sim_days": SIM_DAYS,
        "fa_enabled": True,
        "n_stems": len(stems),
        "n_leaves": len(leaves),
        "n_roots": len(roots),
        "n_tassel_spikes": n_tassel_spikes,
        "n_mainstem_nodes": n_mainstem_nodes,
        "n_leaf_nodes": n_leaf_nodes,
        "n_root_nodes": n_root_nodes,
        "n_segments_total": n_segments_total,
        "n_organs": len(organs),
        "mainstem_length_cm": round(mainstem_length, 4),
        "mainstem_top_z_cm": round(mainstem_top_z, 4),
        "topmost_leaf_insertion_z_cm": round(topmost_leaf_insertion_z, 4),
        "sha256": h.hexdigest(),
    }


def run_capture() -> dict:
    print(f"FA-on D.0 capture: {XML_PATH.name}, seed={SEED}, days={SIM_DAYS}")
    plant = grow_fa(XML_PATH, SIM_DAYS, SEED)
    sig = capture_signature(plant)
    print(f"  sha256={sig['sha256']}")
    print(f"  stems={sig['n_stems']}  leaves={sig['n_leaves']}  roots={sig['n_roots']}  "
          f"tassel_spikes={sig['n_tassel_spikes']}")
    print(f"  mainstem_nodes={sig['n_mainstem_nodes']}  leaf_nodes={sig['n_leaf_nodes']}  "
          f"root_nodes={sig['n_root_nodes']}  nseg_total={sig['n_segments_total']}")
    print(f"  mainstem_length={sig['mainstem_length_cm']:.2f} cm  "
          f"mainstem_top_z={sig['mainstem_top_z_cm']:.2f} cm  "
          f"topmost_leaf_z={sig['topmost_leaf_insertion_z_cm']:.2f} cm")
    return sig


def capture_mode():
    sig = run_capture()
    with OUT_PATH.open("w") as f:
        json.dump(sig, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"  -> {OUT_PATH.relative_to(CPLANTBOX_ROOT)}")


def verify_mode() -> int:
    if not OUT_PATH.exists():
        print(f"MISSING baseline at {OUT_PATH}")
        return 1
    with OUT_PATH.open() as f:
        baseline = json.load(f)
    sig = run_capture()
    drift_pct = 100.0 * abs(sig["n_segments_total"] - baseline["n_segments_total"]) / max(1, baseline["n_segments_total"])
    if sig["sha256"] != baseline["sha256"]:
        print(f"  DIFF: expected {baseline['sha256']} got {sig['sha256']}")
        print(f"  segment count drift: {drift_pct:.2f}%")
        return 1
    print(f"  OK (matches baseline); segment drift: {drift_pct:.2f}%")
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true",
                        help="re-run and compare against stored FA-on baseline")
    args = parser.parse_args()
    if args.verify:
        sys.exit(verify_mode())
    else:
        capture_mode()


if __name__ == "__main__":
    main()
