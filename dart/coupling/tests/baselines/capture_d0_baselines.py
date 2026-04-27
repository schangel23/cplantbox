#!/usr/bin/env python3
"""Capture pre-S3 D.0 baselines for the Fournier-Andrieu multi-species gate.

Plan: PLAN_FOURNIER_ANDRIEU_INTERNODE_KINETICS_2026-04-23.md §D.0.

Runs 6 XMLs that exercise the scalar Stem::simulate path, hashes
(mainstem skeleton xyz + leaf node xyz + segment count) per XML into
baselines/d0_<slug>.json. After S3 rebuilds the .so with the FA branching in
Stem::simulate, re-run this script and diff: every non-FA case must be
byte-identical (Hard Invariant #5).

Seeding uses MappedPlant(seed) ctor — see capture_maize_flagoff_baseline.py
docstring for why setSeed-after-ctor is nondeterministic.

Usage (from /home/lukas/PHD/CPlantBox):
    cpbenv/bin/python3 dart/coupling/tests/baselines/capture_d0_baselines.py [--verify]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from dataclasses import dataclass
from pathlib import Path

BASELINE_DIR = Path(__file__).resolve().parent
COUPLING_DIR = BASELINE_DIR.parent.parent             # .../CPlantBox/dart/coupling
CPLANTBOX_ROOT = COUPLING_DIR.parent.parent           # .../CPlantBox
sys.path.insert(0, str(CPLANTBOX_ROOT))               # so `import dart.coupling...` resolves

import plantbox as pb  # noqa: E402

from dart.coupling.carbon.dvs_partitioning import get_daily_met  # noqa: E402
from dart.coupling.growth.grow import setup_successor_where  # noqa: E402


SEED = 7
T_AIR_DEFAULT = 25.0


@dataclass(frozen=True)
class Case:
    slug: str
    xml: Path
    days: int
    use_daily_met: bool  # only maize cases use Juelich met; others use constant T=25


CASES = [
    Case(
        "wheat_calibrated_130d",
        COUPLING_DIR / "data" / "wheat_calibrated.xml",
        130,
        True,
    ),
    Case(
        "brassica_oleracea_vansteenkiste_2014_60d",
        CPLANTBOX_ROOT / "modelparameter" / "structural" / "plant"
        / "Brassica_oleracea_Vansteenkiste_2014.xml",
        60,
        False,
    ),
    Case(
        "modelparam_4_30d",
        CPLANTBOX_ROOT / "modelparameter" / "structural" / "plant" / "4.xml",
        30,
        False,
    ),
    Case(
        "carbon2020_30d",
        CPLANTBOX_ROOT / "modelparameter" / "structural" / "plant" / "carbon2020.xml",
        30,
        False,
    ),
    Case(
        "legacy_2020_maize_60d",
        CPLANTBOX_ROOT / "modelparameter" / "structural" / "plant" / "2020-maize.xml",
        60,
        True,
    ),
    Case(
        "maize_calibrated_flagoff_130d",
        COUPLING_DIR / "data" / "maize_calibrated.xml",
        130,
        True,
    ),
]


def _pack_xyz(nodes) -> bytes:
    buf = bytearray()
    for n in nodes:
        buf += struct.pack("<ddd", float(n.x), float(n.y), float(n.z))
    return bytes(buf)


def grow_deterministic(xml_path: Path, sim_days: int, seed: int, use_daily_met: bool):
    plant = pb.MappedPlant(seed)
    plant.readParameters(str(xml_path))
    plant.setSeed(seed)
    setup_successor_where(plant)
    plant.initialize()

    met_lookup = get_daily_met(daily_met=None) if use_daily_met else None
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


def capture_signature(plant, case: Case) -> dict:
    organs = plant.getOrgans()
    stems = [o for o in organs if o.organType() == pb.OrganTypes.stem]
    leaves = [o for o in organs if o.organType() == pb.OrganTypes.leaf]
    roots = [o for o in organs if o.organType() == pb.OrganTypes.root]

    mainstem_bytes = b""
    leaf_bytes = b""
    root_bytes = b""
    stem_bytes_all = b""
    n_mainstem_nodes = 0
    n_leaf_nodes = 0
    n_root_nodes = 0
    n_segments_total = 0

    for s in sorted(stems, key=lambda o: o.getId()):
        nodes = list(s.getNodes())
        stem_bytes_all += _pack_xyz(nodes)
        if int(s.getParameter("subType")) == 1 and len(mainstem_bytes) == 0:
            mainstem_bytes = _pack_xyz(nodes)
            n_mainstem_nodes = len(nodes)
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
        "slug": case.slug,
        "xml": str(case.xml.relative_to(CPLANTBOX_ROOT)),
        "seed": SEED,
        "sim_days": case.days,
        "use_daily_met": case.use_daily_met,
        "n_stems": len(stems),
        "n_leaves": len(leaves),
        "n_roots": len(roots),
        "n_mainstem_nodes": n_mainstem_nodes,
        "n_leaf_nodes": n_leaf_nodes,
        "n_root_nodes": n_root_nodes,
        "n_segments_total": n_segments_total,
        "n_organs": len(organs),
        "sha256": h.hexdigest(),
    }


def run_case(case: Case) -> dict:
    print(f"[{case.slug}] xml={case.xml.name} days={case.days}", flush=True)
    plant = grow_deterministic(case.xml, case.days, SEED, case.use_daily_met)
    sig = capture_signature(plant, case)
    print(f"  sha256={sig['sha256']}  stems={sig['n_stems']} leaves={sig['n_leaves']} "
          f"roots={sig['n_roots']} nseg={sig['n_segments_total']}")
    return sig


def capture_mode():
    for case in CASES:
        sig = run_case(case)
        out = BASELINE_DIR / f"d0_{case.slug}.json"
        with out.open("w") as f:
            json.dump(sig, f, indent=2, sort_keys=True)
            f.write("\n")
        print(f"  -> {out.relative_to(CPLANTBOX_ROOT)}")


def verify_mode() -> int:
    failures = []
    for case in CASES:
        baseline_path = BASELINE_DIR / f"d0_{case.slug}.json"
        if not baseline_path.exists():
            print(f"[{case.slug}] MISSING baseline at {baseline_path}")
            failures.append(case.slug)
            continue
        with baseline_path.open() as f:
            baseline = json.load(f)
        sig = run_case(case)
        if sig["sha256"] != baseline["sha256"]:
            print(f"  DIFF: expected {baseline['sha256']} got {sig['sha256']}")
            failures.append(case.slug)
        else:
            print(f"  OK (matches baseline)")
    if failures:
        print(f"\nD.0 FAILED: {len(failures)} case(s) diverged: {failures}")
        return 1
    print("\nD.0 PASSED: all 6 cases bit-for-bit identical to baselines.")
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true",
                        help="re-run all cases and compare against stored baselines")
    args = parser.parse_args()
    if args.verify:
        sys.exit(verify_mode())
    else:
        capture_mode()


if __name__ == "__main__":
    main()
