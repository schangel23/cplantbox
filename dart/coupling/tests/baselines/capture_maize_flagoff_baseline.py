#!/usr/bin/env python3
"""Capture pre-B.1 baseline hash for the Fournier-Andrieu Session 1 smoke test.

Runs maize_calibrated.xml for 60 days under Juelich 2024 daily met forcing,
seed=7, via MappedPlant(seed) (deterministic seeding — see note below). Hashes
(mainstem skeleton xyz + leaf node xyz + total segment count) into
baselines/maize_flagoff_60d_seed7.json.

After Layer B.1+B.2 land (with use_fournier_andrieu_kinetics defaulting false),
the re-captured hash must match byte-for-byte — proves the flag + new literal
defaults do not shift any existing behavior (Hard Invariant #5).

## Determinism note

CPlantBox has a latent issue in `Organism::setSeed(seed)`: it reseeds `gen` but
does NOT update `seed_val`. The tropism RNG in `tropism.cpp:98` seeds itself off
`plant->getSeedVal() + nodeIdx + o->getId()` — so after `setSeed(7)`, the
tropism RNG is still driven by the wall-clock-based `seed_val` set in the
constructor. To get deterministic baseline hashes, we seed via `MappedPlant(7)`
directly, which sets `seed_val=7` from construction. `grow_plant()` currently
uses `setSeed()` after construction and is therefore nondeterministic between
runs (same total node count, different per-node xyz). This baseline script
bypasses that path.

Usage (from /home/lukas/PHD/CPlantBox):
    cpbenv/bin/python3 dart/coupling/tests/baselines/capture_maize_flagoff_baseline.py
"""

from __future__ import annotations

import hashlib
import json
import struct
import sys
from pathlib import Path

COUPLING_DIR = Path(__file__).resolve().parent.parent.parent
REPO_ROOT = COUPLING_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT))

import plantbox as pb  # noqa: E402

from dart.coupling.carbon.dvs_partitioning import get_daily_met  # noqa: E402
from dart.coupling.growth.grow import setup_successor_where  # noqa: E402

XML_PATH = COUPLING_DIR / "data" / "maize_calibrated.xml"
OUT_PATH = Path(__file__).resolve().parent / "maize_flagoff_60d_seed7.json"

SEED = 7
SIM_DAYS = 60
T_AIR_DEFAULT = 25.0


def _pack_xyz(nodes) -> bytes:
    buf = bytearray()
    for n in nodes:
        buf += struct.pack("<ddd", float(n.x), float(n.y), float(n.z))
    return bytes(buf)


def grow_deterministic(xml_path: Path, sim_days: int, seed: int):
    """Deterministic grow — seeds via Plant ctor, mirrors grow.py otherwise."""
    plant = pb.MappedPlant(seed)         # <-- seed_val=7 locked from ctor
    plant.readParameters(str(xml_path))
    plant.setSeed(seed)                  # redundant but harmless; pins gen too
    setup_successor_where(plant)
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
    mainstem_bytes = b""
    leaf_bytes = b""
    n_segments_total = 0
    n_mainstem_nodes = 0
    n_leaf_nodes = 0

    stems = [o for o in organs if o.organType() == pb.OrganTypes.stem]
    leaves = [o for o in organs if o.organType() == pb.OrganTypes.leaf]
    mainstems = [s for s in stems if int(s.getParameter("subType")) == 1]
    if len(mainstems) != 1:
        print(f"WARNING: expected 1 mainstem, got {len(mainstems)}", file=sys.stderr)
    for s in mainstems:
        nodes = list(s.getNodes())
        mainstem_bytes += _pack_xyz(nodes)
        n_mainstem_nodes += len(nodes)
    for lf in sorted(leaves, key=lambda o: o.getId()):
        nodes = list(lf.getNodes())
        leaf_bytes += _pack_xyz(nodes)
        n_leaf_nodes += len(nodes)
    for o in organs:
        n_segments_total += max(0, len(o.getNodes()) - 1)

    h = hashlib.sha256()
    h.update(b"mainstem:")
    h.update(mainstem_bytes)
    h.update(b"|leaves:")
    h.update(leaf_bytes)
    h.update(b"|nseg:")
    h.update(struct.pack("<q", n_segments_total))
    return {
        "xml": str(XML_PATH.relative_to(REPO_ROOT)),
        "seed": SEED,
        "sim_days": SIM_DAYS,
        "n_mainstem_nodes": n_mainstem_nodes,
        "n_leaf_nodes": n_leaf_nodes,
        "n_segments_total": n_segments_total,
        "n_organs": len(organs),
        "sha256": h.hexdigest(),
    }


def main():
    print(f"Baseline capture: {XML_PATH.name}, seed={SEED}, days={SIM_DAYS}")
    plant = grow_deterministic(XML_PATH, SIM_DAYS, SEED)
    sig = capture_signature(plant)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w") as f:
        json.dump(sig, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"\nBaseline signature: {sig['sha256']}")
    print(f"  mainstem nodes: {sig['n_mainstem_nodes']}")
    print(f"  leaf nodes:     {sig['n_leaf_nodes']}")
    print(f"  segments:       {sig['n_segments_total']}")
    print(f"  organs:         {sig['n_organs']}")
    print(f"Saved to: {OUT_PATH}")


if __name__ == "__main__":
    main()
