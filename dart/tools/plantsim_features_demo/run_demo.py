"""PlantSimulation-learnings demo: baseline vs with-new-features at day 130.

Produces two OBJs so group names and triangle distribution can be compared:

- ``maize_day130_baseline.obj``  — all new flags default/off → bit-identical
  to the pre-session pipeline output (organ_N group names).
- ``maize_day130_features.obj`` — ``leaf_fracture=True`` +
  ``enable_senescent_split=True`` → fractured tips + ``senescent_leaf_N``
  groups on leaves past the senescence threshold.

Run:
    cd /home/lukas/PHD/CPlantBox
    source cpbenv/bin/activate
    PYTHONPATH=. python dart/coupling/output/plantsim_features_demo/run_demo.py
"""
from __future__ import annotations
from pathlib import Path
import sys

import plantbox as pb

from dart.coupling.config import DEFAULT_XML
from dart.coupling.geometry import extract_organs_for_lofter, loft_organs
from dart.coupling.growth.grow import setup_successor_where

DAY = 130
SEED = 7
OUT = Path(__file__).resolve().parent


def grow_to(day: int, daily_met: dict):
    plant = pb.MappedPlant()
    plant.readParameters(str(DEFAULT_XML))
    plant.setSeed(SEED)
    setup_successor_where(plant)
    plant.initialize()
    for day_1b in range(1, day + 1):
        T = float(daily_met.get(day_1b, {}).get("T_mean_C", 25.0))
        plant.setAirTemperature(T)
        plant.simulate(1.0, verbose=False)
    return plant


def export(plant, name: str, **extract_kwargs):
    organs = extract_organs_for_lofter(plant, skip_roots=True, **extract_kwargs)
    mesh = loft_organs(organs, stem_sides=8, use_nurbs_backend=True)
    out = OUT / f"{name}.obj"
    mesh.to_obj(str(out), group_by_organ=True)

    group_names = [m["name"] for m in mesh.organ_meta]
    n_senescent = sum(1 for n in group_names if n.startswith("senescent_leaf_"))
    n_healthy_leaves = sum(1 for n in group_names
                           if n.startswith("leaf_") and not n.startswith("senescent_leaf_"))
    n_stem = sum(1 for n in group_names if n.startswith("stem_"))
    n_tassel = sum(1 for n in group_names if n.startswith(("tassel_spike_", "tassel_branch_")))
    n_sheath = sum(1 for n in group_names if n.startswith("sheath_"))

    break_fracs = [o.get("break_fraction", 1.0) for o in organs if o.get("type") == "leaf"]
    n_broken = sum(1 for f in break_fracs if f < 1.0)

    print(f"  → {out.name}")
    print(f"    tris={mesh.n_triangles:6d}  organs={len(organs):3d}")
    print(f"    leaves healthy={n_healthy_leaves:2d}  senescent={n_senescent:2d}")
    print(f"    stem={n_stem}  sheath={n_sheath}  tassel={n_tassel}")
    print(f"    fractured leaves={n_broken} / {len(break_fracs)}  "
          f"(min break_fraction={min(break_fracs):.3f})")
    return mesh


def main():
    from dart.coupling.carbon.dvs_partitioning import get_daily_met
    daily_met = get_daily_met()
    if daily_met is None:
        raise SystemExit("No daily met CSV found; aborting.")

    print(f"[1/2] Growing to day {DAY} (seed={SEED}) …")
    plant = grow_to(DAY, daily_met)
    tt = plant.getAccumulatedTT()
    print(f"      plant TT = {tt:.1f} degCd")

    print("\n[2/2] Exporting OBJs …")
    print("  [A] baseline (all flags default/off)")
    export(plant, f"maize_day{DAY}_baseline")

    print("  [B] fracture + senescent split on")
    # Re-grow to avoid any state-mutation carryover
    plant2 = grow_to(DAY, daily_met)
    export(plant2, f"maize_day{DAY}_features",
           leaf_fracture={"enabled": True, "seed": 42},
           enable_senescent_split=True,
           senescent_rho_threshold=0.50)

    print(f"\nOutputs in {OUT}")


if __name__ == "__main__":
    sys.exit(main() or 0)
