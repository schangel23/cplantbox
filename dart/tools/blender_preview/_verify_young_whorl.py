"""Young-whorl verification (Item 4 partial ship, 2026-04-22).

Exports side-view OBJs at four ages covering V3 -> mature:
- day 15 (V3 target): stacked collars, sheath wraps stem, blades 0-2 post-
  collar splay, blade 3 just peeking from whorl apex.
- day 25 (V5-V6): sheath column still visible, bottom 3 leaves splayed,
  top 2-3 still whorled.
- day 55 (pre-VT): mature silhouette, regression sanity-check.
- day 130 (R3): mature senescence sanity-check (should be unchanged from
  current master's senescence ship at the adapter level).

Run:
    cd /home/lukas/PHD/CPlantBox
    source cpbenv/bin/activate
    PYTHONPATH=. python dart/coupling/output/blender_preview/_verify_young_whorl.py
"""
from __future__ import annotations
from pathlib import Path
import sys
import traceback

import plantbox as pb

from dart.coupling.config import DEFAULT_XML
from dart.coupling.geometry import extract_organs_for_lofter, loft_organs
from dart.coupling.growth.grow import setup_successor_where

DAYS = (15, 20, 25, 55, 130)
SEED = 7
STAGES = Path(__file__).resolve().parent / "stages"


def grow_and_export(day: int, daily_met: dict) -> Path:
    plant = pb.MappedPlant()
    plant.readParameters(str(DEFAULT_XML))
    plant.setSeed(SEED)
    setup_successor_where(plant)
    plant.initialize()

    for day_1b in range(1, day + 1):
        T = float(daily_met.get(day_1b, {}).get("T_mean_C", 25.0))
        plant.setAirTemperature(T)
        plant.simulate(1.0, verbose=False)

    tt = plant.getAccumulatedTT() if hasattr(plant, "getAccumulatedTT") else -1.0
    organs = extract_organs_for_lofter(plant, skip_roots=True)
    n_leaves = sum(1 for o in organs if o.get("type") == "leaf")
    mesh = loft_organs(organs, stem_sides=8, use_nurbs_backend=True)

    out = STAGES / f"young_whorl_day{day:03d}.obj"
    out.parent.mkdir(parents=True, exist_ok=True)
    mesh.to_obj(str(out), group_by_organ=True)
    print(f"day {day:3d}  TT={tt:7.1f} degCd  leaves={n_leaves}  -> {out}")
    return out


def main() -> None:
    from dart.coupling.carbon.dvs_partitioning import get_daily_met
    daily_met = get_daily_met()
    if daily_met is None:
        raise SystemExit("No daily met CSV found; aborting.")

    for d in DAYS:
        try:
            grow_and_export(d, daily_met)
        except BaseException as e:
            print(f"DAY {d} FAILED: {e!r}", flush=True)
            traceback.print_exc()


if __name__ == "__main__":
    sys.exit(main() or 0)
