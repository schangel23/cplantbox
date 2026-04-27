"""Senescence-bend verification (Item 1, PLAN_GEOMETRY_FIDELITY_2026-04-22).

Covers the four R-stages of the two-segment bend:

- day 60  (TT~455 degCd, all ρ=0)  -- R1 baseline, no bend applied
- day 100 (TT~895 degCd, pos 0 ρ≈0.24) -- R2 onset, hook begins
- day 115 (TT~1090 degCd, pos 0 ρ≈0.49) -- R2 peak, tip curls up about hinge
- day 130 (TT~1273 degCd, pos 0 ρ≈0.72) -- R3, hook fading, tip below insertion

Full R4 (ρ=1.0) on pos 0 needs TT >= 1500 degCd (~day 160+ under Juelich met);
validated separately via the pure-skeleton smoke path (see adapter helpers).

Run:
    cd /home/lukas/PHD/CPlantBox
    source cpbenv/bin/activate
    PYTHONPATH=. python dart/coupling/output/blender_preview/_verify_senescence_droop.py
"""
from __future__ import annotations
from pathlib import Path
import sys

import plantbox as pb

from dart.coupling.config import DEFAULT_XML
from dart.coupling.geometry import extract_organs_for_lofter, loft_organs
from dart.coupling.growth.grow import setup_successor_where

DAYS = (60, 100, 115, 130)
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

    tt = plant.getAccumulatedTT()
    organs = extract_organs_for_lofter(plant, skip_roots=True)
    mesh = loft_organs(organs, stem_sides=8, use_nurbs_backend=True)

    out = STAGES / f"senescence_droop_day{day}.obj"
    out.parent.mkdir(parents=True, exist_ok=True)
    mesh.to_obj(str(out), group_by_organ=True)
    print(f"day {day:3d}  TT={tt:7.1f} degCd  -> {out}")
    return out


def main() -> None:
    from dart.coupling.carbon.dvs_partitioning import get_daily_met
    daily_met = get_daily_met()
    if daily_met is None:
        raise SystemExit("No daily met CSV found; aborting.")

    import traceback
    for d in DAYS:
        try:
            grow_and_export(d, daily_met)
        except BaseException as e:
            print(f"DAY {d} FAILED: {e!r}", flush=True)
            traceback.print_exc()


if __name__ == "__main__":
    sys.exit(main() or 0)
