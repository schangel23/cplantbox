#!/usr/bin/env python3
"""Capture a substep-resolved diel trace of the transient_reserve_pool_
and Σ local_C_pool_ across a 48-hour window.

Feeds ``fig_c_diel_reserve`` in ``ch1_figures.py``.  The trace is a JSON
list of one record per substep::

    [{"sim_day": 35, "substep": 0, "An_mmol": 0.123,
      "transient_reserve_pool_mmol": 1.42,
      "local_C_pool_total_mmol": 0.31,
      "is_light": true,
      "Fu_lim_estimate_mmol": 0.045}, ...]

The trace uses ``solve_carbon_partitioning_pm`` with ``advance_plant=False``
so plant geometry stays fixed across all 48 substeps (mirrors the per-hour
diagnostic mode in pm_substep, suitable for a day-2 plant snapshot).
The reserve / local-pool state is read directly off the plant pybind
between substeps.

Usage::

    cpbenv/bin/python dart/coupling/scripts/capture_diel_trace_2026-05-15.py \\
        --bootstrap-day 35 --hours 48 --soil-psi-cm -200 \\
        --out out_diel_trace.json

The default 48-hour window covers two full day-night cycles, allowing
``fig_c_diel_reserve`` to show the store-day / drain-night signature.
Because plant.simulate is NOT called between substeps, the trace is a
diagnostic-only probe; production runs continue to use the daily-batched
extension (Plan §4.3a).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
COUPLING_DIR = SCRIPT_DIR.parent
CPLANTBOX_ROOT = COUPLING_DIR.parent.parent
sys.path.insert(0, str(CPLANTBOX_ROOT))

import numpy as np  # noqa: E402

from dart.coupling.growth.grow import grow_plant  # noqa: E402
from dart.coupling.growth.carbon_growth import (  # noqa: E402
    enable_cw_limited_growth,
)
from dart.coupling.hydraulics.soil_psi import FixedSoilPsi  # noqa: E402
from dart.coupling.carbon.pm_substep import (  # noqa: E402
    solve_carbon_partitioning_pm,
)

MAIZE_XML = COUPLING_DIR / "data" / "maize_calibrated.xml"


def _local_pool_total(plant) -> float:
    return float(sum(max(0.0, float(getattr(o, "local_C_pool_", 0.0)))
                     for o in plant.getOrgans(-1, True)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bootstrap-day", type=int, default=35,
                    help="day to grow_plant to before sampling")
    ap.add_argument("--hours", type=int, default=48,
                    help="how many hourly substeps to capture")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--soil-psi-cm", type=float, default=-200.0,
                    help="static soil ψ for the probe")
    ap.add_argument("--krm1-multiplier", type=float, default=0.01)
    ap.add_argument("--kmfu-multiplier", type=float, default=0.1)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    print(f"Phase 1: grow_plant seed={args.seed} day_0..{args.bootstrap_day}")
    plant = grow_plant(
        xml_path=str(MAIZE_XML),
        simulation_time=args.bootstrap_day,
        seed=args.seed,
        enable_photosynthesis=True,
    )
    enable_cw_limited_growth(plant, wrap_roots=False, wrap_fa=True)

    n_leaf_segs = len(plant.getSegmentIds(4))
    if n_leaf_segs == 0:
        print(f"FAIL: no leaf segments at day {args.bootstrap_day}",
              file=sys.stderr)
        return 1
    An_per_leaf = np.full(n_leaf_segs, 0.002 / n_leaf_segs, dtype=float)
    provider = FixedSoilPsi(psi_cm=args.soil_psi_cm, n_cells=200)

    trace = []
    t0 = time.time()
    for hour in range(args.hours):
        sim_day = args.bootstrap_day + hour // 24
        substep = hour % 24
        # day-cycle approximation: substeps 6..18 = day (light), else
        # night.  PiafMunch's internal PAR cycling matches this when
        # n_substeps=24 and par_umol is the daytime constant.
        is_light = 6 <= substep <= 18
        # Run a single-substep PM solve to update the pools.
        result = solve_carbon_partitioning_pm(
            plant, An_per_leaf, Tair_C=25.0, day=int(sim_day),
            n_substeps=1, advance_plant=False,
            soil_psi_provider=provider, inject_an_target=False,
            krm1_multiplier=args.krm1_multiplier,
            kmfu_multiplier=args.kmfu_multiplier,
            use_buffered_carbon=True,
        )
        if result is None:
            print(f"  hour {hour}: PM bailed", file=sys.stderr)
            continue
        an_mmol = float(result.get("An_total_mmol", 0.0))
        reserve = float(getattr(plant, "transient_reserve_pool_", 0.0))
        local = _local_pool_total(plant)
        fu_lim_est = float(result.get("buffered_fu_delivered_mmol", 0.0))
        trace.append({
            "sim_day": int(sim_day),
            "substep": int(substep),
            "hour": hour,
            "An_mmol": an_mmol,
            "transient_reserve_pool_mmol": reserve,
            "local_C_pool_total_mmol": local,
            "Fu_lim_estimate_mmol": fu_lim_est,
            "is_light": bool(is_light),
        })
        if hour % 12 == 0:
            print(f"  hour {hour}: An={an_mmol:.4f} "
                  f"reserve={reserve:.4f} local={local:.4f} "
                  f"({time.time() - t0:.0f}s elapsed)")
    with args.out.open("w") as f:
        json.dump(trace, f, indent=2)
    print(f"  wrote {len(trace)} substep records → {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
