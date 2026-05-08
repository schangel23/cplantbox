#!/usr/bin/env python3
"""Gate Ch1.PM.5 — D.0 6-XML invariance under PM wrap policy (FA-off subset).

Plan: ``Literature/Chapter 1/Concepts/CPBOXBALENOCOUPLING/
PLAN_PIAFMUNCH_CALIBRATION_2026-05-04.md`` Gate Ch1.PM.5.

Mirrors ``capture_d0_baselines.py`` but injects
``enable_cw_limited_growth(plant, wrap_roots=False, wrap_fa=True)``
between ``initialize()`` and the daily simulate loop. This is the
FA-off invariance check for the PM dispatch path: when
``solve_carbon_partitioning_pm`` runs on a non-FA XML, the wrap helper
fires; the Gate-5 fix in ``carbon_growth.py`` gates the bare-CW
overwrite to ``gf ∈ {gft_negexp, gft_CWLim}`` so existing
``LinearGrowth`` (gf=2) / ``GompertzGrowth`` (gf=4) organs are left
untouched. Empty ``CW_Gr`` then makes ``CWLimitedGrowth::getLength``
fall back to ``ExponentialGrowth`` (``growth.cpp:154``), which is
bit-identical to the original ``ExponentialGrowth`` dispatch for those
organs.

Scope — five FA-OFF XMLs from the D.0 matrix:
    wheat_calibrated_130d, brassica_oleracea_vansteenkiste_2014_60d,
    modelparam_4_30d, carbon2020_30d, legacy_2020_maize_60d.

The sixth case (``maize_calibrated_flagoff_130d``) is **excluded** here:
its slug is misleading. ``maize_calibrated.xml`` carries
``use_fournier_andrieu_kinetics=1`` baked in, so ``Plant::initCallbacks``
mints ``MultiPhaseStem/LeafGrowth`` regardless of the Python helper
state — it is FA-ON. The wrap-without-inject pattern this probe
exercises triggers ``CWLimitedGrowth::getLength``'s empty-``CW_Gr``
fallback to ``ExponentialGrowth``, bypassing the FA demand and
crashing mid-130-day simulate. That's an expected limitation of the
empty-``CW_Gr`` fallback in growth.cpp, not a Gate-5 leak: under real
PM dispatch (``--carbon-solver=pm``) every substep runs ``startPM`` →
``CW_Gr`` populated → ``simulate`` with FA preserved (Gate 4 §G3
bootstrap+inject parity).

Acceptance: 5/5 FA-off XMLs bit-identical to the corresponding
``d0_<slug>.json`` baseline produced by ``capture_d0_baselines.py``.

Usage::

    cpbenv/bin/python3 dart/coupling/tests/baselines/capture_d0_pm_invariance.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

BASELINE_DIR = Path(__file__).resolve().parent
COUPLING_DIR = BASELINE_DIR.parent.parent
CPLANTBOX_ROOT = COUPLING_DIR.parent.parent
sys.path.insert(0, str(CPLANTBOX_ROOT))

# Re-use the cases / hash logic from capture_d0_baselines so any future
# baseline change auto-propagates here.
from dart.coupling.tests.baselines.capture_d0_baselines import (  # noqa: E402
    CASES, SEED, T_AIR_DEFAULT, capture_signature,
)
from dart.coupling.carbon.dvs_partitioning import get_daily_met  # noqa: E402
from dart.coupling.growth.carbon_growth import (  # noqa: E402
    enable_cw_limited_growth,
)
from dart.coupling.growth.grow import setup_successor_where  # noqa: E402

import plantbox as pb  # noqa: E402


# FA-off subset — see module docstring for why maize_calibrated_flagoff_130d
# is excluded from Gate 5 (XML is FA-on under the hood; wrap-without-inject
# triggers an unrelated empty-CW_Gr fallback path covered by Gate 4 §G3).
GATE5_EXCLUDED_SLUGS = {"maize_calibrated_flagoff_130d"}


def grow_deterministic_with_wrap(xml_path: Path, sim_days: int, seed: int,
                                  use_daily_met: bool):
    """Same shape as capture_d0_baselines.grow_deterministic, but injects
    enable_cw_limited_growth(plant, wrap_roots=False, wrap_fa=True)
    after initialize()."""
    plant = pb.MappedPlant(seed)
    plant.readParameters(str(xml_path))
    plant.setSeed(seed)
    setup_successor_where(plant)
    plant.initialize()

    # Gate Ch1.PM.5 — wrap before the simulate loop. Mirrors what
    # solve_carbon_partitioning_pm would do on the first call (via
    # _is_cw_wrapped + enable_cw_limited_growth) when the diurnal CLI
    # runs with --carbon-solver=pm.
    enable_cw_limited_growth(plant, wrap_roots=False, wrap_fa=True)

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


def run_case(case) -> dict:
    print(f"[{case.slug}] xml={case.xml.name} days={case.days}", flush=True)
    plant = grow_deterministic_with_wrap(
        case.xml, case.days, SEED, case.use_daily_met)
    sig = capture_signature(plant, case)
    print(f"  sha256={sig['sha256']}  stems={sig['n_stems']} "
          f"leaves={sig['n_leaves']} roots={sig['n_roots']} "
          f"nseg={sig['n_segments_total']}")
    return sig


def main() -> int:
    failures = []
    skipped = []
    n_run = 0
    for case in CASES:
        if case.slug in GATE5_EXCLUDED_SLUGS:
            print(f"[{case.slug}] SKIPPED (FA-on under the hood; covered by "
                  f"Gate 4 §G3, not by wrap-without-inject)")
            skipped.append(case.slug)
            continue
        baseline_path = BASELINE_DIR / f"d0_{case.slug}.json"
        if not baseline_path.exists():
            print(f"[{case.slug}] MISSING baseline at {baseline_path}")
            failures.append(case.slug)
            continue
        with baseline_path.open() as f:
            baseline = json.load(f)
        sig = run_case(case)
        n_run += 1
        if sig["sha256"] != baseline["sha256"]:
            print(f"  DIFF: expected {baseline['sha256']} got "
                  f"{sig['sha256']}")
            failures.append(case.slug)
        else:
            print(f"  OK (matches baseline)")

    print()
    if failures:
        print(f"Gate Ch1.PM.5 FAILED: {len(failures)}/{n_run} FA-off case(s) "
              f"diverged under wrap policy: {failures}")
        if skipped:
            print(f"  Excluded (FA-on, see module docstring): {skipped}")
        return 1

    print(f"Gate Ch1.PM.5 PASSED ({n_run}/{n_run}): wrap policy bit-identical"
          f" on every FA-off case.")
    if skipped:
        print(f"  Excluded (FA-on, see module docstring): {skipped}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
