#!/usr/bin/env python3
"""§G6 acceptance gate: PM+DuMux real-substep FA reproduction at day 130.

Successor to ``run_g3_with_carbon_parity.py``. Same Phase-1+2 (FA-on
bootstrap to day-N, then `enable_cw_limited_growth`), but Phase-3 calls
the **real** ``solve_carbon_partitioning_pm`` substep loop once per day
instead of injecting synthetic BIG_SUPPLY via ``inject_cw_gr``.

Closes ``PLAN_PIAFMUNCH_DUMUX_COUPLING_2026-05-09 §G6``:

    Bootstrap V3 → day-30 with --soil-mode=static, then day-30 → day-130
    with --soil-mode=dumux (well-watered IC), reproduces the FA-on
    no-carbon oracle within <0.5% on mainstem length and <2% on per-leaf.

Why this test matters
---------------------
§G3 already proves Lock #6 + Lock #9 preserve the FA target shape when
supply ≫ demand (BIG_SUPPLY_CM = 100 cm/step). §G6 replaces that
synthetic abundance with the real PM-internal FvCB + phloem solver. Two
questions answered by one run:

  (a) Does the PM substep loop's constant-PAR 24-h FvCB (par_umol=600)
      produce *enough* daily An that FA-wrapped organs still hit their
      FA target across 100 simulated days? If yes → the G5.1 An↔Rm gap
      observed at single-day V3 doesn't accumulate into season-level
      geometry divergence (the gap is a transient under-supply that the
      plant grows out of as canopy area scales).
  (b) Does the DuMux 3D Richards path stay numerically stable across
      100 days of root-driven dry-down? If yes → the PMDM.5 mechanical
      wiring is production-quality.

A FAIL on this test is informative either way: it locates the gap (PM
under-supplying vs. FA target) and rules out the G5.1 single-day number
being a transient.

Usage (from /home/lukas/PHD/CPlantBox)::

    cpbenv/bin/python3 dart/coupling/tests/baselines/run_g6_pm_dumux_fa_parity.py
    cpbenv/bin/python3 dart/coupling/tests/baselines/run_g6_pm_dumux_fa_parity.py \\
        --bootstrap-day 30 --sim-days 60 --soil-mode static

Server (long full-day-130 DuMux run, ~3 h on nile)::

    cd /media/data/Lukas/CPlantBox && source cpbenv/bin/activate
    python3 dart/coupling/tests/baselines/run_g6_pm_dumux_fa_parity.py \\
        --bootstrap-day 30 --sim-days 130 --soil-mode dumux
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

BASELINE_DIR = Path(__file__).resolve().parent
COUPLING_DIR = BASELINE_DIR.parent.parent
CPLANTBOX_ROOT = COUPLING_DIR.parent.parent
FIXTURES_DIR = COUPLING_DIR / "tests" / "fixtures"
sys.path.insert(0, str(CPLANTBOX_ROOT))

import numpy as np  # noqa: E402

from dart.coupling.growth.grow import grow_plant  # noqa: E402
from dart.coupling.growth.carbon_growth import (  # noqa: E402
    enable_cw_limited_growth,
)
from dart.coupling.carbon.dvs_partitioning import get_daily_met  # noqa: E402
from dart.coupling.carbon.pm_substep import (  # noqa: E402
    solve_carbon_partitioning_pm,
)
from dart.coupling.tests.baselines._oracle_compare import (  # noqa: E402
    per_organ_snapshot,
    compare_against_oracle,
)

ORACLE_PATH = FIXTURES_DIR / "oracle_fa_no_carbon_day130.json"
SEED = 7
DEFAULT_SIM_DAYS = 130
DEFAULT_BOOTSTRAP_DAY = 30

# §G6 plan-doc tolerances (PLAN_PIAFMUNCH_DUMUX_COUPLING_2026-05-09 line 1438):
#   <0.5% drift on mainstem length
#   <2.0% drift on per-leaf lengths
TOL_LEAF_PCT = 2.0
TOL_MAINSTEM_CM = 0.5  # the spec says "<0.5%" — at ~200 cm mainstem this is ~1 cm,
                       # but we keep the absolute cm threshold from §G3 since the
                       # oracle stores mainstem realised_length, not top-z explicitly.

# Synthetic per-leaf-segment An placeholder for ``solve_carbon_partitioning_pm``.
# PM's internal constant-PAR FvCB drives the actual An; this number is only
# stored as ``An_total_mmol_target`` in the returned dict and (when
# inject_an_target=True) used to rescale Ag4Phloem. Default behaviour leaves
# this dormant — the §G6 contract is "PM internal FvCB → abundant supply →
# FA-target-binding growth".
SYNTH_AN_PER_PLANT_MOL = 0.002  # mol CO2/plant/day (representative V3 value)


def _make_provider(soil_mode: str, soil_psi_cm: float):
    """Construct one SoilPsiProvider for the multi-day loop.

    For ``dumux`` we use the same 1×1×100 column shape that the diurnal
    pipeline defaults to under `--soil-mode=dumux`, so the test exercises
    the production grid geometry.
    """
    from dart.coupling.hydraulics.soil_psi import make_provider
    # Match grow.py DEFAULT_SOIL_* (1×1×150) so the plant's seg→cell
    # mapping fits inside the provider's per-cell profile. A smaller
    # provider triggers ``vector::_M_range_check`` inside hm.solve.
    soil_mode = soil_mode.lower()
    if soil_mode == "static":
        return make_provider("fixed", soil_psi_cm=soil_psi_cm, n_cells=150)
    if soil_mode == "dumux":
        return make_provider(
            "dumux",
            soil_psi_cm=soil_psi_cm,
            min_b=(-50.0, -50.0, -150.0),
            max_b=(50.0, 50.0, 0.0),
            cell_number=(1, 1, 150),
        )
    raise ValueError(f"Unknown soil-mode {soil_mode!r}; use static or dumux")


def _synth_an_per_leaf(plant) -> np.ndarray:
    """Per-leaf-segment An vector (mol CO2/d/seg), uniform over emerged leaves."""
    n_leaf_segs = len(plant.getSegmentIds(4))
    if n_leaf_segs == 0:
        return np.array([], dtype=float)
    return np.full(
        n_leaf_segs, SYNTH_AN_PER_PLANT_MOL / n_leaf_segs, dtype=float,
    )


def grow_with_pm(
    bootstrap_day: int,
    sim_days: int,
    soil_mode: str,
    soil_psi_cm: float,
    inject_an_target: bool,
    krm1_multiplier: float = None,
):
    """Phase 1: FA-on no-carbon bootstrap. Phase 2: wrap. Phase 3: PM loop."""
    print(f"Phase 1: bootstrap to day {bootstrap_day} via grow_plant "
          f"(FA-on, no carbon, seed={SEED})")
    plant = grow_plant(
        xml_path=str(COUPLING_DIR / "data" / "maize_calibrated.xml"),
        simulation_time=bootstrap_day,
        seed=SEED,
        enable_photosynthesis=True,
    )

    print("Phase 2: switch to carbon-mode (Lock #9 wrap policy)")
    enable_cw_limited_growth(plant, wrap_roots=False, wrap_fa=True)
    n_wrapped_stem = sum(
        1 for p in plant.getOrganRandomParameter(3)
        if p is not None and getattr(p.f_gf, "demand", None) is not None
    )
    n_wrapped_leaf = sum(
        1 for p in plant.getOrganRandomParameter(4)
        if p is not None and getattr(p.f_gf, "demand", None) is not None
    )
    print(f"  wrapped {n_wrapped_stem} stem RPs + {n_wrapped_leaf} leaf RPs "
          f"with demand=FA")

    print(f"Phase 3: PM substep loop days {bootstrap_day+1}..{sim_days} "
          f"(soil_mode={soil_mode}, psi_init={soil_psi_cm} cm, "
          f"inject_an_target={inject_an_target})")
    provider = _make_provider(soil_mode, soil_psi_cm)
    met_lookup = get_daily_met(daily_met=None)

    t0 = time.time()
    n_pm_calls = 0
    n_pm_fail = 0

    for sim_day in range(bootstrap_day + 1, sim_days + 1):
        T_air = 25.0
        if met_lookup is not None and sim_day in met_lookup:
            T_air = float(met_lookup[sim_day]["T_mean_C"])
        if hasattr(plant, "setAirTemperature"):
            plant.setAirTemperature(T_air)

        # Align provider clock to current sim_day so DuMux advances one day
        # per PM call (24 substeps × dt=1/24). Static providers ignore this.
        if hasattr(provider, "_t_last_days"):
            setattr(provider, "_t_last_days", float(sim_day - 1))

        An_per_leaf_seg = _synth_an_per_leaf(plant)
        if An_per_leaf_seg.size == 0:
            # No emerged leaves yet — skip PM call. The plant has no
            # photosynthetic surface so there is nothing to partition;
            # advance plant geometry by 1 day under bare CWLim+FA
            # (will fall back to FA demand on the wrapped organs because
            # CW_Gr is empty).
            plant.simulate(1.0, False)
            continue

        result = solve_carbon_partitioning_pm(
            plant,
            An_per_leaf_seg,
            Tair_C=T_air,
            day=int(sim_day - 1),
            n_substeps=24,
            advance_plant=True,
            soil_psi_provider=provider,
            inject_an_target=inject_an_target,
            krm1_multiplier=krm1_multiplier,
        )
        n_pm_calls += 1
        if result is None:
            n_pm_fail += 1
            # Solver bailed; advance plant geometry anyway so the day
            # count keeps moving. Drift will surface in the comparator.
            try:
                plant.simulate(0.0, False)
            except Exception:
                pass

        if sim_day % 10 == 0 or sim_day == sim_days:
            elapsed = time.time() - t0
            all_organs = plant.getOrgans(-1, True)
            n_leaves_all = sum(1 for o in all_organs if int(o.organType()) == 4)
            n_leaves_emerged = sum(
                1 for o in all_organs
                if int(o.organType()) == 4 and o.getLength() > 0.01
            )
            mb_str = (f"mb={result['mass_balance_residual_pct']:.2f}%"
                      if result else "mb=FAIL")
            print(f"  day {sim_day}: ok ({elapsed:.0f}s), {mb_str}, "
                  f"organs={len(all_organs)}, "
                  f"leaves={n_leaves_emerged}/{n_leaves_all}, "
                  f"PM fails={n_pm_fail}/{n_pm_calls}")

    print(f"Phase 3 done in {time.time() - t0:.0f}s "
          f"({n_pm_calls} PM calls, {n_pm_fail} failures)")
    return plant


def main():
    parser = argparse.ArgumentParser(
        description=(__doc__ or "").split("\n")[0],
    )
    parser.add_argument("--bootstrap-day", type=int, default=DEFAULT_BOOTSTRAP_DAY,
                        help="day to grow_plant() before switching to PM dispatch")
    parser.add_argument("--sim-days", type=int, default=DEFAULT_SIM_DAYS,
                        help="final simulation day (oracle was captured at 130)")
    parser.add_argument("--soil-mode", choices=("static", "dumux"), default="dumux",
                        help="static = FixedSoilPsi; dumux = DumuxSoilPsi 1×1×100 column")
    parser.add_argument("--soil-psi-cm", type=float, default=-300.0,
                        help="initial soil ψ [cm]; default -300 matches G5 smoke")
    parser.add_argument("--inject-an-target", action="store_true",
                        help="rescale Ag4Phloem to daily-uniform synthetic An "
                             "target (closer to diurnal-realistic supply; "
                             "expected to fail oracle tolerance — diagnostic mode)")
    parser.add_argument("--krm1-multiplier", type=float, default=None,
                        help="scalar multiplier on WOFOST leaf Krm1 (0.030 "
                             "d⁻¹) applied to every PM substep via "
                             "hm.setKrm1([[0.030 * m]]). Default None leaves "
                             "JSON values untouched. Used for α-clip "
                             "diagnostic sweeps (e.g. 0.1 / 0.3) over the "
                             "G6-fast horizon.")
    parser.add_argument("--tol-leaf-pct", type=float, default=TOL_LEAF_PCT)
    parser.add_argument("--tol-mainstem-cm", type=float, default=TOL_MAINSTEM_CM)
    parser.add_argument("--skip-leaves-shorter-than-cm", type=float, default=0.0,
                        help="filter oracle leaves below this length (use >0 for "
                             "intermediate horizons day-60 etc. where late "
                             "emergents carry high relative noise)")
    args = parser.parse_args()

    if not ORACLE_PATH.exists():
        print(f"MISSING oracle at {ORACLE_PATH}")
        print("  Run capture_oracle_fa_no_carbon_day130.py first.")
        return 2

    plant = grow_with_pm(
        args.bootstrap_day,
        args.sim_days,
        args.soil_mode,
        args.soil_psi_cm,
        args.inject_an_target,
        krm1_multiplier=args.krm1_multiplier,
    )
    snap = per_organ_snapshot(plant)
    ok, lines = compare_against_oracle(
        snap,
        ORACLE_PATH,
        tol_leaf_pct=args.tol_leaf_pct,
        tol_mainstem_cm=args.tol_mainstem_cm,
        skip_leaves_shorter_than_cm=args.skip_leaves_shorter_than_cm,
    )

    print()
    print("=" * 78)
    print(f"§G6 PM+DuMux FA-parity check "
          f"(bootstrap={args.bootstrap_day}, days={args.sim_days}, "
          f"soil={args.soil_mode}@{args.soil_psi_cm}cm, "
          f"inject_an={args.inject_an_target})")
    print("=" * 78)
    for line in lines:
        print(line)
    print()
    if ok:
        print("§G6 PASS — PM+DuMux reproduces FA-no-carbon oracle within tolerance.")
        return 0
    print("§G6 FAIL — drift exceeds tolerance. Inspect the list above.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
