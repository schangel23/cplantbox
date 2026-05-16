#!/usr/bin/env python3
"""§S7 calibration sweep: XML-only carbon-buffer knobs vs Liebig closure +
realised-FA fraction at day-130 PM+DuMux.

Plan reference: PLAN_BUFFERED_CARBON_GROWTH_2026-05-15.md §S7.

Sweep is parametric and resumable. Each combo runs Phase-1 bootstrap →
Phase-2 carbon wrap → Phase-3 PM substep loop (driver mirrors
``run_g6_pm_dumux_fa_parity.py``), then writes a CSV row with:

  - the six tunable knobs (c_cost_leaf, c_cost_stem, c_cost_root,
    local_C_pool_capacity_factor, reserve_capacity_factor,
    starch_remob_rate, starch_storage_efficiency, starch_remob_efficiency)
  - per-seed mass-balance audit (cumulative + max-day-residual)
  - realised vs FA-oracle biomass ratio per organ_type and overall
  - PM convergence stats and runtime.

CSV columns are stable; rows are appended. A combo (knob tuple + seed) is
skipped if already present in the CSV (resumable mode).

Usage (cheap preliminary local scan, 30-day horizon, static soil):

  cpbenv/bin/python dart/coupling/scripts/calibrate_c_cost_per_cm_2026-05-15.py \\
      --c-cost-leaf 0.1 0.2 0.35 0.5 0.75 1.0 \\
      --seeds 7 \\
      --bootstrap-day 5 --sim-days 35 --soil-mode static \\
      --out-csv out_calibration_s7_smoke.csv

Server full sweep (day-130 + DuMux + multi-seed):

  cpbenv/bin/python dart/coupling/scripts/calibrate_c_cost_per_cm_2026-05-15.py \\
      --c-cost-leaf 0.20 0.35 0.50 0.75 \\
      --seeds 7 11 13 17 23 \\
      --bootstrap-day 30 --sim-days 130 --soil-mode dumux \\
      --krm1-multiplier 0.01 --kmfu-multiplier 0.1 \\
      --out-csv out_calibration_s7_day130.csv

The acceptance test ``test_s7_calibration.py`` reads the CSV and asserts
that at least one combo meets Liebig ≤1% AND realised-FA ∈ [0.4, 0.9].
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
COUPLING_DIR = SCRIPT_DIR.parent
CPLANTBOX_ROOT = COUPLING_DIR.parent.parent
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
)

ORACLE_PATH = (COUPLING_DIR / "tests" / "fixtures"
               / "oracle_fa_no_carbon_day130.json")
MAIZE_XML = COUPLING_DIR / "data" / "maize_calibrated.xml"

CSV_COLUMNS = [
    # knobs
    "c_cost_leaf", "c_cost_stem", "c_cost_root",
    "local_cap_factor", "local_cap_factor_root", "reserve_cap_factor",
    "starch_remob_rate", "starch_storage_eff", "starch_remob_eff",
    # config
    "seed", "bootstrap_day", "sim_days", "soil_mode", "soil_psi_cm",
    "krm1_mult", "kmfu_mult", "wrap_roots",
    # outcome
    "runtime_s", "n_pm_calls", "n_pm_fail",
    "mainstem_realised_cm", "mainstem_oracle_cm", "mainstem_fraction",
    "sum_leaf_realised_cm", "sum_leaf_oracle_cm", "leaf_fraction",
    "sum_root_realised_cm", "sum_root_oracle_cm", "root_fraction",
    "total_realised_cm", "total_oracle_cm", "realised_fa_fraction",
    "cum_an_mmol", "cum_used_mmol", "cum_mb_residual_pct",
    "max_day_mb_residual_pct", "mean_day_mb_residual_pct",
    "transient_reserve_end_mmol", "local_C_pool_total_end_mmol",
    "cum_rg_realised_mmol_co2",  # Σ realised dL × c_cost across the run
    "status", "error",
]


def _apply_knobs(plant, knobs: Dict[str, float]) -> None:
    """Override RP fields in-memory after grow_plant() loads the XML."""
    # Seed-level (organ type 1)
    for srp in plant.getOrganRandomParameter(1):
        if srp is None:
            continue
        srp.reserve_capacity_factor = knobs["reserve_cap_factor"]
        srp.starch_remob_rate = knobs["starch_remob_rate"]
        srp.starch_storage_efficiency = knobs["starch_storage_eff"]
        srp.starch_remob_efficiency = knobs["starch_remob_eff"]

    # Root (organ type 2)
    for rp in plant.getOrganRandomParameter(2):
        if rp is None:
            continue
        # Roots default cap_factor = 0.0 (dormant). The plan keeps roots
        # outside the buffered pool (§4.1 + §11), so we set c_cost but
        # leave cap_factor at the XML default unless the caller bumped it.
        rp.c_cost_per_cm = knobs["c_cost_root"]
        rp.local_C_pool_capacity_factor = knobs["local_cap_factor_root"]

    # Stem (organ type 3) and Leaf (organ type 4) — only subType >= 2 (real
    # organs; subType 0/1 are template/default rows).
    for ot, knob_key in [(3, "c_cost_stem"), (4, "c_cost_leaf")]:
        for rp in plant.getOrganRandomParameter(ot):
            if rp is None:
                continue
            if int(rp.subType) < 2:
                continue
            rp.c_cost_per_cm = knobs[knob_key]
            rp.local_C_pool_capacity_factor = knobs["local_cap_factor"]


def _summarise_realised(plant) -> Dict[str, float]:
    """Sum realised_length per organ_type from current plant state."""
    snap = per_organ_snapshot(plant)
    sums = {"mainstem": 0.0, "stem_other": 0.0, "leaf": 0.0, "root": 0.0,
            "n_leaf": 0, "n_stem": 0, "n_root": 0}
    for v in snap.values():
        ot = v["organ_type"]
        st = v["subType"]
        L = v["realised_length"]
        if ot == 3:
            sums["n_stem"] += 1
            if st == 1:
                sums["mainstem"] = max(sums["mainstem"], L)
            else:
                sums["stem_other"] += L
        elif ot == 4:
            sums["n_leaf"] += 1
            sums["leaf"] += L
        elif ot == 2:
            sums["n_root"] += 1
            sums["root"] += L
    return sums


def _summarise_oracle(oracle_path: Path) -> Dict[str, float]:
    with oracle_path.open() as f:
        oracle = json.load(f)
    sums = {"mainstem": 0.0, "stem_other": 0.0, "leaf": 0.0, "root": 0.0,
            "n_leaf": 0, "n_stem": 0, "n_root": 0}
    for v in oracle["organs"].values():
        ot = v["organ_type"]
        st = v["subType"]
        L = v["realised_length"]
        if ot == 3:
            sums["n_stem"] += 1
            if st == 1:
                sums["mainstem"] = max(sums["mainstem"], L)
            else:
                sums["stem_other"] += L
        elif ot == 4:
            sums["n_leaf"] += 1
            sums["leaf"] += L
        elif ot == 2:
            sums["n_root"] += 1
            sums["root"] += L
    return sums


def _make_provider(soil_mode: str, soil_psi_cm: float):
    from dart.coupling.hydraulics.soil_psi import make_provider
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
    raise ValueError(f"Unknown soil-mode {soil_mode!r}")


def _synth_an_per_leaf(plant) -> np.ndarray:
    n_leaf_segs = len(plant.getSegmentIds(4))
    if n_leaf_segs == 0:
        return np.array([], dtype=float)
    SYNTH_AN_PER_PLANT_MOL = 0.002  # representative V3 value
    return np.full(n_leaf_segs, SYNTH_AN_PER_PLANT_MOL / n_leaf_segs,
                   dtype=float)


def run_one_combo(knobs: Dict[str, float], seed: int, bootstrap_day: int,
                  sim_days: int, soil_mode: str, soil_psi_cm: float,
                  krm1_mult: Optional[float], kmfu_mult: Optional[float],
                  wrap_roots: bool = False,
                  verbose: bool = True) -> Dict[str, float]:
    """Execute one calibration combo and return CSV row dict."""
    t0 = time.time()
    row: Dict[str, float] = {
        "c_cost_leaf": knobs["c_cost_leaf"],
        "c_cost_stem": knobs["c_cost_stem"],
        "c_cost_root": knobs["c_cost_root"],
        "local_cap_factor": knobs["local_cap_factor"],
        "local_cap_factor_root": knobs["local_cap_factor_root"],
        "reserve_cap_factor": knobs["reserve_cap_factor"],
        "starch_remob_rate": knobs["starch_remob_rate"],
        "starch_storage_eff": knobs["starch_storage_eff"],
        "starch_remob_eff": knobs["starch_remob_eff"],
        "seed": seed,
        "bootstrap_day": bootstrap_day,
        "sim_days": sim_days,
        "soil_mode": soil_mode,
        "soil_psi_cm": soil_psi_cm,
        "krm1_mult": krm1_mult if krm1_mult is not None else "",
        "kmfu_mult": kmfu_mult if kmfu_mult is not None else "",
        "wrap_roots": int(bool(wrap_roots)),
        "status": "OK",
        "error": "",
    }

    try:
        if verbose:
            print(f"  Phase 1: grow_plant seed={seed} day_0..{bootstrap_day}",
                  flush=True)
        plant = grow_plant(
            xml_path=str(MAIZE_XML),
            simulation_time=bootstrap_day,
            seed=seed,
            enable_photosynthesis=True,
        )
        _apply_knobs(plant, knobs)
        enable_cw_limited_growth(plant, wrap_roots=wrap_roots, wrap_fa=True)

        provider = _make_provider(soil_mode, soil_psi_cm)
        met_lookup = get_daily_met(daily_met=None)

        n_pm = 0
        n_pm_fail = 0
        cum_an = 0.0
        cum_used = 0.0
        max_day_residual = 0.0
        sum_day_residual = 0.0
        # Plan §3.2 buffered-side accounting — Σ Rg_realised across the
        # run.  Computed as (post-day Σ length × c_cost) − (pre-day same)
        # so we can decompose Fu_lim into Rg_realised + Δlocal_C +
        # Δreserve + losses without trusting that PM-internal closure
        # implies buffered closure.
        cum_rg_realised_suc = 0.0  # mmol Suc, summed across days
        c_cost_by_ot = {2: knobs["c_cost_root"],
                        3: knobs["c_cost_stem"],
                        4: knobs["c_cost_leaf"]}

        def _length_cost_total(p) -> float:
            """Σ organ.getLength() × c_cost_per_cm[organ_type] [mmol Suc]."""
            tot = 0.0
            for o in p.getOrgans(-1, True):
                ot = int(o.organType())
                if ot in c_cost_by_ot:
                    tot += float(o.getLength()) * c_cost_by_ot[ot]
            return tot

        if verbose:
            print(f"  Phase 3: PM loop day {bootstrap_day+1}..{sim_days}",
                  flush=True)

        for sim_day in range(bootstrap_day + 1, sim_days + 1):
            T_air = 25.0
            if met_lookup is not None and sim_day in met_lookup:
                T_air = float(met_lookup[sim_day]["T_mean_C"])
            if hasattr(plant, "setAirTemperature"):
                plant.setAirTemperature(T_air)
            if hasattr(provider, "_t_last_days"):
                setattr(provider, "_t_last_days", float(sim_day - 1))

            An_seg = _synth_an_per_leaf(plant)
            if An_seg.size == 0:
                plant.simulate(1.0, False)
                continue

            length_cost_pre = _length_cost_total(plant)
            result = solve_carbon_partitioning_pm(
                plant, An_seg, Tair_C=T_air, day=int(sim_day - 1),
                n_substeps=24, advance_plant=True,
                soil_psi_provider=provider, inject_an_target=False,
                krm1_multiplier=krm1_mult, kmfu_multiplier=kmfu_mult,
                use_buffered_carbon=True,
            )
            n_pm += 1
            if result is None:
                n_pm_fail += 1
                try:
                    plant.simulate(0.0, False)
                except Exception:
                    pass
                continue

            # PM-internal Liebig closure (Plan §3.2 PM-side):
            #   An = Rm + Rg(=Fu_lim) + Exud + Mucil + ΔStorage_PM
            # All terms come back from pm_substep already converted to
            # mmol CO2 (Rm/Rg/stem_storage) or mmol Suc (root_exud +
            # dQ_Mucil); we convert Suc terms via SUC_TO_CO2 = 12.
            an = float(result.get("An_total_mmol", 0.0))
            rm = float(result.get("Rm_total_mmol", 0.0))
            rg = float(result.get("Rg_total_mmol", 0.0))
            stor_pm = float(result.get("stem_storage_mmol", 0.0))
            exud_suc = float(np.sum(result.get("root_exud_mmol_d",
                                               np.zeros(1))))
            mucil_suc = float(result.get("dQ_Mucil", 0.0))
            used = rm + rg + stor_pm + (exud_suc + mucil_suc) * 12.0
            cum_an += an
            cum_used += used
            day_residual = (abs(an - used) / an * 100.0) if an > 1e-9 else 0.0
            max_day_residual = max(max_day_residual, day_residual)
            sum_day_residual += day_residual

            # Buffered-side audit: Δ(Σ length × c_cost) = today's
            # Rg_realised in mmol Suc.  Captures the carbon that
            # actually went into structural extension (vs. sat in pools
            # or got remobilised through the reserve).
            length_cost_post = _length_cost_total(plant)
            cum_rg_realised_suc += max(0.0, length_cost_post - length_cost_pre)

            if verbose and (sim_day % 10 == 0 or sim_day == sim_days):
                print(f"    day {sim_day}: An={an:.3f} used={used:.3f} "
                      f"resid={day_residual:.2f}% PMfail={n_pm_fail}/{n_pm}",
                      flush=True)

        oracle_sums = _summarise_oracle(ORACLE_PATH)
        realised_sums = _summarise_realised(plant)

        # Mainstem fraction (ratio against the FA-oracle for the same
        # bootstrap+sim horizon — if sim_days < 130 the oracle is still
        # day-130 so the fraction is a relative anchor, not absolute).
        ms_oracle = oracle_sums["mainstem"]
        ms_realised = realised_sums["mainstem"]
        ms_frac = (ms_realised / ms_oracle) if ms_oracle > 1e-9 else 0.0

        leaf_oracle = oracle_sums["leaf"]
        leaf_realised = realised_sums["leaf"]
        leaf_frac = (leaf_realised / leaf_oracle) if leaf_oracle > 1e-9 else 0.0

        root_oracle = oracle_sums["root"]
        root_realised = realised_sums["root"]
        root_frac = (root_realised / root_oracle) if root_oracle > 1e-9 else 0.0

        total_oracle = (oracle_sums["mainstem"] + oracle_sums["stem_other"]
                        + oracle_sums["leaf"] + oracle_sums["root"])
        total_realised = (realised_sums["mainstem"] + realised_sums["stem_other"]
                          + realised_sums["leaf"] + realised_sums["root"])
        total_frac = (total_realised / total_oracle) if total_oracle > 1e-9 else 0.0

        cum_mb = (abs(cum_an - cum_used) / cum_an * 100.0) if cum_an > 1e-9 else 0.0
        mean_day_residual = (sum_day_residual / n_pm) if n_pm > 0 else 0.0

        row.update({
            "runtime_s": round(time.time() - t0, 1),
            "n_pm_calls": n_pm,
            "n_pm_fail": n_pm_fail,
            "mainstem_realised_cm": round(ms_realised, 4),
            "mainstem_oracle_cm": round(ms_oracle, 4),
            "mainstem_fraction": round(ms_frac, 4),
            "sum_leaf_realised_cm": round(leaf_realised, 4),
            "sum_leaf_oracle_cm": round(leaf_oracle, 4),
            "leaf_fraction": round(leaf_frac, 4),
            "sum_root_realised_cm": round(root_realised, 4),
            "sum_root_oracle_cm": round(root_oracle, 4),
            "root_fraction": round(root_frac, 4),
            "total_realised_cm": round(total_realised, 4),
            "total_oracle_cm": round(total_oracle, 4),
            "realised_fa_fraction": round(total_frac, 4),
            "cum_an_mmol": round(cum_an, 4),
            "cum_used_mmol": round(cum_used, 4),
            "cum_mb_residual_pct": round(cum_mb, 4),
            "max_day_mb_residual_pct": round(max_day_residual, 4),
            "mean_day_mb_residual_pct": round(mean_day_residual, 4),
            "transient_reserve_end_mmol": round(
                float(getattr(plant, "transient_reserve_pool_", 0.0)), 4),
            "local_C_pool_total_end_mmol": round(
                sum(max(0.0, float(getattr(o, "local_C_pool_", 0.0)))
                    for o in plant.getOrgans(-1, True)), 4),
            "cum_rg_realised_mmol_co2": round(cum_rg_realised_suc * 12.0, 4),
        })
    except Exception as exc:  # noqa: BLE001
        row["status"] = "ERROR"
        row["error"] = f"{type(exc).__name__}: {exc}"
        row["runtime_s"] = round(time.time() - t0, 1)
        if verbose:
            print(f"  ERROR: {row['error']}")
            traceback.print_exc()
    return row


def _existing_combos(csv_path: Path) -> set:
    if not csv_path.exists():
        return set()
    keys = set()
    with csv_path.open() as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            try:
                k = (
                    round(float(row["c_cost_leaf"]), 6),
                    round(float(row["c_cost_stem"]), 6),
                    round(float(row["c_cost_root"]), 6),
                    round(float(row["local_cap_factor"]), 6),
                    round(float(row.get("local_cap_factor_root", 0.0) or 0.0), 6),
                    round(float(row["reserve_cap_factor"]), 6),
                    round(float(row["starch_remob_rate"]), 6),
                    round(float(row["starch_storage_eff"]), 6),
                    round(float(row["starch_remob_eff"]), 6),
                    int(row["seed"]),
                    int(row["bootstrap_day"]),
                    int(row["sim_days"]),
                    row["soil_mode"],
                    float(row["soil_psi_cm"]),
                    row["krm1_mult"],
                    row["kmfu_mult"],
                    int(row.get("wrap_roots", 0) or 0),
                )
            except (KeyError, ValueError):
                continue
            keys.add(k)
    return keys


def _row_key(row: Dict[str, object], soil_mode: str, soil_psi_cm: float,
             krm1_mult: Optional[float], kmfu_mult: Optional[float],
             wrap_roots: bool) -> tuple:
    return (
        round(float(row["c_cost_leaf"]), 6),
        round(float(row["c_cost_stem"]), 6),
        round(float(row["c_cost_root"]), 6),
        round(float(row["local_cap_factor"]), 6),
        round(float(row.get("local_cap_factor_root", 0.0)), 6),
        round(float(row["reserve_cap_factor"]), 6),
        round(float(row["starch_remob_rate"]), 6),
        round(float(row["starch_storage_eff"]), 6),
        round(float(row["starch_remob_eff"]), 6),
        int(row["seed"]),
        int(row["bootstrap_day"]),
        int(row["sim_days"]),
        soil_mode,
        float(soil_psi_cm),
        "" if krm1_mult is None else str(krm1_mult),
        "" if kmfu_mult is None else str(kmfu_mult),
        int(bool(wrap_roots)),
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--c-cost-leaf", type=float, nargs="+",
                    default=[0.35])
    ap.add_argument("--c-cost-stem", type=float, nargs="+",
                    default=[0.55])
    ap.add_argument("--c-cost-root", type=float, nargs="+",
                    default=[0.20])
    ap.add_argument("--local-cap-factor", type=float, nargs="+",
                    default=[0.5])
    ap.add_argument("--local-cap-factor-root", type=float, nargs="+",
                    default=[0.0],
                    help="Root capacity factor; default 0 (dormant per §4.1)")
    ap.add_argument("--reserve-cap-factor", type=float, nargs="+",
                    default=[0.04])
    ap.add_argument("--starch-remob-rate", type=float, nargs="+",
                    default=[2.0])
    ap.add_argument("--starch-storage-eff", type=float, nargs="+",
                    default=[0.95])
    ap.add_argument("--starch-remob-eff", type=float, nargs="+",
                    default=[0.98])
    ap.add_argument("--seeds", type=int, nargs="+", default=[7])
    ap.add_argument("--bootstrap-day", type=int, default=30)
    ap.add_argument("--sim-days", type=int, default=130)
    ap.add_argument("--soil-mode", choices=("static", "dumux"),
                    default="dumux")
    ap.add_argument("--soil-psi-cm", type=float, default=-300.0)
    ap.add_argument("--krm1-multiplier", type=float, default=0.01,
                    help="Path B default = 0.01 (G6-fast 5/5 PM PASS).")
    ap.add_argument("--kmfu-multiplier", type=float, default=0.1,
                    help="Path B default = 0.1 (G6-fast 5/5 PM PASS).")
    ap.add_argument("--wrap-roots", action="store_true",
                    help=("Wrap root organs in CWLimitedGrowth so "
                          "c_cost_root + local_cap_factor_root actually "
                          "gate root growth. Default False preserves the "
                          "pre-§S10 native-FA-root path (c_cost_root is "
                          "structurally inert under that default). Required "
                          "to exercise §S10 root-budget falsification."))
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    csv_path = Path(args.out_csv)
    existing = _existing_combos(csv_path)

    if not csv_path.exists():
        with csv_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            w.writeheader()

    knob_grid = list(itertools.product(
        args.c_cost_leaf, args.c_cost_stem, args.c_cost_root,
        args.local_cap_factor, args.local_cap_factor_root,
        args.reserve_cap_factor, args.starch_remob_rate,
        args.starch_storage_eff, args.starch_remob_eff,
        args.seeds,
    ))
    print(f"§S7 sweep: {len(knob_grid)} combos total; "
          f"resume from {len(existing)} cached rows.")
    print(f"Output → {csv_path}")

    for i, (cl, cs, cr, cap, cap_root, rc, rr, se, re_, seed) in enumerate(
            knob_grid, 1):
        knobs = {
            "c_cost_leaf": cl, "c_cost_stem": cs, "c_cost_root": cr,
            "local_cap_factor": cap, "local_cap_factor_root": cap_root,
            "reserve_cap_factor": rc, "starch_remob_rate": rr,
            "starch_storage_eff": se, "starch_remob_eff": re_,
        }
        row_for_key = {
            "c_cost_leaf": cl, "c_cost_stem": cs, "c_cost_root": cr,
            "local_cap_factor": cap, "local_cap_factor_root": cap_root,
            "reserve_cap_factor": rc,
            "starch_remob_rate": rr, "starch_storage_eff": se,
            "starch_remob_eff": re_, "seed": seed,
            "bootstrap_day": args.bootstrap_day, "sim_days": args.sim_days,
        }
        key = _row_key(row_for_key, args.soil_mode, args.soil_psi_cm,
                       args.krm1_multiplier, args.kmfu_multiplier,
                       args.wrap_roots)
        if key in existing:
            if not args.quiet:
                print(f"[{i}/{len(knob_grid)}] SKIP cached "
                      f"cl={cl} seed={seed}")
            continue
        print(f"[{i}/{len(knob_grid)}] cl={cl} cs={cs} cr={cr} cap={cap} "
              f"rc={rc} rr={rr} seed={seed} "
              f"days={args.bootstrap_day}→{args.sim_days} {args.soil_mode}",
              flush=True)
        # Per-combo status sidecar — overwritten each iteration so the
        # latest state is always visible without tailing CSV.
        status_path = csv_path.with_suffix(".status.json")
        status_path.write_text(json.dumps({
            "combo_idx": i, "total_combos": len(knob_grid),
            "c_cost_leaf": cl, "c_cost_stem": cs, "c_cost_root": cr,
            "local_cap_factor": cap, "reserve_cap_factor": rc,
            "starch_remob_rate": rr, "seed": seed,
            "bootstrap_day": args.bootstrap_day,
            "sim_days": args.sim_days,
            "soil_mode": args.soil_mode,
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }, indent=2))
        row = run_one_combo(
            knobs, seed=seed, bootstrap_day=args.bootstrap_day,
            sim_days=args.sim_days, soil_mode=args.soil_mode,
            soil_psi_cm=args.soil_psi_cm,
            krm1_mult=args.krm1_multiplier,
            kmfu_mult=args.kmfu_multiplier,
            wrap_roots=args.wrap_roots,
            verbose=not args.quiet,
        )
        with csv_path.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            w.writerow({k: row.get(k, "") for k in CSV_COLUMNS})
        existing.add(key)

    print(f"§S7 sweep done; {len(existing)} rows in {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
