#!/usr/bin/env python3
"""Fit CPlantBox XML parameters to a Pheno4D plant across all scan dates.

Phase 4 multi-date architecture-level driver. Unlike ``fit_to_reference.py``
(which compares against OBJ exports with skeleton Chamfer), this driver:

  * Loads canonical 11x5x3 CP grids from ``pheno4d_canonical_cps.json`` for
    one Pheno4D plant across every scan date available.
  * Runs CMA-ES over shared stem + leaf parameters.
  * Per sample, grows CPlantBox at every recorded scan day, lofts with the
    NURBS backend, and accumulates ``cp_l2_loss`` over the full
    (date x leaf) grid using ``hungarian_leaf_match`` per stage.
  * Outputs a fitted XML, per-date OBJ exports for eyeball validation,
    and a JSON of per-date per-leaf residuals.

Usage (smoke test):
    python3 dart/coupling/experimental/fit_xml_to_pheno4d.py \\
        --plant M01 --evals 50 \\
        --target-cps /home/lukas/PHD/Resources/Pheno4D/pheno4d_canonical_cps.json \\
        --output dart/coupling/experimental/output/m01_smoke/ -v
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date as _date, datetime
from pathlib import Path

import numpy as np
import cma

_COUPLING_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_COUPLING_DIR.parent))

import plantbox as pb

from dart.coupling.geometry.canonical_cp_grid import N_U, N_V
from dart.coupling.geometry.g1_to_g3 import loft_organs
from dart.coupling.geometry.cplantbox_adapter import extract_organs_for_lofter
from dart.coupling.growth.grow import setup_successor_where
from dart.coupling.experimental.losses.cp_distance import (
    cp_l2_loss, hungarian_leaf_match,
)
from dart.coupling.experimental.fit_to_reference import (
    load_pheno4d_cps_for_plant, ensure_xml_has_all_subtypes,
)

# Architecture-level shared parameters: one number per plant, applied to
# every leaf subtype. Keeps the CMA-ES search space tractable while still
# capturing the dominant growth/architecture modes.
PARAM_NAMES = [
    "stem_lmax", "stem_r", "stem_ln", "stem_lb",
    "leaf_lmax", "leaf_r", "leaf_theta", "leaf_tropismS",
]
DEFAULT_BOUNDS = [
    ("stem_lmax",     50.0, 250.0),
    ("stem_r",         0.5,  10.0),
    ("stem_ln",        4.0,  20.0),
    ("stem_lb",        1.0,  10.0),
    ("leaf_lmax",     20.0, 120.0),
    ("leaf_r",         0.5,   8.0),
    ("leaf_theta",     0.1,   1.4),
    ("leaf_tropismS",  0.001, 0.15),
]
DEFAULT_X0 = [150.0, 2.5, 14.0, 4.0, 60.0, 2.5, 0.7, 0.03]


# ---------------------------------------------------------------------------
# Date / day utilities
# ---------------------------------------------------------------------------

def parse_pheno4d_date(mmdd: str, year: int = 2017) -> _date:
    """Parse a ``'MMDD'`` string (Pheno4D's date key) to a ``date``.

    The year is metadata-level; only relative spacing matters for sim days.
    """
    mmdd = str(mmdd).zfill(4)
    return datetime.strptime(f"{year}{mmdd}", "%Y%m%d").date()


def dates_to_sim_days(dates: list[str], day_offset: float) -> dict[str, float]:
    """Convert Pheno4D date strings to CPlantBox simulation days.

    The earliest date in ``dates`` maps to ``day_offset``; later dates are
    offset by the day-delta from that first scan.
    """
    parsed = sorted((parse_pheno4d_date(d), d) for d in dates)
    first_date = parsed[0][0]
    return {d: day_offset + (dt - first_date).days for dt, d in parsed}


# ---------------------------------------------------------------------------
# CPlantBox driver — grow + loft with NURBS backend at one day
# ---------------------------------------------------------------------------

def _apply_shared_params(plant, params: dict) -> None:
    """Push shared (stem + leaf) params onto every relevant organ random-param."""
    sp = plant.getOrganRandomParameter(3, 1)
    sp.lmax = params["stem_lmax"]
    sp.r = params["stem_r"]
    sp.ln = params["stem_ln"]
    sp.lb = params["stem_lb"]

    # Apply shared leaf params to every subtype present in the XML.
    for st in range(2, 22):
        try:
            lp = plant.getOrganRandomParameter(4, st)
        except Exception:
            break
        lp.lmax = params["leaf_lmax"]
        lp.r = params["leaf_r"]
        lp.theta = params["leaf_theta"]
        lp.tropismS = params["leaf_tropismS"]


def grow_and_get_cps(xml_path: str, day: float, params: dict
                     ) -> tuple[dict[int, np.ndarray], object]:
    """Grow a plant to ``day`` and return ``{organ_id: CPs}`` + mesh."""
    plant = pb.Plant()
    plant.readParameters(str(xml_path))
    _apply_shared_params(plant, params)
    setup_successor_where(plant)
    plant.initialize(False)
    plant.simulate(day)

    organs = extract_organs_for_lofter(plant)
    mesh = loft_organs(organs, use_nurbs_backend=True,
                       subdivide=False, smooth=False)

    pred_cps = {int(oid): np.asarray(cps, dtype=np.float64)
                for oid, cps in (mesh.organ_cps or {}).items()
                if np.asarray(cps).shape == (N_U, N_V, 3)}
    return pred_cps, mesh


# ---------------------------------------------------------------------------
# Multi-date CP objective
# ---------------------------------------------------------------------------

def evaluate_multi_date(params_vec: np.ndarray,
                         bounds: list[tuple[str, float, float]],
                         xml_path: str,
                         target_cps_by_day: dict[float, dict[int, np.ndarray]],
                         penalty: float = 1e4) -> float:
    """Sum of per-stage mean CP-L2 losses across every Pheno4D scan date."""
    params = {name: float(np.clip(params_vec[i], lo, hi))
              for i, (name, lo, hi) in enumerate(bounds)}

    total = 0.0
    n_stages = 0
    for day, target_cps in target_cps_by_day.items():
        try:
            pred_cps, _mesh = grow_and_get_cps(xml_path, day, params)
        except Exception:
            total += penalty
            n_stages += 1
            continue

        if not pred_cps or not target_cps:
            total += penalty
            n_stages += 1
            continue

        pred_keys_sorted = sorted(pred_cps.keys())
        pred_ranks = {oid: r for r, oid in enumerate(pred_keys_sorted)}
        target_ranks = {lab: int(lab) for lab in target_cps}

        match = hungarian_leaf_match(
            pred_cps, target_cps,
            weight_xyz=1.0, weight_arc=0.5, weight_rank=0.5,
            pred_ranks=pred_ranks, target_ranks=target_ranks,
        )
        stage_loss = cp_l2_loss(pred_cps, target_cps, match, reduction="mean")
        total += stage_loss
        n_stages += 1

    # Average across stages so the objective does not trivially scale with
    # scan count.
    return total / max(n_stages, 1)


# ---------------------------------------------------------------------------
# Post-fit export
# ---------------------------------------------------------------------------

def export_obj(mesh, out_path: Path, header: str) -> None:
    """Write a minimal OBJ file (vertices + triangle faces)."""
    with open(out_path, "w") as f:
        f.write(f"# {header}\n")
        for v in mesh.vertices:
            f.write(f"v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}\n")
        for tri in mesh.indices:
            f.write(f"f {tri[0] + 1} {tri[1] + 1} {tri[2] + 1}\n")


def analyze_final(xml_path: str, params: dict,
                   target_cps_by_day: dict[float, dict[int, np.ndarray]],
                   date_by_day: dict[float, str],
                   output_dir: Path) -> dict:
    """Run the fitted params at every scan date; export OBJs + per-leaf residuals."""
    report = {"per_date": [], "mean_cp_l2": None}
    stage_losses = []

    objs_dir = output_dir / "per_date_objs"
    objs_dir.mkdir(parents=True, exist_ok=True)

    for day in sorted(target_cps_by_day.keys()):
        date_str = date_by_day[day]
        try:
            pred_cps, mesh = grow_and_get_cps(xml_path, day, params)
        except Exception as exc:
            report["per_date"].append({
                "date": date_str, "day": day, "error": str(exc),
            })
            continue

        target = target_cps_by_day[day]
        pred_keys_sorted = sorted(pred_cps.keys())
        pred_ranks = {oid: r for r, oid in enumerate(pred_keys_sorted)}
        target_ranks = {lab: int(lab) for lab in target}

        match = hungarian_leaf_match(
            pred_cps, target,
            weight_xyz=1.0, weight_arc=0.5, weight_rank=0.5,
            pred_ranks=pred_ranks, target_ranks=target_ranks,
        )
        stage_loss = cp_l2_loss(pred_cps, target, match, reduction="mean")
        stage_losses.append(stage_loss)

        per_leaf = []
        for p_oid, t_label in match:
            pc = pred_cps[p_oid]  # type: ignore[index]
            tc = target[t_label]  # type: ignore[index]
            diff = pc - tc
            per_leaf.append({
                "pred_organ_id": int(p_oid),  # type: ignore[arg-type]
                "target_label": int(t_label),  # type: ignore[arg-type]
                "ssd": float(np.sum(diff * diff)),
                "rmse_cm": float(np.sqrt(np.mean(diff * diff))),
            })

        report["per_date"].append({
            "date": date_str, "day": day,
            "mean_cp_l2": stage_loss,
            "n_pred_leaves": len(pred_cps),
            "n_target_leaves": len(target),
            "n_matched": len(match),
            "per_leaf": per_leaf,
        })

        obj_path = objs_dir / f"fitted_{date_str}_day{int(day):03d}.obj"
        export_obj(mesh, obj_path, f"Pheno4D {date_str} day {day:.1f}")

    if stage_losses:
        report["mean_cp_l2"] = float(np.mean(stage_losses))
    return report


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit CPlantBox XML to a Pheno4D plant via multi-date CP-L2.")
    parser.add_argument("--plant", required=True,
                        help="Pheno4D plant id (e.g. M01).")
    parser.add_argument("--target-cps",
                        default="/home/lukas/PHD/Resources/Pheno4D/pheno4d_canonical_cps.json",
                        help="Path to pheno4d_canonical_cps.json.")
    parser.add_argument("--xml", default=None,
                        help="Template XML (default: data/maize_calibrated.xml).")
    parser.add_argument("--output", "-o", required=True,
                        help="Output directory.")
    parser.add_argument("--evals", type=int, default=50,
                        help="CMA-ES max evaluations (default: 50).")
    parser.add_argument("--day-offset", type=float, default=20.0,
                        help="Simulation day corresponding to the earliest "
                             "scan date (default: 20).")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    xml_path = args.xml or str(_COUPLING_DIR / "data" / "maize_calibrated.xml")

    print("=" * 70)
    print(f"FIT CPLANTBOX → Pheno4D plant {args.plant}")
    print("=" * 70)

    # 1) Load target CPs per scan date.
    print("\n[1/4] Loading Pheno4D canonical CPs...")
    cps_by_date = load_pheno4d_cps_for_plant(args.target_cps, args.plant)
    if not cps_by_date:
        print(f"  ERROR: no scans for plant {args.plant!r} in {args.target_cps}")
        sys.exit(1)
    print(f"  Found {len(cps_by_date)} scans: {sorted(cps_by_date.keys())}")

    # 2) Map dates → simulation days.
    day_by_date = dates_to_sim_days(list(cps_by_date.keys()), args.day_offset)
    date_by_day = {d: date_str for date_str, d in day_by_date.items()}
    target_cps_by_day = {
        day_by_date[date_str]: cps
        for date_str, cps in cps_by_date.items()
    }
    print(f"  Day-offset {args.day_offset} → simulation days "
          f"{sorted(target_cps_by_day.keys())}")

    # 3) Prepare XML — ensure enough leaf subtypes for the max observed rank.
    max_label = max(int(lab) for scan in cps_by_date.values() for lab in scan)
    # Labels in canonical_cps json are collar-index labels; treat them as
    # 1-indexed positions. Position p needs subType p+1 under CPlantBox.
    positions = list(range(max_label))  # 0..max_label-1 covers labels 1..max_label
    prep_xml = output_dir / "template_prepared.xml"
    ensure_xml_has_all_subtypes(xml_path, str(prep_xml), positions)
    xml_path = str(prep_xml)

    # 4) CMA-ES over shared params, objective = mean CP-L2 across dates.
    bounds = list(DEFAULT_BOUNDS)
    lo = [b[1] for b in bounds]
    hi = [b[2] for b in bounds]
    ranges = [h - l for l, h in zip(lo, hi)]
    x0 = DEFAULT_X0
    x0_scaled = [float(np.clip((v - lo[i]) / ranges[i], 0.02, 0.98))
                 for i, v in enumerate(x0)]

    def objective(x_scaled: np.ndarray) -> float:
        x_real = np.array([lo[i] + x_scaled[i] * ranges[i]
                            for i in range(len(x_scaled))], dtype=np.float64)
        return evaluate_multi_date(x_real, bounds, xml_path,
                                     target_cps_by_day)

    popsize = max(8, 2 * len(bounds))
    es = cma.CMAEvolutionStrategy(x0_scaled, 0.2, {
        "bounds": [0, 1],
        "maxfevals": args.evals,
        "verbose": -9,
        "seed": 42,
        "popsize": popsize,
    })

    print(f"\n[2/4] CMA-ES: popsize={popsize}, maxfevals={args.evals}")
    t0 = time.time()
    gen = 0
    best_fitness = float("inf")
    while not es.stop():
        sols = es.ask()
        fits = [objective(np.array(s)) for s in sols]
        es.tell(sols, fits)
        gen += 1
        if es.result.fbest < best_fitness:
            best_fitness = es.result.fbest
            if args.verbose:
                print(f"  Gen {gen}: best mean CP-L2 = {best_fitness:.3f}")
    elapsed = time.time() - t0

    # Decode best solution.
    best_scaled = es.result.xbest
    best_real = [float(lo[i] + best_scaled[i] * ranges[i])
                 for i in range(len(best_scaled))]
    best_params = {name: best_real[i] for i, (name, _, _) in enumerate(bounds)}

    print(f"\n  CMA-ES finished in {elapsed:.1f}s ({gen} generations)")
    print(f"  Best mean CP-L2: {best_fitness:.3f}")
    print("  Best params:")
    for name, v in best_params.items():
        print(f"    {name:>16s} = {v:.4f}")

    (output_dir / "fitted_params.json").write_text(
        json.dumps({
            "plant": args.plant,
            "best_fitness_mean_cp_l2": best_fitness,
            "generations": gen,
            "evals": es.result.evaluations,
            "params": best_params,
            "day_offset": args.day_offset,
            "dates": sorted(cps_by_date.keys()),
        }, indent=2))

    # 5) Final analysis: run fitted params at every date, export OBJs.
    print(f"\n[3/4] Final analysis + per-date OBJ export...")
    report = analyze_final(xml_path, best_params, target_cps_by_day,
                            date_by_day, output_dir)
    (output_dir / "fit_report.json").write_text(
        json.dumps(report, indent=2))

    print(f"\n[4/4] Wrote:")
    print(f"  Params:  {output_dir / 'fitted_params.json'}")
    print(f"  Report:  {output_dir / 'fit_report.json'}")
    print(f"  OBJs:    {output_dir / 'per_date_objs'}/")
    if report["mean_cp_l2"] is not None:
        print(f"\n  Mean CP-L2 at final params: {report['mean_cp_l2']:.3f}")


if __name__ == "__main__":
    main()
