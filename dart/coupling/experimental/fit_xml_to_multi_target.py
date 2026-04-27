#!/usr/bin/env python3
"""Fused CPlantBox XML fitter: Pheno4D trajectory + MaizeField3D mature shape.

Extends `fit_xml_to_pheno4d.py` with two additions:

  1. A second loss term from MaizeField3D `per_position.median_cps_cm`
     (mature leaf-shape prior, collar-subtracted so it is shape-only).
  2. A `apply_params_to_xml()` writer that emits a ready-to-grow calibrated
     XML alongside the JSON param dump.

Both target datasets live in the canonical 11x5x3 CP grid, so the fused
loss is two sums of squared CP differences with separate weights. Pheno4D
drives architecture dynamics (stem growth, leaf emergence, rank-specific
insertion) across scan dates; MaizeField3D anchors what a mature leaf at
each rank should look like.

Usage (smoke):
    python3 dart/coupling/experimental/fit_xml_to_multi_target.py \\
        --plant M01 --evals 50 \\
        --pheno4d-cps /home/lukas/PHD/Resources/Pheno4D/pheno4d_canonical_cps.json \\
        --maizefield3d-cps /home/lukas/PHD/Resources/MaizeField3d/maizefield3d_canonical_cps.json \\
        --output dart/coupling/experimental/output/m01_fused_smoke/ -v
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import cma

_COUPLING_DIR = Path(__file__).resolve().parent.parent
# Add CPlantBox repo root to sys.path so `dart.coupling.*` resolves when the
# script is invoked directly (not via `python -m ...`).
sys.path.insert(0, str(_COUPLING_DIR.parent.parent))

from dart.coupling.geometry.canonical_cp_grid import N_U, N_V  # noqa: E402
from dart.coupling.experimental.losses.cp_distance import (  # noqa: E402
    cp_l2_loss, hungarian_leaf_match,
)
from dart.coupling.experimental.fit_to_reference import (  # noqa: E402
    load_pheno4d_cps_for_plant, ensure_xml_has_all_subtypes,
)
from dart.coupling.experimental.fit_xml_to_pheno4d import (  # noqa: E402
    DEFAULT_BOUNDS, DEFAULT_X0,
    dates_to_sim_days, grow_and_get_cps, export_obj,
)


# ---------------------------------------------------------------------------
# XML writer: apply 8 fitted params to a template XML and save
# ---------------------------------------------------------------------------

STEM_SHARED = {"stem_lmax": "lmax", "stem_r": "r",
               "stem_ln": "ln", "stem_lb": "lb"}
LEAF_SHARED = {"leaf_lmax": "lmax", "leaf_r": "r",
               "leaf_theta": "theta", "leaf_tropismS": "tropismS"}


def apply_params_to_xml(template_xml: str, out_xml: str,
                         params: dict) -> None:
    """Read ``template_xml``, apply the 8 shared params, write ``out_xml``.

    Stem ``lmax/r/ln/lb`` are written to stem subType=1. Leaf
    ``lmax/r/theta/tropismS`` are written to every leaf subType present
    in the XML (2..N). Stem ``la`` is recomputed as
    ``max(0.1, lmax - lb - (n_leaves-1)*ln)`` so internode count still
    matches leaf count after geometry changes.
    """
    tree = ET.parse(template_xml)
    root = tree.getroot()

    # ---- Stem: subType=1 ----
    stem = root.find("stem[@subType='1']")
    if stem is None:
        raise ValueError(f"No stem[@subType='1'] in {template_xml}")
    for pkey, xml_name in STEM_SHARED.items():
        el = stem.find(f"parameter[@name='{xml_name}']")
        if el is None:
            raise ValueError(
                f"stem subType=1 is missing <parameter name='{xml_name}'/>")
        el.set("value", f"{params[pkey]}")

    # Recompute stem la so total stem length covers all leaf insertions.
    leaf_elems = root.findall("leaf")
    n_leaves = len(leaf_elems)
    lmax = params["stem_lmax"]
    lb = params["stem_lb"]
    ln = params["stem_ln"]
    la_val = max(0.1, lmax - lb - max(0, n_leaves - 1) * ln)
    la_el = stem.find("parameter[@name='la']")
    if la_el is not None:
        la_el.set("value", f"{la_val}")

    # ---- Every leaf subType ----
    for leaf in leaf_elems:
        for pkey, xml_name in LEAF_SHARED.items():
            el = leaf.find(f"parameter[@name='{xml_name}']")
            if el is None:
                # Some XMLs may not have tropismS on every subtype; add it.
                el = ET.SubElement(leaf, "parameter")
                el.set("name", xml_name)
            el.set("value", f"{params[pkey]}")

    tree.write(out_xml, encoding="utf-8", xml_declaration=True)


# ---------------------------------------------------------------------------
# MaizeField3D loader — per-position mature shape prior
# ---------------------------------------------------------------------------

def load_maizefield3d_mature_cps(json_path: str
                                   ) -> dict[int, np.ndarray]:
    """Load MaizeField3D per-position median CPs, collar-subtracted.

    Returns ``{position_0_based: (N_U, N_V, 3) cps_relative_to_collar_midrib}``.
    Collar = cps[0, v_mid, :] so xyz origin is at the blade's proximal
    midrib node. This yields a pure shape signal (orientation + extent
    relative to collar) that can be meaningfully compared against
    CPlantBox-lofted leaves after the same collar subtraction.
    """
    with open(json_path) as f:
        d = json.load(f)
    if d.get("n_u") != N_U or d.get("n_v") != N_V:
        raise ValueError(
            f"MaizeField3D CP file has n_u={d.get('n_u')}, n_v={d.get('n_v')}; "
            f"canonical expected ({N_U}, {N_V})")
    v_mid = N_V // 2
    out: dict[int, np.ndarray] = {}
    for entry in d["per_position"]:
        pos = int(entry["position"])
        cps = np.asarray(entry["median_cps_cm"], dtype=np.float64)
        if cps.shape != (N_U, N_V, 3):
            continue
        collar = cps[0, v_mid, :]
        out[pos] = cps - collar  # broadcast over (u, v, 3)
    return out


def collar_subtract(cps: np.ndarray) -> np.ndarray:
    """Subtract collar-midrib CP from the whole grid; pure shape."""
    v_mid = cps.shape[1] // 2
    return cps - cps[0, v_mid, :]


# ---------------------------------------------------------------------------
# Fused objective
# ---------------------------------------------------------------------------

def evaluate_fused(params_vec: np.ndarray,
                    bounds: list[tuple[str, float, float]],
                    xml_path: str,
                    target_pheno_by_day: dict[float, dict[int, np.ndarray]],
                    target_mf3d_by_position: dict[int, np.ndarray],
                    mature_day: float,
                    w_pheno: float, w_mature: float,
                    penalty: float = 1e4) -> tuple[float, dict]:
    """Combined CP-L2 objective: Pheno4D trajectory + MaizeField3D mature.

    Returns (fitness, diagnostics) where diagnostics carries per-stage
    loss values so the outer loop can log them without rerunning.
    """
    params = {name: float(np.clip(params_vec[i], lo, hi))
              for i, (name, lo, hi) in enumerate(bounds)}

    diag = {"pheno_stages": [], "mature": None}

    # --- Pheno4D trajectory term ---
    pheno_losses: list[float] = []
    for day, target_cps in target_pheno_by_day.items():
        try:
            pred_cps, _mesh = grow_and_get_cps(xml_path, day, params)
        except Exception:
            pheno_losses.append(penalty)
            diag["pheno_stages"].append(
                {"day": day, "loss": penalty, "error": "grow failed"})
            continue
        if not pred_cps or not target_cps:
            pheno_losses.append(penalty)
            diag["pheno_stages"].append(
                {"day": day, "loss": penalty, "error": "no CPs"})
            continue
        pred_keys_sorted = sorted(pred_cps.keys())
        pred_ranks = {oid: r for r, oid in enumerate(pred_keys_sorted)}
        target_ranks = {lab: int(lab) - 1 for lab in target_cps}  # 1-based → 0
        match = hungarian_leaf_match(
            pred_cps, target_cps,
            weight_xyz=1.0, weight_arc=0.5, weight_rank=0.5,
            pred_ranks=pred_ranks, target_ranks=target_ranks,
        )
        L = cp_l2_loss(pred_cps, target_cps, match, reduction="mean")
        pheno_losses.append(L)
        diag["pheno_stages"].append({"day": day, "loss": L, "n_match": len(match)})
    mean_pheno = float(np.mean(pheno_losses)) if pheno_losses else penalty

    # --- MaizeField3D mature shape term (collar-subtracted) ---
    try:
        pred_mature, _mesh_mature = grow_and_get_cps(
            xml_path, mature_day, params)
    except Exception:
        diag["mature"] = {"loss": penalty, "error": "grow failed"}
        mature_loss = penalty
    else:
        if not pred_mature:
            diag["mature"] = {"loss": penalty, "error": "no CPs"}
            mature_loss = penalty
        else:
            pred_shape = {oid: collar_subtract(cps)
                          for oid, cps in pred_mature.items()}
            # Rank mapping: both are 0-based positions here.
            pred_ranks = {oid: r for r, oid in
                          enumerate(sorted(pred_shape.keys()))}
            target_ranks = {pos: pos for pos in target_mf3d_by_position}
            match = hungarian_leaf_match(
                pred_shape, target_mf3d_by_position,
                weight_xyz=1.0, weight_arc=0.5, weight_rank=5.0,
                pred_ranks=pred_ranks, target_ranks=target_ranks,
            )
            if not match:
                diag["mature"] = {"loss": penalty, "error": "no match"}
                mature_loss = penalty
            else:
                mature_loss = cp_l2_loss(pred_shape, target_mf3d_by_position,
                                           match, reduction="mean")
                diag["mature"] = {"loss": mature_loss,
                                  "n_match": len(match)}

    total = w_pheno * mean_pheno + w_mature * mature_loss
    diag["mean_pheno"] = mean_pheno
    diag["mature_loss"] = mature_loss
    diag["total"] = total
    return total, diag


# ---------------------------------------------------------------------------
# Final analysis — per-stage 1:1 comparison report
# ---------------------------------------------------------------------------

def analyze_final_fused(xml_path: str, params: dict,
                         target_pheno_by_day: dict[float, dict[int, np.ndarray]],
                         target_mf3d_by_position: dict[int, np.ndarray],
                         date_by_day: dict[float, str],
                         mature_day: float,
                         output_dir: Path) -> dict:
    """Run fitted params, export per-date OBJs + side-by-side CP diagnostics.

    Per-stage block carries:
      - matched (pred_oid, target_label) pairs
      - per-leaf CP-L2 + per-CP RMSE
      - unmatched leaf counts on both sides
    Mature block carries:
      - matched (pred_oid, position) pairs  (shape-only)
      - per-leaf CP-L2 + per-CP RMSE
    """
    report = {
        "params": params,
        "pheno_stages": [],
        "mature": None,
        "mean_pheno_cp_l2": None,
        "mature_cp_l2": None,
    }
    objs_dir = output_dir / "per_date_objs"
    objs_dir.mkdir(parents=True, exist_ok=True)

    stage_losses = []
    for day in sorted(target_pheno_by_day.keys()):
        date_str = date_by_day[day]
        try:
            pred_cps, mesh = grow_and_get_cps(xml_path, day, params)
        except Exception as exc:
            report["pheno_stages"].append(
                {"date": date_str, "day": day, "error": str(exc)})
            continue
        target = target_pheno_by_day[day]
        pred_keys_sorted = sorted(pred_cps.keys())
        pred_ranks = {oid: r for r, oid in enumerate(pred_keys_sorted)}
        target_ranks = {lab: int(lab) - 1 for lab in target}
        match = hungarian_leaf_match(
            pred_cps, target, weight_xyz=1.0, weight_arc=0.5, weight_rank=0.5,
            pred_ranks=pred_ranks, target_ranks=target_ranks,
        )
        per_leaf = []
        for p_oid, t_lab in match:
            pc = pred_cps[p_oid]  # type: ignore[index]
            tc = target[t_lab]  # type: ignore[index]
            diff = pc - tc
            per_leaf.append({
                "pred_organ_id": int(p_oid),  # type: ignore[arg-type]
                "target_label": int(t_lab),  # type: ignore[arg-type]
                "ssd_cm2": float(np.sum(diff * diff)),
                "rmse_cm": float(np.sqrt(np.mean(np.sum(diff * diff, axis=-1)))),
                "max_cp_err_cm": float(np.max(np.linalg.norm(diff, axis=-1))),
            })
        stage_loss = (cp_l2_loss(pred_cps, target, match, reduction="mean")
                      if match else None)
        if stage_loss is not None:
            stage_losses.append(stage_loss)
        report["pheno_stages"].append({
            "date": date_str, "day": day,
            "mean_cp_l2": stage_loss,
            "n_pred": len(pred_cps), "n_target": len(target),
            "n_matched": len(match),
            "n_pred_unmatched": max(0, len(pred_cps) - len(match)),
            "n_target_unmatched": max(0, len(target) - len(match)),
            "per_leaf": per_leaf,
        })
        obj_path = objs_dir / f"fitted_{date_str}_day{int(day):03d}.obj"
        export_obj(mesh, obj_path, f"Pheno4D {date_str} day {day:.1f}")
    report["mean_pheno_cp_l2"] = (float(np.mean(stage_losses))
                                   if stage_losses else None)

    # --- Mature stage ---
    try:
        pred_mature, mesh_mature = grow_and_get_cps(
            xml_path, mature_day, params)
    except Exception as exc:
        report["mature"] = {"day": mature_day, "error": str(exc)}
    else:
        pred_shape = {oid: collar_subtract(cps)
                      for oid, cps in pred_mature.items()}
        pred_ranks = {oid: r for r, oid in
                      enumerate(sorted(pred_shape.keys()))}
        target_ranks = {pos: pos for pos in target_mf3d_by_position}
        match = hungarian_leaf_match(
            pred_shape, target_mf3d_by_position,
            weight_xyz=1.0, weight_arc=0.5, weight_rank=5.0,
            pred_ranks=pred_ranks, target_ranks=target_ranks,
        )
        per_leaf = []
        for p_oid, pos in match:
            pc = pred_shape[p_oid]  # type: ignore[index]
            tc = target_mf3d_by_position[pos]  # type: ignore[index]
            diff = pc - tc
            per_leaf.append({
                "pred_organ_id": int(p_oid),  # type: ignore[arg-type]
                "mf3d_position": int(pos),  # type: ignore[arg-type]
                "ssd_cm2": float(np.sum(diff * diff)),
                "rmse_cm": float(np.sqrt(np.mean(np.sum(diff * diff, axis=-1)))),
                "max_cp_err_cm": float(np.max(np.linalg.norm(diff, axis=-1))),
            })
        mature_loss = (cp_l2_loss(pred_shape, target_mf3d_by_position,
                                    match, reduction="mean")
                       if match else None)
        report["mature"] = {
            "day": mature_day,
            "mean_cp_l2": mature_loss,
            "n_pred": len(pred_shape),
            "n_target": len(target_mf3d_by_position),
            "n_matched": len(match),
            "per_leaf": per_leaf,
        }
        report["mature_cp_l2"] = mature_loss
        mature_obj = objs_dir / f"fitted_mature_day{int(mature_day):03d}.obj"
        export_obj(mesh_mature, mature_obj,
                   f"MaizeField3D mature reference (day {mature_day:.1f})")

    return report


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fused CP-L2 fitter: Pheno4D trajectory + MaizeField3D mature.")
    parser.add_argument("--plant", required=True,
                        help="Pheno4D plant id (e.g. M01).")
    parser.add_argument("--pheno4d-cps",
                        default="/home/lukas/PHD/Resources/Pheno4D/pheno4d_canonical_cps.json")
    parser.add_argument("--maizefield3d-cps",
                        default="/home/lukas/PHD/Resources/MaizeField3d/maizefield3d_canonical_cps.json")
    parser.add_argument("--xml", default=None,
                        help="Template XML (default: data/maize_calibrated.xml).")
    parser.add_argument("--output", "-o", required=True,
                        help="Output directory.")
    parser.add_argument("--evals", type=int, default=50,
                        help="CMA-ES max evaluations.")
    parser.add_argument("--day-offset", type=float, default=20.0,
                        help="Simulation day for the earliest scan date.")
    parser.add_argument("--mature-day", type=float, default=65.0,
                        help="Simulation day for the MaizeField3D mature term.")
    parser.add_argument("--w-pheno", type=float, default=1.0,
                        help="Weight on the Pheno4D trajectory loss.")
    parser.add_argument("--w-mature", type=float, default=1.0,
                        help="Weight on the MaizeField3D mature loss.")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    xml_path = args.xml or str(_COUPLING_DIR / "data" / "maize_calibrated.xml")

    print("=" * 70)
    print(f"FUSED FIT — Pheno4D plant {args.plant} + MaizeField3D mature")
    print("=" * 70)

    # 1) Load target CPs.
    print("\n[1/5] Loading target CPs")
    cps_by_date = load_pheno4d_cps_for_plant(args.pheno4d_cps, args.plant)
    if not cps_by_date:
        print(f"  ERROR: no Pheno4D scans for plant {args.plant!r}")
        sys.exit(1)
    print(f"  Pheno4D:      {len(cps_by_date)} scans {sorted(cps_by_date.keys())}")
    target_mf3d = load_maizefield3d_mature_cps(args.maizefield3d_cps)
    print(f"  MaizeField3D: {len(target_mf3d)} positions {sorted(target_mf3d.keys())}")

    day_by_date = dates_to_sim_days(list(cps_by_date.keys()), args.day_offset)
    date_by_day = {d: date_str for date_str, d in day_by_date.items()}
    target_pheno_by_day = {
        day_by_date[date_str]: cps for date_str, cps in cps_by_date.items()
    }
    print(f"  Sim days:     {sorted(target_pheno_by_day.keys())}")
    print(f"  Mature day:   {args.mature_day}")
    print(f"  Weights:      w_pheno={args.w_pheno}, w_mature={args.w_mature}")

    # 2) Prepare XML: ensure enough leaf subtypes for both datasets.
    print("\n[2/5] Preparing XML (ensure_xml_has_all_subtypes)")
    max_pheno_rank = max(int(lab) for scan in cps_by_date.values() for lab in scan)
    max_mf3d_rank = max(target_mf3d.keys()) + 1  # 0-based → count
    max_rank = max(max_pheno_rank, max_mf3d_rank)
    positions = list(range(max_rank))  # 0..max_rank-1 covers positions 1..max_rank
    prep_xml = output_dir / "template_prepared.xml"
    ensure_xml_has_all_subtypes(xml_path, str(prep_xml), positions)
    xml_path = str(prep_xml)

    # 3) CMA-ES.
    print("\n[3/5] CMA-ES")
    bounds = list(DEFAULT_BOUNDS)
    lo = [b[1] for b in bounds]
    hi = [b[2] for b in bounds]
    ranges = [h - l for l, h in zip(lo, hi)]
    x0_scaled = [float(np.clip((v - lo[i]) / ranges[i], 0.02, 0.98))
                 for i, v in enumerate(DEFAULT_X0)]

    def objective(x_scaled: np.ndarray) -> float:
        x_real = np.array([lo[i] + x_scaled[i] * ranges[i]
                            for i in range(len(x_scaled))], dtype=np.float64)
        f, _ = evaluate_fused(
            x_real, bounds, xml_path,
            target_pheno_by_day, target_mf3d,
            args.mature_day, args.w_pheno, args.w_mature,
        )
        return f

    popsize = max(8, 2 * len(bounds))
    es = cma.CMAEvolutionStrategy(x0_scaled, 0.2, {
        "bounds": [0, 1],
        "maxfevals": args.evals,
        "verbose": -9,
        "seed": 42,
        "popsize": popsize,
    })
    print(f"  popsize={popsize}, maxfevals={args.evals}")
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
                print(f"    Gen {gen}: best total loss = {best_fitness:.3f}")
    elapsed = time.time() - t0
    best_scaled = es.result.xbest
    best_real = [float(lo[i] + best_scaled[i] * ranges[i])
                 for i in range(len(best_scaled))]
    best_params = {name: best_real[i] for i, (name, _, _) in enumerate(bounds)}
    print(f"  Done in {elapsed:.1f}s, {gen} gens, best total = {best_fitness:.3f}")
    for name, v in best_params.items():
        print(f"    {name:>16s} = {v:.4f}")

    # 4) Final analysis.
    print("\n[4/5] Final analysis + per-stage 1:1 report")
    report = analyze_final_fused(
        xml_path, best_params,
        target_pheno_by_day, target_mf3d,
        date_by_day, args.mature_day, output_dir,
    )
    report["cma"] = {
        "evals": es.result.evaluations, "generations": gen,
        "best_total_loss": best_fitness,
        "weights": {"w_pheno": args.w_pheno, "w_mature": args.w_mature},
    }
    (output_dir / "fit_report.json").write_text(json.dumps(report, indent=2))
    (output_dir / "fitted_params.json").write_text(json.dumps({
        "plant": args.plant, "params": best_params,
        "best_total_loss": best_fitness,
        "mean_pheno_cp_l2": report.get("mean_pheno_cp_l2"),
        "mature_cp_l2": report.get("mature_cp_l2"),
        "day_offset": args.day_offset,
        "mature_day": args.mature_day,
        "dates": sorted(cps_by_date.keys()),
    }, indent=2))

    # 5) Emit fitted XML.
    print("\n[5/5] Writing fitted XML")
    fitted_xml = output_dir / "fitted_maize.xml"
    apply_params_to_xml(xml_path, str(fitted_xml), best_params)
    print(f"  -> {fitted_xml}")

    print("\nDone.")
    print(f"  {output_dir}/fitted_maize.xml          (new calibrated XML)")
    print(f"  {output_dir}/fit_report.json           (per-stage 1:1 report)")
    print(f"  {output_dir}/fitted_params.json        (best params)")
    print(f"  {output_dir}/per_date_objs/*.obj       (per-stage meshes)")


if __name__ == "__main__":
    main()
