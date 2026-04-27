#!/usr/bin/env python3
"""Validation driver — MaizeField3D → canonical 11x5 CP reconstruction error.

Loads ``Resources/MaizeField3d/maizefield3d_canonical_cps.json`` (produced by
``resample_all_to_canonical.py``) and reports per-position and overall
reconstruction-error statistics for the canonical LSQ fits. This is the
reproducible artifact that backs the "Implementation outcomes" numbers in
``NURBS_FITTING_PIPELINE.md`` (Phase 5).

Also verifies CP shape invariants (``(11, 5, 3)`` with no NaN / Inf) — a
light drift guard in case the resampler is re-run with different knot /
degree settings.

Outputs:
  --output PATH     JSON summary (default: tools/output/maizefield3d_validation.json)
  --markdown PATH   Markdown table alongside the JSON (default: same name with .md).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date as _today
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent  # /home/lukas/PHD/CPlantBox
sys.path.insert(0, str(_REPO))

from dart.coupling.geometry.canonical_cp_grid import N_U, N_V

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_INPUT = Path(
    "/home/lukas/PHD/Resources/MaizeField3d/maizefield3d_canonical_cps.json"
)
ACCEPT_MEAN_RMSE_CM = 0.06  # plan acceptance gate


def summarise_stats(vals: list[float]) -> dict:
    if not vals:
        return {"n": 0}
    a = np.asarray(vals, dtype=np.float64)
    return {
        "n": int(a.size),
        "mean": float(a.mean()),
        "median": float(np.median(a)),
        "p95": float(np.percentile(a, 95)),
        "max": float(a.max()),
    }


def validate(json_path: Path) -> dict:
    """Walk every plant×leaf in the dataset; aggregate per-position errors."""
    data = json.loads(json_path.read_text())
    plants = data.get("plants", [])
    if not plants:
        raise RuntimeError(f"No plants in {json_path}")

    per_pos: dict[int, dict[str, list[float]]] = {}
    shape_errors: list[str] = []
    finite_errors: list[str] = []
    total_leaves = 0

    for plant in plants:
        plant_id = plant.get("plant_id", "?")
        for leaf in plant.get("leaves", []):
            total_leaves += 1
            pos = int(leaf["position"])
            cps = np.asarray(leaf["cps_cm"], dtype=np.float64)

            if cps.shape != (N_U, N_V, 3):
                shape_errors.append(f"{plant_id}/pos{pos}: shape={cps.shape}")
                continue
            if not np.all(np.isfinite(cps)):
                finite_errors.append(f"{plant_id}/pos{pos}: non-finite CP")
                continue

            bucket = per_pos.setdefault(pos, {"rmse_cm": [], "max_err_cm": []})
            bucket["rmse_cm"].append(float(leaf["rmse_cm"]))
            bucket["max_err_cm"].append(float(leaf["max_err_cm"]))

    summary = {
        "source": str(json_path),
        "generated_on": _today.today().isoformat(),
        "n_plants": len(plants),
        "n_leaves_total": total_leaves,
        "n_positions": len(per_pos),
        "shape_errors": shape_errors,
        "finite_errors": finite_errors,
        "per_position": [],
    }

    all_rmse: list[float] = []
    all_max: list[float] = []
    for pos in sorted(per_pos.keys()):
        stats_rmse = summarise_stats(per_pos[pos]["rmse_cm"])
        stats_max = summarise_stats(per_pos[pos]["max_err_cm"])
        summary["per_position"].append({
            "position": pos,
            "n": stats_rmse["n"],
            "rmse_cm": stats_rmse,
            "max_err_cm": stats_max,
        })
        all_rmse.extend(per_pos[pos]["rmse_cm"])
        all_max.extend(per_pos[pos]["max_err_cm"])

    summary["overall"] = {
        "rmse_cm": summarise_stats(all_rmse),
        "max_err_cm": summarise_stats(all_max),
        "accept_mean_rmse_threshold_cm": ACCEPT_MEAN_RMSE_CM,
        "accept_passed": (
            bool(all_rmse) and float(np.mean(all_rmse)) < ACCEPT_MEAN_RMSE_CM
        ),
    }
    return summary


def write_markdown(summary: dict, out_path: Path) -> None:
    lines = [
        f"# MaizeField3D canonical-CP validation — {summary['generated_on']}",
        "",
        f"Source: `{summary['source']}`",
        f"Plants: {summary['n_plants']}   Leaves: {summary['n_leaves_total']}   "
        f"Positions tracked: {summary['n_positions']}",
        "",
        "## Per-position RMSE (cm)",
        "",
        "| Position | n | mean | median | p95 | max |",
        "|---------:|---:|------:|--------:|------:|-----:|",
    ]
    for pp in summary["per_position"]:
        r = pp["rmse_cm"]
        lines.append(
            f"| {pp['position']:>2} | {pp['n']:>4} | {r['mean']:.4f} | "
            f"{r['median']:.4f} | {r['p95']:.4f} | {r['max']:.4f} |"
        )

    o = summary["overall"]
    r = o["rmse_cm"]
    m = o["max_err_cm"]
    lines += [
        "",
        "## Overall",
        "",
        f"- Mean RMSE: **{r['mean']:.4f} cm** "
        f"(threshold: {o['accept_mean_rmse_threshold_cm']:.4f} cm, "
        f"{'PASS' if o['accept_passed'] else 'FAIL'})",
        f"- Median RMSE: {r['median']:.4f} cm",
        f"- p95 RMSE: {r['p95']:.4f} cm",
        f"- Max per-leaf error (mean across leaves): {m['mean']:.4f} cm",
        f"- Max per-leaf error (absolute max): {m['max']:.4f} cm",
    ]
    if summary["shape_errors"]:
        lines += ["", f"## Shape errors ({len(summary['shape_errors'])})", ""]
        lines += [f"- {e}" for e in summary["shape_errors"][:20]]
    if summary["finite_errors"]:
        lines += ["", f"## Non-finite errors ({len(summary['finite_errors'])})", ""]
        lines += [f"- {e}" for e in summary["finite_errors"][:20]]

    out_path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(__doc__ or "").strip().splitlines()[0])
    parser.add_argument("--input", default=str(DEFAULT_INPUT),
                        help=f"Input canonical-CP JSON (default: {DEFAULT_INPUT})")
    parser.add_argument("--output", default="tools/output/maizefield3d_validation.json",
                        help="Output JSON summary path.")
    parser.add_argument("--markdown", default=None,
                        help="Output markdown path (default: matches --output with .md).")
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        print(f"ERROR: input JSON not found: {in_path}", file=sys.stderr)
        return 1

    print(f"Loading {in_path}...")
    summary = validate(in_path)

    out_json = Path(args.output)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2))

    out_md = Path(args.markdown) if args.markdown \
        else out_json.with_suffix(".md")
    write_markdown(summary, out_md)

    r = summary["overall"]["rmse_cm"]
    print(f"\nValidated {summary['n_leaves_total']} leaves across "
          f"{summary['n_positions']} positions from {summary['n_plants']} plants.")
    print(f"Mean RMSE: {r['mean']:.4f} cm  "
          f"(threshold {ACCEPT_MEAN_RMSE_CM:.4f} cm — "
          f"{'PASS' if summary['overall']['accept_passed'] else 'FAIL'})")
    print(f"Median RMSE: {r['median']:.4f} cm   p95: {r['p95']:.4f} cm   "
          f"max: {r['max']:.4f} cm")
    print(f"\nWrote: {out_json}")
    print(f"Wrote: {out_md}")
    return 0 if summary["overall"]["accept_passed"] else 2


if __name__ == "__main__":
    sys.exit(main())
