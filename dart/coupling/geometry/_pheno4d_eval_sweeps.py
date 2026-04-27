"""Standalone evaluator for the Pheno4D young-library threshold sweeps.

Companion script for the plan note
``Literature/Chapter 1/Concepts/CPBOXBALENOCOUPLING/PHENO4D_YOUNG_LIBRARY_EVAL_2026-04-25.md``.
Re-runs the planariser + bucket aggregation under varying gate thresholds
and bucket schemes, dumps per-cell stats and visual-QA OBJs, and never
writes to any production artefact (``pheno4d_young_library.npz``,
``maize_calibrated.xml`` and friends are left untouched).

Usage
-----
    python3 -m dart.coupling.geometry._pheno4d_eval_sweeps \
        --out /tmp/pheno4d_eval --sweep all --emit-obj

Layout
------
    <out>/
      sweep_wind/    cells = max_wind_deg sweep
      sweep_qa/      cells = (gate, value) one-axis-at-a-time sweep
      sweep_bucket/  cells = bucket scheme + min_samples (loose thresholds)
      summary.json   cross-sweep digest

Each sweep directory contains ``stats.json``, ``shape_residual.json`` and
(when ``--emit-obj``) an ``obj/`` directory with ``..._median.obj`` and
up to ``--n-visual-samples`` per (cell, bucket) sample OBJs for Blender
inspection.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np

from .canonical_library import (
    N_U,
    N_V,
    _midrib_arc,
    _planarise_pheno4d_fit,
    to_local_frame,
)


# A "very large" threshold acts as `None` for the planariser without
# changing its public API.
_DISABLE = 1.0e9

# Defaults match the shipped library configuration so each sweep keeps
# the non-swept axes at the production setting unless documented otherwise.
DEFAULT_THRESHOLDS = {
    "max_wind_deg": 60.0,
    "tip_z_min": 0.85,
    "x_range_max": 0.25,
    "y_range_max": 0.20,
}

# Sweep 3 uses a relaxed configuration — otherwise the stricter gates
# zero out every bucket below m=0.9 and the bucket structure becomes
# unobservable. Documented in the plan note (§ Sweep 3).
LOOSE_THRESHOLDS = {
    "max_wind_deg": _DISABLE,
    "tip_z_min": 0.30,
    "x_range_max": 0.60,
    "y_range_max": 0.30,
}


# --------------------------------------------------------------------- IO


def load_raw_fits(path: Path) -> list[dict]:
    """Return per-leaf records: ``{plant, scan, label, cps_world, arc}``."""
    data = json.loads(Path(path).read_text())
    out: list[dict] = []
    for scan in data.get("scans", []):
        plant = scan.get("plant_id")
        sdate = scan.get("date") or scan.get("scan_id") or "?"
        for leaf in scan.get("leaves", []):
            cps = leaf.get("cps_cm")
            label = leaf.get("label")
            if cps is None or label is None:
                continue
            arr = np.asarray(cps, dtype=np.float64)
            if arr.shape != (N_U, N_V, 3):
                continue
            arc = _midrib_arc(arr)
            if arc <= 1e-6:
                continue
            out.append({
                "plant": str(plant),
                "scan": str(sdate),
                "label": int(label),
                "cps_world": arr,
                "arc": float(arc),
            })
    return out


def annotate_maturity(records: Sequence[dict]) -> None:
    """In-place: add ``maturity`` (arc / chain max) to each record."""
    chain_max: dict[tuple[str, int], float] = defaultdict(float)
    for r in records:
        key = (r["plant"], r["label"])
        if r["arc"] > chain_max[key]:
            chain_max[key] = r["arc"]
    for r in records:
        lmax = chain_max[(r["plant"], r["label"])]
        r["maturity"] = float(min(r["arc"] / lmax, 1.0)) if lmax > 1e-6 else 0.0


# ---------------------------------------------------------- Process / bucket


def planarise_record(
    rec: dict,
    *,
    max_wind_deg: float,
    tip_z_min: float,
    x_range_max: float,
    y_range_max: float,
) -> tuple[str, np.ndarray | None]:
    """Run a single fit through ``to_local_frame`` + ``_planarise_pheno4d_fit``.

    Returns ``(reason, cps_normalised | None)``. ``reason`` is one of
    ``"frame_fail"``, ``"whorl_reject"``, ``"qa_reject"``, ``"degenerate_arc"``,
    ``"ok"``.
    """
    try:
        cps_local, _, _ = to_local_frame(rec["cps_world"])
    except Exception:
        return "frame_fail", None

    # Probe the whorl gate in isolation by calling the planariser twice:
    # once with QA disabled (only the wind gate fires) and once at the
    # requested QA. Lets the script attribute rejects to the right gate
    # without monkey-patching ``_planarise_pheno4d_fit``.
    wind_only = _planarise_pheno4d_fit(
        cps_local,
        max_wind_deg=max_wind_deg,
        tip_z_min=-_DISABLE,
        x_range_max=_DISABLE,
        y_range_max=_DISABLE,
    )
    if wind_only is None:
        return "whorl_reject", None

    cleaned = _planarise_pheno4d_fit(
        cps_local,
        max_wind_deg=max_wind_deg,
        tip_z_min=tip_z_min,
        x_range_max=x_range_max,
        y_range_max=y_range_max,
    )
    if cleaned is None:
        return "qa_reject", None

    arc_local = _midrib_arc(cleaned)
    if arc_local <= 1e-6:
        return "degenerate_arc", None
    return "ok", cleaned * (1.0 / arc_local)


def bucket_for(maturity: float, scheme: str) -> tuple[int, str]:
    """Return ``(bucket_index, label)`` under the given scheme.

    Schemes:
      - ``"equal-10"``   : 10 equal-width maturity bins (production default)
      - ``"equal-5"``    : 5 equal-width bins (0.2 each)
      - ``"biological-3"`` : <0.3 / 0.3-0.7 / >=0.7
      - ``"per-rank"``   : not maturity — the bucket index is the
        Pheno4D leaf label. Resolved separately by the caller because we
        need access to ``rec['label']`` rather than maturity.
    """
    m = float(np.clip(maturity, 0.0, 1.0))
    if scheme == "equal-10":
        b = int(min(m * 10, 9))
        return b, f"m{b/10:.1f}-{(b+1)/10:.1f}"
    if scheme == "equal-5":
        b = int(min(m * 5, 4))
        return b, f"m{b/5:.1f}-{(b+1)/5:.1f}"
    if scheme == "biological-3":
        if m < 0.3:
            return 0, "m<0.3"
        if m < 0.7:
            return 1, "m0.3-0.7"
        return 2, "m>=0.7"
    raise ValueError(f"unknown scheme {scheme!r} (per-rank handled by caller)")


# ------------------------------------------------------------------- Stats


def hausdorff(a: np.ndarray, b: np.ndarray) -> float:
    """Symmetric Hausdorff distance between two ``(N_U, N_V, 3)`` grids."""
    pa = a.reshape(-1, 3)
    pb = b.reshape(-1, 3)
    diff = pa[:, None, :] - pb[None, :, :]
    d = np.linalg.norm(diff, axis=-1)
    return float(max(d.min(axis=1).max(), d.min(axis=0).max()))


def aggregate_bucket(
    samples: list[np.ndarray],
    reducer: str = "median",
) -> tuple[np.ndarray, float, float]:
    """Median (or mean) CP grid + (mean, max) Hausdorff residual to it."""
    stack = np.stack(samples, axis=0)
    agg = (
        np.median(stack, axis=0) if reducer == "median" else np.mean(stack, axis=0)
    )
    arc = _midrib_arc(agg)
    if arc > 1e-6:
        agg = agg * (1.0 / arc)
    residuals = [hausdorff(s, agg) for s in samples]
    return agg, float(np.mean(residuals)), float(np.max(residuals))


# --------------------------------------------------------------------- OBJ


def write_cp_grid_obj(path: Path, cps: np.ndarray, label: str = "grid") -> None:
    """Write a CP grid ``(N_U, N_V, 3)`` as quads + a midrib line strip."""
    n_u, n_v, _ = cps.shape
    with open(path, "w") as f:
        f.write(f"# {label}\n# shape = ({n_u}, {n_v}, 3)\n")
        f.write(f"g {label}\n")
        for u in range(n_u):
            for v in range(n_v):
                p = cps[u, v]
                f.write(f"v {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")

        def vid(u: int, v: int) -> int:
            return u * n_v + v + 1

        for u in range(n_u - 1):
            for v in range(n_v - 1):
                a = vid(u, v)
                b = vid(u, v + 1)
                c = vid(u + 1, v + 1)
                d = vid(u + 1, v)
                f.write(f"f {a} {b} {c}\n")
                f.write(f"f {a} {c} {d}\n")
        mid_j = n_v // 2
        f.write("g midrib\n")
        for u in range(n_u - 1):
            f.write(f"l {vid(u, mid_j)} {vid(u + 1, mid_j)}\n")


# ------------------------------------------------------------------- Sweep


def run_cell(
    records: Sequence[dict],
    *,
    thresholds: dict,
    scheme: str,
    min_samples_per_bucket: int,
) -> dict:
    """Process every record under one threshold + bucket-scheme combo."""
    bucket_samples: dict[int, list[np.ndarray]] = defaultdict(list)
    bucket_label: dict[int, str] = {}
    bucket_examples: dict[int, list[dict]] = defaultdict(list)
    rejection_hist: Counter = Counter()
    seen_per_bucket: Counter = Counter()

    for rec in records:
        if scheme == "per-rank":
            b = int(rec["label"])
            label = f"rank{b}"
        else:
            b, label = bucket_for(rec["maturity"], scheme)
        bucket_label[b] = label
        seen_per_bucket[b] += 1
        reason, cps_norm = planarise_record(rec, **thresholds)
        rejection_hist[reason] += 1
        if reason != "ok" or cps_norm is None:
            continue
        bucket_samples[b].append(cps_norm)
        bucket_examples[b].append({
            "plant": rec["plant"],
            "scan": rec["scan"],
            "label": rec["label"],
            "maturity": round(rec["maturity"], 4),
            "arc_cm": round(rec["arc"], 3),
            "cps_norm": cps_norm,
        })

    kept_per_bucket = {b: len(v) for b, v in bucket_samples.items()}
    kept_buckets = sorted(
        [b for b, n in kept_per_bucket.items() if n >= min_samples_per_bucket]
    )

    bucket_results = []
    for b in kept_buckets:
        samples = bucket_samples[b]
        agg, mean_res, max_res = aggregate_bucket(samples)
        bucket_results.append({
            "bucket": b,
            "label": bucket_label.get(b, str(b)),
            "n_samples": len(samples),
            "shape_residual_mean": mean_res,
            "shape_residual_max": max_res,
            "median_cps": agg,
            "examples": bucket_examples[b],
        })

    return {
        "thresholds": dict(thresholds),
        "scheme": scheme,
        "min_samples_per_bucket": int(min_samples_per_bucket),
        "rejection_hist": dict(rejection_hist),
        "seen_per_bucket": {
            bucket_label.get(b, str(b)): n
            for b, n in sorted(seen_per_bucket.items())
        },
        "kept_per_bucket": {
            bucket_label.get(b, str(b)): n
            for b, n in sorted(kept_per_bucket.items())
        },
        "bucket_results": bucket_results,
    }


def emit_cell_outputs(
    cell: dict,
    *,
    out_dir: Path,
    cell_tag: str,
    emit_obj: bool,
    n_visual_samples: int,
) -> dict:
    """Drop OBJs (if requested) and return a JSON-serialisable summary."""
    obj_dir = out_dir / "obj"
    if emit_obj:
        obj_dir.mkdir(parents=True, exist_ok=True)

    summary_buckets = []
    for br in cell["bucket_results"]:
        summary = {
            "bucket": br["bucket"],
            "label": br["label"],
            "n_samples": br["n_samples"],
            "shape_residual_mean": br["shape_residual_mean"],
            "shape_residual_max": br["shape_residual_max"],
        }
        summary_buckets.append(summary)

        if emit_obj:
            # Median bucket grid (scaled to 50 cm for visual size).
            tag = f"{cell_tag}_{_safe(br['label'])}_median_n{br['n_samples']}"
            write_cp_grid_obj(
                obj_dir / f"{tag}.obj",
                br["median_cps"] * 50.0,
                label=tag,
            )
            for j, ex in enumerate(br["examples"][:n_visual_samples]):
                stag = (
                    f"{cell_tag}_{_safe(br['label'])}_s{j:02d}"
                    f"_{ex['plant']}_{ex['scan']}_leaf{ex['label']}"
                )
                write_cp_grid_obj(
                    obj_dir / f"{stag}.obj",
                    ex["cps_norm"] * 50.0,
                    label=stag,
                )

    return {
        "cell_tag": cell_tag,
        "thresholds": cell["thresholds"],
        "scheme": cell["scheme"],
        "min_samples_per_bucket": cell["min_samples_per_bucket"],
        "rejection_hist": cell["rejection_hist"],
        "seen_per_bucket": cell["seen_per_bucket"],
        "kept_per_bucket": cell["kept_per_bucket"],
        "buckets": summary_buckets,
    }


def _safe(text: str) -> str:
    return "".join(c if (c.isalnum() or c in "._-") else "_" for c in text)


# ---------------------------------------------------------------- Sweeps


def sweep_wind(records, out_dir, *, emit_obj, n_visual_samples) -> list[dict]:
    """Vary ``max_wind_deg`` only. Bucket scheme = production default."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cells = []
    for wind in (60.0, 180.0, 360.0, _DISABLE):
        thresholds = dict(DEFAULT_THRESHOLDS, max_wind_deg=wind)
        tag = f"wind_{int(wind) if wind != _DISABLE else 'OFF'}"
        cell = run_cell(
            records,
            thresholds=thresholds,
            scheme="equal-10",
            min_samples_per_bucket=1,
        )
        cells.append(
            emit_cell_outputs(
                cell,
                out_dir=out_dir,
                cell_tag=tag,
                emit_obj=emit_obj,
                n_visual_samples=n_visual_samples,
            )
        )
    return cells


def sweep_qa(records, out_dir, *, emit_obj, n_visual_samples) -> list[dict]:
    """One-axis-at-a-time QA sweep (wind held at OFF to surface QA's effect)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cells = []
    sweep_grid = {
        "tip_z_min": [0.30, 0.50, 0.70, 0.85],
        "x_range_max": [0.25, 0.40, 0.60],
        "y_range_max": [0.20, 0.30, 0.50],
    }
    base = dict(DEFAULT_THRESHOLDS)
    base["max_wind_deg"] = _DISABLE  # so QA gates dominate the rejection budget
    for axis, values in sweep_grid.items():
        for v in values:
            thresholds = dict(base, **{axis: v})
            tag = f"qa_{axis}_{v}"
            cell = run_cell(
                records,
                thresholds=thresholds,
                scheme="equal-10",
                min_samples_per_bucket=1,
            )
            cells.append(
                emit_cell_outputs(
                    cell,
                    out_dir=out_dir,
                    cell_tag=tag,
                    emit_obj=emit_obj,
                    n_visual_samples=n_visual_samples,
                )
            )
    return cells


def sweep_bucket(records, out_dir, *, emit_obj, n_visual_samples) -> list[dict]:
    """Bucket scheme + ``min_samples`` sweep under the LOOSE threshold combo.

    LOOSE thresholds are required here — under the production defaults
    every bucket below m=0.9 is empty, so the scheme has no degrees of
    freedom to exercise. Document in plan note § Sweep 3.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    cells = []
    schemes = ["equal-10", "equal-5", "biological-3", "per-rank"]
    for scheme in schemes:
        for min_samples in (2, 3):
            tag = f"bucket_{scheme}_min{min_samples}"
            cell = run_cell(
                records,
                thresholds=dict(LOOSE_THRESHOLDS),
                scheme=scheme,
                min_samples_per_bucket=min_samples,
            )
            cells.append(
                emit_cell_outputs(
                    cell,
                    out_dir=out_dir,
                    cell_tag=tag,
                    emit_obj=emit_obj,
                    n_visual_samples=n_visual_samples,
                )
            )
    return cells


# --------------------------------------------------------------------- IO


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=_json_default)


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    raise TypeError(f"non-serialisable: {type(o)}")


def write_sweep_outputs(name: str, cells: list[dict], out_dir: Path) -> None:
    write_json(out_dir / "stats.json", {"sweep": name, "cells": cells})
    residual = []
    for c in cells:
        for b in c["buckets"]:
            residual.append({
                "cell_tag": c["cell_tag"],
                "scheme": c["scheme"],
                "thresholds": c["thresholds"],
                "bucket": b["label"],
                "n_samples": b["n_samples"],
                "shape_residual_mean": b["shape_residual_mean"],
                "shape_residual_max": b["shape_residual_max"],
            })
    write_json(out_dir / "shape_residual.json", residual)


def acceptance_summary(cells: list[dict]) -> dict:
    """Score every cell against the plan's 4-point acceptance rule.

    Counts a cell as "viable" if buckets covering m=0.3-0.7 each have
    >=5 samples and shape_residual_mean <= 0.15 (arc-fraction). The full
    visual-progression test still requires manual Blender inspection.
    """
    out = []
    for c in cells:
        # Pull buckets that cover m ∈ [0.3, 0.7]. Naming heuristic on the
        # equal-10 / equal-5 / biological-3 schemes used above.
        relevant = []
        for b in c["buckets"]:
            label = b["label"]
            if label.startswith("m"):
                # Either "mLO-HI" or "m<X" / "m>=X" / "mLO-HI"
                relevant.append(b)
        in_window = []
        for b in relevant:
            label = b["label"]
            if label == "m0.3-0.7" or label == "m<0.3" or label == "m>=0.7":
                if label == "m0.3-0.7":
                    in_window.append(b)
            else:
                # "mLO-HI"
                try:
                    body = label[1:]
                    lo, hi = body.split("-")
                    lo_f = float(lo)
                    hi_f = float(hi)
                    if hi_f > 0.3 and lo_f < 0.7:
                        in_window.append(b)
                except Exception:
                    pass
        if not in_window:
            out.append({"cell_tag": c["cell_tag"], "viable": False,
                        "reason": "no buckets cover m=0.3-0.7"})
            continue
        min_n = min(b["n_samples"] for b in in_window)
        max_res = max(b["shape_residual_mean"] for b in in_window)
        viable = (min_n >= 5) and (max_res <= 0.15)
        out.append({
            "cell_tag": c["cell_tag"],
            "viable": bool(viable),
            "n_samples_min_in_window": int(min_n),
            "shape_residual_max_in_window": float(max_res),
            "buckets_in_window": [b["label"] for b in in_window],
        })
    return {"per_cell": out, "any_viable": any(x.get("viable") for x in out)}


# --------------------------------------------------------------------- main


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--pheno4d-json",
        type=Path,
        default=Path("/home/lukas/PHD/Resources/Pheno4D/pheno4d_canonical_cps.json"),
    )
    ap.add_argument("--out", type=Path, default=Path("/tmp/pheno4d_eval"))
    ap.add_argument(
        "--sweep",
        choices=["wind", "qa", "bucket", "all"],
        default="all",
    )
    ap.add_argument("--emit-obj", action="store_true")
    ap.add_argument("--n-visual-samples", type=int, default=3)
    args = ap.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    print(f"loading {args.pheno4d_json}")
    records = load_raw_fits(args.pheno4d_json)
    annotate_maturity(records)
    print(f"  {len(records)} valid (N_U, N_V) leaves across "
          f"{len({(r['plant'], r['label']) for r in records})} chains")

    sweeps_to_run = (
        ["wind", "qa", "bucket"] if args.sweep == "all" else [args.sweep]
    )
    digest = {"records": len(records), "sweeps": {}}
    for name in sweeps_to_run:
        print(f"\n=== sweep: {name}")
        sweep_dir = args.out / f"sweep_{name}"
        if name == "wind":
            cells = sweep_wind(
                records, sweep_dir,
                emit_obj=args.emit_obj,
                n_visual_samples=args.n_visual_samples,
            )
        elif name == "qa":
            cells = sweep_qa(
                records, sweep_dir,
                emit_obj=args.emit_obj,
                n_visual_samples=args.n_visual_samples,
            )
        elif name == "bucket":
            cells = sweep_bucket(
                records, sweep_dir,
                emit_obj=args.emit_obj,
                n_visual_samples=args.n_visual_samples,
            )
        else:
            raise AssertionError(name)
        write_sweep_outputs(name, cells, sweep_dir)
        digest["sweeps"][name] = {
            "n_cells": len(cells),
            "acceptance": acceptance_summary(cells),
        }
        print(f"  {len(cells)} cells; viable = "
              f"{digest['sweeps'][name]['acceptance']['any_viable']}")

    write_json(args.out / "summary.json", digest)
    print(f"\ndone → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
