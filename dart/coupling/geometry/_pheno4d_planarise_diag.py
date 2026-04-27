"""Diagnostic OBJ dumper for the Pheno4D planariser.

Dumps control-polygon OBJs for a handful of Pheno4D fits BEFORE and AFTER
running them through ``_planarise_pheno4d_fit``, plus the aggregated
library bucket. Meshes are the raw CP grid connected by quads — not the
evaluated NURBS surface — which is enough to eyeball whether the
planariser is straightening the midrib, killing the whorl, and
symmetrising the cross-rows.

Usage:
    python3 -m dart.coupling.geometry._pheno4d_planarise_diag \
        --out /tmp/pheno4d_diag \
        --n-samples 3
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .canonical_library import (
    N_U,
    N_V,
    _midrib_arc,
    _planarise_pheno4d_fit,
    load_young_library,
    to_local_frame,
)


def write_cp_grid_obj(path: Path, cps: np.ndarray, label: str = "grid") -> None:
    """Write a CP grid ``(N_U, N_V, 3)`` as an OBJ of quads + midrib line."""
    n_u, n_v, _ = cps.shape
    with open(path, "w") as f:
        f.write(f"# {label}\n")
        f.write(f"# shape = ({n_u}, {n_v}, 3)\n")
        f.write(f"g {label}\n")
        for u in range(n_u):
            for v in range(n_v):
                p = cps[u, v]
                f.write(f"v {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")

        def vid(u: int, v: int) -> int:
            return u * n_v + v + 1  # OBJ 1-indexed

        # Quads between adjacent CPs (fan-triangulated as two tris each).
        f.write("usemtl grid\n")
        for u in range(n_u - 1):
            for v in range(n_v - 1):
                a = vid(u, v)
                b = vid(u, v + 1)
                c = vid(u + 1, v + 1)
                d = vid(u + 1, v)
                f.write(f"f {a} {b} {c}\n")
                f.write(f"f {a} {c} {d}\n")
        # Highlight the midrib column as a line strip (easy to see in viewers).
        mid_j = n_v // 2
        f.write("g midrib\n")
        for u in range(n_u - 1):
            f.write(f"l {vid(u, mid_j)} {vid(u + 1, mid_j)}\n")


def _midrib_y_range_frac(cps: np.ndarray) -> float:
    mid_j = cps.shape[1] // 2
    m = cps[:, mid_j, :]
    arc = _midrib_arc(cps)
    return float((m[:, 1].max() - m[:, 1].min()) / max(arc, 1e-9))


def _midrib_tip_z_frac(cps: np.ndarray) -> float:
    mid_j = cps.shape[1] // 2
    m = cps[:, mid_j, :]
    arc = _midrib_arc(cps)
    return float((m[-1, 2] - m[0, 2]) / max(arc, 1e-9))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--pheno4d-json",
        type=Path,
        default=Path("/home/lukas/PHD/Resources/Pheno4D/pheno4d_canonical_cps.json"),
    )
    ap.add_argument(
        "--library",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "pheno4d_young_library.npz",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("/tmp/pheno4d_diag"),
    )
    ap.add_argument(
        "--n-per-class",
        type=int,
        default=2,
        help="Number of representative fits per class (clean / drooped / whorl).",
    )
    args = ap.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)

    data = json.loads(args.pheno4d_json.read_text())
    raw: list[tuple[str, str, int, np.ndarray]] = []  # (plant, scan, label, cps)
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
            raw.append((plant, sdate, int(label), arr))

    # Classify each fit by pre-planarise metrics, splitting rejects into
    # the gate that actually fired.
    classes: dict[str, list] = {
        "clean": [],       # passes planariser + y_frac/tip_z heuristics
        "drooped": [],     # passes planariser but with visible droop pre-clean
        "whorl": [],       # rejected by early whorl-wrap gate
        "qa_reject": [],   # passed whorl gate, rejected by post-QA
    }
    for entry in raw:
        plant, sdate, label, cps = entry
        try:
            cps_local, _, _ = to_local_frame(cps)
        except Exception:
            continue
        y_frac = _midrib_y_range_frac(cps_local)
        tip_z = _midrib_tip_z_frac(cps_local)

        # Probe the whorl gate in isolation by running the planariser with
        # all QA thresholds disabled.
        loose = _planarise_pheno4d_fit(
            cps_local,
            tip_z_min=-1e9,
            x_range_max=1e9,
            y_range_max=1e9,
        )
        # Same thresholds as the default (the shipped config).
        strict = _planarise_pheno4d_fit(cps_local)

        if loose is None:
            classes["whorl"].append((plant, sdate, label, cps_local, y_frac, tip_z))
        elif strict is None:
            classes["qa_reject"].append((plant, sdate, label, cps_local, y_frac, tip_z))
        elif y_frac < 0.15 and tip_z > 0.9:
            classes["clean"].append((plant, sdate, label, cps_local, y_frac, tip_z))
        else:
            classes["drooped"].append((plant, sdate, label, cps_local, y_frac, tip_z))

    print(
        f"classified — clean: {len(classes['clean'])}, "
        f"drooped: {len(classes['drooped'])}, "
        f"qa_reject: {len(classes['qa_reject'])}, "
        f"whorl_reject: {len(classes['whorl'])}"
    )

    # Dump N per class, before + after planarisation.
    for cls in ("clean", "drooped", "qa_reject", "whorl"):
        samples = classes[cls][: args.n_per_class]
        for i, (plant, sdate, label, cps_local, y_frac, tip_z) in enumerate(samples):
            tag = f"{cls}_{i:02d}_{plant}_{sdate}_leaf{label}"
            raw_obj = args.out / f"{tag}_raw.obj"
            write_cp_grid_obj(raw_obj, cps_local, label=f"{tag}_raw")
            print(f"  wrote {raw_obj.name}  y_frac={y_frac:.3f}  tip_z={tip_z:.3f}")
            planarised = _planarise_pheno4d_fit(cps_local)
            if planarised is None:
                print(f"    → rejected by planariser (expected for whorl-wrapped)")
                continue
            cleaned_obj = args.out / f"{tag}_planarised.obj"
            write_cp_grid_obj(cleaned_obj, planarised, label=f"{tag}_planarised")
            y_after = _midrib_y_range_frac(planarised)
            tip_z_after = _midrib_tip_z_frac(planarised)
            print(f"  wrote {cleaned_obj.name}  y_frac={y_after:.3f}  tip_z={tip_z_after:.3f}")

    # Aggregated library buckets.
    if args.library.exists():
        lib = load_young_library(args.library)
        cps_lib = lib["cps_normalised"]
        centers = lib["bucket_centers"]
        counts = lib["counts"]
        for idx, center in enumerate(centers):
            tag = f"library_bucket_mat{center:.2f}_n{int(counts[idx])}"
            obj = args.out / f"{tag}.obj"
            # arc-normalised grids are length 1; scale to 50 cm for viewing.
            write_cp_grid_obj(obj, cps_lib[idx] * 50.0, label=tag)
            print(f"  wrote {obj.name}")

    print(f"done → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
