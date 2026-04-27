#!/usr/bin/env python3
"""Landscape-smoothness receipt — CP-L2 vs. Chamfer along a 1D parameter sweep.

Generates the artifact that justifies the NURBS fitter refactor in the
Chapter 1 writeup. For one synthetic target plant:

  1. Pick a target leaf (a canonical 11x5x3 CP grid from a synthetic organ).
  2. Sweep a single deformation parameter (``wave_normal_amp`` by default)
     across a 1D range.
  3. At each value, loft via the NURBS backend to get both tessellated
     vertices *and* the canonical CPs.
  4. Compute two losses against the unperturbed target:
       - Bidirectional mean Chamfer on the tessellated vertices
         (via KDTree).
       - ``cp_l2_loss`` on the canonical CP grid.
  5. Plot both loss curves on the same parameter axis.

Expected: the CP-L2 curve should be visibly smoother / more unimodal than
the Chamfer curve, whose jagged structure makes gradient-free optimisers
wander.

Usage:
    python3 tools/compare_cp_vs_chamfer.py \\
        --param wave_normal_amp --range 0.0,1.0,40 \\
        --output tools/output/landscape.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import KDTree

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from dart.coupling.geometry.canonical_cp_grid import N_U, N_V
from dart.coupling.geometry.nurbs_blade import loft_leaf_nurbs
from dart.coupling.experimental.losses.cp_distance import cp_l2_loss


# ---------------------------------------------------------------------------
# Target construction
# ---------------------------------------------------------------------------
def make_synthetic_leaf(organ_id: int = 1, n_skel: int = 21,
                         **deformation_kwargs) -> dict:
    """A gently curved blade with configurable deformations."""
    t = np.linspace(0.0, 1.0, n_skel)
    skel = np.column_stack([
        25.0 * t,
        0.4 * np.sin(np.pi * t),
        30.0 - 8.0 * t + 2.0 * t * t,
    ])
    widths = np.maximum(3.0 * (1 - np.abs(t - 0.5) * 1.2) * (1 - 0.2 * t), 0.2)
    organ = {
        "type": "leaf", "organ_id": organ_id,
        "skeleton": skel, "widths": widths,
        "_orig_segment_map": np.arange(n_skel - 1, dtype=np.int32),
    }
    organ.update(deformation_kwargs)
    return organ


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------
def bidirectional_chamfer(a: np.ndarray, b: np.ndarray) -> float:
    """Symmetric mean-Chamfer over two point clouds."""
    ta = KDTree(a)
    tb = KDTree(b)
    d_ab, _ = tb.query(a)
    d_ba, _ = ta.query(b)
    return float(0.5 * (d_ab.mean() + d_ba.mean()))


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------
def run_sweep(param_name: str, values: np.ndarray,
              target_deform: dict) -> dict:
    """Loft a target leaf once, then sweep one parameter and record losses."""
    target = make_synthetic_leaf(organ_id=0, **target_deform)
    tgt_verts, *_, tgt_cps = loft_leaf_nurbs(target)

    cp_losses: list[float] = []
    chamfer_losses: list[float] = []

    for v in values:
        pred = make_synthetic_leaf(organ_id=1,
                                    **{**target_deform, param_name: float(v)})
        pred_verts, *_, pred_cps = loft_leaf_nurbs(pred)

        cp_losses.append(
            cp_l2_loss({0: pred_cps}, {0: tgt_cps}, [(0, 0)],
                        reduction="mean")
        )
        chamfer_losses.append(bidirectional_chamfer(pred_verts, tgt_verts))

    return {
        "param": param_name,
        "values": values.tolist(),
        "target_deform": target_deform,
        "cp_l2": cp_losses,
        "chamfer": chamfer_losses,
        "n_u": N_U, "n_v": N_V,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def save_plot(result: dict, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.asarray(result["values"])
    cp = np.asarray(result["cp_l2"])
    ch = np.asarray(result["chamfer"])

    fig, (ax_cp, ax_ch) = plt.subplots(
        1, 2, figsize=(10, 4), constrained_layout=True)

    ax_cp.plot(x, cp, color="C0", marker="o", ms=3, lw=1.5)
    ax_cp.set_title("CP-L2 loss")
    ax_cp.set_xlabel(result["param"])
    ax_cp.set_ylabel("cp_l2_loss (mean per leaf)")
    ax_cp.grid(alpha=0.3)

    ax_ch.plot(x, ch, color="C3", marker="s", ms=3, lw=1.5)
    ax_ch.set_title("Bidirectional Chamfer (tessellated)")
    ax_ch.set_xlabel(result["param"])
    ax_ch.set_ylabel("chamfer (cm)")
    ax_ch.grid(alpha=0.3)

    fig.suptitle(
        f"Loss landscape along '{result['param']}' "
        f"(target deform: {result['target_deform']})",
        fontsize=10,
    )
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Smoothness diagnostic
# ---------------------------------------------------------------------------
def smoothness_receipt(values: np.ndarray, ys: np.ndarray) -> dict:
    """How jagged is the curve?  Count sign changes in its first derivative."""
    if len(values) < 3:
        return {"sign_changes": 0, "total_variation": 0.0}
    dy = np.diff(ys)
    tv = float(np.sum(np.abs(dy)))
    sc = int(np.sum(np.sign(dy[:-1]) * np.sign(dy[1:]) < 0))
    return {"sign_changes": sc, "total_variation": tv}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plot CP-L2 vs. Chamfer loss landscape along a parameter sweep.")
    parser.add_argument("--param", default="wave_normal_amp",
                        help="Deformation parameter to sweep. Must be accepted "
                             "by the NURBS lofter organ dict (e.g. "
                             "wave_normal_amp, twist_max, curl_amp).")
    parser.add_argument("--range", default="0.0,1.0,40",
                        help="Comma-separated 'lo,hi,n' (default: 0.0,1.0,40).")
    parser.add_argument("--target-amp", type=float, default=0.4,
                        help="Target value of the swept parameter (default: 0.4). "
                             "The sweep will straddle this value so both losses "
                             "have a minimum inside the range.")
    parser.add_argument("--target-extra", default="",
                        help="Extra ``key=value,...`` deformations applied to "
                             "both target and predictions.")
    parser.add_argument("--output", default="tools/output/landscape.png",
                        help="Output PNG path.")
    parser.add_argument("--json", default=None,
                        help="Optional JSON dump of the raw sweep data "
                             "(default: matches --output with .json).")
    args = parser.parse_args()

    lo_s, hi_s, n_s = args.range.split(",")
    lo, hi, n = float(lo_s), float(hi_s), int(n_s)
    values = np.linspace(lo, hi, n)

    # Target deformations: the swept param set to --target-amp, plus extras.
    target_deform: dict = {args.param: args.target_amp}
    for tok in (t.strip() for t in args.target_extra.split(",") if t.strip()):
        k, v = tok.split("=")
        target_deform[k.strip()] = float(v)

    print(f"Sweeping '{args.param}' across {n} values in [{lo}, {hi}] "
          f"(target {args.param} = {args.target_amp})...")
    result = run_sweep(args.param, values, target_deform)
    cp_smooth = smoothness_receipt(values, np.asarray(result["cp_l2"]))
    ch_smooth = smoothness_receipt(values, np.asarray(result["chamfer"]))
    result["cp_smoothness"] = cp_smooth
    result["chamfer_smoothness"] = ch_smooth

    out_png = Path(args.output)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    save_plot(result, out_png)

    out_json = Path(args.json) if args.json else out_png.with_suffix(".json")
    out_json.write_text(json.dumps(result, indent=2))

    print("\nSmoothness receipt (fewer sign-changes + lower TV ⇒ smoother):")
    print(f"  CP-L2:   sign_changes={cp_smooth['sign_changes']:3d}  "
          f"TV={cp_smooth['total_variation']:.4f}")
    print(f"  Chamfer: sign_changes={ch_smooth['sign_changes']:3d}  "
          f"TV={ch_smooth['total_variation']:.4f}")
    print(f"\nWrote: {out_png}")
    print(f"Wrote: {out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
