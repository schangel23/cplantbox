"""Shift the rank intercepts along uniform-halfwidth PCA modes to make plants wider on average.

Companion of ``dampen_parametric_halfwidth.py``. Where the damper REDUCES the variance
of bimodal width modes (squashing both wide and narrow tails toward the mean), this
script SHIFTS the mean so the narrow tail of the variation lands at-or-above the
original mean. Goal: keep full variation amplitude on modes that uniformly widen
or narrow the leaf, but bias the draw so plants are never narrower than the calibrated
baseline. Trade-off: plants now distribute upward of the old mean (the OLD mean
becomes the LOWER bound of the typical draw), and the average plant is wider.

For each PCA mode k satisfying |sum(hw_block)| / ||hw_block|| > threshold:
  intercept_new[r] = intercept_old[r] + c * sqrt(λ_k) * sign(sum(hw_block)) * U_k

With c = 1.0, a z = -1 draw on mode k now produces (intercept_new + (-1)·σ·U_k) =
intercept_old + 0 = the OLD mean. A z = +1 draw produces intercept_old + 2σ·U_k =
wider than ever. With c = 0.5, the OLD mean sits at z = -0.5 of the new distribution
(narrow tail extends below the old mean by 0.5σ).

This is purely a JSON edit; no C++ change needed. The C++ ``LeafShapeDistribution``
constructor consumes the shifted intercepts as already-fitted means.

Usage:
    python -m dart.coupling.scripts.shift_parametric_halfwidth_intercept \\
        --json dart/coupling/data/maize_leaf_shape_distribution.json \\
        --shift-c 1.0 \\
        --threshold 0.5
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np


def shift(json_path: Path, shift_c: float, threshold: float) -> dict:
    d = json.loads(json_path.read_text())
    pca = d.get("pca_truncation")
    if pca is None or pca.get("K", 0) == 0:
        raise SystemExit(
            f"{json_path} has no PCA truncation block; nothing to shift.")

    U = np.array(pca["pca_components"])
    ev = np.array(pca["pca_eigenvalues"])
    n_cp = d["n_cp_per_axis"]
    hw_start, hw_end = d["coeffs_block_layout"]["halfwidth_norm"]

    shifted_modes = []
    total_shift = np.zeros(U.shape[1])
    for k in range(len(ev)):
        u_k = U[k]
        hw = u_k[hw_start:hw_end]
        hw_norm = float(np.linalg.norm(hw))
        if hw_norm < 1e-12:
            continue
        hw_sum = float(np.sum(hw))
        hw_uniformity = abs(hw_sum) / hw_norm
        if hw_uniformity <= threshold:
            continue
        sign = 1.0 if hw_sum > 0 else -1.0
        sigma = float(np.sqrt(max(ev[k], 0.0)))
        # Shift in coeff space along the widening direction
        delta = shift_c * sigma * sign * u_k
        total_shift += delta
        shifted_modes.append({
            "mode": k,
            "sigma": sigma,
            "hw_uniformity": hw_uniformity,
            "hw_signed_sum": hw_sum,
            "sign": int(sign),
            "delta_norm": float(np.linalg.norm(delta)),
        })

    if not shifted_modes:
        raise SystemExit(
            f"No modes had |sum(hw)|/||hw|| > {threshold}; nothing shifted.")

    # Apply the shift to every rank's intercept
    intercepts = d["intercepts"]
    for rank_key, intercept_list in intercepts.items():
        arr = np.array(intercept_list)
        if arr.shape != total_shift.shape:
            raise SystemExit(
                f"Rank {rank_key} intercept shape {arr.shape} != shift shape "
                f"{total_shift.shape}")
        d["intercepts"][rank_key] = (arr + total_shift).tolist()

    pca["halfwidth_intercept_shift_c"] = shift_c
    pca["halfwidth_intercept_shift_threshold"] = threshold
    pca["halfwidth_intercept_shift_modes"] = [int(m["mode"]) for m in shifted_modes]
    prev_reason = pca.get("drop_reason", "")
    pca["drop_reason"] = (
        prev_reason
        + f" | Intercepts shifted by +c·σ_k·sign(hw_sum)·U_k for "
        f"modes {[m['mode'] for m in shifted_modes]} "
        f"(c={shift_c}, threshold={threshold}); bakes a mean-widening offset "
        f"into per-rank intercepts so the narrow tail of mode-k draws lands at "
        f"the old mean (or above), eliminating sub-baseline narrow plants while "
        f"preserving full PCA variance. Width distribution now skews wider on "
        f"average."
    )
    json_path.write_text(json.dumps(d, indent=2))
    return {"shifted": shifted_modes, "total_shift_norm": float(np.linalg.norm(total_shift))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=Path, required=True)
    ap.add_argument("--shift-c", type=float, default=1.0)
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    result = shift(args.json, args.shift_c, args.threshold)
    print(f"Shifted intercepts along {len(result['shifted'])} modes "
          f"(c={args.shift_c}, threshold={args.threshold}); "
          f"total shift norm = {result['total_shift_norm']:.4f}")
    for m in result["shifted"]:
        print(f"  mode {m['mode']}: sigma {m['sigma']:.4f}  "
              f"hw_uniformity={m['hw_uniformity']:.3f}  "
              f"sign={m['sign']:+d}  delta_norm={m['delta_norm']:.4f}")


if __name__ == "__main__":
    main()
