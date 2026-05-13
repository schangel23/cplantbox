"""Dampen uniform-halfwidth modes in a fitted parametric leaf shape distribution.

Sister of ``dampen_parametric_midrib.py``. The PCA fitter (fix path α, K=8
retained) keeps modes whose eigenvectors load almost entirely on the halfwidth
block with **all coefficients of the same sign** (i.e. the mode shifts the
*entire* halfwidth profile up or down uniformly). Per-plant Gaussian draws on
such a mode produce a bimodal "wide vs narrow" canopy: some seeds land at
z>0 → wider-than-mean leaves, others at z<0 → narrower-than-mean leaves, and
since maize XML has zero stochastic stddev on lmaxs/Width_blades, this is the
ONLY source of plant-to-plant width variance, so the bimodal split dominates
the population. Observed at scale=0.5 / day 180: seeds 42, 45 land in the
narrow mode (total area 4527/4797 cm² blade-only; 5074/5373 blade+midrib),
seeds 43, 44, 46 in the wide mode (total area 5558/5656/5200; 6233/6345/5841).

Detection criterion: ``|sum(hw_block)| / ||hw_block|| > threshold``. A value
near √n_cp ≈ 3.3 means all coefficients identical; 0 means the block is
balanced (some +, some -). Default threshold 0.5 catches uniformly-signed
modes while leaving "narrowing-toward-tip" / "widening-at-base" modes alone
(those have hw_signed_sum near zero because + and - coefficients cancel).

Usage:
    python -m dart.coupling.scripts.dampen_parametric_halfwidth \\
        --json dart/coupling/data/maize_leaf_shape_distribution.json \\
        --damp-factor 0.25 \\
        --threshold 0.5
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np


def dampen(json_path: Path, damp_factor: float, threshold: float) -> dict:
    d = json.loads(json_path.read_text())
    pca = d.get("pca_truncation")
    if pca is None or pca.get("K", 0) == 0:
        raise SystemExit(
            f"{json_path} has no PCA truncation block; nothing to dampen.")

    U = np.array(pca["pca_components"])
    ev = np.array(pca["pca_eigenvalues"])
    n_cp = d["n_cp_per_axis"]
    block = d["coeffs_block_layout"]
    hw_start, hw_end = block["halfwidth_norm"]

    new_ev = ev.copy()
    damped = []
    for k in range(len(ev)):
        u_k = U[k]
        hw = u_k[hw_start:hw_end]
        hw_norm = float(np.linalg.norm(hw))
        if hw_norm < 1e-12:
            continue
        hw_uniformity = float(abs(np.sum(hw)) / hw_norm)
        if hw_uniformity > threshold:
            new_ev[k] = ev[k] * damp_factor
            damped.append({
                "mode": k,
                "sigma_before": float(np.sqrt(ev[k])),
                "sigma_after": float(np.sqrt(new_ev[k])),
                "hw_norm": hw_norm,
                "hw_signed_sum": float(np.sum(hw)),
                "hw_uniformity": hw_uniformity,
            })

    if not damped:
        raise SystemExit(
            f"No modes had |sum(hw)|/||hw|| > {threshold}; nothing damped.")

    total_var = pca.get("total_variance",
                        float(np.sum(pca.get("all_eigenvalues_descending",
                                              ev.tolist()))))
    pca["pca_eigenvalues"] = new_ev.tolist()
    pca["retained_variance_fraction"] = float(np.sum(new_ev) / max(total_var, 1e-30))
    pca["halfwidth_damp_factor"] = damp_factor
    pca["halfwidth_damp_threshold"] = threshold
    pca["halfwidth_damp_modes"] = [int(m["mode"]) for m in damped]
    prev_reason = pca.get("drop_reason", "")
    pca["drop_reason"] = (
        prev_reason
        + f" | Uniform-halfwidth modes "
        f"(|sum(hw_block)|/||hw_block|| > {threshold} in eigenvector) had "
        f"their eigenvalues multiplied by {damp_factor} (sigma x "
        f"{np.sqrt(damp_factor):.2f}) to dampen the bimodal wide-vs-narrow "
        f"failure mode (seeds 42, 45 narrow vs 43, 44, 46 wide at scale=0.5 "
        f"day 180; total area split 4527-4797 vs 5558-5656 cm² blade-only)."
    )
    json_path.write_text(json.dumps(d, indent=2))
    return {"damped": damped, "retained": pca["retained_variance_fraction"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=Path, required=True)
    ap.add_argument("--damp-factor", type=float, default=0.25)
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    result = dampen(args.json, args.damp_factor, args.threshold)
    print(f"Damped {len(result['damped'])} modes (damp_factor={args.damp_factor}, "
          f"threshold={args.threshold}); retained variance = {result['retained']:.4f}")
    for m in result["damped"]:
        print(f"  mode {m['mode']}: sigma {m['sigma_before']:.3f} -> "
              f"{m['sigma_after']:.4f}  hw_uniformity={m['hw_uniformity']:.3f}  "
              f"hw_signed_sum={m['hw_signed_sum']:+.3f}")


if __name__ == "__main__":
    main()
