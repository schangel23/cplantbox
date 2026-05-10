"""Dampen midrib-content modes in a fitted parametric leaf shape distribution.

Post-α fix from PLAN_PARAMETRIC_LEAF_SHAPE_2026-05-09_REV1 ("Post-α: midrib
eigenvalue damping" section). The PCA fitter (fix path α, K=8 retained) keeps
all modes — including modes whose eigenvectors concentrate on the midrib
coefficient blocks (droop / along). Per-plant Gaussian draws on those modes
produced upright-leaf failures at scale=0.3 (canopy5_day180_pca_clamp_scale0.3
seeds 42 and 45 had mean lateral extent 13.6 / 15.7 cm vs ~33 cm for the others
— the midrib's along-axis perturbation pulled leaf tips back toward the stem).

This script identifies eigenmodes with sqrt(droop² + along²) > 0.30 in the
eigenvector and multiplies their eigenvalues by a damping factor (default
0.01 = σ × 0.1) in place. K is unchanged. The fitter does not need to know;
this is a pure postprocessing step the user runs once after each refit.

Usage:
    python -m dart.coupling.scripts.dampen_parametric_midrib \\
        --json dart/coupling/data/maize_leaf_shape_distribution.json \\
        --damp-factor 0.01 \\
        --threshold 0.30
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

    U = np.array(pca["pca_components"])     # K rows × n_components
    ev = np.array(pca["pca_eigenvalues"])   # K
    n_cp = d["n_cp_per_axis"]

    new_ev = ev.copy()
    damped = []
    for k in range(len(ev)):
        u_k = U[k]
        droop_mag = float(np.linalg.norm(u_k[0:n_cp]))
        along_mag = float(np.linalg.norm(u_k[n_cp:2*n_cp]))
        midrib_mag = float(np.sqrt(droop_mag**2 + along_mag**2))
        if midrib_mag > threshold:
            new_ev[k] = ev[k] * damp_factor
            damped.append({
                "mode": k,
                "sigma_before": float(np.sqrt(ev[k])),
                "sigma_after": float(np.sqrt(new_ev[k])),
                "droop_mag": droop_mag,
                "along_mag": along_mag,
                "midrib_mag": midrib_mag,
            })

    if not damped:
        raise SystemExit(
            f"No modes had midrib_mag > {threshold}; nothing damped. "
            "Check the threshold or the eigenvector structure.")

    total_var = pca.get("total_variance",
                        float(np.sum(pca.get("all_eigenvalues_descending",
                                              ev.tolist()))))
    pca["pca_eigenvalues"] = new_ev.tolist()
    pca["retained_variance_fraction"] = float(np.sum(new_ev) / max(total_var, 1e-30))
    pca["midrib_damp_factor"] = damp_factor
    pca["midrib_damp_threshold"] = threshold
    pca["midrib_damp_modes"] = [int(m["mode"]) for m in damped]
    pca["drop_reason"] = (
        f"Midrib-content modes (sqrt(droop^2 + along^2) > {threshold} in "
        f"eigenvector) had their eigenvalues multiplied by {damp_factor} "
        f"(sigma x {np.sqrt(damp_factor):.2f}) to dampen midrib variance. "
        f"Eliminates the upright-leaf failure mode (Gaussian draws on droop "
        f"or along axes producing leaves with reduced lateral spread; "
        f"observed in canopy5_day180_pca_clamp_scale0.3 seeds 42, 45) while "
        f"preserving small biological midrib variation. See "
        f"PLAN_PARAMETRIC_LEAF_SHAPE_2026-05-09_REV1.md "
        f"section 'Post-alpha: midrib eigenvalue damping'."
    )
    json_path.write_text(json.dumps(d, indent=2))
    return {"damped": damped, "retained": pca["retained_variance_fraction"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=Path, required=True)
    ap.add_argument("--damp-factor", type=float, default=0.01)
    ap.add_argument("--threshold", type=float, default=0.30)
    args = ap.parse_args()

    result = dampen(args.json, args.damp_factor, args.threshold)
    print(f"Damped {len(result['damped'])} modes (damp_factor={args.damp_factor}, "
          f"threshold={args.threshold}); retained variance = {result['retained']:.4f}")
    for m in result["damped"]:
        print(f"  mode {m['mode']}: sigma {m['sigma_before']:.3f} -> "
              f"{m['sigma_after']:.4f}  droop={m['droop_mag']:.3f}  "
              f"along={m['along_mag']:.3f}")


if __name__ == "__main__":
    main()
