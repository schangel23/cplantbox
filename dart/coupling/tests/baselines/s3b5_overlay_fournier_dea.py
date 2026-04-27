#!/usr/bin/env python3
"""S3b.5 overlay — sim per-rank target (post-S3b.3 FA-on) vs Fournier 2000 Déa.

Plan: PLAN_FULL_B35_PER_PHYTOMER_BOOKKEEPING_2026-04-23.md §D (S3b.5).

Produces the Ch2 Fig Xb "money figure": per-rank τ_n-axis overlay of the FA
kinetic target against Fournier 2000 Fig 6A Déa observations. Also records
achieved-vs-target residual as S3b.3-downgrade diagnostic.

**What S3b.5 validates.** Under the S3b.3 pragmatic scope downgrade
(`project_fa_s3b3_shipped.md`), `stem.get_phytomer_length(n)` reports the
*scalar-allocator span* for rank n (~uniform across emerged ranks) because
the true per-phytomer mid-stem insertion driver deadlocked on leaf-emergence
↔ FA-kinetic chicken-and-egg. The FA-kinetic signal — the claim S3b.5's
per-rank Déa test exists to validate — lives in `calcLengthPerPhytomer(n)`
(the kinetic target), not in the scalar-allocated achieved span. So the
primary Déa overlay is target-vs-obs; achieved-vs-target is a diagnostic
pane.

**Outputs:**
  * `d2_htt_per_rank_fournier_dea.json` — per-rank RMSE@Δ + peaks + Δ_n,
    frozen as the S3b.5 acceptance baseline
  * `d2_htt_per_rank_plot_s3b.png` — 7-panel per-rank overlay (sim target
    shifted by Δ_n onto Fournier's axis) + achieved-vs-target diagnostic

Run (from /home/lukas/PHD/CPlantBox):
    cpbenv/bin/python3 dart/coupling/tests/baselines/s3b5_overlay_fournier_dea.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

BASELINE_DIR = Path(__file__).resolve().parent
SIM_JSON = BASELINE_DIR / "s3b5_achieved_per_rank.json"
OBS_JSON = BASELINE_DIR / "fournier2000_dea_fig6a_per_rank.json"
OUT_JSON = BASELINE_DIR / "d2_htt_per_rank_fournier_dea.json"
OUT_PNG = BASELINE_DIR / "d2_htt_per_rank_plot_s3b.png"

# Validation tolerances — see plan §D.
RMSE_MAX_PER_RANK_CM = 2.5           # primary: τ_n-axis RMSE per rank 9–15
ABS_AXIS_RESIDUAL_PCT = 25.0         # secondary (non-blocking): shared-axis residual vs obs peak


def load_sim():
    d = json.loads(SIM_JSON.read_text())
    tt_a = np.array([x["tt_andrieu"] for x in d["trajectory"]], dtype=float)
    n_ranks = d["n_ranks"]
    target = {n: np.array([x["target_per_rank_cm"][n - 1] for x in d["trajectory"]], dtype=float)
              for n in range(1, n_ranks + 1)}
    achieved = {n: np.array([x["achieved_per_rank_cm"][n - 1] for x in d["trajectory"]], dtype=float)
                for n in range(1, n_ranks + 1)}
    return tt_a, target, achieved, d


def load_obs():
    d = json.loads(OBS_JSON.read_text())
    tt = np.array(d["tt_deg_cd"], dtype=float)
    per_rank = {int(k): np.array(v, dtype=float) for k, v in d["per_rank_il_cm"].items()}
    return tt, per_rank


def best_offset(sim_tt, sim_il, obs_tt, obs_il, search_range=(-100.0, 500.0), step=1.0):
    """Find delta minimizing RMSE of sim_il(obs_tt + delta) vs obs_il.

    Only observation points where obs_il > 0.5 cm contribute — below that, both
    curves are near zero and the fit has no leverage.

    Returns (delta, rmse_at_delta, rmse_at_zero_offset).
    """
    active = obs_il > 0.5
    if active.sum() < 3:
        return np.nan, np.nan, np.nan
    best_delta = 0.0
    best_rmse = np.inf
    deltas = np.arange(search_range[0], search_range[1] + step, step)
    for delta in deltas:
        interp_sim = np.interp(obs_tt[active] + delta, sim_tt, sim_il, left=0.0, right=np.nan)
        if np.any(np.isnan(interp_sim)):
            continue
        rmse = float(np.sqrt(np.mean((interp_sim - obs_il[active]) ** 2)))
        if rmse < best_rmse:
            best_rmse = rmse
            best_delta = delta
    rmse_zero = float(np.sqrt(np.mean(
        (np.interp(obs_tt[active], sim_tt, sim_il, left=0.0, right=np.nan) - obs_il[active]) ** 2
    )))
    return best_delta, best_rmse, rmse_zero


def abs_axis_residual(sim_tt, sim_il, obs_tt, obs_il, delta):
    """Residual at each obs point after applying (shared) mean delta.

    Returns array of abs residuals normalised by obs peak, in percent.
    """
    active = obs_il > 0.5
    if active.sum() < 1:
        return np.array([])
    interp_sim = np.interp(obs_tt[active] + delta, sim_tt, sim_il, left=0.0, right=np.nan)
    peak = obs_il.max() if obs_il.max() > 0 else 1.0
    return np.abs(interp_sim - obs_il[active]) / peak * 100.0


def main():
    sim_tt, sim_target, sim_achieved, sim_meta = load_sim()
    obs_tt, obs_per_rank = load_obs()

    ranks = sorted(obs_per_rank.keys())
    print(f"S3b.5 overlay: per-rank target (post-S3b.3) vs Fournier 2000 Fig 6A Déa ({len(ranks)} ranks)")
    print(f"  primary bound:  RMSE@Δ ≤ {RMSE_MAX_PER_RANK_CM:.1f} cm per rank (τ_n-axis collapse)")
    print(f"  secondary bound (non-blocking): |residual| / obs_peak ≤ {ABS_AXIS_RESIDUAL_PCT:.0f}% at each sample (mean Δ)")
    print()

    results = {}
    for n in ranks:
        sim_il = sim_target[n]
        obs_il = obs_per_rank[n]
        delta, rmse_at_delta, rmse_at_zero = best_offset(sim_tt, sim_il, obs_tt, obs_il)
        results[n] = {
            "delta_tt_cd": float(delta),
            "rmse_at_delta_cm": float(rmse_at_delta),
            "rmse_at_zero_offset_cm": float(rmse_at_zero),
            "obs_peak_cm": float(obs_il.max()),
            "sim_target_peak_cm": float(sim_il.max()),
            "sim_achieved_peak_cm": float(sim_achieved[n].max()),
            "primary_pass": bool(rmse_at_delta <= RMSE_MAX_PER_RANK_CM),
        }
        status = "PASS" if results[n]["primary_pass"] else "FAIL"
        print(f"  rank {n:2d}: obs peak {obs_il.max():5.1f}  tgt peak {sim_il.max():5.1f}  "
              f"ach peak {sim_achieved[n].max():5.1f}  best Δ={delta:+6.1f} °Cd  "
              f"RMSE(Δ)={rmse_at_delta:4.2f} cm  [{status}]")

    deltas = np.array([r["delta_tt_cd"] for r in results.values() if not np.isnan(r["delta_tt_cd"])])
    mean_delta = float(deltas.mean()) if deltas.size else 0.0
    std_delta = float(deltas.std()) if deltas.size else 0.0

    # Secondary (non-blocking): max per-rank residual at mean Δ.
    max_residual_pct_per_rank = {}
    for n in ranks:
        res = abs_axis_residual(sim_tt, sim_target[n], obs_tt, obs_per_rank[n], mean_delta)
        max_residual_pct_per_rank[n] = float(res.max()) if res.size else float("nan")
    secondary_max = max(v for v in max_residual_pct_per_rank.values() if not np.isnan(v))

    all_primary_pass = all(r["primary_pass"] for r in results.values())
    secondary_pass = secondary_max <= ABS_AXIS_RESIDUAL_PCT

    print()
    print(f"Mean per-rank offset: Δ = {mean_delta:+6.1f} ± {std_delta:4.1f} °Cd")
    print(f"Secondary: max residual at mean Δ = {secondary_max:.1f}%  "
          f"[{'PASS' if secondary_pass else 'FLAG (non-blocking)'}]")
    print(f"Primary (τ_n-axis RMSE ≤ {RMSE_MAX_PER_RANK_CM} cm per rank 9–15): "
          f"{'PASS' if all_primary_pass else 'FAIL'}")

    # Achieved-vs-target diagnostic (S3b.3 downgrade context).
    ach_vs_tgt_day130 = {}
    for n in range(1, sim_meta["n_ranks"] + 1):
        ach_final = float(sim_achieved[n][-1])
        tgt_final = float(sim_target[n][-1])
        ach_vs_tgt_day130[n] = {
            "achieved_cm": ach_final,
            "target_cm": tgt_final,
            "delta_cm": ach_final - tgt_final,
        }

    out = {
        "_meta": {
            "source_sim_json": SIM_JSON.name,
            "source_obs_json": OBS_JSON.name,
            "sim_build": "S3b.3 post-S3b.4 sign-off (FA-on maize_calibrated.xml, seed=7)",
            "validation_quantity": "calcLengthPerPhytomer(n) — FA kinetic target curve",
            "s3b3_downgrade_note": (
                "get_phytomer_length(n) reports scalar-allocator span per "
                "project_fa_s3b3_shipped.md; achieved-vs-target residual logged "
                "but NOT validated against Déa. Primary Déa test is target-vs-obs."
            ),
            "primary_tolerance_cm": RMSE_MAX_PER_RANK_CM,
            "secondary_tolerance_pct": ABS_AXIS_RESIDUAL_PCT,
        },
        "per_rank": {int(n): results[n] for n in ranks},
        "mean_delta_tt_cd": mean_delta,
        "std_delta_tt_cd": std_delta,
        "max_residual_pct_at_mean_delta": max_residual_pct_per_rank,
        "secondary_pass": secondary_pass,
        "primary_pass": all_primary_pass,
        "achieved_vs_target_day130": ach_vs_tgt_day130,
    }
    OUT_JSON.write_text(json.dumps(out, indent=2, default=float))
    print(f"\nSaved acceptance baseline to {OUT_JSON}")

    # --------- PNG render ---------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib unavailable — skipping PNG render")
        return 0 if all_primary_pass else 1

    ncols = 4
    nrows = (len(ranks) + ncols - 1) // ncols + 1  # extra row for achieved-vs-target diagnostic
    fig = plt.figure(figsize=(4 * ncols, 3 * nrows))

    # Per-rank panels
    for idx, n in enumerate(ranks):
        ax = fig.add_subplot(nrows, ncols, idx + 1)
        r = results[n]
        delta = r["delta_tt_cd"]
        ax.plot(sim_tt - delta, sim_target[n], "b-", lw=1.4,
                label=f"sim target (Δ={delta:+.0f})")
        ax.plot(sim_tt - delta, sim_achieved[n], "r-", lw=0.9, alpha=0.55,
                label="sim achieved (S3b.3 span)")
        ax.plot(obs_tt, obs_per_rank[n], "k^", ms=5, label="Fournier 2000 obs")
        ax.fill_between(obs_tt, obs_per_rank[n] - 1.0, obs_per_rank[n] + 1.0,
                        color="gray", alpha=0.2, label="±1 cm digitisation")
        passed = "PASS" if r["primary_pass"] else "FAIL"
        ax.set_title(f"rank {n}  RMSE={r['rmse_at_delta_cm']:.2f} cm  [{passed}]", fontsize=10)
        ax.set_xlim(400, 800)
        ax.set_ylim(-1, 28)
        ax.grid(alpha=0.3)
        if idx == 0:
            ax.legend(fontsize=7, loc="upper left")
        if idx % ncols == 0:
            ax.set_ylabel("IL_n (cm)")
        if idx >= len(ranks) - ncols:
            ax.set_xlabel("TT since emergence (°Cd)")

    # Achieved-vs-target diagnostic row (bottom)
    ax_diag = fig.add_subplot(nrows, 1, nrows)
    ranks_full = list(range(1, sim_meta["n_ranks"] + 1))
    ach_final = [sim_achieved[n][-1] for n in ranks_full]
    tgt_final = [sim_target[n][-1] for n in ranks_full]
    obs_peak = [obs_per_rank[n].max() if n in obs_per_rank else np.nan for n in ranks_full]
    ax_diag.bar(np.array(ranks_full) - 0.25, tgt_final, width=0.25, color="C0", label="sim target (FA kinetic)")
    ax_diag.bar(np.array(ranks_full), ach_final, width=0.25, color="C3", alpha=0.7,
                label="sim achieved (S3b.3 scalar span)")
    ax_diag.bar(np.array(ranks_full) + 0.25, obs_peak, width=0.25, color="0.3", label="Fournier Déa peak")
    ax_diag.set_xlabel("rank n")
    ax_diag.set_ylabel("IL_n at day 130 (cm)")
    ax_diag.set_title("S3b.3 downgrade diagnostic: achieved reflects scalar allocation, not FA kinetic. "
                      "Primary Déa test is target (blue) vs obs (grey).", fontsize=10)
    ax_diag.legend(fontsize=9, loc="upper right")
    ax_diag.grid(alpha=0.3, axis="y")
    ax_diag.set_xticks(ranks_full)

    fig.suptitle(
        f"S3b.5: per-rank target (post-S3b.3 FA-on) vs Fournier 2000 Déa Fig 6A\n"
        f"τ_n-axis RMSE {'PASS' if all_primary_pass else 'FAIL'} "
        f"(≤{RMSE_MAX_PER_RANK_CM} cm per rank 9–15); "
        f"Δ = {mean_delta:+.0f} ± {std_delta:.0f} °Cd (plastochron drift, S3b.1 documented)",
        fontsize=11)
    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    plt.savefig(OUT_PNG, dpi=130)
    print(f"Saved overlay to {OUT_PNG}")

    return 0 if all_primary_pass else 1


if __name__ == "__main__":
    sys.exit(main())
