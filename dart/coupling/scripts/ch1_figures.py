#!/usr/bin/env python3
"""Ch1 deliverable figures.

Original PLAN_CH1_CARBON_DEMAND_2026-05-14 step 8 produced four figures:

  fig1: Mainstem-height vs day        (G6 100-day trajectory)
  fig2: ψ_leaf vs day                 (G6 + drought ψ_init sweep)
  fig3: GPP / Rm / Rg stacked area    (G6 daily carbon partitioning)
  fig4: Biomass vs ψ_init panel       (drought sweep N=9 endpoint biomass)

PLAN_BUFFERED_CARBON_GROWTH_2026-05-15 §S7-S8-Ch1 adds four buffered-
carbon closure figures driven off the §S7 calibration CSV + §S8 drought
sweep + a substep-resolved diel trace:

  fig_a_realised_vs_oracle  — day-130 cumulative realised length vs FA
                              oracle, per organ_type (leaf / stem /
                              root).  Reads the S7 calibration CSV.
  fig_b_drought_response    — ψ-sweep cumulative biomass + transpiration
                              from the S8 drought sweep.  Reads the
                              per-ψ CSV bundle.
  fig_c_diel_reserve        — substep-resolved transient_reserve_pool_
                              charge/drain across a representative
                              48-hour window.  Reads a diel-trace JSON
                              (written by ``capture_diel_trace.py`` or
                              the calibration script when
                              ``--diel-trace`` is set).
  fig_d_mb_audit_table      — table-style figure: per-organ-type
                              cumulative An / Rm / Rg / Δreserve /
                              Δlocal_C / storage_loss / remob_loss /
                              exudation at day 130.  Reads the S7 CSV.

Inputs (paths configurable; CLI flags below):

  --g6-dir DIR          legacy G6 daily-summary directory
  --drought-dir DIR     drought sweep root (one subdir per ψ_init)
  --s7-csv FILE         §S7 calibration CSV
                        (``out_calibration_s7_day130.csv``)
  --s8-csv FILE         §S8 drought sweep CSV (same shape as S7)
  --diel-trace FILE     JSON trace of substep-resolved reserve
                        dynamics (one record per substep)
  --out DIR             where to write fig{1..4} + fig_{a..d} .pdf/.png

Usage::

    cpbenv/bin/python dart/coupling/scripts/ch1_figures.py \\
        --s7-csv out_calibration_s7_day130.csv \\
        --s8-csv out_calibration_s8_drought.csv \\
        --diel-trace out_diel_trace.json \\
        --out latex/FSPM2026_Abstract/figures

Every figure writes a sidecar CSV so plots are reproducible without
re-running the simulation.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def _load_g6_daily(g6_dir: Path) -> pd.DataFrame:
    """Aggregate per-day daily_summary.json + per_plant_carbon CSVs.

    Expected layout::

        g6_dir/
            dayN/
                daily_summary.json     # has 'sim_day', 'mainstem_cm',
                                       # 'psi_leaf_min_cm', etc.
                per_plant_carbon_pm_dayN.csv  # cols: plant_idx, GPP, Rm,
                                              # Rg, daily_An_mol, ...
    """
    rows = []
    for day_dir in sorted(g6_dir.glob("day*")):
        ds = day_dir / "daily_summary.json"
        if not ds.exists():
            continue
        with ds.open() as f:
            d = json.load(f)
        row = {
            "day": int(d.get("sim_day", day_dir.name.removeprefix("day"))),
            "mainstem_cm": float(d.get("mainstem_cm", float("nan"))),
            "psi_leaf_min_cm": float(d.get("psi_leaf_min_cm", float("nan"))),
            "psi_leaf_mean_cm": float(d.get("psi_leaf_mean_cm", float("nan"))),
        }
        ppc = list(day_dir.glob("per_plant_carbon_pm_day*.csv"))
        if ppc:
            df = pd.read_csv(ppc[0])
            row["GPP_mmol"] = float(df.get("daily_An_mol", df.get("An_total_mmol", 0)).sum())
            row["Rm_mmol"] = float(df.get("Rm_total_mmol", 0).sum())
            row["Rg_mmol"] = float(df.get("Rg_total_mmol", 0).sum())
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("day").reset_index(drop=True)


def _load_drought_endpoint(drought_dir: Path) -> pd.DataFrame:
    """Read drought-sweep endpoint biomass by (ψ_init, seed)."""
    summary_path = drought_dir / "summary.json"
    if not summary_path.exists():
        print(f"  no drought summary at {summary_path}; skipping fig4",
              file=sys.stderr)
        return pd.DataFrame()
    with summary_path.open() as f:
        summary = json.load(f)
    rows = []
    for psi_str, seeds in summary.get("matrix", {}).items():
        psi_init = float(psi_str)
        for seed, end in seeds.items():
            rows.append({
                "psi_init_cm": psi_init,
                "seed": int(seed),
                "biomass_mmol": float(end.get("biomass_mmol", float("nan"))),
                "mainstem_cm": float(end.get("mainstem_cm", float("nan"))),
            })
    return pd.DataFrame(rows)


def fig1_mainstem(df: pd.DataFrame, out: Path):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(df["day"], df["mainstem_cm"], "o-", label="PM+DuMux")
    ax.set_xlabel("day")
    ax.set_ylabel("mainstem height [cm]")
    ax.set_title("Ch1 fig 1 — mainstem height vs day")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "fig1_mainstem.pdf")
    fig.savefig(out / "fig1_mainstem.png", dpi=150)
    df[["day", "mainstem_cm"]].to_csv(out / "fig1_mainstem.csv", index=False)
    plt.close(fig)


def fig2_psi_leaf(df: pd.DataFrame, out: Path,
                  drought_df: pd.DataFrame | None = None):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(df["day"], df["psi_leaf_min_cm"], "o-", label="ψ_leaf_min (G6)")
    ax.plot(df["day"], df["psi_leaf_mean_cm"], "s-", label="ψ_leaf_mean (G6)")
    if drought_df is not None and not drought_df.empty:
        # Drought sweep adds per-ψ_init endpoint markers as horizontal
        # bands.
        for psi in sorted(drought_df["psi_init_cm"].unique()):
            ax.axhline(psi, ls="--", alpha=0.4,
                       label=f"ψ_init={psi:.0f} cm")
    ax.set_xlabel("day")
    ax.set_ylabel("ψ_leaf [cm]")
    ax.set_title("Ch1 fig 2 — leaf water potential vs day")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig2_psi_leaf.pdf")
    fig.savefig(out / "fig2_psi_leaf.png", dpi=150)
    df[["day", "psi_leaf_min_cm", "psi_leaf_mean_cm"]].to_csv(
        out / "fig2_psi_leaf.csv", index=False,
    )
    plt.close(fig)


def fig3_carbon_stack(df: pd.DataFrame, out: Path):
    fig, ax = plt.subplots(figsize=(7, 4))
    cols = [c for c in ("Rm_mmol", "Rg_mmol") if c in df.columns]
    if not cols:
        print("  no carbon columns; skipping fig3", file=sys.stderr)
        plt.close(fig)
        return
    bottom = np.zeros(len(df))
    colors = {"Rm_mmol": "tab:orange", "Rg_mmol": "tab:green"}
    for c in cols:
        ax.fill_between(df["day"], bottom, bottom + df[c],
                        label=c.replace("_mmol", ""),
                        color=colors.get(c, "tab:gray"), alpha=0.7)
        bottom = bottom + df[c]
    if "GPP_mmol" in df.columns:
        ax.plot(df["day"], df["GPP_mmol"], "k--", label="GPP", linewidth=1.5)
    ax.set_xlabel("day")
    ax.set_ylabel("mmol CO2 / plant / day")
    ax.set_title("Ch1 fig 3 — daily carbon partitioning")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out / "fig3_carbon_stack.pdf")
    fig.savefig(out / "fig3_carbon_stack.png", dpi=150)
    df.to_csv(out / "fig3_carbon_stack.csv", index=False)
    plt.close(fig)


def fig4_drought_biomass(drought_df: pd.DataFrame, out: Path):
    if drought_df.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    psi_levels = sorted(drought_df["psi_init_cm"].unique())
    means = [drought_df[drought_df["psi_init_cm"] == p]["biomass_mmol"].mean()
             for p in psi_levels]
    stds = [drought_df[drought_df["psi_init_cm"] == p]["biomass_mmol"].std()
            for p in psi_levels]
    ax.errorbar(psi_levels, means, yerr=stds, fmt="o-", capsize=4)
    for _, row in drought_df.iterrows():
        ax.scatter(row["psi_init_cm"], row["biomass_mmol"],
                   alpha=0.4, color="tab:blue", s=15)
    ax.set_xlabel("ψ_init [cm]")
    ax.set_ylabel("end-of-season biomass [mmol CO2]")
    ax.set_title("Ch1 fig 4 — biomass vs ψ_init (drought sweep)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "fig4_biomass_psi.pdf")
    fig.savefig(out / "fig4_biomass_psi.png", dpi=150)
    drought_df.to_csv(out / "fig4_biomass_psi.csv", index=False)
    plt.close(fig)


# ----------------------------------------------------------------------
# §S7-S8-Ch1 buffered-carbon closure figures (a)-(d)
# ----------------------------------------------------------------------

def _best_row(s7_df: pd.DataFrame, mb_max: float = 1.0,
              fa_band: tuple[float, float] = (0.4, 0.9)) -> pd.Series | None:
    """Pick the row whose realised-FA fraction is closest to band midpoint
    AND has cumulative MB ≤ mb_max.  Returns None on empty."""
    if s7_df.empty:
        return None
    ok = s7_df[(s7_df["status"] == "OK")
               & (s7_df["cum_mb_residual_pct"] <= mb_max)]
    if ok.empty:
        return None
    mid = 0.5 * sum(fa_band)
    return ok.iloc[(ok["realised_fa_fraction"] - mid).abs().argsort().iloc[0]]


def fig_a_realised_vs_oracle(s7_df: pd.DataFrame, out: Path) -> None:
    """(a) Day-130 biomass realised vs FA oracle per organ.  Two
    side-by-side bars (realised vs oracle) for leaf / stem (mainstem) /
    root cumulative organ length.  Uses the best in-band row by default."""
    if s7_df.empty:
        return
    row = _best_row(s7_df)
    if row is None:
        # Fall back to the seed=7 / lowest-MB row so the figure is always
        # produced, but annotate that the band is not satisfied.
        row = s7_df.iloc[s7_df["cum_mb_residual_pct"].argmin()]
    organs = ["leaf", "mainstem", "root"]
    realised = [row["sum_leaf_realised_cm"],
                row["mainstem_realised_cm"],
                row["sum_root_realised_cm"]]
    oracle = [row["sum_leaf_oracle_cm"],
              row["mainstem_oracle_cm"],
              row["sum_root_oracle_cm"]]
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(organs))
    ax.bar(x - 0.18, oracle, width=0.36, label="FA oracle",
           color="tab:gray", alpha=0.7)
    ax.bar(x + 0.18, realised, width=0.36, label="closed-loop (buffered)",
           color="tab:green", alpha=0.85)
    ax.set_xticks(x, organs)
    ax.set_ylabel("Σ organ realised length [cm]")
    title = (f"Ch1 (a) — realised vs FA oracle, day-{int(row['sim_days'])} "
             f"(FA-frac={row['realised_fa_fraction']:.3f}, "
             f"MB={row['cum_mb_residual_pct']:.2f}%)")
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "fig_a_realised_vs_oracle.pdf")
    fig.savefig(out / "fig_a_realised_vs_oracle.png", dpi=150)
    pd.DataFrame({"organ": organs, "realised_cm": realised,
                  "oracle_cm": oracle,
                  "fraction": [r / o if o > 0 else float("nan")
                               for r, o in zip(realised, oracle)],
                  }).to_csv(out / "fig_a_realised_vs_oracle.csv", index=False)
    plt.close(fig)


def fig_b_drought_response(s8_df: pd.DataFrame, out: Path) -> None:
    """(b) Drought response curve: ψ-sweep cumulative biomass +
    transpiration proxy.  Expects the S8 sweep CSV (same schema as S7
    plus per-row soil_psi_cm).  Averages over seeds at each ψ."""
    if s8_df.empty:
        return
    s8_df = s8_df[s8_df["status"] == "OK"]
    if s8_df.empty:
        return
    grp = s8_df.groupby("soil_psi_cm")
    psi_levels = sorted(grp.groups.keys(), reverse=True)  # wet → dry
    biomass_mean = [grp.get_group(p)["total_realised_cm"].mean()
                    for p in psi_levels]
    biomass_std = [grp.get_group(p)["total_realised_cm"].std()
                   for p in psi_levels]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar(psi_levels, biomass_mean, yerr=biomass_std, fmt="o-",
                color="tab:green", capsize=4, label="Σ realised length")
    ax2 = ax.twinx()
    if "cum_an_mmol" in s8_df.columns:
        an_mean = [grp.get_group(p)["cum_an_mmol"].mean() for p in psi_levels]
        ax2.plot(psi_levels, an_mean, "s--", color="tab:blue",
                 label="cumulative An (proxy for transpiration)")
        ax2.set_ylabel("Σ An [mmol CO2 / plant]", color="tab:blue")
        ax2.tick_params(axis="y", colors="tab:blue")
    ax.set_xlabel("ψ_soil_init [cm]  (wet ← → dry)")
    ax.set_ylabel("Σ realised length [cm]", color="tab:green")
    ax.tick_params(axis="y", colors="tab:green")
    ax.set_title("Ch1 (b) — drought response (ψ-sweep)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "fig_b_drought_response.pdf")
    fig.savefig(out / "fig_b_drought_response.png", dpi=150)
    out_rows = pd.DataFrame({"psi_cm": psi_levels,
                             "biomass_mean_cm": biomass_mean,
                             "biomass_std_cm": biomass_std})
    out_rows.to_csv(out / "fig_b_drought_response.csv", index=False)
    plt.close(fig)


def fig_c_diel_reserve(trace_path: Path | None, out: Path) -> None:
    """(c) Diel reserve charge/drain trace.  Expects a JSON file with
    list-of-dicts schema::

        [{"sim_day": 35, "substep": 0, "An_mmol": ..., "Fu_lim": ...,
          "transient_reserve_pool_mmol": ..., "local_C_pool_total_mmol": ...,
          "is_light": True}, ...]

    If the trace JSON is missing or empty, the figure is skipped (the
    CLI surfaces a warning)."""
    if trace_path is None or not trace_path.exists():
        return
    with trace_path.open() as f:
        recs = json.load(f)
    if not recs:
        return
    df = pd.DataFrame(recs)
    if "transient_reserve_pool_mmol" not in df.columns:
        print("  diel trace missing transient_reserve_pool_mmol", file=sys.stderr)
        return
    df = df.sort_values(["sim_day", "substep"]).reset_index(drop=True)
    # Hour index from day-substep tuple.
    df["hour"] = (df["sim_day"] - df["sim_day"].min()) * 24 + df["substep"]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(df["hour"], df["transient_reserve_pool_mmol"], "-",
            color="tab:purple", label="transient_reserve_pool_")
    if "local_C_pool_total_mmol" in df.columns:
        ax.plot(df["hour"], df["local_C_pool_total_mmol"], "-",
                color="tab:orange", alpha=0.7, label="Σ local_C_pool_")
    # Shade night substeps (those flagged as ¬is_light if present, else
    # substeps with An < 1% of max).
    if "is_light" in df.columns:
        for _, r in df[df["is_light"] == False].iterrows():  # noqa: E712
            ax.axvspan(r["hour"] - 0.5, r["hour"] + 0.5,
                       color="gray", alpha=0.1, linewidth=0)
    ax.set_xlabel("hour from window start")
    ax.set_ylabel("[mmol Suc]")
    ax.set_title("Ch1 (c) — diel reserve + local-pool charge/drain")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out / "fig_c_diel_reserve.pdf")
    fig.savefig(out / "fig_c_diel_reserve.png", dpi=150)
    df.to_csv(out / "fig_c_diel_reserve.csv", index=False)
    plt.close(fig)


def fig_d_mb_audit_table(s7_df: pd.DataFrame, out: Path) -> None:
    """(d) MB audit table at day 130.  Aggregates per-knob-combo
    cumulative An / used and reports the closure budget.  Renders as a
    matplotlib table for the abstract figure-pack and as a sidecar CSV."""
    if s7_df.empty:
        return
    s7_df = s7_df[s7_df["status"] == "OK"]
    if s7_df.empty:
        return
    show = s7_df.sort_values("cum_mb_residual_pct").head(8).reset_index(drop=True)
    cols = ["c_cost_leaf", "c_cost_stem", "local_cap_factor",
            "reserve_cap_factor", "seed",
            "cum_an_mmol", "cum_used_mmol", "cum_mb_residual_pct",
            "realised_fa_fraction"]
    show = show[[c for c in cols if c in show.columns]]
    fig, ax = plt.subplots(figsize=(9, 0.45 * len(show) + 1.5))
    ax.axis("off")
    tbl = ax.table(
        cellText=[[f"{v:.4g}" if isinstance(v, (int, float)) else str(v)
                   for v in row] for row in show.values.tolist()],
        colLabels=list(show.columns),
        loc="center", cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1.05, 1.4)
    ax.set_title(
        "Ch1 (d) — MB audit (top-8 combos by Liebig closure)",
        pad=12,
    )
    fig.tight_layout()
    fig.savefig(out / "fig_d_mb_audit_table.pdf")
    fig.savefig(out / "fig_d_mb_audit_table.png", dpi=150)
    show.to_csv(out / "fig_d_mb_audit_table.csv", index=False)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="Ch1 deliverable figures")
    p.add_argument("--g6-dir", type=Path,
                   default=Path("dart/coupling/output/g6_full_dumux"))
    p.add_argument("--drought-dir", type=Path,
                   default=Path("dart/coupling/output/drought_sweep"))
    p.add_argument("--s7-csv", type=Path, default=None,
                   help="§S7 calibration CSV (drives figures a + d).")
    p.add_argument("--s8-csv", type=Path, default=None,
                   help="§S8 drought sweep CSV (drives figure b).")
    p.add_argument("--diel-trace", type=Path, default=None,
                   help="JSON trace of substep-resolved reserve dynamics "
                        "(drives figure c).")
    p.add_argument("--out", type=Path, default=Path("figures/ch1"))
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    # Legacy G6/drought-sweep figures.  Skipped silently when the
    # legacy directories don't exist (the §S7-S8 figure pack is the
    # canonical output now).
    legacy_df = _load_g6_daily(args.g6_dir)
    if legacy_df.empty:
        print(f"  no G6 daily data at {args.g6_dir}; legacy fig1-3 skipped",
              file=sys.stderr)
    else:
        legacy_drought_df = _load_drought_endpoint(args.drought_dir)
        fig1_mainstem(legacy_df, args.out)
        fig2_psi_leaf(legacy_df, args.out,
                      legacy_drought_df if not legacy_drought_df.empty else None)
        fig3_carbon_stack(legacy_df, args.out)
        fig4_drought_biomass(legacy_drought_df, args.out)

    # §S7-S8-Ch1 closure figures.
    s7_df = pd.read_csv(args.s7_csv) if args.s7_csv and args.s7_csv.exists() else pd.DataFrame()
    s8_df = pd.read_csv(args.s8_csv) if args.s8_csv and args.s8_csv.exists() else pd.DataFrame()
    if s7_df.empty:
        print(f"  no S7 CSV at {args.s7_csv}; figs (a) + (d) skipped",
              file=sys.stderr)
    else:
        fig_a_realised_vs_oracle(s7_df, args.out)
        fig_d_mb_audit_table(s7_df, args.out)
    if s8_df.empty:
        print(f"  no S8 CSV at {args.s8_csv}; fig (b) skipped",
              file=sys.stderr)
    else:
        fig_b_drought_response(s8_df, args.out)
    if args.diel_trace is None or not args.diel_trace.exists():
        print(f"  no diel trace at {args.diel_trace}; fig (c) skipped",
              file=sys.stderr)
    else:
        fig_c_diel_reserve(args.diel_trace, args.out)

    print(f"  wrote figures to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
