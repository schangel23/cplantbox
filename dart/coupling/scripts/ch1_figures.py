#!/usr/bin/env python3
"""Ch1 deliverable figures (PLAN_CH1_CARBON_DEMAND_2026-05-14 step 8).

Reads outputs from the G6 acceptance + drought-sweep runs and produces
four figures matching the plan-doc spec:

  fig1: Mainstem-height vs day        (G6 100-day trajectory)
  fig2: ψ_leaf vs day                 (G6 + drought ψ_init sweep)
  fig3: GPP / Rm / Rg stacked area    (G6 daily carbon partitioning)
  fig4: Biomass vs ψ_init panel       (drought sweep N=9 endpoint biomass)

Inputs expected (paths configurable; defaults match the production output
locations the diurnal pipeline writes):

  --g6-dir DIR        directory containing per-day {daily_summary.json,
                      hourly_results.csv, per_plant_carbon_pm_dayN.csv}
  --drought-dir DIR   directory containing one subdir per ψ_init with the
                      same per-day shape, plus a top-level summary.json
                      listing the {ψ_init: [seeds]} matrix
  --out DIR           where to write fig{1..4}.pdf and .png

Usage::

    cpbenv/bin/python dart/coupling/scripts/ch1_figures.py \\
        --g6-dir dart/coupling/output/g6_full_dumux/seed7 \\
        --drought-dir dart/coupling/output/drought_sweep \\
        --out figures/ch1

Each figure also writes a sidecar CSV with the underlying values so the
plot is reproducible without re-running the simulation.
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


def main():
    p = argparse.ArgumentParser(description="Ch1 deliverable figures")
    p.add_argument("--g6-dir", type=Path,
                   default=Path("dart/coupling/output/g6_full_dumux"))
    p.add_argument("--drought-dir", type=Path,
                   default=Path("dart/coupling/output/drought_sweep"))
    p.add_argument("--out", type=Path, default=Path("figures/ch1"))
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    df = _load_g6_daily(args.g6_dir)
    if df.empty:
        print(f"  no G6 daily data at {args.g6_dir}; nothing to plot",
              file=sys.stderr)
        return 1
    drought_df = _load_drought_endpoint(args.drought_dir)

    fig1_mainstem(df, args.out)
    fig2_psi_leaf(df, args.out, drought_df if not drought_df.empty else None)
    fig3_carbon_stack(df, args.out)
    fig4_drought_biomass(drought_df, args.out)
    print(f"  wrote figures to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
