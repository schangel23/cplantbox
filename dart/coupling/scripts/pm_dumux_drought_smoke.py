"""Gate Ch1.PMDM.4 — non-trivial drought signal smoke (PM ↔ DuMux).

Plan-doc reference:
  ``Literature/Chapter 1/Concepts/CPBOXBALENOCOUPLING/
   PLAN_PIAFMUNCH_DUMUX_COUPLING_2026-05-09.md`` § "G4 — Non-trivial
  drought signal".

Setup:
- Plant: maize day-55 grown via ``grow_plant`` under BABST_MET (the
  same constant-met fixture used by ``pm_notebook_loop.case_maize`` so
  numbers are comparable to Gate 1-3 calibration runs).
- Soil grid: 3D rectangular box covering day-55 root spread, sized via
  ``--cell-number`` (default 4×4×25 = 400 cells over ±50 cm × ±50 cm
  lateral × 150 cm depth — drops resolution vs phase35_3d_smoke for
  wall-time, since the drought gate is a time-domain signal not a
  spatial pattern).
- BCs: no-flux top + no-flux bottom (closed system). The plan-doc
  spec is "no rain, no evaporation"; we additionally close the bottom
  so the only sink is RWU. This makes the column-mean drying signal
  unambiguous (free drainage would steal mass and confuse the
  drought-vs-drainage attribution).
- IC: ``--psi-init`` cm uniform (default -100 cm = moist, matches the
  plan-doc "starting from a moist column" spec).
- Cadence: ``--days`` outer iterations of
  ``solve_carbon_partitioning_pm`` (default 8). Each call runs 24
  hourly PM substeps with one DuMux advance per substep, advancing
  the plant by 1 day. Provider state persists across calls.

Acceptance gates (printed at the end as PASS / FAIL):
  G4.1: mean ψ_s in the root zone at day N below -500 cm (plant has
        actually pulled water — we started at -100 cm, so a 400+ cm
        drop demonstrates the soil↔plant water loop is closing).
  G4.2: minimum ψ_leaf (xylem) across the full window below -1000 cm
        (drought has reached the plant, not just the soil).
  G4.3: PM mass balance < 5 % on every day (Gate Ch1.PM.3 closure
        survives evolving ψ_s).
  G4.4: Q_Gr / Rg drops by > 30 % between day 1 and day N (drought-
        induced growth slowdown — the Ch2 deliverable signal).

Outputs:
- CSV: ``<output-dir>/pm_dumux_drought_<TAG>.csv`` with columns
  (day, psi_root_zone_mean_cm, psi_root_zone_min_cm,
   psi_leaf_min_cm, psi_leaf_max_cm, psi_leaf_mean_cm,
   An_total_mmol, Rg_total_mmol, Rm_total_mmol, C_ST_mean,
   integrated_rwu_cm3, integrated_transpiration_cm3,
   mass_balance_residual_pct).
- Plot (when ``matplotlib`` is importable): two-panel PNG with
  ψ_root_zone + ψ_leaf vs day, and Rg + mass-balance vs day.

Wall time:
- 4×4×25 grid, 24 substeps × 8 days, day-55 maize: ~10-15 min on
  a Ryzen workstation. Larger grids and longer windows scale linearly.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

# Force line-buffered stdout so progress lines appear immediately when the
# script is piped through ``tee`` or redirected to a file. Default Python
# block-buffers stdout on non-tty pipes, which would hide day-row progress
# until the buffer flushes (i.e. at end-of-program — useless for a 20-min
# run). Equivalent to running with ``PYTHONUNBUFFERED=1``.
try:
    sys.stdout.reconfigure(line_buffering=True)
except (AttributeError, OSError):
    pass

from dart.coupling.carbon.pm_substep import solve_carbon_partitioning_pm
from dart.coupling.config import DEFAULT_XML
from dart.coupling.growth.grow import grow_plant
from dart.coupling.hydraulics.soil_psi import (
    BC_CONSTANT_FLUX,
    DumuxSoilPsi,
)


BABST_MET = {
    d: {"T_mean_C": 20.75, "T_min_C": 19.0, "T_max_C": 22.0,
        "PAR_MJ_m2_d": 30.0 * 0.219, "VPD_kPa": 1.0,
        "RH_pct": 60.0, "Wind_m_s": 0.5}
    for d in range(1, 80)
}
TAIR_C = 20.75


def _root_zone_indices(provider, z_root_cm: float = -75.0) -> np.ndarray:
    """Return cellidxs whose centroid lies in the upper ``-z_root_cm`` cm
    of the column (where the day-55 maize root system is densest under
    BABST_MET).
    """
    nx, ny, nz = provider.cell_number
    minz = provider.min_b[2]
    maxz = provider.max_b[2]
    wz = (maxz - minz) / nz
    idxs = []
    for iz in range(nz):
        zc = minz + (iz + 0.5) * wz
        if zc >= z_root_cm:
            for iy in range(ny):
                for ix in range(nx):
                    idxs.append(iz * nx * ny + iy * nx + ix)
    return np.asarray(idxs, dtype=int)


def _synth_an_per_leaf(plant, an_total_mol: float = 0.025) -> np.ndarray:
    """Per-leaf-segment An vector summing to ``an_total_mol`` (mol CO2/d).

    Day-55 maize at saturating PAR transpires ~8-15 cm³/d and assimilates
    ~25 mmol CO2/d (matches Gate-2 day-55 An ≈ 23 mmol Suc × 12 mol CO2/mol
    Suc / 12 = ~23 mmol CO2 — i.e. PM-internal An; the caller's
    ``An_per_leaf_seg`` argument is stored as ``An_total_mmol_target``
    only, the loop computes its own An). 25 mmol CO2 = 0.025 mol.
    """
    n_leaf_segs = len(plant.getSegmentIds(4))
    if n_leaf_segs == 0:
        return np.array([], dtype=float)
    return np.full(n_leaf_segs, an_total_mol / n_leaf_segs, dtype=float)


def main() -> int:
    p = argparse.ArgumentParser(
        description="Gate Ch1.PMDM.4 — PM ↔ DuMux drought smoke",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--start-day", type=int, default=55,
                   help="Plant age at start of drought window")
    p.add_argument("--days", type=int, default=8,
                   help="Outer iteration count (1 day = 24 PM substeps).")
    p.add_argument("--n-substeps", type=int, default=24,
                   help="PM substeps per day. 24 = hourly (Gate 1-5 default)")
    p.add_argument("--cell-number", type=int, nargs=3, default=(4, 4, 25),
                   metavar=("NX", "NY", "NZ"),
                   help="DuMux 3D grid resolution. Phase 3.5 smoke uses "
                        "8x8x25; default here drops to 4x4x25 for wall-time")
    p.add_argument("--min-b", type=float, nargs=3, default=(-50.0, -50.0, -150.0),
                   metavar=("XMIN", "YMIN", "ZMIN"),
                   help="DuMux box min corner [cm], plant-relative")
    p.add_argument("--max-b", type=float, nargs=3, default=(50.0, 50.0, 0.0),
                   metavar=("XMAX", "YMAX", "ZMAX"),
                   help="DuMux box max corner [cm], plant-relative")
    p.add_argument("--psi-init", type=float, default=-100.0,
                   help="Uniform IC pressure head [cm]")
    p.add_argument("--z-root-cm", type=float, default=-75.0,
                   help="Root-zone depth threshold for ψ_root_zone aggregate")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tag", default="day55_drought",
                   help="Filename tag for outputs")
    p.add_argument("--output-dir", type=Path,
                   default=REPO_ROOT / "dart/coupling/output/drought_smoke",
                   help="Output directory for CSV + plot")
    p.add_argument("--no-plot", action="store_true",
                   help="Skip matplotlib plot generation")
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / f"pm_dumux_drought_{args.tag}.csv"
    plot_path = args.output_dir / f"pm_dumux_drought_{args.tag}.png"

    print("=" * 72)
    print(f"Gate Ch1.PMDM.4 — PM ↔ DuMux drought smoke")
    print("=" * 72)
    print(f"  start_day      : {args.start_day}")
    print(f"  days           : {args.days}")
    print(f"  n_substeps/day : {args.n_substeps}")
    print(f"  grid           : {tuple(args.cell_number)}, "
          f"box {tuple(args.min_b)} → {tuple(args.max_b)} cm")
    print(f"  psi_init       : {args.psi_init} cm (uniform)")
    print(f"  BCs            : no-flux top + no-flux bottom (closed system)")
    print(f"  output         : {csv_path}")
    print()

    # --- Grow plant + build provider with matching seg→cell mapping -----
    print(f"Growing maize day-{args.start_day} (seed={args.seed})...")
    t0 = time.time()
    plant = grow_plant(
        xml_path=str(DEFAULT_XML),
        simulation_time=args.start_day,
        min_stem_nodes=10,
        min_leaf_nodes=4,
        enable_photosynthesis=True,
        seed=args.seed,
        daily_met=BABST_MET,
        T_air_default=TAIR_C,
        soil_min_b=tuple(args.min_b),
        soil_max_b=tuple(args.max_b),
        soil_cell_number=tuple(args.cell_number),
    )
    print(f"  grew in {time.time()-t0:.1f}s; "
          f"n_segments={len(plant.getSegments())}, "
          f"n_leaf_segs={len(plant.getSegmentIds(4))}")

    # Sanity: every root segment should map into the soil grid. Out-of-grid
    # roots silently distort RWU because their flux is excluded from
    # ``integrated_rwu_cm3`` while still consuming xylem water.
    ot = np.asarray(plant.organTypes, dtype=int)
    n_root_out = sum(
        1 for s, c in plant.seg2cell.items()
        if c < 0 and int(ot[s]) == 2
    )
    if n_root_out > 0:
        print(f"  WARNING: {n_root_out} root segments fall outside the soil "
              f"grid — enlarge --min-b/--max-b or shorten --days. RWU "
              f"diagnostic will under-count.")

    print(f"\nBuilding DumuxSoilPsi (closed-system, IC={args.psi_init} cm)...")
    provider = DumuxSoilPsi(
        min_b=tuple(args.min_b),
        max_b=tuple(args.max_b),
        cell_number=tuple(args.cell_number),
        psi_init_cm=args.psi_init,
        top_bc=(BC_CONSTANT_FLUX, 0.0),
        bot_bc=(BC_CONSTANT_FLUX, 0.0),  # closed bottom = no drainage
        periodic=False,
    )
    provider._t_last_days = float(args.start_day)
    n_cells_total = provider.n_cells_total
    print(f"  n_cells_total={n_cells_total}")

    # Capture root-zone cellidx mask (cells with centroid above z_root_cm).
    # Used for the G4.1 ψ_root_zone diagnostic.
    rz_idx = _root_zone_indices(provider, z_root_cm=args.z_root_cm)
    print(f"  root-zone mask: {rz_idx.size}/{n_cells_total} cells "
          f"(centroid z ≥ {args.z_root_cm} cm)")

    # Initial ψ profile (before any sink push).
    psi_initial = provider.get_profile(
        t_days=float(args.start_day), depth_cm=n_cells_total,
    ).copy()
    print(f"  initial ψ_s    : "
          f"min={psi_initial.min():.1f}, mean={psi_initial.mean():.1f}, "
          f"max={psi_initial.max():.1f} cm")

    # --- Daily PM ↔ DuMux loop ------------------------------------------
    # Write CSV row-by-row so partial results survive an interrupt
    # (matters for a multi-day run that may exceed an hour).
    rows = []
    fieldnames = [
        "day",
        "psi_root_zone_mean_cm", "psi_root_zone_min_cm", "psi_root_zone_max_cm",
        "psi_column_mean_cm", "psi_column_min_cm",
        "psi_leaf_min_cm", "psi_leaf_max_cm", "psi_leaf_mean_cm",
        "An_total_mmol", "Rg_total_mmol", "Rm_total_mmol", "C_ST_mean",
        "integrated_rwu_cm3", "integrated_transpiration_cm3",
        "mass_balance_residual_pct", "wall_seconds",
    ]
    csv_file = csv_path.open("w", newline="")
    csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    csv_writer.writeheader()
    csv_file.flush()

    print(f"\n{'='*72}")
    print(f"{'day':>3}  {'ψ_rz_mean':>10}  {'ψ_leaf_min':>11}  "
          f"{'ψ_leaf_mean':>11}  {'An':>8}  {'Rg':>7}  "
          f"{'mb%':>5}  {'∫RWU':>8}  {'wall':>5}")
    print("-" * 72)

    for d_offset in range(args.days):
        day = args.start_day + d_offset
        t_day = time.time()
        An_per_leaf = _synth_an_per_leaf(plant, an_total_mol=0.025)
        if An_per_leaf.size == 0:
            print(f"  day {day}: no leaf segments left, aborting")
            return 1

        result = solve_carbon_partitioning_pm(
            plant, An_per_leaf, Tair_C=TAIR_C, day=day,
            n_substeps=args.n_substeps,
            soil_psi_provider=provider,
        )
        if result is None:
            print(f"  day {day}: PM solver failed, aborting")
            return 1

        # Read post-loop ψ_s. Flush the last substep's pending sink with
        # an extra dt advance so the soil state reflects the day's full
        # uptake.
        psi_post = provider.get_profile(
            t_days=float(day) + 1.0, depth_cm=n_cells_total,
        )
        rz_psi = psi_post[rz_idx]
        wall_d = time.time() - t_day

        row = {
            "day": day,
            "psi_root_zone_mean_cm": float(rz_psi.mean()),
            "psi_root_zone_min_cm": float(rz_psi.min()),
            "psi_root_zone_max_cm": float(rz_psi.max()),
            "psi_column_mean_cm": float(psi_post.mean()),
            "psi_column_min_cm": float(psi_post.min()),
            "psi_leaf_min_cm": (float(result["psi_leaf_min_cm"])
                                if result["psi_leaf_min_cm"] is not None
                                else float("nan")),
            "psi_leaf_max_cm": (float(result["psi_leaf_max_cm"])
                                if result["psi_leaf_max_cm"] is not None
                                else float("nan")),
            "psi_leaf_mean_cm": (float(result["psi_leaf_mean_cm"])
                                 if result["psi_leaf_mean_cm"] is not None
                                 else float("nan")),
            "An_total_mmol": float(result["An_total_mmol"]),
            "Rg_total_mmol": float(result["Rg_total_mmol"]),
            "Rm_total_mmol": float(result["Rm_total_mmol"]),
            "C_ST_mean": float(result["C_ST_mean"]),
            "integrated_rwu_cm3": float(result["integrated_rwu_cm3"]),
            "integrated_transpiration_cm3":
                float(result["integrated_transpiration_cm3"]),
            "mass_balance_residual_pct":
                float(result["mass_balance_residual_pct"]),
            "wall_seconds": wall_d,
        }
        rows.append(row)
        csv_writer.writerow(row)
        csv_file.flush()

        print(f"{day:>3d}  {row['psi_root_zone_mean_cm']:>10.1f}  "
              f"{row['psi_leaf_min_cm']:>11.1f}  "
              f"{row['psi_leaf_mean_cm']:>11.1f}  "
              f"{row['An_total_mmol']:>8.2f}  "
              f"{row['Rg_total_mmol']:>7.3f}  "
              f"{row['mass_balance_residual_pct']:>5.2f}  "
              f"{row['integrated_rwu_cm3']:>8.2f}  "
              f"{wall_d:>4.0f}s")

    csv_file.close()
    print(f"\nCSV: {csv_path}")

    # --- Acceptance gates ------------------------------------------------
    last = rows[-1]
    first = rows[0]

    g4_1 = last["psi_root_zone_mean_cm"] < -500.0
    g4_2_min = min(r["psi_leaf_min_cm"] for r in rows
                   if not np.isnan(r["psi_leaf_min_cm"]))
    g4_2 = g4_2_min < -1000.0
    g4_3 = max(abs(r["mass_balance_residual_pct"]) for r in rows) < 5.0
    rg_first = first["Rg_total_mmol"]
    rg_last = last["Rg_total_mmol"]
    if rg_first > 1e-9:
        rg_drop_pct = 100.0 * (rg_first - rg_last) / rg_first
    else:
        rg_drop_pct = float("nan")
    g4_4 = rg_drop_pct > 30.0

    print(f"\n{'='*72}")
    print("Acceptance gates")
    print("-" * 72)
    print(f"  G4.1  ψ_rz_mean(day {last['day']}) = "
          f"{last['psi_root_zone_mean_cm']:.1f} cm  < -500 cm "
          f"→ {'PASS' if g4_1 else 'FAIL'}")
    print(f"  G4.2  min ψ_leaf over window     = {g4_2_min:.1f} cm  "
          f"< -1000 cm → {'PASS' if g4_2 else 'FAIL'}")
    print(f"  G4.3  max |mass_balance_pct|      = "
          f"{max(abs(r['mass_balance_residual_pct']) for r in rows):.2f} %  "
          f"< 5 %  → {'PASS' if g4_3 else 'FAIL'}")
    print(f"  G4.4  Rg drop day {first['day']}→{last['day']}      = "
          f"{rg_drop_pct:.1f} %      > 30 % → "
          f"{'PASS' if g4_4 else 'FAIL'}")

    all_pass = g4_1 and g4_2 and g4_3 and g4_4
    print(f"\n  Overall: {'PASS' if all_pass else 'FAIL'}")

    # --- Optional plot ---------------------------------------------------
    if not args.no_plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("\n  (matplotlib not available, skipping plot)")
        else:
            days = np.array([r["day"] for r in rows])
            fig, axes = plt.subplots(2, 1, figsize=(8, 6.5), sharex=True)
            ax1, ax2 = axes

            ax1.plot(days, [r["psi_root_zone_mean_cm"] for r in rows],
                     "o-", label="ψ_root_zone mean", color="tab:brown")
            ax1.plot(days, [r["psi_root_zone_min_cm"] for r in rows],
                     "v--", label="ψ_root_zone min", color="tab:brown",
                     alpha=0.5)
            ax1.plot(days, [r["psi_leaf_min_cm"] for r in rows],
                     "s-", label="ψ_leaf min", color="tab:green")
            ax1.plot(days, [r["psi_leaf_mean_cm"] for r in rows],
                     "^-", label="ψ_leaf mean", color="tab:olive", alpha=0.7)
            ax1.axhline(-500, color="gray", ls=":", lw=0.8,
                        label="−500 cm (G4.1 gate)")
            ax1.axhline(-1000, color="gray", ls=":", lw=0.8)
            ax1.set_ylabel("Pressure head [cm]")
            ax1.legend(fontsize=8, loc="upper right")
            ax1.set_title(
                f"Gate Ch1.PMDM.4 drought smoke: day-{args.start_day} maize, "
                f"{args.days} days, IC={args.psi_init} cm closed system"
            )

            ax2_b = ax2.twinx()
            l1 = ax2.plot(days, [r["Rg_total_mmol"] for r in rows],
                          "o-", color="tab:blue", label="Rg [mmol CO2]")
            l2 = ax2.plot(days, [r["An_total_mmol"] for r in rows],
                          "s--", color="tab:cyan", label="An [mmol CO2]",
                          alpha=0.7)
            l3 = ax2_b.plot(
                days,
                [abs(r["mass_balance_residual_pct"]) for r in rows],
                "^:", color="tab:red", label="|mass-balance %|",
            )
            ax2.set_xlabel("Day")
            ax2.set_ylabel("Carbon flux [mmol CO2/d]")
            ax2_b.set_ylabel("|mass balance| [%]", color="tab:red")
            ax2_b.tick_params(axis="y", labelcolor="tab:red")
            ax2_b.axhline(5.0, color="tab:red", ls=":", lw=0.8)
            lines = l1 + l2 + l3
            ax2.legend(lines, [ln.get_label() for ln in lines],
                       fontsize=8, loc="upper left")

            fig.tight_layout()
            fig.savefig(plot_path, dpi=120)
            plt.close(fig)
            print(f"  plot: {plot_path}")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
