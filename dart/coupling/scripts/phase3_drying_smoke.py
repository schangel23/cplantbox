"""Phase 3 — 30-day drying smoke (local, no DART/Baleno).

Two DuMux treatments at maize-day-55, one midday `run_photosynthesis_solve`
per day. Both start uniform at -300 cm and share the same column geometry +
RWU sink-term feedback (Phase 2.9); only the top boundary differs:

    [WET]    DumuxSoilPsi  top = Dirichlet ψ = -300 cm  bot = free-drainage
             — surface re-wets to compensate for RWU; column stays near IC.
    [DRY]    DumuxSoilPsi  top = zero-flux                bot = free-drainage
             — no recharge; RWU drives the column drier over time.

Earlier draft used `FixedSoilPsi(-300)` as the wet control, but that returns
the legacy `np.linspace(-300, -400, 100)` which embeds an artificial -100 cm
"gradient" making deep cells drier than DuMux's uniform IC. Result: deep
roots saw more suction in the legacy wet control than in the DuMux dry
treatment, inverting the drought signal. Same-physics top-BC contrast
removes that artefact.

Smoke gates. Per-day Vcmax / Chl is time-varying (LOPS profile in
coupled.py), so within-treatment day-N vs day-0 comparisons conflate
drought with chemistry — G2/G3 compare DRY vs WET at the **same day**.
G1 is a one-shot legacy-vs-DuMux convention parity check.

    G1  One-shot:  FixedSoilPsi(-300) An ≡ DumuxSoilPsi(uniform -300) An
        within 1 % at day 0
        (matric/total convention; confirms no +z correction in get_profile.)
    G2  Day-N [DRY] transp < 0.95 × Day-N [WET] transp
        (drying engages relative to the same-physics wet control.)
    G3  Day-N [DRY] |ψ_leaf_min| > 1.05 × Day-N [WET] |ψ_leaf_min|
        (water-stress propagates to the leaf.)

What this does NOT prove
    • Actual SIF response — η is computed by Baleno's vegetation plugin,
      not exposed by the C++ Photosynthesis binding. Once the upstream
      chain is verified here, run a full diurnal on the server with
      `--soil-mode dumux --with-sif` to read out ψ→SIF directly.
    • Realistic daily transpiration — we integrate a constant midday
      PAR over 24 h, so daily uptake is ≈ 2-3× a realistic clearsky
      integral. This accelerates drying (by design — the smoke wants
      a signal in 30 days, not 90).

Outputs
    output/phase3_drying/phase3_drying_metrics.csv
    output/phase3_drying/phase3_drying.png      (4 panels)
    output/phase3_drying/phase3_drying_soil_profiles.png

Usage
    python -m dart.coupling.scripts.phase3_drying_smoke              # 30 days
    python -m dart.coupling.scripts.phase3_drying_smoke --days 5     # quick
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from dart.coupling.growth.grow import grow_plant
from dart.coupling.hydraulics.soil_psi import (
    BC_CONSTANT_FLUX, BC_CONSTANT_PRESSURE, BC_FREE_DRAINAGE,
    DumuxSoilPsi, FixedSoilPsi,
)
from dart.coupling.photosynthesis.coupled import run_photosynthesis_solve

XML_PATH = Path(__file__).resolve().parents[1] / "data" / "maize_calibrated.xml"
OUT_DIR = Path(__file__).resolve().parents[1] / "output" / "phase3_drying"

START_DAY = 55          # baseline plant age before drying loop
PSI_INIT_CM = -300.0    # well-watered initial soil head
PAR_MIDDAY = 1500.0     # µmol m⁻² s⁻¹
TLEAF_C = 25.0
RH = 0.7
DEPTH_CM = 100          # locked to coupled.py picker convention (z ∈ [-100, 0])
N_CELLS_Z = 100
COL_HALF_WIDTH_CM = 2.5 # 5×5 cm cross-section (≈ single-plant rhizosphere).
                        # 5×5×100 = 2500 cm³ column; at θ≈0.20 → ~500 cm³ water.
                        # 9.7 cm³/d uptake × 30 days = 290 cm³ ≈ 58 % depletion,
                        # so a 30-day drying signal is measurable. Default
                        # DumuxSoilPsi uses 10×10 cm (field-scale per plant);
                        # too dilute to produce signal in 30 days here.


def _summarise_solve(provider, sim_time: float, res: dict) -> dict:
    """Pull scalar metrics out of a run_photosynthesis_solve result.

    `provider.get_profile` is idempotent at unchanged t_days, so calling
    it here just reads out the post-solve soil state.
    """
    psi_leaf = np.asarray(res["psi_leaf_MPa"])
    profile_cm = provider.get_profile(t_days=float(sim_time),
                                      depth_cm=DEPTH_CM)
    return {
        "day": float(sim_time),
        "An_total_mmol": float(res["An_total_mmol"]),
        "transp_mmol": float(res["transp_mmol"]),
        "psi_leaf_mean_MPa": float(np.mean(psi_leaf)),
        "psi_leaf_min_MPa": float(np.min(psi_leaf)),
        "psi_leaf_max_MPa": float(np.max(psi_leaf)),
        "soil_psi_top_cm": float(profile_cm[0]),
        "soil_psi_mid_cm": float(profile_cm[DEPTH_CM // 2]),
        "soil_psi_bot_cm": float(profile_cm[-1]),
    }


def _run_treatment(plant, provider, label: str, n_days: int,
                   profile_log: list[tuple[float, np.ndarray]]) -> list[dict]:
    rows = []
    for d in range(n_days + 1):  # inclusive: d = 0 .. n_days
        sim_time = START_DAY + d
        res = run_photosynthesis_solve(
            plant, sim_time,
            par=PAR_MIDDAY, tleaf=TLEAF_C, label=f"{label}_d{d:02d}",
            rh=RH, soil_psi_provider=provider,
        )
        if res is None:
            print(f"  [{label}] day {d}: solve FAILED, aborting treatment")
            break
        row = _summarise_solve(provider, sim_time, res)
        row["treatment"] = label
        rows.append(row)
        # Snapshot full soil profile every 5 days for the depth panel.
        if d % 5 == 0 or d == n_days:
            profile_cm = provider.get_profile(float(sim_time), DEPTH_CM)
            profile_log.append((float(sim_time), profile_cm.copy()))
    return rows


def _align_provider_t0(provider, t0_days: float) -> None:
    """Tell DumuxSoilPsi that 'now' is t0_days, not the absolute t=0 default,
    so the first get_profile(t_days=t0_days) call advances dt=0 (no-op)."""
    if hasattr(provider, "_t_last_days"):
        provider._t_last_days = float(t0_days)


def _save_csv(rows_wet: list[dict], rows_dry: list[dict], path: Path) -> None:
    fields = [
        "treatment", "day", "An_total_mmol", "transp_mmol",
        "psi_leaf_mean_MPa", "psi_leaf_min_MPa", "psi_leaf_max_MPa",
        "soil_psi_top_cm", "soil_psi_mid_cm", "soil_psi_bot_cm",
    ]
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows_wet + rows_dry:
            w.writerow({k: r[k] for k in fields})


def _save_trajectory_figure(rows_wet, rows_dry, path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    def _xy(rows, key):
        days = [r["day"] - START_DAY for r in rows]
        ys = [r[key] for r in rows]
        return days, ys

    panels = [
        (axes[0, 0], "psi_leaf_min_MPa", "min leaf ψ [MPa]"),
        (axes[0, 1], "transp_mmol",      "transpiration [mmol d⁻¹]"),
        (axes[1, 0], "An_total_mmol",    "An total [mmol d⁻¹]"),
        (axes[1, 1], "soil_psi_top_cm",  "soil ψ at top cell [cm]"),
    ]
    for ax, key, ylabel in panels:
        x_w, y_w = _xy(rows_wet, key)
        x_d, y_d = _xy(rows_dry, key)
        ax.plot(x_w, y_w, "-o", label="wet (Fixed -300)", color="#1f77b4")
        ax.plot(x_d, y_d, "-s", label="drying (DuMux)",   color="#d62728")
        ax.set_xlabel("days since start")
        ax.set_ylabel(ylabel)
        ax.legend(loc="best", fontsize=8)
        ax.grid(alpha=0.3)

    fig.suptitle(f"Phase 3 — drying smoke (start: maize day {START_DAY}, "
                 f"ψ_init = {PSI_INIT_CM:.0f} cm)", y=0.995)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _save_soil_profile_figure(profile_log_dry, path: Path) -> None:
    import matplotlib.pyplot as plt

    if not profile_log_dry:
        return
    fig, ax = plt.subplots(figsize=(6, 7))
    z = -np.arange(DEPTH_CM)  # 0 .. -(DEPTH_CM-1) cm; element 0 = top
    cmap = plt.get_cmap("viridis")
    days_max = max(t for t, _ in profile_log_dry)
    days_min = min(t for t, _ in profile_log_dry)
    span = max(days_max - days_min, 1.0)
    for sim_t, profile_cm in profile_log_dry:
        norm = (sim_t - days_min) / span
        ax.plot(profile_cm, z, color=cmap(norm),
                label=f"day {int(sim_t - START_DAY):>2d}")
    ax.set_xlabel("soil ψ [cm]")
    ax.set_ylabel("depth from surface [cm]")
    ax.set_title("DRYING treatment — soil ψ profile vs depth")
    ax.legend(loc="best", fontsize=7, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _gate_check(rows_wet, rows_dry, legacy_an) -> tuple[bool, list[str]]:
    """Return (all_pass, [diagnostic lines]).

    Compares DRY vs WET at the same day to factor out time-varying Vcmax
    (LOPS Chl profile) — within-treatment day-N vs day-0 comparisons are
    confounded by chemistry up-regulation.
    """
    msgs = []
    if not rows_wet or not rows_dry:
        return False, ["empty rows — solve failed before gates could run"]

    # G1: legacy FixedSoilPsi vs DuMux uniform-IC, both at day 0.
    g1_dumux0 = rows_dry[0]["An_total_mmol"]
    if legacy_an is None:
        g1_pass = False
        msgs.append("G1 legacy parity: SKIPPED (FixedSoilPsi solve failed)")
    else:
        g1_pct = abs(legacy_an - g1_dumux0) / max(abs(legacy_an), 1e-12) * 100
        g1_pass = g1_pct < 1.0
        msgs.append(f"G1 day-0 An parity: legacy={legacy_an:.4f}, "
                    f"dumux={g1_dumux0:.4f}, |Δ|={g1_pct:.3f}%  "
                    f"→ {'PASS' if g1_pass else 'FAIL'}")

    # G2/G3 compare DRY vs WET at the LAST shared day, so chemistry cancels.
    n_shared = min(len(rows_wet), len(rows_dry))
    last = n_shared - 1
    if last < 1:
        msgs.append("G2/G3 skipped — DRY treatment didn't survive past day 0")
        return False, msgs

    day_label = int(rows_dry[last]["day"] - START_DAY)

    transp_wet = rows_wet[last]["transp_mmol"]
    transp_dry = rows_dry[last]["transp_mmol"]
    g2_ratio = transp_dry / transp_wet if transp_wet > 0 else float("nan")
    g2_pass = g2_ratio < 0.95
    msgs.append(f"G2 day-{day_label} transp DRY vs WET: "
                f"DRY={transp_dry:.2f}, WET={transp_wet:.2f} mmol/d, "
                f"ratio={g2_ratio:.3f}  → {'PASS' if g2_pass else 'FAIL'}  "
                f"(needs DRY/WET < 0.95)")

    psi_wet = abs(rows_wet[last]["psi_leaf_min_MPa"])
    psi_dry = abs(rows_dry[last]["psi_leaf_min_MPa"])
    g3_ratio = psi_dry / psi_wet if psi_wet > 0 else float("nan")
    g3_pass = g3_ratio > 1.05
    msgs.append(f"G3 day-{day_label} |ψ_leaf_min| DRY vs WET: "
                f"DRY={psi_dry:.4f}, WET={psi_wet:.4f} MPa, "
                f"ratio={g3_ratio:.3f}  → {'PASS' if g3_pass else 'FAIL'}  "
                f"(needs DRY/WET > 1.05)")

    return all([g1_pass, g2_pass, g3_pass]), msgs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=30,
                    help="Number of drying days (default 30)")
    ap.add_argument("--out", type=Path, default=OUT_DIR,
                    help=f"Output directory (default {OUT_DIR})")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    print(f"Phase 3 drying smoke — START_DAY={START_DAY}, n_days={args.days}, "
          f"ψ_init={PSI_INIT_CM} cm")

    print(f"\nGrowing maize plant to day {START_DAY}…")
    plant = grow_plant(str(XML_PATH), simulation_time=START_DAY, seed=42,
                       enable_photosynthesis=True)
    print(f"  plant ready: {len(plant.getSegments())} segments")

    profile_log_dry: list[tuple[float, np.ndarray]] = []

    # G1 one-shot: legacy FixedSoilPsi(-300) An at day 0 → captured before
    # the WET/DRY DuMux loops, so the convention parity is independent of
    # any DuMux state evolution.
    print(f"\n[G1 one-shot] FixedSoilPsi({PSI_INIT_CM}) at day {START_DAY}")
    legacy_res = run_photosynthesis_solve(
        plant, START_DAY, par=PAR_MIDDAY, tleaf=TLEAF_C,
        label="FIXED_d00", rh=RH,
        soil_psi_provider=FixedSoilPsi(psi_cm=PSI_INIT_CM),
    )
    legacy_an = float(legacy_res["An_total_mmol"]) if legacy_res else None
    print(f"  legacy An = {legacy_an}")

    print(f"\n[WET] DumuxSoilPsi(top=Dirichlet ψ=-300, bot=free-drainage, "
          f"col={2*COL_HALF_WIDTH_CM:.0f}×{2*COL_HALF_WIDTH_CM:.0f}×{DEPTH_CM} cm)")
    wet_provider = DumuxSoilPsi(
        depth_cm=DEPTH_CM, n_cells_z=N_CELLS_Z,
        psi_init_cm=PSI_INIT_CM,
        col_half_width_cm=COL_HALF_WIDTH_CM,
        top_bc=(BC_CONSTANT_PRESSURE, PSI_INIT_CM),
        bot_bc=(BC_FREE_DRAINAGE, 0.0),
        verbose=False,
    )
    _align_provider_t0(wet_provider, START_DAY)
    rows_wet = _run_treatment(plant, wet_provider, "WET", args.days,
                              profile_log=[])

    print(f"\n[DRY] DumuxSoilPsi(top=zero-flux, bot=free-drainage, "
          f"col={2*COL_HALF_WIDTH_CM:.0f}×{2*COL_HALF_WIDTH_CM:.0f}×{DEPTH_CM} cm)")
    dry_provider = DumuxSoilPsi(
        depth_cm=DEPTH_CM, n_cells_z=N_CELLS_Z,
        psi_init_cm=PSI_INIT_CM,
        col_half_width_cm=COL_HALF_WIDTH_CM,
        top_bc=(BC_CONSTANT_FLUX, 0.0),
        bot_bc=(BC_FREE_DRAINAGE, 0.0),
        verbose=False,
    )
    _align_provider_t0(dry_provider, START_DAY)
    rows_dry = _run_treatment(plant, dry_provider, "DRY", args.days,
                              profile_log=profile_log_dry)

    csv_path = args.out / "phase3_drying_metrics.csv"
    _save_csv(rows_wet, rows_dry, csv_path)
    print(f"\n  CSV  → {csv_path}")

    try:
        traj_path = args.out / "phase3_drying.png"
        _save_trajectory_figure(rows_wet, rows_dry, traj_path)
        print(f"  PNG  → {traj_path}")

        prof_path = args.out / "phase3_drying_soil_profiles.png"
        _save_soil_profile_figure(profile_log_dry, prof_path)
        print(f"  PNG  → {prof_path}")
    except ImportError:
        print("  (matplotlib not available; skipped figures, CSV is sufficient)")

    print("\n=== Smoke gates ===")
    ok, msgs = _gate_check(rows_wet, rows_dry, legacy_an)
    for m in msgs:
        print(f"  {m}")
    print(f"\nPhase 3 smoke: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
