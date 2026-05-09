"""pm_dumux_g5_score.py — Gate Ch1.PMDM.5 production-smoke scorecard.

Post-processes the per-plant PM sidecars written by
``run_production_series_carbon`` (line ~877 in
``dart/coupling/photosynthesis/diurnal.py``) and asserts the G5
acceptance gates against what was actually persisted on disk. Designed
to run *after* a production smoke completes — no PM re-execution.

What this scorer asserts (today, against existing on-disk data):

  * **G5.1 — PM internal carbon mass-balance** per plant per day:
    ``|mass_balance_residual_pct| < 5 %``. Read straight from the
    ``per_plant_carbon_pm_day{N}.csv`` sidecar's
    ``mass_balance_residual_pct`` column.

  * **G5.2 — PM produced data on every fresh day**: at least one plant
    in each requested day's sidecar has a non-empty
    ``mass_balance_residual_pct`` field.

  * **G5.3 — An_target consistency**: ``An_total_mmol`` (PM-organic)
    and ``An_total_mmol_target`` (DART-informed daily An scaled to
    sucrose) within a loose 5× ratio. PM uses constant peak-PAR
    forcing internally so these can drift; the gate flags
    catastrophic mis-scaling (sign flip, factor-100 error) but not
    physically-acceptable hour-of-day differences.

What this scorer **cannot** assert today (the relevant fields are
returned in the ``solve_carbon_partitioning_pm`` dict but the diurnal
pipeline doesn't currently persist them on disk):

  * **G5.4 — RWU conservation** (``rwu_transpiration_residual_pct < 25 %``,
    PMDM.3 anatomy-bound gate).
  * **G5.5 — ψ_leaf instrumentation** (PMDM.4 drought signal).

To score those, apply the diurnal-pipeline persistence patch in
``pm_dumux_g5_score_PATCH.md`` (sister file in this directory) and
re-run the smoke once. The scorer auto-detects the additional columns
when present and adds them to the scorecard.

Usage on nile:

    source /media/data/Lukas/CPlantBox/cpbenv/bin/activate
    cd /media/data/Lukas/CPlantBox
    python3 dart/coupling/scripts/pm_dumux_g5_score.py \\
        --days 20,24 \\
        --output-dir dart/coupling/output/diurnal_carbon

Usage with custom paths:

    python3 -m dart.coupling.scripts.pm_dumux_g5_score \\
        --days 22,26 \\
        --output-dir /media/data/Lukas/CPlantBox/dart/coupling/output/diurnal_carbon

Exit 0 = all gates PASS; exit 1 = at least one gate FAIL. Per-plant
detail printed to stdout.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Optional

# Acceptance gate thresholds (mirror plan-doc §G5 + PMDM.3 deviation).
MASS_BAL_PCT_MAX = 5.0
RWU_CONS_PCT_MAX = 25.0  # plant-side, anatomy-bound (sheath isPseudostem=1)
AN_TARGET_RATIO_MIN = 0.2
AN_TARGET_RATIO_MAX = 5.0


def _parse_float(s: str) -> Optional[float]:
    """CSV cells that the writer left blank become empty strings; treat
    them as missing, not 0.0. Empty mass_balance is a real signal that
    the plant didn't run through PM (S5 fallback or solver failure)."""
    s = s.strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _read_pm_sidecar(csv_path: Path) -> list[dict]:
    """Read per_plant_carbon_pm_day{N}.csv into a list of dicts.

    The writer emits one row per plant slot (15 by current production
    config); rows where the plant did not run through PM have all
    PM-specific fields blank."""
    rows = []
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            entry = {
                "plant_idx": int(row["plant_idx"]),
                "seed": int(row["seed"]),
                "An_total_mmol": _parse_float(row.get("An_total_mmol", "")),
                "An_total_mmol_target": _parse_float(
                    row.get("An_total_mmol_target", "")),
                "sum_Q_S_meso": _parse_float(row.get("sum_Q_S_meso", "")),
                "dQ_S_meso": _parse_float(row.get("dQ_S_meso", "")),
                "dQ_meso": _parse_float(row.get("dQ_meso", "")),
                "dQ_ST": _parse_float(row.get("dQ_ST", "")),
                "mass_balance_residual_pct": _parse_float(
                    row.get("mass_balance_residual_pct", "")),
                # PMDM.3 / .4 fields — auto-detected when sister
                # persistence patch is applied.
                "integrated_rwu_cm3": _parse_float(
                    row.get("integrated_rwu_cm3", "")),
                "integrated_transpiration_cm3": _parse_float(
                    row.get("integrated_transpiration_cm3", "")),
                "rwu_transpiration_residual_pct": _parse_float(
                    row.get("rwu_transpiration_residual_pct", "")),
                "psi_leaf_min_cm": _parse_float(
                    row.get("psi_leaf_min_cm", "")),
                "psi_leaf_mean_cm": _parse_float(
                    row.get("psi_leaf_mean_cm", "")),
            }
            rows.append(entry)
    return rows


def _score_day(rows: list[dict], day: int) -> dict:
    """Score one day's worth of per-plant PM rows.

    Returns a dict with per-gate PASS/FAIL counts + worst-case
    diagnostics. Plants that didn't run through PM (mass_balance is
    None) are excluded from gate counts but reported.
    """
    pm_rows = [r for r in rows if r["mass_balance_residual_pct"] is not None]
    skipped = len(rows) - len(pm_rows)

    has_rwu = any(r["rwu_transpiration_residual_pct"] is not None for r in pm_rows)
    has_psi = any(r["psi_leaf_min_cm"] is not None for r in pm_rows)

    mb_fails = []
    mb_worst = 0.0
    for r in pm_rows:
        mb = abs(r["mass_balance_residual_pct"])
        if mb > mb_worst:
            mb_worst = mb
        if mb >= MASS_BAL_PCT_MAX:
            mb_fails.append((r["plant_idx"], r["seed"], mb))

    rwu_fails = []
    rwu_worst = None
    if has_rwu:
        rwu_worst = 0.0
        for r in pm_rows:
            rwu = r["rwu_transpiration_residual_pct"]
            if rwu is None:
                continue
            if abs(rwu) > rwu_worst:
                rwu_worst = abs(rwu)
            if abs(rwu) >= RWU_CONS_PCT_MAX:
                rwu_fails.append((r["plant_idx"], r["seed"], rwu))

    an_ratio_fails = []
    for r in pm_rows:
        an_pm = r["An_total_mmol"]
        an_tgt = r["An_total_mmol_target"]
        if an_pm is None or an_tgt is None or an_tgt <= 0:
            continue
        ratio = an_pm / an_tgt
        if not (AN_TARGET_RATIO_MIN <= ratio <= AN_TARGET_RATIO_MAX):
            an_ratio_fails.append(
                (r["plant_idx"], r["seed"], an_pm, an_tgt, ratio))

    psi_min = None
    if has_psi:
        non_none = [r["psi_leaf_min_cm"] for r in pm_rows
                    if r["psi_leaf_min_cm"] is not None]
        psi_min = min(non_none) if non_none else None

    return {
        "day": day,
        "n_plants_total": len(rows),
        "n_plants_pm": len(pm_rows),
        "n_plants_skipped": skipped,
        "mb_worst_pct": mb_worst,
        "mb_fails": mb_fails,
        "has_rwu_data": has_rwu,
        "rwu_worst_pct": rwu_worst,
        "rwu_fails": rwu_fails,
        "an_ratio_fails": an_ratio_fails,
        "has_psi_data": has_psi,
        "psi_leaf_min_cm": psi_min,
    }


def _print_scorecard(per_day: list[dict]) -> bool:
    """Print the cross-day scorecard. Return True if all gates PASS."""
    print()
    print("=" * 78)
    print("Gate Ch1.PMDM.5 production-smoke scorecard")
    print("=" * 78)

    all_pass = True

    # Per-day detail.
    for s in per_day:
        day = s["day"]
        n_pm = s["n_plants_pm"]
        n_total = s["n_plants_total"]
        n_skip = s["n_plants_skipped"]
        print(f"\nDay {day}: {n_pm}/{n_total} plants ran PM "
              f"({n_skip} skipped)")
        if n_pm == 0:
            print(f"  G5.2 PM-produced-data       FAIL (0 plants ran PM)")
            all_pass = False
            continue
        else:
            print(f"  G5.2 PM-produced-data       PASS")

        mb_status = "PASS" if not s["mb_fails"] else "FAIL"
        all_pass = all_pass and not s["mb_fails"]
        print(f"  G5.1 mass-balance < {MASS_BAL_PCT_MAX:.0f} %      "
              f"{mb_status}  worst={s['mb_worst_pct']:.2f} %")
        for pi, seed, mb in s["mb_fails"]:
            print(f"        plant {pi:>2} (seed {seed}): "
                  f"|mb|={mb:.2f} %")

        an_status = "PASS" if not s["an_ratio_fails"] else "FAIL"
        all_pass = all_pass and not s["an_ratio_fails"]
        print(f"  G5.3 An_target ratio in band  {an_status}  "
              f"({len(s['an_ratio_fails'])} out-of-band)")
        for pi, seed, an_pm, an_tgt, ratio in s["an_ratio_fails"][:3]:
            print(f"        plant {pi:>2} (seed {seed}): "
                  f"An_pm={an_pm:.2f}, An_tgt={an_tgt:.2f}, "
                  f"ratio={ratio:.3f}")
        if len(s["an_ratio_fails"]) > 3:
            print(f"        (+{len(s['an_ratio_fails'])-3} more)")

        if s["has_rwu_data"]:
            rwu_status = "PASS" if not s["rwu_fails"] else "FAIL"
            all_pass = all_pass and not s["rwu_fails"]
            print(f"  G5.4 RWU conservation < {RWU_CONS_PCT_MAX:.0f} % "
                  f"{rwu_status}  worst={s['rwu_worst_pct']:.2f} %")
            for pi, seed, rwu in s["rwu_fails"]:
                print(f"        plant {pi:>2} (seed {seed}): "
                      f"|rwu_resid|={abs(rwu):.2f} %")
        else:
            print(f"  G5.4 RWU conservation        SKIP  "
                  f"(field not persisted; apply patch)")

        if s["has_psi_data"]:
            psi = s["psi_leaf_min_cm"]
            print(f"  G5.5 ψ_leaf min on day        "
                  f"min={psi:.1f} cm (informational)")
        else:
            print(f"  G5.5 ψ_leaf instrumentation   SKIP  "
                  f"(field not persisted; apply patch)")

    # Cross-day summary.
    print()
    print("-" * 78)
    overall = "PASS" if all_pass else "FAIL"
    print(f"Overall (G5.1 + G5.2 + G5.3, plus G5.4 if data persisted): "
          f"{overall}")
    print("-" * 78)
    return all_pass


def main():
    ap = argparse.ArgumentParser(
        description="Gate Ch1.PMDM.5 production-smoke scorecard.")
    ap.add_argument("--days", type=str, required=True,
                    help="Comma-separated DART days to score, "
                         "e.g. '20,24'.")
    ap.add_argument("--output-dir", type=str,
                    default="dart/coupling/output/diurnal_carbon",
                    help="Production output dir containing day{N}/ "
                         "subdirs (default: relative to CPlantBox root).")
    args = ap.parse_args()

    days = [int(d.strip()) for d in args.days.split(",")]
    out_dir = Path(args.output_dir)

    if not out_dir.is_absolute():
        # Resolve against CPlantBox root, the repo's typical CWD.
        repo_root = Path(__file__).resolve().parents[3]
        out_dir = repo_root / out_dir

    if not out_dir.exists():
        print(f"ERROR: output dir does not exist: {out_dir}",
              file=sys.stderr)
        return 2

    per_day = []
    for d in days:
        sidecar = out_dir / f"day{d}" / f"per_plant_carbon_pm_day{d}.csv"
        if not sidecar.exists():
            print(f"WARNING: missing sidecar for day {d}: {sidecar}",
                  file=sys.stderr)
            continue
        rows = _read_pm_sidecar(sidecar)
        per_day.append(_score_day(rows, d))

    if not per_day:
        print("ERROR: no PM sidecars found for the requested days",
              file=sys.stderr)
        return 2

    ok = _print_scorecard(per_day)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
