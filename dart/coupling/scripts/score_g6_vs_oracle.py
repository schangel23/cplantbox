"""Score a G6-full Path B run against FA oracle PASS criteria.

Parses ``run_g6_pm_dumux_fa_parity`` stdout log and emits a §G6 Path B
acceptance scorecard:

  - convergence: 0 PM-day failures across the 100-day Phase-3 loop
  - day-130 mainstem >= 122 cm (30% of oracle 175 cm)
  - >= 12/15 oracle leaves present with realised_length > 0.01 cm
  - GPP/Rm/Rg bands: not yet emitted by the runner; reported as N/A
    until runner instrumentation lands.

A trajectory milestone table (default horizons 40/70/100/130 d) is
printed from the runner's per-10-day status lines.

Usage::

    python -m dart.coupling.scripts.score_g6_vs_oracle \\
        --log dart/coupling/output/g6full_pathB_run/g6full.log \\
        [--mainstem-floor-cm 122] [--leaf-floor 12] \\
        [--horizons 40,70,100,130] [--json]

Exit codes: 0 = PASS, 1 = FAIL, 2 = log unparseable.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

DAY_RE = re.compile(
    r"day\s+(?P<day>\d+):\s+ok\s+\((?P<wall_s>\d+)s\),\s+"
    r"mb=(?P<mb>FAIL|[-\d.]+%),\s+"
    r"organs=(?P<organs>\d+),\s+"
    r"leaves=(?P<leaves_emerged>\d+)/(?P<leaves_total>\d+),\s+"
    r"PM fails=(?P<pm_fails>\d+)/(?P<pm_calls>\d+)"
    r"(?:,\s+An=(?P<an>[-\d.]+)\s+Rm=(?P<rm>[-\d.]+)\s+Rg=(?P<rg>[-\d.]+))?"
)
PHASE3_DONE_RE = re.compile(
    r"Phase 3 done in (?P<wall>\d+)s \((?P<calls>\d+) PM calls, (?P<fails>\d+) failures\)"
)
MAINSTEM_RE = re.compile(
    r"mainstem realised:\s+oracle=(?P<oracle>[-\d.]+)\s+cm,\s+"
    r"with-carbon=(?P<wc>[-\d.]+)\s+cm,\s+"
    r"Δ=(?P<delta>[-+\d.]+)\s+cm"
)
LEAF_RE = re.compile(
    r"leaf\s+st=(?P<st>\d+):\s+oracle=(?P<oracle>[-\d.]+)\s+cm,\s+"
    r"with-carbon=(?P<wc>[-\d.]+)\s+cm,\s+"
    r"drift=(?P<drift>[\d.]+)%"
)
LEAF_MISSING_RE = re.compile(
    r"FAIL:\s+leaf subType=(?P<st>\d+) missing from with-carbon snapshot"
)

DEFAULT_HORIZONS = (40, 70, 100, 130)
DEFAULT_MAINSTEM_FLOOR_CM = 122.0
DEFAULT_LEAF_FLOOR = 12
DEFAULT_ORACLE_LEAVES = 15


def parse_log(text: str) -> dict:
    days = [
        {
            "day": int(m["day"]),
            "wall_s": int(m["wall_s"]),
            "mb_pct": None if m["mb"] == "FAIL" else float(m["mb"].rstrip("%")),
            "organs": int(m["organs"]),
            "leaves_emerged": int(m["leaves_emerged"]),
            "leaves_total": int(m["leaves_total"]),
            "pm_fails": int(m["pm_fails"]),
            "pm_calls": int(m["pm_calls"]),
            "An_mmol_co2": float(m["an"]) if m["an"] is not None else None,
            "Rm_mmol_co2": float(m["rm"]) if m["rm"] is not None else None,
            "Rg_mmol_co2": float(m["rg"]) if m["rg"] is not None else None,
        }
        for m in DAY_RE.finditer(text)
    ]

    phase3 = PHASE3_DONE_RE.search(text)
    mainstem = MAINSTEM_RE.search(text)
    leaves = [
        {
            "subtype": int(m["st"]),
            "oracle_cm": float(m["oracle"]),
            "with_carbon_cm": float(m["wc"]),
            "drift_pct": float(m["drift"]),
        }
        for m in LEAF_RE.finditer(text)
    ]
    missing = sorted({int(m["st"]) for m in LEAF_MISSING_RE.finditer(text)})

    return {
        "days": days,
        "phase3_wall_s": int(phase3["wall"]) if phase3 else None,
        "phase3_pm_calls": int(phase3["calls"]) if phase3 else None,
        "phase3_failures": int(phase3["fails"]) if phase3 else None,
        "mainstem": (
            {
                "oracle_cm": float(mainstem["oracle"]),
                "with_carbon_cm": float(mainstem["wc"]),
                "delta_cm": float(mainstem["delta"]),
            }
            if mainstem
            else None
        ),
        "leaves_observed": leaves,
        "leaves_missing": missing,
    }


def score(
    parsed: dict,
    *,
    mainstem_floor_cm: float,
    leaf_floor: int,
    oracle_leaves: int,
) -> dict:
    criteria = []

    # 1. Convergence
    fails = parsed["phase3_failures"]
    calls = parsed["phase3_pm_calls"]
    if fails is None:
        criteria.append(("convergence", "INDET",
                         "no Phase-3-done line in log (run truncated?)"))
    elif fails == 0:
        criteria.append(("convergence", "PASS",
                         f"0 PM failures across {calls} calls"))
    else:
        failed_days = [d["day"] for d in parsed["days"] if d["mb_pct"] is None]
        sample = failed_days[:5]
        more = "" if len(failed_days) <= 5 else f" (+{len(failed_days) - 5} more)"
        criteria.append(("convergence", "FAIL",
                         f"{fails}/{calls} PM failures; first failed days {sample}{more}"))

    # 2. day-130 mainstem
    ms = parsed["mainstem"]
    if ms is None:
        criteria.append(("mainstem", "INDET", "no mainstem comparison line in log"))
    else:
        ratio = ms["with_carbon_cm"] / ms["oracle_cm"] if ms["oracle_cm"] else 0.0
        status = "PASS" if ms["with_carbon_cm"] >= mainstem_floor_cm else "FAIL"
        criteria.append((
            "mainstem", status,
            f"{ms['with_carbon_cm']:.2f} cm vs oracle {ms['oracle_cm']:.2f} cm "
            f"({ratio * 100:.1f}% of oracle; floor {mainstem_floor_cm:.0f} cm)",
        ))

    # 3. leaves present
    n_present = sum(1 for L in parsed["leaves_observed"] if L["with_carbon_cm"] > 0.01)
    n_missing = len(parsed["leaves_missing"])
    status = "PASS" if n_present >= leaf_floor else "FAIL"
    criteria.append((
        "leaves_present", status,
        f"{n_present}/{oracle_leaves} present "
        f"({n_missing} missing; floor {leaf_floor}/{oracle_leaves})",
    ))

    # 4. flux bands — physically-plausible carbon trajectory
    flux_days = [d for d in parsed["days"] if d["An_mmol_co2"] is not None]
    if not flux_days:
        criteria.append(("flux_bands", "N/A",
                         "runner did not log per-day An/Rm/Rg "
                         "(pre-instrumentation log)"))
    else:
        problems = []
        for d in flux_days:
            an, rm, rg = d["An_mmol_co2"], d["Rm_mmol_co2"], d["Rg_mmol_co2"]
            if an <= 0:
                problems.append(f"d{d['day']} An<=0 ({an:.3f})")
            if rg < 0:
                problems.append(f"d{d['day']} Rg<0 ({rg:.3f})")
            if rm < 0:
                problems.append(f"d{d['day']} Rm<0 ({rm:.3f})")
            if an > 0 and rm / an > 5.0:
                problems.append(f"d{d['day']} Rm/An={rm/an:.2f}>5 (runaway maint resp)")
        # Final-day Rg must clear a non-collapsed threshold (Path B target
        # was 0.957 mmol CO2/d at single-day day-30; expect day-130 >=
        # that as a minimum for a non-collapsed run).
        final = flux_days[-1]
        rg_floor = 0.5  # mmol CO2/d, conservative non-collapsed Path B threshold
        if final["Rg_mmol_co2"] < rg_floor:
            problems.append(
                f"d{final['day']} Rg={final['Rg_mmol_co2']:.3f} < {rg_floor} "
                f"(collapsed-growth signature)"
            )
        if problems:
            criteria.append(("flux_bands", "FAIL",
                             "; ".join(problems[:6]) +
                             ("" if len(problems) <= 6
                              else f" (+{len(problems) - 6} more)")))
        else:
            an_range = (min(d["An_mmol_co2"] for d in flux_days),
                        max(d["An_mmol_co2"] for d in flux_days))
            rm_range = (min(d["Rm_mmol_co2"] for d in flux_days),
                        max(d["Rm_mmol_co2"] for d in flux_days))
            rg_range = (min(d["Rg_mmol_co2"] for d in flux_days),
                        max(d["Rg_mmol_co2"] for d in flux_days))
            criteria.append((
                "flux_bands", "PASS",
                f"An∈[{an_range[0]:.2f},{an_range[1]:.2f}] "
                f"Rm∈[{rm_range[0]:.2f},{rm_range[1]:.2f}] "
                f"Rg∈[{rg_range[0]:.2f},{rg_range[1]:.2f}] mmol CO2/d "
                f"(no negatives, no Rm/An>5, final Rg >= {rg_floor})"
            ))

    overall = "PASS" if all(s in ("PASS", "N/A") for _, s, _ in criteria) else "FAIL"
    return {"overall": overall, "criteria": criteria}


def horizon_table(parsed: dict, horizons) -> str:
    by_day = {d["day"]: d for d in parsed["days"]}
    rows = [
        "Trajectory milestones (from per-10-day status lines):",
        f"  {'day':>5} {'wall_s':>8} {'mb%':>8} {'organs':>8} "
        f"{'leaves':>11} {'PM_fails':>10} "
        f"{'An':>7} {'Rm':>7} {'Rg':>7} (mmol CO2/d)",
    ]
    for h in horizons:
        d = by_day.get(h)
        if d is None:
            rows.append(f"  {h:>5}   (missing — runner did not log this day)")
            continue
        mb = "FAIL" if d["mb_pct"] is None else f"{d['mb_pct']:.2f}"
        leaves = f"{d['leaves_emerged']:>3}/{d['leaves_total']:<3}"
        fails = f"{d['pm_fails']:>3}/{d['pm_calls']:<5}"
        an = "  --   " if d["An_mmol_co2"] is None else f"{d['An_mmol_co2']:>7.3f}"
        rm = "  --   " if d["Rm_mmol_co2"] is None else f"{d['Rm_mmol_co2']:>7.3f}"
        rg = "  --   " if d["Rg_mmol_co2"] is None else f"{d['Rg_mmol_co2']:>7.3f}"
        rows.append(
            f"  {d['day']:>5} {d['wall_s']:>8} {mb:>8} {d['organs']:>8} "
            f"{leaves:>11} {fails:>10} {an} {rm} {rg}"
        )
    return "\n".join(rows)


def main():
    parser = argparse.ArgumentParser(
        description=(__doc__ or "").splitlines()[0],
    )
    parser.add_argument("--log", required=True, type=Path,
                        help="path to run_g6_pm_dumux_fa_parity stdout log")
    parser.add_argument("--mainstem-floor-cm", type=float,
                        default=DEFAULT_MAINSTEM_FLOOR_CM,
                        help="Path B mainstem PASS floor (default 122 cm)")
    parser.add_argument("--leaf-floor", type=int, default=DEFAULT_LEAF_FLOOR,
                        help="minimum leaves-present count (default 12)")
    parser.add_argument("--oracle-leaves", type=int,
                        default=DEFAULT_ORACLE_LEAVES,
                        help="oracle leaf count denominator (default 15)")
    parser.add_argument("--horizons", type=str,
                        default=",".join(str(h) for h in DEFAULT_HORIZONS),
                        help="comma-separated milestone days for table")
    parser.add_argument("--json", action="store_true",
                        help="emit machine-readable JSON instead of human text")
    args = parser.parse_args()

    if not args.log.exists():
        print(f"ERROR: log not found: {args.log}", file=sys.stderr)
        return 2

    parsed = parse_log(args.log.read_text())
    if not parsed["days"]:
        print(f"ERROR: no per-day status lines parsed from {args.log}",
              file=sys.stderr)
        return 2

    horizons = tuple(int(h) for h in args.horizons.split(","))
    report = score(
        parsed,
        mainstem_floor_cm=args.mainstem_floor_cm,
        leaf_floor=args.leaf_floor,
        oracle_leaves=args.oracle_leaves,
    )

    if args.json:
        print(json.dumps({
            "log": str(args.log),
            "horizons": list(horizons),
            "parsed": parsed,
            "scorecard": report,
        }, indent=2))
        return 0 if report["overall"] == "PASS" else 1

    print("=" * 78)
    print(f"§G6 Path B scorecard — {args.log}")
    print("=" * 78)
    print(horizon_table(parsed, horizons))
    print()
    for name, status, detail in report["criteria"]:
        print(f"  [{status:>5}] {name:<16} {detail}")
    print()
    print(f"  OVERALL: {report['overall']}")
    print("=" * 78)
    return 0 if report["overall"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
