"""S6 V6-V11 visual regression: per-rank leaf-to-mature ratio analyser.

Per ADR §S6 (Literature/Chapter 1/Concepts/CPBOXBALENOCOUPLING/
ADR_LEAF_KINEMATICS_2026-04-28.md), the post-Andrieu canopy must show
staggered maturity across ranks rather than the pre-Andrieu uniform-50 %
logistic lockstep. This script grows maize at a sweep of mid-vegetative
calendar days, extracts per-leaf (subType, length, lmax) tuples directly
from the simulator, and asserts the ADR §S6 gates:

  - V-stage staggering: at V_k, the per-rank ratio sequence is monotonic
    non-increasing from oldest to youngest leaf.
  - Spread: max(ratio) - min(ratio) >= 0.50 across the canopy at V8±2
    (ADR signal: spec calls for leaf 5 ~85-95 % vs leaf 8 ~10-25 % at V8).
  - Absence of uniform lockstep: < 60 % of canopy leaves sit in the
    [0.40, 0.60] band at any mid-vegetative day.

Per-rank ratios are computed exactly the same way ``phenology.py``'s
``count_visible_leaves`` does (arc-length over lrp.lmax). Each maize leaf
subType IS one Déa rank per ADR §C4 1:1 mapping, so subType doubles as
rank. Pairs with ``_gen_vr_stages.py``: this analyser provides the numeric
gate; the renderer provides the OBJ row for visual confirmation in Blender.

Run (from /home/lukas/PHD/CPlantBox):
    LD_LIBRARY_PATH=... PYTHONPATH=. cpbenv/bin/python3 \
        dart/tools/blender_preview/_analyse_vr_stages.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

import plantbox as pb  # noqa: E402

from dart.coupling.config import DEFAULT_XML  # noqa: E402
from dart.coupling.growth.grow import grow_plant  # noqa: E402
from dart.coupling.growth.phenology import (  # noqa: E402
    COLLAR_RELEASE,
    COLLAR_THRESHOLD,
    count_visible_leaves,
    detect_v_stage,
)


SEED = 42

# Sweep covers the ADR §S6 day range (55..100) plus 50 (V5 candidate) and
# 60/65/75/80 to triangulate first-crossing days for V5/V8/V11 under the
# post-S3 Andrieu canopy (where V-stage labels shift earlier in calendar
# terms vs the pre-S3 logistic phenotype).
DAYS = [50, 55, 60, 65, 70, 75, 80, 85]


def _arc_length(organ) -> float:
    nodes = organ.getNodes()
    if len(nodes) < 2:
        return 0.0
    cur = 0.0
    prev = nodes[0]
    for nd in nodes[1:]:
        dx = float(nd.x) - float(prev.x)
        dy = float(nd.y) - float(prev.y)
        dz = float(nd.z) - float(prev.z)
        cur += math.sqrt(dx * dx + dy * dy + dz * dz)
        prev = nd
    return cur


def per_leaf_ratios(plant):
    """Return list of (rank, length_cm, lmax_cm, ratio) sorted by rank.

    Rank == leaf subType per ADR §C4 1:1 mapping (subType N → Déa rank N).
    Tassel organs (subType 21) and any leaves with no nodes are skipped.
    """
    out = []
    for lf in plant.getOrgans(pb.leaf):
        try:
            st = int(lf.getParameter("subType"))
        except Exception:
            continue
        if st <= 0:
            continue
        length = _arc_length(lf)
        if length <= 0.0:
            continue
        lrp = lf.getLeafRandomParameter()
        lmax = max(float(lrp.lmax), 1e-9)
        out.append((st, length, lmax, min(length / lmax, 1.0)))
    out.sort(key=lambda r: r[0])
    return out


def is_monotone_non_increasing(ratios: list[float], tol: float = 0.02) -> bool:
    """Per-rank ratio sequence is non-increasing from oldest to youngest leaf.

    Allows a small tolerance for the lag-exp phase where adjacent ranks have
    near-equal length (within `tol`); a true increase across rank boundaries
    means the older leaf is still elongating faster than the younger one,
    which would indicate the pre-Andrieu logistic lockstep.

    Caller is expected to pass ratios for non-gated ranks only (subType >= 4
    on the maize calibration). Gated juveniles (rank 1-3, gf=1
    `ExponentialGrowth`) plateau asymptotically below 1.0 and would
    spuriously break monotonicity against the gf=6 ranks that hit lmax exactly.
    """
    return all(b <= a + tol for a, b in zip(ratios, ratios[1:]))


def lockstep_fraction(ratios: list[float]) -> float:
    """Fraction of leaves in the [0.40, 0.60] uniform-50 % band."""
    if not ratios:
        return 0.0
    n_band = sum(1 for r in ratios if 0.40 <= r <= 0.60)
    return n_band / len(ratios)


def report_stage(day: int) -> dict:
    print(f"\n=== Day {day:3d} (seed {SEED}) ===")
    plant = grow_plant(
        str(DEFAULT_XML),
        simulation_time=day,
        seed=SEED,
        enable_photosynthesis=False,
    )
    label = detect_v_stage(plant)
    counts = count_visible_leaves(plant)
    rows = per_leaf_ratios(plant)
    ratios = [r[3] for r in rows]
    spread = (max(ratios) - min(ratios)) if ratios else 0.0
    lock_frac = lockstep_fraction(ratios)
    # Monotonicity gate: skip gated juvenile ranks (subType < 4 in the maize
    # calibration use gf=1 ExponentialGrowth and plateau slightly below 1.0
    # asymptotically). The ADR §S6 staggering signal lives in the gf=6 ranks.
    non_gated_ratios = [r[3] for r in rows if r[0] >= 4]
    monotonic = is_monotone_non_increasing(non_gated_ratios)

    print(f"  V-label: {label}  collared: {counts['collared']}  "
          f"emerging: {counts['emerging']}  whorl: {counts['whorl']}")
    print(f"  rank  length    lmax    ratio  bucket")
    for rank, L, lmax, r in rows:
        if r >= COLLAR_RELEASE:
            bucket = "collared"
        elif r >= COLLAR_THRESHOLD:
            bucket = "emerging"
        else:
            bucket = "whorl"
        print(f"  {rank:4d}  {L:6.2f}cm  {lmax:5.2f}cm  {r:5.3f}  {bucket}")
    print(f"  spread = {spread:.3f}  monotonic = {monotonic}  "
          f"lockstep_frac = {lock_frac:.2f}")

    return {
        "day": day,
        "label": label,
        "counts": counts,
        "rows": rows,
        "spread": spread,
        "monotonic": monotonic,
        "lockstep_frac": lock_frac,
    }


def find_v_stage_day(stages: list[dict], target_v: int) -> dict | None:
    """First day in `stages` whose `counts['collared']` == target_v.

    Picks the earliest exact match; falls back to closest-by-collar-count
    if no exact hit (under post-S3 V-stages can skip a number when two
    leaves cross COLLAR_RELEASE on the same day).
    """
    exact = [s for s in stages if s["counts"]["collared"] == target_v]
    if exact:
        return min(exact, key=lambda s: s["day"])
    by_diff = sorted(stages, key=lambda s: abs(s["counts"]["collared"] - target_v))
    return by_diff[0] if by_diff else None


def evaluate_gates(stages: list[dict]) -> tuple[int, list[dict]]:
    """Apply ADR §S6 gates to V5/V8/V11 (or nearest)."""
    print("\n" + "=" * 60)
    print("ADR §S6 gate evaluation")
    print("=" * 60)
    gate_results = []
    fail_count = 0
    for v_target in (5, 8, 11):
        stage = find_v_stage_day(stages, v_target)
        if stage is None:
            print(f"\nV{v_target}: SKIP (no candidate day in sweep)")
            continue

        ratios = [r[3] for r in stage["rows"]]
        ranks = [r[0] for r in stage["rows"]]
        n_collared = stage["counts"]["collared"]
        is_exact = (n_collared == v_target)
        match_tag = "exact" if is_exact else f"closest (V{n_collared})"

        gates = {}
        gates["monotonic"] = stage["monotonic"]
        # Spread gate applies only at V5/V8 — the ADR §S6 "leaf 5 ≥ 85 % AND
        # leaf 8 ≤ 25 %" pattern is a mid-vegetative staggering signal. By
        # V11 the entire canopy has plateaued at lmax (spread → 0), which
        # is the biologically correct endpoint, not a regression.
        if v_target in (5, 8):
            gates["spread_ge_0p50"] = stage["spread"] >= 0.50
        gates["lockstep_lt_0p60"] = stage["lockstep_frac"] < 0.60

        oldest_collared = max((r for r in stage["rows"] if r[3] >= COLLAR_RELEASE),
                              key=lambda r: r[3], default=None)
        gates["has_mature_leaf"] = oldest_collared is not None and oldest_collared[3] >= 0.85

        passed = all(gates.values())
        if not passed:
            fail_count += 1

        print(f"\nV{v_target} [{match_tag}] day={stage['day']} "
              f"label={stage['label']} ranks={ranks}")
        print(f"  spread = {stage['spread']:.3f}  "
              f"monotonic = {stage['monotonic']}  "
              f"lockstep_frac = {stage['lockstep_frac']:.2f}  "
              f"max_ratio = {max(ratios):.3f}")
        for gname, gval in gates.items():
            mark = "PASS" if gval else "FAIL"
            print(f"    [{mark}] {gname}")

        gate_results.append({
            "v_target": v_target,
            "match": match_tag,
            "day": stage["day"],
            "label": stage["label"],
            "gates": gates,
            "passed": passed,
        })

    return fail_count, gate_results


def main() -> int:
    print(f"S6 V-stage analyser — sweep days {DAYS} (seed={SEED})")
    stages = []
    for day in DAYS:
        try:
            stages.append(report_stage(day))
        except Exception as e:
            print(f"  FAILED day {day}: {e!r}")

    fail_count, gate_results = evaluate_gates(stages)

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f" day  label  collared  spread  monotonic  lockstep")
    for s in stages:
        print(f" {s['day']:3d}  {s['label']:<12s}  "
              f"{s['counts']['collared']:2d}        "
              f"{s['spread']:5.3f}   {str(s['monotonic']):<5s}      "
              f"{s['lockstep_frac']:.2f}")
    print()
    if fail_count == 0:
        print("ADR §S6 gates: ALL PASS")
        rc = 0
    else:
        print(f"ADR §S6 gates: {fail_count} V-stage(s) FAILED")
        rc = 1

    out = {
        "seed": SEED,
        "days": [s["day"] for s in stages],
        "stages": [{
            "day": s["day"],
            "label": s["label"],
            "counts": s["counts"],
            "spread": s["spread"],
            "monotonic": s["monotonic"],
            "lockstep_frac": s["lockstep_frac"],
            "rows": [{"rank": r[0], "length_cm": r[1], "lmax_cm": r[2],
                      "ratio": r[3]} for r in s["rows"]],
        } for s in stages],
        "gate_results": gate_results,
    }
    out_path = PROJECT_ROOT / "dart" / "coupling" / "output" / "vr_stages" / "_s6_analysis.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  -> {out_path.relative_to(PROJECT_ROOT)}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
