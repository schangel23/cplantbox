#!/usr/bin/env python3
"""Report no-PM growth deltas over a day window.

Diagnostic purpose: separate native FA/thermal growth potential from
PiafMunch/CWLimitedGrowth carbon gating. If the no-PM stem delta matches the
PM stem delta, the stem is pinned by its native demand schedule. If no-PM grows
more stem, PM/CW routing is suppressing stem demand or supply.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dart.coupling.growth.grow import grow_plant  # noqa: E402


DEFAULT_XML = REPO_ROOT / "dart/coupling/data/maize_calibrated.xml"


def _organ_type_lengths(plant) -> dict[int, float]:
    lengths = {2: 0.0, 3: 0.0, 4: 0.0}
    for organ in plant.getOrgans(-1, True):
        ot = int(organ.organType())
        if ot in lengths:
            lengths[ot] += float(organ.getLength())
    return lengths


def _row(seed: int, start_day: int, end_day: int, xml: Path) -> dict[str, float]:
    plant_start = grow_plant(str(xml), simulation_time=start_day, seed=seed)
    plant_end = grow_plant(str(xml), simulation_time=end_day, seed=seed)
    start = _organ_type_lengths(plant_start)
    end = _organ_type_lengths(plant_end)
    root_d = end[2] - start[2]
    stem_d = end[3] - start[3]
    leaf_d = end[4] - start[4]
    return {
        "seed": seed,
        "start_day": start_day,
        "end_day": end_day,
        "root_len_start_cm": round(start[2], 6),
        "root_len_end_cm": round(end[2], 6),
        "root_dlen_cm": round(root_d, 6),
        "stem_len_start_cm": round(start[3], 6),
        "stem_len_end_cm": round(end[3], 6),
        "stem_dlen_cm": round(stem_d, 6),
        "leaf_len_start_cm": round(start[4], 6),
        "leaf_len_end_cm": round(end[4], 6),
        "leaf_dlen_cm": round(leaf_d, 6),
        "total_dlen_cm": round(root_d + stem_d + leaf_d, 6),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--xml", type=Path, default=DEFAULT_XML)
    ap.add_argument("--seeds", type=int, nargs="+", default=[7])
    ap.add_argument("--start-day", type=int, default=30)
    ap.add_argument("--end-day", type=int, default=55)
    ap.add_argument("--out-csv", required=True)
    args = ap.parse_args()

    rows = [_row(seed, args.start_day, args.end_day, args.xml)
            for seed in args.seeds]
    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(
            f"seed={row['seed']} day {args.start_day}->{args.end_day}: "
            f"root={row['root_dlen_cm']:.3f} cm "
            f"stem={row['stem_dlen_cm']:.3f} cm "
            f"leaf={row['leaf_dlen_cm']:.3f} cm "
            f"total={row['total_dlen_cm']:.3f} cm"
        )
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
