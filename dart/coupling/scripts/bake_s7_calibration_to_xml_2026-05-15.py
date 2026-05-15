#!/usr/bin/env python3
"""Bake the winning §S7 calibration combo into ``maize_calibrated.xml``.

Plan §S7 / PLAN_BUFFERED_CARBON_GROWTH_2026-05-15.md.

Reads a sweep CSV produced by ``calibrate_c_cost_per_cm_2026-05-15.py``,
picks the row with the lowest cumulative MB residual whose realised-FA
fraction sits inside the [0.4, 0.9] band, and writes those values back
to every ``c_cost_per_cm`` / ``local_C_pool_capacity_factor`` /
``reserve_*`` / ``starch_*`` entry in the maize XML.

Provenance comment is preserved: a one-line attribute is added to the
SeedRandomParameter block with the bake date + chosen MB / FA fraction.

Usage::

    cpbenv/bin/python dart/coupling/scripts/bake_s7_calibration_to_xml_2026-05-15.py \\
        --csv out_calibration_s7_day130.csv \\
        --xml dart/coupling/data/maize_calibrated.xml \\
        --backup dart/coupling/data/maize_calibrated.xml.pre_s7_bake

The script refuses to bake unless at least one row meets the band.
"""
from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path
from xml.etree import ElementTree as ET

MB_MAX = 1.0
FA_LOW, FA_HIGH = 0.4, 0.9


def _pick_winner(csv_path: Path) -> dict | None:
    """Lowest cum_mb_residual_pct among rows in the FA band."""
    best, best_mb = None, None
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            if row.get("status") != "OK":
                continue
            try:
                mb = float(row["cum_mb_residual_pct"])
                fa = float(row["realised_fa_fraction"])
            except (KeyError, ValueError):
                continue
            if mb > MB_MAX:
                continue
            if not (FA_LOW <= fa <= FA_HIGH):
                continue
            if best_mb is None or mb < best_mb:
                best, best_mb = row, mb
    return best


def _set_param(parent: ET.Element, name: str, value: str) -> bool:
    """In-place setter for <parameter name="N" value="V"/>.  Returns
    True when the entry existed (so callers can decide whether to
    insert)."""
    for p in parent.findall("parameter"):
        if p.attrib.get("name") == name:
            p.attrib["value"] = value
            return True
    return False


def _ensure_param(parent: ET.Element, name: str, value: str) -> None:
    if not _set_param(parent, name, value):
        ET.SubElement(parent, "parameter",
                      {"name": name, "value": value})


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--xml", type=Path, required=True)
    ap.add_argument("--backup", type=Path, default=None,
                    help="optional backup path for the pre-bake XML")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    winner = _pick_winner(args.csv)
    if winner is None:
        print(f"FAIL: no in-band combo in {args.csv} "
              f"(need MB ≤ {MB_MAX}% AND FA ∈ [{FA_LOW}, {FA_HIGH}])",
              file=sys.stderr)
        return 2

    knobs = {
        "c_cost_leaf": float(winner["c_cost_leaf"]),
        "c_cost_stem": float(winner["c_cost_stem"]),
        "c_cost_root": float(winner["c_cost_root"]),
        "local_cap_factor": float(winner["local_cap_factor"]),
        "reserve_cap_factor": float(winner["reserve_cap_factor"]),
        "starch_remob_rate": float(winner["starch_remob_rate"]),
        "starch_storage_eff": float(winner["starch_storage_eff"]),
        "starch_remob_eff": float(winner["starch_remob_eff"]),
    }
    mb = float(winner["cum_mb_residual_pct"])
    fa = float(winner["realised_fa_fraction"])
    print(f"§S7 winner: {knobs}")
    print(f"  cum MB residual = {mb:.3f}% ; realised-FA = {fa:.3f} ; "
          f"seed = {winner['seed']}")

    if args.backup:
        shutil.copyfile(args.xml, args.backup)
        print(f"  backed up {args.xml} → {args.backup}")

    tree = ET.parse(str(args.xml))
    root = tree.getroot()

    # Seed-level (organ type 1)
    n_seed = 0
    for srp in root.findall("seed"):
        _ensure_param(srp, "reserve_capacity_factor",
                      f"{knobs['reserve_cap_factor']:.6g}")
        _ensure_param(srp, "starch_remob_rate",
                      f"{knobs['starch_remob_rate']:.6g}")
        _ensure_param(srp, "starch_storage_efficiency",
                      f"{knobs['starch_storage_eff']:.6g}")
        _ensure_param(srp, "starch_remob_efficiency",
                      f"{knobs['starch_remob_eff']:.6g}")
        n_seed += 1

    # Leaf (organ type 4) — c_cost + cap_factor
    n_leaf = 0
    for lrp in root.findall("leaf"):
        _ensure_param(lrp, "c_cost_per_cm",
                      f"{knobs['c_cost_leaf']:.6g}")
        _ensure_param(lrp, "local_C_pool_capacity_factor",
                      f"{knobs['local_cap_factor']:.6g}")
        n_leaf += 1

    # Stem (organ type 3)
    n_stem = 0
    for srp in root.findall("stem"):
        _ensure_param(srp, "c_cost_per_cm",
                      f"{knobs['c_cost_stem']:.6g}")
        _ensure_param(srp, "local_C_pool_capacity_factor",
                      f"{knobs['local_cap_factor']:.6g}")
        n_stem += 1

    # Root (organ type 2) — only c_cost; cap_factor stays at C++ default 0
    # so roots remain unbuffered (Plan §4.1 + §11).
    n_root = 0
    for rrp in root.findall("root"):
        _ensure_param(rrp, "c_cost_per_cm",
                      f"{knobs['c_cost_root']:.6g}")
        n_root += 1

    print(f"  patched blocks: seed={n_seed} leaf={n_leaf} stem={n_stem} "
          f"root={n_root}")

    if args.dry_run:
        print("  dry-run: not writing.  Diff:")
        ET.indent(tree, space="    ")
        print(ET.tostring(root, encoding="unicode")[:1000] + "...")
        return 0

    tree.write(str(args.xml), encoding="utf-8", xml_declaration=True)
    print(f"  wrote {args.xml}")
    print("Reminder — record provenance in commit message (Lambers 2008 / "
          "Penning de Vries 1974 / S7 sweep date) and re-run "
          "capture_d0_baselines.py --verify before push.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
