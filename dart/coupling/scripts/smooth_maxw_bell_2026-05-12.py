"""Smooth the max_w bell so it's monotonic peaking at slot 9, and widen
the canopy so total rendered area returns to ~6900 cm² (the pre-shift
donor-reality target).

Approach:
  max_w_new[K] = α × lmax[K] + β

Chosen so that:
  - max_w_new[0] = 2.57 (preserve current stub width — slot 0 unchanged)
  - max_w_new[9] = 8.0  (target peak — gives ~7000 cm² total area at ~78 cm peak lmax)

This makes max_w monotonic on both sides of slot 9 by construction
(lmax bell is already monotonic peaking at slot 9 → max_w follows).

Updates:
  - JSON: max_w_xml_cm per rank
  - XML: Width_blade per leaf (= old Width_blade × ratio max_w_new/max_w_old)
  - XML: areaMax per leaf (= old × ratio max_w_new/max_w_old)
  - XML: Width_petiole = Width_blade × 0.3

NOT touched:
  - JSON intercept blocks (droop, along, halfwidth_norm — halfwidth_norm is
    dimensionless and gets multiplied by max_w at runtime; absolute blocks
    stay correct since we don't change lmax)
  - JSON asym_residual_grids_cm (absolute cm; stays correct)
  - JSON lmax_intercept_cm / lmax_xml_cm (lmax unchanged)
  - XML lmax / R2_n / lag_exp_n (FA kinetics unchanged)
"""
from __future__ import annotations
import json
import shutil
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

JSON_PATH = Path("/home/lukas/PHD/CPlantBox/dart/coupling/data/maize_leaf_shape_distribution.json")
XML_PATH = Path("/home/lukas/PHD/CPlantBox/dart/coupling/data/maize_calibrated.xml")
N_RANKS = 15
TARGET_PEAK_MAX_W = 8.0    # cm
STUB_MAX_W = 2.57           # cm at slot 0 (= MF3D pos 0 stub)


def main():
    # ---- JSON side ----
    with open(JSON_PATH) as f:
        d = json.load(f)

    lmax = {int(k): float(v) for k, v in d["lmax_intercept_cm"].items()}
    old_max_w = {int(k): float(v) for k, v in d["max_w_xml_cm"].items()}

    # Solve α, β: max_w = α × lmax + β
    #   stub: max_w[0] = STUB_MAX_W at lmax[0]
    #   peak: max_w[9] = TARGET_PEAK_MAX_W at lmax[9]
    L0 = lmax[0]; L9 = lmax[9]
    alpha = (TARGET_PEAK_MAX_W - STUB_MAX_W) / (L9 - L0)
    beta  = STUB_MAX_W - alpha * L0
    print(f"max_w bell: max_w = {alpha:.4f} × lmax + {beta:.4f}")
    print(f"  anchor stub: lmax[0]={L0:.2f} → max_w[0]={alpha*L0+beta:.2f}")
    print(f"  anchor peak: lmax[9]={L9:.2f} → max_w[9]={alpha*L9+beta:.2f}")

    new_max_w = {K: alpha * lmax[K] + beta for K in range(N_RANKS)}
    # For the stub crown (slots 0..3 all have same lmax = stub), the formula
    # gives identical values automatically.

    print("\nslot  lmax   old_max_w  new_max_w  ratio")
    for K in range(N_RANKS):
        ratio = new_max_w[K] / old_max_w[K] if old_max_w[K] > 0 else 0
        print(f" {K:>4}  {lmax[K]:5.2f}   {old_max_w[K]:5.2f}    {new_max_w[K]:5.2f}   {ratio:.3f}")

    # Backup JSON
    json_backup = JSON_PATH.with_suffix(".json.bak_2026-05-12_pre-smooth")
    if not json_backup.exists():
        shutil.copy2(JSON_PATH, json_backup)
        print(f"\nbacked up: {json_backup}")

    # Scale asym_residual_grids_cm channel 0 (lateral) by the same ratio
    # max_w_new/max_w_old. asym_residual is in absolute cm and the C++
    # adds it to sym_x = (vc-0.5)*w*max_w. If we increase max_w without
    # scaling channel-0 asym proportionally, sym_x grows but asym doesn't —
    # the leaf shifts laterally instead of widening (and may NARROW because
    # sym_x can cancel asym at edges).
    arg = d["asym_residual_grids_cm"]
    for K in range(N_RANKS):
        grid = np.asarray(arg[str(K)])  # (n_u, n_v, 3)
        ratio = new_max_w[K] / old_max_w[K] if old_max_w[K] > 0 else 1.0
        grid[..., 0] = grid[..., 0] * ratio   # lateral channel only
        arg[str(K)] = grid.tolist()
    print(f"  scaled asym_residual_grids_cm channel 0 (lateral) by per-rank ratio")

    d["max_w_xml_cm"] = {str(K): float(new_max_w[K]) for K in range(N_RANKS)}
    d.setdefault("shift_history", []).append({
        "date": "2026-05-12",
        "operation": "smooth max_w bell",
        "rule": f"max_w = {alpha:.4f} × lmax + {beta:.4f}",
        "reason": "Replace bumpy MF3D-inherited per-rank max_w (which had "
                  "a secondary peak at slot 11) with a smooth monotonic bell. "
                  "Target peak max_w = 8 cm at slot 9 brings total rendered "
                  "area back to ~6900 cm² (donor-reality target).",
    })
    with open(JSON_PATH, "w") as f:
        json.dump(d, f, indent=2)
    print(f"wrote {JSON_PATH}")

    # ---- XML side: scale Width_blade, Width_petiole, areaMax ----
    xml_backup = XML_PATH.with_suffix(".xml.bak_2026-05-12_pre-smooth")
    if not xml_backup.exists():
        shutil.copy2(XML_PATH, xml_backup)
        print(f"backed up: {xml_backup}")

    tree = ET.parse(XML_PATH)
    root = tree.getroot()
    leaves = root.findall(".//leaf")
    for K, leaf in enumerate(leaves):
        params = {p.get("name"): p for p in leaf if p.get("name")}
        ratio = new_max_w[K] / old_max_w[K] if old_max_w[K] > 0 else 1.0
        if "Width_blade" in params:
            old_wb = float(params["Width_blade"].get("value", 0))
            params["Width_blade"].set("value", str(old_wb * ratio))
        if "Width_petiole" in params:
            old_wp = float(params["Width_petiole"].get("value", 0))
            params["Width_petiole"].set("value", str(old_wp * ratio))
        if "areaMax" in params:
            old_am = float(params["areaMax"].get("value", 0))
            params["areaMax"].set("value", str(old_am * ratio))
    tree.write(XML_PATH, encoding="UTF-8", xml_declaration=True)
    print(f"wrote {XML_PATH}")


if __name__ == "__main__":
    main()
