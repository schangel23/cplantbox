"""Smooth-taper lmax across slots 0..9 to widen the four identical-stub
crown left by the 2026-05-12 +3 rank shift. Targets the donor-reality
~6900 cm² total rendered area (current 5769 cm², gap is the crown).

Per-slot lmax targets (monotone interpolation 24.0 → 78.9 cm across
slots 0..9; slots 10..14 already in tapered descent, unchanged):

    slot 0:  24.0   (stub, unchanged)
    slot 1:  35.0
    slot 2:  45.0
    slot 3:  52.5
    slot 4:  60.0
    slot 5:  67.0
    slot 6:  72.5
    slot 7:  76.0
    slot 8:  78.0
    slot 9:  78.9   (peak, unchanged)

max_w follows automatically via the smooth rule baked on 2026-05-12:

    max_w = 0.0989 × lmax + 0.2024

(anchors stub max_w=2.57 cm at lmax=24, peak max_w=8.0 cm at lmax=78.9).

Render formula consequence (leafshape.cpp:260-262):
    sym_x = (v - 0.5) × w × max_w_intercept
    sym_y = m_y × lmax_intercept
    sym_z = m_z × lmax_intercept

So when lmax changes:
  - sym_y, sym_z scale by lmax_ratio
  - sym_x scales by max_w_ratio (via the smooth rule)

To keep the rendered envelope proportional, the asym_residual_grid scales
by the SAME per-channel ratios (lateral = max_w_ratio, OOP+along =
lmax_ratio). Otherwise the asym contribution becomes a smaller fraction of
sym at the new size and the leaf renders narrower/shorter than intended.
This generalises the channel-0 scaling discovery from smooth_maxw_bell_
2026-05-12.py to a joint lmax+max_w change.

Updates (JSON):
    - lmax_intercept_cm[K]       = new_lmax
    - lmax_xml_cm[K]             = new_lmax
    - max_w_xml_cm[K]            = 0.0989 × new_lmax + 0.2024
    - asym_residual_grids_cm[K][..., 0] × max_w_ratio   (lateral)
    - asym_residual_grids_cm[K][..., 1] × lmax_ratio    (OOP/droop)
    - asym_residual_grids_cm[K][..., 2] × lmax_ratio    (along)

Updates (XML):
    - lmax              × lmax_ratio   (FA target; bake then re-derives R2/lag)
    - Width_blade       × max_w_ratio
    - Width_petiole     × max_w_ratio  (= Width_blade × 0.3)
    - areaMax           × max_w_ratio × lmax_ratio
    - surface_cp.x      × max_w_ratio
    - surface_cp.y      × lmax_ratio
    - surface_cp.z      × lmax_ratio

NOT touched:
    - intercepts (33-vec dimensionless splines; invariant to lmax/max_w)
    - covariance, cholesky_factor, pca_truncation (per-plant shape draw)
    - InitBeta, theta, tropism, phyllochron, stem internodes, FA stem fields
    - FA leaf kinetics R1_n / R2_n / lag_exp_n / D_lin_n / T0_n / L_min
      (these are re-derived by `bake_calibration_to_xml.py` after this
      script writes the new XML lmax values — see step note at end).
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

# Smooth max_w rule (from 2026-05-12 smoothing pass).
ALPHA_MAXW = 0.0989
BETA_MAXW = 0.2024

# Target lmax per slot — monotone smooth taper across slots 0..9; the
# descending tail (10..14) stays at its current values.
TAPER_LMAX_TARGETS = {
    0: 24.0,    # stub (unchanged from current 23.94)
    1: 35.0,
    2: 45.0,
    3: 52.5,
    4: 60.0,
    5: 67.0,
    6: 72.5,
    7: 76.0,
    8: 78.0,
    9: 78.9,    # peak (unchanged from current 78.90)
}


def main():
    # ---- Backups ----
    json_backup = JSON_PATH.with_suffix(".json.bak_2026-05-13_pre-taper")
    if not json_backup.exists():
        shutil.copy2(JSON_PATH, json_backup)
        print(f"backed up: {json_backup}")

    xml_backup = XML_PATH.with_suffix(".xml.bak_2026-05-13_pre-taper")
    if not xml_backup.exists():
        shutil.copy2(XML_PATH, xml_backup)
        print(f"backed up: {xml_backup}")

    # ---- Load ----
    with open(JSON_PATH) as f:
        d = json.load(f)

    old_lmax = {int(k): float(v) for k, v in d["lmax_intercept_cm"].items()}
    old_max_w = {int(k): float(v) for k, v in d["max_w_xml_cm"].items()}

    # ---- Compute per-slot new values + ratios ----
    print("\nslot  old_lmax  new_lmax  lmax_ratio  old_max_w  new_max_w  max_w_ratio")
    new_lmax = {}
    new_max_w = {}
    lmax_ratio = {}
    max_w_ratio = {}
    for K in range(N_RANKS):
        new_l = TAPER_LMAX_TARGETS.get(K, old_lmax[K])  # slots 10..14 unchanged
        new_l = float(new_l)
        new_w = ALPHA_MAXW * new_l + BETA_MAXW
        new_lmax[K] = new_l
        new_max_w[K] = new_w
        lmax_ratio[K] = new_l / old_lmax[K] if old_lmax[K] > 0 else 1.0
        max_w_ratio[K] = new_w / old_max_w[K] if old_max_w[K] > 0 else 1.0
        print(f" {K:>4}    {old_lmax[K]:6.2f}   {new_l:6.2f}      {lmax_ratio[K]:.3f}    "
              f"{old_max_w[K]:5.2f}     {new_w:5.2f}      {max_w_ratio[K]:.3f}")

    # ---- JSON updates ----
    # 1) lmax_intercept_cm + lmax_xml_cm + max_w_xml_cm
    d["lmax_intercept_cm"] = {str(K): float(new_lmax[K]) for K in range(N_RANKS)}
    d["lmax_xml_cm"] = {str(K): float(new_lmax[K]) for K in range(N_RANKS)}
    d["max_w_xml_cm"] = {str(K): float(new_max_w[K]) for K in range(N_RANKS)}

    # 2) asym_residual_grids_cm — scale per channel
    arg = d["asym_residual_grids_cm"]
    for K in range(N_RANKS):
        grid = np.asarray(arg[str(K)], dtype=float)  # (n_u, n_v, 3)
        grid[..., 0] = grid[..., 0] * max_w_ratio[K]   # lateral
        grid[..., 1] = grid[..., 1] * lmax_ratio[K]    # OOP / droop
        grid[..., 2] = grid[..., 2] * lmax_ratio[K]    # along
        arg[str(K)] = grid.tolist()
    print("\nscaled asym_residual_grids_cm channels: 0 by max_w_ratio, 1+2 by lmax_ratio")

    d.setdefault("shift_history", []).append({
        "date": "2026-05-13",
        "operation": "smooth taper lmax slots 0..9",
        "rule": (
            "Smooth taper of lmax across slots 0..9 (24 → 78.9 cm) to widen "
            "the 4 identical-stub crown left by the +3 rank shift. max_w "
            f"follows via {ALPHA_MAXW} × lmax + {BETA_MAXW}; asym_residual "
            "channel 0 scaled by max_w_ratio, channels 1+2 by lmax_ratio."
        ),
        "reason": (
            "Restore donor-reality area target of ~6900 cm² total rendered "
            "(was 5769 cm² post-+3-shift). See DIAG_MAIZE_LEAF_BOTTOM_HEAVY_"
            "2026-05-12 §'Future tunings (deferred)' option 1, smooth-taper "
            "variant (slots 0..9 not 0..3) to preserve monotonicity."
        ),
        "lmax_targets": {str(K): float(new_lmax[K]) for K in sorted(TAPER_LMAX_TARGETS)},
    })

    with open(JSON_PATH, "w") as f:
        json.dump(d, f, indent=2)
    print(f"wrote {JSON_PATH}")

    # ---- XML updates ----
    tree = ET.parse(XML_PATH)
    root = tree.getroot()
    leaves = root.findall(".//leaf")
    if len(leaves) != N_RANKS:
        raise RuntimeError(f"expected {N_RANKS} leaves in XML, found {len(leaves)}")

    print("\nslot   XML lmax_old  XML lmax_new  Width_blade_old  Width_blade_new")
    for K, leaf in enumerate(leaves):
        lr = lmax_ratio[K]
        wr = max_w_ratio[K]
        params = {p.get("name"): p for p in leaf if p.get("name")}

        # Scale named scalars
        if "lmax" in params:
            old_lm = float(params["lmax"].get("value", 0.0))
            params["lmax"].set("value", str(old_lm * lr))
            print(f" {K:>4}      {old_lm:6.2f}        {old_lm * lr:6.2f}", end="")
        if "Width_blade" in params:
            old_wb = float(params["Width_blade"].get("value", 0.0))
            params["Width_blade"].set("value", str(old_wb * wr))
            print(f"          {old_wb:5.2f}            {old_wb * wr:5.2f}")
        else:
            print()
        if "Width_petiole" in params:
            old_wp = float(params["Width_petiole"].get("value", 0.0))
            params["Width_petiole"].set("value", str(old_wp * wr))
        if "areaMax" in params:
            old_am = float(params["areaMax"].get("value", 0.0))
            # area ∝ width × length
            params["areaMax"].set("value", str(old_am * wr * lr))

        # Scale surface_cp grid in place
        for cp in leaf.findall("parameter[@name='surface_cp']"):
            x = float(cp.get("x", 0.0)); y = float(cp.get("y", 0.0)); z = float(cp.get("z", 0.0))
            cp.set("x", str(x * wr))
            cp.set("y", str(y * lr))
            cp.set("z", str(z * lr))

    tree.write(XML_PATH, encoding="UTF-8", xml_declaration=True)
    print(f"wrote {XML_PATH}")

    print("\nNEXT: run `python -m dart.coupling.scripts.bake_calibration_to_xml`")
    print("      to re-derive FA leaf kinetics (R2_n, lag_exp_n) from the new lmax.")


if __name__ == "__main__":
    main()
