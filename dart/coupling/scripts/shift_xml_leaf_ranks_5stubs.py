"""Shift per-leaf size + surface_cp parameters in maize_calibrated.xml so
the FA kinetics and rendered shape stay aligned after the JSON +3 shift.

Mirror of shift_leaf_shape_ranks_5stubs.py for the XML side.

Mapping: new XML leaf subType K+2 (= XML slot K) copies from source slot
max(0, K - 3). Fields copied:

  - lmax              (FA growth target length)
  - Width_blade       (half-width for legacy paths)
  - Width_petiole     (= Width_blade × 0.3)
  - areaMax           (static blade area)
  - surface_cp grid   (55 entries: 11x5 CPs, full canonical shape)

NOT shifted (positional, not size-dependent):
  - InitBeta, theta   (per-leaf azimuth / insertion angle)
  - tropismAge, tropismS  (per-leaf bend kinetics)
  - phyllochron_tt    (per-leaf emergence timing)
  - shape_rank_index  (stays = slot K; identity dispatch to shifted JSON)

NOT shifted yet (will be re-derived by bake_calibration_to_xml.py):
  - R1_n, R2_n, lag_exp_n  (FA kinetics derived from new lmax)
  - sigma_extension_n, tau_extension_n  (same)
"""
from __future__ import annotations
import shutil
from pathlib import Path
import xml.etree.ElementTree as ET

XML_PATH = Path("/home/lukas/PHD/CPlantBox/dart/coupling/data/maize_calibrated.xml")
SHIFT = 3
N_LEAVES = 15
FIELDS_TO_COPY = ["lmax", "Width_blade", "Width_petiole", "areaMax"]


def _params_by_name(leaf):
    return {p.get("name"): p for p in leaf if p.get("name") is not None}


def main():
    backup = XML_PATH.with_suffix(".xml.bak_2026-05-12_pre-5stub")
    if not backup.exists():
        shutil.copy2(XML_PATH, backup)
        print(f"backed up: {backup}")

    tree = ET.parse(XML_PATH)
    root = tree.getroot()

    leaves = root.findall(".//leaf")
    if len(leaves) != N_LEAVES:
        raise RuntimeError(f"expected {N_LEAVES} leaves, found {len(leaves)}")

    # Build source-slot snapshot BEFORE mutating (so we read pristine values)
    source_snapshot = []
    for leaf in leaves:
        params = _params_by_name(leaf)
        snap = {}
        for f in FIELDS_TO_COPY:
            if f in params:
                snap[f] = params[f].get("value")
        # surface_cp grid: list of dicts with u, v, x, y, z
        cps = []
        for p in leaf.findall("parameter[@name='surface_cp']"):
            cps.append({
                "u": p.get("u"), "v": p.get("v"),
                "x": p.get("x"), "y": p.get("y"), "z": p.get("z"),
            })
        snap["surface_cp"] = cps
        source_snapshot.append(snap)

    print("\nBefore shift (lmax per slot):")
    for K, snap in enumerate(source_snapshot):
        print(f"  slot {K:>2}: lmax={float(snap.get('lmax', 0)):6.2f}  "
              f"Width_blade={float(snap.get('Width_blade', 0)):5.2f}")

    # Apply the shift: new slot K reads from source slot max(0, K - SHIFT)
    for K, leaf in enumerate(leaves):
        src_K = max(0, K - SHIFT)
        src = source_snapshot[src_K]
        params = _params_by_name(leaf)
        for f in FIELDS_TO_COPY:
            if f in src and f in params:
                params[f].set("value", str(src[f]))
        # Replace surface_cp entries
        for p in list(leaf.findall("parameter[@name='surface_cp']")):
            leaf.remove(p)
        for cp_snap in src["surface_cp"]:
            cp = ET.SubElement(leaf, "parameter")
            cp.set("name", "surface_cp")
            cp.set("u", cp_snap["u"])
            cp.set("v", cp_snap["v"])
            cp.set("x", cp_snap["x"])
            cp.set("y", cp_snap["y"])
            cp.set("z", cp_snap["z"])

    print("\nAfter shift (lmax per slot):")
    for K, leaf in enumerate(leaves):
        params = _params_by_name(leaf)
        print(f"  slot {K:>2}: lmax={float(params['lmax'].get('value')):6.2f}  "
              f"Width_blade={float(params['Width_blade'].get('value')):5.2f}")

    tree.write(XML_PATH, encoding="UTF-8", xml_declaration=True)
    print(f"\nWrote {XML_PATH}")


if __name__ == "__main__":
    main()
