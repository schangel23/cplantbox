"""Per-leaf rescale of surface_cp x-coords so CP-encoded area matches XML areaMax.

Diagnosis (2026-05-08): the surface_cp grids in dart/coupling/data/maize_calibrated.xml
encode peak blade widths roughly half of what XML metadata (Width_blade, areaMax)
asks for — the lateral (x) coordinate convention used by the CP-fit script
disagreed with the Width_blade × 2 = peak convention used by calibrate.py and
the lofter. Result: at day 130 the rendered NURBS surface area is ~44 % of
the calibrated total leaf area, vs DART R1 references at 92–122 %.

Fix: per-leaf, scale all surface_cp x values so

    cp_int_area = arc_along_v_mid × (x_max - x_min) × 0.73

equals areaMax, where arc and 0.73 (mean shape factor) are unchanged. y/z
remain untouched — only the lateral axis is corrected.

Run from CPlantBox repo root:

    python3 dart/coupling/scripts/fix_surface_cp_lateral.py \
        dart/coupling/data/maize_calibrated.xml
"""
from __future__ import annotations

import sys
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

SHAPE_FACTOR = 0.73


def _cp_geometry(cps: np.ndarray) -> tuple[float, float, float]:
    """Return (mid-arc cm, peak lateral span cm, integrated area cm²)."""
    v_unique = sorted(set(cps[:, 1]))
    v_mid = v_unique[len(v_unique) // 2]
    midrib = cps[cps[:, 1] == v_mid]
    midrib = midrib[np.argsort(midrib[:, 0])]
    arc = float(np.sum(np.linalg.norm(np.diff(midrib[:, 2:5], axis=0), axis=1)))
    peak = float(cps[:, 2].max() - cps[:, 2].min())
    area = arc * peak * SHAPE_FACTOR
    return arc, peak, area


def patch(xml_path: Path) -> None:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    print(f"Patching {xml_path}\n")
    print(f"{'leaf':<18} {'lmax':>5} {'aMax':>6} | "
          f"{'cp_pkW_pre':>10} {'cp_area_pre':>11} | "
          f"{'scale':>6} {'cp_pkW_post':>11} {'cp_area_post':>12}")
    for leaf in root.findall('.//leaf'):
        params = leaf.findall("parameter")
        cp_params = [p for p in params if p.get('name') == 'surface_cp']
        if not cp_params:
            continue
        lmax = float(leaf.find("parameter[@name='lmax']").get('value'))
        a_max = float(leaf.find("parameter[@name='areaMax']").get('value'))
        cps = np.array([
            [float(p.get('u')), float(p.get('v')),
             float(p.get('x')), float(p.get('y')), float(p.get('z'))]
            for p in cp_params
        ])
        arc, peak_pre, area_pre = _cp_geometry(cps)
        if area_pre <= 0:
            continue
        scale = a_max / area_pre  # area scales linearly with x_scale
        for p in cp_params:
            p.set('x', f"{float(p.get('x')) * scale:.6f}")
        # Verify
        cps2 = np.array([
            [float(p.get('u')), float(p.get('v')),
             float(p.get('x')), float(p.get('y')), float(p.get('z'))]
            for p in cp_params
        ])
        _, peak_post, area_post = _cp_geometry(cps2)
        print(f"{leaf.get('name'):<18} {lmax:5.1f} {a_max:6.0f} | "
              f"{peak_pre:10.2f} {area_pre:11.0f} | "
              f"{scale:6.3f} {peak_post:11.2f} {area_post:12.0f}")
    tree.write(xml_path, encoding='UTF-8', xml_declaration=True)
    print(f"\nWrote: {xml_path}")


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    patch(Path(sys.argv[1]))
