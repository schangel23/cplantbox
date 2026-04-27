#!/usr/bin/env python3
"""Inject MaizeField3D-derived blade deformation splines into maize_calibrated.xml.

Per CAPABILITY_AUDIT §5.2: populate the four dormant spline sets
(leafOOPCurvature, leafAsymmetry, leafEdgeCurl, leafCrossSection) with
5 knots each, per leaf subType, using MF3D population medians.

Inputs:
  - maizefield3d_audit.json          -> kappa_median, gutter_median_cm,
                                        width_profile_median_cm per position
  - maizefield3d_blade_deformation.json -> curl_amp_median, avg_curl_ramp

Mapping:
  subType 2..12  ->  MF3D position 0..10 (stem-base to top).

Units:
  leafOOPCurvature kappa  : 1/cm  (lofter integrates kappa * ds along binormal)
  leafAsymmetry    offset : cm    (lateral midrib shift along binormal)
  leafEdgeCurl     angle  : rad   (lofter uses tan(angle) * w/2 * (2|frac|)^3)
  leafCrossSection curv   : cm    (additional gutter depth atop adapter's 18% rule)

leafAsymmetry is populated with zeros: MF3D population mean shows no
systematic lateral bias (per-plant curl randomly signs).

leafCurvature (midrib kappa) is explicitly skipped — lofter does not yet
consume it; midrib bending comes from tropism + skeleton injection.
"""

import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

KNOTS = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
SPLINE_NAMES = (
    "leafOOPCurvature",
    "leafAsymmetry",
    "leafEdgeCurl",
    "leafCrossSection",
)


def _resample_to_knots(values, n_src=None):
    """Linearly interpolate a per-station profile onto the 5 knot positions."""
    arr = np.asarray(values, dtype=float)
    if n_src is None:
        n_src = len(arr)
    src_phi = np.linspace(0.0, 1.0, n_src)
    return np.interp(KNOTS, src_phi, arr)


def _edge_curl_angles(curl_ramp, curl_amp_cm, max_width_cm):
    """Convert curl amplitude + ramp shape to per-knot edge-curl angles (rad).

    Lofter: edge displacement ~= tan(angle) * w/2 at the edge (frac=±0.5).
    Inverted: angle = atan(2 * displacement / w).
    """
    ramp5 = _resample_to_knots(curl_ramp, n_src=len(curl_ramp))
    ramp5 = np.clip(ramp5, 0.0, None)
    peak = float(np.max(ramp5)) if np.max(ramp5) > 1e-6 else 1.0
    displacement_cm = (ramp5 / peak) * float(curl_amp_cm)
    w = max(float(max_width_cm), 1e-3)
    return np.arctan(2.0 * displacement_cm / w)


def build_splines_for_position(audit_pos, deform_pos):
    """Build the 4 per-knot spline arrays for one MF3D stem position."""
    oop_kappa = _resample_to_knots(audit_pos["kappa_median"])
    gutter = _resample_to_knots(audit_pos["gutter_median_cm"])
    asym = np.zeros_like(KNOTS)

    width_profile = audit_pos.get("width_profile_median_cm") or []
    max_w = float(max(width_profile)) if width_profile else 6.0
    edge_curl = _edge_curl_angles(
        deform_pos.get("avg_curl_ramp", [0.0] * 10),
        deform_pos.get("curl_amp_median", 0.0),
        max_w,
    )
    return {
        "leafOOPCurvature": ("kappa", oop_kappa),
        "leafAsymmetry": ("offset", asym),
        "leafEdgeCurl": ("angle", edge_curl),
        "leafCrossSection": ("curv", gutter),
    }


def _drop_existing_splines(leaf_elem):
    for name in SPLINE_NAMES:
        for p in list(leaf_elem.findall(f"parameter[@name='{name}']")):
            leaf_elem.remove(p)


def _append_spline(leaf_elem, name, attr, values):
    for phi, v in zip(KNOTS, values):
        p = ET.SubElement(leaf_elem, "parameter")
        p.set("name", name)
        p.set("phi", f"{phi:.4f}")
        p.set(attr, f"{float(v):.6f}")


def inject_splines(xml_path, audit_path, deform_path, output_path=None,
                   start_subtype=2):
    audit = json.loads(Path(audit_path).read_text())["per_position"]
    deform = json.loads(Path(deform_path).read_text())["per_position"]

    tree = ET.parse(xml_path)
    root = tree.getroot()

    injected = 0
    for leaf in root.findall(".//leaf"):
        sub = leaf.get("subType")
        if sub is None:
            continue
        pos = int(sub) - start_subtype
        if pos < 0:
            continue
        audit_pos = audit.get(str(pos))
        if audit_pos is None or pos >= len(deform):
            continue
        splines = build_splines_for_position(audit_pos, deform[pos])
        _drop_existing_splines(leaf)
        for name, (attr, values) in splines.items():
            _append_spline(leaf, name, attr, values)
        injected += 1

    out = Path(output_path) if output_path else Path(xml_path)
    tree.write(out, encoding="utf-8", xml_declaration=True)
    return injected, out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--xml", default="dart/coupling/data/maize_calibrated.xml",
                    help="target maize XML (updated in place if --output omitted)")
    ap.add_argument("--audit",
                    default="Resources/MaizeField3d/maizefield3d_audit.json")
    ap.add_argument("--deformation",
                    default="Resources/MaizeField3d/maizefield3d_blade_deformation.json")
    ap.add_argument("--output", default=None, help="write result here (defaults to --xml)")
    ap.add_argument("--start-subtype", type=int, default=2,
                    help="subType of position 0 (default 2 for maize)")
    args = ap.parse_args()

    n, out = inject_splines(
        args.xml, args.audit, args.deformation,
        output_path=args.output, start_subtype=args.start_subtype,
    )
    print(f"injected splines into {n} leaf subTypes -> {out}")


if __name__ == "__main__":
    main()
