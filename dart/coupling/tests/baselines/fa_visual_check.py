#!/usr/bin/env python3
"""Generate FA-on vs FA-off OBJ meshes for visual comparison.

Grows maize_calibrated under Juelich 2024 met (seed 7) twice — once with
Fournier-Andrieu kinetics on, once with the scalar fallback — at multiple
growth days, lofts via the standard cplantbox_adapter + g1_to_g3 pipeline,
and writes paired OBJs to dart/coupling/output/fa_visual_check/.

Open the OBJ files in Blender, MeshLab, or any 3D viewer to compare.

Usage (from /home/lukas/PHD/CPlantBox):
    cpbenv/bin/python3 dart/coupling/tests/baselines/fa_visual_check.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

BASELINE_DIR = Path(__file__).resolve().parent
COUPLING_DIR = BASELINE_DIR.parent.parent
CPLANTBOX_ROOT = COUPLING_DIR.parent.parent
sys.path.insert(0, str(CPLANTBOX_ROOT))

import plantbox as pb  # noqa: E402

from dart.coupling.carbon.dvs_partitioning import get_daily_met  # noqa: E402
from dart.coupling.geometry.cplantbox_adapter import extract_organs_for_lofter  # noqa: E402
from dart.coupling.geometry.g1_to_g3 import loft_organs  # noqa: E402
from dart.coupling.growth.grow import setup_successor_where, enable_fa_on_mainstem  # noqa: E402,F401


XML_PATH = COUPLING_DIR / "data" / "maize_calibrated.xml"
KINETICS_PATH = COUPLING_DIR / "data" / "phase_III_per_rank.json"
OUT_DIR = COUPLING_DIR / "output" / "fa_visual_check"
SEED = 7
GROWTH_DAYS = [30, 60, 100, 130]
MAX_RANK = 16


# Local load_fa_kinetics + enable_fa_on_mainstem were duplicated by the
# shared helper in ``dart.coupling.growth.grow`` (imported above). Kept the
# re-export so existing call sites continue to work unchanged.


def grow(days: int, fa_on: bool):
    plant = pb.MappedPlant(SEED)
    plant.readParameters(str(XML_PATH))
    plant.setSeed(SEED)
    setup_successor_where(plant)
    if fa_on:
        enable_fa_on_mainstem(plant)
    plant.initialize()

    met_lookup = get_daily_met(daily_met=None)
    total = 0.0
    while total < days:
        step = min(1.0, days - total)
        sim_day_1b = int(total) + 1
        T_air = float(met_lookup[sim_day_1b]["T_mean_C"]) if (met_lookup and sim_day_1b in met_lookup) else 25.0
        plant.setAirTemperature(T_air)
        try:
            plant.simulate(step, False)
            total += step
        except (IndexError, RuntimeError) as e:
            print(f"  simulate() error at day {total + step:.1f}: {e}")
            break
    return plant


def mainstem_top_z(plant):
    for o in plant.getOrgans():
        if o.organType() == pb.OrganTypes.stem and int(o.getParameter("subType")) == 1:
            nodes = list(o.getNodes())
            if nodes:
                return max(float(n.z) for n in nodes)
    return float("nan")


def export_pair(day: int):
    rows = []
    for fa_on in (False, True):
        tag = "fa_on" if fa_on else "fa_off"
        print(f"\n[day {day:>3} | {tag}] growing...")
        plant = grow(day, fa_on)
        top_z = mainstem_top_z(plant)
        print(f"[day {day:>3} | {tag}] mainstem top z = {top_z:.2f} cm")

        organs = extract_organs_for_lofter(plant)
        mesh = loft_organs(organs, target_spacing=0.5)
        out_path = OUT_DIR / f"maize_day{day:03d}_{tag}.obj"
        mesh.to_obj(out_path, group_by_organ=True)
        verts = mesh.vertices
        bbox = (
            float(verts[:, 0].min()), float(verts[:, 0].max()),
            float(verts[:, 1].min()), float(verts[:, 1].max()),
            float(verts[:, 2].min()), float(verts[:, 2].max()),
        )
        print(f"[day {day:>3} | {tag}] {mesh.n_triangles} tris  ->  {out_path.relative_to(CPLANTBOX_ROOT)}")
        print(f"[day {day:>3} | {tag}] bbox cm  x[{bbox[0]:.1f},{bbox[1]:.1f}]  y[{bbox[2]:.1f},{bbox[3]:.1f}]  z[{bbox[4]:.1f},{bbox[5]:.1f}]")
        rows.append((day, tag, top_z, mesh.n_triangles, bbox))
    return rows


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Writing OBJs to: {OUT_DIR}")
    summary = []
    for day in GROWTH_DAYS:
        summary.extend(export_pair(day))

    print("\n=== Summary ===")
    print(f"{'day':>4} {'variant':>7} {'top_z':>8} {'tris':>8}  bbox(cm)")
    for day, tag, top_z, ntri, bbox in summary:
        print(f"{day:>4} {tag:>7} {top_z:>7.2f}  {ntri:>8}  "
              f"x[{bbox[0]:.1f},{bbox[1]:.1f}] y[{bbox[2]:.1f},{bbox[3]:.1f}] z[{bbox[4]:.1f},{bbox[5]:.1f}]")
    print(f"\nDone. Open *.obj in Blender / MeshLab to compare FA-on vs FA-off.")


if __name__ == "__main__":
    main()
