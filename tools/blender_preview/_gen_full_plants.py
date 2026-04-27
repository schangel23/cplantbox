"""Generate full-plant OBJs for the three library variants.

For each of {median, draw42, draw99}:
  1. Calibrate an XML with the corresponding surface_cps library mode
  2. Grow the plant to day 55 with a fixed CPlantBox seed (so the skeleton
     is identical across variants — only the leaf CP library changes)
  3. Loft via the NURBS backend (so surface_cps + muted deformations are used)
  4. Write <variant>.obj

Also writes a legacy-backend OBJ (no surface_cps — falls back to the
leafGeometry-based lofter) for comparison.
"""
from __future__ import annotations
import subprocess
import sys
from pathlib import Path

from dart.coupling.growth.grow import grow_plant, extract_g3_mesh


REPO = Path(__file__).resolve().parents[4]
DATA = REPO / "dart" / "coupling" / "data"
OUT = Path(__file__).resolve().parent
TEMPLATE = REPO / "modelparameter" / "structural" / "plant" / "2020-maize.xml"
MF3D = Path("/home/lukas/PHD/Resources/MaizeField3d/maizefield3d_stats.json")

SIM_DAY = 55
PLANT_SEED = 7  # fixed so skeletons match across variants


def calibrate(out_xml: Path, extra_args: list[str]) -> None:
    cmd = [
        sys.executable, "-m", "dart.coupling.growth.calibrate",
        "--template", str(TEMPLATE),
        "--output", str(out_xml),
        "--maizefield3d", str(MF3D),
    ] + extra_args
    print(f"\n--- Calibrating {out_xml.name} ---")
    subprocess.run(cmd, check=True, cwd=str(REPO), capture_output=True)


def grow_and_export(xml_path: Path, obj_path: Path, use_nurbs: bool) -> None:
    print(f"\n--- Growing + exporting {obj_path.name} "
          f"(nurbs={use_nurbs}) ---")
    plant = grow_plant(xml_path, simulation_time=SIM_DAY, seed=PLANT_SEED)
    mesh, _ = extract_g3_mesh(
        plant,
        use_nurbs_leaf_backend=use_nurbs,
        nurbs_leaf_n_u_eval=30,
        nurbs_leaf_n_v_eval=7,
    )
    mesh.to_obj(str(obj_path))
    print(f"  wrote {obj_path} ({mesh.n_vertices} verts, "
          f"{mesh.n_triangles} tris)")


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    xml_median = OUT / "maize_median.xml"
    xml_d42 = OUT / "maize_draw42.xml"
    xml_d99 = OUT / "maize_draw99.xml"
    xml_coh42 = OUT / "maize_coherent42.xml"
    xml_coh99 = OUT / "maize_coherent99.xml"

    calibrate(xml_median, ["--surface-cps"])  # default: median library
    calibrate(xml_d42, ["--surface-cps-draw-seed", "42"])
    calibrate(xml_d99, ["--surface-cps-draw-seed", "99"])
    calibrate(xml_coh42, ["--surface-cps-draw-coherent-seed", "42"])
    calibrate(xml_coh99, ["--surface-cps-draw-coherent-seed", "99"])

    grow_and_export(xml_median, OUT / "plant_median.obj", use_nurbs=True)
    grow_and_export(xml_d42, OUT / "plant_draw42.obj", use_nurbs=True)
    grow_and_export(xml_d99, OUT / "plant_draw99.obj", use_nurbs=True)
    grow_and_export(xml_coh42, OUT / "plant_coherent42.obj", use_nurbs=True)
    grow_and_export(xml_coh99, OUT / "plant_coherent99.obj", use_nurbs=True)


if __name__ == "__main__":
    main()
