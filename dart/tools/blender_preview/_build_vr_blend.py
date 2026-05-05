"""Build a Blender .blend file with all V/R-stage OBJs from ``vr_stages/``
arranged in a row along +X. Each plant's origin sits at its stem base
(world z=0 in CPlantBox coords, preserved through scaling).

Usage:
    blender --background --python dart/tools/blender_preview/_build_vr_blend.py

Reads OBJs from ``dart/coupling/output/vr_stages/*.obj`` (paired with sidecar
.mtl), saves ``dart/coupling/output/vr_stages/vr_stages_row.blend``.
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path

import bpy

# Output: canonical pipeline location (resolves regardless of CWD).
# Script lives at dart/tools/blender_preview/<name>.py → parents[3] = project root.
HERE = Path(__file__).resolve()
PROJECT_ROOT = HERE.parents[3]
OUT_DIR = PROJECT_ROOT / "dart" / "coupling" / "output" / "vr_stages"
BLEND_PATH = OUT_DIR / "vr_stages_row.blend"
DEFAULT_XML = PROJECT_ROOT / "dart" / "coupling" / "data" / "maize_calibrated.xml"

# Row spacing (Blender meters). Mature maize ≈ 1.5 m wide leaf spread, so
# 1.6 m keeps neighbouring plants from clipping and matches a realistic field
# inter-row distance.
ROW_SPACING_M = 1.6
# CPlantBox world coords are in cm; Blender default unit = m. Scale verts
# during import so 150 cm plants render at 1.5 m.
CM_TO_M = 0.01
# Label height above ground (m).
LABEL_Z = 2.2
LABEL_SCALE = 0.10


_DAY_RE = re.compile(r"day(\d{3})_(.+)")


def parse_stem(stem: str) -> tuple[int, str]:
    """OBJ filename stem 'maize_day055_V8' → (55, 'V8')."""
    m = _DAY_RE.search(stem)
    return (int(m.group(1)), m.group(2)) if m else (9999, stem)


def read_seed_pos_cm(xml_path: Path) -> tuple[float, float]:
    """Pull seedPos.x / seedPos.y (cm) out of a CPlantBox parameter XML.

    Falls back to (0, 0) if the params can't be parsed — matches pre-2026
    XMLs that emerged at world origin.
    """
    try:
        root = ET.parse(xml_path).getroot()
        xy = {"x": 0.0, "y": 0.0}
        for p in root.iter("parameter"):
            name = p.get("name", "")
            if name in ("seedPos.x", "seedPos.y"):
                xy[name[-1]] = float(p.get("value", "0"))
        return xy["x"], xy["y"]
    except (ET.ParseError, OSError, ValueError):
        return 0.0, 0.0


def clear_scene() -> None:
    """Remove default cube/camera/light so we start clean."""
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    for mesh in list(bpy.data.meshes):
        bpy.data.meshes.remove(mesh)
    for light in list(bpy.data.lights):
        bpy.data.lights.remove(light)
    for cam in list(bpy.data.cameras):
        bpy.data.cameras.remove(cam)


def import_and_join(obj_path: Path, name: str, seed_xy_m: tuple[float, float]):
    """Import OBJ + MTL, join all parts into one object, return the joined obj.

    CPlantBox emits world-space vertex coords with the stem base at the XML's
    ``seedPos`` (currently (200, 200, -3) cm in maize_calibrated.xml — non-zero
    so DuMux-Rosi / AgroC voxel grids can centre on the plant). We scale verts
    cm→m on import, then translate mesh data by ``-seed_xy_m`` so the stem
    base sits at local (0, 0) — that way ``object.location`` coincides with
    the stem base and the row + label placement code stays simple.

    A whole-plant translation preserves all relative geometry, so senescent
    leaves drooping below z=0 and tassel billboards above the canopy stay
    rigidly attached to the stem.
    """
    bpy.ops.object.select_all(action="DESELECT")
    bpy.ops.wm.obj_import(filepath=str(obj_path), global_scale=CM_TO_M)
    imported = list(bpy.context.selected_objects)
    if not imported:
        return None

    bpy.context.view_layer.objects.active = imported[0]
    if len(imported) > 1:
        bpy.ops.object.join()
    joined = bpy.context.active_object
    joined.name = name

    dx, dy = seed_xy_m
    if abs(dx) > 1e-6 or abs(dy) > 1e-6:
        mesh = joined.data
        for v in mesh.vertices:
            v.co.x -= dx
            v.co.y -= dy
        mesh.update()
    return joined


def add_label(text: str, x: float, name: str):
    """3-D text object hovering above each plant, facing +Y so a -Y camera reads it."""
    bpy.ops.object.text_add(location=(x, 0.0, LABEL_Z))
    label = bpy.context.active_object
    label.data.body = text
    label.data.align_x = "CENTER"
    label.data.align_y = "CENTER"
    label.scale = (LABEL_SCALE, LABEL_SCALE, LABEL_SCALE)
    label.rotation_euler = (1.5708, 0.0, 0.0)  # 90° around X — text upright
    label.name = name
    return label


def add_ground(row_len: float):
    bpy.ops.mesh.primitive_plane_add(size=row_len + 4.0,
                                     location=(row_len / 2.0, 0.0, 0.0))
    ground = bpy.context.active_object
    ground.name = "ground"
    mat = bpy.data.materials.new(name="ground_mat")
    mat.diffuse_color = (0.25, 0.18, 0.10, 1.0)
    ground.data.materials.append(mat)
    return ground


def add_lighting():
    bpy.ops.object.light_add(type="SUN", location=(0.0, 0.0, 8.0))
    sun = bpy.context.active_object
    sun.name = "sun"
    sun.data.energy = 4.0
    sun.rotation_euler = (0.7, 0.3, 0.0)


def add_camera(row_len: float):
    cam_x = row_len / 2.0
    cam_y = -max(row_len * 0.6, 5.0)
    cam_z = 1.5
    bpy.ops.object.camera_add(location=(cam_x, cam_y, cam_z),
                              rotation=(1.4, 0.0, 0.0))
    cam = bpy.context.active_object
    cam.name = "camera"
    cam.data.lens = 35.0
    bpy.context.scene.camera = cam


def main() -> int:
    if not OUT_DIR.exists():
        print(f"ERROR: {OUT_DIR} not found — run _gen_vr_stages.py first")
        return 1

    objs = sorted(OUT_DIR.glob("maize_day*.obj"),
                  key=lambda p: parse_stem(p.stem))
    if not objs:
        print(f"ERROR: no OBJs in {OUT_DIR}")
        return 1
    print(f"Found {len(objs)} OBJ files in {OUT_DIR}")

    seed_x_cm, seed_y_cm = read_seed_pos_cm(DEFAULT_XML)
    seed_xy_m = (seed_x_cm * CM_TO_M, seed_y_cm * CM_TO_M)
    print(f"seedPos (cm) = ({seed_x_cm:.1f}, {seed_y_cm:.1f}) "
          f"→ recentre offset (m) = ({seed_xy_m[0]:.2f}, {seed_xy_m[1]:.2f})")

    clear_scene()

    for i, obj_path in enumerate(objs):
        day, label = parse_stem(obj_path.stem)
        x = i * ROW_SPACING_M
        name = f"day{day:03d}_{label}"
        print(f"  [{i:2d}] x={x:5.2f}m  {obj_path.name}")

        plant = import_and_join(obj_path, name, seed_xy_m)
        if plant is None:
            print(f"    SKIP: import returned no objects")
            continue
        plant.location.x = x
        # Origin already at stem base (mesh-local 0,0,0); object.location
        # places origin at world (x, 0, 0), so stem base is at world (x, 0, 0).

        add_label(f"day{day:03d}\n{label}", x, f"{name}_label")

    row_len = max(len(objs) * ROW_SPACING_M, 1.0)
    add_ground(row_len)
    add_lighting()
    add_camera(row_len)

    bpy.context.scene.unit_settings.system = "METRIC"
    bpy.context.scene.unit_settings.scale_length = 1.0

    bpy.ops.wm.save_as_mainfile(filepath=str(BLEND_PATH))
    print(f"\nSaved: {BLEND_PATH}")
    print(f"Open with: blender {BLEND_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
