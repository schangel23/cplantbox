"""Blender script: assemble a seed-variation panel from N OBJs side-by-side.

Run as:
    blender --background --python dart/coupling/scripts/blender_seed_variation_panel.py -- \\
        --obj-dir dart/coupling/output/seed_variation_day130 \\
        --rows 4 --cols 5 --row-spacing 1.4 --col-spacing 1.0 \\
        --save-blend dart/coupling/output/seed_variation_day130/panel.blend

How it works:
    1. All ``seed_*.obj`` files in --obj-dir are imported and placed on a
       rows × cols grid (m, Blender units).
    2. Each plant is centred at its grid cell (mesh XY centre subtracted).
    3. A text label ``seed N | <V> | <H> cm`` is added above each plant
       (read from stats.csv if present; otherwise just the seed number).
    4. The .blend is saved so the user can open it interactively.

The script does NOT render — it just assembles the scene. Open in Blender
GUI to inspect.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import bpy

CM_TO_M = 0.01


def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--obj-dir", required=True,
                   help="Directory with seed_*.obj files (and optional stats.csv).")
    p.add_argument("--rows", type=int, default=4)
    p.add_argument("--cols", type=int, default=5)
    p.add_argument("--row-spacing", type=float, default=1.4,
                   help="Between-row spacing (m). ≥ 1.4 m fits a ~2.4 m maize plant.")
    p.add_argument("--col-spacing", type=float, default=1.0,
                   help="Within-row column spacing (m).")
    p.add_argument("--save-blend", required=True,
                   help="Output .blend path.")
    p.add_argument("--label-z", type=float, default=2.6,
                   help="Z (m) at which to place per-plant text labels.")
    return p.parse_args(argv)


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for coll in list(bpy.data.collections):
        bpy.data.collections.remove(coll)
    for mesh in list(bpy.data.meshes):
        bpy.data.meshes.remove(mesh)
    for mat in list(bpy.data.materials):
        bpy.data.materials.remove(mat)


def load_stats(stats_csv: Path) -> dict[int, dict]:
    """seed -> {label, ms_len_cm, ...}"""
    out: dict[int, dict] = {}
    if not stats_csv.exists():
        return out
    with stats_csv.open() as fh:
        for row in csv.DictReader(fh):
            try:
                out[int(row["seed"])] = row
            except (ValueError, KeyError):
                continue
    return out


def import_obj(obj_path: Path) -> list[bpy.types.Object]:
    before = set(bpy.data.objects)
    bpy.ops.wm.obj_import(filepath=str(obj_path), forward_axis='Y', up_axis='Z')
    return [o for o in bpy.data.objects if o not in before]


def place_at_cell(objs: list[bpy.types.Object], cx: float, cy: float) -> None:
    """Translate so the mesh XY centre lands at (cx, cy) m, scale cm→m."""
    if not objs:
        return
    # Find combined XY centre using bound boxes (in object-local cm).
    xs, ys = [], []
    for o in objs:
        for v in o.bound_box:
            xs.append(v[0])
            ys.append(v[1])
    mid_x = (min(xs) + max(xs)) / 2.0
    mid_y = (min(ys) + max(ys)) / 2.0
    for o in objs:
        # The OBJ is in cm; scale → m, then shift mesh centre to (cx, cy).
        o.scale = (CM_TO_M, CM_TO_M, CM_TO_M)
        o.location = (cx - mid_x * CM_TO_M, cy - mid_y * CM_TO_M, 0.0)


def make_label(text: str, x: float, y: float, z: float,
               name: str) -> bpy.types.Object:
    bpy.ops.object.text_add(location=(x, y, z))
    obj = bpy.context.object
    obj.name = name
    obj.data.body = text
    obj.data.size = 0.10
    obj.rotation_euler = (1.5708, 0.0, 0.0)  # face +Y (camera-friendly)
    return obj


def build_panel(args: argparse.Namespace) -> None:
    obj_dir = Path(args.obj_dir)
    stats = load_stats(obj_dir / "stats.csv")
    obj_paths = sorted(obj_dir.glob("seed_*.obj"))
    # Drop DART-routed copies (they're duplicates of the same geometry).
    obj_paths = [p for p in obj_paths if not p.stem.endswith("_dart")]
    if not obj_paths:
        print(f"ERROR: no seed_*.obj in {obj_dir}")
        sys.exit(1)

    print(f"Importing {len(obj_paths)} plants into "
          f"{args.rows}×{args.cols} grid")

    cell_x_offset = -(args.cols - 1) / 2.0 * args.col_spacing
    cell_y_offset = -(args.rows - 1) / 2.0 * args.row_spacing

    for i, obj_path in enumerate(obj_paths):
        if i >= args.rows * args.cols:
            print(f"WARNING: skipping {obj_path.name} — grid full")
            break
        col = i % args.cols
        row = i // args.cols
        cx = cell_x_offset + col * args.col_spacing
        cy = cell_y_offset + row * args.row_spacing

        # seed number from filename "seed_007_VT_emerging.obj"
        try:
            seed = int(obj_path.stem.split("_")[1])
        except (IndexError, ValueError):
            seed = i + 1

        objs = import_obj(obj_path)
        place_at_cell(objs, cx, cy)

        # Label
        st = stats.get(seed, {})
        if st:
            label_text = (
                f"seed {seed} | {st.get('label', '?')} | "
                f"{float(st.get('ms_len_cm', 0)):.0f} cm"
            )
        else:
            label_text = f"seed {seed}"
        make_label(label_text, cx, cy, args.label_z, name=f"label_{seed:03d}")

        print(f"  [{i+1}/{len(obj_paths)}] seed={seed} cell=({col},{row}) "
              f"-> {label_text}")

    # Camera + light so the .blend opens with a sensible default view.
    bpy.ops.object.camera_add(
        location=(0.0, -8.0, 2.5),
        rotation=(1.2, 0.0, 0.0),
    )
    bpy.context.scene.camera = bpy.context.object
    bpy.ops.object.light_add(type='SUN', location=(2.0, -4.0, 6.0))
    bpy.context.object.data.energy = 4.0

    # Save.
    save_path = Path(args.save_blend)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(save_path))
    print(f"\nSaved panel to {save_path}")


def main() -> None:
    args = parse_args()
    clear_scene()
    build_panel(args)


if __name__ == "__main__":
    main()
