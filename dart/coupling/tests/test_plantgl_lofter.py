"""Proof-of-concept: mesh maize organs via PlantGL Extrusion vs numpy lofter.

Takes the same skeleton + widths from a phytomer-mode day-55 plant and lofts
a representative blade, sheath, and stem slice both ways. Writes two OBJs
so the user can eyeball them side-by-side and reports triangle counts +
file sizes.

Run:
    LD_LIBRARY_PATH=$CPBENV/lib/python3.14/site-packages/lib \
    QT_QPA_PLATFORM=offscreen \
    $CPBENV/bin/python3 -m dart.coupling.tests.test_plantgl_lofter
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

# PlantGL needs these before import
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# Explicitly avoid plantgl.all (pulls in gui/_pglgui which isn't built here)
from openalea.plantgl.math import Vector3, Vector2
from openalea.plantgl.scenegraph import (
    Extrusion,
    Polyline,
    Polyline2D,
    Point3Array,
    Point2Array,
)
from openalea.plantgl.algo import Discretizer

# CPlantBox pipeline
import plantbox as pb
from dart.coupling.growth.grow import grow_plant
from dart.coupling.geometry.cplantbox_adapter import extract_organs_for_lofter
from dart.coupling.geometry.g1_to_g3 import loft_organs


OUT_DIR = Path("/home/lukas/PHD/CPlantBox/dart/coupling/output/plantgl_test")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# PlantGL loft kernels
# ---------------------------------------------------------------------------
def _quadset_to_triangles(qs) -> tuple[np.ndarray, np.ndarray]:
    """Convert PlantGL QuadSet -> (verts[N,3], tris[M,3])."""
    verts = np.array([[v.x, v.y, v.z] for v in qs.pointList], dtype=np.float64)
    tris = []
    for q in qs.indexList:
        # QuadSet quad = (i0, i1, i2, i3). Split into two triangles.
        i0, i1, i2, i3 = int(q[0]), int(q[1]), int(q[2]), int(q[3])
        tris.append([i0, i1, i2])
        tris.append([i0, i2, i3])
    return verts, np.array(tris, dtype=np.int64)


def _extrude(axis_pts: np.ndarray, cross_xy: np.ndarray,
             scale_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Run PlantGL Extrusion on the given axis/cross/scale arrays.

    axis_pts:  (N, 3) — centerline in local coords (first point at origin)
    cross_xy:  (M, 2) — closed cross-section polyline in the local x-y plane
    scale_xy:  (N, 2) — per-axis scale factors (sx, sy) applied to cross

    Returns verts (K, 3), tris (T, 3).
    """
    axis = Polyline(Point3Array([Vector3(float(p[0]), float(p[1]), float(p[2]))
                                 for p in axis_pts]))
    cross = Polyline2D(Point2Array([Vector2(float(c[0]), float(c[1]))
                                    for c in cross_xy]))
    scales = Point2Array([Vector2(float(s[0]), float(s[1])) for s in scale_xy])

    ext = Extrusion(axis, cross, scales)
    d = Discretizer()
    ext.apply(d)
    qs = d.result
    if qs is None:
        raise RuntimeError("Extrusion discretization returned None")
    return _quadset_to_triangles(qs)


def plantgl_mesh_blade(skeleton: np.ndarray, widths: np.ndarray,
                       thickness_ratio: float = 0.025) -> tuple[np.ndarray, np.ndarray]:
    """Mesh a blade as a thin rectangular ribbon swept along the skeleton.

    Cross-section = 4-point rectangle centred at origin (x = half-width,
    y = half-thickness). Width taper is handled via scale_xy.
    """
    # Base cross: unit rectangle (±0.5 in x, ±0.5 * thickness_ratio in y)
    t = thickness_ratio
    cross = np.array([
        [-0.5, -0.5 * t],
        [ 0.5, -0.5 * t],
        [ 0.5,  0.5 * t],
        [-0.5,  0.5 * t],
        [-0.5, -0.5 * t],  # close
    ])
    # Per-node scale: x = full blade width, y = 1 (thickness is baked into cross)
    scale_xy = np.column_stack([widths, np.ones_like(widths)])

    # Translate skeleton so first point is origin (PlantGL convention)
    axis = skeleton - skeleton[0]
    return _extrude(axis, cross, scale_xy)


def plantgl_mesh_sheath(skeleton: np.ndarray, radii: np.ndarray,
                        n_arc: int = 24, wrap_deg: float = 330.0
                        ) -> tuple[np.ndarray, np.ndarray]:
    """Mesh a sheath as a partial-arc cylinder swept along the skeleton."""
    # Cross: 330° arc, unit radius, then back to start (open arc — we don't close)
    # Keep it open so PlantGL produces a scroll-like surface (matches sheath_mesher.py)
    theta = np.deg2rad(np.linspace(0, wrap_deg, n_arc))
    cross = np.column_stack([np.cos(theta), np.sin(theta)])
    # Scale per-node: radius uniform in x and y
    scale_xy = np.column_stack([radii, radii])
    axis = skeleton - skeleton[0]
    return _extrude(axis, cross, scale_xy)


def plantgl_mesh_stem(skeleton: np.ndarray, radii: np.ndarray,
                      n_ring: int = 16) -> tuple[np.ndarray, np.ndarray]:
    """Mesh a stem as a full closed cylinder swept along the skeleton."""
    theta = np.linspace(0, 2 * np.pi, n_ring + 1)  # closed loop
    cross = np.column_stack([np.cos(theta), np.sin(theta)])
    scale_xy = np.column_stack([radii, radii])
    axis = skeleton - skeleton[0]
    return _extrude(axis, cross, scale_xy)


# ---------------------------------------------------------------------------
# OBJ writer
# ---------------------------------------------------------------------------
def write_obj(path: Path, verts: np.ndarray, tris: np.ndarray,
              offset: np.ndarray = None, material: str = "blade") -> None:
    """Write a simple OBJ with one material group. tris is 0-indexed."""
    if offset is not None:
        verts = verts + offset
    with open(path, "w") as f:
        f.write(f"# PlantGL-lofted mesh\n")
        f.write(f"usemtl {material}\n")
        for v in verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for t in tris:
            f.write(f"f {t[0] + 1} {t[1] + 1} {t[2] + 1}\n")


def write_obj_multi(path: Path, parts: list[dict]) -> None:
    """Write an OBJ combining multiple meshes with per-part usemtl groups.

    parts: [{"name": str, "material": str, "verts": ndarray, "tris": ndarray}]
    """
    with open(path, "w") as f:
        f.write(f"# PlantGL-lofted plant\n")
        vert_offset = 0
        for part in parts:
            v = part["verts"]
            t = part["tris"]
            f.write(f"g {part['name']}\n")
            f.write(f"usemtl {part['material']}\n")
            for p in v:
                f.write(f"v {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
            for tri in t:
                f.write(f"f {tri[0] + vert_offset + 1} "
                        f"{tri[1] + vert_offset + 1} "
                        f"{tri[2] + vert_offset + 1}\n")
            vert_offset += len(v)


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------
def main():
    print("=== Growing plant (phytomer mode, day 55) ===")
    xml_path = "dart/coupling/data/maize_phytomer.xml"
    plant = grow_plant(xml_path, simulation_time=55, seed=42)

    print("\n=== Extracting organs ===")
    organs = extract_organs_for_lofter(plant, min_stem_nodes=50, min_leaf_nodes=20)
    print(f"  Extracted {len(organs)} organs")

    # Pick representatives: the stem, a mid-plant sheath, a mid-plant blade
    stem_organ = next(o for o in organs if o["type"] == "stem")
    sheaths = [o for o in organs if o.get("part_type") == "sheath"]
    blades = [o for o in organs if o.get("part_type") == "blade"]
    print(f"  Stems: 1, Sheaths: {len(sheaths)}, Blades: {len(blades)}")

    if not sheaths or not blades:
        print("!! no sheath/blade found — is phytomer mode active?")
        return 1

    # ------ PlantGL: FULL PLANT (all organs) ------
    print("\n=== PlantGL Extrusion: full plant ===")
    parts = []
    total_tris = 0

    # Stem
    stem_radii = stem_organ["widths"] / 2.0
    mv, mt = plantgl_mesh_stem(stem_organ["skeleton"], stem_radii)
    parts.append({"name": "stem_0", "material": "stem",
                  "verts": mv + stem_organ["skeleton"][0], "tris": mt})
    total_tris += len(mt)
    print(f"  Stem:   {len(mv):>6} verts, {len(mt):>6} tris")

    # All sheaths
    sheath_tris = 0
    for sh in sheaths:
        radii = sh.get("radii")
        if radii is None:
            radii = sh["widths"] / 2.0
        radii = np.asarray(radii, dtype=np.float64)
        sv, st = plantgl_mesh_sheath(sh["skeleton"], radii)
        parts.append({"name": sh["name"], "material": "sheath",
                      "verts": sv + sh["skeleton"][0], "tris": st})
        sheath_tris += len(st)
    total_tris += sheath_tris
    print(f"  Sheaths: {len(sheaths)} organs, {sheath_tris:>6} tris total")

    # All blades
    blade_tris = 0
    for bl in blades:
        bv, bt = plantgl_mesh_blade(bl["skeleton"], bl["widths"])
        parts.append({"name": bl["name"], "material": "blade",
                      "verts": bv + bl["skeleton"][0], "tris": bt})
        blade_tris += len(bt)
    total_tris += blade_tris
    print(f"  Blades:  {len(blades)} organs, {blade_tris:>6} tris total")

    plantgl_out = OUT_DIR / "full_plant_plantgl.obj"
    write_obj_multi(plantgl_out, parts)
    print(f"  -> {plantgl_out}  ({plantgl_out.stat().st_size / 1024:.1f} KB)")
    print(f"  TOTAL: {total_tris} tris")

    # ------ Numpy lofter (existing pipeline): FULL PLANT ------
    print("\n=== Numpy lofter: full plant ===")
    full_organs = [stem_organ] + sheaths + blades
    mesh_np = loft_organs(full_organs)
    n_verts_np = len(mesh_np.vertices)
    n_tris_np = len(mesh_np.indices)
    print(f"  Combined: {n_verts_np} verts, {n_tris_np} tris")

    numpy_out = OUT_DIR / "full_plant_numpy.obj"
    mesh_np.to_obj(str(numpy_out), write_materials=True)
    print(f"  -> {numpy_out}  ({numpy_out.stat().st_size / 1024:.1f} KB)")

    # ------ Summary ------
    print("\n=== Summary ===")
    plantgl_tri = total_tris
    numpy_tri = n_tris_np
    print(f"  PlantGL: {plantgl_tri:>6} triangles, "
          f"{plantgl_out.stat().st_size / 1024:>7.1f} KB")
    print(f"  Numpy:   {numpy_tri:>6} triangles, "
          f"{numpy_out.stat().st_size / 1024:>7.1f} KB")
    if numpy_tri:
        print(f"  Ratio:   PlantGL/Numpy = {plantgl_tri / numpy_tri:.2f}x tris, "
              f"{plantgl_out.stat().st_size / numpy_out.stat().st_size:.2f}x size")
    return 0


if __name__ == "__main__":
    sys.exit(main())
