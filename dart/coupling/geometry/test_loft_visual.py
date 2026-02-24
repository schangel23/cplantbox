#!/usr/bin/env python3
"""Generate test OBJ meshes for visual inspection of tip geometry.

Run from CPlantBox root:
    source cpbenv/bin/activate
    python3 dart/coupling/geometry/test_loft_visual.py

Produces:
    dart/coupling/output/test_loft_day10.obj
    dart/coupling/output/test_loft_day55.obj
"""

import sys
import os
import numpy as np
from pathlib import Path

# Add CPlantBox to path
cpb_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(cpb_root))

import plantbox as pb
from dart.coupling.geometry.cplantbox_adapter import extract_organs_for_lofter
from dart.coupling.geometry.g1_to_g3 import loft_organs

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

XML = cpb_root / "dart" / "coupling" / "data" / "maize_calibrated.xml"


def grow_and_loft(day, seed=42):
    """Grow a plant and loft it to G3 mesh."""
    plant = pb.MappedPlant()
    plant.readParameters(str(XML))
    plant.setSeed(seed)
    plant.initialize(verbose=False)
    plant.simulate(day, verbose=False)

    organs = extract_organs_for_lofter(plant)
    mesh = loft_organs(organs, target_spacing=0.5)
    return mesh, organs


def report(mesh, label):
    """Print mesh statistics."""
    verts = mesh.vertices
    v0 = verts[mesh.indices[:, 0]]
    v1 = verts[mesh.indices[:, 1]]
    v2 = verts[mesh.indices[:, 2]]
    areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)

    print(f"\n{'=' * 60}")
    print(f"{label}")
    print(f"{'=' * 60}")
    print(f"  Vertices:  {mesh.n_vertices}")
    print(f"  Triangles: {mesh.n_triangles}")
    print(f"  Organs:    {len(mesh.organ_meta)}")

    if len(areas) > 0:
        print(f"\n  Triangle areas (cm²):")
        print(f"    Min:    {areas.min():.6f}")
        print(f"    Max:    {areas.max():.6f}")
        print(f"    Mean:   {areas.mean():.6f}")
        print(f"    Median: {np.median(areas):.6f}")
        for threshold in [0.001, 0.01, 0.05]:
            n_below = np.sum(areas < threshold)
            print(f"    < {threshold} cm²: {n_below} ({100*n_below/len(areas):.1f}%)")


if __name__ == "__main__":
    for day in [10, 55]:
        print(f"\nGrowing day {day} plant...")
        mesh, organs = grow_and_loft(day)

        n_leaves = sum(1 for o in organs if o['type'] == 'leaf')
        print(f"  Leaves: {n_leaves}")

        report(mesh, f"Day {day} G3 Mesh")

        obj_path = OUTPUT_DIR / f"test_loft_day{day}.obj"
        mesh.to_obj(obj_path)
        print(f"  Exported: {obj_path}")
