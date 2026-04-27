"""Generate a small 'canopy' via runtime CP-swap.

Same CPlantBox skeleton seed for all plants — only the MF3D donor varies.
This isolates leaf-shape variation so you can eyeball whether drawing
different donors per plant gives visibly different canopy appearance
without any XML regeneration.
"""
from __future__ import annotations
from pathlib import Path
import sys

import plantbox as pb

from dart.coupling.config import DEFAULT_XML
from dart.coupling.growth.grow import grow_plant, extract_g3_mesh


SIM_DAY = 55
SKEL_SEED = 7
DONOR_SEEDS = [1, 2, 4, 11, 21]  # map 1:1 to pool {430, 455, 446, 216, 374}
OUT = Path(__file__).resolve().parent


def _recenter_to_stem_base(mesh, plant) -> tuple[float, float]:
    """Translate mesh so seed x/y = (0,0); leave z (ground stays at z=0).

    Blender convenience: opening the OBJ drops the plant at the origin
    instead of its original (200, 200, …) world coords.
    """
    stems = plant.getOrgans(pb.stem)
    if not stems:
        return (0.0, 0.0)
    seed_node = stems[0].getNodes()[0]
    sx, sy = float(seed_node.x), float(seed_node.y)
    mesh.vertices[:, 0] -= sx
    mesh.vertices[:, 1] -= sy
    return sx, sy


def main() -> None:
    for donor in DONOR_SEEDS:
        out_obj = OUT / f"runtime_donor{donor}.obj"
        print(f"\n--- donor_seed={donor} -> {out_obj.name} ---")
        plant = grow_plant(
            str(DEFAULT_XML),
            simulation_time=SIM_DAY,
            seed=SKEL_SEED,
            cp_donor_seed=donor,
            cp_donor_mode="draw_coherent",
        )
        mesh, _ = extract_g3_mesh(
            plant,
            use_nurbs_leaf_backend=True,
            nurbs_leaf_n_u_eval=30,
            nurbs_leaf_n_v_eval=7,
        )
        sx, sy = _recenter_to_stem_base(mesh, plant)
        mesh.to_obj(str(out_obj))
        print(f"  wrote {out_obj} "
              f"({mesh.n_vertices} verts, {mesh.n_triangles} tris); "
              f"recentered x-={sx:.1f} y-={sy:.1f}")


if __name__ == "__main__":
    sys.exit(main() or 0)
