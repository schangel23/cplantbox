"""Preview test: young-stage midrib straightening.

For each young day, export two OBJs:
  * ``young_day{D}_current.obj``     — current behaviour (uniform scale of mature CPs)
  * ``young_day{D}_straightened.obj`` — proposed morph applied before lofting

The morph is applied in Python only, purely by mutating ``surface_cps_local``
in the organ dicts between ``extract_organs_for_lofter`` and ``loft_organs``.
No library code is modified.

Run:
    cd /home/lukas/PHD/CPlantBox
    source cpbenv/bin/activate
    python dart/coupling/output/blender_preview/_gen_young_straighten_test.py
"""
from __future__ import annotations
from pathlib import Path
import sys

import numpy as np
import plantbox as pb

from dart.coupling.config import DEFAULT_XML
from dart.coupling.growth.grow import grow_plant
from dart.coupling.geometry import extract_organs_for_lofter, loft_organs


SIM_DAYS = [10, 15, 20, 30, 55]   # 55 = mature reference
SKEL_SEED = 7
DONOR_SEED = 4                     # maps to MF3D plant 446 under draw_coherent
FADE_END = 0.7                     # maturity at which leaf reaches full MF3D droop
STRAIGHTEN_EXPONENT = 1.0          # 1 = linear; >1 = stays straight longer

OUT = Path(__file__).resolve().parent


def _straighten_midrib(cps_local: np.ndarray,
                       current_length: float,
                       mature_length: float,
                       *,
                       fade_end: float = FADE_END,
                       exponent: float = STRAIGHTEN_EXPONENT) -> np.ndarray:
    """Blend a leaf's CP grid toward an erect midrib based on maturity.

    The midrib is the v = n_v//2 column. For each u-row, compute the target
    position on a straight +z midrib with matching arc-length, then shift the
    entire u-row (all v) by ``alpha_young * (target - midrib_original)``.
    This preserves blade width, twist, and cross-row offsets around the
    midrib; only the midrib path is pulled toward straight-up.

    ``cps_local`` frame (matches ``Leaf.cpp::updateNodesFromSurfaceCPs``):
      * +z = insertion tangent (grow-away direction)
      * x/y = lateral / across-midrib

    Args:
        cps_local: (N_U, N_V, 3) CP grid in leaf-local frame (normalised or
            absolute cm — morph is shape-preserving under later scaling).
        current_length: leaf's current midrib arc-length (cm).
        mature_length: leaf's mature lmax (cm).
        fade_end: maturity at which the morph fully relaxes (alpha=0).
        exponent: shape of the straightening decay; 1 = linear, >1 keeps
            young leaves straight for longer.

    Returns:
        New (N_U, N_V, 3) array with midrib blended toward straight +z.
    """
    if mature_length <= 1e-9:
        return cps_local
    maturity = max(0.0, min(current_length / mature_length, 1.0))
    if maturity >= fade_end:
        return cps_local
    alpha = (1.0 - maturity / fade_end) ** exponent

    cps = np.asarray(cps_local, dtype=np.float64).copy()
    v_mid = cps.shape[1] // 2

    midrib = cps[:, v_mid, :].copy()                     # (N_U, 3)
    seg = np.linalg.norm(np.diff(midrib, axis=0), axis=1)
    if seg.sum() < 1e-12:
        return cps_local
    s = np.concatenate([[0.0], np.cumsum(seg)])          # (N_U,), absolute

    # Straight target: same arc-length, along +z in the leaf-local frame.
    target = np.zeros_like(midrib)
    target[:, 2] = s

    delta = target - midrib                              # (N_U, 3) per u-row shift
    cps += alpha * delta[:, None, :]                     # broadcast across N_V
    return cps


def _apply_morph_to_organs(organ_dicts: list[dict]) -> int:
    """Mutate leaf-blade organs in place; return count modified."""
    n_mod = 0
    for organ in organ_dicts:
        cps = organ.get("surface_cps_local")
        if cps is None:
            continue
        curr = float(organ.get("current_length", 0.0))
        mat = float(organ.get("mature_length", 0.0))
        organ["surface_cps_local"] = _straighten_midrib(cps, curr, mat)
        n_mod += 1
    return n_mod


def _recenter(mesh, plant) -> tuple[float, float]:
    stems = plant.getOrgans(pb.stem)
    if not stems:
        return (0.0, 0.0)
    seed_node = stems[0].getNodes()[0]
    sx, sy = float(seed_node.x), float(seed_node.y)
    mesh.vertices[:, 0] -= sx
    mesh.vertices[:, 1] -= sy
    return sx, sy


def _loft_and_write(plant, organ_dicts, out_path: Path, label: str) -> None:
    mesh = loft_organs(
        organ_dicts,
        stem_sides=16,
        use_nurbs_backend=True,
        nurbs_n_u_eval=30,
        nurbs_n_v_eval=7,
    )
    _recenter(mesh, plant)
    mesh.to_obj(str(out_path))
    print(f"  [{label}] wrote {out_path.name} "
          f"({mesh.n_vertices} v, {mesh.n_triangles} t)")


def main() -> None:
    for day in SIM_DAYS:
        print(f"\n=== day {day} (donor_seed={DONOR_SEED}, "
              f"fade_end={FADE_END}, exp={STRAIGHTEN_EXPONENT}) ===")
        plant = grow_plant(
            str(DEFAULT_XML),
            simulation_time=day,
            seed=SKEL_SEED,
            cp_donor_seed=DONOR_SEED,
            cp_donor_mode="draw_coherent",
        )

        # Current behaviour
        organs_current = extract_organs_for_lofter(plant, skip_roots=True)
        _loft_and_write(
            plant, organs_current,
            OUT / f"young_day{day:02d}_current.obj", "current",
        )

        # Proposed straightening
        organs_straight = extract_organs_for_lofter(plant, skip_roots=True)
        n_mod = _apply_morph_to_organs(organs_straight)
        print(f"  straightened {n_mod} blades")
        _loft_and_write(
            plant, organs_straight,
            OUT / f"young_day{day:02d}_straightened.obj", "straight",
        )


if __name__ == "__main__":
    sys.exit(main() or 0)
