"""Preview test: maturity-dependent leaf verticalisation + y-damp.

Young maize leaves emerge nearly vertical from the whorl and only splay
outward as they mature. Pure tangent rotation can't achieve this because
the MF3D CPs encode mature droop in the leaf-local y axis, so a scaled-
down young leaf still has a y-excursion tip even with a vertical tangent.

This preview combines TWO render-time morphs driven by the same alpha:
  1) ``collar_tangent`` is slerped toward ``parent_tangent`` (rotates base)
  2) Leaf-local y component of ``surface_cps_local`` is scaled by
     ``(1 - alpha)`` — flattens the droop component in the leaf-local
     frame without touching blade width (local x) or arc-length (local z)

    maturity m = current_length / mature_length            (clamped [0,1])
    alpha     = max(0, 1 - (m / fade_end) ** exp)
    new_tan   = slerp(collar_tangent, parent_tangent, alpha)
    cps[..., 1] *= (1 - alpha)

Mature leaves (alpha=0) get baseline behaviour; very young (alpha≈1) get
a near-vertical base AND a near-flat blade silhouette.

Run:
    cd /home/lukas/PHD/CPlantBox
    source cpbenv/bin/activate
    python dart/coupling/output/blender_preview/_gen_young_theta_test.py
"""
from __future__ import annotations
from pathlib import Path
import sys

import numpy as np
import plantbox as pb

from dart.coupling.config import DEFAULT_XML
from dart.coupling.growth.grow import grow_plant
from dart.coupling.geometry import extract_organs_for_lofter, loft_organs


SIM_DAYS = [10, 15, 20, 30, 55]
SKEL_SEED = 7
DONOR_SEED = 4                                   # MF3D plant 446

# Profiles: (label, fade_end, exp). Lower exp + higher fade_end = more drastic.
PROFILES = [
    ("base", None, None),                        # no morph (reference)
    ("soft",  1.0, 2.0),                         # gentle: 1 - m^2
    ("hard",  1.0, 1.0),                         # linear: 1 - m  (stronger mid-maturity)
    ("max",   0.9, 0.7),                         # extreme: vertical up to ~m=0.7
]

OUT = Path(__file__).resolve().parent


def _slerp_tangent(t_curr: np.ndarray, t_par: np.ndarray, alpha: float) -> np.ndarray:
    """Spherical interpolation from ``t_curr`` to ``t_par`` by fraction ``alpha``.

    ``alpha=0`` → ``t_curr``; ``alpha=1`` → ``t_par``. Unit vectors assumed.
    """
    c = float(np.clip(np.dot(t_curr, t_par), -1.0, 1.0))
    omega = np.arccos(c)
    if omega < 1e-6:
        return t_curr
    sin_omega = np.sin(omega)
    w_curr = np.sin((1.0 - alpha) * omega) / sin_omega
    w_par = np.sin(alpha * omega) / sin_omega
    out = w_curr * t_curr + w_par * t_par
    n = float(np.linalg.norm(out))
    return out / n if n > 1e-12 else t_curr


def _verticalise_morph(organ_dicts: list[dict],
                       fade_end: float,
                       exp: float) -> int:
    """Apply tangent rotation + leaf-local y-damp, both weighted by alpha(m).

    Returns number of blades modified. Skips organs without the metadata
    needed for the morph (stems, non-blade leaves).
    """
    n_mod = 0
    for organ in organ_dicts:
        if organ.get("type") != "leaf":
            continue
        ct = organ.get("collar_tangent")
        pt = organ.get("parent_tangent")
        cps = organ.get("surface_cps_local")
        if ct is None or pt is None or cps is None:
            continue
        mature = float(organ.get("mature_length", 0.0))
        current = float(organ.get("current_length", 0.0))
        if mature <= 1e-9:
            continue
        m = max(0.0, min(current / mature, 1.0))
        alpha = max(0.0, 1.0 - (m / fade_end) ** exp)
        if alpha <= 0.0:
            continue

        # --- Tangent rotation ---
        t_curr = np.asarray(ct, dtype=np.float64)
        t_par = np.asarray(pt, dtype=np.float64)
        n_c = float(np.linalg.norm(t_curr))
        n_p = float(np.linalg.norm(t_par))
        if n_c >= 1e-9 and n_p >= 1e-9:
            t_curr /= n_c
            t_par /= n_p
            organ["collar_tangent"] = _slerp_tangent(t_curr, t_par, alpha)

        # --- Leaf-local y-damp ---
        # local +z = tangent (arc-length direction), +x = blade width,
        # +y = droop-forward. Scaling y by (1-alpha) flattens the droop
        # component. Width and arc preserved.
        cps_mod = np.asarray(cps, dtype=np.float64).copy()
        cps_mod[..., 1] *= (1.0 - alpha)
        organ["surface_cps_local"] = cps_mod

        n_mod += 1
    return n_mod


def _recenter(mesh, plant) -> None:
    stems = plant.getOrgans(pb.stem)
    if not stems:
        return
    seed_node = stems[0].getNodes()[0]
    mesh.vertices[:, 0] -= float(seed_node.x)
    mesh.vertices[:, 1] -= float(seed_node.y)


def main() -> None:
    for day in SIM_DAYS:
        # Grow once per day; mutation happens on fresh organ_dicts per profile.
        print(f"\n=== day {day} (donor_seed={DONOR_SEED}) ===")
        plant = grow_plant(
            str(DEFAULT_XML),
            simulation_time=day,
            seed=SKEL_SEED,
            cp_donor_seed=DONOR_SEED,
            cp_donor_mode="draw_coherent",
        )

        for label, fade_end, exp in PROFILES:
            out_obj = OUT / f"young_day{day:02d}_{label}.obj"
            organs = extract_organs_for_lofter(plant, skip_roots=True)
            if fade_end is not None:
                n_mod = _verticalise_morph(organs, fade_end, exp)
                print(f"  [{label:>4}] morphed {n_mod} blades "
                      f"(fade_end={fade_end}, exp={exp})")
            else:
                print(f"  [{label:>4}] no morph (baseline)")

            mesh = loft_organs(
                organs, stem_sides=16, use_nurbs_backend=True,
                nurbs_n_u_eval=30, nurbs_n_v_eval=7,
            )
            _recenter(mesh, plant)
            mesh.to_obj(str(out_obj))
            print(f"         wrote {out_obj.name} "
                  f"({mesh.n_vertices} v, {mesh.n_triangles} t)")


if __name__ == "__main__":
    sys.exit(main() or 0)
