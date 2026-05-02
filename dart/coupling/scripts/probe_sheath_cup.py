#!/usr/bin/env python3
"""Probe maize compound sheath cup/stem contact geometry."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dart.coupling.growth.grow import enable_fa_on_mainstem, grow_plant
from dart.coupling.geometry.cplantbox_adapter import extract_organs_for_lofter
from dart.coupling.geometry.g1_to_g3 import loft_organs


XML = ROOT / "dart" / "coupling" / "data" / "maize_calibrated.xml"


def _unit(v):
    v = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v


def _stem_vertices_for_organ(mesh, stem_id):
    tri_mask = mesh.organ_ids == int(stem_id)
    if not np.any(tri_mask):
        return np.empty((0, 3), dtype=np.float64)
    vidx = np.unique(mesh.indices[tri_mask].ravel())
    return mesh.vertices[vidx]


def _actual_stem_radius_at_z(stem_vertices, stem_skeleton, world_z):
    if len(stem_vertices) == 0:
        return None
    stem_skeleton = np.asarray(stem_skeleton, dtype=np.float64)
    target = float(world_z)
    dz = np.abs(stem_vertices[:, 2] - target)
    order = np.argsort(dz)
    keep = order[dz[order] <= 0.08]
    if len(keep) < 8:
        keep = order[: min(16, len(order))]
    pts = stem_vertices[keep]
    skel_z = stem_skeleton[:, 2]
    center = np.array([
        np.interp(target, skel_z, stem_skeleton[:, 0]),
        np.interp(target, skel_z, stem_skeleton[:, 1]),
        target,
    ])
    radii = np.linalg.norm(pts[:, :2] - center[None, :2], axis=1)
    radii = radii[radii > 1e-4]
    if len(radii) == 0:
        return None
    return {
        "n": int(len(radii)),
        "z_mean": float(np.mean(pts[:, 2])),
        "z_abs_err_max": float(np.max(np.abs(pts[:, 2] - target))),
        "r_min": float(np.min(radii)),
        "r_mean": float(np.mean(radii)),
        "r_max": float(np.max(radii)),
    }


def _organ_stem_id(organs):
    for organ in organs:
        if organ.get("type") == "stem":
            return int(organ["organ_id"])
    return None


def _stem_organ(organs):
    for organ in organs:
        if organ.get("type") == "stem":
            return organ
    return None


def _cup_bottom_radii(cps_world, collar_pos, stem_axis):
    stem_axis = _unit(stem_axis)
    collar_pos = np.asarray(collar_pos, dtype=np.float64)
    row = np.asarray(cps_world[0], dtype=np.float64)
    rel = row - collar_pos[None, :]
    axial = rel @ stem_axis
    radial_vec = rel - axial[:, None] * stem_axis[None, :]
    return axial, np.linalg.norm(radial_vec, axis=1)


def main():
    plant = grow_plant(
        str(XML),
        simulation_time=100,
        seed=42,
        enable_photosynthesis=False,
        mutate_lrp_pre_init=lambda p: enable_fa_on_mainstem(p),
    )
    organs = extract_organs_for_lofter(plant, species="maize")
    mesh = loft_organs(organs, use_nurbs_backend=True)

    stem_organ = _stem_organ(organs)
    stem_id = int(stem_organ["organ_id"]) if stem_organ is not None else None
    stem_skeleton = (
        np.asarray(stem_organ["skeleton"], dtype=np.float64)
        if stem_organ is not None
        else np.empty((0, 3))
    )
    stem_vertices = (
        _stem_vertices_for_organ(mesh, stem_id)
        if stem_id is not None
        else np.empty((0, 3))
    )

    candidates = [
        o for o in organs
        if o.get("type") == "leaf"
        and 5 <= int(o.get("organ_id", -1)) <= 10
        and float(o.get("sheath_length_cm") or 0.0) > 0.0
    ]

    print(f"organs={len(organs)} mesh_vertices={len(mesh.vertices)} stem_id={stem_id}")
    print(f"candidate_leaf_ids={[int(o['organ_id']) for o in candidates]}")

    for organ in candidates[:3]:
        oid = int(organ["organ_id"])
        cps = mesh.organ_cps.get(oid)
        collar_pos = np.asarray(organ["collar_pos"], dtype=np.float64)
        stem_axis = _unit(np.asarray(organ.get("parent_tangent", (0.0, 0.0, 1.0)), dtype=np.float64))
        stem_r_callable = organ.get("parent_stem_radius_at_z_cm")
        if cps is None:
            print(f"\nrank={oid} organ_id={oid}: no NURBS cps")
            continue

        axial, cup_r = _cup_bottom_radii(cps, collar_pos, stem_axis)
        z_local = float(np.mean(axial))
        callable_r = float(stem_r_callable(z_local)) if callable(stem_r_callable) else math.nan
        world_z = float(collar_pos[2] + stem_axis[2] * z_local)
        actual = _actual_stem_radius_at_z(stem_vertices, stem_skeleton, world_z)
        actual_txt = "None"
        if actual is not None:
            actual_txt = (
                f"n={actual['n']} z_mean={actual['z_mean']:.4f} "
                f"z_err_max={actual['z_abs_err_max']:.4f} "
                f"r_min/mean/max={actual['r_min']:.4f}/"
                f"{actual['r_mean']:.4f}/{actual['r_max']:.4f}"
            )

        print(f"\nrank={oid} organ_id={oid}")
        print(f"  sheath_length_cm={float(organ.get('sheath_length_cm') or 0.0):.4f}")
        print(f"  stem_radius_cm={float(organ.get('stem_radius_cm') or 0.0):.4f}")
        print(f"  parent_stem_radius_at_z_cm={callable_r:.4f} at z_local={z_local:.4f}")
        print(f"  cup_bottom_world_z={world_z:.4f}")
        print(f"  sheath_provenance={organ.get('sheath_provenance')}")
        print(
            "  cup_bottom_z_local min/mean/max="
            f"{float(np.min(axial)):.4f}/{z_local:.4f}/{float(np.max(axial)):.4f}"
        )
        print(
            "  cup_bottom_cp_radii min/mean/max="
            f"{float(np.min(cup_r)):.4f}/{float(np.mean(cup_r)):.4f}/{float(np.max(cup_r)):.4f}"
        )
        print(f"  actual_stem_vertex_radii_at_z {actual_txt}")
        if actual is not None and callable_r > 0:
            print(f"  actual_mean/callable={actual['r_mean'] / callable_r:.4f}")


if __name__ == "__main__":
    main()
