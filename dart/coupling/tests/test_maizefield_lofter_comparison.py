"""Compare lofter capabilities on REAL maize geometry.

Input: MaizeField3D plant 0317 (10 NURBS leaves, 3x6 control grid each).

For each leaf:
  - Ground truth: tessellate NURBS directly on a dense (n_u x n_v) grid.
  - PlantGL:     Extrude a flat rectangular ribbon along the NURBS midrib using
                 the measured width profile.
  - Numpy:       Feed the same midrib+widths to dart.coupling.geometry.g1_to_g3
                 loft_organs() as a single blade organ.

Outputs (under dart/coupling/output/maizefield_compare/):
  - 0317_ground_truth.obj  (NURBS direct tessellation)
  - 0317_plantgl.obj       (PlantGL Extrusion sweep)
  - 0317_numpy.obj         (numpy lofter, with default blade deformations)
  - 0317_numpy_clean.obj   (numpy lofter, deformations disabled — fair compare)

Reports per-leaf triangle counts, bbox spans, and nearest-point RMS deviation
from the NURBS ground truth.

Run:
    LD_LIBRARY_PATH=$CPBENV/lib/python3.14/site-packages/lib \
    QT_QPA_PLATFORM=offscreen \
    $CPBENV/bin/python3 -m dart.coupling.tests.test_maizefield_lofter_comparison
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# Make sure we can import the MaizeField3D readers
MF3D_DIR = Path("/home/lukas/PHD/Resources/MaizeField3d")
sys.path.insert(0, str(MF3D_DIR))

from extract_maizefield3d_morphology import parse_nurbs_dat  # noqa: E402
from maizefield3d_nurbs_reader import (  # noqa: E402
    leaf_dict_to_geomdl_surface,
    evaluate_midrib,
    evaluate_width_profile,
)

from geomdl import BSpline  # noqa: E402

# PlantGL (skip .all to avoid GUI pull-in)
from openalea.plantgl.math import Vector3, Vector2  # noqa: E402
from openalea.plantgl.scenegraph import (  # noqa: E402
    Extrusion, Polyline, Polyline2D, Point3Array, Point2Array,
)
from openalea.plantgl.algo import Discretizer  # noqa: E402

# Numpy lofter
from dart.coupling.geometry.g1_to_g3 import loft_organs  # noqa: E402
from dart.coupling.geometry.cplantbox_adapter import _leaf_wave_params  # noqa: E402


DAT_PATH = MF3D_DIR / "FielGrwon_ZeaMays_Reconstructed_Surface_dat" / "0317.dat"
OUT_DIR = Path("/home/lukas/PHD/CPlantBox/dart/coupling/output/maizefield_compare")
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_V = 60          # midrib samples along leaf (v direction)
N_U = 24          # cross-section samples across leaf (u direction) for GT
M_TO_CM = 100.0


# ---------------------------------------------------------------------------
# NURBS ground-truth tessellation
# ---------------------------------------------------------------------------
def tessellate_nurbs(surf: BSpline.Surface, n_u: int = N_U, n_v: int = N_V
                     ) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate the NURBS surface on a dense (n_u, n_v) grid and triangulate
    the resulting quad lattice. Returns (verts_cm, tris)."""
    deg_u, deg_v = surf.degree_u, surf.degree_v
    n_u_cp = len(surf.knotvector_u) - deg_u - 1
    n_v_cp = len(surf.knotvector_v) - deg_v - 1
    u_min, u_max = surf.knotvector_u[deg_u], surf.knotvector_u[n_u_cp]
    v_min, v_max = surf.knotvector_v[deg_v], surf.knotvector_v[n_v_cp]

    u_vals = np.linspace(u_min, u_max, n_u)
    v_vals = np.linspace(v_min, v_max, n_v)

    params = [[float(u), float(v)] for v in v_vals for u in u_vals]
    pts = np.array(surf.evaluate_list(params)) * M_TO_CM
    verts = pts  # (n_u*n_v, 3), row-major with u fastest

    tris = []
    for jv in range(n_v - 1):
        for ju in range(n_u - 1):
            i00 = jv * n_u + ju
            i10 = jv * n_u + ju + 1
            i01 = (jv + 1) * n_u + ju
            i11 = (jv + 1) * n_u + ju + 1
            tris.append([i00, i10, i11])
            tris.append([i00, i11, i01])
    return verts, np.array(tris, dtype=np.int64)


# ---------------------------------------------------------------------------
# PlantGL blade kernel (flat ribbon)
# ---------------------------------------------------------------------------
def _quadset_to_tris(qs) -> tuple[np.ndarray, np.ndarray]:
    verts = np.array([[v.x, v.y, v.z] for v in qs.pointList], dtype=np.float64)
    tris = []
    for q in qs.indexList:
        i0, i1, i2, i3 = int(q[0]), int(q[1]), int(q[2]), int(q[3])
        tris.append([i0, i1, i2])
        tris.append([i0, i2, i3])
    return verts, np.array(tris, dtype=np.int64)


def plantgl_blade(skeleton: np.ndarray, widths: np.ndarray,
                  thickness_ratio: float = 0.02
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Extrude a flat rectangular ribbon along the given skeleton. Width is
    applied per-node via scale_xy.x; thickness is fixed via cross coords."""
    t = thickness_ratio
    cross = np.array([
        [-0.5, -0.5 * t],
        [ 0.5, -0.5 * t],
        [ 0.5,  0.5 * t],
        [-0.5,  0.5 * t],
        [-0.5, -0.5 * t],
    ])
    scale_xy = np.column_stack([widths, np.ones_like(widths)])
    axis_local = skeleton - skeleton[0]  # PlantGL expects origin at first node

    axis_pl = Polyline(Point3Array(
        [Vector3(float(p[0]), float(p[1]), float(p[2])) for p in axis_local]))
    cross_pl = Polyline2D(Point2Array(
        [Vector2(float(c[0]), float(c[1])) for c in cross]))
    scales_pa = Point2Array(
        [Vector2(float(s[0]), float(s[1])) for s in scale_xy])

    ext = Extrusion(axis_pl, cross_pl, scales_pa)
    d = Discretizer()
    ext.apply(d)
    qs = d.result
    if qs is None:
        raise RuntimeError("PlantGL Extrusion returned None")
    verts_local, tris = _quadset_to_tris(qs)
    return verts_local + skeleton[0], tris


# ---------------------------------------------------------------------------
# OBJ writer (multi-group)
# ---------------------------------------------------------------------------
def write_obj_multi(path: Path, parts: list[dict], header: str = "") -> None:
    with open(path, "w") as f:
        if header:
            for ln in header.splitlines():
                f.write(f"# {ln}\n")
        vert_offset = 0
        for part in parts:
            v = np.asarray(part["verts"])
            t = np.asarray(part["tris"])
            f.write(f"g {part['name']}\n")
            f.write(f"usemtl {part.get('material', 'blade')}\n")
            for p in v:
                f.write(f"v {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
            for tri in t:
                f.write(f"f {tri[0] + vert_offset + 1} "
                        f"{tri[1] + vert_offset + 1} "
                        f"{tri[2] + vert_offset + 1}\n")
            vert_offset += len(v)


# ---------------------------------------------------------------------------
# Nearest-point RMS distance (query points → target surface vertices)
# ---------------------------------------------------------------------------
def rms_nearest(query: np.ndarray, target: np.ndarray) -> float:
    """Mean(min_i ||q - t_i||) RMS. Uses plain broadcasting — fine for
    a few thousand verts per leaf."""
    # chunk to keep memory bounded
    chunk = 512
    dists = np.empty(len(query), dtype=np.float64)
    for start in range(0, len(query), chunk):
        stop = min(start + chunk, len(query))
        diff = query[start:stop, None, :] - target[None, :, :]
        dists[start:stop] = np.linalg.norm(diff, axis=2).min(axis=1)
    return float(np.sqrt(np.mean(dists ** 2)))


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------
def _skel_widths_from_surf(surf: BSpline.Surface) -> tuple[np.ndarray, np.ndarray]:
    midrib_m = evaluate_midrib(surf, n_samples=N_V)
    widths_m = evaluate_width_profile(surf, n_samples=N_V)
    return midrib_m * M_TO_CM, widths_m * M_TO_CM


def _numpy_loft_blade(name: str, skeleton: np.ndarray, widths: np.ndarray,
                      organ_id: int, with_waves: bool,
                      position: int | None = None
                      ) -> tuple[np.ndarray, np.ndarray]:
    """Run the numpy lofter on a single-leaf organ dict.

    If with_waves=True, inject the production wave/curl/twist/ruffle params
    from cplantbox_adapter._leaf_wave_params (uses hand-tuned fallback since
    we have no deformation_stats here). If False, leave the dict bare — the
    lofter then produces a flat ribbon for apples-to-apples vs PlantGL.
    """
    # Translate to local coords so loft_organs operates near origin and we
    # don't accumulate float error in the smoother.
    origin = skeleton[0].copy()
    skel_local = skeleton - origin
    leaf_length = float(np.sum(np.linalg.norm(np.diff(skel_local, axis=0),
                                               axis=1)))

    organ = {
        "type": "leaf",
        "part_type": "blade",
        "skeleton": skel_local,
        "widths": widths.astype(np.float64),
        "organ_id": organ_id,
        "name": name,
        "node_ids": list(range(len(skel_local))),
    }
    if with_waves:
        # Use a per-leaf RNG so output is reproducible
        rng = np.random.RandomState(1000 + organ_id)
        wave_params = _leaf_wave_params(leaf_length, rng,
                                        position=position,
                                        deformation_stats=None,
                                        species='maize')
        organ.update(wave_params)

    mesh = loft_organs([organ], smooth=True, smooth_iterations=2,
                       subdivide=True, target_spacing=0.5)
    verts = np.asarray(mesh.vertices)
    tris = np.asarray(mesh.indices, dtype=np.int64)
    return verts + origin, tris


def main() -> int:
    if not DAT_PATH.is_file():
        print(f"!! MaizeField3D dat file not found: {DAT_PATH}")
        return 1

    print(f"=== Loading {DAT_PATH.name} ===")
    leaves_raw = parse_nurbs_dat(str(DAT_PATH))
    print(f"  {len(leaves_raw)} NURBS leaves")

    gt_parts, pg_parts, np_parts, np_clean_parts = [], [], [], []
    per_leaf_stats = []

    gt_tris_total = 0
    pg_tris_total = 0
    np_tris_total = 0
    np_clean_tris_total = 0

    for idx, lraw in enumerate(leaves_raw):
        name = lraw["name"]
        surf = leaf_dict_to_geomdl_surface(lraw)

        # 1) Ground truth — NURBS tessellated directly on (N_U, N_V) grid
        gt_verts, gt_tris = tessellate_nurbs(surf, N_U, N_V)
        gt_parts.append({"name": f"{name}_gt", "material": "blade_gt",
                         "verts": gt_verts, "tris": gt_tris})
        gt_tris_total += len(gt_tris)

        # 2) Extract skeleton + widths (in cm, field-position retained)
        skeleton_cm, widths_cm = _skel_widths_from_surf(surf)

        # 3) PlantGL ribbon sweep
        try:
            pg_verts, pg_tris = plantgl_blade(skeleton_cm, widths_cm)
            pg_parts.append({"name": f"{name}_plantgl", "material": "blade_pg",
                             "verts": pg_verts, "tris": pg_tris})
            pg_tris_total += len(pg_tris)
        except Exception as e:
            print(f"  !! PlantGL failed on {name}: {e}")
            pg_verts, pg_tris = np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64)

        # 4) Numpy lofter — production config (wave/curl/twist/ruffle)
        try:
            nv, nt = _numpy_loft_blade(f"{name}_np", skeleton_cm, widths_cm,
                                        organ_id=idx, with_waves=True,
                                        position=idx)
            np_parts.append({"name": f"{name}_np", "material": "blade_np",
                             "verts": nv, "tris": nt})
            np_tris_total += len(nt)
        except Exception as e:
            print(f"  !! Numpy (waves) failed on {name}: {e}")
            nv, nt = np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64)

        # 5) Numpy lofter — flat ribbon (no synthetic deformations; fair vs PlantGL)
        try:
            ncv, nct = _numpy_loft_blade(f"{name}_npc", skeleton_cm, widths_cm,
                                          organ_id=idx, with_waves=False,
                                          position=idx)
            np_clean_parts.append({"name": f"{name}_np_clean",
                                   "material": "blade_np_clean",
                                   "verts": ncv, "tris": nct})
            np_clean_tris_total += len(nct)
        except Exception as e:
            print(f"  !! Numpy (clean) failed on {name}: {e}")
            ncv, nct = np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64)

        # 6) Metrics — nearest-point RMS from each method's verts to GT verts
        pg_rms = rms_nearest(pg_verts, gt_verts) if len(pg_verts) else float("nan")
        np_rms = rms_nearest(nv, gt_verts) if len(nv) else float("nan")
        npc_rms = rms_nearest(ncv, gt_verts) if len(ncv) else float("nan")

        total_len = np.sum(np.linalg.norm(np.diff(skeleton_cm, axis=0), axis=1))
        max_w = float(np.max(widths_cm))

        per_leaf_stats.append({
            "name": name, "len_cm": float(total_len), "max_w_cm": max_w,
            "gt_tris": len(gt_tris), "pg_tris": len(pg_tris),
            "np_tris": len(nt), "npc_tris": len(nct),
            "pg_rms": pg_rms, "np_rms": np_rms, "npc_rms": npc_rms,
        })
        print(f"  {name:>6}  L={total_len:6.1f}cm  w_max={max_w:5.2f}cm  "
              f"GT={len(gt_tris):>4}  PG={len(pg_tris):>4}  "
              f"NP={len(nt):>5}  NPc={len(nct):>5}  |  "
              f"RMS(cm) PG={pg_rms:5.2f} NP={np_rms:5.2f} NPc={npc_rms:5.2f}")

    # ------ Write combined OBJs ------
    print("\n=== Writing OBJs ===")
    for label, parts in [("ground_truth", gt_parts),
                         ("plantgl", pg_parts),
                         ("numpy", np_parts),
                         ("numpy_clean", np_clean_parts)]:
        out = OUT_DIR / f"0317_{label}.obj"
        write_obj_multi(out, parts, header=f"0317 — {label}")
        kb = out.stat().st_size / 1024.0
        print(f"  {label:>12s}: {out}  ({kb:.1f} KB)")

    # ------ Summary ------
    print("\n=== Summary (plant 0317) ===")
    print(f"  {'method':<14}{'tris':>9}{'size (KB)':>12}{'mean RMS (cm)':>16}")
    for label, parts, total_tris, rms_key in [
        ("ground truth", gt_parts, gt_tris_total, None),
        ("PlantGL",      pg_parts, pg_tris_total, "pg_rms"),
        ("numpy",        np_parts, np_tris_total, "np_rms"),
        ("numpy clean",  np_clean_parts, np_clean_tris_total, "npc_rms"),
    ]:
        path = OUT_DIR / f"0317_{label.replace(' ', '_').replace('numpy_clean','numpy_clean').lower()}.obj"
        # fix path name mapping (label → filename)
        fname_map = {"ground truth": "0317_ground_truth.obj",
                     "PlantGL": "0317_plantgl.obj",
                     "numpy": "0317_numpy.obj",
                     "numpy clean": "0317_numpy_clean.obj"}
        path = OUT_DIR / fname_map[label]
        kb = path.stat().st_size / 1024.0
        if rms_key is None:
            rms_str = "    (reference)"
        else:
            rms_vals = [s[rms_key] for s in per_leaf_stats
                        if not np.isnan(s[rms_key])]
            rms_str = f"{np.mean(rms_vals):>14.3f}" if rms_vals else "      n/a"
        print(f"  {label:<14}{total_tris:>9}{kb:>12.1f}{rms_str}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
