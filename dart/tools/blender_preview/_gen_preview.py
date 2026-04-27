"""Generate Blender preview OBJs comparing the library variants.

Produces four leaves laid out side-by-side per position:
  x=0  : median, plain
  x=30 : median, muted-deformations
  x=60 : draw (seed A), plain
  x=90 : draw (seed B), muted-deformations

A separate OBJ is written per leaf position (0..13) so Blender can load
them individually, plus one combined OBJ per variant showing all positions
stacked vertically (z).
"""
from __future__ import annotations
from pathlib import Path

import numpy as np

from dart.coupling.geometry.canonical_library import (
    build_from_maizefield3d, _default_canonical_json,
)
from dart.coupling.geometry.nurbs_blade import loft_leaf_nurbs


OUT = Path(__file__).resolve().parent


def _plain_organ(cps_local: np.ndarray, collar: np.ndarray,
                 mature_len: float = 50.0):
    return {
        "type": "leaf",
        "organ_id": 0,
        "surface_cps_local": cps_local.copy(),
        "collar_pos": collar.copy(),
        "collar_tangent": np.array([1.0, 0.0, 0.0]),
        "parent_tangent": np.array([0.0, 0.0, 1.0]),
        "mature_length": mature_len,
        "current_length": mature_len,
        "skeleton": np.column_stack([
            collar[0] + np.linspace(0, mature_len, 20),
            np.full(20, collar[1]),
            np.full(20, collar[2]),
        ]),
    }


def _deformed_organ(*args, organ_id: int, **kw):
    o = _plain_organ(*args, **kw)
    rng = np.random.RandomState(organ_id * 37 + 7)
    o["organ_id"] = organ_id
    o.update(
        wave_normal_amp=rng.uniform(0.3, 0.6),
        wave_normal_freq=rng.uniform(2.5, 4.0),
        wave_normal_phase=rng.uniform(0, 2 * np.pi),
        twist_max=rng.choice([-1, 1]) * rng.uniform(0.1, 0.25),
        curl_amp=rng.uniform(0.3, 0.6),
        curl_freq=rng.uniform(1.0, 2.0),
        curl_phase=rng.uniform(0, 2 * np.pi),
        curl_onset=0.15,
        ramp_onset=0.15,
        maturity_fraction=1.0,
    )
    return o


def _write_obj(path: Path, verts: np.ndarray, tris: np.ndarray,
               normals: np.ndarray | None = None):
    with path.open("w") as f:
        f.write(f"# {path.name}\n")
        for v in verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        if normals is not None:
            for n in normals:
                f.write(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n")
        for t in tris:
            a, b, c = int(t[0]) + 1, int(t[1]) + 1, int(t[2]) + 1
            if normals is not None:
                f.write(f"f {a}//{a} {b}//{b} {c}//{c}\n")
            else:
                f.write(f"f {a} {b} {c}\n")


def _loft(organ):
    r = loft_leaf_nurbs(organ, n_u_eval=40, n_v_eval=9)
    return r[0], r[1], r[2]  # verts, tris (n_tris,3), normals


def _combine(meshes):
    """Concatenate (verts, tris, normals) with index offsets."""
    V, T, N = [], [], []
    off = 0
    for v, t, n in meshes:
        V.append(v)
        T.append(t + off)
        N.append(n)
        off += v.shape[0]
    return np.vstack(V), np.vstack(T), np.vstack(N)


def main():
    lib_med = build_from_maizefield3d(_default_canonical_json(),
                                      reducer="median")
    lib_d1 = build_from_maizefield3d(_default_canonical_json(),
                                     reducer="draw", draw_seed=42)
    lib_d2 = build_from_maizefield3d(_default_canonical_json(),
                                     reducer="draw", draw_seed=99)

    n_pos = lib_med["cps_local"].shape[0]
    print(f"Library: {n_pos} positions")

    variants = {
        "median_plain": (lib_med, False),
        "median_deformed": (lib_med, True),
        "draw42_plain": (lib_d1, False),
        "draw42_deformed": (lib_d1, True),
        "draw99_deformed": (lib_d2, True),
    }

    # One combined OBJ per variant, stacked vertically by position.
    for name, (lib, deformed) in variants.items():
        meshes = []
        for pos in range(n_pos):
            cps = np.asarray(lib["cps_local"][pos], dtype=np.float64)
            collar = np.array([0.0, 0.0, 10.0 * pos])
            if deformed:
                organ = _deformed_organ(cps, collar, organ_id=pos)
            else:
                organ = _plain_organ(cps, collar)
                organ["organ_id"] = pos
            v, t, n = _loft(organ)
            meshes.append((v, t, n))
        V, T, N = _combine(meshes)
        path = OUT / f"{name}.obj"
        _write_obj(path, V, T, N)
        print(f"  {path.name}: {V.shape[0]} verts, {T.shape[0]} tris")

    # Side-by-side 4-up layout at a single mid-canopy position so the user
    # can compare variants directly on one leaf.
    mid = n_pos // 2
    side_variants = [
        ("median_plain", lib_med, False, 42),
        ("median_deformed", lib_med, True, 42),
        ("draw42_plain", lib_d1, False, 42),
        ("draw42_deformed", lib_d1, True, 42),
    ]
    meshes = []
    for i, (label, lib, deformed, seed) in enumerate(side_variants):
        cps = np.asarray(lib["cps_local"][mid], dtype=np.float64)
        collar = np.array([0.0, 40.0 * i, 0.0])  # space along +y
        if deformed:
            organ = _deformed_organ(cps, collar, organ_id=seed)
        else:
            organ = _plain_organ(cps, collar)
            organ["organ_id"] = i
        print(f"  side-by-side [{i}] {label} at y={collar[1]}")
        v, t, n = _loft(organ)
        meshes.append((v, t, n))
    V, T, N = _combine(meshes)
    path = OUT / f"compare_pos{mid}_four_up.obj"
    _write_obj(path, V, T, N)
    print(f"\n  {path.name}: {V.shape[0]} verts, {T.shape[0]} tris")
    print(f"\n  Layout (y-axis): 0=median/plain  40=median/deformed  "
          f"80=draw42/plain  120=draw42/deformed")


if __name__ == "__main__":
    main()
