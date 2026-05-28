"""Generate (partial, complete) point-cloud pairs from CPlantBox for training a
point-completion network (CRA-PCN) on in-domain maize geometry.

complete = dense area-weighted surface sampling of the lofted whole-plant mesh.
partial  = single-viewpoint Hidden-Point-Removal (Katz et al. 2007) of that
           cloud + 5 mm voxel downsample + mm noise, mimicking the FP4D robot's
           one-sided self-occluded scan.

Run from CPlantBox root with the dev env:
    cpbenv/bin/python dart/coupling/scripts/gen_synth_completion_pairs.py \
        --n 1 --out /home/lukas/pointr/synth --viz
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np

from dart.coupling.growth.grow import grow_plant
from dart.coupling.geometry.cplantbox_adapter import extract_organs_for_lofter
from dart.coupling.geometry.g1_to_g3 import loft_organs

XML = "dart/coupling/data/maize_calibrated.xml"


def sample_mesh(vertices, indices, n):
    """Area-weighted uniform surface sampling of a triangle mesh."""
    v = vertices[indices]                       # (T,3,3)
    a = np.linalg.norm(np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0]), axis=1) * 0.5
    p = a / a.sum()
    tri = np.random.choice(len(indices), n, p=p)
    u = np.random.rand(n, 1); w = np.random.rand(n, 1)
    over = (u + w > 1).ravel(); u[over] = 1 - u[over]; w[over] = 1 - w[over]
    A, B, C = v[tri, 0], v[tri, 1], v[tri, 2]
    return A + u * (B - A) + w * (C - A)


def hpr(points, viewpoint, param=3.2):
    """Katz et al. 2007 Hidden Point Removal — indices visible from viewpoint."""
    from scipy.spatial import ConvexHull
    p = points - viewpoint
    norm = np.linalg.norm(p, axis=1, keepdims=True)
    norm = np.maximum(norm, 1e-9)
    R = norm.max() * (10 ** param)
    flipped = p + 2 * (R - norm) * p / norm
    hull = ConvexHull(np.vstack([flipped, np.zeros((1, 3))]))
    vis = hull.vertices
    return vis[vis < len(points)]


def voxel_down(pc, vox):
    mn = pc.min(0); idx = np.floor((pc - mn) / vox).astype(np.int64)
    _, u = np.unique(idx, axis=0, return_index=True)
    return pc[u]


def make_partial(complete, rng, vox_cm=0.5, noise_mm=1.0):
    """One viewpoint (side + above, FP4D-robot-like), HPR, voxel, noise."""
    c = complete.mean(0)
    h = np.ptp(complete[:, 2]) + 1e-6
    az = rng.uniform(0, 2 * np.pi)
    el = rng.uniform(np.radians(25), np.radians(65))   # robot looks down-ish
    dist = h * rng.uniform(2.0, 3.5)
    vp = c + dist * np.array([np.cos(el) * np.cos(az),
                              np.cos(el) * np.sin(az),
                              np.sin(el)])
    vis = hpr(complete, vp)
    part = complete[vis]
    part = voxel_down(part, vox_cm)
    part = part + rng.normal(0, noise_mm / 10.0, part.shape)   # mm -> cm
    return part


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--out", default="/home/lukas/pointr/synth")
    ap.add_argument("--n_complete", type=int, default=16384)
    ap.add_argument("--day_lo", type=int, default=40)
    ap.add_argument("--day_hi", type=int, default=80)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--viz", action="store_true")
    a = ap.parse_args()
    out = Path(a.out); (out / "complete").mkdir(parents=True, exist_ok=True)
    (out / "partial").mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(a.seed0)
    for i in range(a.n):
        seed = a.seed0 + i
        day = int(rng.integers(a.day_lo, a.day_hi + 1))
        plant = grow_plant(XML, simulation_time=day, seed=seed)
        organs = extract_organs_for_lofter(plant)
        mesh = loft_organs(organs, use_nurbs_backend=True)
        comp = sample_mesh(mesh.vertices, mesh.indices, a.n_complete)
        part = make_partial(comp, rng)
        np.save(out / "complete" / f"plant_{seed:04d}_d{day}.npy", comp.astype(np.float32))
        np.save(out / "partial" / f"plant_{seed:04d}_d{day}.npy", part.astype(np.float32))
        print(f"[{i+1}/{a.n}] seed {seed} day {day}: complete {len(comp)} "
              f"({np.ptp(comp[:,2]):.0f}cm) -> partial {len(part)} "
              f"({100*len(part)/len(comp):.0f}% retained)")
        if a.viz and i == 0:
            import matplotlib; matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, axs = plt.subplots(1, 3, figsize=(14, 7))
            for ax, P, t, col in [(axs[0], comp, f"COMPLETE {len(comp)}", "#1f77b4"),
                                  (axs[1], part, f"PARTIAL {len(part)}", "#d62728"),
                                  (axs[2], None, "overlay y-z", None)]:
                if P is not None:
                    ax.scatter(P[:, 1], P[:, 2], s=0.5, c=col, alpha=0.4)
                else:
                    ax.scatter(comp[:, 1], comp[:, 2], s=0.5, c="#1f77b4", alpha=0.2)
                    ax.scatter(part[:, 1], part[:, 2], s=0.6, c="#d62728", alpha=0.5)
                ax.set_title(t); ax.set_aspect("equal"); ax.set_xlabel("y"); ax.set_ylabel("z")
            fig.suptitle(f"synthetic pair · seed {seed} day {day}")
            vp = out / "preview.png"; fig.savefig(vp, dpi=120, bbox_inches="tight")
            print(f"  viz -> {vp}")


if __name__ == "__main__":
    main()
