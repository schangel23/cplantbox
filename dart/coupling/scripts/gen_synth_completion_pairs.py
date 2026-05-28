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


def occlude_depthbuffer(points, viewpoint, ang_res_deg=0.6, depth_tol_cm=1.5):
    """Angular z-buffer occlusion: keep the frontmost surface per viewing
    direction (+ a depth tolerance), drop anything shadowed behind it.

    This is the physically faithful core of self-occlusion for a triangulation
    scanner — a lower/inner leaf behind an upper leaf along the same line of
    sight is removed, exactly the ground-up self-shadowing FP4D shows.
    """
    d = points - viewpoint
    r = np.linalg.norm(d, axis=1)
    r = np.maximum(r, 1e-9)
    az = np.arctan2(d[:, 1], d[:, 0])
    el = np.arcsin(np.clip(d[:, 2] / r, -1, 1))
    res = np.radians(ang_res_deg)
    ai = np.floor(az / res).astype(np.int64)
    ei = np.floor(el / res).astype(np.int64)
    key = np.stack([ai, ei], 1)
    _, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.ravel()
    bin_min = np.full(inv.max() + 1, np.inf)
    np.minimum.at(bin_min, inv, r)        # nearest range per angular bin
    return np.where(r <= bin_min[inv] + depth_tol_cm)[0]


def scanner_viewpoint(complete, rng, mostly_above=True):
    c = complete.mean(0)
    h = np.ptp(complete[:, 2]) + 1e-6
    az = rng.uniform(0, 2 * np.pi)
    # FP4D gantry looks mostly down the canopy; bias elevation high, some oblique
    el = rng.uniform(np.radians(55), np.radians(85)) if mostly_above \
        else rng.uniform(np.radians(25), np.radians(55))
    dist = h * rng.uniform(2.0, 4.0)
    return c + dist * np.array([np.cos(el) * np.cos(az),
                                np.cos(el) * np.sin(az), np.sin(el)])


def ground_up_dropout(points, rng, strength):
    """Lower canopy hidden under upper leaves: drop probability grows with the
    amount of canopy mass above a point (count of points higher within a column)."""
    from scipy.spatial import KDTree
    z = points[:, 2]
    tree = KDTree(points[:, :2])
    # neighbours in an xy-column; fraction of them that sit above this point
    nb = tree.query_ball_point(points[:, :2], r=3.0)
    above = np.array([np.mean(z[ix] > zi) if ix else 0.0 for ix, zi in zip(nb, z)])
    pdrop = strength * above
    return np.where(rng.random(len(points)) > pdrop)[0]


def sector_loss(points, rng, frac):
    """Remove a contiguous azimuthal wedge about the canopy centroid (one-sided
    scan): the far side of the plant is simply never seen."""
    c = points[:, :2].mean(0)
    ang = np.arctan2(points[:, 1] - c[1], points[:, 0] - c[0])
    a0 = rng.uniform(-np.pi, np.pi)
    width = frac * 2 * np.pi
    dd = np.abs(np.angle(np.exp(1j * (ang - a0))))
    return np.where(dd > width / 2)[0]


def voxel_down(pc, vox):
    mn = pc.min(0); idx = np.floor((pc - mn) / vox).astype(np.int64)
    _, u = np.unique(idx, axis=0, return_index=True)
    return pc[u]


def make_partial(complete, rng, vox_cm=0.5, noise_mm=1.0):
    """Composite FP4D-like occlusion: depth-buffer self-shadowing (always) +
    randomised ground-up dropout + occasional one-sided sector loss, then voxel
    + mm noise. Randomised per sample so the dataset spans realistic patterns."""
    idx = occlude_depthbuffer(
        complete, scanner_viewpoint(complete, rng),
        ang_res_deg=rng.uniform(0.4, 0.9), depth_tol_cm=rng.uniform(1.0, 2.5))
    part = complete[idx]
    # ground-up self-occlusion
    part = part[ground_up_dropout(part, rng, strength=rng.uniform(0.3, 0.8))]
    # one-sided scan ~40% of the time
    if rng.random() < 0.4 and len(part) > 500:
        part = part[sector_loss(part, rng, frac=rng.uniform(0.15, 0.35))]
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
