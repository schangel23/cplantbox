"""Side-by-side MP4 of CPlantBox G1 skeleton (left) and lofted G3 mesh (right)
growing from day 1 to day 180 under FA-on Juelich met.

Run:
    cd /home/lukas/PHD/CPlantBox
    source cpbenv/bin/activate
    PYTHONPATH=. python dart/tools/blender_preview/_gen_growth_animation.py
"""
from __future__ import annotations

import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import plantbox as pb  # type: ignore[import-not-found]
from matplotlib.animation import FFMpegWriter
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

from dart.coupling.config import DEFAULT_XML
from dart.coupling.geometry import extract_organs_for_lofter, loft_organs
from dart.coupling.growth.grow import grow_plant
from dart.coupling.growth.phenology import detect_v_stage

PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = PROJECT_ROOT / "dart" / "coupling" / "output" / "growth_animation"
OUT_FILE = OUT_DIR / "growth.mp4"

SEED = 42
DAYS = list(range(5, 181, 5))   # 5,10,...,180 → 36 frames
FPS = 6
MAX_TRIS_PER_FRAME = 12000      # subsample for matplotlib perf

# Plot extents (cm). Day-180 maize tops out ~155 cm; canopy radius ~70 cm.
# XY recentred per-frame on the plant (seed xy ≈ (200,200) in world coords).
XY_HALFWIDTH = 80.0
Z_LIM = (-2, 200)

COLOR_STEM = "#6b4226"
COLOR_LEAF = "#2e7d32"
COLOR_TASSEL = "#c9a227"


def extract_skeleton_lines(plant) -> tuple[list[np.ndarray], list[str]]:
    """Return (segments, colors) — one entry per organ for Line3DCollection."""
    segments: list[np.ndarray] = []
    colors: list[str] = []
    for organ in plant.getOrgans(pb.stem):
        nodes = [(n.x, n.y, n.z) for n in organ.getNodes()]
        if len(nodes) < 2:
            continue
        params = organ.getOrganRandomParameter()
        name = (params.name or "").lower() if params else ""
        is_tassel = "tassel" in name or "spike" in name or "branch" in name
        color = COLOR_TASSEL if is_tassel else COLOR_STEM
        pts = np.asarray(nodes, dtype=np.float64)
        segments.append(np.stack([pts[:-1], pts[1:]], axis=1))
        colors.append(color)
    for organ in plant.getOrgans(pb.leaf):
        nodes = [(n.x, n.y, n.z) for n in organ.getNodes()]
        if len(nodes) < 2:
            continue
        pts = np.asarray(nodes, dtype=np.float64)
        segments.append(np.stack([pts[:-1], pts[1:]], axis=1))
        colors.append(COLOR_LEAF)
    return segments, colors


def subsample_triangles(verts: np.ndarray, tris: np.ndarray, max_tris: int):
    if len(tris) <= max_tris:
        return verts, tris
    stride = int(np.ceil(len(tris) / max_tris))
    return verts, tris[::stride]


def mesh_polys(verts: np.ndarray, tris: np.ndarray) -> np.ndarray:
    """(M, 3, 3) array of triangle vertex coords for Poly3DCollection."""
    return verts[tris]


def style_axis(ax, title: str, cx: float, cy: float) -> None:
    ax.set_xlim(cx - XY_HALFWIDTH, cx + XY_HALFWIDTH)
    ax.set_ylim(cy - XY_HALFWIDTH, cy + XY_HALFWIDTH)
    ax.set_zlim(*Z_LIM)
    ax.set_box_aspect((1, 1, (Z_LIM[1] - Z_LIM[0]) / (2 * XY_HALFWIDTH)))
    ax.set_title(title, fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zlabel("z [cm]")
    ax.view_init(elev=12, azim=-60)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output: {OUT_FILE}")
    print(f"Frames: {len(DAYS)} days {DAYS[0]}..{DAYS[-1]} step 5")

    fig = plt.figure(figsize=(14, 7), dpi=110)
    ax_skel = fig.add_subplot(1, 2, 1, projection="3d")
    ax_mesh = fig.add_subplot(1, 2, 2, projection="3d")

    writer = FFMpegWriter(fps=FPS, bitrate=4000, codec="libx264")
    t_total = time.time()
    with writer.saving(fig, str(OUT_FILE), dpi=110):
        for i, day in enumerate(DAYS):
            t0 = time.time()
            plant = grow_plant(
                str(DEFAULT_XML),
                simulation_time=day,
                seed=SEED,
                enable_photosynthesis=False,
            )

            label = detect_v_stage(plant)
            tt = plant.getAccumulatedTT() if hasattr(plant, "getAccumulatedTT") else -1.0

            segments, colors = extract_skeleton_lines(plant)

            organs = extract_organs_for_lofter(plant, species="maize", skip_roots=True)
            mesh = loft_organs(organs, stem_sides=8, use_nurbs_backend=True)
            verts, tris = subsample_triangles(
                mesh.vertices, mesh.indices, MAX_TRIS_PER_FRAME
            )

            cx = float(mesh.vertices[:, 0].mean())
            cy = float(mesh.vertices[:, 1].mean())

            ax_skel.clear()
            ax_mesh.clear()
            style_axis(
                ax_skel, f"G1 skeleton — day {day} ({label}, TT={tt:.0f})", cx, cy
            )
            style_axis(ax_mesh, f"G3 lofted mesh — {mesh.n_triangles} tris", cx, cy)

            if segments:
                flat_segs = np.concatenate(segments, axis=0)
                seg_colors = np.concatenate([
                    np.repeat([c], len(s), axis=0) if False else [c] * len(s)
                    for c, s in zip(colors, segments)
                ])
                lc = Line3DCollection(
                    flat_segs, colors=list(seg_colors), linewidths=1.4
                )
                ax_skel.add_collection3d(lc)

            polys = mesh_polys(verts, tris)
            pc = Poly3DCollection(
                polys,
                facecolor="#3b8a3b",
                edgecolor="none",
                alpha=0.65,
                linewidths=0.0,
            )
            ax_mesh.add_collection3d(pc)

            writer.grab_frame()

            elapsed = time.time() - t0
            print(
                f"  [{i + 1:2d}/{len(DAYS)}] day {day:3d}  {label:<14s} "
                f"TT={tt:6.1f}  tris={mesh.n_triangles:6d}  ({elapsed:.1f}s)"
            )

    plt.close(fig)
    print(f"\nDone — wrote {OUT_FILE} in {time.time() - t_total:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
