"""FP4D MonGraphSeg graph-segmentation visual check (2026-05-28).

Runs the faithful MonGraphSeg graph pipeline (mongraphseg_graph) on the test
plant, overlays the skeleton tree (stem + leaf branches) on the cloud, and
fits the vendored NURBS surface per recovered leaf. Companion to
viz_fp4d_segmentation_2026-05-28.py (which drives the geodesic path).

    cpbenv/bin/python dart/coupling/scripts/viz_fp4d_mongraphseg_2026-05-28.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PHENO4D_DIR = Path(__file__).resolve().parents[3] / "src" / "visualisation" / "pheno4d_to_g1"
sys.path.insert(0, str(PHENO4D_DIR))

import loader  # type: ignore  # noqa: E402
import mongraphseg_graph as MG  # type: ignore  # noqa: E402
import nurbs_leaf_fit  # type: ignore  # noqa: E402

SCAN = "/home/lukas/PHD/Resources/PHENOROAM DATA ASSIMILATION May 2026/doi-10.60507-fk2-hyi2ds/Plot04/230621.las"
OUT_DIR = Path("/home/lukas/PHD/CPlantBox/dart/coupling/output/fp4d_mongraphseg_2026-05-28")

ORGAN_COLOURS = {"stem": "#888888", "leaf_1": "#e41a1c", "leaf_2": "#377eb8",
                 "leaf_3": "#4daf4a", "leaf_4": "#984ea3", "leaf_5": "#ff7f00",
                 "leaf_6": "#a65628", "leaf_7": "#f781bf", "leaf_8": "#999900"}
_AXIS = {"xz": (0, 2, "x", "z"), "yz": (1, 2, "y", "z"), "xy": (0, 1, "x", "y")}


def _scatter_organs(ax, organs, debug, view, with_tree=True, legend=True):
    a, b, la, lb = _AXIS[view]
    allp = np.concatenate([p for p in organs.values() if p.size])
    for name, pts in organs.items():
        if pts.size == 0:
            continue
        ax.scatter(pts[:, a], pts[:, b], c=ORGAN_COLOURS.get(name, "#999"),
                   s=0.6, alpha=0.45, linewidths=0, label=f"{name} ({len(pts):,})")
    if with_tree:
        npos = debug["node_positions"]
        sp = debug["stem_path"]
        sp_xy = np.array([npos[n] for n in sp])
        ax.plot(sp_xy[:, a], sp_xy[:, b], "-k", lw=2.0, alpha=0.8)
        for path in debug["leaf_dict"].values():
            lp = np.array([npos[n] for n in path])
            ax.plot(lp[:, a], lp[:, b], "-", color="orange", lw=1.3, alpha=0.9)
    ax.set_xlabel(f"{la} [cm]"); ax.set_ylabel(f"{lb} [cm]")
    ax.set_xlim(allp[:, a].min() - 2, allp[:, a].max() + 2)
    ax.set_ylim(allp[:, b].min() - 2, allp[:, b].max() + 2)
    ax.set_aspect("equal", adjustable="box"); ax.grid(alpha=0.2)
    if legend:
        ax.legend(fontsize=7, loc="best", framealpha=0.85)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    row = loader.load_las(SCAN, height_lo_m=0.15, voxel_m=0.005)
    row_axis, centres = loader.separate_plants_along_row(row)
    plant = loader.crop_plant_window(row, row_axis, centres[9],
                                     window_cm=20.0, cross_row_window_cm=25.0)
    print(f"[viz] plant crop {plant.shape[0]:,} pts, height {np.ptp(plant[:,2]):.1f} cm")

    organs, debug = MG.segment_plant_graph(plant, return_debug=True)
    print(f"[viz] graph: {debug['tree'].number_of_nodes()} tree nodes, "
          f"stem {len(debug['stem_path'])} nodes, {len(debug['leaf_dict'])} leaves")
    for n, p in organs.items():
        if len(p):
            print(f"   {n:8s} {len(p):5,} pts")

    leaf_names = sorted(n for n in organs if n.startswith("leaf"))
    fits = {}
    for name in leaf_names:
        pts = organs[name]
        if pts.shape[0] < nurbs_leaf_fit.DEFAULTS["min_points_per_leaf"]:
            print(f"[viz] {name}: too few points ({pts.shape[0]}), skip NURBS")
            fits[name] = None
            continue
        try:
            srf = nurbs_leaf_fit.fit_leaf_nurbs_surface(pts / 100.0)
            fits[name] = srf
            print(f"[viz] {name}: NURBS RMS {srf['fit_rms']*1000:.2f} mm "
                  f"({srf['n_used']} pts)")
        except Exception as exc:
            print(f"[viz] {name}: NURBS failed ({exc})")
            fits[name] = None

    n_leaves = len(leaf_names)
    leaf_rows = (n_leaves + 1) // 2
    fig = plt.figure(figsize=(16, 4 + 5 * leaf_rows))
    gs = fig.add_gridspec(1 + leaf_rows, 6, height_ratios=[3.2] + [5.0] * leaf_rows,
                          hspace=0.4, wspace=0.55)
    for k, view in enumerate(["xz", "yz", "xy"]):
        ax = fig.add_subplot(gs[0, 2 * k:2 * k + 2])
        _scatter_organs(ax, organs, debug, view, legend=(k == 0))
        ax.set_title(f"MonGraphSeg · {view}")

    for i, name in enumerate(leaf_names):
        r = 1 + i // 2
        c = (i % 2) * 3
        ax = fig.add_subplot(gs[r, c:c + 3], projection="3d")
        pts = organs[name]
        srf = fits[name]
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c="#1f77b4", s=1.2,
                   alpha=0.35, linewidths=0)
        if srf is not None:
            grid = nurbs_leaf_fit.nrbeval_grid(srf, 60, 18) * 100.0
            ax.plot_surface(grid[:, :, 0], grid[:, :, 1], grid[:, :, 2],
                            color="#ff7f0e", alpha=0.55, linewidth=0)
        ax.set_box_aspect((max(np.ptp(pts[:, 0]), 1), max(np.ptp(pts[:, 1]), 1),
                           max(np.ptp(pts[:, 2]), 1)))
        rms = f"RMS {srf['fit_rms']*1000:.1f} mm" if srf else "NURBS skipped"
        ax.set_title(f"{name} · {len(pts):,} pts · z-span {np.ptp(pts[:,2]):.1f} cm · {rms}",
                     fontsize=9)

    fig.suptitle(f"FP4D MonGraphSeg graph segmentation · Plot04/230621 centre[9] · "
                 f"{plant.shape[0]:,} pts · {n_leaves} leaves", fontsize=12, y=0.997)
    out = OUT_DIR / "fp4d_mongraphseg_overview.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] wrote {out}")


if __name__ == "__main__":
    main()
