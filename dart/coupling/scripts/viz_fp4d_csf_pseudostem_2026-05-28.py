"""FP4D CSF + pseudostem-segmenter visual check (2026-05-28).

Loads the test scan with the new CSF (Cloth Simulation Filter) ground method
— DEM-normalised so the basal leaf the flat height cut deleted is recovered —
crops the anchor plant with a lean-robust wide cross window, runs the no-stem
``segment_plant_pseudostem`` segmenter, and renders xz / yz / xy + a per-organ
3-D panel. The yz panel is the one to watch: the pseudostem (black) curl is the
gap-driven artefact the completion work targets.

    cpbenv/bin/python dart/coupling/scripts/viz_fp4d_csf_pseudostem_2026-05-28.py
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

SCAN = "/home/lukas/PHD/Resources/PHENOROAM DATA ASSIMILATION May 2026/doi-10.60507-fk2-hyi2ds/Plot04/230621.las"
OUT_DIR = Path("/home/lukas/PHD/CPlantBox/dart/coupling/output/fp4d_mongraphseg_2026-05-28")
ANCHOR_CROSS = -2.9  # flat-cut centre[9] row coordinate

_CMAP = plt.get_cmap("tab20")
_AXIS = {"xz": (0, 2, "x", "z"), "yz": (1, 2, "y", "z"), "xy": (0, 1, "x", "y")}


def _colour(name):
    if name == "pseudostem":
        return "#222222"
    return _CMAP(int(name.split("_")[1]) % 20)


def _scatter(ax, organs, debug, view, legend=False):
    a, b, la, lb = _AXIS[view]
    allp = np.concatenate([p for p in organs.values() if p.size])
    for name in sorted(organs, key=lambda n: (n != "pseudostem", n)):
        pts = organs[name]
        if pts.size == 0:
            continue
        ax.scatter(pts[:, a], pts[:, b], c=[_colour(name)], s=0.6, alpha=0.5,
                   linewidths=0, label=f"{name} ({len(pts):,})")
    npos = debug["node_positions"]
    for u, v in debug["pseudostem_edges"]:
        p = np.array([npos[u], npos[v]])
        ax.plot(p[:, a], p[:, b], "-k", lw=1.6, alpha=0.85)
    ax.set_xlabel(f"{la} [cm]"); ax.set_ylabel(f"{lb} [cm]")
    ax.set_xlim(allp[:, a].min() - 2, allp[:, a].max() + 2)
    ax.set_ylim(allp[:, b].min() - 2, allp[:, b].max() + 2)
    ax.set_aspect("equal", adjustable="box"); ax.grid(alpha=0.2)
    if legend:
        ax.legend(fontsize=6, loc="best", framealpha=0.85, ncol=2)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    row = loader.load_las(SCAN, voxel_m=0.005, ground_method="csf")
    row_axis, centres = loader.separate_plants_along_row(row)
    c = centres[int(np.argmin(np.abs(centres - ANCHOR_CROSS)))]
    plant = loader.crop_plant_window(row, row_axis, c, window_cm=20.0,
                                     cross_row_window_cm=40.0)
    print(f"[viz] CSF crop {plant.shape[0]:,} pts, height {np.ptp(plant[:,2]):.1f} cm, "
          f"base z {plant[:,2].min():.1f} cm")

    organs, debug = MG.segment_plant_pseudostem(plant, n_skel_nodes=250,
                                                return_debug=True)
    leaf_names = sorted((n for n in organs if n.startswith("leaf")),
                        key=lambda n: int(n.split("_")[1]))
    print(f"[viz] pseudostem {len(organs['pseudostem']):,} pts, {len(leaf_names)} leaves")
    for n in leaf_names:
        p = organs[n]
        print(f"   {n:8s} {len(p):5,} pts  z {p[:,2].min():4.1f}..{p[:,2].max():4.1f}")

    fig = plt.figure(figsize=(17, 7))
    gs = fig.add_gridspec(1, 4, wspace=0.35)
    for k, view in enumerate(["xz", "yz", "xy"]):
        ax = fig.add_subplot(gs[0, k])
        _scatter(ax, organs, debug, view, legend=(k == 0))
        ax.set_title(f"CSF + pseudostem · {view}"
                     + ("  ← curl view" if view == "yz" else ""))
    ax3 = fig.add_subplot(gs[0, 3], projection="3d")
    for name in sorted(organs, key=lambda n: (n != "pseudostem", n)):
        p = organs[name]
        if p.size:
            ax3.scatter(p[:, 0], p[:, 1], p[:, 2], c=[_colour(name)], s=0.8,
                        alpha=0.45, linewidths=0)
    ax3.set_box_aspect((np.ptp(plant[:, 0]), np.ptp(plant[:, 1]), np.ptp(plant[:, 2])))
    ax3.set_title("3-D")
    fig.suptitle(f"FP4D CSF + pseudostem · Plot04/230621 · {plant.shape[0]:,} pts · "
                 f"{len(leaf_names)} leaves · base z={plant[:,2].min():.1f} cm",
                 fontsize=12)
    out = OUT_DIR / "fp4d_csf_pseudostem_overview.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] wrote {out}")


if __name__ == "__main__":
    main()
