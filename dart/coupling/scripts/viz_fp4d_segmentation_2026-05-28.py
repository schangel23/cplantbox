"""FP4D leaf-segmentation visual check (status snapshot 2026-05-28).

Mirrors the diagnostic call in FP4D_LEAF_SEGMENTATION_STATUS_2026-05-28.md:
loads Plot04/230621.las, separates plants along the row, crops centre[9],
runs segment_by_leaf_templates, fits NURBS per candidate, and writes a
multi-panel PNG so the current segmentation quality can be eyeballed.

Run from CPlantBox root with the local env:
    cpbenv/bin/python dart/coupling/scripts/viz_fp4d_segmentation_2026-05-28.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Force in-tree pheno4d_to_g1 (site-packages copy is stale, missing
# leaf_template_refinement). Import submodules directly to bypass the package
# __init__ which pulls plantbox (currently broken on this machine via
# libCPlantBox.so undefined-symbol).
PHENO4D_DIR = Path(__file__).resolve().parents[3] / "src" / "visualisation" / "pheno4d_to_g1"
sys.path.insert(0, str(PHENO4D_DIR))

import loader  # type: ignore  # noqa: E402
import segmenter  # type: ignore  # noqa: F401, E402  (used transitively)
import geodesic_assignment  # type: ignore  # noqa: F401, E402  (used transitively)
import leaf_template_refinement  # type: ignore  # noqa: E402
import nurbs_leaf_fit  # type: ignore  # noqa: E402

load_las = loader.load_las
separate_plants_along_row = loader.separate_plants_along_row
crop_plant_window = loader.crop_plant_window
segment_by_leaf_templates = leaf_template_refinement.segment_by_leaf_templates
write_labelled_xyz = leaf_template_refinement.write_labelled_xyz


SCAN = "/home/lukas/PHD/Resources/PHENOROAM DATA ASSIMILATION May 2026/doi-10.60507-fk2-hyi2ds/Plot04/230621.las"
OUT_DIR = Path(
    "/home/lukas/PHD/CPlantBox/dart/coupling/output/fp4d_segmentation_viz_2026-05-28"
)


# Distinct colours per organ. Stem is grey; leaves cycle through tabs.
ORGAN_COLOURS = {
    "stem": "#888888",
    "leaf_1": "#e41a1c",
    "leaf_2": "#377eb8",
    "leaf_3": "#4daf4a",
    "leaf_4": "#984ea3",
    "leaf_5": "#ff7f00",
    "leaf_6": "#a65628",
}


_AXIS_PAIRS = {"xz": (0, 2, "x", "z"),
               "yz": (1, 2, "y", "z"),
               "xy": (0, 1, "x", "y")}


def _set_data_lim_with_pad(ax, pts, a, b, pad=2.0):
    ax.set_xlim(pts[:, a].min() - pad, pts[:, a].max() + pad)
    ax.set_ylim(pts[:, b].min() - pad, pts[:, b].max() + pad)


def _scatter_height(ax, pts, view="xz", title="", s=0.6):
    a, b, la, lb = _AXIS_PAIRS[view]
    ax.scatter(pts[:, a], pts[:, b], c=pts[:, 2], cmap="viridis",
               s=s, alpha=0.55, linewidths=0)
    ax.set_xlabel(f"{la} [cm]"); ax.set_ylabel(f"{lb} [cm]")
    _set_data_lim_with_pad(ax, pts, a, b)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)
    ax.grid(alpha=0.2)


def _scatter_organs(ax, organs, view="xz", s=0.6, with_legend=True):
    a, b, la, lb = _AXIS_PAIRS[view]
    all_pts = np.concatenate([p for p in organs.values() if p.size], axis=0)
    for name, pts in organs.items():
        if pts.size == 0:
            continue
        c = ORGAN_COLOURS.get(name, "#999999")
        ax.scatter(pts[:, a], pts[:, b], c=c, s=s, alpha=0.5,
                   linewidths=0, label=f"{name} ({len(pts):,})")
    ax.set_xlabel(f"{la} [cm]"); ax.set_ylabel(f"{lb} [cm]")
    _set_data_lim_with_pad(ax, all_pts, a, b)
    ax.set_aspect("equal", adjustable="box")
    if with_legend:
        ax.legend(fontsize=7, loc="best", framealpha=0.85)
    ax.grid(alpha=0.2)


def _leaf_panel(ax, pts_cm, srf, nurbs_mod, title):
    """3-D scatter of one leaf with the NURBS surface overlaid."""
    ax.scatter(pts_cm[:, 0], pts_cm[:, 1], pts_cm[:, 2],
               c="#1f77b4", s=1.2, alpha=0.35, linewidths=0)
    if srf is not None:
        # main_NURBS works in m -> convert back to cm for plotting
        grid = nurbs_mod.nrbeval_grid(srf, 60, 18) * 100.0
        ax.plot_surface(grid[:, :, 0], grid[:, :, 1], grid[:, :, 2],
                        color="#ff7f0e", alpha=0.55, linewidth=0,
                        antialiased=True)
    ax.set_box_aspect((float(np.ptp(pts_cm[:, 0])),
                       float(np.ptp(pts_cm[:, 1])),
                       max(float(np.ptp(pts_cm[:, 2])), 1.0)))
    ax.set_xlabel("x [cm]", fontsize=8)
    ax.set_ylabel("y [cm]", fontsize=8)
    ax.set_zlabel("z [cm]", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_title(title, fontsize=9)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Higher z floor (15 cm vs 10) drops most mulch; cross_row_window_cm
    # bakes the cross-row clip into the loader.
    print(f"[viz] loading {SCAN}")
    row = load_las(SCAN, height_lo_m=0.15, voxel_m=0.005)
    row_axis, centres = separate_plants_along_row(row)
    print(f"[viz] detected {len(centres)} plants along axis {row_axis}")

    if len(centres) <= 9:
        raise SystemExit(f"need at least 10 plant centres; got {len(centres)}")
    plant = crop_plant_window(row, row_axis, centres[9],
                              window_cm=20.0, cross_row_window_cm=25.0)
    print(f"[viz] plant crop: {plant.shape[0]:,} pts, "
          f"height {np.ptp(plant[:, 2]):.1f} cm, "
          f"x ∈ [{plant[:, 0].min():.1f}, {plant[:, 0].max():.1f}] cm")

    organs, _ = segment_by_leaf_templates(plant, return_debug=True)
    summary = [(n, len(p)) for n, p in organs.items()]
    print("[viz] segmentation summary:")
    for n, c in summary:
        print(f"   {n:8s} {c:>6,} pts")

    nurbs_mod = nurbs_leaf_fit
    leaf_names = sorted(n for n in organs if n.startswith("leaf"))
    fits = {}
    for name in leaf_names:
        pts_cm = organs[name]
        if pts_cm.shape[0] < nurbs_mod.DEFAULTS["min_points_per_leaf"]:
            print(f"[viz] {name}: too few points ({pts_cm.shape[0]}), skipping NURBS")
            fits[name] = None
            continue
        try:
            # vendored fitter: units-agnostic, adaptive width bins, denoise.
            srf = nurbs_mod.fit_leaf_nurbs_surface(
                pts_cm / 100.0,  # work in metres -> fit_rms*1000 = mm
                nCtrlU=nurbs_mod.DEFAULTS["nCtrlU"],
                nCtrlV=nurbs_mod.DEFAULTS["nCtrlV"],
                p=nurbs_mod.DEFAULTS["p"],
                q=nurbs_mod.DEFAULTS["q"],
                lam=nurbs_mod.DEFAULTS["lam"],
                leftPrct=nurbs_mod.DEFAULTS["left_percentile"],
                rightPrct=nurbs_mod.DEFAULTS["right_percentile"],
            )
            fits[name] = srf
            print(f"[viz] {name}: NURBS RMS {srf['fit_rms']*1000:.2f} mm "
                  f"({srf['n_used']} pts after denoise)")
        except Exception as exc:
            print(f"[viz] {name}: NURBS fit failed ({exc})")
            fits[name] = None

    # ---------- multi-panel figure -----------------------------------------
    n_leaves = len(leaf_names)
    leaf_rows = (n_leaves + 1) // 2
    fig = plt.figure(figsize=(16, 7 + 5 * leaf_rows))
    gs = fig.add_gridspec(
        nrows=2 + leaf_rows,
        ncols=6,
        height_ratios=[3.0, 3.0] + [5.0] * leaf_rows,
        hspace=0.40, wspace=0.55,
    )

    # Row 0 — raw cloud, three orthogonal views (height colormap)
    ax_raw_xz = fig.add_subplot(gs[0, 0:2])
    ax_raw_yz = fig.add_subplot(gs[0, 2:4])
    ax_raw_xy = fig.add_subplot(gs[0, 4:6])
    _scatter_height(ax_raw_xz, plant, view="xz", title="raw · side (x-z)")
    _scatter_height(ax_raw_yz, plant, view="yz", title="raw · side (y-z)")
    _scatter_height(ax_raw_xy, plant, view="xy", title="raw · top-down (x-y)")

    # Row 1 — labelled cloud, same three views
    ax_lbl_xz = fig.add_subplot(gs[1, 0:2])
    ax_lbl_yz = fig.add_subplot(gs[1, 2:4])
    ax_lbl_xy = fig.add_subplot(gs[1, 4:6])
    _scatter_organs(ax_lbl_xz, organs, view="xz", with_legend=True)
    ax_lbl_xz.set_title("segmented · side (x-z)")
    _scatter_organs(ax_lbl_yz, organs, view="yz", with_legend=False)
    ax_lbl_yz.set_title("segmented · side (y-z)")
    _scatter_organs(ax_lbl_xy, organs, view="xy", with_legend=False)
    ax_lbl_xy.set_title("segmented · top-down (x-y)")

    # Row 2+ — per-leaf NURBS overlays (3-D scatter, 2 per row)
    for i, name in enumerate(leaf_names):
        r = 2 + i // 2
        c = (i % 2) * 3
        ax = fig.add_subplot(gs[r, c:c + 3], projection="3d")
        pts = organs[name]
        srf = fits[name]
        rms = (f"RMS {srf['fit_rms']*1000:.1f} mm"
               if srf is not None else "NURBS skipped")
        zspan = float(np.ptp(pts[:, 2]))
        xy_extent = float(np.linalg.norm(np.ptp(pts[:, :2], axis=0)))
        ax_title = (f"{name} · {len(pts):,} pts · z-span {zspan:.1f} cm · "
                    f"xy {xy_extent:.1f} cm · {rms}")
        _leaf_panel(ax, pts, srf, nurbs_mod, ax_title)

    plant_id = f"Plot04/230621 · centre[9]"
    fig.suptitle(
        f"FP4D leaf-segmentation snapshot · {plant_id} · "
        f"{plant.shape[0]:,} pts · {len(leaf_names)} leaf candidates",
        fontsize=12, y=0.995,
    )
    out_png = OUT_DIR / "fp4d_segmentation_overview.png"
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] wrote {out_png}")

    # Also save a per-leaf labelled .xyz for downstream QC
    label_map = {name: idx for idx, name in enumerate(organs.keys())}
    pts_all = np.concatenate(
        [pts for pts in organs.values() if pts.size], axis=0
    )
    labels_all = np.concatenate(
        [np.full(len(pts), label_map[name], dtype=int)
         for name, pts in organs.items() if pts.size]
    )
    xyz_out = OUT_DIR / "labelled.xyz"
    write_labelled_xyz(pts_all, labels_all, str(xyz_out))
    legend = OUT_DIR / "label_legend.txt"
    legend.write_text("\n".join(f"{idx}\t{name}" for name, idx in label_map.items()) + "\n")
    print(f"[viz] wrote {xyz_out} (legend: {legend})")


if __name__ == "__main__":
    main()
