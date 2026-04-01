"""Per-leaf differentiable fitting for wheat scan reconstruction.

Fixes skeletons from MVS-Pheno PLY extractions. Optimizes width profile,
deformations, and gutter depth via Adam to minimize Chamfer distance
against per-leaf PCD point clouds.

No CPlantBox in the loop. No surface reconstruction needed — raw PCDs
serve as Chamfer targets directly.

Usage:
    cd /media/data/Lukas/CPlantBox
    source cpbenv/bin/activate
    python3 -m dart.coupling.experimental.fitting.fit_wheat_leaves \
        /path/to/parameterextraction/wheat1/ \
        -o wheat1_fitted.obj --json wheat1_fitted.json --steps 300
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from ..diff_lofter.deformations import (
    SPLINE_DEFORM_NAMES,
    compute_deformations_spline,
    make_spline_control_points,
    _interp_linear,
)
from ..diff_lofter.frames import compute_binormal_field, compute_tangents
from ..diff_lofter.lofter import compute_arc_fracs, loft_leaf, loft_stem, resample_skeleton
from ..losses.chamfer import chamfer_distance
from .extract_wheat_width_profiles import (
    parse_ply_skeleton,
    orient_skeleton_to_base,
    bridge_skeleton,
)


STEM_DIAMETER = 0.4     # stem cylinder diameter (cm)
STEM_N_SIDES = 8        # stem cross-section vertices
N_WIDTH_CP = 20         # width profile control points (dense for natural taper)
N_DEFORM_CP = 5         # deformation control points per type
N_CROSS = 11            # cross-section vertices (wheat = smoother)
TARGET_SPACING = 0.2    # skeleton resampling spacing (cm)
WIDTH_MIN = 0.05        # minimum width clamp (cm) — near-zero for pointed tips
WIDTH_MAX = 2.5         # maximum width clamp (cm)
DEFORM_CLAMP = 1.0      # deformation CP clamp
REG_DEFORM = 0.005      # deformation regularization weight
REG_WIDTH_SMOOTH = 0.02 # width smoothness (slightly higher for 20 CPs)


def _load_leaf_targets(scan_dir: str) -> list[dict]:
    """Load per-leaf skeletons and PCD targets from scan directory.

    Args:
        scan_dir: Path to parameterextraction/wheatN/ directory.

    Returns:
        List of dicts with 'skeleton' (N,3), 'target_pts' (M,3),
        'leaf_id', 'stem_id' per leaf.
    """
    import open3d as o3d

    scan_path = Path(scan_dir)
    leaf_dir = scan_path / 'leaf'

    # Load stem attachment points
    attach_pcd = o3d.io.read_point_cloud(str(scan_path / 'leaf_stem_close.pcd'))
    attach_pts = np.asarray(attach_pcd.points)

    # Load stem assignments from result.xlsx
    stem_ids: dict[int, int] = {}
    try:
        import pandas as pd
        df = pd.read_excel(str(scan_path / 'result.xlsx'))
        for i, sid in enumerate(df['leaf_of_stem'].dropna().values):
            stem_ids[i] = int(sid)
    except Exception:
        pass

    # Discover leaves
    leaf_ids = sorted(
        int(p.stem.split('_')[-1])
        for p in leaf_dir.glob('leaf_in_*.ply')
    )

    leaves = []
    for lid in leaf_ids:
        ply_path = leaf_dir / f'leaf_in_{lid}.ply'
        pcd_path = leaf_dir / f'leaf{lid}.pcd'
        if not ply_path.exists() or not pcd_path.exists():
            continue

        # Parse and orient skeleton
        skeleton = parse_ply_skeleton(str(ply_path))
        attach_pt = attach_pts[lid] if lid < len(attach_pts) else attach_pts[0]
        skeleton = orient_skeleton_to_base(skeleton, attach_pt)

        # Load target PCD
        leaf_pcd = o3d.io.read_point_cloud(str(pcd_path))
        target_pts = np.asarray(leaf_pcd.points)

        if skeleton.shape[0] < 3 or target_pts.shape[0] < 10:
            continue

        leaves.append({
            'leaf_id': lid,
            'skeleton': skeleton.astype(np.float32),
            'target_pts': target_pts.astype(np.float32),
            'stem_id': stem_ids.get(lid, -1),
            'stem_attachment': attach_pt.astype(np.float32),
        })

    return leaves


def fit_single_leaf(
    skeleton_np: np.ndarray,
    target_pts_np: np.ndarray,
    n_steps: int = 300,
    lr: float = 0.05,
    device: str = 'cuda',
) -> dict:
    """Fit width + deformations + gutter for a single leaf.

    Skeleton is fixed. Optimized parameters:
      - width_cps: (N_WIDTH_CP,) half-width profile control points
      - deform_cps: {name: (N_DEFORM_CP,)} per deformation type
      - gutter_depth: scalar (midrib V-fold depth)

    Args:
        skeleton_np: (N, 3) ordered skeleton from PLY.
        target_pts_np: (M, 3) leaf PCD points.
        n_steps: Adam optimization steps.
        lr: Learning rate.
        device: torch device.

    Returns:
        Dict with fitted params, best loss, and lofted vertices.
    """
    # Resample skeleton to uniform spacing
    skel_t = torch.tensor(skeleton_np, dtype=torch.float32, device=device)
    dummy_w = torch.ones(skel_t.shape[0], device=device) * 0.5
    skel_r, _ = resample_skeleton(skel_t, dummy_w, target_spacing=TARGET_SPACING)
    n_skel = skel_r.shape[0]

    # Skeleton is fixed (detached)
    skel_r = skel_r.detach()

    # Compute frames (fixed)
    tangents = compute_tangents(skel_r).detach()
    binormals = compute_binormal_field(skel_r, tangents).detach()
    arc_fracs = compute_arc_fracs(skel_r).detach()

    # Subsample target if too large
    if target_pts_np.shape[0] > 3000:
        idx = np.random.RandomState(42).choice(
            target_pts_np.shape[0], 3000, replace=False)
        target_pts_np = target_pts_np[idx]
    target_gpu = torch.tensor(target_pts_np, dtype=torch.float32, device=device)

    # Learnable parameters
    # Width profile: 20 CPs — natural leaf shape init
    # narrow base → widen → max at ~40% → gradual taper → pointed tip
    t = torch.linspace(0, 1, N_WIDTH_CP)
    init_w = 0.6 * torch.sin(torch.pi * t) * (1.0 - 0.3 * t)  # peaks ~40%, tapers at tip
    init_w = torch.clamp(init_w, min=WIDTH_MIN)
    width_cps = init_w.to(device=device, dtype=torch.float32).requires_grad_(True)

    deform_cps = make_spline_control_points(
        n_cp=N_DEFORM_CP, device=device, requires_grad=True)

    gutter_depth = torch.tensor(
        0.1, device=device, dtype=torch.float32).requires_grad_(True)

    opt_params = [width_cps, gutter_depth]
    for v in deform_cps.values():
        opt_params.append(v)

    optimizer = torch.optim.Adam(opt_params, lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_steps, eta_min=lr * 0.01)

    best_loss = float('inf')
    best_state: dict | None = None

    for _step in range(n_steps):
        optimizer.zero_grad()

        # Interpolate width profile along arc length
        widths = _interp_linear(arc_fracs, torch.clamp(width_cps, min=WIDTH_MIN))
        widths = torch.clamp(widths, min=WIDTH_MIN, max=WIDTH_MAX)

        # Compute deformations
        deforms = compute_deformations_spline(arc_fracs, deform_cps)

        # Loft — gutter_depth passed as float (not in the grad graph,
        # but we optimize it via finite-difference-like Adam updates
        # since it only affects vertex positions linearly)
        gd = float(torch.clamp(gutter_depth, min=0.0, max=0.3).item())
        verts = loft_leaf(
            skel_r, widths, deforms, tangents, binormals,
            n_cross=N_CROSS, gutter_depth=gd,
        )

        # Chamfer loss
        chamfer = chamfer_distance(verts, target_gpu)

        # Regularization
        reg = torch.tensor(0.0, device=device)
        for v in deform_cps.values():
            reg = reg + REG_DEFORM * (v ** 2).sum()
        # Width smoothness
        w_diff = width_cps[1:] - width_cps[:-1]
        reg = reg + REG_WIDTH_SMOOTH * (w_diff ** 2).sum()

        loss = chamfer + reg
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Clamp parameters
        with torch.no_grad():
            width_cps.clamp_(WIDTH_MIN, WIDTH_MAX)
            gutter_depth.clamp_(0.0, 0.3)
            for v in deform_cps.values():
                v.clamp_(-DEFORM_CLAMP, DEFORM_CLAMP)

        chamfer_val = chamfer.item()
        if chamfer_val < best_loss:
            best_loss = chamfer_val
            best_state = {
                'width_cps': width_cps.detach().cpu().clone(),
                'gutter_depth': gutter_depth.detach().cpu().item(),
                'deform_cps': {
                    name: deform_cps[name].detach().cpu().clone()
                    for name in SPLINE_DEFORM_NAMES
                },
            }

    assert best_state is not None

    # Generate final mesh with best params
    with torch.no_grad():
        widths = _interp_linear(
            arc_fracs,
            torch.clamp(best_state['width_cps'].to(device), min=WIDTH_MIN),
        )
        widths = torch.clamp(widths, min=WIDTH_MIN, max=WIDTH_MAX)
        cp_best = {
            name: best_state['deform_cps'][name].to(device)
            for name in SPLINE_DEFORM_NAMES
        }
        deforms = compute_deformations_spline(arc_fracs, cp_best)
        final_verts = loft_leaf(
            skel_r, widths, deforms, tangents, binormals,
            n_cross=N_CROSS, gutter_depth=best_state['gutter_depth'],
        )

    return {
        'best_loss': best_loss,
        'n_skel': n_skel,
        'vertices': final_verts.detach().cpu().numpy(),
        'params': {
            'width_cps': best_state['width_cps'].tolist(),
            'gutter_depth': best_state['gutter_depth'],
            'deform_cps': {
                name: best_state['deform_cps'][name].tolist()
                for name in SPLINE_DEFORM_NAMES
            },
        },
    }


def _build_stem_organs(scan_dir: str) -> list[dict]:
    """Build stem cylinder organs from per-tiller PCDs.

    Reads stemN.pcd files, sorts points by Z, resamples to uniform
    spacing, and lofts thin cylinders via loft_stem().

    Args:
        scan_dir: Path to parameterextraction/wheatN/ directory.

    Returns:
        List of organ dicts ready for export_obj.
    """
    import open3d as o3d

    scan_path = Path(scan_dir)
    stem_dir = scan_path / 'stem'
    organs = []

    for stem_file in sorted(stem_dir.glob('stem[0-9]*.pcd')):
        sid = stem_file.stem  # e.g. "stem0"
        pcd = o3d.io.read_point_cloud(str(stem_file))
        pts = np.asarray(pcd.points)
        if pts.shape[0] < 2:
            continue

        # Sort by Z (ascending = bottom to top in raw inverted coords)
        order = np.argsort(pts[:, 2])
        pts = pts[order].astype(np.float32)

        # Resample to uniform spacing
        skel_t = torch.tensor(pts, dtype=torch.float32)
        diam = torch.ones(skel_t.shape[0]) * STEM_DIAMETER
        skel_r, diam_r = resample_skeleton(skel_t, diam, target_spacing=0.3)

        if skel_r.shape[0] < 2:
            continue

        n_rows = skel_r.shape[0]
        tangents = compute_tangents(skel_r)
        binormals = compute_binormal_field(skel_r, tangents)
        verts = loft_stem(skel_r, diam_r, tangents, binormals, n_sides=STEM_N_SIDES)

        organs.append({
            'name': sid,
            'vertices': verts.detach().numpy(),
            'n_rows': n_rows,
            'n_cross': STEM_N_SIDES,
        })
        print(f"  Stem {sid}: {n_rows} nodes, "
              f"Z=[{pts[:, 2].min():.1f}, {pts[:, 2].max():.1f}]",
              file=sys.stderr)

    return organs


def fit_all_leaves(
    scan_dir: str,
    output_obj: str,
    output_json: str | None = None,
    n_steps: int = 300,
    lr: float = 0.05,
    device: str = 'cuda',
) -> dict:
    """Fit all leaves in a wheat scan and export mesh.

    Args:
        scan_dir: Path to parameterextraction/wheatN/ directory.
        output_obj: Output OBJ path.
        output_json: Optional output JSON with fitted parameters.
        n_steps: Adam steps per leaf.
        lr: Learning rate.
        device: torch device.

    Returns:
        Summary dict.
    """
    from ..diff_lofter.export import export_obj

    print(f"Loading leaf data from {scan_dir}...", file=sys.stderr)
    leaves = _load_leaf_targets(scan_dir)
    print(f"  Found {len(leaves)} leaves", file=sys.stderr)

    organs = []
    all_params = []

    # Build stem cylinders first
    print(f"\nBuilding stem cylinders...", file=sys.stderr)
    stem_organs = _build_stem_organs(scan_dir)
    organs.extend(stem_organs)
    print(f"  Added {len(stem_organs)} stems", file=sys.stderr)

    for leaf in leaves:
        lid = leaf['leaf_id']
        n_target = leaf['target_pts'].shape[0]
        print(f"\nFitting leaf {lid} ({n_target} target pts)...",
              file=sys.stderr)

        result = fit_single_leaf(
            leaf['skeleton'],
            leaf['target_pts'],
            n_steps=n_steps,
            lr=lr,
            device=device,
        )

        n_skel = result['n_skel']
        print(f"  -> chamfer={result['best_loss']:.4f} cm, "
              f"n_skel={n_skel}, gutter={result['params']['gutter_depth']:.3f}, "
              f"widths={[round(w, 2) for w in result['params']['width_cps']]}",
              file=sys.stderr)

        organs.append({
            'name': f'leaf_{lid}',
            'vertices': result['vertices'],
            'n_rows': n_skel,
            'n_cross': N_CROSS,
        })

        all_params.append({
            'leaf_id': lid,
            'stem_id': leaf['stem_id'],
            'best_loss': result['best_loss'],
            'n_skel': n_skel,
            **result['params'],
        })

    # Export OBJ
    export_obj(output_obj, organs)

    total_verts = sum(o['vertices'].shape[0] for o in organs)
    total_tris = sum((o['n_rows'] - 1) * 2 * (o['n_cross'] - 1) for o in organs)
    print(f"\nExported: {output_obj}", file=sys.stderr)
    print(f"  Organs: {len(organs)}, Vertices: {total_verts}, "
          f"Triangles: {total_tris}", file=sys.stderr)

    losses = [p['best_loss'] for p in all_params]
    print(f"  Chamfer: mean={np.mean(losses):.4f}, "
          f"median={np.median(losses):.4f}, max={np.max(losses):.4f} cm",
          file=sys.stderr)

    summary = {
        'scan_dir': str(scan_dir),
        'n_leaves': len(organs),
        'total_vertices': total_verts,
        'total_triangles': total_tris,
        'mean_chamfer': float(np.mean(losses)),
        'per_leaf': all_params,
    }

    if output_json:
        with open(output_json, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"  Params: {output_json}", file=sys.stderr)

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Fit wheat leaf geometry to scan point clouds")
    parser.add_argument("scan_dir",
                        help="Path to parameterextraction/wheatN/ directory")
    parser.add_argument("-o", "--output", default="wheat_fitted.obj",
                        help="Output OBJ path")
    parser.add_argument("--json", default=None,
                        help="Output JSON with fitted params")
    parser.add_argument("--steps", type=int, default=300,
                        help="Adam steps per leaf (default: 300)")
    parser.add_argument("--lr", type=float, default=0.05,
                        help="Learning rate (default: 0.05)")
    parser.add_argument("--device", default=None,
                        help="torch device (default: cuda if available)")
    args = parser.parse_args()

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')

    fit_all_leaves(
        args.scan_dir,
        args.output,
        output_json=args.json,
        n_steps=args.steps,
        lr=args.lr,
        device=device,
    )


if __name__ == '__main__':
    main()
