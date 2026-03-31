"""Direct differentiable skeleton fitting — bypass CPlantBox entirely.

Each leaf is parameterized as:
  - insertion_height: Z-coordinate where leaf meets stem
  - theta: insertion angle from vertical (radians)
  - azimuth: rotation around stem (radians)
  - skeleton_cp: (N_SKEL_CP, 3) control points in local leaf frame
  - width_cp: (N_WIDTH_CP,) width profile control points
  - deformation CPs: from spline deformation model

All parameters are gradient-optimized via Adam. No CPlantBox in the loop.

Usage:
    from dart.coupling.experimental.fitting.direct_skeleton import fit_plant_direct
    result = fit_plant_direct(target_points, n_leaves=11)
"""

import math
import sys

import numpy as np
import torch

from ..diff_lofter.deformations import (
    compute_deformations_spline,
    make_spline_control_points,
    _interp_linear,
    SPLINE_DEFORM_NAMES,
)
from ..diff_lofter.frames import compute_binormal_field, compute_tangents
from ..diff_lofter.lofter import compute_arc_fracs, loft_leaf
from ..losses.chamfer import chamfer_distance


N_CURV_CP = 5        # curvature profile control points per leaf
N_WIDTH_CP = 5       # width profile control points per leaf
N_CROSS = 7          # cross-section vertices
N_DENSE = 50         # dense skeleton points after interpolation

# Regularization weights (light — structure does the heavy lifting)
REG_DEFORM = 0.01    # penalize deformation magnitudes
REG_SPACING = 0.01   # monotonic insertion heights


def _cubic_interp(cp: torch.Tensor, n_out: int) -> torch.Tensor:
    """Differentiable cubic B-spline interpolation of control points.

    Args:
        cp: (K, D) control points.
        n_out: number of output points.

    Returns:
        (n_out, D) interpolated points.
    """
    k = cp.shape[0]
    device = cp.device
    dtype = cp.dtype

    # Parameter values for output points
    t_out = torch.linspace(0.0, 1.0, n_out, device=device, dtype=dtype)

    # Map to continuous index
    t_idx = t_out * (k - 1)
    t_idx = torch.clamp(t_idx, 0.0, k - 1.0 - 1e-6)
    idx_lo = t_idx.long()
    idx_hi = (idx_lo + 1).clamp(max=k - 1)
    frac = (t_idx - idx_lo.float()).unsqueeze(1)  # (n_out, 1)

    # Hermite-style interpolation for smoothness
    # Use Catmull-Rom: needs idx-1 and idx+2
    idx_m1 = (idx_lo - 1).clamp(min=0)
    idx_p2 = (idx_hi + 1).clamp(max=k - 1)

    p0 = cp[idx_m1]  # (n_out, D)
    p1 = cp[idx_lo]
    p2 = cp[idx_hi]
    p3 = cp[idx_p2]

    # Catmull-Rom spline
    t = frac
    t2 = t * t
    t3 = t2 * t

    out = 0.5 * (
        (2.0 * p1) +
        (-p0 + p2) * t +
        (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2 +
        (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
    )
    return out


def _build_skeleton(
    insertion_height: torch.Tensor,
    theta: torch.Tensor,
    azimuth: torch.Tensor,
    local_cp: torch.Tensor,
    n_dense: int = N_DENSE,
) -> torch.Tensor:
    """Build a dense 3D skeleton from leaf parameters.

    The skeleton is constructed by:
    1. Interpolating local control points to get a smooth local curve
    2. Rotating by theta (from vertical) and azimuth (around Z)
    3. Translating to insertion point on stem

    Args:
        insertion_height: scalar, Z-coordinate of leaf base
        theta: scalar, angle from vertical (radians)
        azimuth: scalar, rotation around Z (radians)
        local_cp: (N_SKEL_CP, 3) control points in local frame
            x=along leaf, y=lateral, z=vertical deviation
        n_dense: number of output skeleton points

    Returns:
        (n_dense, 3) world-space skeleton points
    """
    # Interpolate control points to dense skeleton
    local_dense = _cubic_interp(local_cp, n_dense)  # (N, 3)

    # Build rotation matrix: first rotate by theta around Y, then by azimuth around Z
    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)
    cos_a = torch.cos(azimuth)
    sin_a = torch.sin(azimuth)

    # Ry(theta) @ Rz(azimuth) — leaf grows outward at angle theta
    # Local x → outward direction, local z → upward deviation
    r00 = cos_a * cos_t
    r01 = -sin_a
    r02 = cos_a * sin_t
    r10 = sin_a * cos_t
    r11 = cos_a
    r12 = sin_a * sin_t
    r20 = -sin_t
    r21 = torch.tensor(0.0, device=theta.device)
    r22 = cos_t

    # Apply rotation
    x = local_dense[:, 0]
    y = local_dense[:, 1]
    z = local_dense[:, 2]

    world_x = r00 * x + r01 * y + r02 * z
    world_y = r10 * x + r11 * y + r12 * z
    world_z = r20 * x + r21 * y + r22 * z + insertion_height

    return torch.stack([world_x, world_y, world_z], dim=1)


def _init_leaf_params(
    n_leaves: int,
    max_height: float,
    device: str = 'cpu',
):
    """Initialize leaf parameters with reasonable maize defaults.

    Returns dict of parameter tensors, all requiring gradients.
    """
    params = {}

    # Insertion heights: evenly spaced along stem
    heights = torch.linspace(5.0, max_height * 0.85, n_leaves,
                             device=device, dtype=torch.float32)
    params['insertion_heights'] = heights.clone().requires_grad_(True)

    # Theta: lower leaves more horizontal (0.8 rad ≈ 45°), upper more vertical (0.4 rad)
    thetas = torch.linspace(0.8, 0.4, n_leaves, device=device, dtype=torch.float32)
    params['thetas'] = thetas.clone().requires_grad_(True)

    # Azimuth: alternating distichous (180° apart) with slight jitter
    azimuths = torch.zeros(n_leaves, device=device, dtype=torch.float32)
    for i in range(n_leaves):
        azimuths[i] = (i % 2) * math.pi + 0.07 * i
    params['azimuths'] = azimuths.clone().requires_grad_(True)

    # Local skeleton control points per leaf
    # Default: straight leaf extending in +x, slight droop in -z at tip
    all_cp = []
    for i in range(n_leaves):
        # Leaf length decreases for lower/upper leaves, longest in middle
        pos_frac = i / max(n_leaves - 1, 1)
        length = 40.0 + 35.0 * math.sin(math.pi * pos_frac)  # 40-75 cm

        cp = torch.zeros(N_SKEL_CP, 3, device=device, dtype=torch.float32)
        for j in range(N_SKEL_CP):
            t = j / (N_SKEL_CP - 1)
            cp[j, 0] = t * length                    # along leaf
            cp[j, 1] = 0.0                            # lateral (zero for now)
            cp[j, 2] = -0.3 * (t ** 2) * length       # gentle droop
        all_cp.append(cp)

    params['skeleton_cps'] = torch.stack(all_cp).requires_grad_(True)  # (n_leaves, N_SKEL_CP, 3)

    # Width profiles per leaf: start narrow, widen, taper at tip
    all_widths = []
    for i in range(n_leaves):
        pos_frac = i / max(n_leaves - 1, 1)
        max_w = 2.0 + 4.0 * math.sin(math.pi * pos_frac)  # 2-6 cm
        w = torch.tensor(
            [max_w * 0.3, max_w * 0.8, max_w, max_w * 0.7, max_w * 0.15],
            device=device, dtype=torch.float32,
        )
        all_widths.append(w)

    params['width_cps'] = torch.stack(all_widths).requires_grad_(True)  # (n_leaves, N_WIDTH_CP)

    # Deformation control points per leaf (init zero)
    deform_cps = {}
    for name in SPLINE_DEFORM_NAMES:
        deform_cps[name] = torch.zeros(
            n_leaves, 5, device=device, dtype=torch.float32, requires_grad=True
        )
    params['deform_cps'] = deform_cps

    return params


def _compute_regularization(params, n_leaves):
    """Compute regularization losses for realistic geometry.

    Light priors — only prevent degenerate shapes:
    - Smooth skeletons (penalize sharp bends, not gentle curves)
    - Skeleton progresses forward (no loops)
    - Tip width small (leaves taper at tip)
    - Positive widths
    - Monotonic insertion heights
    - Reasonable leaf lengths
    - Small deformations (keep leaf surfaces clean)
    """
    device = params['insertion_heights'].device
    reg = torch.tensor(0.0, device=device)

    skel_cps = params['skeleton_cps']  # (n_leaves, N_SKEL_CP, 3)
    w_cps = params['width_cps']  # (n_leaves, N_WIDTH_CP)

    # 1. Skeleton smoothness: penalize second derivative (sharp bends only)
    d2 = skel_cps[:, 2:] - 2 * skel_cps[:, 1:-1] + skel_cps[:, :-2]
    reg = reg + REG_SMOOTH * (d2 ** 2).sum()

    # 2. Skeleton must progress forward: x should be monotonically increasing
    dx = skel_cps[:, 1:, 0] - skel_cps[:, :-1, 0]
    reg = reg + 0.1 * (torch.clamp(-dx + 1.0, min=0.0) ** 2).sum()

    # 3. Tip width should be small (< 1.5 cm) — leaves taper
    reg = reg + REG_TIP * (torch.clamp(w_cps[:, -1] - 1.5, min=0.0) ** 2).sum()

    # 4. Positive widths (hard constraint)
    reg = reg + 1.0 * (torch.clamp(-w_cps + 0.1, min=0.0) ** 2).sum()

    # 5. Insertion heights monotonically increasing
    heights = params['insertion_heights']
    h_diff = heights[1:] - heights[:-1]
    reg = reg + REG_SPACING * (torch.clamp(-h_diff + 2.0, min=0.0) ** 2).sum()

    # 6. Leaf lengths reasonable (20-90 cm)
    for i in range(n_leaves):
        cp = skel_cps[i]
        diffs = cp[1:] - cp[:-1]
        length = torch.linalg.norm(diffs, dim=1).sum()
        reg = reg + REG_LENGTH * (torch.clamp(20.0 - length, min=0.0) ** 2)
        reg = reg + REG_LENGTH * (torch.clamp(length - 90.0, min=0.0) ** 2)

    # 7. Penalize deformation magnitudes — keep surfaces clean
    for name in SPLINE_DEFORM_NAMES:
        reg = reg + REG_DEFORM * (params['deform_cps'][name] ** 2).sum()

    return reg


def _forward_plant(params, n_leaves):
    """Forward pass: params → all leaf vertices."""
    all_verts = []

    for i in range(n_leaves):
        # Build skeleton
        skeleton = _build_skeleton(
            params['insertion_heights'][i],
            params['thetas'][i],
            params['azimuths'][i],
            params['skeleton_cps'][i],
            n_dense=N_DENSE,
        )

        # Compute frames
        tangents = compute_tangents(skeleton)
        binormals = compute_binormal_field(skeleton, tangents)
        arc_fracs = compute_arc_fracs(skeleton)

        # Width profile
        w_cp = params['width_cps'][i]
        widths = _interp_linear(arc_fracs, torch.clamp(w_cp, min=0.1))

        # Deformations
        cp = {name: params['deform_cps'][name][i] for name in SPLINE_DEFORM_NAMES}
        deforms = compute_deformations_spline(arc_fracs, cp)

        # Loft
        verts = loft_leaf(skeleton, widths, deforms, tangents, binormals,
                          n_cross=N_CROSS)
        all_verts.append(verts)

    return torch.cat(all_verts, dim=0)


def fit_plant_direct(
    target_points: np.ndarray,
    n_leaves: int = 11,
    n_steps: int = 500,
    lr: float = 0.1,
    device: str = 'cuda',
    seed: int = 42,
    align_rotation: bool = True,
) -> dict:
    """Fit leaf geometry directly to target point cloud via gradient descent.

    No CPlantBox involved — pure differentiable optimization.

    Args:
        target_points: (N, 3) numpy array, target point cloud in cm
        n_leaves: number of leaves to fit
        n_steps: Adam optimization steps
        lr: learning rate (0.1 works well for this scale)
        device: torch device
        seed: random seed
        align_rotation: try Z-axis rotation alignment first

    Returns:
        dict with fitted parameters, loss history, and exportable data
    """
    torch.manual_seed(seed)

    # Estimate plant height from target
    z_max = float(target_points[:, 2].max())
    print(f"Target: {len(target_points)} pts, height={z_max:.1f} cm", file=sys.stderr)

    # Subsample target
    if len(target_points) > 8000:
        idx = np.random.RandomState(seed).choice(len(target_points), 8000, replace=False)
        target_sub = target_points[idx]
    else:
        target_sub = target_points

    target_gpu = torch.tensor(target_sub, dtype=torch.float32, device=device)

    # Initialize parameters
    params = _init_leaf_params(n_leaves, max_height=z_max, device=device)

    # Rotation alignment
    if align_rotation:
        with torch.no_grad():
            init_verts = _forward_plant(params, n_leaves)
            init_pts = init_verts.cpu().numpy()
        if len(init_pts) > 5000:
            init_pts = init_pts[np.random.RandomState(seed).choice(len(init_pts), 5000, replace=False)]
        from ..targets.pointcloud_loader import align_rotation_z
        target_sub, best_angle = align_rotation_z(target_sub, init_pts, n_angles=72)
        target_gpu = torch.tensor(target_sub, dtype=torch.float32, device=device)
        print(f"Rotation alignment: {best_angle:.0f} deg", file=sys.stderr)

    # Collect all learnable parameters
    opt_params = [
        params['insertion_heights'],
        params['thetas'],
        params['azimuths'],
        params['skeleton_cps'],
        params['width_cps'],
    ]
    for t in params['deform_cps'].values():
        opt_params.append(t)

    optimizer = torch.optim.Adam(opt_params, lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_steps, eta_min=lr * 0.01)

    loss_history = []
    best_loss = float('inf')
    best_state = None

    for step in range(n_steps):
        optimizer.zero_grad()

        gen_verts = _forward_plant(params, n_leaves)
        chamfer = chamfer_distance(gen_verts, target_gpu)
        reg = _compute_regularization(params, n_leaves)
        loss = chamfer + reg

        loss.backward()
        optimizer.step()
        scheduler.step()

        chamfer_val = chamfer.item()
        loss_history.append(chamfer_val)

        if chamfer_val < best_loss:
            best_loss = chamfer_val
            best_state = {
                'insertion_heights': params['insertion_heights'].detach().cpu().clone(),
                'thetas': params['thetas'].detach().cpu().clone(),
                'azimuths': params['azimuths'].detach().cpu().clone(),
                'skeleton_cps': params['skeleton_cps'].detach().cpu().clone(),
                'width_cps': params['width_cps'].detach().cpu().clone(),
                'deform_cps': {
                    name: params['deform_cps'][name].detach().cpu().clone()
                    for name in SPLINE_DEFORM_NAMES
                },
            }

        if step % 50 == 0 or step == n_steps - 1:
            print(f"  step {step:4d}: chamfer={chamfer_val:.3f} cm  "
                  f"reg={reg.item():.3f}  lr={scheduler.get_last_lr()[0]:.5f}",
                  file=sys.stderr)

    print(f"Best Chamfer: {best_loss:.3f} cm", file=sys.stderr)

    return {
        'best_loss': best_loss,
        'loss_history': loss_history,
        'n_leaves': n_leaves,
        'n_steps': n_steps,
        'params': best_state,
    }


def export_fitted_mesh(result: dict, output_path: str):
    """Export the fitted plant as an OBJ mesh.

    Args:
        result: output from fit_plant_direct()
        output_path: path to write .obj file
    """
    import trimesh

    state = result['params']
    n_leaves = result['n_leaves']
    device = 'cpu'

    all_verts = []
    all_faces = []
    vert_offset = 0

    for i in range(n_leaves):
        skeleton = _build_skeleton(
            state['insertion_heights'][i],
            state['thetas'][i],
            state['azimuths'][i],
            state['skeleton_cps'][i],
            n_dense=N_DENSE,
        )

        tangents = compute_tangents(skeleton)
        binormals = compute_binormal_field(skeleton, tangents)
        arc_fracs = compute_arc_fracs(skeleton)

        w_cp = state['width_cps'][i]
        widths = _interp_linear(arc_fracs, torch.clamp(w_cp, min=0.1))

        cp = {name: state['deform_cps'][name][i] for name in SPLINE_DEFORM_NAMES}
        deforms = compute_deformations_spline(arc_fracs, cp)

        verts = loft_leaf(skeleton, widths, deforms, tangents, binormals,
                          n_cross=N_CROSS)
        verts_np = verts.detach().numpy()

        n_skel = N_DENSE
        faces = []
        for j in range(n_skel - 1):
            for k in range(N_CROSS - 1):
                v0 = j * N_CROSS + k
                v1 = v0 + 1
                v2 = (j + 1) * N_CROSS + k
                v3 = v2 + 1
                faces.append([v0 + vert_offset, v2 + vert_offset, v1 + vert_offset])
                faces.append([v1 + vert_offset, v2 + vert_offset, v3 + vert_offset])

        all_verts.append(verts_np)
        all_faces.extend(faces)
        vert_offset += len(verts_np)

    mesh = trimesh.Trimesh(
        vertices=np.concatenate(all_verts),
        faces=np.array(all_faces),
    )
    mesh.export(output_path)
    return mesh


def save_result(result: dict, path: str):
    """Save fitting result to JSON (tensors converted to lists)."""
    import json

    out = {
        'best_loss': result['best_loss'],
        'loss_history': result['loss_history'],
        'n_leaves': result['n_leaves'],
        'n_steps': result['n_steps'],
        'params': {
            'insertion_heights': result['params']['insertion_heights'].tolist(),
            'thetas': result['params']['thetas'].tolist(),
            'azimuths': result['params']['azimuths'].tolist(),
            'skeleton_cps': result['params']['skeleton_cps'].tolist(),
            'width_cps': result['params']['width_cps'].tolist(),
            'deform_cps': {
                name: result['params']['deform_cps'][name].tolist()
                for name in SPLINE_DEFORM_NAMES
            },
        },
    }

    with open(path, 'w') as f:
        json.dump(out, f, indent=2)


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m dart.coupling.experimental.fitting.direct_skeleton <target.stl> [n_leaves] [n_steps]")
        sys.exit(1)

    target_path = sys.argv[1]
    n_leaves = int(sys.argv[2]) if len(sys.argv) > 2 else 11
    n_steps = int(sys.argv[3]) if len(sys.argv) > 3 else 500

    from ..targets.pointcloud_loader import load_pointcloud
    target_pts, _ = load_pointcloud(target_path, n_points=10000)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    result = fit_plant_direct(target_pts, n_leaves=n_leaves, n_steps=n_steps, device=device)

    out_base = target_path.rsplit('.', 1)[0]
    export_fitted_mesh(result, f'{out_base}_direct_fit.obj')
    save_result(result, f'{out_base}_direct_fit.json')
    print(f"Exported: {out_base}_direct_fit.obj + .json")
