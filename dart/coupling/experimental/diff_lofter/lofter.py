"""Differentiable leaf lofter: skeleton + widths + deformations -> vertex positions.

Ports the cross-section sweep from g1_to_g3.py _loft_leaf() (lines 602-806)
to PyTorch. All operations support autograd -- no numpy in the forward pass.

Output: vertex positions (V, 3) suitable for Chamfer distance loss.
Triangle indices, UV coordinates, segment mapping, and Laplacian smoothing
are NOT needed and are omitted.
"""

import math
import torch

# Preprocessing constants (match production lofter)
MIN_SKELETON_SPACING = 0.2   # cm — resample denser skeletons to this
MIN_WIDTH = 0.15             # cm — clamp widths below this (prevents zero-area tips)


def resample_skeleton(
    skeleton: torch.Tensor,
    widths: torch.Tensor,
    target_spacing: float = MIN_SKELETON_SPACING,
    min_width: float = MIN_WIDTH,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Resample skeleton to uniform spacing and clamp minimum width.

    Prevents degenerate triangles by ensuring no two skeleton points are
    closer than target_spacing and no width is below min_width.

    Differentiable (uses linear interpolation).

    Args:
        skeleton: (N, 3) skeleton points.
        widths: (N,) widths per point.
        target_spacing: minimum distance between points (cm).
        min_width: minimum width (cm).

    Returns:
        (skeleton_resampled, widths_resampled) with uniform spacing.
    """
    device = skeleton.device
    dtype = skeleton.dtype

    # Compute cumulative arc length
    diffs = skeleton[1:] - skeleton[:-1]
    seg_lens = torch.linalg.norm(diffs, dim=1)
    cum_len = torch.cat([torch.zeros(1, device=device, dtype=dtype),
                         torch.cumsum(seg_lens, dim=0)])
    total_len = cum_len[-1]

    if total_len < target_spacing * 2:
        # Too short to resample — just clamp widths
        return skeleton, torch.clamp(widths, min=min_width)

    # Number of resampled points
    n_out = max(int(total_len / target_spacing) + 1, 3)
    t_out = torch.linspace(0, total_len, n_out, device=device, dtype=dtype)

    # Truncate at min_width: find where width drops below threshold from the end
    # and stop the skeleton there (pointed tip)
    # This is done after interpolation

    # Linear interpolation of skeleton and widths at new arc positions
    # Find interval for each t_out
    skel_out = torch.zeros(n_out, 3, device=device, dtype=dtype)
    w_out = torch.zeros(n_out, device=device, dtype=dtype)

    for j in range(n_out):
        t = t_out[j]
        # Binary search for interval (use searchsorted)
        idx = torch.searchsorted(cum_len, t).clamp(1, len(cum_len) - 1)
        i0 = idx - 1
        i1 = idx
        seg_len = cum_len[i1] - cum_len[i0]
        frac = (t - cum_len[i0]) / seg_len.clamp(min=1e-12)
        frac = frac.clamp(0, 1)

        skel_out[j] = skeleton[i0] * (1 - frac) + skeleton[i1] * frac
        w_out[j] = widths[i0] * (1 - frac) + widths[i1] * frac

    # Clamp minimum width
    w_out = torch.clamp(w_out, min=min_width)

    # Truncate skeleton where original width was near-zero (tip region)
    # Find last point where interpolated width > min_width * 1.5
    above_min = w_out > min_width * 1.5
    if above_min.any():
        last_good = above_min.nonzero()[-1].item()
        # Keep one extra point past last_good for the tip
        n_keep = min(last_good + 2, n_out)
        skel_out = skel_out[:n_keep]
        w_out = w_out[:n_keep]

    return skel_out, w_out


def _rodrigues_rotate(v: torch.Tensor, k: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """Rodrigues rotation: rotate v around unit axis k by angle theta.

    Fully vectorized and differentiable.

    Args:
        v: (N, 3) vectors to rotate.
        k: (N, 3) unit rotation axes.
        theta: (N,) rotation angles in radians.

    Returns:
        (N, 3) rotated vectors.
    """
    cos_t = torch.cos(theta).unsqueeze(1)  # (N, 1)
    sin_t = torch.sin(theta).unsqueeze(1)  # (N, 1)
    dot_kv = (k * v).sum(dim=1, keepdim=True)  # (N, 1)
    cross_kv = torch.linalg.cross(k, v)  # (N, 3)
    return v * cos_t + cross_kv * sin_t + k * dot_kv * (1.0 - cos_t)


def compute_arc_fracs(skeleton: torch.Tensor) -> torch.Tensor:
    """Compute normalized cumulative arc-length fractions.

    Args:
        skeleton: (N, 3) ordered 3D points.

    Returns:
        (N,) values in [0, 1].
    """
    diffs = skeleton[1:] - skeleton[:-1]  # (N-1, 3)
    seg_lengths = torch.linalg.norm(diffs, dim=1)  # (N-1,)
    cumulative = torch.cat([
        torch.zeros(1, device=skeleton.device, dtype=skeleton.dtype),
        torch.cumsum(seg_lengths, dim=0),
    ])
    total = cumulative[-1]
    if total < 1e-12:
        return torch.linspace(0.0, 1.0, skeleton.shape[0],
                              device=skeleton.device, dtype=skeleton.dtype)
    return cumulative / total


def loft_leaf(
    skeleton: torch.Tensor,
    widths: torch.Tensor,
    deformations: dict[str, torch.Tensor],
    tangents: torch.Tensor,
    binormals: torch.Tensor,
    n_cross: int = 7,
    gutter_depth: float = 0.0,
) -> torch.Tensor:
    """Differentiable leaf lofting: skeleton + widths + deformations -> vertices.

    Cross-section sweep logic from g1_to_g3.py, fully vectorized over both
    skeleton nodes (N) and cross-section positions (C) using broadcasting.

    Args:
        skeleton: (N, 3) node positions.
        widths: (N,) half-widths per node (matching g1_to_g3 convention where
            widths passed to _loft_leaf are full-widths, but the cross-section
            uses frac * w where frac ranges [-0.5, 0.5]).
        deformations: Dict from compute_deformations() with keys:
            wave_normal, wave_lateral, twist, curl, edge_ruffle, fold.
            Each value is (N,).
        tangents: (N, 3) unit tangent vectors.
        binormals: (N, 3) unit binormal vectors.
        n_cross: Number of vertices across the width (default 7).
        gutter_depth: Midrib channel depth in cm (0 = flat).

    Returns:
        (N * n_cross, 3) vertex positions.
    """
    device = skeleton.device
    dtype = skeleton.dtype
    n = skeleton.shape[0]

    # Cross-section fractions: [-0.5, ..., 0.5]
    cross_fracs = torch.linspace(-0.5, 0.5, n_cross, device=device, dtype=dtype)  # (C,)

    # Gutter profile: parabolic, 1 at center, 0 at edges
    cross_gutter = 1.0 - (2.0 * cross_fracs) ** 2  # (C,)

    # Compute normals: cross(tangent, binormal) per node
    normals = torch.linalg.cross(tangents, binormals)  # (N, 3)
    nm_len = torch.linalg.norm(normals, dim=1, keepdim=True)
    normals = normals / torch.clamp(nm_len, min=1e-12)

    # --- Apply twist via Rodrigues rotation ---
    twist_angles = deformations["twist"]  # (N,)
    # Only rotate when twist is non-negligible (always computed for grad flow)
    # Rotate binormal and normal around tangent by twist angle
    bn_twisted = _rodrigues_rotate(binormals, tangents, twist_angles)  # (N, 3)
    nm_twisted = _rodrigues_rotate(normals, tangents, twist_angles)  # (N, 3)
    # Re-normalize for safety
    bn_twisted = bn_twisted / torch.clamp(
        torch.linalg.norm(bn_twisted, dim=1, keepdim=True), min=1e-12)
    nm_twisted = nm_twisted / torch.clamp(
        torch.linalg.norm(nm_twisted, dim=1, keepdim=True), min=1e-12)

    # --- Center displacement: wave_normal + wave_lateral ---
    wave_n = deformations["wave_normal"]  # (N,)
    wave_l = deformations["wave_lateral"]  # (N,)
    center = (
        skeleton
        + wave_n.unsqueeze(1) * nm_twisted
        + wave_l.unsqueeze(1) * bn_twisted
    )  # (N, 3)

    # --- Width-proportional fade factor ---
    max_w = widths.max()
    max_w = torch.clamp(max_w, min=0.01)
    w_fade = widths / max_w  # (N,)

    # --- Vectorized cross-section sweep ---
    # Expand dimensions for broadcasting:
    #   center:     (N, 1, 3)
    #   bn_twisted: (N, 1, 3)
    #   nm_twisted: (N, 1, 3)
    #   widths:     (N, 1)
    #   cross_fracs:(1, C)
    center_exp = center.unsqueeze(1)  # (N, 1, 3)
    bn_exp = bn_twisted.unsqueeze(1)  # (N, 1, 3)
    nm_exp = nm_twisted.unsqueeze(1)  # (N, 1, 3)
    w_exp = widths.unsqueeze(1)  # (N, 1)
    cf_exp = cross_fracs.unsqueeze(0)  # (1, C)
    w_fade_exp = w_fade.unsqueeze(1)  # (N, 1)

    # Lateral displacement: frac * w * binormal
    lateral = cf_exp.unsqueeze(2) * w_exp.unsqueeze(2) * bn_exp  # (N, C, 3)

    # Gutter offset: -gutter_depth * profile * normal * w_fade
    cg_exp = cross_gutter.unsqueeze(0)  # (1, C)
    gutter_offset = (
        -gutter_depth
        * cg_exp.unsqueeze(2)
        * nm_exp
        * w_fade_exp.unsqueeze(2)
    )  # (N, C, 3)

    # Curl: (2 * frac) * curl_factor * normal * w_fade
    curl_factors = deformations["curl"]  # (N,)
    curl_offset = (
        (2.0 * cf_exp).unsqueeze(2)
        * curl_factors.unsqueeze(1).unsqueeze(2)
        * nm_exp
        * w_fade_exp.unsqueeze(2)
    )  # (N, C, 3)

    # Edge ruffle: edge_frac^2 * ruffle_base * normal * w_fade * sign(frac)
    # edge_frac = (2*|frac|)^2, left edge gets flipped sign
    abs_cf = torch.abs(cf_exp)  # (1, C)
    edge_frac = (2.0 * abs_cf) ** 2  # (1, C) - 0 at center, 1 at edges
    ruffle_base = deformations["edge_ruffle"]  # (N,)
    # Sign: left edge (frac < 0) flips the ruffle
    ruffle_sign = torch.sign(cross_fracs)  # (C,)
    # Handle exact zero (midrib): ruffle is zero there anyway via edge_frac
    ruffle_sign = torch.where(ruffle_sign == 0, torch.ones_like(ruffle_sign), ruffle_sign)
    ruffle_offset = (
        edge_frac.unsqueeze(2)
        * ruffle_base.unsqueeze(1).unsqueeze(2)
        * nm_exp
        * w_fade_exp.unsqueeze(2)
        * ruffle_sign.unsqueeze(0).unsqueeze(2)
    )  # (N, C, 3)

    # Fold: sin(pi * |2*frac|) * fold_factor * normal * w_fade
    fold_profile = torch.sin(math.pi * torch.abs(2.0 * cf_exp))  # (1, C)
    fold_factors = deformations["fold"]  # (N,)
    fold_offset = (
        fold_profile.unsqueeze(2)
        * fold_factors.unsqueeze(1).unsqueeze(2)
        * nm_exp
        * w_fade_exp.unsqueeze(2)
    )  # (N, C, 3)

    # --- Assemble vertices ---
    vertices = (
        center_exp
        + lateral
        + gutter_offset
        + curl_offset
        + ruffle_offset
        + fold_offset
    )  # (N, C, 3)

    # Reshape to (N*C, 3)
    return vertices.reshape(-1, 3)


def loft_stem(
    skeleton: torch.Tensor,
    widths: torch.Tensor,
    tangents: torch.Tensor,
    binormals: torch.Tensor,
    n_sides: int = 8,
) -> torch.Tensor:
    """Differentiable stem lofting: skeleton + widths -> cylindrical tube vertices.

    Creates rings of vertices at each skeleton point using circular cross-sections.

    Args:
        skeleton: (N, 3) node positions.
        widths: (N,) diameters per node (radius = width / 2).
        tangents: (N, 3) unit tangent vectors.
        binormals: (N, 3) unit binormal vectors.
        n_sides: Number of vertices per ring (default 8).

    Returns:
        (N * n_sides, 3) vertex positions.
    """
    device = skeleton.device
    dtype = skeleton.dtype
    n = skeleton.shape[0]

    # Compute normals from tangent x binormal
    normals = torch.linalg.cross(tangents, binormals)
    nm_len = torch.linalg.norm(normals, dim=1, keepdim=True)
    normals = normals / torch.clamp(nm_len, min=1e-12)

    # Angles around the tube
    angles = torch.linspace(0, 2 * math.pi, n_sides + 1, device=device, dtype=dtype)[:-1]  # (C,)

    # Radii = width / 2
    radii = widths / 2.0  # (N,)

    # Vectorized ring computation
    # cos(angle) * binormal + sin(angle) * normal, scaled by radius
    cos_a = torch.cos(angles)  # (C,)
    sin_a = torch.sin(angles)  # (C,)

    # Expand for broadcasting: skeleton (N,1,3), binormals (N,1,3), etc.
    center = skeleton.unsqueeze(1)        # (N, 1, 3)
    bn_exp = binormals.unsqueeze(1)       # (N, 1, 3)
    nm_exp = normals.unsqueeze(1)         # (N, 1, 3)
    r_exp = radii.unsqueeze(1).unsqueeze(2)  # (N, 1, 1)

    # Direction vectors for each angle
    cos_exp = cos_a.unsqueeze(0).unsqueeze(2)  # (1, C, 1)
    sin_exp = sin_a.unsqueeze(0).unsqueeze(2)  # (1, C, 1)

    direction = cos_exp * bn_exp + sin_exp * nm_exp  # (N, C, 3)
    vertices = center + r_exp * direction  # (N, C, 3)

    return vertices.reshape(-1, 3)


def loft_plant(
    leaf_skeletons: list[torch.Tensor],
    leaf_widths: list[torch.Tensor],
    leaf_deformations: list[dict[str, torch.Tensor]],
    leaf_tangents: list[torch.Tensor],
    leaf_binormals: list[torch.Tensor],
    n_cross: int = 7,
    gutter_depth: float = 0.0,
) -> torch.Tensor:
    """Loft all leaves of a plant, concatenate vertices.

    Args:
        leaf_skeletons: List of (N_i, 3) skeleton tensors per leaf.
        leaf_widths: List of (N_i,) width tensors per leaf.
        leaf_deformations: List of deformation dicts per leaf.
        leaf_tangents: List of (N_i, 3) tangent tensors per leaf.
        leaf_binormals: List of (N_i, 3) binormal tensors per leaf.
        n_cross: Cross-section vertex count.
        gutter_depth: Midrib channel depth.

    Returns:
        (V_total, 3) concatenated vertex positions from all leaves.
    """
    all_verts = []
    for skel, w, deform, tang, bn in zip(
        leaf_skeletons, leaf_widths, leaf_deformations,
        leaf_tangents, leaf_binormals, strict=True,
    ):
        verts = loft_leaf(skel, w, deform, tang, bn,
                          n_cross=n_cross, gutter_depth=gutter_depth)
        all_verts.append(verts)

    if not all_verts:
        device = leaf_skeletons[0].device if leaf_skeletons else torch.device("cpu")
        dtype = leaf_skeletons[0].dtype if leaf_skeletons else torch.float32
        return torch.empty((0, 3), device=device, dtype=dtype)

    return torch.cat(all_verts, dim=0)
