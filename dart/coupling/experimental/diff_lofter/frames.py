"""Differentiable Frenet frame computation for the lofter.

Ports the gravity-referenced binormal field from g1_to_g3.py to PyTorch,
ensuring all operations support autograd backpropagation.
"""

import torch
import torch.nn.functional as F


def compute_tangents(skeleton: torch.Tensor) -> torch.Tensor:
    """Central-difference tangent vectors, normalized.

    Forward diff at endpoints, central diff for interior points.

    Args:
        skeleton: (N, 3) ordered 3D points.

    Returns:
        (N, 3) unit tangent vectors.
    """
    n = skeleton.shape[0]
    if n < 2:
        return torch.tensor([[0.0, 0.0, 1.0]], device=skeleton.device,
                            dtype=skeleton.dtype).expand(n, 3)

    tangents = torch.empty_like(skeleton)
    # Forward diff at start, backward diff at end
    tangents[0] = skeleton[1] - skeleton[0]
    tangents[-1] = skeleton[-1] - skeleton[-2]
    # Central differences for interior
    if n > 2:
        tangents[1:-1] = skeleton[2:] - skeleton[:-2]

    # Normalize to unit vectors
    lengths = torch.linalg.norm(tangents, dim=1, keepdim=True)
    lengths = torch.clamp(lengths, min=1e-12)
    tangents = tangents / lengths

    return tangents


def _smoothstep(x: torch.Tensor, edge0: float, edge1: float) -> torch.Tensor:
    """Hermite smoothstep: 0 when x <= edge0, 1 when x >= edge1, smooth in between.

    Differentiable everywhere (polynomial).
    """
    t = torch.clamp((x - edge0) / (edge1 - edge0 + 1e-12), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def compute_binormal_field(
    skeleton: torch.Tensor,
    tangents: torch.Tensor,
    smooth_kernel_size: int | None = None,
) -> torch.Tensor:
    """Gravity-referenced binormal field with SVD plane fitting.

    Strategy (matching g1_to_g3.py):
    1. SVD of centered skeleton for best-fit plane normal (stable fallback).
    2. Per-point: project up=[0,0,1] onto plane perpendicular to tangent,
       cross with tangent to get binormal.
    3. Near-vertical tangent: soft-blend between gravity primary and SVD
       plane normal fallback (smoothstep on face-normal magnitude).
    4. Sign consistency via cumulative dot-product sign propagation.
    5. Smooth with 1D conv (replaces scipy uniform_filter1d).
    6. Re-normalize.

    Args:
        skeleton: (N, 3) ordered 3D points.
        tangents: (N, 3) unit tangent vectors.
        smooth_kernel_size: Kernel size for uniform smoothing. If None,
            auto-computed as max(3, N//10) | 1.

    Returns:
        (N, 3) unit binormal vectors.
    """
    n = skeleton.shape[0]
    device = skeleton.device
    dtype = skeleton.dtype

    if n < 2:
        return torch.tensor([[1.0, 0.0, 0.0]], device=device, dtype=dtype).expand(n, 3)

    up = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype)
    fallback = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=dtype)

    # --- Step 1: SVD plane fitting ---
    centroid = skeleton.mean(dim=0)
    centered = skeleton - centroid
    # torch.linalg.svd is differentiable
    _, svals, vh = torch.linalg.svd(centered, full_matrices=False)

    # Plane normal is the right-singular vector with smallest singular value
    svd_pn = vh[2]  # (3,)
    # Check if the plane is well-defined: smallest sv << second sv
    plane_ok = (svals[2] < svals[1] * 0.5) if svals[1] > 1e-8 else False

    # --- Step 2 & 3: Per-point binormal with soft gravity/SVD blending ---
    # Gravity-based face normal: project up onto plane perp to tangent
    # face_normal = up - dot(up, t) * t
    up_expanded = up.unsqueeze(0).expand(n, 3)  # (N, 3)
    dot_up_t = (up_expanded * tangents).sum(dim=1, keepdim=True)  # (N, 1)
    face_normal_gravity = up_expanded - dot_up_t * tangents  # (N, 3)
    fn_len_gravity = torch.linalg.norm(face_normal_gravity, dim=1, keepdim=True)  # (N, 1)

    # Gravity-based binormal: cross(tangent, face_normal_normalized)
    face_normal_gravity_norm = face_normal_gravity / torch.clamp(fn_len_gravity, min=1e-12)
    bn_gravity = torch.linalg.cross(tangents, face_normal_gravity_norm)  # (N, 3)

    # SVD fallback binormal: project svd_pn perp to tangent
    if plane_ok:
        svd_expanded = svd_pn.unsqueeze(0).expand(n, 3)
        dot_svd_t = (svd_expanded * tangents).sum(dim=1, keepdim=True)
        bn_svd = svd_expanded - dot_svd_t * tangents
        bn_svd_len = torch.linalg.norm(bn_svd, dim=1, keepdim=True)
        bn_svd = bn_svd / torch.clamp(bn_svd_len, min=1e-12)
    else:
        # No valid SVD plane; use raw fallback cross
        bn_svd = torch.linalg.cross(
            tangents,
            fallback.unsqueeze(0).expand(n, 3),
        )
        bn_svd_len = torch.linalg.norm(bn_svd, dim=1, keepdim=True)
        bn_svd = bn_svd / torch.clamp(bn_svd_len, min=1e-12)

    # Soft blending: weight = smoothstep(fn_len, 0.1, 0.4)
    # When fn_len_gravity is large (non-vertical tangent), use gravity.
    # When small (near-vertical), blend toward SVD fallback.
    fn_len_squeezed = fn_len_gravity.squeeze(1)  # (N,)
    weight_gravity = _smoothstep(fn_len_squeezed, 0.1, 0.4)  # (N,)
    weight_gravity = weight_gravity.unsqueeze(1)  # (N, 1)

    binormals = weight_gravity * bn_gravity + (1.0 - weight_gravity) * bn_svd  # (N, 3)

    # Normalize
    bn_len = torch.linalg.norm(binormals, dim=1, keepdim=True)
    # Final fallback: if both gravity and SVD gave zero-length, cross with fallback
    degenerate_mask = (bn_len.squeeze(1) < 1e-6)  # (N,)
    if degenerate_mask.any():
        bn_fallback = torch.linalg.cross(
            tangents,
            fallback.unsqueeze(0).expand(n, 3),
        )
        binormals = torch.where(
            degenerate_mask.unsqueeze(1),
            bn_fallback,
            binormals,
        )
        bn_len = torch.linalg.norm(binormals, dim=1, keepdim=True)

    binormals = binormals / torch.clamp(bn_len, min=1e-12)

    # --- Step 4: Sign consistency ---
    # Differentiable approach: compute dot products between consecutive
    # binormals, take sign (detached for gradient), and accumulate via
    # cumprod so all point the same direction as the first.
    if n > 1:
        dots = (binormals[:-1] * binormals[1:]).sum(dim=1)  # (N-1,)
        # Sign: +1 if same direction, -1 if flipped. Detach so gradient
        # flows through the magnitude of binormals, not through sign.
        signs = torch.sign(dots).detach()  # (N-1,)
        # Replace zeros with +1 (parallel vectors)
        signs = torch.where(signs == 0, torch.ones_like(signs), signs)
        # Cumulative product of signs: flip_i = product(signs[0:i])
        cum_signs = torch.cumprod(signs, dim=0)  # (N-1,)
        # First binormal keeps its sign; subsequent ones get flipped as needed
        flip = torch.cat([torch.ones(1, device=device, dtype=dtype), cum_signs])  # (N,)
        binormals = binormals * flip.unsqueeze(1)

    # --- Step 5: Smooth with 1D convolution ---
    if n > 5:
        if smooth_kernel_size is None:
            k = max(3, n // 10)
            k = k | 1  # ensure odd
        else:
            k = smooth_kernel_size
        if k >= 3 and k <= n:
            # F.conv1d: input (batch, channels, length)
            pad = k // 2
            # Uniform kernel
            kernel = torch.ones(1, 1, k, device=device, dtype=dtype) / k
            # Process each of 3 dimensions
            bn_t = binormals.T.unsqueeze(1)  # (3, 1, N)
            # Replicate padding to match scipy 'nearest' mode
            bn_padded = F.pad(bn_t, (pad, pad), mode='replicate')
            bn_smooth = F.conv1d(bn_padded, kernel)  # (3, 1, N)
            binormals = bn_smooth.squeeze(1).T  # (N, 3)

    # --- Step 6: Re-normalize ---
    bn_len = torch.linalg.norm(binormals, dim=1, keepdim=True)
    binormals = binormals / torch.clamp(bn_len, min=1e-12)

    return binormals
