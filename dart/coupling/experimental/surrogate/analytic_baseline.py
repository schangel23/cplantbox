"""Analytic straight-line skeleton baseline for residual prediction.

Provides a simple geometric prior: straight line at the insertion angle
with parabolic gravitropic droop and a power-law width taper.  The
neural surrogate only needs to learn the *residual* between this
baseline and the true CPlantBox skeleton, which greatly reduces the
learning difficulty.
"""

import torch
import math


def analytic_skeleton(
    lmax: torch.Tensor,
    theta: torch.Tensor,
    tropismS: torch.Tensor,
    width_blade: torch.Tensor,
    n_nodes: int = 64,
) -> torch.Tensor:
    """Generate a baseline skeleton per sample in the batch.

    The skeleton is a straight line starting at the origin, extending at
    angle *theta* from the vertical (z-axis) for length *lmax*, with a
    parabolic droop controlled by *tropismS* and a simple width taper.

    Args:
        lmax: ``(B,)`` maximum leaf length [cm].
        theta: ``(B,)`` insertion angle from vertical [rad].
        tropismS: ``(B,)`` gravitropism strength (droop coefficient).
        width_blade: ``(B,)`` maximum half-width [cm].
        n_nodes: Number of nodes along the skeleton.

    Returns:
        ``(B, n_nodes, 4)`` tensor — columns are (x, y, z, half-width).
    """
    B = lmax.shape[0]
    device = lmax.device
    dtype = lmax.dtype

    # Arc-length parameter t in [0, 1]
    t = torch.linspace(0.0, 1.0, n_nodes, device=device, dtype=dtype)  # (n_nodes,)
    t = t.unsqueeze(0).expand(B, -1)  # (B, n_nodes)

    # Arc-length positions
    s = t * lmax.unsqueeze(1)  # (B, n_nodes)

    # Straight-line direction at angle theta from vertical.
    # x = s * sin(theta), y = 0, z = s * cos(theta).
    sin_t = torch.sin(theta).unsqueeze(1)  # (B, 1)
    cos_t = torch.cos(theta).unsqueeze(1)  # (B, 1)

    x = s * sin_t
    y = torch.zeros_like(x)
    z = s * cos_t

    # Parabolic gravitropic droop: z decreases by tropismS * s^2.
    droop = tropismS.unsqueeze(1) * s * s
    z = z - droop

    # Width profile: sqrt taper from max at base to 0 at tip.
    # w(t) = width_blade * (1 - sqrt(t))
    w = width_blade.unsqueeze(1) * (1.0 - torch.sqrt(t))

    # Stack to (B, n_nodes, 4)
    skeleton = torch.stack([x, y, z, w], dim=-1)

    return skeleton
