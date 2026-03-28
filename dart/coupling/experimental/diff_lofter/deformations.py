"""Differentiable leaf deformation computations.

Ports the 6 deformation types from g1_to_g3.py (lines 608-684) to PyTorch.
All operations are element-wise on tensors, fully supporting autograd.
"""

import math
import torch


def compute_deformations(
    arc_fracs: torch.Tensor,
    wave_normal_amp: torch.Tensor,
    wave_normal_freq: float,
    wave_normal_phase: float,
    wave_lateral_amp: torch.Tensor,
    wave_lateral_freq: float,
    wave_lateral_phase: float,
    twist_max: torch.Tensor,
    curl_amp: torch.Tensor,
    curl_freq: float,
    curl_phase: float,
    edge_ruffle_amp: torch.Tensor,
    edge_ruffle_freq: float,
    edge_ruffle_phase: float,
    fold_amp: torch.Tensor,
    fold_freq: float,
    fold_phase: float,
    ramp_onset: float = 0.15,
) -> dict[str, torch.Tensor]:
    """Compute per-skeleton-point deformation values.

    Each deformation follows: amplitude * ramp(t) * sin(2*pi*freq*t + phase)
    where t = arc_fracs (normalized arc length, 0 at base, 1 at tip).

    Ramp: linear onset from ramp_onset to 1.0, clamped at 0.
    Twist uses a quadratic ramp (gentle near base, strong at tip).

    Args:
        arc_fracs: (N,) normalized arc lengths in [0, 1].
        wave_normal_amp: Scalar tensor, amplitude of vertical undulation (cm).
        wave_normal_freq: Frequency (waves per leaf length).
        wave_normal_phase: Phase offset (radians).
        wave_lateral_amp: Scalar tensor, amplitude of lateral sway (cm).
        wave_lateral_freq: Frequency.
        wave_lateral_phase: Phase offset.
        twist_max: Scalar tensor, maximum twist angle (radians) at tip.
        curl_amp: Scalar tensor, curl displacement amplitude (cm).
        curl_freq: Frequency.
        curl_phase: Phase offset.
        edge_ruffle_amp: Scalar tensor, edge ruffle amplitude (cm).
        edge_ruffle_freq: Frequency (high, ~7 waves per leaf).
        edge_ruffle_phase: Phase offset.
        fold_amp: Scalar tensor, internal fold amplitude (cm).
        fold_freq: Frequency.
        fold_phase: Phase offset.
        ramp_onset: Fraction of leaf length where deformation begins.

    Returns:
        Dict with keys:
            wave_normal: (N,) vertical undulation offsets.
            wave_lateral: (N,) lateral sway offsets.
            twist: (N,) twist angles (quadratic ramp).
            curl: (N,) curl factors.
            edge_ruffle: (N,) edge ruffle base values.
            fold: (N,) fold factors.
    """
    two_pi = 2.0 * math.pi

    # Linear ramp: 0 below onset, linearly increasing to 1 at tip
    denom = max(1.0 - ramp_onset, 1e-12)
    ramp = torch.clamp((arc_fracs - ramp_onset) / denom, min=0.0)
    ramp_sq = ramp * ramp  # quadratic for twist

    # Wave normal: vertical undulation
    wave_normal = wave_normal_amp * ramp * torch.sin(
        two_pi * wave_normal_freq * arc_fracs + wave_normal_phase
    )

    # Wave lateral: side-to-side sway
    wave_lateral = wave_lateral_amp * ramp * torch.sin(
        two_pi * wave_lateral_freq * arc_fracs + wave_lateral_phase
    )

    # Twist: quadratic ramp (gentle near base, strong at tip)
    twist = twist_max * ramp_sq

    # Curl: low-frequency asymmetric edge displacement
    curl = curl_amp * ramp * torch.sin(
        two_pi * curl_freq * arc_fracs + curl_phase
    )

    # Edge ruffle: high-frequency undulation at leaf margins
    edge_ruffle = edge_ruffle_amp * ramp * torch.sin(
        two_pi * edge_ruffle_freq * arc_fracs + edge_ruffle_phase
    )

    # Internal fold: cross-sectional curvature variation
    fold = fold_amp * ramp * torch.sin(
        two_pi * fold_freq * arc_fracs + fold_phase
    )

    return {
        "wave_normal": wave_normal,
        "wave_lateral": wave_lateral,
        "twist": twist,
        "curl": curl,
        "edge_ruffle": edge_ruffle,
        "fold": fold,
    }
