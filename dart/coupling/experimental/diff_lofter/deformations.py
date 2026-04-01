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


# --- Spline-based deformations (replacement for sinusoidal model) ---

SPLINE_DEFORM_NAMES = ['wave_normal', 'wave_lateral', 'twist', 'curl', 'edge_ruffle', 'fold']
DEFAULT_N_CP = 5  # number of control points per deformation type


def _interp_linear(arc_fracs: torch.Tensor, cp_values: torch.Tensor) -> torch.Tensor:
    """Differentiable linear interpolation of control points along arc length.

    Control points are assumed to be evenly spaced in [0, 1].

    Args:
        arc_fracs: (N,) normalized arc lengths in [0, 1].
        cp_values: (K,) control point values at positions [0, 1/(K-1), ..., 1].

    Returns:
        (N,) interpolated values.
    """
    k = cp_values.shape[0]
    if k == 1:
        return cp_values[0].expand_as(arc_fracs)

    # Map arc_fracs to continuous index [0, K-1]
    t = arc_fracs * (k - 1)
    t = torch.clamp(t, 0.0, k - 1.0 - 1e-6)
    idx_lo = t.long()                     # (N,) integer floor index
    idx_hi = (idx_lo + 1).clamp(max=k-1)  # (N,) integer ceil index
    frac = t - idx_lo.float()             # (N,) fractional part

    val_lo = cp_values[idx_lo]  # (N,)
    val_hi = cp_values[idx_hi]  # (N,)

    return val_lo + frac * (val_hi - val_lo)


def compute_deformations_spline(
    arc_fracs: torch.Tensor,
    control_points: dict[str, torch.Tensor],
    ramp_onset: float = 0.15,
) -> dict[str, torch.Tensor]:
    """Compute per-skeleton-point deformations via spline control points.

    Replaces the sinusoidal model. Each deformation type has K learnable control
    points evenly spaced in [0,1] along the leaf. Values are linearly interpolated
    and multiplied by a ramp (zero at base, full at tip).

    Args:
        arc_fracs: (N,) normalized arc lengths in [0, 1].
        control_points: Dict mapping deformation name to (K,) control point tensor.
            Required keys: wave_normal, wave_lateral, twist, curl, edge_ruffle, fold.
        ramp_onset: Fraction of leaf length where deformation begins.

    Returns:
        Dict with same keys as compute_deformations(), each (N,) tensor.
    """
    denom = max(1.0 - ramp_onset, 1e-12)
    ramp = torch.clamp((arc_fracs - ramp_onset) / denom, min=0.0)
    ramp_sq = ramp * ramp  # quadratic for twist

    result = {}
    for name in SPLINE_DEFORM_NAMES:
        cp = control_points[name]
        interp = _interp_linear(arc_fracs, cp)
        if name == 'twist':
            result[name] = interp * ramp_sq
        else:
            result[name] = interp * ramp

    return result


def make_spline_control_points(
    n_cp: int = DEFAULT_N_CP,
    device: str = 'cpu',
    requires_grad: bool = True,
) -> dict[str, torch.Tensor]:
    """Create zero-initialized learnable control points for all deformation types.

    Args:
        n_cp: Number of control points per deformation type.
        device: Torch device.
        requires_grad: Whether tensors track gradients.

    Returns:
        Dict mapping deformation name to (n_cp,) tensor.
    """
    return {
        name: torch.zeros(n_cp, device=device, dtype=torch.float32,
                          requires_grad=requires_grad)
        for name in SPLINE_DEFORM_NAMES
    }


# --- Extended deformations for feature search ---

EXTENDED_DEFORM_NAMES = [
    'blade_tilt',           # cross-section tilt angle (rad)
    'midrib_depth',         # gutter depth (cm)
    'asymmetry',            # left/right width offset (cm)
    'out_of_plane_curv',    # binormal-direction curvature (1/cm)
    'edge_curl',            # margin deflection angle (rad)
    'cross_section_profile', # cross-section curvature factor
    'width_taper',          # width multiplier profile
    'tip_taper_onset',      # scalar: where tip taper starts
]


def compute_extended_deformations(
    arc_fracs: torch.Tensor,
    control_points: dict[str, torch.Tensor],
    ramp_onset: float = 0.15,
) -> dict[str, torch.Tensor]:
    """Compute extended deformation values for feature search candidates.

    Only computes deformations for keys present in control_points.
    Missing keys are silently skipped (feature not active).

    Args:
        arc_fracs: (N,) normalized arc lengths [0, 1].
        control_points: Dict mapping feature name to (K,) CP tensor.
            Only features present in this dict are computed.
        ramp_onset: Fraction where deformation begins.

    Returns:
        Dict mapping feature name to (N,) interpolated values.
    """
    denom = max(1.0 - ramp_onset, 1e-12)
    ramp = torch.clamp((arc_fracs - ramp_onset) / denom, min=0.0)
    ramp_sq = ramp * ramp

    result = {}
    for name, cp in control_points.items():
        if name not in EXTENDED_DEFORM_NAMES:
            continue
        interp = _interp_linear(arc_fracs, cp)

        if name == 'width_taper':
            # Width taper is a multiplier, no ramp — applies everywhere
            result[name] = interp
        elif name == 'tip_taper_onset':
            # Scalar — just use the single control point value
            result[name] = interp
        elif name in ('blade_tilt', 'out_of_plane_curv'):
            # Quadratic ramp for structural features
            result[name] = interp * ramp_sq
        else:
            # Linear ramp for the rest
            result[name] = interp * ramp

    return result


def make_extended_control_points(
    active_features: set[str],
    feature_catalog: dict,
    device: str = 'cpu',
    requires_grad: bool = True,
) -> dict[str, torch.Tensor]:
    """Create zero-initialized control points for active extended features.

    Args:
        active_features: Set of feature names to create CPs for.
        feature_catalog: The FEATURE_CATALOG dict with n_cp per feature.
        device: Torch device.
        requires_grad: Whether tensors track gradients.

    Returns:
        Dict mapping feature name to (n_cp,) tensor.
    """
    result = {}
    for name in active_features:
        if name in feature_catalog:
            n_cp = feature_catalog[name]["n_cp"]
            result[name] = torch.zeros(
                n_cp, device=device, dtype=torch.float32,
                requires_grad=requires_grad,
            )
    return result
