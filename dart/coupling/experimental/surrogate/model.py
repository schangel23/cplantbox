"""Neural surrogate: XML params -> leaf skeleton prediction.

Architecture: per-node implicit function.  For each skeleton node, the
network receives the leaf parameters (13 dims) plus a normalised arc
position t in [0, 1] and predicts the residual (dx, dy, dz, dw) relative
to the analytic baseline.  This is equivalent to learning a continuous
function f(params, t) -> (x, y, z, w) for the skeleton.

The key insight is that 14 -> 4 is far easier for an MLP than the
previous 13 -> 256 (which tried to predict all 64 nodes at once and
plateaued at 7.7 cm error).

Shared weights across all nodes AND all leaf positions.
"""

import math

import torch
import torch.nn as nn

from .analytic_baseline import analytic_skeleton
from ..data_generation.param_sampler import LEAF_PARAM_NAMES, N_LEAF_PARAMS, N_POSITIONS


# Indices into the per-leaf param vector for analytic baseline extraction.
_IDX_LMAX = LEAF_PARAM_NAMES.index("lmax")
_IDX_THETA = LEAF_PARAM_NAMES.index("theta")
_IDX_TROPISMS = LEAF_PARAM_NAMES.index("tropismS")
_IDX_WIDTH = LEAF_PARAM_NAMES.index("Width_blade")


def _sinusoidal_encoding(t: torch.Tensor, n_freqs: int = 8) -> torch.Tensor:
    """Positional encoding: [t, sin(2^0*pi*t), cos(2^0*pi*t), ...].

    Helps the MLP learn high-frequency spatial variation along the leaf.

    Args:
        t: (...,) normalised positions in [0, 1].
        n_freqs: Number of frequency bands.

    Returns:
        (..., 1 + 2*n_freqs) encoded positions.
    """
    freqs = 2.0 ** torch.arange(n_freqs, device=t.device, dtype=t.dtype) * math.pi
    t_exp = t.unsqueeze(-1)  # (..., 1)
    args = t_exp * freqs  # (..., n_freqs)
    return torch.cat([t_exp, torch.sin(args), torch.cos(args)], dim=-1)


class SkeletonSurrogate(nn.Module):
    """Per-node implicit function predicting skeleton residuals.

    Input:  ``(B, 13)`` leaf params, evaluated at 64 node positions.
    Output: ``(B, 64, 4)`` delta xyz + delta half-width.

    Internally, each node query is ``(B*64, 13 + 1 + 2*n_freqs)`` —
    leaf params concatenated with sinusoidal-encoded arc position.

    Args:
        n_params: Per-leaf input dimension (default 13).
        n_nodes: Number of skeleton nodes (default 64).
        hidden: Hidden layer width (default 256).
        n_layers: Number of hidden layers (default 4).
        n_freqs: Frequency bands for positional encoding (default 8).
    """

    def __init__(
        self,
        n_params: int = N_LEAF_PARAMS + 1,
        n_nodes: int = 64,
        hidden: int = 256,
        n_layers: int = 4,
        n_freqs: int = 8,
    ) -> None:
        super().__init__()
        self.n_params = n_params
        self.n_nodes = n_nodes
        self.n_freqs = n_freqs

        # Input: leaf params + sinusoidal-encoded node position
        pos_dim = 1 + 2 * n_freqs  # 17 for n_freqs=8
        in_dim = n_params + pos_dim  # 13 + 17 = 30

        layers: list[nn.Module] = []
        d = in_dim
        for _ in range(n_layers):
            layers.append(nn.Linear(d, hidden))
            layers.append(nn.SiLU())
            layers.append(nn.LayerNorm(hidden))
            d = hidden

        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(hidden, 4)  # predict (dx, dy, dz, dw) per node

        # Small init so initial prediction ≈ baseline
        nn.init.zeros_(self.head.bias)
        nn.init.normal_(self.head.weight, std=1e-3)

    def forward(self, params: torch.Tensor) -> torch.Tensor:
        """Predict skeleton residuals for all nodes.

        Args:
            params: ``(B, n_params)`` per-leaf parameter vector.

        Returns:
            ``(B, n_nodes, 4)`` residual (delta xyz + delta width).
        """
        B = params.shape[0]
        device = params.device
        dtype = params.dtype

        # Node positions: [0, 1] uniformly spaced
        t = torch.linspace(0.0, 1.0, self.n_nodes, device=device, dtype=dtype)  # (N,)
        t_enc = _sinusoidal_encoding(t, self.n_freqs)  # (N, 17)

        # Expand and concatenate: each leaf's params paired with each node position
        # params: (B, 1, 13) -> (B, N, 13)
        # t_enc:  (1, N, 17) -> (B, N, 17)
        params_exp = params.unsqueeze(1).expand(B, self.n_nodes, self.n_params)
        t_enc_exp = t_enc.unsqueeze(0).expand(B, self.n_nodes, -1)

        x = torch.cat([params_exp, t_enc_exp], dim=-1)  # (B, N, 30)
        x = x.reshape(B * self.n_nodes, -1)  # (B*N, 30)

        h = self.backbone(x)  # (B*N, hidden)
        out = self.head(h)  # (B*N, 4)

        return out.view(B, self.n_nodes, 4)

    def predict_skeleton(
        self,
        params: torch.Tensor,
        lmax: torch.Tensor,
        theta: torch.Tensor,
        tropismS: torch.Tensor,
        width_blade: torch.Tensor,
    ) -> torch.Tensor:
        """Full prediction: analytic baseline + learned residual.

        Args:
            params: ``(B, n_params)`` normalised per-leaf parameters.
            lmax: ``(B,)`` maximum leaf length.
            theta: ``(B,)`` insertion angle from vertical.
            tropismS: ``(B,)`` gravitropism strength.
            width_blade: ``(B,)`` maximum half-width.

        Returns:
            ``(B, n_nodes, 4)`` absolute skeleton coordinates + widths.
        """
        baseline = analytic_skeleton(lmax, theta, tropismS, width_blade, self.n_nodes)
        residual = self.forward(params)
        return baseline + residual


def prepare_per_leaf_input(
    flat_params: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Unpack a flat (B, 133) batch into per-leaf inputs.

    Returns:
        per_leaf_params: ``(B * 11, 13)`` — leaf params + position.
        lmax: ``(B * 11,)``
        theta: ``(B * 11,)``
        tropismS: ``(B * 11,)``
        width_blade: ``(B * 11,)``
    """
    B = flat_params.shape[0]
    device = flat_params.device
    dtype = flat_params.dtype

    # Split flat vector into per-position blocks.
    # Layout: [pos0 * 12, pos1 * 12, ..., pos10 * 12, stem_ln]
    leaf_block = flat_params[:, : N_POSITIONS * N_LEAF_PARAMS]  # (B, 132)
    leaf_block = leaf_block.view(B, N_POSITIONS, N_LEAF_PARAMS)  # (B, 11, 12)

    # Position encoding: normalised index in [0, 1]
    pos_enc = torch.linspace(0.0, 1.0, N_POSITIONS, device=device, dtype=dtype)
    pos_enc = pos_enc.unsqueeze(0).unsqueeze(-1).expand(B, -1, -1)  # (B, 11, 1)

    per_leaf = torch.cat([leaf_block, pos_enc], dim=-1)  # (B, 11, 13)
    per_leaf = per_leaf.reshape(B * N_POSITIONS, N_LEAF_PARAMS + 1)  # (B*11, 13)

    # Extract baseline parameters
    lmax = leaf_block[:, :, _IDX_LMAX].reshape(-1)
    theta = leaf_block[:, :, _IDX_THETA].reshape(-1)
    tropismS = leaf_block[:, :, _IDX_TROPISMS].reshape(-1)
    width_blade = leaf_block[:, :, _IDX_WIDTH].reshape(-1)

    return per_leaf, lmax, theta, tropismS, width_blade
