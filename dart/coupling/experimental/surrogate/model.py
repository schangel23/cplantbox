"""Neural surrogate: XML params -> leaf skeleton prediction.

Architecture: a shared-weight MLP that takes per-leaf parameters
(12 XML/deformation params + 1 position encoding) and predicts
skeleton *residuals* relative to the analytic baseline.  The full
skeleton is ``baseline + residual``.

Shared weights across leaf positions reduce the parameter count
and encourage generalisation to unseen position/param combinations.
"""

import torch
import torch.nn as nn

from .analytic_baseline import analytic_skeleton
from ..data_generation.param_sampler import LEAF_PARAM_NAMES, N_LEAF_PARAMS, N_POSITIONS


# Indices into the per-leaf param vector for analytic baseline extraction.
_IDX_LMAX = LEAF_PARAM_NAMES.index("lmax")
_IDX_THETA = LEAF_PARAM_NAMES.index("theta")
_IDX_TROPISMS = LEAF_PARAM_NAMES.index("tropismS")
_IDX_WIDTH = LEAF_PARAM_NAMES.index("Width_blade")


class SkeletonSurrogate(nn.Module):
    """Shared-weight MLP predicting skeleton residuals from XML params.

    Input:  ``(B, 13)`` — 12 per-leaf params + 1 normalised position index.
    Output: ``(B, 64, 4)`` — delta xyz + delta half-width relative to
            the analytic baseline.

    Args:
        n_params: Dimension of the per-leaf input vector (default 13).
        n_nodes: Number of skeleton nodes per leaf (default 64).
        hidden: Hidden layer width (default 256).
        n_layers: Number of hidden layers (default 3).
    """

    def __init__(
        self,
        n_params: int = N_LEAF_PARAMS + 1,  # 12 leaf params + 1 position
        n_nodes: int = 64,
        hidden: int = 256,
        n_layers: int = 3,
    ) -> None:
        super().__init__()
        self.n_params = n_params
        self.n_nodes = n_nodes

        layers: list[nn.Module] = []
        in_dim = n_params
        for _ in range(n_layers):
            layers.append(nn.Linear(in_dim, hidden))
            layers.append(nn.SiLU())
            layers.append(nn.LayerNorm(hidden))
            in_dim = hidden

        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(hidden, n_nodes * 4)

        # Initialise head weights small so initial prediction ≈ baseline.
        nn.init.zeros_(self.head.bias)
        nn.init.normal_(self.head.weight, std=1e-3)

    def forward(self, params: torch.Tensor) -> torch.Tensor:
        """Predict skeleton residuals.

        Args:
            params: ``(B, n_params)`` per-leaf parameter vector.

        Returns:
            ``(B, n_nodes, 4)`` residual (delta xyz + delta width).
        """
        h = self.backbone(params)
        out = self.head(h)  # (B, n_nodes * 4)
        return out.view(-1, self.n_nodes, 4)

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
