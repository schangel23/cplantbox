"""Regularization losses to keep fitted parameters near plausible priors."""

import torch


def prior_loss(
    params: torch.Tensor, prior_means: torch.Tensor, prior_stds: torch.Tensor
) -> torch.Tensor:
    """Weighted MSE penalizing deviation from prior distribution.

    Computes sum((params - prior_means)^2 / prior_stds^2).
    Priors are typically MaizeField3D median values and observed standard deviations.

    Args:
        params: (D,) current parameter values
        prior_means: (D,) expected values (e.g., MaizeField3D medians)
        prior_stds: (D,) standard deviations (controls penalty strength)

    Returns:
        Scalar regularization loss.
    """
    return ((params - prior_means) ** 2 / prior_stds**2).sum()
