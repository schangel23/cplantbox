"""PyTorch Dataset for skeleton training data."""

from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from .param_sampler import N_PARAMS


class SkeletonDataset(Dataset):
    """Loads HDF5 skeleton data for surrogate training.

    Each item is a ``(params, skeletons, masks)`` tuple where:

    - ``params``:    ``(133,)`` normalised parameter vector.
    - ``skeletons``: ``(11, 64, 4)`` xyz + half-width per leaf, zero-padded.
    - ``masks``:     ``(11, 64)`` boolean mask for valid nodes.

    Args:
        h5_path: Path to the HDF5 file produced by
            :func:`generate_skeletons.generate_dataset`.
        normalize: If True (default), z-score normalise the parameter
            vectors using training-set statistics.
    """

    def __init__(self, h5_path: str, normalize: bool = True) -> None:
        with h5py.File(h5_path, "r") as f:
            self.params = np.array(f["params"], dtype=np.float32)       # (N, 133)
            self.skeletons = np.array(f["skeletons"], dtype=np.float32) # (N, 11, 64, 4)
            self.masks = np.array(f["masks"], dtype=bool)               # (N, 11, 64)

        assert self.params.shape[1] == N_PARAMS, (
            f"Expected {N_PARAMS} params, got {self.params.shape[1]}"
        )

        # Compute normalization statistics on the full dataset.
        # Callers should compute on train split only if doing train/val
        # split externally; for a single-file workflow this is fine.
        self._mean = self.params.mean(axis=0)
        self._std = self.params.std(axis=0)
        self._std[self._std < 1e-8] = 1.0  # avoid division by zero

        self.normalize = normalize

    def __len__(self) -> int:
        return len(self.params)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        params = self.params[idx].copy()
        if self.normalize:
            params = (params - self._mean) / self._std

        return (
            torch.from_numpy(params),
            torch.from_numpy(self.skeletons[idx]),
            torch.from_numpy(self.masks[idx].astype(np.float32)),
        )

    def get_normalization(self) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(mean, std)`` arrays for parameter denormalization."""
        return self._mean.copy(), self._std.copy()

    def set_normalization(self, mean: np.ndarray, std: np.ndarray) -> None:
        """Override normalization stats (e.g. from training split)."""
        self._mean = mean.copy()
        self._std = std.copy()
        self._std[self._std < 1e-8] = 1.0
