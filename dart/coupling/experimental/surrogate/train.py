"""Training loop for the skeleton surrogate."""

import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, random_split

from ..data_generation.dataset import SkeletonDataset
from ..data_generation.param_sampler import N_LEAF_PARAMS, N_POSITIONS
from .model import SkeletonSurrogate, prepare_per_leaf_input

logger = logging.getLogger(__name__)

N_NODES = 64


def _masked_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """MSE loss on skeleton positions + widths, masked to valid nodes.

    Args:
        pred: ``(B, 11, 64, 4)`` predicted skeletons.
        target: ``(B, 11, 64, 4)`` ground-truth skeletons.
        mask: ``(B, 11, 64)`` valid-node flags (1.0 or 0.0).

    Returns:
        Scalar loss.
    """
    # Expand mask to cover the 4 channels (xyz + width)
    mask4 = mask.unsqueeze(-1).expand_as(pred)  # (B, 11, 64, 4)
    diff = (pred - target) ** 2
    n_valid = mask4.sum().clamp(min=1.0)
    return (diff * mask4).sum() / n_valid


def _position_error_cm(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> float:
    """Mean Euclidean distance (cm) between predicted and target nodes.

    Only xyz channels, masked to valid nodes.
    """
    mask3 = mask.unsqueeze(-1).expand(-1, -1, -1, 3)  # (B, 11, 64, 3)
    diff = (pred[..., :3] - target[..., :3]) ** 2
    dist = (diff * mask3).sum(dim=-1).sqrt()  # (B, 11, 64)
    n_valid = mask.sum().clamp(min=1.0)
    return float((dist * mask).sum() / n_valid)


def train_surrogate(
    dataset_path: str,
    output_dir: str,
    epochs: int = 200,
    batch_size: int = 128,
    lr: float = 1e-3,
    val_split: float = 0.1,
    device: str = "cuda",
    hidden: int = 256,
    n_layers: int = 3,
    patience: int = 20,
) -> dict:
    """Train the skeleton surrogate model.

    Saves:
        - ``best_model.pt`` — model checkpoint with best validation loss.
        - ``norm_stats.npz`` — parameter normalisation statistics.
        - ``train_log.json`` — per-epoch training curves.

    Args:
        dataset_path: Path to HDF5 training data.
        output_dir: Directory for outputs.
        epochs: Maximum training epochs.
        batch_size: Mini-batch size.
        lr: Initial learning rate for AdamW.
        val_split: Fraction of data reserved for validation.
        device: ``"cuda"`` or ``"cpu"``.
        hidden: MLP hidden layer width.
        n_layers: Number of MLP hidden layers.
        patience: Early-stopping patience (epochs without improvement).

    Returns:
        Dict with final metrics: ``best_val_loss``, ``best_epoch``,
        ``best_pos_error_cm``.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        device = "cpu"

    # --- Data ---
    full_dataset = SkeletonDataset(dataset_path, normalize=True)
    n_total = len(full_dataset)
    n_val = max(1, int(n_total * val_split))
    n_train = n_total - n_val

    train_ds, val_ds = random_split(
        full_dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )

    # Apply train-set normalization to val set (same stats).
    mean, std = full_dataset.get_normalization()
    np.savez(str(out / "norm_stats.npz"), mean=mean, std=std)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    logger.info(
        "Dataset: %d total, %d train, %d val", n_total, n_train, n_val
    )

    # --- Model ---
    model = SkeletonSurrogate(
        n_params=N_LEAF_PARAMS + 1,
        n_nodes=N_NODES,
        hidden=hidden,
        n_layers=n_layers,
    ).to(device)

    n_model_params = sum(p.numel() for p in model.parameters())
    logger.info("Model: %d parameters", n_model_params)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)

    # --- Training ---
    best_val_loss = float("inf")
    best_epoch = 0
    best_pos_err = float("inf")
    epochs_no_improve = 0
    log: list[dict] = []

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        # Train
        model.train()
        train_loss_sum = 0.0
        train_n = 0

        for params_batch, skel_batch, mask_batch in train_loader:
            params_batch = params_batch.to(device)
            skel_batch = skel_batch.to(device)
            mask_batch = mask_batch.to(device)

            B = params_batch.shape[0]

            # Unpack into per-leaf inputs
            per_leaf, lmax, theta, tropismS, width_blade = prepare_per_leaf_input(
                params_batch
            )

            # Predict per-leaf skeletons
            pred_flat = model.predict_skeleton(
                per_leaf, lmax, theta, tropismS, width_blade
            )  # (B*11, 64, 4)
            pred = pred_flat.view(B, N_POSITIONS, N_NODES, 4)

            loss = _masked_mse(pred, skel_batch, mask_batch)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss_sum += loss.item() * B
            train_n += B

        scheduler.step()
        train_loss = train_loss_sum / max(train_n, 1)

        # Validate
        model.eval()
        val_loss_sum = 0.0
        val_pos_err_sum = 0.0
        val_n = 0

        with torch.no_grad():
            for params_batch, skel_batch, mask_batch in val_loader:
                params_batch = params_batch.to(device)
                skel_batch = skel_batch.to(device)
                mask_batch = mask_batch.to(device)

                B = params_batch.shape[0]
                per_leaf, lmax, theta, tropismS, width_blade = prepare_per_leaf_input(
                    params_batch
                )
                pred_flat = model.predict_skeleton(
                    per_leaf, lmax, theta, tropismS, width_blade
                )
                pred = pred_flat.view(B, N_POSITIONS, N_NODES, 4)

                vloss = _masked_mse(pred, skel_batch, mask_batch)
                val_loss_sum += vloss.item() * B

                pos_err = _position_error_cm(pred, skel_batch, mask_batch)
                val_pos_err_sum += pos_err * B

                val_n += B

        val_loss = val_loss_sum / max(val_n, 1)
        val_pos_err = val_pos_err_sum / max(val_n, 1)
        elapsed = time.time() - t0

        entry = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_pos_error_cm": val_pos_err,
            "lr": optimizer.param_groups[0]["lr"],
            "elapsed_s": elapsed,
        }
        log.append(entry)

        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            best_epoch = epoch
            best_pos_err = val_pos_err
            epochs_no_improve = 0

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_loss": val_loss,
                    "val_pos_error_cm": val_pos_err,
                    "hidden": hidden,
                    "n_layers": n_layers,
                    "n_params": N_LEAF_PARAMS + 1,
                    "n_nodes": N_NODES,
                },
                str(out / "best_model.pt"),
            )
        else:
            epochs_no_improve += 1

        if epoch <= 5 or epoch % 10 == 0 or improved:
            marker = " *" if improved else ""
            logger.info(
                "Epoch %3d | train %.6f | val %.6f | pos_err %.3f cm | "
                "lr %.2e | %.1fs%s",
                epoch, train_loss, val_loss, val_pos_err,
                optimizer.param_groups[0]["lr"], elapsed, marker,
            )

        if epochs_no_improve >= patience:
            logger.info(
                "Early stopping at epoch %d (patience=%d, best=%d)",
                epoch, patience, best_epoch,
            )
            break

    # Save training log
    with open(str(out / "train_log.json"), "w") as f:
        json.dump(log, f, indent=2)

    logger.info(
        "Training complete. Best epoch %d: val_loss=%.6f, pos_err=%.3f cm",
        best_epoch, best_val_loss, best_pos_err,
    )

    return {
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "best_pos_error_cm": best_pos_err,
    }
