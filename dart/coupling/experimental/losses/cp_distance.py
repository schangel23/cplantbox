"""CP-space L2 loss and Hungarian leaf matcher (Phase 3).

This replaces the bidirectional-Chamfer-on-triangles loss used by the
existing leaf fitting pipeline with a direct L2 distance in the canonical
NURBS control-point space defined by
:mod:`dart.coupling.geometry.canonical_cp_grid`.

Motivation
----------
For a canonical ``(N_U, N_V, 3)`` CP grid, every control point carries an
explicit parametric identity — CP ``[i, j]`` is the knot-vector-anchored
handle at ``(u_i, v_j)``. When two leaves live in the same canonical
parameterisation, their CP grids can be compared element-wise:

    L = (1/M) * Σ_{matched (p, t)} ||CP_pred[p] − CP_target[t]||²

where the inner norm is summed over all ``N_U × N_V`` CPs of each leaf
and ``M`` is the number of matched pairs.

Because the shape has at most ~22 leaves on a maize plant, a Hungarian
assignment on a small centroid-feature cost matrix is microsecond-scale
and gives the optimal one-to-one pairing under a sensible "similarity
under physical layout + rank" cost.

Leaf-match cost features
------------------------
The match cost between predicted leaf ``p`` and target leaf ``t`` is::

    cost(p, t) = w_xyz · ||c_p − c_t||²_norm
                 + w_arc · (a_p − a_t)²_norm
                 + w_rank · (r_p − r_t)²_norm

Features, all normalised to zero mean / unit std over the union of
predicted and target leaves:
  - ``c = (xc, yc, zc)`` — CP-grid centroid in cm.
  - ``a`` — midrib arc length in cm.
  - ``r`` — integer emergence rank (leaf index from the base). If not
    supplied, rank is inferred from centroid-z order per side.

The rank term disambiguates near-symmetric leaves (left/right on maize).

Public surface
--------------
  - :func:`cp_l2_loss` — scalar loss for matched pairs.
  - :func:`per_cp_distance` — per-CP L2 distances for diagnostic plots.
  - :func:`hungarian_leaf_match` — optimal 1-to-1 leaf match.
  - :func:`leaf_centroid`, :func:`leaf_arc_length` — feature helpers.
"""
from __future__ import annotations

from collections.abc import Hashable
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

# `dict[K, V]` is not assignable to `Mapping[Hashable, V]` under pyright
# because `Mapping`'s key parameter is invariant. Using `Any` as the key
# parameter keeps pyright happy without weakening runtime behaviour; every
# concrete `dict[K, np.ndarray]` remains a valid argument.
_CPMap = Mapping[Any, np.ndarray]


# ---------------------------------------------------------------------------
# Feature helpers
# ---------------------------------------------------------------------------
def leaf_centroid(cps: np.ndarray) -> np.ndarray:
    """Return the (3,) centroid of a canonical ``(N_U, N_V, 3)`` CP grid."""
    arr = np.asarray(cps, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"cps must be (N_U, N_V, 3); got {arr.shape}")
    return arr.reshape(-1, 3).mean(axis=0)


def leaf_arc_length(cps: np.ndarray) -> float:
    """Midrib arc length along the central v-column.

    For a canonical ``(N_U=11, N_V=5, 3)`` grid, the midrib column is
    ``v=0.5`` which is CP index ``j = N_V // 2 = 2``. Arc length uses the
    CPs directly — they are handles, not curve samples, but the polyline
    through the midrib CPs is a faithful first-order approximation of the
    evaluated midrib curve and is *identical* across matched canonical
    grids, so its use here as a matching feature is consistent.
    """
    arr = np.asarray(cps, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"cps must be (N_U, N_V, 3); got {arr.shape}")
    mid_j = arr.shape[1] // 2
    midrib = arr[:, mid_j, :]
    diffs = np.diff(midrib, axis=0)
    return float(np.linalg.norm(diffs, axis=1).sum())


def _infer_rank_by_z(cps_by_key: _CPMap) -> dict:
    """Assign integer ranks 0..N-1 by ascending centroid-z."""
    keys = list(cps_by_key.keys())
    zs = np.array([leaf_centroid(cps_by_key[k])[2] for k in keys])
    order = np.argsort(zs)
    ranks = {keys[int(order[i])]: i for i in range(len(keys))}
    return ranks


def _feature_matrix(
    cps_by_key: _CPMap,
    ranks: Mapping[Any, int] | None = None,
) -> tuple[list[Hashable], np.ndarray]:
    """Return ``(keys, F)`` with ``F`` of shape ``(len(keys), 5)`` in columns
    ``[xc, yc, zc, arc_length, rank]``."""
    keys = list(cps_by_key.keys())
    if ranks is None:
        ranks = _infer_rank_by_z(cps_by_key)
    F = np.empty((len(keys), 5), dtype=np.float64)
    for i, k in enumerate(keys):
        cps = cps_by_key[k]
        c = leaf_centroid(cps)
        F[i, 0:3] = c
        F[i, 3] = leaf_arc_length(cps)
        F[i, 4] = float(ranks[k])
    return keys, F


# ---------------------------------------------------------------------------
# Hungarian matcher
# ---------------------------------------------------------------------------
def hungarian_leaf_match(
    pred_cps: _CPMap,
    target_cps: _CPMap,
    weight_xyz: float = 1.0,
    weight_arc: float = 0.5,
    weight_rank: float = 0.5,
    pred_ranks: Mapping[Any, int] | None = None,
    target_ranks: Mapping[Any, int] | None = None,
) -> list[tuple[Hashable, Hashable]]:
    """Optimal 1-to-1 leaf assignment via scipy.optimize.linear_sum_assignment.

    Cost is the weighted squared Euclidean distance in a 5-dimensional
    feature space: xyz centroid, midrib arc length, and rank (see module
    docstring). Features are normalised to zero mean / unit std across
    the union of predicted and target leaves, then weighted.

    When ``len(pred) != len(target)``, the matcher pads the smaller set
    with dummy infinite-cost columns/rows so ``linear_sum_assignment``
    returns only the real-to-real pairs (``min(n_pred, n_target)`` of them).

    Args:
        pred_cps: ``{leaf_key: (N_U, N_V, 3) array}`` of predicted leaves.
        target_cps: same format for target leaves.
        weight_xyz / weight_arc / weight_rank: relative weights on the
            three feature groups (xyz triplet is a single group).
        pred_ranks / target_ranks: optional integer ranks per leaf. If not
            supplied, ranks are inferred from ascending centroid-z.

    Returns:
        List of ``(pred_key, target_key)`` pairs of length
        ``min(len(pred), len(target))``.
    """
    if not pred_cps or not target_cps:
        return []

    p_keys, Fp = _feature_matrix(pred_cps, pred_ranks)
    t_keys, Ft = _feature_matrix(target_cps, target_ranks)

    # Normalise column-wise across the union of both sets for a dimension-
    # independent cost metric.
    stacked = np.vstack([Fp, Ft])
    mean = stacked.mean(axis=0)
    std = stacked.std(axis=0)
    std_safe = np.where(std > 1e-9, std, 1.0)

    Fp_n = (Fp - mean) / std_safe
    Ft_n = (Ft - mean) / std_safe

    # Per-group weights: xyz is a single 3-feature group.
    w = np.array(
        [weight_xyz, weight_xyz, weight_xyz, weight_arc, weight_rank],
        dtype=np.float64,
    )
    w = np.sqrt(w)  # applied to features; the L2 cost squares them back.
    Fp_w = Fp_n * w
    Ft_w = Ft_n * w

    # Pairwise squared-Euclidean cost matrix — shape (n_pred, n_target).
    diff = Fp_w[:, None, :] - Ft_w[None, :, :]
    cost = np.einsum("ijk,ijk->ij", diff, diff)

    row_ind, col_ind = linear_sum_assignment(cost)
    return [(p_keys[int(r)], t_keys[int(c)]) for r, c in zip(row_ind, col_ind)]


# ---------------------------------------------------------------------------
# CP-space L2 loss
# ---------------------------------------------------------------------------
def cp_l2_loss(
    pred_cps: _CPMap,
    target_cps: _CPMap,
    leaf_match: Sequence[tuple[Hashable, Hashable]],
    reduction: str = "mean",
) -> float:
    """Sum (or mean) of squared CP distances over matched leaf pairs.

    Args:
        pred_cps: ``{leaf_key: (N_U, N_V, 3)}`` grids in cm.
        target_cps: same format for target leaves.
        leaf_match: list of matched ``(pred_key, target_key)`` pairs.
        reduction: ``"mean"`` (default) averages over matched leaves so
            the loss does not grow with plant size; ``"sum"`` returns the
            total SSD; ``"mean_per_cp"`` averages over both matched leaves
            **and** the ``N_U * N_V`` CPs per leaf.

    Returns:
        A float loss value. 0.0 for an empty match.
    """
    if not leaf_match:
        return 0.0

    total = 0.0
    n_cps_per_leaf = 0
    for p_id, t_id in leaf_match:
        pc = np.asarray(pred_cps[p_id], dtype=np.float64)
        tc = np.asarray(target_cps[t_id], dtype=np.float64)
        if pc.shape != tc.shape:
            raise ValueError(
                f"shape mismatch pred[{p_id}]={pc.shape} vs target[{t_id}]={tc.shape}"
            )
        diff = pc - tc
        total += float(np.sum(diff * diff))
        n_cps_per_leaf = int(np.prod(pc.shape[:-1]))

    if reduction == "mean":
        return total / len(leaf_match)
    if reduction == "sum":
        return total
    if reduction == "mean_per_cp":
        denom = len(leaf_match) * max(1, n_cps_per_leaf)
        return total / denom
    raise ValueError(f"unknown reduction: {reduction!r}")


def per_cp_distance(
    pred_cps: _CPMap,
    target_cps: _CPMap,
    leaf_match: Sequence[tuple[Hashable, Hashable]],
) -> np.ndarray:
    """Per-CP L2 distance tensor of shape ``(M, N_U, N_V)``.

    Useful for diagnostic heatmaps and for checking which CPs dominate
    the loss (collar, tip, edge, midrib, …).
    """
    if not leaf_match:
        return np.empty((0, 0, 0), dtype=np.float64)

    distances = []
    for p_id, t_id in leaf_match:
        pc = np.asarray(pred_cps[p_id], dtype=np.float64)
        tc = np.asarray(target_cps[t_id], dtype=np.float64)
        if pc.shape != tc.shape:
            raise ValueError(
                f"shape mismatch pred[{p_id}]={pc.shape} vs target[{t_id}]={tc.shape}"
            )
        distances.append(np.linalg.norm(pc - tc, axis=-1))
    return np.stack(distances, axis=0)


__all__ = [
    "leaf_centroid",
    "leaf_arc_length",
    "hungarian_leaf_match",
    "cp_l2_loss",
    "per_cp_distance",
]
