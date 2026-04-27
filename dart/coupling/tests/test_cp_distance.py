"""Unit tests for the CP-space L2 loss and Hungarian leaf matcher (Phase 3).

Covers:
  1. Zero-cost identity: matching a set of CPs against itself gives zero
     loss and per-CP distance.
  2. Known-translation loss: shifting all pred CPs by ``d`` gives loss
     ``d²·N_U·N_V·3 / 3 = d²·N_U·N_V`` per matched leaf (since the squared
     displacement magnitude is ``d²`` per CP with an x-only shift).
  3. Shape-mismatch raises.
  4. Hungarian matcher recovers the identity permutation on shuffled keys.
  5. Hungarian matcher uses rank to disambiguate near-symmetric leaves.
  6. Unequal-cardinality pred/target → returns ``min(|pred|, |target|)``
     real pairs.
  7. ``per_cp_distance`` shape and zero-on-identity.
  8. Reduction modes (mean / sum / mean_per_cp).
"""
from __future__ import annotations

import numpy as np
import pytest

from dart.coupling.experimental.losses import (
    cp_l2_loss,
    hungarian_leaf_match,
    leaf_arc_length,
    leaf_centroid,
    per_cp_distance,
)

N_U, N_V = 11, 5


def _synth_leaf(
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
    length_cm: float = 50.0,
    half_width_cm: float = 3.0,
    twist_deg: float = 0.0,
    sag_cm: float = 5.0,
) -> np.ndarray:
    """Build a smooth canonical (11, 5, 3) CP grid for a synthetic leaf."""
    us = np.linspace(0.0, 1.0, N_U)
    vs = np.linspace(-1.0, 1.0, N_V)
    cps = np.empty((N_U, N_V, 3), dtype=np.float64)
    twist = np.deg2rad(twist_deg)
    for i, u in enumerate(us):
        rot = twist * u
        x_axis = np.array([1.0, 0.0, 0.0])
        y_axis = np.array([0.0, np.cos(rot), np.sin(rot)])
        sag = -sag_cm * 4.0 * u * (1.0 - u)
        for j, v in enumerate(vs):
            w = half_width_cm * (1.0 - 0.4 * u * u)
            p = u * length_cm * x_axis + v * w * y_axis + np.array([0.0, 0.0, sag])
            cps[i, j] = np.asarray(origin, dtype=np.float64) + p
    return cps


# ---------------------------------------------------------------------------
# 1. Identity → zero loss
# ---------------------------------------------------------------------------
def test_identity_gives_zero_loss():
    cps = {0: _synth_leaf(), 1: _synth_leaf(origin=(0, 10, 0))}
    loss = cp_l2_loss(cps, cps, [(0, 0), (1, 1)])
    assert loss == 0.0


def test_identity_per_cp_zero():
    cps = {0: _synth_leaf(), 1: _synth_leaf(origin=(0, 10, 0))}
    d = per_cp_distance(cps, cps, [(0, 0), (1, 1)])
    assert d.shape == (2, N_U, N_V)
    assert np.allclose(d, 0.0)


# ---------------------------------------------------------------------------
# 2. Known-translation loss
# ---------------------------------------------------------------------------
def test_known_translation_sum_loss():
    """Shifting a leaf by 2 cm in +x gives SSD = 4·N_U·N_V per matched leaf."""
    target = {0: _synth_leaf()}
    shifted = target[0].copy()
    shifted[..., 0] += 2.0
    pred = {0: shifted}

    loss_sum = cp_l2_loss(pred, target, [(0, 0)], reduction="sum")
    expected = 4.0 * N_U * N_V  # (2^2) per CP
    assert loss_sum == pytest.approx(expected)


def test_known_translation_mean_per_cp():
    target = {0: _synth_leaf()}
    shifted = target[0].copy()
    shifted[..., 0] += 2.0
    pred = {0: shifted}

    loss = cp_l2_loss(pred, target, [(0, 0)], reduction="mean_per_cp")
    # 4 cm² per CP averaged over all CPs.
    assert loss == pytest.approx(4.0)


def test_known_translation_per_cp_distance():
    target = {0: _synth_leaf()}
    shifted = target[0].copy()
    shifted[..., 0] += 2.0
    pred = {0: shifted}

    d = per_cp_distance(pred, target, [(0, 0)])
    assert d.shape == (1, N_U, N_V)
    assert np.allclose(d, 2.0)


# ---------------------------------------------------------------------------
# 3. Shape-mismatch
# ---------------------------------------------------------------------------
def test_shape_mismatch_raises():
    pred = {0: np.zeros((N_U, N_V, 3))}
    target = {0: np.zeros((N_U + 1, N_V, 3))}
    with pytest.raises(ValueError, match="shape mismatch"):
        cp_l2_loss(pred, target, [(0, 0)])
    with pytest.raises(ValueError, match="shape mismatch"):
        per_cp_distance(pred, target, [(0, 0)])


# ---------------------------------------------------------------------------
# 4. Hungarian identity on shuffled keys
# ---------------------------------------------------------------------------
def test_hungarian_recovers_identity_shuffled_keys():
    """Leaves at well-separated centroids should match correctly despite
    dict insertion-order shuffling."""
    leaves = {
        "A": _synth_leaf(origin=(0, 0, 5), length_cm=30),
        "B": _synth_leaf(origin=(0, 0, 15), length_cm=40),
        "C": _synth_leaf(origin=(0, 0, 25), length_cm=50),
    }
    # Shuffled target dict
    target = {"C": leaves["C"], "A": leaves["A"], "B": leaves["B"]}
    match = hungarian_leaf_match(leaves, target)
    match_dict = dict(match)
    for k in leaves:
        assert match_dict[k] == k, f"expected identity, got {match}"


# ---------------------------------------------------------------------------
# 5. Rank disambiguation
# ---------------------------------------------------------------------------
def test_hungarian_rank_disambiguates_similar_leaves():
    """Two near-identical leaves (same centroid and shape) disambiguated
    only by explicit rank tags."""
    base = _synth_leaf(origin=(0, 0, 10), length_cm=50)
    twin_a = base + np.array([0.0, 0.01, 0.0])  # tiny y-offset
    twin_b = base + np.array([0.0, -0.01, 0.0])

    pred = {"p_low": twin_a, "p_high": twin_b}
    target = {"t_low": twin_a, "t_high": twin_b}

    pred_ranks = {"p_low": 0, "p_high": 1}
    target_ranks = {"t_low": 0, "t_high": 1}

    match = hungarian_leaf_match(
        pred, target,
        weight_xyz=0.01, weight_arc=0.01, weight_rank=10.0,
        pred_ranks=pred_ranks, target_ranks=target_ranks,
    )
    match_dict = dict(match)
    assert match_dict["p_low"] == "t_low"
    assert match_dict["p_high"] == "t_high"


# ---------------------------------------------------------------------------
# 6. Unequal cardinality
# ---------------------------------------------------------------------------
def test_hungarian_unequal_cardinality():
    pred = {
        "a": _synth_leaf(origin=(0, 0, 5), length_cm=30),
        "b": _synth_leaf(origin=(0, 0, 15), length_cm=40),
    }
    target = {
        "x": _synth_leaf(origin=(0, 0, 5), length_cm=30),
    }
    match = hungarian_leaf_match(pred, target)
    assert len(match) == 1
    # The single pair must be the one with the closest centroid match.
    assert match[0][1] == "x"


def test_hungarian_empty_inputs_return_empty_list():
    assert hungarian_leaf_match({}, {}) == []
    leaf = _synth_leaf()
    assert hungarian_leaf_match({0: leaf}, {}) == []
    assert hungarian_leaf_match({}, {0: leaf}) == []


# ---------------------------------------------------------------------------
# 7. Feature helpers
# ---------------------------------------------------------------------------
def test_leaf_centroid_shape():
    base = _synth_leaf(origin=(0, 0, 0))
    shifted = _synth_leaf(origin=(1, 2, 3))
    c_base = leaf_centroid(base)
    c_shifted = leaf_centroid(shifted)
    assert c_shifted.shape == (3,)
    # Translating the leaf must translate the centroid by the same amount.
    assert np.allclose(c_shifted - c_base, np.array([1.0, 2.0, 3.0]), atol=1e-9)


def test_leaf_arc_length_positive_and_close_to_length():
    cps = _synth_leaf(length_cm=50.0, sag_cm=0.0)
    arc = leaf_arc_length(cps)
    # With zero sag, the midrib is a straight line from 0 to length_cm; CPs
    # are spaced uniformly at u-stations, arc should equal length_cm.
    assert arc == pytest.approx(50.0, rel=1e-6)


# ---------------------------------------------------------------------------
# 8. End-to-end: match + loss on synthetic plants
# ---------------------------------------------------------------------------
def test_end_to_end_match_loss_on_similar_plants():
    """Two "plants" with matching leaf counts and small per-leaf offsets:
    Hungarian should align them by rank, CP-L2 loss should be small."""
    pred = {
        i: _synth_leaf(origin=(0, 0, 5 + 10 * i), length_cm=30 + 5 * i)
        for i in range(4)
    }
    target = {
        i + 10: _synth_leaf(origin=(0.5, 0, 5 + 10 * i), length_cm=30 + 5 * i)
        for i in range(4)
    }
    match = hungarian_leaf_match(pred, target)
    loss = cp_l2_loss(pred, target, match, reduction="mean_per_cp")
    # Each CP is offset by 0.5 cm in +x → per-CP L2² = 0.25.
    assert loss == pytest.approx(0.25, rel=1e-6)
    # And the Hungarian pairs must follow rank order (keys are ints 0..3 ↔ 10..13).
    expected_pairs = {(i, i + 10) for i in range(4)}
    assert set(match) == expected_pairs
