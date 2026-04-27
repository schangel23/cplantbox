"""Tests for the random-draw reducer + muted library-path deformations.

Covers:
  1. ``aggregate_library(reducer='draw', draw_seed=s)`` is reproducible for the
     same seed, differs across seeds, and outputs the same shape as median.
  2. Drawn rows are exact copies of per-position input stacks (the reducer
     does not mix plants — it picks one).
  3. ``build_from_maizefield3d`` stamps the seed into the returned dict.
  4. ``loft_leaf_nurbs`` library branch applies ``_apply_deformations`` when
     wave/twist/curl params are present on the organ dict, and the deformed
     mesh differs from the undeformed mesh at sub-centimetre amplitude.
"""
from __future__ import annotations

import numpy as np
import pytest

from dart.coupling.geometry.canonical_library import (
    aggregate_library,
    build_from_maizefield3d,
    _default_canonical_json,
)
from dart.coupling.geometry.canonical_cp_grid import N_U, N_V
from dart.coupling.geometry.nurbs_blade import loft_leaf_nurbs


# ---------------------------------------------------------------------------
# 1–3. random-draw reducer
# ---------------------------------------------------------------------------
def _synthetic_per_position(n_plants: int = 8, n_pos: int = 4):
    """Build a deterministic per-position stack so drawn samples are
    identifiable."""
    rng = np.random.default_rng(0)
    out: dict[int, list[np.ndarray]] = {}
    for pos in range(n_pos):
        stack = []
        for p in range(n_plants):
            # Encode (pos, p) in the CP values so the draw is identifiable.
            grid = rng.normal(size=(N_U, N_V, 3)) + 100.0 * pos + p
            stack.append(grid)
        out[pos] = stack
    return out


def test_draw_reducer_reproducible_same_seed():
    per_pos = _synthetic_per_position()
    a, _, _ = aggregate_library(per_pos, reducer="draw", draw_seed=42)
    b, _, _ = aggregate_library(per_pos, reducer="draw", draw_seed=42)
    assert np.array_equal(a, b)


def test_draw_reducer_differs_across_seeds():
    per_pos = _synthetic_per_position()
    a, _, _ = aggregate_library(per_pos, reducer="draw", draw_seed=42)
    b, _, _ = aggregate_library(per_pos, reducer="draw", draw_seed=99)
    assert not np.array_equal(a, b)


def test_draw_reducer_matches_median_shape():
    per_pos = _synthetic_per_position()
    drawn, counts_d, _ = aggregate_library(per_pos, reducer="draw", draw_seed=7)
    median, counts_m, _ = aggregate_library(per_pos, reducer="median")
    assert drawn.shape == median.shape
    assert np.array_equal(counts_d, counts_m)


def test_draw_reducer_picks_actual_sample():
    """Each output row must equal one of the plants in that position's stack,
    not an interpolation."""
    per_pos = _synthetic_per_position()
    drawn, _, _ = aggregate_library(per_pos, reducer="draw", draw_seed=11)
    positions = sorted(per_pos.keys())
    for i, pos in enumerate(positions):
        stack = np.stack(per_pos[pos], axis=0)
        matches = np.any(np.all(stack == drawn[i], axis=(1, 2, 3)))
        assert matches, f"position {pos} drawn row not found in stack"


def test_draw_reducer_requires_seed():
    per_pos = _synthetic_per_position()
    with pytest.raises(ValueError):
        aggregate_library(per_pos, reducer="draw")  # no seed


def test_build_from_maizefield3d_stamps_seed():
    src = _default_canonical_json()
    if not src.exists():  # CI may not have the 520-plant JSON
        pytest.skip(f"{src} not available")
    lib = build_from_maizefield3d(src, reducer="draw", draw_seed=123)
    assert lib["reducer"] == "draw"
    assert lib["draw_seed"] == 123
    assert lib["cps_local"].shape[1:] == (N_U, N_V, 3)


def test_build_from_maizefield3d_median_seed_is_none():
    src = _default_canonical_json()
    if not src.exists():
        pytest.skip(f"{src} not available")
    lib = build_from_maizefield3d(src, reducer="median")
    assert lib["reducer"] == "median"
    assert lib["draw_seed"] is None


# ---------------------------------------------------------------------------
# 4. muted library-path deformations
# ---------------------------------------------------------------------------
def _lib_organ(cps_local: np.ndarray, deformed: bool):
    organ = {
        "type": "leaf",
        "organ_id": 3,
        "surface_cps_local": cps_local.copy(),
        "collar_pos": np.array([0.0, 0.0, 50.0]),
        "collar_tangent": np.array([1.0, 0.0, 0.0]),
        "parent_tangent": np.array([0.0, 0.0, 1.0]),
        "mature_length": 50.0,
        "current_length": 50.0,
        "skeleton": np.column_stack([
            np.linspace(0, 50, 20), np.zeros(20), np.full(20, 50.0),
        ]),
    }
    if deformed:
        organ.update(
            wave_normal_amp=0.5,
            wave_normal_freq=3.0,
            wave_normal_phase=0.3,
            twist_max=0.15,
            curl_amp=0.4,
            curl_freq=1.2,
            curl_phase=1.0,
            curl_onset=0.15,
            ramp_onset=0.15,
            maturity_fraction=1.0,
        )
    return organ


def _sample_cp_grid():
    src = _default_canonical_json()
    if not src.exists():
        pytest.skip(f"{src} not available")
    lib = build_from_maizefield3d(src, reducer="median")
    mid = lib["cps_local"].shape[0] // 2
    return np.asarray(lib["cps_local"][mid], dtype=np.float64)


def test_library_branch_applies_deformations():
    cps = _sample_cp_grid()
    r_plain = loft_leaf_nurbs(_lib_organ(cps, deformed=False))
    r_def = loft_leaf_nurbs(_lib_organ(cps, deformed=True))
    # Same tessellation shape, but different vertex positions.
    assert r_plain[0].shape == r_def[0].shape
    assert r_plain[3].shape == r_def[3].shape
    assert not np.allclose(r_plain[0], r_def[0])
    assert np.all(np.isfinite(r_def[0]))


def test_library_branch_deformation_amplitude_is_muted():
    """Muted deformations on a 50 cm leaf should displace vertices by a few
    cm at most — not by the full leaf length."""
    cps = _sample_cp_grid()
    r_plain = loft_leaf_nurbs(_lib_organ(cps, deformed=False))
    r_def = loft_leaf_nurbs(_lib_organ(cps, deformed=True))
    disp = np.linalg.norm(r_plain[0] - r_def[0], axis=1)
    # 50 cm leaf — a few cm of ruffle is the intent; >10 cm would mean a bug.
    assert disp.max() < 5.0, f"unexpected large displacement: {disp.max():.2f} cm"
    assert disp.mean() > 0.0


def test_library_branch_mesh_is_finite_with_compound_sheath():
    """Deformations on the compound-sheath path must not produce NaNs."""
    cps = _sample_cp_grid()
    organ = _lib_organ(cps, deformed=True)
    organ["stem_radius_cm"] = 1.5
    organ["sheath_length_cm"] = 5.0
    r = loft_leaf_nurbs(organ)
    assert np.all(np.isfinite(r[0]))
    assert r[3].shape[0] > 0
