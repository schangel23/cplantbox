"""Unit tests for Phase 2a: MaizeField3D → canonical 11×5 CP resampler.

Covers:
  1. Output shape and dtype.
  2. LSQ fit residual on real `.dat` input (must beat the 0.1 cm spec on a
     clean sample leaf).
  3. Axis mapping — the resampled midrib (v=0.5) should run collar→tip,
     matching the source's longitudinal direction.
  4. Orientation convention — result obeys ``enforce_orientation`` (v=0 edge
     on +x_local side).

Skips gracefully if the MaizeField3D `.dat` fixtures are not available
(e.g., on the server where data lives elsewhere, or on CI).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_VAULT = Path(__file__).resolve().parents[4]
_DAT_DIR = _VAULT / "Resources" / "MaizeField3d" / "FielGrwon_ZeaMays_Reconstructed_Surface_dat"
_READER_DIR = _VAULT / "Resources" / "MaizeField3d"


def _ensure_reader_on_path() -> None:
    if str(_READER_DIR) in sys.path:
        return
    sys.path.insert(0, str(_READER_DIR))


@pytest.fixture(scope="module")
def sample_leaf():
    """Parsed first leaf of 0001.dat — skip if data missing."""
    _ensure_reader_on_path()
    if not _DAT_DIR.is_dir():
        pytest.skip(f"MaizeField3D .dat fixtures not found at {_DAT_DIR}")
    from extract_maizefield3d_morphology import parse_nurbs_dat
    leaves = parse_nurbs_dat(str(_DAT_DIR / "0001.dat"))
    if not leaves:
        pytest.skip("0001.dat contained no leaves")
    return leaves[0]


def test_resample_output_shape_and_units(sample_leaf):
    _ensure_reader_on_path()
    from maizefield3d_nurbs_reader import resample_to_canonical

    cps = resample_to_canonical(sample_leaf)
    assert isinstance(cps, np.ndarray)
    assert cps.shape == (11, 5, 3)
    assert cps.dtype == np.float64

    # Resampled output is in cm; a mature maize leaf sits at O(10²) cm span,
    # so mean |z| should be nonzero and the midrib should extend at least
    # 20 cm end-to-end.
    midrib = cps[:, 2, :]
    total_len = float(np.linalg.norm(np.diff(midrib, axis=0), axis=1).sum())
    assert total_len > 20.0, f"midrib length {total_len:.1f} cm unrealistically small"


def test_resample_residual_under_spec(sample_leaf):
    """Spec: LSQ residual must be < 0.1 cm max per clean MaizeField3D leaf."""
    _ensure_reader_on_path()
    from maizefield3d_nurbs_reader import resample_to_canonical

    _, residual = resample_to_canonical(sample_leaf, return_residual=True)
    # Tight bound for the quality-controlled first-plant/first-leaf sample.
    assert residual["rmse_cm"] < 0.05
    assert residual["max_err_cm"] < 0.10


def test_resample_midrib_runs_collar_to_tip(sample_leaf):
    """u=0 should be the collar (stem-side); u=1 should be the tip.

    Control points are handles, not curve samples — so we evaluate the
    canonical surface at ``(u, v=0.5)`` and compare to the source midrib
    at ``(u_src=0.5, v_src=u_c)`` (axis swap).
    """
    _ensure_reader_on_path()
    from dart.coupling.geometry.canonical_cp_grid import (
        cp_grid_to_geomdl_surface,
    )
    from maizefield3d_nurbs_reader import (
        leaf_dict_to_geomdl_surface,
        resample_to_canonical,
    )

    # Source midrib at u_src=0.5 along v_src in [0, 1]
    surf_src = leaf_dict_to_geomdl_surface(sample_leaf)
    us = np.linspace(0, 1, 21)
    src_pts_cm = np.array(surf_src.evaluate_list(
        [[0.5, float(t)] for t in us]
    )) * 100.0

    # Canonical midrib at v=0.5 along u in [0, 1]
    cps = resample_to_canonical(sample_leaf, apply_orientation=False)
    surf_canon = cp_grid_to_geomdl_surface(cps)
    canon_pts = np.array(surf_canon.evaluate_list(
        [[float(u), 0.5] for u in us]
    ))

    diff = np.linalg.norm(canon_pts - src_pts_cm, axis=-1)
    # Matches the resampler's documented residual (< 0.1 cm per leaf on
    # clean MaizeField3D input).
    assert diff.max() < 0.2, (
        f"canonical midrib does not reproduce source midrib: "
        f"max diff {diff.max():.4f} cm"
    )


def test_resample_orientation_matches_convention(sample_leaf):
    """After ``enforce_orientation``, re-applying is a no-op (idempotent)."""
    _ensure_reader_on_path()
    from dart.coupling.geometry.canonical_cp_grid import enforce_orientation
    from maizefield3d_nurbs_reader import resample_to_canonical

    cps = resample_to_canonical(sample_leaf, apply_orientation=True)
    cps_again = enforce_orientation(cps)
    assert np.array_equal(cps, cps_again), "orientation not idempotent"
