"""NPZ-frame bit-identity regression for build_from_maizefield3d.

The baked canonical_leaf_library.npz (and the maize_calibrated.xml
surface_cps derived from it) were built with an early version of
to_local_frame that did NOT rotate each leaf's tip into the canonical
(+y, +z) half-plane and did NOT apply the _default_tip_bounds filter.
Mixing that frame with the post-rotate frame on the same plant produces
a frame mismatch when the NPZ-derived library is used to rebake XML CPs.

This test pins the recovered NPZ-build recipe so future refactors of
to_local_frame don't silently break frame compatibility with the XML.

Recipe: tip_canonical_rotate=False, tip_bounds=no_op,
normalize_arc=False, reducer='median'.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dart.coupling.geometry.canonical_library import (
    _default_canonical_json,
    _passes_lmax_smoothness,
    build_from_maizefield3d,
    load_library,
)


_LIB_NPZ = (
    Path(__file__).resolve().parents[1] / "data" / "canonical_leaf_library.npz"
)


def _no_op_filter(_pos: int) -> tuple[float, float, float, float]:
    return (-1e9, 1e9, -1e9, -1e9)


@pytest.mark.skipif(
    not _default_canonical_json().exists(),
    reason="MaizeField3d canonical_cps.json not available in this checkout",
)
def test_npz_compat_build_bit_identical_to_baked_npz():
    """build_from_maizefield3d in NPZ-compat mode reproduces the NPZ exactly.

    The NPZ has 16 positions: positions 0–1 are stub copies of the original
    pos 0 (prepend_stub_positions.py), positions 2–15 are original 0–13.
    The build path reconstructs the original 14 positions, so we compare
    NPZ[2:16] against build[0:14].
    """
    npz = load_library(_LIB_NPZ)
    cps_npz = np.asarray(npz["cps_local"])  # (16, N_U, N_V, 3)
    counts_npz = np.asarray(npz["counts"])

    lib = build_from_maizefield3d(
        _default_canonical_json(),
        reducer="median",
        normalize_arc=False,
        tip_bounds=_no_op_filter,
        tip_canonical_rotate=False,
    )
    cps_b = np.asarray(lib["cps_local"])
    counts_b = np.asarray(lib["counts"])

    assert cps_b.shape[0] == cps_npz.shape[0] - 2, (
        f"expected build to have {cps_npz.shape[0] - 2} positions "
        f"(NPZ minus 2 prepended stubs); got {cps_b.shape[0]}"
    )

    np.testing.assert_array_equal(
        counts_b,
        counts_npz[2:],
        err_msg="counts diverged → tip_bounds filter is rejecting plants",
    )

    np.testing.assert_allclose(
        cps_b,
        cps_npz[2:],
        atol=0.0,
        rtol=0.0,
        err_msg=(
            "build_from_maizefield3d(NPZ-compat) drifted from the baked "
            "NPZ — to_local_frame's pre-aggregation transform changed. "
            "Any drift here will break XML CPs that were originally baked "
            "from this frame."
        ),
    )


def test_donor_quality_filter_rejects_known_bad_curves():
    """Both panel-seed scan-noise donors must be rejected at tol=0.20.

    Concrete curves drawn from the live MaizeField3D pool — seed-1's
    chosen donor jumps from pos 8 = 30.9 cm to pos 9 = 47.7 cm (descent
    leg goes UP +54 %); seed-4's donor has a single upper-rank spike at
    pos 12 = 88.4 cm that would displace a naive argmax-based peak
    detector. The smoothed-peak filter catches both.
    """
    bad_seed1 = {0: 46.8, 1: 55.4, 2: 61.1, 3: 56.7, 4: 56.0, 5: 50.3,
                 6: 46.3, 7: 42.3, 8: 30.9, 9: 47.7, 10: 39.2, 11: 39.0,
                 12: 30.7, 13: 22.0}
    bad_seed4 = {0: 51.0, 1: 61.8, 2: 70.6, 3: 77.6, 4: 74.5, 5: 78.0,
                 6: 70.6, 7: 77.1, 8: 81.3, 9: 79.3, 10: 71.3, 11: 62.3,
                 12: 88.4, 13: 52.6}
    assert not _passes_lmax_smoothness(bad_seed1, max_jump_frac=0.20)
    assert not _passes_lmax_smoothness(bad_seed4, max_jump_frac=0.20)


def test_donor_quality_filter_accepts_clean_u_curve():
    """A monotonic U-curve must pass at the default tolerance."""
    clean = {0: 64.0, 1: 77.2, 2: 86.3, 3: 95.9, 4: 102.0, 5: 97.2,
             6: 106.9, 7: 104.1, 8: 110.0, 9: 104.5, 10: 97.1, 11: 80.4,
             12: 78.3, 13: 71.2}
    assert _passes_lmax_smoothness(clean, max_jump_frac=0.20)


def test_donor_quality_filter_rejects_too_few_positions():
    """Plants with fewer than min_positions leaves are rejected."""
    sparse = {0: 50.0, 1: 60.0, 2: 70.0}
    assert not _passes_lmax_smoothness(sparse, max_jump_frac=0.20)


@pytest.mark.skipif(
    not _default_canonical_json().exists(),
    reason="MaizeField3d canonical_cps.json not available in this checkout",
)
def test_donor_quality_filter_shrinks_pool():
    """Filter on must reduce per-position counts vs filter off."""
    no_op = lambda _p: (-1e9, 1e9, -1e9, -1e9)  # noqa: E731
    lib_off = build_from_maizefield3d(
        _default_canonical_json(), reducer="median", normalize_arc=False,
        tip_bounds=no_op, tip_canonical_rotate=False,
    )
    lib_on = build_from_maizefield3d(
        _default_canonical_json(), reducer="median", normalize_arc=False,
        tip_bounds=no_op, tip_canonical_rotate=False,
        donor_quality_filter=0.20,
    )
    assert lib_on["counts"].sum() < lib_off["counts"].sum(), (
        "donor_quality_filter=0.20 should reject some leaves"
    )
    assert lib_on["counts"][0] >= int(0.6 * lib_off["counts"][0]), (
        "filter at 0.20 should keep most plants at pos 0 (>60%)"
    )


def test_tip_canonical_rotate_default_keeps_legacy_frame():
    """Default tip_canonical_rotate=True must still yield the rotated frame.

    Sanity check on the new flag's default — callers that don't opt in
    should get the post-rotate behaviour they had before this fix.
    """
    if not _default_canonical_json().exists():
        pytest.skip("MaizeField3d canonical_cps.json not available")
    lib_default = build_from_maizefield3d(
        _default_canonical_json(), reducer="median", normalize_arc=False,
        tip_bounds=_no_op_filter,
    )
    lib_compat = build_from_maizefield3d(
        _default_canonical_json(), reducer="median", normalize_arc=False,
        tip_bounds=_no_op_filter, tip_canonical_rotate=False,
    )
    # Z-column is invariant under tip rotation (rotation is about z-axis).
    np.testing.assert_allclose(
        lib_default["cps_local"][..., 2],
        lib_compat["cps_local"][..., 2],
        atol=1e-9,
        err_msg="z-column should be unaffected by tip_canonical_rotate",
    )
    # The xy magnitudes must differ (rotation collects droop into +y, so
    # rotated peak_y > non-rotated peak_y on a multi-plant median).
    npz_compat_peak_y = np.abs(lib_compat["cps_local"][..., 1]).max()
    rotated_peak_y = np.abs(lib_default["cps_local"][..., 1]).max()
    assert rotated_peak_y > npz_compat_peak_y * 1.05, (
        f"expected rotated peak_y to exceed NPZ-compat by >5%; "
        f"got rotated={rotated_peak_y:.3f} vs npz_compat={npz_compat_peak_y:.3f}"
    )
