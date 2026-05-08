"""NPZ-frame bit-identity regression for build_from_maizefield3d.

The baked canonical_leaf_library.npz (and the maize_calibrated.xml
surface_cps derived from it) were built with an early version of
to_local_frame that did NOT rotate each leaf's tip into the canonical
(+y, +z) half-plane and did NOT apply the _default_tip_bounds filter.
Mixing that frame with the post-rotate frame on the same plant produces
crumbled meshes (cp_swap → XML frame mismatch).

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
            "cp_swap injects donor CPs into XML CPs that live in the NPZ "
            "frame; any drift here will crumble swapped meshes."
        ),
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
