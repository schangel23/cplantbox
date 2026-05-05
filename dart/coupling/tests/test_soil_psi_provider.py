"""Tests for dart.coupling.hydraulics.soil_psi providers (Phase 2)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def fixed():
    from dart.coupling.hydraulics.soil_psi import FixedSoilPsi
    return FixedSoilPsi(psi_cm=-500.0)


def _legacy_linspace(psi_cm, depth_cm):
    return np.linspace(psi_cm, psi_cm - depth_cm, depth_cm)


@pytest.mark.parametrize("psi", [-100.0, -300.0, -500.0, -1500.0])
@pytest.mark.parametrize("depth", [50, 100, 200])
def test_fixed_bit_identical_with_legacy_linspace(psi, depth):
    """FixedSoilPsi.get_profile must match np.linspace bit-for-bit.

    The legacy expression was the production hardcoded site at
    ``coupled.py:226``, ``iterative.py:450``, ``grow.py:613``:
        ``p_s = np.linspace(soil_psi_cm, soil_psi_cm - depth, depth)``
    """
    from dart.coupling.hydraulics.soil_psi import FixedSoilPsi

    legacy = _legacy_linspace(psi, depth)
    new = FixedSoilPsi(psi_cm=psi).get_profile(t_days=0.0, depth_cm=depth)
    assert np.array_equal(legacy, new)


def test_fixed_independent_of_t_days(fixed):
    p0 = fixed.get_profile(0.0, 100)
    p100 = fixed.get_profile(100.0, 100)
    assert np.array_equal(p0, p100)


def test_fixed_update_is_noop(fixed):
    before = fixed.get_profile(0.0, 100)
    fixed.update(t_days=5.0, sink_per_cell=np.full(100, -1e-3))
    after = fixed.get_profile(5.0, 100)
    assert np.array_equal(before, after)


def test_factory_dispatch():
    from dart.coupling.hydraulics.soil_psi import (
        BucketSoilPsi, FixedSoilPsi, make_provider,
    )
    assert isinstance(make_provider("fixed", soil_psi_cm=-500), FixedSoilPsi)
    assert isinstance(make_provider("bucket", soil_psi_cm=-300), BucketSoilPsi)
    with pytest.raises(ValueError):
        make_provider("nonsense")


def test_bucket_drying_monotonic():
    from dart.coupling.hydraulics.soil_psi import BucketSoilPsi
    b = BucketSoilPsi(psi_init_cm=-200.0, psi_target_cm=-1500.0,
                      tau_days=10.0)
    p0 = b.get_profile(0.0, 100)
    p10 = b.get_profile(10.0, 100)
    p30 = b.get_profile(30.0, 100)
    # ψ becomes more negative over time (drying toward target)
    assert p10[0] < p0[0]
    assert p30[0] < p10[0]
    # Asymptotes to target at 100 days (well past 3*tau)
    p100 = b.get_profile(100.0, 100)
    assert abs(p100[0] - (-1500.0)) < 5.0


_DUMUX_BIND = Path(
    "/home/lukas/PHD/dumux-build/dumux/dumux-rosi/build-cmake/cpp/python_binding"
)
_DUMUX_AVAILABLE = (_DUMUX_BIND / "rosi_richards.cpython-314-x86_64-linux-gnu.so").exists()


@pytest.mark.skipif(not _DUMUX_AVAILABLE, reason="rosi_richards.so not built")
def test_dumux_constructs_and_advances():
    from dart.coupling.hydraulics.soil_psi import DumuxSoilPsi

    dum = DumuxSoilPsi(depth_cm=100, n_cells_z=100, psi_init_cm=-100.0,
                       verbose=False)
    p0 = dum.get_profile(t_days=0.0, depth_cm=100)
    p10 = dum.get_profile(t_days=10.0, depth_cm=100)

    assert p0.shape == (100,)
    assert np.all(np.isfinite(p0))
    assert np.all(np.isfinite(p10))
    # Free-drainage from -100 cm: top should dry slightly, bottom is quiescent
    assert p10[0] <= p0[0] + 1e-9
    assert abs(p10[-1] - p0[-1]) < 1.0


@pytest.mark.skipif(not _DUMUX_AVAILABLE, reason="rosi_richards.so not built")
def test_dumux_get_profile_rejects_grid_mismatch():
    from dart.coupling.hydraulics.soil_psi import DumuxSoilPsi
    dum = DumuxSoilPsi(depth_cm=100, n_cells_z=100, psi_init_cm=-200.0,
                       verbose=False)
    with pytest.raises(ValueError):
        dum.get_profile(t_days=0.0, depth_cm=50)


def test_provider_protocol_conformance():
    """Each concrete provider satisfies the SoilPsiProvider Protocol."""
    from dart.coupling.hydraulics.soil_psi import (
        BucketSoilPsi, FixedSoilPsi, SoilPsiProvider,
    )
    # Protocol is duck-typed; these calls should be runtime-callable
    for prov in [FixedSoilPsi(-500.0), BucketSoilPsi()]:
        prof = prov.get_profile(0.0, 100)
        assert prof.shape == (100,)
        prov.update(0.0, np.zeros(100))  # must not raise


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
