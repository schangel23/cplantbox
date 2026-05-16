"""§S9 sink-fullness feedback fixtures (Plan §11.3 escalation).

Fast unit suite for the L-Peach-style Vmaxloading downregulation that
relieves PiafMunch CVODE-feasibility stress under sustained Path B
operation. The acceptance gates live in:

  - this file (unit + XML round-trip + kwarg-default checks)
  - test_s8_d0_6xml_invariance (§S9 must preserve D.0 baselines because
    sink_feedback_enabled defaults False and the new XML field is
    dormant in non-PM code paths)
  - run_g5 / test_s7_calibration (slow gates re-run with
    --sink-feedback-enabled to confirm PM-fail-rate drop)
"""

from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest

from dart.coupling.carbon import pm_substep
from dart.coupling.carbon.pm_substep import (
    DRY_MATTER_FRACTION,
    _reserve_capacity_total,
    _sink_feedback_multiplier,
    solve_carbon_partitioning_pm,
)

CPLANTBOX_ROOT = Path(__file__).resolve().parents[3]
MAIZE_XML = CPLANTBOX_ROOT / "dart/coupling/data/maize_calibrated.xml"


class _Organ:
    def __init__(self, volume):
        self._volume = volume

    def orgVolume(self, _length, _realized):
        return self._volume


def _make_plant(*, factor=1.0, theta=0.8, volume=20.0, reserve=0.0):
    plant = SimpleNamespace()
    plant.transient_reserve_pool_ = reserve
    srp = SimpleNamespace(
        reserve_capacity_factor=factor,
        sink_feedback_theta_full=theta,
    )
    plant.getOrganRandomParameter = lambda ot, idx=None: (
        srp if idx is not None else [srp]
    )
    plant.getOrgans = lambda *a, **kw: [_Organ(volume)]
    return plant


def test_s9_capacity_formula():
    """capacity = reserve_capacity_factor × Σ structural_dry_mass."""
    plant = _make_plant(factor=1.0, volume=20.0)
    cap = _reserve_capacity_total(plant)
    expected = 1.0 * 20.0 * DRY_MATTER_FRACTION
    assert cap == pytest.approx(expected, rel=1e-9)


def test_s9_capacity_zero_when_factor_zero():
    """factor=0 → capacity 0 → multiplier is no-op (1.0)."""
    plant = _make_plant(factor=0.0, reserve=99.0)
    assert _reserve_capacity_total(plant) == 0.0
    assert _sink_feedback_multiplier(plant, 0.8) == 1.0


def test_s9_multiplier_below_threshold():
    """sat ≤ theta_full → no damping (multiplier 1.0)."""
    # cap = 1.0 * 20 * 0.15 = 3.0; sat = 1.5/3 = 0.5
    plant = _make_plant(factor=1.0, volume=20.0, reserve=1.5)
    assert _sink_feedback_multiplier(plant, theta_full=0.8) == 1.0


def test_s9_multiplier_at_threshold():
    """sat == theta_full → still no damping (boundary inclusive).

    The reserve value 3.0×0.8 may round to 2.4000000000000004 so sat
    drifts microscopically above 0.8 — multiplier ends up at 1−ε rather
    than exactly 1. Treated as no-op via approx tolerance.
    """
    plant = _make_plant(factor=1.0, volume=20.0, reserve=3.0 * 0.8)
    assert _sink_feedback_multiplier(plant, theta_full=0.8) == pytest.approx(
        1.0, abs=1e-6
    )


def test_s9_multiplier_linear_between():
    """sat=0.9, theta=0.8 → (1-0.9)/(1-0.8) = 0.5."""
    plant = _make_plant(factor=1.0, volume=20.0, reserve=3.0 * 0.9)
    mult = _sink_feedback_multiplier(plant, theta_full=0.8)
    assert mult == pytest.approx(0.5, rel=1e-9)


def test_s9_multiplier_saturation():
    """sat=1.0 → multiplier 0 (no loading at all)."""
    plant = _make_plant(factor=1.0, volume=20.0, reserve=3.0)
    assert _sink_feedback_multiplier(plant, theta_full=0.8) == 0.0


def test_s9_multiplier_oversaturated_clamped():
    """sat > 1.0 → still clamped to 0 (no negative loading)."""
    plant = _make_plant(factor=1.0, volume=20.0, reserve=99.0)
    assert _sink_feedback_multiplier(plant, theta_full=0.8) == 0.0


def test_s9_theta_one_is_noop():
    """theta_full=1.0 → divide-by-zero guard kicks in (multiplier 1.0)."""
    plant = _make_plant(factor=1.0, volume=20.0, reserve=99.0)
    assert _sink_feedback_multiplier(plant, theta_full=1.0) == 1.0


def test_s9_kwarg_default_off_for_d0_invariance():
    """sink_feedback_enabled default must be False; threading the kwarg
    through solve_carbon_partitioning_pm must not change pre-S9 behaviour
    until a caller opts in. This is the D.0 invariance guarantee that
    test_s8_d0_6xml_invariance relies on.
    """
    sig = inspect.signature(solve_carbon_partitioning_pm)
    assert "sink_feedback_enabled" in sig.parameters
    assert sig.parameters["sink_feedback_enabled"].default is False


def test_s9_xml_round_trip():
    """maize_calibrated.xml carries sink_feedback_theta_full=0.80 via SRP
    bindParameter; round-trip preserves the value.
    """
    pb = pytest.importorskip("plantbox")
    plant = pb.MappedPlant()
    plant.readParameters(str(MAIZE_XML))
    plant.initialize(False)
    srp = plant.getOrganRandomParameter(1, 0)
    assert hasattr(srp, "sink_feedback_theta_full")
    assert float(srp.sink_feedback_theta_full) == pytest.approx(0.80)


def test_s9_srp_bind_default_when_xml_omits_field():
    """A SeedRandomParameter constructed standalone (no XML) defaults to
    0.80 — proves the C++ default lives at the field declaration so any
    XML without the entry still gets the same value (legacy-XML safety).
    """
    pb = pytest.importorskip("plantbox")
    organism = pb.Organism()
    srp = pb.SeedRandomParameter(organism)
    assert float(srp.sink_feedback_theta_full) == pytest.approx(0.80)


def test_s9_helpers_exported():
    """Helpers must be importable from pm_substep for downstream code
    (calibration sweep, slow_s9 acceptance test)."""
    assert callable(pm_substep._sink_feedback_multiplier)
    assert callable(pm_substep._reserve_capacity_total)
