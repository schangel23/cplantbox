"""S2 buffered-carbon plant reserve pybind fixtures."""

from __future__ import annotations

import pytest

from dart.coupling.growth.grow import grow_plant


MAIZE_XML = "dart/coupling/data/maize_calibrated.xml"
WHEAT_XML = "dart/coupling/data/wheat_calibrated.xml"


def test_plant_reserve_pool_default_zero():
    plant = grow_plant(MAIZE_XML, simulation_time=130, seed=42)

    assert plant.transient_reserve_pool_ == 0.0


def test_reserve_pool_readwrite():
    plant = grow_plant(MAIZE_XML, simulation_time=10, seed=42)

    plant.transient_reserve_pool_ = 3.0
    assert plant.transient_reserve_pool_ == 3.0
    plant.transient_reserve_pool_ = 0.0
    assert plant.transient_reserve_pool_ == 0.0


def test_s2_seed_rp_reserve_defaults():
    plant = grow_plant(WHEAT_XML, simulation_time=1, seed=42)
    srp = plant.getOrganRandomParameter(1, 0)

    assert srp.reserve_capacity_factor == pytest.approx(0.04)
    assert srp.starch_remob_rate == pytest.approx(2.0)
    assert srp.starch_storage_efficiency == pytest.approx(0.95)
    assert srp.starch_remob_efficiency == pytest.approx(0.98)
