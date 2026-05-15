"""S6 buffered-carbon XML and pybind fixtures."""

from __future__ import annotations

import pytest

from dart.coupling.carbon.pm_substep import _pool_capacity
from dart.coupling.growth.grow import grow_plant


MAIZE_XML = "dart/coupling/data/maize_calibrated.xml"
WHEAT_XML = "dart/coupling/data/wheat_calibrated.xml"


def test_s6_c_cost_readwrite_leaf_stem():
    plant = grow_plant(MAIZE_XML, simulation_time=1, seed=42)
    leaf_rp = plant.getOrganRandomParameter(4, 2)
    stem_rp = plant.getOrganRandomParameter(3, 1)

    assert leaf_rp.c_cost_per_cm == pytest.approx(0.35)
    assert stem_rp.c_cost_per_cm == pytest.approx(0.55)

    leaf_rp.c_cost_per_cm = 0.41
    stem_rp.c_cost_per_cm = 0.61
    assert leaf_rp.c_cost_per_cm == pytest.approx(0.41)
    assert stem_rp.c_cost_per_cm == pytest.approx(0.61)


def test_s6_capacity_factor_leaf_xml():
    plant = grow_plant(MAIZE_XML, simulation_time=1, seed=42)

    for sub_type in range(2, 17):
        leaf_rp = plant.getOrganRandomParameter(4, sub_type)
        assert leaf_rp.local_C_pool_capacity_factor == pytest.approx(0.5)


def test_s6_capacity_zero_default_on_wheat():
    plant = grow_plant(WHEAT_XML, simulation_time=1, seed=42)

    for organ_type in (2, 3, 4):
        for rp in plant.getOrganRandomParameter(organ_type):
            if rp is not None:
                assert rp.local_C_pool_capacity_factor == pytest.approx(0.0)


def test_s6_reserve_params_in_seed_xml():
    plant = grow_plant(MAIZE_XML, simulation_time=1, seed=42)
    srp = plant.getOrganRandomParameter(1, 0)

    assert srp.reserve_capacity_factor == pytest.approx(0.04)
    assert srp.starch_remob_rate == pytest.approx(2.0)
    assert srp.starch_storage_efficiency == pytest.approx(0.95)
    assert srp.starch_remob_efficiency == pytest.approx(0.98)


def test_s6_capacity_uses_dry_matter_proxy():
    class Organ:
        def orgVolume(self, _length, _realized):
            return 10.0

    class RP:
        local_C_pool_capacity_factor = 0.5

    assert _pool_capacity(Organ(), RP()) == pytest.approx(0.75)
