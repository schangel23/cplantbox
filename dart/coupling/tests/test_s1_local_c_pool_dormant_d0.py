"""S1 buffered-carbon local pool pybind fixtures."""

from __future__ import annotations

import plantbox as pb

from dart.coupling.growth.grow import grow_plant


XML = "dart/coupling/data/maize_calibrated.xml"


def test_organ_local_c_pool_default_zero():
    plant = grow_plant(XML, simulation_time=10, seed=42)

    assert plant.getOrgans()
    assert all(o.local_C_pool_ == 0.0 for o in plant.getOrgans())


def test_pybind_local_c_pool_readwrite():
    plant = grow_plant(XML, simulation_time=10, seed=42)
    organ = plant.getOrgans()[0]

    organ.local_C_pool_ = 1.5
    assert organ.local_C_pool_ == 1.5
    organ.local_C_pool_ = 0.0
    assert organ.local_C_pool_ == 0.0
