"""S3 buffered dispatch fixtures for CWLimitedGrowth."""

from __future__ import annotations

import math

import pytest
import plantbox as pb

from dart.coupling.growth.grow import grow_plant
from dart.coupling.growth.carbon_growth import enable_cw_limited_growth


XML = "dart/coupling/data/maize_calibrated.xml"


def _cw_organs(plant):
    return [
        (o, o.getOrganRandomParameter(), o.getOrganRandomParameter().f_gf)
        for o in plant.getOrgans()
        if isinstance(o.getOrganRandomParameter().f_gf, pb.CWLimitedGrowth)
    ]


def _active_cw_organ(plant, organ_type):
    plant.setAccumulatedTT(plant.getAccumulatedTT() + 10.0)
    plant.setAccumulatedAndrieuTT(plant.getAccumulatedAndrieuTT() + 10.0)
    for organ, rp, gf in _cw_organs(plant):
        if organ.organType() != organ_type or gf.demand is None:
            continue
        current = organ.getLength(False)
        demand = gf.demand.getDemand(
            organ.getAge() + 1.0 / 24.0,
            rp.r,
            organ.param().getK(),
            organ,
        ) - current
        if demand > 1e-4:
            return organ, rp, gf, demand
    raise AssertionError(f"No active CWLimitedGrowth organ type {organ_type} found")


def test_s3_use_local_pool_false_d0_identical():
    plant_a = grow_plant(XML, simulation_time=40, seed=42)
    plant_b = grow_plant(XML, simulation_time=40, seed=42)
    enable_cw_limited_growth(plant_b, wrap_roots=False, wrap_fa=True)

    assert _cw_organs(plant_b)
    for _, _, gf in _cw_organs(plant_b):
        assert gf.use_local_pool is False

    lengths_a = [o.getLength(False) for o in plant_a.getOrgans()]
    lengths_b = [o.getLength(False) for o in plant_b.getOrgans()]
    assert lengths_b == pytest.approx(lengths_a, abs=0.0)


def test_s3_buffered_dispatch_synthetic():
    plant = grow_plant(XML, simulation_time=30, seed=42)
    enable_cw_limited_growth(plant, wrap_roots=False, wrap_fa=True)
    organ, rp, gf, demand = _active_cw_organ(plant, pb.OrganTypes.leaf)
    current = organ.getLength(False)

    rp.c_cost_per_cm = 0.35
    rp.local_C_pool_capacity_factor = 1.0
    gf.use_local_pool = True
    organ.local_C_pool_ = 2.0
    organ.dl_backlog = 5.0

    returned = gf.getLength(
        organ.getAge() + 1.0 / 24.0,
        rp.r,
        organ.param().getK(),
        organ,
    )
    expected_dL = min(demand, 2.0 / 0.35, organ.param().getK() - current)

    assert returned == pytest.approx(current + expected_dL)
    assert organ.local_C_pool_ == pytest.approx(2.0 - expected_dL * 0.35)
    assert organ.dl_backlog == 0.0


def test_s3_buffered_supply_limited_synthetic():
    plant = grow_plant(XML, simulation_time=30, seed=42)
    enable_cw_limited_growth(plant, wrap_roots=False, wrap_fa=True)
    organ, rp, gf, demand = _active_cw_organ(plant, pb.OrganTypes.leaf)
    current = organ.getLength(False)

    rp.c_cost_per_cm = max(100.0, 1.0 / demand)
    rp.local_C_pool_capacity_factor = 1.0
    gf.use_local_pool = True
    organ.local_C_pool_ = 0.1
    organ.dl_backlog = 5.0

    returned = gf.getLength(
        organ.getAge() + 1.0 / 24.0,
        rp.r,
        organ.param().getK(),
        organ,
    )
    expected_dL = 0.1 / rp.c_cost_per_cm

    assert returned == pytest.approx(current + expected_dL)
    assert organ.local_C_pool_ == pytest.approx(0.0, abs=1e-12)
    assert organ.dl_backlog == 0.0


def test_s3_buffered_supersedes_per_rank_stem():
    plant = grow_plant(XML, simulation_time=30, seed=42)
    enable_cw_limited_growth(plant, wrap_roots=False, wrap_fa=True)
    organ, rp, gf, _ = _active_cw_organ(plant, pb.OrganTypes.stem)

    rp.c_cost_per_cm = 0.35
    rp.local_C_pool_capacity_factor = 1.0
    gf.use_local_pool = True
    gf.CW_Gr_per_n = {organ.getId(): [0.0] * 32}
    organ.local_C_pool_ = 5.0
    organ.dl_backlog_per_n = [math.pi, math.e, 1.0]
    before_pool = organ.local_C_pool_
    before_backlog = list(organ.dl_backlog_per_n)

    returned = gf.getLength(
        organ.getAge() + 1.0 / 24.0,
        rp.r,
        organ.param().getK(),
        organ,
    )

    assert returned > organ.getLength(False)
    assert organ.local_C_pool_ < before_pool
    assert list(organ.dl_backlog_per_n) == pytest.approx(before_backlog)
