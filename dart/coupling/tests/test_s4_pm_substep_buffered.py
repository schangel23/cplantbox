"""S4 buffered-carbon PM substep integration fixtures."""

from __future__ import annotations

import pytest

from dart.coupling.carbon import pm_substep
from dart.coupling.carbon.pm_substep import solve_carbon_partitioning_pm
from dart.coupling.growth.carbon_growth import enable_cw_limited_growth
from dart.coupling.growth.grow import grow_plant
from dart.coupling.tests.test_pm_substep_dispatch import _synth_an_per_leaf


XML = "dart/coupling/data/maize_calibrated.xml"


def test_s4_local_pool_fills_each_substep():
    plant = grow_plant(XML, simulation_time=21, seed=42, enable_photosynthesis=True)
    enable_cw_limited_growth(plant, wrap_roots=False, wrap_fa=True)
    n_active = pm_substep._activate_buffered_growth(plant)
    plant.setAccumulatedTT(plant.getAccumulatedTT() + 10.0)
    plant.setAccumulatedAndrieuTT(plant.getAccumulatedAndrieuTT() + 10.0)

    demands = pm_substep._compute_fa_demands(plant, 1.0 / 24.0)
    result = pm_substep._allocate_fu_lim(plant, 1.0, demands)

    assert n_active > 0
    assert result["delivered_mmol"] > 0.0
    assert pm_substep._local_pool_total(plant) > 0.0


def test_s4_mass_balance_day1_buffered():
    plant = grow_plant(XML, simulation_time=21, seed=42, enable_photosynthesis=True)
    result = solve_carbon_partitioning_pm(
        plant,
        _synth_an_per_leaf(plant),
        Tair_C=20.75,
        day=21,
        n_substeps=24,
        use_buffered_carbon=True,
    )

    assert result is not None
    assert result["buffered_growth_active"] is True
    assert result["buffered_growth_active_organs"] > 0
    assert result["buffered_fu_delivered_mmol"] > 0.0
    assert abs(result["mass_balance_residual_pct"]) <= 2.0
    for key in (
        "transient_reserve_pool_mmol",
        "local_C_pool_total_mmol",
        "reserve_delta_mmol",
        "local_C_pool_delta_mmol",
        "storage_loss_mmol",
        "remob_loss_mmol",
    ):
        assert key in result
        assert result[key] == pytest.approx(float(result[key]))


def test_s4_buffered_can_be_disabled():
    plant = grow_plant(XML, simulation_time=21, seed=42, enable_photosynthesis=True)
    result = solve_carbon_partitioning_pm(
        plant,
        _synth_an_per_leaf(plant),
        Tair_C=20.75,
        day=21,
        n_substeps=4,
        use_buffered_carbon=False,
    )

    assert result is not None
    assert result["buffered_growth_active"] is False
    assert result["buffered_growth_active_organs"] == 0
    assert result["buffered_fu_delivered_mmol"] == 0.0
