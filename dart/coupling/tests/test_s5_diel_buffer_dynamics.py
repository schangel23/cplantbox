"""S5 buffered-carbon diel reserve dynamics fixtures."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from dart.coupling.carbon import pm_substep


class _Organ:
    def __init__(self, oid, volume, pool=0.0):
        self._oid = oid
        self._volume = volume
        self.local_C_pool_ = pool

    def getId(self):
        return self._oid

    def orgVolume(self, _length, _realized):
        return self._volume


class _Plant:
    def __init__(self):
        self.transient_reserve_pool_ = 0.0
        self.srp = SimpleNamespace(
            starch_storage_efficiency=0.95,
            starch_remob_rate=2.0,
            starch_remob_efficiency=0.98,
        )

    def getOrganRandomParameter(self, organ_type, sub_type):
        assert organ_type == 1
        assert sub_type == 0
        return self.srp


def _patch_organs(monkeypatch, organs):
    rows = [
        (
            organ,
            SimpleNamespace(local_C_pool_capacity_factor=1.0),
            None,
        )
        for organ in organs
    ]
    monkeypatch.setattr(pm_substep, "_iter_bufferable_organs", lambda _plant: iter(rows))


def test_s5_reserve_charges_during_day(monkeypatch):
    plant = _Plant()
    charge_organ = _Organ(1, volume=0.2)
    remob_sink = _Organ(2, volume=5.0)
    _patch_organs(monkeypatch, [charge_organ, remob_sink])

    reserve_trace = []
    for _substep in range(12):
        pm_substep._reserve_remob_step(
            plant, 1.0 / 24.0, prior_fu_lim=1.0, starch_remob_threshold=0.1
        )
        pm_substep._allocate_fu_lim(plant, 0.5, {1: 1.0, 2: 0.0})
        reserve_trace.append(plant.transient_reserve_pool_)

    assert reserve_trace[0] > 0.0
    assert reserve_trace == sorted(reserve_trace)


def test_s5_reserve_drains_during_night(monkeypatch):
    plant = _Plant()
    charge_organ = _Organ(1, volume=0.2)
    remob_sink = _Organ(2, volume=5.0)
    _patch_organs(monkeypatch, [charge_organ, remob_sink])

    for _substep in range(12):
        pm_substep._allocate_fu_lim(plant, 0.5, {1: 1.0, 2: 0.0})

    reserve_before_night = plant.transient_reserve_pool_
    delivered_at_night = []
    reserve_trace = []
    for _substep in range(12):
        alloc = pm_substep._allocate_fu_lim(plant, 0.0, {1: 1.0, 2: 1.0})
        delivered_at_night.append(alloc["delivered_mmol"])
        pm_substep._reserve_remob_step(
            plant, 1.0 / 24.0, prior_fu_lim=0.0, starch_remob_threshold=0.1
        )
        reserve_trace.append(plant.transient_reserve_pool_)

    assert max(delivered_at_night) == pytest.approx(0.0)
    assert reserve_trace[0] < reserve_before_night
    assert reserve_trace == sorted(reserve_trace, reverse=True)


def test_s5_multi_day_reserve_signal(monkeypatch):
    plant = _Plant()
    charge_organ = _Organ(1, volume=0.2)
    remob_sink = _Organ(2, volume=5.0)
    _patch_organs(monkeypatch, [charge_organ, remob_sink])

    plant.transient_reserve_pool_ = 1.0
    cloudy_start = plant.transient_reserve_pool_
    for _substep in range(24):
        pm_substep._reserve_remob_step(
            plant, 1.0 / 24.0, prior_fu_lim=0.0, starch_remob_threshold=0.1
        )
    cloudy_delta = plant.transient_reserve_pool_ - cloudy_start

    sunny_start = plant.transient_reserve_pool_
    for _substep in range(12):
        pm_substep._reserve_remob_step(
            plant, 1.0 / 24.0, prior_fu_lim=1.0, starch_remob_threshold=0.1
        )
        pm_substep._allocate_fu_lim(plant, 0.8, {1: 1.0, 2: 0.0})
    sunny_delta = plant.transient_reserve_pool_ - sunny_start

    assert cloudy_delta < 0.0
    assert sunny_delta > 0.0
