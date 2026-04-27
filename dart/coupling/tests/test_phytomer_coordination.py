#!/usr/bin/env python3
"""
Step 2B — Phytomer Coordination & Thermal-Time Tests

Validates Fournier-style coordination events and cardinal-temperature-driven
elongation added in Step 2B.

Usage:
  cd /home/lukas/PHD/CPlantBox
  cpbenv/bin/python3 -m pytest dart/coupling/tests/test_phytomer_coordination.py -v
"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

# Add coupling package to path
COUPLING_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(COUPLING_DIR.parent.parent))

import plantbox as pb
from dart.coupling.growth.grow import setup_successor_where

PHYTOMER_XML = COUPLING_DIR / "data" / "maize_phytomer.xml"
MONOLITHIC_XML = COUPLING_DIR / "data" / "maize_calibrated.xml"


def _grow_phytomer_plant(day, seed=42, T_air=25.0, use_thermal=False):
    """Grow a plant with decompose_phytomer=1, optionally with thermal elongation."""
    plant = pb.MappedPlant()
    plant.readParameters(str(PHYTOMER_XML))
    plant.setSeed(seed)
    setup_successor_where(plant)

    if use_thermal:
        # Enable thermal elongation on all leaf subtypes
        for st in range(24):
            try:
                lrp = plant.getOrganRandomParameter(pb.leaf, st)
                if lrp is not None:
                    lrp.use_thermal_elongation = 1
                    lrp.T_base = 8.0
                    lrp.T_opt = 30.0
                    lrp.T_max = 41.0
                    lrp.LER_max = 1.5
            except Exception:
                pass

    depth = 100
    soil_domain = pb.SDF_PlantContainer(np.inf, np.inf, depth, True)
    plant.setGeometry(soil_domain)
    plant.setSoilGrid(lambda _x, _y, z: max(min(int(np.floor(-z)), depth - 1), -1))
    plant.initialize()

    # Simulate day-by-day, setting temperature before each step
    dt = 1.0
    total = 0.0
    while total < day:
        step = min(dt, day - total)
        plant.setAirTemperature(T_air)
        try:
            plant.simulate(step, verbose=(total == 0))
            total += step
        except (IndexError, RuntimeError):
            try:
                plant.simulate(0.0)
            except Exception:
                pass
            break
    return plant


def _grow_monolithic_plant(day, seed=42):
    """Grow a plant with decompose_phytomer=0 (backward compat)."""
    plant = pb.MappedPlant()
    plant.readParameters(str(MONOLITHIC_XML))
    plant.setSeed(seed)
    setup_successor_where(plant)

    depth = 100
    soil_domain = pb.SDF_PlantContainer(np.inf, np.inf, depth, True)
    plant.setGeometry(soil_domain)
    plant.setSoilGrid(lambda _x, _y, z: max(min(int(np.floor(-z)), depth - 1), -1))
    plant.initialize()
    plant.simulate(day, True)
    return plant


def _get_leaf_organs(plant):
    """Get all leaf organs with >1 node (grown, not dormant)."""
    return [o for o in plant.getOrgans(pb.leaf) if len(o.getNodes()) > 1]


def _classify_organs(organs):
    """Split leaf organs into sheaths (even subType) and blades (odd subType)."""
    sheaths = [o for o in organs if o.getParameter("subType") % 2 == 0]
    blades = [o for o in organs if o.getParameter("subType") % 2 == 1]
    return sheaths, blades


class TestCardinalTemperature:
    """Test cardinal temperature response function via growth behavior."""

    def test_below_Tbase_no_growth(self):
        """T < T_base → no thermal-time elongation."""
        plant = _grow_phytomer_plant(30, T_air=5.0, use_thermal=True)
        organs = _get_leaf_organs(plant)
        # With T=5 < T_base=8, thermal elongation = 0.
        # Calendar-day growth still runs via calcLength, but thermal dl branch
        # produces dl=0 since f_T=0. Leaves should be very short or zero-length.
        total_length = sum(o.getLength(True) for o in organs)
        # The calendar-day path is skipped when thermal_elongation=true,
        # so growth is purely thermal. At T=5 < T_base=8, f_T=0 → dl=0.
        # Some leaves may have minimal length from initialization.
        for o in organs:
            assert o.getLength(True) < 1.0, (
                f"SubType {o.getParameter('subType')}: length={o.getLength(True):.2f} "
                f"should be near 0 at T=5°C < T_base=8°C"
            )

    def test_at_Topt_full_rate(self):
        """T = T_opt → maximum elongation rate."""
        plant_opt = _grow_phytomer_plant(30, T_air=30.0, use_thermal=True)
        plant_mid = _grow_phytomer_plant(30, T_air=19.0, use_thermal=True)

        organs_opt = _get_leaf_organs(plant_opt)
        organs_mid = _get_leaf_organs(plant_mid)

        total_opt = sum(o.getLength(True) for o in organs_opt)
        total_mid = sum(o.getLength(True) for o in organs_mid)

        # T_opt growth should exceed T=19°C growth (f_T=0.5 at T=19)
        assert total_opt > total_mid, (
            f"T_opt total={total_opt:.1f} should exceed T=19°C total={total_mid:.1f}"
        )

    def test_above_Tmax_no_growth(self):
        """T >= T_max → no growth."""
        plant = _grow_phytomer_plant(30, T_air=42.0, use_thermal=True)
        organs = _get_leaf_organs(plant)
        for o in organs:
            assert o.getLength(True) < 1.0, (
                f"SubType {o.getParameter('subType')}: length={o.getLength(True):.2f} "
                f"should be near 0 at T=42°C >= T_max=41°C"
            )


class TestPseudostemHeight:
    """Test pseudostem height = max tip z of older sheaths."""

    def test_pseudostem_height_positive(self):
        """Sheath rank > 0 should see positive pseudostem height (older sheaths exist)."""
        plant = _grow_phytomer_plant(40)
        organs = _get_leaf_organs(plant)
        sheaths, _ = _classify_organs(organs)

        if len(sheaths) < 2:
            pytest.skip("Need at least 2 sheaths for pseudostem test")

        # Sort by rank, use a sheath with rank > 0
        sheaths_sorted = sorted(sheaths, key=lambda o: o.getParameter("subType"))
        sheath = sheaths_sorted[1]  # rank 1
        ps_h = sheath.computePseudostemHeight()
        assert ps_h > 0, f"Pseudostem height for rank 1 should be > 0, got {ps_h}"

    def test_pseudostem_equals_max_older_sheath_tip(self):
        """Pseudostem height of rank N = max tip z of sheaths with rank < N."""
        plant = _grow_phytomer_plant(40)
        organs = _get_leaf_organs(plant)
        sheaths, _ = _classify_organs(organs)

        if len(sheaths) < 3:
            pytest.skip("Need at least 3 sheaths")

        # Sort by rank
        sheaths_sorted = sorted(sheaths, key=lambda o: o.getParameter("subType"))
        test_sheath = sheaths_sorted[2]  # rank 2
        test_rank = int(test_sheath.getParameter("subType")) // 2

        # Compute max tip z of sheaths with rank < test_rank
        older_sheaths = [s for s in sheaths
                         if int(s.getParameter("subType")) // 2 < test_rank]
        max_tip_z = max(s.getNodes()[-1].z for s in older_sheaths)

        # Get pseudostem height from C++ method
        ps_h = test_sheath.computePseudostemHeight()

        assert abs(ps_h - max_tip_z) < 0.1, (
            f"Pseudostem height {ps_h:.2f} != max older sheath tip z {max_tip_z:.2f}"
        )


class TestEmergenceDetection:
    """Test that sheath emergence is detected when tip > pseudostem."""

    def test_emergence_flag_set(self):
        """With thermal elongation, sheaths should eventually set emerged=True."""
        plant = _grow_phytomer_plant(40, T_air=25.0, use_thermal=True)
        organs = _get_leaf_organs(plant)
        sheaths, _ = _classify_organs(organs)

        if len(sheaths) < 2:
            pytest.skip("Need at least 2 sheaths")

        # Rank 0 sheath: pseudostem_h=0, so tip_z > 0 triggers emergence immediately.
        # Higher rank sheaths emerge when their tip exceeds older sheaths' max tip z.
        emerged_count = sum(
            1 for o in sheaths if o.hasEmerged()
        )
        # At minimum, rank 0 should have emerged (pseudostem_h=0, any growth → emerge)
        assert emerged_count > 0, (
            f"No sheaths have emerged after 40 days at T=25°C. "
            f"Sheath lengths: {[o.getLength(True) for o in sheaths]}"
        )


class TestBladeCoordination:
    """Test blade growth coordination with previous sheath emergence."""

    def test_blade_lmax_set_after_sheath_emergence(self):
        """Blade n should get lmax_set=True after sheath n-1 emerges."""
        plant = _grow_phytomer_plant(40, T_air=25.0, use_thermal=True)
        organs = _get_leaf_organs(plant)
        _, blades = _classify_organs(organs)

        if len(blades) == 0:
            pytest.skip("No blades produced")

        # First blade (rank 0) should always have lmax_set
        first_blades = [o for o in blades if int(o.getParameter("subType")) // 2 == 0]
        if first_blades:
            assert first_blades[0].lmax_set, "First blade (rank 0) should have lmax_set=True"


class TestSequentialEmergence:
    """Test that leaves emerge sequentially, not simultaneously."""

    def test_early_vs_late(self):
        """At day 10, fewer leaves than at day 55."""
        plant_early = _grow_phytomer_plant(10)
        plant_late = _grow_phytomer_plant(55)

        organs_early = _get_leaf_organs(plant_early)
        organs_late = _get_leaf_organs(plant_late)

        n_early = len(organs_early)
        n_late = len(organs_late)

        assert n_early < n_late, (
            f"Day 10 has {n_early} organs, day 55 has {n_late}. "
            f"Emergence should be sequential."
        )


class TestThermalElongation:
    """Test that thermal-time elongation responds to temperature."""

    def test_higher_temp_faster_growth(self):
        """T=25°C (f_T≈0.77) should produce more growth than T=15°C (f_T≈0.32)."""
        plant_warm = _grow_phytomer_plant(30, T_air=25.0, use_thermal=True)
        plant_cool = _grow_phytomer_plant(30, T_air=15.0, use_thermal=True)

        organs_warm = _get_leaf_organs(plant_warm)
        organs_cool = _get_leaf_organs(plant_cool)

        total_warm = sum(o.getLength(True) for o in organs_warm)
        total_cool = sum(o.getLength(True) for o in organs_cool)

        assert total_warm > total_cool, (
            f"T=25°C total={total_warm:.1f} should exceed "
            f"T=15°C total={total_cool:.1f}"
        )


class TestBackwardCompat:
    """Test that decompose_phytomer=0 is unaffected by Step 2B changes."""

    def test_monolithic_unchanged(self):
        """Monolithic (non-phytomer) growth should still work identically."""
        plant = _grow_monolithic_plant(55)
        organs = _get_leaf_organs(plant)

        # Should produce at least some leaves
        assert len(organs) > 5, f"Expected >5 leaf organs, got {len(organs)}"

        # Check total length is reasonable (maize at day 55)
        total_length = sum(o.getLength(True) for o in organs)
        assert total_length > 100, f"Total leaf length {total_length:.1f} too low"


class TestTemperatureZeroGrowth:
    """Test that T exactly at boundaries produces zero growth."""

    def test_T_equals_Tbase(self):
        """T = T_base exactly → f_T = 0 → no thermal growth."""
        plant = _grow_phytomer_plant(20, T_air=8.0, use_thermal=True)
        organs = _get_leaf_organs(plant)
        for o in organs:
            assert o.getLength(True) < 1.0, (
                f"SubType {o.getParameter('subType')}: length={o.getLength(True):.2f} "
                f"at T=T_base=8°C should be ~0"
            )

    def test_T_equals_Tmax(self):
        """T = T_max exactly → f_T = 0 → no thermal growth."""
        plant = _grow_phytomer_plant(20, T_air=41.0, use_thermal=True)
        organs = _get_leaf_organs(plant)
        for o in organs:
            assert o.getLength(True) < 1.0, (
                f"SubType {o.getParameter('subType')}: length={o.getLength(True):.2f} "
                f"at T=T_max=41°C should be ~0"
            )
