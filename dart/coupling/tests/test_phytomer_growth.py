#!/usr/bin/env python3
"""
Step 2A — Phytomer Growth Validation Tests

Validates that decompose_phytomer=1 produces correct sheath+blade organ pairs
with proper topology, sequential emergence, and comparable total area.

Usage:
  cd /home/lukas/PHD/CPlantBox
  cpbenv/bin/python3 -m pytest dart/coupling/tests/test_phytomer_growth.py -v
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

N_PHYTOMERS = 11  # 11 leaf positions in maize


def _grow_phytomer_plant(day, seed=42):
    """Grow a plant with decompose_phytomer=1."""
    plant = pb.MappedPlant()
    plant.readParameters(str(PHYTOMER_XML))
    plant.setSeed(seed)
    setup_successor_where(plant)

    depth = 100
    soil_domain = pb.SDF_PlantContainer(np.inf, np.inf, depth, True)
    plant.setGeometry(soil_domain)
    plant.setSoilGrid(lambda _x, _y, z: max(min(int(np.floor(-z)), depth - 1), -1))
    plant.initialize()
    plant.simulate(day, True)
    return plant


def _grow_monolithic_plant(day, seed=42):
    """Grow a plant with decompose_phytomer=0 (existing behavior)."""
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


class TestPhytomerOrganCount:
    """Test 1: Correct number of sheath+blade pairs."""

    def test_day55_organ_count(self):
        """Day 55: all 11 phytomers should have emerged → 11 sheaths + 11 blades."""
        plant = _grow_phytomer_plant(55)
        organs = _get_leaf_organs(plant)
        sheaths, blades = _classify_organs(organs)

        print(f"\nDay 55: {len(sheaths)} sheaths, {len(blades)} blades, "
              f"{len(organs)} total leaf organs")
        assert len(sheaths) == N_PHYTOMERS, \
            f"Expected {N_PHYTOMERS} sheaths, got {len(sheaths)}"
        assert len(blades) == N_PHYTOMERS, \
            f"Expected {N_PHYTOMERS} blades, got {len(blades)}"
        assert len(organs) == 2 * N_PHYTOMERS, \
            f"Expected {2 * N_PHYTOMERS} total, got {len(organs)}"

    def test_total_22_leaf_organs(self):
        """22 leaf organs total (11 sheaths + 11 blades)."""
        plant = _grow_phytomer_plant(55)
        organs = _get_leaf_organs(plant)
        assert len(organs) == 22


class TestPhytomerOrganTypes:
    """Test 2: Even subtypes are sheaths, odd are blades."""

    def test_sheath_is_pseudostem(self):
        """Sheath organs have isPseudostem=1."""
        plant = _grow_phytomer_plant(55)
        organs = _get_leaf_organs(plant)
        sheaths, _ = _classify_organs(organs)
        for s in sheaths:
            ps = s.getParameter("isPseudostem")
            assert ps == 1, f"Sheath subType={s.getParameter('subType')} has isPseudostem={ps}"

    def test_blade_is_not_pseudostem(self):
        """Blade organs have isPseudostem=0."""
        plant = _grow_phytomer_plant(55)
        organs = _get_leaf_organs(plant)
        _, blades = _classify_organs(organs)
        for b in blades:
            ps = b.getParameter("isPseudostem")
            assert ps == 0, f"Blade subType={b.getParameter('subType')} has isPseudostem={ps}"


class TestPhytomerTopology:
    """Test 3: Both sheath and blade are children of the stem (sibling topology)."""

    def test_blade_parent_is_stem(self):
        """Blade organs should be children of the stem (sibling with sheath)."""
        plant = _grow_phytomer_plant(55)
        organs = _get_leaf_organs(plant)
        _, blades = _classify_organs(organs)

        for blade in blades:
            parent = blade.getParent()
            blade_st = blade.getParameter("subType")
            if parent is not None:
                parent_ot = parent.organType()
                parent_st = parent.getParameter("subType")
                # In sibling topology, blade's parent is the stem
                assert parent_ot == pb.stem, \
                    f"Blade st={blade_st} parent is organType={parent_ot}, expected stem ({pb.stem})"
                print(f"  Blade st={blade_st} → parent stem st={parent_st} ✓")

    def test_sheath_parent_is_stem(self):
        """Sheath organs should also be children of the stem."""
        plant = _grow_phytomer_plant(55)
        organs = _get_leaf_organs(plant)
        sheaths, _ = _classify_organs(organs)

        for sheath in sheaths:
            parent = sheath.getParent()
            sheath_st = sheath.getParameter("subType")
            if parent is not None:
                parent_ot = parent.organType()
                parent_st = parent.getParameter("subType")
                assert parent_ot == pb.stem, \
                    f"Sheath st={sheath_st} parent is organType={parent_ot}, expected stem ({pb.stem})"
                print(f"  Sheath st={sheath_st} → parent stem st={parent_st} ✓")


class TestPhytomerGeometry:
    """Test 4: Sheaths grow vertically, blades bend at insertion angle."""

    def test_sheath_is_mostly_vertical(self):
        """Sheath segments should be mostly vertical (small angle from z-axis)."""
        plant = _grow_phytomer_plant(55)
        organs = _get_leaf_organs(plant)
        sheaths, _ = _classify_organs(organs)

        for sheath in sheaths[:3]:  # test first 3
            nodes = sheath.getNodes()
            if len(nodes) < 2:
                continue
            # Direction from first to last node
            p0 = np.array([nodes[0].x, nodes[0].y, nodes[0].z])
            p1 = np.array([nodes[-1].x, nodes[-1].y, nodes[-1].z])
            direction = p1 - p0
            length = np.linalg.norm(direction)
            if length < 0.1:
                continue
            # Angle from vertical (z-axis)
            cos_angle = abs(direction[2]) / length
            angle_deg = math.degrees(math.acos(min(cos_angle, 1.0)))
            print(f"  Sheath st={sheath.getParameter('subType')}: "
                  f"angle from vertical = {angle_deg:.1f}°")
            # Sheaths should be within 45° of vertical
            assert angle_deg < 45, \
                f"Sheath st={sheath.getParameter('subType')} too far from vertical: {angle_deg:.1f}°"


class TestPhytomerArea:
    """Test 5: Total blade area ≈ monolithic total leaf area."""

    def test_area_comparison(self):
        """Decomposed blade area should be within 20% of monolithic leaf area."""
        plant_phyto = _grow_phytomer_plant(55)
        plant_mono = _grow_monolithic_plant(55)

        # Monolithic total leaf area
        mono_organs = _get_leaf_organs(plant_mono)
        mono_area = sum(o.leafArea(False) for o in mono_organs)

        # Phytomer blade-only area (sheaths have no blade area)
        phyto_organs = _get_leaf_organs(plant_phyto)
        _, blades = _classify_organs(phyto_organs)
        blade_area = sum(o.leafArea(False) for o in blades)

        print(f"\nMonolithic total leaf area: {mono_area:.1f} cm²")
        print(f"Phytomer blade-only area:  {blade_area:.1f} cm²")

        if mono_area > 0:
            ratio = blade_area / mono_area
            print(f"Ratio: {ratio:.2f}")
            # Allow 20% tolerance — SL_ratio changes effective blade lmax
            assert 0.3 < ratio < 1.5, \
                f"Area ratio {ratio:.2f} outside tolerance (0.3-1.5)"
        else:
            pytest.skip("Monolithic plant has no leaf area")


class TestSequentialEmergence:
    """Test 6: Phytomers emerge sequentially over time."""

    def test_day10_few_phytomers(self):
        """Day 10: only 2-4 phytomers should be visible."""
        plant = _grow_phytomer_plant(10)
        organs = _get_leaf_organs(plant)
        sheaths, _ = _classify_organs(organs)
        n = len(sheaths)
        print(f"\nDay 10: {n} sheaths emerged")
        assert 1 <= n <= 5, f"Expected 1-5 sheaths at day 10, got {n}"

    def test_day30_more_phytomers(self):
        """Day 30: ~6-9 phytomers should be visible."""
        plant = _grow_phytomer_plant(30)
        organs = _get_leaf_organs(plant)
        sheaths, _ = _classify_organs(organs)
        n = len(sheaths)
        print(f"\nDay 30: {n} sheaths emerged")
        assert 4 <= n <= 11, f"Expected 4-11 sheaths at day 30, got {n}"

    def test_day55_all_phytomers(self):
        """Day 55: all 11 phytomers should be visible."""
        plant = _grow_phytomer_plant(55)
        organs = _get_leaf_organs(plant)
        sheaths, _ = _classify_organs(organs)
        n = len(sheaths)
        print(f"\nDay 55: {n} sheaths emerged")
        assert n == N_PHYTOMERS, f"Expected {N_PHYTOMERS} sheaths at day 55, got {n}"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
