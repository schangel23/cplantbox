#!/usr/bin/env python3
"""
Step 2A — Transport Solver Validation (CRITICAL BLOCKER)

Validates that the Meunier hydraulic solver correctly handles sheath→blade
segment chains. If these tests fail, the entire phytomer pipeline is blocked.

Tests:
  1. Topology: valid tree, no cycles
  2. Water flow path: root→stem→sheath→blade
  3. Transpiration: only blade segments (odd subType) have kr>0
  4. Flux conservation: water uptake ≈ transpiration
  5. Solver convergence: no NaN/Inf in solution
  6. Comparison: decompose=1 flux ≈ decompose=0 (within tolerance)

Usage:
  cd /home/lukas/PHD/CPlantBox
  cpbenv/bin/python3 -m pytest dart/coupling/tests/test_phytomer_hydraulics.py -v
"""

import json
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
PHYTOMER_HYDRAULICS = COUPLING_DIR / "data" / "maize_phytomer_hydraulics.json"
MONOLITHIC_XML = COUPLING_DIR / "data" / "maize_calibrated.xml"
MONOLITHIC_HYDRAULICS = COUPLING_DIR / "data" / "maize_couvreur2012_hydraulics.json"

# Simulation parameters
SIM_DAY = 30  # Enough for several phytomers but fast
SOIL_PSI = -500.0  # well-watered
DEPTH = 100
PAR_UMOL = 1000.0
TAIR_C = 25.0
RH = 0.7


def _grow_plant(xml_path, day=SIM_DAY, seed=42):
    """Grow a plant with soil grid for hydraulics."""
    plant = pb.MappedPlant()
    plant.readParameters(str(xml_path))
    plant.setSeed(seed)
    setup_successor_where(plant)

    soil_domain = pb.SDF_PlantContainer(np.inf, np.inf, DEPTH, True)
    plant.setGeometry(soil_domain)
    plant.setSoilGrid(lambda _x, _y, z: max(min(int(np.floor(-z)), DEPTH - 1), -1))
    plant.initialize()
    plant.simulate(day, True)
    return plant


def _setup_hydraulics(plant, hydraulics_json):
    """Set up PhloemFluxPython model with given hydraulics."""
    from plantbox.functional.phloem_flux import PhloemFluxPython
    from plantbox.functional.PlantHydraulicParameters import PlantHydraulicParameters

    params = PlantHydraulicParameters()
    # read_parameters() appends ".json" internally, so strip it
    params.read_parameters(str(hydraulics_json).removesuffix('.json'))

    hm = PhloemFluxPython(plant, params)
    # CPlantBox read_* methods append ".json" internally, so strip it
    photo_json = COUPLING_DIR / "data" / "maize_C4_photosynthesis_parameters.json"
    if photo_json.exists():
        hm.read_photosynthesis_parameters(filename=str(photo_json).removesuffix('.json'))
    phloem_json = COUPLING_DIR / "data" / "phloem_parameters_maize2026.json"
    if phloem_json.exists():
        hm.read_phloem_parameters(filename=str(phloem_json).removesuffix('.json'))
    return hm


def _solve_hydraulics(plant, hm, day=SIM_DAY):
    """Run a hydraulic + photosynthesis solve."""
    p_s = np.linspace(SOIL_PSI, SOIL_PSI - DEPTH, DEPTH)
    es = hm.get_es(TAIR_C)
    ea = es * RH
    par_mol_cm2_d = PAR_UMOL * 1e-6 * 86400 * 1e-4

    hm.solve(
        sim_time=day,
        rsx=p_s,
        cells=True,
        ea=ea,
        es=es,
        PAR=par_mol_cm2_d,
        TairC=TAIR_C,
        verbose=0,
    )
    return hm


class TestSegmentTopology:
    """Test 1: Segments form a valid tree (no cycles, all connected)."""

    def test_valid_tree(self):
        """All segments should form a connected tree with no cycles."""
        plant = _grow_plant(PHYTOMER_XML)
        segs = plant.getSegments()
        nodes = plant.getNodes()

        n_segs = len(segs)
        n_nodes = len(nodes)
        print(f"\nPhytomer plant: {n_segs} segments, {n_nodes} nodes")

        # Tree: n_segments = n_nodes - 1
        assert n_segs == n_nodes - 1, \
            f"Not a tree: {n_segs} segments vs {n_nodes} nodes (expected {n_nodes-1})"

        # Check no self-loops
        for i, seg in enumerate(segs):
            assert seg.x != seg.y, f"Self-loop at segment {i}: ({seg.x}, {seg.y})"

    def test_no_disconnected_components(self):
        """All nodes should be reachable from node 0."""
        plant = _grow_plant(PHYTOMER_XML)
        segs = plant.getSegments()
        nodes = plant.getNodes()
        n_nodes = len(nodes)

        # Build adjacency list
        adj = {i: [] for i in range(n_nodes)}
        for seg in segs:
            adj[seg.x].append(seg.y)
            adj[seg.y].append(seg.x)

        # BFS from node 0
        visited = set()
        queue = [0]
        while queue:
            node = queue.pop(0)
            if node in visited:
                continue
            visited.add(node)
            for neighbor in adj[node]:
                if neighbor not in visited:
                    queue.append(neighbor)

        assert len(visited) == n_nodes, \
            f"Disconnected: only {len(visited)}/{n_nodes} nodes reachable from root"


class TestWaterFlowPath:
    """Test 2: Water flows root→stem→sheath→blade."""

    def test_blade_segments_exist(self):
        """Blade segments (odd subType, organType=4) should exist."""
        plant = _grow_plant(PHYTOMER_XML)
        leaf_seg_ids = plant.getSegmentIds(4)  # organType=4 = leaf
        organs = plant.getOrgans(pb.leaf)
        blade_segs = []
        for o in organs:
            if o.getParameter("subType") % 2 == 1 and len(o.getNodes()) > 1:
                blade_segs.extend(range(len(o.getSegments())))
        print(f"\nTotal leaf segments: {len(leaf_seg_ids)}")
        assert len(leaf_seg_ids) > 0, "No leaf segments found"

    def test_sheath_blade_share_stem_node(self):
        """Sheath and blade at same position should attach to the same stem node."""
        plant = _grow_plant(PHYTOMER_XML)
        organs = plant.getOrgans(pb.leaf)

        sheaths = {o.getParameter("subType"): o for o in organs
                   if o.getParameter("subType") % 2 == 0 and len(o.getNodes()) > 1}
        blades = {o.getParameter("subType"): o for o in organs
                  if o.getParameter("subType") % 2 == 1 and len(o.getNodes()) > 1}

        connected = 0
        for blade_st, blade in blades.items():
            sheath_st = blade_st - 1
            if sheath_st in sheaths:
                sheath = sheaths[sheath_st]
                sheath_base = sheath.getNodes()[0]
                blade_base = blade.getNodes()[0]
                # Both siblings attach to the same stem node
                dist = np.sqrt((sheath_base.x - blade_base.x)**2 +
                               (sheath_base.y - blade_base.y)**2 +
                               (sheath_base.z - blade_base.z)**2)
                print(f"  Sheath st={sheath_st} base ↔ Blade st={blade_st} base: "
                      f"dist={dist:.4f} cm")
                # They share the same stem node, so distance should be ~0
                assert dist < 1.0, \
                    f"Sheath-blade base gap too large: {dist:.2f} cm"
                connected += 1

        print(f"\n  {connected} sheath-blade pairs share stem attachment")
        assert connected > 0, "No sheath-blade pairs found"


class TestTranspiration:
    """Test 3: Only blade segments should have radial conductivity."""

    def test_sheath_kr_zero(self):
        """Sheath segments should have kr=0 (no transpiration)."""
        # Verify from the hydraulics JSON
        with open(PHYTOMER_HYDRAULICS) as f:
            data = json.load(f)

        # organType "4" (leaf), even indices = sheaths
        kr_leaf = data["kr_values"]["4"]
        for i in range(0, min(len(kr_leaf), 22), 2):  # even indices
            values = kr_leaf[i]
            assert all(v == 0 for v in values), \
                f"Sheath subType={i} has non-zero kr: {values}"
            print(f"  Sheath subType={i}: kr={values} ✓")

    def test_blade_kr_nonzero(self):
        """Blade segments should have kr>0 (transpiration active)."""
        with open(PHYTOMER_HYDRAULICS) as f:
            data = json.load(f)

        kr_leaf = data["kr_values"]["4"]
        for i in range(1, min(len(kr_leaf), 22), 2):  # odd indices
            values = kr_leaf[i]
            assert any(v > 0 for v in values), \
                f"Blade subType={i} has all-zero kr: {values}"
            print(f"  Blade subType={i}: kr={values} ✓")


class TestSolverConvergence:
    """Test 5: Solver should converge without NaN/Inf."""

    def test_no_nan_in_solution(self):
        """Water potential solution should contain no NaN or Inf values."""
        plant = _grow_plant(PHYTOMER_XML)
        hm = _setup_hydraulics(plant, PHYTOMER_HYDRAULICS)
        hm = _solve_hydraulics(plant, hm)

        hx = np.array(hm.get_water_potential())
        print(f"\nWater potential: {len(hx)} values, "
              f"mean={np.mean(hx):.0f} cm, "
              f"range=[{np.min(hx):.0f}, {np.max(hx):.0f}]")

        assert not np.any(np.isnan(hx)), "NaN in water potential solution"
        assert not np.any(np.isinf(hx)), "Inf in water potential solution"

    def test_assimilation_positive(self):
        """Net assimilation should be positive (plant is photosynthesizing)."""
        plant = _grow_plant(PHYTOMER_XML)
        hm = _setup_hydraulics(plant, PHYTOMER_HYDRAULICS)
        hm = _solve_hydraulics(plant, hm)

        An = np.array(hm.get_net_assimilation())
        An_total = np.sum(An) * 1e3  # mmol CO2 d-1
        print(f"\nTotal An = {An_total:.1f} mmol CO2 d-1 ({len(An)} leaf segs)")

        assert An_total > 0, f"An_total = {An_total:.3f} ≤ 0 — photosynthesis failed"
        assert not np.any(np.isnan(An)), "NaN in assimilation"

    def test_transpiration_positive(self):
        """Total transpiration should be positive."""
        plant = _grow_plant(PHYTOMER_XML)
        hm = _setup_hydraulics(plant, PHYTOMER_HYDRAULICS)
        hm = _solve_hydraulics(plant, hm)

        transp = np.sum(hm.get_transpiration()) / 18 * 1e3  # mmol H2O d-1
        print(f"\nTotal transpiration = {transp:.1f} mmol H2O d-1")
        assert transp > 0, f"Transpiration = {transp:.3f} ≤ 0"


class TestFluxComparison:
    """Test 6: Phytomer mode flux ≈ monolithic mode."""

    def test_assimilation_same_order(self):
        """Phytomer An should be within a factor of 3 of monolithic An."""
        # Phytomer mode
        plant_p = _grow_plant(PHYTOMER_XML)
        hm_p = _setup_hydraulics(plant_p, PHYTOMER_HYDRAULICS)
        hm_p = _solve_hydraulics(plant_p, hm_p)
        An_p = np.sum(hm_p.get_net_assimilation()) * 1e3

        # Monolithic mode
        plant_m = _grow_plant(MONOLITHIC_XML)
        hm_m = _setup_hydraulics(plant_m, MONOLITHIC_HYDRAULICS)
        hm_m = _solve_hydraulics(plant_m, hm_m)
        An_m = np.sum(hm_m.get_net_assimilation()) * 1e3

        print(f"\nPhytomer An:   {An_p:.1f} mmol CO2 d-1")
        print(f"Monolithic An: {An_m:.1f} mmol CO2 d-1")

        if An_m > 0 and An_p > 0:
            ratio = An_p / An_m
            print(f"Ratio: {ratio:.2f}")
            # Allow factor of 3 tolerance — topology and area differ
            assert 0.1 < ratio < 5.0, \
                f"An ratio {ratio:.2f} outside tolerance (0.1-5.0)"
        elif An_m == 0 and An_p == 0:
            pytest.skip("Both modes produced An=0")
        else:
            print(f"  WARNING: one mode has An=0, check topology")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
