#!/usr/bin/env python3
"""
Step 4 — CPlantBox to MTG Converter Tests

Validates that cplantbox_to_mtg produces correct MTG topology, scale hierarchy,
properties, and round-trip serialization.

Usage:
  cd /home/lukas/PHD/CPlantBox
  cpbenv/bin/python3 -m pytest dart/coupling/tests/test_cplantbox_to_mtg.py -v
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

# Add coupling package to path
COUPLING_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(COUPLING_DIR.parent.parent))

import plantbox as pb
from dart.coupling.geometry.cplantbox_to_mtg import (
    cplantbox_to_mtg, write_mtg_file, read_mtg_with_arrays,
)
from dart.coupling.growth.grow import setup_successor_where

PHYTOMER_XML = COUPLING_DIR / "data" / "maize_phytomer.xml"
MONOLITHIC_XML = COUPLING_DIR / "data" / "maize_calibrated.xml"

N_PHYTOMERS = 11


def _grow_phytomer_plant(day=55, seed=42):
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


def _grow_monolithic_plant(day=55, seed=42):
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


@pytest.fixture(scope="module")
def phytomer_mtg():
    plant = _grow_phytomer_plant()
    return cplantbox_to_mtg(plant, decompose_phytomer=True)


@pytest.fixture(scope="module")
def monolithic_mtg():
    plant = _grow_monolithic_plant()
    return cplantbox_to_mtg(plant, decompose_phytomer=False)


# ---------------------------------------------------------------------------
# TestMTGCreation
# ---------------------------------------------------------------------------

class TestMTGCreation:
    """MTG not empty for phytomer + monolithic plants."""

    def test_phytomer_mtg_not_empty(self, phytomer_mtg):
        verts = [v for v in phytomer_mtg.vertices() if phytomer_mtg.scale(v) > 0]
        assert len(verts) > 0, "Phytomer MTG has no vertices"

    def test_monolithic_mtg_not_empty(self, monolithic_mtg):
        verts = [v for v in monolithic_mtg.vertices() if monolithic_mtg.scale(v) > 0]
        assert len(verts) > 0, "Monolithic MTG has no vertices"


# ---------------------------------------------------------------------------
# TestScaleHierarchy
# ---------------------------------------------------------------------------

class TestScaleHierarchy:
    """Correct vertex counts at each scale."""

    def test_one_plant(self, phytomer_mtg):
        plants = list(phytomer_mtg.vertices(scale=1))
        assert len(plants) == 1, f"Expected 1 plant, got {len(plants)}"

    def test_one_axis(self, phytomer_mtg):
        axes = list(phytomer_mtg.vertices(scale=2))
        assert len(axes) == 1, f"Expected 1 axis, got {len(axes)}"

    def test_eleven_metamers(self, phytomer_mtg):
        metamers = list(phytomer_mtg.vertices(scale=3))
        assert len(metamers) == N_PHYTOMERS, \
            f"Expected {N_PHYTOMERS} metamers, got {len(metamers)}"

    def test_phytomer_organ_count(self, phytomer_mtg):
        """11 internodes + 11 sheaths + 11 blades = 33 organs."""
        organs = list(phytomer_mtg.vertices(scale=4))
        organ_types = [phytomer_mtg.property('organ_type').get(v) for v in organs]
        n_i = organ_types.count('internode')
        n_s = organ_types.count('sheath')
        n_b = organ_types.count('blade')
        print(f"\nOrgans: {n_i}I + {n_s}S + {n_b}B = {len(organs)} total")
        assert n_i == N_PHYTOMERS, f"Expected {N_PHYTOMERS} internodes, got {n_i}"
        assert n_s == N_PHYTOMERS, f"Expected {N_PHYTOMERS} sheaths, got {n_s}"
        assert n_b == N_PHYTOMERS, f"Expected {N_PHYTOMERS} blades, got {n_b}"

    def test_monolithic_organ_count(self, monolithic_mtg):
        """11 internodes + 11 leaves = 22 organs."""
        organs = list(monolithic_mtg.vertices(scale=4))
        organ_types = [monolithic_mtg.property('organ_type').get(v) for v in organs]
        n_i = organ_types.count('internode')
        n_l = organ_types.count('leaf')
        print(f"\nOrgans: {n_i}I + {n_l}L = {len(organs)} total")
        assert n_i == N_PHYTOMERS, f"Expected {N_PHYTOMERS} internodes, got {n_i}"
        assert n_l == N_PHYTOMERS, f"Expected {N_PHYTOMERS} leaves, got {n_l}"


# ---------------------------------------------------------------------------
# TestEdgeTypes
# ---------------------------------------------------------------------------

class TestEdgeTypes:
    """Correct edge types between vertices."""

    def test_metamer_succession(self, phytomer_mtg):
        """Consecutive metamers connected by '<' edges."""
        metamers = sorted(phytomer_mtg.vertices(scale=3))
        for i in range(1, len(metamers)):
            parent = phytomer_mtg.parent(metamers[i])
            assert parent == metamers[i - 1], \
                f"Metamer {metamers[i]} parent is {parent}, expected {metamers[i-1]}"
            edge = phytomer_mtg.property('edge_type').get(metamers[i])
            assert edge == '<', f"Metamer edge_type is '{edge}', expected '<'"

    def test_internode_decomposition(self, phytomer_mtg):
        """Internode is a component (decomposition) of its metamer."""
        for metamer in phytomer_mtg.vertices(scale=3):
            components = list(phytomer_mtg.components(metamer))
            assert len(components) > 0, f"Metamer {metamer} has no components"
            # First component should be the internode
            first_label = phytomer_mtg.label(components[0])
            assert first_label == 'I', \
                f"First component label is '{first_label}', expected 'I'"

    def test_sheath_blade_branching(self, phytomer_mtg):
        """Sheaths and blades are '+' children of internodes."""
        for organ_vid in phytomer_mtg.vertices(scale=4):
            label = phytomer_mtg.label(organ_vid)
            if label in ('S', 'B'):
                edge = phytomer_mtg.property('edge_type').get(organ_vid)
                assert edge == '+', \
                    f"Organ {organ_vid} ({label}) edge_type is '{edge}', expected '+'"


# ---------------------------------------------------------------------------
# TestProperties
# ---------------------------------------------------------------------------

class TestProperties:
    """Organ properties are correctly attached."""

    def test_skeleton_is_ndarray(self, phytomer_mtg):
        for vid in phytomer_mtg.vertices(scale=4):
            skel = phytomer_mtg.property('skeleton').get(vid)
            assert isinstance(skel, np.ndarray), \
                f"Vertex {vid} skeleton is {type(skel)}, expected ndarray"
            assert skel.ndim == 2 and skel.shape[1] == 3, \
                f"Vertex {vid} skeleton shape {skel.shape}, expected (N, 3)"

    def test_length_positive(self, phytomer_mtg):
        for vid in phytomer_mtg.vertices(scale=4):
            length = phytomer_mtg.property('length').get(vid)
            label = phytomer_mtg.label(vid)
            if label != 'I':  # internodes can be very short
                assert length is not None and length > 0, \
                    f"Vertex {vid} ({label}) length={length}"

    def test_organ_type_correct(self, phytomer_mtg):
        label_to_type = {'I': 'internode', 'S': 'sheath', 'B': 'blade'}
        for vid in phytomer_mtg.vertices(scale=4):
            label = phytomer_mtg.label(vid)
            expected = label_to_type.get(label)
            actual = phytomer_mtg.property('organ_type').get(vid)
            assert actual == expected, \
                f"Vertex {vid} label={label} organ_type='{actual}', expected '{expected}'"

    def test_position_not_all_zeros(self, phytomer_mtg):
        all_zero = 0
        for vid in phytomer_mtg.vertices(scale=4):
            px = phytomer_mtg.property('position_x').get(vid, 0)
            py = phytomer_mtg.property('position_y').get(vid, 0)
            pz = phytomer_mtg.property('position_z').get(vid, 0)
            if px == 0 and py == 0 and pz == 0:
                all_zero += 1
        total = len(list(phytomer_mtg.vertices(scale=4)))
        assert all_zero < total, "All organs have position (0,0,0)"


# ---------------------------------------------------------------------------
# TestScaleNavigation
# ---------------------------------------------------------------------------

class TestScaleNavigation:
    """MTG scale navigation works correctly."""

    def test_complex_organ_to_metamer(self, phytomer_mtg):
        """complex(organ) → metamer at scale 3."""
        for organ_vid in phytomer_mtg.vertices(scale=4):
            metamer = phytomer_mtg.complex(organ_vid)
            assert phytomer_mtg.scale(metamer) == 3, \
                f"complex({organ_vid}) is at scale {phytomer_mtg.scale(metamer)}, expected 3"

    def test_components_metamer_to_organs(self, phytomer_mtg):
        """components(metamer) → [I, S, B] at scale 4."""
        for metamer in phytomer_mtg.vertices(scale=3):
            components = list(phytomer_mtg.components(metamer))
            labels = [phytomer_mtg.label(v) for v in components]
            assert 'I' in labels, f"Metamer {metamer} missing internode: {labels}"

    def test_components_plant_to_axis(self, phytomer_mtg):
        """components(plant) → [axis] at scale 2."""
        plant_vid = list(phytomer_mtg.vertices(scale=1))[0]
        components = list(phytomer_mtg.components(plant_vid))
        assert len(components) == 1
        assert phytomer_mtg.scale(components[0]) == 2


# ---------------------------------------------------------------------------
# TestRoundTrip
# ---------------------------------------------------------------------------

class TestRoundTrip:
    """Write + read preserves topology and properties."""

    def test_write_read_pkl(self, phytomer_mtg):
        """Pickle roundtrip preserves vertex count."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_mtg"
            write_mtg_file(phytomer_mtg, path)
            g2 = read_mtg_with_arrays(path)
            for scale in range(5):
                orig = len(list(phytomer_mtg.vertices(scale=scale)))
                loaded = len(list(g2.vertices(scale=scale)))
                assert orig == loaded, \
                    f"Scale {scale}: {orig} → {loaded} vertices"

    def test_scalar_properties_survive(self, phytomer_mtg):
        """Scalar properties preserved after roundtrip."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_mtg"
            write_mtg_file(phytomer_mtg, path)
            g2 = read_mtg_with_arrays(path)
            orig_lengths = phytomer_mtg.property('length')
            loaded_lengths = g2.property('length')
            for vid, val in orig_lengths.items():
                assert vid in loaded_lengths, f"Missing length for vid {vid}"
                assert abs(loaded_lengths[vid] - val) < 1e-10

    def test_array_properties_survive(self, phytomer_mtg):
        """Array properties preserved via npz companion."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_mtg"
            write_mtg_file(phytomer_mtg, path)
            g2 = read_mtg_with_arrays(path)
            orig_skels = phytomer_mtg.property('skeleton')
            loaded_skels = g2.property('skeleton')
            for vid, arr in orig_skels.items():
                assert vid in loaded_skels, f"Missing skeleton for vid {vid}"
                np.testing.assert_array_equal(loaded_skels[vid], arr)

    def test_topology_survives(self, phytomer_mtg):
        """Parent-child relationships preserved after roundtrip."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_mtg"
            write_mtg_file(phytomer_mtg, path)
            g2 = read_mtg_with_arrays(path)
            for vid in phytomer_mtg.vertices(scale=3):
                orig_children = sorted(phytomer_mtg.children(vid))
                loaded_children = sorted(g2.children(vid))
                assert orig_children == loaded_children, \
                    f"Children of {vid}: {orig_children} → {loaded_children}"


# ---------------------------------------------------------------------------
# TestInternodeSplitting
# ---------------------------------------------------------------------------

class TestInternodeSplitting:
    """Stem correctly split into per-phytomer internodes."""

    def test_one_internode_per_phytomer(self, phytomer_mtg):
        """Each metamer has exactly one internode."""
        for metamer in phytomer_mtg.vertices(scale=3):
            components = list(phytomer_mtg.components(metamer))
            internodes = [v for v in components
                          if phytomer_mtg.label(v) == 'I']
            assert len(internodes) == 1, \
                f"Metamer {metamer} has {len(internodes)} internodes"

    def test_concatenated_lengths_approx_stem(self):
        """Sum of internode lengths ≈ total stem length."""
        plant = _grow_phytomer_plant()
        g = cplantbox_to_mtg(plant, decompose_phytomer=True)

        # Total internode length from MTG
        total_internode = sum(
            g.property('length').get(v, 0)
            for v in g.vertices(scale=4)
            if g.label(v) == 'I'
        )

        # Actual stem length from CPlantBox
        stem = plant.getOrgans(pb.stem)[0]
        nodes = stem.getNodes()
        skel = np.array([[n.x, n.y, n.z] for n in nodes])
        stem_length = float(np.sum(np.linalg.norm(np.diff(skel, axis=0), axis=1)))

        ratio = total_internode / stem_length if stem_length > 0 else 0
        print(f"\nInternode sum: {total_internode:.2f} cm, "
              f"Stem: {stem_length:.2f} cm, ratio: {ratio:.3f}")
        assert 0.8 < ratio < 1.2, \
            f"Internode sum / stem length = {ratio:.3f}, expected ~1.0"

    def test_internode_skeleton_shape(self, phytomer_mtg):
        """Each internode has skeleton shape (N, 3) with N >= 2."""
        for vid in phytomer_mtg.vertices(scale=4):
            if phytomer_mtg.label(vid) != 'I':
                continue
            skel = phytomer_mtg.property('skeleton').get(vid)
            assert skel is not None, f"Internode {vid} has no skeleton"
            assert skel.shape[0] >= 2, \
                f"Internode {vid} skeleton has {skel.shape[0]} points (need >= 2)"
            assert skel.shape[1] == 3
