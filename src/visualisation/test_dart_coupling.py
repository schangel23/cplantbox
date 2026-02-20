"""
Test script for CPlantBox → DART coupling workflow.

This script tests the complete workflow without requiring DART installation.
Uses dummy DART results to verify the mapping works correctly.
"""

import sys
import numpy as np
import plantbox as pb
from dart_coupling import (
    export_plant_for_dart,
    map_dart_to_cplantbox_segments,
    validate_segment_mapping
)


def test_basic_export():
    """Test basic export functionality."""
    print("=" * 70)
    print("TEST 1: Basic Export")
    print("=" * 70)

    # Create simple plant
    plant = pb.MappedPlant()
    plant.readParameters("modelparameter/structural/plant/Zea_mays_1_Leitner_2010a.xml")
    plant.initialize()
    plant.simulate(10)  # Short simulation for testing

    print(f"Created plant with {len(plant.getOrgans())} organs")

    # Export
    vis = pb.PlantVisualiser(plant)
    mapping = export_plant_for_dart(vis, plant, "test_plant.obj")

    # Verify files created
    import os
    assert os.path.exists("test_plant.obj"), "OBJ file not created"
    assert os.path.exists("test_plant_mapping.json"), "Mapping file not created"

    print("\n✓ Export successful")
    return mapping, plant


def test_mapping_validation(mapping, plant):
    """Test mapping validation."""
    print("\n" + "=" * 70)
    print("TEST 2: Mapping Validation")
    print("=" * 70)

    is_valid = validate_segment_mapping("test_plant_mapping.json", plant)
    assert is_valid, "Mapping validation failed"

    print("\n✓ Mapping validation passed")


def test_dart_feedback(mapping, plant):
    """Test DART result feedback to segments."""
    print("\n" + "=" * 70)
    print("TEST 3: DART → CPlantBox Feedback")
    print("=" * 70)

    # Simulate DART results (random APAR per triangle in W/m²)
    n_triangles = mapping["total_triangles"]
    dart_triangle_apar_wm2 = np.random.uniform(50, 300, n_triangles)  # W/m²

    print(f"Generated dummy DART results for {n_triangles} triangles")
    print(f"  APAR range: {dart_triangle_apar_wm2.min():.1f} - {dart_triangle_apar_wm2.max():.1f} W/m²")

    # Map to segments with unit conversion
    segment_ppfd = map_dart_to_cplantbox_segments(
        dart_triangle_apar_wm2,
        "test_plant_mapping.json",
        plant,
        convert_units=True
    )

    print(f"\nMapped to segments and converted units:")
    print(f"  Array shape: {segment_ppfd.shape}")
    print(f"  PPFD range: {segment_ppfd.min():.6f} - {segment_ppfd.max():.6f} mol photons cm⁻² d⁻¹")

    # Verify array size
    expected_segments = mapping["total_segments"]
    assert segment_ppfd.shape[0] == expected_segments, \
        f"Segment array size mismatch: {segment_ppfd.shape[0]} != {expected_segments}"

    print("\n✓ DART feedback mapping successful")
    return segment_ppfd


def test_segment_order_consistency(mapping, plant):
    """Test that segment order matches CPlantBox internal order."""
    print("\n" + "=" * 70)
    print("TEST 4: Segment Order Consistency")
    print("=" * 70)

    # Check if using MappedPlant
    if not hasattr(plant, 'getSegmentIds'):
        print("⚠  Not a MappedPlant - skipping global segment order test")
        return

    # Get global segment IDs
    global_seg_ids = plant.getSegmentIds()
    print(f"MappedPlant has {len(global_seg_ids)} segments in global order")

    # Check mapping has global segment indices
    triangle_to_segment = mapping["triangle_to_segment"]
    has_global_indices = any(
        "global_segment_index" in tri_data
        for tri_data in triangle_to_segment.values()
    )

    if has_global_indices:
        print("✓ Mapping includes global segment indices")
    else:
        print("⚠  Mapping missing global segment indices - using sequential order")

    # Verify no duplicate global indices
    if has_global_indices:
        global_indices = set()
        for tri_data in triangle_to_segment.values():
            if "global_segment_index" in tri_data:
                idx = tri_data["global_segment_index"]
                assert idx not in global_indices, f"Duplicate global segment index: {idx}"
                global_indices.add(idx)

        print(f"✓ All global segment indices unique ({len(global_indices)} segments)")


def test_unit_conversion(segment_ppfd):
    """Test unit conversion validation."""
    print("\n" + "=" * 70)
    print("TEST 5: Unit Conversion Validation")
    print("=" * 70)

    # Conversion already done by map_dart_to_cplantbox_segments
    # Just validate the values are reasonable

    print(f"CPlantBox PPFD [mol cm⁻² d⁻¹]: {segment_ppfd.min():.6f} - {segment_ppfd.max():.6f}")

    # Sanity check: daily PPFD should be reasonable
    # Typical values: 0.01 - 0.15 mol photons cm⁻² d⁻¹
    # (50-300 W/m² × 4.6 / 1e6 × 86400 / 10000 = 0.020 - 0.119)
    assert segment_ppfd.max() < 0.2, "PPFD too high - check conversion"
    assert segment_ppfd.min() > 0, "PPFD should be positive"
    assert segment_ppfd.min() > 0.01, "PPFD too low - check conversion"

    print("\n✓ Unit conversion validation passed")
    print(f"  Expected range for 50-300 W/m²: 0.020 - 0.119 mol photons cm⁻² d⁻¹")
    print(f"  Actual range: {segment_ppfd.min():.6f} - {segment_ppfd.max():.6f}")

    return segment_ppfd


def test_triangle_to_segment_logic(mapping):
    """Test triangle-to-segment assignment logic."""
    print("\n" + "=" * 70)
    print("TEST 6: Triangle-to-Segment Assignment")
    print("=" * 70)

    triangle_to_segment = mapping["triangle_to_segment"]

    # Check each triangle is assigned to exactly one segment
    for tri_idx, tri_data in triangle_to_segment.items():
        assert "organ_id" in tri_data, f"Triangle {tri_idx} missing organ_id"
        assert "local_segment_index" in tri_data, f"Triangle {tri_idx} missing local_segment_index"
        assert "node_ids" in tri_data, f"Triangle {tri_idx} missing node_ids"
        assert len(tri_data["node_ids"]) == 3, f"Triangle {tri_idx} should have 3 node IDs"

    print(f"Checked {len(triangle_to_segment)} triangles")
    print("✓ All triangles properly assigned to segments")


def cleanup():
    """Remove test files."""
    import os
    for f in ["test_plant.obj", "test_plant_mapping.json", "test_plant.mtl"]:
        if os.path.exists(f):
            os.remove(f)
            print(f"Cleaned up {f}")


def run_all_tests():
    """Run complete test suite."""
    print("\n" + "🧪" * 35)
    print("CPlantBox → DART Coupling Test Suite")
    print("🧪" * 35 + "\n")

    try:
        # Test 1: Export
        mapping, plant = test_basic_export()

        # Test 2: Validation
        test_mapping_validation(mapping, plant)

        # Test 3: DART feedback
        segment_ppfd = test_dart_feedback(mapping, plant)

        # Test 4: Segment order
        test_segment_order_consistency(mapping, plant)

        # Test 5: Unit conversion validation
        ppfd_daily = test_unit_conversion(segment_ppfd)

        # Test 6: Assignment logic
        test_triangle_to_segment_logic(mapping)

        print("\n" + "=" * 70)
        print("✅ ALL TESTS PASSED")
        print("=" * 70)

    except Exception as e:
        print("\n" + "=" * 70)
        print("❌ TEST FAILED")
        print("=" * 70)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Cleanup
        print("\nCleaning up test files...")
        cleanup()


if __name__ == "__main__":
    run_all_tests()
