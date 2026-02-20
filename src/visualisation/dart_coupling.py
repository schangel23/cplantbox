"""
CPlantBox → DART Coupling using Baker's Approach
==================================================

This module provides functions to couple CPlantBox with DART radiative transfer
using the validated approach from Baker et al. (2025).

Key features:
- Export plant as OBJ with per-organ grouping
- Save triangle-to-segment mapping using node IDs
- Map DART per-triangle APAR back to CPlantBox segments
- Handle MappedPlant global segment ordering for photosynthesis

Based on Baker et al. (2025) Synavis methodology.
"""

import plantbox as pb
import numpy as np
import json
from plantbox.visualisation.vis_tools import WavefrontFromPlantGeometry


def export_plant_for_dart(vis: pb.PlantVisualiser, plant: pb.Plant,
                          filename: str, colour=None, resolution="organ",
                          geometry_resolution=16, leaf_resolution=30):
    """
    Export plant to OBJ with triangle-to-segment mapping for DART coupling.

    This uses the existing CPlantBox export but adds mapping data needed
    to feed DART results back to photosynthesis module.

    Args:
        vis: PlantVisualiser instance
        plant: Plant or MappedPlant instance
        filename: Output OBJ filename (e.g., "plant.obj")
        colour: Optional dict mapping organ types to RGB colors
        resolution: "organ" (default) or "type"
        geometry_resolution: Number of sides for stem cylinders (default 16)
        leaf_resolution: Leaf mesh resolution (default 30)

    Creates:
        - plant.obj: Wavefront OBJ file with per-organ groups
        - plant_mapping.json: Triangle-to-segment mapping for DART feedback

    Returns:
        Dictionary with export statistics

    Usage:
        vis = pb.PlantVisualiser(plant)
        stats = export_plant_for_dart(vis, plant, "maize.obj")
    """

    # Set mesh resolution before export
    vis.SetGeometryResolution(geometry_resolution)
    vis.SetLeafResolution(leaf_resolution)

    # Export using existing CPlantBox function (skip texcoords — DART doesn't support them)
    WavefrontFromPlantGeometry(vis, plant, filename, resolution=resolution, colour=colour,
                               skip_texcoords=True)

    # Build triangle-to-segment mapping
    mapping = _build_triangle_to_segment_mapping(vis, plant)

    # Save mapping
    mapping_file = filename.replace('.obj', '_mapping.json')
    with open(mapping_file, 'w') as f:
        json.dump(mapping, f, indent=2)

    print(f"Exported plant to {filename}")
    print(f"Saved triangle-to-segment mapping to {mapping_file}")
    print(f"  Total triangles: {mapping['total_triangles']}")
    print(f"  Total segments: {mapping['total_segments']}")
    print(f"  Total organs: {len(mapping['organs'])}")

    return mapping


def _build_triangle_to_segment_mapping(vis: pb.PlantVisualiser, plant: pb.Plant):
    """
    Build mapping from triangle indices to segment information using node IDs.

    This is the core of Baker's approach: using GetGeometryNodeIds() to link
    the 3D mesh triangles back to the 1D segment structure.

    Args:
        vis: PlantVisualiser instance
        plant: Plant instance

    Returns:
        Dictionary with triangle mapping data
    """

    organs = [o for o in plant.getOrgans() if o.organType() != 1]

    triangle_to_segment = {}  # tri_idx → (organ_id, local_seg_idx, global_seg_idx)
    global_triangle_counter = 0
    organ_info = {}

    for organ in organs:
        organ_id = organ.getId()
        organ_type = organ.organType()
        organ_segments = organ.getSegments()  # Vector2i with .x/.y = global node IDs

        # Compute geometry for this organ
        vis.ResetGeometry()
        vis.ComputeGeometryForOrgan(organ_id)

        # Get geometry data - KEY APIs from Baker's approach
        nodeids = np.array(vis.GetGeometryNodeIds())  # Maps vertices → global node IDs
        indices = np.reshape(vis.GetGeometryIndices(), (-1, 3))  # Triangles
        texcoords = np.reshape(vis.GetGeometryTextureCoordinates(), (-1, 2))  # UV coords

        if len(indices) == 0:
            continue

        # Build segment lookup by node pairs (seg.x/y are already global node IDs)
        segment_lookup = {}  # (node1, node2) → local_seg_idx
        for local_seg_idx, seg in enumerate(organ_segments):
            segment_lookup[(seg.x, seg.y)] = local_seg_idx
            segment_lookup[(seg.y, seg.x)] = local_seg_idx

        # Map each triangle to its segment
        for tri_idx_local, (v0, v1, v2) in enumerate(indices):
            # Get node IDs for this triangle's vertices
            tri_node_ids = [nodeids[v0], nodeids[v1], nodeids[v2]]

            # Find which segment these nodes belong to
            best_match = None
            max_matches = 0

            for local_seg_idx, seg in enumerate(organ_segments):
                seg_nodes = {seg.x, seg.y}
                matches = sum(1 for n in tri_node_ids if n in seg_nodes)

                if matches > max_matches:
                    max_matches = matches
                    best_match = local_seg_idx

            if best_match is not None:
                best_seg = organ_segments[best_match]
                # Average UV of the triangle's 3 vertices
                # U = cross-section position (0.5=midline), V = along-segment (0=base, 1=tip)
                tri_uv = texcoords[[v0, v1, v2]].mean(axis=0)
                triangle_to_segment[global_triangle_counter] = {
                    "organ_id": int(organ_id),
                    "organ_type": int(organ_type),
                    "local_segment_index": int(best_match),
                    "node_ids": [int(n) for n in tri_node_ids],
                    "segment_nodes": [int(best_seg.x), int(best_seg.y)],
                    "uv": [float(tri_uv[0]), float(tri_uv[1])]
                }

            global_triangle_counter += 1

        # Store organ info
        organ_type_name = {pb.root: "root", pb.stem: "stem", pb.leaf: "leaf"}.get(organ_type, "unknown")
        organ_info[str(organ_id)] = {
            "organ_type": int(organ_type),
            "organ_type_name": organ_type_name,
            "num_segments": len(organ_segments),
            "num_triangles": len(indices)
        }

    # Build global segment order using sequential organ ordering
    segment_to_global = {}
    global_counter = 0
    for organ in organs:
        organ_id = organ.getId()
        num_segs = len(organ.getSegments())
        for local_seg_idx in range(num_segs):
            segment_to_global[(organ_id, local_seg_idx)] = global_counter
            global_counter += 1

    # Add global segment indices to triangle mapping
    if segment_to_global is not None:
        for tri_idx in triangle_to_segment:
            tri_data = triangle_to_segment[tri_idx]
            key = (tri_data["organ_id"], tri_data["local_segment_index"])
            if key in segment_to_global:
                tri_data["global_segment_index"] = segment_to_global[key]

    return {
        "total_triangles": global_triangle_counter,
        "total_segments": sum(info["num_segments"] for info in organ_info.values()),
        "organs": organ_info,
        "triangle_to_segment": triangle_to_segment,
        "metadata": {
            "description": "Triangle-to-segment mapping for CPlantBox-DART coupling",
            "uses_global_segment_order": True,
            "method": "Baker et al. (2025) approach using node IDs"
        }
    }


def convert_dart_to_cplantbox_units(apar_wm2: np.ndarray) -> np.ndarray:
    """
    Convert DART APAR from W/m² to CPlantBox photosynthesis units.

    DART outputs: W/m² (radiometric irradiance)
    CPlantBox needs: mol photons cm⁻² d⁻¹ (photon flux)

    Conversion steps:
    1. W/m² → μmol photons m⁻² s⁻¹ (energy to photon flux in PAR)
    2. μmol m⁻² s⁻¹ → mol m⁻² s⁻¹ (divide by 1e6)
    3. mol m⁻² s⁻¹ → mol m⁻² d⁻¹ (multiply by 86400)
    4. mol m⁻² d⁻¹ → mol cm⁻² d⁻¹ (divide by 10000)

    Args:
        apar_wm2: APAR in W/m² (from DART)

    Returns:
        PPFD in mol photons cm⁻² d⁻¹ (for CPlantBox)

    Notes:
        - Conversion factor 4.6 μmol/J is typical for PAR (400-700 nm) under sunlight
        - See McCree (1972), Thimijan & Heins (1983) for PAR conversion factors
        - DART's spectral output allows more precise conversion if needed
    """
    # W/m² to μmol photons m⁻² s⁻¹
    # For PAR (400-700 nm), typical conversion: ~4.6 μmol photons per Joule
    # This assumes average PAR spectral quality
    PAR_CONVERSION_FACTOR = 4.6  # μmol photons per Joule in PAR range

    ppfd_umol_m2_s = apar_wm2 * PAR_CONVERSION_FACTOR

    # μmol m⁻² s⁻¹ → mol m⁻² s⁻¹
    ppfd_mol_m2_s = ppfd_umol_m2_s / 1e6

    # mol m⁻² s⁻¹ → mol m⁻² d⁻¹
    ppfd_mol_m2_d = ppfd_mol_m2_s * 86400  # seconds per day

    # mol m⁻² d⁻¹ → mol cm⁻² d⁻¹
    ppfd_mol_cm2_d = ppfd_mol_m2_d / 10000  # cm² per m²

    return ppfd_mol_cm2_d


def map_dart_to_cplantbox_segments(dart_triangle_apar: np.ndarray,
                                   mapping_file: str,
                                   plant: pb.Plant = None,
                                   convert_units: bool = True):
    """
    Map DART per-triangle APAR results back to CPlantBox segment array.

    This aggregates triangle-level DART results to segment level and orders
    them correctly for CPlantBox photosynthesis module.

    Args:
        dart_triangle_apar: Array of APAR values per triangle from DART [W/m²]
        mapping_file: Path to *_mapping.json file from export
        plant: Optional Plant instance for validation
        convert_units: If True, converts W/m² to mol photons cm⁻² d⁻¹ for photosynthesis

    Returns:
        Array of APAR values in CPlantBox segment order
        - If convert_units=False: [W/m²]
        - If convert_units=True: [mol photons cm⁻² d⁻¹]

    Usage:
        # After DART simulation
        dart_apar_wm2 = load_dart_results()  # DART outputs W/m²
        mapping_file = "maize_mapping.json"

        # Get segment APAR with unit conversion for photosynthesis
        segment_ppfd = map_dart_to_cplantbox_segments(
            dart_apar_wm2, mapping_file, plant, convert_units=True
        )

        # Feed to photosynthesis
        photo.Qlight = segment_ppfd
    """

    # Load mapping
    with open(mapping_file, 'r') as f:
        mapping = json.load(f)

    triangle_to_segment = mapping["triangle_to_segment"]
    total_segments = mapping["total_segments"]

    # Aggregate triangles to segments
    segment_apar_dict = {}  # (organ_id, local_seg_idx) → [apar_values]

    for tri_idx_str, tri_data in triangle_to_segment.items():
        tri_idx = int(tri_idx_str)

        if tri_idx >= len(dart_triangle_apar):
            print(f"WARNING: Triangle {tri_idx} not in DART results")
            continue

        apar_value = dart_triangle_apar[tri_idx]

        organ_id = tri_data["organ_id"]
        local_seg_idx = tri_data["local_segment_index"]
        key = (organ_id, local_seg_idx)

        if key not in segment_apar_dict:
            segment_apar_dict[key] = []
        segment_apar_dict[key].append(apar_value)

    # Average APAR per segment (area-weighted would be more accurate)
    segment_apar_averaged = {}
    for key, values in segment_apar_dict.items():
        segment_apar_averaged[key] = np.mean(values)

    print(f"Aggregated {len(dart_triangle_apar)} triangles → {len(segment_apar_averaged)} segments")

    # Convert to array in correct order
    if mapping["metadata"]["uses_global_segment_order"]:
        # Use global segment indices from mapping
        segment_apar_array = np.zeros(total_segments)

        for tri_data in triangle_to_segment.values():
            if "global_segment_index" in tri_data:
                global_idx = tri_data["global_segment_index"]
                organ_id = tri_data["organ_id"]
                local_seg_idx = tri_data["local_segment_index"]
                key = (organ_id, local_seg_idx)

                if key in segment_apar_averaged:
                    segment_apar_array[global_idx] = segment_apar_averaged[key]
    else:
        # Fallback: sequential order (may not match MappedPlant!)
        print("WARNING: Using sequential segment order - may not match MappedPlant!")
        segment_apar_array = np.zeros(total_segments)

        sorted_keys = sorted(segment_apar_averaged.keys())
        for idx, key in enumerate(sorted_keys):
            segment_apar_array[idx] = segment_apar_averaged[key]

    # Validate if plant provided
    if plant is not None:
        validate_segment_mapping(mapping_file, plant)

    # Convert units if requested
    if convert_units:
        segment_apar_array = convert_dart_to_cplantbox_units(segment_apar_array)
        print(f"Converted W/m² → mol photons cm⁻² d⁻¹")
        print(f"  Range: {segment_apar_array.min():.6f} - {segment_apar_array.max():.6f} mol photons cm⁻² d⁻¹")

    return segment_apar_array


def validate_segment_mapping(mapping_file: str, plant: pb.Plant):
    """
    Validate that the saved mapping matches the current plant structure.

    Args:
        mapping_file: Path to *_mapping.json
        plant: CPlantBox Plant instance

    Returns:
        True if valid, False otherwise
    """
    with open(mapping_file, 'r') as f:
        mapping = json.load(f)

    organs = [o for o in plant.getOrgans() if o.organType() != 1]

    # Check organ count
    if len(organs) != len(mapping["organs"]):
        print(f"ERROR: Organ count mismatch! Plant: {len(organs)}, Mapping: {len(mapping['organs'])}")
        return False

    # Check segment counts per organ
    for organ in organs:
        organ_id = str(organ.getId())
        if organ_id not in mapping["organs"]:
            print(f"ERROR: Organ {organ_id} not found in mapping")
            return False

        organ_segments = organ.getSegments()
        mapped_segs = mapping["organs"][organ_id]["num_segments"]

        if len(organ_segments) != mapped_segs:
            print(f"ERROR: Organ {organ_id} segment count mismatch! "
                  f"Plant: {len(organ_segments)}, Mapping: {mapped_segs}")
            return False

    print(f"✓ Mapping validated: {len(organs)} organs, {mapping['total_segments']} segments")
    return True


def load_dart_triangle_results(dart_output_file: str, variable_name='aPAR'):
    """
    Load DART per-triangle results from NetCDF output.

    Args:
        dart_output_file: Path to DART radiativeBudget_Triangles.nc file
        variable_name: Variable to extract (default: 'aPAR')

    Returns:
        Array of values per triangle

    Note:
        This is a placeholder - actual implementation depends on DART output format.
        Use pytools4dart to read DART NetCDF files.
    """
    try:
        import netCDF4 as nc
        dataset = nc.Dataset(dart_output_file)
        triangle_apar = dataset.variables[variable_name][:]
        dataset.close()
        return np.array(triangle_apar)
    except ImportError:
        print("ERROR: netCDF4 not installed. Install with: pip install netCDF4")
        raise
    except Exception as e:
        print(f"ERROR loading DART results: {e}")
        raise


# ============================================================================
# Complete Workflow Example
# ============================================================================

def example_cplantbox_dart_workflow():
    """
    Complete example of CPlantBox → DART → CPlantBox coupling workflow
    using Baker's approach.
    """
    print("=" * 70)
    print("CPlantBox → DART Coupling Workflow (Baker's Approach)")
    print("=" * 70)

    # Step 1: Create plant
    print("\nStep 1: Create Plant")
    print("-" * 70)
    plant = pb.MappedPlant()
    path = "modelparameter/structural/plant/"
    plant.readParameters(path + "Zea_mays_1_Leitner_2010.xml")
    plant.initialize()
    plant.simulate(30)
    print(f"✓ Plant simulated: {len(plant.getOrgans())} organs")

    # Step 2: Export with mapping
    print("\nStep 2: Export to OBJ with Triangle-to-Segment Mapping")
    print("-" * 70)
    vis = pb.PlantVisualiser(plant)
    mapping = export_plant_for_dart(vis, plant, "maize_dart.obj")

    # Step 3: Run DART (placeholder)
    print("\nStep 3: DART Simulation")
    print("-" * 70)
    print("In practice, you would:")
    print("  1. Import maize_dart.obj into DART/pytools4dart")
    print("  2. Assign optical properties (PROSPECT model)")
    print("  3. Run radiative transfer simulation")
    print("  4. Extract radiativeBudget_Triangles.nc")
    print("\nGenerating dummy DART results for demonstration...")

    # Dummy DART results (W/m²)
    n_triangles = mapping["total_triangles"]
    dart_triangle_apar_wm2 = np.random.uniform(50, 300, n_triangles)  # W/m²
    print(f"✓ Simulated DART results: {n_triangles} triangles")
    print(f"  APAR range: {dart_triangle_apar_wm2.min():.1f} - {dart_triangle_apar_wm2.max():.1f} W/m²")

    # Step 4: Map back to segments with unit conversion
    print("\nStep 4: Map DART Results to CPlantBox Segments")
    print("-" * 70)
    segment_ppfd = map_dart_to_cplantbox_segments(
        dart_triangle_apar_wm2, "maize_dart_mapping.json", plant, convert_units=True
    )
    print(f"✓ PPFD array for photosynthesis: shape={segment_ppfd.shape}")

    # Step 5: Feed to photosynthesis
    print("\nStep 5: Use with CPlantBox Photosynthesis")
    print("-" * 70)
    print("To use with photosynthesis module:")
    print("")
    print("  from plantbox.functional.Photosynthesis import PhotosynthesisPython")
    print("")
    print("  # segment_ppfd is already in correct units (mol photons cm⁻² d⁻¹)")
    print("  # No additional conversion needed!")
    print("")
    print("  # Initialize photosynthesis")
    print("  photo = PhotosynthesisPython(plant, params)")
    print("")
    print("  # Set per-segment light (already converted from W/m²)")
    print("  photo.Qlight = segment_ppfd")
    print("")
    print("  # Solve")
    print("  photo.solve(sim_time=30, ...)")
    print("")
    print("  # Get net assimilation per segment")
    print("  An = photo.get_net_assimilation()  # mol CO₂ d⁻¹ per segment")

    print("\n" + "=" * 70)
    print("✓ Workflow Complete!")
    print("=" * 70)


if __name__ == "__main__":
    example_cplantbox_dart_workflow()
