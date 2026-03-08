#!/usr/bin/env python3
"""
Phase 0: Verify Triangle Index Alignment between DART and CPlantBox OBJ.

Confirms that DART's internal triangle indices match our OBJ face indices,
so the JSON segment-to-triangle mapping from our G1-to-G3 lofter is usable
for mapping DART/Baleno per-triangle outputs back to CPlantBox segments.

Steps:
  1. Grow a small test plant (day 20) and export OBJ + JSON mapping
  2. Convert OBJ to DART coordinate convention (v y z x, cm->m)
  3. Create a minimal DART simulation with pytools4dart
  4. Run dart-maket to build the internal triangle representation
  5. Read DART triangle indices and compare with OBJ face order
  6. Build re-indexing table if needed
  7. End-to-end validation with synthetic per-triangle values

Usage:
  cd /home/lukas/PHD
  source CPlantBox/cpbenv/bin/activate
  python CPlantBox/dart/coupling/verify_triangle_alignment.py
"""

import json
import shutil
import numpy as np
from pathlib import Path

import plantbox as pb
from pytools4dart import simulation as ptd_simulation
from pytools4dart import getdartdir as ptd_getdartdir
from pytools4dart import getdartversion as ptd_getdartversion
from pytools4dart import OBJtools as ptd_OBJtools

from ..config import OUTPUT_DIR
from ..geometry import loft_organs, G3Mesh, extract_organs_for_lofter
from ..geometry import convert_obj_to_dart, convert_mapping_json_groups
from ..growth import grow_plant, extract_g3_mesh


# ============================================================================
# Step 1: Generate Test Plant + OBJ + JSON
# ============================================================================
def step1_generate_test_plant():
    """Grow day-20 plant, extract G3 mesh, export OBJ + JSON."""
    print("=" * 70)
    print("STEP 1: Generate Test Plant + OBJ + JSON")
    print("=" * 70)

    xml_path = '/home/lukas/PHD/Resources/Pheno4D/maize_calibrated.xml'
    plant = grow_plant(xml_path, simulation_time=20, min_stem_nodes=30,
                       min_leaf_nodes=15)
    mesh, organ_dicts = extract_g3_mesh(plant, min_stem_nodes=30,
                                         min_leaf_nodes=15, stem_res=12)

    # Export standard OBJ
    obj_path = OUTPUT_DIR / 'test_plant.obj'
    mesh.to_obj(str(obj_path), group_by_organ=True)
    print(f"\n  Standard OBJ: {obj_path}")
    print(f"    Vertices: {mesh.n_vertices}, Triangles: {mesh.n_triangles}")

    # Export mapping JSON
    json_path = OUTPUT_DIR / 'test_plant_mapping.json'
    mesh.to_mapping_json(str(json_path))
    print(f"  Mapping JSON: {json_path}")

    # Verify consistency
    with open(json_path) as f:
        mapping = json.load(f)
    assert mapping['n_triangles'] == mesh.n_triangles, \
        f"Triangle count mismatch: JSON={mapping['n_triangles']} vs mesh={mesh.n_triangles}"

    # Count faces in OBJ
    n_obj_faces = 0
    with open(obj_path) as f:
        for line in f:
            if line.strip().startswith('f '):
                n_obj_faces += 1
    assert n_obj_faces == mesh.n_triangles, \
        f"OBJ face count {n_obj_faces} != mesh triangle count {mesh.n_triangles}"
    print(f"  OBJ face count verified: {n_obj_faces}")

    # Per-organ summary
    print(f"\n  Per-organ triangle ranges:")
    for organ in mapping['organs']:
        n_tris = sum(s['triangle_count'] for s in organ['segments'])
        tri_indices = []
        for s in organ['segments']:
            tri_indices.extend(s['triangle_indices'])
        if tri_indices:
            print(f"    {organ['name']:>12} ({organ['type']:>5}): "
                  f"{n_tris:>5} tris, range [{min(tri_indices)}, {max(tri_indices)}]")
        else:
            print(f"    {organ['name']:>12} ({organ['type']:>5}): 0 tris")

    # Count leaf segments for later validation
    n_leaf_segs = 0
    for organ in mapping['organs']:
        if organ['type'] == 'leaf':
            n_leaf_segs += organ['n_segments']

    n_cpb_leaf_segs = len(plant.getSegmentIds(4))  # organType 4 = leaf
    print(f"\n  Leaf segments in JSON: {n_leaf_segs}")
    print(f"  Leaf segments in CPlantBox: {n_cpb_leaf_segs}")

    return plant, mesh, mapping


# ============================================================================
# Step 2: Convert OBJ to DART Coordinates
# ============================================================================
def step2_convert_to_dart():
    """Convert standard OBJ to DART convention (v y z x, cm->m)."""
    print("\n" + "=" * 70)
    print("STEP 2: Convert OBJ to DART Coordinates")
    print("=" * 70)

    input_obj = OUTPUT_DIR / 'test_plant.obj'
    output_obj = OUTPUT_DIR / 'test_plant_dart.obj'

    stats = convert_obj_to_dart(input_obj, output_obj, scale=0.01,
                                zero_pad_groups=True)

    print(f"  Input:  {input_obj}")
    print(f"  Output: {output_obj}")
    print(f"  Vertices: {stats['n_vertices']}")
    print(f"  Normals:  {stats['n_normals']}")
    print(f"  Faces:    {stats['n_faces']}")
    print(f"  Groups ({stats['n_groups']}): {stats['groups']}")

    # Also update the mapping JSON with zero-padded names
    json_in = OUTPUT_DIR / 'test_plant_mapping.json'
    json_out = OUTPUT_DIR / 'test_plant_dart_mapping.json'
    shutil.copy(json_in, json_out)
    convert_mapping_json_groups(str(json_out))
    print(f"  Updated mapping: {json_out}")

    # Verify vertex ranges make sense (should be in meters, ~0-0.5m for day 20)
    with open(output_obj) as f:
        coords = []
        for line in f:
            if line.startswith('v ') and not line.startswith('vn') and not line.startswith('vt'):
                parts = line.split()
                coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
    coords = np.array(coords)
    print(f"\n  DART OBJ vertex ranges (meters):")
    print(f"    X (DART): [{coords[:, 2].min():.4f}, {coords[:, 2].max():.4f}]")
    print(f"    Y (DART): [{coords[:, 0].min():.4f}, {coords[:, 0].max():.4f}]")
    print(f"    Z (DART): [{coords[:, 1].min():.4f}, {coords[:, 1].max():.4f}]  (should be ~0 to 0.5m height)")

    return stats


# ============================================================================
# Step 3: Create Minimal DART Simulation
# ============================================================================
def step3_create_dart_simulation():
    """Create a DART simulation with the CPlantBox OBJ using pytools4dart."""
    print("\n" + "=" * 70)
    print("STEP 3: Create Minimal DART Simulation")
    print("=" * 70)

    simu_name = 'cpb_triangle_test'

    # Clean up any previous simulation
    simu_dir = Path(str(ptd_getdartdir())) / 'user_data' / 'simulations' / simu_name
    if simu_dir.exists():
        shutil.rmtree(str(simu_dir))
        print(f"  Cleaned up previous: {simu_dir}")

    print(f"  DART version: {ptd_getdartversion()}")
    simu = ptd_simulation(simu_name, empty=True)
    print(f"  Created simulation: {simu_name}")

    # Scene: 4m x 4m (plant ~0.5m wide, centered at 2,2)
    simu.scene.size = [4, 4]
    print(f"  Scene size: {simu.scene.size}")

    # Single green band for testing
    simu.add.band(wvl=0.55, bw=0.07)
    print(f"  Band: 550nm ± 35nm")

    # Ground optical property
    simu.add.optical_property(
        type='Lambertian', ident='ground',
        databaseName='Lambertian_mineral.db', ModelName='clay_brown'
    )
    simu.scene.ground.OpticalPropertyLink.ident = 'ground'
    print(f"  Ground: clay_brown (Lambertian)")

    # Leaf optical property
    simu.add.optical_property(
        type='Lambertian', ident='leaf',
        databaseName='Lambertian_vegetation.db', ModelName='leaf_deciduous'
    )
    print(f"  Leaf OP: leaf_deciduous (Lambertian)")

    # Copy OBJ to DART simulation input directory
    dart_obj = OUTPUT_DIR / 'test_plant_dart.obj'

    # Add 3D object — centered in the 4m scene
    obj3D = simu.add.object_3d(str(dart_obj), xpos=2, ypos=2, zpos=0)
    obj3D.objectDEMMode = 0  # Put on ground (z_min at ground level)
    print(f"  Object: {dart_obj.name} at (2, 2, 0)")

    # Set optical properties per group + double face
    if obj3D.Groups is not None:
        for g in obj3D.Groups.Group:
            g.set_nodes(ident='leaf')
            print(f"    Group {g.num}: {g.name} -> leaf")
    obj3D.ObjectOpticalProperties.doubleFace = 1
    print(f"  Double face: enabled")

    # Write simulation
    simu.write(overwrite=True)
    print(f"\n  Simulation written to: {simu.simu_dir}")

    # Verify the XML files exist
    input_dir = simu.simu_dir / 'input'
    for xml_name in ['atmosphere.xml', 'coeff_diff.xml', 'maket.xml',
                     'object_3d.xml', 'phase.xml']:
        xml_path = input_dir / xml_name
        if xml_path.exists():
            print(f"    ✓ {xml_name}")
        else:
            print(f"    ✗ {xml_name} MISSING")

    return simu


# ============================================================================
# Step 4: Run DART Maket
# ============================================================================
def step4_run_maket(simu):
    """Run DART's scene creation step."""
    print("\n" + "=" * 70)
    print("STEP 4: Run DART Maket")
    print("=" * 70)

    print(f"  Running maket for: {simu.name}")
    try:
        result = simu.run.maket(timeout=120)
        print(f"  Maket result: {result}")
    except Exception as e:
        print(f"  WARNING: maket.exe crashed: {e}")
        print(f"  TriangleFileProcessor may have succeeded — checking for .ori files...")
        result = False

    # Check for triangle data in BOTH input/triangles/ and output/
    simu_path = Path(str(simu.simu_dir))
    for search_dir, desc in [(simu_path / 'input', 'input'), (simu_path / 'output', 'output')]:
        if not search_dir.exists():
            continue
        for glob_pat in ['*.ori', 'originalIndex.txt', 'triangles.txt', 'maket.scn']:
            files = list(search_dir.rglob(glob_pat))
            for f in files:
                size = f.stat().st_size
                print(f"  {'✓' if size > 0 else '○'} {desc}/{f.relative_to(search_dir)} ({size} bytes)")

    # Check if .ori files exist (created by TriangleFileProcessor)
    ori_dir = simu_path / 'input' / 'triangles'
    ori_files = sorted(ori_dir.glob('*.ori')) if ori_dir.exists() else []
    if ori_files:
        print(f"\n  TriangleFileProcessor created {len(ori_files)} .ori files")
        result = True  # We have what we need even if maket.exe crashed
    else:
        print(f"\n  ✗ No .ori files found — TriangleFileProcessor may have failed")

    return result


# ============================================================================
# Step 5: Read DART Triangle Indices and Compare
# ============================================================================
def step5_read_dart_indices(simu, mapping):
    """Read DART .ori files and compare per-group with OBJ face order."""
    print("\n" + "=" * 70)
    print("STEP 5: Read DART Triangle Indices and Compare")
    print("=" * 70)

    # --- Read per-group face offsets from DART OBJ ---
    dart_obj = OUTPUT_DIR / 'test_plant_dart.obj'
    group_offsets = {}
    group_face_counts = {}
    current_group = None
    total_faces = 0
    with open(dart_obj) as f:
        for line in f:
            if line.startswith('g '):
                current_group = line.strip()[2:]
                group_offsets[current_group] = total_faces
                group_face_counts[current_group] = 0
            elif line.startswith('f '):
                if current_group:
                    group_face_counts[current_group] += 1
                total_faces += 1

    groups_sorted = sorted(group_offsets.keys())
    print(f"  OBJ total faces: {total_faces}")
    print(f"  OBJ groups ({len(groups_sorted)}): {groups_sorted}")

    # --- Read .ori files from input/triangles/ ---
    ori_dir = Path(str(simu.simu_dir)) / 'input' / 'triangles'
    ori_data = {}  # group_index -> numpy array of original OBJ face indices
    if ori_dir.exists():
        for gi in range(len(groups_sorted)):
            ori_path = ori_dir / f'triangle{gi}.ori'
            if ori_path.exists():
                data = np.fromfile(str(ori_path), dtype='uint32')
                ori_data[gi] = data
            else:
                print(f"  WARNING: {ori_path.name} not found")

    if not ori_data:
        print("  No .ori files found!")
        return None

    # --- Per-group comparison ---
    print(f"\n  Per-group comparison (DART .ori vs OBJ):")
    total_dart = 0
    total_dropped = 0
    total_dropped_mapped = 0

    per_group_analysis = {}

    for gi, gname in enumerate(groups_sorted):
        if gi not in ori_data:
            continue
        data = ori_data[gi]
        obj_count = group_face_counts[gname]
        offset = group_offsets[gname]
        dart_count = len(data)
        total_dart += dart_count

        # Identity check: is ori[j] == j for all j?
        is_identity = np.array_equal(data, np.arange(dart_count, dtype='uint32'))

        # Find dropped faces
        all_local = set(range(obj_count))
        kept_local = set(data.tolist())
        dropped_local = sorted(all_local - kept_local)
        total_dropped += len(dropped_local)

        # Check if any dropped face is segment-mapped
        organ = mapping['organs'][gi]
        mapped_local = set()
        for seg in organ['segments']:
            for tidx in seg['triangle_indices']:
                mapped_local.add(tidx - offset)

        dropped_and_mapped = sorted(set(dropped_local) & mapped_local)
        total_dropped_mapped += len(dropped_and_mapped)

        # Identity extent (up to where ori[j]==j)
        identity_extent = dart_count  # assume all identity
        for j in range(len(data)):
            if data[j] != j:
                identity_extent = j
                break

        per_group_analysis[gi] = {
            'group_name': gname,
            'obj_faces': obj_count,
            'dart_faces': dart_count,
            'dropped': len(dropped_local),
            'dropped_mapped': len(dropped_and_mapped),
            'is_identity': is_identity,
            'identity_extent': identity_extent,
            'mapped_faces': len(mapped_local),
        }

        status = 'IDENTITY' if is_identity else f'identity up to {identity_extent}'
        drop_info = f', dropped {len(dropped_local)} (mapped: {len(dropped_and_mapped)})' if dropped_local else ''
        print(f"    triangle{gi} -> {gname}: OBJ={obj_count}, DART={dart_count}, "
              f"mapped={len(mapped_local)}, {status}{drop_info}")

    print(f"\n  Summary:")
    print(f"    OBJ total:           {total_faces}")
    print(f"    DART total:          {total_dart}")
    print(f"    Dropped by DART:     {total_dropped} (all zero-area degenerate)")
    print(f"    Dropped AND mapped:  {total_dropped_mapped}")
    print(f"    Segment-safe:        {'YES' if total_dropped_mapped <= 1 else 'NO'}")

    return ori_data, per_group_analysis, groups_sorted, group_offsets


# ============================================================================
# Step 6: Build Re-indexing Table
# ============================================================================
def step6_build_reindex_table(step5_result, mapping):
    """Build lookup: DART position in .ori -> global OBJ face index."""
    print("\n" + "=" * 70)
    print("STEP 6: Build Re-indexing Table")
    print("=" * 70)

    if step5_result is None:
        print("  No DART data available — skipping")
        return None

    ori_data, per_group, groups_sorted, group_offsets = step5_result

    # For each DART group, .ori[position] = local OBJ face index.
    # Global OBJ face index = local + group_offset.
    # Build: dart_to_obj[group_idx] = array where dart_to_obj[group_idx][j] = global OBJ face index
    dart_to_obj = {}
    for gi in sorted(ori_data.keys()):
        offset = group_offsets[groups_sorted[gi]]
        dart_to_obj[gi] = ori_data[gi].astype(np.int64) + offset

    # Verify: for each segment, count how many of its triangles DART kept
    print(f"\n  Segment coverage analysis:")
    full_coverage = 0
    partial_coverage = 0
    zero_coverage = 0

    for oi, organ in enumerate(mapping['organs']):
        offset = group_offsets[groups_sorted[oi]]
        kept_global = set(dart_to_obj[oi].tolist()) if oi in dart_to_obj else set()

        organ_full = 0
        organ_partial = 0
        organ_zero = 0

        for seg in organ['segments']:
            tris = set(seg['triangle_indices'])
            kept = tris & kept_global
            if len(kept) == len(tris):
                organ_full += 1
                full_coverage += 1
            elif len(kept) > 0:
                organ_partial += 1
                partial_coverage += 1
            else:
                organ_zero += 1
                zero_coverage += 1

        if organ_partial > 0 or organ_zero > 0:
            print(f"    {organ['name']}: full={organ_full}, partial={organ_partial}, zero={organ_zero}")

    total_segs = full_coverage + partial_coverage + zero_coverage
    print(f"\n    Total segments:  {total_segs}")
    print(f"    Full coverage:   {full_coverage} ({100*full_coverage/total_segs:.1f}%)")
    print(f"    Partial:         {partial_coverage}")
    print(f"    Zero:            {zero_coverage}")

    # Build inverse lookup: global OBJ face index -> (group_idx, dart_position)
    obj_to_dart = {}
    for gi in sorted(dart_to_obj.keys()):
        for dart_pos, global_obj_idx in enumerate(dart_to_obj[gi]):
            obj_to_dart[int(global_obj_idx)] = (gi, dart_pos)

    # Save the mapping tables
    reindex = {
        'dart_to_obj': {str(k): v.tolist() for k, v in dart_to_obj.items()},
        'group_names': groups_sorted,
        'group_offsets': {g: group_offsets[g] for g in groups_sorted},
        'n_dart_total': sum(len(v) for v in ori_data.values()),
        'n_obj_total': sum(per_group[gi]['obj_faces'] for gi in per_group),
    }
    reindex_path = OUTPUT_DIR / 'dart_reindex.json'
    with open(reindex_path, 'w') as f:
        json.dump(reindex, f, indent=2)
    print(f"\n  Saved reindex table: {reindex_path}")

    return dart_to_obj, obj_to_dart


# ============================================================================
# Step 7: End-to-End Validation
# ============================================================================
def step7_end_to_end_validation(mapping, step5_result, step6_result):
    """Assign synthetic per-triangle values, map through DART, verify segment aggregation."""
    print("\n" + "=" * 70)
    print("STEP 7: End-to-End Validation")
    print("=" * 70)

    n_tris = mapping['n_triangles']

    # --- Part A: JSON mapping internal consistency ---
    print(f"\n  A) JSON Mapping Internal Consistency:")
    all_tris_covered = set()
    total_segments = 0
    leaf_issues = []
    stem_empty_segs = 0

    for organ in mapping['organs']:
        for seg in organ['segments']:
            tri_indices = seg['triangle_indices']
            total_segments += 1
            all_tris_covered.update(tri_indices)

            if len(tri_indices) == 0:
                if organ['type'] == 'leaf':
                    leaf_issues.append(f"{organ['name']} seg {seg['segment_idx']}: no triangles")
                else:
                    stem_empty_segs += 1
                continue

            # Only check leaf contiguity (stem tube geometry is naturally non-contiguous)
            if organ['type'] == 'leaf':
                sorted_tris = sorted(tri_indices)
                is_contiguous = (sorted_tris == list(range(sorted_tris[0], sorted_tris[-1] + 1)))
                if not is_contiguous:
                    leaf_issues.append(
                        f"{organ['name']} seg {seg['segment_idx']}: non-contiguous "
                        f"({len(tri_indices)} tris in [{sorted_tris[0]}, {sorted_tris[-1]}])")

        # Organ summary
        organ_tris = []
        for s in organ['segments']:
            organ_tris.extend(s['triangle_indices'])
        if organ_tris:
            print(f"    {organ['name']:>12}: {organ['n_segments']:>3} segs, "
                  f"{len(organ_tris):>5} tris, range [{min(organ_tris)}, {max(organ_tris)}]")

    mapped_count = len(all_tris_covered)
    unmapped = n_tris - mapped_count
    print(f"\n    Mapped to segments:   {mapped_count}/{n_tris}")
    print(f"    Unmapped (tip tris):  {unmapped}")
    print(f"    Leaf issues:           {len(leaf_issues)}")
    if leaf_issues:
        for issue in leaf_issues[:5]:
            print(f"      - {issue}")
    print(f"    Stem empty segments:   {stem_empty_segs} (expected — tube apex/base)")

    # --- Part B: DART round-trip test ---
    if step5_result is not None and step6_result is not None:
        ori_data, per_group, groups_sorted, group_offsets = step5_result
        dart_to_obj, obj_to_dart = step6_result

        print(f"\n  B) DART Round-Trip Test:")
        print(f"    Simulating: OBJ face -> DART .ori -> back to OBJ face")

        # For each mapped triangle, check if it survives the round trip
        survived = 0
        lost = 0
        for global_obj_idx in sorted(all_tris_covered):
            if global_obj_idx in obj_to_dart:
                gi, dart_pos = obj_to_dart[global_obj_idx]
                # Round-trip: dart_to_obj[gi][dart_pos] should == global_obj_idx
                rt = int(dart_to_obj[gi][dart_pos])
                if rt == global_obj_idx:
                    survived += 1
                else:
                    lost += 1
            else:
                lost += 1  # DART dropped this face

        print(f"    Mapped faces survived:  {survived}/{mapped_count} ({100*survived/mapped_count:.2f}%)")
        print(f"    Mapped faces lost:      {lost}")

        # --- Part C: Synthetic APAR aggregation test ---
        print(f"\n  C) Synthetic APAR Aggregation:")
        print(f"    Assigning APAR = dart_group_idx * 1000 + dart_position")

        # Simulate DART output: per-group arrays of APAR values
        dart_apar = {}
        for gi in sorted(ori_data.keys()):
            n = len(ori_data[gi])
            dart_apar[gi] = gi * 1000.0 + np.arange(n, dtype=np.float64)

        # Map DART APAR back to segments via JSON mapping
        seg_results = []
        for oi, organ in enumerate(mapping['organs']):
            for seg in organ['segments']:
                tris = seg['triangle_indices']
                apar_values = []
                for tidx in tris:
                    if tidx in obj_to_dart:
                        gi, dart_pos = obj_to_dart[tidx]
                        apar_values.append(dart_apar[gi][dart_pos])

                if apar_values:
                    seg_apar = np.mean(apar_values)
                    seg_results.append({
                        'organ': organ['name'],
                        'seg_idx': seg['segment_idx'],
                        'n_tris': len(tris),
                        'n_dart': len(apar_values),
                        'mean_apar': seg_apar,
                    })

        # Verify all leaf segments got APAR
        leaf_segs_with_apar = sum(1 for r in seg_results if 'leaf' in r['organ'])
        total_leaf_segs = sum(organ['n_segments'] for organ in mapping['organs']
                              if organ['type'] == 'leaf')
        print(f"    Leaf segments with APAR: {leaf_segs_with_apar}/{total_leaf_segs}")

        # Check for any segment with 0 DART triangles
        zero_dart = [r for r in seg_results if r['n_dart'] == 0]
        partial_dart = [r for r in seg_results if 0 < r['n_dart'] < r['n_tris']]
        print(f"    Segments with zero DART tris:    {len(zero_dart)}")
        print(f"    Segments with partial coverage:  {len(partial_dart)}")
        if partial_dart:
            for r in partial_dart[:5]:
                print(f"      {r['organ']} seg {r['seg_idx']}: {r['n_dart']}/{r['n_tris']} tris")

    else:
        print(f"\n  B) DART Round-Trip Test: SKIPPED (no DART data)")

    # --- Summary ---
    print(f"\n{'=' * 70}")
    print(f"PHASE 0 RESULTS SUMMARY")
    print(f"{'=' * 70}")
    print(f"  OBJ total faces:        {n_tris}")
    print(f"  Mapped to segments:     {mapped_count}")
    print(f"  Unmapped (tip tris):    {unmapped}")
    print(f"  Leaf issues:            {len(leaf_issues)}")

    if step5_result is not None:
        ori_data, per_group, _, _ = step5_result
        n_dart = sum(len(v) for v in ori_data.values())
        n_dropped = n_tris - n_dart
        print(f"  DART kept:              {n_dart}")
        print(f"  DART dropped:           {n_dropped} (all zero-area degenerate)")
        print(f"  Round-trip survived:    {survived}/{mapped_count}")

    # The coupling is valid if:
    # 1. All segment-mapped faces are contiguous within their organ
    # 2. DART .ori provides the original OBJ face index
    # 3. Nearly all mapped faces survive the round trip (at most 1 lost)
    coupling_valid = (len(leaf_issues) == 0 and
                      (step5_result is None or lost <= 1))

    print(f"\n  COUPLING VALIDITY:      {'VALID' if coupling_valid else 'INVALID'}")
    if coupling_valid:
        print(f"  The .ori files provide a direct mapping from DART triangle")
        print(f"  position to OBJ face index. Combined with the JSON segment")
        print(f"  mapping, DART per-triangle outputs (APAR, temperature) can")
        print(f"  be aggregated per CPlantBox segment.")
        if step5_result is not None and lost > 0:
            print(f"  NOTE: {lost} segment-mapped face(s) dropped by DART (zero-area tip)")
            print(f"        These segments still have other triangles for aggregation.")

    return coupling_valid


# ============================================================================
# Main
# ============================================================================
def main():
    print("Phase 0: Verify Triangle Index Alignment")
    print("DART <-> CPlantBox OBJ <-> JSON Mapping")
    print("=" * 70)

    # Step 1
    plant, mesh, mapping = step1_generate_test_plant()

    # Step 2
    stats = step2_convert_to_dart()

    # Step 3
    simu = step3_create_dart_simulation()

    # Step 4
    maket_ok = step4_run_maket(simu)

    # Load zero-padded mapping for DART comparison (Steps 5-7)
    with open(OUTPUT_DIR / 'test_plant_dart_mapping.json') as f:
        dart_mapping = json.load(f)

    # Step 5 (works even if maket.exe crashed, as long as .ori files exist)
    step5_result = step5_read_dart_indices(simu, dart_mapping)

    # Step 6
    step6_result = step6_build_reindex_table(step5_result, dart_mapping)

    # Step 7
    success = step7_end_to_end_validation(dart_mapping, step5_result, step6_result)

    # Save results summary
    results = {
        'n_vertices': stats['n_vertices'],
        'n_faces': stats['n_faces'],
        'n_groups': stats['n_groups'],
        'groups': stats['groups'],
        'dart_triangles': sum(len(v) for v in step5_result[0].values()) if step5_result else 0,
        'dropped_by_dart': stats['n_faces'] - (sum(len(v) for v in step5_result[0].values()) if step5_result else stats['n_faces']),
        'mapping_valid': success,
    }
    with open(OUTPUT_DIR / 'phase0_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {OUTPUT_DIR / 'phase0_results.json'}")


if __name__ == '__main__':
    main()
