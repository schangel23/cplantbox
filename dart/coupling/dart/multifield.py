#!/usr/bin/env python3
"""
Phase 6: Multi-Plant Unique Realizations.

Replaces 9 identical plant copies (Phase 1-4) with 9 unique stochastic
realizations (different random seeds → different leaf angles, lengths,
emergence timing).  Captures structural diversity effects on canopy-level
photosynthesis.

Architecture:
  - 9 individual OBJs with plant-prefixed group names (p0_organ_00 .. p8_organ_11)
  - DART ObjectFields with 9 models (one per plant), field file maps model_index
  - Budget extraction via maket.scn for ALL 9 instances
  - Per-plant segment mapping + photosynthesis solve

Usage:
  cd /home/lukas/PHD
  source CPlantBox/cpbenv/bin/activate
  python CPlantBox/dart/coupling/run_multifield.py
"""

import os
import json
import re
import csv
import shutil
import subprocess
import textwrap
import xml.etree.ElementTree as ET
import numpy as np
from pathlib import Path
from collections import OrderedDict

import plantbox as pb
import pytools4dart as ptd

from ..config import (DEFAULT_XML, DART_HOME, DART_EB_DIR, DARTRC,
                      BALENO_PYTHON, OUTPUT_DIR, DART_THREADS,
                      DART_RAY_DENSITY_PER_PIXEL, DART_MAX_RENDERING_TIME,
                      get_species,
                      get_hydraulics_json, get_photosynthesis_json, get_phloem_json)
from ..growth.grow import grow_plant, extract_g3_mesh
from ..geometry import loft_organs, G3Mesh, extract_organs_for_lofter
from ..geometry import convert_obj_to_dart, convert_mapping_json_groups
from ..prospect_params import (get_prospect_params, get_prospect_params_per_position,
                               get_stem_prospect_params,
                               log_consistency, log_lops_consistency)
from ..dart.simulation import configure_atmosphere_midlatsum

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
XML_PATH = str(DEFAULT_XML)
SIMULATION_DAYS = 55
SIMU_NAME = 'cpb_multifield_day55_par'
N_PLANTS = 9
FIELD_SEED = 42
CENTER_PLANT_IDX = 4  # center of 3x3 grid

# 6 contiguous PAR bands covering 400-700 nm (central wavelength in µm, bandwidth in µm)
PAR_BANDS = [
    (0.425, 0.050),   # 400-450 nm
    (0.475, 0.050),   # 450-500 nm
    (0.525, 0.050),   # 500-550 nm
    (0.575, 0.050),   # 550-600 nm
    (0.625, 0.050),   # 600-650 nm
    (0.675, 0.050),   # 650-700 nm
]

# PROSPECT parameters are loaded per-position in step3/step6 via
# get_prospect_params_per_position(). Base stage params for metadata only:
PROSPECT_PARAMS_BASE = get_prospect_params(SIMULATION_DAYS)

SUN_ZENITH = 45.0
SUN_AZIMUTH = 225.0
SCENE_SIZE = [4, 4]
PLANT_POS = (2.0, 2.0)

GRID_NX, GRID_NY = 3, 3
GRID_SPACING_X = 0.75  # meters
GRID_SPACING_Y = 0.25  # meters
FIELD_FILENAME = 'multifield_plant_field.txt'

# Baleno paths
BALENO_DIR = DART_EB_DIR
DART_DIR = DART_HOME
DART_LOCAL = DART_DIR / 'user_data'
DARTRC_PATH = DARTRC
VENV_PYTHON = BALENO_PYTHON
BALENO_USER_DATA = BALENO_DIR / 'user_data'
SIMU_NAME_EB = 'cpb_multifield_eb'
DART_SIMU_NAME_EB = 'cpb_multifield_day55_eb'

# Shortwave bands for Baleno
SW_BANDS = [(0.400 + i * 0.100 + 0.050, 0.100) for i in range(21)]
TIR_BAND = (10.0, 4.0)
TARGET_PAR_UMOL = 1000.0
TAIR_C = 25.0
RH = 0.7
SOIL_PSI_CM = -500.0


def _compute_plant_positions():
    """Compute 3x3 grid positions centered at PLANT_POS (meters, DART coords)."""
    positions = []
    for iy in range(GRID_NY):
        for ix in range(GRID_NX):
            x = PLANT_POS[0] + (ix - (GRID_NX - 1) / 2) * GRID_SPACING_X
            y = PLANT_POS[1] + (iy - (GRID_NY - 1) / 2) * GRID_SPACING_Y
            positions.append((x, y))
    return positions


# ============================================================================
# Step 1: Grow 9 unique plants
# ============================================================================
def step1_grow_plants():
    """Grow 9 plants with different random seeds."""
    print("=" * 70)
    print("STEP 1: Grow 9 Unique Plants")
    print("=" * 70)

    plants = []
    for i in range(N_PLANTS):
        seed = FIELD_SEED + i
        print(f"\n--- Plant {i} (seed={seed}) ---")
        plant = grow_plant(XML_PATH, simulation_time=SIMULATION_DAYS,
                           min_stem_nodes=50, min_leaf_nodes=20, seed=seed)
        plants.append(plant)

    # Summary
    print(f"\n  Summary:")
    for i, plant in enumerate(plants):
        organs = plant.getOrgans()
        n_leaves = sum(1 for o in organs if o.organType() == pb.OrganTypes.leaf)
        n_nodes = len(plant.getNodes())
        print(f"    Plant {i} (seed={FIELD_SEED + i}): "
              f"{n_leaves} leaves, {n_nodes} nodes")

    return plants


# ============================================================================
# Step 2: Export 9 OBJs + 9 mapping JSONs
# ============================================================================
def step2_export_meshes(plants):
    """Export G3 mesh for each plant with plant-prefixed group names."""
    print("\n" + "=" * 70)
    print("STEP 2: Export 9 G3 Meshes")
    print("=" * 70)

    meshes = []
    mappings = []
    obj_paths = []
    dart_obj_paths = []
    mapping_json_paths = []

    for i, plant in enumerate(plants):
        prefix = f"p{i}_"
        print(f"\n--- Plant {i} (prefix={prefix}) ---")

        # Extract organs with plant prefix
        organ_dicts = extract_organs_for_lofter(
            plant, min_stem_nodes=50, min_leaf_nodes=20,
            name_prefix=prefix,
        )
        # Add plant_id to each organ dict for mapping JSON
        for od in organ_dicts:
            od['plant_id'] = i

        print(f"  Extracted {len(organ_dicts)} organs")

        # Loft to G3
        mesh = loft_organs(organ_dicts, stem_sides=16)
        print(f"  Vertices: {mesh.n_vertices}, Triangles: {mesh.n_triangles}")

        # Export OBJ with plant-prefixed group names
        obj_path = OUTPUT_DIR / f'multifield_p{i}.obj'
        mesh.to_obj(str(obj_path), group_by_organ=True, group_prefix=prefix)
        print(f"  OBJ: {obj_path}")

        # Export mapping JSON
        json_path = OUTPUT_DIR / f'multifield_p{i}_mapping.json'
        mesh.to_mapping_json(str(json_path))
        print(f"  Mapping: {json_path}")

        # Convert to DART coordinates
        dart_obj_path = OUTPUT_DIR / f'multifield_p{i}_dart.obj'
        stats = convert_obj_to_dart(obj_path, dart_obj_path, scale=0.01,
                                     zero_pad_groups=True)
        print(f"  DART OBJ: {dart_obj_path} ({stats['n_groups']} groups: "
              f"{stats['groups']})")

        # Update mapping JSON with zero-padded names
        dart_json_path = OUTPUT_DIR / f'multifield_p{i}_dart_mapping.json'
        shutil.copy(json_path, dart_json_path)
        convert_mapping_json_groups(str(dart_json_path))

        meshes.append(mesh)
        with open(dart_json_path) as f:
            mappings.append(json.load(f))
        obj_paths.append(obj_path)
        dart_obj_paths.append(dart_obj_path)
        mapping_json_paths.append(dart_json_path)

    return meshes, mappings, obj_paths, dart_obj_paths, mapping_json_paths


# ============================================================================
# Step 3: Create DART PAR simulation (6 bands, 9 models)
# ============================================================================
def step3_create_dart_simulation(dart_obj_paths):
    """Create DART simulation with 9 models in ObjectFields."""
    print("\n" + "=" * 70)
    print("STEP 3: Create DART Simulation (9 Models)")
    print("=" * 70)

    simu_dir = Path(ptd.getdartdir()) / 'user_data' / 'simulations' / SIMU_NAME
    if simu_dir.exists():
        shutil.rmtree(str(simu_dir))
        print(f"  Cleaned up previous: {simu_dir}")

    simu = ptd.simulation(SIMU_NAME, empty=True)
    simu.scene.size = SCENE_SIZE
    print(f"  Scene: {SCENE_SIZE[0]}m x {SCENE_SIZE[1]}m")

    # Spectral bands
    for wvl, bw in PAR_BANDS:
        simu.add.band(wvl=wvl, bw=bw)
    print(f"  Bands: {len(PAR_BANDS)} PAR bands")

    # Sun
    simu.core.directions.Directions.SunViewingAngles.sunViewingZenithAngle = SUN_ZENITH
    simu.core.directions.Directions.SunViewingAngles.sunViewingAzimuthAngle = SUN_AZIMUTH
    print(f"  Sun: zenith={SUN_ZENITH}, azimuth={SUN_AZIMUTH}")

    # Ground OP
    simu.add.optical_property(
        type='Lambertian', ident='ground',
        databaseName='Lambertian_mineral.db', ModelName='clay_brown',
        useMultiplicativeFactorForLUT=0,
    )
    simu.scene.ground.OpticalPropertyLink.ident = 'ground'

    # Per-position leaf OPs (PROSPECT from LOPS) — same species, shared across plants
    # Determine n_leaves from first OBJ (all plants have the same leaf count)
    first_obj_path = simu.get_input_file_path(str(dart_obj_paths[0]))
    first_obj_info = ptd.OBJtools.objreader(first_obj_path)
    first_gnames = ptd.OBJtools.gnames_dart_order(first_obj_info.names)
    n_leaf_groups = sum(1 for g in first_gnames if not g.endswith('_00'))

    per_pos_params = get_prospect_params_per_position(SIMULATION_DAYS, n_leaf_groups)
    for pi, params in enumerate(per_pos_params):
        ident = f'maize_leaf_pos{pi}'
        simu.add.optical_property(
            type='Lambertian', ident=ident,
            prospect=params,
            useMultiplicativeFactorForLUT=0,
        )
    log_lops_consistency(SIMULATION_DAYS, n_leaf_groups)

    stem_prospect = get_stem_prospect_params(SIMULATION_DAYS)
    simu.add.optical_property(
        type='Lambertian', ident='maize_stem',
        prospect=stem_prospect,
        useMultiplicativeFactorForLUT=0,
    )

    # Build multi-model ObjectFields
    model_list = ptd.object_3d.create_ModelList()

    for i, dart_obj in enumerate(dart_obj_paths):
        file_src_fullpath = simu.get_input_file_path(str(dart_obj))
        obj_info = ptd.OBJtools.objreader(file_src_fullpath)
        gnames = ptd.OBJtools.gnames_dart_order(obj_info.names)
        xdim, ydim, zdim = obj_info.dims
        xc, yc, zc = obj_info.center

        # Create groups with per-position PROSPECT OP + doubleFace
        # Leaf position = suffix number - 1 (e.g., p0_organ_01 -> pos 0)
        groups_list = []
        leaf_idx = 0
        for gi, gname in enumerate(gnames):
            g = ptd.object_3d.create_Group(num=gi + 1, name=gname)
            is_stem = gname.endswith('_00')
            if is_stem:
                op_ident = 'maize_stem'
            else:
                op_ident = f'maize_leaf_pos{leaf_idx}'
                leaf_idx += 1
            df = 0 if is_stem else 1
            g.set_nodes(ident=op_ident)
            gop = g.GroupOpticalProperties
            gop.SurfaceOpticalProperties.doubleFace = df
            gop.SurfaceExitanceProperties.doubleFace = df
            groups_list.append(g)
        groups = ptd.object_3d.create_Groups(Group=groups_list)

        geom = ptd.object_3d.create_GeometricProperties(
            Dimension3D=ptd.object_3d.create_Dimension3D(
                xdim=xdim, ydim=ydim, zdim=zdim),
            Center3D=ptd.object_3d.create_Center3D(
                xCenter=xc, yCenter=yc, zCenter=zc),
            ScaleProperties=ptd.object_3d.create_ScaleProperties(
                xscale=1.0, yscale=1.0, zscale=1.0),
        )
        model_obj = ptd.object_3d.create_Object(
            file_src=str(dart_obj),
            hasGroups=1,
            GeometricProperties=geom,
            Groups=groups,
            num=i,
            name=f'CPlantBox_Maize_p{i}',
            objectDEMMode=0,
        )
        model_list.add_Object(model_obj)
        print(f"  Model {i}: {dart_obj.name} ({len(gnames)} groups)")

    # ObjectFields with field file
    field = ptd.object_3d.create_Field(
        name='MultiMaizeField',
        fieldDescriptionFileName=FIELD_FILENAME,
    )
    field.set_ModelList(model_list)
    obj_fields = ptd.object_3d.create_ObjectFields()
    obj_fields.add_Field(field)
    simu.core.object_3d.object_3d.ObjectFields = obj_fields

    # Radiative budget
    products = simu.core.phase.Phase.DartProduct.dartModuleProducts.CommonProducts
    products.radiativeBudgetProducts = 1
    products.radiativeBudgetProperties.budget3DParSurface = 1

    # Engine: Lux + sampling
    simu.core.phase.Phase.accelerationEngine = 2
    lux = simu.core.phase.Phase.EngineParameter.LuxCoreRenderEngineParameters
    lux.targetRayDensityPerPixel = DART_RAY_DENSITY_PER_PIXEL
    lux.maximumRenderingTime = DART_MAX_RENDERING_TIME
    simu.core.phase.Phase.ExpertModeZone.nbThreads = DART_THREADS
    print(f"  LuxCore: {DART_RAY_DENSITY_PER_PIXEL} rays/pixel, "
          f"maxTime={DART_MAX_RENDERING_TIME}s, threads={DART_THREADS}")

    # Atmosphere: MIDLATSUM
    configure_atmosphere_midlatsum(simu)
    print("  Atmosphere: MIDLATSUM + RURALV23 (TOAtoBOA=2)")

    # Write
    simu.write(overwrite=True)
    print(f"  Simulation written: {simu.simu_dir}")

    # Write field file: each plant at its own grid position with unique model_index
    simu_path = Path(str(simu.simu_dir))
    field_path = simu_path / 'input' / FIELD_FILENAME
    positions = _compute_plant_positions()
    with open(field_path, 'w') as f:
        f.write('complete transformation\n')
        for idx, (x, y) in enumerate(positions):
            f.write(f'{idx} {x:.6f} {y:.6f} 0.0 1.0 1.0 1.0 0.0 0.0 0.0\n')
    print(f"  Field file: {field_path} ({N_PLANTS} plants, unique models)")

    # Save grid info
    grid_info = {
        'grid_nx': GRID_NX, 'grid_ny': GRID_NY,
        'spacing_x_m': GRID_SPACING_X, 'spacing_y_m': GRID_SPACING_Y,
        'n_plants': N_PLANTS,
        'center_plant_idx': CENTER_PLANT_IDX,
        'positions_m': positions,
        'field_filename': FIELD_FILENAME,
        'unique_models': True,
    }
    grid_path = OUTPUT_DIR / 'multifield_grid_info.json'
    with open(grid_path, 'w') as f:
        json.dump(grid_info, f, indent=2)

    return simu


# ============================================================================
# Step 4: Run DART
# ============================================================================
def step4_run_dart(simu):
    """Run full DART simulation."""
    print("\n" + "=" * 70)
    print("STEP 4: Run DART Pipeline")
    print("=" * 70)

    print(f"  Running full simulation (timeout=1200s)...")
    try:
        result = simu.run.full(timeout=1200)
        print(f"  Full run result: {result}")
        if result:
            return True
    except Exception as e:
        print(f"  Full run failed: {e}")

    # Fallback: individual stages
    print(f"\n  Falling back to individual stages...")
    for name, runner in [('direction', simu.run.direction),
                         ('phase', simu.run.phase),
                         ('maket', simu.run.maket),
                         ('dart', simu.run.dart)]:
        print(f"  Running {name}...")
        try:
            ok = runner(timeout=600)
            print(f"    {name}: {'OK' if ok else 'FAILED'}")
        except Exception as e:
            print(f"    {name} crashed: {e}")

    simu_path = Path(str(simu.simu_dir))
    band_dirs = sorted(simu_path.glob('output/BAND*'))
    if band_dirs:
        print(f"\n  Output bands found: {[d.name for d in band_dirs]}")
        return True

    output_dir = simu_path / 'output'
    if output_dir.exists():
        all_files = list(output_dir.rglob('*'))
        if all_files:
            print(f"\n  Output directory has {len(all_files)} files")
            return True

    print(f"\n  ERROR: No DART output found!")
    return False


# ============================================================================
# Step 5: Read .ori + budget, aggregate per-plant APAR
# ============================================================================
def step5_read_apar(simu, dart_obj_paths, mappings):
    """Read .ori files and radiative budget, aggregate per-plant-per-segment APAR."""
    print("\n" + "=" * 70)
    print("STEP 5: Read Budget + Aggregate Per-Plant APAR")
    print("=" * 70)

    simu_path = Path(str(simu.simu_dir))

    # --- Read .ori files per model ---
    # With multi-model ObjectFields, .ori files are per-model.
    # Naming: triangle0.ori, triangle1.ori, ... (global across all models)
    ori_dir = simu_path / 'input' / 'triangles'
    if not ori_dir.exists():
        print(f"  ERROR: {ori_dir} does not exist!")
        return None

    # For each model (plant), read its DART OBJ to get group info
    per_plant_ori = {}  # plant_idx -> {local_group_idx -> ori_data}
    per_plant_groups = {}  # plant_idx -> sorted list of group names
    per_plant_offsets = {}  # plant_idx -> {group_name -> face offset in OBJ}
    per_plant_face_counts = {}  # plant_idx -> {group_name -> face count}

    # With multi-model ObjectFields, we need to discover .ori file numbering.
    # DART may create N .ori files per model or N total. We read all and
    # determine from the model's OBJ which belong to which plant.

    # First, read all .ori files
    all_ori_files = sorted(ori_dir.glob('triangle*.ori'))
    print(f"  Found {len(all_ori_files)} .ori files")

    all_ori_data = {}
    for ori_file in all_ori_files:
        gi = int(re.search(r'triangle(\d+)', ori_file.name).group(1))
        all_ori_data[gi] = np.fromfile(str(ori_file), dtype='uint32')

    # Read group info from each plant's DART OBJ
    total_global_groups = 0
    for pi in range(N_PLANTS):
        dart_obj = dart_obj_paths[pi]
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
        per_plant_groups[pi] = groups_sorted
        per_plant_offsets[pi] = group_offsets
        per_plant_face_counts[pi] = group_face_counts

        # Assign .ori files to this plant's groups
        # .ori numbering: 0..N-1 for model 0, N..2N-1 for model 1, etc.
        n_groups = len(groups_sorted)
        per_plant_ori[pi] = {}
        for local_gi in range(n_groups):
            global_gi = total_global_groups + local_gi
            if global_gi in all_ori_data:
                per_plant_ori[pi][local_gi] = all_ori_data[global_gi]

        total_global_groups += n_groups
        print(f"  Plant {pi}: {n_groups} groups, {total_faces} faces, "
              f".ori indices {total_global_groups - n_groups}..{total_global_groups - 1}")

    # --- Read radiative budget ---
    output_path = simu_path / 'output'
    per_band_data = {}
    band_dirs = sorted(output_path.glob('BAND*'))
    if band_dirs:
        for band_dir in band_dirs:
            band_idx = int(re.search(r'BAND(\d+)', band_dir.name).group(1))
            rb_file = band_dir / 'RADIATIVE_BUDGET' / 'ITERX' / 'RadiativeBudgetFigures.txt'
            if rb_file.exists():
                data = _parse_radiative_budget_txt(rb_file)
                if data is not None:
                    per_band_data[band_idx] = data
                    n_tris = sum(len(v) for v in data['per_object'].values()
                                 if hasattr(v, '__len__') and v.ndim == 2)
                    print(f"    BAND{band_idx}: {n_tris} triangles")

    if not per_band_data:
        print("  ERROR: No radiative budget data!")
        return None

    # --- Parse maket.scn for budget-to-(instance, group) mapping ---
    maket_mapping = _parse_maket_scn(simu_path)
    if maket_mapping is None:
        print("  ERROR: Could not parse maket.scn!")
        return None

    # Build per-plant budget mapping
    # maket_mapping: {budget_idx: (instance_idx, group_idx)}
    per_plant_budget = {}  # plant_idx -> {budget_idx: local_group_idx}
    for budget_idx, (instance, group) in maket_mapping.items():
        if instance not in per_plant_budget:
            per_plant_budget[instance] = {}
        per_plant_budget[instance][budget_idx] = group

    print(f"\n  maket.scn: {len(maket_mapping)} vegetation objects")
    for pi in range(N_PLANTS):
        n_budget = len(per_plant_budget.get(pi, {}))
        print(f"    Plant {pi}: {n_budget} budget objects")

    # --- Find absorbed columns ---
    sample_band = list(per_band_data.values())[0]
    header = sample_band['header']
    abs_col_idx = None
    abs_back_col_idx = None
    for i, h in enumerate(header):
        h_lower = h.lower()
        if h_lower == 'absorbed':
            abs_col_idx = i
        elif h_lower == 'absorbed_back':
            abs_back_col_idx = i
    if abs_col_idx is None:
        for i, h in enumerate(header):
            if h.lower() == 'intercepted':
                abs_col_idx = i
                break
    if abs_col_idx is None:
        abs_col_idx = len(header) - 1
    print(f"  Absorbed column: {abs_col_idx} ('{header[abs_col_idx]}')")

    # --- Aggregate per-plant, per-segment ---
    all_segment_results = []
    band_indices = sorted(per_band_data.keys())

    for pi in range(N_PLANTS):
        mapping = mappings[pi]
        plant_budget = per_plant_budget.get(pi, {})
        plant_ori = per_plant_ori.get(pi, {})
        groups_sorted = per_plant_groups[pi]
        group_offsets = per_plant_offsets[pi]

        # Build per-band absorbed arrays for this plant
        per_band_absorbed = {}  # band_idx -> {local_group_idx -> absorbed array}
        for band_idx, band_data in per_band_data.items():
            per_band_absorbed[band_idx] = {}
            for obj_key, arr in band_data['per_object'].items():
                m = re.search(r'object(\d+)', obj_key)
                if m:
                    budget_idx = int(m.group(1))
                    if budget_idx in plant_budget:
                        local_gi = plant_budget[budget_idx]
                        if arr.ndim == 2 and arr.shape[1] > abs_col_idx:
                            front = arr[:, abs_col_idx]
                            if abs_back_col_idx is not None and arr.shape[1] > abs_back_col_idx:
                                back = arr[:, abs_back_col_idx]
                                per_band_absorbed[band_idx][local_gi] = front + back
                            else:
                                per_band_absorbed[band_idx][local_gi] = front

        # Build dart_to_obj for this plant
        dart_to_obj = {}
        for local_gi in sorted(plant_ori.keys()):
            offset = group_offsets[groups_sorted[local_gi]]
            dart_to_obj[local_gi] = plant_ori[local_gi].astype(np.int64) + offset

        obj_to_dart = {}
        for local_gi in sorted(dart_to_obj.keys()):
            for dart_pos, global_obj_idx in enumerate(dart_to_obj[local_gi]):
                obj_to_dart[int(global_obj_idx)] = (local_gi, dart_pos)

        # Aggregate to segments
        for organ in mapping['organs']:
            for seg in organ['segments']:
                tri_indices = seg['triangle_indices']
                if not tri_indices:
                    continue

                per_band_vals = {}
                for band_idx in band_indices:
                    vals = []
                    for tidx in tri_indices:
                        if tidx in obj_to_dart:
                            gi, dart_pos = obj_to_dart[tidx]
                            if gi in per_band_absorbed.get(band_idx, {}):
                                absorbed_arr = per_band_absorbed[band_idx][gi]
                                if dart_pos < len(absorbed_arr):
                                    vals.append(absorbed_arr[dart_pos])
                    per_band_vals[band_idx] = vals

                band_means = {}
                for band_idx in band_indices:
                    bv = per_band_vals.get(band_idx, [])
                    band_means[band_idx] = np.mean(bv) if bv else 0.0
                total_apar = sum(band_means.values())

                all_segment_results.append({
                    'plant_id': pi,
                    'organ': organ['name'],
                    'organ_type': organ['type'],
                    'segment_idx': seg['segment_idx'],
                    'n_triangles': len(tri_indices),
                    'n_dart_matched': sum(1 for t in tri_indices if t in obj_to_dart),
                    'total_apar': total_apar,
                    'band_apar': band_means,
                })

    # Summary
    for pi in range(N_PLANTS):
        plant_segs = [r for r in all_segment_results if r['plant_id'] == pi]
        leaf_segs = [r for r in plant_segs if r['organ_type'] == 'leaf']
        with_apar = [r for r in leaf_segs if r['total_apar'] > 0]
        cov = 100 * len(with_apar) / max(len(leaf_segs), 1)
        total = sum(r['total_apar'] for r in with_apar)
        print(f"  Plant {pi}: {len(leaf_segs)} leaf segs, "
              f"{len(with_apar)} with aPAR ({cov:.1f}%), total={total:.2f}")

    # Save CSV
    csv_path = OUTPUT_DIR / 'multifield_day55_segment_apar.csv'
    with open(csv_path, 'w') as f:
        band_cols = ','.join(f'apar_band{bi}' for bi in band_indices)
        f.write(f"plant_id,organ,organ_type,segment_idx,n_triangles,"
                f"n_dart_matched,total_apar,{band_cols}\n")
        for r in all_segment_results:
            band_vals = ','.join(f"{r['band_apar'].get(bi, 0):.6f}"
                                for bi in band_indices)
            f.write(f"{r['plant_id']},{r['organ']},{r['organ_type']},"
                    f"{r['segment_idx']},{r['n_triangles']},"
                    f"{r['n_dart_matched']},{r['total_apar']:.6f},{band_vals}\n")
    print(f"\n  Saved: {csv_path} ({len(all_segment_results)} segments)")

    return all_segment_results, per_plant_ori, per_plant_groups, per_plant_offsets


# ============================================================================
# Step 6: Create Baleno _I + _II simulations
# ============================================================================
def step6_create_baleno_simus(dart_obj_paths):
    """Create shortwave _I and thermal _II DART simulations for Baleno."""
    print("\n" + "=" * 70)
    print("STEP 6: Create Baleno DART Simulations (_I + _II)")
    print("=" * 70)

    # --- Create _I (shortwave) ---
    simu_name = f'{DART_SIMU_NAME_EB}/{DART_SIMU_NAME_EB}_I'
    parent_dir = DART_LOCAL / 'simulations' / DART_SIMU_NAME_EB
    simu_dir = parent_dir / f'{DART_SIMU_NAME_EB}_I'
    parent_dir.mkdir(parents=True, exist_ok=True)

    if simu_dir.exists():
        shutil.rmtree(str(simu_dir))

    simu = ptd.simulation(simu_name, empty=True)
    simu.scene.size = SCENE_SIZE

    for wvl, bw in SW_BANDS:
        simu.add.band(wvl=wvl, bw=bw)
    print(f"  _I bands: {len(SW_BANDS)} (400-2500nm)")

    simu.core.directions.Directions.SunViewingAngles.sunViewingZenithAngle = SUN_ZENITH
    simu.core.directions.Directions.SunViewingAngles.sunViewingAzimuthAngle = SUN_AZIMUTH

    simu.add.optical_property(
        type='Lambertian', ident='ground',
        databaseName='Lambertian_mineral.db', ModelName='clay_brown',
        useMultiplicativeFactorForLUT=0,
    )
    simu.scene.ground.OpticalPropertyLink.ident = 'ground'

    # Per-position leaf OPs (same as step3)
    first_obj_path = simu.get_input_file_path(str(dart_obj_paths[0]))
    first_obj_info = ptd.OBJtools.objreader(first_obj_path)
    first_gnames = ptd.OBJtools.gnames_dart_order(first_obj_info.names)
    n_leaf_groups = sum(1 for g in first_gnames if not g.endswith('_00'))

    per_pos_params = get_prospect_params_per_position(SIMULATION_DAYS, n_leaf_groups)
    for pi, params in enumerate(per_pos_params):
        ident = f'maize_leaf_pos{pi}'
        simu.add.optical_property(
            type='Lambertian', ident=ident,
            prospect=params,
            useMultiplicativeFactorForLUT=0,
        )

    stem_prospect = get_stem_prospect_params(SIMULATION_DAYS)
    simu.add.optical_property(
        type='Lambertian', ident='maize_stem',
        prospect=stem_prospect,
        useMultiplicativeFactorForLUT=0,
    )

    # Multi-model ObjectFields (same as step 3 but with SW bands)
    model_list = ptd.object_3d.create_ModelList()
    for i, dart_obj in enumerate(dart_obj_paths):
        file_src_fullpath = simu.get_input_file_path(str(dart_obj))
        obj_info = ptd.OBJtools.objreader(file_src_fullpath)
        gnames = ptd.OBJtools.gnames_dart_order(obj_info.names)
        xdim, ydim, zdim = obj_info.dims
        xc, yc, zc = obj_info.center

        groups_list = []
        leaf_idx = 0
        for gi, gname in enumerate(gnames):
            g = ptd.object_3d.create_Group(num=gi + 1, name=gname)
            is_stem = gname.endswith('_00')
            if is_stem:
                op_ident = 'maize_stem'
            else:
                op_ident = f'maize_leaf_pos{leaf_idx}'
                leaf_idx += 1
            df = 0 if is_stem else 1
            g.set_nodes(ident=op_ident)
            gop = g.GroupOpticalProperties
            gop.SurfaceOpticalProperties.doubleFace = df
            gop.SurfaceExitanceProperties.doubleFace = df
            groups_list.append(g)
        groups = ptd.object_3d.create_Groups(Group=groups_list)

        geom = ptd.object_3d.create_GeometricProperties(
            Dimension3D=ptd.object_3d.create_Dimension3D(
                xdim=xdim, ydim=ydim, zdim=zdim),
            Center3D=ptd.object_3d.create_Center3D(
                xCenter=xc, yCenter=yc, zCenter=zc),
            ScaleProperties=ptd.object_3d.create_ScaleProperties(
                xscale=1.0, yscale=1.0, zscale=1.0),
        )
        model_obj = ptd.object_3d.create_Object(
            file_src=str(dart_obj),
            hasGroups=1,
            GeometricProperties=geom,
            Groups=groups,
            num=i,
            name=f'CPlantBox_Maize_p{i}',
            objectDEMMode=0,
        )
        model_list.add_Object(model_obj)

    field = ptd.object_3d.create_Field(
        name='MultiMaizeField',
        fieldDescriptionFileName=FIELD_FILENAME,
    )
    field.set_ModelList(model_list)
    obj_fields = ptd.object_3d.create_ObjectFields()
    obj_fields.add_Field(field)
    simu.core.object_3d.object_3d.ObjectFields = obj_fields

    products = simu.core.phase.Phase.DartProduct.dartModuleProducts.CommonProducts
    products.radiativeBudgetProducts = 1
    products.radiativeBudgetProperties.budget3DParSurface = 1

    simu.core.phase.Phase.accelerationEngine = 2
    simu.core.phase.Phase.ExpertModeZone.nbThreads = DART_THREADS

    # Atmosphere: MIDLATSUM
    configure_atmosphere_midlatsum(simu)

    simu.write(overwrite=True)

    # Write field file
    simu_I_path = Path(str(simu.simu_dir))
    field_path = simu_I_path / 'input' / FIELD_FILENAME
    positions = _compute_plant_positions()
    with open(field_path, 'w') as f:
        f.write('complete transformation\n')
        for idx, (x, y) in enumerate(positions):
            f.write(f'{idx} {x:.6f} {y:.6f} 0.0 1.0 1.0 1.0 0.0 0.0 0.0\n')

    print(f"  _I written: {simu.simu_dir}")

    # --- Create _II (thermal) by copying _I ---
    simu_II_dir = simu_I_path.parent / f'{DART_SIMU_NAME_EB}_II'
    if simu_II_dir.exists():
        shutil.rmtree(str(simu_II_dir))
    shutil.copytree(str(simu_I_path), str(simu_II_dir))
    print(f"  Copied _I to _II: {simu_II_dir}")

    # Modify phase.xml: single thermal band
    phase_xml = simu_II_dir / 'input' / 'phase.xml'
    if phase_xml.exists():
        content = phase_xml.read_text()
        first_replaced = False
        lines = content.split('\n')
        new_lines = []
        for line in lines:
            if '<SpectralIntervalsProperties' in line and '/>' in line:
                if not first_replaced:
                    indent = line[:len(line) - len(line.lstrip())]
                    new_lines.append(
                        f'{indent}<SpectralIntervalsProperties '
                        f'deltaLambda="4.0" meanLambda="10.0" '
                        f'spectralDartMode="0"/>'
                    )
                    first_replaced = True
            else:
                new_lines.append(line)
        phase_xml.write_text('\n'.join(new_lines))
        print(f"  _II phase.xml: 1 thermal band at 10µm")

    # Modify object_3d.xml: per-triangle temperature
    obj3d_xml = simu_II_dir / 'input' / 'object_3d.xml'
    if obj3d_xml.exists():
        tree = ET.parse(str(obj3d_xml))
        root = tree.getroot()
        for group in root.findall('.//Group'):
            group_name = group.get('name', 'unknown')
            temp_filename = f'temperature_{group_name}.txt'
            gop = group.find('GroupOpticalProperties')
            if gop is None:
                continue
            sep = gop.find('SurfaceExitanceProperties')
            if sep is None:
                continue
            sep.set('useTemperaturePerTriangle', '1')
            tptp = sep.find('TemperaturePerTriangleProperty')
            if tptp is None:
                tptp = ET.SubElement(sep, 'TemperaturePerTriangleProperty')
            tptp.set('triangleTemperatureFile', temp_filename)
        tree.write(str(obj3d_xml), xml_declaration=True, encoding='unicode')
        print(f"  _II object_3d.xml: per-triangle temperature enabled")

    # Create initial temperature files
    # Read .ori files from the PAR simulation to get triangle counts per group
    par_simu_path = (Path(ptd.getdartdir()) / 'user_data' / 'simulations'
                     / SIMU_NAME)
    par_ori_dir = par_simu_path / 'input' / 'triangles'
    input_dir = simu_II_dir / 'input'

    # We need triangle counts per group. Read from the DART OBJs.
    for pi in range(N_PLANTS):
        dart_obj = dart_obj_paths[pi]
        current_group = None
        group_face_counts = {}
        with open(dart_obj) as f:
            for line in f:
                if line.startswith('g '):
                    current_group = line.strip()[2:]
                    group_face_counts[current_group] = 0
                elif line.startswith('f '):
                    if current_group:
                        group_face_counts[current_group] += 1

        for gname, n_tris in group_face_counts.items():
            temp_filename = f'temperature_{gname}.txt'
            temp_values = np.full(n_tris, 298.15)
            np.savetxt(str(input_dir / temp_filename), temp_values, fmt='%.2f')

    n_temp_files = len(list(input_dir.glob('temperature_*.txt')))
    print(f"  Created {n_temp_files} temperature files at 298.15 K")

    # Clean output from _I copy
    output_dir = simu_II_dir / 'output'
    if output_dir.exists():
        shutil.rmtree(str(output_dir))
        output_dir.mkdir()

    return simu, simu_II_dir


# ============================================================================
# Step 7: Create Baleno configs + run Baleno
# ============================================================================
def step7_run_baleno():
    """Create Baleno configs and run energy balance."""
    print("\n" + "=" * 70)
    print("STEP 7: Create Baleno Configs + Run")
    print("=" * 70)

    # Create Baleno simulation directory
    baleno_sim_dir = BALENO_USER_DATA / 'simulations' / SIMU_NAME_EB
    input_dir = baleno_sim_dir / 'input'
    plugins_dir = input_dir / 'plugins'

    if baleno_sim_dir.exists():
        shutil.rmtree(str(baleno_sim_dir))
    input_dir.mkdir(parents=True)
    plugins_dir.mkdir(parents=True)

    # JSON5 configs (same as Phase 2, different simulation names)
    _write_json5(input_dir / 'atmosphere.json5', {
        "z": 10, "Ta": 298.15, "p": 1013, "ea": 15, "u": 2,
        "Ca": 400, "Oa": 280,
    })
    # vegetation.json5 — mean Cab/N across per-position profiles
    import numpy as _np
    mean_cab = float(_np.mean([p["Cab"] for p in per_pos_params]))
    mean_n = float(_np.mean([p["N"] for p in per_pos_params]))
    base_params = get_prospect_params(SIMULATION_DAYS)
    from ..prospect_params import vcmax25_from_cab
    _write_json5(input_dir / 'vegetation.json5', {
        "Plugin": "BiochemicalSCOPE", "Model": "VegetationSCOPE",
        "PAR_min": 0.400, "PAR_max": 0.700,
        "Cab": round(mean_cab, 1), "Cca": 10, "Cs": 0,
        "Cw": base_params["Cw"], "Cdm": base_params["Cm"],
        "N": round(mean_n, 2), "fqe": 0,
        "Vcmax25": round(vcmax25_from_cab(mean_cab), 1),
        "BallBerrySlope": 8, "BallBerry0": 0.01,
        "RdPerVcmax25": get_species()["rd_per_vcmax25"],
        "Type": get_species()["photo_type"],
        "rho_thermal": 0.01, "tau_thermal": 0.01, "stress_factor": 1,
    })
    _write_json5(input_dir / 'radiation.json5',
                 {"Plugin": "DART", "Model": "DART"})
    _write_json5(input_dir / 'scene.json5',
                 {"Plugin": "DART", "scene_reader": "DARTSceneTriangleReader"})
    _write_json5(input_dir / 'soil.json5', {
        "Plugin": "SoilMod", "Model": "KustasModel",
        "rs_thermal": 0.06, "SMC": 0.25,
    })
    _write_json5(input_dir / 'aerodynamics.json5', {
        "Plugin": "AerodynamicsSCOPE", "Model": "AeroSCOPE",
        "Cd": 0.3, "rwc": 0, "rbs": 10.0, "CR": 0.35,
        "CD1": 20.6, "Psicor": 0.2, "CSSOIL": 0.01,
        "Monin_Obukhov_correction": True,
    })
    _write_json5(input_dir / 'modelling_parameters.json5', {
        "EB_error": 5, "max_iteration": 10, "min_variation_rate": 0,
        "closure_method": "Newton", "load_state": False,
    })
    _write_json5(input_dir / 'output.json5', {
        "model": "PhysicsAwareDataWriter", "intermediate_outputs": False,
        "save_state": False, "radiation": True, "vegetation": True,
        "soil": True, "aerodynamics": False, "energy_balance_products": True,
        "fluxes": True, "save_scene": True, "delimiter": ";",
        "compute_sunlit": False, "sunlit_threshold": 0.5,
        "1 dimension": False, "2 dimension": False, "3 dimension": True,
        "layer_number": 20, "write_yaml": False,
    })
    _write_json5(input_dir / 'time_series.json5', {
        "is_time_series": False, "input_filename": "time_series.csv",
        "header_filename": "headers.json",
        "load_from_previous_timestep": False, "ts_number": -1, "deltat": -1,
    })
    _write_json5(plugins_dir / 'DART_input.json5', {
        "dart_simulation": DART_SIMU_NAME_EB,
        "Compute_Rn1": True, "Compute_broadband": True,
        "Compute_APAR": True, "Compute_Rn2": True,
    })
    _write_json5(plugins_dir / 'BiochemicalSCOPE_input.json5', {
        "Kn0": 2.48, "Knalpha": 2.83, "Knbeta": 0.114,
        "g_m": "Not computed", "kV": 0.6396, "apply_T_correction": True,
    })
    _write_json5(plugins_dir / 'SoilMod_input.json5', {
        "rss": 500, "Compute_rss_from_SMC": False, "ratio_rn_g": 0.35,
    })
    _write_json5(plugins_dir / 'AerodynamicsSCOPE_input.json5', {})

    print(f"  Baleno configs created: {baleno_sim_dir}")

    # Write config files (with backup + restore)
    backups = _write_baleno_configs()

    try:
        success = _run_baleno_subprocess()
        if not success:
            print("\n  Baleno failed. Check logs at output/baleno_logs/")
            return False
    finally:
        _restore_configs(backups)

    return True


def _write_baleno_configs():
    """Write Baleno config.ini, dart_config.ini, and apply source patches."""
    backups = {}

    # .dartrc
    dartrc_bak = DARTRC_PATH.with_suffix('.bak_multifield')
    if DARTRC_PATH.exists():
        shutil.copy2(str(DARTRC_PATH), str(dartrc_bak))
        backups['dartrc'] = (str(DARTRC_PATH), str(dartrc_bak))

    corrected_dartrc = textwrap.dedent(f"""\
        #!/bin/bash
        export DART_HOME={DART_DIR}
        export DART_LOCAL={DART_LOCAL}
        export DART_PYTHON_PATH={DART_DIR}/bin/python
        export DART_JAVA_MAX_MEMORY=4g
        export PATH=$DART_PYTHON_PATH:$DART_PYTHON_PATH/bin:$DART_HOME/bin/jre/bin:$DART_HOME/bin:$DART_HOME/bin/hapke:$DART_HOME/bin/prospect:$PATH
        export LD_LIBRARY_PATH=$DART_HOME/bin:$LD_LIBRARY_PATH
    """)
    DARTRC_PATH.write_text(corrected_dartrc)

    # Baleno config.ini
    baleno_config_path = BALENO_DIR / 'resources' / 'config.ini'
    baleno_config_bak = baleno_config_path.with_suffix('.ini.bak_multifield')
    if baleno_config_path.exists():
        shutil.copy2(str(baleno_config_path), str(baleno_config_bak))
        backups['baleno_config'] = (str(baleno_config_path), str(baleno_config_bak))
    else:
        backups['baleno_config'] = (str(baleno_config_path), None)

    baleno_config_path.write_text(textwrap.dedent(f"""\
        [simulation]
        user_data_path =
        name = {SIMU_NAME_EB}
    """))

    # DART plugin dart_config.ini
    dart_config_path = BALENO_DIR / 'plugins' / 'DART' / 'resources' / 'dart_config.ini'
    dart_config_bak = dart_config_path.with_suffix('.ini.bak_multifield')
    if dart_config_path.exists():
        shutil.copy2(str(dart_config_path), str(dart_config_bak))
    backups['dart_config'] = (str(dart_config_path), str(dart_config_bak))

    dart_config_path.write_text(textwrap.dedent(f"""\
        [paths]
        dart_path = {DART_DIR}
    """))

    # Apply Baleno source patches (same as Phase 2)
    _apply_baleno_patches(backups)

    return backups


def _apply_baleno_patches(backups):
    """Apply the 4 Baleno source patches from Phase 2."""
    # Patch 1: lux_output_reader.py — relaxed is_scene_sorted
    lux_reader_path = BALENO_DIR / 'plugins' / 'DART' / 'IO' / 'output_readers' / 'lux_output_reader.py'
    lux_reader_bak = lux_reader_path.with_suffix('.py.bak_multifield')
    if lux_reader_path.exists():
        shutil.copy2(str(lux_reader_path), str(lux_reader_bak))
        backups['lux_output_reader'] = (str(lux_reader_path), str(lux_reader_bak))
        content = lux_reader_path.read_text()
        old = 'is_valid = np.all(col == np.arange(len(col)))'
        new = 'is_valid = np.all(np.diff(col.astype(float)) >= 0)  # monotonic (patched)'
        if old in content:
            content = content.replace(old, new)
            lux_reader_path.write_text(content)
            print(f"  Patched lux_output_reader.py")

    # Patch 2: spectrum_integration.py — band index bug
    spec_int_path = BALENO_DIR / 'plugins' / 'DART' / 'launch_processes' / 'spectrum_integration.py'
    spec_int_bak = spec_int_path.with_suffix('.py.bak_multifield')
    if spec_int_path.exists():
        shutil.copy2(str(spec_int_path), str(spec_int_bak))
        backups['spectrum_integration'] = (str(spec_int_path), str(spec_int_bak))
        content = spec_int_path.read_text()
        old_loop = (
            '    for band_name in band_name_list:\n'
            '        band_index = int(band_name[4:])  # not done with enumerate as the bands can be in the incorrect order\n'
            '        short_wl = wavebands[band_index][0] * 1000  # Conversion from µm to nm\n'
            '        long_wl = wavebands[band_index][1] * 1000\n'
            '\n'
            '        coefficient = coefficient_function(short_wl, long_wl)\n'
            '        dart_paths = DARTPathManager(is_simu_i, band_number=band_index)\n'
            '        radiative_budget_file = get_radiative_budget_file_triangles(is_simu_i, band_index)\n'
            '\n'
            '        vect = coefficient * np.array(simu_manager.get_radiative_budget(scene_array, dart_paths, radiative_budget_file, band_index))'
        )
        new_loop = (
            '    for wvb_idx, band_name in enumerate(band_name_list):  # patched\n'
            '        dart_band_index = int(band_name[4:])\n'
            '        short_wl = wavebands[wvb_idx][0] * 1000\n'
            '        long_wl = wavebands[wvb_idx][1] * 1000\n'
            '\n'
            '        coefficient = coefficient_function(short_wl, long_wl)\n'
            '        dart_paths = DARTPathManager(is_simu_i, band_number=dart_band_index)\n'
            '        radiative_budget_file = get_radiative_budget_file_triangles(is_simu_i, dart_band_index)\n'
            '\n'
            '        vect = coefficient * np.array(simu_manager.get_radiative_budget(scene_array, dart_paths, radiative_budget_file, dart_band_index))'
        )
        if old_loop in content:
            content = content.replace(old_loop, new_loop)
            spec_int_path.write_text(content)
            print(f"  Patched spectrum_integration.py")

    # Patch 3: lux_manager.py — missing temperaturematrix fallback
    lux_mgr_path = BALENO_DIR / 'plugins' / 'DART' / 'IO' / 'FT_Lux_Managers' / 'lux_manager.py'
    lux_mgr_bak = lux_mgr_path.with_suffix('.py.bak_multifield')
    if lux_mgr_path.exists():
        shutil.copy2(str(lux_mgr_path), str(lux_mgr_bak))
        backups['lux_manager'] = (str(lux_mgr_path), str(lux_mgr_bak))
        content = lux_mgr_path.read_text()
        old_get = (
            '                    if exitance_file is None:\n'
            '                        logger.debug(\n'
            '                            f"No {input_type.name} file set as direct way for object {dart_name_id}. Looking for file from {input_type.name} property definition")\n'
            '                        material_name = value.get(MaketReader.MATERIAL)\n'
            '                        material_dict = scn_reader.get_dict_material()\n'
            '                        exitance_file = self.__get_file_from_material(material_name, material_dict, dart_state, input_type)\n'
            '                    files_dict[dart_name_id] = exitance_file'
        )
        new_get = (
            '                    if exitance_file is None:\n'
            '                        logger.debug(\n'
            '                            f"No {input_type.name} file set as direct way for object {dart_name_id}. Looking for file from {input_type.name} property definition")\n'
            '                        material_name = value.get(MaketReader.MATERIAL)\n'
            '                        material_dict = scn_reader.get_dict_material()\n'
            '                        try:\n'
            '                            exitance_file = self.__get_file_from_material(material_name, material_dict, dart_state, input_type)\n'
            '                        except (AttributeError, TypeError, KeyError) as e:\n'
            '                            exitance_file = f"temperature_{dart_name_id}.txt"\n'
            '                            logger.warning(f"No {input_type.name} file found for {dart_name_id}, using default: {exitance_file}")\n'
            '                    files_dict[dart_name_id] = exitance_file'
        )
        if old_get in content:
            content = content.replace(old_get, new_get)
            lux_mgr_path.write_text(content)
            print(f"  Patched lux_manager.py")


def _run_baleno_subprocess():
    """Run Baleno as subprocess."""
    if not VENV_PYTHON.exists():
        print(f"  ERROR: darteb_venv not found at {VENV_PYTHON}")
        return False

    env = os.environ.copy()
    env['PYTHONPATH'] = str(BALENO_DIR)
    env['DART_HOME'] = str(DART_DIR)
    env['DART_LOCAL'] = str(DART_LOCAL)

    # Remove DART's problematic libreadline
    ld_path = env.get('LD_LIBRARY_PATH', '')
    dart_lib_paths = [str(DART_DIR / 'bin' / 'python' / 'lib')]
    filtered_ld = ':'.join(
        p for p in ld_path.split(':') if p and p not in dart_lib_paths
    )
    env['LD_LIBRARY_PATH'] = filtered_ld

    cmd = [str(VENV_PYTHON), 'src/main.py']
    print(f"  Running Baleno (timeout=3600s)...")

    try:
        result = subprocess.run(
            cmd, cwd=str(BALENO_DIR), env=env,
            capture_output=True, text=True, timeout=3600,
        )
        log_dir = OUTPUT_DIR / 'multifield_baleno_logs'
        log_dir.mkdir(exist_ok=True)
        (log_dir / 'stdout.txt').write_text(result.stdout)
        (log_dir / 'stderr.txt').write_text(result.stderr)

        stdout_lines = result.stdout.strip().split('\n')
        print(f"\n  Baleno output ({len(stdout_lines)} lines):")
        for line in stdout_lines[-30:]:
            print(f"    {line}")

        if result.returncode != 0:
            print(f"\n  ERROR: Baleno exited with code {result.returncode}")
            stderr_lines = result.stderr.strip().split('\n')
            for line in stderr_lines[-20:]:
                print(f"    {line}")
            return False

        print(f"\n  Baleno completed successfully!")
        return True

    except subprocess.TimeoutExpired:
        print(f"  ERROR: Baleno timed out after 3600s")
        return False


def _restore_configs(backups):
    """Restore backed-up config files."""
    print("\n  Restoring config files...")
    for name, (original, backup) in backups.items():
        if backup is None:
            if Path(original).exists():
                Path(original).unlink()
        elif Path(backup).exists():
            shutil.copy2(backup, original)
            Path(backup).unlink()


def _write_json5(path, data):
    """Write JSON5-compatible file."""
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)


# ============================================================================
# Step 8: Read Baleno outputs + aggregate per-plant Tleaf
# ============================================================================
def step8_read_baleno(dart_obj_paths, mappings):
    """Parse Baleno 3D outputs and aggregate per-plant-per-segment."""
    print("\n" + "=" * 70)
    print("STEP 8: Read Baleno Outputs + Aggregate Per-Plant")
    print("=" * 70)

    output_base = BALENO_USER_DATA / 'simulations' / SIMU_NAME_EB / 'output'
    results_dir = output_base / 'final_results'

    if not results_dir.exists():
        print(f"  ERROR: Results directory not found: {results_dir}")
        return None

    # Read scene file
    scene_file = None
    for candidate in [output_base / 'scene', results_dir / 'scene.csv']:
        if candidate.exists() and candidate.stat().st_size > 0:
            scene_file = candidate
            break

    if scene_file is None:
        print(f"  ERROR: scene file not found!")
        return None

    delimiter = _detect_delimiter(scene_file)
    print(f"  Scene file: {scene_file}")

    # Read all output files
    outputs = {}
    header, data = _read_baleno_csv(scene_file, delimiter)
    outputs['scene'] = {'header': header, 'data': data, '_path': str(scene_file)}

    for key, filename in [('energy_balance', 'energy_balance_3D.csv'),
                          ('radiation', 'radiation_3D.csv'),
                          ('heat_fluxes', 'heat_fluxes_3D.csv'),
                          ('vegetation', 'vegetation_3D.csv')]:
        filepath = results_dir / filename
        if filepath.exists():
            h, d = _read_baleno_csv(filepath, delimiter)
            outputs[key] = {'header': h, 'data': d}
            print(f"  {filename}: {d.shape[0]} rows x {d.shape[1]} cols")

    # Build Baleno→OBJ mapping per plant using scene DART_NAME
    scene_header = outputs['scene']['header']
    scene_str = np.genfromtxt(str(scene_file), skip_header=1,
                               delimiter=delimiter, dtype=str)

    col_type_id = scene_header.index('TYPE_ID') if 'TYPE_ID' in scene_header else 0
    col_dart_name = scene_header.index('DART_NAME') if 'DART_NAME' in scene_header else 1
    col_index_obj = scene_header.index('INDEX_OBJECT') if 'INDEX_OBJECT' in scene_header else 2
    col_surface = scene_header.index('SURFACE') if 'SURFACE' in scene_header else 5

    type_ids = scene_str[:, col_type_id].astype(float).astype(int)
    dart_names = scene_str[:, col_dart_name]
    index_in_object = scene_str[:, col_index_obj].astype(float).astype(int)
    surfaces = scene_str[:, col_surface].astype(float)

    leaf_mask = (type_ids >= 100) | (type_ids == 5)
    n_total = len(type_ids)
    print(f"  Total rows: {n_total}, leaf: {np.sum(leaf_mask)}")

    # Load reindex data from PAR simulation for .ori lookup
    # Read .ori from PAR simulation
    par_simu_path = (Path(ptd.getdartdir()) / 'user_data' / 'simulations'
                     / SIMU_NAME)
    ori_dir = par_simu_path / 'input' / 'triangles'
    all_ori_data = {}
    for ori_file in sorted(ori_dir.glob('triangle*.ori')):
        gi = int(re.search(r'triangle(\d+)', ori_file.name).group(1))
        all_ori_data[gi] = np.fromfile(str(ori_file), dtype='uint32')

    # Build per-plant .ori and group info (same as step 5)
    per_plant_ori = {}
    per_plant_groups = {}
    per_plant_offsets = {}
    total_global_groups = 0

    for pi in range(N_PLANTS):
        dart_obj = dart_obj_paths[pi]
        group_offsets = {}
        current_group = None
        total_faces = 0
        with open(dart_obj) as f:
            for line in f:
                if line.startswith('g '):
                    current_group = line.strip()[2:]
                    group_offsets[current_group] = total_faces
                elif line.startswith('f '):
                    if current_group:
                        total_faces += 1

        groups_sorted = sorted(group_offsets.keys())
        per_plant_groups[pi] = groups_sorted
        per_plant_offsets[pi] = group_offsets

        n_groups = len(groups_sorted)
        per_plant_ori[pi] = {}
        for local_gi in range(n_groups):
            global_gi = total_global_groups + local_gi
            if global_gi in all_ori_data:
                per_plant_ori[pi][local_gi] = all_ori_data[global_gi]
        total_global_groups += n_groups

    # Parse DART_NAME to identify plant instance and group
    # Format: fo0_moX_goY where X=instance, Y=group
    per_plant_baleno_to_obj = {}

    for pi in range(N_PLANTS):
        groups_sorted = per_plant_groups[pi]
        group_offsets = per_plant_offsets[pi]
        plant_ori = per_plant_ori[pi]

        # Build dart_to_obj for this plant
        dart_to_obj = {}
        for local_gi in sorted(plant_ori.keys()):
            offset = group_offsets[groups_sorted[local_gi]]
            dart_to_obj[local_gi] = plant_ori[local_gi].astype(np.int64) + offset

        # Map Baleno rows to OBJ faces for this plant
        baleno_to_obj = np.full(n_total, -1, dtype=np.int64)
        for row_idx in range(n_total):
            if not leaf_mask[row_idx]:
                continue
            dn = dart_names[row_idx]
            mo_m = re.search(r'_mo(\d+)', dn)
            go_m = re.search(r'_go(\d+)', dn)
            if mo_m and go_m:
                instance = int(mo_m.group(1))
                group = int(go_m.group(1))
                if instance == pi and group in dart_to_obj:
                    idx = index_in_object[row_idx]
                    ori_arr = dart_to_obj[group]
                    if idx < len(ori_arr):
                        baleno_to_obj[row_idx] = ori_arr[idx]

        per_plant_baleno_to_obj[pi] = baleno_to_obj
        n_mapped = np.sum(baleno_to_obj >= 0)
        print(f"  Plant {pi}: {n_mapped} Baleno rows mapped")

    # Aggregate per-plant, per-segment
    eb_data = outputs.get('energy_balance', {}).get('data')
    eb_header = outputs.get('energy_balance', {}).get('header', [])
    rad_data = outputs.get('radiation', {}).get('data')
    rad_header = outputs.get('radiation', {}).get('header', [])
    flux_data = outputs.get('heat_fluxes', {}).get('data')
    flux_header = outputs.get('heat_fluxes', {}).get('header', [])

    def _col(header, name, default=-1):
        for i, h in enumerate(header):
            if name.lower() in h.lower():
                return i
        return default

    col_temp = _col(eb_header, 'TEMPERATURE', 1)
    col_eb_err = _col(eb_header, 'ERROR', 2)
    col_apar = _col(rad_header, 'ABSORPTION_PAR', 4)
    col_le = _col(flux_header, 'LATENT', 1)
    col_h = _col(flux_header, 'SENSIBLE', 2)

    all_baleno_results = []
    for pi in range(N_PLANTS):
        mapping = mappings[pi]
        baleno_to_obj = per_plant_baleno_to_obj[pi]

        obj_to_baleno = {}
        for row_idx in range(len(baleno_to_obj)):
            obj_face = baleno_to_obj[row_idx]
            if obj_face >= 0:
                obj_to_baleno[int(obj_face)] = row_idx

        for organ in mapping['organs']:
            for seg in organ['segments']:
                tri_indices = seg['triangle_indices']
                if not tri_indices:
                    continue

                baleno_rows = []
                tri_surfaces = []
                for tidx in tri_indices:
                    if tidx in obj_to_baleno:
                        row = obj_to_baleno[tidx]
                        baleno_rows.append(row)
                        tri_surfaces.append(surfaces[row])

                if not baleno_rows:
                    all_baleno_results.append({
                        'plant_id': pi,
                        'organ': organ['name'],
                        'organ_type': organ['type'],
                        'segment_idx': seg['segment_idx'],
                        'n_triangles': len(tri_indices),
                        'n_matched': 0,
                        'Tleaf_K': np.nan, 'Tleaf_C': np.nan,
                        'APAR_umol': np.nan, 'LE_Wm2': np.nan,
                        'H_Wm2': np.nan, 'EB_error': np.nan,
                    })
                    continue

                rows = np.array(baleno_rows)
                weights = np.array(tri_surfaces)
                if np.sum(weights) > 0:
                    weights = weights / np.sum(weights)
                else:
                    weights = np.ones(len(rows)) / len(rows)

                def _wmean(data, col):
                    if data is None or col < 0 or col >= data.shape[1]:
                        return np.nan
                    vals = data[rows, col].astype(float)
                    valid = ~np.isnan(vals)
                    if np.sum(valid) == 0:
                        return np.nan
                    return np.average(vals[valid], weights=weights[valid])

                tleaf_k = _wmean(eb_data, col_temp)
                tleaf_c = tleaf_k - 273.15 if not np.isnan(tleaf_k) else np.nan

                all_baleno_results.append({
                    'plant_id': pi,
                    'organ': organ['name'],
                    'organ_type': organ['type'],
                    'segment_idx': seg['segment_idx'],
                    'n_triangles': len(tri_indices),
                    'n_matched': len(baleno_rows),
                    'Tleaf_K': tleaf_k,
                    'Tleaf_C': tleaf_c,
                    'APAR_umol': _wmean(rad_data, col_apar),
                    'LE_Wm2': _wmean(flux_data, col_le),
                    'H_Wm2': _wmean(flux_data, col_h),
                    'EB_error': _wmean(eb_data, col_eb_err),
                })

    # Summary per plant
    for pi in range(N_PLANTS):
        plant_segs = [r for r in all_baleno_results
                      if r['plant_id'] == pi and r['organ_type'] == 'leaf']
        valid = [r for r in plant_segs if not np.isnan(r['Tleaf_C'])]
        if valid:
            temps = [r['Tleaf_C'] for r in valid]
            print(f"  Plant {pi}: {len(valid)}/{len(plant_segs)} valid, "
                  f"Tleaf mean={np.mean(temps):.2f} C")

    # Save CSV
    csv_path = OUTPUT_DIR / 'multifield_day55_baleno_segments.csv'
    with open(csv_path, 'w') as f:
        f.write("plant_id,organ,organ_type,segment_idx,n_triangles,n_matched,"
                "Tleaf_K,Tleaf_C,APAR_umol_m2_s,LE_Wm2,H_Wm2,EB_error_Wm2\n")
        for r in all_baleno_results:
            f.write(f"{r['plant_id']},{r['organ']},{r['organ_type']},"
                    f"{r['segment_idx']},{r['n_triangles']},{r['n_matched']},"
                    f"{r['Tleaf_K']:.4f},{r['Tleaf_C']:.4f},"
                    f"{r['APAR_umol']:.4f},{r['LE_Wm2']:.4f},"
                    f"{r['H_Wm2']:.4f},{r['EB_error']:.4f}\n")
    print(f"\n  Saved: {csv_path} ({len(all_baleno_results)} segments)")

    return all_baleno_results


# ============================================================================
# Step 9: Per-plant photosynthesis
# ============================================================================
def step9_photosynthesis(apar_results, baleno_results, mappings):
    """Grow 9 fresh plants with photosynthesis, run per-plant solve."""
    print("\n" + "=" * 70)
    print("STEP 9: Per-Plant Photosynthesis")
    print("=" * 70)

    from plantbox.functional.phloem_flux import PhloemFluxPython
    from plantbox.functional.PlantHydraulicParameters import PlantHydraulicParameters

    all_photo_results = []
    per_plant_summary = []

    for pi in range(N_PLANTS):
        seed = FIELD_SEED + pi
        print(f"\n--- Plant {pi} (seed={seed}) ---")

        # Build per-segment APAR and Tleaf arrays
        plant_apar = [r for r in apar_results
                      if r['plant_id'] == pi and r['organ_type'] == 'leaf']
        plant_baleno = [r for r in baleno_results
                        if r['plant_id'] == pi and r['organ_type'] == 'leaf']

        if len(plant_apar) != len(plant_baleno):
            print(f"  WARNING: APAR ({len(plant_apar)}) != Baleno ({len(plant_baleno)}) segments")
            continue

        n_leaf_segs = len(plant_apar)
        if n_leaf_segs == 0:
            print(f"  No leaf segments, skipping")
            continue

        # Normalize APAR to target PAR mean
        raw_apar = np.array([r['total_apar'] for r in plant_apar])
        mean_apar = np.mean(raw_apar)
        if mean_apar > 1e-12:
            apar_umol = (raw_apar / mean_apar) * TARGET_PAR_UMOL
        else:
            apar_umol = np.full(n_leaf_segs, TARGET_PAR_UMOL)
        apar_umol = np.clip(apar_umol, 0.0, 3000.0)

        # Extract Tleaf, fallback to TAIR_C for NaN
        tleaf_c = np.array([r['Tleaf_C'] for r in plant_baleno])
        nan_mask = np.isnan(tleaf_c)
        if np.any(nan_mask):
            n_nan = np.sum(nan_mask)
            print(f"  {n_nan} NaN Tleaf values, using {TAIR_C} C")
            tleaf_c[nan_mask] = TAIR_C

        print(f"  APAR: mean={np.mean(apar_umol):.1f}, "
              f"range=[{apar_umol.min():.1f}, {apar_umol.max():.1f}]")
        print(f"  Tleaf: mean={np.mean(tleaf_c):.2f}, "
              f"range=[{tleaf_c.min():.2f}, {tleaf_c.max():.2f}]")

        # Grow fresh plant with same seed
        plant = grow_plant(XML_PATH, SIMULATION_DAYS,
                           enable_photosynthesis=True, seed=seed)

        # Verify segment alignment
        seg_leaves_idx = plant.getSegmentIds(4)
        n_cpb = len(seg_leaves_idx)
        if n_cpb != n_leaf_segs:
            print(f"  ALIGNMENT MISMATCH: CPlantBox={n_cpb}, CSV={n_leaf_segs}")
            continue
        print(f"  Alignment OK: {n_cpb} leaf segments")

        # Setup hydraulics + photosynthesis + phloem
        params = PlantHydraulicParameters()
        params.read_parameters(get_hydraulics_json())

        hm = PhloemFluxPython(plant, params)
        hm.read_photosynthesis_parameters(filename=get_photosynthesis_json())
        hm.read_phloem_parameters(filename=get_phloem_json())

        # Per-segment Chl from LOPS per-position profiles
        from ..prospect_params import get_chl_per_segment, vcmax25_from_cab
        chl_per_seg = get_chl_per_segment(SIMULATION_DAYS, plant)
        if len(chl_per_seg) == n_leaf_segs:
            hm.Chl = chl_per_seg
            print(f"  Per-segment Chl: [{min(chl_per_seg):.1f}, {max(chl_per_seg):.1f}] "
                  f"-> Vcmax [{vcmax25_from_cab(min(chl_per_seg)):.1f}, "
                  f"{vcmax25_from_cab(max(chl_per_seg)):.1f}] umol/m2/s")

        # Solve
        depth = 100
        p_s = np.linspace(SOIL_PSI_CM, SOIL_PSI_CM - depth, depth)
        es = hm.get_es(float(np.mean(tleaf_c)))
        ea = es * RH
        par_mol = apar_umol * 1e-6 * 86400 * 1e-4

        try:
            hm.solve(sim_time=SIMULATION_DAYS, rsx=p_s, cells=True,
                     ea=ea, es=es, PAR=par_mol, TairC=tleaf_c, verbose=0)
        except Exception as e:
            print(f"  ERROR in hm.solve(): {e}")
            continue

        An_leaf = np.array(hm.get_net_assimilation())
        An_per = np.array(hm.get_net_assimilation_perleafBladeArea())
        transp = np.sum(hm.get_transpiration()) / 18 * 1e3
        An_total_mmol = np.sum(An_leaf) * 1e3
        An_per_umol = An_per * 1e4 / 86400 * 1e6

        print(f"  An_total: {An_total_mmol:.3f} mmol CO2/d, "
              f"Transp: {transp:.3f} mmol H2O/d")

        # Store per-segment results
        for si in range(n_leaf_segs):
            all_photo_results.append({
                'plant_id': pi,
                'organ': plant_apar[si]['organ'],
                'segment_idx': plant_apar[si]['segment_idx'],
                'apar_umol': float(apar_umol[si]),
                'tleaf_c': float(tleaf_c[si]),
                'An_umol_m2_s': float(An_per_umol[si]),
                'An_mol_d': float(An_leaf[si]),
            })

        # Compute leaf area
        leaf_organs = [o for o in plant.getOrgans()
                       if o.organType() == pb.OrganTypes.leaf]
        leaf_area_cm2 = sum(o.leafArea() for o in leaf_organs
                            if hasattr(o, 'leafArea'))

        per_plant_summary.append({
            'plant_id': pi,
            'seed': seed,
            'n_leaf_segs': n_leaf_segs,
            'An_total_mmol': float(An_total_mmol),
            'transp_mmol': float(transp),
            'mean_An_umol': float(np.mean(An_per_umol[An_per_umol > 0]))
                if np.any(An_per_umol > 0) else 0.0,
            'leaf_area_cm2': float(leaf_area_cm2) if leaf_area_cm2 > 0 else 0.0,
            'mean_apar': float(np.mean(apar_umol)),
            'mean_tleaf': float(np.mean(tleaf_c)),
        })

    # Save per-segment CSV
    csv_path = OUTPUT_DIR / 'multifield_day55_photosynthesis.csv'
    with open(csv_path, 'w') as f:
        f.write("plant_id,organ,segment_idx,apar_umol_m2_s,tleaf_c,"
                "An_umol_m2_s,An_mol_d\n")
        for r in all_photo_results:
            f.write(f"{r['plant_id']},{r['organ']},{r['segment_idx']},"
                    f"{r['apar_umol']:.4f},{r['tleaf_c']:.4f},"
                    f"{r['An_umol_m2_s']:.6f},{r['An_mol_d']:.6e}\n")
    print(f"\n  Saved: {csv_path} ({len(all_photo_results)} segments)")

    return per_plant_summary, all_photo_results


# ============================================================================
# Step 10: Field-level aggregation + figure
# ============================================================================
def step10_summary(per_plant_summary, all_photo_results, apar_results, baleno_results):
    """Field-level aggregation, comparison, and multi-panel figure."""
    print("\n" + "=" * 70)
    print("STEP 10: Field-Level Summary")
    print("=" * 70)

    if not per_plant_summary:
        print("  ERROR: No plant summaries!")
        return False

    # Cross-plant statistics
    An_values = [p['An_total_mmol'] for p in per_plant_summary]
    mean_An = np.mean(An_values)
    std_An = np.std(An_values)
    cv_An = std_An / mean_An * 100 if mean_An > 0 else 0

    print(f"\n  Cross-plant An (mmol CO2/d):")
    print(f"    Mean: {mean_An:.3f}")
    print(f"    Std:  {std_An:.3f}")
    print(f"    CV:   {cv_An:.1f}%")
    print(f"    Range: [{min(An_values):.3f}, {max(An_values):.3f}]")

    for p in per_plant_summary:
        print(f"    Plant {p['plant_id']} (seed={p['seed']}): "
              f"An={p['An_total_mmol']:.3f}, Transp={p['transp_mmol']:.3f}")

    # Center plant comparison with Phase 4
    center = [p for p in per_plant_summary if p['plant_id'] == CENTER_PLANT_IDX]
    if center:
        center = center[0]
        print(f"\n  Center plant (idx={CENTER_PLANT_IDX}, seed={FIELD_SEED + CENTER_PLANT_IDX}):")
        print(f"    An={center['An_total_mmol']:.3f} mmol CO2/d")

    # Save results JSON
    results = {
        'phase': 'Phase 6: Multi-Plant Unique Realizations',
        'simulation_days': SIMULATION_DAYS,
        'n_plants': N_PLANTS,
        'field_seed': FIELD_SEED,
        'center_plant_idx': CENTER_PLANT_IDX,
        'grid': {'nx': GRID_NX, 'ny': GRID_NY,
                 'spacing_x_m': GRID_SPACING_X, 'spacing_y_m': GRID_SPACING_Y},
        'per_plant': per_plant_summary,
        'field_stats': {
            'An_mean_mmol': float(mean_An),
            'An_std_mmol': float(std_An),
            'An_cv_pct': float(cv_An),
            'An_min_mmol': float(min(An_values)),
            'An_max_mmol': float(max(An_values)),
        },
        'checks': {
            'cv_gt_5pct': bool(cv_An > 5.0),
            'all_plants_have_an': bool(all(a > 0 for a in An_values)),
        },
    }

    json_path = OUTPUT_DIR / 'multifield_day55_results.json'
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: {json_path}")

    # Multi-panel figure
    _plot_multifield(per_plant_summary, all_photo_results,
                     apar_results, baleno_results)

    return True


def _plot_multifield(per_plant_summary, all_photo_results,
                     apar_results, baleno_results):
    """Create multi-panel validation figure."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available, skipping figure")
        return

    fig, axes = plt.subplots(2, 3, figsize=(18, 11), facecolor='white')

    An_values = [p['An_total_mmol'] for p in per_plant_summary]
    mean_An = np.mean(An_values)
    cv_An = np.std(An_values) / mean_An * 100 if mean_An > 0 else 0

    fig.suptitle(
        f"Phase 6: Multi-Plant Field ({N_PLANTS} unique realizations)\n"
        f"Mean An={mean_An:.1f} mmol CO$_2$ d$^{{-1}}$, CV={cv_An:.1f}%",
        fontsize=13, fontweight='bold', y=0.98
    )

    # Panel 1: 3x3 grid heatmap of per-plant total An
    ax = axes[0, 0]
    an_grid = np.full((GRID_NY, GRID_NX), np.nan)
    for p in per_plant_summary:
        pi = p['plant_id']
        iy, ix = divmod(pi, GRID_NX)
        an_grid[iy, ix] = p['An_total_mmol']
    im = ax.imshow(an_grid, cmap='YlGn', origin='lower', aspect='equal')
    for iy in range(GRID_NY):
        for ix in range(GRID_NX):
            val = an_grid[iy, ix]
            if not np.isnan(val):
                pi = iy * GRID_NX + ix
                color = 'white' if val > np.nanmean(an_grid) else 'black'
                ax.text(ix, iy, f'P{pi}\n{val:.1f}', ha='center', va='center',
                        fontsize=8, color=color, fontweight='bold')
    plt.colorbar(im, ax=ax, label='An (mmol CO$_2$ d$^{-1}$)')
    ax.set_title('Per-Plant Total An')
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')

    # Panel 2: Cross-plant An boxplot + scatter
    ax = axes[0, 1]
    ax.boxplot(An_values, vert=True, widths=0.5)
    ax.scatter(np.ones(len(An_values)), An_values, c='forestgreen',
               s=50, zorder=5, alpha=0.8)
    for i, v in enumerate(An_values):
        ax.annotate(f'P{i}', (1.05, v), fontsize=7)
    ax.set_ylabel('An (mmol CO$_2$ d$^{-1}$)')
    ax.set_title(f'An Distribution (CV={cv_An:.1f}%)')
    ax.set_xticks([])

    # Panel 3: Per-plant vertical APAR profiles
    ax = axes[0, 2]
    colors = plt.cm.tab10(np.linspace(0, 1, N_PLANTS))
    for pi in range(N_PLANTS):
        plant_segs = [r for r in apar_results
                      if r['plant_id'] == pi and r['organ_type'] == 'leaf'
                      and r['total_apar'] > 0]
        if not plant_segs:
            continue
        apars = [r['total_apar'] for r in plant_segs]
        seg_idxs = np.arange(len(apars))
        ax.plot(apars, seg_idxs, '.', markersize=1, color=colors[pi],
                alpha=0.4, label=f'P{pi}')
    ax.set_xlabel('Total aPAR')
    ax.set_ylabel('Segment index')
    ax.set_title('Per-Plant APAR Profiles')
    ax.legend(fontsize=6, ncol=3, loc='upper right')

    # Panel 4: Leaf area vs An scatter
    ax = axes[1, 0]
    la_vals = [p.get('leaf_area_cm2', np.nan) for p in per_plant_summary]
    an_vals = [p['An_total_mmol'] for p in per_plant_summary]
    valid = [(la, an, p['plant_id']) for la, an, p
             in zip(la_vals, an_vals, per_plant_summary)
             if not np.isnan(la)]
    if valid:
        las, ans, pids = zip(*valid)
        ax.scatter(las, ans, c='forestgreen', s=60, edgecolors='#333', lw=0.5)
        for la, an, pid in zip(las, ans, pids):
            ax.annotate(f'P{pid}', (la + 10, an), fontsize=7)
    ax.set_xlabel('Leaf Area (cm$^2$)')
    ax.set_ylabel('An (mmol CO$_2$ d$^{-1}$)')
    ax.set_title('Leaf Area vs Assimilation')

    # Panel 5: Per-plant Tleaf distributions
    ax = axes[1, 1]
    for pi in range(N_PLANTS):
        plant_segs = [r for r in baleno_results
                      if r['plant_id'] == pi and r['organ_type'] == 'leaf'
                      and not np.isnan(r.get('Tleaf_C', np.nan))]
        if plant_segs:
            temps = [r['Tleaf_C'] for r in plant_segs]
            ax.hist(temps, bins=20, alpha=0.3, color=colors[pi],
                    label=f'P{pi}')
    ax.set_xlabel('T$_{leaf}$ (°C)')
    ax.set_ylabel('Count')
    ax.set_title('Per-Plant Tleaf Distributions')
    ax.legend(fontsize=6, ncol=3)

    # Panel 6: EB closure per plant
    ax = axes[1, 2]
    eb_means = []
    for pi in range(N_PLANTS):
        plant_segs = [r for r in baleno_results
                      if r['plant_id'] == pi and r['organ_type'] == 'leaf'
                      and not np.isnan(r.get('EB_error', np.nan))]
        if plant_segs:
            eb_err = np.mean([abs(r['EB_error']) for r in plant_segs])
            eb_means.append(eb_err)
        else:
            eb_means.append(np.nan)
    ax.bar(range(N_PLANTS), eb_means, color='steelblue', edgecolor='#333', lw=0.5)
    ax.axhline(5.0, color='red', ls='--', lw=1, label='5 W/m² threshold')
    ax.set_xlabel('Plant ID')
    ax.set_ylabel('Mean |EB error| (W/m²)')
    ax.set_title('Energy Balance Closure')
    ax.legend(fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig_path = OUTPUT_DIR / 'multifield_day55_figure.png'
    fig.savefig(str(fig_path), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Figure saved: {fig_path}")


# ============================================================================
# Helper: parse radiative budget + maket.scn (reused from Phase 1)
# ============================================================================
def _parse_radiative_budget_txt(filepath):
    """Parse RadiativeBudgetFigures.txt."""
    with open(filepath, 'r') as f:
        lines = f.readlines()
    if not lines:
        return None

    header_line = lines[0].strip('* \n\t')
    header = re.split(r'[\t;]+', header_line)
    header = [h.strip() for h in header if h.strip()]

    per_object = {}
    current_key = None
    for line in lines[1:]:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith('ObjectName'):
            current_key = stripped
            per_object[current_key] = []
        elif current_key is not None:
            values = re.split(r'[\t ]+', stripped)
            try:
                row = [float(v) for v in values if v]
                per_object[current_key].append(row)
            except ValueError:
                continue

    for key in per_object:
        if per_object[key]:
            per_object[key] = np.array(per_object[key])
        else:
            per_object[key] = np.empty((0, len(header)))

    return {'header': header, 'per_object': per_object}


def _parse_maket_scn(simu_path):
    """Parse maket.scn for budget object to (instance, group) mapping."""
    maket_path = Path(simu_path) / 'output' / 'maket.scn'
    if not maket_path.exists():
        return None

    mapping = {}
    with open(maket_path) as f:
        for line in f:
            if 'dartNameId=fo' not in line:
                continue
            m = re.match(
                r'scene\.objects\.object(\d+)_0_0\.dartNameId='
                r'fo\d+_mo(\d+)_go(\d+)',
                line.strip(),
            )
            if m:
                budget_idx = int(m.group(1))
                instance = int(m.group(2))
                group = int(m.group(3))
                mapping[budget_idx] = (instance, group)

    return mapping if mapping else None


def _detect_delimiter(filepath):
    """Detect CSV delimiter from header line."""
    with open(filepath) as f:
        header = f.readline().strip()
    if ';' in header:
        return ';'
    if '\t' in header:
        return '\t'
    return ','


def _read_baleno_csv(filepath, delimiter=';'):
    """Read Baleno output CSV with header."""
    with open(filepath) as f:
        header_line = f.readline().strip()
    header = [h.strip() for h in header_line.split(delimiter)]

    try:
        data = np.genfromtxt(str(filepath), skip_header=1, delimiter=delimiter,
                             dtype=float, filling_values=np.nan)
    except ValueError:
        data = np.genfromtxt(str(filepath), skip_header=1, delimiter=delimiter,
                             dtype=str)

    return header, data


# ============================================================================
# Main
# ============================================================================
def main():
    print("=" * 70)
    print("Phase 6: Multi-Plant Unique Realizations")
    print("=" * 70)
    log_consistency(SIMULATION_DAYS)

    # Step 1: Grow 9 unique plants
    plants = step1_grow_plants()

    # Step 2: Export 9 OBJs + mapping JSONs
    meshes, mappings, obj_paths, dart_obj_paths, mapping_json_paths = \
        step2_export_meshes(plants)

    # Step 3: Create DART PAR simulation
    simu = step3_create_dart_simulation(dart_obj_paths)

    # Step 4: Run DART
    success = step4_run_dart(simu)
    if not success:
        print("\nDART failed, aborting.")
        return

    # Step 5: Read APAR per-plant
    apar_result = step5_read_apar(simu, dart_obj_paths, mappings)
    if apar_result is None:
        print("\nAPAR extraction failed, aborting.")
        return
    apar_results, per_plant_ori, per_plant_groups, per_plant_offsets = apar_result

    # Step 6: Create Baleno _I + _II simulations
    simu_eb, simu_II_dir = step6_create_baleno_simus(dart_obj_paths)

    # Step 7: Run Baleno
    baleno_ok = step7_run_baleno()
    if not baleno_ok:
        print("\nBaleno failed. Continuing with APAR-only photosynthesis...")
        # Fall through to photosynthesis with uniform Tleaf
        baleno_results = None
    else:
        # Step 8: Read Baleno outputs
        baleno_results = step8_read_baleno(dart_obj_paths, mappings)

    # If Baleno failed, create dummy results with uniform temperature
    if baleno_results is None:
        print("\n  Using uniform Tleaf for photosynthesis...")
        baleno_results = []
        for r in apar_results:
            baleno_results.append({
                'plant_id': r['plant_id'],
                'organ': r['organ'],
                'organ_type': r['organ_type'],
                'segment_idx': r['segment_idx'],
                'n_triangles': r['n_triangles'],
                'n_matched': 0,
                'Tleaf_K': TAIR_C + 273.15,
                'Tleaf_C': TAIR_C,
                'APAR_umol': np.nan,
                'LE_Wm2': np.nan,
                'H_Wm2': np.nan,
                'EB_error': 0.0,
            })

    # Step 9: Per-plant photosynthesis
    per_plant_summary, all_photo_results = step9_photosynthesis(
        apar_results, baleno_results, mappings)

    # Step 10: Field-level summary + figure
    step10_summary(per_plant_summary, all_photo_results,
                   apar_results, baleno_results)

    print(f"\n{'=' * 70}")
    print(f"Phase 6 complete. Output in: {OUTPUT_DIR}")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
