#!/usr/bin/env python3
"""
Phase 1: Standalone DART Simulation from CPlantBox OBJ.

Grows a CPlantBox maize plant (day 55), exports a G3 mesh, converts to DART
coordinates, creates a DART simulation with PROSPECT leaf optics and 6 PAR
bands, runs the full radiative transfer pipeline, reads per-triangle absorbed
radiation, and aggregates it to per-CPlantBox-segment aPAR.

Usage:
  cd /home/lukas/PHD
  source CPlantBox/cpbenv/bin/activate
  python CPlantBox/dart/coupling/create_dart_simulation.py
"""

import json
import re
import shutil
import numpy as np
from pathlib import Path
import plantbox as pb
import pytools4dart as ptd

from .. import config as _cfg
from ..config import DEFAULT_XML, OUTPUT_DIR
from ..growth import grow_plant, extract_g3_mesh
from ..geometry import convert_obj_to_dart, convert_mapping_json_groups
from ..prospect_params import (get_prospect_params, get_prospect_params_per_position,
                               get_stem_prospect_params,
                               get_tassel_prospect_params,
                               get_midrib_prospect_params,
                               get_senescent_leaf_prospect_params,
                               log_consistency, log_lops_consistency)
from .parsers import (parse_radiative_budget_txt, parse_maket_scn,
                      compute_plant_positions,
                      PLANT_POS, GRID_NX, GRID_NY, GRID_SPACING_X, GRID_SPACING_Y)

# ---------------------------------------------------------------------------
# Atmosphere helper
# ---------------------------------------------------------------------------

def configure_exact_date(simu, calendar_date, hour_utc, minute_utc,
                         lat=50.92, lon=6.36):
    """Configure DART to compute sun angles from date/time/location.

    Uses DART's built-in solar geometry (exactDate=1) instead of manually
    injecting sun zenith/azimuth angles.  Requires sunAzimuthalOffset=-90
    to correct for DART's azimuth convention vs geographic convention
    (see pytools4dart use_case_7, line 153-156).

    Args:
        simu: ptd.simulation object (not yet written).
        calendar_date: datetime.date with year/month/day.
        hour_utc: Hour in UTC (int).
        minute_utc: Minute (int).
        lat: Latitude (default: Juelich 50.92).
        lon: Longitude (default: Juelich 6.36).
    """
    simu.core.maket.set_nodes(latitude=lat, longitude=lon)
    simu.core.directions.set_nodes(exactDate=1)
    simu.core.directions.set_nodes(
        year=calendar_date.year, month=calendar_date.month,
        day=calendar_date.day, hour=int(hour_utc), minute=int(minute_utc),
        second=0, localTime=0, timezone=0, daylightSavingTime=0)
    simu.core.directions.set_nodes(sunAzimuthalOffset=-90)


def configure_atmosphere_midlatsum(simu, co2_ppm=None):
    """Configure DART atmosphere: MIDLATSUM + RURALV23, TOAtoBOA=2.

    Args:
        simu: pytools4dart simulation object.
        co2_ppm: CO2 mixing ratio (ppm). If None, uses active site config.
    """
    if co2_ppm is None:
        from .. import config as _cfg
        co2_ppm = _cfg.DEFAULT_CO2_PPM

    # Enable atmospheric RT simulation
    simu.core.phase.Phase.AtmosphereRadiativeTransfer.TOAtoBOA = 2

    # Gas model: Mid-Latitude Summer (typeOfAtmosphere=1 → database mode)
    atmo = simu.core.atmosphere.Atmosphere.IsAtmosphere
    atmo.AtmosphericOpticalPropertyModel.gasModelName = 'MIDLATSUM'
    atmo.AtmosphericOpticalPropertyModel.gasCumulativeModelName = 'MIDLATSUM'
    atmo.AtmosphericOpticalPropertyModel.temperatureModelName = 'MIDLATSUM'
    atmo.AtmosphericOpticalPropertyModel.co2MixRate = co2_ppm

    # Aerosol model: MIDLATSUM + Rural (visibility 23km)
    # Aerosols live under Atmosphere.Aerosol (separate from IsAtmosphere)
    simu.core.atmosphere.Atmosphere.Aerosol.AerosolProperties[0].aerosolsModelName = 'MIDLATSUM_RURALV23'


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
XML_PATH = str(DEFAULT_XML)
SIMULATION_DAYS = 55
SIMU_NAME = 'cpb_maize_day55_par'

# 6 contiguous PAR bands covering 400-700 nm (central wavelength in µm, bandwidth in µm)
PAR_BANDS = [
    (0.425, 0.050),   # 400-450 nm  violet-blue
    (0.475, 0.050),   # 450-500 nm  blue
    (0.525, 0.050),   # 500-550 nm  cyan-green
    (0.575, 0.050),   # 550-600 nm  green-yellow
    (0.625, 0.050),   # 600-650 nm  orange-red
    (0.675, 0.050),   # 650-700 nm  red
]

# PROSPECT parameters: loaded from shared growth-stage table
PROSPECT_PARAMS = get_prospect_params(SIMULATION_DAYS)

# Sun geometry: for standalone testing only.
# Diurnal pipeline uses configure_exact_date() (DART exactDate=1 mode).
SUN_ZENITH = 45.0
SUN_AZIMUTH = 225.0

# Scene geometry
SCENE_SIZE = [4.0, 2.25]  # meters (≥0.75 m border each side for 3×5 grid)
CELL_SIZE = 0.5            # meters (not directly settable via simple API; scene.size controls it)
N_PLANTS = GRID_NX * GRID_NY       # 15 (3 rows × 5 along-row)
CENTER_PLANT_IDX = N_PLANTS // 2   # 7 (0-indexed, center of 3x5)
FIELD_FILENAME = 'plant_field.txt'


def _write_field_file(path):
    """Write DART ObjectField position file for the plant grid.

    Format: 'complete transformation' header, then per-instance lines:
      model_index xpos ypos zpos xscale yscale zscale xrot yrot zrot
    """
    positions = compute_plant_positions()
    with open(path, 'w') as f:
        f.write('complete transformation\n')
        for x, y, yrot in positions:
            f.write(f'0 {x:.6f} {y:.6f} 0.0 1.0 1.0 1.0 0.0 {yrot:.2f} 0.0\n')
    return positions


# ============================================================================
# Step 1: Grow plant + export G3 mesh
# ============================================================================
def step1_grow_and_export():
    """Grow day-55 maize plant, extract G3 mesh, export OBJ + JSON."""
    print("=" * 70)
    print("STEP 1: Grow Plant + Export G3 Mesh")
    print("=" * 70)

    plant = grow_plant(XML_PATH, simulation_time=SIMULATION_DAYS,
                       min_stem_nodes=50, min_leaf_nodes=20)
    mesh, organ_dicts = extract_g3_mesh(plant, min_stem_nodes=50,
                                         min_leaf_nodes=20, stem_res=16)

    # Export standard OBJ
    obj_path = OUTPUT_DIR / 'maize_day55.obj'
    mesh.to_obj(str(obj_path), group_by_organ=True)
    print(f"\n  OBJ: {obj_path}")
    print(f"    Vertices: {mesh.n_vertices}, Triangles: {mesh.n_triangles}")

    # Export mapping JSON
    json_path = OUTPUT_DIR / 'maize_day55_mapping.json'
    mesh.to_mapping_json(str(json_path))
    print(f"  Mapping JSON: {json_path}")

    # Verify
    with open(json_path) as f:
        mapping = json.load(f)
    assert mapping['n_triangles'] == mesh.n_triangles, \
        f"Mismatch: JSON={mapping['n_triangles']} vs mesh={mesh.n_triangles}"

    n_groups = len(mapping['organs'])
    n_leaf_organs = sum(1 for o in mapping['organs'] if o['type'] == 'leaf')
    print(f"  Groups: {n_groups} ({n_leaf_organs} leaf, {n_groups - n_leaf_organs} stem)")

    # Save grid info for downstream steps
    positions = compute_plant_positions()
    grid_info = {
        'grid_nx': GRID_NX, 'grid_ny': GRID_NY,
        'spacing_x_m': GRID_SPACING_X, 'spacing_y_m': GRID_SPACING_Y,
        'n_plants': N_PLANTS,
        'center_plant_idx': CENTER_PLANT_IDX,
        'positions_m': positions,
        'n_groups_per_plant': n_groups,
        'field_filename': FIELD_FILENAME,
    }
    grid_path = OUTPUT_DIR / 'grid_info.json'
    with open(grid_path, 'w') as f:
        json.dump(grid_info, f, indent=2)
    print(f"  Grid info: {grid_path} ({N_PLANTS} plants, center={CENTER_PLANT_IDX})")

    return plant, mesh, mapping


# ============================================================================
# Step 2: Convert OBJ to DART coordinates
# ============================================================================
def step2_convert_to_dart():
    """Convert OBJ to DART convention (v y z x, cm->m, zero-padded groups)."""
    print("\n" + "=" * 70)
    print("STEP 2: Convert OBJ to DART Coordinates")
    print("=" * 70)

    input_obj = OUTPUT_DIR / 'maize_day55.obj'
    output_obj = OUTPUT_DIR / 'maize_day55_dart.obj'

    stats = convert_obj_to_dart(input_obj, output_obj, scale=0.01,
                                zero_pad_groups=True)

    print(f"  Vertices: {stats['n_vertices']}")
    print(f"  Faces:    {stats['n_faces']}")
    print(f"  Groups ({stats['n_groups']}): {stats['groups']}")

    # Update mapping JSON with zero-padded names
    json_in = OUTPUT_DIR / 'maize_day55_mapping.json'
    json_out = OUTPUT_DIR / 'maize_day55_dart_mapping.json'
    shutil.copy(json_in, json_out)
    convert_mapping_json_groups(str(json_out))
    print(f"  DART mapping: {json_out}")

    return stats


# ============================================================================
# Step 3: Create DART simulation with pytools4dart
# ============================================================================
def step3_create_dart_simulation():
    """Create DART simulation with 6 PAR bands, PROSPECT leaf optics."""
    print("\n" + "=" * 70)
    print("STEP 3: Create DART Simulation")
    print("=" * 70)

    # Clean up previous
    simu_dir = Path(ptd.getdartdir()) / 'user_data' / 'simulations' / SIMU_NAME
    if simu_dir.exists():
        shutil.rmtree(str(simu_dir))
        print(f"  Cleaned up previous: {simu_dir}")

    print(f"  DART version: {ptd.getdartversion()}")
    simu = ptd.simulation(SIMU_NAME, empty=True)
    print(f"  Simulation: {SIMU_NAME}")

    # Scene size
    simu.scene.size = SCENE_SIZE
    print(f"  Scene: {SCENE_SIZE[0]}m x {SCENE_SIZE[1]}m")

    # --- Spectral bands ---
    for wvl, bw in PAR_BANDS:
        simu.add.band(wvl=wvl, bw=bw)
        print(f"  Band: {wvl*1000:.0f}nm ± {bw*1000/2:.0f}nm")

    # --- Sun direction ---
    simu.core.directions.Directions.SunViewingAngles.sunViewingZenithAngle = SUN_ZENITH
    simu.core.directions.Directions.SunViewingAngles.sunViewingAzimuthAngle = SUN_AZIMUTH
    print(f"  Sun: zenith={SUN_ZENITH}°, azimuth={SUN_AZIMUTH}°")

    # --- Ground optical property ---
    simu.add.optical_property(
        type='Lambertian', ident='ground',
        databaseName='Lambertian_mineral.db', ModelName='clay_brown',
        useMultiplicativeFactorForLUT=0,
    )
    simu.scene.ground.OpticalPropertyLink.ident = 'ground'
    print(f"  Ground: clay_brown (Lambertian)")

    # --- Per-position leaf optical properties (PROSPECT from LOPS data) ---
    # Count leaf groups (non-_00 suffixes) to determine n_leaves
    dart_obj = OUTPUT_DIR / 'maize_day55_dart.obj'
    file_src_fullpath = simu.get_input_file_path(str(dart_obj))
    obj_info = ptd.OBJtools.objreader(file_src_fullpath)
    gnames = ptd.OBJtools.gnames_dart_order(obj_info.names)
    n_leaf_groups = sum(1 for g in gnames
                        if not g.endswith('_00')
                        and not g.endswith('_midrib')
                        and not g.startswith(('tassel_spike_', 'tassel_branch_')))

    per_pos_params = get_prospect_params_per_position(SIMULATION_DAYS, n_leaf_groups)
    leaf_idx = 0
    for i, params in enumerate(per_pos_params):
        ident = f'maize_leaf_pos{i}'
        simu.add.optical_property(
            type='Lambertian', ident=ident,
            prospect=params,
            useMultiplicativeFactorForLUT=0,
        )
        print(f"  Leaf OP: {ident} (Cab={params['Cab']:.1f}, N={params['N']:.2f})")
    log_lops_consistency(SIMULATION_DAYS, n_leaf_groups)

    stem_prospect = get_stem_prospect_params(SIMULATION_DAYS)
    simu.add.optical_property(
        type='Lambertian', ident='maize_stem',
        prospect=stem_prospect,
        useMultiplicativeFactorForLUT=0,
    )
    print(f"  Stem OP: maize_stem (PROSPECT Cab={stem_prospect['Cab']:.0f},"
          f" N={stem_prospect['N']:.1f}, CBrown={stem_prospect['CBrown']:.3f})")

    tassel_prospect = get_tassel_prospect_params(SIMULATION_DAYS)
    simu.add.optical_property(
        type='Lambertian', ident='maize_tassel',
        prospect=tassel_prospect,
        useMultiplicativeFactorForLUT=0,
    )
    print(f"  Tassel OP: maize_tassel (PROSPECT Cab={tassel_prospect['Cab']:.0f},"
          f" N={tassel_prospect['N']:.1f}, CBrown={tassel_prospect['CBrown']:.3f})")

    # Midrib OP — only registered when the OBJ contains ``*_midrib``
    # sub-groups (lofter emits these when ``midrib_amps_cm > 0``). Default
    # PROSPECT lowers Cab to ~30 % of blade and bumps N + Cw + Cm to
    # reflect the dense vascular tissue.
    _has_midrib = any(g.endswith('_midrib') for g in gnames)
    if _has_midrib:
        midrib_prospect = get_midrib_prospect_params(SIMULATION_DAYS)
        simu.add.optical_property(
            type='Lambertian', ident='maize_leaf_midrib',
            prospect=midrib_prospect,
            useMultiplicativeFactorForLUT=0,
        )
        print(f"  Midrib OP: maize_leaf_midrib "
              f"(PROSPECT Cab={midrib_prospect['Cab']:.0f}, "
              f"N={midrib_prospect['N']:.1f}, "
              f"CBrown={midrib_prospect['CBrown']:.3f})")

    # Senescent-leaf OP is only registered when the OBJ actually contains
    # ``senescent_leaf_*`` groups (learnings §3.2). When the adapter is
    # run with enable_senescent_split=False the prefix never appears and
    # this block is skipped, keeping the DART scene bit-identical to the
    # pre-feature baseline.
    _has_senescent = any(g.startswith('senescent_leaf_') for g in gnames)
    if _has_senescent:
        senescent_prospect = get_senescent_leaf_prospect_params(SIMULATION_DAYS)
        simu.add.optical_property(
            type='Lambertian', ident='maize_leaf_senescent',
            prospect=senescent_prospect,
            useMultiplicativeFactorForLUT=0,
        )
        print(f"  Senescent-leaf OP: maize_leaf_senescent "
              f"(PROSPECT Cab={senescent_prospect['Cab']:.0f}, "
              f"N={senescent_prospect['N']:.1f}, "
              f"CBrown={senescent_prospect['CBrown']:.3f})")

    # --- 3D Object via ObjectFields (plant grid) ---
    xdim, ydim, zdim = obj_info.dims
    xc, yc, zc = obj_info.center
    print(f"  OBJ: {dart_obj.name} ({len(gnames)} groups, "
          f"dims={xdim:.2f}x{ydim:.2f}x{zdim:.2f}m)")

    # Create group list with per-position optical properties + doubleFace.
    # Routing precedence: midrib suffix (catches both blade- and senescent-
    # leaf midribs) > tassel prefix > senescent prefix > stem (organ_00)
    # > per-position leaf.
    #
    # Midrib runs single-sided (doubleFace=0): the painted optical stripe
    # is meant as an adaxial-only feature, so the back face stays
    # optically inactive and the underside reads as bare blade rather
    # than a second painted ridge. Trade-off: a thin (~5–15 % × W)
    # central strip becomes RT-transparent from below — narrow enough
    # not to bias canopy fluxes, but worth noting if you ever try to
    # close the leaf-level energy balance to <1 % at the strip itself.
    groups_list = []
    leaf_idx = 0
    for gi, gname in enumerate(gnames):
        g = ptd.object_3d.create_Group(num=gi + 1, name=gname)
        if gname.endswith('_midrib'):
            op_ident = 'maize_leaf_midrib'
            df = 0
        elif gname.startswith(('tassel_spike_', 'tassel_branch_')):
            op_ident = 'maize_tassel'
            df = 1
        elif gname.startswith('senescent_leaf_'):
            op_ident = 'maize_leaf_senescent'
            df = 1
        elif gname.endswith('_00'):
            op_ident = 'maize_stem'
            df = 0
        else:
            op_ident = f'maize_leaf_pos{leaf_idx}'
            leaf_idx += 1
            df = 1
        g.set_nodes(ident=op_ident)
        gop = g.GroupOpticalProperties
        gop.SurfaceOpticalProperties.doubleFace = df
        gop.SurfaceExitanceProperties.doubleFace = df
        groups_list.append(g)
        print(f"    Group {gi+1}: {gname} -> {op_ident}, doubleFace={df}")
    groups = ptd.object_3d.create_Groups(Group=groups_list)

    # Create model object for ObjectFields (position comes from field file)
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
        num=0,
        name='CPlantBox_Maize',
        objectDEMMode=0,
    )

    # Create ObjectFields with the field file
    model_list = ptd.object_3d.create_ModelList()
    model_list.add_Object(model_obj)
    field = ptd.object_3d.create_Field(
        name='MaizeField',
        fieldDescriptionFileName=FIELD_FILENAME,
    )
    field.set_ModelList(model_list)
    obj_fields = ptd.object_3d.create_ObjectFields()
    obj_fields.add_Field(field)
    simu.core.object_3d.object_3d.ObjectFields = obj_fields

    positions = compute_plant_positions()
    print(f"  ObjectFields: {N_PLANTS} plants, center idx={CENTER_PLANT_IDX}")
    print(f"  Grid: {GRID_NX}x{GRID_NY}, spacing {GRID_SPACING_X}m x {GRID_SPACING_Y}m")
    print(f"  Center plant at ({positions[CENTER_PLANT_IDX][0]:.2f}, "
          f"{positions[CENTER_PLANT_IDX][1]:.2f})m")

    # --- Radiative budget products ---
    products = simu.core.phase.Phase.DartProduct.dartModuleProducts.CommonProducts
    products.radiativeBudgetProducts = 1
    # Enable per-triangle budget
    products.radiativeBudgetProperties.budget3DParSurface = 1
    print(f"  Radiative budget: enabled (per-triangle)")

    # --- Engine: use Lux (accelerationEngine=2 for forward+backward) ---
    simu.core.phase.Phase.accelerationEngine = 2
    print(f"  Engine: Lux (accelerationEngine=2)")

    # --- LuxCore sampling ---
    lux = simu.core.phase.Phase.EngineParameter.LuxCoreRenderEngineParameters
    lux.targetRayDensityPerPixel = _cfg.DART_RAY_DENSITY_PER_PIXEL
    lux.maximumRenderingTime = _cfg.DART_MAX_RENDERING_TIME
    print(f"  LuxCore sampling: {_cfg.DART_RAY_DENSITY_PER_PIXEL} rays/pixel, "
          f"maxTime={_cfg.DART_MAX_RENDERING_TIME}s")

    # --- Threads ---
    simu.core.phase.Phase.ExpertModeZone.nbThreads = _cfg.DART_THREADS
    print(f"  Threads: {_cfg.DART_THREADS}")

    # --- Atmosphere: MIDLATSUM ---
    configure_atmosphere_midlatsum(simu)
    print("  Atmosphere: MIDLATSUM + RURALV23 (TOAtoBOA=2)")

    # --- Write simulation ---
    simu.write(overwrite=True)
    print(f"\n  Simulation written to: {simu.simu_dir}")

    # --- Write field file into simulation input directory ---
    simu_path = Path(str(simu.simu_dir))
    field_path = simu_path / 'input' / FIELD_FILENAME
    _write_field_file(field_path)
    print(f"  Field file: {field_path} ({N_PLANTS} positions)")

    # --- Verify XML configuration ---
    _verify_xml(simu)

    return simu


def _verify_xml(simu):
    """Read back XML to confirm key settings."""
    print(f"\n  XML verification:")
    simu_path = Path(str(simu.simu_dir))

    # Check object_3d.xml for doubleFace
    obj3d_xml = simu_path / 'input' / 'object_3d.xml'
    if obj3d_xml.exists():
        content = obj3d_xml.read_text()
        df_count = content.count('doubleFace="1"')
        df0_count = content.count('doubleFace="0"')
        print(f"    object_3d.xml: doubleFace=1 ({df_count}x), doubleFace=0 ({df0_count}x)")
        if df0_count > 0:
            print(f"    WARNING: {df0_count} groups still have doubleFace=0!")

    # Check phase.xml for radiativeBudgetProducts
    phase_xml = simu_path / 'input' / 'phase.xml'
    if phase_xml.exists():
        content = phase_xml.read_text()
        if 'radiativeBudgetProducts="1"' in content:
            print(f"    phase.xml: radiativeBudgetProducts=1 OK")
        else:
            print(f"    WARNING: radiativeBudgetProducts not set to 1!")
        if 'budget3DParSurface="1"' in content:
            print(f"    phase.xml: budget3DParSurface=1 OK")
        else:
            print(f"    WARNING: budget3DParSurface not set to 1!")

        # Count bands
        band_count = content.count('SpectralIntervalsProperties')
        print(f"    phase.xml: ~{band_count // 2} spectral bands")

    # Check coeff_diff.xml for useMultiplicativeFactorForLUT
    cd_xml = simu_path / 'input' / 'coeff_diff.xml'
    if cd_xml.exists():
        content = cd_xml.read_text()
        lut1 = content.count('useMultiplicativeFactorForLUT="1"')
        lut0 = content.count('useMultiplicativeFactorForLUT="0"')
        print(f"    coeff_diff.xml: useMultiplicativeFactorForLUT=0 ({lut0}x), =1 ({lut1}x)")
        if lut1 > 0:
            print(f"    WARNING: {lut1} optical properties have useMultiplicativeFactorForLUT=1!")


# ============================================================================
# Step 4: Run DART pipeline
# ============================================================================
def step4_run_dart(simu):
    """Run full DART simulation (direction -> phase -> maket -> dart)."""
    print("\n" + "=" * 70)
    print("STEP 4: Run DART Pipeline")
    print("=" * 70)

    # Try full pipeline first
    print(f"  Running full simulation (timeout=600s)...")
    try:
        result = simu.run.full(timeout=600)
        print(f"  Full run result: {result}")
        if result:
            return True
    except Exception as e:
        print(f"  Full run failed: {e}")

    # Fallback: run stages individually to isolate the crash
    print(f"\n  Falling back to individual stages...")
    stages = [
        ('direction', simu.run.direction),
        ('phase', simu.run.phase),
        ('maket', simu.run.maket),
        ('dart', simu.run.dart),
    ]
    for name, runner in stages:
        print(f"  Running {name}...")
        try:
            ok = runner(timeout=300)
            print(f"    {name}: {'OK' if ok else 'FAILED'}")
            if not ok:
                print(f"    Continuing despite {name} failure...")
        except Exception as e:
            print(f"    {name} crashed: {e}")

    # Check if output exists
    simu_path = Path(str(simu.simu_dir))
    band_dirs = sorted(simu_path.glob('output/BAND*'))
    if band_dirs:
        print(f"\n  Output bands found: {[d.name for d in band_dirs]}")
        return True
    else:
        # Check if there's a Lux-style output (single combined output)
        output_dir = simu_path / 'output'
        if output_dir.exists():
            all_files = list(output_dir.rglob('*'))
            print(f"\n  Output directory has {len(all_files)} files")
            for f in all_files[:20]:
                print(f"    {f.relative_to(output_dir)}")
            return True

    print(f"\n  ERROR: No DART output found!")
    return False


# ============================================================================
# Step 5: Read .ori reindex tables
# ============================================================================
def step5_read_ori(simu, mapping):
    """Read .ori files for DART-to-OBJ triangle reindexing."""
    print("\n" + "=" * 70)
    print("STEP 5: Read .ori Reindex Tables")
    print("=" * 70)

    simu_path = Path(str(simu.simu_dir))
    ori_dir = simu_path / 'input' / 'triangles'

    if not ori_dir.exists():
        print(f"  ERROR: {ori_dir} does not exist!")
        return None

    # Read group info from DART OBJ
    dart_obj = OUTPUT_DIR / 'maize_day55_dart.obj'
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
    print(f"  OBJ: {total_faces} faces, {len(groups_sorted)} groups")

    # Read .ori files
    ori_data = {}
    for gi in range(len(groups_sorted)):
        ori_path = ori_dir / f'triangle{gi}.ori'
        if ori_path.exists():
            data = np.fromfile(str(ori_path), dtype='uint32')
            ori_data[gi] = data
        else:
            print(f"  WARNING: {ori_path.name} not found")

    if not ori_data:
        print("  ERROR: No .ori files found!")
        return None

    # Build reindex tables
    # dart_to_obj[gi][dart_pos] = global OBJ face index
    dart_to_obj = {}
    for gi in sorted(ori_data.keys()):
        offset = group_offsets[groups_sorted[gi]]
        dart_to_obj[gi] = ori_data[gi].astype(np.int64) + offset

    # obj_to_dart[global_obj_face] = (group_idx, dart_position)
    obj_to_dart = {}
    for gi in sorted(dart_to_obj.keys()):
        for dart_pos, global_obj_idx in enumerate(dart_to_obj[gi]):
            obj_to_dart[int(global_obj_idx)] = (gi, dart_pos)

    n_dart_total = sum(len(v) for v in ori_data.values())
    print(f"  DART triangles: {n_dart_total}")
    print(f"  Dropped by DART: {total_faces - n_dart_total} (degenerate)")

    # Per-group summary
    for gi, gname in enumerate(groups_sorted):
        if gi in ori_data:
            print(f"    triangle{gi} ({gname}): OBJ={group_face_counts[gname]}, "
                  f"DART={len(ori_data[gi])}")

    # Detect per-plant group structure from grid info
    grid_path = OUTPUT_DIR / 'grid_info.json'
    n_groups_per_plant = len(groups_sorted)
    center_groups = list(range(len(groups_sorted)))  # default: all groups
    if grid_path.exists():
        with open(grid_path) as gf:
            grid_info = json.load(gf)
        n_plants = grid_info['n_plants']
        center_idx = grid_info['center_plant_idx']
        n_groups_per_plant = grid_info['n_groups_per_plant']
        if len(groups_sorted) == n_plants * n_groups_per_plant:
            center_start = center_idx * n_groups_per_plant
            center_groups = list(range(center_start, center_start + n_groups_per_plant))
            print(f"\n  Grid: {n_plants} plants, {n_groups_per_plant} groups/plant")
            print(f"  Center plant (idx {center_idx}): "
                  f"groups {center_start}..{center_start + n_groups_per_plant - 1}")
        else:
            # ObjectFields: .ori files are per-model (shared across instances)
            # Center plant filtering happens in step 7 via maket.scn
            print(f"\n  Grid: {n_plants} plants, {n_groups_per_plant} groups/model")
            print(f"  .ori files: {len(groups_sorted)} (per-model, shared across instances)")
            print(f"  Center plant filtering: via maket.scn in step 7")

    # Save reindex
    reindex = {
        'dart_to_obj': {str(k): v.tolist() for k, v in dart_to_obj.items()},
        'obj_to_dart_sample': {str(k): list(v) for k, v in list(obj_to_dart.items())[:10]},
        'group_names': groups_sorted,
        'group_offsets': {g: group_offsets[g] for g in groups_sorted},
        'n_dart_total': n_dart_total,
        'n_obj_total': total_faces,
        'center_groups': center_groups,
        'n_groups_per_plant': n_groups_per_plant,
    }
    reindex_path = OUTPUT_DIR / 'maize_day55_reindex.json'
    with open(reindex_path, 'w') as f:
        json.dump(reindex, f, indent=2)
    print(f"\n  Saved: {reindex_path}")

    return dart_to_obj, obj_to_dart, groups_sorted, group_offsets, ori_data


# ============================================================================
# Step 6: Read per-triangle radiative budget
# ============================================================================
def step6_read_radiative_budget(simu):
    """Parse DART per-triangle radiative budget output for each band."""
    print("\n" + "=" * 70)
    print("STEP 6: Read Per-Triangle Radiative Budget")
    print("=" * 70)

    simu_path = Path(str(simu.simu_dir))
    output_path = simu_path / 'output'

    # Discover output structure
    print(f"  Scanning: {output_path}")

    # DART organizes output per band: output/BAND{N}/RADIATIVE_BUDGET/ITERX/
    # The text file is: RadiativeBudgetFigures.txt
    # For Lux engine, it may also be in netcdf format

    per_band_data = {}

    # Try text-based radiative budget first (BAND0, BAND1, ...)
    band_dirs = sorted(output_path.glob('BAND*'))
    if band_dirs:
        print(f"  Found {len(band_dirs)} band directories")
        for band_dir in band_dirs:
            band_idx = int(re.search(r'BAND(\d+)', band_dir.name).group(1))
            rb_file = band_dir / 'RADIATIVE_BUDGET' / 'ITERX' / 'RadiativeBudgetFigures.txt'
            if rb_file.exists():
                data = parse_radiative_budget_txt(rb_file)
                if data is not None:
                    per_band_data[band_idx] = data
                    print(f"    BAND{band_idx}: {sum(len(v) for v in data['per_object'].values())} triangles")
            else:
                # Check for any file in RADIATIVE_BUDGET
                rb_dir = band_dir / 'RADIATIVE_BUDGET'
                if rb_dir.exists():
                    files = list(rb_dir.rglob('*'))
                    print(f"    BAND{band_idx}: RadiativeBudgetFigures.txt not found, "
                          f"but {len(files)} files in RADIATIVE_BUDGET/")
                    for f in files[:5]:
                        print(f"      {f.name} ({f.stat().st_size} bytes)")

    # Try NetCDF format
    if not per_band_data:
        netcdf_dir = output_path / 'netcdf' / 'radiativeBudget'
        if netcdf_dir.exists():
            nc_files = list(netcdf_dir.glob('*.nc'))
            print(f"  Found {len(nc_files)} NetCDF files in {netcdf_dir}")
            for nc_file in nc_files:
                print(f"    {nc_file.name} ({nc_file.stat().st_size} bytes)")
            # NetCDF reading would require netCDF4; implement if txt format not available
            if nc_files:
                per_band_data = _parse_radiative_budget_netcdf(nc_files, len(PAR_BANDS))

    # If still no data, list all output files for debugging
    if not per_band_data:
        print(f"\n  WARNING: No radiative budget data found!")
        print(f"  Full output listing:")
        for f in sorted(output_path.rglob('*')):
            if f.is_file():
                print(f"    {f.relative_to(output_path)} ({f.stat().st_size} bytes)")

    return per_band_data


def _parse_radiative_budget_netcdf(nc_files, n_bands):
    """Parse radiative budget from NetCDF files."""
    try:
        import h5py
    except ImportError:
        print("  h5py not available, cannot parse NetCDF radiative budget")
        return {}

    per_band_data = {}
    for nc_file in nc_files:
        if 'Triangle' in nc_file.name or 'triangle' in nc_file.name:
            try:
                f = h5py.File(str(nc_file), 'r')
                # Navigate to Triangles/ITERX group
                triangles_group = f
                for group_name in ['Triangles', 'ITERX']:
                    if group_name in triangles_group:
                        triangles_group = triangles_group[group_name]

                # Read object list
                if 'Objects_list' in triangles_group:
                    obj_list = triangles_group['Objects_list'][()].tobytes().decode('utf-8').split(';')
                else:
                    obj_list = []

                # Read per-band data
                for band_idx in range(n_bands):
                    # Try different naming conventions
                    for fmt in [f'Band_{band_idx}', f'Band_{band_idx:02d}',
                                f'Band_{band_idx:03d}']:
                        if fmt in triangles_group:
                            dset = triangles_group[fmt]
                            data = dset[:]
                            header = dset.attrs['_tableHeader'].split(';') if '_tableHeader' in dset.attrs else []

                            # Split by object
                            per_object = {}
                            for oi, obj_name in enumerate(obj_list):
                                mask = data[:, 0] == oi
                                per_object[f'ObjectName {obj_name}'] = data[mask]

                            per_band_data[band_idx] = {
                                'header': header,
                                'per_object': per_object,
                            }
                            break

                f.close()
            except Exception as e:
                print(f"  Error reading {nc_file.name}: {e}")

    return per_band_data


def _build_budget_to_ori_mapping(per_band_data, ori_data,
                                  maket_mapping=None, center_plant_idx=None):
    """Build mapping from DART budget object index to .ori group index.

    When *maket_mapping* is provided (multi-plant ObjectFields), uses the
    exact mapping from ``maket.scn`` filtered to the center plant.  Otherwise
    falls back to greedy matching by triangle count (single-plant case).
    """
    if maket_mapping is not None and center_plant_idx is not None:
        # ---- Multi-plant: exact mapping via maket.scn ----
        budget_to_ori = {}
        for budget_idx, (instance, group) in maket_mapping.items():
            if instance == center_plant_idx and group in ori_data:
                budget_to_ori[budget_idx] = group

        # Reporting
        n_total_veg = len(maket_mapping)
        n_center = len(budget_to_ori)
        n_neighbor = n_total_veg - n_center

        sample_band = list(per_band_data.values())[0]
        all_budget_ids = set()
        for obj_key in sample_band['per_object']:
            m2 = re.search(r'object(\d+)', obj_key)
            if m2:
                all_budget_ids.add(int(m2.group(1)))
        n_ground = len(all_budget_ids) - n_total_veg

        print(f"  maket.scn: {n_total_veg} vegetation objects, "
              f"{n_center} center plant (instance {center_plant_idx}), "
              f"{n_neighbor} neighbors, {n_ground} ground/scene")
        return budget_to_ori

    # ---- Single-plant fallback: greedy match by triangle count ----
    sample_band = list(per_band_data.values())[0]
    budget_sizes = {}
    for obj_key, arr in sample_band['per_object'].items():
        m = re.search(r'object(\d+)', obj_key)
        if m:
            budget_idx = int(m.group(1))
            budget_sizes[budget_idx] = arr.shape[0] if arr.ndim == 2 else 1

    ori_sizes = {gi: len(data) for gi, data in ori_data.items()}

    budget_to_ori = {}
    used_ori = set()
    for budget_idx, bsize in sorted(budget_sizes.items()):
        for gi in sorted(ori_sizes.keys()):
            if gi in used_ori:
                continue
            if bsize == ori_sizes[gi]:
                budget_to_ori[budget_idx] = gi
                used_ori.add(gi)
                break

    unmatched = set(budget_sizes.keys()) - set(budget_to_ori.keys())
    if unmatched:
        for bi in unmatched:
            print(f"  Skipping budget object{bi} ({budget_sizes[bi]} tris, "
                  f"likely ground/scene)")

    return budget_to_ori


# ============================================================================
# Step 7: Aggregate per-triangle aPAR to per-segment
# ============================================================================
def step7_aggregate_to_segments(per_band_data, step5_result, mapping):
    """Map DART per-triangle absorbed radiation to CPlantBox segments."""
    print("\n" + "=" * 70)
    print("STEP 7: Aggregate Per-Triangle aPAR to Per-Segment")
    print("=" * 70)

    if not per_band_data:
        print("  ERROR: No radiative budget data to aggregate!")
        return None

    dart_to_obj, obj_to_dart, groups_sorted, group_offsets, ori_data = step5_result

    # Load center plant group indices from reindex
    reindex_path = OUTPUT_DIR / 'maize_day55_reindex.json'
    center_groups = None
    if reindex_path.exists():
        with open(reindex_path) as f:
            reindex = json.load(f)
        center_groups = set(reindex.get('center_groups', []))
        if center_groups:
            print(f"  Center plant groups: {sorted(center_groups)}")

    # Find the "absorbed" columns in the radiative budget
    # Typical columns: TriangleIndex, Intercepted, Scattered, Emitted, Absorbed,
    #   Intercepted_back, Scattered_back, Emitted_back, Absorbed_back, surface_area_[m2]
    sample_band = list(per_band_data.values())[0]
    header = sample_band['header']
    print(f"  Budget columns: {header}")

    # Find Absorbed (front) and Absorbed_back columns
    abs_col_idx = None
    abs_back_col_idx = None
    for i, h in enumerate(header):
        h_lower = h.lower()
        if h_lower == 'absorbed':
            abs_col_idx = i
        elif h_lower == 'absorbed_back':
            abs_back_col_idx = i

    if abs_col_idx is None:
        # Try intercepted as fallback
        for i, h in enumerate(header):
            if h.lower() == 'intercepted':
                abs_col_idx = i
                print(f"  WARNING: No 'Absorbed' column, using 'Intercepted' (col {i})")
                break
    if abs_col_idx is None:
        print(f"  ERROR: Cannot find absorbed column in {header}")
        print(f"  Using last column as fallback")
        abs_col_idx = len(header) - 1

    print(f"  Using front: col {abs_col_idx} ('{header[abs_col_idx]}')")
    if abs_back_col_idx is not None:
        print(f"  Using back:  col {abs_back_col_idx} ('{header[abs_back_col_idx]}')")
    else:
        print(f"  No back-face absorption column found")

    # Build mapping from DART budget object index to .ori group index.
    # For multi-plant ObjectFields: parse maket.scn for exact center-plant
    # mapping.  Single-plant: fall back to greedy triangle-count matching.
    simu_path = (Path(ptd.getdartdir()) / 'user_data' / 'simulations'
                 / SIMU_NAME)
    maket_mapping = parse_maket_scn(simu_path)
    budget_to_ori = _build_budget_to_ori_mapping(
        per_band_data, ori_data,
        maket_mapping=maket_mapping,
        center_plant_idx=CENTER_PLANT_IDX,
    )
    print(f"\n  Budget-to-ORI mapping ({len(budget_to_ori)} matched):")
    for budget_idx, ori_gi in sorted(budget_to_ori.items()):
        print(f"    object{budget_idx} -> triangle{ori_gi}.ori ({groups_sorted[ori_gi]})")

    # Build per-band, per-ORI-group absorbed arrays
    # Key: .ori group index (matching obj_to_dart lookup)
    per_band_absorbed = {}  # band_idx -> {ori_group_idx -> numpy array of absorbed values}

    for band_idx, band_data in per_band_data.items():
        per_band_absorbed[band_idx] = {}
        for obj_key, arr in band_data['per_object'].items():
            # Extract budget object index from name
            # Format: "ObjectName: object0_0_0" -> budget index 0
            m = re.search(r'object(\d+)', obj_key)
            if m:
                budget_idx = int(m.group(1))
                # Map to .ori group index (skip unmapped objects like ground)
                if budget_idx not in budget_to_ori:
                    continue
                gi = budget_to_ori[budget_idx]
                if arr.ndim == 2 and arr.shape[1] > abs_col_idx:
                    # Sum front + back face absorption (doubleFace=1)
                    front = arr[:, abs_col_idx]
                    if abs_back_col_idx is not None and arr.shape[1] > abs_back_col_idx:
                        back = arr[:, abs_back_col_idx]
                        per_band_absorbed[band_idx][gi] = front + back
                    else:
                        per_band_absorbed[band_idx][gi] = front
                elif arr.ndim == 1 and len(arr) > abs_col_idx:
                    val = arr[abs_col_idx]
                    if abs_back_col_idx is not None and len(arr) > abs_back_col_idx:
                        val += arr[abs_back_col_idx]
                    per_band_absorbed[band_idx][gi] = np.array([val])

    # Build center-plant-only obj_to_dart mapping.
    # With ObjectFields, all instances share the same OBJ face indices.
    # Filter to only center plant's .ori groups so we aggregate only its budget.
    if center_groups:
        center_obj_to_dart = {
            obj_idx: (gi, dart_pos)
            for obj_idx, (gi, dart_pos) in obj_to_dart.items()
            if gi in center_groups
        }
        print(f"  Center plant OBJ-to-DART entries: {len(center_obj_to_dart)} "
              f"(of {len(obj_to_dart)} total)")
    else:
        center_obj_to_dart = obj_to_dart

    # Now aggregate to segments using the JSON mapping + reindex
    segment_results = []
    band_indices = sorted(per_band_absorbed.keys())

    for oi, organ in enumerate(mapping['organs']):
        for seg in organ['segments']:
            tri_indices = seg['triangle_indices']  # global OBJ face indices
            if not tri_indices:
                continue

            # For each band, collect absorbed values for this segment's triangles
            # Use center_obj_to_dart to only get center plant's radiation
            per_band_vals = {}
            for band_idx in band_indices:
                vals = []
                for tidx in tri_indices:
                    if tidx in center_obj_to_dart:
                        gi, dart_pos = center_obj_to_dart[tidx]
                        if gi in per_band_absorbed.get(band_idx, {}):
                            absorbed_arr = per_band_absorbed[band_idx][gi]
                            if dart_pos < len(absorbed_arr):
                                vals.append(absorbed_arr[dart_pos])
                per_band_vals[band_idx] = vals

            # Compute mean absorbed across all bands (total aPAR)
            all_vals = []
            for band_idx in band_indices:
                all_vals.extend(per_band_vals.get(band_idx, []))

            # Per-band means
            band_means = {}
            for band_idx in band_indices:
                bv = per_band_vals.get(band_idx, [])
                band_means[band_idx] = np.mean(bv) if bv else 0.0

            # Total aPAR = sum of band means (W/m² across all bands)
            total_apar = sum(band_means.values())

            segment_results.append({
                'organ': organ['name'],
                'organ_type': organ['type'],
                'segment_idx': seg['segment_idx'],
                'n_triangles': len(tri_indices),
                'n_dart_matched': sum(1 for t in tri_indices if t in obj_to_dart),
                'total_apar': total_apar,
                'band_apar': band_means,
            })

    # Summary
    leaf_segs = [r for r in segment_results if r['organ_type'] == 'leaf']
    leaf_with_apar = [r for r in leaf_segs if r['total_apar'] > 0]
    stem_segs = [r for r in segment_results if r['organ_type'] == 'stem']

    print(f"\n  Leaf segments: {len(leaf_segs)}")
    print(f"  Leaf segments with aPAR > 0: {len(leaf_with_apar)} "
          f"({100 * len(leaf_with_apar) / max(len(leaf_segs), 1):.1f}%)")
    print(f"  Stem segments: {len(stem_segs)}")

    if leaf_with_apar:
        apar_values = np.array([r['total_apar'] for r in leaf_with_apar])
        print(f"\n  Leaf aPAR statistics:")
        print(f"    Mean:   {np.mean(apar_values):.4f}")
        print(f"    Median: {np.median(apar_values):.4f}")
        print(f"    Min:    {np.min(apar_values):.4f}")
        print(f"    Max:    {np.max(apar_values):.4f}")

        # Per-band summary
        print(f"\n  Per-band mean absorbed (leaf segments):")
        for band_idx in band_indices:
            band_vals = [r['band_apar'][band_idx] for r in leaf_with_apar
                         if r['band_apar'].get(band_idx, 0) > 0]
            if band_vals:
                wvl = PAR_BANDS[band_idx][0] * 1000 if band_idx < len(PAR_BANDS) else 0
                print(f"    Band {band_idx} ({wvl:.0f}nm): "
                      f"mean={np.mean(band_vals):.4f}, n={len(band_vals)}")

    return segment_results


# ============================================================================
# Step 8: Summary + sanity checks + output
# ============================================================================
def step8_summary_and_output(segment_results, per_band_data, mapping):
    """Run sanity checks and save results."""
    print("\n" + "=" * 70)
    print("STEP 8: Summary + Sanity Checks + Output")
    print("=" * 70)

    if segment_results is None:
        print("  ERROR: No segment results to summarize!")
        return False

    leaf_segs = [r for r in segment_results if r['organ_type'] == 'leaf']
    leaf_with_apar = [r for r in leaf_segs if r['total_apar'] > 0]
    band_indices = sorted(per_band_data.keys()) if per_band_data else []

    # --- Sanity Check 1: Coverage ---
    # Some leaf segments may have aPAR=0 due to self-shading or
    # segments with no triangles mapped (tip extensions). 70% is reasonable.
    coverage = len(leaf_with_apar) / max(len(leaf_segs), 1) * 100
    check1 = coverage > 70
    print(f"\n  Check 1: Leaf segment coverage = {coverage:.1f}% "
          f"{'PASS' if check1 else 'FAIL'} (threshold: >70%)")

    # --- Sanity Check 2: Spectral absorption pattern ---
    # With PROSPECT, chlorophyll causes a "green dip": absorption at 550nm
    # should be lower than linear interpolation between blue (475nm) and red
    # (680nm). Absolute values may increase from blue to red due to the solar
    # spectral slope, so we test relative dip, not absolute ordering.
    check2 = True
    if len(band_indices) >= 3 and leaf_with_apar:
        blue_mean = np.mean([r['band_apar'].get(1, 0) for r in leaf_with_apar])   # 475nm
        green_mean = np.mean([r['band_apar'].get(2, 0) for r in leaf_with_apar])  # 525nm
        red_mean = np.mean([r['band_apar'].get(5, 0) for r in leaf_with_apar])    # 675nm

        # Interpolated value at 525nm between 475nm and 675nm
        interp_green = blue_mean + (red_mean - blue_mean) * (525 - 475) / (675 - 475)
        green_dip = (interp_green - green_mean) / interp_green * 100 if interp_green > 0 else 0
        check2 = green_dip > 1.0  # at least 1% dip at green
        print(f"  Check 2: Spectral green-dip (PROSPECT chlorophyll signature):")
        print(f"    Blue (475nm):  {blue_mean:.4f}")
        print(f"    Green (525nm): {green_mean:.4f}")
        print(f"    Red (675nm):   {red_mean:.4f}")
        print(f"    Interpolated green: {interp_green:.4f}")
        print(f"    Green dip: {green_dip:.1f}%")
        print(f"    {'PASS' if check2 else 'FAIL'} (threshold: >1% dip)")
    else:
        print(f"  Check 2: Spectral pattern - SKIPPED (insufficient band data)")

    # --- Sanity Check 3: Total plant aPAR magnitude ---
    check3 = True
    if leaf_with_apar:
        total_apar = sum(r['total_apar'] for r in leaf_with_apar)
        print(f"  Check 3: Total plant aPAR = {total_apar:.2f}")
        print(f"    (Physically reasonable range depends on DART output units)")

    # --- Save per-segment CSV ---
    csv_path = OUTPUT_DIR / 'maize_day55_segment_apar.csv'
    with open(csv_path, 'w') as f:
        # Header
        band_cols = ','.join(f'apar_band{bi}' for bi in band_indices)
        f.write(f"organ,organ_type,segment_idx,n_triangles,n_dart_matched,"
                f"total_apar,{band_cols}\n")
        for r in segment_results:
            band_vals = ','.join(f"{r['band_apar'].get(bi, 0):.6f}"
                                for bi in band_indices)
            f.write(f"{r['organ']},{r['organ_type']},{r['segment_idx']},"
                    f"{r['n_triangles']},{r['n_dart_matched']},"
                    f"{r['total_apar']:.6f},{band_vals}\n")
    print(f"\n  Saved: {csv_path} ({len(segment_results)} segments)")

    # --- Save results JSON ---
    results = {
        'simulation_name': SIMU_NAME,
        'simulation_days': SIMULATION_DAYS,
        'xml_path': str(XML_PATH),
        'par_bands': [{'wvl_um': w, 'bw_um': b} for w, b in PAR_BANDS],
        'prospect_params': PROSPECT_PARAMS,
        'sun_zenith': SUN_ZENITH,
        'sun_azimuth': SUN_AZIMUTH,
        'plant_grid': {
            'nx': GRID_NX, 'ny': GRID_NY,
            'n_plants': N_PLANTS,
            'center_plant_idx': CENTER_PLANT_IDX,
            'spacing_x_m': GRID_SPACING_X,
            'spacing_y_m': GRID_SPACING_Y,
        },
        'n_triangles_obj': mapping['n_triangles'],
        'n_leaf_segments': len(leaf_segs),
        'n_leaf_segments_with_apar': len(leaf_with_apar),
        'coverage_pct': coverage,
        'checks': {
            'coverage_90pct': bool(check1),
            'spectral_pattern': bool(check2),
        },
    }

    if leaf_with_apar:
        apar_values = [r['total_apar'] for r in leaf_with_apar]
        results['apar_stats'] = {
            'mean': float(np.mean(apar_values)),
            'median': float(np.median(apar_values)),
            'min': float(np.min(apar_values)),
            'max': float(np.max(apar_values)),
            'total': float(sum(apar_values)),
        }

        # Per-band stats
        results['per_band_stats'] = {}
        for bi in band_indices:
            bv = [r['band_apar'].get(bi, 0) for r in leaf_with_apar
                  if r['band_apar'].get(bi, 0) > 0]
            if bv:
                results['per_band_stats'][f'band{bi}'] = {
                    'wvl_nm': PAR_BANDS[bi][0] * 1000 if bi < len(PAR_BANDS) else 0,
                    'mean': float(np.mean(bv)),
                    'min': float(np.min(bv)),
                    'max': float(np.max(bv)),
                    'n_segments': len(bv),
                }

    results_path = OUTPUT_DIR / 'phase1_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {results_path}")

    # --- Final verdict ---
    all_pass = check1 and check2
    print(f"\n{'=' * 70}")
    print(f"PHASE 1 RESULT: {'PASS' if all_pass else 'PARTIAL'}")
    print(f"{'=' * 70}")
    if all_pass:
        print(f"  DART radiative transfer on CPlantBox mesh completed successfully.")
        print(f"  Per-segment aPAR is available for coupling to CPlantBox photosynthesis.")
    else:
        if not check1:
            print(f"  WARNING: Low leaf segment coverage ({coverage:.1f}%)")
        if not check2:
            print(f"  WARNING: Unexpected spectral absorption pattern")

    return all_pass


# ============================================================================
# Reusable API for diurnal loop
# ============================================================================

def create_dart_simulation(obj_path, mapping_json_path, simu_name,
                           sun_zenith, sun_azimuth,
                           prospect_params, scene_size, grid_info,
                           par_bands=None, field_filename='plant_field.txt'):
    """Create and write a DART simulation with given parameters.

    This is the reusable counterpart of step3_create_dart_simulation().
    Returns the ptd.simulation object (already written to disk).

    Args:
        obj_path: Path to DART-convention OBJ file.
        mapping_json_path: Path to DART mapping JSON.
        simu_name: DART simulation name (can contain '/' for subdirs).
        sun_zenith: Sun zenith angle in degrees.
        sun_azimuth: Sun azimuth angle in degrees.
        prospect_params: Dict with PROSPECT parameters for leaf optics.
        scene_size: [x, y] scene size in meters.
        grid_info: Dict with grid_nx, grid_ny, n_plants, center_plant_idx,
                   positions_m, n_groups_per_plant, field_filename.
        par_bands: List of (center_wvl_um, bandwidth_um) tuples.
                   Defaults to PAR_BANDS.
        field_filename: Name of field position file.

    Returns:
        ptd.simulation object.
    """
    if par_bands is None:
        par_bands = PAR_BANDS

    # Clean up previous
    simu_dir = Path(ptd.getdartdir()) / 'user_data' / 'simulations' / simu_name
    if simu_dir.exists():
        shutil.rmtree(str(simu_dir))

    simu = ptd.simulation(simu_name, empty=True)
    simu.scene.size = list(scene_size)

    # Spectral bands
    for wvl, bw in par_bands:
        simu.add.band(wvl=wvl, bw=bw)

    # Sun direction
    simu.core.directions.Directions.SunViewingAngles.sunViewingZenithAngle = sun_zenith
    simu.core.directions.Directions.SunViewingAngles.sunViewingAzimuthAngle = sun_azimuth

    # Ground optical property
    simu.add.optical_property(
        type='Lambertian', ident='ground',
        databaseName='Lambertian_mineral.db', ModelName='clay_brown',
        useMultiplicativeFactorForLUT=0,
    )
    simu.scene.ground.OpticalPropertyLink.ident = 'ground'

    # Leaf optical property (PROSPECT)
    simu.add.optical_property(
        type='Lambertian', ident='maize_leaf',
        prospect=prospect_params,
        useMultiplicativeFactorForLUT=0,
    )
    stem_prospect = get_stem_prospect_params(SIMULATION_DAYS)
    simu.add.optical_property(
        type='Lambertian', ident='maize_stem',
        prospect=stem_prospect,
        useMultiplicativeFactorForLUT=0,
    )
    tassel_prospect = get_tassel_prospect_params(SIMULATION_DAYS)
    simu.add.optical_property(
        type='Lambertian', ident='maize_tassel',
        prospect=tassel_prospect,
        useMultiplicativeFactorForLUT=0,
    )

    # 3D Object via ObjectFields
    file_src_fullpath = simu.get_input_file_path(str(obj_path))
    obj_info = ptd.OBJtools.objreader(file_src_fullpath)
    gnames = ptd.OBJtools.gnames_dart_order(obj_info.names)
    xdim, ydim, zdim = obj_info.dims
    xc, yc, zc = obj_info.center

    # Midrib OP — registered on demand, mirrors site-1 dispatch.
    _has_midrib = any(g.endswith('_midrib') for g in gnames)
    if _has_midrib:
        midrib_prospect = get_midrib_prospect_params(SIMULATION_DAYS)
        simu.add.optical_property(
            type='Lambertian', ident='maize_leaf_midrib',
            prospect=midrib_prospect,
            useMultiplicativeFactorForLUT=0,
        )

    # Senescent-leaf OP — registered on demand, see site 1 note.
    _has_senescent = any(g.startswith('senescent_leaf_') for g in gnames)
    if _has_senescent:
        senescent_prospect = get_senescent_leaf_prospect_params(SIMULATION_DAYS)
        simu.add.optical_property(
            type='Lambertian', ident='maize_leaf_senescent',
            prospect=senescent_prospect,
            useMultiplicativeFactorForLUT=0,
        )

    groups_list = []
    for gi, gname in enumerate(gnames):
        g = ptd.object_3d.create_Group(num=gi + 1, name=gname)
        if gname.endswith('_midrib'):
            # Adaxial-only stripe — see ``run_simulation`` for the trade-off.
            op_ident, df = 'maize_leaf_midrib', 0
        elif gname.startswith(('tassel_spike_', 'tassel_branch_')):
            op_ident, df = 'maize_tassel', 1
        elif gname.startswith('senescent_leaf_'):
            op_ident, df = 'maize_leaf_senescent', 1
        elif gname.endswith('_00'):
            op_ident, df = 'maize_stem', 0
        else:
            op_ident, df = 'maize_leaf', 1
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
        file_src=str(obj_path),
        hasGroups=1,
        GeometricProperties=geom,
        Groups=groups,
        num=0,
        name='CPlantBox_Maize',
        objectDEMMode=0,
    )

    model_list = ptd.object_3d.create_ModelList()
    model_list.add_Object(model_obj)
    field = ptd.object_3d.create_Field(
        name='MaizeField',
        fieldDescriptionFileName=field_filename,
    )
    field.set_ModelList(model_list)
    obj_fields = ptd.object_3d.create_ObjectFields()
    obj_fields.add_Field(field)
    simu.core.object_3d.object_3d.ObjectFields = obj_fields

    # Radiative budget products
    products = simu.core.phase.Phase.DartProduct.dartModuleProducts.CommonProducts
    products.radiativeBudgetProducts = 1
    products.radiativeBudgetProperties.budget3DParSurface = 1

    # Engine: Lux + sampling
    simu.core.phase.Phase.accelerationEngine = 2
    lux = simu.core.phase.Phase.EngineParameter.LuxCoreRenderEngineParameters
    lux.targetRayDensityPerPixel = _cfg.DART_RAY_DENSITY_PER_PIXEL
    lux.maximumRenderingTime = _cfg.DART_MAX_RENDERING_TIME
    simu.core.phase.Phase.ExpertModeZone.nbThreads = _cfg.DART_THREADS

    # Atmosphere: MIDLATSUM
    configure_atmosphere_midlatsum(simu)

    # Write simulation
    simu.write(overwrite=True)

    # Write field file
    simu_path = Path(str(simu.simu_dir))
    field_path = simu_path / 'input' / field_filename
    with open(field_path, 'w') as f:
        f.write('complete transformation\n')
        for pos in grid_info['positions_m']:
            x, y = pos[0], pos[1]
            yrot = pos[2] if len(pos) > 2 else 0.0
            f.write(f'0 {x:.6f} {y:.6f} 0.0 1.0 1.0 1.0 0.0 {yrot:.2f} 0.0\n')

    return simu


def run_dart_full(simu, timeout=600):
    """Run all DART stages (direction + phase + maket + dart).

    Returns True if output was generated.
    """
    try:
        result = simu.run.full(timeout=timeout)
        if result:
            return True
    except Exception as e:
        print(f"  Full run failed: {e}")

    # Fallback: individual stages
    for name, runner in [
        ('direction', simu.run.direction),
        ('phase', simu.run.phase),
        ('maket', simu.run.maket),
        ('dart', simu.run.dart),
    ]:
        try:
            runner(timeout=300)
        except Exception as e:
            print(f"  {name} crashed: {e}")

    simu_path = Path(str(simu.simu_dir))
    return bool(list(simu_path.glob('output/BAND*')))


def update_sun_and_rerun(simu, sun_zenith, sun_azimuth, timeout=300):
    """Update sun angles in a DART simulation and re-run RT (skip maket).

    Geometry is unchanged, so maket can be skipped.  Only direction + phase +
    dart need to re-run.

    Args:
        simu: ptd.simulation object (already written + maket'd).
        sun_zenith: New sun zenith angle (degrees).
        sun_azimuth: New sun azimuth angle (degrees).
        timeout: Per-stage timeout in seconds.

    Returns:
        True if DART output was generated.
    """
    import xml.etree.ElementTree as ET

    simu_path = Path(str(simu.simu_dir))
    directions_xml = simu_path / 'input' / 'directions.xml'

    if directions_xml.exists():
        tree = ET.parse(str(directions_xml))
        root = tree.getroot()
        # Find SunViewingAngles element
        for elem in root.iter('SunViewingAngles'):
            elem.set('sunViewingZenithAngle', f'{sun_zenith:.6f}')
            elem.set('sunViewingAzimuthAngle', f'{sun_azimuth:.6f}')
        tree.write(str(directions_xml), xml_declaration=True, encoding='unicode')

    # Re-run direction + phase + dart (skip maket — geometry unchanged)
    for name, runner in [
        ('direction', simu.run.direction),
        ('phase', simu.run.phase),
        ('dart', simu.run.dart),
    ]:
        try:
            ok = runner(timeout=timeout)
            if not ok:
                print(f"  WARNING: {name} returned False")
        except Exception as e:
            print(f"  {name} error: {e}")

    return bool(list(simu_path.glob('output/BAND*')))


def update_datetime_and_rerun(simu, calendar_date, hour_utc, minute_utc,
                               timeout=600):
    """Update ExactDateHour in directions.xml and re-run RT (skip maket).

    For use with exactDate=1 mode. Edits the ExactDateHour XML element
    instead of SunViewingAngles, then re-runs direction+phase+dart.

    Args:
        simu: ptd.simulation object (already written + maket'd).
        calendar_date: datetime.date with year/month/day.
        hour_utc: Hour in UTC (int).
        minute_utc: Minute (int).
        timeout: Per-stage timeout in seconds.

    Returns:
        True if DART output was generated.
    """
    import xml.etree.ElementTree as ET

    simu_path = Path(str(simu.simu_dir))
    directions_xml = simu_path / 'input' / 'directions.xml'

    if directions_xml.exists():
        tree = ET.parse(str(directions_xml))
        root = tree.getroot()
        for elem in root.iter('ExactDateHour'):
            elem.set('year', str(calendar_date.year))
            elem.set('month', str(calendar_date.month))
            elem.set('day', str(calendar_date.day))
            elem.set('hour', str(int(hour_utc)))
            elem.set('minute', str(int(minute_utc)))
            elem.set('second', '0')
        tree.write(str(directions_xml), xml_declaration=True, encoding='unicode')

    # Re-run direction + phase + dart (skip maket — geometry unchanged)
    for name, runner in [
        ('direction', simu.run.direction),
        ('phase', simu.run.phase),
        ('dart', simu.run.dart),
    ]:
        try:
            ok = runner(timeout=timeout)
            if not ok:
                print(f"  WARNING: {name} returned False")
        except Exception as e:
            print(f"  {name} error: {e}")

    return bool(list(simu_path.glob('output/BAND*')))


def read_ori_reindex(simu, dart_obj_path):
    """Read .ori reindex tables from a DART simulation.

    Returns:
        dict with keys: dart_to_obj, obj_to_dart, groups_sorted,
        group_offsets, ori_data, group_face_counts.
        Returns None on failure.
    """
    simu_path = Path(str(simu.simu_dir))
    ori_dir = simu_path / 'input' / 'triangles'
    if not ori_dir.exists():
        return None

    # Parse OBJ for group structure
    group_offsets = {}
    group_face_counts = {}
    current_group = None
    total_faces = 0
    with open(dart_obj_path) as f:
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

    # Read .ori files
    ori_data = {}
    for gi in range(len(groups_sorted)):
        ori_path = ori_dir / f'triangle{gi}.ori'
        if ori_path.exists():
            ori_data[gi] = np.fromfile(str(ori_path), dtype='uint32')

    if not ori_data:
        return None

    # Build reindex tables
    dart_to_obj = {}
    for gi in sorted(ori_data.keys()):
        offset = group_offsets[groups_sorted[gi]]
        dart_to_obj[gi] = ori_data[gi].astype(np.int64) + offset

    obj_to_dart = {}
    for gi in sorted(dart_to_obj.keys()):
        for dart_pos, global_obj_idx in enumerate(dart_to_obj[gi]):
            obj_to_dart[int(global_obj_idx)] = (gi, dart_pos)

    return {
        'dart_to_obj': dart_to_obj,
        'obj_to_dart': obj_to_dart,
        'groups_sorted': groups_sorted,
        'group_offsets': group_offsets,
        'ori_data': ori_data,
        'group_face_counts': group_face_counts,
    }


def read_and_aggregate_apar(simu, mapping, reindex_info,
                             center_plant_idx=4, par_bands=None):
    """Read DART radiative budget and aggregate to per-segment aPAR array.

    Combines step6 + step7 logic. Returns a 1-D numpy array of aPAR values
    with one entry per leaf segment (in CSV/organ creation order).

    Args:
        simu: ptd.simulation object (after DART run).
        mapping: Parsed mapping JSON dict (with 'organs').
        reindex_info: Dict from read_ori_reindex().
        center_plant_idx: Which plant instance to extract (for multi-plant).
        par_bands: List of (wvl, bw) tuples for band info. Defaults to PAR_BANDS.

    Returns:
        np.ndarray of per-leaf-segment total aPAR (W/m²), or None on failure.
    """
    if par_bands is None:
        par_bands = PAR_BANDS

    # Read radiative budget
    simu_path = Path(str(simu.simu_dir))
    output_path = simu_path / 'output'
    per_band_data = {}

    band_dirs = sorted(output_path.glob('BAND*'))
    for band_dir in band_dirs:
        band_idx = int(re.search(r'BAND(\d+)', band_dir.name).group(1))
        rb_file = band_dir / 'RADIATIVE_BUDGET' / 'ITERX' / 'RadiativeBudgetFigures.txt'
        if rb_file.exists():
            data = parse_radiative_budget_txt(rb_file)
            if data is not None:
                per_band_data[band_idx] = data

    if not per_band_data:
        # Try NetCDF fallback
        netcdf_dir = output_path / 'netcdf' / 'radiativeBudget'
        if netcdf_dir.exists():
            nc_files = list(netcdf_dir.glob('*.nc'))
            if nc_files:
                per_band_data = _parse_radiative_budget_netcdf(nc_files, len(par_bands))

    if not per_band_data:
        return None

    ori_data = reindex_info['ori_data']
    obj_to_dart = reindex_info['obj_to_dart']
    groups_sorted = reindex_info['groups_sorted']

    # Build budget-to-ORI mapping
    maket_mapping = parse_maket_scn(simu_path)
    budget_to_ori = _build_budget_to_ori_mapping(
        per_band_data, ori_data,
        maket_mapping=maket_mapping,
        center_plant_idx=center_plant_idx,
    )

    # Find absorbed column
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

    # Build per-band absorbed arrays
    band_indices = sorted(per_band_data.keys())
    per_band_absorbed = {}
    for band_idx, band_data in per_band_data.items():
        per_band_absorbed[band_idx] = {}
        for obj_key, arr in band_data['per_object'].items():
            m = re.search(r'object(\d+)', obj_key)
            if m:
                budget_idx = int(m.group(1))
                if budget_idx not in budget_to_ori:
                    continue
                gi = budget_to_ori[budget_idx]
                if arr.ndim == 2 and arr.shape[1] > abs_col_idx:
                    front = arr[:, abs_col_idx]
                    if abs_back_col_idx is not None and arr.shape[1] > abs_back_col_idx:
                        per_band_absorbed[band_idx][gi] = front + arr[:, abs_back_col_idx]
                    else:
                        per_band_absorbed[band_idx][gi] = front

    # Aggregate to segments
    segment_apar = []
    for organ in mapping['organs']:
        if organ['type'] != 'leaf':
            continue
        for seg in organ['segments']:
            tri_indices = seg['triangle_indices']
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

            segment_apar.append(sum(band_means.values()))

    return np.array(segment_apar)


# ============================================================================
# Multi-plant API (9 unique realizations)
# ============================================================================

def create_dart_simulation_multi(obj_paths, mapping_json_paths, simu_name,
                                  sun_zenith=None, sun_azimuth=None,
                                  prospect_params=None, scene_size=None,
                                  grid_info=None,
                                  par_bands=None,
                                  field_filename='plant_field.txt',
                                  calendar_date=None, hour_utc=None,
                                  minute_utc=None, lat=50.92, lon=6.36,
                                  use_exact_date=True):
    """Create DART simulation with multiple unique plant models.

    Each OBJ becomes a separate model in ObjectFields; the field file maps
    ``model_index`` 0..N-1 to grid positions (one instance per model).

    Args:
        obj_paths: List of DART-convention OBJ paths (one per plant).
        mapping_json_paths: List of mapping JSON paths (one per plant).
        simu_name: DART simulation name.
        sun_zenith / sun_azimuth: Sun angles (degrees). Used when
            use_exact_date=False (backward compat).
        prospect_params: Dict with PROSPECT parameters.
        scene_size: [x, y] in meters.
        grid_info: Dict with positions_m, etc.
        par_bands: List of (center_wvl_um, bw_um) tuples.
        field_filename: Name of the field position file.
        calendar_date: datetime.date for exactDate mode.
        hour_utc: Hour in UTC (int) for exactDate mode.
        minute_utc: Minute (int) for exactDate mode.
        lat: Latitude (default: Juelich 50.92).
        lon: Longitude (default: Juelich 6.36).
        use_exact_date: If True (default), use DART's built-in solar
            geometry via exactDate=1. Requires calendar_date/hour/minute.

    Returns:
        ptd.simulation object (written to disk).
    """
    if par_bands is None:
        par_bands = PAR_BANDS

    n_plants = len(obj_paths)

    # Clean up previous
    simu_dir = Path(ptd.getdartdir()) / 'user_data' / 'simulations' / simu_name
    if simu_dir.exists():
        shutil.rmtree(str(simu_dir))

    simu = ptd.simulation(simu_name, empty=True)
    simu.scene.size = list(scene_size)

    # Spectral bands
    for wvl, bw in par_bands:
        simu.add.band(wvl=wvl, bw=bw)

    # Sun direction
    if use_exact_date and calendar_date is not None:
        configure_exact_date(simu, calendar_date, hour_utc, minute_utc,
                             lat=lat, lon=lon)
    else:
        simu.core.directions.Directions.SunViewingAngles.sunViewingZenithAngle = sun_zenith
        simu.core.directions.Directions.SunViewingAngles.sunViewingAzimuthAngle = sun_azimuth

    # Ground OP
    simu.add.optical_property(
        type='Lambertian', ident='ground',
        databaseName='Lambertian_mineral.db', ModelName='clay_brown',
        useMultiplicativeFactorForLUT=0,
    )
    simu.scene.ground.OpticalPropertyLink.ident = 'ground'

    # Leaf OP (PROSPECT)
    simu.add.optical_property(
        type='Lambertian', ident='maize_leaf',
        prospect=prospect_params,
        useMultiplicativeFactorForLUT=0,
    )
    stem_prospect = get_stem_prospect_params(SIMULATION_DAYS)
    simu.add.optical_property(
        type='Lambertian', ident='maize_stem',
        prospect=stem_prospect,
        useMultiplicativeFactorForLUT=0,
    )
    tassel_prospect = get_tassel_prospect_params(SIMULATION_DAYS)
    simu.add.optical_property(
        type='Lambertian', ident='maize_tassel',
        prospect=tassel_prospect,
        useMultiplicativeFactorForLUT=0,
    )

    # Build multi-model ObjectFields (one model per plant).
    # Scan the plant meshes first so the senescent OP is registered once
    # per simulation (not per plant) if any of them carry senescent groups.
    _has_senescent = False
    for dart_obj in obj_paths:
        _fs = simu.get_input_file_path(str(dart_obj))
        _info = ptd.OBJtools.objreader(_fs)
        if any(g.startswith('senescent_leaf_')
               for g in ptd.OBJtools.gnames_dart_order(_info.names)):
            _has_senescent = True
            break
    if _has_senescent:
        senescent_prospect = get_senescent_leaf_prospect_params(SIMULATION_DAYS)
        simu.add.optical_property(
            type='Lambertian', ident='maize_leaf_senescent',
            prospect=senescent_prospect,
            useMultiplicativeFactorForLUT=0,
        )

    model_list = ptd.object_3d.create_ModelList()

    for i, dart_obj in enumerate(obj_paths):
        file_src_fullpath = simu.get_input_file_path(str(dart_obj))
        obj_info = ptd.OBJtools.objreader(file_src_fullpath)
        gnames = ptd.OBJtools.gnames_dart_order(obj_info.names)
        xdim, ydim, zdim = obj_info.dims
        xc, yc, zc = obj_info.center

        groups_list = []
        for gi, gname in enumerate(gnames):
            g = ptd.object_3d.create_Group(num=gi + 1, name=gname)
            if gname.startswith(('tassel_spike_', 'tassel_branch_')):
                op_ident, df = 'maize_tassel', 1
            elif gname.startswith('senescent_leaf_'):
                op_ident, df = 'maize_leaf_senescent', 1
            elif gname.endswith('_00'):
                op_ident, df = 'maize_stem', 0
            else:
                op_ident, df = 'maize_leaf', 1
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

    # ObjectFields with field file
    field = ptd.object_3d.create_Field(
        name='MaizeField',
        fieldDescriptionFileName=field_filename,
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
    lux.targetRayDensityPerPixel = _cfg.DART_RAY_DENSITY_PER_PIXEL
    lux.maximumRenderingTime = _cfg.DART_MAX_RENDERING_TIME
    simu.core.phase.Phase.ExpertModeZone.nbThreads = _cfg.DART_THREADS

    # Atmosphere: MIDLATSUM
    configure_atmosphere_midlatsum(simu)

    # Write simulation
    simu.write(overwrite=True)

    # Write field file: one model per position (model_index = position index)
    simu_path = Path(str(simu.simu_dir))
    field_path = simu_path / 'input' / field_filename
    with open(field_path, 'w') as f_out:
        f_out.write('complete transformation\n')
        for idx, pos in enumerate(grid_info['positions_m']):
            x, y = pos[0], pos[1]
            yrot = pos[2] if len(pos) > 2 else 0.0
            f_out.write(f'{idx} {x:.6f} {y:.6f} 0.0 1.0 1.0 1.0 0.0 {yrot:.2f} 0.0\n')

    print(f"  Multi-plant DART simulation: {simu_name} ({n_plants} models)")
    return simu


def read_ori_reindex_multi(simu, dart_obj_paths):
    """Read .ori reindex tables for a multi-model DART simulation.

    With multi-model ObjectFields, .ori files are numbered sequentially
    across models: plant 0 gets groups 0..G0-1, plant 1 gets G0..G0+G1-1, etc.

    Args:
        simu: ptd.simulation object (after DART run).
        dart_obj_paths: List of DART OBJ paths (one per plant).

    Returns:
        List of per-plant reindex dicts (same structure as read_ori_reindex()),
        or None on failure.
    """
    simu_path = Path(str(simu.simu_dir))
    ori_dir = simu_path / 'input' / 'triangles'
    if not ori_dir.exists():
        return None

    # Read all .ori files
    all_ori_data = {}
    for ori_file in sorted(ori_dir.glob('triangle*.ori')):
        gi = int(re.search(r'triangle(\d+)', ori_file.name).group(1))
        all_ori_data[gi] = np.fromfile(str(ori_file), dtype='uint32')

    if not all_ori_data:
        return None

    # Build per-plant reindex info
    results = []
    total_global_groups = 0

    for pi, dart_obj in enumerate(dart_obj_paths):
        # Parse OBJ for group structure
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
        n_groups = len(groups_sorted)

        # Assign .ori files for this plant
        ori_data = {}
        for local_gi in range(n_groups):
            global_gi = total_global_groups + local_gi
            if global_gi in all_ori_data:
                ori_data[local_gi] = all_ori_data[global_gi]

        total_global_groups += n_groups

        # Build reindex tables
        dart_to_obj = {}
        for gi in sorted(ori_data.keys()):
            offset = group_offsets[groups_sorted[gi]]
            dart_to_obj[gi] = ori_data[gi].astype(np.int64) + offset

        obj_to_dart = {}
        for gi in sorted(dart_to_obj.keys()):
            for dart_pos, global_obj_idx in enumerate(dart_to_obj[gi]):
                obj_to_dart[int(global_obj_idx)] = (gi, dart_pos)

        results.append({
            'dart_to_obj': dart_to_obj,
            'obj_to_dart': obj_to_dart,
            'groups_sorted': groups_sorted,
            'group_offsets': group_offsets,
            'ori_data': ori_data,
            'group_face_counts': group_face_counts,
        })

    print(f"  Read .ori for {len(results)} plants "
          f"({total_global_groups} total groups)")
    return results


def read_and_aggregate_apar_multi(simu, mappings, reindex_infos, par_bands=None):
    """Read DART budget and aggregate per-segment aPAR for ALL plants.

    Uses maket.scn to map budget objects to per-plant groups (same approach
    as multifield.py step5).

    Args:
        simu: ptd.simulation object (after DART run).
        mappings: List of mapping JSON dicts (one per plant).
        reindex_infos: List of per-plant reindex dicts from read_ori_reindex_multi().
        par_bands: PAR band definitions.

    Returns:
        List of np.ndarray (one per-leaf-segment aPAR array per plant),
        or None on failure.
    """
    if par_bands is None:
        par_bands = PAR_BANDS

    n_plants = len(mappings)

    # Read radiative budget
    simu_path = Path(str(simu.simu_dir))
    output_path = simu_path / 'output'
    per_band_data = {}

    band_dirs = sorted(output_path.glob('BAND*'))
    for band_dir in band_dirs:
        band_idx = int(re.search(r'BAND(\d+)', band_dir.name).group(1))
        rb_file = band_dir / 'RADIATIVE_BUDGET' / 'ITERX' / 'RadiativeBudgetFigures.txt'
        if rb_file.exists():
            data = parse_radiative_budget_txt(rb_file)
            if data is not None:
                per_band_data[band_idx] = data

    if not per_band_data:
        return None

    # Parse maket.scn for budget → (instance, group) mapping
    maket_mapping = parse_maket_scn(simu_path)
    if maket_mapping is None:
        print("  WARNING: No maket.scn mapping — cannot extract per-plant aPAR")
        return None

    # Build per-plant budget mapping: {plant_idx: {budget_idx: local_group_idx}}
    per_plant_budget = {}
    for budget_idx, (instance, group) in maket_mapping.items():
        if instance not in per_plant_budget:
            per_plant_budget[instance] = {}
        per_plant_budget[instance][budget_idx] = group

    # Find absorbed columns
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

    band_indices = sorted(per_band_data.keys())

    # Aggregate per-plant
    all_plant_apar = []
    for pi in range(n_plants):
        mapping = mappings[pi]
        plant_budget = per_plant_budget.get(pi, {})
        reindex = reindex_infos[pi]
        obj_to_dart = reindex['obj_to_dart']

        # Build per-band absorbed arrays for this plant
        per_band_absorbed = {}
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
                            if (abs_back_col_idx is not None
                                    and arr.shape[1] > abs_back_col_idx):
                                per_band_absorbed[band_idx][local_gi] = (
                                    front + arr[:, abs_back_col_idx])
                            else:
                                per_band_absorbed[band_idx][local_gi] = front

        # Aggregate to leaf segments
        segment_apar = []
        for organ in mapping['organs']:
            if organ['type'] != 'leaf':
                continue
            for seg in organ['segments']:
                tri_indices = seg['triangle_indices']
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
                segment_apar.append(sum(band_means.values()))

        all_plant_apar.append(np.array(segment_apar))

    return all_plant_apar


# ============================================================================
# Main
# ============================================================================
def main():
    print("Phase 1: Standalone DART Simulation from CPlantBox OBJ")
    print("=" * 70)
    log_consistency(SIMULATION_DAYS)

    # Step 1
    plant, mesh, mapping = step1_grow_and_export()

    # Step 2
    stats = step2_convert_to_dart()

    # Load DART-convention mapping for Steps 5-7
    with open(OUTPUT_DIR / 'maize_day55_dart_mapping.json') as f:
        dart_mapping = json.load(f)

    # Step 3
    simu = step3_create_dart_simulation()

    # Step 4
    dart_ok = step4_run_dart(simu)

    # Step 5
    step5_result = step5_read_ori(simu, dart_mapping)
    if step5_result is None:
        print("\nERROR: Cannot proceed without .ori reindex tables!")
        return

    # Step 6
    per_band_data = step6_read_radiative_budget(simu)

    # Step 7
    segment_results = step7_aggregate_to_segments(per_band_data, step5_result,
                                                   dart_mapping)

    # Step 8
    success = step8_summary_and_output(segment_results, per_band_data,
                                        dart_mapping)

    print(f"\nOutput files:")
    for f in sorted(OUTPUT_DIR.glob('maize_day55*')):
        size = f.stat().st_size
        print(f"  {f.name} ({size:,} bytes)")


if __name__ == '__main__':
    main()
