#!/usr/bin/env python3
"""
Baleno Energy Balance — Production API.

Provides reusable functions for the diurnal coupling pipeline:
  - setup_baleno_full / setup_baleno_full_multi: create DART _I/_II + Baleno configs
  - write_baleno_config_files / restore_config_files: manage .dartrc + INI files
  - run_baleno_subprocess: run Baleno as subprocess
  - read_baleno_tleaf / read_baleno_tleaf_multi: read per-segment Tleaf
  - update_baleno_atmosphere / update_baleno_datetime_and_rerun_I: timestep updates
  - run_baleno_with_external_gs: ExternalGS one-shot wrapper

Legacy standalone demo (steps 1-9) is in baleno_standalone.py.
"""

import os
import json
import re
import shutil
import subprocess
import textwrap
import xml.etree.ElementTree as ET
import numpy as np
from pathlib import Path
import pytools4dart as ptd

from .. import config as _cfg
from ..config import (DART_HOME, DART_EB_DIR, DARTRC, BALENO_PYTHON,
                      get_species)
from ..prospect_params import (get_prospect_params, get_prospect_params_per_position,
                               get_stem_prospect_params, vcmax25_from_cab)
from ..dart.simulation import configure_atmosphere_midlatsum
from .parsers import detect_delimiter, write_json5

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BALENO_DIR = DART_EB_DIR
DART_DIR = DART_HOME
DART_LOCAL = DART_DIR / 'user_data'
DARTRC_PATH = DARTRC
VENV_PYTHON = BALENO_PYTHON

SIMU_NAME_EB = 'cpb_maize_eb'            # Baleno simulation name
DART_SIMU_NAME = 'cpb_maize_day55_eb'    # DART simulation container name

# Baleno has TWO user_data directories:
# 1. BALENO_DIR/user_data — Baleno's own (JSON5 configs, EB outputs)
# 2. DART_LOCAL (= DART_HOME/user_data) — DART simulations (_I, _II subdirs)
#
# config.ini user_data_path MUST be empty so check_existing_simulation()
# resolves to BALENO_DIR/user_data where the JSON5 input files live.
# DARTPathManager separately reads DART_LOCAL from .dartrc for DART outputs.
BALENO_USER_DATA = BALENO_DIR / 'user_data'

# Shortwave bands: 21 bands from 400-2500 nm (100 nm each)
# Band edges: [0.400, 0.500, 0.600, ..., 2.500] µm
# Center wavelength = edge + bandwidth/2 (e.g., 0.450 for [0.400, 0.500])
SW_BANDS = [(0.400 + i * 0.100 + 0.050, 0.100) for i in range(21)]

# Thermal band: single band at 10 µm (8-12 µm window)
TIR_BAND = (10.0, 4.0)  # center 10 µm, bandwidth 4 µm


# ============================================================================
# Config file management (shared by production pipeline and standalone demo)
# ============================================================================

def write_baleno_config_files():
    """Write Baleno config.ini and DART dart_config.ini.

    Also fixes the .dartrc to have correct absolute paths for DART_LOCAL.
    Returns backup paths for restoration.
    """
    print("\n" + "=" * 70)
    print("STEP 4: Write Config Files")
    print("=" * 70)

    backups = {}

    # --- Fix .dartrc ---
    # The existing .dartrc has relative paths (/home/user_data) that don't
    # resolve correctly. Write a corrected version with absolute paths.
    dartrc_bak = DARTRC_PATH.with_suffix('.bak_baleno')
    if DARTRC_PATH.exists():
        shutil.copy2(str(DARTRC_PATH), str(dartrc_bak))
        backups['dartrc'] = (str(DARTRC_PATH), str(dartrc_bak))
        print(f"  Backed up .dartrc to {dartrc_bak}")

    corrected_dartrc = textwrap.dedent(f"""\
        #!/bin/bash
        # DART Configuration File (corrected for Baleno)
        export DART_HOME={DART_DIR}
        export DART_LOCAL={DART_LOCAL}
        export DART_PYTHON_PATH={DART_DIR}/bin/python
        export DART_JAVA_MAX_MEMORY=4g
        export PATH=$DART_PYTHON_PATH:$DART_PYTHON_PATH/bin:$DART_HOME/bin/jre/bin:$DART_HOME/bin:$DART_HOME/bin/hapke:$DART_HOME/bin/prospect:$PATH
        export LD_LIBRARY_PATH=$DART_HOME/bin:$LD_LIBRARY_PATH
    """)
    DARTRC_PATH.write_text(corrected_dartrc)
    print(f"  Fixed .dartrc: DART_LOCAL={DART_LOCAL}")

    # --- Baleno config.ini ---
    baleno_config_path = BALENO_DIR / 'resources' / 'config.ini'
    baleno_config_bak = baleno_config_path.with_suffix('.ini.bak_baleno')
    if baleno_config_path.exists():
        shutil.copy2(str(baleno_config_path), str(baleno_config_bak))
        backups['baleno_config'] = (str(baleno_config_path), str(baleno_config_bak))
        print(f"  Backed up config.ini")
    else:
        # File doesn't exist — mark for deletion on restore
        backups['baleno_config'] = (str(baleno_config_path), None)
        print(f"  config.ini did not exist (will be cleaned up)")

    baleno_config_content = textwrap.dedent(f"""\
        [simulation]
        user_data_path =
        name = {SIMU_NAME_EB}
    """)
    baleno_config_path.write_text(baleno_config_content)
    print(f"  Wrote config.ini: user_data_path=(empty -> BALENO_DIR/user_data), "
          f"name={SIMU_NAME_EB}")

    # --- DART plugin dart_config.ini ---
    dart_config_path = BALENO_DIR / 'plugins' / 'DART' / 'resources' / 'dart_config.ini'
    dart_config_bak = dart_config_path.with_suffix('.ini.bak_baleno')
    if dart_config_path.exists():
        shutil.copy2(str(dart_config_path), str(dart_config_bak))
    else:
        # No existing file — check for .example
        example = dart_config_path.with_suffix('.example.ini')
        if example.exists():
            shutil.copy2(str(example), str(dart_config_bak))
    backups['dart_config'] = (str(dart_config_path), str(dart_config_bak))

    dart_config_content = textwrap.dedent(f"""\
        [paths]
        dart_path = {DART_DIR}
    """)
    dart_config_path.write_text(dart_config_content)
    print(f"  Wrote dart_config.ini: dart_path={DART_DIR}")

    # --- Patch is_scene_sorted() in Baleno ---
    # Baleno bug: is_scene_sorted() expects INDEX_OBJECT to be exactly [0,1,2,...,N-1]
    # per DART_NAME group, but DART's maket step removes degenerate triangles, creating
    # gaps (e.g., [0,1,2,...,3398,3400,3402,3404,3406]). The actual data mapping uses
    # DART_NAME (not INDEX_OBJECT), so relaxing to monotonic-increasing is safe.
    lux_reader_path = BALENO_DIR / 'plugins' / 'DART' / 'IO' / 'output_readers' / 'lux_output_reader.py'
    lux_reader_bak = lux_reader_path.with_suffix('.py.bak_baleno')
    if lux_reader_path.exists():
        shutil.copy2(str(lux_reader_path), str(lux_reader_bak))
        backups['lux_output_reader'] = (str(lux_reader_path), str(lux_reader_bak))

        content = lux_reader_path.read_text()
        # Replace the strict sequential check with monotonic-increasing check
        old_check = 'is_valid = np.all(col == np.arange(len(col)))'
        new_check = 'is_valid = np.all(np.diff(col.astype(float)) >= 0)  # monotonic (patched for OBJ with gaps)'
        if old_check in content:
            content = content.replace(old_check, new_check)
            lux_reader_path.write_text(content)
            print(f"  Patched is_scene_sorted(): relaxed to monotonic check")
        elif new_check not in content:
            print(f"  Note: is_scene_sorted patch target not found (DART version may differ)")

    # --- Patch spectrum_integration.py: band index bug ---
    # Baleno bug: radiative_budget_integration() uses DART band number (from band name)
    # to index into PAR-filtered waveband list. With multi-band simulations, PAR bands
    # have indices > 0 but the filtered list starts at 0.
    # Fix: use enumerate index for wavebands, DART band number for file paths.
    spec_int_path = BALENO_DIR / 'plugins' / 'DART' / 'launch_processes' / 'spectrum_integration.py'
    spec_int_bak = spec_int_path.with_suffix('.py.bak_baleno')
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
            '    for wvb_idx, band_name in enumerate(band_name_list):  # patched: use enumerate for filtered list\n'
            '        dart_band_index = int(band_name[4:])  # DART band number for file paths\n'
            '        short_wl = wavebands[wvb_idx][0] * 1000  # Conversion from µm to nm (use filtered list index)\n'
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
            print(f"  Patched spectrum_integration.py: band index bug fix")
        elif 'enumerate(band_name_list)' not in content:
            print(f"  Note: spectrum_integration band loop patch target not found (DART version may differ)")

    # --- Patch lux_manager.py: missing temperaturematrix fallback ---
    # When _II simulation doesn't have temperaturematrix entries in phase.scn
    # (e.g., DART didn't generate them despite useTemperaturePerTriangle=1),
    # __get_file_from_material() crashes with AttributeError on None.
    # Patch: return a default filename based on dart_name_id when lookup fails.
    lux_mgr_path = BALENO_DIR / 'plugins' / 'DART' / 'IO' / 'FT_Lux_Managers' / 'lux_manager.py'
    lux_mgr_bak = lux_mgr_path.with_suffix('.py.bak_baleno')
    if lux_mgr_path.exists():
        shutil.copy2(str(lux_mgr_path), str(lux_mgr_bak))
        backups['lux_manager'] = (str(lux_mgr_path), str(lux_mgr_bak))

        content = lux_mgr_path.read_text()

        # Patch __get_input_files: when exitance_file is None from both
        # direct lookup and material lookup, generate a default filename
        # rather than crashing.
        old_get_input = (
            '                    if exitance_file is None:\n'
            '                        logger.debug(\n'
            '                            f"No {input_type.name} file set as direct way for object {dart_name_id}. Looking for file from {input_type.name} property definition")\n'
            '                        material_name = value.get(MaketReader.MATERIAL)\n'
            '                        material_dict = scn_reader.get_dict_material()\n'
            '                        exitance_file = self.__get_file_from_material(material_name, material_dict, dart_state, input_type)\n'
            '                    files_dict[dart_name_id] = exitance_file'
        )
        new_get_input = (
            '                    if exitance_file is None:\n'
            '                        logger.debug(\n'
            '                            f"No {input_type.name} file set as direct way for object {dart_name_id}. Looking for file from {input_type.name} property definition")\n'
            '                        material_name = value.get(MaketReader.MATERIAL)\n'
            '                        material_dict = scn_reader.get_dict_material()\n'
            '                        try:\n'
            '                            exitance_file = self.__get_file_from_material(material_name, material_dict, dart_state, input_type)\n'
            '                        except (AttributeError, TypeError, KeyError) as e:\n'
            '                            # Fallback: generate default temp file path (patched for CPlantBox coupling)\n'
            '                            exitance_file = f"temperature_{dart_name_id}.txt"\n'
            '                            logger.warning(f"No {input_type.name} file found via material lookup for {dart_name_id}, using default: {exitance_file}")\n'
            '                    files_dict[dart_name_id] = exitance_file'
        )
        if old_get_input in content:
            content = content.replace(old_get_input, new_get_input)
            lux_mgr_path.write_text(content)
            print(f"  Patched lux_manager.py: fallback for missing temperaturematrix")
        elif 'Fallback: generate default temp file path' not in content:
            print(f"  Note: lux_manager __get_input_files patch target not found (DART version may differ)")

    return backups


def restore_config_files(backups):
    """Restore all backed-up config files."""
    print("\n  Restoring config files...")
    for name, (original, backup) in backups.items():
        if backup is None:
            # File didn't exist before — delete our created version
            if Path(original).exists():
                Path(original).unlink()
                print(f"    Cleaned up: {name}")
        elif Path(backup).exists():
            shutil.copy2(backup, original)
            Path(backup).unlink()
            print(f"    Restored: {name}")



# ============================================================================
# Production API: setup, run, read (used by diurnal.py / iterative.py)
# ============================================================================

def setup_baleno_full(obj_path, mapping_json, reindex_json, grid_info_path,
                      prospect_params, sun_zenith=None, sun_azimuth=None,
                      scene_size=(4, 4), plant_pos=(2.0, 2.0),
                      dart_simu_name='cpb_maize_diurnal_eb',
                      baleno_simu_name='cpb_diurnal_eb',
                      field_filename='plant_field.txt',
                      calendar_date=None, hour_utc=None, minute_utc=None,
                      lat=50.92, lon=6.36, use_exact_date=True):
    """Create _I, _II DART simulations and Baleno config files.

    This is the reusable counterpart of steps 1-4. Returns paths and backup
    info needed for running and cleaning up.

    Args:
        obj_path: Path to DART-convention OBJ.
        mapping_json: Path to DART mapping JSON.
        reindex_json: Path to Phase 1 reindex JSON.
        grid_info_path: Path to grid_info.json.
        prospect_params: PROSPECT parameter dict.
        sun_zenith, sun_azimuth: Initial sun angles. Used when
            use_exact_date=False (backward compat).
        scene_size: Scene size in meters.
        plant_pos: Center plant position.
        dart_simu_name: Name for the DART _I/_II container.
        baleno_simu_name: Name for the Baleno simulation.
        field_filename: Name of the plant field file.
        calendar_date: datetime.date for exactDate mode.
        hour_utc: Hour in UTC (int) for exactDate mode.
        minute_utc: Minute (int) for exactDate mode.
        lat: Latitude (default: Juelich 50.92).
        lon: Longitude (default: Juelich 6.36).
        use_exact_date: If True (default), use DART's built-in solar
            geometry via exactDate=1.

    Returns:
        dict with: simu_I (ptd.simulation), simu_II_dir (Path),
        baleno_sim_dir (Path), backups (dict for restore_config_files),
        dart_simu_name (str), baleno_simu_name (str).
    """
    import json as _json

    # Step 1: Create _I simulation
    simu_I_name = f'{dart_simu_name}/{dart_simu_name}_I'
    parent_dir = DART_LOCAL / 'simulations' / dart_simu_name
    simu_I_dir = parent_dir / f'{dart_simu_name}_I'
    parent_dir.mkdir(parents=True, exist_ok=True)
    if simu_I_dir.exists():
        shutil.rmtree(str(simu_I_dir))

    simu_I = ptd.simulation(simu_I_name, empty=True)
    simu_I.scene.size = list(scene_size)

    # 21 shortwave bands
    for wvl, bw in SW_BANDS:
        simu_I.add.band(wvl=wvl, bw=bw)

    # Sun direction
    if use_exact_date and calendar_date is not None:
        from ..dart.simulation import configure_exact_date
        configure_exact_date(simu_I, calendar_date, hour_utc, minute_utc,
                             lat=lat, lon=lon)
    else:
        simu_I.core.directions.Directions.SunViewingAngles.sunViewingZenithAngle = sun_zenith
        simu_I.core.directions.Directions.SunViewingAngles.sunViewingAzimuthAngle = sun_azimuth

    simu_I.add.optical_property(
        type='Lambertian', ident='ground',
        databaseName='Lambertian_mineral.db', ModelName='clay_brown',
        useMultiplicativeFactorForLUT=0,
    )
    simu_I.scene.ground.OpticalPropertyLink.ident = 'ground'
    simu_I.add.optical_property(
        type='Lambertian', ident='maize_leaf',
        prospect=prospect_params,
        useMultiplicativeFactorForLUT=0,
    )
    stem_prospect = get_stem_prospect_params(55)
    simu_I.add.optical_property(
        type='Lambertian', ident='maize_stem',
        prospect=stem_prospect,
        useMultiplicativeFactorForLUT=0,
    )

    # 3D Object
    file_src_fullpath = simu_I.get_input_file_path(str(obj_path))
    obj_info = ptd.OBJtools.objreader(file_src_fullpath)
    gnames = ptd.OBJtools.gnames_dart_order(obj_info.names)
    xdim, ydim, zdim = obj_info.dims
    xc, yc, zc = obj_info.center

    groups_list = []
    for gi, gname in enumerate(gnames):
        g = ptd.object_3d.create_Group(num=gi + 1, name=gname)
        is_stem = gname.endswith('_00')
        op_ident = 'maize_stem' if is_stem else 'maize_leaf'
        df = 0 if is_stem else 1
        g.set_nodes(ident=op_ident)
        gop = g.GroupOpticalProperties
        gop.SurfaceOpticalProperties.doubleFace = df
        gop.SurfaceExitanceProperties.doubleFace = df
        groups_list.append(g)
    groups = ptd.object_3d.create_Groups(Group=groups_list)

    geom = ptd.object_3d.create_GeometricProperties(
        Dimension3D=ptd.object_3d.create_Dimension3D(xdim=xdim, ydim=ydim, zdim=zdim),
        Center3D=ptd.object_3d.create_Center3D(xCenter=xc, yCenter=yc, zCenter=zc),
        ScaleProperties=ptd.object_3d.create_ScaleProperties(xscale=1.0, yscale=1.0, zscale=1.0),
    )
    model_obj = ptd.object_3d.create_Object(
        file_src=str(obj_path), hasGroups=1, GeometricProperties=geom,
        Groups=groups, num=0, name='CPlantBox_Maize', objectDEMMode=0,
    )

    model_list = ptd.object_3d.create_ModelList()
    model_list.add_Object(model_obj)
    field = ptd.object_3d.create_Field(name='MaizeField', fieldDescriptionFileName=field_filename)
    field.set_ModelList(model_list)
    obj_fields = ptd.object_3d.create_ObjectFields()
    obj_fields.add_Field(field)
    simu_I.core.object_3d.object_3d.ObjectFields = obj_fields

    products = simu_I.core.phase.Phase.DartProduct.dartModuleProducts.CommonProducts
    products.radiativeBudgetProducts = 1
    products.radiativeBudgetProperties.budget3DParSurface = 1
    simu_I.core.phase.Phase.accelerationEngine = 2
    lux = simu_I.core.phase.Phase.EngineParameter.LuxCoreRenderEngineParameters
    lux.targetRayDensityPerPixel = _cfg.DART_RAY_DENSITY_PER_PIXEL
    lux.maximumRenderingTime = _cfg.DART_MAX_RENDERING_TIME
    simu_I.core.phase.Phase.ExpertModeZone.nbThreads = _cfg.DART_THREADS

    # Atmosphere: MIDLATSUM
    configure_atmosphere_midlatsum(simu_I)

    simu_I.write(overwrite=True)

    # Write field file
    simu_I_path = Path(str(simu_I.simu_dir))
    field_path = simu_I_path / 'input' / field_filename
    if Path(grid_info_path).exists():
        with open(grid_info_path) as gf:
            gi = _json.load(gf)
        with open(field_path, 'w') as ff:
            ff.write('complete transformation\n')
            for x, y in gi['positions_m']:
                ff.write(f'0 {x:.6f} {y:.6f} 0.0 1.0 1.0 1.0 0.0 0.0 0.0\n')
    else:
        with open(field_path, 'w') as ff:
            ff.write('complete transformation\n')
            ff.write(f'0 {plant_pos[0]:.6f} {plant_pos[1]:.6f} 0.0 1.0 1.0 1.0 0.0 0.0 0.0\n')

    # Step 2: Create _II (copy + modify)
    simu_II_dir = _create_simu_II(simu_I, dart_simu_name, str(reindex_json))

    # Step 3: Create Baleno configs
    baleno_sim_dir = _create_baleno_configs(baleno_simu_name, dart_simu_name)

    # Step 4: Write config files
    backups = write_baleno_config_files()

    return {
        'simu_I': simu_I,
        'simu_II_dir': simu_II_dir,
        'baleno_sim_dir': baleno_sim_dir,
        'backups': backups,
        'dart_simu_name': dart_simu_name,
        'baleno_simu_name': baleno_simu_name,
    }


def setup_baleno_full_multi(obj_paths, mapping_json_paths, reindex_json_paths,
                             grid_info_path, prospect_params,
                             sun_zenith=None, sun_azimuth=None,
                             scene_size=(4, 4),
                             dart_simu_name='cpb_maize_diurnal_eb',
                             baleno_simu_name='cpb_diurnal_eb',
                             field_filename='plant_field.txt',
                             calendar_date=None, hour_utc=None, minute_utc=None,
                             lat=50.92, lon=6.36, use_exact_date=True):
    """Create _I, _II DART sims and Baleno configs for multi-plant scene.

    Like setup_baleno_full() but with 9 unique OBJ models in ObjectFields,
    matching create_dart_simulation_multi() pattern. A single Baleno run
    computes energy balance for ALL plants simultaneously.

    Args:
        obj_paths: List of DART-convention OBJ paths (one per plant).
        mapping_json_paths: List of mapping JSON paths (one per plant).
        reindex_json_paths: List of reindex JSON paths (one per plant).
        grid_info_path: Path to grid_info.json.
        prospect_params: PROSPECT parameter dict.
        sun_zenith, sun_azimuth: Initial sun angles (backward compat).
        scene_size: Scene size in meters.
        dart_simu_name: Name for the DART _I/_II container.
        baleno_simu_name: Name for the Baleno simulation.
        field_filename: Name of the plant field file.
        calendar_date: datetime.date for exactDate mode.
        hour_utc, minute_utc: UTC time for exactDate mode.
        lat, lon: Location coordinates.
        use_exact_date: If True, use DART's exactDate=1 solar geometry.

    Returns:
        dict with: simu_I, simu_II_dir, baleno_sim_dir, backups,
        dart_simu_name, baleno_simu_name, obj_paths, reindex_json_paths.
    """
    import json as _json

    n_plants = len(obj_paths)

    # Step 1: Create _I simulation with multiple models
    simu_I_name = f'{dart_simu_name}/{dart_simu_name}_I'
    parent_dir = DART_LOCAL / 'simulations' / dart_simu_name
    simu_I_dir = parent_dir / f'{dart_simu_name}_I'
    parent_dir.mkdir(parents=True, exist_ok=True)
    if simu_I_dir.exists():
        shutil.rmtree(str(simu_I_dir))

    simu_I = ptd.simulation(simu_I_name, empty=True)
    simu_I.scene.size = list(scene_size)

    # 21 shortwave bands
    for wvl, bw in SW_BANDS:
        simu_I.add.band(wvl=wvl, bw=bw)

    # Sun direction
    if use_exact_date and calendar_date is not None:
        from ..dart.simulation import configure_exact_date
        configure_exact_date(simu_I, calendar_date, hour_utc, minute_utc,
                             lat=lat, lon=lon)
    else:
        simu_I.core.directions.Directions.SunViewingAngles.sunViewingZenithAngle = sun_zenith
        simu_I.core.directions.Directions.SunViewingAngles.sunViewingAzimuthAngle = sun_azimuth

    # Ground OP
    simu_I.add.optical_property(
        type='Lambertian', ident='ground',
        databaseName='Lambertian_mineral.db', ModelName='clay_brown',
        useMultiplicativeFactorForLUT=0,
    )
    simu_I.scene.ground.OpticalPropertyLink.ident = 'ground'

    # Leaf + stem OPs
    simu_I.add.optical_property(
        type='Lambertian', ident='maize_leaf',
        prospect=prospect_params,
        useMultiplicativeFactorForLUT=0,
    )
    stem_prospect = get_stem_prospect_params(55)
    simu_I.add.optical_property(
        type='Lambertian', ident='maize_stem',
        prospect=stem_prospect,
        useMultiplicativeFactorForLUT=0,
    )

    # Build multi-model ObjectFields (one model per plant)
    model_list = ptd.object_3d.create_ModelList()

    for i, dart_obj in enumerate(obj_paths):
        file_src_fullpath = simu_I.get_input_file_path(str(dart_obj))
        obj_info = ptd.OBJtools.objreader(file_src_fullpath)
        gnames = ptd.OBJtools.gnames_dart_order(obj_info.names)
        xdim, ydim, zdim = obj_info.dims
        xc, yc, zc = obj_info.center

        groups_list = []
        for gi, gname in enumerate(gnames):
            g = ptd.object_3d.create_Group(num=gi + 1, name=gname)
            is_stem = gname.endswith('_00')
            op_ident = 'maize_stem' if is_stem else 'maize_leaf'
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
            file_src=str(dart_obj), hasGroups=1, GeometricProperties=geom,
            Groups=groups, num=i, name=f'CPlantBox_Maize_p{i}',
            objectDEMMode=0,
        )
        model_list.add_Object(model_obj)

    # ObjectFields with field file
    field = ptd.object_3d.create_Field(
        name='MaizeField', fieldDescriptionFileName=field_filename)
    field.set_ModelList(model_list)
    obj_fields = ptd.object_3d.create_ObjectFields()
    obj_fields.add_Field(field)
    simu_I.core.object_3d.object_3d.ObjectFields = obj_fields

    # Radiative budget + LuxCore sampling
    products = simu_I.core.phase.Phase.DartProduct.dartModuleProducts.CommonProducts
    products.radiativeBudgetProducts = 1
    products.radiativeBudgetProperties.budget3DParSurface = 1
    simu_I.core.phase.Phase.accelerationEngine = 2
    lux = simu_I.core.phase.Phase.EngineParameter.LuxCoreRenderEngineParameters
    lux.targetRayDensityPerPixel = _cfg.DART_RAY_DENSITY_PER_PIXEL
    lux.maximumRenderingTime = _cfg.DART_MAX_RENDERING_TIME
    simu_I.core.phase.Phase.ExpertModeZone.nbThreads = _cfg.DART_THREADS

    # Atmosphere: MIDLATSUM
    configure_atmosphere_midlatsum(simu_I)

    simu_I.write(overwrite=True)

    # Write field file: one model per position (model_index = position index)
    simu_I_path = Path(str(simu_I.simu_dir))
    field_path = simu_I_path / 'input' / field_filename
    if Path(grid_info_path).exists():
        with open(grid_info_path) as gf:
            gi = _json.load(gf)
        with open(field_path, 'w') as ff:
            ff.write('complete transformation\n')
            for idx, (x, y) in enumerate(gi['positions_m']):
                ff.write(f'{idx} {x:.6f} {y:.6f} 0.0 1.0 1.0 1.0 0.0 0.0 0.0\n')
    else:
        with open(field_path, 'w') as ff:
            ff.write('complete transformation\n')
            ff.write('0 2.0 2.0 0.0 1.0 1.0 1.0 0.0 0.0 0.0\n')

    # Step 2: Create _II (thermal) — uses combined reindex from all plants
    combined_reindex = _build_combined_reindex(reindex_json_paths)
    combined_reindex_path = simu_I_path.parent / 'combined_reindex.json'
    with open(combined_reindex_path, 'w') as f:
        _json.dump(combined_reindex, f, indent=2)
    simu_II_dir = _create_simu_II(simu_I, dart_simu_name, str(combined_reindex_path))

    # Step 3: Create Baleno configs
    baleno_sim_dir = _create_baleno_configs(baleno_simu_name, dart_simu_name)

    # Step 4: Write config files
    backups = write_baleno_config_files()

    return {
        'simu_I': simu_I,
        'simu_II_dir': simu_II_dir,
        'baleno_sim_dir': baleno_sim_dir,
        'backups': backups,
        'dart_simu_name': dart_simu_name,
        'baleno_simu_name': baleno_simu_name,
        'obj_paths': obj_paths,
        'reindex_json_paths': reindex_json_paths,
    }


def _build_combined_reindex(reindex_json_paths):
    """Combine per-plant reindex JSONs into one for _II temperature files.

    The _II simulation sees all plant groups sequentially numbered.
    Plant 0 gets groups 0..G0-1, plant 1 gets G0..G0+G1-1, etc.
    """
    import json as _json

    combined = {'dart_to_obj': {}, 'group_names': []}
    global_gi = 0
    for pi, rpath in enumerate(reindex_json_paths):
        with open(rpath) as f:
            ri = _json.load(f)
        n_groups = len(ri['group_names'])
        for local_gi in range(n_groups):
            gname = ri['group_names'][local_gi]
            combined['dart_to_obj'][str(global_gi)] = ri['dart_to_obj'][str(local_gi)]
            combined['group_names'].append(gname)
            global_gi += 1

    return combined


def _create_simu_II(simu_I, dart_simu_name, reindex_json_path):
    """Create _II (thermal) simulation from _I."""
    import json as _json

    simu_I_dir = Path(str(simu_I.simu_dir))
    simu_II_dir = simu_I_dir.parent / f'{dart_simu_name}_II'

    if simu_II_dir.exists():
        shutil.rmtree(str(simu_II_dir))
    shutil.copytree(str(simu_I_dir), str(simu_II_dir))

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
                        f'spectralDartMode="0"/>')
                    first_replaced = True
            else:
                new_lines.append(line)
        phase_xml.write_text('\n'.join(new_lines))

    # Modify object_3d.xml: per-triangle temperature
    obj3d_xml = simu_II_dir / 'input' / 'object_3d.xml'
    if obj3d_xml.exists():
        tree = ET.parse(str(obj3d_xml))
        root = tree.getroot()
        for group in root.findall('.//Group'):
            group_name = group.get('name', 'unknown')
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
            tptp.set('triangleTemperatureFile', f'temperature_{group_name}.txt')
        tree.write(str(obj3d_xml), xml_declaration=True, encoding='unicode')

    # Create initial temperature files
    input_dir = simu_II_dir / 'input'
    with open(reindex_json_path) as f:
        reindex = _json.load(f)
    for gi_str, obj_indices in reindex['dart_to_obj'].items():
        gi = int(gi_str)
        group_name = reindex['group_names'][gi] if gi < len(reindex['group_names']) else f'group_{gi}'
        n_tris = len(obj_indices)
        np.savetxt(str(input_dir / f'temperature_{group_name}.txt'),
                   np.full(n_tris, 298.15), fmt='%.2f')

    # Clean output
    output_dir = simu_II_dir / 'output'
    if output_dir.exists():
        shutil.rmtree(str(output_dir))
        output_dir.mkdir()

    return simu_II_dir


def _create_baleno_configs(baleno_simu_name, dart_simu_name):
    """Create Baleno simulation configs (reusable version of step3)."""
    baleno_sim_dir = BALENO_USER_DATA / 'simulations' / baleno_simu_name
    input_dir = baleno_sim_dir / 'input'
    plugins_dir = input_dir / 'plugins'

    if baleno_sim_dir.exists():
        shutil.rmtree(str(baleno_sim_dir))
    input_dir.mkdir(parents=True)
    plugins_dir.mkdir(parents=True)

    write_json5(input_dir / 'atmosphere.json5', {
        "z": 10, "Ta": 298.15, "p": 1013, "ea": 15, "u": 2,
        "Ca": 400, "Oa": 280,
    })
    # Use mean Cab/N across per-position LOPS profiles
    import numpy as _np
    _per_pos = get_prospect_params_per_position(55, 11)
    _mean_cab = float(_np.mean([p["Cab"] for p in _per_pos]))
    _mean_n = float(_np.mean([p["N"] for p in _per_pos]))
    _base_p = get_prospect_params(55)
    write_json5(input_dir / 'vegetation.json5', {
        "Plugin": "BiochemicalSCOPE", "Model": "VegetationSCOPE",
        "PAR_min": 0.400, "PAR_max": 0.700,
        "Cab": round(_mean_cab, 1), "Cca": 10, "Cs": 0,
        "Cw": _base_p["Cw"], "Cdm": _base_p["Cm"],
        "N": round(_mean_n, 2), "fqe": 0.01,
        "Vcmax25": round(vcmax25_from_cab(_mean_cab), 1),
        "BallBerrySlope": 8,
        "BallBerry0": 0.01,
        "RdPerVcmax25": get_species()["rd_per_vcmax25"],
        "Type": get_species()["photo_type"],
        "rho_thermal": 0.01, "tau_thermal": 0.01, "stress_factor": 1,
    })
    write_json5(input_dir / 'radiation.json5', {"Plugin": "DART", "Model": "DART"})
    write_json5(input_dir / 'scene.json5', {"Plugin": "DART", "scene_reader": "DARTSceneTriangleReader"})
    write_json5(input_dir / 'soil.json5', {
        "Plugin": "SoilMod", "Model": "KustasModel",
        "rs_thermal": 0.06, "SMC": 0.25,
    })
    write_json5(input_dir / 'aerodynamics.json5', {
        "Plugin": "AerodynamicsSCOPE", "Model": "AeroSCOPE",
        "Cd": 0.3, "rwc": 0, "rbs": 10.0, "CR": 0.35,
        "CD1": 20.6, "Psicor": 0.2, "CSSOIL": 0.01,
        "Monin_Obukhov_correction": True,
    })
    write_json5(input_dir / 'modelling_parameters.json5', {
        "EB_error": 5, "max_iteration": 20, "min_variation_rate": 0,
        "closure_method": "Newton", "load_state": False,
    })
    write_json5(input_dir / 'output.json5', {
        "model": "PhysicsAwareDataWriter", "intermediate_outputs": False,
        "save_state": False, "radiation": True, "vegetation": True,
        "soil": True, "aerodynamics": False, "energy_balance_products": True,
        "fluxes": True, "save_scene": True, "delimiter": ";",
        "compute_sunlit": False, "sunlit_threshold": 0.5,
        "1 dimension": False, "2 dimension": False, "3 dimension": True,
        "layer_number": 20, "write_yaml": False,
    })
    write_json5(input_dir / 'time_series.json5', {
        "is_time_series": False, "input_filename": "time_series.csv",
        "header_filename": "headers.json",
        "load_from_previous_timestep": False, "ts_number": -1, "deltat": -1,
    })
    write_json5(plugins_dir / 'DART_input.json5', {
        "dart_simulation": dart_simu_name,
        "Compute_Rn1": True, "Compute_broadband": True,
        "Compute_APAR": True, "Compute_Rn2": True,
    })
    write_json5(plugins_dir / 'BiochemicalSCOPE_input.json5', {
        "Kn0": 2.48, "Knalpha": 2.83, "Knbeta": 0.114,
        "g_m": "Not computed", "kV": 0.6396, "apply_T_correction": True,
    })
    write_json5(plugins_dir / 'SoilMod_input.json5', {
        "rss": 500, "Compute_rss_from_SMC": False, "ratio_rn_g": 0.35,
    })
    write_json5(plugins_dir / 'AerodynamicsSCOPE_input.json5', {})

    return baleno_sim_dir


def update_baleno_atmosphere(baleno_sim_dir, T_air_K, ea_hPa, wind_ms,
                              p_hPa=1013, Ca_ppm=400):
    """Update Baleno atmosphere.json5 with new meteorological conditions.

    Args:
        baleno_sim_dir: Path to Baleno simulation directory.
        T_air_K: Air temperature in Kelvin.
        ea_hPa: Water vapour pressure in hPa.
        wind_ms: Wind speed in m/s.
        p_hPa: Atmospheric pressure in hPa.
        Ca_ppm: CO2 concentration in ppm.
    """
    import json as _json
    atmo_path = Path(baleno_sim_dir) / 'input' / 'atmosphere.json5'
    atmo = {
        "z": 10, "Ta": float(T_air_K), "p": float(p_hPa),
        "ea": float(ea_hPa), "u": float(wind_ms),
        "Ca": float(Ca_ppm), "Oa": 280,
    }
    with open(atmo_path, 'w') as f:
        _json.dump(atmo, f, indent=4)


def update_baleno_sun_and_rerun_I(simu_I, sun_zenith, sun_azimuth, timeout=300):
    """Update sun angles in _I DART simulation and re-run RT (skip maket).

    Same approach as update_sun_and_rerun from create_dart_simulation.
    """
    simu_I_path = Path(str(simu_I.simu_dir))
    directions_xml = simu_I_path / 'input' / 'directions.xml'

    if directions_xml.exists():
        tree = ET.parse(str(directions_xml))
        root = tree.getroot()
        for elem in root.iter('SunViewingAngles'):
            elem.set('sunViewingZenithAngle', f'{sun_zenith:.6f}')
            elem.set('sunViewingAzimuthAngle', f'{sun_azimuth:.6f}')
        tree.write(str(directions_xml), xml_declaration=True, encoding='unicode')

    # Clean partial DART outputs to prevent corruption from killed processes
    output_dir = simu_I_path / 'output'
    if output_dir.exists():
        for partial in output_dir.glob('*.nc.tmp'):
            partial.unlink(missing_ok=True)

    for name, runner in [
        ('direction', simu_I.run.direction),
        ('phase', simu_I.run.phase),
        ('dart', simu_I.run.dart),
    ]:
        try:
            runner(timeout=timeout)
        except Exception as e:
            print(f"  Baleno _I {name} error: {e}")
            return False
    return True


def update_baleno_datetime_and_rerun_I(simu_I, calendar_date, hour_utc,
                                        minute_utc, timeout=300):
    """Update ExactDateHour in _I DART simulation and re-run RT (skip maket).

    For use with exactDate=1 mode. Edits the ExactDateHour XML element
    instead of SunViewingAngles.

    The timeout applies per-module (direction, phase, dart). For large scenes
    the dart module dominates; pass a higher timeout for scenes with >200k
    visibility particles (e.g. day 26+ with 9 mature plants).
    """
    simu_I_path = Path(str(simu_I.simu_dir))
    directions_xml = simu_I_path / 'input' / 'directions.xml'

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

    # Clean partial DART outputs to prevent corruption from killed processes
    output_dir = simu_I_path / 'output'
    if output_dir.exists():
        for partial in output_dir.glob('*.nc.tmp'):
            partial.unlink(missing_ok=True)

    for name, runner in [
        ('direction', simu_I.run.direction),
        ('phase', simu_I.run.phase),
        ('dart', simu_I.run.dart),
    ]:
        try:
            runner(timeout=timeout)
        except Exception as e:
            print(f"  Baleno _I {name} error: {e}")
            return False
    return True


def run_baleno_subprocess(baleno_simu_name=None, timeout=3600):
    """Run Baleno as subprocess. Returns True on success."""
    if not VENV_PYTHON.exists():
        print(f"  ERROR: darteb_venv not found at {VENV_PYTHON}")
        return False

    env = os.environ.copy()
    env['PYTHONPATH'] = str(BALENO_DIR)
    env['DART_HOME'] = str(DART_DIR)
    env['DART_LOCAL'] = str(DART_LOCAL)

    # Remove DART's problematic libreadline from LD_LIBRARY_PATH
    ld_path = env.get('LD_LIBRARY_PATH', '')
    dart_lib_paths = [str(DART_DIR / 'bin' / 'python' / 'lib')]
    filtered_ld = ':'.join(
        p for p in ld_path.split(':')
        if p and p not in dart_lib_paths
    )
    env['LD_LIBRARY_PATH'] = filtered_ld

    cmd = [str(VENV_PYTHON), '-m', 'src.main']
    import time as _time
    t0 = _time.time()
    print(f"    Baleno subprocess: {' '.join(cmd)}")
    print(f"    cwd={BALENO_DIR}, timeout={timeout}s")
    try:
        result = subprocess.run(
            cmd, cwd=str(BALENO_DIR), env=env,
            capture_output=True, text=True, timeout=timeout,
        )
        elapsed = _time.time() - t0

        if result.returncode != 0:
            print(f"  Baleno error (exit {result.returncode}, {elapsed:.0f}s)")
            stderr_lines = result.stderr.strip().split('\n')
            for line in stderr_lines[-10:]:
                print(f"    {line}")
            return False

        # Baleno exits 0 even on failure (e.g. "does not exist. Exit").
        # Check stdout/stderr for known failure patterns.
        combined = result.stdout + result.stderr
        failure_patterns = ['does not exist', 'Error', 'Traceback', 'KeyError']
        for pattern in failure_patterns:
            if pattern in combined:
                print(f"  Baleno exited 0 but output indicates failure ({elapsed:.0f}s):")
                for line in combined.strip().split('\n')[-15:]:
                    print(f"    {line}")
                return False

        print(f"    Baleno completed in {elapsed:.1f}s")
        return True
    except subprocess.TimeoutExpired:
        print(f"  Baleno timed out after {timeout}s")
        return False
    except Exception as e:
        print(f"  Baleno subprocess error: {e}")
        return False


def read_baleno_tleaf(baleno_sim_dir, mapping_json_path, reindex_json_path,
                      grid_info_path=None, center_plant_idx=4,
                      tair_c=None, apar_shaded_threshold=10.0):
    """Read Baleno outputs and aggregate per-triangle Tleaf to per-segment.

    Combines steps 6-8 logic into one call.

    Args:
        tair_c: Air temperature (°C) for fallback on rejected triangles.
            None = use 25.0 for backward compatibility.
        apar_shaded_threshold: APAR below this (µmol/m²/s) = shaded triangle.
            Shaded triangles with high EB_ERROR get Tair instead of rejection.

    Returns:
        np.ndarray of per-leaf-segment Tleaf (°C), or None on failure.
    """
    import json as _json

    output_base = Path(baleno_sim_dir) / 'output'
    results_dir = output_base / 'final_results'

    if not results_dir.exists():
        return None

    # Read scene
    scene_file = None
    for candidate in [output_base / 'scene', results_dir / 'scene.csv', results_dir / 'scene']:
        if candidate.exists() and candidate.stat().st_size > 0:
            scene_file = candidate
            break
    if scene_file is None:
        return None

    delimiter = detect_delimiter(scene_file)
    scene_str = np.genfromtxt(str(scene_file), skip_header=1, delimiter=delimiter, dtype=str)

    with open(scene_file) as f:
        header_line = f.readline().strip()
    scene_header = [h.strip() for h in header_line.split(delimiter)]

    col_type_id = scene_header.index('TYPE_ID') if 'TYPE_ID' in scene_header else 0
    col_dart_name = scene_header.index('DART_NAME') if 'DART_NAME' in scene_header else 1
    col_index_obj = scene_header.index('INDEX_OBJECT') if 'INDEX_OBJECT' in scene_header else 2

    type_ids = scene_str[:, col_type_id].astype(float).astype(int)
    dart_names = scene_str[:, col_dart_name]
    index_in_object = scene_str[:, col_index_obj].astype(float).astype(int)

    # Read energy balance
    eb_file = results_dir / 'energy_balance_3D.csv'
    if not eb_file.exists():
        return None

    with open(eb_file) as f:
        eb_header_line = f.readline().strip()
    eb_header = [h.strip() for h in eb_header_line.split(delimiter)]
    eb_data = np.genfromtxt(str(eb_file), skip_header=1, delimiter=delimiter,
                            dtype=float, filling_values=np.nan)

    col_temp = -1
    col_eb_err = -1
    for i, h in enumerate(eb_header):
        if 'temperature' in h.lower():
            col_temp = i
        if 'error' in h.lower():
            col_eb_err = i
    if col_temp < 0:
        col_temp = 1

    # Read radiation_3D.csv for shaded-triangle detection
    rad_file = results_dir / 'radiation_3D.csv'
    rad_data = None
    col_apar = -1
    if rad_file.exists():
        with open(rad_file) as f:
            rad_header = [h.strip() for h in f.readline().strip().split(delimiter)]
        rad_data = np.genfromtxt(str(rad_file), skip_header=1, delimiter=delimiter,
                                 dtype=float, filling_values=np.nan)
        for i, h in enumerate(rad_header):
            if 'absorption' in h.lower() and 'par' in h.lower():
                col_apar = i

    # Load reindex
    with open(reindex_json_path) as f:
        reindex = _json.load(f)
    dart_to_obj = {}
    for gi_str, obj_indices in reindex['dart_to_obj'].items():
        dart_to_obj[int(gi_str)] = np.array(obj_indices, dtype=np.int64)
    group_names = reindex['group_names']

    # Identify leaf triangles
    leaf_mask = (type_ids >= 100) | (type_ids == 5)
    leaf_dart_names = set(dart_names[leaf_mask])

    # Build DART name → .ori group mapping (ObjectFields)
    sample_dn = next(iter(leaf_dart_names), '')
    is_object_fields = '_mo' in sample_dn and '_go' in sample_dn

    dart_name_to_ori = {}
    center_dart_names = set()
    ori_group_counts = {gi: len(arr) for gi, arr in dart_to_obj.items()}

    if is_object_fields:
        ci = center_plant_idx
        if grid_info_path and Path(grid_info_path).exists():
            with open(grid_info_path) as gf:
                gi_data = _json.load(gf)
            ci = gi_data.get('center_plant_idx', center_plant_idx)

        for dn in leaf_dart_names:
            mo_m = re.search(r'_mo(\d+)', dn)
            go_m = re.search(r'_go(\d+)', dn)
            if mo_m and go_m:
                instance = int(mo_m.group(1))
                group = int(go_m.group(1))
                if group in ori_group_counts:
                    dart_name_to_ori[dn] = group
                    if instance == ci:
                        center_dart_names.add(dn)
    else:
        used_ori = set()
        dn_counts = {}
        for dn in leaf_dart_names:
            dn_counts[dn] = np.sum(dart_names == dn)
        for dn in sorted(dn_counts.keys()):
            for gi in sorted(ori_group_counts.keys()):
                if gi in used_ori:
                    continue
                if dn_counts[dn] == ori_group_counts[gi]:
                    dart_name_to_ori[dn] = gi
                    used_ori.add(gi)
                    break
        center_dart_names = set(dart_name_to_ori.keys())

    # Build baleno_to_obj mapping (center plant only)
    n_total = len(type_ids)
    baleno_to_obj = np.full(n_total, -1, dtype=np.int64)
    for row_idx in range(n_total):
        if not leaf_mask[row_idx]:
            continue
        dn = dart_names[row_idx]
        if dn not in center_dart_names or dn not in dart_name_to_ori:
            continue
        gi = dart_name_to_ori[dn]
        idx = index_in_object[row_idx]
        ori_arr = dart_to_obj[gi]
        if idx < len(ori_arr):
            baleno_to_obj[row_idx] = ori_arr[idx]

    # Build reverse mapping
    obj_to_baleno = {}
    for row_idx in range(n_total):
        obj_face = baleno_to_obj[row_idx]
        if obj_face >= 0:
            obj_to_baleno[int(obj_face)] = row_idx

    # Aggregate to segments
    with open(mapping_json_path) as f:
        seg_mapping = _json.load(f)

    # EB_ERROR threshold: reject triangles where Baleno didn't converge.
    # Well-converged triangles have |EB_ERROR| < 3 W/m²; non-converged
    # ones reach 80–160 W/m² and produce non-physical Tleaf (80–150 °C).
    # Tiered approach: shaded triangles (low APAR) with high EB_ERROR get
    # Tair fallback (physically correct — no shortwave heating). Only lit
    # triangles with high EB_ERROR are truly rejected.
    EB_ERR_MAX = 20.0  # W/m²
    fallback_temp_c = tair_c if tair_c is not None else 25.0
    n_rejected = 0
    n_shaded_fallback = 0

    def _is_shaded(row):
        """Check if triangle is shaded (low APAR)."""
        if rad_data is None or col_apar < 0:
            return False
        if row >= rad_data.shape[0] or col_apar >= rad_data.shape[1]:
            return False
        apar = rad_data[row, col_apar]
        return not np.isnan(apar) and apar < apar_shaded_threshold

    segment_tleaf = []
    for organ in seg_mapping['organs']:
        if organ['type'] != 'leaf':
            continue
        for seg in organ['segments']:
            tri_indices = seg['triangle_indices']
            temps = []
            for tidx in tri_indices:
                if tidx in obj_to_baleno:
                    row = obj_to_baleno[tidx]
                    if row < eb_data.shape[0] and col_temp < eb_data.shape[1]:
                        t = eb_data[row, col_temp]
                        if np.isnan(t):
                            continue
                        # Check EB convergence
                        if col_eb_err >= 0 and col_eb_err < eb_data.shape[1]:
                            err = abs(eb_data[row, col_eb_err])
                            if err > EB_ERR_MAX:
                                if _is_shaded(row):
                                    # Shaded triangle: Tleaf ≈ Tair is correct
                                    temps.append(fallback_temp_c)
                                    n_shaded_fallback += 1
                                else:
                                    n_rejected += 1
                                continue
                        temps.append(t - 273.15)  # K → °C
            if temps:
                segment_tleaf.append(float(np.mean(temps)))
            else:
                segment_tleaf.append(fallback_temp_c)

    if n_shaded_fallback > 0 or n_rejected > 0:
        print(f"  Tleaf: {n_shaded_fallback} shaded->Tair({fallback_temp_c:.1f}C), "
              f"{n_rejected} lit rejected (|EB_ERROR|>{EB_ERR_MAX} W/m²)")

    return np.array(segment_tleaf)


def read_baleno_outputs_multi(baleno_sim_dir, mapping_json_paths,
                               reindex_json_paths, n_plants,
                               tair_c=None, apar_shaded_threshold=10.0,
                               read_fluorescence=False):
    """Read Baleno outputs: per-plant Tleaf + optionally eta (fluorescence).

    When read_fluorescence=True, also reads vegetation_3D.csv column 6
    (FLUORESCENCE) and aggregates eta to per-segment arrays. Returns per-triangle
    raw data dicts for optional triangle-level SIF output.

    Args:
        baleno_sim_dir: Path to Baleno simulation directory.
        mapping_json_paths: List of mapping JSON paths (one per plant).
        reindex_json_paths: List of reindex JSON paths (one per plant).
        n_plants: Number of plants.
        tair_c: Air temperature (C) for fallback.
        apar_shaded_threshold: APAR below this = shaded triangle.
        read_fluorescence: If True, also read vegetation_3D.csv for eta.

    Returns:
        dict with:
          'tleaf': list of n_plants np.ndarray (per-segment Tleaf C)
          'eta': list of n_plants np.ndarray (per-segment eta) [only if read_fluorescence]
          'tri_data_raw': list of n_plants lists of per-tri dicts [only if read_fluorescence]
        Or None on failure.
    """
    import json as _json
    import re as _re

    output_base = Path(baleno_sim_dir) / 'output'
    results_dir = output_base / 'final_results'
    if not results_dir.exists():
        print(f"  read_baleno_outputs_multi: results dir not found")
        return None

    # Read scene file
    scene_file = None
    for candidate in [output_base / 'scene', results_dir / 'scene.csv',
                      results_dir / 'scene']:
        if candidate.exists() and candidate.stat().st_size > 0:
            scene_file = candidate
            break
    if scene_file is None:
        print(f"  read_baleno_outputs_multi: scene file not found")
        return None

    delimiter = detect_delimiter(scene_file)
    scene_str = np.genfromtxt(str(scene_file), skip_header=1,
                               delimiter=delimiter, dtype=str)
    with open(scene_file) as f:
        header_line = f.readline().strip()
    scene_header = [h.strip() for h in header_line.split(delimiter)]

    col_type_id = scene_header.index('TYPE_ID') if 'TYPE_ID' in scene_header else 0
    col_dart_name = scene_header.index('DART_NAME') if 'DART_NAME' in scene_header else 1
    col_index_obj = scene_header.index('INDEX_OBJECT') if 'INDEX_OBJECT' in scene_header else 2

    type_ids = scene_str[:, col_type_id].astype(float).astype(int)
    dart_names = scene_str[:, col_dart_name]
    index_in_object = scene_str[:, col_index_obj].astype(float).astype(int)
    leaf_mask = (type_ids >= 100) | (type_ids == 5)
    n_total = len(type_ids)

    # Read energy balance
    eb_file = results_dir / 'energy_balance_3D.csv'
    if not eb_file.exists():
        print(f"  read_baleno_outputs_multi: energy_balance_3D.csv not found")
        return None

    with open(eb_file) as f:
        eb_header = [h.strip() for h in f.readline().strip().split(delimiter)]
    eb_data = np.genfromtxt(str(eb_file), skip_header=1, delimiter=delimiter,
                            dtype=float, filling_values=np.nan)

    col_temp = -1
    col_eb_err = -1
    for i, h in enumerate(eb_header):
        if 'temperature' in h.lower():
            col_temp = i
        if 'error' in h.lower():
            col_eb_err = i
    if col_temp < 0:
        col_temp = 1

    # Read radiation for shaded detection
    rad_file = results_dir / 'radiation_3D.csv'
    rad_data = None
    col_apar = -1
    if rad_file.exists():
        with open(rad_file) as f:
            rad_header = [h.strip() for h in f.readline().strip().split(delimiter)]
        rad_data = np.genfromtxt(str(rad_file), skip_header=1,
                                  delimiter=delimiter, dtype=float,
                                  filling_values=np.nan)
        for i, h in enumerate(rad_header):
            if 'absorption' in h.lower() and 'par' in h.lower():
                col_apar = i

    # Read vegetation (fluorescence) if requested
    veg_data = None
    col_eta = -1
    col_an_veg = -1
    if read_fluorescence:
        veg_file = results_dir / 'vegetation_3D.csv'
        if veg_file.exists():
            with open(veg_file) as f:
                veg_header = [h.strip() for h in f.readline().strip().split(delimiter)]
            veg_data = np.genfromtxt(str(veg_file), skip_header=1,
                                      delimiter=delimiter, dtype=float,
                                      filling_values=np.nan)
            for i, h in enumerate(veg_header):
                if 'fluorescence' in h.lower():
                    col_eta = i
                if 'net photosynthesis' in h.lower():
                    col_an_veg = i
            if col_eta < 0:
                col_eta = 6  # default FLUORESCENCE column index

    # Build per-plant reindex
    per_plant_dart_to_obj = {}
    for pi in range(n_plants):
        with open(reindex_json_paths[pi]) as f:
            ri = _json.load(f)
        dart_to_obj = {}
        for gi_str, obj_indices in ri['dart_to_obj'].items():
            dart_to_obj[int(gi_str)] = np.array(obj_indices, dtype=np.int64)
        per_plant_dart_to_obj[pi] = dart_to_obj

    EB_ERR_MAX = 20.0
    fallback_temp_c = tair_c if tair_c is not None else 25.0

    def _is_shaded(row):
        if rad_data is None or col_apar < 0:
            return False
        if row >= rad_data.shape[0] or col_apar >= rad_data.shape[1]:
            return False
        apar = rad_data[row, col_apar]
        return not np.isnan(apar) and apar < apar_shaded_threshold

    all_plant_tleaf = []
    all_plant_eta = []
    all_plant_tri_raw = []

    for pi in range(n_plants):
        dart_to_obj = per_plant_dart_to_obj[pi]

        # Map Baleno rows to OBJ faces for this plant
        baleno_to_obj = np.full(n_total, -1, dtype=np.int64)
        for row_idx in range(n_total):
            if not leaf_mask[row_idx]:
                continue
            dn = dart_names[row_idx]
            mo_m = _re.search(r'_mo(\d+)', dn)
            go_m = _re.search(r'_go(\d+)', dn)
            if mo_m and go_m:
                instance = int(mo_m.group(1))
                group = int(go_m.group(1))
                if instance == pi and group in dart_to_obj:
                    idx = index_in_object[row_idx]
                    ori_arr = dart_to_obj[group]
                    if idx < len(ori_arr):
                        baleno_to_obj[row_idx] = ori_arr[idx]

        # Build reverse mapping
        obj_to_baleno = {}
        for row_idx in range(n_total):
            obj_face = baleno_to_obj[row_idx]
            if obj_face >= 0:
                obj_to_baleno[int(obj_face)] = row_idx

        # Aggregate to segments
        with open(mapping_json_paths[pi]) as f:
            seg_mapping = _json.load(f)

        segment_tleaf = []
        segment_eta = []
        segment_tri_data = []  # per-segment aggregated tri info
        plant_tri_raw = []     # flat list of per-triangle dicts

        for organ in seg_mapping['organs']:
            if organ['type'] != 'leaf':
                continue
            for seg in organ['segments']:
                tri_indices = seg['triangle_indices']
                temps = []
                etas = []
                seg_tri_apar = []
                seg_tri_count = 0
                seg_tri_sunlit = 0

                for tidx in tri_indices:
                    if tidx not in obj_to_baleno:
                        continue
                    row = obj_to_baleno[tidx]
                    if row >= eb_data.shape[0]:
                        continue

                    # Temperature
                    if col_temp < eb_data.shape[1]:
                        t = eb_data[row, col_temp]
                        if not np.isnan(t):
                            if col_eb_err >= 0 and col_eb_err < eb_data.shape[1]:
                                err = abs(eb_data[row, col_eb_err])
                                if err > EB_ERR_MAX:
                                    if _is_shaded(row):
                                        temps.append(fallback_temp_c)
                                    continue
                            temps.append(t - 273.15)

                    # Fluorescence
                    if read_fluorescence and veg_data is not None:
                        if row < veg_data.shape[0] and col_eta < veg_data.shape[1]:
                            eta_val = veg_data[row, col_eta]
                            if not np.isnan(eta_val):
                                etas.append(eta_val)

                        # Per-tri raw data
                        apar_tri = 0.0
                        if rad_data is not None and col_apar >= 0:
                            if row < rad_data.shape[0]:
                                a = rad_data[row, col_apar]
                                if not np.isnan(a):
                                    apar_tri = float(a)
                        an_tri = 0.0
                        if col_an_veg >= 0 and row < veg_data.shape[0]:
                            av = veg_data[row, col_an_veg]
                            if not np.isnan(av):
                                an_tri = float(av)
                        tleaf_tri = temps[-1] if temps else fallback_temp_c
                        eta_tri = etas[-1] if etas else 0.0
                        area_tri = seg.get('triangle_area_cm2', 1.0)
                        if isinstance(area_tri, list):
                            ti_local = tri_indices.index(tidx) if tidx in tri_indices else 0
                            area_tri = area_tri[ti_local] if ti_local < len(area_tri) else 1.0

                        plant_tri_raw.append({
                            'tri_idx': tidx,
                            'segment_idx': len(segment_tleaf),
                            'apar_Wm2': apar_tri,
                            'tleaf_C': tleaf_tri,
                            'eta': eta_tri,
                            'An_umol': an_tri,
                            'area_cm2': float(area_tri) if not isinstance(area_tri, list) else 1.0,
                        })

                        seg_tri_count += 1
                        apar_umol = apar_tri * 4.57
                        seg_tri_apar.append(apar_umol)
                        if apar_umol > apar_shaded_threshold:
                            seg_tri_sunlit += 1

                segment_tleaf.append(float(np.mean(temps)) if temps else fallback_temp_c)

                if read_fluorescence:
                    segment_eta.append(float(np.mean(etas)) if etas else 0.0)
                    segment_tri_data.append({
                        'n_triangles': seg_tri_count,
                        'total_area_cm2': seg.get('total_area_cm2', 0.0),
                        'mean_apar_umol': float(np.mean(seg_tri_apar)) if seg_tri_apar else 0.0,
                        'n_sunlit': seg_tri_sunlit,
                        'n_total': seg_tri_count,
                    })

        all_plant_tleaf.append(np.array(segment_tleaf))
        if read_fluorescence:
            all_plant_eta.append(np.array(segment_eta))
            all_plant_tri_raw.append(plant_tri_raw)

    result = {'tleaf': all_plant_tleaf}
    if read_fluorescence:
        result['eta'] = all_plant_eta
        result['tri_data'] = [segment_tri_data for _ in range(n_plants)]  # placeholder
        result['tri_data_raw'] = all_plant_tri_raw
    return result


def read_baleno_tleaf_multi(baleno_sim_dir, mapping_json_paths,
                             reindex_json_paths, n_plants,
                             tair_c=None, apar_shaded_threshold=10.0):
    """Read Baleno outputs and return per-plant per-segment Tleaf arrays.

    Thin wrapper around read_baleno_outputs_multi() for backward compatibility.

    Returns:
        List of n_plants np.ndarray (per-leaf-segment Tleaf in °C),
        or None on failure.
    """
    result = read_baleno_outputs_multi(
        baleno_sim_dir, mapping_json_paths, reindex_json_paths, n_plants,
        tair_c=tair_c, apar_shaded_threshold=apar_shaded_threshold,
        read_fluorescence=False)
    if result is None:
        return None
    return result['tleaf']


def log_baleno_diagnostics(baleno_sim_dir, tleaf_per_segment, tair_c):
    """Log diagnostic stats from Baleno EB output files.

    Reads energy_balance_3D.csv and reports convergence, Tleaf offset,
    and energy flux statistics to help diagnose EB anomalies.
    """
    results_dir = Path(baleno_sim_dir) / 'output' / 'final_results'
    eb_file = results_dir / 'energy_balance_3D.csv'
    if not eb_file.exists():
        print(f"    [EB diag] energy_balance_3D.csv not found")
        return

    delimiter = detect_delimiter(eb_file)
    with open(eb_file) as f:
        eb_header = [h.strip() for h in f.readline().strip().split(delimiter)]
    eb_data = np.genfromtxt(str(eb_file), skip_header=1, delimiter=delimiter,
                            dtype=float, filling_values=np.nan)
    if eb_data.ndim < 2 or eb_data.shape[0] == 0:
        print(f"    [EB diag] empty EB data")
        return

    # Find columns by name
    def _col(names):
        for n in names:
            for i, h in enumerate(eb_header):
                if n.lower() in h.lower():
                    return i
        return -1

    col_temp = _col(['temperature'])
    col_err = _col(['error'])
    col_h = _col(['sensible', 'H'])
    col_le = _col(['latent', 'lE', 'LE'])
    col_rn = _col(['net_radiation', 'Rn', 'rn'])

    print(f"    [EB diag] {eb_data.shape[0]} triangles, Tair={tair_c:.1f}C")

    # Read radiation_3D.csv for shaded/lit breakdown
    rad_file = results_dir / 'radiation_3D.csv'
    rad_data_diag = None
    col_apar_diag = -1
    if rad_file.exists():
        with open(rad_file) as f:
            rad_hdr = [h.strip() for h in f.readline().strip().split(delimiter)]
        rad_data_diag = np.genfromtxt(str(rad_file), skip_header=1,
                                      delimiter=delimiter, dtype=float,
                                      filling_values=np.nan)
        for i, h in enumerate(rad_hdr):
            if 'absorption' in h.lower() and 'par' in h.lower():
                col_apar_diag = i

    if col_err >= 0:
        eb_err = eb_data[:, col_err]
        valid = ~np.isnan(eb_err)
        if np.any(valid):
            n_converged = int(np.sum(np.abs(eb_err[valid]) < 3.0))
            n_high_err = int(np.sum(np.abs(eb_err[valid]) > 20.0))
            # Break down high-error triangles into shaded vs lit
            n_shaded = 0
            n_lit = 0
            if rad_data_diag is not None and col_apar_diag >= 0:
                high_err_mask = valid & (np.abs(eb_err) > 20.0)
                high_err_idx = np.where(high_err_mask)[0]
                for idx in high_err_idx:
                    if (idx < rad_data_diag.shape[0] and
                            col_apar_diag < rad_data_diag.shape[1]):
                        apar = rad_data_diag[idx, col_apar_diag]
                        if not np.isnan(apar) and apar < 10.0:
                            n_shaded += 1
                        else:
                            n_lit += 1
                    else:
                        n_lit += 1
            print(f"    [EB diag] EB_ERROR: mean={np.nanmean(eb_err):.2f}, "
                  f"std={np.nanstd(eb_err):.2f} W/m2")
            print(f"    [EB diag] converged (|err|<3)={n_converged}, "
                  f"high_err (|err|>20)={n_high_err}"
                  f"{f' (shaded={n_shaded}, lit={n_lit})' if n_shaded + n_lit > 0 else ''}")

    if col_temp >= 0:
        temps_k = eb_data[:, col_temp]
        valid = ~np.isnan(temps_k)
        if np.any(valid):
            temps_c = temps_k[valid] - 273.15
            print(f"    [EB diag] Tleaf(all tri): mean={np.mean(temps_c):.2f}C, "
                  f"std={np.std(temps_c):.2f}C, "
                  f"offset={np.mean(temps_c) - tair_c:+.2f}C vs Tair")

    if tleaf_per_segment is not None and len(tleaf_per_segment) > 0:
        print(f"    [EB diag] Tleaf(segments): mean={np.mean(tleaf_per_segment):.2f}C, "
              f"std={np.std(tleaf_per_segment):.2f}C, "
              f"offset={np.mean(tleaf_per_segment) - tair_c:+.2f}C vs Tair")

    for label, col in [('H', col_h), ('lE', col_le), ('Rn', col_rn)]:
        if col >= 0:
            vals = eb_data[:, col]
            valid = ~np.isnan(vals)
            if np.any(valid):
                print(f"    [EB diag] {label}: mean={np.nanmean(vals):.1f}, "
                      f"std={np.nanstd(vals):.1f} W/m2")


def run_baleno_with_external_gs(gs_per_segment, mapping_json_path,
                                reindex_json_path, baleno_sim_dir,
                                baleno_simu_name,
                                grid_info_path=None, center_plant_idx=4,
                                timeout=1800):
    """Run Baleno energy balance with externally-provided stomatal conductance.

    Writes per-triangle rcw CSV, updates Baleno vegetation config to use the
    ExternalGS plugin, runs Baleno subprocess, and reads resulting Tleaf.

    Args:
        gs_per_segment: array of gs [mol CO2/m²/s] per leaf segment.
        mapping_json_path: Path to DART mapping JSON.
        reindex_json_path: Path to .ori reindex JSON.
        baleno_sim_dir: Path to Baleno simulation directory.
        baleno_simu_name: Name of the Baleno simulation.
        grid_info_path: Path to grid_info.json (optional).
        center_plant_idx: Index of center plant in grid.
        timeout: Subprocess timeout in seconds.

    Returns:
        np.ndarray of per-leaf-segment Tleaf [°C], or None on failure.
    """
    from ..photosynthesis.iterative import (
        segment_gs_to_triangle_gs, write_triangle_gs_csv,
        RCW_MAX,
    )

    # Convert segment gs -> triangle rcw
    tri_result = segment_gs_to_triangle_gs(
        gs_per_segment, mapping_json_path, reindex_json_path)

    # Write CSV
    gs_csv_path = Path(baleno_sim_dir) / 'input' / 'external_gs.csv'
    write_triangle_gs_csv(tri_result['rcw_per_triangle'], gs_csv_path)

    # Update vegetation.json5 to use ExternalGS plugin (mean Cab/N from LOPS)
    import numpy as _np
    _per_pos = get_prospect_params_per_position(55, 11)
    _mean_cab = float(_np.mean([p["Cab"] for p in _per_pos]))
    _mean_n = float(_np.mean([p["N"] for p in _per_pos]))
    _base_p = get_prospect_params(55)
    input_dir = Path(baleno_sim_dir) / 'input'
    write_json5(input_dir / 'vegetation.json5', {
        "Plugin": "ExternalGS",
        "Model": "VegetationExternalGS",
        "PAR_min": 0.400, "PAR_max": 0.700,
        "Cab": round(_mean_cab, 1), "Cca": 10, "Cs": 0,
        "Cw": _base_p["Cw"], "Cdm": _base_p["Cm"],
        "N": round(_mean_n, 2), "fqe": 0.01,
        "Vcmax25": round(vcmax25_from_cab(_mean_cab), 1),
        "BallBerrySlope": 8, "BallBerry0": 0.01,
        "RdPerVcmax25": get_species()["rd_per_vcmax25"],
        "Type": get_species()["photo_type"],
        "rho_thermal": 0.01, "tau_thermal": 0.01, "stress_factor": 1,
    })

    # Write plugin-specific input
    plugins_dir = input_dir / 'plugins'
    plugins_dir.mkdir(parents=True, exist_ok=True)
    write_json5(plugins_dir / 'ExternalGS_input.json5', {
        "gs_file": str(gs_csv_path),
        "fallback_rcw": 100.0,
    })

    # Write config.ini
    baleno_config_path = BALENO_DIR / 'resources' / 'config.ini'
    import textwrap as _textwrap
    baleno_config_path.write_text(_textwrap.dedent(f"""\
        [simulation]
        user_data_path =
        name = {baleno_simu_name}
    """))

    # Run Baleno
    ok = run_baleno_subprocess(timeout=timeout)
    if not ok:
        return None

    # Read Tleaf
    return read_baleno_tleaf(
        str(baleno_sim_dir), mapping_json_path, reindex_json_path,
        grid_info_path=grid_info_path,
        center_plant_idx=center_plant_idx,
    )
