#!/usr/bin/env python3
"""
Phase 2: Baleno Energy Balance for Per-Triangle Tleaf + APAR.

Runs the Baleno (DART-EB) energy balance on the Phase 1 CPlantBox maize
geometry. Baleno computes per-triangle leaf temperature via iterative energy
balance coupling DART radiative transfer with SCOPE-based photosynthesis.

Prerequisites:
  - Phase 1 completed (maize_day55_dart.obj, _dart_mapping.json, _reindex.json)
  - darteb_venv at /home/lukas/PHD/darteb_venv (Python 3.12 + Baleno deps)
  - DART installed at /home/lukas/DART
  - pytools4dart importable (cpbenv)

Usage:
  cd /home/lukas/PHD
  source CPlantBox/cpbenv/bin/activate
  python CPlantBox/dart/coupling/run_baleno.py
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

from ..config import DART_HOME, DART_EB_DIR, DARTRC, BALENO_PYTHON, OUTPUT_DIR, DART_THREADS, get_species
from ..prospect_params import (get_prospect_params, get_prospect_params_per_position,
                               get_stem_prospect_params,
                               log_consistency, log_lops_consistency, vcmax25_from_cab)
from ..dart.simulation import configure_atmosphere_midlatsum

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

# Baleno user_data base: when config.ini user_data_path is empty, both
# main.py and PathManager resolve to BALENO_DIR / 'user_data'.
# We MUST use empty user_data_path because main.py:check_existing_simulation()
# and PathManager construct different paths when user_data_path is non-empty
# (main.py skips the 'user_data/' subdirectory — Baleno bug).
BALENO_USER_DATA = BALENO_DIR / 'user_data'

# Pre-existing Phase 1 outputs
OBJ_PATH = OUTPUT_DIR / 'maize_day55_dart.obj'
MAPPING_JSON = OUTPUT_DIR / 'maize_day55_dart_mapping.json'
REINDEX_JSON = OUTPUT_DIR / 'maize_day55_reindex.json'

# PROSPECT parameters: loaded from shared growth-stage table
# Baleno uses the same day as Phase 1 (day 55)
PROSPECT_PARAMS = get_prospect_params(55)

# Sun geometry (same as Phase 1)
SUN_ZENITH = 45.0
SUN_AZIMUTH = 225.0
SCENE_SIZE = [4, 4]
PLANT_POS = (2.0, 2.0)

# Plant grid (same as Phase 1 — import from grid_info.json at runtime)
GRID_INFO_PATH = OUTPUT_DIR / 'grid_info.json'
FIELD_FILENAME = 'plant_field.txt'

# Shortwave bands: 21 bands from 400-2500 nm (100 nm each)
# Band edges: [0.400, 0.500, 0.600, ..., 2.500] µm
# Center wavelength = edge + bandwidth/2 (e.g., 0.450 for [0.400, 0.500])
SW_BANDS = [(0.400 + i * 0.100 + 0.050, 0.100) for i in range(21)]

# Thermal band: single band at 10 µm (8-12 µm window)
TIR_BAND = (10.0, 4.0)  # center 10 µm, bandwidth 4 µm


# ============================================================================
# Step 1: Create _I DART simulation (shortwave, 21 bands)
# ============================================================================
def step1_create_simu_I():
    """Create shortwave DART simulation with 21 bands covering 400-2500nm."""
    print("=" * 70)
    print("STEP 1: Create DART Simulation _I (Shortwave)")
    print("=" * 70)

    simu_name = f'{DART_SIMU_NAME}/{DART_SIMU_NAME}_I'
    parent_dir = DART_LOCAL / 'simulations' / DART_SIMU_NAME
    simu_dir = parent_dir / f'{DART_SIMU_NAME}_I'

    parent_dir.mkdir(parents=True, exist_ok=True)

    if simu_dir.exists():
        shutil.rmtree(str(simu_dir))
        print(f"  Cleaned up previous: {simu_dir}")

    simu = ptd.simulation(simu_name, empty=True)
    simu.scene.size = SCENE_SIZE

    # 21 shortwave bands
    for wvl, bw in SW_BANDS:
        simu.add.band(wvl=wvl, bw=bw)
    print(f"  Bands: {len(SW_BANDS)} ({SW_BANDS[0][0]*1000:.0f}-"
          f"{(SW_BANDS[-1][0]+SW_BANDS[-1][1])*1000:.0f}nm)")

    # Sun
    simu.core.directions.Directions.SunViewingAngles.sunViewingZenithAngle = SUN_ZENITH
    simu.core.directions.Directions.SunViewingAngles.sunViewingAzimuthAngle = SUN_AZIMUTH

    # Ground optical property
    simu.add.optical_property(
        type='Lambertian', ident='ground',
        databaseName='Lambertian_mineral.db', ModelName='clay_brown',
        useMultiplicativeFactorForLUT=0,
    )
    simu.scene.ground.OpticalPropertyLink.ident = 'ground'

    # Per-position leaf optical properties (PROSPECT from LOPS data)
    file_src_fullpath = simu.get_input_file_path(str(OBJ_PATH))
    obj_info = ptd.OBJtools.objreader(file_src_fullpath)
    gnames = ptd.OBJtools.gnames_dart_order(obj_info.names)
    n_leaf_groups = sum(1 for g in gnames if not g.endswith('_00'))

    per_pos_params = get_prospect_params_per_position(55, n_leaf_groups)
    for i, params in enumerate(per_pos_params):
        ident = f'maize_leaf_pos{i}'
        simu.add.optical_property(
            type='Lambertian', ident=ident,
            prospect=params,
            useMultiplicativeFactorForLUT=0,
        )
        print(f"    Leaf OP: {ident} (Cab={params['Cab']:.1f}, N={params['N']:.2f})")
    log_lops_consistency(55, n_leaf_groups)

    stem_prospect = get_stem_prospect_params(55)
    simu.add.optical_property(
        type='Lambertian', ident='maize_stem',
        prospect=stem_prospect,
        useMultiplicativeFactorForLUT=0,
    )

    # 3D Object via ObjectFields (same grid as Phase 1)
    xdim, ydim, zdim = obj_info.dims
    xc, yc, zc = obj_info.center

    # Create groups with per-position optical properties + doubleFace
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
        print(f"    Group {gi+1}: {gname} -> {op_ident}, doubleFace={df}")
    groups = ptd.object_3d.create_Groups(Group=groups_list)
    print(f"  OBJ groups: {len(gnames)}")

    # Create model object for ObjectFields
    geom = ptd.object_3d.create_GeometricProperties(
        Dimension3D=ptd.object_3d.create_Dimension3D(
            xdim=xdim, ydim=ydim, zdim=zdim),
        Center3D=ptd.object_3d.create_Center3D(
            xCenter=xc, yCenter=yc, zCenter=zc),
        ScaleProperties=ptd.object_3d.create_ScaleProperties(
            xscale=1.0, yscale=1.0, zscale=1.0),
    )
    model_obj = ptd.object_3d.create_Object(
        file_src=str(OBJ_PATH),
        hasGroups=1,
        GeometricProperties=geom,
        Groups=groups,
        num=0,
        name='CPlantBox_Maize',
        objectDEMMode=0,
    )

    # Load grid info from Phase 1
    if GRID_INFO_PATH.exists():
        with open(GRID_INFO_PATH) as f:
            grid_info = json.load(f)
        n_plants = grid_info['n_plants']
        print(f"  Grid: {grid_info['grid_nx']}x{grid_info['grid_ny']}, "
              f"{n_plants} plants")
    else:
        print(f"  WARNING: grid_info.json not found, using single plant")
        n_plants = 1

    # Create ObjectFields
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

    # Radiative budget products (per-triangle)
    products = simu.core.phase.Phase.DartProduct.dartModuleProducts.CommonProducts
    products.radiativeBudgetProducts = 1
    products.radiativeBudgetProperties.budget3DParSurface = 1

    # Engine: Lux
    simu.core.phase.Phase.accelerationEngine = 2
    simu.core.phase.Phase.ExpertModeZone.nbThreads = DART_THREADS

    # Atmosphere: MIDLATSUM
    configure_atmosphere_midlatsum(simu)

    simu.write(overwrite=True)

    # Write field file into simulation input directory
    simu_path = Path(str(simu.simu_dir))
    field_path = simu_path / 'input' / FIELD_FILENAME
    if GRID_INFO_PATH.exists():
        with open(GRID_INFO_PATH) as f:
            grid_info = json.load(f)
        with open(field_path, 'w') as f:
            f.write('complete transformation\n')
            for x, y in grid_info['positions_m']:
                f.write(f'0 {x:.6f} {y:.6f} 0.0 1.0 1.0 1.0 0.0 0.0 0.0\n')
        print(f"  Field file: {field_path} ({grid_info['n_plants']} positions)")
    else:
        # Fallback: single plant at PLANT_POS
        with open(field_path, 'w') as f:
            f.write('complete transformation\n')
            f.write(f'0 {PLANT_POS[0]:.6f} {PLANT_POS[1]:.6f} 0.0 '
                    f'1.0 1.0 1.0 0.0 0.0 0.0\n')
        print(f"  Field file: {field_path} (single plant fallback)")

    print(f"  Written: {simu.simu_dir}")
    return simu


# ============================================================================
# Step 2: Create _II DART simulation (thermal, 1 band)
# ============================================================================
def step2_create_simu_II(simu_I):
    """Create thermal DART simulation by copying _I and reconfiguring for TIR.

    Baleno needs a _II simulation for iterative thermal emission. We copy the
    _I simulation structure and modify:
    - phase.xml: single thermal band at 10µm
    - object_3d.xml: enable useTemperaturePerTriangle=1 with per-group temp files
    - Create initial temperature files (298.15 K) in INPUT directory
    """
    print("\n" + "=" * 70)
    print("STEP 2: Create DART Simulation _II (Thermal)")
    print("=" * 70)

    simu_I_dir = Path(str(simu_I.simu_dir))
    simu_II_dir = simu_I_dir.parent / f'{DART_SIMU_NAME}_II'

    if simu_II_dir.exists():
        shutil.rmtree(str(simu_II_dir))

    # Copy _I to _II
    shutil.copytree(str(simu_I_dir), str(simu_II_dir))
    print(f"  Copied _I to _II: {simu_II_dir}")

    # --- Modify phase.xml: single thermal band ---
    phase_xml = simu_II_dir / 'input' / 'phase.xml'
    if phase_xml.exists():
        content = phase_xml.read_text()

        band_count = len(re.findall(r'<SpectralIntervalsProperties[^/]*/>', content))
        print(f"  Original bands: {band_count}")

        if band_count > 0:
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
            content = '\n'.join(new_lines)

        phase_xml.write_text(content)
        print(f"  Modified phase.xml: 1 thermal band at 10µm ± 2µm")
    else:
        print(f"  WARNING: phase.xml not found at {phase_xml}")

    # --- Modify object_3d.xml: enable per-triangle temperature ---
    # Baleno writes per-object temperature files. DART reads them during RT
    # when useTemperaturePerTriangle="1" and TemperaturePerTriangleProperty
    # specifies the filename.
    obj3d_xml = simu_II_dir / 'input' / 'object_3d.xml'
    if obj3d_xml.exists():
        content = obj3d_xml.read_text()

        # Get group names from the OBJ (organ_00 through organ_11)
        tree = ET.parse(str(obj3d_xml))
        root = tree.getroot()
        groups = root.findall('.//Group')
        print(f"  Found {len(groups)} groups in object_3d.xml")

        for group in groups:
            group_name = group.get('name', 'unknown')
            temp_filename = f'temperature_{group_name}.txt'
            gop = group.find('GroupOpticalProperties')
            if gop is None:
                continue
            sep = gop.find('SurfaceExitanceProperties')
            if sep is None:
                continue

            # Enable per-triangle temperature
            sep.set('useTemperaturePerTriangle', '1')

            # Add TemperaturePerTriangleProperty element if not present
            tptp = sep.find('TemperaturePerTriangleProperty')
            if tptp is None:
                tptp = ET.SubElement(sep, 'TemperaturePerTriangleProperty')
            tptp.set('triangleTemperatureFile', temp_filename)
            print(f"    {group_name}: useTemperaturePerTriangle=1, "
                  f"file={temp_filename}")

        tree.write(str(obj3d_xml), xml_declaration=True, encoding='unicode')
        print(f"  Modified object_3d.xml: per-triangle temperature enabled")

    # --- Create initial temperature files in INPUT directory ---
    # Each group needs a temperature file with one value per triangle.
    # Initial temperature = 298.15 K (25°C). Baleno will overwrite these.
    input_dir = simu_II_dir / 'input'
    with open(REINDEX_JSON) as f:
        reindex = json.load(f)

    for gi_str, obj_indices in reindex['dart_to_obj'].items():
        gi = int(gi_str)
        group_name = reindex['group_names'][gi] if gi < len(reindex['group_names']) else f'group_{gi}'
        temp_filename = f'temperature_{group_name}.txt'
        n_triangles = len(obj_indices)
        temp_values = np.full(n_triangles, 298.15)
        np.savetxt(str(input_dir / temp_filename), temp_values, fmt='%.2f')
        print(f"    Created {temp_filename}: {n_triangles} triangles at 298.15 K")

    print(f"  Initial temperature files created in {input_dir}")

    # Ensure field file exists in _II input directory (copied from _I)
    field_file_II = simu_II_dir / 'input' / FIELD_FILENAME
    if not field_file_II.exists():
        field_file_I = simu_I_dir / 'input' / FIELD_FILENAME
        if field_file_I.exists():
            shutil.copy2(str(field_file_I), str(field_file_II))
            print(f"  Copied field file to _II")

    # Clean output from _I copy (Baleno will regenerate)
    output_dir = simu_II_dir / 'output'
    if output_dir.exists():
        shutil.rmtree(str(output_dir))
        output_dir.mkdir()
        print(f"  Cleaned output directory")

    print(f"  Simulation _II: {simu_II_dir}")
    return simu_II_dir


# ============================================================================
# Step 3: Create Baleno simulation directory + JSON5 configs
# ============================================================================
def step3_create_baleno_configs():
    """Create Baleno simulation directory with all JSON5 config files."""
    print("\n" + "=" * 70)
    print("STEP 3: Create Baleno Simulation Configs")
    print("=" * 70)

    baleno_sim_dir = BALENO_USER_DATA / 'simulations' / SIMU_NAME_EB
    input_dir = baleno_sim_dir / 'input'
    plugins_dir = input_dir / 'plugins'

    # Clean and recreate
    if baleno_sim_dir.exists():
        shutil.rmtree(str(baleno_sim_dir))
    input_dir.mkdir(parents=True)
    plugins_dir.mkdir(parents=True)

    # --- Main JSON5 configs ---

    # atmosphere.json5
    _write_json5(input_dir / 'atmosphere.json5', {
        "z": 10,
        "Ta": 298.15,      # 25°C in Kelvin
        "p": 1013,          # hPa (standard atmosphere)
        "ea": 15,           # hPa (water vapor pressure)
        "u": 2,             # m/s wind speed
        "Ca": 400,          # ppm CO2
        "Oa": 280,          # per mille O2
    })

    # vegetation.json5 — C4 maize with mean PROSPECT params across positions
    # Per-position spectral effects are already in the DART OPs; Baleno's SCOPE
    # config uses mean Cab/N.  The iterative Tuzet loop uses CPlantBox's
    # per-segment Vcmax anyway.
    import numpy as np
    mean_cab = float(np.mean([p["Cab"] for p in per_pos_params]))
    mean_n = float(np.mean([p["N"] for p in per_pos_params]))
    base_params = get_prospect_params(55)
    _write_json5(input_dir / 'vegetation.json5', {
        "Plugin": "BiochemicalSCOPE",
        "Model": "VegetationSCOPE",
        "PAR_min": 0.400,
        "PAR_max": 0.700,
        "Cab": round(mean_cab, 1),
        "Cca": 10,
        "Cs": 0,
        "Cw": base_params["Cw"],
        "Cdm": base_params["Cm"],
        "N": round(mean_n, 2),
        "fqe": 0,
        "Vcmax25": round(vcmax25_from_cab(mean_cab), 1),
        "BallBerrySlope": 8,
        "BallBerry0": 0.01,
        "RdPerVcmax25": get_species()["rd_per_vcmax25"],
        "Type": get_species()["photo_type"],
        "rho_thermal": 0.01,
        "tau_thermal": 0.01,
        "stress_factor": 1,
    })
    print(f"  Baleno vegetation.json5: mean Cab={mean_cab:.1f}, N={mean_n:.2f}")

    # radiation.json5
    _write_json5(input_dir / 'radiation.json5', {
        "Plugin": "DART",
        "Model": "DART",
    })

    # scene.json5
    _write_json5(input_dir / 'scene.json5', {
        "Plugin": "DART",
        "scene_reader": "DARTSceneTriangleReader",
    })

    # soil.json5
    _write_json5(input_dir / 'soil.json5', {
        "Plugin": "SoilMod",
        "Model": "KustasModel",
        "rs_thermal": 0.06,
        "SMC": 0.25,
    })

    # aerodynamics.json5
    _write_json5(input_dir / 'aerodynamics.json5', {
        "Plugin": "AerodynamicsSCOPE",
        "Model": "AeroSCOPE",
        "Cd": 0.3,
        "rwc": 0,
        "rbs": 10.0,
        "CR": 0.35,
        "CD1": 20.6,
        "Psicor": 0.2,
        "CSSOIL": 0.01,
        "Monin_Obukhov_correction": True,
    })

    # modelling_parameters.json5
    _write_json5(input_dir / 'modelling_parameters.json5', {
        "EB_error": 5,
        "max_iteration": 20,
        "min_variation_rate": 0,
        "closure_method": "Newton",
        "load_state": False,
    })

    # output.json5 — enable 3D outputs for all products
    _write_json5(input_dir / 'output.json5', {
        "model": "PhysicsAwareDataWriter",
        "intermediate_outputs": False,
        "save_state": False,
        "radiation": True,
        "vegetation": True,
        "soil": True,
        "aerodynamics": False,
        "energy_balance_products": True,
        "fluxes": True,
        "save_scene": True,
        "delimiter": ";",
        "compute_sunlit": False,
        "sunlit_threshold": 0.5,
        "1 dimension": False,
        "2 dimension": False,
        "3 dimension": True,
        "layer_number": 20,
        "write_yaml": False,
    })

    # time_series.json5
    _write_json5(input_dir / 'time_series.json5', {
        "is_time_series": False,
        "input_filename": "time_series.csv",
        "header_filename": "headers.json",
        "load_from_previous_timestep": False,
        "ts_number": -1,
        "deltat": -1,
    })

    # --- Plugin JSON5 configs ---

    # DART_input.json5
    _write_json5(plugins_dir / 'DART_input.json5', {
        "dart_simulation": DART_SIMU_NAME,
        "Compute_Rn1": True,
        "Compute_broadband": True,
        "Compute_APAR": True,
        "Compute_Rn2": True,
    })

    # BiochemicalSCOPE_input.json5
    _write_json5(plugins_dir / 'BiochemicalSCOPE_input.json5', {
        "Kn0": 2.48,
        "Knalpha": 2.83,
        "Knbeta": 0.114,
        "g_m": "Not computed",
        "kV": 0.6396,
        "apply_T_correction": True,
    })

    # SoilMod_input.json5
    _write_json5(plugins_dir / 'SoilMod_input.json5', {
        "rss": 500,
        "Compute_rss_from_SMC": False,
        "ratio_rn_g": 0.35,
    })

    # AerodynamicsSCOPE_input.json5 (empty — uses defaults, but file must exist)
    _write_json5(plugins_dir / 'AerodynamicsSCOPE_input.json5', {})

    print(f"  Baleno simulation: {baleno_sim_dir}")
    print(f"  Main configs: {len(list(input_dir.glob('*.json5')))} files")
    print(f"  Plugin configs: {len(list(plugins_dir.glob('*.json5')))} files")
    return baleno_sim_dir


def _write_json5(path, data):
    """Write a JSON5-compatible file (JSON with comments support)."""
    import json
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"    Created: {path.name}")


# ============================================================================
# Step 4: Fix .dartrc and write Baleno INI config files
# ============================================================================
def step4_write_config_files():
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

    # user_data_path MUST be empty so both main.py and PathManager
    # resolve to BALENO_DIR/user_data/ (see Baleno path inconsistency bug)
    baleno_config_content = textwrap.dedent(f"""\
        [simulation]
        user_data_path =
        name = {SIMU_NAME_EB}
    """)
    baleno_config_path.write_text(baleno_config_content)
    print(f"  Wrote config.ini: user_data_path=(empty), name={SIMU_NAME_EB}")

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
        else:
            print(f"  WARNING: Could not find is_scene_sorted check to patch")

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
        else:
            print(f"  WARNING: Could not find band loop to patch in spectrum_integration.py")

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
        else:
            print(f"  WARNING: Could not find __get_input_files code to patch in lux_manager.py")

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
# Step 5: Run Baleno as subprocess
# ============================================================================
def step5_run_baleno():
    """Run Baleno (DART-EB) energy balance simulation as subprocess."""
    print("\n" + "=" * 70)
    print("STEP 5: Run Baleno Energy Balance")
    print("=" * 70)

    # Verify prerequisites
    if not VENV_PYTHON.exists():
        print(f"  ERROR: darteb_venv not found at {VENV_PYTHON}")
        return False

    # Build environment for Baleno subprocess
    env = os.environ.copy()
    env['PYTHONPATH'] = str(BALENO_DIR)
    env['DART_HOME'] = str(DART_DIR)
    env['DART_LOCAL'] = str(DART_LOCAL)

    # Remove DART's problematic libreadline from LD_LIBRARY_PATH
    # (DART ships libreadline.so.8.0 which is incompatible with system 8.3)
    ld_path = env.get('LD_LIBRARY_PATH', '')
    dart_lib_paths = [str(DART_DIR / 'bin' / 'python' / 'lib')]
    filtered_ld = ':'.join(
        p for p in ld_path.split(':')
        if p and p not in dart_lib_paths
    )
    env['LD_LIBRARY_PATH'] = filtered_ld

    cmd = [str(VENV_PYTHON), '-m', 'src.main']
    print(f"  Command: {' '.join(cmd)}")
    print(f"  CWD: {BALENO_DIR}")
    print(f"  Python: {VENV_PYTHON}")
    print(f"  Timeout: 3600s")

    try:
        result = subprocess.run(
            cmd,
            cwd=str(BALENO_DIR),
            env=env,
            capture_output=True,
            text=True,
            timeout=3600,
        )

        # Save stdout/stderr for debugging
        log_dir = OUTPUT_DIR / 'baleno_logs'
        log_dir.mkdir(exist_ok=True)
        (log_dir / 'stdout.txt').write_text(result.stdout)
        (log_dir / 'stderr.txt').write_text(result.stderr)

        # Print last 50 lines of stdout
        stdout_lines = result.stdout.strip().split('\n')
        print(f"\n  Baleno output ({len(stdout_lines)} lines):")
        for line in stdout_lines[-50:]:
            print(f"    {line}")

        if result.returncode != 0:
            print(f"\n  ERROR: Baleno exited with code {result.returncode}")
            stderr_lines = result.stderr.strip().split('\n')
            print(f"  Stderr ({len(stderr_lines)} lines):")
            for line in stderr_lines[-30:]:
                print(f"    {line}")
            return False

        print(f"\n  Baleno completed successfully!")
        return True

    except subprocess.TimeoutExpired:
        print(f"  ERROR: Baleno timed out after 3600s")
        return False
    except Exception as e:
        print(f"  ERROR: {e}")
        return False


# ============================================================================
# Step 6: Read Baleno 3D output CSVs
# ============================================================================
def step6_read_baleno_outputs():
    """Parse Baleno's 3D output CSVs from final_results/ and scene from output/."""
    print("\n" + "=" * 70)
    print("STEP 6: Read Baleno Outputs")
    print("=" * 70)

    output_base = BALENO_USER_DATA / 'simulations' / SIMU_NAME_EB / 'output'
    results_dir = output_base / 'final_results'

    if not results_dir.exists():
        print(f"  ERROR: Results directory not found: {results_dir}")
        if output_base.exists():
            print(f"  Contents of output/:")
            for f in sorted(output_base.rglob('*')):
                if f.is_file():
                    print(f"    {f.relative_to(output_base)} ({f.stat().st_size:,} bytes)")
        return None

    print(f"  Results directory: {results_dir}")
    for f in sorted(results_dir.glob('*')):
        print(f"    {f.name} ({f.stat().st_size:,} bytes)")

    outputs = {}

    # Scene file: Baleno writes scene data to output/scene (semicolon-delimited,
    # no .csv extension), NOT to final_results/scene.csv.
    scene_file = None
    for candidate in [
        output_base / 'scene',               # Baleno default location
        results_dir / 'scene.csv',            # fallback
        results_dir / 'scene',                # fallback
    ]:
        if candidate.exists() and candidate.stat().st_size > 0:
            scene_file = candidate
            break

    delimiter = ';'  # Baleno default
    if scene_file is not None:
        delimiter = _detect_delimiter(scene_file)
        print(f"  Scene file: {scene_file}")
        header, data = _read_baleno_csv(scene_file, delimiter)
        outputs['scene'] = {'header': header, 'data': data, '_path': str(scene_file)}
        print(f"  scene: {data.shape[0]} rows x {data.shape[1]} cols")
        print(f"    Columns: {header}")
    else:
        print(f"  WARNING: scene file not found in output/ or final_results/")

    print(f"  Detected delimiter: {repr(delimiter)}")

    # Parse remaining output files from final_results/
    file_map = {
        'energy_balance': 'energy_balance_3D.csv',
        'radiation': 'radiation_3D.csv',
        'heat_fluxes': 'heat_fluxes_3D.csv',
        'vegetation': 'vegetation_3D.csv',
    }

    for key, filename in file_map.items():
        filepath = results_dir / filename
        if filepath.exists():
            header, data = _read_baleno_csv(filepath, delimiter)
            outputs[key] = {'header': header, 'data': data}
            print(f"  {filename}: {data.shape[0]} rows x {data.shape[1]} cols")
            print(f"    Columns: {header}")
        else:
            print(f"  WARNING: {filename} not found")

    return outputs


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
    """Read a Baleno output CSV with header."""
    with open(filepath) as f:
        header_line = f.readline().strip()
    header = [h.strip() for h in header_line.split(delimiter)]

    # Try to read as numeric, falling back to string for scene columns
    try:
        data = np.genfromtxt(
            str(filepath), skip_header=1, delimiter=delimiter,
            dtype=float, filling_values=np.nan,
        )
    except ValueError:
        # Scene file may have string columns (DART_NAME)
        data = np.genfromtxt(
            str(filepath), skip_header=1, delimiter=delimiter,
            dtype=str,
        )

    return header, data


# ============================================================================
# Step 7: Map Baleno rows to OBJ face indices
# ============================================================================
def step7_build_mapping(outputs):
    """Build mapping from Baleno row index → OBJ face index.

    Baleno's scene is sorted by lexsort(INDEX_OBJECT, DART_NAME, TYPE_ID).
    The scene.csv contains these columns. We match vegetation triangles
    to .ori group indices by DART_NAME, then use INDEX_OBJECT as the
    triangle position within the .ori group to get the OBJ face index.
    """
    print("\n" + "=" * 70)
    print("STEP 7: Build Baleno→OBJ Index Mapping")
    print("=" * 70)

    # Load Phase 1 reindex data
    with open(REINDEX_JSON) as f:
        reindex = json.load(f)

    dart_to_obj = {}
    for gi_str, obj_indices in reindex['dart_to_obj'].items():
        dart_to_obj[int(gi_str)] = np.array(obj_indices, dtype=np.int64)

    group_names = reindex['group_names']
    group_offsets = {g: reindex['group_offsets'][g] for g in group_names}

    # Read scene data
    if 'scene' not in outputs:
        print("  ERROR: No scene.csv in Baleno outputs!")
        return None

    scene_header = outputs['scene']['header']
    scene_data = outputs['scene']['data']
    print(f"  Scene: {scene_data.shape[0]} triangles")
    print(f"  Scene columns: {scene_header}")

    # Find column indices
    # SceneColumns: TYPE_ID=0, DART_NAME=1, INDEX_OBJECT=2, GLOBAL_INDEX=3,
    #               SUNLIT=4, SURFACE=5, HEIGHT_GROUND=6
    col_type_id = scene_header.index('TYPE_ID') if 'TYPE_ID' in scene_header else 0
    col_dart_name = scene_header.index('DART_NAME') if 'DART_NAME' in scene_header else 1
    col_index_obj = scene_header.index('INDEX_OBJECT') if 'INDEX_OBJECT' in scene_header else 2
    col_surface = scene_header.index('SURFACE') if 'SURFACE' in scene_header else 5

    # Re-read scene as string to get DART_NAME correctly
    # Use the path resolved by step6 (stored in outputs), fall back to search
    scene_path = outputs['scene'].get('_path') if 'scene' in outputs else None
    if scene_path is None:
        # Search for scene file
        output_base = BALENO_USER_DATA / 'simulations' / SIMU_NAME_EB / 'output'
        for candidate in [output_base / 'scene', output_base / 'final_results' / 'scene.csv']:
            if candidate.exists():
                scene_path = str(candidate)
                break
    scene_path = Path(scene_path)
    delimiter = _detect_delimiter(scene_path)
    scene_str = np.genfromtxt(
        str(scene_path), skip_header=1, delimiter=delimiter, dtype=str,
    )

    # Extract columns
    type_ids = scene_str[:, col_type_id].astype(float).astype(int)
    dart_names = scene_str[:, col_dart_name]
    index_in_object = scene_str[:, col_index_obj].astype(float).astype(int)
    surfaces = scene_str[:, col_surface].astype(float)

    # Load center plant group info from the already-loaded reindex
    center_groups = None
    n_groups_per_plant = len(group_names)
    if 'center_groups' in reindex:
        center_groups = set(reindex['center_groups'])
        n_groups_per_plant = reindex.get('n_groups_per_plant', n_groups_per_plant)
        print(f"  Center plant groups: {sorted(center_groups)}")

    # Identify leaf triangles (type_id >= 100 or == 5)
    leaf_mask = (type_ids >= 100) | (type_ids == 5)
    n_leaf = np.sum(leaf_mask)
    n_total = len(type_ids)
    print(f"  Leaf triangles: {n_leaf} / {n_total}")

    # Group DART names for leaf triangles
    leaf_dart_names = set(dart_names[leaf_mask])
    print(f"  Unique leaf DART names: {sorted(leaf_dart_names)}")

    # Count triangles per DART name (leaf only)
    dart_name_counts = {}
    for dn in leaf_dart_names:
        dart_name_counts[dn] = np.sum(dart_names == dn)

    # Count triangles per .ori group
    ori_group_counts = {gi: len(arr) for gi, arr in dart_to_obj.items()}

    # Match DART names to .ori group indices.
    # For ObjectFields (multi-plant), DART_NAME contains instance/group info:
    #   fo0_moX_goY  where X=instance, Y=group (matching .ori numbering)
    # For single-plant, DART_NAME is the group name: organ_00, organ_01, ...
    sample_dn = next(iter(leaf_dart_names), '')
    is_object_fields = '_mo' in sample_dn and '_go' in sample_dn

    dart_name_to_ori = {}
    center_dart_names = set()

    if is_object_fields:
        # ---- Multi-plant: parse mo/go from DART_NAME directly ----
        grid_info = {}
        if GRID_INFO_PATH.exists():
            with open(GRID_INFO_PATH) as gf:
                grid_info = json.load(gf)
        center_idx = grid_info.get('center_plant_idx', 4)

        for dn in leaf_dart_names:
            mo_m = re.search(r'_mo(\d+)', dn)
            go_m = re.search(r'_go(\d+)', dn)
            if mo_m and go_m:
                instance = int(mo_m.group(1))
                group = int(go_m.group(1))
                if group in ori_group_counts:
                    dart_name_to_ori[dn] = group
                    if instance == center_idx:
                        center_dart_names.add(dn)

        n_instances = len(set(
            int(re.search(r'_mo(\d+)', dn).group(1))
            for dn in dart_name_to_ori if re.search(r'_mo(\d+)', dn)
        ))
        print(f"\n  ObjectFields detected: {n_instances} instances, "
              f"center={center_idx}")
        print(f"  DART name → .ori group mapping "
              f"({len(dart_name_to_ori)} total, "
              f"{len(center_dart_names)} center plant):")
        for dn in sorted(center_dart_names):
            gi = dart_name_to_ori[dn]
            gname = group_names[gi] if gi < len(group_names) else f'?{gi}'
            print(f"    {dn} -> triangle{gi}.ori ({gname})")
    else:
        # ---- Single-plant: greedy match by triangle count ----
        used_ori = set()
        for dn in sorted(dart_name_counts.keys()):
            dn_count = dart_name_counts[dn]
            for gi in sorted(ori_group_counts.keys()):
                if gi in used_ori:
                    continue
                if dn_count == ori_group_counts[gi]:
                    dart_name_to_ori[dn] = gi
                    used_ori.add(gi)
                    break

        print(f"\n  DART name → .ori group mapping "
              f"({len(dart_name_to_ori)} matched):")
        for dn, gi in sorted(dart_name_to_ori.items(), key=lambda x: x[1]):
            gname = group_names[gi] if gi < len(group_names) else f'?{gi}'
            print(f"    {dn} -> triangle{gi}.ori ({gname}, "
                  f"{dart_name_counts[dn]} tris)")

        center_dart_names = set(dart_name_to_ori.keys())

    # Unmatched DART names (likely ground/soil objects)
    unmatched = leaf_dart_names - set(dart_name_to_ori.keys())
    if unmatched:
        print(f"  Unmatched leaf DART names: {sorted(unmatched)}")

    # Build per-row mapping: baleno_row → OBJ face index
    # Only map center plant's DART names so we aggregate only its data
    baleno_to_obj = np.full(n_total, -1, dtype=np.int64)
    for row_idx in range(n_total):
        if not leaf_mask[row_idx]:
            continue
        dn = dart_names[row_idx]
        if dn not in center_dart_names:
            continue
        if dn not in dart_name_to_ori:
            continue
        gi = dart_name_to_ori[dn]
        idx = index_in_object[row_idx]
        ori_arr = dart_to_obj[gi]
        if idx < len(ori_arr):
            baleno_to_obj[row_idx] = ori_arr[idx]

    n_mapped = np.sum(baleno_to_obj >= 0)
    n_center_leaf = sum(1 for i in range(n_total)
                        if leaf_mask[i] and dart_names[i] in center_dart_names)
    print(f"\n  Center plant leaf triangles: {n_center_leaf}")
    print(f"  Mapped: {n_mapped} / {n_center_leaf} center plant leaf triangles "
          f"({100*n_mapped/max(n_center_leaf, 1):.1f}%)")

    return {
        'baleno_to_obj': baleno_to_obj,
        'leaf_mask': leaf_mask,
        'surfaces': surfaces,
        'type_ids': type_ids,
    }


# ============================================================================
# Step 8: Aggregate per-triangle → per-segment
# ============================================================================
def step8_aggregate(outputs, mapping_info):
    """Aggregate Baleno per-triangle results to CPlantBox per-segment arrays."""
    print("\n" + "=" * 70)
    print("STEP 8: Aggregate Per-Triangle → Per-Segment")
    print("=" * 70)

    # Load Phase 1 segment→triangle mapping
    with open(MAPPING_JSON) as f:
        seg_mapping = json.load(f)

    baleno_to_obj = mapping_info['baleno_to_obj']
    leaf_mask = mapping_info['leaf_mask']
    surfaces = mapping_info['surfaces']

    # Build reverse mapping: OBJ face → Baleno row
    obj_to_baleno = {}
    for row_idx in range(len(baleno_to_obj)):
        obj_face = baleno_to_obj[row_idx]
        if obj_face >= 0:
            obj_to_baleno[int(obj_face)] = row_idx

    print(f"  OBJ→Baleno reverse mapping: {len(obj_to_baleno)} entries")

    # Extract per-triangle arrays from Baleno outputs
    # Energy balance: TEMPERATURE (col 1), EB_ERROR (col 2)
    eb_data = outputs.get('energy_balance', {}).get('data')
    eb_header = outputs.get('energy_balance', {}).get('header', [])

    # Radiation: RADIATIVE_BUDGET_I, _II, _TOT, ABSORPTION_PAR, SUNLIT, TIR_EMISSIVITY
    rad_data = outputs.get('radiation', {}).get('data')
    rad_header = outputs.get('radiation', {}).get('header', [])

    # Heat fluxes: LATENT, SENSIBLE, GROUND_HEAT, SURFACE_RESISTANCE
    flux_data = outputs.get('heat_fluxes', {}).get('data')
    flux_header = outputs.get('heat_fluxes', {}).get('header', [])

    # Vegetation: STOMATAL_RESISTANCE, ..., NET_PHOTOSYNTHESIS, ...
    veg_data = outputs.get('vegetation', {}).get('data')
    veg_header = outputs.get('vegetation', {}).get('header', [])

    # Helper to find column index
    def _col(header, name, default=-1):
        for i, h in enumerate(header):
            if name.lower() in h.lower():
                return i
        return default

    # Identify key columns
    col_temp = _col(eb_header, 'TEMPERATURE', 1)
    col_eb_err = _col(eb_header, 'ERROR', 2)
    col_apar = _col(rad_header, 'ABSORPTION_PAR', 4)
    col_rn_tot = _col(rad_header, 'RADIATIVE_BUDGET_TOT', 3)
    col_le = _col(flux_header, 'LATENT', 1)
    col_h = _col(flux_header, 'SENSIBLE', 2)
    col_an = _col(veg_header, 'NET_PHOTOSYNTHESIS', 4)

    print(f"  Column indices: Temp={col_temp}, aPAR={col_apar}, "
          f"Rn={col_rn_tot}, LE={col_le}, H={col_h}, An={col_an}")

    # Aggregate per segment
    segment_results = []
    for organ in seg_mapping['organs']:
        for seg in organ['segments']:
            tri_indices = seg['triangle_indices']  # global OBJ face indices
            if not tri_indices:
                continue

            # Collect Baleno rows for this segment's triangles
            baleno_rows = []
            tri_surfaces = []
            for tidx in tri_indices:
                if tidx in obj_to_baleno:
                    row = obj_to_baleno[tidx]
                    baleno_rows.append(row)
                    tri_surfaces.append(surfaces[row])

            if not baleno_rows:
                segment_results.append({
                    'organ': organ['name'],
                    'organ_type': organ['type'],
                    'segment_idx': seg['segment_idx'],
                    'n_triangles': len(tri_indices),
                    'n_matched': 0,
                    'Tleaf_K': np.nan,
                    'Tleaf_C': np.nan,
                    'APAR_umol': np.nan,
                    'Rn_Wm2': np.nan,
                    'LE_Wm2': np.nan,
                    'H_Wm2': np.nan,
                    'An_umol': np.nan,
                    'EB_error': np.nan,
                })
                continue

            rows = np.array(baleno_rows)
            weights = np.array(tri_surfaces)
            if np.sum(weights) > 0:
                weights = weights / np.sum(weights)
            else:
                weights = np.ones(len(rows)) / len(rows)

            # Area-weighted mean for each variable
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

            segment_results.append({
                'organ': organ['name'],
                'organ_type': organ['type'],
                'segment_idx': seg['segment_idx'],
                'n_triangles': len(tri_indices),
                'n_matched': len(baleno_rows),
                'Tleaf_K': tleaf_k,
                'Tleaf_C': tleaf_c,
                'APAR_umol': _wmean(rad_data, col_apar),
                'Rn_Wm2': _wmean(rad_data, col_rn_tot),
                'LE_Wm2': _wmean(flux_data, col_le),
                'H_Wm2': _wmean(flux_data, col_h),
                'An_umol': _wmean(veg_data, col_an),
                'EB_error': _wmean(eb_data, col_eb_err),
            })

    # Summary
    leaf_segs = [r for r in segment_results if r['organ_type'] == 'leaf']
    leaf_valid = [r for r in leaf_segs if not np.isnan(r['Tleaf_C'])]

    print(f"\n  Total segments: {len(segment_results)}")
    print(f"  Leaf segments: {len(leaf_segs)}")
    print(f"  Leaf segments with valid Tleaf: {len(leaf_valid)} "
          f"({100*len(leaf_valid)/max(len(leaf_segs),1):.1f}%)")

    if leaf_valid:
        temps = np.array([r['Tleaf_C'] for r in leaf_valid])
        apars = np.array([r['APAR_umol'] for r in leaf_valid
                          if not np.isnan(r['APAR_umol'])])
        eb_errs = np.array([abs(r['EB_error']) for r in leaf_valid
                            if not np.isnan(r['EB_error'])])

        print(f"\n  Tleaf (°C): mean={np.mean(temps):.2f}, "
              f"min={np.min(temps):.2f}, max={np.max(temps):.2f}, "
              f"std={np.std(temps):.2f}")
        if len(apars) > 0:
            print(f"  APAR (µmol/m²/s): mean={np.mean(apars):.2f}, "
                  f"min={np.min(apars):.2f}, max={np.max(apars):.2f}")
        if len(eb_errs) > 0:
            print(f"  EB error (W/m²): mean={np.mean(eb_errs):.2f}, "
                  f"max={np.max(eb_errs):.2f}")

    return segment_results


# ============================================================================
# Step 9: Save results
# ============================================================================
def step9_save_results(segment_results):
    """Save per-segment results to CSV and JSON."""
    print("\n" + "=" * 70)
    print("STEP 9: Save Results")
    print("=" * 70)

    leaf_segs = [r for r in segment_results if r['organ_type'] == 'leaf']
    leaf_valid = [r for r in leaf_segs if not np.isnan(r['Tleaf_C'])]

    # --- CSV ---
    csv_path = OUTPUT_DIR / 'maize_day55_baleno_segments.csv'
    with open(csv_path, 'w') as f:
        f.write("organ,organ_type,segment_idx,n_triangles,n_matched,"
                "Tleaf_K,Tleaf_C,APAR_umol_m2_s,Rn_Wm2,"
                "LE_Wm2,H_Wm2,An_umol_CO2_m2_s,EB_error_Wm2\n")
        for r in segment_results:
            f.write(f"{r['organ']},{r['organ_type']},{r['segment_idx']},"
                    f"{r['n_triangles']},{r['n_matched']},"
                    f"{r['Tleaf_K']:.4f},{r['Tleaf_C']:.4f},"
                    f"{r['APAR_umol']:.4f},{r['Rn_Wm2']:.4f},"
                    f"{r['LE_Wm2']:.4f},{r['H_Wm2']:.4f},"
                    f"{r['An_umol']:.4f},{r['EB_error']:.4f}\n")
    print(f"  CSV: {csv_path} ({len(segment_results)} segments)")

    # --- JSON results ---
    results = {
        'phase': 2,
        'description': 'Baleno energy balance per-triangle Tleaf + APAR',
        'baleno_simulation': SIMU_NAME_EB,
        'dart_simulation': DART_SIMU_NAME,
        'n_sw_bands': len(SW_BANDS),
        'tir_band_um': TIR_BAND[0],
        'prospect_params': PROSPECT_PARAMS,
        'atmosphere': {'Ta_K': 298.15, 'p_hPa': 1013, 'ea_hPa': 15,
                       'u_ms': 2, 'Ca_ppm': 400},
        'n_leaf_segments': len(leaf_segs),
        'n_leaf_segments_valid': len(leaf_valid),
        'coverage_pct': 100 * len(leaf_valid) / max(len(leaf_segs), 1),
    }

    # Verification checks
    checks = {}
    if leaf_valid:
        temps = np.array([r['Tleaf_C'] for r in leaf_valid])
        results['Tleaf_stats'] = {
            'mean_C': float(np.mean(temps)),
            'std_C': float(np.std(temps)),
            'min_C': float(np.min(temps)),
            'max_C': float(np.max(temps)),
        }

        # Check 1: Tleaf range (sunlit ~28-35C, shaded ~24-27C with Tair=25C)
        checks['tleaf_range_reasonable'] = bool(
            np.min(temps) > 15 and np.max(temps) < 50
        )

        # Check 2: EB closure
        eb_errs = [abs(r['EB_error']) for r in leaf_valid
                   if not np.isnan(r['EB_error'])]
        if eb_errs:
            mean_err = float(np.mean(eb_errs))
            results['eb_error_mean_Wm2'] = mean_err
            checks['eb_closure_5Wm2'] = mean_err < 5.0

        # Check 3: Coverage
        checks['coverage_95pct'] = results['coverage_pct'] > 95.0

        # Check 4: APAR available
        apars = [r['APAR_umol'] for r in leaf_valid
                 if not np.isnan(r['APAR_umol'])]
        if apars:
            results['APAR_stats'] = {
                'mean_umol': float(np.mean(apars)),
                'min_umol': float(np.min(apars)),
                'max_umol': float(np.max(apars)),
            }
            checks['apar_positive'] = float(np.mean(apars)) > 0

    results['checks'] = checks
    all_pass = all(checks.values()) if checks else False
    results['all_checks_pass'] = all_pass

    json_path = OUTPUT_DIR / 'phase2_results.json'
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  JSON: {json_path}")

    # --- Verdict ---
    print(f"\n{'=' * 70}")
    print(f"PHASE 2 RESULT: {'PASS' if all_pass else 'PARTIAL'}")
    print(f"{'=' * 70}")
    for check_name, passed in checks.items():
        print(f"  {check_name}: {'PASS' if passed else 'FAIL'}")

    if all_pass:
        print(f"\n  Baleno energy balance completed successfully.")
        print(f"  Per-segment Tleaf and APAR available for CPlantBox coupling.")
    else:
        print(f"\n  Some checks failed — review results.")

    return all_pass


# ============================================================================
# Reusable API for diurnal loop
# ============================================================================

def setup_baleno_full(obj_path, mapping_json, reindex_json, grid_info_path,
                      prospect_params, sun_zenith, sun_azimuth,
                      scene_size=(4, 4), plant_pos=(2.0, 2.0),
                      dart_simu_name='cpb_maize_diurnal_eb',
                      baleno_simu_name='cpb_diurnal_eb',
                      field_filename='plant_field.txt'):
    """Create _I, _II DART simulations and Baleno config files.

    This is the reusable counterpart of steps 1-4. Returns paths and backup
    info needed for running and cleaning up.

    Args:
        obj_path: Path to DART-convention OBJ.
        mapping_json: Path to DART mapping JSON.
        reindex_json: Path to Phase 1 reindex JSON.
        grid_info_path: Path to grid_info.json.
        prospect_params: PROSPECT parameter dict.
        sun_zenith, sun_azimuth: Initial sun angles.
        scene_size: Scene size in meters.
        plant_pos: Center plant position.
        dart_simu_name: Name for the DART _I/_II container.
        baleno_simu_name: Name for the Baleno simulation.
        field_filename: Name of the plant field file.

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
    simu_I.core.phase.Phase.ExpertModeZone.nbThreads = DART_THREADS

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
    backups = step4_write_config_files()

    return {
        'simu_I': simu_I,
        'simu_II_dir': simu_II_dir,
        'baleno_sim_dir': baleno_sim_dir,
        'backups': backups,
        'dart_simu_name': dart_simu_name,
        'baleno_simu_name': baleno_simu_name,
    }


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

    _write_json5(input_dir / 'atmosphere.json5', {
        "z": 10, "Ta": 298.15, "p": 1013, "ea": 15, "u": 2,
        "Ca": 400, "Oa": 280,
    })
    # Use mean Cab/N across per-position LOPS profiles
    import numpy as _np
    _per_pos = get_prospect_params_per_position(55, 11)
    _mean_cab = float(_np.mean([p["Cab"] for p in _per_pos]))
    _mean_n = float(_np.mean([p["N"] for p in _per_pos]))
    _base_p = get_prospect_params(55)
    _write_json5(input_dir / 'vegetation.json5', {
        "Plugin": "BiochemicalSCOPE", "Model": "VegetationSCOPE",
        "PAR_min": 0.400, "PAR_max": 0.700,
        "Cab": round(_mean_cab, 1), "Cca": 10, "Cs": 0,
        "Cw": _base_p["Cw"], "Cdm": _base_p["Cm"],
        "N": round(_mean_n, 2), "fqe": 0,
        "Vcmax25": round(vcmax25_from_cab(_mean_cab), 1),
        "BallBerrySlope": 8,
        "BallBerry0": 0.01,
        "RdPerVcmax25": get_species()["rd_per_vcmax25"],
        "Type": get_species()["photo_type"],
        "rho_thermal": 0.01, "tau_thermal": 0.01, "stress_factor": 1,
    })
    _write_json5(input_dir / 'radiation.json5', {"Plugin": "DART", "Model": "DART"})
    _write_json5(input_dir / 'scene.json5', {"Plugin": "DART", "scene_reader": "DARTSceneTriangleReader"})
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
        "EB_error": 5, "max_iteration": 20, "min_variation_rate": 0,
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
        "dart_simulation": dart_simu_name,
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

    for name, runner in [
        ('direction', simu_I.run.direction),
        ('phase', simu_I.run.phase),
        ('dart', simu_I.run.dart),
    ]:
        try:
            runner(timeout=timeout)
        except Exception as e:
            print(f"  Baleno _I {name} error: {e}")


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
    try:
        result = subprocess.run(
            cmd, cwd=str(BALENO_DIR), env=env,
            capture_output=True, text=True, timeout=timeout,
        )
        if result.returncode != 0:
            print(f"  Baleno error (exit {result.returncode})")
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
                print(f"  Baleno exited 0 but output indicates failure:")
                for line in combined.strip().split('\n')[-15:]:
                    print(f"    {line}")
                return False

        return True
    except subprocess.TimeoutExpired:
        print(f"  Baleno timed out after {timeout}s")
        return False
    except Exception as e:
        print(f"  Baleno subprocess error: {e}")
        return False


def read_baleno_tleaf(baleno_sim_dir, mapping_json_path, reindex_json_path,
                      grid_info_path=None, center_plant_idx=4):
    """Read Baleno outputs and aggregate per-triangle Tleaf to per-segment.

    Combines steps 6-8 logic into one call.

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

    delimiter = _detect_delimiter(scene_file)
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
    EB_ERR_MAX = 20.0  # W/m²
    n_rejected = 0

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
                        # Reject non-converged triangles
                        if col_eb_err >= 0 and col_eb_err < eb_data.shape[1]:
                            err = abs(eb_data[row, col_eb_err])
                            if err > EB_ERR_MAX:
                                n_rejected += 1
                                continue
                        temps.append(t - 273.15)  # K → °C
            if temps:
                segment_tleaf.append(float(np.mean(temps)))
            else:
                segment_tleaf.append(25.0)  # fallback

    if n_rejected > 0:
        print(f"  Tleaf: rejected {n_rejected} triangles with |EB_ERROR| > {EB_ERR_MAX} W/m²")

    return np.array(segment_tleaf)


# ============================================================================
# Main
# ============================================================================
def main():
    print("Phase 2: Baleno Energy Balance for Per-Triangle Tleaf")
    print("=" * 70)
    log_consistency(55)

    # Verify prerequisites
    for path, name in [
        (OBJ_PATH, 'Phase 1 OBJ'),
        (MAPPING_JSON, 'Phase 1 mapping'),
        (REINDEX_JSON, 'Phase 1 reindex'),
        (VENV_PYTHON, 'darteb_venv'),
    ]:
        if not path.exists():
            print(f"ERROR: {name} not found: {path}")
            return
    print("  All prerequisites found.\n")

    # Step 1: Create _I simulation
    simu_I = step1_create_simu_I()

    # Step 2: Create _II simulation
    simu_II_dir = step2_create_simu_II(simu_I)

    # Step 3: Create Baleno configs
    baleno_sim_dir = step3_create_baleno_configs()

    # Step 4: Write config files (with backup)
    backups = step4_write_config_files()

    try:
        # Step 5: Run Baleno
        success = step5_run_baleno()
        if not success:
            print("\nBaleno failed. Check logs at output/baleno_logs/")
            return
    finally:
        # Always restore config files
        restore_config_files(backups)

    # Step 6: Read outputs
    outputs = step6_read_baleno_outputs()
    if outputs is None:
        print("\nERROR: Could not read Baleno outputs!")
        return

    # Step 7: Build mapping
    mapping_info = step7_build_mapping(outputs)
    if mapping_info is None:
        print("\nERROR: Could not build Baleno→OBJ mapping!")
        return

    # Step 8: Aggregate
    segment_results = step8_aggregate(outputs, mapping_info)

    # Step 9: Save
    success = step9_save_results(segment_results)

    print(f"\nOutput files:")
    for f in sorted(OUTPUT_DIR.glob('*baleno*')) + sorted(OUTPUT_DIR.glob('phase2*')):
        size = f.stat().st_size
        print(f"  {f.name} ({size:,} bytes)")


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
    _write_json5(input_dir / 'vegetation.json5', {
        "Plugin": "ExternalGS",
        "Model": "VegetationExternalGS",
        "PAR_min": 0.400, "PAR_max": 0.700,
        "Cab": round(_mean_cab, 1), "Cca": 10, "Cs": 0,
        "Cw": _base_p["Cw"], "Cdm": _base_p["Cm"],
        "N": round(_mean_n, 2), "fqe": 0,
        "Vcmax25": round(vcmax25_from_cab(_mean_cab), 1),
        "BallBerrySlope": 8, "BallBerry0": 0.01,
        "RdPerVcmax25": get_species()["rd_per_vcmax25"],
        "Type": get_species()["photo_type"],
        "rho_thermal": 0.01, "tau_thermal": 0.01, "stress_factor": 1,
    })

    # Write plugin-specific input
    plugins_dir = input_dir / 'plugins'
    plugins_dir.mkdir(parents=True, exist_ok=True)
    _write_json5(plugins_dir / 'ExternalGS_input.json5', {
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


if __name__ == '__main__':
    main()
