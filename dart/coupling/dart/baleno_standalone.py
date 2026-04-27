#!/usr/bin/env python3
"""
Legacy standalone Baleno demo (Phase 2 steps 1-9 + main).

Archived from baleno.py during Session 6 cleanup. These functions are NOT
used by the production pipeline (diurnal.py / iterative.py) but are needed
for the CLI command: python -m dart.coupling baleno

Production API remains in baleno.py.
"""

import json
import re
import shutil
import numpy as np
from pathlib import Path
import pytools4dart as ptd
import xml.etree.ElementTree as ET

from .. import config as _cfg
from ..config import (DART_HOME, DART_EB_DIR, DARTRC, OUTPUT_DIR,
                      get_species)
from ..prospect_params import (get_prospect_params, get_prospect_params_per_position,
                               get_stem_prospect_params,
                               get_tassel_prospect_params,
                               get_midrib_prospect_params,
                               log_consistency, log_lops_consistency, vcmax25_from_cab)
from ..dart.simulation import configure_atmosphere_midlatsum
from .parsers import detect_delimiter, read_baleno_csv, write_json5
from .baleno import (write_baleno_config_files, restore_config_files,
                     BALENO_DIR, DART_DIR, DART_LOCAL, DARTRC_PATH,
                     VENV_PYTHON, BALENO_USER_DATA,
                     SW_BANDS, TIR_BAND, SIMU_NAME_EB, DART_SIMU_NAME)

# ---------------------------------------------------------------------------
# Legacy-only constants (not used in production pipeline)
# ---------------------------------------------------------------------------
OBJ_PATH = OUTPUT_DIR / 'maize_day55_dart.obj'
MAPPING_JSON = OUTPUT_DIR / 'maize_day55_dart_mapping.json'
REINDEX_JSON = OUTPUT_DIR / 'maize_day55_reindex.json'
PROSPECT_PARAMS = get_prospect_params(55)
SUN_ZENITH = 45.0
SUN_AZIMUTH = 225.0
SCENE_SIZE = [4.0, 2.25]
PLANT_POS = (SCENE_SIZE[0] / 2, SCENE_SIZE[1] / 2)
GRID_INFO_PATH = OUTPUT_DIR / 'grid_info.json'
FIELD_FILENAME = 'plant_field.txt'


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
    n_leaf_groups = sum(1 for g in gnames
                        if not g.endswith('_00')
                        and not g.endswith('_midrib')
                        and not g.startswith(('tassel_spike_', 'tassel_branch_')))

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
    tassel_prospect = get_tassel_prospect_params(55)
    simu.add.optical_property(
        type='Lambertian', ident='maize_tassel',
        prospect=tassel_prospect,
        useMultiplicativeFactorForLUT=0,
    )

    # Midrib OP — registered on demand.
    _has_midrib = any(g.endswith('_midrib') for g in gnames)
    if _has_midrib:
        midrib_prospect = get_midrib_prospect_params(55)
        simu.add.optical_property(
            type='Lambertian', ident='maize_leaf_midrib',
            prospect=midrib_prospect,
            useMultiplicativeFactorForLUT=0,
        )

    # 3D Object via ObjectFields (same grid as Phase 1)
    xdim, ydim, zdim = obj_info.dims
    xc, yc, zc = obj_info.center

    # Create groups with per-position optical properties + doubleFace.
    # Routing precedence: midrib suffix > tassel prefix > stem (organ_00)
    # > per-position leaf.
    groups_list = []
    leaf_idx = 0
    for gi, gname in enumerate(gnames):
        g = ptd.object_3d.create_Group(num=gi + 1, name=gname)
        if gname.endswith('_midrib'):
            op_ident, df = 'maize_leaf_midrib', 1
        elif gname.startswith(('tassel_spike_', 'tassel_branch_')):
            op_ident, df = 'maize_tassel', 1
        elif gname.endswith('_00'):
            op_ident, df = 'maize_stem', 0
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

    # Engine: Lux + sampling
    simu.core.phase.Phase.accelerationEngine = 2
    lux = simu.core.phase.Phase.EngineParameter.LuxCoreRenderEngineParameters
    lux.targetRayDensityPerPixel = _cfg.DART_RAY_DENSITY_PER_PIXEL
    lux.maximumRenderingTime = _cfg.DART_MAX_RENDERING_TIME
    simu.core.phase.Phase.ExpertModeZone.nbThreads = _cfg.DART_THREADS

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
            for pos in grid_info['positions_m']:
                x, y = pos[0], pos[1]
                yrot = pos[2] if len(pos) > 2 else 0.0
                f.write(f'0 {x:.6f} {y:.6f} 0.0 1.0 1.0 1.0 0.0 {yrot:.2f} 0.0\n')
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
    - phase.xml: single thermal band at 10um
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
        print(f"  Modified phase.xml: 1 thermal band at 10um +/- 2um")
    else:
        print(f"  WARNING: phase.xml not found at {phase_xml}")

    # --- Modify object_3d.xml: enable per-triangle temperature ---
    obj3d_xml = simu_II_dir / 'input' / 'object_3d.xml'
    if obj3d_xml.exists():
        content = obj3d_xml.read_text()

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

            sep.set('useTemperaturePerTriangle', '1')

            tptp = sep.find('TemperaturePerTriangleProperty')
            if tptp is None:
                tptp = ET.SubElement(sep, 'TemperaturePerTriangleProperty')
            tptp.set('triangleTemperatureFile', temp_filename)
            print(f"    {group_name}: useTemperaturePerTriangle=1, "
                  f"file={temp_filename}")

        tree.write(str(obj3d_xml), xml_declaration=True, encoding='unicode')
        print(f"  Modified object_3d.xml: per-triangle temperature enabled")

    # --- Create initial temperature files in INPUT directory ---
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
    write_json5(input_dir / 'atmosphere.json5', {
        "z": 10,
        "Ta": 298.15,      # 25C in Kelvin
        "p": 1013,          # hPa (standard atmosphere)
        "ea": 15,           # hPa (water vapor pressure)
        "u": 2,             # m/s wind speed
        "Ca": 400,          # ppm CO2
        "Oa": 280,          # per mille O2
    })

    # vegetation.json5
    per_pos_params = get_prospect_params_per_position(55, 11)
    mean_cab = float(np.mean([p["Cab"] for p in per_pos_params]))
    mean_n = float(np.mean([p["N"] for p in per_pos_params]))
    base_params = get_prospect_params(55)
    write_json5(input_dir / 'vegetation.json5', {
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
    write_json5(input_dir / 'radiation.json5', {
        "Plugin": "DART",
        "Model": "DART",
    })

    # scene.json5
    write_json5(input_dir / 'scene.json5', {
        "Plugin": "DART",
        "scene_reader": "DARTSceneTriangleReader",
    })

    # soil.json5
    write_json5(input_dir / 'soil.json5', {
        "Plugin": "SoilMod",
        "Model": "KustasModel",
        "rs_thermal": 0.06,
        "SMC": 0.25,
    })

    # aerodynamics.json5
    write_json5(input_dir / 'aerodynamics.json5', {
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
    write_json5(input_dir / 'modelling_parameters.json5', {
        "EB_error": 5,
        "max_iteration": 20,
        "min_variation_rate": 0,
        "closure_method": "Newton",
        "load_state": False,
    })

    # output.json5 -- enable 3D outputs for all products
    write_json5(input_dir / 'output.json5', {
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
    write_json5(input_dir / 'time_series.json5', {
        "is_time_series": False,
        "input_filename": "time_series.csv",
        "header_filename": "headers.json",
        "load_from_previous_timestep": False,
        "ts_number": -1,
        "deltat": -1,
    })

    # --- Plugin JSON5 configs ---

    # DART_input.json5
    write_json5(plugins_dir / 'DART_input.json5', {
        "dart_simulation": DART_SIMU_NAME,
        "Compute_Rn1": True,
        "Compute_broadband": True,
        "Compute_APAR": True,
        "Compute_Rn2": True,
    })

    # BiochemicalSCOPE_input.json5
    write_json5(plugins_dir / 'BiochemicalSCOPE_input.json5', {
        "Kn0": 2.48,
        "Knalpha": 2.83,
        "Knbeta": 0.114,
        "g_m": "Not computed",
        "kV": 0.6396,
        "apply_T_correction": True,
    })

    # SoilMod_input.json5
    write_json5(plugins_dir / 'SoilMod_input.json5', {
        "rss": 500,
        "Compute_rss_from_SMC": False,
        "ratio_rn_g": 0.35,
    })

    # AerodynamicsSCOPE_input.json5 (empty -- uses defaults, but file must exist)
    write_json5(plugins_dir / 'AerodynamicsSCOPE_input.json5', {})

    print(f"  Baleno simulation: {baleno_sim_dir}")
    print(f"  Main configs: {len(list(input_dir.glob('*.json5')))} files")
    print(f"  Plugin configs: {len(list(plugins_dir.glob('*.json5')))} files")
    return baleno_sim_dir


# ============================================================================
# Step 5: Run Baleno as subprocess
# ============================================================================
def step5_run_baleno():
    """Run Baleno (DART-EB) energy balance simulation as subprocess."""
    import os
    import subprocess

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

    # Scene file
    scene_file = None
    for candidate in [
        output_base / 'scene',
        results_dir / 'scene.csv',
        results_dir / 'scene',
    ]:
        if candidate.exists() and candidate.stat().st_size > 0:
            scene_file = candidate
            break

    delimiter = ';'  # Baleno default
    if scene_file is not None:
        delimiter = detect_delimiter(scene_file)
        print(f"  Scene file: {scene_file}")
        header, data = read_baleno_csv(scene_file, delimiter)
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
            header, data = read_baleno_csv(filepath, delimiter)
            outputs[key] = {'header': header, 'data': data}
            print(f"  {filename}: {data.shape[0]} rows x {data.shape[1]} cols")
            print(f"    Columns: {header}")
        else:
            print(f"  WARNING: {filename} not found")

    return outputs


# ============================================================================
# Step 7: Map Baleno rows to OBJ face indices
# ============================================================================
def step7_build_mapping(outputs):
    """Build mapping from Baleno row index -> OBJ face index.

    Baleno's scene is sorted by lexsort(INDEX_OBJECT, DART_NAME, TYPE_ID).
    The scene.csv contains these columns. We match vegetation triangles
    to .ori group indices by DART_NAME, then use INDEX_OBJECT as the
    triangle position within the .ori group to get the OBJ face index.
    """
    print("\n" + "=" * 70)
    print("STEP 7: Build Baleno->OBJ Index Mapping")
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
    col_type_id = scene_header.index('TYPE_ID') if 'TYPE_ID' in scene_header else 0
    col_dart_name = scene_header.index('DART_NAME') if 'DART_NAME' in scene_header else 1
    col_index_obj = scene_header.index('INDEX_OBJECT') if 'INDEX_OBJECT' in scene_header else 2
    col_surface = scene_header.index('SURFACE') if 'SURFACE' in scene_header else 5

    # Re-read scene as string to get DART_NAME correctly
    scene_path = outputs['scene'].get('_path') if 'scene' in outputs else None
    if scene_path is None:
        output_base = BALENO_USER_DATA / 'simulations' / SIMU_NAME_EB / 'output'
        for candidate in [output_base / 'scene', output_base / 'final_results' / 'scene.csv']:
            if candidate.exists():
                scene_path = str(candidate)
                break
    scene_path = Path(scene_path)
    delimiter = detect_delimiter(scene_path)
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
        print(f"  DART name -> .ori group mapping "
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

        print(f"\n  DART name -> .ori group mapping "
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

    # Build per-row mapping: baleno_row -> OBJ face index
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
# Step 8: Aggregate per-triangle -> per-segment
# ============================================================================
def step8_aggregate(outputs, mapping_info):
    """Aggregate Baleno per-triangle results to CPlantBox per-segment arrays."""
    print("\n" + "=" * 70)
    print("STEP 8: Aggregate Per-Triangle -> Per-Segment")
    print("=" * 70)

    # Load Phase 1 segment->triangle mapping
    with open(MAPPING_JSON) as f:
        seg_mapping = json.load(f)

    baleno_to_obj = mapping_info['baleno_to_obj']
    leaf_mask = mapping_info['leaf_mask']
    surfaces = mapping_info['surfaces']

    # Build reverse mapping: OBJ face -> Baleno row
    obj_to_baleno = {}
    for row_idx in range(len(baleno_to_obj)):
        obj_face = baleno_to_obj[row_idx]
        if obj_face >= 0:
            obj_to_baleno[int(obj_face)] = row_idx

    print(f"  OBJ->Baleno reverse mapping: {len(obj_to_baleno)} entries")

    # Extract per-triangle arrays from Baleno outputs
    eb_data = outputs.get('energy_balance', {}).get('data')
    eb_header = outputs.get('energy_balance', {}).get('header', [])
    rad_data = outputs.get('radiation', {}).get('data')
    rad_header = outputs.get('radiation', {}).get('header', [])
    flux_data = outputs.get('heat_fluxes', {}).get('data')
    flux_header = outputs.get('heat_fluxes', {}).get('header', [])
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
            tri_indices = seg['triangle_indices']
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

        print(f"\n  Tleaf (C): mean={np.mean(temps):.2f}, "
              f"min={np.min(temps):.2f}, max={np.max(temps):.2f}, "
              f"std={np.std(temps):.2f}")
        if len(apars) > 0:
            print(f"  APAR (umol/m2/s): mean={np.mean(apars):.2f}, "
                  f"min={np.min(apars):.2f}, max={np.max(apars):.2f}")
        if len(eb_errs) > 0:
            print(f"  EB error (W/m2): mean={np.mean(eb_errs):.2f}, "
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

        # Check 1: Tleaf range
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
        print(f"\n  Some checks failed -- review results.")

    return all_pass


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
    backups = write_baleno_config_files()

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
        print("\nERROR: Could not build Baleno->OBJ mapping!")
        return

    # Step 8: Aggregate
    segment_results = step8_aggregate(outputs, mapping_info)

    # Step 9: Save
    success = step9_save_results(segment_results)

    print(f"\nOutput files:")
    for f in sorted(OUTPUT_DIR.glob('*baleno*')) + sorted(OUTPUT_DIR.glob('phase2*')):
        size = f.stat().st_size
        print(f"  {f.name} ({size:,} bytes)")


if __name__ == '__main__':
    main()
