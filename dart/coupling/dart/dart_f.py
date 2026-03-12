#!/usr/bin/env python3
"""DART-F: Fluorescence Radiative Transfer — Level 2 TOC SIF Radiance.

After Level 1 converges (eta per triangle available), runs a second DART
simulation in fluorescence mode. Uses Fluspect + per-triangle eta file to
trace fluorescence photons through the 3D canopy, producing top-of-canopy
SIF radiance at O2-A (760nm) and O2-B (687nm).
"""

import json
import shutil
import time
import numpy as np
from pathlib import Path

import pytools4dart as ptd

from .. import config as _cfg
from ..config import DART_HOME
from ..prospect_params import get_prospect_params, get_stem_prospect_params
from .simulation import configure_atmosphere_midlatsum, configure_exact_date


# SIF target bands (nm)
SIF_TARGET_687 = 687.0
SIF_TARGET_760 = 760.0

# Spectral range for fluorescence (nm)
SIF_WVL_MIN = 400.0
SIF_WVL_MAX = 850.0
SIF_BAND_WIDTH = 2.0  # nm per band


def write_eta_file(output_path, eta_per_triangle):
    """Write DART-format eta file.

    Format: first line 'N 1 1', then N eta values (one per line).
    Reference: /home/lukas/Downloads/.../eta_27.txt
    """
    output_path = Path(output_path)
    n = len(eta_per_triangle)
    with open(output_path, 'w') as f:
        f.write(f"{n} 1 1\n")
        for v in eta_per_triangle:
            f.write(f"{v:.18e}\n")


def collect_per_triangle_eta(iter_results):
    """Collect flat array of per-triangle eta from iterative coupling results.

    Concatenates all plants' tri_data_raw eta values in order.

    Args:
        iter_results: list of per-plant result dicts from iterative coupling.

    Returns:
        np.ndarray of eta values, one per triangle across all plants.
    """
    all_eta = []
    for r in iter_results:
        if r is None:
            continue
        tri_raw = r.get('tri_data_raw')
        if tri_raw is None:
            continue
        for td in tri_raw:
            all_eta.append(td.get('eta', 0.0))
    return np.array(all_eta, dtype=np.float64)


def create_dart_f_simulation(obj_paths, prospect_params, eta_file_path,
                              calendar_date, hour, minute, lat, lon,
                              scene_size, grid_info=None,
                              mapping_json_paths=None,
                              field_filename='plant_field.txt',
                              fqe=0.01, simu_name=None,
                              day=55):
    """Create DART fluorescence simulation for TOC SIF radiance.

    Mirrors create_dart_simulation_multi() exactly (ground OP, PROSPECT leaf/stem
    OPs, exact date solar geometry, ObjectFields with num/name, field file writing,
    LuxCore params, atmosphere) but adds:
    - ~225 spectral bands (400-850nm at 2nm) for Fluspect spectral resolution
    - fluorescenceProducts=1, useCombinedYield=1 on Coeff_diff
    - isFluorescent=1, FluorescenceYields, WindProfileEta on leaf OP
    - lut.properties fluorescence storage flag

    Args:
        obj_paths: list of OBJ file paths (one per plant).
        prospect_params: dict with PROSPECT parameters (Cab, Car, Cw, Cm, N).
        eta_file_path: Path to DART-format eta file.
        calendar_date: datetime.date for exactDate solar geometry.
        hour, minute: UTC time (int).
        lat, lon: location coordinates.
        scene_size: [x_m, y_m] scene dimensions.
        grid_info: grid info dict with 'positions_m'.
        mapping_json_paths: list of mapping JSON paths (unused, kept for API compat).
        field_filename: field placement file name.
        fqe: fluorescence quantum efficiency (default 0.01).
        simu_name: optional simulation name.
        day: growth day for stem PROSPECT params (default 55).

    Returns:
        pytools4dart simulation object.
    """
    if simu_name is None:
        simu_name = f'cpb_dart_f_{calendar_date}_{hour:02d}{minute:02d}'

    n_plants = len(obj_paths)

    # Clean up previous simulation directory (matches simulation.py:1632-1634)
    simu_dir = Path(ptd.getdartdir()) / 'user_data' / 'simulations' / simu_name
    if simu_dir.exists():
        shutil.rmtree(str(simu_dir))

    simu = ptd.simulation(simu_name, empty=True)
    simu.scene.size = list(scene_size)

    # Spectral bands: 400-850nm at 2nm resolution
    n_bands = int((SIF_WVL_MAX - SIF_WVL_MIN) / SIF_BAND_WIDTH)
    for i in range(n_bands):
        wvl = SIF_WVL_MIN + i * SIF_BAND_WIDTH + SIF_BAND_WIDTH / 2.0
        simu.add.band(wvl=wvl / 1000.0, bw=SIF_BAND_WIDTH / 1000.0)

    # Enable fluorescence products on Coeff_diff
    cd = simu.core.coeff_diff.Coeff_diff
    cd.fluorescenceProducts = 1
    cd.useCombinedYield = 1

    # Solar geometry via exact date (matches simulation.py:1644-1646)
    configure_exact_date(simu, calendar_date, hour, minute, lat=lat, lon=lon)

    # Ground OP (matches simulation.py:1651-1657)
    simu.add.optical_property(
        type='Lambertian', ident='ground',
        databaseName='Lambertian_mineral.db', ModelName='clay_brown',
        useMultiplicativeFactorForLUT=0,
    )
    simu.scene.ground.OpticalPropertyLink.ident = 'ground'

    # Leaf OP via prospect= kwarg (matches simulation.py:1660-1664), then overlay fluorescence
    leaf_op = simu.add.optical_property(
        type='Lambertian', ident='leaf_fluorescent',
        prospect=prospect_params,
        useMultiplicativeFactorForLUT=0,
    )

    # Overlay fluorescence on the ProspectExternalModule created by prospect= kwarg
    pem = leaf_op.Lambertian.ProspectExternalModule
    pem.isFluorescent = 1
    pem.FluorescenceYields = ptd.coeff_diff.create_FluorescenceYields(
        forceYields=1,
        Yield=ptd.coeff_diff.create_Yield(yieldPS=fqe),
    )
    pem.WindProfileEta = ptd.coeff_diff.create_WindProfileEta(
        useBioClimaticWeighting=1,
        BioClimaticWeighting=ptd.coeff_diff.create_BioClimaticWeighting(
            profileFilePath=str(eta_file_path),
        ),
    )

    # Stem OP via prospect= kwarg (matches simulation.py:1665-1670, non-fluorescent)
    stem_prospect = get_stem_prospect_params(day)
    simu.add.optical_property(
        type='Lambertian', ident='stem_bark',
        prospect=stem_prospect,
        useMultiplicativeFactorForLUT=0,
    )

    # Build multi-model ObjectFields (matches simulation.py:1673-1722)
    model_list = ptd.object_3d.create_ModelList()
    for pi, obj_path in enumerate(obj_paths):
        obj_path = Path(obj_path)
        if not obj_path.exists():
            continue

        file_src_fullpath = simu.get_input_file_path(str(obj_path))
        dart_obj = ptd.OBJtools.objreader(file_src_fullpath)
        gnames = ptd.OBJtools.gnames_dart_order(dart_obj.names)
        xdim, ydim, zdim = dart_obj.dims
        xc, yc, zc = dart_obj.center

        groups_list = []
        for gi, gname in enumerate(gnames):
            g = ptd.object_3d.create_Group(num=gi + 1, name=gname)
            is_stem = gname.endswith('_00')
            op_ident = 'stem_bark' if is_stem else 'leaf_fluorescent'
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
            file_src=str(obj_path),
            hasGroups=1,
            GeometricProperties=geom,
            Groups=groups,
            num=pi,
            name=f'CPlantBox_Maize_p{pi}',
            objectDEMMode=0,
        )
        model_list.add_Object(model_obj)

    field = ptd.object_3d.create_Field(
        name='SIF_Field',
        fieldDescriptionFileName=field_filename,
    )
    field.set_ModelList(model_list)
    obj_fields = ptd.object_3d.create_ObjectFields()
    obj_fields.add_Field(field)
    simu.core.object_3d.object_3d.ObjectFields = obj_fields

    # Engine: LuxCore + sampling (matches simulation.py:1730-1734)
    simu.core.phase.Phase.accelerationEngine = 2
    lux = simu.core.phase.Phase.EngineParameter.LuxCoreRenderEngineParameters
    lux.targetRayDensityPerPixel = _cfg.DART_RAY_DENSITY_PER_PIXEL
    lux.maximumRenderingTime = _cfg.DART_MAX_RENDERING_TIME
    lux.pixelSize = 0.05  # 5cm image resolution (LuxCore ignores DartInputParameters)
    simu.core.phase.Phase.ExpertModeZone.nbThreads = _cfg.DART_THREADS

    # Atmosphere: MIDLATSUM (matches simulation.py:1737)
    configure_atmosphere_midlatsum(simu)

    # Write simulation
    simu.write(overwrite=True)

    # Write field file: one model per position (matches simulation.py:1744-1748)
    simu_path = Path(str(simu.simu_dir))
    field_path = simu_path / 'input' / field_filename
    with open(field_path, 'w') as f_out:
        f_out.write('complete transformation\n')
        if grid_info and 'positions_m' in grid_info:
            for idx, (x, y) in enumerate(grid_info['positions_m']):
                f_out.write(f'{idx} {x:.6f} {y:.6f} 0.0 1.0 1.0 1.0 0.0 0.0 0.0\n')

    # Write lut.properties for fluorescence storage
    lut_props_path = simu_path / 'input' / 'lut.properties'
    lut_props_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lut_props_path, 'a') as f:
        f.write('\nlut.store.fluorescence=true\n')

    print(f"  DART-F simulation: {simu_name} ({n_plants} models, {n_bands} bands)")
    return simu


def run_dart_f(simu, timeout=None):
    """Run DART-F simulation (direction + phase + dart).

    Args:
        simu: pytools4dart simulation object.
        timeout: subprocess timeout in seconds. None = no limit.
    """
    print(f"  Running DART-F fluorescence simulation...")
    t0 = time.time()
    try:
        simu.run.full(timeout=timeout)
        elapsed = time.time() - t0
        print(f"  DART-F completed in {elapsed:.1f}s")
        return True
    except Exception as e:
        print(f"  DART-F failed: {e}")
        return False


def _wvl_to_band_idx(target_nm):
    """Convert wavelength [nm] to DART band index (0-based)."""
    return round((target_nm - SIF_WVL_MIN - SIF_BAND_WIDTH / 2.0) / SIF_BAND_WIDTH)


def _find_netcdf(output_dir):
    """Find the latest DART NetCDF image file in output directory."""
    nc_dir = Path(output_dir) / 'netcdf' / 'image'
    if not nc_dir.exists():
        return None
    nc_files = list(nc_dir.glob('image_dart_*.nc'))
    if not nc_files:
        fallback = nc_dir / 'image_dart.nc'
        return fallback if fallback.exists() else None
    def _num(f):
        try:
            return int(f.stem.split('_')[-1])
        except (ValueError, IndexError):
            return -1
    return max(nc_files, key=_num)


def _find_sif_path(f):
    """Auto-detect the SIF fluorescence group path inside an HDF5 file."""
    test_band = _wvl_to_band_idx(SIF_TARGET_760)

    # Try hardcoded paths first (fast path)
    for ima in ['ima001', 'ima002']:
        for rad in ['Reflectance', 'Radiance']:
            path = (f'Fluorescence/{ima}_VZ=000_0_VA=000_0'
                    f'/BOA/ITERX/{rad}/PSI')
            try:
                _ = f[f'{path}/Band_{test_band:03d}/image']
                return path
            except KeyError:
                continue

    # Auto-discover: walk Fluorescence group for any ima with PSI bands
    if 'Fluorescence' not in f:
        return None
    for ima_name in sorted(f['Fluorescence'].keys(), reverse=True):
        if ima_name == 'BRFmap':
            continue
        for iterx in ['ITERX', 'COUPL', 'ITER1', 'ITER2']:
            for rad in ['Reflectance', 'Radiance']:
                path = f'Fluorescence/{ima_name}/BOA/{iterx}/{rad}/PSI'
                try:
                    _ = f[f'{path}/Band_{test_band:03d}/image']
                    return path
                except KeyError:
                    continue
    return None


def _find_reflectance_path(f):
    """Auto-detect the reflectance group path inside an HDF5 file."""
    candidates = []
    # Build candidates from what's actually in the file
    if 'MajorImages' in f:
        for ima_name in f['MajorImages'].keys():
            if ima_name.startswith('ima'):
                for it in ['ITERX', 'ITER1', 'ITER2']:
                    candidates.append(
                        f'MajorImages/{ima_name}/BOA/{it}/Reflectance')
    # Static fallbacks
    for ima in ['ima002', 'ima001']:
        for it in ['ITERX', 'ITER1']:
            p = f'MajorImages/{ima}_VZ=000_0_VA=000_0/BOA/{it}/Reflectance'
            if p not in candidates:
                candidates.append(p)
    test_band = _wvl_to_band_idx(650)
    for path in candidates:
        try:
            _ = f[f'{path}/Band_{test_band:03d}/image']
            return path
        except KeyError:
            continue
    return None


def _get_band_image(f, group_path, band_idx):
    """Extract a 2-D band image from an HDF5 file.

    Returns:
        np.ndarray (float64) or None.
    """
    try:
        key = f'{group_path}/Band_{band_idx:03d}/image'
        data = f[key][:]
        return data.astype(np.float64)
    except KeyError:
        return None


def read_sif_radiance(simu, target_bands_nm=None):
    """Read TOC SIF radiance from DART-F NetCDF output.

    Finds the latest ``image_dart*.nc`` under the simulation output,
    auto-detects the fluorescence group path, and extracts per-pixel
    SIF images at the target wavelengths.

    Uses h5py instead of netCDF4 to avoid libnetcdf-C compatibility issues
    with DART-written HDF5/NetCDF4 files.

    Args:
        simu: pytools4dart simulation object.
        target_bands_nm: list of target wavelengths [nm]. Default [687, 760].

    Returns:
        dict with keys:
            ``SIF_<wvl>_Wm2sr`` — vegetation-mean SIF radiance per band,
            ``SIF_total_Wm2sr`` — sum of all target bands,
            ``sif_images``      — {wvl_nm: 2-D np.ndarray} raw pixel maps,
            ``nc_path``         — Path to the NetCDF file read.
        Empty dict on failure.
    """
    import h5py

    if target_bands_nm is None:
        target_bands_nm = [SIF_TARGET_687, SIF_TARGET_760]

    simu_path = Path(simu.simu_dir)
    output_dir = simu_path / 'output'

    nc_path = _find_netcdf(output_dir)
    if nc_path is None:
        print(f"  No DART-F NetCDF found in {output_dir / 'netcdf' / 'image'}")
        return {}

    try:
        f = h5py.File(str(nc_path), 'r')
    except Exception as e:
        print(f"  Failed to open NetCDF {nc_path}: {e}")
        return {}

    try:
        sif_path = _find_sif_path(f)
        if sif_path is None:
            print(f"  Could not find SIF group in {nc_path.name}")
            return {}

        result = {'nc_path': str(nc_path), 'sif_images': {}}
        for target_nm in target_bands_nm:
            band_idx = _wvl_to_band_idx(target_nm)
            img = _get_band_image(f, sif_path, band_idx)
            key = f'SIF_{int(target_nm)}_Wm2sr'
            if img is not None:
                # Vegetation-pixel mean (positive values only)
                veg = img[img > 0]
                result[key] = float(np.mean(veg)) if veg.size > 0 else 0.0
                result['sif_images'][int(target_nm)] = img
            else:
                result[key] = 0.0

        # Also grab reflectance images if available
        refl_path = _find_reflectance_path(f)
        if refl_path is not None:
            result['reflectance_images'] = {}
            for wvl in [450, 550, 650, 850]:
                bidx = _wvl_to_band_idx(wvl)
                rim = _get_band_image(f, refl_path, bidx)
                if rim is not None:
                    result['reflectance_images'][wvl] = rim

        sif_687 = result.get('SIF_687_Wm2sr', 0.0)
        sif_760 = result.get('SIF_760_Wm2sr', 0.0)
        result['SIF_total_Wm2sr'] = sif_687 + sif_760

        return result

    except Exception as e:
        print(f"  Failed to read DART-F output: {e}")
        import traceback
        traceback.print_exc()
        return {}
    finally:
        f.close()
