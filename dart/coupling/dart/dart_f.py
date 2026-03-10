#!/usr/bin/env python3
"""DART-F: Fluorescence Radiative Transfer — Level 2 TOC SIF Radiance.

After Level 1 converges (eta per triangle available), runs a second DART
simulation in fluorescence mode. Uses Fluspect + per-triangle eta file to
trace fluorescence photons through the 3D canopy, producing top-of-canopy
SIF radiance at O2-A (760nm) and O2-B (687nm).
"""

import json
import time
import numpy as np
from pathlib import Path

import pytools4dart as ptd

from ..config import (DART_HOME, DART_RAY_DENSITY_PER_PIXEL,
                      DART_MAX_RENDERING_TIME, DART_THREADS)
from ..prospect_params import get_prospect_params
from .simulation import configure_atmosphere_midlatsum


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
                              fqe=0.02, simu_name=None):
    """Create DART fluorescence simulation for TOC SIF radiance.

    Key differences from Phase 1 (PAR) simulation:
    - ~225 spectral bands (400-850nm at 2nm) for Fluspect spectral resolution
    - PROSPECT+Fluspect leaf optics: isFluorescent=1
    - FluorescenceYields forceYields=1, combined yield yieldPS=fqe
    - WindProfileEta useBioClimaticWeighting=1, profileFilePath=eta_file
    - Coeff_diff: fluorescenceProducts=1, useCombinedYield=1

    Args:
        obj_paths: list of OBJ file paths (one per plant).
        prospect_params: dict with PROSPECT parameters (Cab, Car, Cw, Cm, N).
        eta_file_path: Path to DART-format eta file.
        calendar_date: date string (YYYY-MM-DD).
        hour, minute: UTC time.
        lat, lon: location coordinates.
        scene_size: [x_m, y_m] scene dimensions.
        grid_info: grid info dict (positions, etc.).
        mapping_json_paths: list of mapping JSON paths.
        field_filename: field placement file name.
        fqe: fluorescence quantum efficiency (default 0.02).
        simu_name: optional simulation name.

    Returns:
        pytools4dart simulation object.
    """
    if simu_name is None:
        simu_name = f'cpb_dart_f_{calendar_date}_{hour:02d}{minute:02d}'

    # Spectral bands: 400-850nm at 2nm resolution
    n_bands = int((SIF_WVL_MAX - SIF_WVL_MIN) / SIF_BAND_WIDTH)

    simu = ptd.simulation(name=simu_name, empty=True)

    # Add spectral bands
    for i in range(n_bands):
        wvl = SIF_WVL_MIN + i * SIF_BAND_WIDTH + SIF_BAND_WIDTH / 2.0
        simu.add.band(wvl=wvl / 1000.0, bw=SIF_BAND_WIDTH / 1000.0)

    # Enable fluorescence products on Coeff_diff
    cd = simu.core.coeff_diff.Coeff_diff
    cd.fluorescenceProducts = 1
    cd.useCombinedYield = 1

    # Add Lambertian optical properties with PROSPECT+Fluspect + fluorescence
    cab = prospect_params.get('Cab', 55.0)
    car = prospect_params.get('Car', 10.0)
    cw = prospect_params.get('Cw', 0.012)
    cm = prospect_params.get('Cm', 0.005)
    n_struct = prospect_params.get('N', 1.8)

    # Create leaf optical property with fluorescence
    op = simu.add.optical_property(
        type='Lambertian',
        ident='leaf_fluorescent',
        databaseName='Lambertian_vegetation.db',
        ModelName='leaf_deciduous',
        useMultiplicativeFactorForLUT=0,
    )

    # Access the Lambertian OP and configure PROSPECT+fluorescence
    lmf_container = cd.Surfaces.LambertianMultiFunctions
    lmf_items = lmf_container.LambertianMulti if lmf_container else []
    if lmf_items and len(lmf_items) > 0:
        lmf = lmf_items[-1]
        lamb = lmf.Lambertian

        # Set up ProspectExternalModule with fluorescence
        prospect_mod = ptd.coeff_diff.create_ProspectExternalModule(
            useProspectExternalModule=1,
            isFluorescent=1,
            ProspectExternParameters=ptd.coeff_diff.create_ProspectExternParameters(
                inputProspectFile='Prospect_Fluspect/Optipar2021_ProspectPRO.txt',
                Cab=cab, Car=car, Cw=cw, Cm=cm, N=n_struct,
            ),
            FluorescenceYields=ptd.coeff_diff.create_FluorescenceYields(
                forceYields=1,
                Yield=ptd.coeff_diff.create_Yield(yieldPS=fqe),
            ),
            WindProfileEta=ptd.coeff_diff.create_WindProfileEta(
                useBioClimaticWeighting=1,
                BioClimaticWeighting=ptd.coeff_diff.create_BioClimaticWeighting(
                    profileFilePath=str(eta_file_path),
                ),
            ),
        )
        lamb.ProspectExternalModule = prospect_mod

    # Add stem optical property (non-fluorescent)
    simu.add.optical_property(
        type='Lambertian',
        ident='stem_bark',
        databaseName='Lambertian_vegetation.db',
        ModelName='bark_deciduous',
        useMultiplicativeFactorForLUT=0,
    )

    # Scene configuration
    simu.core.phase.Phase.calculatorMethod = 0  # forward
    phase = simu.core.phase.Phase
    phase.ExpertModeZone.nbThreads = DART_THREADS

    # LuxCore sampling settings
    lux = phase.EngineParameter.LuxCoreRenderEngineParameters
    lux.targetRayDensityPerPixel = DART_RAY_DENSITY_PER_PIXEL
    lux.maximumRenderingTime = DART_MAX_RENDERING_TIME

    # Scene size and pixel resolution (5 cm)
    simu.scene.size = list(scene_size)
    pixel_size = 0.05  # 5 cm
    phase.DartInputParameters.imageResolution = pixel_size

    # Solar geometry via exact date
    phase.ExpertModeZone.ExpertModeZone_TypeOfIllumination = 0
    ds = simu.core.directions.Directions
    ds.SunViewingAngles.sunViewingAzimuthAngle = 0  # overridden by exact date
    ds.SunViewingAngles.sunViewingZenithAngle = 30  # overridden by exact date

    # Add 3D objects via ObjectFields (with per-group doubleFace)
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

    # Configure atmosphere
    configure_atmosphere_midlatsum(simu)

    # Write simulation
    simu.write(overwrite=True)

    # Write lut.properties for fluorescence storage
    lut_props_path = Path(simu.getsimupath()) / 'input' / 'lut.properties'
    lut_props_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lut_props_path, 'a') as f:
        f.write('\nlut.store.fluorescence=true\n')

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


def _find_sif_path(ds):
    """Auto-detect the SIF fluorescence group path inside a NetCDF dataset."""
    for ima in ['ima002', 'ima001']:
        path = (f'Fluorescence/{ima}_VZ=000_0_VA=000_0'
                f'/BOA/ITERX/Radiance/PSI')
        try:
            test_band = _wvl_to_band_idx(SIF_TARGET_760)
            _ = ds[f'{path}/Band_{test_band:03d}']['image']
            return path
        except (KeyError, IndexError):
            continue
    return None


def _find_reflectance_path(ds):
    """Auto-detect the reflectance group path inside a NetCDF dataset."""
    candidates = []
    # Build candidates from what's actually in the file
    if 'MajorImages' in ds.groups:
        for ima_name in ds.groups['MajorImages'].groups.keys():
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
            _ = ds[f'{path}/Band_{test_band:03d}']['image']
            return path
        except (KeyError, IndexError):
            continue
    return None


def _get_band_image(ds, group_path, band_idx):
    """Extract a 2-D band image from a NetCDF dataset.

    Returns:
        np.ndarray (float64) or None.
    """
    try:
        key = f'{group_path}/Band_{band_idx:03d}'
        data = ds[key]['image'][:]
        if hasattr(data, 'filled'):
            data = data.filled(np.nan)
        return data.astype(np.float64)
    except (KeyError, IndexError):
        return None


def read_sif_radiance(simu, target_bands_nm=None):
    """Read TOC SIF radiance from DART-F NetCDF output.

    Finds the latest ``image_dart*.nc`` under the simulation output,
    auto-detects the fluorescence group path, and extracts per-pixel
    SIF images at the target wavelengths.

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
    from netCDF4 import Dataset

    if target_bands_nm is None:
        target_bands_nm = [SIF_TARGET_687, SIF_TARGET_760]

    simu_path = Path(simu.getsimupath())
    output_dir = simu_path / 'output'

    nc_path = _find_netcdf(output_dir)
    if nc_path is None:
        print(f"  No DART-F NetCDF found in {output_dir / 'netcdf' / 'image'}")
        return {}

    try:
        ds = Dataset(str(nc_path), 'r')
    except Exception as e:
        print(f"  Failed to open NetCDF {nc_path}: {e}")
        return {}

    try:
        sif_path = _find_sif_path(ds)
        if sif_path is None:
            print(f"  Could not find SIF group in {nc_path.name}")
            return {}

        result = {'nc_path': str(nc_path), 'sif_images': {}}
        for target_nm in target_bands_nm:
            band_idx = _wvl_to_band_idx(target_nm)
            img = _get_band_image(ds, sif_path, band_idx)
            key = f'SIF_{int(target_nm)}_Wm2sr'
            if img is not None:
                # Vegetation-pixel mean (positive values only)
                veg = img[img > 0]
                result[key] = float(np.mean(veg)) if veg.size > 0 else 0.0
                result['sif_images'][int(target_nm)] = img
            else:
                result[key] = 0.0

        # Also grab reflectance images if available
        refl_path = _find_reflectance_path(ds)
        if refl_path is not None:
            result['reflectance_images'] = {}
            for wvl in [450, 550, 650, 850]:
                bidx = _wvl_to_band_idx(wvl)
                rim = _get_band_image(ds, refl_path, bidx)
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
        ds.close()
