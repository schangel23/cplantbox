#!/usr/bin/env python3
"""SIF spatial analysis and visualization from DART-F NetCDF output.

Produces TOC SIF radiance maps (F687, F760), F760/F687 ratio maps,
vegetation-masked statistics, and optional sensor-resolution aggregation.

Adapted from SIFVISUJAN.py (bark-beetle SIF analysis pipeline).
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

from ..dart.dart_f import (
    SIF_WVL_MIN, SIF_WVL_MAX, SIF_BAND_WIDTH,
    SIF_TARGET_687, SIF_TARGET_760,
    _find_netcdf, _find_sif_path, _find_reflectance_path,
    _get_band_image, _wvl_to_band_idx,
)


# ---------------------------------------------------------------------------
# Vegetation masking thresholds
# ---------------------------------------------------------------------------
VEG_F760_MIN = 0.0001   # W/m2/sr/um — minimum for vegetation classification
VEG_F687_MIN = 0.00005
RATIO_F760_MIN = 0.0005  # stricter thresholds for ratio maps
RATIO_F687_MIN = 0.0002

# Percentiles for colorbar scaling (avoid outlier-driven colorbars)
SCALE_PCT_LO = 2
SCALE_PCT_HI = 98

# Scene pixel size (must match dart_f.py imageResolution)
DART_PIXEL_SIZE_M = 0.05  # 5 cm

# Diagnostic SIF bands for integration
BANDS = {
    'Red (682-692)': (682, 692),
    'Far-red (755-765)': (755, 765),
}

# Plot style
plt.rcParams.update({
    'font.size': 11,
    'axes.spines.top': False,
    'axes.spines.right': False,
})


# ---------------------------------------------------------------------------
# Vegetation masking
# ---------------------------------------------------------------------------
def create_vegetation_mask(f687, f760, strict=False):
    """Boolean mask: True where pixel is vegetation.

    Args:
        f687, f760: 2-D SIF images.
        strict: use tighter thresholds (for ratio maps).
    """
    if strict:
        return (f760 > RATIO_F760_MIN) & (f687 > RATIO_F687_MIN)
    return (f760 > VEG_F760_MIN) & (f687 > VEG_F687_MIN)


# ---------------------------------------------------------------------------
# Sensor-resolution aggregation
# ---------------------------------------------------------------------------
def aggregate_pixels(data, agg_x, agg_y):
    """Aggregate 2-D array by block-averaging (nanmean).

    Args:
        data: 2-D array at native resolution.
        agg_x, agg_y: number of pixels to average in x and y.

    Returns:
        2-D array at coarser resolution.
    """
    ny, nx = data.shape
    ny_t = (ny // agg_y) * agg_y
    nx_t = (nx // agg_x) * agg_x
    trimmed = data[:ny_t, :nx_t]
    reshaped = trimmed.reshape(ny_t // agg_y, agg_y, nx_t // agg_x, agg_x)
    return np.nanmean(reshaped, axis=(1, 3))


# ---------------------------------------------------------------------------
# Core analysis: load + compute metrics from a single DART-F run
# ---------------------------------------------------------------------------
def load_sif_from_netcdf(nc_path):
    """Load F687, F760, and reflectance images from a DART-F NetCDF file.

    Args:
        nc_path: Path to ``image_dart*.nc``.

    Returns:
        dict with keys 'F687', 'F760', 'reflectance' (dict of wvl->image),
        'nc_path', 'pixel_size_m'.  None on failure.
    """
    import h5py

    nc_path = Path(nc_path)
    if not nc_path.exists():
        print(f"  NetCDF not found: {nc_path}")
        return None

    f = h5py.File(str(nc_path), 'r')
    try:
        sif_path = _find_sif_path(f)
        if sif_path is None:
            print(f"  No SIF group found in {nc_path.name}")
            return None

        result = {
            'nc_path': str(nc_path),
            'pixel_size_m': DART_PIXEL_SIZE_M,
        }

        for label, target_nm in [('F687', SIF_TARGET_687),
                                  ('F760', SIF_TARGET_760)]:
            bidx = _wvl_to_band_idx(target_nm)
            img = _get_band_image(f, sif_path, bidx)
            result[label] = img

        refl_path = _find_reflectance_path(f)
        result['reflectance'] = {}
        if refl_path is not None:
            for wvl in [450, 550, 650, 850]:
                rim = _get_band_image(f, refl_path, _wvl_to_band_idx(wvl))
                if rim is not None:
                    result['reflectance'][wvl] = rim

        return result
    finally:
        f.close()


def compute_sif_metrics(data):
    """Compute scalar SIF metrics from loaded images.

    Args:
        data: dict from ``load_sif_from_netcdf``.

    Returns:
        dict with mean F687, F760, ratio, pixel counts, etc.
    """
    f687 = data.get('F687')
    f760 = data.get('F760')
    if f687 is None or f760 is None:
        return {}

    veg = create_vegetation_mask(f687, f760)
    strict = create_vegetation_mask(f687, f760, strict=True)

    n_total = veg.size
    n_veg = int(np.sum(veg))

    with np.errstate(divide='ignore', invalid='ignore'):
        ratio_img = np.where(strict, f760 / f687, np.nan)

    return {
        'n_pixels_total': n_total,
        'n_pixels_veg': n_veg,
        'frac_veg': round(n_veg / max(n_total, 1), 4),
        'F687_mean': float(np.nanmean(f687[veg])) if n_veg else 0.0,
        'F760_mean': float(np.nanmean(f760[veg])) if n_veg else 0.0,
        'F687_std': float(np.nanstd(f687[veg])) if n_veg else 0.0,
        'F760_std': float(np.nanstd(f760[veg])) if n_veg else 0.0,
        'F760_F687_ratio': float(np.nanmean(ratio_img[strict])) if np.any(strict) else 0.0,
        'pixel_size_m': data.get('pixel_size_m', DART_PIXEL_SIZE_M),
    }


# ---------------------------------------------------------------------------
# Visualization: spatial SIF maps
# ---------------------------------------------------------------------------
def plot_sif_maps(data, output_dir, label='', scene_size_m=None):
    """Plot F687, F760, and F760/F687 ratio maps with vegetation masking.

    Produces:
        ``SIF_F760_{label}.png`` — F760 map
        ``SIF_F687_{label}.png`` — F687 map
        ``SIF_ratio_{label}.png`` — F760/F687 ratio map
        ``SIF_overview_{label}.png`` — 2x2 panel (F760, F687, ratio, RGB/CIR)

    Args:
        data: dict from ``load_sif_from_netcdf``.
        output_dir: directory to write PNGs.
        label: suffix for filenames (e.g. 'day55_1200').
        scene_size_m: [x, y] for axis labels in metres. Auto-detected if None.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    f687 = data.get('F687')
    f760 = data.get('F760')
    if f687 is None or f760 is None:
        print("  No SIF images to plot")
        return

    veg = create_vegetation_mask(f687, f760)
    strict = create_vegetation_mask(f687, f760, strict=True)

    # Mask non-vegetation as NaN (renders white)
    f687_masked = np.where(veg, f687, np.nan)
    f760_masked = np.where(veg, f760, np.nan)

    with np.errstate(divide='ignore', invalid='ignore'):
        ratio_masked = np.where(strict, f760 / f687, np.nan)

    # Axis extent in metres
    pix = data.get('pixel_size_m', DART_PIXEL_SIZE_M)
    ny, nx = f760.shape
    if scene_size_m is not None:
        extent = [0, scene_size_m[0], scene_size_m[1], 0]
    else:
        extent = [0, nx * pix, ny * pix, 0]

    suffix = f'_{label}' if label else ''

    # --- Percentile-based colorbars from vegetation pixels only ---
    veg_f760 = f760_masked[~np.isnan(f760_masked)]
    veg_f687 = f687_masked[~np.isnan(f687_masked)]
    veg_ratio = ratio_masked[~np.isnan(ratio_masked)]

    def _pctl(arr, lo=SCALE_PCT_LO, hi=SCALE_PCT_HI):
        if arr.size == 0:
            return 0, 1
        return float(np.percentile(arr, lo)), float(np.percentile(arr, hi))

    f760_vmin, f760_vmax = _pctl(veg_f760)
    f687_vmin, f687_vmax = _pctl(veg_f687)
    rat_vmin, rat_vmax = _pctl(veg_ratio)

    # --- 2x2 overview panel ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # F760
    im0 = axes[0, 0].imshow(f760_masked, cmap='plasma', extent=extent,
                             vmin=f760_vmin, vmax=f760_vmax)
    axes[0, 0].set_title('F760 (O2-A)')
    fig.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04,
                 label='W/m\u00b2/sr/\u03bcm')

    # F687
    im1 = axes[0, 1].imshow(f687_masked, cmap='viridis', extent=extent,
                             vmin=f687_vmin, vmax=f687_vmax)
    axes[0, 1].set_title('F687 (O2-B)')
    fig.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04,
                 label='W/m\u00b2/sr/\u03bcm')

    # F760/F687 ratio
    im2 = axes[1, 0].imshow(ratio_masked, cmap='RdYlGn', extent=extent,
                             vmin=rat_vmin, vmax=rat_vmax)
    axes[1, 0].set_title('F760 / F687')
    fig.colorbar(im2, ax=axes[1, 0], fraction=0.046, pad=0.04, label='ratio')

    # RGB or CIR composite
    refl = data.get('reflectance', {})
    if 650 in refl and 550 in refl and 450 in refl:
        r, g, b = refl[650], refl[550], refl[450]
        rgb_max = max(r.max(), g.max(), b.max(), 1e-9)
        rgb = np.clip(np.dstack([r, g, b]) / rgb_max, 0, 1)
        axes[1, 1].imshow(rgb, extent=extent)
        axes[1, 1].set_title('RGB (650, 550, 450 nm)')
    elif 850 in refl and 650 in refl and 550 in refl:
        nir, r, g = refl[850], refl[650], refl[550]
        cir_max = max(nir.max(), r.max(), g.max(), 1e-9)
        cir = np.clip(np.dstack([nir, r, g]) / cir_max, 0, 1)
        axes[1, 1].imshow(cir, extent=extent)
        axes[1, 1].set_title('CIR (NIR, Red, Green)')
    else:
        axes[1, 1].text(0.5, 0.5, 'No reflectance data',
                        ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Reflectance')

    for ax in axes.flat:
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')

    metrics = compute_sif_metrics(data)
    fig.suptitle(
        f'DART-F TOC SIF   |   '
        f'F760={metrics.get("F760_mean", 0):.4f}   '
        f'F687={metrics.get("F687_mean", 0):.4f}   '
        f'ratio={metrics.get("F760_F687_ratio", 0):.2f}   '
        f'veg={metrics.get("frac_veg", 0) * 100:.0f}%',
        fontsize=11, fontweight='bold',
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out = output_dir / f'SIF_overview{suffix}.png'
    fig.savefig(out, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"    Saved {out.name}")

    return str(out)


def plot_sif_aggregated(data, output_dir, label='',
                        agg_x=10, agg_y=20, agg_label='0.5m x 1m'):
    """Plot native vs sensor-aggregated SIF comparison.

    Produces a 2-row figure: native (5 cm) on top, aggregated below.

    Args:
        data: dict from ``load_sif_from_netcdf``.
        output_dir: directory to write PNGs.
        label: filename suffix.
        agg_x, agg_y: aggregation factors in x and y.
        agg_label: human-readable label for the coarse resolution.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    f687 = data.get('F687')
    f760 = data.get('F760')
    if f687 is None or f760 is None:
        return

    veg = create_vegetation_mask(f687, f760)
    f760_masked = np.where(veg, f760, np.nan)
    f687_masked = np.where(veg, f687, np.nan)

    f760_agg = aggregate_pixels(f760_masked, agg_x, agg_y)
    f687_agg = aggregate_pixels(f687_masked, agg_x, agg_y)

    # Consistent scaling from native-resolution vegetation pixels
    vals = f760_masked[~np.isnan(f760_masked)]
    if vals.size == 0:
        return
    vmin = float(np.percentile(vals, SCALE_PCT_LO))
    vmax = float(np.percentile(vals, SCALE_PCT_HI))

    aspect_agg = agg_y / agg_x  # display aspect for rectangular pixels

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Native F760
    im0 = axes[0, 0].imshow(f760_masked, cmap='plasma', vmin=vmin, vmax=vmax)
    axes[0, 0].set_title('F760 native (5 cm)')
    axes[0, 0].axis('off')

    # Aggregated F760
    im1 = axes[1, 0].imshow(f760_agg, cmap='plasma', vmin=vmin, vmax=vmax,
                             aspect=aspect_agg)
    axes[1, 0].set_title(f'F760 aggregated ({agg_label})')
    axes[1, 0].axis('off')

    # Native F687
    vals687 = f687_masked[~np.isnan(f687_masked)]
    vmin687 = float(np.percentile(vals687, SCALE_PCT_LO)) if vals687.size else 0
    vmax687 = float(np.percentile(vals687, SCALE_PCT_HI)) if vals687.size else 1

    im2 = axes[0, 1].imshow(f687_masked, cmap='viridis', vmin=vmin687, vmax=vmax687)
    axes[0, 1].set_title('F687 native (5 cm)')
    axes[0, 1].axis('off')

    # Aggregated F687
    im3 = axes[1, 1].imshow(f687_agg, cmap='viridis', vmin=vmin687, vmax=vmax687,
                             aspect=aspect_agg)
    axes[1, 1].set_title(f'F687 aggregated ({agg_label})')
    axes[1, 1].axis('off')

    fig.colorbar(im0, ax=axes[:, 0], fraction=0.03, pad=0.04,
                 label='F760 [W/m\u00b2/sr/\u03bcm]')
    fig.colorbar(im2, ax=axes[:, 1], fraction=0.03, pad=0.04,
                 label='F687 [W/m\u00b2/sr/\u03bcm]')

    suffix = f'_{label}' if label else ''
    fig.suptitle('Native vs Aggregated SIF', fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out = output_dir / f'SIF_native_vs_aggregated{suffix}.png'
    fig.savefig(out, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"    Saved {out.name}")


# ---------------------------------------------------------------------------
# High-level entry point for the pipeline
# ---------------------------------------------------------------------------
def analyze_dart_f_output(simu_or_nc_path, output_dir, label='',
                          scene_size_m=None, aggregate=False,
                          agg_x=10, agg_y=20, agg_label='0.5m x 1m'):
    """All-in-one: load NetCDF, compute metrics, produce plots.

    Can be called with either a pytools4dart simulation object
    or a direct path to the NetCDF file.

    Args:
        simu_or_nc_path: pytools4dart simulation or str/Path to NetCDF.
        output_dir: directory for output PNGs and JSON.
        label: filename suffix (e.g. 'day55_1200').
        scene_size_m: [x, y] scene extent for axis labels.
        aggregate: if True, also produce sensor-aggregated comparison.
        agg_x, agg_y: aggregation factors.
        agg_label: label for the coarser resolution.

    Returns:
        dict with metrics + paths, or empty dict on failure.
    """
    import json as _json

    output_dir = Path(output_dir)

    # Resolve NetCDF path
    if isinstance(simu_or_nc_path, (str, Path)):
        nc_path = Path(simu_or_nc_path)
        if nc_path.is_dir():
            nc_path = _find_netcdf(nc_path)
    else:
        # pytools4dart simulation object
        simu_path = Path(simu_or_nc_path.simu_dir)
        nc_path = _find_netcdf(simu_path / 'output')

    if nc_path is None:
        print("  No DART-F NetCDF found for analysis")
        return {}

    data = load_sif_from_netcdf(nc_path)
    if data is None:
        return {}

    metrics = compute_sif_metrics(data)
    print(f"    F760={metrics.get('F760_mean', 0):.5f}  "
          f"F687={metrics.get('F687_mean', 0):.5f}  "
          f"ratio={metrics.get('F760_F687_ratio', 0):.2f}  "
          f"veg={metrics.get('frac_veg', 0) * 100:.0f}%")

    # Spatial maps
    plot_sif_maps(data, output_dir, label=label, scene_size_m=scene_size_m)

    if aggregate:
        plot_sif_aggregated(data, output_dir, label=label,
                            agg_x=agg_x, agg_y=agg_y, agg_label=agg_label)

    # Save metrics JSON
    suffix = f'_{label}' if label else ''
    metrics_path = output_dir / f'SIF_metrics{suffix}.json'
    with open(metrics_path, 'w') as f:
        _json.dump(metrics, f, indent=2)

    metrics['overview_png'] = str(output_dir / f'SIF_overview{suffix}.png')
    metrics['metrics_json'] = str(metrics_path)
    return metrics
