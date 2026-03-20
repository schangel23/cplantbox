#!/usr/bin/env python3
"""SIF output writers — per-segment and per-triangle CSVs + summary stats."""

import csv
import numpy as np
from pathlib import Path


def write_segment_sif_csv(output_path, iter_result, plant_idx,
                          clearsky_par_wm2, fqe=0.01, sunlit_frac=0.2):
    """Write per-segment SIF CSV for one plant at one timestep.

    Columns: segment_idx, n_triangles, total_area_cm2,
             apar_umol, tleaf_C, gs_mol, An_umol, eta, SIF_W_m2, f_sunlit,
             psi_leaf_MPa
    """
    output_path = Path(output_path)

    an = iter_result.get('an_per_segment')
    tleaf = iter_result.get('tleaf_per_segment')
    gs = iter_result.get('gs_per_segment')
    eta = iter_result.get('eta_per_segment')
    psi = iter_result.get('psi_leaf_MPa')

    if an is None or eta is None:
        return

    n_segs = len(an)

    # Per-segment tri_data for area/apar if available
    tri_data = iter_result.get('tri_data')

    fieldnames = ['segment_idx', 'n_triangles', 'total_area_cm2',
                  'apar_umol', 'tleaf_C', 'gs_mol', 'An_umol',
                  'eta', 'SIF_W_m2', 'f_sunlit', 'psi_leaf_MPa']

    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for si in range(n_segs):
            eta_val = float(eta[si]) if si < len(eta) else 0.0
            an_val = float(an[si]) if si < len(an) else 0.0
            tleaf_val = float(tleaf[si]) if tleaf is not None and si < len(tleaf) else 25.0
            gs_val = float(gs[si]) if gs is not None and si < len(gs) else 0.0
            psi_val = float(psi[si]) if psi is not None and si < len(psi) else None

            # Per-segment aggregated values from tri_data
            n_tris = 0
            area_cm2 = 0.0
            apar_umol = 0.0
            sif_wm2 = 0.0
            n_sunlit = 0
            n_total = 0
            if tri_data is not None and si < len(tri_data):
                td = tri_data[si]
                n_tris = td.get('n_triangles', 0)
                area_cm2 = td.get('total_area_cm2', 0.0)
                apar_umol = td.get('mean_apar_umol', 0.0)
                n_sunlit = td.get('n_sunlit', 0)
                n_total = td.get('n_total', max(n_tris, 1))
                # SIF = eta * fqe * apar (W/m2 PAR)
                apar_wm2 = apar_umol / 4.57 if apar_umol > 0 else 0.0
                sif_wm2 = eta_val * fqe * apar_wm2

            f_sunlit = float(n_sunlit) / max(n_total, 1)

            writer.writerow({
                'segment_idx': si,
                'n_triangles': n_tris,
                'total_area_cm2': round(area_cm2, 4),
                'apar_umol': round(apar_umol, 2),
                'tleaf_C': round(tleaf_val, 2),
                'gs_mol': round(gs_val, 6),
                'An_umol': round(an_val, 4),
                'eta': round(eta_val, 6),
                'SIF_W_m2': round(sif_wm2, 6),
                'f_sunlit': round(f_sunlit, 3),
                'psi_leaf_MPa': round(psi_val, 4) if psi_val is not None else '',
            })


def write_triangle_sif_csv(output_path, tri_data_raw, plant_idx,
                           clearsky_par_wm2, fqe=0.01, sunlit_frac=0.2,
                           apar_shaded_threshold_umol=10.0):
    """Write per-triangle SIF CSV for one plant (opt-in, large file).

    tri_data_raw: list of dicts per triangle with keys:
        tri_idx, segment_idx, apar_Wm2, tleaf_C, eta, area_cm2
    """
    output_path = Path(output_path)
    if tri_data_raw is None or len(tri_data_raw) == 0:
        return

    # Fixed threshold: Baleno apar_Wm2 is per-triangle absorbed energy,
    # not comparable to clearsky PAR.  Use absolute threshold (10 µmol).
    par_threshold = apar_shaded_threshold_umol

    fieldnames = ['tri_idx', 'segment_idx', 'apar_Wm2', 'tleaf_C',
                  'eta', 'SIF_W_m2', 'sunlit', 'area_cm2']

    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for td in tri_data_raw:
            apar_wm2 = td.get('apar_Wm2', 0.0)
            eta_val = td.get('eta', 0.0)
            sif_wm2 = eta_val * fqe * apar_wm2
            # Use DART sunlit classification if available, else threshold
            sunlit_frac = td.get('sunlit_frac')
            if sunlit_frac is not None:
                sunlit = 1 if sunlit_frac > 0.5 else 0
            else:
                apar_umol = apar_wm2 * 4.57
                sunlit = 1 if apar_umol > par_threshold else 0
            writer.writerow({
                'tri_idx': td.get('tri_idx', 0),
                'segment_idx': td.get('segment_idx', 0),
                'apar_Wm2': round(apar_wm2, 4),
                'tleaf_C': round(td.get('tleaf_C', 25.0), 2),
                'eta': round(eta_val, 6),
                'SIF_W_m2': round(sif_wm2, 6),
                'sunlit': sunlit,
                'area_cm2': round(td.get('area_cm2', 0.0), 4),
            })


def compute_sunlit_shaded_summary(iter_results, clearsky_par_wm2,
                                  fqe=0.01, sunlit_frac=0.2,
                                  ground_area_m2=None,
                                  apar_shaded_threshold_umol=10.0):
    """Compute sunlit/shaded summary stats across all plants.

    Args:
        iter_results: list of per-plant result dicts from iterative coupling.
            Each must have 'tri_data_raw' (list of per-tri dicts).
        clearsky_par_wm2: clearsky PAR at this timestep [W/m2].
        fqe: fluorescence quantum efficiency.
        sunlit_frac: (unused, kept for API compat) formerly fraction of clearsky PAR.
        ground_area_m2: total ground area [m2] for canopy-level SIF.
        apar_shaded_threshold_umol: fixed aPAR threshold [µmol] below which
            a triangle is classified as shaded.  Default 10 µmol matches
            _is_shaded() in baleno.py.

    Returns:
        dict with sunlit/shaded stats + canopy SIF, or empty dict on failure.
    """
    if iter_results is None:
        return {}

    # Fixed threshold: Baleno apar_Wm2 is per-triangle absorbed energy,
    # not comparable to clearsky PAR.  Use absolute threshold (10 µmol).
    par_threshold = apar_shaded_threshold_umol

    all_apar_sunlit = []
    all_apar_shaded = []
    all_tleaf_sunlit = []
    all_tleaf_shaded = []
    all_an_sunlit = []
    all_an_shaded = []
    all_eta_sunlit = []
    all_eta_shaded = []
    total_sif_area_weighted = 0.0
    total_leaf_area_cm2 = 0.0
    n_tri_sunlit = 0
    n_tri_shaded = 0

    for r in iter_results:
        if r is None:
            continue
        tri_raw = r.get('tri_data_raw')
        if tri_raw is None:
            continue

        for td in tri_raw:
            apar_wm2 = td.get('apar_Wm2', 0.0)
            apar_umol = apar_wm2 * 4.57
            tleaf_c = td.get('tleaf_C', 25.0)
            eta_val = td.get('eta', 0.0)
            area_cm2 = td.get('area_cm2', 1.0)
            an_val = td.get('An_umol', 0.0)

            sif_tri = eta_val * fqe * apar_wm2 * area_cm2
            total_sif_area_weighted += sif_tri
            total_leaf_area_cm2 += area_cm2

            # Use DART sunlit classification if available, else threshold
            sunlit_frac = td.get('sunlit_frac')
            if sunlit_frac is not None:
                is_sunlit = sunlit_frac > 0.5
            else:
                is_sunlit = apar_umol > par_threshold
            if is_sunlit:
                n_tri_sunlit += 1
                all_apar_sunlit.append(apar_umol)
                all_tleaf_sunlit.append(tleaf_c)
                all_an_sunlit.append(an_val)
                all_eta_sunlit.append(eta_val)
            else:
                n_tri_shaded += 1
                all_apar_shaded.append(apar_umol)
                all_tleaf_shaded.append(tleaf_c)
                all_an_shaded.append(an_val)
                all_eta_shaded.append(eta_val)

    n_total = n_tri_sunlit + n_tri_shaded
    if n_total == 0:
        return {}

    f_sunlit_area = float(n_tri_sunlit) / n_total

    # Canopy-level SIF: area-weighted sum / ground area
    if ground_area_m2 is not None and ground_area_m2 > 0:
        sif_canopy = total_sif_area_weighted / (ground_area_m2 * 1e4)  # cm2->m2
    elif total_leaf_area_cm2 > 0:
        sif_canopy = total_sif_area_weighted / total_leaf_area_cm2
    else:
        sif_canopy = 0.0

    def _safe_mean(arr):
        return float(np.mean(arr)) if arr else 0.0

    return {
        'n_tri_sunlit': n_tri_sunlit,
        'n_tri_shaded': n_tri_shaded,
        'f_sunlit_area': round(f_sunlit_area, 4),
        'mean_apar_sunlit': round(_safe_mean(all_apar_sunlit), 2),
        'mean_apar_shaded': round(_safe_mean(all_apar_shaded), 2),
        'mean_tleaf_sunlit': round(_safe_mean(all_tleaf_sunlit), 2),
        'mean_tleaf_shaded': round(_safe_mean(all_tleaf_shaded), 2),
        'mean_An_sunlit': round(_safe_mean(all_an_sunlit), 4),
        'mean_An_shaded': round(_safe_mean(all_an_shaded), 4),
        'mean_eta_sunlit': round(_safe_mean(all_eta_sunlit), 6),
        'mean_eta_shaded': round(_safe_mean(all_eta_shaded), 6),
        'SIF_canopy_W_m2': round(sif_canopy, 6),
    }
