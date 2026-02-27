#!/usr/bin/env python3
"""
Phase 4: CPlantBox Photosynthesis with Spatially-Resolved DART/Baleno Inputs.

Feeds per-segment APAR (from Phase 1 DART radiative transfer) and per-segment
Tleaf (from Phase 2 Baleno energy balance) into CPlantBox's FvCB photosynthesis
model.  Compares against a uniform-input baseline to quantify the impact of 3D
light distribution on whole-plant carbon assimilation.

Critical segment mapping:
  - CSV leaf rows are in organ creation order (leaf_1, leaf_2, ..., leaf_11)
  - CPlantBox getSegmentIds(4) iterates getOrgans(4) in the same creation order
  - For day-55 mature leaves (>>20 native nodes), NO resampling occurs in the
    lofter → lofter segments == CPlantBox segments exactly
  - Mapping JSON stores n_orig_segs = len(node_ids) - 1 = CPlantBox native count

Usage:
  cd /home/lukas/PHD
  source CPlantBox/cpbenv/bin/activate
  python CPlantBox/dart/coupling/run_coupled_photosynthesis.py
"""

import sys
import csv
import json
import numpy as np
from pathlib import Path
from collections import OrderedDict

import plantbox as pb

from ..config import DEFAULT_XML, OUTPUT_DIR, get_hydraulics_json, get_photosynthesis_json, get_phloem_json
from ..growth.grow import grow_plant
from ..prospect_params import (get_chl_for_photosynthesis, get_chl_per_segment,
                               get_prospect_params, log_consistency, log_lops_consistency,
                               vcmax25_from_cab)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
XML_PATH = str(DEFAULT_XML)
PHASE1_CSV = OUTPUT_DIR / 'maize_day55_segment_apar.csv'
PHASE2_CSV = OUTPUT_DIR / 'maize_day55_baleno_segments.csv'

SIMULATION_DAYS = 55
TARGET_PAR_UMOL = 1000.0   # µmol/m²/s mean PAR for normalization
TAIR_C = 25.0              # uniform baseline temperature
RH = 0.7
SOIL_PSI_CM = -500.0


# ============================================================================
# Step 1: Load Phase 1 + Phase 2 CSVs
# ============================================================================
def load_phase_csvs(phase1_path, phase2_path):
    """Parse Phase 1 APAR and Phase 2 Tleaf CSVs, filter to leaf rows.

    Returns dict with matched leaf rows, row count, and per-organ segment counts.
    """
    with open(phase1_path) as f:
        p1_rows = list(csv.DictReader(f))
    with open(phase2_path) as f:
        p2_rows = list(csv.DictReader(f))

    p1_leaf = [r for r in p1_rows if r['organ_type'] == 'leaf']
    p2_leaf = [r for r in p2_rows if r['organ_type'] == 'leaf']

    assert len(p1_leaf) == len(p2_leaf), \
        f"Phase 1/2 leaf row count mismatch: {len(p1_leaf)} vs {len(p2_leaf)}"

    # Verify organs match row-by-row
    for i, (r1, r2) in enumerate(zip(p1_leaf, p2_leaf)):
        assert r1['organ'] == r2['organ'] and r1['segment_idx'] == r2['segment_idx'], \
            f"Row {i}: P1={r1['organ']}:{r1['segment_idx']}, P2={r2['organ']}:{r2['segment_idx']}"

    # Build per-organ segment counts (preserving insertion order)
    organ_seg_counts = OrderedDict()
    for r in p1_leaf:
        name = r['organ']
        organ_seg_counts[name] = organ_seg_counts.get(name, 0) + 1

    return {
        'p1_leaf': p1_leaf,
        'p2_leaf': p2_leaf,
        'n_leaf_rows': len(p1_leaf),
        'organ_seg_counts': organ_seg_counts,
    }


# ============================================================================
# Step 3: Verify segment alignment (CRITICAL)
# ============================================================================
def verify_segment_alignment(plant, csv_data):
    """Assert total and per-organ CPlantBox segment counts match CSV.

    The i-th leaf organ (0-indexed) maps to CSV organ leaf_{i+1}
    (because stem_0 takes organ_counter=0 in the adapter).

    Aborts with diagnostic table on mismatch.
    """
    seg_leaves_idx = plant.getSegmentIds(4)
    n_cpb = len(seg_leaves_idx)
    n_csv = csv_data['n_leaf_rows']

    print(f"\n=== Segment Alignment Verification ===")
    print(f"  CPlantBox leaf segments: {n_cpb}")
    print(f"  CSV leaf rows:           {n_csv}")

    # Per-organ comparison
    leaf_organs = [o for o in plant.getOrgans() if o.organType() == pb.OrganTypes.leaf]
    cpb_organ_counts = OrderedDict()
    for i, organ in enumerate(leaf_organs):
        name = f"leaf_{i + 1}"
        cpb_organ_counts[name] = len(organ.getSegments())

    csv_counts = csv_data['organ_seg_counts']

    print(f"\n  {'Organ':<12} {'CPlantBox':>10} {'CSV':>10} {'Match':>6}")
    print(f"  {'-' * 40}")
    all_match = True
    for name in csv_counts:
        cpb_n = cpb_organ_counts.get(name, -1)
        csv_n = csv_counts[name]
        match = cpb_n == csv_n
        if not match:
            all_match = False
        print(f"  {name:<12} {cpb_n:>10} {csv_n:>10} {'OK' if match else 'FAIL':>6}")

    if n_cpb != n_csv or not all_match:
        print(f"\n  ALIGNMENT FAILURE — aborting.")
        print(f"  CPlantBox has {len(leaf_organs)} leaf organs, "
              f"CSV has {len(csv_counts)} organs.")
        sys.exit(1)

    print(f"\n  ALIGNMENT OK ({n_cpb} segments, {len(csv_counts)} organs)")
    return seg_leaves_idx


# ============================================================================
# Step 4: Build input arrays
# ============================================================================
def build_apar_array(csv_data, target_par_umol):
    """Normalize Phase 1 APAR fractions to a target PAR mean.

    Phase 1 total_apar values are DART absorption fractions (not absolute
    µmol/m²/s).  We normalize the spatial pattern so that the mean equals
    target_par_umol, enabling a fair comparison against a uniform baseline
    with the same mean.
    """
    total_apar = np.array([float(r['total_apar']) for r in csv_data['p1_leaf']])
    mean_apar = np.mean(total_apar)
    relative = total_apar / mean_apar
    apar_umol = relative * target_par_umol
    apar_umol = np.clip(apar_umol, 0.0, 3000.0)  # physical cap

    print(f"\n=== APAR Array ===")
    print(f"  Raw Phase 1: mean={mean_apar:.4f}, "
          f"range=[{total_apar.min():.4f}, {total_apar.max():.4f}]")
    print(f"  Scaled (µmol/m²/s): mean={np.mean(apar_umol):.1f}, "
          f"range=[{apar_umol.min():.1f}, {apar_umol.max():.1f}]")

    return apar_umol


def build_tleaf_array(csv_data):
    """Extract Tleaf_C array from Phase 2 Baleno output."""
    tleaf = np.array([float(r['Tleaf_C']) for r in csv_data['p2_leaf']])

    print(f"\n=== Tleaf Array ===")
    print(f"  Range: [{tleaf.min():.2f}, {tleaf.max():.2f}] °C, "
          f"mean={np.mean(tleaf):.2f} °C")

    return tleaf


# ============================================================================
# Steps 5/6: Run photosynthesis solve
# ============================================================================
def run_photosynthesis_solve(plant, sim_time, par, tleaf, label,
                             rh=0.7, soil_psi_cm=-500.0):
    """Setup hydraulics + FvCB photosynthesis and run hm.solve().

    Args:
        plant: pb.MappedPlant (grown, with soil grid).
        sim_time: days simulated.
        par: scalar or array of PAR [µmol/m²/s].
        tleaf: scalar or array of leaf temperature [°C].
        label: string label for logging.
        rh: relative humidity [0–1].
        soil_psi_cm: uniform soil water potential [cm].

    Returns dict with An_leaf, An_per_umol, An_total_mmol, transp_mmol.
    """
    from plantbox.functional.phloem_flux import PhloemFluxPython
    from plantbox.functional.PlantHydraulicParameters import PlantHydraulicParameters

    print(f"\n=== Photosynthesis Solve: {label} ===")

    # --- Hydraulic parameters ---
    params = PlantHydraulicParameters()
    params.read_parameters(get_hydraulics_json())

    # --- Photosynthesis + phloem model ---
    hm = PhloemFluxPython(plant, params)
    hm.read_photosynthesis_parameters(filename=get_photosynthesis_json())
    hm.read_phloem_parameters(filename=get_phloem_json())

    # Override Chl from LOPS per-position profiles (per-segment mode)
    chl_per_seg = get_chl_per_segment(sim_time, plant)
    seg_leaves_idx_check = plant.getSegmentIds(4)
    if len(chl_per_seg) == len(seg_leaves_idx_check):
        hm.Chl = chl_per_seg
    else:
        # Fallback to uniform if mismatch
        hm.Chl = [get_chl_for_photosynthesis(sim_time)]

    cab_arr = hm.Chl if len(hm.Chl) > 1 else hm.Chl * len(seg_leaves_idx_check)
    vcmax_min = vcmax25_from_cab(min(cab_arr))
    vcmax_max = vcmax25_from_cab(max(cab_arr))
    print(f"  PhotoType={'C4' if hm.PhotoType == 1 else 'C3'}, "
          f"Vcmax range=[{vcmax_min:.1f}, {vcmax_max:.1f}] µmol m-2 s-1 "
          f"({len(hm.Chl)} Chl values)")

    # --- Soil water potential ---
    depth = 100
    p_s = np.linspace(soil_psi_cm, soil_psi_cm - depth, depth)

    # --- Vapour pressure ---
    if np.isscalar(tleaf):
        es = hm.get_es(tleaf)
    else:
        es = hm.get_es(float(np.mean(tleaf)))
    ea = es * rh

    # --- PAR conversion: µmol/m²/s → mol/cm²/d ---
    if np.isscalar(par):
        par_mol = par * 1e-6 * 86400 * 1e-4
        print(f"  PAR: {par:.1f} µmol/m²/s (uniform) = {par_mol:.4e} mol/cm²/d")
    else:
        par_mol = par * 1e-6 * 86400 * 1e-4
        print(f"  PAR: array, mean={np.mean(par):.1f} µmol/m²/s, "
              f"range=[{par.min():.1f}, {par.max():.1f}]")

    # --- Pre-solve size assertions (prevent silent broadcast bug) ---
    seg_leaves_idx = plant.getSegmentIds(4)
    n_leaf_segs = len(seg_leaves_idx)

    if not np.isscalar(par_mol):
        assert len(par_mol) == n_leaf_segs, \
            f"PAR array size {len(par_mol)} != leaf segments {n_leaf_segs}"
    if not np.isscalar(tleaf):
        assert len(tleaf) == n_leaf_segs, \
            f"Tleaf array size {len(tleaf)} != leaf segments {n_leaf_segs}"

    if np.isscalar(tleaf):
        print(f"  Tleaf: {tleaf:.1f} °C (uniform)")
    else:
        print(f"  Tleaf: array, mean={np.mean(tleaf):.2f} °C, "
              f"range=[{tleaf.min():.2f}, {tleaf.max():.2f}]")
    print(f"  es={es:.2f}, ea={ea:.2f}, RH={rh*100:.0f}%")
    print(f"  Leaf segments: {n_leaf_segs}")

    # --- Solve ---
    try:
        hm.solve(
            sim_time=sim_time,
            rsx=p_s,
            cells=True,
            ea=ea,
            es=es,
            PAR=par_mol,
            TairC=tleaf,
            verbose=0,
        )
    except Exception as e:
        print(f"  ERROR in hm.solve(): {e}")
        return None

    # --- Extract results ---
    An_leaf = np.array(hm.get_net_assimilation())         # mol CO2/d per segment
    An_per = np.array(hm.get_net_assimilation_perleafBladeArea())  # mol CO2/cm²/d
    transp = np.sum(hm.get_transpiration()) / 18 * 1e3    # mmol H2O/d

    An_total_mmol = np.sum(An_leaf) * 1e3                 # mmol CO2/d
    An_per_umol = An_per * 1e4 / 86400 * 1e6              # µmol CO2/m²/s

    print(f"\n  --- Results ({label}) ---")
    print(f"  Total An:      {An_total_mmol:.3f} mmol CO2/d")
    print(f"  Transpiration: {transp:.3f} mmol H2O/d")
    print(f"  Leaf segments: {len(An_leaf)}")

    active = An_per_umol[An_per_umol > 0]
    if len(active) > 0:
        print(f"  Active:        {len(active)} / {len(An_per_umol)}")
        print(f"  Mean An:       {np.mean(active):.2f} µmol CO2 m-2 s-1")
        print(f"  Range:         [{np.min(active):.2f}, {np.max(active):.2f}]")

    return {
        'An_leaf': An_leaf,
        'An_per_umol': An_per_umol,
        'An_total_mmol': An_total_mmol,
        'transp_mmol': transp,
        'n_segs': len(An_leaf),
    }


# ============================================================================
# Step 7: Compare results
# ============================================================================
def compare_results(uniform, informed, apar_umol, tleaf_c, csv_data):
    """Compute comparison metrics between uniform and 3D-informed runs."""
    print(f"\n{'=' * 70}")
    print(f"COMPARISON: Uniform vs 3D-Informed Photosynthesis")
    print(f"{'=' * 70}")

    An_u = uniform['An_total_mmol']
    An_3d = informed['An_total_mmol']
    delta_pct = (An_3d - An_u) / An_u * 100

    print(f"\n  Whole-plant total An:")
    print(f"    Uniform:  {An_u:.3f} mmol CO2/d")
    print(f"    3D:       {An_3d:.3f} mmol CO2/d")
    print(f"    Change:   {delta_pct:+.2f}%")

    print(f"\n  Transpiration:")
    tr_u = uniform['transp_mmol']
    tr_3d = informed['transp_mmol']
    tr_delta = (tr_3d - tr_u) / tr_u * 100 if abs(tr_u) > 1e-12 else 0.0
    print(f"    Uniform:  {tr_u:.3f} mmol H2O/d")
    print(f"    3D:       {tr_3d:.3f} mmol H2O/d")
    print(f"    Change:   {tr_delta:+.2f}%")

    # Per-segment distribution stats
    print(f"\n  Per-segment An (µmol CO2 m-2 s-1):")
    for label, res in [("Uniform", uniform), ("3D", informed)]:
        an = res['An_per_umol']
        print(f"    {label:8s}: mean={np.mean(an):.2f}, median={np.median(an):.2f}, "
              f"std={np.std(an):.2f}, range=[{np.min(an):.2f}, {np.max(an):.2f}]")

    # Sunlit/shaded split
    mean_apar = np.mean(apar_umol)
    sunlit_mask = apar_umol >= mean_apar
    shaded_mask = ~sunlit_mask

    print(f"\n  Sunlit/shaded split (threshold: {mean_apar:.1f} µmol/m²/s):")
    for lbl, mask in [("Sunlit", sunlit_mask), ("Shaded", shaded_mask)]:
        n = np.sum(mask)
        an_u = np.mean(uniform['An_per_umol'][mask]) if n > 0 else 0
        an_3d = np.mean(informed['An_per_umol'][mask]) if n > 0 else 0
        print(f"    {lbl:7s}: n={n:>5}, An_uniform={an_u:.2f}, "
              f"An_3D={an_3d:.2f} µmol m-2 s-1")

    # Per-organ breakdown
    print(f"\n  Per-organ breakdown:")
    print(f"  {'Organ':<12} {'N_segs':>7} {'An_U (mmol)':>12} "
          f"{'An_3D (mmol)':>13} {'Change':>8}")
    print(f"  {'-' * 55}")

    organ_names = list(csv_data['organ_seg_counts'].keys())
    offset = 0
    organ_metrics = []
    for name in organ_names:
        n = csv_data['organ_seg_counts'][name]
        an_u = np.sum(uniform['An_leaf'][offset:offset + n]) * 1e3
        an_3d = np.sum(informed['An_leaf'][offset:offset + n]) * 1e3
        change = (an_3d - an_u) / max(abs(an_u), 1e-12) * 100
        mean_apar_org = float(np.mean(apar_umol[offset:offset + n]))
        mean_tleaf_org = float(np.mean(tleaf_c[offset:offset + n]))

        print(f"  {name:<12} {n:>7} {an_u:>12.3f} {an_3d:>13.3f} {change:>+7.1f}%")

        organ_metrics.append({
            'organ': name,
            'n_segments': n,
            'An_uniform_mmol': float(an_u),
            'An_3d_mmol': float(an_3d),
            'change_pct': float(change),
            'mean_apar_umol': mean_apar_org,
            'mean_tleaf_c': mean_tleaf_org,
        })
        offset += n

    return {
        'An_uniform_mmol': float(An_u),
        'An_3d_mmol': float(An_3d),
        'change_pct': float(delta_pct),
        'transp_uniform_mmol': float(tr_u),
        'transp_3d_mmol': float(tr_3d),
        'transp_change_pct': float(tr_delta),
        'per_organ': organ_metrics,
    }


# ============================================================================
# Plotting
# ============================================================================
def plot_comparison(uniform, informed, apar_umol, tleaf_c, csv_data,
                    comparison, output_path):
    """Create 6-panel comparison figure."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(18, 11), facecolor='white')
    fig.suptitle(
        f"Phase 4: CPlantBox Photosynthesis — Uniform vs 3D-Informed\n"
        f"An_uniform={comparison['An_uniform_mmol']:.1f}, "
        f"An_3D={comparison['An_3d_mmol']:.1f} mmol CO$_2$ d$^{{-1}}$ "
        f"({comparison['change_pct']:+.1f}%)",
        fontsize=13, fontweight='bold', y=0.98
    )

    # --- Panel 1: APAR vs An scatter (light response curve) ---
    ax = axes[0, 0]
    ax.scatter(apar_umol, informed['An_per_umol'],
               s=1, alpha=0.3, c='forestgreen', label='3D-informed')
    ax.axhline(np.mean(uniform['An_per_umol']),
               color='orange', ls='--', lw=1.5, label='Uniform mean')
    ax.set_xlabel('aPAR (µmol m$^{-2}$ s$^{-1}$)')
    ax.set_ylabel('An (µmol CO$_2$ m$^{-2}$ s$^{-1}$)')
    ax.set_title('Light Response (APAR vs An)')
    ax.legend(fontsize=8, loc='lower right')
    ax.set_xlim(0, 3200)

    # --- Panel 2: Per-organ An comparison (grouped bars) ---
    ax = axes[0, 1]
    organ_labels = [m['organ'].replace('leaf_', 'L') for m in comparison['per_organ']]
    n_org = len(organ_labels)
    x = np.arange(n_org)
    w = 0.35
    an_u = [m['An_uniform_mmol'] for m in comparison['per_organ']]
    an_3d = [m['An_3d_mmol'] for m in comparison['per_organ']]
    ax.bar(x - w / 2, an_u, w, color='orange', edgecolor='#333', lw=0.5,
           label='Uniform')
    ax.bar(x + w / 2, an_3d, w, color='forestgreen', edgecolor='#333', lw=0.5,
           label='3D')
    ax.set_xticks(x)
    ax.set_xticklabels(organ_labels, fontsize=8)
    ax.set_ylabel('An (mmol CO$_2$ d$^{-1}$)')
    ax.set_title('Per-Organ Net Assimilation')
    ax.legend(fontsize=8)

    # --- Panel 3: Per-organ % change ---
    ax = axes[0, 2]
    changes = [m['change_pct'] for m in comparison['per_organ']]
    colors = ['forestgreen' if c >= 0 else 'firebrick' for c in changes]
    ax.bar(x, changes, color=colors, edgecolor='#333', lw=0.5)
    ax.axhline(0, color='k', lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(organ_labels, fontsize=8)
    ax.set_ylabel('Change (%)')
    ax.set_title('3D vs Uniform Change per Organ')

    # --- Panel 4: APAR distribution histogram ---
    ax = axes[1, 0]
    ax.hist(apar_umol, bins=60, color='gold', edgecolor='#333', lw=0.3,
            alpha=0.8)
    ax.axvline(TARGET_PAR_UMOL, color='red', ls='--', lw=1.5,
               label=f'Mean={TARGET_PAR_UMOL:.0f}')
    ax.set_xlabel('aPAR (µmol m$^{-2}$ s$^{-1}$)')
    ax.set_ylabel('Count')
    ax.set_title('APAR Distribution (from DART)')
    ax.legend(fontsize=8)

    # --- Panel 5: Tleaf distribution ---
    ax = axes[1, 1]
    ax.hist(tleaf_c, bins=40, color='salmon', edgecolor='#333', lw=0.3,
            alpha=0.8)
    ax.axvline(np.mean(tleaf_c), color='red', ls='--', lw=1.5,
               label=f'Mean={np.mean(tleaf_c):.2f} °C')
    ax.set_xlabel('T$_{leaf}$ (°C)')
    ax.set_ylabel('Count')
    ax.set_title('Leaf Temperature Distribution (from Baleno)')
    ax.legend(fontsize=8)

    # --- Panel 6: Per-segment An difference (3D - uniform) ---
    ax = axes[1, 2]
    diff = informed['An_per_umol'] - uniform['An_per_umol']
    ax.scatter(apar_umol, diff, s=1, alpha=0.3, c='steelblue')
    ax.axhline(0, color='k', lw=0.5)
    ax.set_xlabel('aPAR (µmol m$^{-2}$ s$^{-1}$)')
    ax.set_ylabel('$\\Delta$An (µmol CO$_2$ m$^{-2}$ s$^{-1}$)')
    ax.set_title('Per-Segment An Difference (3D − Uniform)')

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(str(output_path), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Figure saved: {output_path}")


# ============================================================================
# Main
# ============================================================================
def main():
    print("=" * 70)
    print("Phase 4: CPlantBox Photosynthesis with DART/Baleno Inputs")
    print("=" * 70)
    log_consistency(SIMULATION_DAYS)

    # ------------------------------------------------------------------
    # Step 1: Load Phase 1 + Phase 2 CSVs
    # ------------------------------------------------------------------
    print("\n--- Step 1: Load Phase 1 + Phase 2 CSVs ---")
    csv_data = load_phase_csvs(PHASE1_CSV, PHASE2_CSV)
    print(f"  Loaded {csv_data['n_leaf_rows']} leaf rows from both CSVs")
    print(f"  Organs: {list(csv_data['organ_seg_counts'].keys())}")

    # ------------------------------------------------------------------
    # Step 2: Grow day-55 plant for UNIFORM baseline
    # ------------------------------------------------------------------
    print("\n--- Step 2: Grow Day-55 Plant (UNIFORM baseline) ---")
    plant_u = grow_plant(XML_PATH, SIMULATION_DAYS, enable_photosynthesis=True)

    # ------------------------------------------------------------------
    # Step 3: Verify segment alignment
    # ------------------------------------------------------------------
    print("\n--- Step 3: Verify Segment Alignment ---")
    seg_leaves_idx = verify_segment_alignment(plant_u, csv_data)

    # ------------------------------------------------------------------
    # Step 4: Build input arrays
    # ------------------------------------------------------------------
    print("\n--- Step 4: Build Input Arrays ---")
    apar_umol = build_apar_array(csv_data, TARGET_PAR_UMOL)
    tleaf_c = build_tleaf_array(csv_data)

    # ------------------------------------------------------------------
    # Step 5: Run UNIFORM baseline
    # ------------------------------------------------------------------
    print("\n--- Step 5: Run UNIFORM Baseline ---")
    uniform = run_photosynthesis_solve(
        plant_u, SIMULATION_DAYS,
        par=TARGET_PAR_UMOL, tleaf=TAIR_C,
        label="UNIFORM",
        rh=RH, soil_psi_cm=SOIL_PSI_CM,
    )
    if uniform is None:
        print("UNIFORM solve failed, aborting")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Step 6: Grow FRESH plant + run 3D-INFORMED
    # ------------------------------------------------------------------
    print("\n--- Step 6: Grow Fresh Plant + Run 3D-Informed ---")
    plant_3d = grow_plant(XML_PATH, SIMULATION_DAYS, enable_photosynthesis=True)

    # Verify alignment on fresh plant too
    seg_3d = plant_3d.getSegmentIds(4)
    assert len(seg_3d) == csv_data['n_leaf_rows'], \
        f"Fresh plant seg count {len(seg_3d)} != CSV {csv_data['n_leaf_rows']}"
    print(f"  Fresh plant alignment OK ({len(seg_3d)} leaf segments)")

    informed = run_photosynthesis_solve(
        plant_3d, SIMULATION_DAYS,
        par=apar_umol, tleaf=tleaf_c,
        label="3D-INFORMED",
        rh=RH, soil_psi_cm=SOIL_PSI_CM,
    )
    if informed is None:
        print("3D-INFORMED solve failed, aborting")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Step 7: Compare and export
    # ------------------------------------------------------------------
    print("\n--- Step 7: Compare and Export ---")
    comparison = compare_results(uniform, informed, apar_umol, tleaf_c, csv_data)

    # --- Per-segment CSV ---
    csv_out = OUTPUT_DIR / 'maize_day55_coupled_photosynthesis.csv'
    organ_names = list(csv_data['organ_seg_counts'].keys())
    with open(csv_out, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'organ', 'segment_idx',
            'apar_umol_m2_s', 'tleaf_c',
            'An_uniform_umol_m2_s', 'An_3d_umol_m2_s',
            'An_uniform_mol_d', 'An_3d_mol_d',
        ])
        offset = 0
        for name in organ_names:
            n = csv_data['organ_seg_counts'][name]
            for i in range(n):
                idx = offset + i
                writer.writerow([
                    name, i,
                    f"{apar_umol[idx]:.4f}",
                    f"{tleaf_c[idx]:.4f}",
                    f"{uniform['An_per_umol'][idx]:.6f}",
                    f"{informed['An_per_umol'][idx]:.6f}",
                    f"{uniform['An_leaf'][idx]:.6e}",
                    f"{informed['An_leaf'][idx]:.6e}",
                ])
            offset += n
    print(f"  CSV: {csv_out} ({offset} rows)")

    # --- JSON summary ---
    json_out = OUTPUT_DIR / 'maize_day55_coupled_results.json'
    results = {
        'phase': 'Phase 4: Coupled Photosynthesis',
        'simulation_days': SIMULATION_DAYS,
        'target_par_umol': TARGET_PAR_UMOL,
        'tair_c_uniform': TAIR_C,
        'rh': RH,
        'soil_psi_cm': SOIL_PSI_CM,
        'n_leaf_segments': csv_data['n_leaf_rows'],
        'n_organs': len(organ_names),
        'apar_stats': {
            'mean': float(np.mean(apar_umol)),
            'std': float(np.std(apar_umol)),
            'min': float(np.min(apar_umol)),
            'max': float(np.max(apar_umol)),
        },
        'tleaf_stats': {
            'mean': float(np.mean(tleaf_c)),
            'std': float(np.std(tleaf_c)),
            'min': float(np.min(tleaf_c)),
            'max': float(np.max(tleaf_c)),
        },
        'uniform': {
            'An_total_mmol': float(uniform['An_total_mmol']),
            'transp_mmol': float(uniform['transp_mmol']),
        },
        'informed_3d': {
            'An_total_mmol': float(informed['An_total_mmol']),
            'transp_mmol': float(informed['transp_mmol']),
        },
        'comparison': comparison,
    }
    with open(json_out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  JSON: {json_out}")

    # --- Comparison plot ---
    plot_path = OUTPUT_DIR / 'maize_day55_coupled_comparison.png'
    plot_comparison(uniform, informed, apar_umol, tleaf_c, csv_data,
                    comparison, plot_path)

    # --- Final summary ---
    print(f"\n{'=' * 70}")
    print(f"Phase 4 Complete!")
    print(f"  An_uniform = {uniform['An_total_mmol']:.3f} mmol CO2/d")
    print(f"  An_3D      = {informed['An_total_mmol']:.3f} mmol CO2/d")
    print(f"  Change     = {comparison['change_pct']:+.2f}%")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
