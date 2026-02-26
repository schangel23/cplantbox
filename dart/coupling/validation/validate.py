#!/usr/bin/env python3
"""
Phase 5: Coupling Validation — Physical Consistency and Biological Realism.

Validates the CPlantBox-Baleno coupling by checking:
  1. Vertical APAR/An profiles (should decrease top → bottom)
  2. Energy balance closure (Rn ≈ LE + H per segment)
  3. Transpiration consistency: Baleno LE vs CPlantBox Ev
  4. 3D light distribution visualization (An_3D coloured scatter)
  5. Sunlit vs shaded photosynthesis statistics

All prerequisite CSVs from Phases 1, 2, and 4 are read from output/.
The script re-grows the day-55 plant (for Z-heights) and re-runs hm.solve()
with 3D inputs to obtain per-segment transpiration (Ev).

Usage:
  cd /home/lukas/PHD
  source CPlantBox/cpbenv/bin/activate
  python CPlantBox/dart/coupling/validate_coupling.py
"""

import csv
import json
import numpy as np
from pathlib import Path
from collections import OrderedDict

import plantbox as pb

from ..config import DEFAULT_XML, OUTPUT_DIR, PHOTO_PATH, get_phloem_json
from ..growth.grow import grow_plant

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
XML_PATH   = str(DEFAULT_XML)

PHASE1_CSV = OUTPUT_DIR / 'maize_day55_segment_apar.csv'
PHASE2_CSV = OUTPUT_DIR / 'maize_day55_baleno_segments.csv'
PHASE4_CSV = OUTPUT_DIR / 'maize_day55_coupled_photosynthesis.csv'
OBJ_PATH   = OUTPUT_DIR / 'maize_day55.obj'
MAP_PATH   = OUTPUT_DIR / 'maize_day55_mapping.json'

SIMULATION_DAYS  = 55
TARGET_PAR_UMOL  = 1000.0   # µmol/m²/s — used to normalise Phase 1 APAR
TAIR_C           = 25.0
RH               = 0.7
SOIL_PSI_CM      = -500.0

SUNLIT_THRESHOLD = 500.0      # µmol/m²/s
LAMBDA_J_MOL     = 44000.0   # latent heat of H2O vaporisation [J/mol]
MW_H2O           = 18.0       # g/mol
SECS_PER_DAY     = 86400.0


# ============================================================================
# Step 1: Load and merge CSVs
# ============================================================================
def load_and_merge_csvs():
    """Load Phase 1, 2, 4 CSVs; filter to leaf rows; join by (organ, segment_idx).

    Returns:
        merged      : list of dicts, one per leaf segment (ordered as in Phase 4 CSV)
        organ_counts: OrderedDict  organ_name → n_segments
    """
    print("\n--- Step 1: Load and Merge CSVs ---")

    with open(PHASE1_CSV) as f:
        p1_all = list(csv.DictReader(f))
    with open(PHASE2_CSV) as f:
        p2_all = list(csv.DictReader(f))
    with open(PHASE4_CSV) as f:
        p4_all = list(csv.DictReader(f))

    p1_leaf = [r for r in p1_all if r['organ_type'] == 'leaf']
    p2_leaf = [r for r in p2_all if r['organ_type'] == 'leaf']
    p4_leaf = p4_all   # Phase 4 CSV is leaf-only

    n = len(p4_leaf)
    assert len(p1_leaf) == n, f"Phase 1 leaf rows {len(p1_leaf)} != Phase 4 {n}"
    assert len(p2_leaf) == n, f"Phase 2 leaf rows {len(p2_leaf)} != Phase 4 {n}"

    merged = []
    for i, (r1, r2, r4) in enumerate(zip(p1_leaf, p2_leaf, p4_leaf)):
        assert r4['organ'] == r2['organ'] and r4['segment_idx'] == r2['segment_idx'], (
            f"Row {i}: P4=({r4['organ']},{r4['segment_idx']}) "
            f"P2=({r2['organ']},{r2['segment_idx']})"
        )
        merged.append({
            'organ':        r4['organ'],
            'segment_idx':  int(r4['segment_idx']),
            # Phase 4 columns (APAR already in µmol/m²/s)
            'apar_umol_m2_s': float(r4['apar_umol_m2_s']),
            'tleaf_c':        float(r4['tleaf_c']),
            'An_uniform':     float(r4['An_uniform_umol_m2_s']),
            'An_3d':          float(r4['An_3d_umol_m2_s']),
            # Phase 2 Baleno columns
            'Rn_Wm2':         float(r2['Rn_Wm2']),
            'LE_Wm2':         float(r2['LE_Wm2']),
            'H_Wm2':          float(r2['H_Wm2']),
            'EB_error_Wm2':   float(r2['EB_error_Wm2']),
        })

    organ_counts = OrderedDict()
    for r in merged:
        organ_counts[r['organ']] = organ_counts.get(r['organ'], 0) + 1

    print(f"  Leaf segments: {n}")
    print(f"  Organs ({len(organ_counts)}): {list(organ_counts.keys())}")
    return merged, organ_counts


# ============================================================================
# Step 2: Grow plant — extract segment Z-heights and XYZ midpoints
# ============================================================================
def get_segment_positions(plant):
    """Return Z-heights and XYZ midpoints for all leaf segments.

    Iterates leaf organs in creation order (same order as CSV rows).

    Returns:
        z_cm   : ndarray (n_segs,)  — height above ground [cm]
        xyz_cm : ndarray (n_segs, 3) — X, Y, Z midpoints [cm]
    """
    print("\n--- Step 2: Extract Segment Positions ---")

    all_nodes = np.array(plant.getNodes())   # (N, 3), cm
    leaf_organs = [o for o in plant.getOrgans()
                   if o.organType() == pb.OrganTypes.leaf]

    x, y, z = [], [], []
    for organ in leaf_organs:
        for seg in organ.getSegments():
            n0 = all_nodes[seg.x]
            n1 = all_nodes[seg.y]
            x.append((n0[0] + n1[0]) / 2.0)
            y.append((n0[1] + n1[1]) / 2.0)
            z.append((n0[2] + n1[2]) / 2.0)

    z_cm   = np.array(z)
    xyz_cm = np.stack([x, y, z], axis=1)

    print(f"  Leaf segments: {len(z_cm)}")
    print(f"  Z range: [{z_cm.min():.1f}, {z_cm.max():.1f}] cm")
    print(f"  X range: [{xyz_cm[:, 0].min():.1f}, {xyz_cm[:, 0].max():.1f}] cm")
    print(f"  Y range: [{xyz_cm[:, 1].min():.1f}, {xyz_cm[:, 1].max():.1f}] cm")
    return z_cm, xyz_cm


# ============================================================================
# Step 3: Compute per-segment leaf areas from OBJ mesh
# ============================================================================
def compute_segment_areas(merged):
    """Parse OBJ + mapping JSON; return per-segment leaf area in cm² and m².

    Uses a (organ_name, segment_idx) lookup so the result order matches `merged`.
    Triangle area = 0.5 * ||(v1-v0) × (v2-v0)||  [cm²].
    """
    print("\n--- Step 3: Compute Segment Leaf Areas ---")

    # --- Parse OBJ vertices ---
    verts = []
    with open(OBJ_PATH) as f:
        for line in f:
            if line.startswith('v '):
                parts = line.split()
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
    verts = np.array(verts)   # (N_v, 3) cm

    # --- Parse OBJ faces (1-indexed → 0-indexed) ---
    faces = []
    with open(OBJ_PATH) as f:
        for line in f:
            if line.startswith('f '):
                parts = line.split()[1:]
                faces.append([int(p.split('/')[0]) - 1 for p in parts])
    faces = np.array(faces)   # (N_f, 3)

    print(f"  OBJ: {len(verts)} vertices, {len(faces)} faces")

    # --- Load mapping JSON ---
    with open(MAP_PATH) as f:
        mapping = json.load(f)

    # --- Build (organ_name, segment_idx) → area_cm2 lookup ---
    area_lookup = {}
    for organ_info in mapping['organs']:
        if organ_info['type'] != 'leaf':
            continue
        oname = organ_info['name']
        for seg_info in organ_info['segments']:
            tri_idx = seg_info['triangle_indices']
            if len(tri_idx) == 0:
                area_cm2 = 0.0
            else:
                tri_idx = np.array(tri_idx, dtype=int)
                v0 = verts[faces[tri_idx, 0]]
                v1 = verts[faces[tri_idx, 1]]
                v2 = verts[faces[tri_idx, 2]]
                cross = np.cross(v1 - v0, v2 - v0)
                area_cm2 = float(np.sum(0.5 * np.linalg.norm(cross, axis=1)))
            area_lookup[(oname, seg_info['segment_idx'])] = area_cm2

    # --- Build array aligned with merged rows ---
    seg_area_cm2 = np.array([
        area_lookup.get((r['organ'], r['segment_idx']), 0.0)
        for r in merged
    ])
    seg_area_m2 = seg_area_cm2 * 1e-4

    missing = int(np.sum(seg_area_cm2 == 0))
    print(f"  Leaf segments in lookup: {len(area_lookup)}")
    print(f"  Segments with zero area: {missing}")
    print(f"  Total leaf area: {np.sum(seg_area_m2):.4f} m²  "
          f"({np.sum(seg_area_cm2):.0f} cm²)")
    print(f"  Mean per segment: {np.mean(seg_area_cm2):.2f} cm²")
    return seg_area_cm2, seg_area_m2


# ============================================================================
# Step 4: Re-run 3D photosynthesis solve — extract per-segment transpiration
# ============================================================================
def get_per_segment_transpiration(plant, apar_umol, tleaf_c):
    """Run hm.solve() with 3D APAR/Tleaf; return per-leaf-segment Ev [cm³/d].

    Returns None if the solve fails (caller should skip LE-vs-Ev comparison).
    """
    print("\n--- Step 4: Re-run 3D Solve for Per-Segment Transpiration ---")

    from plantbox.functional.phloem_flux import PhloemFluxPython
    from plantbox.functional.PlantHydraulicParameters import PlantHydraulicParameters

    params = PlantHydraulicParameters()
    params.read_parameters(PHOTO_PATH + "maize_couvreur2012_hydraulics")

    hm = PhloemFluxPython(plant, params)
    hm.read_photosynthesis_parameters(
        filename=PHOTO_PATH + "maize_C4_photosynthesis_parameters")
    hm.read_phloem_parameters(filename=get_phloem_json())

    depth = 100
    p_s = np.linspace(SOIL_PSI_CM, SOIL_PSI_CM - depth, depth)
    es  = hm.get_es(float(np.mean(tleaf_c)))
    ea  = es * RH

    par_mol = apar_umol * 1e-6 * SECS_PER_DAY * 1e-4   # → mol/cm²/d

    seg_leaves_idx = plant.getSegmentIds(4)
    n_leaf = len(seg_leaves_idx)
    assert len(par_mol)  == n_leaf, f"PAR size {len(par_mol)} != {n_leaf}"
    assert len(tleaf_c)  == n_leaf, f"Tleaf size {len(tleaf_c)} != {n_leaf}"

    print(f"  Leaf segments: {n_leaf}")
    print(f"  PAR mean: {np.mean(apar_umol):.1f} µmol/m²/s")
    print(f"  Tleaf mean: {np.mean(tleaf_c):.2f} °C")

    try:
        hm.solve(
            sim_time=SIMULATION_DAYS,
            rsx=p_s,
            cells=True,
            ea=ea,
            es=es,
            PAR=par_mol,
            TairC=tleaf_c,
            verbose=0,
        )
    except Exception as e:
        print(f"  WARNING: hm.solve() raised: {e}")
        return None

    transp_raw = np.array(hm.get_transpiration())
    print(f"  get_transpiration() size={len(transp_raw)}, "
          f"sum={np.sum(transp_raw):.3f} cm³/d")

    # Determine whether the array covers all segments or only leaf segments
    if len(transp_raw) == n_leaf:
        transp_leaf = transp_raw
    elif len(transp_raw) > n_leaf:
        transp_leaf = transp_raw[list(seg_leaves_idx)]
    else:
        print(f"  WARNING: unexpected transpiration array size {len(transp_raw)}")
        return None

    total_mmol = float(np.sum(transp_leaf) / MW_H2O * 1000)
    print(f"  Total Ev: {total_mmol:.1f} mmol H2O/d")
    return transp_leaf   # cm³/d, ordered as CSV leaf rows


# ============================================================================
# Step 5: Convert Baleno LE_Wm2 → mmol H2O/d per segment
# ============================================================================
def compute_le_baleno_mmol(merged, seg_area_m2):
    """LE [W/m²] × area [m²] × 86400 [s/d] / 44000 [J/mol] × 1000 [mmol/mol]."""
    print("\n--- Step 5: Baleno LE → mmol H2O/d ---")
    le_wm2    = np.array([r['LE_Wm2'] for r in merged])
    le_mmol_d = le_wm2 * seg_area_m2 * SECS_PER_DAY / LAMBDA_J_MOL * 1000.0
    print(f"  LE range: [{le_wm2.min():.2f}, {le_wm2.max():.2f}] W/m²")
    print(f"  Total LE: {np.sum(le_mmol_d):.1f} mmol H2O/d")
    return le_mmol_d


# ============================================================================
# Step 6: Create 6-panel validation figure
# ============================================================================
def create_validation_figure(merged, z_cm, xyz_cm, seg_area_m2,
                             le_mmol_d, ev_leaf_mmol_d,
                             organ_counts, output_path):
    """Create Phase 5 validation figure (6 panels, 20×12 inches)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from mpl_toolkits.mplot3d import Axes3D   # noqa: F401 (registers 3d projection)

    print("\n--- Step 6: Create Validation Figure ---")

    n      = len(merged)
    apar   = np.array([r['apar_umol_m2_s'] for r in merged])
    an_3d  = np.array([r['An_3d']           for r in merged])
    an_uni = np.array([r['An_uniform']       for r in merged])
    rn     = np.array([r['Rn_Wm2']          for r in merged])
    le_wm2 = np.array([r['LE_Wm2']          for r in merged])
    h_wm2  = np.array([r['H_Wm2']           for r in merged])
    organs = [r['organ'] for r in merged]

    sunlit = apar > SUNLIT_THRESHOLD

    # Per-organ colours (tab20, max 20 organs)
    organ_names   = list(organ_counts.keys())
    cmap_org      = plt.cm.tab20
    organ_color   = {nm: cmap_org(i / len(organ_names))
                     for i, nm in enumerate(organ_names)}
    seg_rgba      = np.array([organ_color[o] for o in organs])

    # Vertical bins (0–190 cm in 10 cm steps)
    bins   = np.arange(0, 200, 10)
    bin_id = np.digitize(z_cm, bins)   # bin_id=1 → [0,10), 2 → [10,20), …

    fig = plt.figure(figsize=(20, 12), facecolor='white')
    fig.suptitle(
        f"Phase 5: CPlantBox–Baleno Coupling Validation  |  Day {SIMULATION_DAYS}\n"
        f"{n} leaf segments  ·  {len(organ_names)} leaf organs",
        fontsize=13, fontweight='bold', y=0.99,
    )
    gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.36,
                          left=0.06, right=0.97, top=0.93, bottom=0.06)

    # ------------------------------------------------------------------
    # TL  Vertical profile: APAR and An vs Z-height
    # ------------------------------------------------------------------
    ax = fig.add_subplot(gs[0, 0])

    apar_prof, an3d_prof, anuni_prof, z_cent = [], [], [], []
    for b in np.unique(bin_id):
        mask = bin_id == b
        if np.sum(mask) < 2:
            continue
        apar_prof.append(np.mean(apar[mask]))
        an3d_prof.append(np.mean(an_3d[mask]))
        anuni_prof.append(np.mean(an_uni[mask]))
        z_low = bins[b - 1] if (b - 1) < len(bins) else bins[-1]
        z_cent.append(float(z_low + 5))

    apar_prof  = np.array(apar_prof)
    an3d_prof  = np.array(an3d_prof)
    anuni_prof = np.array(anuni_prof)
    z_cent     = np.array(z_cent)

    c_apar = '#D4A010'
    c_3d   = '#228B22'
    c_uni  = '#E07020'

    ln1 = ax.plot(apar_prof, z_cent, 'o-', color=c_apar, lw=2, ms=5, label='aPAR')
    ax.set_xlabel('aPAR (µmol m$^{-2}$ s$^{-1}$)', color=c_apar, fontsize=9)
    ax.tick_params(axis='x', colors=c_apar)
    ax.set_ylabel('Height (cm)', fontsize=9)
    ax.set_title('Vertical APAR & An Profile', fontsize=10)
    ax.set_ylim(0, max(z_cent.max() * 1.08, 10))

    ax2 = ax.twiny()
    ln2 = ax2.plot(an3d_prof, z_cent, 's--', color=c_3d, lw=2, ms=5, label='An 3D')
    ln3 = ax2.plot(anuni_prof, z_cent, '^:', color=c_uni, lw=1.5, ms=4, label='An unif.')
    ax2.set_xlabel('An (µmol CO$_2$ m$^{-2}$ s$^{-1}$)', color=c_3d, fontsize=9)
    ax2.tick_params(axis='x', colors=c_3d)

    lns  = ln1 + ln2 + ln3
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, fontsize=7, loc='upper left')

    # ------------------------------------------------------------------
    # TC  LE (Baleno) vs Ev (CPlantBox) per organ
    # ------------------------------------------------------------------
    ax = fig.add_subplot(gs[0, 1])
    if ev_leaf_mmol_d is not None:
        offset = 0
        le_org, ev_org, labels_short = [], [], []
        for nm, ns in organ_counts.items():
            le_org.append(float(np.sum(le_mmol_d[offset:offset + ns])))
            ev_org.append(float(np.sum(ev_leaf_mmol_d[offset:offset + ns])))
            labels_short.append(nm.replace('leaf_', 'L'))
            offset += ns

        le_org = np.array(le_org)
        ev_org = np.array(ev_org)

        ax.scatter(ev_org, le_org,
                   c=[organ_color[nm] for nm in organ_counts],
                   s=60, zorder=5, edgecolors='#333', lw=0.5)
        for i, lbl in enumerate(labels_short):
            ax.annotate(lbl, (ev_org[i], le_org[i]),
                        fontsize=6, ha='center', va='bottom',
                        xytext=(0, 3), textcoords='offset points')

        lo = min(ev_org.min(), le_org.min()) * 0.85
        hi = max(ev_org.max(), le_org.max()) * 1.12
        ax.plot([lo, hi], [lo, hi], 'k--', lw=1.2, label='1:1')
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel('CPlantBox Ev (mmol H$_2$O d$^{-1}$)', fontsize=9)
        ax.set_ylabel('Baleno LE (mmol H$_2$O d$^{-1}$)', fontsize=9)
        ax.legend(fontsize=8)

        tot_le = float(np.sum(le_org))
        tot_ev = float(np.sum(ev_org))
        pct    = (tot_le - tot_ev) / max(abs(tot_ev), 1e-9) * 100
        ax.set_title(f'LE vs Ev per Organ\n'
                     f'LE={tot_le:.0f}  Ev={tot_ev:.0f} mmol d$^{{-1}}$ '
                     f'({pct:+.1f}%)', fontsize=9)
    else:
        ax.text(0.5, 0.5, 'Ev unavailable\n(hm.solve failed)',
                ha='center', va='center', transform=ax.transAxes, fontsize=10)
        ax.set_title('LE vs Ev per Organ', fontsize=10)

    # ------------------------------------------------------------------
    # TR  Energy balance closure: Rn vs (LE + H)
    # ------------------------------------------------------------------
    ax = fig.add_subplot(gs[0, 2])
    leh = le_wm2 + h_wm2
    ax.scatter(leh, rn, s=2, alpha=0.35, c=seg_rgba)

    lo = min(float(leh.min()), float(rn.min()))
    hi = max(float(leh.max()), float(rn.max()))
    pad = (hi - lo) * 0.05
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], 'k--', lw=1.5, label='1:1')
    ax.set_xlabel('LE + H (W m$^{-2}$)', fontsize=9)
    ax.set_ylabel('R$_n$ (W m$^{-2}$)', fontsize=9)
    ax.legend(fontsize=8)

    mean_eb_err = float(np.mean(np.abs(rn - leh)))
    ax.set_title(f'Energy Balance Closure\n'
                 f'Mean |Rn−(LE+H)| = {mean_eb_err:.2f} W m$^{{-2}}$', fontsize=9)

    # ------------------------------------------------------------------
    # BL  3D isometric scatter coloured by An_3D
    # ------------------------------------------------------------------
    ax = fig.add_subplot(gs[1, 0], projection='3d')
    sc = ax.scatter(xyz_cm[:, 0], xyz_cm[:, 1], xyz_cm[:, 2],
                    c=an_3d, cmap='viridis', s=3, alpha=0.7,
                    vmin=np.percentile(an_3d, 2), vmax=np.percentile(an_3d, 98))
    cb = fig.colorbar(sc, ax=ax, shrink=0.55, pad=0.08, aspect=20)
    cb.set_label('An$_{3D}$ (µmol CO$_2$ m$^{-2}$ s$^{-1}$)', fontsize=7)
    cb.ax.tick_params(labelsize=6)
    ax.set_xlabel('X (cm)', fontsize=7, labelpad=1)
    ax.set_ylabel('Y (cm)', fontsize=7, labelpad=1)
    ax.set_zlabel('Z (cm)', fontsize=7, labelpad=1)
    ax.tick_params(labelsize=6)
    ax.view_init(elev=20, azim=135)
    ax.set_title('3D An$_{3D}$ Distribution', fontsize=10)

    # ------------------------------------------------------------------
    # BC  Sunlit vs shaded boxplots
    # ------------------------------------------------------------------
    ax = fig.add_subplot(gs[1, 1])
    data_groups = [
        an_uni[sunlit],   # sunlit  / uniform
        an_3d[sunlit],    # sunlit  / 3D
        an_uni[~sunlit],  # shaded  / uniform
        an_3d[~sunlit],   # shaded  / 3D
    ]
    positions  = [1, 2, 4, 5]
    box_colors = [c_uni, c_3d, c_uni, c_3d]

    bp = ax.boxplot(data_groups, positions=positions, widths=0.6,
                    patch_artist=True, showfliers=False,
                    medianprops=dict(color='black', lw=2))
    for patch, col in zip(bp['boxes'], box_colors):
        patch.set_facecolor(col)
        patch.set_alpha(0.7)

    ax.set_xticks(positions)
    ax.set_xticklabels(['Sunlit\nUniform', 'Sunlit\n3D',
                        'Shaded\nUniform', 'Shaded\n3D'], fontsize=8)
    ax.axvline(3, color='gray', lw=1, ls='--', alpha=0.6)
    ax.set_ylabel('An (µmol CO$_2$ m$^{-2}$ s$^{-1}$)', fontsize=9)
    n_sun, n_sha = int(np.sum(sunlit)), int(np.sum(~sunlit))
    ax.set_title(f'Sunlit (n={n_sun}) vs Shaded (n={n_sha})\n'
                 f'Threshold = {SUNLIT_THRESHOLD:.0f} µmol m$^{{-2}}$ s$^{{-1}}$',
                 fontsize=9)
    ax.legend(handles=[Patch(fc=c_uni, alpha=0.7, label='Uniform'),
                       Patch(fc=c_3d,  alpha=0.7, label='3D')],
              fontsize=8, loc='upper right')

    # ------------------------------------------------------------------
    # BR  Per-organ LE (Baleno) vs Ev (CPlantBox) grouped bars
    # ------------------------------------------------------------------
    ax = fig.add_subplot(gs[1, 2])
    short_labels = [nm.replace('leaf_', 'L') for nm in organ_counts]
    x = np.arange(len(short_labels))
    w = 0.35

    offset = 0
    le_per_org = []
    for ns in organ_counts.values():
        le_per_org.append(float(np.sum(le_mmol_d[offset:offset + ns])))
        offset += ns

    ax.bar(x - w / 2, le_per_org, w,
           color='steelblue', edgecolor='#333', lw=0.5, label='Baleno LE')

    if ev_leaf_mmol_d is not None:
        offset = 0
        ev_per_org = []
        for ns in organ_counts.values():
            ev_per_org.append(float(np.sum(ev_leaf_mmol_d[offset:offset + ns])))
            offset += ns
        ax.bar(x + w / 2, ev_per_org, w,
               color='coral', edgecolor='#333', lw=0.5, label='CPlantBox Ev')

    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, fontsize=7)
    ax.set_ylabel('H$_2$O flux (mmol d$^{-1}$)', fontsize=9)
    ax.set_title('Per-Organ Water Fluxes', fontsize=10)
    ax.legend(fontsize=8)

    # ------------------------------------------------------------------
    fig.savefig(str(output_path), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


# ============================================================================
# Step 7: Export validation CSV and phase5_results.json
# ============================================================================
def export_outputs(merged, z_cm, seg_area_cm2, ev_leaf_mmol_d,
                   le_mmol_d, organ_counts):
    """Write maize_day55_validation.csv and phase5_results.json."""
    print("\n--- Step 7: Export Outputs ---")

    seg_area_m2 = seg_area_cm2 * 1e-4
    apar    = np.array([r['apar_umol_m2_s'] for r in merged])
    an_3d   = np.array([r['An_3d']           for r in merged])
    an_uni  = np.array([r['An_uniform']       for r in merged])
    rn      = np.array([r['Rn_Wm2']          for r in merged])
    le_wm2  = np.array([r['LE_Wm2']          for r in merged])
    h_wm2   = np.array([r['H_Wm2']           for r in merged])
    sunlit  = apar > SUNLIT_THRESHOLD

    if ev_leaf_mmol_d is not None:
        ev_mmol_d = ev_leaf_mmol_d
        ev_cm3_d  = ev_mmol_d * MW_H2O / 1000.0
    else:
        ev_mmol_d = np.zeros(len(merged))
        ev_cm3_d  = np.zeros(len(merged))

    # --- Validation CSV ---
    csv_out = OUTPUT_DIR / 'maize_day55_validation.csv'
    with open(csv_out, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'organ', 'segment_idx',
            'apar_umol_m2_s', 'tleaf_c',
            'An_3d_umol_m2_s', 'An_uniform_umol_m2_s',
            'Rn_Wm2', 'LE_Wm2', 'H_Wm2', 'EB_error_Wm2',
            'z_cm', 'seg_area_cm2', 'Ev_cm3_d', 'LE_baleno_mmol_d', 'sunlit',
        ])
        for i, r in enumerate(merged):
            writer.writerow([
                r['organ'], r['segment_idx'],
                f"{r['apar_umol_m2_s']:.4f}",
                f"{r['tleaf_c']:.4f}",
                f"{r['An_3d']:.6f}",
                f"{r['An_uniform']:.6f}",
                f"{r['Rn_Wm2']:.4f}",
                f"{r['LE_Wm2']:.4f}",
                f"{r['H_Wm2']:.4f}",
                f"{r['EB_error_Wm2']:.4f}",
                f"{z_cm[i]:.2f}",
                f"{seg_area_cm2[i]:.4f}",
                f"{ev_cm3_d[i]:.6f}",
                f"{le_mmol_d[i]:.6f}",
                int(sunlit[i]),
            ])
    print(f"  CSV: {csv_out}  ({len(merged)} rows)")

    # --- Vertical profile statistics ---
    bins   = np.arange(0, 200, 10)
    bin_id = np.digitize(z_cm, bins)
    prof_z, prof_an = [], []
    for b in np.unique(bin_id):
        mask = bin_id == b
        if np.sum(mask) >= 2:
            z_low = bins[b - 1] if (b - 1) < len(bins) else bins[-1]
            prof_z.append(float(z_low + 5))
            prof_an.append(float(np.mean(an_3d[mask])))

    slope = float(np.polyfit(prof_z, prof_an, 1)[0]) if len(prof_z) >= 2 else 0.0

    # --- Energy balance per organ ---
    leh = le_wm2 + h_wm2
    eb_abs = np.abs(rn - leh)
    eb_per_organ = {}
    offset = 0
    for nm, ns in organ_counts.items():
        eb_per_organ[nm] = float(np.mean(eb_abs[offset:offset + ns]))
        offset += ns

    # --- Sunlit / shaded stats ---
    n_sun = int(np.sum(sunlit))
    n_sha = int(np.sum(~sunlit))

    def safe_mean(arr, mask):
        return float(np.mean(arr[mask])) if np.sum(mask) > 0 else 0.0

    # --- Transpiration comparison ---
    tot_le = float(np.sum(le_mmol_d))
    tot_ev = float(np.sum(ev_mmol_d)) if ev_leaf_mmol_d is not None else None
    pct_diff = (
        (tot_le - tot_ev) / max(abs(tot_ev), 1e-9) * 100
        if tot_ev is not None else None
    )

    results = {
        'phase': 'Phase 5: Coupling Validation',
        'simulation_days': SIMULATION_DAYS,
        'n_leaf_segments': len(merged),
        'n_organs': len(organ_counts),
        'total_leaf_area_m2': float(np.sum(seg_area_m2)),
        'transpiration': {
            'LE_baleno_total_mmol_d': tot_le,
            'Ev_cpb_total_mmol_d':   tot_ev,
            'pct_difference':         pct_diff,
        },
        'energy_balance_closure': {
            'mean_abs_error_Wm2': float(np.mean(eb_abs)),
            'max_abs_error_Wm2':  float(np.max(eb_abs)),
            'per_organ':          eb_per_organ,
        },
        'vertical_profile': {
            'an_3d_gradient_umol_m2_s_per_cm': slope,
            'z_range_cm':   [float(z_cm.min()), float(z_cm.max())],
            'n_bins':        len(prof_z),
            'profile_z_cm':  prof_z,
            'profile_an_3d': prof_an,
        },
        'sunlit_shaded': {
            'threshold_umol_m2_s':  SUNLIT_THRESHOLD,
            'n_sunlit':             n_sun,
            'n_shaded':             n_sha,
            'sunlit_frac':          float(n_sun / len(merged)),
            'mean_An_3d_sunlit':    safe_mean(an_3d, sunlit),
            'mean_An_3d_shaded':    safe_mean(an_3d, ~sunlit),
            'mean_An_uni_sunlit':   safe_mean(an_uni, sunlit),
            'mean_An_uni_shaded':   safe_mean(an_uni, ~sunlit),
        },
        'apar_stats': {
            'mean': float(np.mean(apar)),
            'std':  float(np.std(apar)),
            'min':  float(np.min(apar)),
            'max':  float(np.max(apar)),
        },
    }

    json_out = OUTPUT_DIR / 'phase5_results.json'
    with open(json_out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  JSON: {json_out}")
    return results


# ============================================================================
# Main
# ============================================================================
def main():
    print("=" * 70)
    print("Phase 5: CPlantBox–Baleno Coupling Validation")
    print("=" * 70)

    # Step 1: Load and merge CSVs
    merged, organ_counts = load_and_merge_csvs()

    # Step 2: Grow day-55 plant; extract segment positions
    print("\n--- Growing Day-55 Plant ---")
    plant = grow_plant(XML_PATH, SIMULATION_DAYS, enable_photosynthesis=True)
    z_cm, xyz_cm = get_segment_positions(plant)
    assert len(z_cm) == len(merged), (
        f"Z-height count {len(z_cm)} != CSV rows {len(merged)}"
    )

    # Step 3: Compute leaf areas from OBJ
    seg_area_cm2, seg_area_m2 = compute_segment_areas(merged)
    assert len(seg_area_cm2) == len(merged), (
        f"Area count {len(seg_area_cm2)} != CSV rows {len(merged)}"
    )

    # Step 4: Re-run 3D solve for per-segment Ev
    apar_umol = np.array([r['apar_umol_m2_s'] for r in merged])
    tleaf_c   = np.array([r['tleaf_c']         for r in merged])
    ev_cm3_d  = get_per_segment_transpiration(plant, apar_umol, tleaf_c)

    ev_leaf_mmol_d = None
    if ev_cm3_d is not None:
        ev_leaf_mmol_d = ev_cm3_d / MW_H2O * 1000.0   # cm³/d → mmol/d

    # Step 5: Baleno LE → mmol H2O/d
    le_mmol_d = compute_le_baleno_mmol(merged, seg_area_m2)

    # Step 6: Validation figure
    fig_path = OUTPUT_DIR / 'maize_day55_validation_figure.png'
    create_validation_figure(
        merged, z_cm, xyz_cm, seg_area_m2,
        le_mmol_d, ev_leaf_mmol_d,
        organ_counts, fig_path,
    )

    # Step 7: Export CSV + JSON
    results = export_outputs(
        merged, z_cm, seg_area_cm2,
        ev_leaf_mmol_d, le_mmol_d, organ_counts,
    )

    # Summary
    print(f"\n{'=' * 70}")
    print(f"Phase 5 Complete!")
    print(f"  Total leaf area:    {results['total_leaf_area_m2']:.4f} m²")
    print(f"  EB mean error:      "
          f"{results['energy_balance_closure']['mean_abs_error_Wm2']:.2f} W/m²")
    print(f"  An gradient:        "
          f"{results['vertical_profile']['an_3d_gradient_umol_m2_s_per_cm']:.4f} "
          f"µmol/m²/s per cm height")
    print(f"  Sunlit/Shaded:      "
          f"{results['sunlit_shaded']['n_sunlit']} / "
          f"{results['sunlit_shaded']['n_shaded']}")
    if results['transpiration']['Ev_cpb_total_mmol_d'] is not None:
        print(f"  LE (Baleno):        "
              f"{results['transpiration']['LE_baleno_total_mmol_d']:.1f} mmol H2O/d")
        print(f"  Ev (CPlantBox):     "
              f"{results['transpiration']['Ev_cpb_total_mmol_d']:.1f} mmol H2O/d")
        print(f"  LE vs Ev diff:      "
              f"{results['transpiration']['pct_difference']:+.1f}%")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
