#!/usr/bin/env python3
"""
Calibrate maize.xml by merging MaizeField3D mature dimensions with Pheno4D
early-growth dynamics.

Data sources:
  - MaizeField3D (required): Target dimensions — lmax, Width_blade, areaMax,
    leaf_geometry, stem params.  520 plants at anthesis, 11 leaf positions.
  - Pheno4D trajectory (optional): Growth dynamics — r (growth rate per
    position), tropismS (curvature), theta (insertion angle).  7 timesteps
    over days 0-12, up to 4 leaf positions observed.

Merge strategy:
  1. MaizeField3D provides the structural parameters for all 11 positions
  2. Pheno4D (when provided) overrides r, tropismS, theta for observed
     positions (0-N)
  3. Positions beyond Pheno4D coverage get lmax-scaled extrapolated dynamics

Output: A single maize_calibrated.xml with per-position leaf subtypes that
grows correctly from seedling to mature.

Usage:
  # Both datasets (recommended)
  python calibrate_maize_xml.py \
      --maizefield3d ../MaizeField3d/maizefield3d_stats.json \
      --trajectory Maize01_g1_output/trajectory/trajectory.json \
      --template /home/lukas/PHD/CPlantBox/modelparameter/structural/plant/maize.xml \
      --output maize_calibrated.xml

  # MaizeField3D only (r defaults to 4.0 for all positions)
  python calibrate_maize_xml.py \
      --maizefield3d ../MaizeField3d/maizefield3d_stats.json \
      --template /home/lukas/PHD/CPlantBox/modelparameter/structural/plant/maize.xml \
      --output maize_calibrated.xml
"""

import json
import math
import numpy as np
import xml.etree.ElementTree as ET
import argparse


DEFAULT_LEAF_GEOMETRY = [
    (-4.0, 0.10),
    (-2.7, 0.70),
    (-1.3, 0.95),
    (0.0, 1.00),
    (1.3, 0.90),
    (2.7, 0.45),
    (4.0, 0.02)
]


def load_trajectory(filepath):
    """Load trajectory.json produced by batch_extract_morphology.py."""
    with open(filepath, 'r') as f:
        return json.load(f)


def _integrate_leaf_profile_shape_factor(leaf_geometry):
    """Compute shape factor from leaf geometry profile via trapezoidal integration.

    The leaf geometry profile is a list of (phi, x) pairs where x is the
    normalized width at station phi. The shape factor is the mean x value
    across the profile, representing the fraction of the bounding rectangle
    (lmax × max_width) that the leaf actually covers.

    Returns ~0.73 for typical maize leaf profiles.
    """
    if not leaf_geometry or len(leaf_geometry) < 2:
        return 0.73  # reasonable default for maize

    phi_vals = [g[0] for g in leaf_geometry]
    x_vals = [g[1] for g in leaf_geometry]
    phi_range = phi_vals[-1] - phi_vals[0]
    if phi_range < 1e-6:
        return 0.73

    # Trapezoidal integration (compatible with numpy 1.x and 2.x)
    integral = 0.0
    for i in range(len(phi_vals) - 1):
        integral += 0.5 * (x_vals[i] + x_vals[i + 1]) * (phi_vals[i + 1] - phi_vals[i])
    return integral / phi_range


def load_maizefield3d_stats(filepath):
    """Load pre-computed MaizeField3D per-position stats.

    Returns (per_position, stem_stats) where per_position is a list of dicts
    and stem_stats contains stem lmax/ln/lb.

    Corrects Width_blade and areaMax using median_max_width_cm and profile
    integration to avoid double-counting of taper. The JSON stores Width_blade
    from median(median_widths) which already includes taper (~75% of max),
    and areaMax applies an additional 0.7 shape factor. The fix uses
    median_max_width_cm (true max width) and integrates the leaf profile
    for the correct shape factor.
    """
    with open(filepath, 'r') as f:
        data = json.load(f)

    per_position = []
    for s in data['per_position']:
        if s is None:
            continue
        lg = s.get('leaf_geometry')
        if lg:
            lg = [(g[0], g[1]) for g in lg]

        # Correct Width_blade and areaMax if median_max_width_cm available
        median_max = s.get('median_max_width_cm')
        lmax = s['lmax']
        if median_max is not None and lg:
            shape_factor = _integrate_leaf_profile_shape_factor(lg)
            width_blade = median_max / 2.0
            area_max = lmax * median_max * shape_factor
            width_petiole = width_blade * 0.3
        else:
            # Fallback to stored values
            width_blade = s['Width_blade']
            area_max = s['areaMax']
            width_petiole = s.get('Width_petiole', width_blade * 0.3)
            shape_factor = None

        per_position.append({
            'position': s['position'],
            'lmax': lmax,
            'r': s.get('r', 4.0),
            'Width_blade': width_blade,
            'Width_petiole': width_petiole,
            'areaMax': area_max,
            'theta': s['theta'],
            'tropismS': s.get('tropismS', 0.15),
            'tropismAge': s.get('tropismAge', 5.0),
            'leaf_geometry': lg,
            'median_length_cm': s.get('median_length_cm'),
        })

    stem_stats = data.get('stem', {})

    print(f"  Loaded MaizeField3D stats: {data.get('n_plants', '?')} plants, "
          f"{len(per_position)} positions")
    print(f"  Stem: lmax={stem_stats.get('lmax', '?')}, ln={stem_stats.get('ln', '?')}, "
          f"lb={stem_stats.get('lb', '?')}")
    print(f"  Leaf width correction (median_max_width → Width_blade, profile-integrated areaMax):")
    for p in per_position:
        pos = p['position']
        orig = data['per_position'][pos]
        if orig and orig.get('median_max_width_cm'):
            print(f"    Pos {pos:>2}: Wb {orig['Width_blade']:.2f} -> {p['Width_blade']:.2f}, "
                  f"aM {orig['areaMax']:.1f} -> {p['areaMax']:.1f}")
        else:
            print(f"    Pos {pos:>2}: (no correction, using stored values)")

    return per_position, stem_stats


def extract_pheno4d_dynamics(trajectory):
    """Extract per-position growth dynamics from Pheno4D trajectory.

    Only extracts the parameters that represent temporal dynamics:
      - r: growth rate (cm/day) from dl/dt between timesteps
      - tropismS: droop strength from measured curvature
      - theta: insertion angle from stem skeleton

    Does NOT extract lmax, Width_blade, areaMax, leaf_geometry — those come
    from MaizeField3D.

    Returns:
        list of dicts with {position, r, tropismS, theta}
    """
    timesteps = trajectory.get('timesteps', trajectory.get('scans', []))

    # Collect per-position data across all timesteps
    pos_data = {}

    for ts in timesteps:
        day = ts['day']
        lengths = ts.get('leaf_lengths_cm', [])
        angles_deg = ts.get('leaf_angles_deg', [])
        tropism_hints = ts.get('leaf_tropism_hints', [])

        for i in range(len(lengths)):
            if i not in pos_data:
                pos_data[i] = {
                    'lengths_over_time': [],
                    'angles_deg': [],
                    'tropismS_hints': [],
                }

            pos_data[i]['lengths_over_time'].append((day, lengths[i]))

            if i < len(angles_deg):
                pos_data[i]['angles_deg'].append(angles_deg[i])
            if i < len(tropism_hints):
                pos_data[i]['tropismS_hints'].append(
                    tropism_hints[i].get('tropismS', 0.05))

    dynamics = []
    for i in sorted(pos_data.keys()):
        pd = pos_data[i]

        # Growth rate from length trajectory
        pts = sorted(pd['lengths_over_time'])
        r = 2.0  # default
        if len(pts) > 1:
            days_arr = np.array([p[0] for p in pts])
            lens_arr = np.array([p[1] for p in pts])
            dt = np.diff(days_arr)
            dl = np.diff(lens_arr)
            valid = (dt > 0) & (dl > 0)
            if valid.any():
                r = max(0.5, float(np.mean(dl[valid] / dt[valid])))

        # Tropism: measured curvature hints, scaled for gravitropism (tropismT=1)
        # With pure gravitropism, lower values needed than age-switching (tropismT=6)
        raw_tropS = float(np.mean(pd['tropismS_hints'])) if pd['tropismS_hints'] else 0.05
        tropismS = np.clip(raw_tropS * 3.0, 0.05, 0.25)

        # Theta: mean of angles, min 30 deg (position-dependent floor applied later)
        # With gravitropism (tropismT=1), theta determines emergence angle
        # Young leaves should be more erect, old leaves more horizontal
        angles = pd['angles_deg'] if pd['angles_deg'] else [50.0]
        theta = float(np.radians(max(np.mean(angles), 30.0)))

        dynamics.append({
            'position': i,
            'r': r,
            'tropismS': tropismS,
            'theta': theta,
        })

    print(f"\n  Pheno4D dynamics extracted for {len(dynamics)} positions:")
    print(f"    {'Pos':>3} {'r':>6} {'tropS':>7} {'theta':>7}")
    for d in dynamics:
        print(f"    {d['position']:>3} {d['r']:>6.1f} {d['tropismS']:>7.3f} "
              f"{d['theta']:>7.2f}")

    return dynamics


def merge_pheno4d_dynamics(mf3d_positions, pheno4d_dynamics):
    """Merge Pheno4D growth dynamics into MaizeField3D structural parameters.

    For positions covered by Pheno4D: use measured r, tropismS, theta.
    For positions beyond Pheno4D: extrapolate using lmax-scaled hybrid.

    The lmax-scaled hybrid works as follows:
      - Take the growth rate at the last observed Pheno4D position as reference
      - Scale it by the ratio of lmax at the target position to lmax at the
        reference position
      - Rationale: bigger leaves (higher lmax) grow proportionally faster

    Args:
        mf3d_positions: list of MaizeField3D position dicts (modified in place)
        pheno4d_dynamics: list of dicts from extract_pheno4d_dynamics()
    """
    # Build lookup by position index
    dyn_by_pos = {d['position']: d for d in pheno4d_dynamics}
    n_pheno4d = len(pheno4d_dynamics)

    if n_pheno4d == 0:
        print("\n  No Pheno4D dynamics to merge")
        return

    # Reference position for lmax-scaled extrapolation: last observed
    last_dyn = pheno4d_dynamics[-1]
    ref_pos = last_dyn['position']
    ref_r = last_dyn['r']
    ref_tropS = last_dyn['tropismS']
    ref_theta = last_dyn['theta']

    # Get lmax at reference position from MaizeField3D
    ref_lmax = None
    for p in mf3d_positions:
        if p['position'] == ref_pos:
            ref_lmax = p['lmax']
            break
    if ref_lmax is None or ref_lmax < 1.0:
        ref_lmax = 50.0  # safety fallback

    print(f"\n  Merging Pheno4D dynamics into MaizeField3D structure:")
    print(f"  Reference position: {ref_pos} (r={ref_r:.1f}, lmax={ref_lmax:.1f})")
    print(f"    {'Pos':>3} {'r_old':>6} {'r_new':>6} {'tropS_old':>9} {'tropS_new':>9} "
          f"{'theta_old':>9} {'theta_new':>9} {'source':>12}")

    for p in mf3d_positions:
        pos = p['position']
        old_r = p['r']
        old_tropS = p['tropismS']
        old_theta = p['theta']

        if pos in dyn_by_pos:
            # Direct override from Pheno4D measurement
            d = dyn_by_pos[pos]
            p['r'] = d['r']
            p['tropismS'] = d['tropismS']
            p['theta'] = d['theta']
            source = "Pheno4D"
        else:
            # lmax-scaled extrapolation
            lmax_ratio = p['lmax'] / ref_lmax
            p['r'] = np.clip(ref_r * lmax_ratio, 0.5, 8.0)
            p['tropismS'] = np.clip(ref_tropS * lmax_ratio, 0.02, 0.25)
            # Theta: blend reference toward target (position-dependent floor applied later)
            blend = min(1.0, (pos - ref_pos) / max(1, len(mf3d_positions) - ref_pos - 1))
            target_theta = math.radians(75.0)
            p['theta'] = ref_theta + blend * (target_theta - ref_theta)
            source = f"extrapolated"

        print(f"    {pos:>3} {old_r:>6.1f} {p['r']:>6.1f} "
              f"{old_tropS:>9.3f} {p['tropismS']:>9.3f} "
              f"{old_theta:>9.2f} {p['theta']:>9.2f} {source:>12}")


def load_pheno4d_emergence_overrides(path, min_n_plants=2):
    """Read Pheno4D emergence time series and return per-position overrides.

    Drop-in replacement for the 3.0-day / 57.9 degCd phyllochron heuristics in
    :func:`generate_per_leaf_xml`. Only positions that pass ``min_n_plants``
    (default 2) are emitted; positions beyond Pheno4D's 7-plant / 4-label
    coverage fall back to the existing heuristic in the caller.

    Source JSON is produced by
    ``Resources/Pheno4D/extract_emergence_timeseries.py``. The loader reads
    the per-position aggregate block, NOT the raw per-plant scatter, so
    downstream consumers only see calibrated scalars.

    Args:
        path: Path to pheno4d_emergence_timeseries.json.
        min_n_plants: Drop positions with fewer plants than this (default 2).

    Returns:
        ``{position (int): {'ldelay_days': float, 'tt_emergence_degCd': float,
                            'phyllochron_days_from_prev': float,
                            'phyllochron_tt_degCd_from_prev': float,
                            'n_plants': int, 'leaf_label': int}}``

        ``position`` is the CPlantBox position index (``leaf_label - 2``),
        matching the indexing used by ``generate_per_leaf_xml``.
    """
    with open(path, 'r') as f:
        blob = json.load(f)
    if 'per_position' not in blob:
        raise ValueError(
            f"{path} does not carry a 'per_position' block; "
            f"regenerate via Resources/Pheno4D/extract_emergence_timeseries.py"
        )
    overrides = {}
    for entry in blob['per_position'].values():
        n = int(entry.get('n_plants', 0))
        if n < min_n_plants:
            continue
        pos = int(entry['cplantbox_position'])
        overrides[pos] = {
            'ldelay_days': float(entry['emergence_day_mean']),
            'tt_emergence_degCd': float(entry['tt_emergence_degCd']),
            'phyllochron_days_from_prev': float(
                entry.get('phyllochron_days_from_prev', 0.0)
            ),
            'phyllochron_tt_degCd_from_prev': float(
                entry.get('phyllochron_tt_degCd_from_prev', 0.0)
            ),
            'n_plants': n,
            'leaf_label': int(entry.get('leaf_label', pos + 2)),
        }
    return overrides, blob.get('tt_assumption', {})


def merge_pheno4d_emergence_overrides(mf3d_positions, overrides, tt_meta=None):
    """Stamp per-position Pheno4D emergence overrides onto the calibration
    stats. ``generate_per_leaf_xml`` reads ``ldelay_override_days`` and
    ``tt_emergence_override_degCd`` from each stats dict when emitting the
    per-leaf XML.

    Coverage gaps. Pheno4D only observes the lowest 2–3 leaf positions
    reliably; upper positions have no measurement. To keep the emergence
    sequence monotonic across the whole plant, the mean per-prev
    phyllochron from the covered positions is extrapolated upward: each
    uncovered position ``p > last_covered`` gets
    ``ldelay[last_covered] + (p - last_covered) * mean_phyllochron_days``.
    Lower positions (below ``min(covered)``) fall back to the existing
    ``pos * phyllochron`` heuristic — Pheno4D's baseline already anchors
    position 0 at day 0, so no gap there in practice.

    Modifies ``mf3d_positions`` in place.
    """
    if not overrides:
        print("\n  No Pheno4D emergence overrides to merge")
        return
    covered_positions = sorted(overrides.keys())
    last_covered = covered_positions[-1]
    # Mean per-prev phyllochron from covered positions (skip pos 0, whose
    # "from prev" is definitionally 0 as the anchor).
    phyll_samples = [
        overrides[p]['phyllochron_days_from_prev']
        for p in covered_positions
        if overrides[p]['phyllochron_days_from_prev'] > 0
    ]
    mean_phyll_days = (
        sum(phyll_samples) / len(phyll_samples) if phyll_samples else 0.0
    )
    gdd_per_day = (tt_meta or {}).get('GDD_per_day_degCd', 0.0)

    print(f"\n  Merging Pheno4D emergence overrides "
          f"({len(overrides)} covered positions, "
          f"mean phyllochron = {mean_phyll_days:.2f} d "
          f"→ extrapolated to positions > {last_covered}):")
    print(f"    {'Pos':>3} {'SubT':>4} {'n':>3} "
          f"{'ldelay_d':>10} {'tt_emerge':>12} {'source':>14}")
    last_covered_ldelay = overrides[last_covered]['ldelay_days']
    for p in mf3d_positions:
        pos = p['position']
        if pos in overrides:
            ov = overrides[pos]
            p['ldelay_override_days'] = ov['ldelay_days']
            p['tt_emergence_override_degCd'] = ov['tt_emergence_degCd']
            n_str = str(ov['n_plants'])
            source = "Pheno4D"
        elif pos > last_covered and mean_phyll_days > 0:
            ld = last_covered_ldelay + (pos - last_covered) * mean_phyll_days
            tt = ld * gdd_per_day
            p['ldelay_override_days'] = ld
            p['tt_emergence_override_degCd'] = tt
            n_str = "-"
            source = "extrapolated"
        else:
            continue
        print(f"    {pos:>3} {pos + 2:>4} {n_str:>3} "
              f"{p['ldelay_override_days']:>10.2f} "
              f"{p['tt_emergence_override_degCd']:>12.1f} {source:>14}")


def load_pheno4d_stem_overrides(path, min_n_plants_internode=2,
                                 min_n_samples_internode=3,
                                 min_n_plants_stem_r=3):
    """Read Pheno4D stem-related scalar overrides from the timeseries JSON.

    Pulls two orthogonal signals:

    - ``per_internode_length``: median Δz (collar_z[L+1] − collar_z[L])
      per upper-label, pooled across (plant, scan) pairs. Pheno4D covers
      lower internodes only (below L3, L4 in current data). A single
      ``ln_override_cm`` is returned as the median across all usable
      upper-labels — CPlantBox's stem XML carries one ``ln`` value, so
      we collapse to a single representative lower-internode length.
      Accept the resulting undershoot on upper internodes as a
      documented trade-off (Chapter 1 focus is V1–V10 window).
    - ``stem_elongation_fit``: exponential-approach fit of top-collar z
      (frame-invariant stack height) pooled across plants. Returns
      ``r_stem_override_cm_per_day`` when ``r_usable`` is set. ``lmax``
      is NEVER returned — Pheno4D's early-V-stage window can't identify
      the mature stem ceiling (fit hits the upper bound).

    Returns a dict with only the keys we're confident about; callers
    stamp them onto ``mf3d_stem_stats``. Both keys are optional.
    """
    with open(path, 'r') as f:
        blob = json.load(f)

    out = {}

    # Internode length → single ln override (median of usable upper-labels).
    per_internode = blob.get('per_internode_length', {})
    usable_medians = []
    covered_upper_labels = []
    for entry in per_internode.values():
        if not entry.get('usable'):
            continue
        if (entry.get('n_plants', 0) < min_n_plants_internode
                or entry.get('n_samples_clean', 0) < min_n_samples_internode):
            continue
        usable_medians.append(float(entry['ln_median_cm']))
        covered_upper_labels.append(int(entry['upper_leaf_label']))
    if usable_medians:
        out['ln_override_cm'] = float(np.median(usable_medians))
        out['ln_override_covered_upper_labels'] = sorted(covered_upper_labels)
        out['ln_override_per_upper_label_medians'] = usable_medians

    # Stem elongation rate → r override (lmax explicitly NOT exported).
    stem_fit = blob.get('stem_elongation_fit', {})
    if stem_fit.get('r_usable') and stem_fit.get('n_plants', 0) >= min_n_plants_stem_r:
        out['r_stem_override_cm_per_day'] = float(stem_fit['r_cm_per_day'])
        out['r_stem_override_std'] = float(stem_fit.get('r_std_cm_per_day', 0.0))
        out['r_stem_override_n_plants'] = int(stem_fit['n_plants'])
        out['r_stem_override_max_observed_stack_cm'] = float(
            stem_fit.get('max_observed_stack_cm', 0.0)
        )

    return out


def merge_pheno4d_stem_overrides(mf3d_stem_stats, stem_ov):
    """Stamp Pheno4D stem overrides onto ``mf3d_stem_stats`` in place.

    Keys added (when present in ``stem_ov``): ``ln_override``, ``r_override``.
    The consumer (stem update in ``generate_per_leaf_xml``) reads these
    and, when present, prefers them over the hard-wired defaults. Absent
    keys mean no change; fallback path is byte-identical to pre-change.
    """
    if not stem_ov:
        print("\n  No Pheno4D stem overrides to merge")
        return
    print(f"\n  Merging Pheno4D stem overrides:")
    if 'ln_override_cm' in stem_ov:
        mf3d_stem_stats['ln_override'] = stem_ov['ln_override_cm']
        labels = stem_ov.get('ln_override_covered_upper_labels', [])
        print(f"    ln_override = {stem_ov['ln_override_cm']:.2f} cm "
              f"(median of upper-labels {labels})")
    if 'r_stem_override_cm_per_day' in stem_ov:
        mf3d_stem_stats['r_override'] = stem_ov['r_stem_override_cm_per_day']
        print(f"    r_stem_override = "
              f"{stem_ov['r_stem_override_cm_per_day']:.3f} "
              f"± {stem_ov['r_stem_override_std']:.3f} cm/day "
              f"(n_plants={stem_ov['r_stem_override_n_plants']}, "
              f"max_obs_stack={stem_ov['r_stem_override_max_observed_stack_cm']:.1f} cm)")


def apply_maize_base_taper(geometry):
    """Apply realistic maize leaf base tapering to a leaf geometry profile.

    Maize leaves start narrow at the ligule (blade-sheath junction),
    expand to maximum width around 1/3 from the base, then taper to a
    pointed tip.
    """
    if not geometry or len(geometry) < 2:
        return DEFAULT_LEAF_GEOMETRY

    phi_vals = [g[0] for g in geometry]
    phi_min = min(phi_vals)
    phi_max = max(phi_vals)
    phi_range = phi_max - phi_min
    if phi_range < 1e-6:
        return DEFAULT_LEAF_GEOMETRY

    result = []
    for phi, x in geometry:
        t = (phi - phi_min) / phi_range

        if t < 0.35:
            s = t / 0.35
            envelope = 0.10 + 0.90 * (s * s * (3 - 2 * s))
        else:
            envelope = 1.0

        result.append((phi, round(x * envelope, 3)))

    return result


def update_xml_parameter(elem, name, value, dev=None):
    """Update or create a parameter in XML element."""
    param = elem.find(f".//parameter[@name='{name}']")
    if param is not None:
        param.set('value', str(value))
        if dev is not None:
            param.set('dev', str(dev))
    else:
        new_param = ET.SubElement(elem, 'parameter')
        new_param.set('name', name)
        new_param.set('value', str(value))
        if dev is not None:
            new_param.set('dev', str(dev))


def generate_per_leaf_xml(root, per_position_stats, fallback_geometry, gf=3,
                          thermal_emergence=False, phyllochron_tt=57.9,
                          surface_cps_by_pos=None):
    """Generate one leaf subType per position and update stem successor rules.

    Args:
        root: XML root element
        per_position_stats: list of dicts with per-position parameters
        fallback_geometry: default leaf geometry profile [(phi, x), ...]
        gf: growth function type (1=exp, 2=linear, 3=CWLimited, 4=Gompertz)
        thermal_emergence: if True, emit `use_thermal_emergence=1` +
            `tt_emergence = pos * phyllochron_tt` instead of calendar-day ldelay.
            Requires Plant::getAccumulatedTT (C++ side).
        phyllochron_tt: thermal-time phyllochron [degCd], Dos Santos 2022 = 57.9 for maize.
        surface_cps_by_pos: optional mapping ``{position: (N_U, N_V, 3) leaf-local
            CPs}``. When provided, each generated leaf subtype gets a
            ``<parameter name="surface_cp" .../>`` block with the flattened
            CP grid (u-major), plus ``surface_n_u/n_v/deg_u/deg_v`` scalars.
    """
    # Remove ALL existing leaf elements
    for leaf_elem in root.findall('leaf'):
        root.remove(leaf_elem)

    # Remove branch stem subtypes (2-5) — they interfere with lateral creation
    for stem_elem in root.findall('stem'):
        st = stem_elem.get('subType', '0')
        if st not in ('0', '1'):
            root.remove(stem_elem)

    # Shared parameters for all maize leaf subtypes.
    # NOTE: CPlantBox computes leaf beta as
    #   beta = phytomerID*pi*rotBeta + pi*rand()*betaDev + initBeta*pi
    # so InitBeta / RotBeta / BetaDev are all in units of pi, NOT radians.
    # With unique subtypes per position, phytomerID is always 0, so RotBeta
    # has no effect — InitBeta (set per-leaf below, in units of pi) is what
    # places each leaf. BetaDev=0.02 gives ~3.6 deg of jitter (tight
    # distichous — prevents adjacent same-rank blades from drifting into
    # each other).
    shared_params = {
        'geometryN': '100',
        'gf': str(gf),
        'isPseudostem': '0',
        'lnf': '0',
        'parametrisationType': '1',
        'shapeType': '2',
        'tropismT': '1',  # gravitropism — leaves emerge at theta and droop gently
        'BetaDev': '0.02',
        'InitBeta': '0',
        'RotBeta': '0',
        'a': '0.04',
        'dx': '0.1',
        'dxMin': '1e-06',
        'la': '0',
        'lb': '0',
        'ln': '1',
        'rlt': '1e+09',
        'tropismN': '3',  # smoother curves
    }

    n_leaves = len(per_position_stats)
    # Maize has DISTICHOUS phyllotaxis: leaves alternate 180° (two-ranked).
    # NOT golden angle (137.5°) — that's for spiral phyllotaxis (sunflower).
    # Give same-rank leaves a small helical fan-out so a drooping upper
    # blade (e.g. pos 6) passes to the side of the one beneath it (pos 4)
    # instead of landing on top of it.
    # InitBeta is stored in UNITS OF PI (CPlantBox does `initBeta*pi`
    # internally), so pos%2 gives the 0/pi alternation as 0.0/1.0.
    rank_spread_pi = 3.0 / 180.0  # 3 deg/rank → ~6 deg between same-rank neighbours
    # Per-leaf azimuthal jitter: keep small so adjacent same-rank leaves
    # (e.g. pos 4 and pos 6) can't drift toward each other and clip.
    rng = np.random.RandomState(42)
    max_pos = max(s['position'] for s in per_position_stats) + 1
    leaf_jitter_pi = rng.uniform(-4.0 / 180.0, 4.0 / 180.0,
                                  size=max(max_pos, n_leaves))

    for stats in per_position_stats:
        pos = stats['position']
        sub_type = pos + 2

        leaf_elem = ET.SubElement(root, 'leaf')
        leaf_elem.set('name', f'maize_leaf_L{pos}')
        leaf_elem.set('subType', str(sub_type))

        rank_idx = pos // 2  # 0,0,1,1,2,2,3,3,...
        # base_angle in units of pi: pos even -> 0, pos odd -> 1 (=180°)
        base_angle_pi = float(pos % 2) + rank_idx * rank_spread_pi
        init_beta = (base_angle_pi + leaf_jitter_pi[pos]) % 2.0

        for name, value in shared_params.items():
            param = ET.SubElement(leaf_elem, 'parameter')
            param.set('name', name)
            if name == 'InitBeta':
                param.set('value', str(init_beta))
            else:
                param.set('value', value)

        # Emergence schedule: calendar-day (default) or thermal-time gated.
        # Calendar mode: phyllochron ~3 d × position → ldelay (CPlantBox honours
        # via delayDefinitionShoot=2 / dd_time_self).
        # Thermal mode: tt_emergence = position × phyllochron_tt; CPlantBox gates
        # the leaf's first growth step on plant accumulated TT crossing the threshold.
        # ldelay is still emitted as a fallback (used only if seed delayDefinitionShoot
        # disagrees), set to 0 so it cannot mask the TT gate.
        # Per-position Pheno4D overrides (from
        # merge_pheno4d_emergence_overrides) take precedence over the
        # heuristic when present. Stats dict carries at most one of
        # ``ldelay_override_days`` / ``tt_emergence_override_degCd``; missing
        # key ⇒ fall back to the original pos-times-scalar heuristic.
        if thermal_emergence:
            tt_val = stats.get('tt_emergence_override_degCd',
                               pos * phyllochron_tt)
            param = ET.SubElement(leaf_elem, 'parameter')
            param.set('name', 'use_thermal_emergence')
            param.set('value', '1')
            param = ET.SubElement(leaf_elem, 'parameter')
            param.set('name', 'tt_emergence')
            param.set('value', str(tt_val))
            param = ET.SubElement(leaf_elem, 'parameter')
            param.set('name', 'ldelay')
            param.set('value', '0')
        else:
            ldelay = stats.get('ldelay_override_days')
            if ldelay is None:
                phyllochron = stats.get('phyllochron', 3.0)
                ldelay = pos * phyllochron
            param = ET.SubElement(leaf_elem, 'parameter')
            param.set('name', 'ldelay')
            param.set('value', str(ldelay))

        # Position-specific parameters
        for name, value in [
            ('lmax', stats['lmax']),
            ('r', stats['r']),
            ('Width_blade', stats['Width_blade']),
            ('Width_petiole', stats['Width_petiole']),
            ('areaMax', stats['areaMax']),
        ]:
            param = ET.SubElement(leaf_elem, 'parameter')
            param.set('name', name)
            param.set('value', str(value))

        # theta with dev — 15% deviation adds natural insertion angle variation
        param = ET.SubElement(leaf_elem, 'parameter')
        param.set('name', 'theta')
        param.set('value', str(stats['theta']))
        param.set('dev', str(stats['theta'] * 0.15))

        # tropism — 30% deviation so same-rank leaves droop differently
        param = ET.SubElement(leaf_elem, 'parameter')
        param.set('name', 'tropismS')
        param.set('value', str(stats['tropismS']))
        param.set('dev', str(stats['tropismS'] * 0.3))

        param = ET.SubElement(leaf_elem, 'parameter')
        param.set('name', 'tropismAge')
        param.set('value', str(stats['tropismAge']))

        # Curvature spline profile (from reverse-engineering analysis)
        curv_spline = stats.get('curvature_spline')
        if curv_spline and 'phi' in curv_spline and 'kappa' in curv_spline:
            for phi_val, kappa_val in zip(curv_spline['phi'], curv_spline['kappa']):
                cp = ET.SubElement(leaf_elem, 'parameter')
                cp.set('name', 'leafCurvature')
                cp.set('phi', f"{phi_val:.4f}")
                cp.set('kappa', f"{kappa_val:.6f}")

        # leafGeometry profile. MaizeField3D peak-aligned aggregation
        # already captures real base narrowing and tip taper — don't apply
        # the synthetic base-taper envelope (would double-count shape).
        geometry = stats.get('leaf_geometry') or fallback_geometry
        if not stats.get('leaf_geometry'):
            geometry = apply_maize_base_taper(geometry)
        for phi, x in geometry:
            geom_param = ET.SubElement(leaf_elem, 'parameter')
            geom_param.set('name', 'leafGeometry')
            geom_param.set('phi', f"{phi:.1f}")
            geom_param.set('x', f"{x:.2f}")

        # Phase B: optional 2D surface CP grid (leaf-local frame).
        # The lofter/simulate code prefer this block over leafGeometry when
        # present; leafGeometry stays as the fallback for consumers that
        # haven't been updated to the 2D path.
        if surface_cps_by_pos is not None and pos in surface_cps_by_pos:
            cps = np.asarray(surface_cps_by_pos[pos], dtype=np.float64)
            if cps.ndim != 3 or cps.shape[-1] != 3:
                raise ValueError(
                    f"surface_cps_by_pos[{pos}] must be (N_U, N_V, 3); got {cps.shape}"
                )
            # Rescale local CPs so the library's intrinsic midrib arc matches
            # this subtype's ``lmax``. This makes ``scale = current_length /
            # lmax`` the correct runtime scaling at both the C++
            # (Leaf::updateNodesFromSurfaceCPs) and Python (loft_leaf_nurbs
            # library path) layers — callers don't need to know the library's
            # native length, which varies per MaizeField3D position.
            n_u_local, n_v_local = cps.shape[0], cps.shape[1]
            v_mid_idx = n_v_local // 2
            midrib = cps[:, v_mid_idx, :]
            lib_arc = float(np.sum(np.linalg.norm(np.diff(midrib, axis=0), axis=1)))
            target_lmax = float(stats['lmax'])
            if lib_arc > 1e-9 and target_lmax > 1e-9:
                cps = cps * (target_lmax / lib_arc)
            for name, value in [
                ('surface_n_u', n_u_local),
                ('surface_n_v', n_v_local),
                ('surface_deg_u', 3),
                ('surface_deg_v', 2),
            ]:
                p = ET.SubElement(leaf_elem, 'parameter')
                p.set('name', name)
                p.set('value', str(value))
            for i_u in range(n_u_local):
                for i_v in range(n_v_local):
                    cp_param = ET.SubElement(leaf_elem, 'parameter')
                    cp_param.set('name', 'surface_cp')
                    u_norm = i_u / max(n_u_local - 1, 1)
                    v_norm = i_v / max(n_v_local - 1, 1)
                    cp_param.set('u', f"{u_norm:.4f}")
                    cp_param.set('v', f"{v_norm:.4f}")
                    cp_param.set('x', f"{cps[i_u, i_v, 0]:.6f}")
                    cp_param.set('y', f"{cps[i_u, i_v, 1]:.6f}")
                    cp_param.set('z', f"{cps[i_u, i_v, 2]:.6f}")

    # Stem successor placeholder — actual per-position rules are set by
    # setup_successor_where() in grow.py at runtime via Python API
    stem_elem = root.find(".//stem[@subType='1']")
    if stem_elem is not None:
        for old_succ in stem_elem.findall(".//parameter[@name='successor']"):
            stem_elem.remove(old_succ)

        succ = ET.SubElement(stem_elem, 'parameter')
        succ.set('name', 'successor')
        succ.set('ruleId', '0')
        succ.set('numLat', '1')
        succ.set('Where', '')
        succ.set('subType', '2')
        succ.set('organType', '4')
        succ.set('percentage', '1')

    print(f"\n  Generated {n_leaves} leaf subtypes (subType {2}..{n_leaves + 1})")

    return n_leaves


def generate_phytomer_leaf_xml(root, per_position_stats, fallback_geometry, gf=3):
    """Generate sheath+blade subType pairs for phytomer decomposition.

    Each phytomer position P gets:
      - Sheath: subType = 2*P (even), isPseudostem=1, successor → blade
      - Blade:  subType = 2*P+1 (odd), isPseudostem=0, no successors

    The sheath's successor mechanism creates the blade at its tip.

    Args:
        root: XML root element
        per_position_stats: list of dicts with per-position parameters
        fallback_geometry: default leaf geometry profile [(phi, x), ...]
        gf: growth function type (1=exp, 2=linear, 3=CWLimited, 4=Gompertz)
    """
    # Remove ALL existing leaf elements
    for leaf_elem in root.findall('leaf'):
        root.remove(leaf_elem)

    # Remove branch stem subtypes (2-5)
    for stem_elem in root.findall('stem'):
        st = stem_elem.get('subType', '0')
        if st not in ('0', '1'):
            root.remove(stem_elem)

    # Dornbusch 2011 SL ratio polynomial (wheat-derived, flat 0.4 for maize)
    SL_RATIO = 0.4

    n_leaves = len(per_position_stats)

    # Phyllotaxis (same as monolithic mode). InitBeta is in units of pi.
    rank_spread_pi = 3.0 / 180.0
    rng = np.random.RandomState(42)
    max_pos = max(s['position'] for s in per_position_stats) + 1
    leaf_jitter_pi = rng.uniform(-4.0 / 180.0, 4.0 / 180.0,
                                  size=max(max_pos, n_leaves))

    for stats in per_position_stats:
        pos = stats['position']
        total_lmax = stats['lmax']
        sheath_lmax = SL_RATIO * total_lmax
        blade_lmax = (1.0 - SL_RATIO) * total_lmax

        sheath_st = 2 * pos
        blade_st = 2 * pos + 1

        # --- Phyllotaxis (shared azimuth for sheath+blade) ---
        rank_idx = pos // 2
        base_angle_pi = float(pos % 2) + rank_idx * rank_spread_pi
        init_beta = (base_angle_pi + leaf_jitter_pi[pos]) % 2.0

        # Per-position phyllochron delay
        phyllochron = stats.get('phyllochron', 3.0)
        ldelay = pos * phyllochron

        # Sheath growth rate: proportional to total r scaled by SL_RATIO
        sheath_r = max(stats['r'] * SL_RATIO, 0.3)
        # Blade growth rate: the remainder
        blade_r = max(stats['r'] * (1.0 - SL_RATIO), 0.5)

        # Sheath elongation duration → blade delay
        sheath_duration = sheath_lmax / max(sheath_r, 0.1)

        # ---- SHEATH LEAF (even subType) ----
        sheath_elem = ET.SubElement(root, 'leaf')
        sheath_elem.set('name', f'maize_sheath_P{pos}')
        sheath_elem.set('subType', str(sheath_st))

        # Thermal-time parameters (Step 2B)
        thermal_params = {}
        if stats.get('use_thermal_elongation', False):
            thermal_params = {
                'use_thermal_elongation': '1',
                'T_base': str(stats.get('T_base', 8.0)),
                'T_opt': str(stats.get('T_opt', 30.0)),
                'T_max': str(stats.get('T_max', 41.0)),
                'LER_max': str(stats.get('LER_max', 1.5)),
                'phyllochron_tt': str(stats.get('phyllochron_tt', 57.9)),
                'sl_ratio': str(SL_RATIO),
            }

        sheath_params = {
            'geometryN': '100',
            'gf': str(gf),
            'isPseudostem': '1',
            'lnf': '0',
            'parametrisationType': '1',
            'shapeType': '2',
            'tropismT': '1',
            'tropismS': '0',        # straight up (sheath is structural)
            'tropismN': '3',
            'BetaDev': '0.02',
            'InitBeta': str(init_beta),
            'RotBeta': '0',
            'a': '0.04',
            'dx': '0.1',
            'dxMin': '1e-06',
            'la': '0',
            'lb': str(sheath_lmax),  # full length is basal zone (no laterals)
            'ln': '1',
            'lmax': str(sheath_lmax),
            'r': str(sheath_r),
            'rlt': '1e+09',
            'Width_blade': '0',     # no blade area on sheath
            'Width_petiole': str(stats.get('Width_petiole', 0.35)),
            'areaMax': '0',
            'ldelay': str(ldelay),
            'theta': str(math.radians(5.0)),  # nearly vertical
            'tropismAge': '1000',   # no age-based droop
            **thermal_params,
        }

        for name, value in sheath_params.items():
            param = ET.SubElement(sheath_elem, 'parameter')
            param.set('name', name)
            param.set('value', value)

        # No successor on sheath — blade is created by stem (sibling topology)

        # ---- BLADE LEAF (odd subType) ----
        blade_elem = ET.SubElement(root, 'leaf')
        blade_elem.set('name', f'maize_blade_P{pos}')
        blade_elem.set('subType', str(blade_st))

        blade_params = {
            'geometryN': '100',
            'gf': str(gf),
            'isPseudostem': '0',
            'lnf': '0',
            'parametrisationType': '1',
            'shapeType': '2',
            'tropismT': '1',
            'tropismN': '3',
            'BetaDev': '0.02',
            'InitBeta': str(init_beta),
            'RotBeta': '0',
            'a': '0.04',
            'dx': '0.1',
            'dxMin': '1e-06',
            'la': str(blade_lmax),  # full length is apical zone
            'lb': '0',
            'ln': '1',
            'lmax': str(blade_lmax),
            'r': str(blade_r),
            'rlt': '1e+09',
            'Width_blade': str(stats['Width_blade']),
            'Width_petiole': str(stats.get('Width_petiole', stats['Width_blade'] * 0.3)),
            'areaMax': str(stats['areaMax'] * (1.0 - SL_RATIO)),  # blade-only area
            'ldelay': str(sheath_duration),  # dormant until sheath finishes
            **thermal_params,
        }

        for name, value in blade_params.items():
            param = ET.SubElement(blade_elem, 'parameter')
            param.set('name', name)
            param.set('value', value)

        # theta with dev (same as monolithic mode)
        param = ET.SubElement(blade_elem, 'parameter')
        param.set('name', 'theta')
        param.set('value', str(stats['theta']))
        param.set('dev', str(stats['theta'] * 0.15))

        # tropismS with dev
        param = ET.SubElement(blade_elem, 'parameter')
        param.set('name', 'tropismS')
        param.set('value', str(stats['tropismS']))
        param.set('dev', str(stats['tropismS'] * 0.3))

        # tropismAge
        param = ET.SubElement(blade_elem, 'parameter')
        param.set('name', 'tropismAge')
        param.set('value', str(stats['tropismAge']))

        # Curvature spline (blade only)
        curv_spline = stats.get('curvature_spline')
        if curv_spline and 'phi' in curv_spline and 'kappa' in curv_spline:
            for phi_val, kappa_val in zip(curv_spline['phi'], curv_spline['kappa']):
                cp = ET.SubElement(blade_elem, 'parameter')
                cp.set('name', 'leafCurvature')
                cp.set('phi', f"{phi_val:.4f}")
                cp.set('kappa', f"{kappa_val:.6f}")

        # leafGeometry (blade only — sheath has no blade area).
        # MaizeField3D peak-aligned profile already encodes real shape;
        # only fall back to synthetic base taper when using the default.
        geometry = stats.get('leaf_geometry') or fallback_geometry
        if not stats.get('leaf_geometry'):
            geometry = apply_maize_base_taper(geometry)
        for phi, x in geometry:
            geom_param = ET.SubElement(blade_elem, 'parameter')
            geom_param.set('name', 'leafGeometry')
            geom_param.set('phi', f"{phi:.1f}")
            geom_param.set('x', f"{x:.2f}")

    # Stem successor: route to sheath subtypes (even numbers)
    stem_elem = root.find(".//stem[@subType='1']")
    if stem_elem is not None:
        for old_succ in stem_elem.findall(".//parameter[@name='successor']"):
            stem_elem.remove(old_succ)

        succ = ET.SubElement(stem_elem, 'parameter')
        succ.set('name', 'successor')
        succ.set('ruleId', '0')
        succ.set('numLat', '1')
        succ.set('Where', '')
        succ.set('subType', '0')  # first sheath
        succ.set('organType', '4')
        succ.set('percentage', '1')

    # Set decompose_phytomer flag on seed
    seed_elem = root.find('.//seed')
    if seed_elem is not None:
        update_xml_parameter(seed_elem, 'decompose_phytomer', 1)

    print(f"\n  Generated {n_leaves} phytomer pairs (sheath+blade)")
    print(f"  Sheath subTypes (even): {[2*s['position'] for s in per_position_stats]}")
    print(f"  Blade subTypes (odd):   {[2*s['position']+1 for s in per_position_stats]}")
    print(f"  SL_ratio = {SL_RATIO} (flat maize placeholder)")

    return n_leaves


def load_reverse_engineer_traits(traits_path, gaps_path=None):
    """Load calibration data from reverse-engineering extracted traits.

    Uses extracted_traits.json (from reverse_engineer_maize.py) as the primary
    data source instead of MaizeField3D. Extracts per-position stats from the
    most mature stage, growth rates from trajectories across stages.

    Args:
        traits_path: Path to extracted_traits.json
        gaps_path: Optional path to gaps_cplantbox.json (for curvature splines)

    Returns:
        (per_position, stem_stats) in the same format as load_maizefield3d_stats()
    """
    import json
    from pathlib import Path as _Path

    data = json.loads(_Path(traits_path).read_text())

    # Find all stages sorted
    stage_keys = sorted(data.keys(), key=lambda k: int(k.split("_")[1]))

    # Find most mature stage (last one)
    mature = data[stage_keys[-1]]
    mature_leaves = mature['leaves']

    # Collect per-leaf trajectories across stages for growth rate fitting
    positions = sorted(set(l['position'] for l in mature_leaves))
    days = [data[sk]['day_estimate'] for sk in stage_keys]

    per_position = []
    for pos in positions:
        # Get this leaf at mature stage
        leaf_mature = next((l for l in mature_leaves if l['position'] == pos), None)
        if not leaf_mature or leaf_mature['length'] < 2:
            continue

        # Collect length trajectory across stages
        lengths = []
        for sk in stage_keys:
            stage_leaves = data[sk]['leaves']
            leaf = next((l for l in stage_leaves if l['position'] == pos), None)
            lengths.append(leaf['length'] if leaf else 0)

        # Fit growth rate from trajectory
        lmax = max(lengths)
        r = 4.0  # default
        lengths_arr = np.array(lengths)
        days_arr = np.array(days)
        mask = lengths_arr > 0.5
        if mask.sum() >= 3:
            from scipy.optimize import curve_fit
            def gomp(t, r_fit):
                e_ = np.exp(1.0)
                c = r_fit * e_ / max(lmax, 1)
                t_m = np.log(max(lmax / max(r_fit, 0.01), 1)) / max(c, 1e-6)
                return lmax * np.exp(-np.exp(-c * (t - t_m)))
            try:
                popt, _ = curve_fit(gomp, days_arr[mask], lengths_arr[mask],
                                    p0=[2.0], bounds=([0.1], [15.0]), maxfev=3000)
                r = float(popt[0])
            except (RuntimeError, ValueError):
                r = 4.0

        # Width profile → leafGeometry (along-axis parametrisation)
        wp = leaf_mature['width_profile_normalized']
        if len(wp) == 10:
            leaf_geometry = [(i / 9.0, wp[i] * 0.5) for i in range(10)]
        else:
            leaf_geometry = None

        # Max width (half-width for Width_blade)
        max_w = leaf_mature['max_width']

        stats = {
            'position': pos - 1,  # 0-indexed for calibration
            'lmax': lmax,
            'r': r,
            'Width_blade': max_w / 2.0,
            'Width_petiole': max_w * 0.15,
            'areaMax': leaf_mature['area'],
            'theta': leaf_mature['insertion_angle'],
            'tropismS': float(np.mean(leaf_mature['curvature_profile']))
                        if leaf_mature['curvature_profile'] else 0.03,
            'tropismAge': 5.0,
            'leaf_geometry': leaf_geometry,
        }
        per_position.append(stats)

    # Stem from mature stage
    stem_data = mature.get('stem', {})
    stem_stats = {
        'lmax': stem_data.get('height', 180.0),
        'ln': float(np.mean(stem_data['internode_lengths']))
              if stem_data.get('internode_lengths') else 14.5,
        'lb': 30.0,
        'n_leaves': len(per_position),
    }

    # Merge curvature splines from gaps
    if gaps_path:
        curv_by_pos = load_reverse_engineer_gaps(gaps_path)
        for p in per_position:
            # gaps uses 1-indexed positions, per_position uses 0-indexed
            if (p['position'] + 1) in curv_by_pos:
                p['curvature_spline'] = curv_by_pos[p['position'] + 1]

    n_curv = sum(1 for p in per_position if 'curvature_spline' in p)
    print(f"  Loaded reverse-engineering traits: {len(stage_keys)} stages, "
          f"{len(per_position)} leaf positions")
    print(f"  Stem: height={stem_stats['lmax']:.0f}cm, ln={stem_stats['ln']:.1f}cm")
    if n_curv:
        print(f"  Curvature splines: {n_curv} leaves")
    for p in per_position:
        print(f"    Pos {p['position']:>2}: lmax={p['lmax']:.1f}, "
              f"Wb={p['Width_blade']:.1f}, r={p['r']:.1f}, "
              f"theta={math.degrees(p['theta']):.0f}°")

    return per_position, stem_stats


def load_reverse_engineer_gaps(gaps_path):
    """Load curvature splines from reverse-engineering gap analysis.

    Args:
        gaps_path: Path to gaps_cplantbox.json from reverse_engineer_maize.py

    Returns:
        dict mapping leaf position -> curvature spline dict {phi: [...], kappa: [...]}
    """
    import json
    from pathlib import Path as _Path
    gaps = json.loads(_Path(gaps_path).read_text())
    curvature_by_pos = {}
    for g in gaps:
        if g['parameter'] == 'leafCurvaturePhi/Kappa':
            spline = g.get('extracted_values', {}).get('spline')
            if spline and 'phi' in spline and 'kappa' in spline:
                for pos in g['leaf_positions']:
                    curvature_by_pos[pos] = spline
    return curvature_by_pos


def calibrate_maize_xml(template_path, output_path, trajectory_path=None,
                        maizefield3d_path=None, max_positions=None,
                        reverse_engineer_path=None, use_gompertz=False,
                        reverse_engineer_traits_path=None,
                        decompose_phytomer=False,
                        thermal_emergence=False, phyllochron_tt=57.9,
                        surface_cps_library=None,
                        surface_cps_draw_seed=None,
                        surface_cps_draw_coherent_seed=None,
                        pheno4d_phyllochron_path=None,
                        pheno4d_phyllochron_min_plants=2,
                        pheno4d_stem_r=False):
    """
    Calibrate maize.xml from MaizeField3D stats, optionally enriched with
    Pheno4D growth dynamics.

    Args:
        template_path: Path to original maize.xml
        output_path: Path to write calibrated maize.xml
        trajectory_path: Path to trajectory.json (optional, for dynamics)
        maizefield3d_path: Path to maizefield3d_stats.json (required)
        max_positions: Max leaf positions (default: auto from MaizeField3D)
    """
    # Load calibration data — prefer OBJ reverse-engineering over MaizeField3D
    if reverse_engineer_traits_path:
        per_position, mf3d_stem_stats = load_reverse_engineer_traits(
            reverse_engineer_traits_path, gaps_path=reverse_engineer_path)
        source_name = "OBJ reverse-engineering"
    elif maizefield3d_path:
        per_position, mf3d_stem_stats = load_maizefield3d_stats(maizefield3d_path)
        source_name = "MaizeField3D"
        # Optionally merge Pheno4D dynamics
        if trajectory_path:
            trajectory = load_trajectory(trajectory_path)
            pheno4d_dynamics = extract_pheno4d_dynamics(trajectory)
            merge_pheno4d_dynamics(per_position, pheno4d_dynamics)
            source_name += " + Pheno4D"
        # Optionally merge reverse-engineering curvature splines only
        if reverse_engineer_path:
            curv_by_pos = load_reverse_engineer_gaps(reverse_engineer_path)
            for p in per_position:
                if p['position'] in curv_by_pos:
                    p['curvature_spline'] = curv_by_pos[p['position']]
            n_curv = sum(1 for p in per_position if 'curvature_spline' in p)
            source_name += f" + Reverse-Eng ({n_curv} curvature splines)"
    else:
        raise ValueError("Either --reverse-engineer-traits or --maizefield3d is required")

    # Optionally merge Pheno4D emergence-time overrides (per-position
    # ldelay / tt_emergence). When absent or unspecified, the existing
    # heuristic in generate_per_leaf_xml runs unchanged.
    if pheno4d_phyllochron_path:
        overrides, tt_meta = load_pheno4d_emergence_overrides(
            pheno4d_phyllochron_path,
            min_n_plants=pheno4d_phyllochron_min_plants,
        )
        merge_pheno4d_emergence_overrides(per_position, overrides,
                                          tt_meta=tt_meta)
        if overrides:
            source_name += f" + Pheno4D-emergence({len(overrides)})"
        if tt_meta:
            print(f"  Pheno4D TT assumption: T_mean={tt_meta.get('T_mean_C')} C, "
                  f"T_base={tt_meta.get('T_base_C')} C, "
                  f"GDD={tt_meta.get('GDD_per_day_degCd')} degCd/day")

        # Stem scalar overrides from the same JSON.
        # ln override: REMOVED 2026-04-19. CPlantBox stem has a single
        # `ln` parameter that represents the MATURE internode target; the
        # plant elongates toward it over time. Pheno4D's 5.14 cm is an
        # early-V-stage snapshot of an internode mid-elongation, not a
        # mature target — substituting it permanently stunts the stem
        # (54 cm at day 55 vs real maize 150-200 cm). Stage-dependent
        # compression should be calibrated via `delayNGStart` /
        # `delayNGEnd` against Pheno4D's top-collar trajectory, not via
        # `ln`. Extractor still emits `per_internode_length` for that
        # future path; currently unused.
        # r override: opt-in via --pheno4d-stem-r. Same regime-mismatch
        # caveat (early-V-stage slope doesn't extrapolate to maturity)
        # but useful for early-stage-only runs.
        stem_ov = load_pheno4d_stem_overrides(pheno4d_phyllochron_path)
        stem_ov.pop('ln_override_cm', None)
        stem_ov.pop('ln_override_covered_upper_labels', None)
        stem_ov.pop('ln_override_per_upper_label_medians', None)
        if not pheno4d_stem_r:
            stem_ov.pop('r_stem_override_cm_per_day', None)
            stem_ov.pop('r_stem_override_std', None)
            stem_ov.pop('r_stem_override_n_plants', None)
            stem_ov.pop('r_stem_override_max_observed_stack_cm', None)
        merge_pheno4d_stem_overrides(mf3d_stem_stats, stem_ov)
        if stem_ov and 'r_stem_override_cm_per_day' in stem_ov:
            source_name += " + Pheno4D-stem(r)"

    if max_positions is not None:
        per_position = per_position[:max_positions]

    # Per-plant lmax/width override for draw / draw_coherent modes.
    # The canonical CP library stores each plant's real leaf shapes, and
    # we scale those by `current_length` at loft time. If we keep the
    # MF3D median lmax, we're stretching a short plant's shape up to the
    # median length — e.g. plant 237's 58 cm pos-2 blade inflated to
    # 74 cm. The fix: when a draw/coherent seed is set, use the chosen
    # plant's actual per-position lmax and max_width so size matches shape.
    if (surface_cps_draw_seed is not None
            or surface_cps_draw_coherent_seed is not None):
        from dart.coupling.geometry.canonical_library import (
            build_from_maizefield3d as _bld,
            _default_canonical_json as _def_cj,
        )
        if surface_cps_draw_coherent_seed is not None:
            _seed = int(surface_cps_draw_coherent_seed)
            _lib_for_sizing = _bld(
                _def_cj(), reducer="draw_coherent", draw_seed=_seed,
            )
            _mode = f"coherent seed={_seed}"
        else:
            assert surface_cps_draw_seed is not None
            _seed = int(surface_cps_draw_seed)
            _lib_for_sizing = _bld(
                _def_cj(), reducer="draw", draw_seed=_seed,
            )
            _mode = f"draw seed={_seed}"
        _metrics = _lib_for_sizing.get("chosen_metrics_cm")
        _lib_positions = list(_lib_for_sizing["positions"])
        if _metrics is not None:
            print(f"\n  Per-plant lmax/width override ({_mode}):")
            print(f"    {'pos':>3} {'lmax_old':>8} {'lmax_new':>8} "
                  f"{'Wb_old':>6} {'Wb_new':>6}")
            for p in per_position:
                if p['position'] not in _lib_positions:
                    continue
                idx = _lib_positions.index(p['position'])
                lm_new, max_w_new = float(_metrics[idx][0]), float(_metrics[idx][1])
                if lm_new <= 0.0 or max_w_new <= 0.0:
                    continue
                lm_old = p['lmax']
                wb_old = p['Width_blade']
                wb_new = max_w_new / 2.0
                # areaMax scales as lmax × max_width (shape factor preserved)
                if lm_old > 0 and wb_old > 0:
                    p['areaMax'] = p.get('areaMax', 0.0) * (
                        (lm_new * max_w_new) / (lm_old * wb_old * 2.0)
                    )
                p['lmax'] = lm_new
                p['Width_blade'] = wb_new
                p['Width_petiole'] = wb_new * 0.3
                if p.get('median_length_cm') is not None:
                    p['median_length_cm'] = lm_new
                print(f"    {p['position']:>3} {lm_old:>8.1f} {lm_new:>8.1f} "
                      f"{wb_old:>6.2f} {wb_new:>6.2f}")

    if use_gompertz:
        source_name += " + Gompertz(gf=4)"

    print(f"\n=== Calibration Statistics — Source: {source_name} ===")

    print(f"\nStem (from MaizeField3D):")
    for k, v in mf3d_stem_stats.items():
        print(f"  {k}: {v}")

    print(f"\nPer-Position Leaf Stats (after merge):")
    print(f"  {'Pos':>3} {'SubT':>4} {'lmax':>7} {'Width':>7} {'theta':>7} "
          f"{'tropS':>7} {'tropAge':>7} {'r':>6} {'area':>7}")
    for stats in per_position:
        print(f"  {stats['position']:>3} {stats['position']+2:>4} "
              f"{stats['lmax']:>7.1f} {stats['Width_blade']:>7.2f} "
              f"{stats['theta']:>7.2f} {stats['tropismS']:>7.3f} "
              f"{stats['tropismAge']:>7.1f} {stats['r']:>6.1f} "
              f"{stats['areaMax']:>7.1f}")

    # Parse XML template
    tree = ET.parse(template_path)
    root = tree.getroot()

    # Set delayDefinitionShoot=2 (dd_time_self)
    seed_elem = root.find('.//seed')
    if seed_elem is not None:
        update_xml_parameter(seed_elem, 'delayDefinitionShoot', 2)
        print(f"\n  Seed delayDefinitionShoot = 2 (dd_time_self)")

    # Update stem from MaizeField3D
    stem_elem = root.find(".//stem[@subType='1']")
    if mf3d_stem_stats and stem_elem is not None:
        print(f"\nUpdating stem subType=1 (from MaizeField3D)...")
        s_lmax = mf3d_stem_stats.get('lmax', 180.0)
        # ln: MF3D mature-internode estimate (~15 cm). Pheno4D ln override
        # was briefly wired 2026-04-19 and backed out same day — `ln` is
        # the mature target, not a snapshot, so early-V-stage data cannot
        # parameterise it (see stem-overrides block above).
        s_ln = mf3d_stem_stats.get('ln', 15.0)
        # r: default 2.5 unless --pheno4d-stem-r is set. Pheno4D r_override
        # is top-collar-z rate, a lower bound on stem-tip r (apex sits
        # above top collar inside the whorl). Pulls young-stage stem
        # height down toward Pheno4D observations but undershoots mature.
        s_r = mf3d_stem_stats.get('r_override', 2.5)
        # Override lb: MaizeField3D lb=30 is mature-plant stretched internodes.
        # lb=12 lifts pos-0 collar ~9 cm above ground (seed at z=-3), giving
        # a mature pos-0 blade (~53 cm, drooping tip.z/arc ≈ 0.33) enough
        # clearance to stay above z=0. lb=4 placed the collar at z=+1 and
        # pushed drooping tips ~3-18 cm underground across all library modes.
        s_lb = 12.0
        update_xml_parameter(stem_elem, 'lmax', s_lmax, dev=s_lmax * 0.1)
        update_xml_parameter(stem_elem, 'r', s_r, dev=max(s_r * 0.1, 0.1))
        update_xml_parameter(stem_elem, 'ln', s_ln, dev=s_ln * 0.05)
        update_xml_parameter(stem_elem, 'lb', s_lb)
        update_xml_parameter(stem_elem, 'dx', 0.1)
        r_src = "Pheno4D" if 'r_override' in mf3d_stem_stats else "default"
        print(f"  lmax={s_lmax}, ln={s_ln:.2f}[MF3D], "
              f"lb={s_lb}, r={s_r:.3f}[{r_src}]")

    # Phyllotaxy: distichous (180° alternating, two-ranked) — correct for maize.
    # CPlantBox reads RotBeta / BetaDev in UNITS OF PI (internally multiplies
    # by pi), so RotBeta=1 means one half-turn per phytomer and BetaDev=0.02
    # gives ~3.6 deg of jitter (tight distichous). Stored-as-radians would be
    # pi^2 per step — the historical radians-as-pi-units bug that scrambled
    # phyllotaxy.
    if stem_elem is not None:
        update_xml_parameter(stem_elem, 'RotBeta', 1.0)
        update_xml_parameter(stem_elem, 'BetaDev', 0.02)

    # Stem la to control leaf count
    n_leaves = len(per_position)
    if stem_elem is not None:
        lmax_param = stem_elem.find(".//parameter[@name='lmax']")
        lb_param = stem_elem.find(".//parameter[@name='lb']")
        ln_param = stem_elem.find(".//parameter[@name='ln']")
        if lmax_param is not None and lb_param is not None and ln_param is not None:
            stem_lmax = float(lmax_param.get('value') or 0.0)
            stem_lb = float(lb_param.get('value') or 0.0)
            stem_ln = float(ln_param.get('value') or 0.0)
            la_val = stem_lmax - stem_lb - (n_leaves - 1) * stem_ln - 0.1
            la_val = max(0.1, la_val)
            update_xml_parameter(stem_elem, 'la', la_val)
            print(f"  Stem la = {la_val:.1f} (controls {n_leaves} leaves)")

    # Theta with empirical CPB-bias correction.
    # Validator measures base tangent as angle over the first 10% of the arc.
    # At day 55, this 10% was laid down early and has been exposed to
    # gravitropism for ~30 days, so the measured tangent differs from the
    # specified XML theta by up to ±12° per position (observed from a
    # diagnostic run). To make the CPB-observed base angle match the MF3D
    # measured insertion angle, pre-subtract the per-position empirical
    # deviation: θ_XML = θ_target − Δ_empirical. Clamp widened to [22°, 55°]
    # so over-corrections (e.g. pos 8 needs +9° bump) aren't capped away.
    # Deviations measured with seed=42 and the current growth parameters;
    # order is position 0..10.
    empirical_delta_deg = [4.4, 2.6, -1.2, 4.0, 9.7, 0.1, 12.4, 1.5, -9.2, 1.6, 4.2]
    n_pos = len(per_position)
    print(f"\n  Theta with empirical CPB-bias correction:")
    for p in per_position:
        pos = p['position']
        old_theta = p['theta']
        target_deg = math.degrees(old_theta)
        delta = empirical_delta_deg[pos] if pos < len(empirical_delta_deg) else 0.0
        corrected = target_deg - delta
        corrected = max(22.0, min(55.0, corrected))
        p['theta'] = math.radians(corrected)
        print(f"    Pos {pos}: target {target_deg:.1f}° - Δ{delta:+.1f}° -> "
              f"XML θ={corrected:.1f}°")

    # Position-dependent tropismS: low values so that only the tip bends,
    # not a uniform U-shaped arc. Real maize leaves are straight blades
    # with a gentle nod at the very tip — NOT symmetric parabolas.
    for p in per_position:
        pos = p['position']
        t = pos / max(n_pos - 1, 1)
        # Lower leaves (pos 0): 0.03 — gentle tip droop
        # Upper leaves (pos n-1): 0.01 — barely perceptible
        p['tropismS'] = 0.03 - t * 0.02

    # Bottom-leaf lmax taper + hard cap: pos 0-1 were historically tapered
    # because MF3D bucket 0 (53 cm) was being applied at the soil-level
    # insertion, pushing drooping tips underground. With the 16-position
    # library's prepended stubs (pos 0-1 synthesised as scaled copies of
    # MF3D pos 0), pos 0-1 arrive pre-tapered (~24 / 35 cm mature) and
    # must NOT be tapered again — double-tapering here reduced pos 0 to
    # ~11 cm, which senescence then shrank to ~5 cm (invisible at day 130).
    # Keep pos 2-3 which still carry the raw MF3D shape at sub-optimal
    # insertion heights after the shift.
    LMAX_TAPER = {2: 0.85, 3: 0.90}
    LMAX_CAP_CM = {2: 50.0, 3: 50.0}
    for p in per_position:
        pos = p['position']
        factor = LMAX_TAPER.get(pos)
        if factor is None:
            continue
        cap = LMAX_CAP_CM.get(pos, float('inf'))
        old_lmax = p['lmax']
        new_lmax = min(old_lmax * factor, cap)
        if new_lmax >= old_lmax:
            continue
        scale = new_lmax / old_lmax
        p['lmax'] = new_lmax
        p['areaMax'] = p.get('areaMax', 0.0) * scale
        if p.get('median_length_cm') is not None:
            p['median_length_cm'] = p['median_length_cm'] * scale
        print(f"  Bottom-leaf taper pos {pos}: lmax "
              f"{old_lmax:.1f} -> {new_lmax:.1f} cm (x{scale:.2f})")

    # Ground penetration clamping with stem params
    # Use the actual lb value from the stem (overridden above)
    clamp_lb = 12.0
    s_ln = mf3d_stem_stats.get('ln', 15.0) if mf3d_stem_stats else 14.5
    # Arc-through-space geometry: real maize blades emerge upward at theta,
    # arc over, then the tip droops. For pos 0 (insertion_h=4, theta=40°,
    # lmax=53, measured tip_droop=12.9 cm), the straight phase (85% of lmax
    # at cos(40°)=0.77 upward) peaks at z=4+0.85·53·0.77=38.7 cm; tip then
    # droops ~13 cm to z≈26 cm — well above ground. The old multiplier 5.0
    # modelled stick-then-drop-straight and over-constrained bottom leaves
    # (pos 0: 53 cm target clamped to 28 cm). Bump to 10.0 so clamp only
    # catches pathological cases, not the real arc geometry.
    for p in per_position:
        insertion_h = clamp_lb + p['position'] * s_ln
        sin_theta = max(math.sin(p['theta']), 0.3)
        max_lmax = insertion_h * 10.0 / sin_theta
        if p['lmax'] > max_lmax and insertion_h > 0:
            old_lmax = p['lmax']
            scale = max_lmax / old_lmax
            p['lmax'] = max_lmax
            p['areaMax'] *= scale
            p['tropismS'] *= scale
            # Do NOT reduce theta — shorter leaves should still emerge outward
            print(f"  Position {p['position']}: clamped lmax {old_lmax:.1f} -> {max_lmax:.1f} cm "
                  f"(insertion height ~{insertion_h:.1f} cm, theta={math.degrees(p['theta']):.0f} deg)")
        if insertion_h > 0:
            max_tropS = 0.04 * insertion_h / max(p['lmax'], 1.0)
            if p['tropismS'] > max_tropS:
                p['tropismS'] = max_tropS

    # Per-position r (elongation rate) tuning.
    # MF3D `r=4.0` is a scalar derived assuming all leaves have the full 55-day
    # growth window. Upper leaves emerge later (ldelay = pos × phyllochron) so
    # t_available = 55 - pos·3 days. With uniform r=4, the negative-exponential
    # growth L(t) = lmax·(1 − exp(−r·t/lmax)) leaves upper positions at
    # ~82-86% of lmax vs. MF3D median target ~91%. Back-solve r per position
    # to hit the measured median_length_cm at day 55:
    #   r = −(lmax/t_avail) · ln(1 − target/lmax)
    # Sanity-bound r to [2.0, 6.0] so single-position outliers don't blow up
    # the tropismAge clamp downstream.
    phyllochron_days = 3.0
    t_total = 55.0
    # Systematic correction: analytic L(t) = lmax·(1−exp(−r·t/lmax)) overestimates
    # CPB's simulated length by ~4% (tropism arc, lb consumption, discretization).
    # Empirical calibration against validator: scaling r by 1.08 aligns all
    # positions within ±1% of target.
    r_correction = 1.08
    print(f"\n  Per-position r tuning (target = median_length at day {t_total:.0f}):")
    for p in per_position:
        pos = p['position']
        lmax = p['lmax']
        target = p.get('median_length_cm')
        t_avail = max(t_total - pos * phyllochron_days, 10.0)
        if target and 0 < target < lmax:
            ratio = 1.0 - target / lmax
            r_new = -(lmax / t_avail) * math.log(max(ratio, 1e-6)) * r_correction
            r_new = float(np.clip(r_new, 2.0, 6.0))
            old_r = p['r']
            p['r'] = round(r_new, 3)
            print(f"    Pos {pos}: r {old_r:.2f} -> {p['r']:.2f} "
                  f"(target {target:.1f} cm, t_avail {t_avail:.0f} d)")

    # Position-dependent tropismAge: controls WHERE on the leaf droop begins.
    # The leaf grows perfectly straight (at theta) for tropismAge days.
    # After that, new growth gets gravitropism → only the TIP droops.
    #
    # Real maize leaves are straight blades — 85-95% of length is rigid,
    # only the outermost 5-15% shows gentle downward curvature.
    # Combined with low tropismS, this avoids the uniform U-shaped arc.
    for p in per_position:
        pos = p['position']
        t = pos / max(n_pos - 1, 1)
        r = max(p['r'], 0.5)
        lmax = p['lmax']
        # Lower leaves: 85% straight (tip droop on outer 15%)
        # Upper leaves: 95% straight (barely any tip droop)
        straight_frac = 0.85 + t * 0.10
        t_straight = straight_frac * lmax / r
        # Clamp: min 12 days, max 70 days
        p['tropismAge'] = round(np.clip(t_straight, 12.0, 70.0), 1)

    # Optional: load local-frame CP library for Phase-B surface_cp emission.
    surface_cps_by_pos = None
    if (surface_cps_library is not None
            or surface_cps_draw_seed is not None
            or surface_cps_draw_coherent_seed is not None):
        from pathlib import Path as _PathLib
        from dart.coupling.geometry.canonical_library import (
            load_library, build_from_maizefield3d, _default_canonical_json,
        )
        if surface_cps_draw_coherent_seed is not None:
            canonical_json = _default_canonical_json()
            lib = build_from_maizefield3d(
                canonical_json, reducer="draw_coherent",
                draw_seed=int(surface_cps_draw_coherent_seed),
            )
            source_desc = (f"coherent-draw from {canonical_json.name} "
                           f"(seed={surface_cps_draw_coherent_seed}, "
                           f"single plant across all positions)")
        elif surface_cps_draw_seed is not None:
            canonical_json = _default_canonical_json()
            lib = build_from_maizefield3d(
                canonical_json, reducer="draw",
                draw_seed=int(surface_cps_draw_seed),
            )
            source_desc = (f"random-draw from {canonical_json.name} "
                           f"(seed={surface_cps_draw_seed}, "
                           f"independent per position)")
        else:
            if surface_cps_library == '__default__' or surface_cps_library is None:
                lib_path = (_PathLib(__file__).resolve().parents[1]
                            / 'data' / 'canonical_leaf_library.npz')
            else:
                lib_path = _PathLib(surface_cps_library)
            lib = load_library(lib_path)
            source_desc = str(lib_path)
        surface_cps_by_pos = {
            int(p): lib['cps_local'][idx]
            for idx, p in enumerate(lib['positions'])
        }
        print(f"\n  Surface CP library loaded from {source_desc}")
        print(f"    {len(surface_cps_by_pos)} positions, grid {lib['n_u']}x{lib['n_v']}")

    # Generate per-leaf subtypes
    gf_val = 4 if use_gompertz else 3
    if decompose_phytomer:
        generate_phytomer_leaf_xml(root, per_position, DEFAULT_LEAF_GEOMETRY,
                                   gf=gf_val)
    else:
        generate_per_leaf_xml(root, per_position, DEFAULT_LEAF_GEOMETRY,
                              gf=gf_val,
                              thermal_emergence=thermal_emergence,
                              phyllochron_tt=phyllochron_tt,
                              surface_cps_by_pos=surface_cps_by_pos)

    # Write calibrated XML
    indent_xml(root)
    tree.write(output_path, encoding='utf-8', xml_declaration=True)

    print(f"\nCalibrated maize.xml written to: {output_path}")
    print(f"\nSummary:")
    print(f"  - {n_leaves} leaf subtypes (subType 2..{n_leaves + 1})")
    print(f"  - Bottom leaves: lmax={per_position[0]['lmax']:.0f} cm, r={per_position[0]['r']:.1f}")
    if n_leaves > 1:
        mid = n_leaves // 2
        print(f"  - Middle leaves: lmax={per_position[mid]['lmax']:.0f} cm, r={per_position[mid]['r']:.1f}")
        print(f"  - Top leaves: lmax={per_position[-1]['lmax']:.0f} cm, r={per_position[-1]['r']:.1f}")
    if trajectory_path:
        print(f"  - Growth rates: Pheno4D-measured + lmax-scaled extrapolation")
    else:
        print(f"  - Growth rates: uniform 4.0 cm/day (no Pheno4D trajectory)")


def indent_xml(elem, level=0):
    """Add pretty-printing indentation to XML."""
    i = "\n" + level * "    "
    children = list(elem)
    if children:
        if not elem.text or not elem.text.strip():
            elem.text = i + "    "
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for child in children:
            indent_xml(child, level + 1)
        last = children[-1]
        if not last.tail or not last.tail.strip():
            last.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i


def main():
    parser = argparse.ArgumentParser(
        description='Calibrate maize.xml from MaizeField3D + optional Pheno4D dynamics')
    parser.add_argument('--maizefield3d', default=None,
                        help='Path to maizefield3d_stats.json')
    parser.add_argument('--reverse-engineer-traits', default=None,
                        help='Path to extracted_traits.json from reverse_engineer_maize.py (preferred over --maizefield3d)')
    parser.add_argument('--reverse-engineer-gaps', default=None,
                        help='Path to gaps_cplantbox.json (for curvature splines)')
    parser.add_argument('--trajectory', default=None,
                        help='Path to trajectory.json (optional, for growth dynamics with MaizeField3D)')
    parser.add_argument('--template', required=True, help='Path to template maize.xml')
    parser.add_argument('--output', required=True, help='Path to output calibrated maize.xml')
    parser.add_argument('--max-positions', type=int, default=None,
                        help='Max leaf positions (default: auto)')
    parser.add_argument('--gompertz', action='store_true',
                        help='Use Gompertz growth function (gf=4) instead of CWLimited (gf=3)')
    parser.add_argument('--decompose-phytomer', action='store_true',
                        help='Generate sheath+blade pairs (phytomer decomposition mode)')
    parser.add_argument('--thermal-emergence', action='store_true',
                        help='Gate leaf emergence on plant thermal time '
                             '(use_thermal_emergence=1, tt_emergence=pos*phyllochron_tt) '
                             'instead of calendar-day ldelay. Requires plant.setAirTemperature() '
                             'fed per simulate() step.')
    parser.add_argument('--phyllochron-tt', type=float, default=57.9,
                        help='Thermal-time phyllochron [degCd] for --thermal-emergence '
                             '(default 57.9, Dos Santos 2022 maize)')
    parser.add_argument('--surface-cps', default=None, nargs='?', const='__default__',
                        help='Embed per-position 2D NURBS CP grids from a leaf-local-frame '
                             'canonical library. Pass a path to .npz or omit for the default '
                             '(coupling/data/canonical_leaf_library.npz).')
    parser.add_argument('--surface-cps-draw-coherent-seed', type=int, default=None,
                        help='Pick ONE random plant and use its CPs for every '
                             'position. Produces a coherent real plant silhouette '
                             '(sizes match across positions). Requires plants '
                             'that cover every requested position.')
    parser.add_argument('--surface-cps-draw-seed', type=int, default=None,
                        help='Instead of loading the aggregated .npz, build the library '
                             'in-memory by random-drawing one plant per position from '
                             'canonical_leaf_library.json. Takes precedence over '
                             '--surface-cps. Preserves real per-plant correlations that '
                             'median aggregation smooths out.')
    parser.add_argument('--pheno4d-phyllochron', default=None,
                        help='Path to pheno4d_emergence_timeseries.json (from '
                             'Resources/Pheno4D/extract_emergence_timeseries.py). When '
                             'provided, per-position ldelay / tt_emergence override the '
                             'default 3.0-d / 57.9-degCd heuristic for positions passing '
                             '--pheno4d-min-plants. Absent positions fall back to heuristic.')
    parser.add_argument('--pheno4d-min-plants', type=int, default=2,
                        help='Minimum Pheno4D plants per position required to trust the '
                             'override (default 2).')
    parser.add_argument('--pheno4d-stem-r', action='store_true',
                        help='Also override stem r from Pheno4D top-collar elongation '
                             'fit (~0.94 cm/day). OFF by default because the fit is an '
                             'early-V-stage rate; applying it to late stages undershoots '
                             'mature stem height by ~50%%. Use for early-stage-only runs '
                             '(days 10-30). --pheno4d-phyllochron always overrides ln '
                             '(lower-internode median ~5 cm) which is regime-robust.')

    args = parser.parse_args()

    calibrate_maize_xml(
        template_path=args.template,
        output_path=args.output,
        trajectory_path=args.trajectory,
        maizefield3d_path=args.maizefield3d,
        max_positions=args.max_positions,
        reverse_engineer_path=args.reverse_engineer_gaps,
        reverse_engineer_traits_path=args.reverse_engineer_traits,
        use_gompertz=args.gompertz,
        decompose_phytomer=args.decompose_phytomer,
        thermal_emergence=args.thermal_emergence,
        phyllochron_tt=args.phyllochron_tt,
        surface_cps_library=args.surface_cps,
        surface_cps_draw_seed=args.surface_cps_draw_seed,
        surface_cps_draw_coherent_seed=args.surface_cps_draw_coherent_seed,
        pheno4d_phyllochron_path=args.pheno4d_phyllochron,
        pheno4d_phyllochron_min_plants=args.pheno4d_min_plants,
        pheno4d_stem_r=args.pheno4d_stem_r,
    )


if __name__ == '__main__':
    main()
