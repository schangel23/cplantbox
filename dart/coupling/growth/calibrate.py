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


def generate_per_leaf_xml(root, per_position_stats, fallback_geometry, gf=3):
    """Generate one leaf subType per position and update stem successor rules.

    Args:
        root: XML root element
        per_position_stats: list of dicts with per-position parameters
        fallback_geometry: default leaf geometry profile [(phi, x), ...]
        gf: growth function type (1=exp, 2=linear, 3=CWLimited, 4=Gompertz)
    """
    # Remove ALL existing leaf elements
    for leaf_elem in root.findall('leaf'):
        root.remove(leaf_elem)

    # Remove branch stem subtypes (2-5) — they interfere with lateral creation
    for stem_elem in root.findall('stem'):
        st = stem_elem.get('subType', '0')
        if st not in ('0', '1'):
            root.remove(stem_elem)

    # Shared parameters for all maize leaf subtypes
    shared_params = {
        'geometryN': '100',
        'gf': str(gf),
        'isPseudostem': '0',
        'lnf': '0',
        'parametrisationType': '1',
        'shapeType': '2',
        'tropismT': '1',  # gravitropism — leaves emerge at theta and droop gently
        'BetaDev': '0.22',
        'InitBeta': '0',
        'RotBeta': str(math.pi / 2),
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
    # Within each rank, add progressive fan-out so same-side leaves don't
    # overlap, plus per-leaf azimuthal jitter for natural look.
    distichous_angle = math.pi  # 180°
    rank_spread = math.radians(4.0)  # 4° systematic fan-out per rank
    # Per-leaf azimuthal jitter: real maize deviates ±8-15° from perfect
    # two-ranked pattern due to stem twist, growth asymmetry, mechanical
    # interactions with neighboring leaves, and wind.  Deterministic seed
    # for reproducible XML output.
    rng = np.random.RandomState(42)
    max_pos = max(s['position'] for s in per_position_stats) + 1
    leaf_jitter_deg = rng.uniform(-12.0, 12.0, size=max(max_pos, n_leaves))

    for stats in per_position_stats:
        pos = stats['position']
        sub_type = pos + 2

        leaf_elem = ET.SubElement(root, 'leaf')
        leaf_elem.set('name', f'maize_leaf_L{pos}')
        leaf_elem.set('subType', str(sub_type))

        rank_idx = pos // 2  # 0,0,1,1,2,2,3,3,...
        base_angle = pos * distichous_angle + rank_idx * rank_spread
        jitter = math.radians(leaf_jitter_deg[pos])
        init_beta = (base_angle + jitter) % (2 * math.pi)

        for name, value in shared_params.items():
            param = ET.SubElement(leaf_elem, 'parameter')
            param.set('name', name)
            if name == 'InitBeta':
                param.set('value', str(init_beta))
            else:
                param.set('value', value)

        # Per-position ldelay: maize phyllochron ~3 days
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

        # leafGeometry profile with base tapering
        geometry = stats.get('leaf_geometry') or fallback_geometry
        geometry = apply_maize_base_taper(geometry)
        for phi, x in geometry:
            geom_param = ET.SubElement(leaf_elem, 'parameter')
            geom_param.set('name', 'leafGeometry')
            geom_param.set('phi', f"{phi:.1f}")
            geom_param.set('x', f"{x:.2f}")

    # Stem successor placeholder (overridden by Python API at runtime)
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
                        reverse_engineer_traits_path=None):
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

    if max_positions is not None:
        per_position = per_position[:max_positions]

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
        s_ln = mf3d_stem_stats.get('ln', 15.0)
        # Override lb: MaizeField3D lb=30 is for mature plants with elongated
        # internodes. For growth simulation, lb=4 allows leaves to appear
        # very early on the stem, simulating the real maize rosette phase
        # where the growing point stays near the soil surface.
        s_lb = 4.0
        update_xml_parameter(stem_elem, 'lmax', s_lmax, dev=s_lmax * 0.1)
        # Stem r=2.5: slower than previous 5.0 to avoid overly tall stems
        # at early stages. Day 5: ~12cm, Day 15: ~34cm, Day 60: ~108cm.
        # Real maize stays compact until V6 (~day 30), then elongates.
        update_xml_parameter(stem_elem, 'r', 2.5, dev=0.25)
        update_xml_parameter(stem_elem, 'ln', s_ln, dev=s_ln * 0.05)
        update_xml_parameter(stem_elem, 'lb', s_lb)
        update_xml_parameter(stem_elem, 'dx', 0.1)
        print(f"  lmax={s_lmax}, ln={s_ln}, lb={s_lb}, r=2.5")

    # Phyllotaxy: distichous (180° alternating, two-ranked) — correct for maize
    if stem_elem is not None:
        distichous_angle = math.pi  # 180°
        update_xml_parameter(stem_elem, 'RotBeta', distichous_angle)
        update_xml_parameter(stem_elem, 'BetaDev', 0.22)

    # Stem la to control leaf count
    n_leaves = len(per_position)
    if stem_elem is not None:
        lmax_param = stem_elem.find(".//parameter[@name='lmax']")
        lb_param = stem_elem.find(".//parameter[@name='lb']")
        ln_param = stem_elem.find(".//parameter[@name='ln']")
        if lmax_param is not None and lb_param is not None and ln_param is not None:
            stem_lmax = float(lmax_param.get('value'))
            stem_lb = float(lb_param.get('value'))
            stem_ln = float(ln_param.get('value'))
            la_val = stem_lmax - stem_lb - (n_leaves - 1) * stem_ln - 0.1
            la_val = max(0.1, la_val)
            update_xml_parameter(stem_elem, 'la', la_val)
            print(f"  Stem la = {la_val:.1f} (controls {n_leaves} leaves)")

    # Position-dependent theta based on real maize growth stages.
    # Lower leaves (older) emerge more horizontally; upper leaves (younger)
    # are more erect. Real maize insertion angles are ~30-50° from vertical.
    # The reference (UIUC growth stages) shows leaves projecting clearly
    # outward from the stem — NOT nearly vertical (that creates a bouquet).
    n_pos = len(per_position)
    print(f"\n  Position-dependent theta (outward projection):")
    for p in per_position:
        pos = p['position']
        t = pos / max(n_pos - 1, 1)
        old_theta = p['theta']
        # Lower leaves (pos 0): 45° — project outward, will droop with age
        # Upper leaves (pos n-1): 25° — more erect, younger
        # Bell curve: middle leaves slightly wider than top
        p['theta'] = math.radians(45 - t * 20)
        print(f"    Pos {pos}: theta {math.degrees(old_theta):.0f} -> "
              f"{math.degrees(p['theta']):.0f} deg")

    # Position-dependent tropismS: low values so that only the tip bends,
    # not a uniform U-shaped arc. Real maize leaves are straight blades
    # with a gentle nod at the very tip — NOT symmetric parabolas.
    for p in per_position:
        pos = p['position']
        t = pos / max(n_pos - 1, 1)
        # Lower leaves (pos 0): 0.03 — gentle tip droop
        # Upper leaves (pos n-1): 0.01 — barely perceptible
        p['tropismS'] = 0.03 - t * 0.02

    # Ground penetration clamping with stem params
    # Use the actual lb value from the stem (overridden to 4.0 above)
    clamp_lb = 4.0
    s_ln = mf3d_stem_stats.get('ln', 15.0) if mf3d_stem_stats else 14.5
    for p in per_position:
        insertion_h = clamp_lb + p['position'] * s_ln
        # Theta-aware clamping: erect leaves (small theta) can be longer
        # because they grow upward and won't reach the ground.
        # Base multiplier 5.0 (relaxed from 3.5 because young leaves are
        # now erect with low tropismS — much less ground penetration risk)
        sin_theta = max(math.sin(p['theta']), 0.3)
        max_lmax = insertion_h * 5.0 / sin_theta
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

    # Generate per-leaf subtypes
    generate_per_leaf_xml(root, per_position, DEFAULT_LEAF_GEOMETRY,
                          gf=4 if use_gompertz else 3)

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
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "    "
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for child in elem:
            indent_xml(child, level + 1)
        if not child.tail or not child.tail.strip():
            child.tail = i
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
    )


if __name__ == '__main__':
    main()
