"""CPlantBox adapter for the G1-to-G3 lofting pipeline.

Extracts skeleton and width data from CPlantBox plant objects into
organ dicts compatible with g1_to_g3.loft_organs().
"""

import json
import numpy as np
from pathlib import Path

from ..config import MAIZEFIELD3D_DEFORMATION, MAIZEFIELD3D_STEM_PROFILE

DEFAULT_MIN_WIDTH = 0.1  # cm, fallback when Width_blade=0

# Default path for MaizeField3D blade deformation stats
_DEFAULT_DEFORMATION_JSON = MAIZEFIELD3D_DEFORMATION

# Default path for MaizeField3D stem radius profile
_DEFAULT_STEM_PROFILE_JSON = MAIZEFIELD3D_STEM_PROFILE

_deformation_cache = {}
_stem_profile_cache = {}


def load_deformation_stats(path=None):
    """Load MaizeField3D blade deformation statistics.

    Args:
        path: Path to JSON file. Defaults to MaizeField3d/maizefield3d_blade_deformation.json.

    Returns:
        List of per-position stat dicts, or None if file not found.
    """
    if path is None:
        path = _DEFAULT_DEFORMATION_JSON
    path = Path(path)
    key = str(path)
    if key in _deformation_cache:
        return _deformation_cache[key]
    if not path.exists():
        _deformation_cache[key] = None
        return None
    with open(path) as f:
        data = json.load(f)
    stats = data.get('per_position', [])
    _deformation_cache[key] = stats
    return stats


def load_stem_profile(path=None):
    """Load MaizeField3D stem radius profile.

    Args:
        path: Path to JSON file. Defaults to MaizeField3d/maizefield3d_stem_profile.json.

    Returns:
        Dict with 'height_frac' and 'radius_cm' arrays (whorl-smoothed), or None.
    """
    if path is None:
        path = _DEFAULT_STEM_PROFILE_JSON
    path = Path(path)
    key = str(path)
    if key in _stem_profile_cache:
        return _stem_profile_cache[key]
    if not path.exists():
        _stem_profile_cache[key] = None
        return None
    with open(path) as f:
        data = json.load(f)
    agg = data.get('aggregate', {})
    h = np.array(agg.get('height_frac', []))
    # Use envelope_radius (median) — the visual profile including sheaths.
    r = np.array(agg['envelope_radius_cm']['median'])
    if len(h) == 0 or len(r) == 0:
        _stem_profile_cache[key] = None
        return None

    # Smooth out the whorl peak (75-95% height).  The leaf sheaths in that
    # zone are already rendered as separate leaf geometry, so including
    # them in the stem would create double geometry.  Replace the whorl
    # region with a smooth linear interpolation between the flanking
    # values (just below the whorl and above it).
    whorl_lo, whorl_hi = 0.75, 0.95
    lo_idx = np.searchsorted(h, whorl_lo, side='right') - 1
    hi_idx = np.searchsorted(h, whorl_hi, side='left')
    if 0 <= lo_idx < len(h) and hi_idx < len(h):
        r_lo = r[lo_idx]
        r_hi = r[hi_idx]
        for i in range(lo_idx + 1, hi_idx):
            t = (h[i] - h[lo_idx]) / max(h[hi_idx] - h[lo_idx], 1e-6)
            r[i] = r_lo + t * (r_hi - r_lo)

    profile = {'height_frac': h, 'radius_cm': r}
    _stem_profile_cache[key] = profile
    return profile


def apply_stem_profile(skeleton, widths, stem_profile):
    """Replace uniform stem widths with measured radius profile.

    Maps each skeleton node's Z position to a normalized height fraction
    and interpolates the measured radius profile.

    Args:
        skeleton: (N, 3) array of stem node positions.
        widths: (N,) array of current widths (diameter, cm).
        stem_profile: Dict from load_stem_profile() with 'height_frac' and 'radius_cm'.

    Returns:
        (N,) array of new widths (diameter = 2 * interpolated radius).
    """
    z = skeleton[:, 2]
    z_min, z_max = z.min(), z.max()
    z_range = z_max - z_min
    if z_range < 1.0:  # Less than 1 cm stem — don't modify
        return widths

    # Normalize Z to [0, 1] fraction
    z_frac = (z - z_min) / z_range

    # Interpolate measured radius at each node position
    h = stem_profile['height_frac']
    r = stem_profile['radius_cm']
    radii = np.interp(z_frac, h, r)

    return 2.0 * radii  # diameter


def _leaf_wave_params(leaf_length, rng, position=None, deformation_stats=None):
    """Generate blade deformation parameters for a single leaf.

    When deformation_stats are provided (from MaizeField3D extraction), uses
    measured distributions. Otherwise falls back to hand-tuned ranges.

    Note: curl is always hand-tuned because the MaizeField3D 3-point NURBS
    cross-sections are inherently symmetric and can't capture asymmetric curl.

    Args:
        leaf_length: Total arc length of the leaf skeleton (cm)
        rng: numpy RandomState for repeatable per-leaf variation
        position: Leaf position index (0-10, bottom to top). Used to look up
            position-specific measured stats.
        deformation_stats: List of per-position stat dicts from
            load_deformation_stats(), or None for hand-tuned fallback.
    """
    # Length-dependent intensity: short leaves (~20 cm) get ~0.2x,
    # long leaves (~70 cm) get ~1.0x. Clamp to [0.15, 1.0].
    intensity = float(np.clip((leaf_length - 15.0) / 55.0, 0.15, 1.0))

    # Check if we have measured stats for this position
    stats = None
    if deformation_stats is not None and position is not None:
        if 0 <= position < len(deformation_stats) and deformation_stats[position] is not None:
            stats = deformation_stats[position]

    if stats is not None:
        # --- Data-driven parameters from MaizeField3D ---
        # Per-leaf random multiplier for inter-leaf variation (0.6x - 1.6x).
        # Prevents the "uniform spiral" look where every leaf deforms identically.
        variation = rng.uniform(0.6, 1.6)

        # Edge ruffle: measured median +/- std, clamped non-negative
        ruffle_med = stats['ruffle_amp_median']
        ruffle_std = stats['ruffle_amp_std']
        edge_ruffle_amp = max(0.0, rng.normal(ruffle_med, ruffle_std)) * variation

        # Fold: measured median +/- std
        fold_med = stats['fold_amp_median']
        fold_std = stats['fold_amp_std']
        fold_amp = max(0.0, rng.normal(fold_med, fold_std)) * variation

        # Twist: measured total twist in degrees, convert to radians
        twist_med = stats['twist_total_median']
        twist_std = stats['twist_total_std']
        twist_deg = max(0.0, rng.normal(twist_med, twist_std))
        twist_dir = rng.choice([-1, 1])
        twist_max = twist_dir * np.radians(twist_deg)

        # Curl: position-dependent asymmetric edge curl from JSON.
        # These are hand-estimated values since NURBS can't capture curl.
        curl_med = stats.get('curl_amp_median', 0.0)
        curl_std = stats.get('curl_amp_std', 0.0)
        if curl_med > 0:
            curl_amp = max(0.0, rng.normal(curl_med, curl_std)) * variation
        else:
            # Fallback if curl not in JSON
            curl_amp = rng.uniform(0.5, 1.5) * intensity
        curl_onset = stats.get('curl_onset_median', 0.15)

        # Ramp onset from measured data
        ramp_onset = stats.get('ruffle_onset_median', 0.05)
    else:
        # --- Hand-tuned fallback ---
        variation = rng.uniform(0.6, 1.6)
        edge_ruffle_amp = rng.uniform(0.6, 1.8) * intensity * variation
        fold_amp = rng.uniform(0.3, 1.0) * intensity * variation
        twist_dir = rng.choice([-1, 1])
        twist_max = twist_dir * np.radians(rng.uniform(15, 40)) * intensity
        curl_amp = rng.uniform(0.5, 1.5) * intensity * variation
        curl_onset = 0.15
        ramp_onset = 0.15

    # Vertical undulation — midrib stays mostly straight (always hand-tuned,
    # not captured by cross-section analysis)
    normal_amp = leaf_length * rng.uniform(0.005, 0.015) * intensity
    normal_freq = rng.uniform(1.5, 2.5)
    normal_phase = rng.uniform(0, 2 * np.pi)

    # Lateral sway — very subtle (always hand-tuned)
    lateral_amp = leaf_length * rng.uniform(0.003, 0.008) * intensity
    lateral_freq = rng.uniform(1.0, 2.0)
    lateral_phase = rng.uniform(0, 2 * np.pi)

    # Curl frequency and phase — wider range for more inter-leaf variation
    curl_freq = rng.uniform(0.6, 2.0)
    curl_phase = rng.uniform(0, 2 * np.pi)

    # Frequencies and phases are always random (measured data gives amplitudes only)
    # Wider frequency ranges than before for less uniform appearance
    edge_ruffle_freq = rng.uniform(1.8, 4.5)
    edge_ruffle_phase = rng.uniform(0, 2 * np.pi)
    fold_freq = rng.uniform(0.8, 2.5)
    fold_phase = rng.uniform(0, 2 * np.pi)

    return {
        "wave_normal_amp": normal_amp,
        "wave_normal_freq": normal_freq,
        "wave_normal_phase": normal_phase,
        "wave_lateral_amp": lateral_amp,
        "wave_lateral_freq": lateral_freq,
        "wave_lateral_phase": lateral_phase,
        "twist_max": twist_max,
        "curl_amp": curl_amp,
        "curl_freq": curl_freq,
        "curl_phase": curl_phase,
        "curl_onset": curl_onset,
        "edge_ruffle_amp": edge_ruffle_amp,
        "edge_ruffle_freq": edge_ruffle_freq,
        "edge_ruffle_phase": edge_ruffle_phase,
        "fold_amp": fold_amp,
        "fold_freq": fold_freq,
        "fold_phase": fold_phase,
        "ramp_onset": ramp_onset,
    }


def extract_organs_from_plant(plant, deformation_stats=None) -> list[dict]:
    """Extract organ skeletons and widths from a CPlantBox MappedPlant.

    Args:
        plant: pb.MappedPlant instance
        deformation_stats: Optional list of per-position deformation stat dicts
            from load_deformation_stats(). If None, attempts to load from default path.

    Returns list of organ dicts with keys: type, skeleton, widths, organ_id, name
    """
    import plantbox as pb

    # Auto-load deformation stats if not provided
    if deformation_stats is None:
        deformation_stats = load_deformation_stats()

    organs = []
    organ_counter = 0
    leaf_position = 0  # track leaf position for deformation lookup

    # Stems
    for organ in plant.getOrgans(pb.stem):
        nodes = organ.getNodes()
        if len(nodes) < 2:
            continue
        skeleton = np.array([[n.x, n.y, n.z] for n in nodes])
        radius = organ.getParameter("a")
        widths = np.full(len(nodes), 2.0 * radius)

        # Global node IDs for DART→CPlantBox segment mapping
        node_ids = [int(nid) for nid in organ.getNodeIds()]

        organs.append({
            "type": "stem",
            "skeleton": skeleton,
            "widths": widths,
            "organ_id": organ_counter,
            "name": f"stem_{organ_counter}",
            "node_ids": node_ids,
        })
        organ_counter += 1

    # Leaves
    for organ in plant.getOrgans(pb.leaf):
        nodes = organ.getNodes()
        if len(nodes) < 2:
            continue

        lrp = organ.getLeafRandomParameter()
        width_blade = lrp.Width_blade

        # Skip broken leaf subtypes with near-zero Width_blade
        if width_blade < 0.01:
            continue

        skeleton = np.array([[n.x, n.y, n.z] for n in nodes])
        phi = np.array(lrp.leafGeometryPhi)
        x = np.array(lrp.leafGeometryX)

        if len(phi) > 0 and len(x) > 0 and width_blade > 0:
            # Compute arc lengths for each node
            diffs = np.diff(skeleton, axis=0)
            seg_lengths = np.linalg.norm(diffs, axis=1)
            cumulative = np.concatenate([[0.0], np.cumsum(seg_lengths)])
            total_length = cumulative[-1]

            if total_length < 1e-12:
                widths = np.full(len(nodes), width_blade * 2.0)
            else:
                fracs = cumulative / total_length
                # Map fraction [0,1] to phi range
                phi_min, phi_max = phi.min(), phi.max()
                node_phi = phi_min + fracs * (phi_max - phi_min)
                # Interpolate x from the leafGeometry profile
                rel_widths = np.interp(node_phi, phi, x)
                # Width_blade is half-width in CPlantBox -> multiply by 2 for full width
                widths = rel_widths * width_blade * 2.0
        else:
            widths = np.full(len(nodes), max(width_blade * 2.0, DEFAULT_MIN_WIDTH))

        # Global node IDs for DART→CPlantBox segment mapping
        node_ids = [int(nid) for nid in organ.getNodeIds()]

        # Leaf blade waviness + twist: natural variation per leaf.
        # Uses organ_counter as seed for repeatable but varied parameters.
        leaf_length = np.sum(np.linalg.norm(np.diff(skeleton, axis=0), axis=1))
        rng = np.random.RandomState(organ_counter * 37 + 7)
        wave_params = _leaf_wave_params(
            leaf_length, rng,
            position=leaf_position,
            deformation_stats=deformation_stats,
        )

        organs.append({
            "type": "leaf",
            "skeleton": skeleton,
            "widths": widths,
            "organ_id": organ_counter,
            "name": f"leaf_{organ_counter}",
            "node_ids": node_ids,
            **wave_params,
        })
        organ_counter += 1
        leaf_position += 1

    return organs


def resample_skeleton_uniform(skeleton, widths, min_nodes):
    """
    Resample skeleton to ensure minimum node count with uniform arc-length spacing.

    Args:
        skeleton: Nx3 array of node positions
        widths: N array of widths at each node
        min_nodes: Minimum number of nodes after resampling

    Returns:
        resampled_skeleton, resampled_widths
    """
    if len(skeleton) >= min_nodes:
        return skeleton, widths

    # Compute arc lengths
    diffs = np.diff(skeleton, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    total_length = cumulative[-1]

    if total_length < 1e-12:
        # Degenerate skeleton, just duplicate last node
        return skeleton, widths

    # Resample at uniform arc-length intervals
    target_s = np.linspace(0, total_length, min_nodes)

    # Interpolate positions
    resampled_skeleton = np.zeros((min_nodes, 3))
    for i in range(3):
        resampled_skeleton[:, i] = np.interp(target_s, cumulative, skeleton[:, i])

    # Interpolate widths
    resampled_widths = np.interp(target_s, cumulative, widths)

    return resampled_skeleton, resampled_widths


def extract_organs_for_lofter(plant, min_stem_nodes=50, min_leaf_nodes=20,
                              skip_roots=True, deformation_stats=None,
                              stem_profile=None, name_prefix=""):
    """
    Extract organs from CPlantBox with resampling for high-resolution meshes.

    This is the recommended function for the G1→G3 pipeline when you want
    maximum mesh detail and realism.

    Args:
        plant: pb.MappedPlant
        min_stem_nodes: Minimum nodes per stem organ (controls tube smoothness)
        min_leaf_nodes: Minimum nodes per leaf organ (controls leaf surface detail)
        skip_roots: If True, exclude root organs from the output (default True).
                    Roots can remain functional in the CPlantBox simulation for
                    water uptake / photosynthesis without appearing in the G3 mesh.
        deformation_stats: Optional list of per-position deformation stat dicts
            from load_deformation_stats(). If None, attempts to load from default path.
        stem_profile: Optional dict from load_stem_profile(). If None, attempts
            to load from default path. Provides measured radius-vs-height taper.

    Returns:
        List of organ dicts with resampled skeletons
    """
    import plantbox as pb

    # Auto-load deformation stats if not provided
    if deformation_stats is None:
        deformation_stats = load_deformation_stats()

    # Auto-load stem profile if not provided
    if stem_profile is None:
        stem_profile = load_stem_profile()

    organs = []
    organ_counter = 0
    leaf_position = 0

    # Stems with resampling
    for organ in plant.getOrgans(pb.stem):
        nodes = organ.getNodes()
        if len(nodes) < 2:
            continue

        skeleton = np.array([[n.x, n.y, n.z] for n in nodes])
        radius = organ.getParameter("a")
        widths = np.full(len(nodes), 2.0 * radius)

        # Preserve original node_ids before resampling (for DART mapping)
        node_ids = [int(nid) for nid in organ.getNodeIds()]

        # Resample for smooth tubes
        skeleton, widths = resample_skeleton_uniform(skeleton, widths, min_stem_nodes)

        # Apply measured stem radius profile (MaizeField3D taper)
        if stem_profile is not None:
            widths = apply_stem_profile(skeleton, widths, stem_profile)

        # Stem maturity scaling: young stems are thinner than mature ones.
        # The MaizeField3D profile gives mature-plant radii — scale down
        # proportionally to how much of the stem has grown.
        stem_length = np.sum(np.linalg.norm(np.diff(skeleton, axis=0), axis=1))
        lmax = organ.getParameter("lmax")
        if lmax > 1.0:
            stem_maturity = min(stem_length / lmax, 1.0)
            width_scale = max(0.20, stem_maturity ** 0.35)
            widths *= width_scale

        organs.append({
            "type": "stem",
            "skeleton": skeleton,
            "widths": widths,
            "organ_id": organ_counter,
            "name": f"{name_prefix}stem_{organ_counter}",
            "node_ids": node_ids,
        })
        organ_counter += 1

    # Leaves with resampling
    for organ in plant.getOrgans(pb.leaf):
        nodes = organ.getNodes()
        if len(nodes) < 2:
            continue

        lrp = organ.getLeafRandomParameter()
        width_blade = lrp.Width_blade

        # Skip broken leaf subtypes
        if width_blade < 0.01:
            continue

        skeleton = np.array([[n.x, n.y, n.z] for n in nodes])
        phi = np.array(lrp.leafGeometryPhi)
        x = np.array(lrp.leafGeometryX)

        # Compute widths using leafGeometry profile
        if len(phi) > 0 and len(x) > 0:
            diffs = np.diff(skeleton, axis=0)
            seg_lengths = np.linalg.norm(diffs, axis=1)
            cumulative = np.concatenate([[0.0], np.cumsum(seg_lengths)])
            total_length = cumulative[-1]

            if total_length > 1e-12:
                fracs = cumulative / total_length
                phi_min, phi_max = phi.min(), phi.max()
                node_phi = phi_min + fracs * (phi_max - phi_min)
                rel_widths = np.interp(node_phi, phi, x)
                widths = rel_widths * width_blade * 2.0  # half-width → full width
            else:
                widths = np.full(len(nodes), width_blade * 2.0)
        else:
            widths = np.full(len(nodes), max(width_blade * 2.0, DEFAULT_MIN_WIDTH))

        # Maturity-based width scaling: young/emerging leaves are narrow
        # (rolled inside the whorl) and only reach full width when nearly
        # mature.  Without this, a 5 cm emerging leaf gets full 7 cm width.
        lmax = max(lrp.lmax, 1.0)
        current_length = np.sum(np.linalg.norm(np.diff(skeleton, axis=0), axis=1))
        maturity = min(current_length / lmax, 1.0)
        if maturity < 0.95:
            # Power curve: gentle ramp that keeps young leaves very narrow
            # mat=0.10→15%, mat=0.25→35%, mat=0.50→62%, mat=0.80→87%
            unfurl = max(maturity ** 0.6, 0.08)
            widths *= unfurl

        # Preserve original node_ids before resampling (for DART mapping)
        node_ids = [int(nid) for nid in organ.getNodeIds()]

        # Resample for detailed leaf surfaces
        skeleton, widths = resample_skeleton_uniform(skeleton, widths, min_leaf_nodes)

        # Trim leaf tips that droop below ground (z < 0).
        # Never trim the base — the leaf must stay attached to the stem
        # even if the attachment point is below ground (seed depth).
        # Only check the tail end of the skeleton for below-ground nodes.
        n_skel = len(skeleton)
        if n_skel > 2 and skeleton[-1, 2] < 0:
            # Find last above-ground node (searching from the tip backward)
            last_above = n_skel - 1
            while last_above > 0 and skeleton[last_above, 2] < 0:
                last_above -= 1
            if last_above < 1:
                continue  # entire leaf underground, skip
            # Interpolate crossing point
            p_above = skeleton[last_above]
            p_below = skeleton[last_above + 1]
            dz = p_below[2] - p_above[2]
            if abs(dz) > 1e-12:
                t_interp = np.clip(-p_above[2] / dz, 0.0, 1.0)
                crossing = p_above + t_interp * (p_below - p_above)
                w_crossing = widths[last_above] + t_interp * (widths[last_above + 1] - widths[last_above])
                skeleton = np.vstack([skeleton[:last_above + 1], crossing[np.newaxis]])
                widths = np.concatenate([widths[:last_above + 1], [w_crossing]])
            else:
                skeleton = skeleton[:last_above + 1]
                widths = widths[:last_above + 1]

        # Pointed tip taper: real maize leaves taper linearly over the last
        # ~30% of blade length, like a spear tip with straight converging
        # edges.  The MaizeField3D profiles maintain ~80% width until 84%
        # then drop abruptly — replace with a gradual linear taper.
        n_skel = len(skeleton)
        if n_skel >= 4:
            diffs_tip = np.diff(skeleton, axis=0)
            seg_lens = np.linalg.norm(diffs_tip, axis=1)
            cum_len = np.concatenate([[0.0], np.cumsum(seg_lens)])
            total_len = cum_len[-1]
            if total_len > 1e-6:
                taper_start = 0.70  # taper begins at 70% of leaf length
                for i in range(n_skel):
                    frac = cum_len[i] / total_len
                    if frac > taper_start:
                        # Linear taper: 1 at taper_start → 0 at tip
                        t = (frac - taper_start) / (1.0 - taper_start)
                        envelope = 1.0 - t
                        widths[i] = min(widths[i], widths[i] * envelope)
            # Truncate skeleton where width falls below min_tip_width.
            # Previously appended a zero-width extension point + clamped to
            # 0.01 cm, but this produced degenerate slivers (~0.0001 cm²)
            # that cause Baleno's Newton solver to diverge (Tleaf > 80 °C).
            # Instead, cut the skeleton at the last point with meaningful
            # width — the taper already makes the tip look pointed.
            min_tip_width = 0.15  # cm (1.5 mm) — smallest non-degenerate width
            last_good = len(widths) - 1
            while last_good > 0 and widths[last_good] < min_tip_width:
                last_good -= 1
            if last_good < len(widths) - 1 and last_good >= 1:
                # Interpolate to find where width crosses min_tip_width
                # for a clean truncation point
                next_idx = last_good + 1
                w0, w1 = widths[last_good], widths[next_idx]
                if abs(w0 - w1) > 1e-8:
                    t_interp = (min_tip_width - w0) / (w1 - w0)
                    t_interp = np.clip(t_interp, 0.0, 1.0)
                    cut_point = skeleton[last_good] + t_interp * (skeleton[next_idx] - skeleton[last_good])
                    skeleton = np.vstack([skeleton[:last_good + 1], cut_point[np.newaxis]])
                    widths = np.concatenate([widths[:last_good + 1], [min_tip_width]])
                else:
                    skeleton = skeleton[:last_good + 1]
                    widths = widths[:last_good + 1]

        # Clamp minimum width — safety net after truncation.
        widths = np.maximum(widths, 0.15)

        # Trim node_ids to match skeleton length after ground-clipping and
        # tip truncation.  node_ids was captured before these operations,
        # so it may be longer than the skeleton.  The mapping JSON uses
        # len(node_ids)-1 as the segment count — any segments beyond the
        # skeleton length would get zero triangles.
        if len(node_ids) > len(skeleton):
            node_ids = node_ids[:len(skeleton)]

        # Leaf blade waviness + twist
        leaf_length = np.sum(np.linalg.norm(np.diff(skeleton, axis=0), axis=1))
        rng = np.random.RandomState(organ_counter * 37 + 7)
        wave_params = _leaf_wave_params(
            leaf_length, rng,
            position=leaf_position,
            deformation_stats=deformation_stats,
        )

        # Scale wave/deformation amplitudes by maturity: young emerging
        # leaves are smooth and straight — undulation, curl, twist, and
        # ruffle only develop as the blade matures and stiffens.
        if maturity < 0.95:
            amp_keys = [
                'wave_normal_amp', 'wave_lateral_amp', 'twist_max',
                'curl_amp', 'edge_ruffle_amp', 'fold_amp',
            ]
            for k in amp_keys:
                if k in wave_params:
                    wave_params[k] *= unfurl

        # Extract spline-based geometry features from CPlantBox LeafRandomParameter
        # (leafOOPCurvPhi/Kappa, leafAsymmetry, leafEdgeCurl, leafCrossSection).
        # These are empty lists if not set in the XML — the lofter treats empty as no-op.
        spline_features = {}
        for attr_phi, attr_val, key in [
            ('leafOOPCurvPhi', 'leafOOPCurvKappa', 'oop_curv_spline'),
            ('leafAsymmetryPhi', 'leafAsymmetryOffset', 'asymmetry_spline'),
            ('leafEdgeCurlPhi', 'leafEdgeCurlAngle', 'edge_curl_spline'),
            ('leafCrossSectionPhi', 'leafCrossSectionCurv', 'cross_section_spline'),
        ]:
            phi_vals = list(getattr(lrp, attr_phi, []))
            data_vals = list(getattr(lrp, attr_val, []))
            if phi_vals and data_vals and len(phi_vals) == len(data_vals):
                spline_features[key] = {
                    'phi': np.array(phi_vals),
                    'values': np.array(data_vals),
                }

        organs.append({
            "type": "leaf",
            "skeleton": skeleton,
            "widths": widths,
            "organ_id": organ_counter,
            "name": f"{name_prefix}leaf_{organ_counter}",
            "node_ids": node_ids,
            **wave_params,
            **spline_features,
        })
        organ_counter += 1
        leaf_position += 1

    # Collect leaf attachment Z heights for internode modulation.
    leaf_bases_z = sorted([o['skeleton'][0, 2] for o in organs if o['type'] == 'leaf'])

    # Trim stem above the highest leaf attachment point.
    # CPlantBox grows the stem beyond the last leaf, but the bare tip
    # looks unrealistic in the mesh.  Truncate to highest_leaf_base + margin.
    if leaf_bases_z:
        max_leaf_z = max(leaf_bases_z)
        margin = 2.0  # cm above last leaf attachment
        cut_z = max_leaf_z + margin
        for o in organs:
            if o['type'] != 'stem':
                continue
            skel = o['skeleton']
            if skel[-1, 2] <= cut_z:
                continue
            # Find last node at or below cut_z
            above = np.where(skel[:, 2] > cut_z)[0]
            if len(above) == 0:
                continue
            first_above = above[0]
            if first_above < 2:
                continue  # don't trim to nothing
            # Interpolate crossing point
            p_below = skel[first_above - 1]
            p_above = skel[first_above]
            dz = p_above[2] - p_below[2]
            if abs(dz) > 1e-12:
                t = (cut_z - p_below[2]) / dz
                crossing = p_below + t * (p_above - p_below)
                w_cross = o['widths'][first_above - 1] + t * (
                    o['widths'][first_above] - o['widths'][first_above - 1])
                o['skeleton'] = np.vstack([skel[:first_above], crossing[np.newaxis]])
                o['widths'] = np.concatenate([o['widths'][:first_above], [w_cross]])
            else:
                o['skeleton'] = skel[:first_above]
                o['widths'] = o['widths'][:first_above]

    # Pass leaf attachment heights to stems for internode modulation.
    if leaf_bases_z:
        for o in organs:
            if o['type'] == 'stem':
                o['node_heights_z'] = list(leaf_bases_z)

    # Roots (optional, usually skip for shoot-only viz)
    if skip_roots:
        return organs

    for organ in plant.getOrgans(pb.root):
        nodes = organ.getNodes()
        if len(nodes) < 2:
            continue

        skeleton = np.array([[n.x, n.y, n.z] for n in nodes])
        radius = organ.getParameter("a")
        widths = np.full(len(nodes), 2.0 * radius)

        # Preserve original node_ids before resampling
        node_ids = [int(nid) for nid in organ.getNodeIds()]

        # Coarser resampling for roots (less critical for visualization)
        skeleton, widths = resample_skeleton_uniform(skeleton, widths, max(20, min_stem_nodes // 2))

        organs.append({
            "type": "root",
            "skeleton": skeleton,
            "widths": widths,
            "organ_id": organ_counter,
            "name": f"root_{organ_counter}",
            "node_ids": node_ids,
        })
        organ_counter += 1

    return organs
