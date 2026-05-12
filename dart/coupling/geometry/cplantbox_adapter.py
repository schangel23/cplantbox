"""CPlantBox adapter for the G1-to-G3 lofting pipeline.

Extracts skeleton and width data from CPlantBox plant objects into
organ dicts compatible with g1_to_g3.loft_organs().
"""

import json
import math
import os
import numpy as np
from pathlib import Path

from ..config import MAIZEFIELD3D_DEFORMATION, MAIZEFIELD3D_STEM_PROFILE
from .g1_to_g3 import _stem_internode_width_scale_at_z

DEFAULT_MIN_WIDTH = 0.1  # cm, fallback when Width_blade=0


# Default leaf-fracture configuration — ports the DART PlantSimulation
# stochastic break-point model (maize_growth.py:105-107,:124 — see
# DART_PLANTSIMULATION_LEARNINGS.md §3.1). Upper-canopy leaves are more
# exposed to wind/rain than lower ranks, so ``break_prob_high`` >
# ``break_prob_low``. The field-measured survival fraction sits in the
# range [``break_loc_small``, ``break_loc_big``] of the original lmax.
# The dict is treated as a full spec: enable with ``enabled=True`` and
# optionally override any subset of the parameters.
DEFAULT_LEAF_FRACTURE = {
    "enabled": False,
    "seed": 1234,
    "leaf_split": 6,       # rank < leaf_split uses break_prob_low
    "break_prob_low": 0.05,
    "break_prob_high": 0.20,
    "break_loc_small": 0.55,
    "break_loc_big": 0.90,
}


def _resolve_fracture_cfg(cfg):
    """Return a filled-in fracture config dict, or None if disabled.

    Accepts ``None`` (default), ``False``/``True`` (shortcut toggles),
    or a partial dict that gets merged over :data:`DEFAULT_LEAF_FRACTURE`.
    """
    if cfg in (None, False):
        return None
    if cfg is True:
        cfg = {"enabled": True}
    if not isinstance(cfg, dict):
        raise TypeError(
            f"leaf_fracture must be None/bool/dict, got {type(cfg).__name__}"
        )
    merged = {**DEFAULT_LEAF_FRACTURE, **cfg}
    if not merged.get("enabled", False):
        return None
    return merged


def _env_bool(name: str) -> bool:
    """Parse boolean-like env vars: 1/0, true/false, yes/no, on/off."""
    val = os.environ.get(name, "").strip().lower()
    return val in ("1", "true", "yes", "on")


def get_plantsim_feature_kwargs_from_env() -> dict:
    """Build the kwargs dict for :func:`extract_organs_for_lofter` based on
    ``COUPLING_*`` env vars set by the CLI (see ``__main__.py``).

    Keeps the default behaviour (no flags → empty-ish dict → features off)
    so call sites can splat unconditionally:

        organs = extract_organs_for_lofter(plant,
                                           **get_plantsim_feature_kwargs_from_env())

    Recognised env vars:

    - ``COUPLING_LEAF_FRACTURE`` (bool)         — master toggle for §3.1.
    - ``COUPLING_LEAF_FRACTURE_SEED`` (int)     — RNG seed.
    - ``COUPLING_LEAF_FRACTURE_PROB_LOW`` (float)
    - ``COUPLING_LEAF_FRACTURE_PROB_HIGH`` (float)
    - ``COUPLING_LEAF_FRACTURE_BREAK_LO`` (float)
    - ``COUPLING_LEAF_FRACTURE_BREAK_HI`` (float)
    - ``COUPLING_LEAF_FRACTURE_SPLIT`` (int)    — rank threshold.
    - ``COUPLING_SENESCENT_SPLIT`` (bool)       — master toggle for §3.2.
    - ``COUPLING_SENESCENT_RHO_THRESHOLD`` (float).
    """
    kwargs: dict = {}

    if _env_bool("COUPLING_LEAF_FRACTURE"):
        cfg: dict = {"enabled": True}
        for env_key, cfg_key, caster in (
            ("COUPLING_LEAF_FRACTURE_SEED",       "seed",             int),
            ("COUPLING_LEAF_FRACTURE_SPLIT",      "leaf_split",       int),
            ("COUPLING_LEAF_FRACTURE_PROB_LOW",   "break_prob_low",   float),
            ("COUPLING_LEAF_FRACTURE_PROB_HIGH",  "break_prob_high",  float),
            ("COUPLING_LEAF_FRACTURE_BREAK_LO",   "break_loc_small",  float),
            ("COUPLING_LEAF_FRACTURE_BREAK_HI",   "break_loc_big",    float),
        ):
            raw = os.environ.get(env_key)
            if raw is not None and raw != "":
                try:
                    cfg[cfg_key] = caster(raw)
                except ValueError:
                    print(f"  [plantsim-features] ignoring malformed {env_key}={raw!r}")
        kwargs["leaf_fracture"] = cfg

    if _env_bool("COUPLING_SENESCENT_SPLIT"):
        kwargs["enable_senescent_split"] = True
        thr = os.environ.get("COUPLING_SENESCENT_RHO_THRESHOLD")
        if thr:
            try:
                kwargs["senescent_rho_threshold"] = float(thr)
            except ValueError:
                print(f"  [plantsim-features] ignoring malformed "
                      f"COUPLING_SENESCENT_RHO_THRESHOLD={thr!r}")

    return kwargs


def _apply_leaf_fracture(skeleton, widths, position, rng, cfg):
    """Stochastically truncate a leaf skeleton at a break point.

    Mirrors the DART PlantSimulation ``Bernoulli(p) × Uniform(lo, hi)``
    fracture model (maize_growth.py:105-107, :124). Returns a
    ``(new_skeleton, new_widths, break_fraction)`` triple with
    ``break_fraction == 1.0`` for intact leaves.

    The cut is an arc-length truncation: the remaining leaf ends with
    the width it had at the break point (torn-leaf profile), which the
    lofter caps naturally with its last cross-section.
    """
    p_break = cfg["break_prob_low"] if position < cfg["leaf_split"] else cfg["break_prob_high"]
    if rng.random() >= p_break:
        return skeleton, widths, 1.0

    frac = float(rng.uniform(cfg["break_loc_small"], cfg["break_loc_big"]))

    skel = np.asarray(skeleton, dtype=np.float64)
    wid = np.asarray(widths, dtype=np.float64)
    if len(skel) < 2:
        return skel, wid, 1.0

    # Arc-length cut — interpolate to keep the truncation independent of
    # the original node spacing.
    diffs = np.diff(skel, axis=0)
    seg_lens = np.linalg.norm(diffs, axis=1)
    cumul = np.concatenate([[0.0], np.cumsum(seg_lens)])
    total = cumul[-1]
    if total <= 1e-9:
        return skel, wid, 1.0
    cut = frac * total

    keep = np.searchsorted(cumul, cut, side="right")
    if keep < 2:
        keep = 2  # preserve at least one segment so the lofter has geometry
    if keep >= len(skel):
        return skel, wid, 1.0

    new_skel = skel[:keep].copy()
    new_wid = wid[:keep].copy()

    # Place the final node exactly at the cut so the truncated leaf
    # length honours ``break_fraction * lmax``.
    remainder = cut - cumul[keep - 1]
    seg = seg_lens[keep - 1] if keep - 1 < len(seg_lens) else 0.0
    if seg > 1e-9 and 0.0 < remainder < seg:
        t = remainder / seg
        new_skel[-1] = skel[keep - 1] + t * (skel[keep] - skel[keep - 1])
        new_wid[-1] = wid[keep - 1] + t * (wid[keep] - wid[keep - 1])

    return new_skel, new_wid, frac

# Default path for MaizeField3D blade deformation stats
_DEFAULT_DEFORMATION_JSON = MAIZEFIELD3D_DEFORMATION

# Default path for MaizeField3D stem radius profile
_DEFAULT_STEM_PROFILE_JSON = MAIZEFIELD3D_STEM_PROFILE

_deformation_cache = {}
_stem_profile_cache = {}
_reference_profiles_cache = {}


_DEFAULT_REFERENCE_PROFILES_JSON = Path(
    "/home/lukas/PHD/Resources/MaizeField3d/maizefield3d_reference_profiles.json"
)

_VIDAL_SHEATH_JSON = Path(__file__).resolve().parents[1] / "data" / "vidal_per_rank_sheath_cm.json"


def load_vidal_sheath_lengths(path=None):
    """Load Vidal 2021 SupData1 per-rank mature sheath lengths (M40+M52 averaged).

    Returns:
        Dict mapping ``rank`` (int, 0-indexed: S0..S20) -> sheath_length_cm (float),
        or ``None`` if the JSON isn't present. Provenance: vidal_2021_per_rank.
    """
    if path is None:
        path = _VIDAL_SHEATH_JSON
    path = Path(path)
    key = "vidal:" + str(path)
    if key in _reference_profiles_cache:
        return _reference_profiles_cache[key]
    if not path.exists():
        _reference_profiles_cache[key] = None
        return None
    with open(path) as f:
        data = json.load(f)
    per_rank = data.get("per_rank", [])
    out = {}
    for entry in per_rank:
        rank = int(entry["rank"])
        L = entry.get("sheath_length_cm")
        if L is None or L <= 0:
            continue
        out[rank] = float(L)
    _reference_profiles_cache[key] = out
    return out


def load_reference_profiles(path=None):
    """Load MaizeField3D per-position reference profiles (sheath length etc.).

    For maize sheath length the Vidal 2021 SupData1 cultivar-averaged values
    (load_vidal_sheath_lengths) take precedence at the consumer site
    (cplantbox_adapter.py ~line 2002). MF3D medians remain as fallback for
    positions Vidal doesn't cover, and as the only source for non-sheath
    fields if any are added later.

    Returns:
        Dict mapping ``position`` (int) -> ``{"sheath_length_cm_median": float}``,
        or ``None`` if the JSON isn't present.
    """
    if path is None:
        path = _DEFAULT_REFERENCE_PROFILES_JSON
    path = Path(path)
    key = str(path)
    if key in _reference_profiles_cache:
        return _reference_profiles_cache[key]
    if not path.exists():
        _reference_profiles_cache[key] = None
        return None
    with open(path) as f:
        data = json.load(f)
    per_position = data.get("per_position", [])
    out = {}
    for entry in per_position:
        pos = int(entry["position"])
        sheath = entry.get("sheath_length_cm", {}) or {}
        median = sheath.get("median")
        if median is None or median <= 0:
            continue
        out[pos] = {"sheath_length_cm_median": float(median)}
    _reference_profiles_cache[key] = out
    return out


def _parent_tangent_at_collar(organ, collar_pos):
    """Unit tangent of the parent organ (stem) at the leaf's collar.

    Used by the compound sheath so the NURBS tube wraps the actual stem
    axis at the collar, not the leaf's blade tangent (which can be
    horizontal or drooping). Finds the two consecutive parent-node
    positions bracketing the collar and returns the chord direction.
    Falls back to world +z when the parent is missing or degenerate.
    """
    try:
        parent = organ.getParent()
    except Exception:
        parent = None
    default = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if parent is None:
        return default
    try:
        nodes = parent.getNodes()
        if len(nodes) < 2:
            return default
        pts = np.array([[n.x, n.y, n.z] for n in nodes], dtype=np.float64)
        collar = np.asarray(collar_pos, dtype=np.float64)
        # Nearest node to the collar; tangent = chord to the adjacent node
        # that lies *above* it (in arc-length along the stem).
        d = np.linalg.norm(pts - collar[None, :], axis=1)
        i = int(np.argmin(d))
        if i + 1 < len(pts):
            tan = pts[i + 1] - pts[i]
        elif i - 1 >= 0:
            tan = pts[i] - pts[i - 1]
        else:
            return default
        n = float(np.linalg.norm(tan))
        if n < 1e-9:
            return default
        return tan / n
    except Exception:
        return default


def _stem_radius_at_collar_cm(organ, collar_z=None, stem_profile=None, fallback=0.3):
    """Return the parent-stem **rendered** radius at the leaf's collar, in cm.

    The sheath NURBS must wrap the *visible* stem cylinder, not the bare
    CPlantBox ``a`` parameter. The stem mesh (``_loft_stem``) applies the
    MF3D envelope profile plus a maturity width scaling; this helper
    replicates both so the sheath sits flush with the stem surface.

    When ``collar_z`` or ``stem_profile`` is unavailable, falls back to
    ``Stem::getRadiusAt(0.0)`` (Phase E.3), then ``parent.param().a``.
    """
    parent = None
    try:
        parent = organ.getParent()
    except Exception:
        parent = None
    if parent is None:
        return fallback

    # Rendered-radius path: must match _loft_stem's width pipeline.
    if stem_profile is not None and collar_z is not None:
        try:
            p_nodes = parent.getNodes()
            if len(p_nodes) >= 2:
                zs = np.array([n.z for n in p_nodes])
                z_min, z_max = float(zs.min()), float(zs.max())
                if z_max - z_min >= 1.0:
                    z_frac = float(np.clip(
                        (float(collar_z) - z_min) / (z_max - z_min), 0.0, 1.0,
                    ))
                    radius_cm = float(np.interp(
                        z_frac, stem_profile["height_frac"],
                        stem_profile["radius_cm"],
                    ))
                    # Maturity scaling: width_scale = max(0.08, maturity**0.8)
                    # Steeper exponent + lower floor than the original
                    # (0.20, 0.35): real maize seedlings are ~2 mm at V3
                    # (day 10–15, maturity ~0.06) whereas the old curve
                    # pinned anything below maturity=0.1 at 20 % of mature.
                    try:
                        stem_len = float(np.sum(np.linalg.norm(
                            np.diff(np.array([[n.x, n.y, n.z]
                                              for n in p_nodes]), axis=0),
                            axis=1,
                        )))
                        lmax = float(parent.getParameter("lmax"))
                        if lmax > 1.0:
                            maturity = min(stem_len / lmax, 1.0)
                            width_scale = max(0.08, maturity ** 0.8)
                            radius_cm *= width_scale
                    except Exception:
                        pass
                    return radius_cm
        except Exception:
            pass

    # Phase E.3 API
    try:
        return float(parent.getRadiusAt(0.0))
    except Exception:
        pass
    # Fallback to SpecificParameter.a
    try:
        return float(parent.param().a)
    except Exception:
        pass
    return fallback


def _make_stem_radius_at_z_callable(
    organ, collar_pos, stem_axis_world, stem_profile, fallback=0.3,
    node_heights_z=None,
):
    """Return ``stem_r(z_local_cm) -> cm`` for the compound leaf cup.

    ``z_local_cm`` is signed along the stem axis at the collar: ``0`` at
    the collar, negative below. Mirrors ``_stem_radius_at_collar_cm``'s
    rendered-radius pipeline (stem profile + maturity scaling) so the
    cup sits flush with ``_loft_stem``'s output at every height the
    sheath covers. Falls back to a constant (collar radius, then
    ``Stem::getRadiusAt(0)``, then ``param.a``) when the profile or
    parent stem geometry is unavailable.
    """
    def _constant_fallback():
        r0 = _stem_radius_at_collar_cm(
            organ, collar_z=float(collar_pos[2]),
            stem_profile=stem_profile, fallback=fallback,
        )
        return lambda _z_local: r0

    try:
        parent = organ.getParent()
    except Exception:
        parent = None
    if parent is None:
        return _constant_fallback()
    if stem_profile is None:
        return _constant_fallback()
    try:
        p_nodes = parent.getNodes()
    except Exception:
        return _constant_fallback()
    if len(p_nodes) < 2:
        return _constant_fallback()

    try:
        zs = np.array([n.z for n in p_nodes], dtype=np.float64)
        z_min, z_max = float(zs.min()), float(zs.max())
        stem_z_range = z_max - z_min
        if stem_z_range < 1.0:
            return _constant_fallback()
        stem_len = float(np.sum(np.linalg.norm(
            np.diff(np.array([[n.x, n.y, n.z] for n in p_nodes]), axis=0),
            axis=1,
        )))
        lmax = float(parent.getParameter("lmax"))
    except Exception:
        return _constant_fallback()

    width_scale = 1.0
    if lmax > 1.0:
        maturity = min(stem_len / lmax, 1.0)
        width_scale = max(0.08, maturity ** 0.8)

    collar_z_world = float(collar_pos[2])
    axis_z = float(np.asarray(stem_axis_world, dtype=np.float64)[2])
    h_frac = np.asarray(stem_profile["height_frac"], dtype=np.float64)
    r_prof = np.asarray(stem_profile["radius_cm"], dtype=np.float64)

    def stem_r_at_z_local(z_local):
        world_z = collar_z_world + axis_z * float(z_local)
        z_frac = (world_z - z_min) / stem_z_range
        if z_frac < 0.0:
            z_frac = 0.0
        elif z_frac > 1.0:
            z_frac = 1.0
        r = float(np.interp(z_frac, h_frac, r_prof))
        internode_scale = _stem_internode_width_scale_at_z(
            world_z, node_heights_z,
        )
        return r * width_scale * internode_scale

    return stem_r_at_z_local


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


def _leaf_wave_params(leaf_length, rng, position=None, deformation_stats=None,
                      species='maize'):
    """Generate blade deformation parameters for a single leaf.

    When deformation_stats are provided (from MaizeField3D extraction), uses
    measured distributions. Otherwise falls back to hand-tuned ranges.
    Species-aware: wheat leaves are much stiffer/straighter than maize.

    Note: curl is always hand-tuned because the MaizeField3D 3-point NURBS
    cross-sections are inherently symmetric and can't capture asymmetric curl.

    Args:
        leaf_length: Total arc length of the leaf skeleton (cm)
        rng: numpy RandomState for repeatable per-leaf variation
        position: Leaf position index (0-10, bottom to top). Used to look up
            position-specific measured stats.
        deformation_stats: List of per-position stat dicts from
            load_deformation_stats(), or None for hand-tuned fallback.
        species: 'maize' or 'wheat'. Wheat uses much smaller amplitudes.
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

        # Length-based intensity mirrors the hand-tuned branch: MF3D medians
        # are measured on mature blades, so short young leaves must attenuate
        # or they inherit full mature amplitude on a 3 cm blade.
        edge_ruffle_amp = max(0.0, rng.normal(stats['ruffle_amp_median'], stats['ruffle_amp_std'])) * variation * intensity

        fold_amp = max(0.0, rng.normal(stats['fold_amp_median'], stats['fold_amp_std'])) * variation * intensity

        twist_deg = max(0.0, rng.normal(stats['twist_total_median'], stats['twist_total_std']))
        twist_dir = rng.choice([-1, 1])
        twist_max = twist_dir * np.radians(twist_deg) * intensity

        # Curl: position-dependent asymmetric edge curl from JSON.
        # These are hand-estimated values since NURBS can't capture curl.
        curl_med = stats.get('curl_amp_median', 0.0)
        curl_std = stats.get('curl_amp_std', 0.0)
        if curl_med > 0:
            curl_amp = max(0.0, rng.normal(curl_med, curl_std)) * variation * intensity
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
    # not captured by cross-section analysis).
    # Independent L/R phases mirror bltree.lsy:9-10 (ZLeft, ZRight with
    # phase_L, phase_R drawn independently) so the two blade edges undulate
    # out-of-sync instead of moving as a rigid ribbon.
    normal_amp = leaf_length * rng.uniform(0.080, 0.150) * intensity
    normal_freq = rng.uniform(4.0, 6.5)
    normal_phase_L = rng.uniform(0, 2 * np.pi)
    normal_phase_R = rng.uniform(0, 2 * np.pi)
    normal_phase = normal_phase_L  # legacy key kept for back-compat

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

    # Species scaling: wheat leaves are stiff, erect, and mostly flat.
    # Dampen all deformation amplitudes to ~15-25% of maize values.
    if species == 'wheat':
        scale = 0.2
        normal_amp *= scale
        lateral_amp *= scale
        twist_max *= scale
        curl_amp *= scale * 0.5  # wheat has almost no curl
        edge_ruffle_amp *= scale
        fold_amp *= scale

    # NURBS_WAVE_GAIN is no longer applied here. It is applied in
    # nurbs_blade._apply_deformations, gated by per-leaf maturity AND
    # plant TT so the gain only amplifies mature blades on mature plants
    # and leaves young blades / young plants on their calibrated profile.
    # Default gain=1.0 remains a no-op everywhere.

    return {
        "wave_normal_amp": normal_amp,
        "wave_normal_freq": normal_freq,
        "wave_normal_phase": normal_phase,
        "wave_normal_phase_L": normal_phase_L,
        "wave_normal_phase_R": normal_phase_R,
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


def extract_organs_from_plant(plant, deformation_stats=None, species='maize',
                              leaf_fracture=None) -> list[dict]:
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

    fracture_cfg = _resolve_fracture_cfg(leaf_fracture)
    fracture_rng = (np.random.default_rng(fracture_cfg["seed"])
                    if fracture_cfg is not None else None)

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
            species=species,
        )

        break_fraction = 1.0
        if fracture_rng is not None:
            skeleton, widths, break_fraction = _apply_leaf_fracture(
                skeleton, widths, leaf_position, fracture_rng, fracture_cfg,
            )

        leaf_entry = {
            "type": "leaf",
            "skeleton": skeleton,
            "widths": widths,
            "organ_id": organ_counter,
            "name": f"leaf_{organ_counter}",
            "node_ids": node_ids,
            "break_fraction": break_fraction,
            **wave_params,
        }
        if break_fraction < 1.0:
            # Fractured tip invalidates the analytical H_top bound — the
            # predicted tip sits at H_ins + sin(theta)·lmax, the torn
            # leaf ends earlier along the arc.
            leaf_entry["check_h_top_invariant"] = False
        organs.append(leaf_entry)
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


def load_fitted_params(json_path):
    """Load per-plant fitted params from diff_lofter optimization.

    The JSON contains structural leaf params and per-leaf deformation CPs
    from gradient-fitted optimization against a scan.

    Args:
        json_path: Path to fitted params JSON (e.g. multistage_fit_v2_0001.json).

    Returns:
        Dict with 'leaf_params', 'stem_params', and per-stage 'deform_params'.
    """
    with open(json_path) as f:
        return json.load(f)


YOUNG_MORPH_FADE_END = 1.0
YOUNG_MORPH_EXP = 2.0


def _build_seedling_first_leaf_cps(n_u, n_v, length_cm=5.5, max_width_cm=1.9,
                                   width_peak_u=0.55,
                                   base_width_frac=0.18, shoulder_frac=0.85,
                                   n_cap_rows=3,
                                   gutter_depth_cm=0.18, midrib_arc_cm=0.45):
    """Construct a ‘V1 first leaf’ NURBS CP grid with a half-disc round tip.

    Rank 1 in the calibrated MF3D library carries a 24-cm sword shape
    (mature size) which renders implausibly long at V3 plants where leaf 1
    should appear as a small lance-rounded blade tapering into a visible
    sheath collar with a fully **rounded apex** (Nielsen 2004 V-stage
    reference). This helper builds a fresh small CP grid that replaces
    the mature library shape for ``leaf_position == 0`` only.

    The ``n_u`` U-rows are split into two zones along the midrib:

      * **Body** (rows 0 .. ``n_u - n_cap_rows - 1``): the leaf body
        with width envelope:
          - smoothstep base → peak: ``base_width_frac`` → 1.0  on
            u ∈ [0, ``width_peak_u``]
          - smoothstep peak → shoulder: 1.0 → ``shoulder_frac`` on
            u ∈ [``width_peak_u``, body_end]
        z is linear over the body length = ``length_cm − cap_radius``.

      * **Round cap** (last ``n_cap_rows``): a true half-disc cap.
        The cap radius equals the blade half-width at the shoulder
        (``shoulder_frac · max_width / 2``). Each cap row sits at angle
        θ_k = (k+1)/n_cap_rows · π/2 around the cap centre, with
            z = body_end_z + r · sin(θ_k)
            half-width = r · cos(θ_k)
        This traces a quarter-circle in the (z, x) plane on each side
        of the midrib, so the silhouette closes as a true half-disc
        instead of a taper or rectangular cutoff. The very last row
        sits at θ = π/2 → width 0 at z = length_cm (the tip apex), and
        the standard ``nurbs_blade`` last-row pinch keeps the surface
        clean. ``rounded_tip`` is forced False on the entry so the
        edge-widening logic doesn't re-inflate our cap rows.

    Cross-section gutter + midrib arc are unchanged (parabolic +y bow
    along the midrib + edges curl up toward +y), giving real leaf
    cross-section character.

    Frame convention: +z midrib forward, +x lateral, +y droop axis.
    """
    cps = np.zeros((n_u, n_v, 3), dtype=np.float64)
    half_v = max(1, (n_v - 1) // 2)
    half_width_cm = 0.5 * max_width_cm
    # Half-disc cap radius = half-width at the shoulder
    cap_radius = shoulder_frac * half_width_cm
    body_end_z = max(length_cm - cap_radius, 0.0)
    n_cap = max(1, min(int(n_cap_rows), n_u - 2))
    n_body = n_u - n_cap  # rows 0 .. n_body - 1 form the body
    last_body_idx = n_body - 1

    def _body_env(u):
        if u <= width_peak_u:
            t = u / max(width_peak_u, 1e-6)
            s = t * t * (3.0 - 2.0 * t)
            return base_width_frac + (1.0 - base_width_frac) * s
        t = (u - width_peak_u) / max(1.0 - width_peak_u, 1e-6)
        s = t * t * (3.0 - 2.0 * t)
        return 1.0 - (1.0 - shoulder_frac) * s

    def _row(iu):
        if iu < n_body:
            u = iu / max(1, last_body_idx)  # 0..1 across body
            env = _body_env(u)
            half_at_u = half_width_cm * env
            z_at_u = body_end_z * u
            return half_at_u, z_at_u, env
        # Cap row k = 1 .. n_cap (skips θ=0 since shoulder is the body end)
        k = iu - n_body + 1
        theta = (k / n_cap) * (math.pi / 2.0)
        half_at_u = cap_radius * math.cos(theta)
        z_at_u = body_end_z + cap_radius * math.sin(theta)
        # Pass shoulder envelope through to gutter so depth doesn't
        # snap to 0 across the cap; modulate by cos(theta) so the cap
        # itself stays smooth.
        env = shoulder_frac * math.cos(theta)
        return half_at_u, z_at_u, env

    for iu in range(n_u):
        half_at_u, z_at_u, env = _row(iu)
        # Position along midrib (used by midrib arc).
        u_global = z_at_u / max(length_cm, 1e-9)
        midrib_y = midrib_arc_cm * 4.0 * u_global * (1.0 - u_global)
        for iv in range(n_v):
            t = (iv - half_v) / half_v
            gutter_y = -gutter_depth_cm * env * (1.0 - t * t)
            cps[iu, iv, 0] = half_at_u * t
            cps[iu, iv, 1] = midrib_y + gutter_y
            cps[iu, iv, 2] = z_at_u
    return cps


def _slerp_tangent(t_curr: np.ndarray, t_par: np.ndarray, alpha: float) -> np.ndarray:
    """Spherical interpolation from ``t_curr`` to ``t_par`` by fraction ``alpha``.

    ``alpha=0`` → ``t_curr``; ``alpha=1`` → ``t_par``. Unit vectors assumed.
    Mirrors `_gen_young_theta_test.py` — kept in sync deliberately.
    """
    c = float(np.clip(np.dot(t_curr, t_par), -1.0, 1.0))
    omega = np.arccos(c)
    if omega < 1e-6:
        return t_curr
    sin_omega = np.sin(omega)
    w_curr = np.sin((1.0 - alpha) * omega) / sin_omega
    w_par = np.sin(alpha * omega) / sin_omega
    out = w_curr * t_curr + w_par * t_par
    n = float(np.linalg.norm(out))
    return out / n if n > 1e-12 else t_curr


# Per-position GDD onset for the senescence wave. Onsets propagate from base
# to top at ~150 °Cd per leaf rank, so positions 0–2 senesce around V7–V11
# (700–1000 °Cd) and the wave reaches the ear-zone leaves (pos 6–8) only
# around VT/R1 (1600–1900 °Cd). At physiological maturity (~R6, TT ≈ 1900
# °Cd) about half the canopy is actively senescing while the upper leaves
# are still healthy — matches the field-observed acropetal senescence
# pattern. Beyond pos 15 the dict returns None and ``_senescence_progress``
# stays at 0 (CPlantBox shoot tops out at 16 leaves under FA-on calibration).
MAIZE_SENESCENCE_ONSET_TT = {
    0: 700.0,  1:  850.0,  2: 1000.0,  3: 1150.0,
    4: 1300.0, 5: 1450.0,  6: 1600.0,  7: 1750.0,
    8: 1900.0, 9: 2050.0, 10: 2200.0, 11: 2350.0,
   12: 2500.0, 13: 2650.0, 14: 2800.0, 15: 2950.0,
}
# GDD span R1 → R4. Drives the senescence-progress scalar ρ ∈ [0, 1].
SENESCENCE_SPAN_TT = 800.0
# Soft cap on ρ. With the world-frame flip handling the dominant
# rotation (and the two-segment bend disabled), we can let ρ run the
# full 0→1 range so late stages get proper droop. Severity (wilt/freq/
# width/length) is bounded by the WILT_BOOST/FREQ_BOOST/SHRINK constants
# being softened — no extra cap needed here.
SENESCENCE_RHO_CEILING = 1.0
# Two-segment bend parameters (see PLAN_GEOMETRY_FIDELITY_2026-04-22 §Item 1).
# The basal segment rotates downward about the collar; the distal segment
# counter-rotates upward about a hinge point along the skeleton. The hinge
# walks slightly distal as ρ advances. The tip-lift amplitude follows a hat
# function peaking at ρ=0.5 (R2 "hook") and returns to zero at R4 (straight-
# down ribbon). Basal max is 150° (not just 90°) because maize leaves emerge
# at ~60° above horizontal — 90° rotation only reaches horizontal; 150° is
# needed so the tangent truly points straight down at R4.
SENESCENCE_THETA_BASAL_MAX = 0.0           # disabled — replaced by world-frame flip
SENESCENCE_THETA_DISTAL_MAX = 0.0          # disabled — replaced by world-frame flip
SENESCENCE_S_BEND_R1 = 0.33                # hinge at ~1/3 arc length at onset
SENESCENCE_S_BEND_R4 = 0.43                # hinge walks slightly distal by end-phase
# Basal-rotation floor — unused while the two-segment bend is disabled.
SENESCENCE_BASAL_FLOOR = 0.0
# Senescence rotation: pure downward pitch (insertion-angle reduction) +
# leaf-degradation primitives (width/length shrink, wilt/freq from later
# constants). The midrib-axis flip and natural-arch inversion experiments
# are kept as zeroed-out machinery for easy reactivation but are not used
# in the production senescence path.
#
# DROOP PITCH — rotation around local +x (world-horizontal perpendicular
# to leaf). Reduces the leaf's emergence pitch progressively, so a leaf
# that naturally emerges at +60° drops toward horizontal then below.
MIDRIB_ROLL_DEG = 0.0                  # disabled — no midrib-axis flip
SENESCENCE_DROOP_BASE_DEG = 0.0        # no immediate snap; pitch ramps from 0
SENESCENCE_DROOP_PROGRESS_DEG = 90.0   # ρ=1 → 90° pitch (a +60° leaf reaches
                                        # -30° world pitch; basal leaves drape
                                        # toward / onto the soil)
# Arch flattening — perpendicular (local +y) component of CPs scaled by
# (1 - FLATTEN·√ρ). Sqrt curve flattens FAST in the moderate-ρ band
# (0.3–0.7) where pitch is enough to look senescent but the natural arch
# is still big enough to read as a J-hook. Low-ρ leaves (just senescing)
# keep most of their arch so they still look alive-ish.
# 0.0 = keep natural arch; 1.0 = fully flatten at ρ=1.
SENESCENCE_ARCH_FLATTEN = 1.0
# Wilting strength: factor on wave/curl/ruffle/fold amps vs. mature
# baseline, scaled (1 + BOOST·ρ). With BOOST=0.5 + RHO_CEILING=1.0 the
# effective max is 1.5×. Softened from 1.5 (which gave 2.5× and still
# produced visible crinkle on the half-shrunk senescent ribbons at ρ=1).
# ``twist_max`` is intentionally excluded from this loop because it is
# an angular quantity and the lofter rotates the cross-section frame
# globally — boosting past ~70° flips the blade on its side and yields
# a midrib ridge with flat triangulated cheeks.
SENESCENCE_WILT_BOOST = 0.5
# Width/size collapse (Item 6, PLAN_GEOMETRY_FIDELITY_2026-04-22). At ρ=1 the
# blade shrinks to the floor. Width shrink applied to cps_local x-axis
# (lateral) and to the legacy widths array. Length shrink applied to cps_local
# z-axis (midrib) and to the legacy skeleton via arc truncation. Length floor
# is 0.25 (leaves get small/stubby rather than long/thin at R4, matching the
# reference image). Width floor kept at 0.30 so narrow ribbons still render
# with positive triangle area for DART.
SENESCENCE_WIDTH_SHRINK = 0.50             # at ρ=1, width *= 0.50; floor below
SENESCENCE_WIDTH_FLOOR = 0.20
SENESCENCE_LENGTH_SHRINK = 0.50            # at ρ=1, length *= 0.50; floor below
SENESCENCE_LENGTH_FLOOR = 0.20
# Ground plane clamp: blades must not punch through world z=ground level.
# Currently set generous (-200 cm) so the senescence bend's natural
# tip-up→tip-down progression is visible without pancaking. Later we can
# replace the per-CP z-clamp with a "lay flat from contact point" fold so
# late-phase leaves rest on the soil naturally without squashing.
SENESCENCE_GROUND_Z = -200.0
# Wave-frequency boost for senescent blades. The NURBS lofter reads
# ``wave_normal_freq`` (default 3.5) and ``curl_freq`` (default 2.0). Scale
# by (1 + BOOST·ρ). At BOOST=0.5 + RHO_CEILING=1.0 the effective max is
# 1.5× → wave freq ~5.3, curl freq ~3.0. Softened from 1.5 (which hit
# 2.5× = ~9 cycles/blade and produced fractal crinkle the n_cross=5–7
# tessellation couldn't resolve, surfacing as triangulated shards).
SENESCENCE_FREQ_BOOST = 0.5
# Baseline wave mute applied to NURBS-backend leaves (even mature/turgid).
# Original value 0.35 was calibrated when the library CPs were assumed to
# carry most of the surface detail, but mature blades looked glassy-smooth.
# Bumped to 0.85 so mature leaves display most of their natural wave /
# twist baseline. The earlier ρ-lerp toward 1.0 was removed because
# stacked on top of WILT_BOOST it pushed senescent amps to ~3× mature —
# senescent leaves should not get *more* wave than mature ones.
NURBS_WAVE_MUTE_BASELINE = 1.0
NURBS_CURL_MUTE_BASELINE = 0.85

# NURBS_WAVE_GAIN now lives in nurbs_blade.py (applied inside
# _apply_deformations, gated by per-leaf maturity + plant TT so gain
# affects mature blades on mature plants only). Constant kept here for
# back-compat / discoverability.
NURBS_WAVE_GAIN = float(os.environ.get("NURBS_WAVE_GAIN", "1.0"))


def _senescence_progress(position, plant_tt, species="maize"):
    """Senescence progress ρ ∈ [0, 1] for a leaf at ``position``.

    ρ=0 at R1 (pre-onset), ρ=1 at R4 (fully senesced). Maize-only for now;
    other species return 0.0. Drives the two-segment bend (Item 1 of
    PLAN_GEOMETRY_FIDELITY_2026-04-22), the arc flattening, and the wilt
    boost applied to wave/curl/ruffle amplitudes.
    """
    if species != "maize" or plant_tt is None:
        return 0.0
    onset = MAIZE_SENESCENCE_ONSET_TT.get(int(position))
    if onset is None:
        return 0.0
    excess = float(plant_tt) - float(onset)
    if excess <= 0.0:
        return 0.0
    rho_raw = excess / SENESCENCE_SPAN_TT
    return float(min(rho_raw, SENESCENCE_RHO_CEILING))


def _senescence_bend_params(rho):
    """Two-segment bend parameters for senescence progress ρ.

    Returns ``(theta_basal, theta_distal, s_bend)``:
    - ``theta_basal``: downward rotation (rad) of the whole skeleton about
      the collar. Linear in ρ, capped at ``SENESCENCE_THETA_BASAL_MAX``.
    - ``theta_distal``: upward counter-rotation (rad) of the distal segment
      about the hinge node, relative to the basal tangent. Hat function
      peaking at ρ=0.5 (R2 hook), zero at ρ=0 (R1) and ρ=1 (R4 ribbon).
    - ``s_bend``: fractional arc-length of the hinge along the skeleton.
    """
    rho_c = float(min(max(rho, 0.0), 1.0))
    # Phase coordinate τ = ρ_capped / RHO_CEILING ∈ [0, 1]. The full
    # early→mid→late bend progression happens across our capped useful range
    # of ρ (since ρ saturates at SENESCENCE_RHO_CEILING for severity reasons,
    # but we still want the bend curves to traverse all phases). Wilt /
    # freq / width / length still scale with ρ_capped — those want bounded
    # severity. Only the geometric bend uses τ.
    if SENESCENCE_RHO_CEILING > 1e-9:
        tau = min(rho_c / SENESCENCE_RHO_CEILING, 1.0)
    else:
        tau = rho_c
    basal_factor = max(SENESCENCE_BASAL_FLOOR, tau) if rho_c > 0.0 else 0.0
    theta_basal = SENESCENCE_THETA_BASAL_MAX * basal_factor
    # Hat function in τ: 0 at τ=0 (early), peak THETA_DISTAL_MAX at τ=0.5
    # (mid-phase R2 hook → tip lifts above collar), 0 at τ=1 (late, leaf
    # fully drooped without hook).
    theta_distal = SENESCENCE_THETA_DISTAL_MAX * 4.0 * tau * (1.0 - tau)
    s_bend = SENESCENCE_S_BEND_R1 + (
        SENESCENCE_S_BEND_R4 - SENESCENCE_S_BEND_R1
    ) * tau
    return theta_basal, theta_distal, s_bend


def _senescence_rotation_angles(rho):
    """Return (roll_rad, droop_rad) for the senescence rotation pair.

    Roll = full 180° flip around the midrib axis as soon as ρ>0 (snap).
    Droop = linear ramp 0° → SENESCENCE_DROOP_DEG as ρ goes 0 → 1
    (gradual downward pitch, user's "lower inner angle" at late stages).
    """
    if rho <= 0.0:
        return 0.0, 0.0
    rho_c = float(min(max(rho, 0.0), 1.0))
    roll = float(np.deg2rad(MIDRIB_ROLL_DEG))
    droop = float(np.deg2rad(
        SENESCENCE_DROOP_BASE_DEG
        + SENESCENCE_DROOP_PROGRESS_DEG * rho_c
    ))
    return roll, droop


def _apply_senescence_rotations_cps_local(cps_local, roll_rad, droop_rad):
    """Apply the senescence roll-then-droop pair to a local NURBS CP grid.

    Roll around local +z (= midrib direction at collar) preserves the
    emergence angle exactly. Droop around local +x (= horizontal axis
    perpendicular to leaf) pitches the rolled leaf downward.

    The Rodrigues rotation around +x with NEGATIVE angle pitches the tip
    toward local +y (forward-and-down in world for typical leaf tangents),
    so we apply with -droop_rad for downward pitch.
    """
    cps = np.asarray(cps_local, dtype=np.float64).copy()
    if cps.size == 0:
        return cps
    origin = np.zeros(3, dtype=np.float64)
    if abs(roll_rad) > 1e-9:
        axis_z = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        cps = _rodrigues_rotate(
            cps.reshape(-1, 3), origin, axis_z, roll_rad,
        ).reshape(cps.shape)
    if abs(droop_rad) > 1e-9:
        axis_x = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        cps = _rodrigues_rotate(
            cps.reshape(-1, 3), origin, axis_x, -droop_rad,
        ).reshape(cps.shape)
    return cps


def _clamp_cps_above_ground(cps_local, collar_pos_world, tangent_world,
                            min_world_z=SENESCENCE_GROUND_Z):
    """Clamp any CP that would land below world z=``min_world_z``.

    Mirrors the local-to-world transform from ``nurbs_blade.from_local_frame``:
    builds the same (x_local_w, y_local_w, tangent) basis, maps each CP to
    world, clamps z, and maps back. Applied after the senescence bend so that
    dropping blades flatten along the soil plane instead of punching through.
    """
    cps = np.asarray(cps_local, dtype=np.float64).copy()
    if cps.size == 0:
        return cps
    t = np.asarray(tangent_world, dtype=np.float64).reshape(3)
    n_t = float(np.linalg.norm(t))
    if n_t < 1e-12:
        return cps
    t = t / n_t
    up = np.array([0.0, 0.0, 1.0])
    x_lw = np.cross(t, up)
    x_len = float(np.linalg.norm(x_lw))
    if x_len < 1e-6:
        alt = (np.array([1.0, 0.0, 0.0]) if abs(t[0]) < 0.9
               else np.array([0.0, 1.0, 0.0]))
        x_lw = np.cross(t, alt)
        x_len = float(np.linalg.norm(x_lw))
    x_lw = x_lw / max(x_len, 1e-12)
    y_lw = np.cross(t, x_lw)
    y_lw = y_lw / max(float(np.linalg.norm(y_lw)), 1e-12)
    R = np.column_stack([x_lw, y_lw, t])            # world = R @ local
    collar = np.asarray(collar_pos_world, dtype=np.float64).reshape(3)
    flat = cps.reshape(-1, 3)
    world = flat @ R.T + collar                      # (N, 3)
    world[:, 2] = np.maximum(world[:, 2], float(min_world_z))
    back = (world - collar) @ R                      # local = R.T @ (world-collar)
    return back.reshape(cps.shape)


def _push_cps_outside_stem(cps_local, collar_pos_world, tangent_world,
                           stem_xy_at_z, stem_radius_cm, margin_cm=0.3):
    """Push any blade cross-section whose points span the stem cylinder.

    Works in world frame. ``stem_xy_at_z(z)`` returns the stem axis ``(x, y)``
    at world height ``z`` via linear interpolation through the parent stem
    nodes. For each u-slice of the NURBS CP grid (a cross-section of the
    blade at constant midrib arc), we compute the "outward" direction from
    stem axis to the slice centroid. If any point in the slice has a signed
    projection onto that direction smaller than ``stem_radius + margin``,
    the WHOLE slice is translated outward (along that same direction) so
    the nearest v-edge sits exactly at ``stem_radius + margin``. This
    prevents a wrap-around blade from dipping through the stem axis between
    CPs — a simple per-CP radial push would leave the interpolated surface
    passing near the axis even when individual CPs are outside.
    """
    cps = np.asarray(cps_local, dtype=np.float64).copy()
    if cps.size == 0 or stem_xy_at_z is None or stem_radius_cm is None:
        return cps
    t = np.asarray(tangent_world, dtype=np.float64).reshape(3)
    n_t = float(np.linalg.norm(t))
    if n_t < 1e-12:
        return cps
    t = t / n_t
    up = np.array([0.0, 0.0, 1.0])
    x_lw = np.cross(t, up)
    x_len = float(np.linalg.norm(x_lw))
    if x_len < 1e-6:
        alt = (np.array([1.0, 0.0, 0.0]) if abs(t[0]) < 0.9
               else np.array([0.0, 1.0, 0.0]))
        x_lw = np.cross(t, alt)
        x_len = float(np.linalg.norm(x_lw))
    x_lw = x_lw / max(x_len, 1e-12)
    y_lw = np.cross(t, x_lw)
    y_lw = y_lw / max(float(np.linalg.norm(y_lw)), 1e-12)
    R = np.column_stack([x_lw, y_lw, t])
    collar = np.asarray(collar_pos_world, dtype=np.float64).reshape(3)
    n_u, n_v = cps.shape[0], cps.shape[1]
    flat = cps.reshape(-1, 3)
    world = (flat @ R.T + collar).reshape(n_u, n_v, 3)

    r_min = float(stem_radius_cm) + float(margin_cm)
    # Fallback outward direction: horizontal projection of the tangent.
    t_horiz_xy = np.array([t[0], t[1]])
    n_th = float(np.linalg.norm(t_horiz_xy))
    t_horiz_xy = t_horiz_xy / n_th if n_th > 1e-9 else np.array([1.0, 0.0])

    for iu in range(n_u):
        # Centroid XY of the u-slice
        cx = float(np.mean(world[iu, :, 0]))
        cy = float(np.mean(world[iu, :, 1]))
        z_avg = float(np.mean(world[iu, :, 2]))
        axis_xy = stem_xy_at_z(z_avg)
        dx = cx - axis_xy[0]
        dy = cy - axis_xy[1]
        d_centroid = float(np.hypot(dx, dy))
        if d_centroid > 1e-6:
            # Outward unit vector = centroid - axis, normalised.
            ox, oy = dx / d_centroid, dy / d_centroid
        else:
            # Centroid coincides with stem axis — use tangent horizontal proj.
            ox, oy = float(t_horiz_xy[0]), float(t_horiz_xy[1])

        # Signed projections of each v-point onto outward direction, centred
        # on the stem axis. min projection is the "stem-ward" edge.
        proj = (world[iu, :, 0] - axis_xy[0]) * ox + (world[iu, :, 1] - axis_xy[1]) * oy
        p_min = float(np.min(proj))
        if p_min < r_min:
            shift = r_min - p_min
            world[iu, :, 0] += shift * ox
            world[iu, :, 1] += shift * oy

    back = (world.reshape(-1, 3) - collar) @ R
    return back.reshape(cps.shape)


def _make_stem_xy_at_z(organ):
    """Return ``stem_xy(z_world) -> (x, y)`` interpolator or ``None``.

    Linearly interpolates the parent stem's node XY through the nodes'
    z-range; out-of-range z snaps to the nearest endpoint. Falls back to
    ``None`` when no parent or < 2 nodes — ``_push_cps_outside_stem`` then
    becomes a no-op.
    """
    try:
        parent = organ.getParent()
    except Exception:
        return None
    if parent is None:
        return None
    try:
        nodes = parent.getNodes()
    except Exception:
        return None
    if len(nodes) < 2:
        return None
    xs = np.array([n.x for n in nodes], dtype=np.float64)
    ys = np.array([n.y for n in nodes], dtype=np.float64)
    zs = np.array([n.z for n in nodes], dtype=np.float64)
    # Sort by z so np.interp works (stem nodes are typically already sorted).
    order = np.argsort(zs)
    zs_s, xs_s, ys_s = zs[order], xs[order], ys[order]

    def stem_xy(z_world):
        return (
            float(np.interp(float(z_world), zs_s, xs_s)),
            float(np.interp(float(z_world), zs_s, ys_s)),
        )
    return stem_xy


def _capsule_chain_sdf_batch(points, capsules_arr):
    """Vectorised SDF + outward gradient for points vs a capsule chain.

    ``points``: ``(P, 3)`` world-frame query positions.
    ``capsules_arr``: ``(C, 7)`` packed ``[p0_xyz, p1_xyz, r]`` rows.

    Returns ``(sdf, grad)`` with shapes ``(P,)`` and ``(P, 3)``. ``sdf`` is
    the minimum signed distance over all capsules (negative inside);
    ``grad`` is the unit outward direction from the closest capsule's
    surface. Used by ``_relax_cps_against_obstacles`` to push penetrating
    CPs outward along ``-grad`` (gradient already points away from the
    capsule axis, so the push direction *is* ``+grad``).
    """
    P = np.asarray(points, dtype=np.float64)
    if P.size == 0 or capsules_arr.size == 0:
        n = P.shape[0] if P.ndim == 2 else 0
        return (np.full(n, np.inf, dtype=np.float64),
                np.zeros((n, 3), dtype=np.float64))
    P0 = capsules_arr[:, 0:3]                              # (C, 3)
    P1 = capsules_arr[:, 3:6]                              # (C, 3)
    R = capsules_arr[:, 6]                                 # (C,)
    D = P1 - P0                                            # (C, 3)
    L2 = np.einsum('cd,cd->c', D, D) + 1e-18               # (C,)
    diff = P[:, None, :] - P0[None, :, :]                  # (P, C, 3)
    t = np.einsum('pcd,cd->pc', diff, D) / L2[None, :]     # (P, C)
    np.clip(t, 0.0, 1.0, out=t)
    closest = P0[None, :, :] + t[:, :, None] * D[None, :, :]  # (P, C, 3)
    v = P[:, None, :] - closest                            # (P, C, 3)
    dist = np.linalg.norm(v, axis=2)                       # (P, C)
    sdf = dist - R[None, :]                                # (P, C)
    idx = np.argmin(sdf, axis=1)                           # (P,)
    rows = np.arange(P.shape[0])
    sdf_min = sdf[rows, idx]                               # (P,)
    v_min = v[rows, idx]                                   # (P, 3)
    d_min = dist[rows, idx]                                # (P,)
    safe = d_min > 1e-9
    grad = np.zeros_like(v_min)
    grad[safe] = v_min[safe] / d_min[safe, None]
    # Fallback gradient for query points that sit on a capsule axis: pick
    # the world horizontal that is most perpendicular to the closest-
    # capsule's axis. Cheap and avoids NaNs without per-row branches.
    if (~safe).any():
        D_min = D[idx[~safe]]
        L_min = np.linalg.norm(D_min, axis=1, keepdims=True) + 1e-9
        D_unit = D_min / L_min
        # take cross with z-up; if degenerate, fall back to x-axis
        up = np.array([0.0, 0.0, 1.0])
        perp = np.cross(D_unit, up[None, :])
        n_perp = np.linalg.norm(perp, axis=1, keepdims=True)
        weak = (n_perp[:, 0] < 1e-6)
        perp[weak] = np.array([1.0, 0.0, 0.0])
        n_perp = np.linalg.norm(perp, axis=1, keepdims=True) + 1e-9
        grad[~safe] = perp / n_perp
    return sdf_min, grad


def _local_to_world_basis(tangent_world):
    """Return ``(R, ok)`` where world = R @ local + collar.

    Mirrors the basis used by ``_clamp_cps_above_ground`` /
    ``_push_cps_outside_stem`` / ``nurbs_blade.from_local_frame`` so all
    CP transforms agree. ``R`` columns are ``(x_local_w, y_local_w, t)``.
    Returns ``ok=False`` when the tangent is degenerate.
    """
    t = np.asarray(tangent_world, dtype=np.float64).reshape(3)
    n_t = float(np.linalg.norm(t))
    if n_t < 1e-12:
        return np.eye(3), False
    t = t / n_t
    up = np.array([0.0, 0.0, 1.0])
    x_lw = np.cross(t, up)
    x_len = float(np.linalg.norm(x_lw))
    if x_len < 1e-6:
        alt = (np.array([1.0, 0.0, 0.0]) if abs(t[0]) < 0.9
               else np.array([0.0, 1.0, 0.0]))
        x_lw = np.cross(t, alt)
        x_len = float(np.linalg.norm(x_lw))
    x_lw = x_lw / max(x_len, 1e-12)
    y_lw = np.cross(t, x_lw)
    y_lw = y_lw / max(float(np.linalg.norm(y_lw)), 1e-12)
    R = np.column_stack([x_lw, y_lw, t])
    return R, True


def _stem_capsule_chain(stem_organ_dict, margin_cm=0.0):
    """Build capsule list ``[(p0, p1, r), ...]`` from a stem organ dict.

    Each segment of the resampled stem skeleton becomes one capsule with
    radius = average half-width of its endpoints (widths are diameters)
    plus an optional ``margin_cm`` buffer. Tassel segments are typically
    excluded by the caller; this helper does no filtering itself.
    """
    skel = np.asarray(stem_organ_dict.get('skeleton'), dtype=np.float64)
    wid = np.asarray(stem_organ_dict.get('widths'), dtype=np.float64)
    if skel.ndim != 2 or skel.shape[0] < 2 or wid.shape[0] != skel.shape[0]:
        return []
    out = []
    for i in range(skel.shape[0] - 1):
        r = 0.5 * 0.5 * (wid[i] + wid[i + 1]) + float(margin_cm)
        out.append((skel[i].copy(), skel[i + 1].copy(), float(max(r, 0.01))))
    return out


def _leaf_capsule_chain(cps_local, collar_pos_world, tangent_world,
                        margin_cm=0.0):
    """Build capsule list approximating a leaf's swept volume in world frame.

    Uses the midrib column (``v = n_v // 2``) as the capsule chain centre-
    line, with capsule radius = mean of the two leaf-edge offsets at that
    u-row (so a pinched tip stays narrow, a wide mid-blade gets a thicker
    capsule). The blade is approximated as a circular cylinder around its
    midrib — overestimates collision envelope perpendicular to the blade
    plane but lets a single SDF query handle leaf-leaf push without
    per-orientation bookkeeping.
    """
    cps = np.asarray(cps_local, dtype=np.float64)
    if cps.ndim != 3 or cps.shape[0] < 2 or cps.shape[1] < 2:
        return []
    R, ok = _local_to_world_basis(tangent_world)
    if not ok:
        return []
    collar = np.asarray(collar_pos_world, dtype=np.float64).reshape(3)
    n_u, n_v = cps.shape[0], cps.shape[1]
    mid_v = n_v // 2
    midrib_l = cps[:, mid_v, :]                           # (n_u, 3) leaf-local
    midrib_w = midrib_l @ R.T + collar                    # (n_u, 3) world
    # Half-width per u-row: average distance from midrib to v=0 and v=-1.
    edge0 = cps[:, 0, :]
    edge1 = cps[:, -1, :]
    hw = 0.5 * (np.linalg.norm(edge0 - midrib_l, axis=1)
                + np.linalg.norm(edge1 - midrib_l, axis=1))   # (n_u,)
    out = []
    for i in range(n_u - 1):
        r = 0.5 * (hw[i] + hw[i + 1]) + float(margin_cm)
        out.append((midrib_w[i].copy(), midrib_w[i + 1].copy(),
                    float(max(r, 0.05))))
    return out


def _relax_cps_against_obstacles(cps_local, current_skel,
                                 collar_pos_world, tangent_world,
                                 capsules,
                                 margin_cm=0.3, n_iter=5,
                                 lambda_smooth=0.3):
    """SDF-driven relaxation of a NURBS CP grid against capsule obstacles.

    For each of ``n_iter`` passes:
      1. project leaf-local CPs to world via the same basis the lofter
         uses;
      2. query ``_capsule_chain_sdf_batch`` for signed distance + outward
         gradient at every CP;
      3. push penetrating CPs (sdf < 0) along ``+grad`` by
         ``|sdf| + small_overshoot``;
      4. apply a 5-point Laplacian smoothing pass over the ``(u, v)`` CP
         lattice (skipping the basal row at ``u=0`` to keep the collar
         attachment intact);
      5. early-exit if the maximum penetration depth is < 1e-3 cm.

    The world-frame skeleton ``current_skel`` (used for DART segment-ID
    mapping) gets a per-node radial push against the same capsule field
    so the segment map tracks the rendered surface.

    Capsules whose axes coincide with this leaf's own midrib are filtered
    out of ``capsules`` by the caller — this helper assumes everything in
    the list is a real obstacle.
    """
    cps = np.asarray(cps_local, dtype=np.float64).copy()
    if cps.size == 0 or not capsules:
        return cps, np.asarray(current_skel, dtype=np.float64).copy()
    R, ok = _local_to_world_basis(tangent_world)
    if not ok:
        return cps, np.asarray(current_skel, dtype=np.float64).copy()
    collar = np.asarray(collar_pos_world, dtype=np.float64).reshape(3)

    cap_arr = np.asarray(
        [[*p0, *p1, float(r) + float(margin_cm)] for (p0, p1, r) in capsules],
        dtype=np.float64,
    )

    n_u, n_v, _ = cps.shape
    flat = cps.reshape(-1, 3)
    world = (flat @ R.T + collar).reshape(n_u, n_v, 3)

    for _it in range(int(n_iter)):
        sdf, grad = _capsule_chain_sdf_batch(
            world.reshape(-1, 3), cap_arr,
        )
        sdf = sdf.reshape(n_u, n_v)
        grad = grad.reshape(n_u, n_v, 3)
        # Pin the basal CP row (u=0) — it's the collar and must stay
        # glued to the parent stem. Without this, a young blade whose
        # collar sits ON the stem capsule (sdf<0 by stem_radius+margin)
        # would have its base pushed radially outward by ~r_stem each
        # pass, which flips a near-vertical young leaf back to a mature-
        # looking splay and breaks the whorl posture.
        sdf[0, :] = 0.0
        max_pen = float(-sdf.min()) if sdf.size else 0.0
        if max_pen < 1e-3:
            break
        # Push: only penetrating CPs move; small overshoot avoids
        # infinite oscillation when a CP sits exactly on an obstacle
        # surface that the next iteration's smoothing might tap back in.
        pen_depth = np.where(sdf < 0.0, -sdf + 5e-4, 0.0)
        world = world + grad * pen_depth[..., None]

        # Laplacian smoothing across the blade width (v direction only).
        # Smoothing along u (the midrib) was found to bleed midrib arc
        # length away by ~3-4% per iteration since adjacent u-rows aren't
        # in general coplanar — averaging them straightens the midrib and
        # shortens the blade. v-smoothing across the width is safe: it
        # keeps the cross-section taut after a per-CP push without
        # affecting blade length.
        if n_v >= 3:
            smoothed = world.copy()
            smoothed[:, 1:-1, :] = world[:, 1:-1, :] + lambda_smooth * 0.5 * (
                world[:, :-2, :] + world[:, 2:, :]
                - 2.0 * world[:, 1:-1, :]
            )
            world = smoothed

    flat_back = (world.reshape(-1, 3) - collar) @ R
    cps_out = flat_back.reshape(cps.shape)

    skel = np.asarray(current_skel, dtype=np.float64).copy()
    if skel.ndim == 2 and skel.shape[0] >= 1 and skel.shape[1] == 3:
        for _it in range(int(n_iter)):
            sdf, grad = _capsule_chain_sdf_batch(skel, cap_arr)
            # Pin skel[0] (collar) for the same reason the basal CP row
            # is pinned: the collar physically attaches to the stem and
            # may sit on the stem surface (sdf<0). Pushing it radially
            # would tilt the leaf away from the whorl posture set by the
            # Gap-2 maturity-coupled theta. Internal nodes still relax.
            sdf[0] = 0.0
            max_pen = float(-sdf.min()) if sdf.size else 0.0
            if max_pen < 1e-3:
                break
            pen = np.where(sdf < 0.0, -sdf + 5e-4, 0.0)
            skel = skel + grad * pen[:, None]

    return cps_out, skel


def _apply_two_segment_bend_cps_local(cps_local, theta_basal, theta_distal,
                                      s_bend):
    """Apply the senescence two-segment bend to a leaf-local NURBS CP grid.

    ``cps_local`` has shape ``(N_U, N_V, 3)`` where the u-axis runs along the
    midrib (local +z), the v-axis runs across the blade (local +x), and local
    +y is perpendicular to both (roughly the blade normal). The rotation axis
    is local +x, which `from_local_frame` maps to a horizontal axis in world
    — the same conceptual droop axis used for the world-frame skeleton bend.

    Negative basal rotation pitches the tip toward local -y (world-down for
    typical leaf tangents); positive distal counter-rotation about the hinge
    row lifts the distal portion back toward local +y (R2 hook).
    """
    cps = np.asarray(cps_local, dtype=np.float64).copy()
    if cps.size == 0 or (abs(theta_basal) < 1e-9 and abs(theta_distal) < 1e-9):
        return cps
    axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    origin = np.zeros(3, dtype=np.float64)
    if abs(theta_basal) > 1e-9:
        flat = _rodrigues_rotate(cps.reshape(-1, 3), origin, axis, -theta_basal)
        cps = flat.reshape(cps.shape)
    if abs(theta_distal) < 1e-9:
        return cps
    mid_col = cps.shape[1] // 2
    midrib = cps[:, mid_col, :]
    diffs = np.diff(midrib, axis=0)
    seg = np.linalg.norm(diffs, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(cum[-1])
    if total < 1e-9:
        return cps
    target = total * float(np.clip(s_bend, 0.0, 1.0))
    hinge_idx = int(np.searchsorted(cum, target))
    hinge_idx = max(1, min(hinge_idx, cps.shape[0] - 2))
    hinge_pos = midrib[hinge_idx].copy()
    distal = cps[hinge_idx:, :, :]
    rotated = _rodrigues_rotate(
        distal.reshape(-1, 3), hinge_pos, axis, +theta_distal,
    ).reshape(distal.shape)
    cps[hinge_idx:, :, :] = rotated
    return cps


def _apply_two_segment_bend(skeleton, pivot, axis, theta_basal, theta_distal,
                            s_bend):
    """Apply the senescence two-segment bend to a skeleton in place-safe form.

    First rotates the full skeleton about ``pivot`` by ``theta_basal`` using
    ``axis`` (downward). Then rotates the distal portion — nodes at arc
    position ≥ ``s_bend`` — by ``-theta_distal`` about the hinge node using
    the same axis (upward relative to the basal tangent). Returns a new
    (N, 3) array. Degrades to a single-pivot rotation when ``theta_distal``
    is ~0 (i.e. at R1 and R4).
    """
    sk = np.asarray(skeleton, dtype=np.float64).copy()
    if len(sk) < 2 or (abs(theta_basal) < 1e-9 and abs(theta_distal) < 1e-9):
        return sk
    if abs(theta_basal) > 1e-9:
        sk = _rodrigues_rotate(sk, pivot, axis, theta_basal)
    if abs(theta_distal) < 1e-9:
        return sk
    diffs = np.diff(sk, axis=0)
    seg = np.linalg.norm(diffs, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(cum[-1])
    if total < 1e-9:
        return sk
    target = total * float(np.clip(s_bend, 0.0, 1.0))
    bend_idx = int(np.searchsorted(cum, target))
    bend_idx = max(1, min(bend_idx, len(sk) - 2))
    distal = sk[bend_idx:]
    sk[bend_idx:] = _rodrigues_rotate(distal, sk[bend_idx], axis, -theta_distal)
    return sk


def _rotation_axis_for_droop(reference_tangent):
    """Horizontal axis perpendicular to the horizontal projection of ``reference_tangent``.

    Rotating about this axis by a positive angle (right-hand rule) pitches
    the leaf tip downward without changing its azimuthal heading.
    """
    ref = np.asarray(reference_tangent, dtype=np.float64).reshape(3)
    ref_xy = np.array([ref[0], ref[1], 0.0])
    nxy = float(np.linalg.norm(ref_xy))
    if nxy < 1e-9:
        ref_xy = np.array([1.0, 0.0, 0.0])
    else:
        ref_xy = ref_xy / nxy
    return np.array([-ref_xy[1], ref_xy[0], 0.0])


def _rodrigues_rotate(points, pivot, axis, angle_rad):
    """Rigid-body rotation of ``points`` about ``pivot`` around unit ``axis``."""
    pts = np.asarray(points, dtype=np.float64)
    was_1d = pts.ndim == 1
    pts2 = pts.reshape(-1, 3)
    pivot = np.asarray(pivot, dtype=np.float64).reshape(3)
    k = np.asarray(axis, dtype=np.float64).reshape(3)
    kn = float(np.linalg.norm(k))
    if kn < 1e-12 or abs(angle_rad) < 1e-12:
        return pts.copy()
    k = k / kn
    v = pts2 - pivot
    cos_t = float(np.cos(angle_rad))
    sin_t = float(np.sin(angle_rad))
    dot_kv = v @ k
    cross_kv = np.cross(k, v)
    rotated = v * cos_t + cross_kv * sin_t + np.outer(dot_kv, k) * (1.0 - cos_t)
    out = rotated + pivot
    return out.reshape(3) if was_1d else out


def extract_organs_for_lofter(plant, min_stem_nodes=50, min_leaf_nodes=20,
                              skip_roots=True, deformation_stats=None,
                              stem_profile=None, name_prefix="",
                              fitted_params=None, species='maize',
                              young_morph='soft', leaf_fracture=None,
                              enable_senescent_split=False,
                              senescent_rho_threshold=0.50):
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
        fitted_params: Optional dict from load_fitted_params(). When provided,
            per-leaf gradient-fitted deformation CPs override the random sinusoidal
            model for dramatically better mesh realism.

    Returns:
        List of organ dicts with resampled skeletons
    """
    import plantbox as pb

    young_morph_profiles = {
        'off':  (None, None),
        # 'soft' aligned with the C++ kYoungFadeEnd=0.7 / kYoungExp=2.0 gate
        # so mature leaves (m >= 0.7) are bit-for-bit untouched. The original
        # preview profile used fade_end=1.0 which still morphs m=0.5 leaves
        # and visibly shrinks mature plants — avoid for production.
        'soft': (0.7, 2.0),
        # Preview profiles retained for _gen_young_theta_test.py compatibility.
        'soft-preview': (1.0, 2.0),
        'hard': (1.0, 1.0),
        'max':  (0.9, 0.7),
    }
    if young_morph not in young_morph_profiles:
        raise ValueError(
            f"young_morph={young_morph!r} not in {list(young_morph_profiles)}"
        )
    ym_fade_end, ym_exp = young_morph_profiles[young_morph]

    # Auto-load deformation stats if not provided
    if deformation_stats is None:
        deformation_stats = load_deformation_stats()

    # Auto-load stem profile if not provided
    if stem_profile is None:
        stem_profile = load_stem_profile()

    fracture_cfg = _resolve_fracture_cfg(leaf_fracture)
    fracture_rng = (np.random.default_rng(fracture_cfg["seed"])
                    if fracture_cfg is not None else None)

    organs = []
    organ_counter = 0
    leaf_position = 0

    # Accumulated thermal time drives the per-position senescence droop
    # applied below (Item 1 of PLAN_GEOMETRY_FIDELITY_2026-04-22). Builds
    # without getAccumulatedTT leave the droop disabled (falls back to 0.0).
    plant_tt = None
    if hasattr(plant, "getAccumulatedTT"):
        try:
            plant_tt = float(plant.getAccumulatedTT())
        except Exception:
            plant_tt = None

    # S3b.8 (2026-04-24): the young-stage render-time z-compression that used
    # to live here has been removed. Real structural geometry from the
    # plastochron-driven initiation (S3b.7) combined with the shrunk mature-
    # calibrated p.lb=2 cm stub and the internodalGrowth basal_zero_ranks
    # gate (S3b.8 C++) now produces anatomically correct V-stage plants:
    # ranks 1-4 stay pinned at basal_internode_cm=1.0 cm, ranks 5+ elongate
    # under FA kinetics. No shim needed. The dict is kept as an empty
    # placeholder so the leaf-pass consumer below stays unchanged without
    # adding a hasattr guard on every leaf (dict lookups return 0.0 via
    # the .get(gid, 0.0) default and short-circuit at the "if _z_shift !=
    # 0.0" check).
    stem_node_z_shift: dict[int, float] = {}

    # Stems with resampling
    for organ in plant.getOrgans(pb.stem):
        nodes = organ.getNodes()
        if len(nodes) < 2:
            continue

        stem_subtype = int(organ.getParameter("subType"))
        is_tassel_spike = stem_subtype == 20
        is_tassel_branch = stem_subtype == 21
        is_tassel = is_tassel_spike or is_tassel_branch

        skeleton = np.array([[n.x, n.y, n.z] for n in nodes])
        radius = organ.getParameter("a")
        widths = np.full(len(nodes), 2.0 * radius)

        # Preserve original node_ids before resampling (for DART mapping)
        node_ids = [int(nid) for nid in organ.getNodeIds()]

        # Resample for smooth tubes
        skeleton, widths = resample_skeleton_uniform(skeleton, widths, min_stem_nodes)

        # Apply measured stem radius profile (MaizeField3D taper) — main stem only.
        # The profile is calibrated from mature mainstem scans (thick base → thin top);
        # applying it to the tassel spike/branches would assign mainstem-scale radii
        # to 0.1–0.3 cm organs. Tassel keeps the simple 2*a uniform radius.
        if stem_profile is not None and not is_tassel:
            widths = apply_stem_profile(skeleton, widths, stem_profile)

        # Apply a linear base→tip taper on tassel organs so tips stay thin
        # (plan §10 approved silhouette: spike 0.5→0.07 cm diameter,
        # branch 0.22→0.03 cm diameter). Uses skeleton arc length to place
        # each node on [0, 1] and interpolates the taper factor.
        if is_tassel:
            n_skel = len(skeleton)
            if n_skel > 1:
                diffs = np.diff(skeleton, axis=0)
                seg_len = np.linalg.norm(diffs, axis=1)
                arc = np.concatenate([[0.0], np.cumsum(seg_len)])
                total = arc[-1]
                frac = arc / total if total > 1e-9 else np.zeros(n_skel)
                tip_ratio = 0.14
                taper = 1.0 + frac * (tip_ratio - 1.0)
                widths = widths * taper

        # Stem maturity scaling: young stems are thinner than mature ones.
        # The MaizeField3D profile gives mature-plant radii — scale down
        # proportionally to how much of the stem has grown. Formula kept
        # in sync with _stem_radius_at_collar_cm and _make_stem_radius_at_z_callable.
        # Tassel spike/branch use a higher floor (0.70) because they emerge
        # near-mature in real maize — the young-stage compression that is
        # correct for the vegetative main stem (floor 0.08) would otherwise
        # keep branch diameters below the degenerate-triangle filter
        # (0.001 cm²) for most of the grain-fill window and wipe their tubes
        # from the mesh.
        stem_length = np.sum(np.linalg.norm(np.diff(skeleton, axis=0), axis=1))
        lmax = organ.getParameter("lmax")
        stem_maturity_value = 1.0  # default for the young-compress check
        if lmax > 1.0:
            stem_maturity_value = min(stem_length / lmax, 1.0)
            maturity_floor = 0.70 if is_tassel else 0.08
            width_scale = max(maturity_floor, stem_maturity_value ** 0.8)
            widths *= width_scale

        # S3b.8: render-time young-stage z-compression removed. V-stage
        # structural geometry is correct after the plastochron-initiation
        # + shrunk-p.lb + internodalGrowth basal_zero_ranks gate chain, so
        # the mainstem skeleton goes through unshifted. FA-off scalar-burst
        # geometry (collar-coincident at p.lb) now honestly reports what
        # the scalar path produces rather than cosmetically hiding it.

        if is_tassel_spike:
            organ_name = f"{name_prefix}tassel_spike_{organ_counter}"
        elif is_tassel_branch:
            organ_name = f"{name_prefix}tassel_branch_{organ_counter}"
        else:
            organ_name = f"{name_prefix}stem_{organ_counter}"

        organs.append({
            "type": "stem",
            "part_type": "stem",
            "skeleton": skeleton,
            "widths": widths,
            "organ_id": organ_counter,
            "name": organ_name,
            "node_ids": node_ids,
        })
        organ_counter += 1

    # SDF-driven collision relaxation: builds an obstacle list of stem +
    # already-placed leaf capsules. Each new leaf's CPs are pushed out of
    # any obstacle they penetrate before being handed to the lofter, so
    # young blades wrap around the stem and mid-canopy leaves deflect
    # around their neighbours instead of clipping through them. Tassel
    # spike + branches are excluded — they live above the canopy and the
    # leaf relaxation never sees them. The list grows as we walk leaves
    # in CPlantBox emission order (acropetal for maize), so each new
    # leaf only sees obstacles that were placed earlier.
    # Stem capsules use zero structural margin — young leaves are
    # supposed to sit flush against the stem (sheath wrap pinches the
    # collar at gap < 0.2 cm) and any positive margin here detaches them
    # visibly, exposing the stem mesh between leaves. The relaxation
    # call below adds a per-pass ``margin_cm`` so we still push CPs
    # that would otherwise interpenetrate; that knob covers the buffer.
    collision_obstacles: list = []
    for _o in organs:
        if _o.get('type') != 'stem':
            continue
        if _o.get('part_type') in ('tassel_spike', 'tassel_branch'):
            continue
        collision_obstacles.extend(_stem_capsule_chain(_o, margin_cm=0.0))

    leaf_attachment_z_for_modulation = []
    for _leaf in plant.getOrgans(pb.leaf):
        try:
            if _leaf.getParameter("isPseudostem") == 1:
                continue
            _nodes = _leaf.getNodes()
            if len(_nodes) >= 2:
                leaf_attachment_z_for_modulation.append(float(_nodes[0].z))
        except Exception:
            continue
    leaf_attachment_z_for_modulation = sorted(leaf_attachment_z_for_modulation)

    # Leaves with resampling
    for organ in plant.getOrgans(pb.leaf):
        nodes = organ.getNodes()
        if len(nodes) < 2:
            continue

        lrp = organ.getLeafRandomParameter()
        width_blade = lrp.Width_blade

        # Detect sheaths (isPseudostem=1) — route to sheath mesher
        is_sheath = organ.getParameter("isPseudostem") == 1
        if is_sheath:
            skeleton = np.array([[n.x, n.y, n.z] for n in nodes])
            if np.sum(np.linalg.norm(np.diff(skeleton, axis=0), axis=1)) < 0.5:
                continue  # skip negligible sheaths

            # Sheath radii: must wrap AROUND the stem, so radius = stem_radius + gap.
            # Look up stem radius at this sheath's z-positions from the stem organ's
            # width profile (already resampled and possibly profile-scaled).
            stem_skel = None
            stem_widths = None
            for o in organs:
                if o['type'] == 'stem':
                    stem_skel = o['skeleton']
                    stem_widths = o['widths']
                    break

            n_skel = len(skeleton)
            # Sheath wraps around the stem with a visible offset that tapers
            # toward the collar. Reference geometry: base_gap ~0.4 cm,
            # collar_gap ~0.15 cm (from Step 0 PlantGL headless test).
            base_gap = 0.40   # cm, sheath stands off from stem at base
            collar_gap = 0.15  # cm, tighter wrap near collar/ligule
            gap_taper = np.linspace(base_gap, collar_gap, n_skel)
            if stem_skel is not None and stem_widths is not None:
                # Interpolate stem radius at each sheath z-position
                stem_z = stem_skel[:, 2]
                stem_radii = stem_widths / 2.0  # widths are diameters
                sheath_z = skeleton[:, 2]
                interp_stem_r = np.interp(sheath_z, stem_z, stem_radii,
                                          left=stem_radii[0], right=stem_radii[-1])
                radii = interp_stem_r + gap_taper
            else:
                # Fallback: use stem organ parameter
                stem_a = 0.5  # cm, conservative default
                for s in plant.getOrgans(pb.stem):
                    stem_a = s.getParameter("a")
                    break
                radii = np.full(n_skel, stem_a) + gap_taper

            node_ids = [int(nid) for nid in organ.getNodeIds()]
            skeleton, radii = resample_skeleton_uniform(skeleton, radii, min_stem_nodes)

            organs.append({
                "type": "sheath",
                "part_type": "sheath",
                "skeleton": skeleton,
                "widths": radii * 2.0,  # full diameter for consistency
                "radii": radii,
                "wrap_angle": np.radians(330),
                "overlap_angle": np.radians(30),
                "sheath_thickness": 0.04,
                "stem_skeleton": stem_skel,
                "organ_id": organ_counter,
                "name": f"{name_prefix}sheath_{organ_counter}",
                "node_ids": node_ids,
            })
            organ_counter += 1
            continue

        # Skip broken leaf subtypes
        if width_blade < 0.01:
            continue

        # Native 2D surface CP path (Phase C). When the leaf's LRP carries a
        # populated ``surface_cps`` grid (written by calibrate.py via the
        # local-frame library), we bypass the skeleton-resampling + sword-taper
        # + gutter-spline pipeline and let the lofter place the library grid
        # directly at the organ's current collar frame.
        # Prefer per-leaf maturity-blended CPs (C++ young-stage flat-template
        # blend in Leaf::getEffectiveSurfaceCPs). Fall back to the LRP's mature
        # grid on older builds without the binding.
        try:
            surface_cps_flat = list(organ.getEffectiveSurfaceCPs())
        except AttributeError:
            surface_cps_flat = list(getattr(lrp, 'surface_cps', []))
        if surface_cps_flat:
            n_u = int(lrp.surface_n_u)
            n_v = int(lrp.surface_n_v)
            if len(surface_cps_flat) != n_u * n_v:
                # Mismatched grid; fall through to legacy skeleton path.
                pass
            else:
                cps_local = np.empty((n_u, n_v, 3), dtype=np.float64)
                for k, p in enumerate(surface_cps_flat):
                    i_u, i_v = divmod(k, n_v)
                    cps_local[i_u, i_v] = (p.x, p.y, p.z)
                collar = organ.getNode(0)
                tangent = organ.getiHeading0()
                node_ids = [int(nid) for nid in organ.getNodeIds()]
                mature_length = float(lrp.lmax)
                current_skel = np.array([[n.x, n.y, n.z] for n in nodes])
                current_length = float(
                    np.sum(np.linalg.norm(np.diff(current_skel, axis=0), axis=1))
                )

                # Rank-1 seedling first leaf override: replace the MF3D
                # 24-cm mature sword with a small rounded ovate blade so
                # the V-stage render shows a Nielsen-reference V1 leaf #1
                # (rounded, broad, attached at a visible sheath collar)
                # rather than a long pointed adult leaf at z ≈ 0. The
                # sheath wrap (compound path) already provides the collar
                # geometry from the MF3D ``sheath_length_cm_median`` at
                # position 0 (≈ 4 cm). All other ranks are untouched.
                # Mature regression on the calibrated XML is preserved at
                # all higher ranks; rank 1 itself diverges from FA-off
                # baseline but the user's V-stage reference takes priority
                # over the MF3D mature snapshot at this position.
                if leaf_position == 0:
                    SEEDLING_LENGTH_CM = 5.5
                    SEEDLING_MAX_WIDTH_CM = 1.9
                    cps_local = _build_seedling_first_leaf_cps(
                        n_u, n_v,
                        length_cm=SEEDLING_LENGTH_CM,
                        max_width_cm=SEEDLING_MAX_WIDTH_CM,
                    )
                    # Override length tracking so the lofter does not
                    # rescale the freshly-built seedling shape (scale=1).
                    mature_length = SEEDLING_LENGTH_CM
                    current_length = SEEDLING_LENGTH_CM
                    # Shrink the world-frame skeleton along the leaf
                    # tangent so DART seg IDs and SDF capsules track the
                    # rendered surface. Nodes beyond the seedling cap are
                    # collapsed onto the tip.
                    if len(current_skel) >= 2:
                        deltas = np.diff(current_skel, axis=0)
                        seg_lens = np.linalg.norm(deltas, axis=1)
                        cumlen = np.concatenate(([0.0], np.cumsum(seg_lens)))
                        if cumlen[-1] > SEEDLING_LENGTH_CM and cumlen[-1] > 1e-9:
                            shrink = SEEDLING_LENGTH_CM / cumlen[-1]
                            current_skel = current_skel.copy()
                            for i_n in range(1, len(current_skel)):
                                current_skel[i_n] = (
                                    current_skel[0]
                                    + (current_skel[i_n] - current_skel[0]) * shrink
                                )

                # Pheno4D young-stage blending was prototyped here but
                # disabled 2026-04-19: Pheno4D NURBS fits capture real-
                # world canopy droop/whorl curl (p50 lateral excursion
                # = 45 % of arc length), whereas MF3D fits are nearly
                # planar. A linear cross-fade between the two regimes
                # produces mangled whorled young leaves even with the
                # arc-renormalisation step. The builder + library helpers
                # in ``canonical_library`` are kept for a future rebuild
                # from a curated / planarised subset. See
                # ``NATIVE_SURFACE_CPS_IMPLEMENTATION.md`` Known Gap #5
                # for the follow-up.

                # Compound sheath+blade path: look up per-position sheath
                # length from MaizeField3D references and pull stem radius
                # from the parent stem's Phase E.3 getRadiusAt API. The
                # simple subType schema is ``position = subType - 2``. If
                # either source is missing, the lofter falls back to the
                # blade-only NURBS surface.
                position = max(int(getattr(lrp, "subType", 2)) - 2, 0)
                refs = load_reference_profiles()
                vidal_sheaths = load_vidal_sheath_lengths()
                sheath_length_cm = None
                sheath_provenance = None
                # Prefer Vidal 2021 SupData1 cultivar-averaged sheath lengths
                # (vidal_per_rank_sheath_cm.json) over MF3D medians for maize.
                # Vidal rank index == leaf position (S0 = position 0).
                if vidal_sheaths is not None and position in vidal_sheaths:
                    sheath_length_cm = vidal_sheaths[position]
                    sheath_provenance = "vidal_per_rank"
                elif refs is not None and position in refs:
                    sheath_length_cm = refs[position]["sheath_length_cm_median"]
                    sheath_provenance = "mf3d_median"
                stem_radius_cm = _stem_radius_at_collar_cm(
                    organ, collar_z=collar.z, stem_profile=stem_profile,
                )
                collar_pos_np = np.array([collar.x, collar.y, collar.z])
                # Young-stage stem compression: if the parent mainstem node
                # was z-shifted during extraction, apply the same shift to
                # this leaf's collar + whole skeleton so blade+sheath track
                # the compressed stem seamlessly. The first node_id of a
                # leaf IS the global id of its parent-stem insertion node.
                _leaf_first_nid = node_ids[0] if node_ids else -1
                _z_shift = stem_node_z_shift.get(int(_leaf_first_nid), 0.0)
                if _z_shift != 0.0:
                    collar_pos_np = collar_pos_np.copy()
                    collar_pos_np[2] -= _z_shift
                    current_skel = current_skel.copy()
                    current_skel[:, 2] -= _z_shift
                parent_tangent_np = _parent_tangent_at_collar(organ, collar_pos_np)
                parent_stem_r_callable = _make_stem_radius_at_z_callable(
                    organ, collar_pos_np, parent_tangent_np, stem_profile,
                    node_heights_z=leaf_attachment_z_for_modulation,
                )
                sheath_cup_max_length_cm = None
                if sheath_length_cm is not None and stem_radius_cm > 0.0:
                    sheath_cup_max_length_cm = 2.5 * float(stem_radius_cm)
                    lower_nodes = [
                        float(z) for z in leaf_attachment_z_for_modulation
                        if float(z) < float(collar_pos_np[2]) - 1e-6
                    ]
                    if lower_nodes:
                        prev_gap = float(collar_pos_np[2]) - max(lower_nodes)
                        if prev_gap > 1e-6:
                            sheath_cup_max_length_cm = min(
                                sheath_cup_max_length_cm,
                                0.95 * prev_gap,
                            )

                # Muted procedural deformations layered over the data-driven
                # CP grid. The base shape already encodes the large-scale
                # blade curvature (gutter, droop, taper) so we only add
                # small-amplitude ruffle/twist/wave to break up the
                # smoothness that median aggregation introduces. Applied in
                # leaf-local frame inside the lofter via _apply_deformations.
                rng = np.random.RandomState(organ_counter * 37 + 7)
                wave_params = _leaf_wave_params(
                    current_length, rng,
                    position=leaf_position,
                    deformation_stats=deformation_stats,
                    species=species,
                )
                # Muting: the CP library already captures gross shape, so
                # scale down the wave/ruffle/twist amplitudes. Curl stays
                # near-full because asymmetric edge curl is never in the
                # NURBS data (3-point cross-sections are symmetric).
                # Senescent leaves (ρ>0) lerp out of the mute so the ribbon-
                # streamer flutter at R4 isn't neutered by the turgid-leaf
                # calibration. `_senescence_progress` is cheap and
                # non-destructive, so compute it here and reuse below.
                rho_senesce = _senescence_progress(
                    leaf_position, plant_tt, species=species,
                )
                mute = NURBS_WAVE_MUTE_BASELINE
                curl_mute = NURBS_CURL_MUTE_BASELINE
                wave_params["wave_normal_amp"] *= mute
                wave_params["twist_max"] *= mute
                wave_params["curl_amp"] *= curl_mute
                if "gutter_depths" in wave_params:
                    wave_params["gutter_depths"] = (
                        np.asarray(wave_params["gutter_depths"]) * mute
                    )

                # Maturity fraction for nurbs_blade._apply_deformations. The
                # lofter attenuates deformations by maturity**0.6; without
                # this hook it defaults to 1.0 and young blades inherit full
                # mature amplitude.
                nurbs_maturity = min(current_length / max(mature_length, 1.0), 1.0)

                # Young-stage posture (PLAN_YOUNG_LEAF_PHYSICS_2026-04-25 §Gap 2):
                # Maturity-coupled effective theta. The leaf insertion angle
                # smoothly opens from THETA_YOUNG (near-vertical, 8° off
                # stem axis) at m≤0.3 to the leaf's mature theta_rank at
                # m≥0.7 (smoothstep). This matches the biology — young
                # blades are essentially vertical inside the whorl, then
                # splay to their rank-determined posture as they mature —
                # and replaces the earlier two-regime collar gate which
                # pulled too sharply toward vertical and lost the Nielsen
                # splay for V3+ leaves.
                #
                # We rotate BOTH the collar tangent (used to place cps_local
                # in world space and to seed the SDF capsule chain) AND
                # the underlying world-frame skeleton (used for DART
                # segment IDs and leaf capsule generation), so SDF
                # collision against the stem capsule, and the rendered
                # NURBS surface, and the segment-mapping skeleton all
                # agree on where the blade actually sits.
                #
                # ym_fade_end/ym_exp survives as an opt-out ceiling:
                # young_morph='off' gives ym_fade_end=None which skips
                # the whole block, leaving mature regression bit-identical.
                THETA_YOUNG_FROM_STEM = math.radians(5.0)  # near-vertical = 5° off stem axis (was 8°, pushed for tighter whorl)
                # Window widened 2026-05-12 from [0.95, 0.99] → [0.80, 1.00]: the
                # 4 % window was C¹ continuous but invisible because FA growth
                # lands maturity at 0.92 one day and 1.00 the next, skipping the
                # interior entirely → leaf snapped from vertical to splayed in
                # one render frame (slot-3 world dx pop 5.16 → 31.78 cm at
                # day 57→58, see DIAG_MAIZE_LEAF_BOTTOM_HEAVY_2026-05-12).
                M_YOUNG_LO = 0.80   # below: fully young theta
                M_YOUNG_HI = 1.00   # above: fully mature theta_rank
                collar_tangent_out = np.array([tangent.x, tangent.y, tangent.z],
                                              dtype=np.float64)
                # Seedling first leaf: use the tropism-evolved skeleton
                # direction as the placement frame. ``organ.getiHeading0()``
                # returns the leaf's INITIAL insertion heading (theta_rank
                # ≈ 35° from stem), but by V3 the cplantbox skeleton has
                # gravitropically opened to ~60-65° splay. Using the saved
                # initial heading paints the seedling at the upright
                # insertion direction, contradicting the live skeleton and
                # the Nielsen reference where leaf #1 lies near-horizontal.
                if leaf_position == 0 and len(current_skel) >= 2:
                    # Target splay angle for the seedling first leaf:
                    # 60° from vertical (≈30° above horizontal) so the
                    # blade tip clears the soil line at z=0 yet still
                    # reads as a splayed not-upright leaf 1. Pure overall
                    # skeleton direction (~90°, fully horizontal) sinks
                    # the leaf below ground; pure initial heading (~35°)
                    # is the upright look the user wants to avoid. We
                    # take the cplantbox skeleton's azimuth and rebuild
                    # the tangent at the target splay so the leaf points
                    # in the same compass direction as the simulation
                    # placed it but at our chosen elevation.
                    SEEDLING_SPLAY_DEG = 60.0
                    skel_dir = current_skel[-1] - current_skel[0]
                    horiz = np.array([skel_dir[0], skel_dir[1], 0.0])
                    horiz_norm = float(np.linalg.norm(horiz))
                    if horiz_norm > 1e-6:
                        horiz_unit = horiz / horiz_norm
                        splay_rad = math.radians(SEEDLING_SPLAY_DEG)
                        sin_s = math.sin(splay_rad)
                        cos_s = math.cos(splay_rad)
                        collar_tangent_out = np.array([
                            horiz_unit[0] * sin_s,
                            horiz_unit[1] * sin_s,
                            cos_s,
                        ], dtype=np.float64)
                if ym_fade_end is not None and ym_exp is not None:
                    profile_alpha = max(
                        0.0,
                        1.0 - (nurbs_maturity / ym_fade_end) ** ym_exp,
                    )
                    n_c = float(np.linalg.norm(collar_tangent_out))
                    n_p = float(np.linalg.norm(parent_tangent_np))
                    # Skip the maturity-coupled theta morph for the V1
                    # seedling first leaf (rank 1). Gap 2 pulls young
                    # leaves toward THETA_YOUNG ≈ 8° off the stem axis to
                    # form the V-stage whorl, which is correct for ranks
                    # 2+ that are still rolled inside the older sheaths.
                    # Rank 1 is the first leaf to emerge — by V3 it has
                    # already opened to its natural splay, so forcing it
                    # near-vertical produces an unnatural "pencil at the
                    # base" look that contradicts the Nielsen reference.
                    # Lifted profile_alpha gate (2026-05-02): the original 'soft'
                    # profile clamped profile_alpha to 0 at m=0.7, blocking the
                    # whorl-posture morph for leaves with m in [0.7, 1.0]. With
                    # the new collar-emergence smoothstep [M_YOUNG_LO=0.95,
                    # M_YOUNG_HI=0.99], we want the morph to fire across all
                    # maturities so leaves stay vertical until collar-emergence.
                    # Smoothstep itself handles the fade — profile_alpha gate is
                    # redundant.
                    can_morph = (
                        n_c >= 1e-9 and n_p >= 1e-9
                        and leaf_position != 0
                    )
                    if can_morph:
                        leaf_unit = collar_tangent_out / n_c
                        stem_unit = parent_tangent_np / n_p
                        cos_now = float(np.clip(np.dot(leaf_unit, stem_unit), -1.0, 1.0))
                        theta_now = math.acos(cos_now)  # current leaf-stem angle [rad]

                        # smoothstep on m ∈ [M_YOUNG_LO, M_YOUNG_HI]; s=0 → young, s=1 → mature
                        t = max(0.0, min(1.0, (nurbs_maturity - M_YOUNG_LO)
                                                / (M_YOUNG_HI - M_YOUNG_LO)))
                        s = t * t * (3.0 - 2.0 * t)
                        effective_theta = (1.0 - s) * THETA_YOUNG_FROM_STEM + s * theta_now
                        # slerp(leaf_unit, stem_unit, alpha) sits at angle
                        # theta_now*(1-alpha) from the stem; solve for alpha.
                        alpha = max(0.0, min(1.0, 1.0 - effective_theta / max(theta_now, 1e-9)))

                        if alpha > 1e-4:
                            collar_tangent_new = _slerp_tangent(leaf_unit, stem_unit, alpha)
                            # Rotate world-frame skeleton about the collar by the
                            # same incremental rotation (leaf_unit → collar_tangent_new),
                            # so SDF capsules and DART seg IDs co-locate with the
                            # rendered blade.
                            cross = np.cross(leaf_unit, collar_tangent_new)
                            n_cross = float(np.linalg.norm(cross))
                            cos_step = float(np.clip(np.dot(leaf_unit, collar_tangent_new),
                                                     -1.0, 1.0))
                            step_angle = math.acos(cos_step)
                            if n_cross > 1e-9 and step_angle > 1e-6:
                                rot_axis = cross / n_cross
                                current_skel = _rodrigues_rotate(
                                    current_skel, collar_pos_np,
                                    rot_axis, step_angle,
                                )
                            collar_tangent_out = collar_tangent_new
                            # Damp the leaf-local y (droop axis) of the CP grid
                            # so the surface itself looks pre-droop in the whorl
                            # — same effect as the prior morph block. cps_local
                            # is in leaf-local frame: +z midrib, +x lateral, ±y droop.
                            cps_local = cps_local.copy()
                            cps_local[..., 1] *= (1.0 - alpha)

                # Senescence two-segment bend (Item 1, PLAN_GEOMETRY_FIDELITY_2026-04-22).
                # Four-part: (1) shrink the blade width (Item 6, ribbon/
                # streamer character); (2) bend the CP grid in leaf-local
                # frame — basal rotation about the collar, distal counter-
                # rotation about the hinge row; (3) boost wilt-like wave /
                # curl / twist amplitudes; (4) mirror the same bend on the
                # world-frame skeleton so DART segment IDs track the
                # rendered geometry. ``collar_tangent`` and
                # ``parent_tangent`` stay unrotated — the CP-local bend is
                # internal to the blade; the sheath wrap continues to hug
                # the upright stalk. (Earlier iterations applied an
                # arc_keep y-axis flatten to prevent a single-rotation
                # fishhook; the two-segment bend handles that geometrically
                # so arc_keep is gone — it was cutting the effective
                # rotation-arm length roughly in half.)
                if rho_senesce > 0.0:
                    theta_basal, theta_distal, s_bend = _senescence_bend_params(
                        rho_senesce,
                    )
                    width_scale = max(
                        SENESCENCE_WIDTH_FLOOR,
                        1.0 - SENESCENCE_WIDTH_SHRINK * rho_senesce,
                    )
                    length_scale = max(
                        SENESCENCE_LENGTH_FLOOR,
                        1.0 - SENESCENCE_LENGTH_SHRINK * rho_senesce,
                    )
                    cps_local = cps_local.copy()
                    cps_local[..., 0] *= width_scale   # lateral (blade width)
                    cps_local[..., 1] *= length_scale  # forward-arc component
                    cps_local[..., 2] *= length_scale  # midrib-along-tangent
                    # Arch flattening with sqrt curve: kicks in fast at
                    # moderate ρ to kill the J-hook (mid-pitch + retained
                    # arch) but leaves low-ρ leaves mostly natural.
                    arch_scale = max(
                        0.0,
                        1.0 - SENESCENCE_ARCH_FLATTEN * float(np.sqrt(rho_senesce)),
                    )
                    cps_local[..., 1] *= arch_scale
                    cps_local = _apply_two_segment_bend_cps_local(
                        cps_local, theta_basal, theta_distal, s_bend,
                    )
                    # Senescence rotation: pure downward pitch (insertion-
                    # angle reduction). MIDRIB_ROLL_DEG=0 so no flip; only
                    # the droop pitch rotates the (now-flattened) leaf.
                    roll_theta, droop_theta = _senescence_rotation_angles(
                        rho_senesce,
                    )
                    cps_local = _apply_senescence_rotations_cps_local(
                        cps_local, roll_theta, droop_theta,
                    )
                    # Ground plane clamp: keep all CPs at or above z=0 in world.
                    cps_local = _clamp_cps_above_ground(
                        cps_local, collar_pos_np, collar_tangent_out,
                    )
                    # Stem-radius nudge previously lived here for the
                    # senescent path only; the SDF-driven relaxation pass
                    # below now handles stem clipping for ALL leaves
                    # (young whorl + mid-canopy + senescent), so the
                    # senescence-specific push has been removed.
                    wilt_boost = 1.0 + SENESCENCE_WILT_BOOST * rho_senesce
                    freq_boost = 1.0 + SENESCENCE_FREQ_BOOST * rho_senesce
                    for _k in ('wave_normal_amp', 'wave_lateral_amp',
                               'curl_amp', 'edge_ruffle_amp', 'fold_amp'):
                        if _k in wave_params:
                            wave_params[_k] = wave_params[_k] * wilt_boost
                    # NURBS lofter reads these as cycles along the midrib —
                    # senescent blades crinkle at high spatial frequency.
                    wave_params['wave_normal_freq'] = float(
                        wave_params.get('wave_normal_freq', 3.5)
                    ) * freq_boost
                    wave_params['curl_freq'] = float(
                        wave_params.get('curl_freq', 2.0)
                    ) * freq_boost
                    droop_axis = _rotation_axis_for_droop(collar_tangent_out)
                    current_skel = _apply_two_segment_bend(
                        current_skel, collar_pos_np, droop_axis,
                        theta_basal, theta_distal, s_bend,
                    )
                    # Mirror the senescence rotation pair on the segment-mapping
                    # skeleton so DART segment IDs track the rotated geometry.
                    # Roll = around the leaf tangent at collar (= midrib
                    # direction in world). Droop = around the horizontal
                    # droop_axis already computed for the disabled bend.
                    if abs(roll_theta) > 1e-9:
                        tangent_world = np.asarray(
                            collar_tangent_out, dtype=np.float64,
                        )
                        tangent_world = tangent_world / max(
                            float(np.linalg.norm(tangent_world)), 1e-12,
                        )
                        current_skel = _rodrigues_rotate(
                            np.asarray(current_skel, dtype=np.float64),
                            collar_pos_np, tangent_world, roll_theta,
                        )
                    if abs(droop_theta) > 1e-9:
                        current_skel = _rodrigues_rotate(
                            np.asarray(current_skel, dtype=np.float64),
                            collar_pos_np, droop_axis, -droop_theta,
                        )
                    # Also clamp the segment-mapping skeleton to ground so DART
                    # segment IDs don't dangle below the soil either.
                    current_skel = np.asarray(current_skel, dtype=np.float64).copy()
                    current_skel[:, 2] = np.maximum(
                        current_skel[:, 2], SENESCENCE_GROUND_Z,
                    )
                    # Skeleton stem-radius nudge also subsumed by the SDF
                    # relaxation below (the helper pushes both CPs and
                    # the world-frame skeleton against the same capsule
                    # field, so DART segment IDs stay co-located with the
                    # rendered geometry).

                # SDF-driven obstacle relaxation: stem cylinder + already-
                # placed leaves act as capsule obstacles. The CP grid and
                # the world-frame skeleton are jointly pushed out of any
                # capsule they penetrate. For very young blades (m < 0.30)
                # the whole leaf is *physically* inside the whorl,
                # surrounded by older sheaths and the stem; pushing it
                # radially outward there would undo the upright Gap-2
                # posture and produce the splayed-V3 we're trying to
                # eliminate. So we skip relaxation in the in-whorl regime
                # and let the parametric posture stand. Mid-canopy and
                # mature blades (m ≥ 0.30) still relax against their
                # neighbours and stem capsule.
                relax_threshold = 0.30
                # Skip SDF relaxation for senescent leaves — the senescence
                # bend deliberately routes the leaf into / through the stem
                # cylinder envelope (R2 hook tucks the distal segment back
                # toward the stem, full droop drapes basal CPs against the
                # base of the stalk). Capsule-based push-out fights the
                # bend and ends up flattening the U-shape into a downward
                # ribbon.
                if (collision_obstacles and nurbs_maturity >= relax_threshold
                        and rho_senesce <= 0.0):
                    # margin shrinks toward the threshold so post-collar
                    # blades emerging out of the whorl still get a soft
                    # nudge rather than a hard push, then settle to the
                    # full mature margin once they're clearly above.
                    soft = max(
                        0.0,
                        min(1.0, (nurbs_maturity - relax_threshold) / 0.20),
                    )
                    margin = 0.05 * soft  # 0..0.05 cm as m goes 0.30→0.50
                    cps_local, current_skel = _relax_cps_against_obstacles(
                        cps_local, current_skel,
                        collar_pos_np, collar_tangent_out,
                        collision_obstacles,
                        margin_cm=margin, n_iter=5, lambda_smooth=0.3,
                    )
                collision_obstacles.extend(_leaf_capsule_chain(
                    cps_local, collar_pos_np, collar_tangent_out,
                    margin_cm=0.0,
                ))

                midrib = cps_local[:, n_v // 2, :]
                midrib_arc = float(np.sum(np.linalg.norm(np.diff(midrib, axis=0), axis=1)))
                is_normalized = midrib_arc < 2.0

                # §3.1 leaf fracture on the NURBS path. Truncating the
                # rigid (N_U, N_V, 3) CP grid in-place would break its
                # shape contract, so we shorten via ``current_length``
                # (the backend already scales CPs by
                # ``current_length/mature_length``) and clip the world-
                # frame skeleton/node_ids so segment mapping tracks the
                # shorter blade. This produces a uniformly-shortened
                # blade (tip preserved) rather than a true torn end —
                # approximate but visually consistent with fracture.
                break_fraction_nurbs = 1.0
                if fracture_rng is not None:
                    _skel_arr = np.array([[n.x, n.y, n.z]
                                          for n in organ.getNodes()])
                    _wid_stub = np.ones(len(_skel_arr))
                    _, _, break_fraction_nurbs = _apply_leaf_fracture(
                        _skel_arr, _wid_stub, leaf_position,
                        fracture_rng, fracture_cfg,
                    )
                    if break_fraction_nurbs < 1.0:
                        current_length = current_length * break_fraction_nurbs
                        # Truncate world-frame skeleton + node_ids to the
                        # surviving arc fraction so the segment map stays
                        # in sync with the mesh the NURBS backend emits.
                        _diffs = np.diff(current_skel, axis=0)
                        _seg = np.linalg.norm(_diffs, axis=1)
                        _cum = np.concatenate([[0.0], np.cumsum(_seg)])
                        _total = float(_cum[-1]) if _cum.size else 0.0
                        if _total > 1e-9:
                            _cut = break_fraction_nurbs * _total
                            _keep = int(np.searchsorted(_cum, _cut, side="right"))
                            _keep = max(2, min(_keep, len(current_skel)))
                            current_skel = current_skel[:_keep].copy()
                            node_ids = node_ids[:_keep]

                # Healthy/withered tag (learnings §3.2): reuse the already-
                # computed ``rho_senesce`` — no extra plant inspection.
                # Default-off keeps the OBJ group names and downstream
                # DART routing bit-identical to baseline captures.
                _is_senescent = (
                    enable_senescent_split
                    and rho_senesce > senescent_rho_threshold
                )
                _leaf_part_type = "blade_senescent" if _is_senescent else "blade"
                _leaf_name = (
                    f"{name_prefix}senescent_leaf_{organ_counter}"
                    if _is_senescent
                    else f"{name_prefix}leaf_{organ_counter}"
                )

                # Raised central midrib for NURBS path: lifts the v=0.5 CP
                # against the baked-in gutter and depresses v=0.25 / v=0.75
                # by half that to deepen the surrounding trough so the rib
                # stands out (see nurbs_blade._apply_deformations §1b).
                #
                # Anatomy targets (real maize):
                #   - rib width is ~15 % of local width near the sheath
                #     junction, narrowing to ~5 % at the tip
                #   - lower/older leaves and the ear leaf show prominent
                #     midribs; upper leaves (flag etc) are muted
                #   - bump height tapers along the leaf so the rib is
                #     wider AND taller at the base, fading toward the tip
                _midrib_amp_scale_nurbs_base = 0.20 if species == 'maize' else 0.0

                # Rank-dependent prominence (maize only). Position 0..N
                # along the stem; ear-leaf zone is around position 9-12 in
                # mature plants, lower leaves are 0-3, upper/flag are 13+.
                if species == 'maize' and _midrib_amp_scale_nurbs_base > 0:
                    _p = int(leaf_position)
                    if _p <= 3:
                        _rank_factor = 1.10   # lower leaves — prominent
                    elif _p <= 8:
                        _rank_factor = 0.85   # mid-canopy
                    elif _p <= 12:
                        _rank_factor = 1.25   # ear-leaf zone — peak
                    else:
                        _rank_factor = 0.65   # upper leaves — muted
                else:
                    _rank_factor = 1.0
                _midrib_amp_scale_nurbs = (
                    _midrib_amp_scale_nurbs_base * _rank_factor
                )

                _n_skel_n = cps_local.shape[0]
                if _midrib_amp_scale_nurbs > 0.0:
                    _w_max_local = float(np.max(np.linalg.norm(
                        cps_local[:, -1, :] - cps_local[:, cps_local.shape[1] // 2, :],
                        axis=1,
                    ))) * 2.0  # full width = 2 × edge offset
                    _w_max_local = max(_w_max_local, 0.5)
                    # Bump-height taper: 1.0 at base → 0.6 at tip
                    _arc = np.linspace(0.0, 1.0, _n_skel_n)
                    _amp_taper = 1.0 - 0.4 * _arc
                    # Basal ramp: blade skeleton starts at the collar, so
                    # arc=0 sits at the sheath boundary. Fade in from 0
                    # over the first 15% of arc (smoothstep) to keep the
                    # rib off the collar transition.
                    _basal_t = np.clip(_arc / 0.15, 0.0, 1.0)
                    _basal_ramp = _basal_t * _basal_t * (3.0 - 2.0 * _basal_t)
                    _midrib_amps_nurbs = (
                        _w_max_local * _midrib_amp_scale_nurbs
                        * _amp_taper * _basal_ramp
                    )
                    _midrib_amps_nurbs *= max(0.0, min(1.0, nurbs_maturity)) ** 0.6
                else:
                    _midrib_amps_nurbs = np.zeros(_n_skel_n)

                # Band-width taper: half-width in v ∈ [0,1] coordinates.
                # 0.075 at base ⇒ rib spans v ∈ [0.425, 0.575] (15 % of width)
                # 0.025 at tip  ⇒ rib spans v ∈ [0.475, 0.525] (5 % of width)
                # Resampled inside the lofter to match n_u_eval.
                if species == 'maize':
                    _midrib_band_v_per_u = np.linspace(0.075, 0.025,
                                                       _n_skel_n)
                else:
                    _midrib_band_v_per_u = np.full(_n_skel_n, 0.025)

                _entry = {
                    "type": "leaf",
                    "part_type": _leaf_part_type,
                    "organ_id": organ_counter,
                    "name": _leaf_name,
                    "node_ids": node_ids,
                    "use_nurbs_backend": True,
                    "surface_cps_local": cps_local,
                    "surface_cps_normalized": is_normalized,
                    "surface_n_u": n_u,
                    "surface_n_v": n_v,
                    "surface_deg_u": int(lrp.surface_deg_u),
                    "surface_deg_v": int(lrp.surface_deg_v),
                    "midrib_amps_cm": _midrib_amps_nurbs,
                    "midrib_half_width": 0.10,
                    "midrib_band_v_frac": _midrib_band_v_per_u,
                    # Pre-2026-04-25 the rounded_tip flag was set for
                    # leaf_position==0 to get a blunt tip envelope from
                    # nurbs_blade.py. With the seedling override below
                    # we build a fully rounded ogive ourselves; the
                    # rounded_tip path's edge-widening would re-inflate
                    # our deliberately-narrow tip rows back to a flat
                    # cutoff. Force False for the seedling so the
                    # standard last-row pinch fires and produces a
                    # smooth point at u=1.
                    "rounded_tip": False,
                    "collar_pos": collar_pos_np,
                    "collar_tangent": collar_tangent_out,
                    "parent_tangent": parent_tangent_np,
                    "mature_length": mature_length,
                    "current_length": current_length,
                    "maturity_fraction": nurbs_maturity,
                    "plant_tt_cd": plant_tt,
                    "skeleton": current_skel,
                    "stem_radius_cm": stem_radius_cm,
                    "sheath_length_cm": sheath_length_cm,
                    "sheath_cup_max_length_cm": sheath_cup_max_length_cm,
                    "sheath_provenance": sheath_provenance,
                    "parent_stem_radius_at_z_cm": parent_stem_r_callable,
                    "break_fraction": break_fraction_nurbs,
                    **wave_params,
                }
                if break_fraction_nurbs < 1.0:
                    _entry["check_h_top_invariant"] = False
                organs.append(_entry)
                organ_counter += 1
                leaf_position += 1
                continue

        skeleton = np.array([[n.x, n.y, n.z] for n in nodes])

        # Senescence two-segment bend (Item 1, PLAN_GEOMETRY_FIDELITY_2026-04-22).
        # Basal portion rotates downward about the collar by θ_basal(ρ);
        # distal portion counter-rotates upward about the hinge node by
        # θ_distal(ρ), which peaks at ρ=0.5 (R2 hook) and returns to zero
        # at R4. Length shrink (Item 6) truncates the skeleton to
        # length_scale·arc BEFORE the bend so the bend operates on the
        # already-shrunk leaf. Applied before ground clip / tip taper /
        # resampling so those steps see the bent geometry and the
        # ``z < 0`` trim fires on the truly drooping portion of the leaf.
        rho_senesce = _senescence_progress(
            leaf_position, plant_tt, species=species,
        )
        if rho_senesce > 0.0 and len(skeleton) >= 2:
            theta_basal, theta_distal, s_bend = _senescence_bend_params(
                rho_senesce,
            )
            length_scale = max(
                SENESCENCE_LENGTH_FLOOR,
                1.0 - SENESCENCE_LENGTH_SHRINK * rho_senesce,
            )
            if length_scale < 1.0 - 1e-6:
                # Truncate skeleton to length_scale × total arc, interpolating
                # on the last remaining segment. widths are (re)computed from
                # this skeleton's arc just below via the leafGeometry profile
                # so we don't need to truncate a widths array here.
                _diffs = np.diff(skeleton, axis=0)
                _seg = np.linalg.norm(_diffs, axis=1)
                _cum = np.concatenate([[0.0], np.cumsum(_seg)])
                _target = float(_cum[-1]) * length_scale
                _cut = int(np.searchsorted(_cum, _target))
                _cut = max(1, min(_cut, len(skeleton) - 1))
                _prev = _cum[_cut - 1]; _step = _cum[_cut] - _prev
                _alpha = 0.0 if _step < 1e-9 else (_target - _prev) / _step
                _interp_pt = skeleton[_cut - 1] + _alpha * (
                    skeleton[_cut] - skeleton[_cut - 1]
                )
                skeleton = np.vstack([skeleton[:_cut], _interp_pt[np.newaxis]])
            tip_dir = skeleton[-1] - skeleton[0]
            droop_axis = _rotation_axis_for_droop(tip_dir)
            skeleton = _apply_two_segment_bend(
                skeleton, skeleton[0], droop_axis,
                theta_basal, theta_distal, s_bend,
            )
            # Senescence rotation pair (legacy quad-ribbon path).
            roll_theta, droop_theta = _senescence_rotation_angles(rho_senesce)
            if abs(roll_theta) > 1e-9:
                tangent_world = tip_dir / max(
                    float(np.linalg.norm(tip_dir)), 1e-12,
                )
                skeleton = _rodrigues_rotate(
                    np.asarray(skeleton, dtype=np.float64),
                    skeleton[0], tangent_world, roll_theta,
                )
            if abs(droop_theta) > 1e-9:
                skeleton = _rodrigues_rotate(
                    np.asarray(skeleton, dtype=np.float64),
                    skeleton[0], droop_axis, -droop_theta,
                )
            # Clamp any dropped node to ground. The later z<0 trim handles
            # downstream resampled skeletons; this pre-clamp keeps the bend
            # step's output self-consistent for segment mapping.
            skeleton = np.asarray(skeleton, dtype=np.float64).copy()
            skeleton[:, 2] = np.maximum(skeleton[:, 2], SENESCENCE_GROUND_Z)

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
        # Width unfurls faster than length grows — a maize blade reaches
        # full width by the time it's ~70% of lmax, then keeps elongating.
        # Saturate unfurl at maturity=0.7 so late-expansion leaves stay at
        # full width (matches MF3D medians).
        lmax = max(lrp.lmax, 1.0)
        current_length = np.sum(np.linalg.norm(np.diff(skeleton, axis=0), axis=1))
        maturity = min(current_length / lmax, 1.0)
        unfurl = min(max((maturity / 0.7) ** 0.6, 0.08), 1.0)
        if maturity < 0.7:
            widths *= unfurl

        # Sword-shape ceiling for emerging leaves: maize blades emerge from
        # the whorl rolled tight, so peak width is bounded by length/aspect
        # until the leaf unfurls. MF3D mature aspect ~12-13:1, so this cap is
        # inactive for mature leaves and only narrows young/short blades.
        max_w = float(np.max(widths)) if len(widths) > 0 else 0.0
        seedling_aspect = 12.0
        length_cap = current_length / seedling_aspect
        if max_w > length_cap > 0:
            widths *= (length_cap / max_w)

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

        # Pointed tip taper: only needed when the incoming width profile does
        # not already taper at the tip. The peak-aligned MF3D leaf_geometry
        # already encodes a natural taper from 1.0 at peak down to ~0.05 × max
        # at the tip — applying this envelope on top double-counts and cost us
        # ~6% of rendered blade area. Gate on widths[-1] / max < 15%.
        n_skel = len(skeleton)
        if n_skel >= 4:
            diffs_tip = np.diff(skeleton, axis=0)
            seg_lens = np.linalg.norm(diffs_tip, axis=1)
            cum_len = np.concatenate([[0.0], np.cumsum(seg_lens)])
            total_len = cum_len[-1]
            w_max = float(np.max(widths)) if len(widths) else 0.0
            tip_already_tapered = (
                w_max > 1e-6 and widths[-1] / w_max < 0.15
            )
            if total_len > 1e-6 and not tip_already_tapered:
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
            species=species,
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

        # Senescence wilting boost + width shrink (Items 1 & 6,
        # PLAN_GEOMETRY_FIDELITY_2026-04-22). Senescent basal leaves
        # flutter / curl / ruffle / twist visibly more than mature turgid
        # ones, and the blade narrows into the ribbon/streamer silhouette.
        # ``rho_senesce`` was computed earlier in the legacy path before
        # skeleton manipulation.
        if rho_senesce > 0.0:
            _boost = 1.0 + SENESCENCE_WILT_BOOST * rho_senesce
            _freq = 1.0 + SENESCENCE_FREQ_BOOST * rho_senesce
            for k in ('wave_normal_amp', 'wave_lateral_amp', 'curl_amp',
                      'edge_ruffle_amp', 'fold_amp'):
                if k in wave_params:
                    wave_params[k] *= _boost
            for k in ('wave_normal_freq', 'wave_lateral_freq', 'curl_freq',
                      'edge_ruffle_freq', 'fold_freq'):
                if k in wave_params:
                    wave_params[k] = float(wave_params[k]) * _freq
            _width_scale = max(
                SENESCENCE_WIDTH_FLOOR,
                1.0 - SENESCENCE_WIDTH_SHRINK * rho_senesce,
            )
            widths = np.asarray(widths, dtype=np.float64) * _width_scale

        # Midrib gutter: concave U-shaped cross-section typical of maize leaves.
        # Depth scales with blade width — wider leaves have a deeper channel.
        # Values from MaizeField3D: ~0.3-0.8 cm for 4-6 cm wide blades.
        n_skel = len(skeleton)
        gutter_depth_scale = 0.28 if species == 'maize' else 0.06  # fraction of width
        gutter_depths = widths * gutter_depth_scale
        if maturity < 0.95:
            gutter_depths *= unfurl  # young leaves are flat

        # Raised central midrib (the visible rib running along the leaf
        # centerline). Combined with `gutter_depths`, gives the maize
        # cross-section: V-channel + raised ridge at the midline.
        # Amplitude scales with blade width: ~10% × W puts the rib visibly
        # above the surrounding gutter trough (gutter depth = 0.18 × W) yet
        # still below the leaf edges, matching the real maize cross-section.
        # Linear arc taper toward the tip mirrors the anatomy (rib thinner
        # distally).
        midrib_amp_scale = 0.20 if species == 'maize' else 0.0
        midrib_arc_taper_min = 0.6  # fraction of base amp retained at tip
        # The blade skeleton starts at the collar (top of the sheath), so
        # arc=0 is the sheath–blade boundary. A non-zero amplitude there
        # makes the rib bleed into the visible collar transition. Fade in
        # from 0 over the first MIDRIB_BASAL_ONSET of the arc (smoothstep)
        # so the rib only emerges once the blade is clearly above the
        # sheath collar.
        midrib_basal_onset = 0.15
        if midrib_amp_scale > 0.0 and n_skel > 0:
            arc_frac = np.linspace(0.0, 1.0, n_skel)
            taper = 1.0 - (1.0 - midrib_arc_taper_min) * arc_frac
            basal_t = np.clip(arc_frac / max(midrib_basal_onset, 1e-6), 0.0, 1.0)
            basal_ramp = basal_t * basal_t * (3.0 - 2.0 * basal_t)  # smoothstep
            midrib_amps_cm = widths * midrib_amp_scale * taper * basal_ramp
            if maturity < 0.95:
                midrib_amps_cm *= unfurl  # young leaves: smaller ridge
        else:
            midrib_amps_cm = np.zeros(n_skel)

        # Extract spline-based geometry features from CPlantBox LeafRandomParameter
        # (leafOOPCurvPhi/Kappa, leafAsymmetry, leafEdgeCurl, leafCrossSection).
        # These are empty lists if not set in the XML — the lofter treats empty as no-op.
        # Band-width taper (15 % at base → 5 % at tip), same as the NURBS
        # path. Quad-ribbon lofter resamples to its own n_cross-1 strips.
        if species == 'maize':
            _midrib_band_per_u_qr = np.linspace(0.0075, 0.0025, n_skel)
        else:
            _midrib_band_per_u_qr = np.full(n_skel, 0.025)
        spline_features = {
            'gutter_depths': gutter_depths,
            'midrib_amps_cm': midrib_amps_cm,
            'midrib_half_width': 0.01,
            'midrib_band_v_frac': _midrib_band_per_u_qr,
        }
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

        # Fitted deformation CPs from diff_lofter optimization (if available).
        # These override the random sinusoidal model with gradient-fitted splines.
        # Also includes extended feature CPs (OOP, asymmetry, edge_curl, cross_section).
        fitted_extras = {}
        if fitted_params is not None:
            leaf_key = str(leaf_position)

            # Format 1: per_leaf_avg_cps from fit_to_reference.py --with-deformations
            avg_cps = fitted_params.get('per_leaf_avg_cps', {})
            if leaf_key in avg_cps:
                fitted_extras['fitted_extended_cps'] = avg_cps[leaf_key]

            # Format 2: stage_results from multistage_optimizer (legacy)
            if not fitted_extras:
                stages = fitted_params.get('stage_results', [])
                if stages:
                    final_deforms = stages[-1].get('deform_params', {})
                    if leaf_key in final_deforms:
                        leaf_deforms = final_deforms[leaf_key]
                        cps = leaf_deforms.get('control_points', {})
                        if cps:
                            fitted_extras['fitted_deform_cps'] = cps
                        ext_cps = leaf_deforms.get('extended_cp', {})
                        if ext_cps:
                            fitted_extras['fitted_extended_cps'] = ext_cps

        break_fraction = 1.0
        if fracture_rng is not None:
            skeleton, widths, break_fraction = _apply_leaf_fracture(
                skeleton, widths, leaf_position, fracture_rng, fracture_cfg,
            )
            # Truncating the skeleton shortens node_ids — keep the first
            # N ids so downstream segment mapping doesn't point past the
            # new end-of-leaf.
            if break_fraction < 1.0:
                node_ids = node_ids[:len(skeleton)]

        # Healthy/withered tag (learnings §3.2). See note in the NURBS
        # branch above.
        _is_senescent = (
            enable_senescent_split
            and rho_senesce > senescent_rho_threshold
        )
        _leaf_part_type = "blade_senescent" if _is_senescent else "blade"
        _leaf_name = (
            f"{name_prefix}senescent_leaf_{organ_counter}"
            if _is_senescent
            else f"{name_prefix}leaf_{organ_counter}"
        )

        leaf_entry = {
            "type": "leaf",
            "part_type": _leaf_part_type,
            "skeleton": skeleton,
            "widths": widths,
            "organ_id": organ_counter,
            "name": _leaf_name,
            "node_ids": node_ids,
            "break_fraction": break_fraction,
            **wave_params,
            **spline_features,
            **fitted_extras,
        }
        if break_fraction < 1.0:
            leaf_entry["check_h_top_invariant"] = False
        organs.append(leaf_entry)
        organ_counter += 1
        leaf_position += 1

    # Collect leaf attachment Z heights for internode modulation.
    leaf_bases_z = sorted([o['skeleton'][0, 2] for o in organs if o['type'] == 'leaf'])

    # Highest tassel-spike base — used both for stem-trim extension and for
    # suppressing the lofter's bare-stub clip below the tassel insertion.
    tassel_base_z = max(
        (o['skeleton'][0, 2] for o in organs
         if o.get('name', '').startswith('tassel_spike_')),
        default=None,
    )

    # When a tassel is attached above the last leaf with a bare-stem gap,
    # translate the whole tassel (spike + branches) down so the spike base
    # sits at max_leaf_z + JOINT_PAD cm. Biologically the CPlantBox gap is
    # covered by the pseudostem sheath in a real plant — we don't render
    # that sheath, so pulling the tassel down to the last leaf avoids a
    # visible bare-stem stub.
    # spike base shifted 3 cm below last leaf so post-smooth mesh base
    # lands at the leaf (smoothing + cap-shrink eat ~3 cm off each tube end)
    JOINT_PAD = -3.0
    tassel_shift_z = 0.0
    if leaf_bases_z and tassel_base_z is not None:
        target_base_z = max(leaf_bases_z) + JOINT_PAD
        if tassel_base_z > target_base_z:
            tassel_shift_z = target_base_z - tassel_base_z  # negative
            for o in organs:
                if o.get('name', '').startswith(('tassel_spike_', 'tassel_branch_')):
                    o['skeleton'] = np.asarray(o['skeleton'], dtype=np.float64).copy()
                    o['skeleton'][:, 2] += tassel_shift_z
            tassel_base_z = target_base_z  # keep downstream logic consistent

    # Cosmetic stem trim removed (Plan B.3 / S4 of
    # PLAN_PEDUNCLE_EXUBERANCE_2026-04-27.md, 2026-04-27).  The
    # structural fix in StemRandomParameter::realize() (FA-aware ln_
    # sizing + tassel-peduncle gate) and Stem::simulate (per-rank Phase
    # IV cessation gate) now keep the mainstem skeleton at topmost-leaf
    # insertion z + ~2 cm by construction, so the lofter no longer needs
    # to mask apex overshoot here.  HI#4 closes via the C++ skeleton
    # itself, not via post-hoc geometry surgery.  See
    # project_peduncle_exuberance_root_cause memory for the diagnostic
    # and the commit history (per-rank cessation gate + ln basal_floor +
    # tassel-peduncle gate) for the underlying fix.


    # Pass leaf attachment heights to stems for internode modulation.
    if leaf_bases_z:
        for o in organs:
            if o['type'] == 'stem':
                o['node_heights_z'] = list(leaf_bases_z)

    # When a tassel is attached, tell the lofter's _clip_stem_above_top_leaf
    # not to trim the mainstem below the tassel insertion — otherwise the
    # stem amputates above the last leaf and leaves the tassel suspended
    # in air.  Applies only to the mainstem-style stem (non-tassel).
    if tassel_base_z is not None:
        for o in organs:
            if o['type'] == 'stem' and not o.get('name', '').startswith(
                    ('tassel_spike_', 'tassel_branch_')):
                o['no_clip_above_z'] = float(tassel_base_z) + 3.0

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
            "part_type": "root",
            "skeleton": skeleton,
            "widths": widths,
            "organ_id": organ_counter,
            "name": f"root_{organ_counter}",
            "node_ids": node_ids,
        })
        organ_counter += 1

    return organs
