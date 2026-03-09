"""
Growth-stage PROSPECT parameter tables — species-aware.

Single source of truth for PROSPECT biochemical parameters used by both
DART optical properties and CPlantBox photosynthesis (Cab → Vcmax).

Species selection is driven by config.get_species().

Per-leaf-position profiles from LOPS measurements (Eko, LOPS_maize_P1.xlsx)
provide Cab and N gradients along the canopy for maize at V6, V10, and R1.

Literature references:
  - Feret et al. (2008) PROSPECT-4 / PROSPECT-5
  - Berger et al. (2018) maize Cab ranges across growth stages
  - Giraud et al. (2023) wheat Cab/Car ranges
  - CPlantBox Vcmax conversion: Vcmax = (VcmaxrefChl1 * Cab + VcmaxrefChl2) * 1e-6 mol/m²/s
"""

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from .config import get_species, get_species_name

# ---------------------------------------------------------------------------
# Runtime overrides — set by PipelineRunner or dashboard before pipeline runs.
# _STAGES_OVERRIDE replaces the entire stages list for _get_stages().
# Each entry must have: day_range, label, Cab, Car, Cw, Cm, N, CBrown, anthocyanin.
# ---------------------------------------------------------------------------
_STAGES_OVERRIDE: list | None = None
_VCMAX_CHL1_OVERRIDE: float | None = None
_VCMAX_CHL2_OVERRIDE: float | None = None


def set_overrides(
    stages: list | None = None,
    vcmax_chl1: float | None = None,
    vcmax_chl2: float | None = None,
) -> None:
    """Set runtime PROSPECT overrides.

    Parameters
    ----------
    stages : list or None
        Full replacement for the species' stage list.  Each dict must have
        day_range (tuple), label (str), and the 7 PROSPECT keys.
        Pass None to clear and revert to built-in defaults.
    vcmax_chl1, vcmax_chl2 : float or None
        Override Vcmax-Chl linear model coefficients.
    """
    global _STAGES_OVERRIDE, _VCMAX_CHL1_OVERRIDE, _VCMAX_CHL2_OVERRIDE
    _STAGES_OVERRIDE = stages
    _VCMAX_CHL1_OVERRIDE = vcmax_chl1
    _VCMAX_CHL2_OVERRIDE = vcmax_chl2


def clear_overrides() -> None:
    """Clear all runtime overrides (revert to stage-based lookup)."""
    set_overrides(None, None, None)


# Growth-stage PROSPECT parameters per species.
# day_range is half-open: [start, end).
_PROSPECT_STAGES = {
    "maize": [
        {
            "day_range": (0, 15),
            "label": "seedling",
            "Cab": 30.0, "Car": 6.0, "Cw": 0.008, "Cm": 0.006,
            "N": 1.6, "CBrown": 0.0, "anthocyanin": 0.0,
        },
        {
            "day_range": (15, 35),
            "label": "early_vegetative",
            "Cab": 45.0, "Car": 8.0, "Cw": 0.010, "Cm": 0.008,
            "N": 1.5, "CBrown": 0.0, "anthocyanin": 0.0,
        },
        {
            "day_range": (35, 56),
            "label": "late_vegetative",
            "Cab": 55.0, "Car": 10.0, "Cw": 0.012, "Cm": 0.010,
            "N": 1.4, "CBrown": 0.0, "anthocyanin": 0.0,
        },
        {
            "day_range": (56, 9999),
            "label": "mature",
            "Cab": 50.0, "Car": 12.0, "Cw": 0.015, "Cm": 0.012,
            "N": 1.3, "CBrown": 0.1, "anthocyanin": 0.0,
        },
    ],
    "wheat": [
        {
            "day_range": (0, 20),
            "label": "seedling",
            "Cab": 25.0, "Car": 5.0, "Cw": 0.007, "Cm": 0.005,
            "N": 1.8, "CBrown": 0.0, "anthocyanin": 0.0,
        },
        {
            "day_range": (20, 50),
            "label": "tillering",
            "Cab": 38.0, "Car": 7.0, "Cw": 0.009, "Cm": 0.007,
            "N": 1.6, "CBrown": 0.0, "anthocyanin": 0.0,
        },
        {
            "day_range": (50, 80),
            "label": "stem_elongation",
            "Cab": 48.0, "Car": 9.0, "Cw": 0.011, "Cm": 0.009,
            "N": 1.5, "CBrown": 0.0, "anthocyanin": 0.0,
        },
        {
            "day_range": (80, 110),
            "label": "heading",
            "Cab": 45.0, "Car": 10.0, "Cw": 0.013, "Cm": 0.010,
            "N": 1.4, "CBrown": 0.0, "anthocyanin": 0.0,
        },
        {
            "day_range": (110, 9999),
            "label": "senescence",
            "Cab": 30.0, "Car": 12.0, "Cw": 0.015, "Cm": 0.014,
            "N": 1.3, "CBrown": 0.3, "anthocyanin": 0.0,
        },
    ],
}

# Backward compatibility: PROSPECT_STAGES still points to maize by default
PROSPECT_STAGES = _PROSPECT_STAGES["maize"]


def _get_stages() -> list:
    """Return PROSPECT stages list for the active species.

    If a runtime override is active (via set_overrides), returns the
    override list instead of the built-in defaults.
    """
    if _STAGES_OVERRIDE is not None:
        return _STAGES_OVERRIDE
    name = get_species_name()
    if name not in _PROSPECT_STAGES:
        raise ValueError(
            f"No PROSPECT stages for species '{name}'. "
            f"Available: {list(_PROSPECT_STAGES.keys())}"
        )
    return _PROSPECT_STAGES[name]


def vcmax25_from_cab(cab_ug_cm2: float) -> float:
    """Compute Vcmax at 25C [umol/m2/s] from chlorophyll content [ug/cm2].

    Uses species-specific linear model: Vcmax = VcmaxrefChl1 * Cab + VcmaxrefChl2.
    Respects runtime overrides from set_overrides().
    Maize (C4): Cab=55 -> Vcmax~39.4 umol/m2/s.
    Wheat (C3): Cab=48 -> Vcmax~69.8 umol/m2/s.
    """
    sp = get_species()
    chl1 = _VCMAX_CHL1_OVERRIDE if _VCMAX_CHL1_OVERRIDE is not None else sp["vcmax_chl1"]
    chl2 = _VCMAX_CHL2_OVERRIDE if _VCMAX_CHL2_OVERRIDE is not None else sp["vcmax_chl2"]
    return chl1 * cab_ug_cm2 + chl2


def get_prospect_params(day: float) -> dict:
    """Return PROSPECT parameter dict for a given simulation day.

    Returns keys: Cab, Car, Cw, Cm, N, CBrown, anthocyanin.
    Suitable for passing directly to pytools4dart's ``prospect=`` kwarg.

    Respects per-stage runtime overrides via set_overrides().
    """
    _KEYS = ("Cab", "Car", "Cw", "Cm", "N", "CBrown", "anthocyanin")
    stages = _get_stages()
    for stage in stages:
        lo, hi = stage["day_range"]
        if lo <= day < hi:
            return {k: stage[k] for k in _KEYS}
    # Fallback to last stage
    stage = stages[-1]
    return {k: stage[k] for k in _KEYS}


def get_stem_prospect_params(day: float) -> dict:
    """Return PROSPECT parameter dict for stem optical properties.

    For maize: reads Cab, N, CBrown from LOPS stem entry for the growth stage.
    For non-maize or missing data: returns base stage params with low Cab.

    Returns keys: Cab, Car, Cw, Cm, N, CBrown, anthocyanin.
    Suitable for passing to pytools4dart's ``prospect=`` kwarg.
    """
    base = get_prospect_params(day)
    stage = get_lops_stage(day)
    if stage is not None and "stem" in stage:
        stem = stage["stem"]
        return {
            "Cab": stem["Cab"],
            "Car": base["Car"],
            "Cw": base["Cw"],
            "Cm": base["Cm"],
            "N": stem["N"],
            "CBrown": stem["CBrown"],
            "anthocyanin": base["anthocyanin"],
        }
    # Fallback: low Cab stem (green but not as much as leaves)
    return {**base, "Cab": base["Cab"] * 0.5, "N": base["N"] + 0.3}


def get_chl_for_photosynthesis(day: float) -> float:
    """Return Cab in CPlantBox internal units (ug/cm2) for a given simulation day.

    CPlantBox's C++ ``Chl`` member is in ug/cm2 (default 55.0).
    The Vcmax formula is: Vcrefmax = (VcmaxrefChl1 * Chl + VcmaxrefChl2) * 1e-6 [mol/m2/s].

    NOTE (2026-02-20): Previously returned Cab * 1e-6 (g/cm2), but CPlantBox
    expects ug/cm2. The *1e-6 was a unit bug that reduced Vcmax from ~39.4
    to ~4.2 umol/m2/s (a 10x error).
    """
    params = get_prospect_params(day)
    return params["Cab"]


def log_consistency(day: float) -> None:
    """Print a verification line showing Cab -> Vcmax chain."""
    sp = get_species()
    params = get_prospect_params(day)
    cab = params["Cab"]
    vcmax = sp["vcmax_chl1"] * cab + sp["vcmax_chl2"]  # umol/m2/s
    stages = _get_stages()
    stage_label = "unknown"
    for stage in stages:
        lo, hi = stage["day_range"]
        if lo <= day < hi:
            stage_label = stage["label"]
            break
    species_name = get_species_name()
    print(f"  PROSPECT [{species_name}/{stage_label}] Cab={cab:.1f} ug/cm2"
          f" -> Vcmax_ref={vcmax:.1f} umol/m2/s ({sp['photo_type']})")


# ==========================================================================
# Per-leaf-position PROSPECT profiles (LOPS data)
# ==========================================================================
_LOPS_DATA_PATH = Path(__file__).parent / 'data' / 'lops_prospect_profiles.json'


@lru_cache(maxsize=1)
def _load_lops_profiles() -> dict:
    """Load and cache LOPS per-position PROSPECT profiles."""
    with open(_LOPS_DATA_PATH) as f:
        return json.load(f)


def get_lops_stage(day: float) -> dict | None:
    """Select LOPS stage dict by day range.  Returns None for non-maize."""
    if get_species_name() != "maize":
        return None
    data = _load_lops_profiles()
    for stage in data["stages"].values():
        lo, hi = stage["day_range"]
        if lo <= day < hi:
            return stage
    return None


def get_prospect_params_per_position(day: float, n_leaves: int) -> list[dict]:
    """Return list of n_leaves PROSPECT dicts with per-position Cab and N.

    For maize: interpolates LOPS Cab/N profiles to n_leaves positions.
    For non-maize or if LOPS data is unavailable: returns uniform dicts.

    Each dict has keys: Cab, Car, Cw, Cm, N, CBrown, anthocyanin.
    """
    base = get_prospect_params(day)  # uniform stage params (Cw, Cm, Car, ...)

    stage = get_lops_stage(day)
    if stage is None or n_leaves < 1:
        return [dict(base) for _ in range(max(n_leaves, 1))]

    lops_data = _load_lops_profiles()
    const = lops_data["constant_params"]

    # LOPS positions (1-based, bottom-to-top)
    positions = stage["positions"]
    lops_pos = np.array([p["position"] for p in positions], dtype=float)
    lops_cab = np.array([p["Cab"] for p in positions], dtype=float)
    lops_n = np.array([p["N"] for p in positions], dtype=float)

    # Normalize both to [0, 1]
    lops_frac = (lops_pos - lops_pos.min()) / max(lops_pos.max() - lops_pos.min(), 1.0)
    # CPlantBox positions: 0 = bottom, n_leaves-1 = top
    if n_leaves == 1:
        cpb_frac = np.array([0.5])
    else:
        cpb_frac = np.linspace(0.0, 1.0, n_leaves)

    # Interpolate Cab and N
    interp_cab = np.interp(cpb_frac, lops_frac, lops_cab)
    interp_n = np.interp(cpb_frac, lops_frac, lops_n)

    result = []
    for i in range(n_leaves):
        d = {
            "Cab": float(interp_cab[i]),
            "N": float(interp_n[i]),
            "Car": const["Car"],
            "Cw": base["Cw"],
            "Cm": base["Cm"],
            "CBrown": const["CBrown"],
            "anthocyanin": const["anthocyanin"],
        }
        result.append(d)
    return result


def get_chl_per_segment(day: float, plant) -> list[float]:
    """Build per-leaf-segment Chl array from LOPS profiles.

    Returns list of length n_leaf_segments, suitable for hm.Chl = [...].
    Each organ's segments get the Cab value of that organ's canopy position.

    CPlantBox's getMeanOrSegData() in Photosynthesis.h switches to per-segment
    mode when Chl.size() == seg_leaves_idx.size().
    """
    import plantbox as pb

    leaf_organs = [o for o in plant.getOrgans() if o.organType() == pb.OrganTypes.leaf]
    n_leaves = len(leaf_organs)

    if n_leaves == 0:
        return [get_chl_for_photosynthesis(day)]

    per_pos = get_prospect_params_per_position(day, n_leaves)

    chl_per_seg = []
    for i, organ in enumerate(leaf_organs):
        n_segs = len(organ.getSegments())
        cab = per_pos[i]["Cab"]
        chl_per_seg.extend([cab] * n_segs)

    return chl_per_seg


def log_lops_consistency(day: float, n_leaves: int) -> None:
    """Print per-position Cab -> Vcmax chain for LOPS profiles."""
    sp = get_species()
    per_pos = get_prospect_params_per_position(day, n_leaves)
    stage = get_lops_stage(day)
    stage_label = stage["label"] if stage else "unknown"

    print(f"  LOPS per-position [{stage_label}] ({n_leaves} leaves):")
    for i, params in enumerate(per_pos):
        cab = params["Cab"]
        n_val = params["N"]
        vcmax = sp["vcmax_chl1"] * cab + sp["vcmax_chl2"]
        pos_label = "bottom" if i == 0 else ("top" if i == n_leaves - 1 else f"pos {i+1}")
        print(f"    {pos_label}: Cab={cab:.1f}, N={n_val:.2f} -> Vcmax={vcmax:.1f} umol/m2/s")
