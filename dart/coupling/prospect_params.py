"""
Growth-stage PROSPECT parameter tables — species-aware.

Single source of truth for PROSPECT biochemical parameters used by both
DART optical properties and CPlantBox photosynthesis (Cab → Vcmax).

Species selection is driven by config.get_species().

Literature references:
  - Feret et al. (2008) PROSPECT-4 / PROSPECT-5
  - Berger et al. (2018) maize Cab ranges across growth stages
  - Giraud et al. (2023) wheat Cab/Car ranges
  - CPlantBox Vcmax conversion: Vcmax = (VcmaxrefChl1 * Cab + VcmaxrefChl2) * 1e-6 mol/m²/s
"""

from .config import get_species, get_species_name

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
    """Return PROSPECT stages list for the active species."""
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
    Maize (C4): Cab=55 -> Vcmax~39.4 umol/m2/s.
    Wheat (C3): Cab=48 -> Vcmax~69.8 umol/m2/s.
    """
    sp = get_species()
    return sp["vcmax_chl1"] * cab_ug_cm2 + sp["vcmax_chl2"]


def get_prospect_params(day: float) -> dict:
    """Return PROSPECT parameter dict for a given simulation day.

    Returns keys: Cab, Car, Cw, Cm, N, CBrown, anthocyanin.
    Suitable for passing directly to pytools4dart's ``prospect=`` kwarg.
    """
    stages = _get_stages()
    for stage in stages:
        lo, hi = stage["day_range"]
        if lo <= day < hi:
            return {
                k: stage[k]
                for k in ("Cab", "Car", "Cw", "Cm", "N", "CBrown", "anthocyanin")
            }
    # Fallback to last stage
    stage = stages[-1]
    return {
        k: stage[k]
        for k in ("Cab", "Car", "Cw", "Cm", "N", "CBrown", "anthocyanin")
    }


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
