"""
Growth-stage PROSPECT parameter table for maize (C4).

Single source of truth for PROSPECT biochemical parameters used by both
DART optical properties and CPlantBox photosynthesis (Cab → Vcmax).

Literature references:
  - Feret et al. (2008) PROSPECT-4 / PROSPECT-5
  - Berger et al. (2018) maize Cab ranges across growth stages
  - CPlantBox Vcmax conversion: Vcmax = (0.64 * Cab + 4.165) * 1e-6 mol/m²/s
"""

# Growth-stage PROSPECT parameters for maize C4.
# day_range is half-open: [start, end).
PROSPECT_STAGES = [
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
]

# CPlantBox Vcmax linear coefficients (from maize_C4_photosynthesis_parameters.json)
_VCMAX_CHL1 = 0.64     # slope
_VCMAX_CHL2 = 4.165    # intercept


def vcmax25_from_cab(cab_ug_cm2: float) -> float:
    """Compute Vcmax at 25°C [µmol/m²/s] from chlorophyll content [µg/cm²].

    Uses the same linear model as CPlantBox: Vcmax = VcmaxrefChl1 * Cab + VcmaxrefChl2.
    At Cab=55 → Vcmax≈39.4 µmol/m²/s.
    """
    return _VCMAX_CHL1 * cab_ug_cm2 + _VCMAX_CHL2


def get_prospect_params(day: float) -> dict:
    """Return PROSPECT parameter dict for a given simulation day.

    Returns keys: Cab, Car, Cw, Cm, N, CBrown, anthocyanin.
    Suitable for passing directly to pytools4dart's ``prospect=`` kwarg.
    """
    for stage in PROSPECT_STAGES:
        lo, hi = stage["day_range"]
        if lo <= day < hi:
            return {
                k: stage[k]
                for k in ("Cab", "Car", "Cw", "Cm", "N", "CBrown", "anthocyanin")
            }
    # Fallback to last stage
    stage = PROSPECT_STAGES[-1]
    return {
        k: stage[k]
        for k in ("Cab", "Car", "Cw", "Cm", "N", "CBrown", "anthocyanin")
    }


def get_chl_for_photosynthesis(day: float) -> float:
    """Return Cab in CPlantBox internal units (µg/cm²) for a given simulation day.

    CPlantBox's C++ ``Chl`` member is in µg/cm² (default 55.0).
    The Vcmax formula is: Vcrefmax = (VcmaxrefChl1 * Chl + VcmaxrefChl2) * 1e-6 [mol/m²/s].
    At Cab=55 µg/cm²: Vcmax_ref ≈ 39.4 µmol/m²/s.

    NOTE (2026-02-20): Previously returned Cab * 1e-6 (g/cm²), but CPlantBox
    expects µg/cm². The *1e-6 was a unit bug that reduced Vcmax from ~39.4
    to ~4.2 µmol/m²/s (a 10x error).
    """
    params = get_prospect_params(day)
    return params["Cab"]


def log_consistency(day: float) -> None:
    """Print a verification line showing Cab → Vcmax chain."""
    params = get_prospect_params(day)
    cab = params["Cab"]
    vcmax = (_VCMAX_CHL1 * cab + _VCMAX_CHL2)  # µmol/m²/s
    stage_label = "unknown"
    for stage in PROSPECT_STAGES:
        lo, hi = stage["day_range"]
        if lo <= day < hi:
            stage_label = stage["label"]
            break
    print(f"  PROSPECT [{stage_label}] Cab={cab:.1f} µg/cm²"
          f" → Vcmax_ref={vcmax:.1f} µmol/m²/s")
