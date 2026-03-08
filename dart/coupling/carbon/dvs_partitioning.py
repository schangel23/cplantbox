"""WOFOST/SUCROS DVS-based carbon partitioning as fallback/comparison baseline.

Development Versus Stage (DVS) lookup tables from WOFOST crop parameters
(github.com/ajwdewit/WOFOST_crop_parameters/maize.yaml).

Partitions gross primary production into organs using fixed DVS-dependent
fractions, with temperature-dependent maintenance respiration.

DVS can be computed from calendar days (legacy) or from accumulated Growing
Degree Days (GDD), which is the standard WOFOST approach and accounts for
actual temperature variation across the growing season.
"""

import numpy as np


# ---------------------------------------------------------------------------
# DVS lookup tables for maize (WOFOST/SUCROS)
# Format: [(DVS, fraction), ...]
# DVS: 0=emergence, 1=flowering, 2=maturity
# ---------------------------------------------------------------------------

# Fraction to roots
FRTB = [
    (0.00, 0.40), (0.10, 0.37), (0.20, 0.34), (0.30, 0.31),
    (0.40, 0.27), (0.50, 0.23), (0.60, 0.19), (0.70, 0.15),
    (0.80, 0.10), (0.90, 0.06), (1.00, 0.00), (2.00, 0.00),
]

# Fraction to leaves (of remaining after roots)
FLTB = [
    (0.00, 0.62), (0.33, 0.62), (0.88, 0.15), (0.95, 0.15),
    (1.10, 0.10), (1.20, 0.00), (2.00, 0.00),
]

# Fraction to stems (of remaining after roots)
FSTB = [
    (0.00, 0.38), (0.33, 0.38), (0.88, 0.85), (0.95, 0.85),
    (1.10, 0.40), (1.20, 0.00), (2.00, 0.00),
]

# Fraction to storage organs (of remaining after roots)
FOTB = [
    (0.00, 0.00), (0.95, 0.00), (1.10, 0.50), (1.20, 1.00),
    (2.00, 1.00),
]

# Maintenance respiration coefficients [kg CH2O / kg DW / d at Tref=25C]
RML = 0.030   # leaves
RMS = 0.015   # stems
RMR = 0.015   # roots
RMO = 0.010   # storage organs

# Conversion efficiencies [kg DW / kg CH2O]
CVL = 0.680   # leaves
CVS = 0.658   # stems
CVR = 0.690   # roots
CVO = 0.671   # storage organs

# Reference temperature for maintenance respiration
TREF = 25.0  # C


def _interp_table(table, dvs):
    """Linearly interpolate a DVS lookup table."""
    xs = [t[0] for t in table]
    ys = [t[1] for t in table]
    return float(np.interp(dvs, xs, ys))


def _dvs_from_day(day, emergence_day=7, flowering_day=65, maturity_day=120):
    """Estimate DVS from simulation day using calendar-day approximation (LEGACY).

    DVS 0 = emergence, 1 = flowering, 2 = maturity.
    Linear interpolation between milestones.

    Prefer dvs_from_gdd() when daily temperatures are available.
    """
    if day <= emergence_day:
        return 0.0
    elif day <= flowering_day:
        return (day - emergence_day) / (flowering_day - emergence_day)
    elif day <= maturity_day:
        return 1.0 + (day - flowering_day) / (maturity_day - flowering_day)
    else:
        return 2.0


# ---------------------------------------------------------------------------
# GDD-based DVS (standard WOFOST thermal time approach)
# ---------------------------------------------------------------------------

# Maize GDD thresholds (°C·day, base 8°C)
# Sources: WOFOST maize.yaml, Lizaso et al. 2018, Jones & Kiniry 1986
GDD_EMERGENCE = 60.0     # ~60 GDD from sowing to emergence
GDD_FLOWERING = 800.0    # ~800 GDD from sowing to flowering (tasseling)
GDD_MATURITY = 1600.0    # ~1600 GDD from sowing to physiological maturity
TBASE_MAIZE = 8.0        # base temperature (°C)
TMAX_MAIZE = 34.0        # upper cutoff — no extra GDD above this


def compute_gdd_day(T_mean_C, Tbase=TBASE_MAIZE, Tmax=TMAX_MAIZE):
    """Compute Growing Degree Days for a single day.

    Args:
        T_mean_C: Daily mean temperature (°C).
        Tbase: Base temperature below which no development occurs.
        Tmax: Upper cutoff (temperatures above this don't contribute extra).

    Returns:
        GDD contribution for the day (°C·day).
    """
    T_eff = min(T_mean_C, Tmax)
    return max(0.0, T_eff - Tbase)


def accumulate_gdd(daily_T_mean_list, Tbase=TBASE_MAIZE, Tmax=TMAX_MAIZE):
    """Accumulate GDD from a list of daily mean temperatures.

    Args:
        daily_T_mean_list: List/array of daily mean temperatures (°C),
            one per day from sowing (index 0 = sowing day).
        Tbase: Base temperature.
        Tmax: Upper cutoff.

    Returns:
        Cumulative GDD (°C·day).
    """
    return sum(compute_gdd_day(T, Tbase, Tmax) for T in daily_T_mean_list)


def dvs_from_gdd(gdd_accumulated, gdd_emergence=GDD_EMERGENCE,
                 gdd_flowering=GDD_FLOWERING, gdd_maturity=GDD_MATURITY):
    """Compute DVS from accumulated Growing Degree Days.

    DVS 0 = emergence, 1 = flowering, 2 = maturity.

    Args:
        gdd_accumulated: Total accumulated GDD from sowing (°C·day).
        gdd_emergence: GDD threshold for emergence.
        gdd_flowering: GDD threshold for flowering.
        gdd_maturity: GDD threshold for maturity.

    Returns:
        DVS value (0-2).
    """
    if gdd_accumulated <= gdd_emergence:
        return 0.0
    elif gdd_accumulated <= gdd_flowering:
        return (gdd_accumulated - gdd_emergence) / (gdd_flowering - gdd_emergence)
    elif gdd_accumulated <= gdd_maturity:
        return 1.0 + (gdd_accumulated - gdd_flowering) / (gdd_maturity - gdd_flowering)
    else:
        return 2.0


def dvs_for_day(sim_day, gdd_accumulated=None, **kwargs):
    """Unified DVS computation: uses GDD if available, falls back to calendar day.

    Args:
        sim_day: Simulation day (days since sowing).
        gdd_accumulated: Accumulated GDD from sowing (°C·day), or None.
        **kwargs: Passed to _dvs_from_day or dvs_from_gdd.

    Returns:
        DVS value (0-2).
    """
    if gdd_accumulated is not None:
        return dvs_from_gdd(gdd_accumulated, **kwargs)
    return _dvs_from_day(sim_day, **kwargs)


# ---------------------------------------------------------------------------
# Daily met CSV for real-weather GDD accumulation
# ---------------------------------------------------------------------------

# Default daily met file (Jülich 2024 growing season)
_DEFAULT_DAILY_MET = None  # lazy-loaded


def load_daily_met(csv_path=None):
    """Load daily met CSV with per-day temperature, humidity, and wind.

    Expected columns: sim_day, date, T_min_C, T_max_C, T_mean_C,
        RH_min, RH_max, RH_mean, wind_mean_kmh, wind_max_kmh.

    Returns:
        dict mapping sim_day (int) -> dict with all met fields.
    """
    import csv as csv_mod
    from pathlib import Path

    if csv_path is None:
        csv_path = Path(__file__).parent.parent / 'data' / 'juelich_2024_daily_met.csv'

    result = {}
    with open(csv_path) as f:
        reader = csv_mod.DictReader(f)
        for row in reader:
            day = int(row['sim_day'])
            result[day] = {
                'T_mean_C': float(row['T_mean_C']),
                'T_min_C': float(row['T_min_C']),
                'T_max_C': float(row['T_max_C']),
            }
            # Optional fields (backward compat with older CSV)
            if 'RH_min' in row:
                result[day]['RH_min'] = float(row['RH_min']) / 100.0
                result[day]['RH_max'] = float(row['RH_max']) / 100.0
            if 'wind_mean_kmh' in row:
                result[day]['wind_mean_ms'] = float(row['wind_mean_kmh']) / 3.6
                result[day]['wind_max_ms'] = float(row['wind_max_kmh']) / 3.6
    return result


def get_daily_met(sim_day=None, daily_met=None):
    """Get daily met data dict, auto-loading from default CSV if needed.

    Args:
        sim_day: If provided, return the met dict for this specific day.
            If None, return the full daily_met lookup.
        daily_met: Pre-loaded dict from load_daily_met(), or None to auto-load.

    Returns:
        If sim_day is None: full dict mapping day -> met dict, or None.
        If sim_day is given: met dict for that day, or None.
    """
    global _DEFAULT_DAILY_MET
    if daily_met is None:
        if _DEFAULT_DAILY_MET is None:
            try:
                _DEFAULT_DAILY_MET = load_daily_met()
            except FileNotFoundError:
                return None
        daily_met = _DEFAULT_DAILY_MET

    if sim_day is None:
        return daily_met
    return daily_met.get(sim_day)


def gdd_at_day(sim_day, daily_met=None):
    """Compute accumulated GDD up to sim_day using real daily temperatures.

    Falls back to None (triggering calendar-day DVS) if no met data available.

    Args:
        sim_day: Simulation day (days since sowing, 1-based).
        daily_met: dict from load_daily_met(), or None to auto-load.

    Returns:
        Accumulated GDD (float), or None if met data unavailable.
    """
    dm = get_daily_met(daily_met=daily_met)
    if dm is None:
        return None

    gdd = 0.0
    for d in range(1, sim_day + 1):
        day_data = dm.get(d)
        if day_data is None:
            return None  # incomplete data
        gdd += compute_gdd_day(day_data['T_mean_C'])
    return gdd


def partition_carbon_dvs(GPP_mmol_CO2_d, DVS_or_day, organ_biomass=None,
                         Tair_C=25.0, is_day=True, gdd_accumulated=None):
    """Partition gross primary production using WOFOST DVS tables.

    Args:
        GPP_mmol_CO2_d: Gross primary production [mmol CO2 d-1] (= An_total + Rd).
            If only net assimilation is available, pass it directly — Rd will be
            estimated internally.
        DVS_or_day: Development versus stage (0-2) or simulation day (if is_day=True).
        organ_biomass: dict with keys 'leaf', 'stem', 'root', 'storage' in [g DW].
            If None, uses rough estimates based on day.
        Tair_C: Air temperature [C] for Q10 scaling of maintenance respiration.
        is_day: If True, DVS_or_day is interpreted as simulation day.
        gdd_accumulated: Accumulated GDD from sowing (°C·day). If provided,
            DVS is computed from thermal time instead of calendar days.

    Returns:
        dict with same interface as QuasiSteadyPhloem.solve().
    """
    if gdd_accumulated is not None:
        DVS = dvs_from_gdd(gdd_accumulated)
    elif is_day:
        DVS = _dvs_from_day(DVS_or_day)
    else:
        DVS = DVS_or_day

    # DVS-dependent partitioning fractions
    FR_root = _interp_table(FRTB, DVS)
    FR_remain = 1.0 - FR_root
    FR_leaf = _interp_table(FLTB, DVS) * FR_remain
    FR_stem = _interp_table(FSTB, DVS) * FR_remain
    FR_storage = _interp_table(FOTB, DVS) * FR_remain

    # Normalize to ensure sum = 1.0
    FR_total = FR_root + FR_leaf + FR_stem + FR_storage
    if FR_total > 0:
        FR_root /= FR_total
        FR_leaf /= FR_total
        FR_stem /= FR_total
        FR_storage /= FR_total

    # Estimate organ biomass if not provided
    if organ_biomass is None:
        # Rough estimates for maize at given day [g DW]
        day = DVS_or_day if is_day else int(DVS * 60)
        total_dw = max(1.0, day * 2.5)  # ~2.5 g DW/d accumulation
        organ_biomass = {
            'leaf': total_dw * 0.35,
            'stem': total_dw * 0.35,
            'root': total_dw * 0.25,
            'storage': total_dw * 0.05,
        }

    # Q10 temperature correction for maintenance respiration
    Q10 = 2.0
    Tref = TREF
    temp_factor = Q10 ** ((Tair_C - Tref) / 10.0)

    # Maintenance respiration [mmol CO2 d-1]
    # Convert: kg CH2O / kg DW / d * g DW * 1000 mmol/mol / 30 g/mol CH2O * 1e-3
    # Simplified: rate [d-1] * biomass [g] * 1000/30 = mmol CH2O d-1
    # 1 mol CH2O = 1 mol CO2 (in respiration)
    ch2o_to_mmol = 1000.0 / 30.0  # g CH2O -> mmol CO2
    Rm_leaf = RML * organ_biomass['leaf'] * ch2o_to_mmol * temp_factor
    Rm_stem = RMS * organ_biomass['stem'] * ch2o_to_mmol * temp_factor
    Rm_root = RMR * organ_biomass['root'] * ch2o_to_mmol * temp_factor
    Rm_storage = RMO * organ_biomass.get('storage', 0) * ch2o_to_mmol * temp_factor
    Rm_total = Rm_leaf + Rm_stem + Rm_root + Rm_storage

    # Available for growth = GPP - Rm
    available = max(0.0, GPP_mmol_CO2_d - Rm_total)

    # Growth respiration + structural growth per organ
    # Growth = available * fraction * conversion_efficiency
    # Growth_resp = available * fraction * (1 - conversion_efficiency)
    Rg_leaf = available * FR_leaf * (1.0 - CVL)
    Rg_stem = available * FR_stem * (1.0 - CVS)
    Rg_root = available * FR_root * (1.0 - CVR)
    Rg_storage = available * FR_storage * (1.0 - CVO)
    Rg_total = Rg_leaf + Rg_stem + Rg_root + Rg_storage

    growth = available - Rg_total  # structural growth

    # Carbon balance
    total_out = Rm_total + Rg_total + growth
    balance_error = abs(GPP_mmol_CO2_d - total_out) / max(GPP_mmol_CO2_d, 1e-6)

    return {
        'Rm_total_mmol': Rm_total,
        'Rm_leaf': Rm_leaf,
        'Rm_stem': Rm_stem,
        'Rm_root': Rm_root,
        'Rm_storage': Rm_storage,
        'Rg_total_mmol': Rg_total,
        'FR_leaf': FR_leaf,
        'FR_stem': FR_stem,
        'FR_root': FR_root,
        'FR_storage': FR_storage,
        'root_resp_profile_mmol_d': np.array([Rm_root]),  # single layer
        'root_exud_mmol_d': np.array([0.0]),  # not modeled in DVS
        'root_dead_mmol_d': np.array([0.0]),
        'growth_mmol_d': growth,
        'carbon_balance_error': balance_error,
        'C_ST_mean': np.nan,  # not applicable
        'DVS': DVS,
        'partitioning_source': 'dvs_wofost',
    }
