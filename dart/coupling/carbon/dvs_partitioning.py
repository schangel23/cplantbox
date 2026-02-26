"""WOFOST/SUCROS DVS-based carbon partitioning as fallback/comparison baseline.

Development Versus Stage (DVS) lookup tables from WOFOST crop parameters
(github.com/ajwdewit/WOFOST_crop_parameters/maize.yaml).

Partitions gross primary production into organs using fixed DVS-dependent
fractions, with temperature-dependent maintenance respiration.
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
    """Estimate DVS from simulation day using thermal-time approximation.

    DVS 0 = emergence, 1 = flowering, 2 = maturity.
    Linear interpolation between milestones.
    """
    if day <= emergence_day:
        return 0.0
    elif day <= flowering_day:
        return (day - emergence_day) / (flowering_day - emergence_day)
    elif day <= maturity_day:
        return 1.0 + (day - flowering_day) / (maturity_day - flowering_day)
    else:
        return 2.0


def partition_carbon_dvs(GPP_mmol_CO2_d, DVS_or_day, organ_biomass=None,
                         Tair_C=25.0, is_day=True):
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

    Returns:
        dict with same interface as QuasiSteadyPhloem.solve().
    """
    DVS = _dvs_from_day(DVS_or_day) if is_day else DVS_or_day

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
