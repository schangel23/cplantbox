"""Unit conversions for CPlantBox -> AgroC interface.

Each function maps a CPlantBox output quantity to the unit convention
expected by AgroC Fortran modules (plants.f90, soilco2.f90).
"""

# Molar mass of CO2 [kg/mol]
MOLCO2 = 0.0440098

# Molar mass of sucrose [kg/mmol]
SUC_MOLAR_MASS_KG = 342.3e-6  # 342.3 g/mol = 342.3e-6 kg/mmol

# Carbon fraction of dry matter (WOFOST convention)
C_FRACTION_DM = 0.467


def mmol_co2_to_mol_co2_per_cm3(total_mmol, layer_vol_cm3):
    """Convert total mmol CO2/d to volumetric mol CO2/cm3/d.

    Maps to AgroC ``rnodert(ri)`` (plants.f90:1639) — root respiration
    distributed per soil layer volume.

    Args:
        total_mmol: total flux [mmol CO2/d] for one layer.
        layer_vol_cm3: layer volume [cm3].

    Returns:
        mol CO2 / cm3 / d
    """
    return total_mmol / 1000.0 / layer_vol_cm3


def mmol_suc_to_kg_c_per_cm3(total_mmol_suc, layer_vol_cm3):
    """Convert total mmol sucrose/d to kg C / cm3 / d.

    Maps to AgroC ``rnodexu(ri)`` (plants.f90:1577) — root exudation
    carbon per soil layer volume.

    Args:
        total_mmol_suc: total flux [mmol sucrose/d] for one layer.
        layer_vol_cm3: layer volume [cm3].

    Returns:
        kg C / cm3 / d
    """
    kg_suc = total_mmol_suc * SUC_MOLAR_MASS_KG
    kg_c = kg_suc * C_FRACTION_DM
    return kg_c / layer_vol_cm3


def mmol_co2_to_mol_co2_per_cm2(total_mmol, ground_area_cm2):
    """Convert total mmol CO2/d to mol CO2 / cm2 ground / d.

    Maps to AgroC ``GPP`` and ``aboveground_respiration`` —
    canopy-level fluxes normalised by ground area.

    Args:
        total_mmol: total flux [mmol CO2/d].
        ground_area_cm2: ground area per plant [cm2].

    Returns:
        mol CO2 / cm2 / d
    """
    return total_mmol / 1000.0 / ground_area_cm2
