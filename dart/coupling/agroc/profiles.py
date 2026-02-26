"""Vertical soil-profile computations for AgroC coupling.

All functions follow the same pattern:
  1. Get root length per layer via ``pb.SegmentAnalyser.distribution("length", ...)``
  2. Compute fraction per layer: ``frac = root_length_per_layer / sum``
  3. Distribute total flux: ``per_layer = total * frac``
  4. Convert units via ``unit_conversions``
  5. Return dict with profile array + conservation check values
"""

import numpy as np

try:
    import plantbox as pb
except ImportError:
    pb = None

from . import unit_conversions as uc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _root_length_fractions(plant, n_layers, depth_cm):
    """Compute root length fraction per depth layer.

    Args:
        plant: pb.MappedPlant (grown, with roots).
        n_layers: number of vertical soil layers.
        depth_cm: maximum soil depth [cm] (positive).

    Returns:
        fractions: np.array of shape (n_layers,), sums to 1.0 (or all-zero
                   if no roots exist).
        root_length: np.array of root length per layer [cm].
    """
    ana = pb.SegmentAnalyser(plant)
    ana.filter("organType", pb.root)

    # distribution("length", top, bot, n, exact)
    # CPlantBox z-convention: surface = 0, depth = negative
    root_length = np.array(
        ana.distribution("length", 0.0, -depth_cm, n_layers, True)
    )

    total = float(np.sum(root_length))
    if total <= 0:
        return np.zeros(n_layers), root_length

    fractions = root_length / total
    return fractions, root_length


def _layer_geometry(n_layers, depth_cm, row_spacing_cm, plant_spacing_cm):
    """Compute layer depths and volumes.

    Returns:
        dict with layer_thickness_cm, layer_vol_cm3, ground_area_cm2,
        depth_top_cm, depth_bot_cm, depth_mid_cm arrays.
    """
    thickness = depth_cm / n_layers
    ground_area = row_spacing_cm * plant_spacing_cm
    layer_vol = thickness * ground_area

    depth_top = np.linspace(0, depth_cm - thickness, n_layers)
    depth_bot = depth_top + thickness
    depth_mid = (depth_top + depth_bot) / 2.0

    return {
        "layer_thickness_cm": thickness,
        "layer_vol_cm3": layer_vol,
        "ground_area_cm2": ground_area,
        "depth_top_cm": depth_top,
        "depth_bot_cm": depth_bot,
        "depth_mid_cm": depth_mid,
    }


# ---------------------------------------------------------------------------
# Root respiration profile
# ---------------------------------------------------------------------------

def compute_root_respiration_profile(plant, Rm_root_mmol, Rg_root_mmol,
                                     n_layers=20, depth_cm=100.0,
                                     row_spacing_cm=75.0,
                                     plant_spacing_cm=20.0):
    """Distribute root respiration (Rm + Rg) across soil layers.

    Maps to AgroC ``rnodert(ri)`` (plants.f90:1639).

    Args:
        plant: pb.MappedPlant.
        Rm_root_mmol: maintenance respiration of roots [mmol CO2/d].
        Rg_root_mmol: growth respiration of roots [mmol CO2/d].
        n_layers, depth_cm, row_spacing_cm, plant_spacing_cm: grid geometry.

    Returns:
        dict with:
            profile_mol_co2_per_cm3_d: np.array (n_layers,)
            total_input_mmol: Rm + Rg total used as input
            profile_sum_mmol: sum of distributed flux (for conservation check)
    """
    total_mmol = Rm_root_mmol + Rg_root_mmol
    fracs, _ = _root_length_fractions(plant, n_layers, depth_cm)
    geom = _layer_geometry(n_layers, depth_cm, row_spacing_cm, plant_spacing_cm)

    per_layer_mmol = total_mmol * fracs  # mmol CO2/d per layer

    profile = np.array([
        uc.mmol_co2_to_mol_co2_per_cm3(per_layer_mmol[i], geom["layer_vol_cm3"])
        for i in range(n_layers)
    ])

    # Conservation: back-compute total from profile
    profile_sum = float(np.sum(profile * geom["layer_vol_cm3"])) * 1000.0

    return {
        "profile_mol_co2_per_cm3_d": profile,
        "total_input_mmol": total_mmol,
        "profile_sum_mmol": profile_sum,
    }


# ---------------------------------------------------------------------------
# Root exudation profile
# ---------------------------------------------------------------------------

def compute_root_exudation_profile(plant, exud_total_mmol_suc,
                                   n_layers=20, depth_cm=100.0,
                                   row_spacing_cm=75.0,
                                   plant_spacing_cm=20.0):
    """Distribute root exudation across soil layers.

    Maps to AgroC ``rnodexu(ri)`` (plants.f90:1577).
    Source: ``carbon_result['root_exud_mmol_d']``.

    Args:
        plant: pb.MappedPlant.
        exud_total_mmol_suc: total root exudation [mmol sucrose/d].
        n_layers, depth_cm, row_spacing_cm, plant_spacing_cm: grid geometry.

    Returns:
        dict with:
            profile_kg_c_per_cm3_d: np.array (n_layers,)
            total_input_mmol_suc: input total
            profile_sum_kg_c: sum of distributed flux
    """
    fracs, _ = _root_length_fractions(plant, n_layers, depth_cm)
    geom = _layer_geometry(n_layers, depth_cm, row_spacing_cm, plant_spacing_cm)

    per_layer_mmol_suc = exud_total_mmol_suc * fracs

    profile = np.array([
        uc.mmol_suc_to_kg_c_per_cm3(per_layer_mmol_suc[i], geom["layer_vol_cm3"])
        for i in range(n_layers)
    ])

    profile_sum = float(np.sum(profile * geom["layer_vol_cm3"]))

    return {
        "profile_kg_c_per_cm3_d": profile,
        "total_input_mmol_suc": exud_total_mmol_suc,
        "profile_sum_kg_c": profile_sum,
    }


# ---------------------------------------------------------------------------
# Dead root carbon profile
# ---------------------------------------------------------------------------

def compute_root_dead_carbon_profile(plant, dead_root_mmol_suc,
                                     n_layers=20, depth_cm=100.0,
                                     row_spacing_cm=75.0,
                                     plant_spacing_cm=20.0):
    """Distribute dead root carbon input across soil layers.

    Maps to AgroC ``rnodedeadw(ri)`` (plants.f90:1599).
    Source: ``carbon_result['root_dead_mmol_d']``.

    Args:
        plant: pb.MappedPlant.
        dead_root_mmol_suc: total dead root carbon [mmol sucrose/d].
        n_layers, depth_cm, row_spacing_cm, plant_spacing_cm: grid geometry.

    Returns:
        dict with:
            profile_kg_c_per_cm3_d: np.array (n_layers,)
            total_input_mmol_suc: input total
            profile_sum_kg_c: sum of distributed flux
    """
    fracs, _ = _root_length_fractions(plant, n_layers, depth_cm)
    geom = _layer_geometry(n_layers, depth_cm, row_spacing_cm, plant_spacing_cm)

    per_layer_mmol_suc = dead_root_mmol_suc * fracs

    profile = np.array([
        uc.mmol_suc_to_kg_c_per_cm3(per_layer_mmol_suc[i], geom["layer_vol_cm3"])
        for i in range(n_layers)
    ])

    profile_sum = float(np.sum(profile * geom["layer_vol_cm3"]))

    return {
        "profile_kg_c_per_cm3_d": profile,
        "total_input_mmol_suc": dead_root_mmol_suc,
        "profile_sum_kg_c": profile_sum,
    }


# ---------------------------------------------------------------------------
# Root water uptake profile
# ---------------------------------------------------------------------------

def compute_root_water_uptake_profile(hm, plant, n_layers=20, depth_cm=100.0,
                                      row_spacing_cm=75.0,
                                      plant_spacing_cm=20.0):
    """Bin per-segment root water uptake into soil layers.

    Unlike respiration/exudation profiles, this is NOT RLD-weighted but
    spatially resolved from the hydraulic solve via ``hm.radial_fluxes()``
    which returns per-segment plant-exterior exchanges [cm3/d].

    Args:
        hm: PhloemFluxPython (after solve).
        plant: pb.MappedPlant.
        n_layers, depth_cm, row_spacing_cm, plant_spacing_cm: grid geometry.

    Returns:
        dict with:
            profile_cm3_per_cm3_d: np.array (n_layers,) — water uptake per
                layer volume per day [cm3 H2O / cm3 soil / d].
            total_uptake_cm3_d: total root water uptake [cm3/d].
    """
    geom = _layer_geometry(n_layers, depth_cm, row_spacing_cm, plant_spacing_cm)
    thickness = geom["layer_thickness_cm"]

    if hm is None:
        return {
            "profile_cm3_per_cm3_d": np.zeros(n_layers),
            "total_uptake_cm3_d": 0.0,
        }

    # Per-segment radial fluxes [cm3/d] — all organ types, all segments
    radial = np.array(hm.radial_fluxes())
    organ_types = np.array(plant.organTypes)
    nodes = plant.getNodes()
    segs = plant.getSegments()

    # Segment z-midpoints
    z_mids = np.array([
        (nodes[seg.x].z + nodes[seg.y].z) / 2.0
        for seg in segs
    ])

    # Only root segments (organType == 2), below ground (z < 0)
    root_mask = (organ_types == 2) & (z_mids < 0)

    profile = np.zeros(n_layers)
    for i in range(n_layers):
        z_top = -i * thickness         # e.g. 0, -5, -10, ...
        z_bot = -(i + 1) * thickness   # e.g. -5, -10, -15, ...
        layer_mask = root_mask & (z_mids <= z_top) & (z_mids > z_bot)
        # Sum absolute uptake per layer, divide by layer volume
        uptake_layer = float(np.sum(np.abs(radial[layer_mask])))
        profile[i] = uptake_layer / geom["layer_vol_cm3"]

    total_uptake = float(np.sum(np.abs(radial[root_mask])))

    return {
        "profile_cm3_per_cm3_d": profile,
        "total_uptake_cm3_d": total_uptake,
    }


# ---------------------------------------------------------------------------
# Aboveground fluxes
# ---------------------------------------------------------------------------

def compute_aboveground_fluxes(carbon_result, An_total_mmol, ground_area_cm2):
    """Compute canopy-level GPP and aboveground respiration.

    Maps to AgroC ``GPP`` and ``aboveground_respiration`` variables.

    Args:
        carbon_result: dict from ``solve_carbon_partitioning()``.
        An_total_mmol: total net assimilation [mmol CO2/d].
        ground_area_cm2: ground area per plant [cm2].

    Returns:
        dict with:
            GPP_mol_co2_per_cm2_d: gross primary production per ground area.
            aboveground_resp_mol_co2_per_cm2_d: leaf + stem respiration per
                ground area (Rm_leaf + Rm_stem).
    """
    GPP = uc.mmol_co2_to_mol_co2_per_cm2(An_total_mmol, ground_area_cm2)

    Rm_leaf = carbon_result.get("Rm_leaf", 0.0)
    Rm_stem = carbon_result.get("Rm_stem", 0.0)
    above_resp = uc.mmol_co2_to_mol_co2_per_cm2(
        Rm_leaf + Rm_stem, ground_area_cm2
    )

    return {
        "GPP_mol_co2_per_cm2_d": GPP,
        "aboveground_resp_mol_co2_per_cm2_d": above_resp,
    }
