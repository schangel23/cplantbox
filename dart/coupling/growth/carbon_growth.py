"""Carbon-feedback growth mode: CWLimitedGrowth (gf=3).

Provides functions to switch plants to carbon-limited growth, inject
per-organ CW_Gr maps from phloem solver output, and run a single
daily carbon-limited growth step.

When CW_Gr is empty, CWLimitedGrowth falls back to ExponentialGrowth
(preserving segment creation, tropism, branching). Only constrains
total organ length when CW_Gr entries are present.
"""

import numpy as np


def enable_cw_limited_growth(plant):
    """Switch all organ types to CWLimitedGrowth (gf=3).

    Args:
        plant: pb.MappedPlant instance.
    """
    import plantbox as pb

    for ot in [2, 3, 4]:  # root, stem, leaf
        for param in plant.getOrganRandomParameter(ot):
            if param is not None:
                param.f_gf = plant.createGrowthFunction(
                    pb.GrowthFunctionType.CWLim)


def inject_cw_gr(plant, organ_growth_map):
    """Fill CW_Gr maps on each organ type's f_gf instances.

    PiafMunch copies the SAME map to ALL subtypes of each organ type
    (runPM.cpp:643-652). We replicate this pattern.

    Args:
        plant: pb.MappedPlant instance.
        organ_growth_map: {2: {orgID: dL}, 3: {...}, 4: {...}}
    """
    for ot in [2, 3, 4]:
        cw_map = organ_growth_map.get(ot, {})
        for param in plant.getOrganRandomParameter(ot):
            if param is not None:
                param.f_gf.CW_Gr = cw_map


def step_plant_carbon(plant, An_leaf, sim_day, tair_c=25.0, dt=1.0,
                      warm_start=None, gdd_accumulated=None):
    """One daily carbon-limited growth step.

    1. Phloem solve with An_leaf -> Rg_node
    2. compute_organ_growth_map -> per-organ length increments
    3. inject_cw_gr -> fill CW_Gr on all organ types
    4. plant.simulate(dt) -> CPlantBox reads CW_Gr

    Args:
        plant: persistent pb.MappedPlant (with gf=3 set).
        An_leaf: per-leaf-segment An (mol CO2/d).
        sim_day: current simulation day.
        tair_c: air temperature [C].
        dt: timestep (days).
        warm_start: optional C_ST warm-start dict.
        gdd_accumulated: Accumulated GDD from sowing (°C·day). If provided,
            DVS is computed from thermal time instead of calendar days.

    Returns:
        carbon_result dict from phloem solver (includes Rg_node, C_ST stats).
    """
    from ..carbon.phloem_steady import QuasiSteadyPhloem

    solver = QuasiSteadyPhloem(plant, sim_day=sim_day,
                                gdd_accumulated=gdd_accumulated)
    result = solver.solve(An_leaf, Tair_C=tair_c, sim_day=sim_day,
                          warm_start=warm_start)

    growth_map = solver.compute_organ_growth_map(result['Rg_node'])
    inject_cw_gr(plant, growth_map)

    # Step with error recovery (same pattern as grow.py:140-153)
    try:
        plant.simulate(dt)
    except (IndexError, RuntimeError) as e:
        print(f"  Warning: simulate() error at day {sim_day}: {e}")
        try:
            plant.simulate(0.0)  # re-sync nodes
        except Exception:
            pass

    return result
