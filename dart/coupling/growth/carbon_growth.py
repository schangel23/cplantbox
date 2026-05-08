"""Carbon-feedback growth mode: CWLimitedGrowth (gf=3).

Provides functions to switch plants to carbon-limited growth, inject
per-organ CW_Gr maps from phloem solver output, and run a single
daily carbon-limited growth step.

When CW_Gr is empty, CWLimitedGrowth falls back to ExponentialGrowth
(preserving segment creation, tropism, branching). Only constrains
total organ length when CW_Gr entries are present.
"""

import numpy as np


def enable_cw_limited_growth(plant, wrap_roots=False, wrap_fa=True):
    """Switch all organ types to CWLimitedGrowth (Lock #9 wrap policy).

    PLAN_S5_SINK_SOURCE_COUPLING_2026-05-02 §S4 / ADR_LEAF_KINEMATICS_2026-04-28 §S5.
    Replaces the pre-Lock-#9 blanket overwrite (which silently clobbered
    the FA target by discarding `MultiPhase{Stem,Leaf}Growth` instances
    that `Plant::initCallbacks` had just minted) with a strict per-organ-type
    policy:

      * root (ot=2):
          bare CWLimitedGrowth() — preserves pre-Lock-#9 semantics.
          Override with ``wrap_roots=True`` for future root demand laws.
          See [[project_root_path_preservation]] for why bare-CWLim is the
          contract for roots.

      * stem/leaf (ot=3,4):
          if existing ``f_gf`` is ``MultiPhaseStemGrowth`` /
          ``MultiPhaseLeafGrowth`` and ``wrap_fa=True`` → wrap with
          ``demand=existing`` so getLength returns ``min(FA_target, supply)``.
          Otherwise bare ``CWLimitedGrowth()`` (pre-Lock-#9 semantics for
          non-FA XMLs — strict isinstance, D2 decision a).

    The strict wrap predicate keeps the 17+ non-maize XMLs in
    ``gui/cplantbox/params/`` bit-identical under carbon mode. Future
    demand laws (wheat FA, sorghum, ...) opt in by adding their class to
    the wrap allowlist here.

    Timing requirement
    ------------------
    Must run **after** ``Plant::initialize()`` (which dispatches through
    ``Plant::initCallbacks`` and mints ``MultiPhase{Stem,Leaf}Growth``
    via the ``gft_multi_phase_stem`` / ``gft_multi_phase_leaf`` factory
    types). Calling this helper before initialize would find the original
    ``gft_negexp`` ``ExponentialGrowth`` GF on the params, miss the FA
    wrap entirely, and reproduce the silent-clobber bug.

    The single production call site at
    ``dart/coupling/photosynthesis/diurnal.py:1627`` already runs after
    ``grow_plant(...)`` (which drives ``MappedPlant.initialize()``).

    Args:
        plant: pb.MappedPlant instance.
        wrap_roots: When True, also build CWLimitedGrowth(demand=existing)
            for roots whose ``f_gf`` is in the FA wrap allowlist. Default
            False to preserve [[project_root_path_preservation]].
        wrap_fa: When True (default), wrap MultiPhase{Stem,Leaf}Growth
            instances. Set False for an emergency rollback to pre-Lock-#9
            blanket-bare semantics across all organ types.

    Gate Ch1.PM.5 — preservable-GF allowlist for the bare path
    -----------------------------------------------------------
    Empty ``CW_Gr`` makes ``CWLimitedGrowth::getLength`` fall back to
    ``ExponentialGrowth::getLength`` (``growth.cpp:154``) — so overwriting
    an existing ``f_gf`` with bare ``CWLimitedGrowth()`` is bit-identical
    only when the existing GF was already ``ExponentialGrowth`` (gf=1) or
    ``CWLimitedGrowth`` (gf=3). For ``LinearGrowth`` (gf=2) or
    ``GompertzGrowth`` (gf=4) the overwrite would silently switch the
    organ's length curve to ExponentialGrowth-equivalent, breaking the G7
    D.0 6-XML invariance gate (CLAUDE.md "Bit-identical regression on
    native paths"). The bare branch is therefore gated on the int ``gf``
    enum: only gf ∈ {1, 3} get overwritten; LinearGrowth/Gompertz/etc.
    are left untouched.
    """
    import plantbox as pb

    # Allowlist of demand GFs the wrap recognises. Adding a new demand law
    # (e.g. WheatFournierAndrieu) is opt-in by appending here.
    fa_wrap_classes = (pb.MultiPhaseStemGrowth, pb.MultiPhaseLeafGrowth)

    # gf int → GrowthFunction class lookup (matches Plant.h::GrowthFunctionTypes).
    # Only ExponentialGrowth and CWLimitedGrowth fall back cleanly to
    # ExponentialGrowth-equivalent under empty CW_Gr, so only these can be
    # safely overwritten with bare CWLimitedGrowth().
    preservable_gf_ints = (1, 3)  # gft_negexp, gft_CWLim

    for ot in [2, 3, 4]:  # root, stem, leaf
        for param in plant.getOrganRandomParameter(ot):
            if param is None:
                continue
            existing = getattr(param, "f_gf", None)
            should_wrap = (
                wrap_fa
                and (ot in (3, 4) or wrap_roots)
                and isinstance(existing, fa_wrap_classes)
            )
            if should_wrap:
                param.f_gf = pb.CWLimitedGrowth(demand=existing)
            elif int(getattr(param, "gf", 0)) in preservable_gf_ints:
                param.f_gf = pb.CWLimitedGrowth()
            # else: LinearGrowth / Gompertz / unknown — leave alone so the
            # native length curve is preserved (Gate Ch1.PM.5 invariance).


def inject_cw_gr(plant, organ_growth_map, per_rank_map=None):
    """Fill CW_Gr (per-organ) and optionally CW_Gr_per_n (per-rank) maps.

    PiafMunch copies the SAME map to ALL subtypes of each organ type
    (runPM.cpp:643-652). We replicate this pattern for the per-organ
    scalar path. PLAN_PER_RANK_CARBON_FA_2026-05-03 §S5 extends the
    helper with an optional per-rank supply map for FA-on stems —
    populating CW_Gr_per_n activates the per-rank cap dispatch
    inside ``CWLimitedGrowth::getLength`` (S3).

    Per-rank semantics
    ------------------
    Only stem RPs whose ``f_gf`` is a wrap-aware ``CWLimitedGrowth``
    receive the per-rank map. Non-FA stems (bare ``CWLimitedGrowth``
    without a demand) and roots/leaves are unaffected: their
    ``CW_Gr_per_n`` stays empty, so ``getLength`` falls through to
    today's per-organ Lock #6 path. This preserves §G3 bit-identical
    regression for all non-FA XMLs and roots.

    Args:
        plant: pb.MappedPlant instance.
        organ_growth_map: {2: {orgID: dL}, 3: {...}, 4: {...}}
        per_rank_map: {orgID: [0.0, dL_n=1, ...]} — index-1-based per-rank
            supply [cm], stems only. Defaults to None (per-organ path only,
            i.e., S5 HEAD semantics).
    """
    for ot in [2, 3, 4]:
        cw_map = organ_growth_map.get(ot, {})
        for param in plant.getOrganRandomParameter(ot):
            if param is not None:
                param.f_gf.CW_Gr = cw_map
    if per_rank_map is None:
        return
    # Per-rank only attaches to FA-wrapped stems. Iterate stem RPs only;
    # leaves/roots have one rank ≡ one organ so per-rank is meaningless.
    for param in plant.getOrganRandomParameter(3):
        if param is None:
            continue
        growth_fn = getattr(param, "f_gf", None)
        if not hasattr(growth_fn, "CW_Gr_per_n"):
            continue
        if getattr(growth_fn, "demand", None) is None:
            # Bare CWLimitedGrowth (non-FA stem) — leave per-rank empty so
            # CWLimitedGrowth::getLength routes through the per-organ path.
            growth_fn.CW_Gr_per_n = {}
            continue
        # PiafMunch-style: same per-rank map shared across all FA stem
        # subTypes (mainstem, branches, ...). Filter by stem-orgIDs that
        # actually appear in the map; orgIDs are unique per-organ so the
        # filter is a no-op when the map already only contains stem ids.
        growth_fn.CW_Gr_per_n = {oid: vec for oid, vec in per_rank_map.items()}


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

    # PLAN_PER_RANK_CARBON_FA_2026-05-03 §S5: route the per-rank map
    # through inject_cw_gr so FA-on stems get the sub-organ supply
    # dispatch. Non-FA stems and leaves/roots stay on the per-organ path
    # (per_rank_map keyed by stem orgID only; non-stem RPs untouched).
    growth_map, per_rank_map = solver.compute_organ_growth_map(
        result['Rg_node'], return_per_rank=True
    )
    inject_cw_gr(plant, growth_map, per_rank_map=per_rank_map)

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
