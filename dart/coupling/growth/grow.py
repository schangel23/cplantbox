#!/usr/bin/env python3
"""
Grow a maize plant using Pheno4D-calibrated maize.xml, then extract high-quality G3 mesh.

This is Baker's approach:
  1. CPlantBox generates G1 skeleton from parametric growth model
  2. G1→G3 lofter adds realistic geometry (tubes + leaf surfaces)
  3. Export to OBJ with UV mapping for DART
  4. Render side-by-side G1 | G3 comparison PNG

No skeleton injection. Just parametric growth with calibrated parameters.

Usage:
  python grow_calibrated_plant.py \
      --xml maize_calibrated.xml \
      --days 30 \
      --output maize_day30 \
      --resolution fine
"""

import json
import numpy as np
from pathlib import Path
import argparse

import plantbox as pb

from ..config import HYDRAULICS_PATH, DEFAULT_XML, get_hydraulics_json, get_photosynthesis_json, get_phloem_json
from ..geometry import loft_organs, G3Mesh, extract_organs_for_lofter
from ..geometry.cplantbox_adapter import get_plantsim_feature_kwargs_from_env
from ..prospect_params import get_chl_per_segment, vcmax25_from_cab
from .cp_swap import apply_donor_cps


FA_KINETICS_PATH = Path(__file__).resolve().parent.parent / "data" / "phase_III_per_rank.json"
# 15 = number of leaf-bearing phytomers on the maize_calibrated mainstem
# (subType 2..16, 15 leaves). The 16th phytomer carries the tassel and has
# no leaf child, so its per-rank cessation latch never fires; trimming the
# stem arrays to 15 lets the global mainstem cessation_age_ gate fire when
# all 15 leaves have collared (closes the ba2188fd dangling-rank bug).
FA_DEFAULT_MAX_RANK = 15


def enable_fa_on_mainstem(plant, kinetics_path=None, max_rank=FA_DEFAULT_MAX_RANK,
                          verbose=False):
    """Enable Fournier-Andrieu per-phytomer internode kinetics on mainstem.

    Flips the C++ `use_fournier_andrieu_kinetics` flag on the mainstem's
    StemRandomParameter (subType=1) and populates the three per-rank kinetic
    tables (`internode_v_n`, `internode_D_n`, `internode_IL_final`) from
    ``data/phase_III_per_rank.json``.

    Must be called BEFORE ``plant.initialize()``. Combines with:
      * S3b.7 plastochron-driven rank initiation (gated by
        `use_fournier_andrieu_kinetics`), and
      * S3b.8 ``Stem::internodalGrowth`` basal_zero_ranks gate (same flag),
    to deliver anatomically correct V-stage geometry directly from the C++
    model. XML-level changes (``lb``, ``basal_internode_cm``,
    ``plastochron_andrieu``) apply unconditionally; the behavioural gates
    only fire when this flag is True.

    Silent no-op on non-maize plants (no mainstem subType=1 with FA attrs)
    and on older ``.so`` builds that pre-date the FA branch. Safe to call
    on any plant.

    Args:
        plant: pb.MappedPlant (or equivalent) **before** ``initialize()``.
        kinetics_path: Path to phase_III_per_rank.json. Defaults to the
            vault copy at ``data/phase_III_per_rank.json``.
        max_rank: Number of mainstem phytomers to configure (default 16,
            matching maize_calibrated.xml).
        verbose: Print a one-line confirmation on success.

    Returns:
        bool — True if FA was enabled, False if skipped (non-maize, no
        kinetics JSON, or old .so).
    """
    try:
        srp = plant.getOrganRandomParameter(pb.OrganTypes.stem, 1)
    except Exception:
        if verbose:
            print("  FA: no mainstem subType=1 — skipping.")
        return False
    if not hasattr(srp, "use_fournier_andrieu_kinetics"):
        if verbose:
            print("  FA: .so predates FA support — skipping.")
        return False

    path = Path(kinetics_path) if kinetics_path else FA_KINETICS_PATH
    if not path.exists():
        if verbose:
            print(f"  FA: kinetics JSON missing at {path} — skipping.")
        return False

    with path.open() as f:
        data = json.load(f)
    v_table = data["v_n_cm_per_degCd"]["expt_1B_primary"]
    d_table = data["D_n_degCd"]["values"]
    il_table = data["IL_final_cross_check_cm"]["values"]

    def _fill(table, fallback_key, fallback_value):
        return [float(table.get(str(n), table.get(fallback_key, fallback_value)))
                for n in range(1, max_rank + 1)]

    srp.use_fournier_andrieu_kinetics = True
    srp.internode_v_n = _fill(v_table, "15", 0.18)
    srp.internode_D_n = _fill(d_table, "15", 79.0)
    il_final = _fill(il_table, "15", 16.0)
    # Plan B.3 (peduncle exuberance, 2026-04-27): basal_zero_ranks get
    # IL_final=0 so Stem::simulate's fa_sum doesn't pick up the phantom
    # 16 cm fallback for ranks the JSON doesn't cover (1..4 for Déa).
    # Without this gate fa_sum ≈ 4*16 + Σ IL_final[5..16] = 64 + 200 = 264 cm,
    # which keeps targetlength chasing the apex 60 cm above the
    # branching-zone cap and fuels the apical-zone bleed observed at
    # 207 cm day-130. Pinning these to 0 makes calcLengthPerPhytomer return
    # 0 for those ranks; the basal_step seed (basal_internode_cm) still
    # appears in length_per_n via the plastochron loop, so geometry is
    # unchanged — only the targetlength forecast contracts.
    basal_zero = list(getattr(srp, "basal_zero_ranks", [1, 2, 3, 4]))
    for r in basal_zero:
        if 1 <= r <= len(il_final):
            il_final[r - 1] = 0.0
    srp.internode_IL_final = il_final
    if verbose:
        print(f"  FA kinetics: enabled on mainstem subType=1 (max_rank={max_rank})")
    return True


def enable_fa_on_leaves(plant, verbose=False):
    """Enable Fournier-Andrieu logistic length kinetics on every leaf.

    PLAN_YOUNG_LEAF_PHYSICS_2026-04-25 §Gap 1. Per-leaf logistic
        length(TT) = lmax / (1 + exp(-(TT - tau_n) / sigma_n))
    keyed off each leaf's existing ``tt_emergence`` and ``phyllochron_tt``:
        tau_n   = tt_emergence + 2 * phyllochron_tt   (m=0.5 two phyllochrons after emergence)
        sigma_n = phyllochron_tt                       (4σ ≈ 4 phyllochrons full rise)

    The 2-phyllochron offset is the empirical finding from the day-33 V3
    sanity sweep (2026-04-25): smaller offsets (½..1 phyllochron) grow leaves
    *faster* than the linear ``r*dt`` baseline because the logistic's middle
    rises quickly; biologists' intuition is that a young leaf takes ~3-4
    phyllochrons from emergence to full size, so τ at the midpoint (2
    phyllochrons after emergence) and σ = 1 phyllochron (4σ = 4 phyllochrons
    full rise) reproduces that picture. At V3 (TT≈232) lower ranks form a
    decreasing size cascade (~21, 25, 19, 12, 6 cm) instead of all-at-lmax.

    Saturates to lmax at large TT (m → 1 within ~6% for the flag leaf at
    day 130; bit-identical for ranks 1-13). Lower ranks at V3 stay visibly
    small.

    Silent no-op on LRPs without the FA fields (older .so) or with
    ``tt_emergence < 0`` (FA-emergence disabled). Must run BEFORE
    ``plant.initialize()``.

    Args:
        plant: pb.MappedPlant pre-initialize().
        verbose: print one summary line per affected LRP.

    Returns:
        int — count of LRPs configured.
    """
    n_done = 0
    try:
        leaf_lrps = plant.getOrganRandomParameter(pb.OrganTypes.leaf)
    except Exception:
        if verbose:
            print("  FA leaves: no LRPs — skipping.")
        return 0
    for lrp in leaf_lrps:
        if not hasattr(lrp, "use_fa_kinetics"):
            if verbose and n_done == 0:
                print("  FA leaves: .so predates leaf-side FA — skipping.")
            return 0
        # Skip blade-less rolls: subType < 2 is a placeholder/sheath in this
        # XML convention; FA-on with lmax=0 still saturates safely but pollutes
        # the verbose output, so just skip them.
        if float(getattr(lrp, "lmax", 0.0)) <= 1e-6:
            continue
        tt_em = float(getattr(lrp, "tt_emergence", -1.0))
        if tt_em < 0.0:
            # Leaf emerges by ldelay, not TT — FA's TT clock is not meaningful
            # for it. Leave scalar.
            continue
        phyll = float(getattr(lrp, "phyllochron_tt", 57.9))
        if phyll <= 0.0:
            phyll = 57.9
        lrp.use_fa_kinetics = 1
        lrp.tau_extension_n = tt_em + 2.0 * phyll
        lrp.sigma_extension_n = phyll
        n_done += 1
        if verbose:
            print(f"    leaf subType={lrp.subType}: tau={lrp.tau_extension_n:.1f} "
                  f"sigma={lrp.sigma_extension_n:.1f} (tt_em={tt_em:.1f}, "
                  f"phyll={phyll:.1f})")
    if verbose and n_done > 0:
        print(f"  FA kinetics: enabled on {n_done} leaf LRPs")
    return n_done


def setup_successor_where(plant):
    """Set deterministic per-position successorWhere on the mainstem.

    The XML contains leaf subtypes 2..N (one per position) but only a
    placeholder successor rule.  This function replaces it with per-position
    rules via the Python API so that linking node 0 gets subType 2,
    linking node 1 gets subType 3, etc.

    In phytomer decomposition mode (decompose_phytomer=1), only sheath
    subtypes (even) are stem successors — blades are created by sheath
    successor rules in the XML.

    If a tassel spike subType (20) is present in the XML, an extra successor
    rule is appended at the final linking node (one past the flag leaf)
    pointing at it. The spike's own successor → branch (subType 21) is kept
    in the XML and must not be overwritten here — only mainstem subType 1 is
    modified.
    """
    TASSEL_SPIKE_SUBTYPE = 20
    has_tassel = any(p.subType == TASSEL_SPIKE_SUBTYPE
                     for p in plant.getOrganRandomParameter(pb.stem))

    # Check if phytomer decomposition is active
    seed_params = plant.getOrganRandomParameter(pb.seed)
    decompose = False
    if seed_params:
        decompose = getattr(seed_params[0], 'decompose_phytomer', 0) == 1

    if decompose:
        # Phytomer mode: stem creates both sheath + blade at each position.
        # Uses two separate successor rules per position (both fire at same
        # linking node). CPlantBox's getLateralType() is probabilistic, so
        # successorNo=2 with [sheath, blade] would pick the same type twice.
        # Instead: rule 2*i → sheath, rule 2*i+1 → blade.
        sheath_subtypes = []
        blade_subtypes = []
        for p in plant.getOrganRandomParameter(pb.leaf):
            if p.getParameter('isPseudostem') == 1:
                sheath_subtypes.append(p.subType)
            else:
                blade_subtypes.append(p.subType)
        sheath_subtypes.sort()
        blade_subtypes.sort()

        if not sheath_subtypes or not blade_subtypes:
            print("  No sheath/blade subtypes found, skipping successorWhere")
            return

        n_phytomers = min(len(sheath_subtypes), len(blade_subtypes))
        import math
        for p in plant.getOrganRandomParameter(pb.stem):
            if p.subType == 1:
                # Two rules per position: one for sheath, one for blade
                successor_st = []
                successor_ot = []
                successor_p = []
                successor_no = []
                successor_where = []
                for i in range(n_phytomers):
                    # Rule 2*i: sheath at position i
                    successor_st.append([sheath_subtypes[i]])
                    successor_ot.append([4])
                    successor_p.append([1.0])
                    successor_no.append(1)
                    successor_where.append([float(i)])
                    # Rule 2*i+1: blade at position i
                    successor_st.append([blade_subtypes[i]])
                    successor_ot.append([4])
                    successor_p.append([1.0])
                    successor_no.append(1)
                    successor_where.append([float(i)])
                # Extra rule: tassel spike at node n_phytomers (past flag leaf)
                if has_tassel:
                    successor_st.append([TASSEL_SPIKE_SUBTYPE])
                    successor_ot.append([3])  # organType 3 = stem
                    successor_p.append([1.0])
                    successor_no.append(1)
                    successor_where.append([float(n_phytomers)])
                p.successorST = successor_st
                p.successorOT = successor_ot
                p.successorP = successor_p
                p.successorNo = successor_no
                p.successorWhere = successor_where
                p.RotBeta = math.pi
                p.BetaDev = 0.22
                plant.setOrganRandomParameter(p)
                tassel_note = f" +1 tassel rule at node {n_phytomers}" if has_tassel else ""
                print(f"  successorWhere (phytomer): stem subType={p.subType}, "
                      f"{n_phytomers} positions, 2 rules each{tassel_note} "
                      f"(sheath {sheath_subtypes[0]}+blade {blade_subtypes[0]} ... "
                      f"sheath {sheath_subtypes[-1]}+blade {blade_subtypes[-1]})")
    else:
        # Monolithic mode: stem → leaf subtypes with Width_blade > 0
        leaf_subtypes = []
        for p in plant.getOrganRandomParameter(pb.leaf):
            if p.subType >= 2 and p.Width_blade > 0.01:
                leaf_subtypes.append(p.subType)
        leaf_subtypes.sort()

        if not leaf_subtypes:
            print("  No calibrated leaf subtypes found, skipping successorWhere")
            return

        import math
        n_leaves = len(leaf_subtypes)
        for p in plant.getOrganRandomParameter(pb.stem):
            if p.subType == 1:
                succ_st = [[st] for st in leaf_subtypes]
                succ_ot = [[4] for _ in leaf_subtypes]
                succ_p = [[1.0] for _ in leaf_subtypes]
                succ_no = [1] * n_leaves
                succ_where = [[float(i)] for i in range(n_leaves)]
                if has_tassel:
                    succ_st.append([TASSEL_SPIKE_SUBTYPE])
                    succ_ot.append([3])  # organType 3 = stem
                    succ_p.append([1.0])
                    succ_no.append(1)
                    succ_where.append([float(n_leaves)])
                p.successorST = succ_st
                p.successorOT = succ_ot
                p.successorP = succ_p
                p.successorNo = succ_no
                p.successorWhere = succ_where
                p.RotBeta = math.pi
                p.BetaDev = 0.22
                plant.setOrganRandomParameter(p)
                tassel_note = f" +1 tassel rule at node {n_leaves}" if has_tassel else ""
                print(f"  successorWhere: stem subType={p.subType}, {n_leaves} leaf rules{tassel_note} "
                      f"(node 0->subType {leaf_subtypes[0]}, ..., "
                      f"node {n_leaves-1}->subType {leaf_subtypes[-1]})")


def init_plant(xml_path=None, seed=None, enable_photosynthesis=True,
               cp_donor_seed=None, cp_donor_mode="draw_coherent",
               soil_min_b=(-50.0, -50.0, -150.0),
               soil_max_b=(50.0, 50.0, 0.0),
               soil_cell_number=(1, 1, 150)):
    """Create and initialize a plant without growing. For carbon-limited mode.

    Same setup as grow_plant() but stops after initialize().
    Returns plant at day 0.

    Args:
        xml_path: Path to calibrated XML. Defaults to DEFAULT_XML.
        seed: Optional random seed for reproducibility.
        enable_photosynthesis: Enable soil grid for photosynthesis (default True).
        cp_donor_seed: If set, swap leaf surface_cps (and lmax/Width_blade/
            areaMax) for a donor drawn from the MF3D canonical library.
            Lets a canopy render N plants with per-plant leaf-shape variation
            without regenerating the XML. None → use whatever CPs are in XML.
        cp_donor_mode: Reducer for the donor draw. ``"draw_coherent"`` (default)
            picks a single MF3D plant covering all positions; ``"draw"`` draws
            independently per position; ``"median"`` uses the pool median.
        soil_min_b, soil_max_b, soil_cell_number: 3D rectangular grid for the
            soil seg→cell mapping (cm, cm, ints). Defaults to a 1×1×100
            vertical column with **±50 cm lateral OOD bounds** (Phase 3.5):
            cellidx only depends on z when ``cell_number_xy = 1``, so the
            lateral box just controls which roots are eligible for RWU.
            ±50 cm captures a single maize plant's root spread; the legacy
            ``_picker`` was lateral-blind, and ±5 cm cropped most roots out.
            Pass ``(8, 8, 25)`` etc. to opt into true 3D heterogeneity.

    Returns:
        pb.MappedPlant at day 0, initialized and ready for simulate().
    """
    if xml_path is None:
        xml_path = str(DEFAULT_XML)

    plant = pb.MappedPlant()
    plant.readParameters(str(xml_path))

    if seed is not None:
        plant.setSeed(seed)

    if cp_donor_seed is not None:
        apply_donor_cps(plant, donor_seed=cp_donor_seed, mode=cp_donor_mode,
                        verbose=False)

    if enable_photosynthesis:
        depth = float(soil_max_b[2] - soil_min_b[2])
        soil_domain = pb.SDF_PlantContainer(np.inf, np.inf, depth, True)
        plant.setGeometry(soil_domain)

        # Shift the soil grid so it's centered on the plant's XY seed
        # position. Maize calibrated XML places seedPos at (200, 200, -3)
        # for field-coordinate compatibility; soil_min_b/max_b describe the
        # box *relative to the seed*. Without this shift the entire root
        # system maps to cellidx=-1 (out-of-domain) and RWU silently falls
        # to zero — symptom Phase 3.5 hit when first replacing the legacy
        # _picker (which was lateral-blind) with setRectangularGrid.
        srp = plant.getOrganRandomParameter(pb.OrganTypes.seed, 0)
        sx, sy = float(srp.seedPos.x), float(srp.seedPos.y)
        shifted_min = (soil_min_b[0] + sx, soil_min_b[1] + sy, soil_min_b[2])
        shifted_max = (soil_max_b[0] + sx, soil_max_b[1] + sy, soil_max_b[2])

        # Upstream-canonical 3D pattern: setRectangularGrid delegates the
        # seg→cell mapping to MappedSegments::soil_index_, replacing the
        # hand-rolled _picker lambda + setSoilGrid pair retired in Phase 3.5
        # (PLAN_DUMUX_INTEGRATION_2026-05-05.md §"Phase 3.5").
        plant.setRectangularGrid(
            pb.Vector3d(*shifted_min),
            pb.Vector3d(*shifted_max),
            pb.Vector3d(*soil_cell_number),
            False,  # cut=False: no segment splitting at cell boundaries (preserves
                    # legacy graph for parity; flip to True only when per-cell
                    # uptake granularity at sub-segment scale matters)
        )

    plant.initialize()
    return plant


def grow_plant(xml_path, simulation_time, min_stem_nodes=50, min_leaf_nodes=20,
               enable_photosynthesis=False, seed=None,
               cp_donor_seed=None, cp_donor_mode="draw_coherent",
               daily_met=None, T_air_default=25.0,
               mutate_lrp_pre_init=None,
               soil_min_b=(-50.0, -50.0, -150.0),
               soil_max_b=(50.0, 50.0, 0.0),
               soil_cell_number=(1, 1, 150)):
    """Grow a CPlantBox plant from calibrated XML.

    Args:
        cp_donor_seed: Optional seed selecting an MF3D donor plant whose
            per-position leaf surface_cps (plus lmax / Width_blade / areaMax)
            are swapped into this plant before ``initialize()``. Leaves the
            XML on disk untouched. None → use the XML's baked-in CPs.
        cp_donor_mode: Donor reducer mode (``"draw_coherent"``, ``"draw"``,
            or ``"median"``).
        daily_met: Optional pre-loaded daily-met dict (``sim_day -> {T_mean_C,
            T_min_C, T_max_C, ...}``) from ``load_daily_met()``. When None,
            ``get_daily_met()`` auto-loads the default daily-met CSV
            (``juelich_2024_daily_met.csv``). Each 1-day ``simulate()`` step
            reads ``T_mean_C`` for the current day and calls
            ``plant.setAirTemperature`` so CPlantBox's thermal-time accumulator
            reflects the real weather. Falls back to ``T_air_default`` on any
            day that has no met entry.
        T_air_default: Fallback air temperature (°C) for days with no met
            data. Also used when no met source is available at all.
        soil_min_b, soil_max_b, soil_cell_number: 3D rectangular grid for the
            soil seg→cell mapping (cm, cm, ints). Defaults to a 1×1×100
            vertical column with **±50 cm lateral OOD bounds** (Phase 3.5):
            cellidx only depends on z when ``cell_number_xy = 1``, so the
            lateral box just controls which roots are eligible for RWU.
            ±50 cm captures a single maize plant's root spread; the legacy
            ``_picker`` was lateral-blind, and ±5 cm cropped most roots out.
            Pass e.g. ``(-50,-50,-150)/(50,50,0)/(8,8,25)`` for true 3D
            heterogeneity. Phase 3.5+ canonical pattern.
    """
    print(f"=== Growing Plant ===")
    print(f"  XML: {xml_path}")
    print(f"  Simulation time: {simulation_time} days")
    if seed is not None:
        print(f"  Seed: {seed}")
    if cp_donor_seed is not None:
        print(f"  CP donor: mode={cp_donor_mode}, seed={cp_donor_seed}")
    if enable_photosynthesis:
        print(f"  Photosynthesis: ENABLED (soil grid active)")

    plant = pb.MappedPlant()
    plant.readParameters(str(xml_path))

    if seed is not None:
        plant.setSeed(seed)

    if cp_donor_seed is not None:
        apply_donor_cps(plant, donor_seed=cp_donor_seed, mode=cp_donor_mode,
                        verbose=True)

    # Final pre-initialize hook for tests / harnesses that need to flip
    # LRP fields before f_gf is minted in plant.initialize() →
    # Plant::initCallbacks. Used by the S0.3 dispatch parity harness
    # (ADR_LEAF_KINEMATICS_2026-04-28); leave None for production runs.
    if mutate_lrp_pre_init is not None:
        mutate_lrp_pre_init(plant)

    # Soil geometry — must be set BEFORE plant.initialize() when using photosynthesis.
    # Roots are excluded from the G3 mesh (skip_roots=True in adapter) but kept in
    # the simulation for water uptake.
    if enable_photosynthesis:
        depth = float(soil_max_b[2] - soil_min_b[2])
        soil_domain = pb.SDF_PlantContainer(np.inf, np.inf, depth, True)
        plant.setGeometry(soil_domain)

        # Shift the soil grid so it's centered on the plant's XY seed
        # position (see init_plant for rationale). Without this shift the
        # entire root system can map to cellidx=-1 if the XML places seedPos
        # outside the soil_min_b/max_b box (e.g. maize_calibrated.xml has
        # seedPos=(200, 200, -3)).
        srp = plant.getOrganRandomParameter(pb.OrganTypes.seed, 0)
        sx, sy = float(srp.seedPos.x), float(srp.seedPos.y)
        shifted_min = (soil_min_b[0] + sx, soil_min_b[1] + sy, soil_min_b[2])
        shifted_max = (soil_max_b[0] + sx, soil_max_b[1] + sy, soil_max_b[2])

        # Upstream-canonical 3D pattern (Phase 3.5,
        # PLAN_DUMUX_INTEGRATION_2026-05-05.md): setRectangularGrid delegates
        # the seg→cell mapping to MappedSegments::soil_index_, replacing the
        # hand-rolled _picker lambda + setSoilGrid pair retired in this phase.
        plant.setRectangularGrid(
            pb.Vector3d(*shifted_min),
            pb.Vector3d(*shifted_max),
            pb.Vector3d(*soil_cell_number),
            False,  # cut=False: no segment splitting at cell boundaries (preserves
                    # legacy graph for parity; flip to True only when per-cell
                    # uptake granularity at sub-segment scale matters)
        )

    plant.initialize()

    # Resolve daily-met source. None → auto-load default CSV (may return None
    # when no CSV is configured); falls through to T_air_default in that case.
    from ..carbon.dvs_partitioning import get_daily_met
    met_lookup = get_daily_met(daily_met=daily_met) if daily_met is None else daily_met
    if met_lookup is not None:
        n_met_days = len(met_lookup)
        print(f"  Met forcing: {n_met_days} days of daily T_mean available for TT accumulator")
    else:
        print(f"  Met forcing: none found — using constant T_air={T_air_default} C")

    # Use incremental simulation with error recovery.
    # CPlantBox has a vector bounds bug with >8 leaf subtypes during initial
    # lateral creation. Incremental steps + catch allow partial growth.
    dt = 1.0  # 1-day steps
    total_simulated = 0.0
    while total_simulated < simulation_time:
        step = min(dt, simulation_time - total_simulated)
        # Feed today's T_mean to the CPlantBox TT accumulator.
        # Sim-day convention: 1-based (day 1 = first 24h of growth).
        sim_day_1b = int(total_simulated) + 1
        if met_lookup is not None:
            day_met = met_lookup.get(sim_day_1b)
            T_air = float(day_met['T_mean_C']) if day_met else T_air_default
        else:
            # Honor a legacy per-plant override if something set it upstream.
            T_air = getattr(plant, '_current_T_air', T_air_default)
        if hasattr(plant, 'setAirTemperature'):
            plant.setAirTemperature(T_air)
        try:
            plant.simulate(step, verbose=(total_simulated == 0))
            total_simulated += step
        except (IndexError, RuntimeError) as e:
            print(f"  Warning: simulate() error at day {total_simulated + step:.1f}: {e}")
            print(f"  Continuing with {total_simulated:.1f} days simulated")
            # Re-sync nodes after error
            try:
                plant.simulate(0.0)
            except Exception:
                pass
            break

    organs = plant.getOrgans()
    n_stems = sum(1 for o in organs if o.organType() == pb.OrganTypes.stem)
    n_leaves = sum(1 for o in organs if o.organType() == pb.OrganTypes.leaf)
    n_roots = sum(1 for o in organs if o.organType() == pb.OrganTypes.root)

    print(f"\n  Stems: {n_stems}, Leaves: {n_leaves}, Roots: {n_roots} (excluded from G3)")
    print(f"  Total nodes: {len(plant.getNodes())}")

    # Print per-leaf stats for verification
    leaf_organs = [o for o in organs if o.organType() == pb.OrganTypes.leaf]
    if leaf_organs:
        print(f"\n  Per-leaf summary:")
        print(f"  {'#':>3} {'SubType':>7} {'Length':>8} {'Nodes':>5}")
        for j, leaf in enumerate(leaf_organs):
            st = leaf.getParameter("subType")
            length = leaf.getLength(False)
            n_nodes = len(leaf.getNodes())
            print(f"  {j:>3} {st:>7.0f} {length:>8.1f} cm {n_nodes:>5}")

    return plant


def extract_g3_mesh(plant, min_stem_nodes=50, min_leaf_nodes=20, stem_res=16,
                    include_roots=False, use_nurbs_leaf_backend=False,
                    nurbs_leaf_n_u_eval=30, nurbs_leaf_n_v_eval=7):
    """Extract G1 skeleton from CPlantBox and loft to G3 mesh.

    Args:
        include_roots: If True, include root geometry in the mesh.
                       Default False (shoot only, roots excluded for DART).
        use_nurbs_leaf_backend: If True, loft leaves via the canonical 11×5
            PlantGL NurbsPatch backend (experimental — off by default).
        nurbs_leaf_n_u_eval, nurbs_leaf_n_v_eval: Tessellation resolution
            for the NURBS backend.
    """
    print(f"\n=== Extracting G3 Mesh ===")

    organ_dicts = extract_organs_for_lofter(
        plant,
        min_stem_nodes=min_stem_nodes,
        min_leaf_nodes=min_leaf_nodes,
        skip_roots=not include_roots,
        **get_plantsim_feature_kwargs_from_env(),
    )

    label = "shoot + root" if include_roots else "shoot only"
    print(f"  Extracted {len(organ_dicts)} organs ({label})")

    mesh = loft_organs(
        organ_dicts,
        stem_sides=stem_res,
        use_nurbs_backend=use_nurbs_leaf_backend,
        nurbs_n_u_eval=nurbs_leaf_n_u_eval,
        nurbs_n_v_eval=nurbs_leaf_n_v_eval,
    )

    print(f"  Vertices: {mesh.n_vertices}, Triangles: {mesh.n_triangles}")

    return mesh, organ_dicts


def extract_root_dicts(plant, min_root_nodes=20):
    """Extract root organ dicts for visualization."""
    root_dicts = extract_organs_for_lofter(
        plant,
        min_stem_nodes=min_root_nodes,
        min_leaf_nodes=min_root_nodes,
        skip_roots=False
    )
    # Keep only roots
    return [o for o in root_dicts if o['type'] == 'root']


def run_photosynthesis(plant, sim_time, output_prefix,
                       par_umol=1000.0, tair_c=25.0, rh=0.7,
                       soil_psi_cm=-500.0,
                       soil_psi_provider=None):
    """Set up hydraulics + C4 photosynthesis and run a single solve.

    Uses:
      - couvreur2012.json  : maize root hydraulics (Doussan 1998 via Couvreur 2012)
      - maize_C4_photosynthesis_parameters.json : PhotoType=1, alpha=0.05

    @param plant          pb.MappedPlant (grown, with soil grid)
    @param sim_time       days simulated (for age-dependent conductivities)
    @param output_prefix  path prefix for CSV output
    @param par_umol       PAR [umol photons m-2 s-1] — uniform over all leaves
    @param tair_c         Air / leaf temperature [°C]
    @param rh             Relative humidity [0–1]
    @param soil_psi_cm    Uniform soil water potential [cm] — -500 = well-watered
    """
    print(f"\n=== Photosynthesis Solve ===")
    print(f"  PAR={par_umol} umol m-2 s-1, T={tair_c}°C, RH={rh*100:.0f}%")
    print(f"  Soil psi={soil_psi_cm} cm  (hydraulics: couvreur2012 / C4 params)")

    from plantbox.functional.phloem_flux import PhloemFluxPython
    from plantbox.functional.PlantHydraulicParameters import PlantHydraulicParameters

    # --- Hydraulic parameters ---
    params = PlantHydraulicParameters()
    params.read_parameters(get_hydraulics_json())

    # --- Photosynthesis + phloem model ---
    hm = PhloemFluxPython(plant, params)
    hm.read_photosynthesis_parameters(filename=get_photosynthesis_json())
    hm.read_phloem_parameters(filename=get_phloem_json())

    # Per-segment Chl from LOPS per-position profiles
    chl_per_seg = get_chl_per_segment(sim_time, plant)
    seg_leaves_check = plant.getSegmentIds(4)
    if len(chl_per_seg) == len(seg_leaves_check):
        hm.Chl = chl_per_seg
        cab_min, cab_max = min(chl_per_seg), max(chl_per_seg)
        vcmax_range = f"[{vcmax25_from_cab(cab_min):.1f}, {vcmax25_from_cab(cab_max):.1f}]"
        print(f"  PhotoType={'C4' if hm.PhotoType == 1 else 'C3'}, "
              f"Vcmax range={vcmax_range} umol m-2 s-1 "
              f"(Cab range=[{cab_min:.1f}, {cab_max:.1f}] ug/cm2, "
              f"{len(chl_per_seg)} segs)")
    else:
        vcmax_umol = (hm.VcmaxrefChl1 * hm.Chl[0] + hm.VcmaxrefChl2)
        print(f"  PhotoType={'C4' if hm.PhotoType == 1 else 'C3'}, "
              f"Vcmax~{vcmax_umol:.1f} umol m-2 s-1 (Chl={hm.Chl[0]:.1f} ug/cm2)")

    # --- Soil water potential vector ---
    if soil_psi_provider is None:
        from ..hydraulics.soil_psi import FixedSoilPsi
        soil_psi_provider = FixedSoilPsi(psi_cm=soil_psi_cm)
    n_cells = int(soil_psi_provider.n_cells_total)
    p_s = soil_psi_provider.get_profile(t_days=float(sim_time), depth_cm=n_cells)

    # --- Weather ---
    es = hm.get_es(tair_c)
    ea = es * rh

    # PAR conversion: umol m-2 s-1  → mol cm-2 d-1
    par_mol_cm2_d = par_umol * 1e-6 * 86400 * 1e-4

    # --- Solve photosynthesis + hydraulics ---
    try:
        hm.solve(
            sim_time=sim_time,
            rsx=p_s,
            cells=True,
            ea=ea,
            es=es,
            PAR=par_mol_cm2_d,
            TairC=tair_c,
            verbose=0,
        )
    except Exception as e:
        print(f"  ERROR in hm.solve(): {e}")
        return None

    # Close the soil↔plant water loop: aggregate per-segment radial fluxes
    # into a per-cell sink and feed it back to the provider. No-op for
    # FixedSoilPsi/BucketSoilPsi.
    from ..hydraulics.soil_psi import push_rwu_sink_to_provider
    push_rwu_sink_to_provider(hm, sim_time, p_s, soil_psi_provider,
                              n_cells=n_cells, verbose=False)

    # --- Results ---
    # NB: get_net_assimilation() returns per-leaf-segment (size = n_leaf_segs),
    # NOT per-all-segments.  Indexed 0..n_leaf-1, matching seg_leaves_idx order.
    An_leaf  = np.array(hm.get_net_assimilation())       # mol CO2 d-1 per leaf seg
    An_per   = np.array(hm.get_net_assimilation_perleafBladeArea())  # mol CO2 cm-2 d-1
    hx_all   = np.array(hm.get_water_potential())
    transp   = np.sum(hm.get_transpiration()) / 18 * 1e3  # mmol H2O d-1

    An_total_mmol = np.sum(An_leaf) * 1e3  # mmol CO2 d-1 whole plant
    n_leaf_segs = len(An_leaf)

    # Convert An_per to umol m-2 s-1:  mol cm-2 d-1 * 1e4 cm2/m2 / 86400 s/d * 1e6
    An_per_umol = An_per * 1e4 / 86400 * 1e6  # umol CO2 m-2 s-1

    print(f"\n  --- Results ---")
    print(f"  Total net assimilation : {An_total_mmol:.3f} mmol CO2 d-1")
    print(f"  Total transpiration    : {transp:.3f} mmol H2O d-1")
    print(f"  Leaf-blade segments    : {n_leaf_segs}")
    if n_leaf_segs > 0:
        nonzero = An_per_umol[An_per_umol > 0]
        print(f"  Active segments        : {len(nonzero)} / {n_leaf_segs}")
        if len(nonzero) > 0:
            print(f"  Mean An (active)       : {np.mean(nonzero):.2f} umol CO2 m-2 s-1")
            print(f"  Min/Max An             : {np.min(nonzero):.2f} / {np.max(nonzero):.2f} umol m-2 s-1")
    print(f"  Mean xylem psi         : {np.mean(hx_all):.0f} cm")

    # --- Per-leaf organ summary ---
    # Use plant.organTypes and plant.subTypes arrays (aligned with getSegments())
    # to map An values to individual leaf organs.  get_net_assimilation() returns
    # one value per leaf segment, ordered by their position in getSegments()
    # filtered to organType==4.
    ot_arr = np.array(plant.organTypes)   # per-segment organ type
    st_arr = np.array(plant.subTypes)     # per-segment sub type
    leaf_mask = (ot_arr == 4)             # True for leaf segments
    leaf_global_indices = np.where(leaf_mask)[0]  # global seg indices of leaves
    # An arrays are indexed 0..n_leaf-1, same order as leaf_global_indices
    assert len(leaf_global_indices) == n_leaf_segs, \
        f"Mismatch: {len(leaf_global_indices)} vs {n_leaf_segs}"

    # Map global seg index -> An array index (for leaf segs only)
    global_to_an = {int(gi): ai for ai, gi in enumerate(leaf_global_indices)}

    organs = plant.getOrgans()
    leaf_organs = [o for o in organs if o.organType() == pb.OrganTypes.leaf]
    lbs = np.array(plant.leafBladeSurface)

    print(f"\n  Per-leaf An:")
    print(f"  {'#':>3} {'SubType':>7} {'Length':>8} {'Segs':>5} "
          f"{'An_mean_umol':>12} {'An_sum_mmol':>12}")

    organ_data = []
    for j, leaf in enumerate(leaf_organs):
        st = int(leaf.getParameter("subType"))
        length = leaf.getLength(False)
        width = leaf.getParameter("Width_blade")

        # Find An indices for this organ's segments via subType match
        organ_leaf_mask = leaf_mask & (st_arr == st)
        organ_global_indices = np.where(organ_leaf_mask)[0]
        an_indices = [global_to_an[int(gi)] for gi in organ_global_indices
                      if int(gi) in global_to_an]

        # Blade area for this organ
        blade_area = sum(lbs[gi] for gi in organ_global_indices if gi < len(lbs)) * 2

        if an_indices:
            An_org = An_per_umol[an_indices]
            An_mean = np.mean(An_org)
            An_sum  = np.sum(An_leaf[an_indices]) * 1e3
            psi_segs = [hx_all[gi] for gi in organ_global_indices
                        if 0 <= gi < len(hx_all)]
            psi_mean = np.mean(psi_segs) if psi_segs else 0.0
        else:
            An_mean = 0.0
            An_sum  = 0.0
            psi_mean = 0.0

        print(f"  {j:>3} {st:>7} {length:>8.1f} cm {len(an_indices):>5} "
              f"{An_mean:>12.2f} {An_sum:>12.4f}")

        organ_data.append({
            'index': j, 'subtype': st, 'length': length,
            'width': width, 'blade_area': blade_area,
            'An_sum_mmol': An_sum, 'psi_mean': psi_mean,
            'n_segs': len(an_indices),
        })

    # --- Save CSV (leaf segments only) ---
    if output_prefix is not None:
        csv_path = Path(output_prefix).with_suffix('.csv')
        header = "leaf_seg_idx,global_seg_idx,An_mol_d,An_umol_m2_s,psi_cm"
        rows = []
        for i in range(n_leaf_segs):
            gi = int(leaf_global_indices[i])
            psi = hx_all[gi] if 0 <= gi < len(hx_all) else 0.0
            rows.append(f"{i},{gi},{An_leaf[i]:.6e},{An_per_umol[i]:.4f},{psi:.2f}")
        csv_path.write_text(header + "\n" + "\n".join(rows))
        print(f"\n  CSV: {csv_path} ({len(rows)} leaf segments)")

        # --- Plot ---
        from .render import plot_photosynthesis
        plot_photosynthesis(
            organ_data=organ_data,
            An_per_umol=An_per_umol,
            hx_all=hx_all,
            seg_leaves_idx=list(leaf_global_indices),
            An_total_mmol=An_total_mmol,
            transp=transp,
            par_umol=par_umol,
            tair_c=tair_c,
            rh=rh,
            sim_time=sim_time,
            output_prefix=output_prefix,
        )

    return hm


def export_mesh(mesh, output_prefix, compact_obj=True):
    """Export G3 mesh to OBJ + JSON mapping files.

    With ``compact_obj=True`` (default) the OBJ omits per-vertex normals
    and UVs and uses 4-decimal float precision — DART/Baleno don't read
    either field, and the saving is ~65 % file size with no geometry
    change. Pass ``compact_obj=False`` for the full-fat encoding (kept
    for debugging / external tools that want vertex normals).
    """
    from ..geometry.g1_to_g3 import COMPACT_OBJ_KWARGS
    output_dir = Path(output_prefix).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    obj_path = Path(output_prefix).with_suffix('.obj')
    json_path = Path(output_prefix).with_suffix('.json')

    obj_kwargs = COMPACT_OBJ_KWARGS if compact_obj else {}
    mesh.to_obj(str(obj_path), group_by_organ=True, **obj_kwargs)
    mesh.to_mapping_json(str(json_path))

    print(f"\n=== Exported ===")
    print(f"  OBJ:  {obj_path} ({mesh.n_triangles} triangles)")
    print(f"  JSON: {json_path} ({len(mesh.organ_meta)} organs)")


def export_g1_skeleton(plant, output_prefix):
    """Export G1 skeleton as thin-tube OBJ."""
    organ_dicts = extract_organs_for_lofter(
        plant, min_stem_nodes=2, min_leaf_nodes=2, skip_roots=True,
    )

    for organ in organ_dicts:
        organ['widths'] = np.full(len(organ['skeleton']), 0.04)

    g1_mesh = loft_organs(organ_dicts, stem_sides=4)

    g1_path = Path(output_prefix).with_name(
        Path(output_prefix).stem + '_g1'
    ).with_suffix('.obj')
    from ..geometry.g1_to_g3 import COMPACT_OBJ_KWARGS
    g1_mesh.to_obj(str(g1_path), group_by_organ=True, **COMPACT_OBJ_KWARGS)

    print(f"  G1 OBJ: {g1_path} ({g1_mesh.n_triangles} triangles)")
    return g1_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description='Grow maize plant with calibrated parameters and extract G3 mesh'
    )
    parser.add_argument('--xml', required=True, help='Path to calibrated maize.xml')
    parser.add_argument('--days', type=float, default=30, help='Simulation time (days)')
    parser.add_argument('--output', required=True, help='Output prefix (e.g., maize_day30)')
    parser.add_argument('--resolution', choices=['coarse', 'medium', 'fine', 'ultra'],
                       default='fine', help='Mesh resolution')
    parser.add_argument('--export-g1', action='store_true',
                       help='Export G1 skeleton OBJ alongside G3 mesh')
    parser.add_argument('--no-png', action='store_true',
                       help='Skip PNG rendering (only export OBJ + JSON)')
    parser.add_argument('--svg', action='store_true',
                       help='Export SVG vector graphic (G1 skeleton | G3 mesh side-by-side)')
    parser.add_argument('--publication', action='store_true',
                       help='Export publication-quality SVG (white bg, scale bar, (a)/(b) labels)')
    parser.add_argument('--animate', action='store_true',
                       help='Export animated SVG showing growth from day 1 to --days')
    parser.add_argument('--animate-step', type=int, default=5,
                       help='Days between animation frames (default: 5)')
    parser.add_argument('--frame-dur', type=float, default=0.5,
                       help='Seconds per animation frame (default: 0.5)')
    parser.add_argument('--photosynthesis', action='store_true',
                       help='Run C4 photosynthesis solve after growth (requires soil grid)')
    parser.add_argument('--par', type=float, default=1000.0,
                       help='PAR for photosynthesis solve [umol m-2 s-1] (default: 1000)')
    parser.add_argument('--tair', type=float, default=25.0,
                       help='Air temperature for photosynthesis solve [°C] (default: 25)')
    parser.add_argument('--rh', type=float, default=0.7,
                       help='Relative humidity for photosynthesis solve [0-1] (default: 0.7)')
    parser.add_argument('--leuning', action='store_true',
                       help='(deprecated — PhloemFluxPython is now the default solver)')
    parser.add_argument('--include-roots-in-mesh', action='store_true',
                       help='Include root geometry in G3 mesh export (default: shoot only)')
    parser.add_argument('--cp-donor-seed', type=int, default=None,
                       help='Seed selecting an MF3D donor plant whose leaf CPs '
                            'overwrite the XML defaults at runtime. None → no swap.')
    parser.add_argument('--cp-donor-mode', type=str, default='draw_coherent',
                       choices=['draw_coherent', 'draw', 'median'],
                       help='Donor reducer mode (default: draw_coherent — one '
                            'MF3D plant covers all positions).')
    parser.add_argument('--no-auto-stage', action='store_true',
                       help='Disable automatic V-stage label appending to '
                            'output prefix (default: append _V<n> or '
                            '_VT_emerging|_VT_mature|_VT_senescent).')
    parser.add_argument('--no-compact-obj', action='store_true',
                       help='Emit full-fat OBJ (per-vertex normals + UVs, '
                            '6-decimal precision). Default is compact: '
                            'no vn/vt, 4-decimal — DART/Baleno consume '
                            'this identically and files are ~65%% smaller.')
    args = parser.parse_args()

    resolution_presets = {
        'coarse': {'min_stem_nodes': 30, 'min_leaf_nodes': 15, 'stem_res': 12},
        'medium': {'min_stem_nodes': 50, 'min_leaf_nodes': 20, 'stem_res': 16},
        'fine': {'min_stem_nodes': 100, 'min_leaf_nodes': 40, 'stem_res': 20},
        'ultra': {'min_stem_nodes': 200, 'min_leaf_nodes': 80, 'stem_res': 32}
    }

    preset = resolution_presets[args.resolution]

    print("=" * 60)
    print("CPlantBox -> G1 -> G3 Pipeline (Calibrated Growth)")
    print("=" * 60)

    # Grow plant
    plant = grow_plant(
        xml_path=args.xml,
        simulation_time=args.days,
        min_stem_nodes=preset['min_stem_nodes'],
        min_leaf_nodes=preset['min_leaf_nodes'],
        enable_photosynthesis=args.photosynthesis,
        cp_donor_seed=args.cp_donor_seed,
        cp_donor_mode=args.cp_donor_mode,
    )

    # Auto-stage label: append phenology stage to output prefix so output
    # filenames carry both day and V-stage. Driven by the same collar-count
    # the lofter uses for material assignment.
    if not args.no_auto_stage:
        from .phenology import detect_v_stage
        stage_label = detect_v_stage(plant)
        args.output = f"{args.output}_{stage_label}"
        print(f"  Phenology stage: {stage_label}")
        print(f"  Output prefix:   {args.output}")

    # Extract G3 mesh (also returns organ_dicts for rendering)
    mesh, organ_dicts = extract_g3_mesh(
        plant,
        min_stem_nodes=preset['min_stem_nodes'],
        min_leaf_nodes=preset['min_leaf_nodes'],
        stem_res=preset['stem_res'],
        include_roots=args.include_roots_in_mesh,
    )

    # Export OBJ + JSON
    export_mesh(mesh, args.output, compact_obj=not args.no_compact_obj)

    # Optional: G1 skeleton OBJ
    if args.export_g1:
        export_g1_skeleton(plant, args.output)

    # Extract root dicts for visualization
    root_dicts = extract_root_dicts(plant, min_root_nodes=preset.get('min_stem_nodes', 50) // 2)
    print(f"  Roots extracted: {len(root_dicts)} organs")

    # Render G1 | G3 | Roots comparison PNG (default on)
    if not args.no_png:
        from .render import render_comparison_png
        render_comparison_png(organ_dicts, mesh, args.output, args.days,
                             root_dicts=root_dicts)

    # Optional: SVG vector graphic
    if args.svg:
        from .render import render_comparison_svg
        render_comparison_svg(organ_dicts, mesh, args.output, args.days,
                              root_dicts=root_dicts)

    # Optional: publication-quality SVG
    if args.publication:
        from .render import render_publication_svg
        render_publication_svg(organ_dicts, mesh, args.output, args.days,
                               root_dicts=root_dicts)

    # Optional: animated SVG (runs its own simulation loop)
    if args.animate:
        from .render import render_animated_svg
        render_animated_svg(
            xml_path=args.xml,
            max_days=args.days,
            output_prefix=args.output,
            preset=preset,
            day_step=args.animate_step,
            frame_dur=args.frame_dur,
        )

    # Optional: C4 photosynthesis solve
    if args.photosynthesis:
        photo_prefix = args.output + '_photosynthesis'
        run_photosynthesis(
            plant=plant,
            sim_time=args.days,
            output_prefix=photo_prefix,
            par_umol=args.par,
            tair_c=args.tair,
            rh=args.rh,
        )

    print("\n" + "=" * 60)
    print("Pipeline complete")
    print("=" * 60)


if __name__ == '__main__':
    main()
