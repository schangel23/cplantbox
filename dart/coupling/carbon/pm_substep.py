"""PiafMunch substep-loop carbon partitioning (Gate Ch1.PM.4).

Mirrors the pm_notebook_loop.run_loop pattern (24 hourly substeps with
useCWGr=True + plant.simulate(dt) between substeps) and packages the
result into the same dict shape as ``solve_carbon_partitioning`` so the
diurnal pipeline's CSV writers, AgroC export, and summary JSON consume
PM output unchanged.

Plan-doc: ``Literature/Chapter 1/Concepts/CPBOXBALENOCOUPLING/
PLAN_PIAFMUNCH_CALIBRATION_2026-05-04.md``, Gate Ch1.PM.4.

The substep loop replaces the S5 quasi-steady solver call at
``photosynthesis/diurnal.py:_run_per_plant_carbon`` (line ~909) and the
carbon-feedback DART-day partitioning at
``run_production_series_carbon`` (line ~2166) when
``carbon_solver='pm'``. The default solver remains ``'s5'`` for Gate 4
so the existing production path stays bit-identical.

Q_out block layout (mirror of pm_notebook_loop.run_loop comment, derived
from ``src/external/PiafMunch/solve.cpp:191-207``):

    block 0 Q_ST, 1 Q_Mesophyll, 2 Q_RespMaint, 3 Q_Exudation,
    block 4 Q_Growthtot, 5 Q_RespMaintmax, 6 Q_Growthtotmax,
    block 7 Q_S_Mesophyll, 8 Q_S_ST, 9 Q_Mucil.

Mass balance (Gate Ch1.PM.3 closure, < 1 % residual on V3/day-55/day-130):

    dAn ≈ dRm + dGr + dExud + dQ_ST + dQ_meso + dQ_S_meso

Design choices vs ``pm_notebook_loop.case_maize``:
  * Builds a fresh ``PhloemFluxPython`` internally; does not reuse the
    diurnal-pipeline ``hm`` from ``run_photosynthesis``.
  * Calls ``enable_cw_limited_growth(plant, wrap_roots=False)``
    idempotently before the loop (Lock #6 + Lock #9 wrap policy from
    [[project_root_path_preservation]] / [[project_s5_sink_source_shipped]]).
  * Uses constant peak-condition PAR + Tair across all substeps, matching
    ``pm_notebook_loop.case_maize`` so production numbers are comparable
    to Gate 1-3 calibration runs. The 24-h cumulative AnSum that emerges
    from PM may differ from the diurnal pipeline's DART-coupled daily An;
    PM-organic numbers are reported (no post-scaling).
  * ``plant.simulate(dt)`` advances the plant during the loop. For the
    parametric path the plant is throwaway. For the carbon-feedback path
    the caller must skip any subsequent ``step_plant_carbon`` to avoid
    double-advancement.
"""
import os
from pathlib import Path

import numpy as np

# Suc → CO2: 1 mmol Suc fully oxidised → 12 mmol CO2 (matches phloem_steady).
SUC_TO_CO2 = 12.0

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_DIR = REPO_ROOT / "dart/coupling/scripts"


def _suppress_io():
    """Mirror pm_notebook_loop._suppress: silence PM C++ stdout/stderr."""
    o1 = os.dup(1)
    o2 = os.dup(2)
    dn = os.open(os.devnull, os.O_WRONLY)
    os.dup2(dn, 1)
    os.dup2(dn, 2)
    return o1, o2, dn


def _restore_io(o1, o2, dn):
    os.dup2(o1, 1)
    os.dup2(o2, 2)
    os.close(dn)
    os.close(o1)
    os.close(o2)


def _is_cw_wrapped(plant):
    """Idempotency probe: did ``enable_cw_limited_growth`` already run?

    The shipping ``enable_cw_limited_growth`` is not idempotent (its
    else-branch unconditionally overwrites with bare ``CWLimitedGrowth``,
    which would drop the FA demand from a previously-wrapped plant). Use
    this probe before calling the wrap helper a second time on a plant
    that may have come from ``run_production_series_carbon``'s persistent
    pool.
    """
    import plantbox as pb
    for ot in (3, 4):  # stem + leaf
        for param in plant.getOrganRandomParameter(ot):
            if param is None:
                continue
            if isinstance(getattr(param, "f_gf", None), pb.CWLimitedGrowth):
                return True
    return False


def solve_carbon_partitioning_pm(plant, An_per_leaf_seg, Tair_C=25.0,
                                  day=55, warm_start=None,
                                  gdd_accumulated=None,
                                  par_umol=600.0, soil_psi_cm=-500.0,
                                  n_substeps=24, advance_plant=True,
                                  pm_filename=None,
                                  pm_atol=1e-6, pm_rtol=1e-4,
                                  Vmaxloading=0.20, beta_loading=2.0,
                                  solver=32, soil_psi_provider=None):
    """Run a 24-substep PiafMunch loop and return an S5-shaped carbon dict.

    Args:
        plant: pb.MappedPlant grown to ``day``.
        An_per_leaf_seg: per-leaf-segment An [mol CO2/d/seg] from the
            diurnal pipeline. Stored as ``An_total_mmol_target`` in the
            returned dict; PM's internal photosynthesis produces its own
            ``An_total_mmol`` which may differ.
        Tair_C: constant air temperature [°C] across all substeps.
        day: simulation day; sets the substep loop's ``sim_init``.
        warm_start: ignored (PM resets internal state per loop).
        gdd_accumulated: ignored (kept for signature parity with
            ``solve_carbon_partitioning``).
        par_umol: constant PAR [μmol photons m⁻² s⁻¹]. Default 600 matches
            ``pm_notebook_loop.case_maize``.
        soil_psi_cm: uniform soil pressure head [cm]; default -500. Used
            only when ``soil_psi_provider`` is ``None``.
        n_substeps: substep count over a 1-day window. Default 24 (1 h).
        advance_plant: when True (default), ``plant.simulate(dt)`` runs
            inside the loop, consuming CW_Gr and advancing the plant by
            ~1 day under carbon-limited growth.
        pm_filename: scratch file path for ``hm.startPM``. Defaults to
            ``dart/coupling/scripts/_pm_substep_loop.txt``.
        pm_atol, pm_rtol: PiafMunch LSODA tolerances. Defaults match
            ``configure_maize`` from ``pm_notebook_loop``.
        Vmaxloading, beta_loading, solver: phloem solver overrides; same
            defaults as ``pm_notebook_loop.case_maize``.
        soil_psi_provider: optional ``SoilPsiProvider`` (see
            ``hydraulics.soil_psi``). When ``None`` (default), the loop
            uses a static 200-cell linspace gradient anchored at
            ``soil_psi_cm`` — bit-identical with the pre-Gate-Ch1.PMDM.1
            behaviour. When supplied, the loop reads
            ``provider.get_profile(t_days=sim)`` per substep AND pushes
            per-cell RWU sinks back via ``push_rwu_sink_to_provider``
            after each ``hm.solve`` (Gate Ch1.PMDM.2), so the next
            substep's ``get_profile`` advances ``DumuxSoilPsi`` against
            the previous substep's transpiration sink — closing the
            soil↔plant water loop. ``FixedSoilPsi`` / ``BucketSoilPsi``
            short-circuit the sink push to a no-op, so passing one of
            those is functionally equivalent to ``soil_psi_provider=None``
            (modulo their static return shape). The caller is
            responsible for aligning ``provider._t_last_days`` with
            ``day`` before this function is called (the production
            diurnal pipeline owns this in G5).

    Returns:
        carbon_result dict with the same keys as
        ``QuasiSteadyPhloem.solve``, plus PM-specific instrumentation
        from Gate Ch1.PM.3 + plant-water conservation from Gate
        Ch1.PMDM.3:

            - ``An_total_mmol_target``: caller's daily target (mmol CO2)
            - ``sum_Q_S_meso``: 24-h end Σ Q_S_Mesophyll (mmol Suc)
            - ``dQ_S_meso``, ``dQ_meso``, ``dQ_ST``: 24-h sucrose-pool
              deltas (mmol Suc/d)
            - ``mass_balance_residual_pct``
            - ``integrated_rwu_cm3``: 24-h integrated root water uptake
              (∑ root-segment radial fluxes for segs with cellidx≥0,
              cm³). Negative = water left the soil → roots. Tracked
              uniformly across all provider modes.
            - ``integrated_transpiration_cm3``: 24-h integrated leaf
              transpiration (∑hm.get_transpiration(), cm³). Positive.
            - ``rwu_transpiration_residual_pct``:
              100·|∫RWU+∫Ev|/|∫Ev| — the steady-state plant water
              balance closure. Gate Ch1.PMDM.3 expects < 2 % under
              well-watered DuMux IC.

        Returns ``None`` on solver failure (caller falls back to S5 or
        marks the plant as no-result, mirroring S5's exception path).
    """
    from plantbox.functional.phloem_flux import PhloemFluxPython
    from plantbox.functional.PlantHydraulicParameters import (
        PlantHydraulicParameters,
    )
    from ..config import (
        get_hydraulics_json, get_photosynthesis_json, get_phloem_json,
    )
    from ..growth.carbon_growth import enable_cw_limited_growth

    # Lock #6 + Lock #9 wrap. Skipped if a previous caller (e.g.
    # ``run_production_series_carbon`` at line 1643) already wrapped the
    # plant, since ``enable_cw_limited_growth`` is not idempotent under
    # double-call.
    if not _is_cw_wrapped(plant):
        enable_cw_limited_growth(plant, wrap_roots=False, wrap_fa=True)

    # Hydraulics + phloem + photosynthesis configuration
    # (mirror of pm_notebook_loop.case_maize cell pattern).
    params_h = PlantHydraulicParameters()
    params_h.read_parameters(get_hydraulics_json())
    hm = PhloemFluxPython(plant, params_h, psiXylInit=-500, ciInit=350e-6)
    hm.read_photosynthesis_parameters(filename=get_photosynthesis_json())
    hm.read_phloem_parameters(filename=get_phloem_json())
    hm.atol = pm_atol
    hm.rtol = pm_rtol
    hm.Vmaxloading = Vmaxloading
    hm.beta_loading = beta_loading
    hm.solver = solver

    sim_init = float(day)
    dt = 1.0 / float(n_substeps)
    sim_max = sim_init + 1.0 - 0.5 * dt  # n_substeps total iterations

    # Soil psi profile.
    #
    # Default (soil_psi_provider=None): same shape as
    # pm_notebook_loop.case_maize — 200 layers from soil_psi_cm to
    # soil_psi_cm-200 (well-watered top → drier bottom), built once
    # before the loop. Bit-identical with pre-Gate-Ch1.PMDM.1 behaviour
    # and with all Gate 1-5 calibration runs.
    #
    # Provider branch (Gate Ch1.PMDM.2): when a SoilPsiProvider is
    # supplied, the static build is skipped; ``p_s`` is refreshed inside
    # the substep loop via ``soil_psi_provider.get_profile(t_days=sim)``
    # *and* the per-segment radial fluxes from the previous solve are
    # aggregated into per-cell sinks via ``push_rwu_sink_to_provider``
    # (see ``hydraulics.soil_psi``). For static providers this push is
    # a documented no-op so ``--soil-mode fixed`` stays bit-identical
    # with the linspace branch (modulo provider return shape). For
    # ``DumuxSoilPsi`` the push closes the soil↔plant water loop:
    # substep N's RWU sink drives substep N+1's RichardsSP advance
    # inside ``get_profile``.
    from ..hydraulics.soil_psi import push_rwu_sink_to_provider

    if soil_psi_provider is None:
        p_s = np.linspace(soil_psi_cm, soil_psi_cm - 200.0, 200)
        n_cells = 200
    else:
        # ``p_s`` is rebuilt fresh per substep inside the loop; seed with
        # an empty array of the provider's length so the type checker
        # tracks ``np.ndarray`` cleanly across the get_profile / solve /
        # push hand-off.
        n_cells = int(soil_psi_provider.n_cells_total)
        p_s = np.empty(n_cells, dtype=float)

    # PAR conversion: μmol m⁻² s⁻¹ → mol cm⁻² d⁻¹
    par_mol_cm2_d = par_umol * 1e-6 * 86400 * 1e-4

    if pm_filename is None:
        SCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
        pm_filename = str(SCRIPTS_DIR / "_pm_substep_loop.txt")

    Tair_K = float(Tair_C) + 273.15
    es = hm.get_es(float(Tair_C))
    ea = es * 0.6  # constant RH=60% across substeps

    AnSum_suc = 0.0
    nt_first = -1
    Q_ST_first = 0.0
    Q_meso_first = 0.0
    Q_S_meso_first = 0.0

    # Gate Ch1.PMDM.3 conservation diagnostics: accumulate per-substep
    # transpiration (sum of leaf radial fluxes) and root water uptake
    # (sum of root-segment radial fluxes mapped to a soil cell). Tracked
    # for every provider mode so the returned dict always exposes the
    # plant-side water balance — meaningful for DumuxSoilPsi (where
    # ψ_s evolves), informational for static providers.
    integrated_transp_cm3 = 0.0
    integrated_rwu_cm3 = 0.0

    sim = sim_init
    n_done = 0
    while sim <= sim_max + 1e-9:
        # Refresh per-substep soil profile when a provider is supplied.
        # Static providers (Fixed/Bucket) return a length-validated array
        # bit-identically to the legacy linspace; DumuxSoilPsi advances
        # its internal RichardsSP solver here using whatever sink was
        # registered by the previous substep's
        # ``push_rwu_sink_to_provider`` call (zero on substep 0 because
        # nothing has been pushed yet — first DuMux advance runs against
        # an empty source).
        if soil_psi_provider is not None:
            p_s = soil_psi_provider.get_profile(t_days=float(sim))

        # 1. Photosynthesis solve at this substep.
        fdpair = _suppress_io()
        try:
            hm.solve(sim_time=sim, rsx=p_s, cells=True, ea=ea, es=es,
                     PAR=par_mol_cm2_d, TairC=float(Tair_C), verbose=0)
        except Exception as e:
            _restore_io(*fdpair)
            print(f"  PM-substep solve error at sim={sim:.4f}: {e}")
            return None
        else:
            _restore_io(*fdpair)

        # 1b. Close the soil↔plant water loop (Gate Ch1.PMDM.2): aggregate
        # per-segment radial fluxes into per-cell sinks and push to the
        # provider. The next substep's get_profile reads the updated ψ_s.
        # No-op for FixedSoilPsi/BucketSoilPsi (helper short-circuits on
        # static providers), keeping --soil-mode fixed bit-identical.
        if soil_psi_provider is not None:
            push_rwu_sink_to_provider(
                hm, float(sim), p_s, soil_psi_provider, n_cells=n_cells,
            )

        # 1c. Conservation diagnostics (Gate Ch1.PMDM.3). Mirrors the
        # manual aggregation inside push_rwu_sink_to_provider so the
        # numbers are tracked uniformly across all provider modes (the
        # push helper short-circuits to {} on static providers).
        # Steady-state plant water balance: ∑(leaf Ev) ≈ −∑(root RWU);
        # the daily integrals of both signals are returned for the
        # G3 closure assertion.
        ev_arr = np.asarray(hm.get_transpiration(), dtype=float)
        integrated_transp_cm3 += float(np.sum(ev_arr)) * dt
        out_flux_arr = np.asarray(hm.radial_fluxes(), dtype=float)
        seg_ot_arr = np.asarray(hm.ms.organTypes, dtype=int)
        rwu_substep_cm3_d = 0.0
        for s_idx, c_idx in hm.ms.seg2cell.items():
            if c_idx >= 0 and int(seg_ot_arr[s_idx]) == 2:
                rwu_substep_cm3_d += float(out_flux_arr[s_idx])
        integrated_rwu_cm3 += rwu_substep_cm3_d * dt

        # 2. Accumulate An (notebook pattern: AnSum += sum(Ag4Phloem) * dt).
        Ag = np.array(hm.Ag4Phloem)
        AnSum_suc += float(np.sum(Ag)) * dt

        # 3. PiafMunch substep.
        fdpair = _suppress_io()
        try:
            ret = hm.startPM(sim, sim + dt, 1, Tair_K, True, pm_filename)
        finally:
            _restore_io(*fdpair)
        if ret != 1:
            print(f"  PM-substep startPM returned {ret} at sim={sim:.4f}")
            return None

        # Capture initial-substep state once (delta-storage accounting).
        # Re-read nt each substep: plant grows under advance_plant=True so
        # node count rises across the loop. Initial Q_* totals are taken
        # at the FIRST substep's nt; final totals at the LAST substep's
        # (larger) nt — this is sucrose mass on the entire plant graph at
        # each instant, not per-node, so the comparison stays valid.
        if nt_first < 0:
            nt_first = len(plant.getNodes())
            Q_ST_first = float(np.sum(np.array(hm.Q_out[0:nt_first])))
            Q_meso_first = float(np.sum(np.array(
                hm.Q_out[nt_first:(2 * nt_first)])))
            Q_S_meso_first = float(np.sum(np.array(
                hm.Q_out[(7 * nt_first):(8 * nt_first)])))

        # 4. plant.simulate(dt) consumes CW_Gr (carbon-limited growth).
        if advance_plant:
            fdpair = _suppress_io()
            try:
                plant.simulate(dt, False)
            finally:
                _restore_io(*fdpair)

        sim += dt
        n_done += 1

    if n_done == 0:
        return None

    # Final-substep readout (re-read nt; plant may have grown).
    nt = len(plant.getNodes())
    Q_ST_arr = np.array(hm.Q_out[0:nt])
    Q_meso_arr = np.array(hm.Q_out[nt:(2 * nt)])
    Q_Rm_arr = np.array(hm.Q_out[(2 * nt):(3 * nt)])
    Q_Exud_arr = np.array(hm.Q_out[(3 * nt):(4 * nt)])
    Q_Gr_arr = np.array(hm.Q_out[(4 * nt):(5 * nt)])
    Q_Grmax_arr = np.array(hm.Q_out[(6 * nt):(7 * nt)])
    Q_S_meso_arr = np.array(hm.Q_out[(7 * nt):(8 * nt)])
    C_ST_arr = np.array(hm.C_ST)

    # Per-segment organ types → per-node organ types (root/stem/leaf masks).
    # plant.organTypes is per-segment (length n_segs); Q_out is per-node
    # (length nt = n_segs + 1). Map seg organType to its child node, the
    # same convention pm_notebook_loop._audit_psixyl_summary uses.
    seg_ot = np.array(plant.organTypes, dtype=int)
    n_segs = seg_ot.size
    node_ot = np.zeros(nt, dtype=int)
    if n_segs + 1 == nt:
        node_ot[1:] = seg_ot
    else:
        m = min(n_segs, nt - 1)
        node_ot[1:m + 1] = seg_ot[:m]

    mask_root = node_ot == 2
    mask_stem = node_ot == 3
    mask_leaf = node_ot == 4

    # Cumulative-from-zero deltas (Q_*bu == 0 at substep 1 in the notebook
    # pattern, and PiafMunch's Q_RespMaint / Q_Exudation / Q_Growthtot are
    # cumulative from the loop start).
    dRm_root = float(np.sum(Q_Rm_arr[mask_root]))
    dRm_stem = float(np.sum(Q_Rm_arr[mask_stem]))
    dRm_leaf = float(np.sum(Q_Rm_arr[mask_leaf]))
    dRm_total = dRm_root + dRm_stem + dRm_leaf

    dGr_root = float(np.sum(Q_Gr_arr[mask_root]))
    dGr_stem = float(np.sum(Q_Gr_arr[mask_stem]))
    dGr_leaf = float(np.sum(Q_Gr_arr[mask_leaf]))
    dGr_total = dGr_root + dGr_stem + dGr_leaf

    dExud_total = float(np.sum(Q_Exud_arr))

    # 24h sucrose-pool deltas.
    dQ_ST = float(np.sum(Q_ST_arr)) - Q_ST_first
    dQ_meso = float(np.sum(Q_meso_arr)) - Q_meso_first
    dQ_S_meso = float(np.sum(Q_S_meso_arr)) - Q_S_meso_first
    dStorage = dQ_ST + dQ_meso + dQ_S_meso

    # FR fractions — S5 convention: storage attributed to FR_stem so the
    # CSV / JSON writers don't need to learn a PM-specific schema. (PM's
    # storage is biologically mostly leaf mesophyll; for analysis the raw
    # dQ_meso / dQ_S_meso are still in the dict.)
    total_usage = dRm_total + dGr_total + dExud_total + dStorage
    if total_usage > 0:
        FR_leaf = (dRm_leaf + dGr_leaf) / total_usage
        FR_stem = (dRm_stem + dGr_stem + dStorage) / total_usage
        FR_root = (dRm_root + dGr_root + dExud_total) / total_usage
        FR_storage = 0.0
    else:
        FR_leaf = FR_stem = FR_root = FR_storage = 0.0

    # Mass balance vs PM-organic AnSum (Gate Ch1.PM.3 closure check).
    if AnSum_suc > 0:
        mb_residual = abs(
            AnSum_suc - (dRm_total + dGr_total + dExud_total + dStorage)
        ) / AnSum_suc
    else:
        mb_residual = 0.0

    # Gate Ch1.PMDM.3 plant-water residual: |∫RWU + ∫Ev| / |∫Ev|.
    # Steady-state expectation: ∑root_radial_flux_per_cell ≈ −∑Ev so the
    # signed sum is near zero. Reported even when ∫Ev is tiny (V0 plants
    # before leaf emergence) — guarded against div-by-zero.
    if abs(integrated_transp_cm3) > 1e-12:
        rwu_transp_residual = abs(
            integrated_rwu_cm3 + integrated_transp_cm3
        ) / abs(integrated_transp_cm3)
    else:
        rwu_transp_residual = 0.0

    # Convert sucrose → CO2 for the S5 contract. Keep root_exud_mmol_d in
    # mmol Suc (downstream AgroC export expects sucrose for kg-C-via-molar
    # conversion, matching the S5 path).
    S = SUC_TO_CO2
    An_total_mmol_target = float(np.sum(An_per_leaf_seg)) * 1000.0  # mol→mmol

    return {
        # S5-shape (mmol CO2 unless otherwise noted) ----------------------
        "Rm_total_mmol": dRm_total * S,
        "Rm_leaf": dRm_leaf * S,
        "Rm_stem": dRm_stem * S,
        "Rm_root": dRm_root * S,
        "Rm_storage": 0.0,
        "Rg_total_mmol": dGr_total * S,
        "stem_storage_mmol": dStorage * S,
        "FR_leaf": FR_leaf,
        "FR_stem": FR_stem,
        "FR_root": FR_root,
        "FR_storage": FR_storage,
        "root_resp_profile_mmol_d": np.array([dRm_root * S]),
        "root_exud_mmol_d": np.array([dExud_total]),  # mmol Suc/d
        "root_dead_mmol_d": np.array([0.0]),
        "growth_mmol_d": dGr_total * S,
        "carbon_balance_error": float(mb_residual),
        "C_ST_mean": float(np.mean(C_ST_arr)),
        "C_ST_min": float(np.min(C_ST_arr)),
        "C_ST_max": float(np.max(C_ST_arr)),
        "n_iterations": int(n_done),
        "converged": True,
        "max_delta": 0.0,
        "total_loading_mmol": dExud_total * S,  # placeholder; PM doesn't
                                                # expose the loading-only
                                                # flux separately
        "starch_surplus_mmol": dQ_S_meso * S,
        "total_An_mmol_suc": float(AnSum_suc),
        "seed_reserve_mmol": 0.0,
        "partitioning_source": "piafmunch_substep",
        "Rg_node": Q_Gr_arr.copy(),
        "Q_Grmax_node": Q_Grmax_arr.copy(),
        "DVS": None,
        "An_total_mmol": float(AnSum_suc * S),  # PM-organic, mmol CO2
        # PM-specific extras (Gate Ch1.PM.3 / .4 instrumentation) ---------
        "An_total_mmol_target": An_total_mmol_target,
        "sum_Q_S_meso": float(np.sum(Q_S_meso_arr)),
        "dQ_S_meso": float(dQ_S_meso),
        "dQ_meso": float(dQ_meso),
        "dQ_ST": float(dQ_ST),
        "mass_balance_residual_pct": float(mb_residual * 100.0),
        # Gate Ch1.PMDM.3 conservation diagnostics (24-h integrals) -------
        # ∫Ev > 0 (water leaves leaves), ∫RWU < 0 (water leaves soil into
        # roots); the signed sum should be ~0 at steady state.
        "integrated_rwu_cm3": float(integrated_rwu_cm3),
        "integrated_transpiration_cm3": float(integrated_transp_cm3),
        "rwu_transpiration_residual_pct": float(rwu_transp_residual * 100.0),
    }
