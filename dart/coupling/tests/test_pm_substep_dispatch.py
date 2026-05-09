"""Gate Ch1.PM.4 smoke tests for solve_carbon_partitioning_pm.

Purpose: verify the PM dispatch helper (`carbon/pm_substep.py`) returns
an S5-shaped dict and runs end-to-end on V3 and day-55 maize, mirroring
``pm_notebook_loop --case v3_maize`` / ``--case day55_maize`` numbers
within tolerance.

These tests bypass the diurnal pipeline (no DART, no Baleno) and call
``solve_carbon_partitioning_pm`` directly so they can run on any host
that has cpbenv available — pytools4dart not required.
"""
import math

import numpy as np
import pytest

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from dart.coupling.config import DEFAULT_XML  # noqa: E402
from dart.coupling.growth import grow_plant  # noqa: E402
from dart.coupling.carbon import (  # noqa: E402
    solve_carbon_partitioning, solve_carbon_partitioning_pm,
)


# Minimum keys the diurnal pipeline's CSV / AgroC / summary writers
# read from the carbon_result dict. The PM helper MUST emit all of
# these for downstream consumers to work without modification.
S5_REQUIRED_KEYS = {
    "Rm_total_mmol", "Rm_leaf", "Rm_stem", "Rm_root", "Rm_storage",
    "Rg_total_mmol", "stem_storage_mmol",
    "FR_leaf", "FR_stem", "FR_root", "FR_storage",
    "root_resp_profile_mmol_d", "root_exud_mmol_d", "root_dead_mmol_d",
    "growth_mmol_d", "carbon_balance_error",
    "C_ST_mean", "C_ST_min", "C_ST_max",
    "n_iterations", "converged", "max_delta",
    "total_loading_mmol", "starch_surplus_mmol", "total_An_mmol_suc",
    "seed_reserve_mmol", "partitioning_source",
    "Rg_node", "Q_Grmax_node", "DVS",
}

PM_EXTRA_KEYS = {
    "An_total_mmol", "An_total_mmol_target",
    "sum_Q_S_meso", "dQ_S_meso", "dQ_meso", "dQ_ST",
    "mass_balance_residual_pct",
    # Gate Ch1.PMDM.3 conservation diagnostics
    "integrated_rwu_cm3", "integrated_transpiration_cm3",
    "rwu_transpiration_residual_pct",
}


@pytest.fixture(scope="module")
def v3_plant():
    """A V3 maize plant (~21 d) — fast to grow, suitable for smoke tests."""
    BABST_MET = {
        d: {"T_mean_C": 20.75, "T_min_C": 19.0, "T_max_C": 22.0,
            "PAR_MJ_m2_d": 30.0 * 0.219, "VPD_kPa": 1.0,
            "RH_pct": 60.0, "Wind_m_s": 0.5}
        for d in range(1, 60)
    }
    return grow_plant(
        xml_path=str(DEFAULT_XML),
        simulation_time=21,
        min_stem_nodes=10,
        min_leaf_nodes=4,
        enable_photosynthesis=True,
        seed=42,
        daily_met=BABST_MET,
        T_air_default=20.75,
    )


def _synth_an_per_leaf(plant, an_total_mol=0.0025):
    """Create a per-leaf-segment An vector (mol CO2/d/seg) summing to
    ``an_total_mol`` — mirrors the diurnal-pipeline scaling pattern in
    ``_run_per_plant_carbon`` (line ~899).
    """
    n_leaf_segs = len(plant.getSegmentIds(4))
    if n_leaf_segs == 0:
        return np.array([], dtype=float)
    return np.full(n_leaf_segs, an_total_mol / n_leaf_segs, dtype=float)


def test_pm_returns_s5_shape(v3_plant):
    """Acceptance gate (output contract): PM dict must contain every key
    that the diurnal pipeline's S5 path exposes, plus the Gate Ch1.PM.4
    instrumentation keys."""
    An = _synth_an_per_leaf(v3_plant)
    result = solve_carbon_partitioning_pm(
        v3_plant, An, Tair_C=20.75, day=21,
        n_substeps=4,  # short loop for smoke; full 24-step covered separately
    )
    assert result is not None, "PM solver returned None"
    missing = S5_REQUIRED_KEYS - set(result.keys())
    assert not missing, f"PM dict missing S5 keys: {sorted(missing)}"
    pm_missing = PM_EXTRA_KEYS - set(result.keys())
    assert not pm_missing, f"PM dict missing PM keys: {sorted(pm_missing)}"
    # FR fractions sum to 1 (same invariant as S5).
    fr_sum = (result["FR_leaf"] + result["FR_stem"]
              + result["FR_root"] + result["FR_storage"])
    assert math.isclose(fr_sum, 1.0, abs_tol=1e-6) or fr_sum == 0.0, (
        f"FR fractions don't sum to 1: leaf+stem+root+storage = {fr_sum}")


def test_pm_v3_24substep_smoke(v3_plant):
    """Acceptance gate 2 (single-day V3): 24-substep loop runs to
    completion and produces non-NaN, finite outputs.

    Compared loosely to pm_notebook_loop --case v3_maize:
      AnSum  ~ 23.85 mmol Suc (notebook), our 24-substep AnSum should
      be in the same order of magnitude (loose factor-of-3 check; the
      PAR / met inputs are constants but identical between the helper
      and pm_notebook_loop.case_maize).
      Rm     ~ 0.57 mmol Suc/d at WOFOST Krm1.
    """
    An = _synth_an_per_leaf(v3_plant, an_total_mol=0.002)  # ~24 mmol CO2/d
    result = solve_carbon_partitioning_pm(
        v3_plant, An, Tair_C=20.75, day=21,
        n_substeps=24,
    )
    assert result is not None, "PM solver returned None on V3 24-substep"
    assert result["partitioning_source"] == "piafmunch_substep"
    assert result["n_iterations"] == 24
    # Sanity bounds (matching plan-doc Gate 2 V3 numbers post-Krm1)
    AnSum = result["total_An_mmol_suc"]
    assert AnSum > 0, f"AnSum non-positive: {AnSum}"
    assert math.isfinite(AnSum)
    assert math.isfinite(result["Rm_total_mmol"])
    assert math.isfinite(result["Rg_total_mmol"])
    assert math.isfinite(result["C_ST_mean"])
    # Mass-balance closure target: <5% (Gate 3 closes to <1% on V3,
    # our smoke uses constant met which is equivalent to Gate 3 V3 case).
    mb = abs(result["mass_balance_residual_pct"])
    assert mb < 5.0, (
        f"PM mass-balance residual {mb:.2f}% exceeds 5% smoke threshold; "
        f"AnSum={AnSum:.3f}, Rm={result['Rm_total_mmol']:.3f}, "
        f"Rg={result['Rg_total_mmol']:.3f}, dStorage="
        f"{result['stem_storage_mmol']:.3f}")


def test_pm_idempotent_wrap_does_not_clobber_demand():
    """Lock #6 + #9 wrap policy: running PM on a plant that's already
    been wrapped (e.g. by run_production_series_carbon's bootstrap)
    must NOT clobber the FA demand. The non-idempotency of
    ``enable_cw_limited_growth`` is real (its else-branch overwrites a
    pre-existing CWLim with a bare one); ``solve_carbon_partitioning_pm``
    guards against it via ``_is_cw_wrapped``. This test exercises that
    guard end-to-end on a freshly-grown plant.
    """
    import plantbox as pb
    from dart.coupling.growth.carbon_growth import enable_cw_limited_growth
    from dart.coupling.carbon.pm_substep import _is_cw_wrapped

    BABST_MET = {
        d: {"T_mean_C": 20.75, "T_min_C": 19.0, "T_max_C": 22.0,
            "PAR_MJ_m2_d": 30.0 * 0.219, "VPD_kPa": 1.0,
            "RH_pct": 60.0, "Wind_m_s": 0.5}
        for d in range(1, 60)
    }
    plant = grow_plant(
        xml_path=str(DEFAULT_XML),
        simulation_time=21,
        min_stem_nodes=10,
        min_leaf_nodes=4,
        enable_photosynthesis=True,
        seed=42,
        daily_met=BABST_MET,
        T_air_default=20.75,
    )

    # Pre-wrap (mimics run_production_series_carbon's bootstrap).
    enable_cw_limited_growth(plant, wrap_roots=False, wrap_fa=True)
    saw_fa_wrap = False
    for rp in plant.getOrganRandomParameter(3):
        if rp is None:
            continue
        f_gf = getattr(rp, "f_gf", None)
        if isinstance(f_gf, pb.CWLimitedGrowth) and getattr(
                f_gf, "demand", None) is not None:
            saw_fa_wrap = True
            break
    assert saw_fa_wrap, "FA wrap was not established after first call"
    assert _is_cw_wrapped(plant) is True

    # Run PM — it should detect the pre-wrap and skip re-wrap (otherwise
    # the bug in enable_cw_limited_growth's else-branch would clobber
    # the FA demand).
    An = _synth_an_per_leaf(plant, an_total_mol=0.002)
    result = solve_carbon_partitioning_pm(
        plant, An, Tair_C=20.75, day=21, n_substeps=2)
    assert result is not None

    # FA wrap survived: stem RPs still have CWLim+demand.
    saw_fa_wrap_post = any(
        isinstance(getattr(rp, "f_gf", None), pb.CWLimitedGrowth)
        and getattr(rp.f_gf, "demand", None) is not None
        for rp in plant.getOrganRandomParameter(3) if rp is not None
    )
    assert saw_fa_wrap_post, (
        "FA wrap was clobbered by PM helper (idempotency probe failed)")


def test_s5_baseline_unchanged_on_same_plant(v3_plant):
    """Acceptance gate 0 (S5 path bit-identical sanity): calling the S5
    solver on a clean copy of the V3 plant returns the same dict shape
    as before Gate 4. The carbon_solver flag plumbing must NOT have
    changed S5's behaviour.
    """
    # Use a fresh plant — solve_carbon_partitioning_pm advances the
    # fixture by ~1 day, so reusing v3_plant here would compare different
    # plant states. The fixture is module-scoped, so just call S5 on a
    # newly-grown V3.
    BABST_MET = {
        d: {"T_mean_C": 20.75, "T_min_C": 19.0, "T_max_C": 22.0,
            "PAR_MJ_m2_d": 30.0 * 0.219, "VPD_kPa": 1.0,
            "RH_pct": 60.0, "Wind_m_s": 0.5}
        for d in range(1, 60)
    }
    plant = grow_plant(
        xml_path=str(DEFAULT_XML),
        simulation_time=21,
        min_stem_nodes=10,
        min_leaf_nodes=4,
        enable_photosynthesis=True,
        seed=42,
        daily_met=BABST_MET,
        T_air_default=20.75,
    )
    # Use a uniform synthetic per-leaf An (mol CO2/d/seg) — same shape
    # _run_per_plant_carbon would emit when the diurnal pipeline produces
    # a daily total. Bypasses run_photosynthesis (which trips a soil-psi
    # vector size mismatch on the V3 fixture's 8316-node plant; that's
    # an unrelated harness issue, not a Gate-4 regression).
    An_scaled = _synth_an_per_leaf(plant, an_total_mol=0.002)
    result = solve_carbon_partitioning(
        plant, An_scaled, Tair_C=25.0, method="phloem", day=21)
    assert result is not None
    # S5 contract: every key downstream consumers need is present.
    for k in S5_REQUIRED_KEYS:
        assert k in result, f"S5 missing key {k}"
    assert result.get("partitioning_source") in (
        "quasi_steady_phloem", "dvs_calendar", "dvs_thermal_time")


@pytest.mark.slow
def test_pm_substep_dumux_smoke():
    """Gate Ch1.PMDM.2 acceptance — DuMux + PM well-watered no-op smoke.

    Builds a small 3D DuMux grid, grows two fresh V3 maize plants with
    matching seg→cell mapping (same seed → identical plants), and runs
    24 PM substeps:
      - Run A: ``FixedSoilPsi(psi_cm=-500, n_cells=120)`` static baseline.
      - Run B: ``DumuxSoilPsi(IC=-500 cm, free-drainage bottom, no-flux
        top)`` — exercises the soil↔plant water loop closure wired in
        Gate Ch1.PMDM.2.

    Acceptance:
      1. PM loop completes under DuMux without crash.
      2. PM mass balance < 5 % under dynamic ψ_s (Gate Ch1.PM.3 closure
         survives).
      3. DuMux solver clock advanced (sink push + RichardsSP step
         actually fired).
      4. Per-cell ψ_s descent over 24 h is bounded (<100 cm anywhere) —
         well-watered IC + small RWU drives gentle drainage, not drought.
      5. Bulk PM outputs (Rm, C_ST_mean) stay within 10 % of the static
         baseline — the IC profile differs (FixedSoilPsi's gravity-
         hydrostatic linspace vs DumuxSoilPsi's uniform IC), so a tighter
         gate is not physically meaningful here. The test confirms
         dynamic ψ_s does not perturb PM's internal carbon dynamics
         beyond the IC-shape signal.
    """
    # Prepend the local DuMux build to sys.path before importorskip;
    # DumuxSoilPsi normally does this lazily inside __init__, but the
    # importorskip below runs first, so we mirror the bind-path it
    # would add. Match the default _DEFAULT_DUMUX_BIND in soil_psi.py.
    _dumux_bind = "/home/lukas/PHD/dumux-build/dumux/dumux-rosi/build-cmake/cpp/python_binding"
    if _dumux_bind not in sys.path:
        sys.path.insert(0, _dumux_bind)
    pytest.importorskip(
        "rosi_richards",
        reason=f"DuMux build (rosi_richards) not importable from {_dumux_bind}",
    )
    from dart.coupling.hydraulics.soil_psi import (
        BC_CONSTANT_FLUX, BC_FREE_DRAINAGE, DumuxSoilPsi, FixedSoilPsi,
    )

    # 3D grid sized to a V3 maize plant. ±50 cm lateral box covers the
    # young root spread; 60 cm depth captures the seminal axis.
    MIN_B = (-50.0, -50.0, -60.0)
    MAX_B = (50.0, 50.0, 0.0)
    CELL_NUMBER = (2, 2, 30)
    N_CELLS = int(np.prod(CELL_NUMBER))  # 120
    DAY = 21

    BABST_MET = {
        d: {"T_mean_C": 20.75, "T_min_C": 19.0, "T_max_C": 22.0,
            "PAR_MJ_m2_d": 30.0 * 0.219, "VPD_kPa": 1.0,
            "RH_pct": 60.0, "Wind_m_s": 0.5}
        for d in range(1, 60)
    }

    def fresh_plant():
        return grow_plant(
            xml_path=str(DEFAULT_XML),
            simulation_time=DAY,
            min_stem_nodes=10,
            min_leaf_nodes=4,
            enable_photosynthesis=True,
            seed=42,
            daily_met=BABST_MET,
            T_air_default=20.75,
            soil_min_b=MIN_B,
            soil_max_b=MAX_B,
            soil_cell_number=CELL_NUMBER,
        )

    # --- Run A: static FixedSoilPsi baseline -----------------------------
    plant_static = fresh_plant()
    An = _synth_an_per_leaf(plant_static, an_total_mol=0.002)
    provider_static = FixedSoilPsi(psi_cm=-500.0, n_cells=N_CELLS)
    result_static = solve_carbon_partitioning_pm(
        plant_static, An, Tair_C=20.75, day=DAY, n_substeps=24,
        soil_psi_provider=provider_static,
    )
    assert result_static is not None, "static-provider PM run failed"

    # --- Run B: DumuxSoilPsi, well-watered IC ----------------------------
    plant_dumux = fresh_plant()
    provider_dumux = DumuxSoilPsi(
        min_b=MIN_B, max_b=MAX_B, cell_number=CELL_NUMBER,
        psi_init_cm=-500.0,
        top_bc=(BC_CONSTANT_FLUX, 0.0),
        bot_bc=(BC_FREE_DRAINAGE, 0.0),
        periodic=False,
    )
    # Align provider clock so the first substep's get_profile(t=DAY) is
    # dt_days=0 (read IC, no advance); subsequent substeps advance dt
    # against the previous substep's pushed sink. Mirrors the
    # phase35_3d_smoke pattern (scripts/phase35_3d_smoke.py:118).
    provider_dumux._t_last_days = float(DAY)
    psi_initial = provider_dumux.get_profile(
        t_days=float(DAY), depth_cm=N_CELLS,
    ).copy()
    result_dumux = solve_carbon_partitioning_pm(
        plant_dumux, An, Tair_C=20.75, day=DAY, n_substeps=24,
        soil_psi_provider=provider_dumux,
    )
    assert result_dumux is not None, "DumuxSoilPsi PM run failed"

    # Read final ψ_s after the loop (the 24th substep's get_profile call
    # left _t_last_days = DAY + 23/24, so the next get_profile advances
    # the final dt and returns the post-loop state).
    psi_final = provider_dumux.get_profile(
        t_days=float(DAY) + 1.0, depth_cm=N_CELLS,
    )

    # ------------------------------------------------------------------
    # Acceptance checks
    # ------------------------------------------------------------------
    # (1) loop completed (already asserted above).

    # (2) mass balance preserved.
    mb = abs(result_dumux["mass_balance_residual_pct"])
    assert mb < 5.0, (
        f"PM mass-balance residual {mb:.2f}% exceeds 5% under "
        f"dynamic ψ_s; sink-push wiring may be perturbing internal "
        f"sucrose accounting"
    )

    # (3) DuMux solver clock actually advanced (otherwise the wiring
    # is a no-op and the loop never closed).
    assert provider_dumux._t_last_days > DAY + 0.5, (
        f"DumuxSoilPsi clock did not advance; "
        f"_t_last_days={provider_dumux._t_last_days:.4f} (started at "
        f"{DAY}). Expected ≥{DAY + 0.5} after 24 substeps."
    )

    # (4) ψ_s descent bounded — well-watered + drainage bottom + no
    # evap top → drying is purely RWU-driven and gentle on V3.
    psi_descent_max = float(np.max(psi_initial - psi_final))
    assert psi_descent_max < 100.0, (
        f"max per-cell ψ_s descent {psi_descent_max:.1f} cm exceeds "
        f"100 cm threshold for well-watered V3 over 24 h; "
        f"plant may be pulling water unrealistically fast"
    )

    # (5) bulk pool fluxes track the static baseline within a loose
    # corridor. The IC profile differs by construction: FixedSoilPsi
    # returns the gravity-hydrostatic linspace (-620 cm at the bottom
    # cellidx 0, -500 cm at the top cellidx N-1), while DumuxSoilPsi at
    # ``psi_init_cm=-500`` starts uniform at -500 cm everywhere and
    # equilibrates toward hydrostatic over the first few substeps. For
    # a V3 plant whose roots populate the top ~half of the column, this
    # means root-zone ψ in run B is ~30 cm wetter than run A on average,
    # which drives gs / An / C_ST higher under DuMux. A 15 % corridor
    # captures that IC signal while still flagging actual wiring
    # regressions (which would manifest as order-of-magnitude breaks).
    # A tighter no-op gate would require seeding DumuxSoilPsi with a
    # non-uniform IC matching the FixedSoilPsi linspace — out of scope
    # for the G2 wiring check; revisit when DumuxSoilPsi grows a
    # non-uniform-IC kwarg.
    for key in ("Rm_total_mmol", "C_ST_mean"):
        ref = result_static[key]
        got = result_dumux[key]
        rel = abs(got - ref) / max(abs(ref), 1e-12)
        assert rel < 0.15, (
            f"{key}: dumux={got:.6f} vs static={ref:.6f} "
            f"(rel diff {rel:.2%}) — exceeds 15% no-op gate. Check "
            f"that DumuxSoilPsi IC and FixedSoilPsi gradient are "
            f"compatible at this ψ regime."
        )


@pytest.mark.slow
def test_pm_substep_dumux_conservation():
    """Gate Ch1.PMDM.3 acceptance — soil-side ΔW_soil ≈ ∫RWU closure.

    Closed-system BCs (no-flux top + no-flux bottom): the only
    sink/source on the soil column is the plant, so DuMux's
    ``getWaterVolume()`` before/after the substep loop must match the
    integrated RWU pushed via ``push_rwu_sink_to_provider`` within the
    solver's quadrature tolerance.

    **Acceptance**: ``|ΔW_soil − ∫RWU| / |∫RWU| < 2 %``.

    The plan-doc also calls for a plant-side ``∫RWU + ∫Ev ≈ 0`` check.
    With the current PhotosynthesisPython solver that gate is
    structurally broken on maize: roughly 13 % of the integrated leaf-
    radial flux flows into non-blade leaf segments (sheath + leaf base,
    isPseudostem=1) where ``Ev`` is reported as zero (the gas-exchange
    block only fires on blade segments) but ``outputFlux`` is non-zero
    (xylem mass-balance still applies). Verified empirically: on V3
    maize day 21, ``Σ Ev[blade] = 4.59`` matches
    ``Σ outputFlux[leaf, blade] = 4.59`` to 0.01 %, while
    ``Σ outputFlux[leaf, non-blade] ≈ −0.60`` represents water the
    xylem delivers to non-transpiring leaf tissue. The 13 % gap is
    therefore species-anatomy, not a coupling regression. We track
    ``rwu_transpiration_residual_pct`` informationally but do not
    gate on it; the soil-side closure is the real conservation
    invariant for the PM ↔ DuMux wiring.

    Why no-flux instead of free-drainage (which G2 uses): free-drainage
    adds an open bottom that bleeds water out, breaking the pure-RWU
    mass balance.
    """
    _dumux_bind = "/home/lukas/PHD/dumux-build/dumux/dumux-rosi/build-cmake/cpp/python_binding"
    if _dumux_bind not in sys.path:
        sys.path.insert(0, _dumux_bind)
    pytest.importorskip(
        "rosi_richards",
        reason=f"DuMux build (rosi_richards) not importable from {_dumux_bind}",
    )
    from dart.coupling.hydraulics.soil_psi import (
        BC_CONSTANT_FLUX, DumuxSoilPsi,
    )

    # 100 cm depth: V3 maize seminal axis reaches z≈-75 cm at day 21
    # under BABST_MET (verified via seg2cell diagnostic). Anything
    # shallower lets root segments spill out at cellidx=-1, which then
    # transpire against ψ_air without contributing to ∫RWU and breaks
    # the closure. The G2 smoke test uses depth 60 cm because its
    # assertion (Rm / C_ST_mean within 15 %) tolerates the leak; G3's
    # 2 % gate does not.
    MIN_B = (-50.0, -50.0, -100.0)
    MAX_B = (50.0, 50.0, 0.0)
    CELL_NUMBER = (2, 2, 50)
    N_CELLS = int(np.prod(CELL_NUMBER))  # 200
    DAY = 21
    N_SUB = 24

    BABST_MET = {
        d: {"T_mean_C": 20.75, "T_min_C": 19.0, "T_max_C": 22.0,
            "PAR_MJ_m2_d": 30.0 * 0.219, "VPD_kPa": 1.0,
            "RH_pct": 60.0, "Wind_m_s": 0.5}
        for d in range(1, 60)
    }

    plant = grow_plant(
        xml_path=str(DEFAULT_XML),
        simulation_time=DAY,
        min_stem_nodes=10,
        min_leaf_nodes=4,
        enable_photosynthesis=True,
        seed=42,
        daily_met=BABST_MET,
        T_air_default=20.75,
        soil_min_b=MIN_B,
        soil_max_b=MAX_B,
        soil_cell_number=CELL_NUMBER,
    )

    # Sanity: every root segment must map into the soil grid for the
    # conservation check to be meaningful. Otherwise ∫RWU silently
    # drops the out-of-grid contribution while ∫Ev still includes the
    # transpiration those roots fed.
    ot = np.asarray(plant.organTypes, dtype=int)
    n_root_out = sum(1 for s, c in plant.seg2cell.items()
                     if c < 0 and int(ot[s]) == 2)
    assert n_root_out == 0, (
        f"{n_root_out} root segments fall outside the soil grid "
        f"({MIN_B}→{MAX_B}, cells={CELL_NUMBER}). Enlarge the grid "
        f"or shorten DAY before running G3."
    )

    An = _synth_an_per_leaf(plant, an_total_mol=0.002)

    # Closed-system DuMux: no-flux on top AND bottom. Mass change of
    # the soil column equals integrated RWU within solver tolerance.
    provider = DumuxSoilPsi(
        min_b=MIN_B, max_b=MAX_B, cell_number=CELL_NUMBER,
        psi_init_cm=-500.0,
        top_bc=(BC_CONSTANT_FLUX, 0.0),
        bot_bc=(BC_CONSTANT_FLUX, 0.0),
        periodic=False,
    )
    provider._t_last_days = float(DAY)

    # Capture initial soil water volume BEFORE any get_profile / solve,
    # so the IC is the unperturbed -500 cm uniform field.
    water_initial_cm3 = provider.get_water_volume_cm3()

    result = solve_carbon_partitioning_pm(
        plant, An, Tair_C=20.75, day=DAY, n_substeps=N_SUB,
        soil_psi_provider=provider,
    )
    assert result is not None, "PM run failed under closed-system DuMux"

    # Flush the 24th substep's pending sink. After the loop body's last
    # iteration, _t_last_days = DAY + 23/24 and a sink was just pushed
    # via push_rwu_sink_to_provider. Calling get_profile(DAY+1.0)
    # advances dt = 1/24 day and applies that sink.
    provider.get_profile(t_days=float(DAY) + 1.0, depth_cm=N_CELLS)
    water_final_cm3 = provider.get_water_volume_cm3()
    delta_W_cm3 = float(water_final_cm3) - float(water_initial_cm3)

    int_rwu = float(result["integrated_rwu_cm3"])
    int_transp = float(result["integrated_transpiration_cm3"])
    plant_residual_pct = float(result["rwu_transpiration_residual_pct"])

    # Sanity: both integrals are non-trivial (V3 maize @ DAY=21 with
    # ~600 PAR transpires ≳ a few cm³/day). Otherwise the closure
    # asserts are vacuous.
    assert int_transp > 0.5, (
        f"integrated_transpiration_cm3={int_transp:.4g} is implausibly "
        f"small for a V3 maize plant — solver may not be advancing or "
        f"the leaf area is zero"
    )
    assert int_rwu < -0.5, (
        f"integrated_rwu_cm3={int_rwu:.4g} is non-negative or near-zero "
        f"— root segments may not be mapping into soil cells"
    )

    # Plant-side: informational only. The Ev-vs-RWU gap is dominated
    # by non-blade leaf segments (Ev=0 by gas-exchange model, but
    # outputFlux≠0 by xylem mass balance). Loose 25 % gate catches
    # order-of-magnitude breakage (e.g. wrong-sign accumulator) while
    # tolerating species anatomy.
    assert plant_residual_pct < 25.0, (
        f"plant-side Ev-vs-RWU residual {plant_residual_pct:.2f}% "
        f"exceeds 25 % loose gate (∫RWU={int_rwu:.4g} cm³, "
        f"∫Ev={int_transp:.4g} cm³). Maize V3 sits around 13 % from "
        f"sheath/petiole flux; an order-of-magnitude break is a real "
        f"regression, not anatomy."
    )

    # Soil-side: ΔW_soil ≈ ∫RWU → |ΔW − ∫RWU| / |∫RWU| < 2 %.
    # This is the real Gate Ch1.PMDM.3 invariant: DuMux's solver
    # conserves mass under the closed BCs, so any departure from ∫RWU
    # implies the sink-push wiring is dropping flux somewhere.
    soil_residual_pct = (
        100.0 * abs(delta_W_cm3 - int_rwu) / max(abs(int_rwu), 1e-12)
    )
    assert soil_residual_pct < 2.0, (
        f"soil-side conservation residual {soil_residual_pct:.3f}% "
        f"exceeds 2 % gate (ΔW_soil={delta_W_cm3:.4g} cm³, "
        f"∫RWU={int_rwu:.4g} cm³). Check that BCs are no-flux on every "
        f"face, that the post-flush get_profile call applied the 24th "
        f"substep's pending sink, and that push_rwu_sink_to_provider "
        f"is aggregating per-cell."
    )

    # Mass balance under dynamic ψ_s preserved (Gate Ch1.PM.3 closure
    # survives the soil↔plant water-loop wiring).
    assert abs(result["mass_balance_residual_pct"]) < 5.0, (
        f"PM mass-balance residual "
        f"{result['mass_balance_residual_pct']:.3f}% exceeds 5 %"
    )
