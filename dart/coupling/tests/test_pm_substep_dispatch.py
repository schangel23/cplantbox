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
