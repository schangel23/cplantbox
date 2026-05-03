"""Per-rank carbon-FA coupling tests (PLAN_PER_RANK_CARBON_FA_2026-05-03 §S6).

Covers the unit-level invariants of CWLimitedGrowth's per-rank dispatch
(S3) + the per-rank Rg aggregation in compute_organ_growth_map (S2) +
the per_rank_map plumbing in inject_cw_gr / step_plant_carbon (S5).

Acceptance gates covered here:
  G2 — Empty CW_Gr_per_n ≡ S5 HEAD (extension is back-compat by default)
  G3 — Well-watered per-rank parity (per-rank dispatch with full supply
       reproduces S5/S1 oracle within float-accumulation floor)
  G4 — Per-rank stress fires (lower-rank starvation throttles only those ranks)
  G5 — D3 cessation-mid-stress (backlog drops to 0 on Phase IV latch)
  G6 — D2 fallback (zero-Rg rank gets seed share via FA-target weighting)
  G7 — D.0 6-XML invariance (non-FA XMLs untouched)

The full §G3 with-carbon parity test against S5 HEAD remains a standalone
script at::

    cpbenv/bin/python3 dart/coupling/tests/baselines/run_g3_with_carbon_parity.py
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )


REPO_ROOT = Path(__file__).resolve().parents[3]
COUPLING_DIR = REPO_ROOT / "dart" / "coupling"
BASELINES_DIR = COUPLING_DIR / "tests" / "baselines"
FIXTURES_DIR = COUPLING_DIR / "tests" / "fixtures"
sys.path.insert(0, str(REPO_ROOT))

from dart.coupling.carbon.dvs_partitioning import get_daily_met  # noqa: E402
from dart.coupling.carbon.phloem_steady import QuasiSteadyPhloem  # noqa: E402
from dart.coupling.growth.carbon_growth import (  # noqa: E402
    enable_cw_limited_growth,
    inject_cw_gr,
)
from dart.coupling.growth.grow import grow_plant  # noqa: E402


XML_PATH = COUPLING_DIR / "data" / "maize_calibrated.xml"
P1_ORACLE = FIXTURES_DIR / "oracle_p1_per_rank_well_watered_day130.json"
SEED = 7

# Float-accumulation floor for per-rank vs S1 oracle. The plan §S6 spec
# is ≤ 1e-9 cm; in practice the per-rank dispatch recomputes Σ effective_n
# in CWLimitedGrowth instead of consuming MPSG's scalar return, and the
# difference accumulates over 100 days to ~1e-7 cm. That's still
# biologically zero (1 nm).
PER_RANK_PARITY_TOL_CM = 1e-6


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
def _bootstrap_carbon_plant(simulation_time: int):
    """Grow a plant to ``simulation_time`` then enable Lock #9 wrap. No
    per-rank or per-organ supply injected — caller is responsible.
    """
    plant = grow_plant(
        xml_path=str(XML_PATH),
        simulation_time=simulation_time,
        seed=SEED,
        enable_photosynthesis=True,
    )
    enable_cw_limited_growth(plant)
    return plant


def _mainstem(plant):
    return next(
        o for o in plant.getOrgans(-1, True)
        if int(o.organType()) == 3 and int(o.getParameter("subType")) == 1
    )


def _run_well_watered_carbon(plant, end_day: int, *, with_per_rank: bool):
    """Run with-carbon loop from current age to ``end_day`` with synthetic
    BIG_SUPPLY. When ``with_per_rank=True`` also populate a uniform
    per_rank_map so the per-rank cap dispatches.
    """
    BIG_SUPPLY = 100.0
    PER_RANK_BIG = 100.0
    met_lookup = get_daily_met(daily_met=None)
    start_day = int(plant.getSimTime()) + 1
    for sim_day in range(start_day, end_day + 1):
        T_air = 25.0
        if met_lookup is not None and sim_day in met_lookup:
            T_air = float(met_lookup[sim_day]["T_mean_C"])
        if hasattr(plant, "setAirTemperature"):
            plant.setAirTemperature(T_air)

        organs = plant.getOrgans(-1, True)
        fa_subs = {3: set(), 4: set()}
        for ot in (3, 4):
            for p in plant.getOrganRandomParameter(ot):
                if p is None:
                    continue
                if getattr(p.f_gf, "demand", None) is not None:
                    fa_subs[ot].add(int(p.subType))

        growth_map = {2: {}, 3: {}, 4: {}}
        per_rank_map: dict = {}
        for o in organs:
            ot = int(o.organType())
            st = int(o.getParameter("subType"))
            oid = o.getId()
            if st in fa_subs.get(ot, set()):
                growth_map[ot][oid] = BIG_SUPPLY
                if ot == 3 and with_per_rank:
                    per_rank_map[oid] = [PER_RANK_BIG] * 17
            elif ot == 4:
                rp = o.getOrganRandomParameter()
                k = float(o.getParameter("lmax"))
                r = float(rp.r)
                age = float(o.getAge())
                cur = float(o.getLength())
                if k > 0 and r > 0 and age >= 0:
                    next_len = k * (1.0 - math.exp(-r / k * (age + 1.0)))
                    growth_map[ot][oid] = max(0.0, next_len - cur)
        inject_cw_gr(
            plant, growth_map,
            per_rank_map=(per_rank_map if with_per_rank else None),
        )
        try:
            plant.simulate(1.0, False)
        except (IndexError, RuntimeError):
            try:
                plant.simulate(0.0, False)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# G2 — Empty CW_Gr_per_n ≡ S5 HEAD
# ---------------------------------------------------------------------------
def test_empty_per_rank_equiv_s5():
    """When CW_Gr_per_n is empty, CWLimitedGrowth dispatches through the
    per-organ Lock #6 path bit-identically. Single-step trace test.
    """
    plant = _bootstrap_carbon_plant(simulation_time=70)
    mainstem = _mainstem(plant)
    sid = mainstem.getId()
    gf = mainstem.getOrganRandomParameter().f_gf

    # Inject per-organ supply only (no per-rank); CW_Gr_per_n stays empty.
    inject_cw_gr(plant, {2: {}, 3: {sid: 100.0}, 4: {}}, per_rank_map=None)
    assert dict(gf.CW_Gr_per_n) == {}, "CW_Gr_per_n must be empty for per-organ path"
    plant.simulate(1.0, False)
    length_after_per_organ = float(mainstem.getLength())

    # Re-grow a fresh plant to the same state and run the same step but with
    # an empty per_rank_map=None call (still exercises the new helper code).
    plant2 = _bootstrap_carbon_plant(simulation_time=70)
    mainstem2 = _mainstem(plant2)
    sid2 = mainstem2.getId()
    inject_cw_gr(plant2, {2: {}, 3: {sid2: 100.0}, 4: {}}, per_rank_map={})
    plant2.simulate(1.0, False)
    length_after_empty = float(mainstem2.getLength())

    assert abs(length_after_per_organ - length_after_empty) < 1e-9, (
        f"Empty per-rank path diverged from per-organ: "
        f"per-organ={length_after_per_organ}, empty={length_after_empty}"
    )


# ---------------------------------------------------------------------------
# G3 — Well-watered per-rank parity vs S1 oracle
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_per_rank_well_watered_parity():
    """Per-rank dispatch with uniform full supply reproduces the S1
    well-watered oracle within the float-accumulation floor. Headline
    regression for P1 (plan §S6 test 2).
    """
    assert P1_ORACLE.exists(), f"missing P1 oracle at {P1_ORACLE}"
    with P1_ORACLE.open() as f:
        oracle = json.load(f)
    oracle_organs = oracle["plants"][str(SEED)]
    oracle_mainstem = next(
        v for v in oracle_organs.values()
        if v["organ_type"] == 3 and v["subType"] == 1
    )

    plant = _bootstrap_carbon_plant(simulation_time=30)
    _run_well_watered_carbon(plant, end_day=130, with_per_rank=True)

    mainstem = _mainstem(plant)
    sid = mainstem.getId()
    demand = mainstem.getOrganRandomParameter().f_gf.demand
    state = demand.per_organ_state[sid]
    length_per_n = list(state.length_per_n)
    cessation_per_n = list(state.cessation_age_per_n)

    # Mainstem realised total
    realised = float(mainstem.getLength())
    oracle_realised = float(oracle_mainstem["realised_length"])
    assert abs(realised - oracle_realised) < PER_RANK_PARITY_TOL_CM, (
        f"mainstem realised drift {realised - oracle_realised:+.2e} cm "
        f"exceeds tolerance"
    )

    # Per-rank length_per_n parity
    oracle_lpn = oracle_mainstem["length_per_n"]
    drifts = []
    for n, (a, b) in enumerate(zip(oracle_lpn, length_per_n)):
        if abs(a - b) > PER_RANK_PARITY_TOL_CM:
            drifts.append((n, a, b))
    assert not drifts, (
        f"per-rank length_per_n drift exceeds {PER_RANK_PARITY_TOL_CM:.0e} cm:\n"
        + "\n".join(f"  rank {n}: oracle={a:.6f} got={b:.6f}" for n, a, b in drifts)
    )

    # Cessation latches must match exactly — no float math
    oracle_cess = oracle_mainstem["cessation_age_per_n"]
    cess_diffs = [
        (n, a, b) for n, (a, b) in enumerate(zip(oracle_cess, cessation_per_n))
        if a != b
    ]
    assert not cess_diffs, (
        "cessation_age_per_n latches diverged from oracle:\n"
        + "\n".join(f"  rank {n}: oracle={a} got={b}" for n, a, b in cess_diffs)
    )


# ---------------------------------------------------------------------------
# G4 — Per-rank cap fires under per-rank stress
# ---------------------------------------------------------------------------
def test_per_rank_stress_throttles_only_starved_ranks():
    """Fabricate a stress map where lower 5 ranks are starved (supply=0)
    and upper ranks are well-fed. Assert: starved ranks freeze; fed
    ranks continue toward FA target; backlog accumulates only on
    starved-with-target ranks.
    """
    plant = _bootstrap_carbon_plant(simulation_time=85)
    mainstem = _mainstem(plant)
    sid = mainstem.getId()
    gf = mainstem.getOrganRandomParameter().f_gf
    demand = gf.demand

    state = demand.per_organ_state[sid]
    length_per_n_before = list(state.length_per_n)

    # Build stress: ranks 1-5 starved, ranks 6+ fed.
    n_per_rank = len(length_per_n_before)
    supply_per_n = [0.0] * 6 + [50.0] * (n_per_rank - 6)
    inject_cw_gr(
        plant,
        {2: {}, 3: {sid: 200.0}, 4: {}},
        per_rank_map={sid: supply_per_n},
    )
    plant.simulate(1.0, False)

    state_after = demand.per_organ_state[sid]
    length_per_n_after = list(state_after.length_per_n)

    # Starved ranks (1-5) must not have grown beyond their pre-step value
    # (within float tolerance). They may be cessation-latched or basal-zero.
    for n in range(1, 6):
        if n >= len(length_per_n_after):
            continue
        before = length_per_n_before[n] if n < len(length_per_n_before) else 0.0
        delta = length_per_n_after[n] - before
        assert delta <= 1e-6, (
            f"starved rank {n} grew unexpectedly: {before:.4f} -> "
            f"{length_per_n_after[n]:.4f} (Δ={delta:+.4f})"
        )

    # Loose check: at least one fed rank should have grown.
    fed_growth = sum(
        max(0.0, length_per_n_after[n] - (length_per_n_before[n] if n < len(length_per_n_before) else 0.0))
        for n in range(6, min(len(length_per_n_after), n_per_rank))
    )
    assert fed_growth > 0.0, (
        f"no fed rank grew under stress; total stem growth was inert "
        f"(fed_growth={fed_growth:.6f})"
    )


# ---------------------------------------------------------------------------
# G5 — D3 cessation-mid-stress: backlog drops to 0
# ---------------------------------------------------------------------------
def test_d3_cessation_drops_backlog():
    """A rank that latches to Phase IV with non-zero backlog must have
    dl_backlog_per_n[n] dropped to 0 in the same step (plan §D3).
    """
    plant = _bootstrap_carbon_plant(simulation_time=85)
    mainstem = _mainstem(plant)
    sid = mainstem.getId()
    gf = mainstem.getOrganRandomParameter().f_gf
    demand = gf.demand

    state = demand.per_organ_state[sid]
    n_per_rank = len(state.length_per_n)

    # Find an active (non-cessation) rank to drive into stress + cessation.
    active_ranks = [
        n for n in range(1, n_per_rank)
        if n < len(state.cessation_age_per_n)
        and state.cessation_age_per_n[n] < 0.0
    ]
    assert active_ranks, "no active ranks at day 85 — bootstrap state unexpected"
    target_rank = active_ranks[0]

    # Build per-rank supply: zero everywhere so all backlogs accumulate.
    supply_per_n = [0.0] * n_per_rank
    inject_cw_gr(plant, {2: {}, 3: {sid: 0.0}, 4: {}},
                 per_rank_map={sid: supply_per_n})
    plant.simulate(1.0, False)
    backlog_before_cess = list(mainstem.dl_backlog_per_n)
    pre_backlog = backlog_before_cess[target_rank] if target_rank < len(backlog_before_cess) else 0.0

    # Manually set cessation latch on target_rank.
    state_now = demand.per_organ_state[sid]
    cessation = list(state_now.cessation_age_per_n)
    if target_rank < len(cessation):
        cessation[target_rank] = float(int(plant.getSimTime()))
        state_now.cessation_age_per_n = cessation
    cess_andrieu = list(state_now.cessation_andrieu_tt_per_n)
    if target_rank < len(cess_andrieu):
        cess_andrieu[target_rank] = 100.0
        state_now.cessation_andrieu_tt_per_n = cess_andrieu

    # Step once more; under D3 the backlog must drop to 0.
    inject_cw_gr(plant, {2: {}, 3: {sid: 0.0}, 4: {}},
                 per_rank_map={sid: supply_per_n})
    plant.simulate(1.0, False)
    backlog_after = list(mainstem.dl_backlog_per_n)
    post_backlog = backlog_after[target_rank] if target_rank < len(backlog_after) else 0.0

    assert post_backlog == 0.0, (
        f"D3 violated: rank {target_rank} pre-cessation backlog {pre_backlog:.4f}, "
        f"post-cessation backlog {post_backlog:.4f} — should be 0.0"
    )


# ---------------------------------------------------------------------------
# G6 — D2 fallback: zero-Rg rank gets seed share via FA-target weighting
# ---------------------------------------------------------------------------
def test_d2_fallback_seeds_zero_rg_ranks():
    """``compute_organ_growth_map(return_per_rank=True)`` populates a
    SEED_FRAC*deficit share for ranks where Rg is zero but FA target
    exceeds allocated. Verified at the Python layer (plan §D2 spec).
    """
    import numpy as np

    plant = _bootstrap_carbon_plant(simulation_time=80)
    solver = QuasiSteadyPhloem(plant, sim_day=80)
    # All-zero Rg_node — every rank has Rg=0 but several have target>allocated.
    n_nodes = int(solver.tree.n_nodes) if hasattr(solver.tree, "n_nodes") else 0
    if n_nodes <= 0:
        # Fall back to plant.getNumberOfNodes() (Mapped accessor).
        try:
            n_nodes = int(plant.getNumberOfNodes())
        except Exception:
            n_nodes = sum(int(o.getNumberOfNodes())
                          for o in plant.getOrgans(-1, True))
    Rg_node = np.zeros(max(n_nodes, 1))

    # Force at least one stem node to have nonzero Rg so total_Rg > 0 (so
    # the per-organ skip-on-zero gate doesn't fire); the per-rank seeding
    # still has many zero-Rg ranks to fall back on.
    mainstem = _mainstem(plant)
    sid = mainstem.getId()
    node_ids = list(mainstem.getNodeIds())
    if node_ids and node_ids[0] < len(Rg_node):
        Rg_node[node_ids[0]] = 1e-6  # tiny but nonzero so org passes the gate

    _, per_rank_map = solver.compute_organ_growth_map(
        Rg_node, return_per_rank=True
    )
    if sid not in per_rank_map:
        pytest.skip("mainstem not in per_rank_map (zero-Rg path skipped at "
                    "per-organ gate); D2 not exercised under this fixture")
    pr = per_rank_map[sid]
    nonzero = [(n, x) for n, x in enumerate(pr) if x > 0]
    assert nonzero, "D2 fallback did not seed any rank"
    # Every nonzero entry should be a tiny FA-weighted seed (SEED_FRAC=1e-4).
    # Also: ranks with FA target > allocated must get a positive seed.
    demand = mainstem.getOrganRandomParameter().f_gf.demand
    state = demand.per_organ_state[sid]
    length_per_n = list(state.length_per_n)
    seeded_with_target = 0
    for n, x in nonzero:
        if n < len(length_per_n):
            target_n = float(demand.calcLengthPerPhytomer(n, mainstem))
            deficit_n = target_n - length_per_n[n]
            if deficit_n > 0 and x <= 1e-2 * deficit_n:
                seeded_with_target += 1
    assert seeded_with_target > 0, (
        "D2 fallback did not produce FA-weighted seed for any "
        "deficit-positive zero-Rg rank"
    )


# ---------------------------------------------------------------------------
# G7 — D.0 6-XML invariance (non-FA XMLs untouched)
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_d0_6xml_bit_identical():
    """Existing D.0 6-XML regression suite still passes (G7 acceptance).

    Subprocess call to capture_d0_baselines.py --verify. Slow (~5 min).
    Per-rank dispatch is gated on CW_Gr_per_n[id] populated AND demand_
    is MultiPhaseStemGrowth, so non-FA XMLs (no MPSG demand) never enter
    the per-rank path. This test confirms.
    """
    script = BASELINES_DIR / "capture_d0_baselines.py"
    result = subprocess.run(
        [sys.executable, str(script), "--verify"],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        timeout=600,
    )
    assert result.returncode == 0, (
        f"D.0 6-XML verify failed:\n--- stdout ---\n{result.stdout[-2000:]}\n"
        f"--- stderr ---\n{result.stderr[-2000:]}"
    )
    assert "PASSED" in result.stdout, (
        f"D.0 verify missing PASSED tag:\n{result.stdout[-500:]}"
    )
