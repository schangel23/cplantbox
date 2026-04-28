#!/usr/bin/env python3
"""S3b.2 + S3b.3 — per-phytomer bookkeeping test battery (B.5').

Plan: PLAN_FULL_B35_PER_PHYTOMER_BOOKKEEPING_2026-04-23.md §A.

S3b.2 shipped the infrastructure: per-phytomer fields on Stem, Option 1
targetlength, per-rank monotonic latch, pybind surface. S3b.3 added
post-hoc node_to_phytomer tagging against the scalar allocator's linking-
node spans, per-rank cessation latch sampling, and the mid-stem lateral
`pending_lateral_pni_override_` hook. This battery covers both stages:

  T1 — per-rank length_per_n advances monotonically (no retreat during
       Phase IV decay). S3b.3: length_per_n is re-summed from the actual
       node geometry each step; monotonicity holds trivially because the
       scalar allocator only grows nodes. S3b.3 also extends length_per_n
       to cover the apical (peduncle) rank, so its size is
       n_ranks+1 + 1 (one extra for the apical tag == n_linking_nodes),
       or longer if the stem has fewer linking nodes realized.
  T4 — Hard Invariant #5 (S3b-redefined): getLength(True) ≈ basal_length_
       + Σ length_per_n after every simulate(dt) step, within dxMin rounding.
       Under S3b.3 this holds to machine precision because length_per_n is
       re-summed directly from the nodes vector at step end.
  T5 — Stem::computeInsertionIndexForRank cold-start fallback: returns
       nodes.size() when node_to_phytomer has no rank-(n-1) entries.
       S3b.3 populates node_to_phytomer via post-hoc tagging, so the
       fallback is tested on ranks whose predecessor has zero tagged
       entries (e.g. querying rank n+2 where only rank n has nodes).

FA-off scalar-path regression (Hard Invariant #1) is covered by the full
6-XML D.0 suite; run `cpbenv/bin/python3 dart/coupling/tests/baselines/
capture_d0_baselines.py --verify` before signing off on S3b.3. Not replicated
inline here to avoid drifting from the capture script's exact hash format.

Explicitly DEFERRED to S3b future sessions (S3b.3 shipped pragmatic tagging
against the scalar allocator's mid-stem inserts, not a full per-rank
insertion driver):
  T2 — same-timestep multi-rank initiation std::sort ordering (needs a
       true per-rank driver that collects initiating ranks and processes
       them in ascending sort order; the scalar allocator bursts all
       laterals simultaneously regardless of TT, so T2's scenario doesn't
       apply to production runs).
  T3 — parentNI shift on mid-stem insert (existing internodalGrowth path
       already relies on addNode shift=true; B.5' T3 would exercise it
       against a per-rank driver that isn't present).

Usage:
    cpbenv/bin/python3 -m pytest dart/coupling/tests/test_fa_per_phytomer_bookkeeping.py -v
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

COUPLING_DIR = Path(__file__).resolve().parent.parent
CPLANTBOX_ROOT = COUPLING_DIR.parent.parent
sys.path.insert(0, str(CPLANTBOX_ROOT))

import plantbox as pb  # noqa: E402

from dart.coupling.carbon.dvs_partitioning import get_daily_met  # noqa: E402
from dart.coupling.growth.grow import setup_successor_where  # noqa: E402


XML_FA = COUPLING_DIR / "data" / "maize_calibrated.xml"
KINETICS_PATH = COUPLING_DIR / "data" / "phase_III_per_rank.json"
SEED = 7
N_RANKS = 16
# S3b.3 achieved machine-precision HI#5 (6e-12 cm). S3b.7 temporarily relaxed
# the tolerance to 3e-1 cm because its basal_internode_cm-sized rank seed
# conflicted with phytomer 0's p.ln[0]=0 cap, leaving a ~0.24 cm bookkeeping
# residual. S3b.8 eliminates the conflict: (a) the XML lb=2 → no 12 cm
# scalar-burst basal; (b) internodalGrowth under FA-on routes any leftover dl
# (from basal_zero_ranks blocking + p.ln-capped elongating ranks) into the
# apical zone via createSegments instead of dropping it as a warning, so the
# `length` accumulator and realized geometry stay bit-identical. Back to 5e-3.
INVARIANT_TOL_CM = 5e-3   # 50 μm — machine precision under S3b.8 bookkeeping


# --------------------------------------------------------------------------- helpers
def _fill_per_rank(tab: dict, default: float) -> list[float]:
    """Build a length-N_RANKS array from the phase_III_per_rank.json tables."""
    arr = [default] * N_RANKS
    numeric_keys = [int(k) for k in tab.keys() if k.isdigit()]
    for n in range(1, N_RANKS + 1):
        if str(n) in tab:
            arr[n - 1] = float(tab[str(n)])
        else:
            near = min(numeric_keys, key=lambda k: abs(k - n))
            arr[n - 1] = float(tab[str(near)])
    return arr


def _configure_fa_mainstem(plant: "pb.MappedPlant") -> None:
    srp = plant.getOrganRandomParameter(pb.OrganTypes.stem, 1)
    srp.use_fournier_andrieu_kinetics = True
    k = json.loads(KINETICS_PATH.read_text())
    srp.internode_v_n = _fill_per_rank(k["v_n_cm_per_degCd"]["expt_1B_primary"], 0.2)
    srp.internode_D_n = _fill_per_rank(k["D_n_degCd"]["values"], 50.0)
    srp.internode_IL_final = _fill_per_rank(k["IL_final_cross_check_cm"]["values"], 16.0)


def _run_faon(days: int):
    """Run FA-on maize_calibrated for `days` days under Juelich met.

    Yields (day, mainstem_Stem) on each step so per-step invariants can be
    asserted without re-running the sim per-test.
    """
    plant = pb.MappedPlant(SEED)
    plant.readParameters(str(XML_FA))
    setup_successor_where(plant)
    _configure_fa_mainstem(plant)
    plant.initialize(False)
    met = get_daily_met()
    for day in range(1, days + 1):
        m = met.get(day) if met else None
        if m is not None:
            plant.setAirTemperature(float(m["T_mean_C"]))
        plant.simulate(1.0, False)
        stems = [o for o in plant.getOrgans(pb.OrganTypes.stem) if int(o.getParameter("subType")) == 1]
        if stems:
            yield day, stems[0]


# --------------------------------------------------------------------------- T1
def test_t1_per_rank_monotonic_latch():
    """T1: length_per_n[n] is non-decreasing across 130 simulate(dt) steps.

    Decision 2 in the plan: the raw Phase IV decay (IL_end_III > IL_final case)
    would cause calcLengthPerPhytomer(n) to drop after Phase III peak. The
    latched length_per_n[n] must NOT drop — S3b.5's per-rank τ_n validation
    against Fournier 2000 Déa consumes the latched value, not the raw target.

    Under S3b.3 length_per_n is re-summed from the scalar allocator's node
    geometry each step, and the allocator only grows nodes (no deletion), so
    monotonicity holds by construction across ranks that have node tags.
    Size can exceed N_RANKS+1 when the apical (peduncle) span gets tagged
    with n_linking_nodes — we check monotonicity on the first N_RANKS+1
    entries which cover the Fournier-indexed ranks.
    """
    prev = [0.0] * (N_RANKS + 2)
    violations: list[tuple[int, int, float, float]] = []  # (day, rank, old, new)
    for day, stem in _run_faon(days=130):
        # S0.5b: FA per-organ kinetic state lives on the MultiPhaseStemGrowth GF.
        fa = stem.getFaState()
        assert fa is not None, "FA-on stem must have a MultiPhaseStemGrowth state"
        lpn = list(fa.length_per_n)
        assert len(lpn) >= N_RANKS + 1, (
            f"length_per_n size {len(lpn)} < n_ranks+1 = {N_RANKS+1}; "
            "S3b.3 may size larger to cover apical rank tag"
        )
        check_upto = min(len(lpn), N_RANKS + 2) - 1
        for n in range(1, check_upto + 1):
            if lpn[n] < prev[n] - 1e-12:
                violations.append((day, n, prev[n], lpn[n]))
            prev[n] = lpn[n]
    assert not violations, (
        f"T1 FAIL: monotonic latch violated on {len(violations)} (day, rank) pairs; "
        f"first 3: {violations[:3]}"
    )


# --------------------------------------------------------------------------- T4
def test_t4_invariant_5_every_step():
    """T4: getLength(True) ≈ basal_length_ + Σ length_per_n (± INVARIANT_TOL_CM).

    This is the S3b-redefined Hard Invariant #5. Under Option 1 bootstrap,
    `basal_length_` tracks the 0 → p.lb basal-stub growth driven by the
    scalar allocator; above p.lb the Σ latched length_per_n tracks the
    distributed FA kinetic sum that drives targetlength. Violations > dxMin
    would indicate either the scalar allocator is distributing length the
    FA targetlength formula doesn't account for, or the latch is diverging
    from realized geometry.
    """
    max_viol = 0.0
    worst: tuple[int, float, float, float] | None = None
    for day, stem in _run_faon(days=130):
        realized = stem.getLength(True)
        # S0.5b: FA bookkeeping reads from the MultiPhaseStemGrowth GF state.
        fa = stem.getFaState()
        assert fa is not None
        basal = fa.basal_length
        s = sum(fa.length_per_n)
        v = abs(realized - (basal + s))
        if v > max_viol:
            max_viol = v
            worst = (day, realized, basal, s)
    assert max_viol < INVARIANT_TOL_CM, (
        f"T4 FAIL: max |L - (basal + Σ length_per_n)| = {max_viol:.4e} cm "
        f"exceeds tol {INVARIANT_TOL_CM} cm; worst (day, L, basal, Σ) = {worst}"
    )


# --------------------------------------------------------------------------- T5
def test_t5_cold_start_insertion_fallback():
    """T5: computeInsertionIndexForRank(n) returns nodes.size() when
    node_to_phytomer contains no rank-(n-1) entries.

    Under S3b.3 post-hoc tagging, node_to_phytomer is populated at the end
    of every simulate(dt) via a span walk over localId_linking_nodes. The
    cold-start fallback is exercised by querying a rank whose predecessor
    has zero tagged nodes — a rank far above the currently-initiated range
    will never have a rank-(n-1) entry, so computeInsertionIndexForRank
    falls through to nodes.size() (apex append).
    """
    plant = pb.MappedPlant(SEED)
    plant.readParameters(str(XML_FA))
    setup_successor_where(plant)
    _configure_fa_mainstem(plant)
    plant.initialize(False)
    plant.simulate(1.0, False)
    stems = [o for o in plant.getOrgans(pb.OrganTypes.stem) if int(o.getParameter("subType")) == 1]
    assert stems, "no mainstem after day 1"
    stem = stems[0]
    ntp = list(stem.node_to_phytomer)
    n_nodes = len(stem.getNodes())
    assert len(ntp) == n_nodes, (
        f"node_to_phytomer size {len(ntp)} ≠ nodes {n_nodes} (S3b.3 tagging must "
        "keep arrays in lockstep)"
    )
    # Find highest tag present, then query well beyond it — guaranteed
    # cold-start since rank (query_rank - 1) has no entries.
    max_tag = max(ntp) if ntp else 0
    for query_rank in (max_tag + 2, max_tag + 10, 100):
        idx = stem.computeInsertionIndexForRank(query_rank)
        assert idx == n_nodes, (
            f"T5 FAIL: computeInsertionIndexForRank({query_rank}) = {idx}, "
            f"expected {n_nodes} (nodes.size()); max_tag in ntp = {max_tag}"
        )


# --------------------------------------------------------------------------- T6 (S3b.3)
def test_t6_synthetic_per_rank_cessation():
    """T6: synthetic tt_cessation=800 exercises per-rank cessation latches.

    Production XML has tt_cessation=1500 which under Juelich 2024 never
    fires by day 130 (Andrieu Tb=9.8 axis climbs to ~1300 °Cd), so the
    per-rank cessation path stays inactive in production — this test
    lowers the gate to force the path to run.

    Expected: under tt_cessation=800, low-rank internodes (which initiated
    earliest, so tau_n = plant_andrieu_tt − init_tt_n crosses 800 first)
    latch their own cessation_andrieu_tt_per_n[n] at some day before 130,
    while high-rank internodes (late-initiated, smaller tau_n) may remain
    unlatched. The per-rank latch field must be >= 0 for at least one
    early rank, and must be ordered: rank_m latches before rank_n when m<n.
    """
    plant = pb.MappedPlant(SEED)
    plant.readParameters(str(XML_FA))
    setup_successor_where(plant)
    _configure_fa_mainstem(plant)
    # Lower the cessation gate to a value that DOES fire under Juelich met.
    srp = plant.getOrganRandomParameter(pb.OrganTypes.stem, 1)
    srp.tt_cessation = 800.0
    plant.initialize(False)
    met = get_daily_met()
    for day in range(1, 131):
        m = met.get(day) if met else None
        if m is not None:
            plant.setAirTemperature(float(m["T_mean_C"]))
        plant.simulate(1.0, False)
    stems = [o for o in plant.getOrgans(pb.OrganTypes.stem) if int(o.getParameter("subType")) == 1]
    assert stems, "no mainstem after 130d"
    stem = stems[0]
    # S0.5b: per-rank cessation latches read from the GF state.
    fa = stem.getFaState()
    assert fa is not None
    cess_tt = list(fa.cessation_andrieu_tt_per_n)
    # At least one early rank must have latched (cessation_andrieu_tt_per_n[n] >= 0).
    latched_ranks = [n for n in range(1, min(len(cess_tt), N_RANKS + 1))
                     if cess_tt[n] >= 0.0]
    assert latched_ranks, (
        f"T6 FAIL: no ranks latched under tt_cessation=800 after 130d; "
        f"cess_tt[1..16] = {cess_tt[1:N_RANKS+1]}"
    )
    # Ordering: for ranks m<n both latched, cess_tt[m] must be <= cess_tt[n]
    # (early ranks hit tau_n=800 earlier at lower plant TT). The LATCH value
    # is plant_andrieu_tt at the moment tau_n crossed tt_cessation; since
    # earlier ranks have smaller init_tt_n, they reach tau_n=tt_cessation at
    # smaller plant_andrieu_tt.
    sorted_ranks = sorted(latched_ranks)
    for i in range(len(sorted_ranks) - 1):
        m, n = sorted_ranks[i], sorted_ranks[i + 1]
        assert cess_tt[m] <= cess_tt[n] + 1e-6, (
            f"T6 FAIL: per-rank cessation ordering violated at ({m},{n}): "
            f"cess_tt[{m}]={cess_tt[m]:.3f} > cess_tt[{n}]={cess_tt[n]:.3f}"
        )
    # Stem must still have produced some growth (not stuck at 0-length).
    assert stem.getLength(True) > 50.0, (
        f"T6 FAIL: synthetic cessation froze all growth (L={stem.getLength(True):.1f} cm); "
        "should be >50 cm since early phytomers elongate before tt_cessation=800 fires"
    )

