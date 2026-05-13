"""test_g6_pmdm_fa_parity.py — slow pytests for Gate Ch1.PMDM.6 acceptance.

§G6 of PLAN_PIAFMUNCH_DUMUX_COUPLING_2026-05-09.md: does the PM+DuMux
substep dispatch reproduce the FA-on no-carbon oracle across a full
season? Three flavours of increasing horizon and cost:

  G6-fast   day-30 → day-35   static @ -300 cm   ~30 s    sanity gate
  G6-mid    day-30 → day-60   static @ -300 cm   ~15 min  intermediate
  G6-full   day-30 → day-130  dumux  @ -300 cm   ~3 h     headline gate

The headline gate (`test_g6_full_dumux`) is the one that answers the
"does the G5.1 An↔Rm gap matter over a season" question. The fast and
mid variants exist to catch regressions before burning 3 h of server
wall time.

Run on the local box (fast only)::

    cd /home/lukas/PHD/CPlantBox
    cpbenv/bin/python -m pytest dart/coupling/tests/test_g6_pmdm_fa_parity.py \\
        -k g6_fast -v

Run on nile (all three)::

    cd /media/data/Lukas/CPlantBox
    source cpbenv/bin/activate
    python3 -m pytest dart/coupling/tests/test_g6_pmdm_fa_parity.py -v

The full-DuMux test skips automatically when ``rosi_richards`` isn't
importable (matches the pattern in ``test_g5_acceptance.py``).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from dart.coupling.tests.baselines.run_g6_pm_dumux_fa_parity import (  # noqa: E402
    grow_with_pm,
    ORACLE_PATH,
)
from dart.coupling.tests.baselines._oracle_compare import (  # noqa: E402
    per_organ_snapshot,
    compare_against_oracle,
)


def _rosi_richards_available() -> bool:
    """Mirror of test_g5_acceptance._rosi_richards_available."""
    try:
        from dart.coupling.hydraulics.soil_psi import DumuxSoilPsi
        DumuxSoilPsi(
            min_b=(-1, -1, -3), max_b=(1, 1, 0),
            cell_number=(1, 1, 3), psi_init_cm=-300.0,
        )
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# G6-fast — bootstrap day-30 → day-35, static ψ, mainstem-only sanity gate.
# Goal: catch regressions in <60 s on the local box.
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_g6_fast_static_5days():
    """5-day PM substep loop with static ψ — sanity gate.

    Day-35 is too early for late-emerging leaves so we use a relaxed
    per-leaf tolerance (~10%) and rely on mainstem + emerged-blade
    drift to flag wiring regressions. The full-season gates carry the
    actual §G6 spec thresholds.
    """
    plant = grow_with_pm(
        bootstrap_day=30,
        sim_days=35,
        soil_mode="static",
        soil_psi_cm=-300.0,
        inject_an_target=False,
    )
    snap = per_organ_snapshot(plant)
    ok, lines = compare_against_oracle(
        snap,
        ORACLE_PATH,
        tol_leaf_pct=10.0,            # relaxed — 5 days is well under canopy
        tol_mainstem_cm=2.0,          # relaxed — mainstem still extending
        skip_leaves_shorter_than_cm=5.0,
    )
    print("\n".join(lines))
    assert ok, (
        "G6-fast FAILED: PM+static reproduction drifted from FA oracle "
        "at day 35. See test stdout for per-organ deltas."
    )


# ---------------------------------------------------------------------------
# G6-mid — bootstrap day-30 → day-60, static ψ.
# Intermediate horizon (~15 min local). Tighter tolerance — by day-60 the
# canopy has 8-10 emerged blades and the mainstem is past its first phase.
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_g6_mid_static_30days():
    """30-day PM substep loop with static ψ at -300 cm.

    By day-60 the maize plant has reached V8-ish stage. This is the
    earliest horizon where FA-target binding is meaningful on multiple
    ranks. If PM under-supplies relative to FA, drift accumulates here
    and is well clear of the noise floor.
    """
    plant = grow_with_pm(
        bootstrap_day=30,
        sim_days=60,
        soil_mode="static",
        soil_psi_cm=-300.0,
        inject_an_target=False,
    )
    snap = per_organ_snapshot(plant)
    ok, lines = compare_against_oracle(
        snap,
        ORACLE_PATH,
        tol_leaf_pct=5.0,            # tighter than fast
        tol_mainstem_cm=1.0,
        skip_leaves_shorter_than_cm=2.0,
    )
    print("\n".join(lines))
    assert ok, (
        "G6-mid FAILED: PM+static drift > 5% per leaf or > 1 cm "
        "mainstem at day 60. The An↔Rm gap is propagating into "
        "season-level geometry."
    )


# ---------------------------------------------------------------------------
# G6-full — bootstrap day-30 → day-130 with DumuxSoilPsi.
# Headline §G6 gate. ~3 h wall on nile.
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.skipif(
    not _rosi_richards_available(),
    reason="rosi_richards binding not available on this host (local box)",
)
def test_g6_full_dumux_100days():
    """100-day PM+DuMux substep loop. The headline §G6 gate.

    Tolerances per plan §G6 line 1438:
      mainstem <0.5%   (≈ 1 cm at ~200 cm)
      per-leaf <2%

    The day-130 oracle has 16 leaves at full mature canopy. If this
    PASSes, the An↔Rm gap observed at single-day V3 is a transient
    that the plant grows out of as canopy area scales. If it FAILS,
    PM is structurally under-supplying and Ch2 calibration work
    (Krm1 / Baleno aggregation / ψ_init) is on the critical path
    for Ch1 closure.
    """
    plant = grow_with_pm(
        bootstrap_day=30,
        sim_days=130,
        soil_mode="dumux",
        soil_psi_cm=-300.0,
        inject_an_target=False,
    )
    snap = per_organ_snapshot(plant)
    ok, lines = compare_against_oracle(
        snap,
        ORACLE_PATH,
        tol_leaf_pct=2.0,            # plan §G6 spec
        tol_mainstem_cm=1.0,         # ~0.5% of 200 cm mainstem
        skip_leaves_shorter_than_cm=0.0,
    )
    print("\n".join(lines))
    assert ok, (
        "G6-FULL FAILED: PM+DuMux day-130 drift outside §G6 tolerance. "
        "The G5.1 An↔Rm gap accumulates into season-level geometry "
        "divergence — Ch2 calibration is on the critical path."
    )
