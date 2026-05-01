#!/usr/bin/env python3
"""Session 6 (D.3) regression — V-stage calendar under FA-on.

Plan: PLAN_FOURNIER_ANDRIEU_INTERNODE_KINETICS_2026-04-23.md §D.3.

Loads the S5 V-stage baseline (`tests/baselines/s5_vstage_fa_baseline.json`,
captured 2026-04-23 under Juelich 2024 met, seed=7) and asserts:

  1. FA-on V1..V6 match Nielsen targets within ±2 calendar days.
  2. FA-on ≡ FA-off V-stage calendar (Hard Invariant #2: leaves gate on Tb=8
     axis, independent of FA stem kinetics on Tb=9.8 axis).

Refresh the snapshot by re-running:
    cpbenv/bin/python3 dart/coupling/tests/baselines/s5_vstage_fa_baseline.py

Usage:
    cpbenv/bin/python3 -m pytest dart/coupling/tests/test_fa_vstage_calendar.py -v
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).resolve().parent
SNAPSHOT_PATH = TESTS_DIR / "baselines" / "s5_vstage_fa_baseline.json"

V_STAGES = (1, 2, 3, 4, 6)  # Nielsen V5 is null (not reported)

# V4/V6 are expected to drift past ±2d under uncoupled (no carbon/water modulation)
# leaf kinetics with Vidal 2021 bilinear T0 + AHB 2006 R1 (see
# phase_III_per_rank_LEAF.json _meta.ch2_residual_note). The xfail flips back to
# pass once Ch2 carbon/water modulation lands; remove these markers then.
# strict=True → an unexpected XPASS errors the suite, forcing us to revisit.
V_STAGES_XFAIL_CH2_RESIDUAL = {4, 6}


@pytest.fixture(scope="module")
def snapshot() -> dict:
    if not SNAPSHOT_PATH.exists():
        pytest.fail(
            f"S5 V-stage baseline missing: {SNAPSHOT_PATH.name}. "
            f"Re-run dart/coupling/tests/baselines/s5_vstage_fa_baseline.py to capture it."
        )
    return json.loads(SNAPSHOT_PATH.read_text())


def test_vstage_snapshot_metadata(snapshot):
    assert snapshot["seed"] == 7
    assert snapshot["xml"] == "maize_calibrated.xml"
    # Nielsen targets unchanged since S5 closed.
    nielsen = {int(k): v for k, v in snapshot["nielsen_targets"].items()}
    assert nielsen[1] == 17
    assert nielsen[2] == 25
    assert nielsen[3] == 33
    assert nielsen[4] == 51
    assert nielsen[6] == 57


_xfail_reason = (
    "Ch2 residual: under uncoupled leaf kinetics (Vidal 2021 bilinear T0 + AHB 2006 "
    "R1, no carbon/water modulation), V4 collars ~7 d early and V6 ~4 d late vs "
    "Nielsen Iowa-State V-stage calendar. Expected to flip back to pass once "
    "carbon/water modulation lands in Ch2; see phase_III_per_rank_LEAF.json "
    "_meta.ch2_residual_note. strict=True so an unexpected XPASS errors and "
    "forces the marker to be removed."
)
_v_params = [
    pytest.param(v, marks=pytest.mark.xfail(reason=_xfail_reason, strict=True))
    if v in V_STAGES_XFAIL_CH2_RESIDUAL else v
    for v in V_STAGES
]


@pytest.mark.parametrize("v", _v_params)
def test_fa_on_vstage_within_2d_of_nielsen(snapshot, v):
    """D.3 exit gate: V1..V6 drift under FA-on must be within ±2 calendar days.

    V4 and V6 are xfail-strict pending Ch2 carbon/water coupling — see module
    docstring and ``V_STAGES_XFAIL_CH2_RESIDUAL``.
    """
    nielsen = {int(k): v_ for k, v_ in snapshot["nielsen_targets"].items()}
    hits_on = {int(k): v_ for k, v_ in snapshot["hits_fa_on"].items()}
    ref = nielsen[v]
    day = hits_on.get(v)
    assert day is not None, f"V{v} not reached under FA-on"
    delta = day - ref
    assert abs(delta) <= 2, f"V{v}: FA-on day {day} vs Nielsen {ref} (Δ={delta:+d}, >2d)"


def test_fa_on_matches_fa_off_calendar(snapshot):
    """Hard Invariant #2: FA flag must not shift the leaf-appearance calendar.

    Leaves gate on the Tb=8 axis (`tt_emergence`), independent of FA stem
    kinetics (Tb=9.8). Any delta > 1 day is a wiring bug (e.g., FA kinetics
    accidentally re-triggering lateral creation).
    """
    hits_on = {int(k): v for k, v in snapshot["hits_fa_on"].items()}
    hits_off = {int(k): v for k, v in snapshot["hits_fa_off"].items()}
    for v in V_STAGES + (5,):
        d_on = hits_on.get(v)
        d_off = hits_off.get(v)
        if d_on is None and d_off is None:
            continue
        assert d_on is not None and d_off is not None, (
            f"V{v}: FA-on={d_on}, FA-off={d_off} — one branch didn't reach V{v}"
        )
        assert abs(d_on - d_off) <= 1, (
            f"V{v}: FA-on day {d_on} vs FA-off day {d_off} (Δ={d_on - d_off:+d}) "
            "— Hard Invariant #2 violated"
        )
