#!/usr/bin/env python3
"""Session 4 (B.6) regression — tassel + peduncle scope invariants.

Plan: PLAN_FOURNIER_ANDRIEU_INTERNODE_KINETICS_2026-04-23.md §B.6 + §D.1.

Loads the frozen B.6 snapshot at `tests/baselines/b6_tassel_peduncle_scope.json`
and asserts the three durable invariants surfaced by the day-130 maize
FA-on/FA-off comparison under Juelich met (seed 7):

  1. Peduncle is in mainstem subType=1 (apical zone above topmost leaf).
  2. Tassel spike (subType=20) emerges between calendar day 120 and 130.
  3. Topmost leaf insertion z = 150.35 ± 0.5 cm (calendar-anchored, Hard
     Invariant #2: bit-identical between FA-on and FA-off).

Plus two characterization bounds that detect drift in the FA kinetics:

  4. Mainstem top z under FA-on is in 187–197 cm (documents +18.8 cm peduncle
     exuberance from §B.6; not a calibration target).
  5. Peduncle FA-vs-scalar error is in 10–25 cm (the quantified mismatch).

Refresh the snapshot by re-running:
    cpbenv/bin/python3 dart/coupling/tests/baselines/b6_tassel_peduncle_scope.py

Usage:
    cpbenv/bin/python3 -m pytest dart/coupling/tests/test_tassel_peduncle_scope.py -v
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).resolve().parent
SNAPSHOT_PATH = TESTS_DIR / "baselines" / "b6_tassel_peduncle_scope.json"


@pytest.fixture(scope="module")
def snapshot() -> dict:
    if not SNAPSHOT_PATH.exists():
        pytest.fail(
            f"B.6 snapshot missing: {SNAPSHOT_PATH.name}. "
            f"Re-run dart/coupling/tests/baselines/b6_tassel_peduncle_scope.py to capture it."
        )
    with SNAPSHOT_PATH.open() as f:
        return json.load(f)


def test_b6_snapshot_is_for_maize_calibrated(snapshot):
    assert snapshot["xml"].endswith("maize_calibrated.xml"), (
        f"Snapshot was captured against {snapshot['xml']!r}, "
        "but the B.6 invariants are calibrated for maize_calibrated.xml."
    )
    assert snapshot["sim_days"] == 130
    assert snapshot["seed"] == 7


def test_peduncle_is_in_mainstem_subtype1(snapshot):
    """Failure mode 3: confirm peduncle (apical zone above topmost leaf node)
    lives in mainstem subType=1, not subType=20. Both FA-on and FA-off should
    show a non-trivial apical zone above the topmost leaf insertion."""
    fa_on_pedun = snapshot["fa_on"]["peduncle_length_cm"]
    fa_off_pedun = snapshot["fa_off"]["peduncle_length_cm"]
    assert fa_on_pedun is not None and fa_off_pedun is not None
    assert fa_on_pedun > 5.0, (
        f"FA-on peduncle = {fa_on_pedun:.2f} cm; XML la=22 cm so we expect a "
        "substantial apical zone above the topmost leaf. If <5 cm, peduncle "
        "may have moved to subType=20 (XML refactor?)."
    )
    assert fa_off_pedun > 5.0
    assert snapshot["discovery"]["peduncle_in_mainstem_subtype1"] is True


def test_tassel_spike_first_node_below_mainstem_top(snapshot):
    """Tassel spike inserts on the mainstem at a node *below* mainstem top
    (peduncle is above the tassel insertion). Sanity-checks the structural
    relationship: peduncle ⊂ mainstem subType=1."""
    fa_on = snapshot["fa_on"]
    assert fa_on["n_tassel_spikes"] >= 1, "FA-on should produce a tassel spike"
    spike_first_z = fa_on["tassel_spikes"][0]["first_node_z_cm"]
    mainstem_top = fa_on["mainstem_top_z_cm"]
    assert spike_first_z < mainstem_top, (
        f"Tassel spike first-node z ({spike_first_z:.2f}) should be below "
        f"mainstem top z ({mainstem_top:.2f}); peduncle sits above the spike "
        "insertion in mainstem subType=1."
    )


def test_tassel_emergence_in_window(snapshot):
    """Failure modes 1 + 2: under Juelich 2024 met, tassel must emerge
    between calendar day 120 and 130 in both FA-on and FA-off."""
    for label in ("fa_on", "fa_off"):
        td = snapshot[label]["tassel_emerge_day"]
        assert td is not None, f"{label}: tassel never emerged within {snapshot['sim_days']} d"
        assert 120 <= td <= 130, f"{label}: tassel_emerge_day={td} outside [120, 130]"


def test_topmost_leaf_insertion_z_calendar_anchor(snapshot):
    """D.1 endpoint observable (post-S4 correction): topmost leaf insertion z
    is the calendar-anchored Hard-Invariant-#2 target, not mainstem top z.
    Must be 150.35 ± 0.5 cm and bit-identical between FA-on and FA-off."""
    for label in ("fa_on", "fa_off"):
        leaf_zs = [li["insertion_z_cm"] for li in snapshot[label]["leaf_insertions"]]
        assert leaf_zs, f"{label}: no mainstem leaves attached"
        topmost_leaf_z = max(leaf_zs)
        assert abs(topmost_leaf_z - 150.35) < 0.5, (
            f"{label}: topmost leaf z = {topmost_leaf_z:.3f} cm, "
            f"|delta from 150.35| = {abs(topmost_leaf_z - 150.35):.3f} cm > 0.5"
        )

    # Hard Invariant #2: leaf calendar must be insensitive to the FA flag.
    fa_on_top = max(li["insertion_z_cm"] for li in snapshot["fa_on"]["leaf_insertions"])
    fa_off_top = max(li["insertion_z_cm"] for li in snapshot["fa_off"]["leaf_insertions"])
    assert abs(fa_on_top - fa_off_top) < 0.05, (
        f"Topmost leaf z drifted between FA-on ({fa_on_top:.3f}) and "
        f"FA-off ({fa_off_top:.3f}); Hard Invariant #2 violated."
    )


def test_mainstem_top_z_fa_on_documented_range(snapshot):
    """Documentary bound on FA-on mainstem top z (~192 cm at day 130 with
    current XML). Not a calibration target — characterizes the +18.8 cm
    peduncle exuberance from §B.6. Wide bounds (±5 cm) so a future
    rank-16 IL_final cap doesn't break this without a snapshot refresh."""
    top_fa = snapshot["fa_on"]["mainstem_top_z_cm"]
    assert 187.0 <= top_fa <= 197.0, (
        f"FA-on mainstem top z = {top_fa:.2f} cm outside documented 187–197 range. "
        "Did the FA kinetics change? Refresh snapshot if intentional."
    )


def test_peduncle_fa_vs_scalar_error_quantified(snapshot):
    """The §B.6 peduncle kinetic error: FA-on peduncle - FA-off peduncle.
    Empirically 18.80 cm at day 130 (Juelich met, seed 7). Bounds 10–25 cm
    to detect kinetics drift without being brittle to small calibration
    tweaks. If a Chapter-2 mitigation (rank-16 IL_final cap) lands, this
    assert tightens or moves to <2 cm."""
    err = snapshot["discovery"]["peduncle_kinetic_error_cm"]
    assert 10.0 <= err <= 25.0, (
        f"Peduncle FA-vs-scalar error = {err:.2f} cm outside expected 10–25 range. "
        "Either FA kinetics drifted, scalar lmax/r changed, or rank-16 cap was "
        "applied. Refresh snapshot + re-baseline if intentional."
    )
