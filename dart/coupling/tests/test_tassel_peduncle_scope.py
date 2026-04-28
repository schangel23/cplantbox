#!/usr/bin/env python3
"""Session 4 (B.6) regression — tassel + peduncle scope invariants.

Plan: PLAN_FOURNIER_ANDRIEU_INTERNODE_KINETICS_2026-04-23.md §B.6 + §D.1.
Updated 2026-04-28 for post-S1-S4 peduncle-exuberance fix (HI#4 ln
basal_floor + B.3 per-rank Phase IV cessation + tassel-peduncle gate).

Loads the frozen B.6 snapshot at `tests/baselines/b6_tassel_peduncle_scope.json`
and asserts the durable invariants surfaced by the day-130 maize
FA-on/FA-off comparison under Juelich met (seed 7):

  1. Peduncle is in mainstem subType=1 (apical zone above topmost leaf):
     FA-off retains the legacy ~40 cm zone; FA-on collapses it to <5 cm
     post-S1-S4.
  2. Tassel spike (subType=20) emerges between calendar day 120 and 130.
  3. Topmost leaf insertion z anchors per-path: FA-on ≈ 187 cm (apex),
     FA-off ≈ 140 cm (legacy peduncle below the apex). The pre-S1-S4
     'bit-identical FA-on/off' form of HI#2 is intentionally retired —
     the divergence is the visible signature of the peduncle collapse.

Plus two characterization bounds that detect drift in the FA kinetics:

  4. Mainstem top z under FA-on is in 187–197 cm.
  5. Peduncle FA-vs-scalar error is in 30–45 cm (quantifies the FA-only
     peduncle collapse; pre-fix this was ~19 cm of FA *exuberance*).

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
    is structurally tracked on mainstem subType=1 (not migrated to subType=20).

    Post-S1-S4 (peduncle-exuberance fix):
      - FA-off keeps the legacy ~40 cm apical zone (XML la=22 + scalar burst).
      - FA-on collapses the peduncle to <5 cm via Phase IV cessation: the
        topmost leaf rides up to the apex and the apical-zone block stops
        bleeding length.
    """
    fa_on_pedun = snapshot["fa_on"]["peduncle_length_cm"]
    fa_off_pedun = snapshot["fa_off"]["peduncle_length_cm"]
    assert fa_on_pedun is not None and fa_off_pedun is not None
    assert fa_off_pedun > 5.0, (
        f"FA-off peduncle = {fa_off_pedun:.2f} cm; XML la=22 cm so we expect "
        "a substantial apical zone above the topmost leaf on the legacy path."
    )
    assert fa_on_pedun < 5.0, (
        f"FA-on peduncle = {fa_on_pedun:.2f} cm; expected <5 cm post-S1-S4 "
        "peduncle-exuberance fix (Phase IV cessation should drop the apical "
        "zone). Did B.3 per-rank cessation regress?"
    )


def test_tassel_spike_first_node_at_or_below_mainstem_top(snapshot):
    """Tassel spike inserts on the mainstem at or below mainstem top.

    Post-S1-S4: with FA-on peduncle collapsed to ~1 cm, the spike attaches
    right at the apex (first_node_z ≈ mainstem_top_z), so we accept
    equality with a small float tolerance. Pre-fix this was strict-less-
    than because the peduncle was a thick apical zone above the spike.
    """
    fa_on = snapshot["fa_on"]
    assert fa_on["n_tassel_spikes"] >= 1, "FA-on should produce a tassel spike"
    spike_first_z = fa_on["tassel_spikes"][0]["first_node_z_cm"]
    mainstem_top = fa_on["mainstem_top_z_cm"]
    assert spike_first_z <= mainstem_top + 1e-6, (
        f"Tassel spike first-node z ({spike_first_z:.2f}) should be at or "
        f"below mainstem top z ({mainstem_top:.2f})."
    )


def test_tassel_emergence_in_window(snapshot):
    """Failure modes 1 + 2: under Juelich 2024 met, tassel must emerge
    between calendar day 120 and 130 in both FA-on and FA-off."""
    for label in ("fa_on", "fa_off"):
        td = snapshot[label]["tassel_emerge_day"]
        assert td is not None, f"{label}: tassel never emerged within {snapshot['sim_days']} d"
        assert 120 <= td <= 130, f"{label}: tassel_emerge_day={td} outside [120, 130]"


def test_topmost_leaf_insertion_z_post_fix_anchors(snapshot):
    """D.1 endpoint observable, post-S1-S4 peduncle-exuberance fix.

    The pre-fix HI#2 form ('topmost leaf z = 150.35 ± 0.5, bit-identical
    FA-on/off') is intentionally retired: B.3 per-rank Phase IV cessation
    is FA-only, so it lifts FA-on leaves to the apex (~187 cm) while
    leaving FA-off on the legacy calendar anchor (~140 cm). The ~46 cm
    divergence is the visible signature of the fix.

    HI#2 in its post-fix form: per-path anchor + a minimum FA-on/off gap.
    """
    EXPECTED_TOPMOST_Z = {"fa_on": 186.96, "fa_off": 140.36}
    TOLERANCE_CM = 1.0
    actual = {}
    for label in ("fa_on", "fa_off"):
        leaf_zs = [li["insertion_z_cm"] for li in snapshot[label]["leaf_insertions"]]
        assert leaf_zs, f"{label}: no mainstem leaves attached"
        topmost_leaf_z = max(leaf_zs)
        actual[label] = topmost_leaf_z
        expected = EXPECTED_TOPMOST_Z[label]
        assert abs(topmost_leaf_z - expected) < TOLERANCE_CM, (
            f"{label}: topmost leaf z = {topmost_leaf_z:.3f} cm, "
            f"expected {expected:.2f} ± {TOLERANCE_CM} cm "
            f"(|delta| = {abs(topmost_leaf_z - expected):.3f})"
        )

    # FA-only divergence is the load-bearing post-fix signature.  If it
    # collapses, B.3 per-rank cessation has likely regressed.
    delta = actual["fa_on"] - actual["fa_off"]
    assert delta > 40.0, (
        f"FA-on/off topmost-leaf delta = {delta:.2f} cm; expected >40 cm. "
        "Did the S1-S4 peduncle-exuberance fix regress?"
    )


def test_mainstem_top_z_fa_on_documented_range(snapshot):
    """Documentary bound on FA-on mainstem top z (~188 cm at day 130 with
    current XML, post-S1-S4). Not a calibration target — pre-fix the
    rationale was '+18.8 cm peduncle exuberance'; post-fix the same
    range still holds because Phase IV cessation freezes the mainstem
    once cessation_age_ latches (length stays at lb + Σln + epsilonDx).
    Wide bounds (±5 cm) so a Chapter-2 rank-16 IL_final cap doesn't
    break this without a snapshot refresh."""
    top_fa = snapshot["fa_on"]["mainstem_top_z_cm"]
    assert 187.0 <= top_fa <= 197.0, (
        f"FA-on mainstem top z = {top_fa:.2f} cm outside documented 187–197 range. "
        "Did the FA kinetics change? Refresh snapshot if intentional."
    )


def test_peduncle_fa_vs_scalar_error_quantified(snapshot):
    """§B.6 peduncle kinetic error: |FA-on peduncle - FA-off peduncle|.

    Pre-S1-S4 (peduncle exuberance bug): ~18.8 cm of FA *exuberance*
    (FA-on > FA-off, FA peduncle ~42 cm vs scalar ~23 cm).

    Post-S1-S4 (peduncle exuberance fix, 2026-04-27): error inverts and
    grows to ~38.8 cm — FA-on peduncle collapses to ~1 cm (Phase IV
    cessation), FA-off keeps the legacy ~40 cm peduncle (untouched by
    the fix). Bounds 30–45 cm capture the post-fix regime with margin
    for small calibration drift.
    """
    err = snapshot["discovery"]["peduncle_kinetic_error_cm"]
    assert 30.0 <= err <= 45.0, (
        f"Peduncle FA-vs-scalar error = {err:.2f} cm outside expected 30–45 range. "
        "Either FA kinetics drifted, scalar lmax/r changed, or B.3 per-rank "
        "cessation regressed. Refresh snapshot + re-baseline if intentional."
    )
