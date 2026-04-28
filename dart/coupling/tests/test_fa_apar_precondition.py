#!/usr/bin/env python3
"""Session 6 (D.4) — FA-flag structural footprint characterization.

Plan: PLAN_FOURNIER_ANDRIEU_INTERNODE_KINETICS_2026-04-23.md §D.4.
Updated 2026-04-28 for the post-S1-S4 peduncle-exuberance fix.

**History.** Pre-S1-S4 the FA-on/FA-off difference was confined to the
apical peduncle (~+19 cm of thin leafless stem above the canopy under
FA-on). Hard Invariant #2 in its early form held leaf insertion zs
bit-identical FA-on/off, and a sub-percent APAR delta was argued
geometrically (the only delta was a single thin cylinder above the
absorbing leaf volume).

**Post-S1-S4.** The peduncle-exuberance fix is structurally invasive
on the FA-on path: B.2 makes p.ln FA-aware (per-rank ln vector
shrinks under FA-on so the stem fits the IL_final budget without an
exuberant apex), B.3 fires Phase IV cessation per rank, and HI#4
adds an ln basal_floor. Net effect at day 130 (Juelich, seed 7):
    FA-off topmost leaf z ≈ 140 cm, peduncle ≈ 40 cm
    FA-on  topmost leaf z ≈ 187 cm, peduncle ≈  1 cm

The 'leaves are bit-identical FA-on/off' precondition is GONE — that
property was the visible signature of the bug, not an invariant.
The post-fix APAR argument cannot be made geometrically; it needs a
real DART A/B run.

**What remains testable.** Two things:
  1. The fix is FA-only (FA-off has cessation_age_d == -1, isActive
     stays True per the legacy length-vs-getK rule).
  2. The peduncle still sits above the topmost leaf (no peduncle-into-
     canopy regression).

**D.4 status.** Deferred to Chapter 2 as a server-side DART A/B run:
3×3 grid FA-on vs FA-off, integrate daily total aPAR across leaf
segments, assert |Δ mean total APAR| / mean within an empirically
established band. Server path: `/media/data/Lukas/CPlantBox/dart/
coupling/` with `python3 -m dart.coupling diurnal --growth-days 130
--with-carbon` on both branches.

Usage:
    cpbenv/bin/python3 -m pytest dart/coupling/tests/test_fa_apar_precondition.py -v
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
            "Refresh via dart/coupling/tests/baselines/b6_tassel_peduncle_scope.py."
        )
    return json.loads(SNAPSHOT_PATH.read_text())


def test_fix_is_fa_only(snapshot):
    """Pre-fix the FA flag was supposed to leave leaf positions and peduncle
    geometry untouched (cosmetic kinetic forecast only). Post-S1-S4 the
    fix is FA-only by design: FA-off cessation never fires, so its
    mainstem keeps the legacy length-vs-getK active rule. This guards
    against the fix accidentally leaking into the FA-off path.
    """
    fa_off = snapshot["fa_off"]
    cessation_age = fa_off["cessation"]["cessation_age_d"]
    assert cessation_age is None or cessation_age < 0.0, (
        f"FA-off cessation_age_d = {cessation_age}; expected -1 (no FA "
        "cessation). The cessation gate may be leaking into the legacy path."
    )
    # FA-off active flag follows the legacy length-vs-getK rule.  At
    # day 130 with mainstem_length=183.23 cm and getK ≈ 213 cm, the
    # mainstem should still report active=True.
    assert fa_off["mainstem_is_active"] is True, (
        f"FA-off mainstem isActive() = {fa_off['mainstem_is_active']!r}; "
        "expected True (legacy length-vs-getK comparison should not have "
        "tripped). Did the cessation guard accidentally clear FA-off active?"
    )


def test_post_fix_structural_delta_is_distributed(snapshot):
    """Post-S1-S4 characterization: the FA-on/off structural delta is now
    distributed across the stem (B.2 FA-aware p.ln + B.3 cessation), not
    concentrated in the apical peduncle. Pre-fix, mainstem_top_z_delta ≈
    peduncle_kinetic_error within 0.05 cm. Post-fix, |peduncle delta| is
    larger than |mainstem-top delta| because FA-on lifts leaves into what
    used to be the exuberant peduncle zone.

    If these collapse back to near-equality, B.2 (FA-aware p.ln) has
    likely regressed and FA is again only changing the apex.
    """
    discovery = snapshot["discovery"]
    top_diff = discovery["mainstem_top_diff_cm"]
    ped_err = discovery["peduncle_kinetic_error_cm"]
    assert ped_err > top_diff + 5.0, (
        f"Peduncle kinetic error ({ped_err:.3f} cm) should exceed mainstem-top "
        f"delta ({top_diff:.3f} cm) by >5 cm under the post-S1-S4 fix; "
        f"observed gap = {ped_err - top_diff:.3f} cm. Did B.2/B.3 regress?"
    )


def test_peduncle_is_above_topmost_leaf(snapshot):
    """The peduncle must live ABOVE the topmost leaf insertion, else the
    APAR-invariance argument collapses (leaf-overlapping stem shifts APAR).
    """
    for label in ("fa_on", "fa_off"):
        insertions = snapshot[label]["leaf_insertions"]
        topmost_leaf_z = max(li["insertion_z_cm"] for li in insertions)
        mainstem_top = snapshot[label]["mainstem_top_z_cm"]
        assert mainstem_top >= topmost_leaf_z - 0.001, (
            f"{label}: mainstem_top_z ({mainstem_top:.3f}) below topmost "
            f"leaf insertion ({topmost_leaf_z:.3f}) — peduncle would be "
            "within the canopy, breaking the APAR-invariance premise."
        )
