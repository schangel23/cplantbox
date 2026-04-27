#!/usr/bin/env python3
"""Session 6 (D.4) — APAR delta sanity via geometric precondition.

Plan: PLAN_FOURNIER_ANDRIEU_INTERNODE_KINETICS_2026-04-23.md §D.4.

**Why this isn't a DART run.** The plan's D.4 ("APAR within 5% of
cosmetic-z-compression baseline") was written when cosmetic-z was still in
the code as an A/B target. Cosmetic-z was removed 2026-04-22 and replaced
by the structural FA model (S3 status note). That makes the literal D.4
untestable; the useful invariant is the GEOMETRIC precondition that pins
APAR delta to a sub-percent effect regardless of any DART run.

**The precondition.** Hard Invariant #2: leaf emergence gates on the Tb=8
axis (`tt_emergence`), while FA stem kinetics run on the Andrieu Tb=9.8
axis. The two axes are decoupled — FA stem kinetics cannot shift leaf
insertion z. All 16 mainstem leaves produce bit-identical insertion z
between FA-on and FA-off at day 130 (verified in the B.6 snapshot, Δ =
0.000000 cm at every rank).

The only FA-on vs FA-off geometric difference is the peduncle: +18.80 cm
of thin leafless stem ABOVE the canopy (apex, where the tassel inserts).
Radiatively this is a single narrow vertical cylinder above the APAR-
absorbing leaf volume. Expected APAR delta: well under 1% — any true
5%+ delta under DART would indicate a structural bug not captured by the
snapshot (e.g., tassel re-insertion or stem-width perturbation).

**When to replace with a real DART run.** If Chapter 2 adds a sensitivity
study that toggles FA on the same plant, re-land D.4 as a server-side
pipeline: 3×3 grid FA-on vs FA-off, integrate daily total aPAR across
leaf segments, assert |Δ mean total APAR| / mean < 5%. Server path:
`/media/data/Lukas/CPlantBox/dart/coupling/` with `python3 -m dart.coupling
diurnal --growth-days 130 --with-carbon` on both branches.

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


def test_all_mainstem_leaves_bit_identical_fa_flag(snapshot):
    """Every mainstem leaf insertion z must be bit-identical between FA-on
    and FA-off. This is what pins the APAR delta to sub-percent: the leaf
    volume absorbing aPAR is geometrically frozen by the FA flag.

    If any leaf diverges, either:
      (a) FA kinetics are leaking into the leaf-appearance gate (Hard
          Invariant #2 violated), or
      (b) lateral creation timing has become flag-sensitive (B.3.5 bug).
    """
    on = sorted(snapshot["fa_on"]["leaf_insertions"], key=lambda l: l["parentNI"])
    off = sorted(snapshot["fa_off"]["leaf_insertions"], key=lambda l: l["parentNI"])
    assert len(on) == len(off), (
        f"Leaf count drift: FA-on={len(on)} FA-off={len(off)}"
    )
    max_delta = 0.0
    for lon, loff in zip(on, off):
        assert lon["subType"] == loff["subType"], (
            f"subType drift at parentNI~{lon['parentNI']}: "
            f"FA-on={lon['subType']} FA-off={loff['subType']}"
        )
        delta = abs(lon["insertion_z_cm"] - loff["insertion_z_cm"])
        max_delta = max(max_delta, delta)
    assert max_delta < 0.001, (
        f"Max leaf insertion z drift = {max_delta:.6f} cm (expect bit-identical). "
        "FA flag is shifting leaf positions — Hard Invariant #2 at risk."
    )


def test_peduncle_is_only_structural_delta(snapshot):
    """The §B.6 peduncle gap is the ONLY FA-on vs FA-off mainstem-length
    delta: mainstem_top_z_delta ≈ peduncle_kinetic_error.

    Both come from `discovery`. If these diverge, there's another source of
    structural drift that we're not accounting for in the APAR argument.
    """
    discovery = snapshot["discovery"]
    top_diff = discovery["mainstem_top_diff_cm"]
    ped_err = discovery["peduncle_kinetic_error_cm"]
    gap = abs(top_diff - ped_err)
    assert gap < 0.05, (
        f"Mainstem-top delta ({top_diff:.3f} cm) disagrees with peduncle "
        f"kinetic error ({ped_err:.3f} cm) by {gap:.3f} cm. There is a "
        "second source of FA-flag structural drift beyond the peduncle."
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
