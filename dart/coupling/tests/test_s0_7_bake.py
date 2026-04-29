#!/usr/bin/env python3
"""S0.7 — Calibration bake script (Lock #3 Half A).

Operationalises D6: the runtime contract is pure XML + C++. After bake,
``pb.MappedPlant("maize_calibrated.xml").initialize().simulate(N)`` reproduces
``grow_plant("maize_calibrated.xml", simulation_time=N)`` without any Python
configuration step between ``readParameters`` and ``initialize``.

Tests:
  T1  round-trip: bake → load baked XML → diff every mutated LRP field
      against in-memory post-bake state. Diff must be empty (exact float
      equality after the precision fix in OrganRandomParameter::writeXML).

  T2  hard gate (post-S3): 25-day grow side-by-side, both paths reading
      the BAKED XML.
      Path A — ``grow_plant(BAKED, 25, daily_met={}, ...)`` (post-S3 runtime
        path: pipeline driver, no pre-init Python helpers, met-free
        deterministic loop).
      Path B — load BAKED XML, day-by-day simulate(1.0) for 25 days
        (pure XML + C++ contract, no pipeline driver).
      Compare node positions: both paths must produce the same plant to
      < 1e-9 cm per node. 25 days exercises FA mainstem kinetics, leaf
      logistics, plastochron-driven rank initiation, basal_zero gate, and
      the successorWhere phyllotaxy — the full Lock #3 Half A surface
      now living on the baked XML.

The simulation length is kept at 25 days (rather than 130) for runtime;
later 130-day end-to-end checks live in the diurnal-pipeline acceptance
suite, not in the unit-test path.
"""
from __future__ import annotations

import math
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

import plantbox as pb

from dart.coupling.config import DEFAULT_XML
from dart.coupling.growth.grow import grow_plant
from dart.coupling.scripts.bake_calibration_to_xml import bake


HARD_GATE_DAYS = 25
HARD_GATE_SEED = 42
NODE_TOLERANCE_CM = 1e-9


def _stem_subtype1_snapshot(plant) -> dict:
    """Snapshot the mutated stem-subType=1 fields the bake touches."""
    srp = plant.getOrganRandomParameter(pb.OrganTypes.stem, 1)
    return {
        "use_fournier_andrieu_kinetics": int(getattr(srp, "use_fournier_andrieu_kinetics")),
        "internode_v_n": tuple(srp.internode_v_n),
        "internode_D_n": tuple(srp.internode_D_n),
        "internode_IL_final": tuple(srp.internode_IL_final),
        "basal_zero_ranks": tuple(srp.basal_zero_ranks),
        "RotBeta": float(srp.RotBeta),
        "BetaDev": float(srp.BetaDev),
        "successorST": tuple(tuple(r) for r in srp.successorST),
        "successorOT": tuple(tuple(r) for r in srp.successorOT),
        "successorP": tuple(tuple(r) for r in srp.successorP),
        "successorNo": tuple(srp.successorNo),
        "successorWhere": tuple(tuple(r) for r in srp.successorWhere),
    }


def _leaf_snapshot(plant) -> dict:
    """Snapshot per-leaf-LRP FA fields the bake touches."""
    out = {}
    for lrp in plant.getOrganRandomParameter(pb.OrganTypes.leaf):
        out[int(lrp.subType)] = {
            "use_fa_kinetics": int(getattr(lrp, "use_fa_kinetics")),
            "tau_extension_n": float(getattr(lrp, "tau_extension_n")),
            "sigma_extension_n": float(getattr(lrp, "sigma_extension_n")),
        }
    return out


@pytest.fixture
def baked_pair(tmp_path):
    """Bake a copy of the canonical maize XML and return both sides.

    Returns:
        (master_xml, baked_xml, post_bake_state) where ``post_bake_state``
        is the in-memory snapshot of the plant whose state was written to
        ``baked_xml`` (so T1 can compare it against the on-disk reload).
    """
    master_xml = tmp_path / "maize_calibrated.master.xml"
    baked_xml = tmp_path / "maize_calibrated.baked.xml"
    shutil.copy(DEFAULT_XML, master_xml)

    bake(master_xml, baked_xml, verbose=False)

    # Re-do the bake's pre-write side in memory so we have an in-memory
    # snapshot to diff against the on-disk baked.xml. Importing the helpers
    # one more time (rather than caching the bake's plant) keeps the test
    # honest about what the bake "really" does.
    from dart.coupling.growth.grow import (
        enable_fa_on_leaves,
        enable_fa_on_mainstem,
        setup_successor_where,
    )

    plant_pre = pb.MappedPlant()
    plant_pre.readParameters(str(master_xml))
    setup_successor_where(plant_pre)
    enable_fa_on_mainstem(plant_pre, verbose=False)
    enable_fa_on_leaves(plant_pre, verbose=False)
    pre_state = {
        "stem": _stem_subtype1_snapshot(plant_pre),
        "leaf": _leaf_snapshot(plant_pre),
    }

    return master_xml, baked_xml, pre_state


def test_T1_round_trip_diff_is_empty(baked_pair):
    """T1: bake → load baked → diff every mutated field. Must be empty."""
    _, baked_xml, pre_state = baked_pair

    plant_post = pb.MappedPlant()
    plant_post.readParameters(str(baked_xml))
    post_state = {
        "stem": _stem_subtype1_snapshot(plant_post),
        "leaf": _leaf_snapshot(plant_post),
    }

    # Stem: every key must match exactly.
    for key, want in pre_state["stem"].items():
        got = post_state["stem"][key]
        assert got == want, (
            f"Stem field {key!r} did not round-trip: pre={want!r} post={got!r}"
        )

    # Leaf: every subType must match.
    pre_leaves = pre_state["leaf"]
    post_leaves = post_state["leaf"]
    assert set(pre_leaves) == set(post_leaves), (
        f"Leaf subType set diverged: pre={sorted(pre_leaves)} "
        f"post={sorted(post_leaves)}"
    )
    for st, want in pre_leaves.items():
        got = post_leaves[st]
        assert got == want, (
            f"Leaf subType={st} fields did not round-trip: "
            f"pre={want!r} post={got!r}"
        )


def _node_array(plant) -> np.ndarray:
    """Return all plant nodes as an (N, 3) array of doubles."""
    nodes = plant.getNodes()
    return np.asarray([[n.x, n.y, n.z] for n in nodes], dtype=float)


def _grow_no_helpers(xml_path: Path, days: int, seed: int) -> "pb.MappedPlant":
    """Grow without any Python pre-init configuration (pure XML + C++).

    Mirrors ``grow_plant``'s daily-stepped simulate loop so the comparison is
    apples-to-apples (same dt structure → same RNG draws for tropism), but
    skips ``setup_successor_where`` / ``enable_fa_on_mainstem`` /
    ``enable_fa_on_leaves`` and skips met forcing.
    """
    plant = pb.MappedPlant()
    plant.readParameters(str(xml_path))
    plant.setSeed(seed)
    plant.initialize()
    for _ in range(days):
        plant.simulate(1.0, verbose=False)
    return plant


def _grow_via_pipeline_driver(xml_path: Path, days: int, seed: int) -> "pb.MappedPlant":
    """Grow via grow_plant() with met disabled (daily_met={} kills met loop).

    Post-S3, ``grow_plant`` is a thin pipeline driver — no pre-init Python
    configuration, just a daily-stepped simulate loop. Compared to
    ``_grow_no_helpers`` the only differences are the (disabled) met-forcing
    branch and the (no-op) cp-donor / soil-grid hooks; both paths must
    produce a bit-identical plant on the baked XML.
    """
    return grow_plant(
        str(xml_path),
        simulation_time=days,
        seed=seed,
        enable_photosynthesis=False,
        daily_met={},
    )


def test_T2_hard_gate_pure_xml_matches_grow_plant(baked_pair):
    """T2: D6's operational closure (post-S3).

    grow_plant(BAKED, days) == pure_XML(BAKED, days)   per-node, < 1e-9 cm.

    Both paths read the baked XML; ``grow_plant`` no longer mutates the
    plant between ``readParameters`` and ``initialize``, so it must
    reproduce the pure-XML invocation byte-for-byte.
    """
    _, baked_xml, _ = baked_pair

    plant_driver = _grow_via_pipeline_driver(baked_xml, HARD_GATE_DAYS, HARD_GATE_SEED)
    plant_pure = _grow_no_helpers(baked_xml, HARD_GATE_DAYS, HARD_GATE_SEED)

    nodes_a = _node_array(plant_driver)
    nodes_b = _node_array(plant_pure)

    assert nodes_a.shape == nodes_b.shape, (
        f"Node-count diverged: driver={nodes_a.shape[0]} pure={nodes_b.shape[0]}"
    )
    diff = np.abs(nodes_a - nodes_b)
    max_diff = float(diff.max()) if diff.size else 0.0
    assert max_diff < NODE_TOLERANCE_CM, (
        f"Pure-XML and grow_plant diverged: max per-node L_inf diff "
        f"{max_diff:.3e} cm exceeds {NODE_TOLERANCE_CM} cm "
        f"(at index {int(np.unravel_index(diff.argmax(), diff.shape)[0])})"
    )
