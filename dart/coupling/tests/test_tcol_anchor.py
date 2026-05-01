#!/usr/bin/env python3
"""Acceptance tests for the Vidal-empirical t_col stem-coordinator anchor.

Plan: PLAN_VIDAL_TCOL_STEM_ANCHOR_2026-05-01.md §"Tests added or updated".

Verifies that:
  1. LeafRandomParameter::t_col_emp_Cd exists, defaults to -1.0, and
     round-trips through XML I/O via OrganRandomParameter::readXML/writeXML.
  2. maize_calibrated.xml carries non-default t_col_emp_Cd for the
     Andrieu-fitted leaf subtypes (4..16) populated by the bake step.
  3. Topmost-leaf insertion z at day 100 sits substantially above 50 cm
     (post-fix invariant; pre-fix value was 54.10 cm but with upper-rank
     internodes collapsed to ~1 cm spacing — the gain shows in the per-rank
     spacing distribution, not just the topmost insz).
  4. Per-rank cessation latches all fire: mainstem cessation_age_ ≥ 0
     by day 180 (was -1.0 indefinitely pre-S6).

Usage::

    cpbenv/bin/python3 -m pytest dart/coupling/tests/test_tcol_anchor.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

COUPLING_DIR = Path(__file__).resolve().parent.parent
CPLANTBOX_ROOT = COUPLING_DIR.parent.parent
sys.path.insert(0, str(CPLANTBOX_ROOT))

import plantbox as pb  # noqa: E402

from dart.coupling.growth.grow import grow_plant  # noqa: E402

XML = COUPLING_DIR / "data" / "maize_calibrated.xml"
SEED = 42


def test_t_col_emp_cd_field_default_is_negative():
    """LeafRandomParameter::t_col_emp_Cd defaults to -1.0 (sentinel: disabled)."""
    lrp = pb.LeafRandomParameter(pb.MappedPlant())
    assert hasattr(lrp, "t_col_emp_Cd"), (
        "LeafRandomParameter is missing t_col_emp_Cd field; rebuild .so?"
    )
    assert lrp.t_col_emp_Cd == pytest.approx(-1.0), (
        f"expected default t_col_emp_Cd == -1.0, got {lrp.t_col_emp_Cd}"
    )


def test_t_col_emp_cd_round_trips_via_xml(tmp_path):
    """Writing and reading t_col_emp_Cd via XML preserves the value."""
    plant = pb.MappedPlant()
    plant.readParameters(str(XML))
    out = tmp_path / "round_trip.xml"
    plant.writeParameters(str(out))

    plant2 = pb.MappedPlant()
    plant2.readParameters(str(out))
    src = {int(l.subType): l.t_col_emp_Cd
           for l in plant.getOrganRandomParameter(pb.OrganTypes.leaf)
           if l.subType > 0}
    dst = {int(l.subType): l.t_col_emp_Cd
           for l in plant2.getOrganRandomParameter(pb.OrganTypes.leaf)
           if l.subType > 0}
    assert src == dst, f"XML round-trip lost t_col_emp_Cd: {src} != {dst}"


def test_maize_xml_has_baked_t_col_for_andrieu_ranks():
    """maize_calibrated.xml carries t_col_emp_Cd > 0 on the 13 Andrieu ranks."""
    plant = pb.MappedPlant()
    plant.readParameters(str(XML))
    populated = sorted(
        int(l.subType) for l in plant.getOrganRandomParameter(pb.OrganTypes.leaf)
        if l.subType >= 4 and l.t_col_emp_Cd >= 0.0
    )
    expected = list(range(4, 17))  # subTypes 4..16, i.e. Andrieu ranks 4..16
    assert populated == expected, (
        f"expected baked t_col_emp_Cd on subTypes 4..16, got {populated}"
    )


def test_topmost_leaf_insertion_well_above_pre_fix():
    """Day 100 topmost leaf insertion z is well above the pre-fix value (was 54 cm
    with upper ranks collapsed at 1 cm spacing). Post-fix should restore proper
    upper-rank spacing — encoded as ≥ 60 cm topmost insz."""
    plant = grow_plant(str(XML), simulation_time=100, seed=SEED,
                       enable_photosynthesis=False)
    leaves = sorted(plant.getOrgans(pb.OrganTypes.leaf),
                    key=lambda l: int(l.getParameter("subType")))
    topmost = leaves[-1]
    insz = float(topmost.getNodes()[0].z)
    assert insz >= 60.0, (
        f"topmost leaf insz {insz:.2f} cm <= 60 cm; suggests upper-rank internodes "
        "collapsed to plastochron-seed (~1 cm spacing). "
        "Vidal empirical t_col + cessation symmetry should restore proper FA Phase III growth."
    )


def test_mainstem_cessation_age_fires_post_vt():
    """At day 180 mainstem cessation_age_ ≥ 0 (was -1.0 indefinitely with the
    n_ranks=16 dangling-rank bug)."""
    plant = grow_plant(str(XML), simulation_time=180, seed=SEED,
                       enable_photosynthesis=False)
    mainstem = next(s for s in plant.getOrgans(pb.OrganTypes.stem)
                    if int(s.getParameter("subType")) == 1)
    assert mainstem.cessation_age_ >= 0.0, (
        f"mainstem cessation_age_ = {mainstem.cessation_age_:.2f}; expected >= 0 by "
        "day 180. Indicates one of the per-rank cessation latches did not fire — "
        "check that internode_v_n.size() == n_leaves on mainstem (should be 15)."
    )
