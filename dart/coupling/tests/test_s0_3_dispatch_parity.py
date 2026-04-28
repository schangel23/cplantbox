"""S0.3 dispatch-parity test (ADR_LEAF_KINEMATICS_2026-04-28).

Runs the same maize_calibrated.xml grow under both
``stem_growth_dispatch == 0`` (shadow if-branch in Stem::simulate, today's
code path) and ``stem_growth_dispatch == 1`` (MultiPhaseStemGrowth GF
dispatched via f_gf->getLength). Asserts that mainstem nodes, total
length, the global cessation latch, and the per-rank length_per_n vector
match bit-identically — which is the gate before S0.4 makes the GF the
default and S0.5 deletes the shadow branch.

Run:
    cpbenv/bin/python3 -m pytest dart/coupling/tests/test_s0_3_dispatch_parity.py -v
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

XML_PATH = COUPLING_DIR / "data" / "maize_calibrated.xml"
# 130 days exercises the full FA pipeline: leaf emergence, plastochron-
# driven rank initiation, per-rank latching, global cessation gate
# (~day 109 under Juelich met), peduncle handling, and tassel emergence.
SIM_DAYS = 130
SEED = 42


def _set_dispatch(plant, dispatch: int) -> None:
    """Flip stem_growth_dispatch on the mainstem subType=1 LRP.

    Must be called after readParameters but BEFORE initialize so
    Plant::initCallbacks mints the right f_gf instance.
    """
    srp = plant.getOrganRandomParameter(pb.OrganTypes.stem, 1)
    srp.stem_growth_dispatch = dispatch


def _grow(dispatch: int):
    """Fresh grow with the requested dispatch flag.

    Uses ``grow_plant`` with ``mutate_lrp_pre_init`` so the dispatch flag
    is applied between readParameters and initialize.
    """

    def _mutate(plant):
        _set_dispatch(plant, dispatch)

    return grow_plant(
        str(XML_PATH),
        simulation_time=SIM_DAYS,
        seed=SEED,
        enable_photosynthesis=False,
        use_fa=True,
        mutate_lrp_pre_init=_mutate,
    )


def _mainstem(plant):
    """Return the maize mainstem (subType=1)."""
    for organ in plant.getOrgans(pb.OrganTypes.stem):
        if organ.getParameter("subType") == 1:
            return organ
    raise RuntimeError("no mainstem subType=1")


def _node_array(stem):
    return [(n.x, n.y, n.z) for n in stem.getNodes()]


@pytest.fixture(scope="module")
def shadow_plant():
    return _grow(0)


@pytest.fixture(scope="module")
def gf_plant():
    return _grow(1)


def test_mainstem_node_count(shadow_plant, gf_plant):
    a = _mainstem(shadow_plant).getNumberOfNodes()
    b = _mainstem(gf_plant).getNumberOfNodes()
    assert a == b, f"mainstem node count diverged: shadow={a} gf={b}"


def test_mainstem_node_positions(shadow_plant, gf_plant):
    a = _node_array(_mainstem(shadow_plant))
    b = _node_array(_mainstem(gf_plant))
    assert len(a) == len(b)
    diffs = [(i, ax, bx) for i, (ax, bx) in enumerate(zip(a, b)) if ax != bx]
    assert not diffs, (
        f"{len(diffs)}/{len(a)} mainstem nodes diverged; first 3: {diffs[:3]}"
    )


def test_mainstem_total_length(shadow_plant, gf_plant):
    a = _mainstem(shadow_plant).getLength(True)
    b = _mainstem(gf_plant).getLength(True)
    assert a == b, f"mainstem length diverged: shadow={a:.12g} gf={b:.12g}"


def test_cessation_latches(shadow_plant, gf_plant):
    a = _mainstem(shadow_plant)
    b = _mainstem(gf_plant)
    assert a.cessation_age_ == b.cessation_age_, (
        f"global cessation_age_ diverged: shadow={a.cessation_age_} gf={b.cessation_age_}"
    )
    assert a.cessation_andrieu_tt_ == b.cessation_andrieu_tt_, (
        "global cessation_andrieu_tt_ diverged: "
        f"shadow={a.cessation_andrieu_tt_} gf={b.cessation_andrieu_tt_}"
    )
    a_per = list(a.cessation_age_per_n)
    b_per = list(b.cessation_age_per_n)
    assert a_per == b_per, f"per-rank cessation_age_per_n diverged: shadow={a_per} gf={b_per}"


def test_length_per_n(shadow_plant, gf_plant):
    a = list(_mainstem(shadow_plant).length_per_n)
    b = list(_mainstem(gf_plant).length_per_n)
    assert a == b, f"length_per_n diverged: shadow={a} gf={b}"


def test_lateral_spawn_flags(shadow_plant, gf_plant):
    a = list(_mainstem(shadow_plant).lateral_spawned_per_n)
    b = list(_mainstem(gf_plant).lateral_spawned_per_n)
    assert a == b, f"lateral_spawned_per_n diverged: shadow={a} gf={b}"
