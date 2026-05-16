"""§S10 root-in-buffered-path fixtures.

Restores the native CPlantBox carbon-economy symmetry: roots, stems and
leaves all participate in the buffered allocation. Roots use
``CWLimitedGrowth`` (bare wrap) → ``ExponentialGrowth`` native potential
as their unimpeded demand; stems/leaves keep FA as their demand model
via the same wrapper.

Plan reference: PLAN_BUFFERED_CARBON_GROWTH_2026-05-15.md §11.3.2
restoration / "RETRACTED" entry replaced by this fix.

The acceptance gates live in:

  - this file (unit + XML round-trip + iterator/demand-helper checks)
  - test_s8_d0_6xml_invariance (§S10 must preserve D.0 baselines: the
    XML knobs are read but never dispatched when PM is off)
  - test_g5_acceptance (slow gate, run after §S10 commit to confirm
    single-day suite is unchanged when roots participate)
  - downstream §S7 sweep CSV (slow_s10, gated behind RUN_S10_LIVE=1)
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from dart.coupling.carbon import pm_substep
from dart.coupling.carbon.pm_substep import (
    _compute_growth_demands,
    _compute_fa_demands,
    _iter_bufferable_organs,
)

CPLANTBOX_ROOT = Path(__file__).resolve().parents[3]
MAIZE_XML = CPLANTBOX_ROOT / "dart/coupling/data/maize_calibrated.xml"


def test_s10_iter_yields_roots():
    """After §S10 the iterator yields organs whose CWLimitedGrowth has no
    demand attached (i.e. roots wrapped as bare CWLimitedGrowth).
    """
    pb = pytest.importorskip("plantbox")
    plant = pb.MappedPlant()
    plant.readParameters(str(MAIZE_XML))
    plant.initialize(False)
    plant.simulate(5.0, False)

    from dart.coupling.growth.carbon_growth import enable_cw_limited_growth
    enable_cw_limited_growth(plant, wrap_roots=False, wrap_fa=True)

    organ_types_seen = set()
    for organ, rp, gf in _iter_bufferable_organs(plant):
        organ_types_seen.add(int(organ.organType()))
    assert 2 in organ_types_seen, "roots must be yielded after §S10"
    assert 3 in organ_types_seen or 4 in organ_types_seen, (
        "stems or leaves must still be yielded"
    )


def test_s10_compute_growth_demands_returns_root_demand():
    """`_compute_growth_demands` returns positive demand for roots when
    their RP carries non-zero local_C_pool_capacity_factor.
    """
    pb = pytest.importorskip("plantbox")
    plant = pb.MappedPlant()
    plant.readParameters(str(MAIZE_XML))
    plant.initialize(False)
    plant.simulate(20.0, False)

    from dart.coupling.growth.carbon_growth import enable_cw_limited_growth
    enable_cw_limited_growth(plant, wrap_roots=False, wrap_fa=True)

    demands = _compute_growth_demands(plant, dt=1.0)
    # The plant has roots by day 20; at least one root must have a
    # non-zero demand in the dict.
    root_ids = [o.getId() for o in plant.getOrgans() if o.organType() == 2]
    root_demands = [demands.get(int(oid), 0.0) for oid in root_ids]
    assert any(d > 0.0 for d in root_demands), (
        "no root demand returned — §S10 wiring broken"
    )


def test_s10_xml_root_rps_have_cost_and_capacity():
    """maize_calibrated.xml seeds each root RP with c_cost_per_cm=0.20
    and local_C_pool_capacity_factor=0.5 (matched to shoot defaults).
    """
    pb = pytest.importorskip("plantbox")
    plant = pb.MappedPlant()
    plant.readParameters(str(MAIZE_XML))
    plant.initialize(False)
    for rp in plant.getOrganRandomParameter(2):
        if rp is None or int(rp.subType) < 1:
            # subType=0 is the C++ template; XML-defined root RPs start
            # at subType=1 (taproot). Only XML-loaded RPs carry the new
            # buffered-carbon fields.
            continue
        assert float(rp.c_cost_per_cm) == pytest.approx(0.20), (
            f"root subType={rp.subType} c_cost_per_cm should be 0.20, "
            f"got {rp.c_cost_per_cm}"
        )
        assert float(rp.local_C_pool_capacity_factor) == pytest.approx(0.5), (
            f"root subType={rp.subType} cap_factor should be 0.5, "
            f"got {rp.local_C_pool_capacity_factor}"
        )


def test_s10_alias_preserved():
    """`_compute_fa_demands` remains importable as a backwards-compatible
    alias for `_compute_growth_demands` so external callers don't break.
    """
    assert _compute_fa_demands is _compute_growth_demands


def test_s10_filter_dropped_in_iter():
    """Sanity: the `gf.demand is None` filter is gone. Constructing a
    fake plant with a bare CWLimitedGrowth on each organ yields all of
    them, not zero.
    """
    pb = pytest.importorskip("plantbox")

    class _FakeOrgan:
        def __init__(self, rp):
            self._rp = rp

        def getOrganRandomParameter(self):
            return self._rp

    rp = SimpleNamespace(f_gf=pb.CWLimitedGrowth())  # bare wrap, demand None
    organ = _FakeOrgan(rp)

    class _FakePlant:
        def getOrgans(self):
            return [organ]

    yielded = list(_iter_bufferable_organs(_FakePlant()))
    assert len(yielded) == 1, "bare CWLimitedGrowth must still yield"


def test_s10_d0_invariance_promise():
    """The §S10 XML edits (c_cost_per_cm + cap_factor on roots) must not
    perturb D.0 baseline hashes — root organs receive the values via
    bindParameter but the buffered dispatch only fires under
    use_buffered_carbon=True, which D.0 never sets.

    This test documents the invariance promise; the actual D.0 verify
    runs as test_s8_d0_6xml_invariance after each S-step.
    """
    pb = pytest.importorskip("plantbox")
    plant = pb.MappedPlant()
    plant.readParameters(str(MAIZE_XML))
    plant.initialize(False)
    # Root RPs received the new fields ...
    rp = plant.getOrganRandomParameter(2, 1)
    assert hasattr(rp, "c_cost_per_cm")
    assert hasattr(rp, "local_C_pool_capacity_factor")
    # ... but their f_gf is still ExponentialGrowth (gf=1) at this stage:
    # carbon_growth.enable_cw_limited_growth would swap it to bare
    # CWLimitedGrowth, but pure D.0 runs (grow_plant without PM wrap)
    # never call that helper, so the native growth curve is preserved.
    assert int(rp.gf) == 1


def test_s10_helpers_exported():
    assert callable(pm_substep._iter_bufferable_organs)
    assert callable(pm_substep._compute_growth_demands)
    assert callable(pm_substep._compute_fa_demands)
