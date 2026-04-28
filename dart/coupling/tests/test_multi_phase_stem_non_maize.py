#!/usr/bin/env python3
"""Non-maize opt-in canary for MultiPhaseStemGrowth (ADR §S0.8 + §D8).

Loads an existing CPlantBox wheat XML, programmatically opts the mainstem
into MultiPhaseStemGrowth with placeholder per-rank arrays whose values
are deliberately non-maize-specific (the test does not pretend to be wheat
physiology — it only asserts the new GF dispatches cleanly under a foreign
calibration). Lock #1 is exercised by stamping ``delayNGEndAxis=TT`` so the
merged Andrieu-TT cessation gate fires.

Acts as the canary that catches species-specificity smuggled into the
stem growth function: anything that hardcodes maize plastochron, basal-zero
ranks, or 16-rank lookup tables will surface here as a crash, a NaN, or a
warning about unhandled fields.

Lock #8 (ldelayAxis=TT for leaf birth-gate symmetry) is not yet shipped on
the C++ side — covered by a follow-up commit. Today the leaf side stays on
the existing ``use_thermal_emergence`` + ``tt_emergence`` path, exercised
indirectly through ``Plant::createGrowthFunction`` resolution but without
an axis-style flag.

Usage (from /home/lukas/PHD/CPlantBox):
    cpbenv/bin/python3 -m pytest dart/coupling/tests/test_multi_phase_stem_non_maize.py -v
"""

from __future__ import annotations

import math
import sys
import warnings
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).resolve().parent
COUPLING_DIR = TESTS_DIR.parent
CPLANTBOX_ROOT = COUPLING_DIR.parent.parent
sys.path.insert(0, str(CPLANTBOX_ROOT))

import plantbox as pb  # noqa: E402

WHEAT_XML = CPLANTBOX_ROOT / "gui" / "cplantbox" / "params" / "Triticum_aestivum_a_Bingham_2011.xml"
SIM_DAYS = 30
SEED = 42

PLACEHOLDER_V_N = [0.012, 0.014, 0.016, 0.018, 0.020]   # cm/°Cd, deliberately non-maize
PLACEHOLDER_D_N = [40.0, 50.0, 60.0, 70.0, 80.0]        # °Cd
PLACEHOLDER_IL_FINAL = [0.0, 0.0, 1.5, 3.0, 5.0]        # cm; ranks 1-2 basal-zero, then small monotonic
PLACEHOLDER_BASAL_ZERO_RANKS = [1, 2]                   # non-maize default (maize uses [1,2,3,4])
TT_CESSATION_DEGCD = 600.0                              # well below maize 1500, ensures gate fires under wheat met


def _opt_in_stem_to_multi_phase(stem_rp: pb.StemRandomParameter) -> None:
    """Stamp the placeholder MultiPhaseStemGrowth opt-in onto a StemRandomParameter.

    Mirrors what ``maize_calibrated.xml`` carries post-S0.7 bake, but with
    smaller, non-maize values so any hardcoded 16-rank lookup or maize-only
    plastochron drops a NaN or a warning."""
    stem_rp.use_fournier_andrieu_kinetics = 1
    stem_rp.internode_v_n = PLACEHOLDER_V_N
    stem_rp.internode_D_n = PLACEHOLDER_D_N
    stem_rp.internode_IL_final = PLACEHOLDER_IL_FINAL
    stem_rp.basal_zero_ranks = PLACEHOLDER_BASAL_ZERO_RANKS
    # Lock #1: delayNGEnd reinterpreted as Andrieu-TT cessation threshold when
    # delayNGEndAxis=TT. Existing wheat XMLs default delayNGEndAxis=Calendar
    # so this is the only knob that flips the merged-gate semantics.
    stem_rp.delayNGEnd = TT_CESSATION_DEGCD
    stem_rp.delayNGEndAxis = pb.DelayAxis.TT


@pytest.fixture(scope="module")
def opted_in_plant():
    """Load wheat, opt the mainstem into MultiPhaseStemGrowth, simulate 30 days.

    Captures Python-level warnings and any C++-side stderr messages tagged
    with the FA dispatch path."""
    assert WHEAT_XML.exists(), f"missing fixture XML: {WHEAT_XML}"

    plant = pb.MappedPlant(SEED)
    plant.readParameters(str(WHEAT_XML))

    stem_rps = plant.getOrganRandomParameter(pb.OrganTypes.stem)
    assert stem_rps, "no stem RPs in wheat XML — fixture is wrong"
    mainstem_rp = next((rp for rp in stem_rps if rp.subType == 1), None)
    assert mainstem_rp is not None, "wheat XML lacks stem subType=1 mainstem"
    _opt_in_stem_to_multi_phase(mainstem_rp)

    plant.setSeed(SEED)
    plant.initialize()

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        plant.simulate(float(SIM_DAYS), False)

    return plant, captured


def _all_node_xyz(plant) -> list[tuple[float, float, float]]:
    out: list[tuple[float, float, float]] = []
    for organ in plant.getOrgans():
        for n in organ.getNodes():
            out.append((float(n.x), float(n.y), float(n.z)))
    return out


def test_simulate_completes_without_crash(opted_in_plant):
    plant, _captured = opted_in_plant
    organs = plant.getOrgans()
    assert len(organs) > 0, "simulate produced zero organs — initialise+sim collapsed"


def test_no_nan_positions_anywhere(opted_in_plant):
    plant, _captured = opted_in_plant
    bad = [
        xyz for xyz in _all_node_xyz(plant)
        if not (math.isfinite(xyz[0]) and math.isfinite(xyz[1]) and math.isfinite(xyz[2]))
    ]
    assert not bad, f"{len(bad)} non-finite node positions; first few: {bad[:5]}"


def test_no_python_warnings(opted_in_plant):
    """Python-level warnings (DeprecationWarning, UserWarning, etc.) must be empty.

    C++ stderr (createLeafRadialGeometry, MappedPlant::initializeLB, etc.) is
    not captured by ``warnings.catch_warnings`` and is *not* asserted here —
    those are pre-existing native diagnostics shared with the matrix sweep."""
    _plant, captured = opted_in_plant
    leaked = [w for w in captured if w.category is not pytest.PytestUnraisableExceptionWarning]
    assert not leaked, f"unexpected Python warnings: {[(w.category.__name__, str(w.message)) for w in leaked]}"


def test_mainstem_grew_some(opted_in_plant):
    """Sanity: with placeholder kinetics the wheat mainstem must elongate beyond p.lb.

    Non-zero growth proves the GF dispatch actually executes (a silent fallback
    to ExponentialGrowth would either match the pre-S0 baseline length or stay
    pinned at p.lb depending on path).  Pre-S0 wheat baseline has 5 stem nodes;
    we just require >= 1 elongated segment."""
    plant, _captured = opted_in_plant
    stems = [o for o in plant.getOrgans() if o.organType() == pb.OrganTypes.stem]
    assert stems, "no stems after simulate"
    mainstem = next((s for s in stems if s.getParameter("subType") == 1.0), None)
    if mainstem is None:
        mainstem = stems[0]
    assert len(list(mainstem.getNodes())) >= 2, "mainstem failed to produce any segments"


def test_mainstem_uses_multi_phase_growth_function(opted_in_plant):
    """Round-trip check: after Plant::initialize, the mainstem RP's f_gf must be MultiPhaseStemGrowth.

    Catches the failure mode where the FA flag survives readXML but
    Plant::createGrowthFunction silently picked gft_negexp (e.g. enum int
    drifted, gft_eff branch reverted)."""
    plant, _captured = opted_in_plant
    stem_rps = plant.getOrganRandomParameter(pb.OrganTypes.stem)
    mainstem_rp = next(rp for rp in stem_rps if rp.subType == 1)
    assert isinstance(mainstem_rp.f_gf, pb.MultiPhaseStemGrowth), (
        f"expected MultiPhaseStemGrowth f_gf after FA opt-in, got {type(mainstem_rp.f_gf).__name__}"
    )


def test_no_nan_in_per_organ_fa_state(opted_in_plant):
    """FA per-organ state on the GF must be NaN-free (catches uninitialised reads).

    `MultiPhaseStemGrowth::per_organ_state` is keyed by organ id and holds the
    rolling Phase III length / cessation latch / phytomer bookkeeping. A NaN in
    any of these vectors indicates a bug in either the basal-zero gate, the
    plastochron-driven rank initiation, or the dispatch ordering with respect
    to relative→absolute coordinate flipping."""
    plant, _captured = opted_in_plant
    stem_rps = plant.getOrganRandomParameter(pb.OrganTypes.stem)
    mainstem_rp = next(rp for rp in stem_rps if rp.subType == 1)
    gf = mainstem_rp.f_gf
    assert isinstance(gf, pb.MultiPhaseStemGrowth)
    for organ_id, state in gf.per_organ_state.items():
        for vec_name in ("length_per_n", "epsilonDx_per_n",
                         "cessation_age_per_n", "cessation_andrieu_tt_per_n",
                         "initiation_andrieu_tt_per_n"):
            vec = getattr(state, vec_name)
            for i, v in enumerate(vec):
                assert math.isfinite(float(v)), (
                    f"non-finite value in per_organ_state[{organ_id}].{vec_name}[{i}] = {v}"
                )
