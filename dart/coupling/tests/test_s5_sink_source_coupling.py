"""S5 sink-source coupling tests (PLAN_S5_SINK_SOURCE_COUPLING_2026-05-02 §S6).

Covers the unit-level invariants of Lock #6 (CWLimitedGrowth(demand=…)) +
Lock #9 (enable_cw_limited_growth strict wrap policy) + §M2 dl_backlog.
The full §G3 with-carbon parity test (FA-on no-carbon vs FA-on with-carbon
@ day 130 against the S5 oracle) is too long for pytest — it runs as a
standalone script at::

    cpbenv/bin/python3 dart/coupling/tests/baselines/run_g3_with_carbon_parity.py

Acceptance gates covered here:
  G1 — D.0 6-XML bit-identical (subprocess against capture_d0_baselines.py --verify)
  G4 — dl_backlog accumulates under stress + drains on recovery
  G5 — Lock #9 three-way coexistence: FA stems wrapped, FA leaves wrapped,
       roots bare; no cross-organ-type CW_Gr contamination
  G6 — source-grep guard: no production code path bypasses enable_cw_limited_growth
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

# Register the `slow` marker for the subprocess-based G1 verifies. Avoids
# PytestUnknownMarkWarning without pulling in a project-level conftest.
def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )


REPO_ROOT = Path(__file__).resolve().parents[3]
COUPLING_DIR = REPO_ROOT / "dart" / "coupling"
BASELINES_DIR = COUPLING_DIR / "tests" / "baselines"
sys.path.insert(0, str(REPO_ROOT))

import plantbox as pb  # noqa: E402

from dart.coupling.growth.carbon_growth import (  # noqa: E402
    enable_cw_limited_growth,
)
from dart.coupling.growth.grow import grow_plant  # noqa: E402

XML_PATH = COUPLING_DIR / "data" / "maize_calibrated.xml"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def maize_day_15():
    """Bootstrap a maize plant via production grow_plant() to day 15.

    Module-scoped so all tests share one bootstrap (~10 s grow). Tests
    that need a clean post-wrap plant should not mutate this fixture.
    """
    plant = grow_plant(
        xml_path=str(XML_PATH),
        simulation_time=15,
        seed=7,
        enable_photosynthesis=True,
    )
    return plant


# ---------------------------------------------------------------------------
# G5 — Lock #9 three-way coexistence
# ---------------------------------------------------------------------------
def test_lock9_wrap_predicate_strict_isinstance(maize_day_15):
    """Strict wrap fires only on MultiPhase{Stem,Leaf}Growth instances.

    PLAN §S4 / D2 decision (a). Verified end-to-end: roots stay bare
    (Lock #9 default `wrap_roots=False` preserves
    [[project_root_path_preservation]]); mainstem subType=1 + FA leaf
    subtypes 4..16 wrap with their FA demand; tassel spike + scalar
    leaves stay bare.
    """
    enable_cw_limited_growth(maize_day_15)

    expected_wrap = {
        # (organ_type, subType): expected demand class name
        (3, 1): "MultiPhaseStemGrowth",       # mainstem FA
    }
    # FA leaf subtypes (the maize_calibrated XML wires FA on subtypes 4..16)
    for st in range(4, 17):
        expected_wrap[(4, st)] = "MultiPhaseLeafGrowth"

    expected_bare = {
        # mainstem children + scalar leaves + all roots → bare CWLim
        (3, 0), (3, 20), (3, 21),
        (4, 0), (4, 2), (4, 3),
    }
    expected_bare.update({(2, st) for st in range(0, 6)})  # all roots

    seen_wrap = {}
    seen_bare = set()
    for ot in (2, 3, 4):
        for p in maize_day_15.getOrganRandomParameter(ot):
            if p is None:
                continue
            assert isinstance(p.f_gf, pb.CWLimitedGrowth), (
                f"ot={ot} subType={p.subType}: f_gf is "
                f"{type(p.f_gf).__name__}, expected CWLimitedGrowth"
            )
            key = (ot, int(p.subType))
            if p.f_gf.demand is None:
                seen_bare.add(key)
            else:
                seen_wrap[key] = type(p.f_gf.demand).__name__

    # Every key in expected_wrap should be wrapped with the right demand.
    for key, demand_class in expected_wrap.items():
        assert key in seen_wrap, f"expected wrap on ot={key[0]} st={key[1]}; got bare"
        assert seen_wrap[key] == demand_class, (
            f"ot={key[0]} st={key[1]}: wrapped with {seen_wrap[key]}, "
            f"expected {demand_class}"
        )
    # Every key in expected_bare should be bare.
    for key in expected_bare:
        assert key in seen_bare, (
            f"expected bare on ot={key[0]} st={key[1]}; got wrapped with "
            f"{seen_wrap.get(key, '???')}"
        )


def test_lock9_no_cross_organ_type_cw_gr_contamination(maize_day_15):
    """CW_Gr maps must be type-keyed, not subtype-keyed.

    PLAN §S6 test 4: ``inject_cw_gr`` writes the SAME map to all
    subtypes of a given organ type (PiafMunch's runPM.cpp:643-652
    convention). After enable_cw_limited_growth, all RPs of a given
    organ type should share the same CW_Gr identity per type, not
    leak across types.
    """
    enable_cw_limited_growth(maize_day_15)

    # Populate a distinguishable CW_Gr per organ type
    test_maps = {2: {1: 1.0}, 3: {1: 2.0}, 4: {1: 3.0}}
    from dart.coupling.growth.carbon_growth import inject_cw_gr
    inject_cw_gr(maize_day_15, test_maps)

    for ot, expected_value in [(2, 1.0), (3, 2.0), (4, 3.0)]:
        for p in maize_day_15.getOrganRandomParameter(ot):
            if p is None:
                continue
            cw = p.f_gf.CW_Gr
            assert 1 in cw, (
                f"ot={ot} subType={p.subType}: CW_Gr missing key 1"
            )
            assert cw[1] == expected_value, (
                f"ot={ot} subType={p.subType}: CW_Gr[1] is {cw[1]}, "
                f"expected {expected_value} — possible cross-type leak"
            )


def test_lock9_wrap_roots_kwarg_off_keeps_roots_bare():
    """wrap_roots=False (default) must leave root f_gf bare.

    Regression for the [[project_root_path_preservation]] contract.
    """
    plant = grow_plant(
        xml_path=str(XML_PATH),
        simulation_time=5,
        seed=7,
        enable_photosynthesis=True,
    )
    enable_cw_limited_growth(plant, wrap_roots=False)
    for p in plant.getOrganRandomParameter(2):
        if p is None:
            continue
        assert isinstance(p.f_gf, pb.CWLimitedGrowth)
        assert p.f_gf.demand is None, (
            f"root subType={p.subType}: demand={type(p.f_gf.demand).__name__}; "
            f"expected None when wrap_roots=False"
        )


# ---------------------------------------------------------------------------
# G4 — dl_backlog accumulates under stress + drains on recovery
# ---------------------------------------------------------------------------
def _find_fa_leaf(plant):
    """Pick a mainstem-attached FA leaf with non-trivial growth headroom."""
    candidates = []
    for o in plant.getOrgans():
        if o.organType() != int(pb.OrganTypes.leaf):
            continue
        if int(o.getParameter("subType")) < 4:
            continue
        cur = o.getLength()
        lmax = float(o.getParameter("lmax"))
        if cur >= lmax - 0.5:
            continue
        candidates.append((cur, o))
    if not candidates:
        raise RuntimeError("no FA leaf with growth headroom found")
    # Pick the most-developed candidate so the FA target this step is
    # firmly above 0 (avoids the pre-emergence phase where demand=0 and
    # the cap can't fire).
    candidates.sort(key=lambda t: t[0], reverse=True)
    return candidates[0][1]


def test_dl_backlog_zero_when_demand_is_null(maize_day_15):
    """Sanity: bare CWLimitedGrowth (demand_==nullptr) must not touch dl_backlog.

    PLAN §M2: dl_backlog is the sink-source mismatch carry; when there's
    no demand wrap (root case, scalar-leaf case), the accumulator must
    stay at the default 0.0 and the backward-compat path is preserved
    for non-FA XMLs / roots.
    """
    enable_cw_limited_growth(maize_day_15)
    for o in maize_day_15.getOrgans():
        # Only assert on non-FA organs whose param has demand=None.
        rp = o.getOrganRandomParameter()
        if not isinstance(rp.f_gf, pb.CWLimitedGrowth):
            continue
        if rp.f_gf.demand is not None:
            continue
        assert o.dl_backlog == 0.0, (
            f"organ id={o.getId()} ot={o.organType()} st={o.getParameter('subType')}: "
            f"dl_backlog={o.dl_backlog} on bare-CW-supply organ; expected 0.0"
        )


def test_dl_backlog_accumulates_under_synthetic_stress():
    """Inject CW_Gr below the FA target and verify backlog grows.

    Builds a plant fresh (this test mutates supply, so doesn't share the
    module fixture), wraps it, picks an FA leaf in mid-growth, and
    injects a CW_Gr that's smaller than the FA demand for that step.
    Verifies post-step:
      * Realised length grew by exactly the supply amount (cap fired).
      * dl_backlog accumulated the unmet portion.
    Then injects a generous CW_Gr and verifies backlog drains to 0.
    """
    plant = grow_plant(
        xml_path=str(XML_PATH),
        simulation_time=30,
        seed=7,
        enable_photosynthesis=True,
    )
    enable_cw_limited_growth(plant)

    leaf = _find_fa_leaf(plant)
    leaf_id = leaf.getId()
    leaf_rp = leaf.getOrganRandomParameter()
    starting_length = leaf.getLength()
    starting_backlog = leaf.dl_backlog

    # Inject a tiny supply (much less than the per-step FA target). All
    # organ types use the same map; the leaf's f_gf reads its own id.
    SUPPLY_CM = 0.01
    cw_map = {leaf_id: SUPPLY_CM}
    leaf_rp.f_gf.CW_Gr = cw_map

    plant.simulate(1.0, False)

    after_stress_length = leaf.getLength()
    after_stress_backlog = leaf.dl_backlog
    delivered = after_stress_length - starting_length

    assert delivered > 0, (
        f"leaf {leaf_id}: no growth delivered ({delivered} cm); supply was {SUPPLY_CM}"
    )
    assert delivered <= SUPPLY_CM + 1e-6, (
        f"leaf {leaf_id}: delivered {delivered} cm > injected supply {SUPPLY_CM}; "
        "supply cap should bind under synthetic stress"
    )
    assert after_stress_backlog > starting_backlog, (
        f"leaf {leaf_id}: dl_backlog did not accumulate under stress "
        f"(was {starting_backlog}, now {after_stress_backlog})"
    )

    # Recovery — inject generous supply (large enough to cover backlog +
    # this step's FA target). Backlog should drain to ~0.
    GENEROUS_CM = 50.0
    leaf_rp.f_gf.CW_Gr = {leaf_id: GENEROUS_CM}
    plant.simulate(1.0, False)
    assert leaf.dl_backlog < 1e-6, (
        f"leaf {leaf_id}: dl_backlog={leaf.dl_backlog} did not drain after "
        f"generous supply step"
    )


# ---------------------------------------------------------------------------
# G6 — source-grep guard
# ---------------------------------------------------------------------------
def test_no_blanket_f_gf_overwrite_in_production_code():
    """No production code path may bypass enable_cw_limited_growth.

    Searches dart/coupling/ for `param.f_gf = ...` assignments and asserts
    each is inside ``carbon_growth.py::enable_cw_limited_growth`` (the
    sanctioned wrap helper) or inside a test/baseline script.

    Catches regressions like the pre-Lock-#9 blanket overwrite where a
    new feature path bypasses the wrap policy and re-introduces the
    silent-FA-clobber.
    """
    suspicious: list[tuple[Path, int, str]] = []
    pattern = re.compile(r"\bf_gf\s*=")
    skip_dirs = {"tests", "experimental", "__pycache__"}
    sanctioned_files = {COUPLING_DIR / "growth" / "carbon_growth.py"}

    for path in COUPLING_DIR.rglob("*.py"):
        if any(part in skip_dirs for part in path.relative_to(COUPLING_DIR).parts):
            continue
        if path in sanctioned_files:
            continue
        text = path.read_text()
        for lineno, line in enumerate(text.splitlines(), 1):
            if pattern.search(line):
                # Allow comments / docstrings — only assignments matter
                stripped = line.lstrip()
                if stripped.startswith("#"):
                    continue
                suspicious.append((path, lineno, line.rstrip()))

    if suspicious:
        msg = "Unsanctioned f_gf assignments found (must go through enable_cw_limited_growth):\n"
        for path, lineno, line in suspicious:
            msg += f"  {path.relative_to(REPO_ROOT)}:{lineno}: {line}\n"
        pytest.fail(msg)


# ---------------------------------------------------------------------------
# G1 — D.0 6-XML bit-identical
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_d0_6xml_bit_identical():
    """Existing D.0 6-XML regression suite still passes (G1 acceptance gate).

    Subprocess call to capture_d0_baselines.py --verify. Slow (~5 min)
    so guarded by `-m "not slow"`-style markers.
    """
    script = BASELINES_DIR / "capture_d0_baselines.py"
    result = subprocess.run(
        [sys.executable, str(script), "--verify"],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        timeout=600,
    )
    assert result.returncode == 0, (
        f"D.0 6-XML verify failed:\n--- stdout ---\n{result.stdout[-2000:]}\n"
        f"--- stderr ---\n{result.stderr[-2000:]}"
    )
    assert "PASSED" in result.stdout, f"D.0 verify missing PASSED tag:\n{result.stdout[-500:]}"


@pytest.mark.slow
def test_s5_oracle_bit_identical():
    """S5 oracle re-capture matches the stored fixture (G1 acceptance gate).

    Subprocess call to capture_oracle_fa_no_carbon_day130.py --verify.
    Slow (~3 min). This is the "FA-on no-carbon path is unchanged"
    regression — proves Lock #6 + Lock #9 + dl_backlog don't disturb
    the production geometry path.
    """
    script = BASELINES_DIR / "capture_oracle_fa_no_carbon_day130.py"
    result = subprocess.run(
        [sys.executable, str(script), "--verify"],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        timeout=600,
    )
    assert result.returncode == 0, (
        f"S5 oracle verify failed:\n--- stdout ---\n{result.stdout[-2000:]}\n"
        f"--- stderr ---\n{result.stderr[-2000:]}"
    )
    assert "OK (matches oracle)" in result.stdout, (
        f"S5 oracle verify did not match:\n{result.stdout[-1000:]}"
    )
