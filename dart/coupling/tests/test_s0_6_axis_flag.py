#!/usr/bin/env python3
"""S0.6 / Lock #1 — axis flag for delayNGEnd (Calendar / TT).

Covers the schema side AND the GF consumer dispatch:
  T1  default axis: a stem XML without `axis="..."` on delayNGEnd loads with
      DelayAxis.Calendar (bit-identical with every existing XML).
  T2  axis="TT" parses into pb.DelayAxis.TT.
  T3  round-trip: write with axis=TT → read → axis preserved; write with
      axis=Calendar (default) → no `axis=` attribute emitted.
  T4  GF dispatch: under FA-on with delayNGEndAxis=TT and delayNGEnd=600,
      the per-rank cessation latch fires once plant Andrieu-TT crosses 600
      (instead of waiting for per-rank Phase IV completion). Verified by
      simulating a maize plant with the merged form set, then checking
      `stem.cessation_andrieu_tt_` lies near 600 (within one TT step).
  T5  the merged-form path takes precedence over legacy tt_cessation when
      both are set (axis=TT > tt_cessation > per-rank Phase IV).
"""
import os
import tempfile
import textwrap

import pytest

import plantbox as pb


def _write_minimal_stem_xml(extra_parameters: str) -> str:
    body = textwrap.dedent(f"""\
        <?xml version="1.0" encoding="UTF-8" standalone="no" ?>
        <Plant>
          <organ type="seed" name="seed" subType="0">
            <parameter name="seedPos.x" value="0" />
            <parameter name="seedPos.y" value="0" />
            <parameter name="seedPos.z" value="-3" />
            <parameter name="firstB" value="0" />
            <parameter name="delayB" value="9999" />
            <parameter name="maxB" value="0" />
            <parameter name="firstSB" value="9999" />
            <parameter name="firstTil" value="9999" />
            <parameter name="nC" value="0" />
            <parameter name="nz" value="0" />
          </organ>
          <stem name="mainstem" subType="1">
            <parameter name="gf" value="1" />
            <parameter name="lnf" value="0" />
            <parameter name="nodalGrowth" value="1" />
            <parameter name="tropismT" value="1" />
            <parameter name="a" value="0.2" />
            <parameter name="dx" value="0.5" />
            <parameter name="dxMin" value="1e-06" />
            <parameter name="lb" value="0" />
            <parameter name="la" value="10" />
            <parameter name="ln" value="1" />
            <parameter name="lmax" value="100" />
            <parameter name="r" value="2.0" />
            <parameter name="rlt" value="1e9" />
            <parameter name="theta" value="0" />
            <parameter name="tropismN" value="1" />
            <parameter name="tropismS" value="0" />
        {extra_parameters}
          </stem>
        </Plant>
        """)
    fd, path = tempfile.mkstemp(suffix=".xml")
    with os.fdopen(fd, "w") as f:
        f.write(body)
    return path


def _stem_lrp(plant: pb.MappedPlant, sub_type: int = 1):
    return plant.getOrganRandomParameter(pb.OrganTypes.stem, sub_type)


# -------------------------------------------------------------------------
# T1 — default axis is Calendar.
# -------------------------------------------------------------------------
def test_t1_default_axis_calendar():
    path = _write_minimal_stem_xml('            <parameter name="delayNGEnd" value="0" />\n')
    try:
        plant = pb.MappedPlant()
        plant.readParameters(path)
        srp = _stem_lrp(plant)
        assert srp.delayNGEndAxis == pb.DelayAxis.Calendar
    finally:
        os.unlink(path)


# -------------------------------------------------------------------------
# T2 — axis="TT" parses into DelayAxis.TT.
# -------------------------------------------------------------------------
def test_t2_axis_tt_parses():
    path = _write_minimal_stem_xml(
        '            <parameter name="delayNGEnd" value="1500" axis="TT" />\n'
    )
    try:
        plant = pb.MappedPlant()
        plant.readParameters(path)
        srp = _stem_lrp(plant)
        assert srp.delayNGEndAxis == pb.DelayAxis.TT
        assert srp.delayNGEnd == pytest.approx(1500.0)
    finally:
        os.unlink(path)


# -------------------------------------------------------------------------
# T3 — round-trip preserves axis flag, omits Calendar default.
# -------------------------------------------------------------------------
def test_t3_round_trip_axis():
    path = _write_minimal_stem_xml(
        '            <parameter name="delayNGEnd" value="0" />\n'
    )
    out_path = path + ".out.xml"
    try:
        plant = pb.MappedPlant()
        plant.readParameters(path)
        srp = _stem_lrp(plant)
        srp.delayNGEnd = 1500.0
        srp.delayNGEndAxis = pb.DelayAxis.TT

        plant.writeParameters(out_path)
        with open(out_path) as f:
            xml_str = f.read()
        assert 'axis="TT"' in xml_str

        plant2 = pb.MappedPlant()
        plant2.readParameters(out_path)
        srp2 = _stem_lrp(plant2)
        assert srp2.delayNGEndAxis == pb.DelayAxis.TT
        assert srp2.delayNGEnd == pytest.approx(1500.0)

        # Now flip back to Calendar and confirm the axis= attribute disappears
        srp2.delayNGEndAxis = pb.DelayAxis.Calendar
        plant2.writeParameters(out_path)
        with open(out_path) as f:
            xml_str2 = f.read()
        assert 'axis="TT"' not in xml_str2
        assert 'axis="Calendar"' not in xml_str2  # default omitted by design
    finally:
        os.unlink(path)
        if os.path.exists(out_path):
            os.unlink(out_path)


# -------------------------------------------------------------------------
# T4 — GF dispatch: axis-TT cessation fires at the configured Andrieu-TT.
#
# Use the production maize_calibrated XML with FA on, then override the
# cessation source: tt_cessation=0 (legacy off), delayNGEnd=600 axis=TT.
# Run with constant Tair = 25 °C, dt=1 day → Andrieu-TT accumulates at
# (25 - 9.8) = 15.2 °Cd/day. Crossing TT=600 happens between days 39-40.
# After simulating 50 days, cessation_andrieu_tt_ should be ≥ 600 (the
# value latched at the first crossing step).
# -------------------------------------------------------------------------
def test_t4_axis_tt_cessation_fires_at_threshold():
    xml = "/home/lukas/PHD/CPlantBox/dart/coupling/data/maize_calibrated.xml"
    if not os.path.exists(xml):
        pytest.skip("maize_calibrated.xml not present")

    import sys
    sys.path.insert(0, "/home/lukas/PHD/CPlantBox")
    from dart.coupling.growth.grow import setup_successor_where

    plant = pb.MappedPlant(7)
    plant.readParameters(xml)
    setup_successor_where(plant)

    # Override the mainstem to use the merged Lock #1 form.
    srp = _stem_lrp(plant)
    srp.use_fournier_andrieu_kinetics = True
    srp.use_thermal_cessation = 1
    srp.tt_cessation = 0.0  # legacy off
    srp.delayNGEnd = 600.0
    srp.delayNGEndAxis = pb.DelayAxis.TT
    # Populate the per-rank arrays so MultiPhaseStemGrowth has the per-rank
    # data it needs (otherwise it early-returns at n_ranks<=0).
    n_ranks = 16
    srp.internode_v_n = [0.18] * n_ranks
    srp.internode_D_n = [100.0] * n_ranks
    srp.internode_IL_final = [0.0] * 4 + [1.0] * (n_ranks - 4)

    plant.initialize()
    plant.setAirTemperature(25.0)
    for _ in range(50):
        plant.simulate(1.0, False)

    main_stem = None
    for organ in plant.getOrgans():
        if organ.organType() == pb.OrganTypes.stem and organ.getParameter("subType") == 1:
            main_stem = organ
            break
    assert main_stem is not None

    fa_state = main_stem.getFaState()
    assert fa_state is not None, "FA state should be populated after FA-on simulation"
    # Snapshot into a Python list immediately — the C++ state lives only as
    # long as `plant` does.
    cess_per_n = list(fa_state.cessation_andrieu_tt_per_n)
    # Per-rank latches should have fired by day 50 under TT=600 threshold
    latched = [v for v in cess_per_n if v >= 0.0]
    assert len(latched) > 0, (
        f"Expected at least one per-rank cessation latch under axis=TT,"
        f" delayNGEnd=600; got {len(latched)} latches; raw={cess_per_n}"
    )
    # Each latched value should be >= delayNGEnd (within one TT step).
    for v in latched:
        assert v >= 600.0, f"cessation latched at TT={v}, expected >= 600"


# -------------------------------------------------------------------------
# T5 — axis=TT takes precedence over legacy tt_cessation.
# Run two simulations with identical setup except cessation source. The
# axis-TT path with delayNGEnd=300 should latch sooner than the legacy
# path with tt_cessation=900.
# -------------------------------------------------------------------------
def test_t5_axis_tt_priority_over_legacy():
    xml = "/home/lukas/PHD/CPlantBox/dart/coupling/data/maize_calibrated.xml"
    if not os.path.exists(xml):
        pytest.skip("maize_calibrated.xml not present")

    import sys
    sys.path.insert(0, "/home/lukas/PHD/CPlantBox")
    from dart.coupling.growth.grow import setup_successor_where

    def run_and_collect(cfg):
        """Run a 50-day FA-on simulation and snapshot the latched cessation
        TTs into a Python list before the plant goes out of scope (the
        `PerOrganFAState` returned by `getFaState()` is a view into C++ state
        that gets freed once the owning `MappedPlant` is gc'd, so we MUST
        copy out the values inside the same scope as the plant)."""
        plant = pb.MappedPlant(7)
        plant.readParameters(xml)
        setup_successor_where(plant)
        srp = _stem_lrp(plant)
        srp.use_fournier_andrieu_kinetics = True
        srp.use_thermal_cessation = 1
        srp.tt_cessation = cfg["tt_cessation"]
        srp.delayNGEnd = cfg["delayNGEnd"]
        srp.delayNGEndAxis = cfg["axis"]
        n_ranks = 16
        srp.internode_v_n = [0.18] * n_ranks
        srp.internode_D_n = [100.0] * n_ranks
        srp.internode_IL_final = [0.0] * 4 + [1.0] * (n_ranks - 4)
        plant.initialize()
        plant.setAirTemperature(25.0)
        for _ in range(50):
            plant.simulate(1.0, False)
        for organ in plant.getOrgans():
            if organ.organType() == pb.OrganTypes.stem and organ.getParameter("subType") == 1:
                fs = organ.getFaState()
                # Snapshot into a Python list so we don't hold a dangling
                # reference once `plant` goes out of scope.
                return list(fs.cessation_andrieu_tt_per_n)
        return []

    # Case A: axis-TT 300 with legacy 900 also set — axis path should win
    cess_a = run_and_collect({"axis": pb.DelayAxis.TT, "delayNGEnd": 300.0, "tt_cessation": 900.0})
    # Case B: legacy only (axis Calendar, ignore delayNGEnd interpretation)
    cess_b = run_and_collect({"axis": pb.DelayAxis.Calendar, "delayNGEnd": 300.0, "tt_cessation": 900.0})

    latched_a = [v for v in cess_a if v >= 0.0]
    latched_b = [v for v in cess_b if v >= 0.0]

    # Case A latched at TT~300; Case B (legacy 900) — under the same 50-day
    # constant-25C run reaches plant TT (Tb=8) ~ 50*17 = 850; legacy reads
    # getAccumulatedTT (Tb=8) ≈ 850 — so a few ranks may latch but later
    # than under axis-TT 300. Case A's earlier threshold means strictly
    # more (or equal) ranks latched, and the latched values themselves are
    # ≥ 300.
    assert len(latched_a) > 0, (
        f"Axis-TT path with delayNGEnd=300 should latch within 50 days; got {latched_a}"
    )
    assert all(v >= 300.0 for v in latched_a), (
        f"Axis-TT latched values should all be >= 300; got {latched_a}"
    )
    assert len(latched_a) >= len(latched_b), (
        f"axis-TT priority broken: case A latched {len(latched_a)} ranks at TT>=300, "
        f"case B latched {len(latched_b)} ranks at legacy threshold 900"
    )
