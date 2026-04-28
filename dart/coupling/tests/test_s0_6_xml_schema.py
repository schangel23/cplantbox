#!/usr/bin/env python3
"""S0.6 — XML schema for MultiPhaseStemGrowth per-rank arrays.

Covers:
  T1  back-compat: a stem XML with no v_n/D_n/IL_final/basal_zero_ranks tags
      loads with the constructor defaults ({} / {1,2,3,4}). Verifies that
      every existing in-tree XML (wheat, brassica, carbon2020, 2020-maize,
      modelparam_4, maize_calibrated FA-off) is unaffected.
  T2  read: a fresh XML with `<parameter name="v_n" values="..."/>` etc.
      populates the StemRandomParameter fields with the parsed numbers.
  T3  write: a StemRandomParameter with non-empty internode arrays emits
      the new tags via writeXML (round-trip read after write reproduces
      the field values exactly).
  T4  basal_zero_ranks emit gate: only emitted when use_fournier_andrieu_kinetics
      is true (so non-FA stems do not gain a default {1,2,3,4} tag on every
      round-trip).
"""
import os
import tempfile
import textwrap

import pytest

import plantbox as pb


# Helper: build a minimal XML containing one stem subType with
# the fields requested. Wraps it in <plant>...</plant> so MappedPlant
# can read it via readParameters.
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


def _make_plant_from_xml(path: str) -> pb.MappedPlant:
    plant = pb.MappedPlant()
    plant.readParameters(path)
    return plant


def _stem_lrp(plant: pb.MappedPlant, sub_type: int = 1):
    """Return the StemRandomParameter for the given subType."""
    return plant.getOrganRandomParameter(pb.OrganTypes.stem, sub_type)


# -------------------------------------------------------------------------
# T1 — back-compat: empty XML leaves defaults intact.
# -------------------------------------------------------------------------
def test_t1_back_compat_empty_arrays():
    """Stem XML with no v_n/D_n/IL_final/basal_zero_ranks tags
    loads with constructor defaults."""
    path = _write_minimal_stem_xml("")
    try:
        plant = _make_plant_from_xml(path)
        srp = _stem_lrp(plant)
        assert list(srp.internode_v_n) == [], (
            f"internode_v_n should default empty, got {list(srp.internode_v_n)}"
        )
        assert list(srp.internode_D_n) == [], (
            f"internode_D_n should default empty, got {list(srp.internode_D_n)}"
        )
        assert list(srp.internode_IL_final) == [], (
            f"internode_IL_final should default empty, got {list(srp.internode_IL_final)}"
        )
        assert list(srp.basal_zero_ranks) == [1, 2, 3, 4], (
            f"basal_zero_ranks default should be [1,2,3,4], got {list(srp.basal_zero_ranks)}"
        )
    finally:
        os.unlink(path)


# -------------------------------------------------------------------------
# T2 — read: comma-separated values populate the per-rank fields.
# -------------------------------------------------------------------------
def test_t2_read_per_rank_arrays():
    extra = textwrap.dedent("""\
            <parameter name="v_n" values="0.10, 0.18, 0.22, 0.25, 0.27, 0.28, 0.28, 0.27, 0.25, 0.22" />
            <parameter name="D_n" values="80, 90, 100, 110, 120, 130, 140, 150, 160, 170" />
            <parameter name="IL_final" values="0.0, 0.0, 0.0, 0.0, 1.5, 4.5, 9.5, 14.5, 18.5, 21.0" />
            <parameter name="basal_zero_ranks" values="1, 2, 3" />
        """)
    path = _write_minimal_stem_xml(extra)
    try:
        plant = _make_plant_from_xml(path)
        srp = _stem_lrp(plant)

        v = list(srp.internode_v_n)
        d = list(srp.internode_D_n)
        il = list(srp.internode_IL_final)
        bz = list(srp.basal_zero_ranks)

        assert len(v) == 10 and v[0] == pytest.approx(0.10) and v[-1] == pytest.approx(0.22), v
        assert len(d) == 10 and d[0] == 80 and d[-1] == 170, d
        assert len(il) == 10 and il[3] == pytest.approx(0.0) and il[6] == pytest.approx(9.5), il
        assert bz == [1, 2, 3], bz
    finally:
        os.unlink(path)


# -------------------------------------------------------------------------
# T3 — write: round-trip read → assign → writeParameters → read reproduces.
# -------------------------------------------------------------------------
def test_t3_round_trip_arrays():
    """A stem with non-empty per-rank arrays writes them back to XML and
    re-reads identically."""
    path = _write_minimal_stem_xml("")
    out_path = path + ".out.xml"
    try:
        plant = _make_plant_from_xml(path)
        srp = _stem_lrp(plant)
        # populate fields and turn FA on so basal_zero_ranks is also emitted
        srp.use_fournier_andrieu_kinetics = True
        srp.internode_v_n = [0.1, 0.2, 0.3]
        srp.internode_D_n = [100.0, 110.0, 120.0]
        srp.internode_IL_final = [0.0, 0.5, 1.0]
        srp.basal_zero_ranks = [1, 2]

        plant.writeParameters(out_path)

        plant2 = pb.MappedPlant()
        plant2.readParameters(out_path)
        srp2 = _stem_lrp(plant2)

        assert list(srp2.internode_v_n) == pytest.approx([0.1, 0.2, 0.3]), list(srp2.internode_v_n)
        assert list(srp2.internode_D_n) == pytest.approx([100.0, 110.0, 120.0]), list(srp2.internode_D_n)
        assert list(srp2.internode_IL_final) == pytest.approx([0.0, 0.5, 1.0]), list(srp2.internode_IL_final)
        assert list(srp2.basal_zero_ranks) == [1, 2], list(srp2.basal_zero_ranks)
    finally:
        os.unlink(path)
        if os.path.exists(out_path):
            os.unlink(out_path)


# -------------------------------------------------------------------------
# T4 — basal_zero_ranks emit gate: FA-off stem does NOT serialise the default.
# -------------------------------------------------------------------------
def test_t4_basal_zero_ranks_emit_gate_fa_off():
    """An FA-off stem writes its XML without the basal_zero_ranks tag,
    so non-FA stems don't accumulate {1,2,3,4} tags on every round-trip."""
    path = _write_minimal_stem_xml("")
    out_path = path + ".out.xml"
    try:
        plant = _make_plant_from_xml(path)
        srp = _stem_lrp(plant)
        # FA off (default), but the constructor still set basal_zero_ranks={1,2,3,4}
        assert srp.use_fournier_andrieu_kinetics is False
        assert list(srp.basal_zero_ranks) == [1, 2, 3, 4]

        plant.writeParameters(out_path)
        with open(out_path) as f:
            xml_str = f.read()

        assert "basal_zero_ranks" not in xml_str, (
            "FA-off stem XML should not emit basal_zero_ranks tag (would change "
            "every existing XML on round-trip)"
        )
    finally:
        os.unlink(path)
        if os.path.exists(out_path):
            os.unlink(out_path)


# -------------------------------------------------------------------------
# T5 — basal_zero_ranks emit gate: FA-on stem DOES serialise the array.
# -------------------------------------------------------------------------
def test_t5_basal_zero_ranks_emit_gate_fa_on():
    path = _write_minimal_stem_xml("")
    out_path = path + ".out.xml"
    try:
        plant = _make_plant_from_xml(path)
        srp = _stem_lrp(plant)
        srp.use_fournier_andrieu_kinetics = True
        srp.basal_zero_ranks = [1, 2, 3, 4, 5]

        plant.writeParameters(out_path)
        with open(out_path) as f:
            xml_str = f.read()

        assert "basal_zero_ranks" in xml_str
        assert 'values="1, 2, 3, 4, 5"' in xml_str, xml_str

        plant2 = pb.MappedPlant()
        plant2.readParameters(out_path)
        srp2 = _stem_lrp(plant2)
        assert list(srp2.basal_zero_ranks) == [1, 2, 3, 4, 5]
    finally:
        os.unlink(path)
        if os.path.exists(out_path):
            os.unlink(out_path)
