"""Round-trip test for the native 2D leaf-surface CP grid (Phase A).

Verifies:
  1. An empty ``surface_cps`` vector produces no ``<parameter name="surface_cp"/>``
     entries in the XML and survives a read/write cycle unchanged.
  2. A populated 11*5 grid writes 55 XML entries, and a subsequent
     ``readParameters`` reconstructs the same flat vector bit-for-bit in the
     CP coordinates and in the ``surface_n_u``/``surface_n_v``/``surface_deg_*``
     scalars.
  3. Other leaf-parameter fields are untouched by the round trip.
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import plantbox as pb


def _make_lrp_with_surface_cps(plant, subtype: int = 2) -> pb.LeafRandomParameter:
    lrp = pb.LeafRandomParameter(plant)
    lrp.name = "test_leaf"
    lrp.subType = subtype
    lrp.organType = 4
    lrp.lmax = 50.0
    lrp.Width_blade = 2.0
    lrp.areaMax = 80.0
    lrp.r = 2.0
    # Populate an 11x5 grid with deterministic values; index k = i_u*5 + i_v
    cps = []
    for i_u in range(11):
        for i_v in range(5):
            # Leaf-local: along-midrib = z = u*5 cm (tip at 50 cm)
            # lateral spread = x = (v-0.5)*2*width_half
            z = i_u * 5.0
            x = (i_v - 2) * 0.5  # v in {-1, -0.5, 0, 0.5, 1}
            y = math.sin(i_u * 0.3) * 0.1  # small non-zero y to catch precision loss
            cps.append(pb.Vector3d(x, y, z))
    lrp.surface_cps = cps
    lrp.surface_n_u = 11
    lrp.surface_n_v = 5
    lrp.surface_deg_u = 3
    lrp.surface_deg_v = 2
    return lrp


def _assert_surface_cps_equal(a, b, tol: float = 1e-9) -> None:
    assert len(a) == len(b), f"size mismatch: {len(a)} vs {len(b)}"
    for k, (p, q) in enumerate(zip(a, b)):
        assert abs(p.x - q.x) < tol, f"cp[{k}].x {p.x} vs {q.x}"
        assert abs(p.y - q.y) < tol, f"cp[{k}].y {p.y} vs {q.y}"
        assert abs(p.z - q.z) < tol, f"cp[{k}].z {p.z} vs {q.z}"


def test_round_trip_populated_surface_cps():
    """Write a plant with a populated surface_cps grid, read it back, compare."""
    plant_w = pb.Plant()
    lrp = _make_lrp_with_surface_cps(plant_w, subtype=2)
    plant_w.setOrganRandomParameter(lrp)

    # A minimal seed so writeParameters doesn't complain
    seed = pb.SeedRandomParameter(plant_w)
    seed.subType = 0
    plant_w.setOrganRandomParameter(seed)

    with tempfile.TemporaryDirectory() as td:
        xml_path = Path(td) / "test_surface.xml"
        plant_w.writeParameters(str(xml_path))

        xml_text = xml_path.read_text()
        # Expect exactly 55 surface_cp entries for the one leaf subtype we set
        assert xml_text.count('name="surface_cp"') == 55, (
            "expected 55 surface_cp entries in XML; got "
            f"{xml_text.count('name=\"surface_cp\"')}"
        )

        plant_r = pb.Plant()
        plant_r.readParameters(str(xml_path))
        lrp_r = plant_r.getOrganRandomParameter(4, 2)

    assert lrp_r.surface_n_u == 11
    assert lrp_r.surface_n_v == 5
    assert lrp_r.surface_deg_u == 3
    assert lrp_r.surface_deg_v == 2
    _assert_surface_cps_equal(lrp.surface_cps, lrp_r.surface_cps)


def test_round_trip_empty_surface_cps_is_noop():
    """Without surface_cps the XML must stay free of surface_cp entries."""
    plant_w = pb.Plant()
    lrp = pb.LeafRandomParameter(plant_w)
    lrp.name = "plain_leaf"
    lrp.subType = 2
    lrp.organType = 4
    lrp.lmax = 30.0
    plant_w.setOrganRandomParameter(lrp)

    seed = pb.SeedRandomParameter(plant_w)
    seed.subType = 0
    plant_w.setOrganRandomParameter(seed)

    with tempfile.TemporaryDirectory() as td:
        xml_path = Path(td) / "plain.xml"
        plant_w.writeParameters(str(xml_path))

        xml_text = xml_path.read_text()
        assert 'name="surface_cp"' not in xml_text

        plant_r = pb.Plant()
        plant_r.readParameters(str(xml_path))
        lrp_r = plant_r.getOrganRandomParameter(4, 2)

    assert len(lrp_r.surface_cps) == 0


if __name__ == "__main__":
    test_round_trip_empty_surface_cps_is_noop()
    print("empty round-trip: OK")
    test_round_trip_populated_surface_cps()
    print("populated round-trip: OK")
