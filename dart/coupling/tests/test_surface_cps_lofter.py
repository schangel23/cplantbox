"""Phase C: the lofter consumes native surface_cps from the LRP.

Checks that when a plant's ``LeafRandomParameter`` carries a populated
``surface_cps`` grid, the adapter extracts it into ``surface_cps_local`` +
collar frame metadata and the lofter produces a non-degenerate mesh via
``loft_leaf_nurbs``'s library path (not the quad-ribbon fallback).

Uses the calibrated maize XML produced with ``--surface-cps`` as a
realistic end-to-end fixture.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import plantbox as pb

from dart.coupling.geometry.cplantbox_adapter import extract_organs_for_lofter
from dart.coupling.geometry.g1_to_g3 import loft_organs


_REPO = Path(__file__).resolve().parents[3]


def _make_calibrated_xml_with_surface_cps(tmp_dir: Path) -> Path:
    """Build a calibrated maize XML with surface_cps populated."""
    out_path = tmp_dir / "maize_surface.xml"
    template = _REPO / "modelparameter" / "structural" / "plant" / "2020-maize.xml"
    mf3d = Path("/home/lukas/PHD/Resources/MaizeField3d/maizefield3d_stats.json")
    env = {**os.environ}
    subprocess.run(
        [
            sys.executable, "-m", "dart.coupling", "calibrate",
            "--template", str(template),
            "--output", str(out_path),
            "--maizefield3d", str(mf3d),
            "--surface-cps",
        ],
        check=True, cwd=str(_REPO), env=env,
        capture_output=True,
    )
    assert out_path.exists()
    # Sanity: 11 subtypes * 55 CPs = 605 surface_cp entries
    n_cps = out_path.read_text().count('name="surface_cp"')
    assert n_cps == 605, f"expected 605 surface_cp entries; got {n_cps}"
    return out_path


def _grow_plant(xml_path: Path, days: int = 30) -> pb.Plant:
    plant = pb.Plant()
    plant.readParameters(str(xml_path))
    plant.initialize(True)
    plant.simulate(days)
    return plant


def test_adapter_emits_library_organ_dict():
    """A leaf with surface_cps must be extracted via the library path."""
    with tempfile.TemporaryDirectory() as td:
        xml_path = _make_calibrated_xml_with_surface_cps(Path(td))
        plant = _grow_plant(xml_path, days=30)
        organs = extract_organs_for_lofter(plant)

    leaves_lib = [o for o in organs if o.get("surface_cps_local") is not None]
    leaves_total = [o for o in organs if o["type"] == "leaf"]
    assert len(leaves_total) > 0, "plant must have grown at least one leaf"
    assert len(leaves_lib) == len(leaves_total), (
        f"expected all {len(leaves_total)} leaves via library; "
        f"got {len(leaves_lib)} library + "
        f"{len(leaves_total) - len(leaves_lib)} skeleton"
    )

    leaf = leaves_lib[0]
    assert leaf["surface_cps_local"].shape == (11, 5, 3)
    assert leaf["use_nurbs_backend"] is True
    assert leaf["collar_pos"].shape == (3,)
    assert leaf["collar_tangent"].shape == (3,)
    assert leaf["mature_length"] > 0.0
    assert leaf["current_length"] > 0.0


def test_lofter_produces_mesh_via_library_path():
    """loft_organs must accept library-CP organ dicts and emit triangles."""
    with tempfile.TemporaryDirectory() as td:
        xml_path = _make_calibrated_xml_with_surface_cps(Path(td))
        plant = _grow_plant(xml_path, days=30)
        organs = extract_organs_for_lofter(plant)
        mesh = loft_organs(organs, smooth=False)

    assert len(mesh.vertices) > 0
    assert len(mesh.indices) > 0
    v = np.asarray(mesh.vertices)
    tris = np.asarray(mesh.indices)
    # Pick out triangles contributed by leaves (by organ_id) — sheath/stem
    # meshers may produce tight triangles that would fail the area check.
    organ_ids = np.asarray(mesh.organ_ids)
    leaf_ids = {o["organ_id"] for o in organs
                if o["type"] == "leaf" and o.get("surface_cps_local") is not None}
    mask = np.isin(organ_ids, list(leaf_ids))
    leaf_tris = tris[mask]
    assert len(leaf_tris) > 0, "no leaf triangles produced"
    e1 = v[leaf_tris[:, 1]] - v[leaf_tris[:, 0]]
    e2 = v[leaf_tris[:, 2]] - v[leaf_tris[:, 0]]
    areas = 0.5 * np.linalg.norm(np.cross(e1, e2), axis=1)
    assert float(areas.min()) > 1e-6, f"degenerate tri min area {areas.min()}"


if __name__ == "__main__":
    test_adapter_emits_library_organ_dict()
    print("adapter library path: OK")
    test_lofter_produces_mesh_via_library_path()
    print("lofter library path: OK")
