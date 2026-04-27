"""Phase D: native C++ CP-driven midrib evolution during ``simulate``.

When a leaf's ``LeafRandomParameter`` carries a populated 2D surface CP grid,
``Plant::simulate`` must re-project internal midrib nodes onto the library-
derived midrib *after* ``rel2abs()`` so collars and tangents are in world
coordinates. This test:

  1. Grows a calibrated maize with ``surface_cps`` set.
  2. Verifies no NaN node positions.
  3. Verifies a CP-enabled run differs from a plain-XML run in midrib
     geometry (the library polyline shape is observable, not just the quad
     ribbon's width profile).
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import plantbox as pb


_REPO = Path(__file__).resolve().parents[3]


def _calibrate(tmp_dir: Path, *, surface_cps: bool) -> Path:
    out = tmp_dir / ("maize_sd.xml" if surface_cps else "maize_plain.xml")
    args = [
        sys.executable, "-m", "dart.coupling", "calibrate",
        "--template",
        str(_REPO / "modelparameter" / "structural" / "plant" / "2020-maize.xml"),
        "--output", str(out),
        "--maizefield3d",
        "/home/lukas/PHD/Resources/MaizeField3d/maizefield3d_stats.json",
    ]
    if surface_cps:
        args.append("--surface-cps")
    subprocess.run(
        args, check=True, cwd=str(_REPO), capture_output=True,
        env={**os.environ},
    )
    return out


def _grow_and_collect(xml: Path, days: int = 30):
    plant = pb.Plant()
    plant.readParameters(str(xml))
    plant.initialize(True)
    plant.simulate(days)
    leaves = plant.getOrgans(pb.leaf)
    out = []
    for lf in leaves:
        nn = lf.getNumberOfNodes()
        pts = np.array([
            [lf.getNode(i).x, lf.getNode(i).y, lf.getNode(i).z] for i in range(nn)
        ])
        out.append(pts)
    return out


def test_no_nan_after_native_cp_evolve():
    with tempfile.TemporaryDirectory() as td:
        xml = _calibrate(Path(td), surface_cps=True)
        leaves = _grow_and_collect(xml, days=30)
    assert len(leaves) > 0, "expected at least one leaf grown"
    for i, pts in enumerate(leaves):
        assert not np.isnan(pts).any(), f"leaf {i} has NaN node positions"


def test_cp_midrib_differs_from_plain():
    """With the CP grid active, midrib node coordinates must differ from the
    tropism-only baseline. We compare per-leaf distance sets (scan ordering
    is deterministic within a given XML but the calibration reshapes geometry
    between runs, so compare gross shape statistics on matched indices)."""
    with tempfile.TemporaryDirectory() as td_a, tempfile.TemporaryDirectory() as td_b:
        xml_cps = _calibrate(Path(td_a), surface_cps=True)
        xml_plain = _calibrate(Path(td_b), surface_cps=False)
        leaves_cps = _grow_and_collect(xml_cps, days=30)
        leaves_plain = _grow_and_collect(xml_plain, days=30)

    assert len(leaves_cps) == len(leaves_plain) > 0

    # At least one leaf must have a notably different midrib path.
    any_diff = False
    for a, b in zip(leaves_cps, leaves_plain):
        n = min(len(a), len(b))
        if n < 3:
            continue
        da = np.linalg.norm(np.diff(a[:n], axis=0), axis=1).sum()
        db = np.linalg.norm(np.diff(b[:n], axis=0), axis=1).sum()
        # Arc length or intermediate point positions should differ.
        mid_a = a[n // 2]
        mid_b = b[n // 2]
        if abs(da - db) > 0.1 or np.linalg.norm(mid_a - mid_b) > 0.5:
            any_diff = True
            break
    assert any_diff, "CP-driven run produced identical midribs to plain run"


if __name__ == "__main__":
    test_no_nan_after_native_cp_evolve()
    print("no NaN: OK")
    test_cp_midrib_differs_from_plain()
    print("CP midrib differs from plain: OK")
