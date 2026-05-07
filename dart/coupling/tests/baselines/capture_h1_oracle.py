#!/usr/bin/env python3
"""S0 — capture the pre-change H=1.0 oracle for the cultivar-height plan.

Plan: PLAN_CULTIVAR_HEIGHT_FACTOR_2026-05-07.md §S0.

Grows the production maize_calibrated.xml at day 130 across 5 seeds with the
current CPlantBox HEAD (no `cultivar_height_factor` field present yet), and
serialises the per-seed geometry needed by the S5 G2 bit-identical regression
to `dart/coupling/tests/fixtures/oracle_h1_well_watered_day130.json`.

Captured per seed:
  - z_max:           max node z across all organs [cm]
  - topmost_leaf_z:  max z of any node belonging to a Leaf [cm]
  - n_leaves:        number of leaves attached to mainstem (subType==1)
  - mainstem_length: mainstem.getLength(True) [cm]
  - phytomer_lengths: latched length_per_n from MultiPhaseStemGrowth FA state
  - leaf_insertion_zs: z of the leaf base node for every mainstem leaf

The G2 test reproduces this fixture rank-by-rank within 1e-9 cm under the
default H=1.0 fallback added in S1+S2.

Usage (from CPlantBox repo root):
    source cpbenv/bin/activate
    python3 dart/coupling/tests/baselines/capture_h1_oracle.py
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
COUPLING_DIR = REPO_ROOT / "dart" / "coupling"
FIXTURES_DIR = COUPLING_DIR / "tests" / "fixtures"
sys.path.insert(0, str(REPO_ROOT))

from dart.coupling.growth.grow import grow_plant  # noqa: E402

XML_PATH = COUPLING_DIR / "data" / "maize_calibrated.xml"
SEEDS = [1, 2, 3, 4, 5]
SIM_DAYS = 130


def _git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def _xml_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mainstem(plant):
    """Return the mainstem (stem with subType==1)."""
    for o in plant.getOrgans(-1, True):
        try:
            if int(o.organType()) == 3 and int(o.getParameter("subType")) == 1:
                return o
        except Exception:
            continue
    raise RuntimeError("Mainstem (subType=1) not found")


def _capture_one(seed: int) -> dict:
    plant = grow_plant(
        xml_path=str(XML_PATH),
        simulation_time=SIM_DAYS,
        seed=seed,
        enable_photosynthesis=True,
    )

    # Walk all nodes for z_max
    nodes = plant.getNodes()
    z_max = max(float(n.z) for n in nodes)

    # Mainstem geometry
    ms = _mainstem(plant)
    mainstem_length = float(ms.getLength(True))

    # FA per-phytomer state
    fa = ms.getFaState()
    if fa is None:
        raise RuntimeError(f"seed={seed}: mainstem has no FA state")
    # length_per_n is 1-based with index 0 as a sentinel; emit ranks 1..N
    length_per_n = list(fa.length_per_n)
    # Drop trailing zeros below the rank cap so the fixture stays compact
    # but keep the canonical 1-based slice for indexing simplicity.
    phytomer_lengths = [float(x) for x in length_per_n]

    # Leaf-side stats: walk children of mainstem only (mainstem subType==1)
    children = []
    for ci in range(ms.getNumberOfChildren()):
        children.append(ms.getChild(ci))
    leaf_children = [c for c in children if int(c.organType()) == 4]
    n_leaves = len(leaf_children)

    leaf_insertion_zs: list[float] = []
    topmost_leaf_z = -1e30
    for c in leaf_children:
        ln = c.getNodes()
        if not ln:
            continue
        # First node of a leaf is its insertion (collar) node
        leaf_insertion_zs.append(float(ln[0].z))
        for n in ln:
            if float(n.z) > topmost_leaf_z:
                topmost_leaf_z = float(n.z)

    if topmost_leaf_z < -1e29:
        topmost_leaf_z = float("nan")

    return {
        "z_max": z_max,
        "topmost_leaf_z": topmost_leaf_z,
        "n_leaves": n_leaves,
        "mainstem_length": mainstem_length,
        "phytomer_lengths": phytomer_lengths,
        "leaf_insertion_zs": leaf_insertion_zs,
    }


def main() -> int:
    head = _git_head()
    xml_sha = _xml_sha256(XML_PATH)
    per_seed: dict[str, dict] = {}
    z_maxes: list[float] = []
    for seed in SEEDS:
        print(f"[capture_h1_oracle] seed={seed} growing day {SIM_DAYS} ...")
        rec = _capture_one(seed)
        per_seed[str(seed)] = rec
        z_maxes.append(rec["z_max"])
        print(
            f"  z_max={rec['z_max']:.4f} cm  topmost_leaf_z={rec['topmost_leaf_z']:.4f} cm  "
            f"n_leaves={rec['n_leaves']}  mainstem_length={rec['mainstem_length']:.4f} cm"
        )

    n = len(z_maxes)
    mean = sum(z_maxes) / n
    var = sum((z - mean) ** 2 for z in z_maxes) / max(n - 1, 1)
    sd = var**0.5

    out = {
        "meta": {
            "slug": "oracle_h1_well_watered_day130",
            "head_sha": head,
            "xml": "dart/coupling/data/maize_calibrated.xml",
            "xml_sha256": xml_sha,
            "seeds": SEEDS,
            "sim_days": SIM_DAYS,
            "plan": "PLAN_CULTIVAR_HEIGHT_FACTOR_2026-05-07.md §S0",
        },
        "stats": {
            "z_max_mean": mean,
            "z_max_sd": sd,
        },
        "per_seed": per_seed,
    }
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIXTURES_DIR / "oracle_h1_well_watered_day130.json"
    out_path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(f"\n[capture_h1_oracle] wrote {out_path}")
    print(f"[capture_h1_oracle] z_max mean={mean:.4f} cm  sd={sd:.4f} cm  (n={n})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
