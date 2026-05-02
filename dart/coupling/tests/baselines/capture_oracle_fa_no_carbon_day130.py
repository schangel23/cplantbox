#!/usr/bin/env python3
"""S5 oracle capture: FA-on no-carbon maize_calibrated, 130 days.

Captures the per-organ FA target geometry that the upcoming `--with-carbon`
runs (post-Lock #6 + Lock #9) must reproduce within tolerance under
well-watered Juelich met. This is the regression target for
PLAN_S5_SINK_SOURCE_COUPLING_2026-05-02 §G3 / §S6 test 2.

The oracle must be captured **before** any S2..S5 C++ change so it is
truly bit-identical to today's known-good FA-only geometric path
(production grow_plant → MultiPhase{Stem,Leaf}Growth dispatch, no
CWLimitedGrowth wrapping).

Why not reuse `d0_maize_calibrated_faon_s3b_130d.json`?
    That file stores aggregate scalars only (mainstem_top_z, n_organs,
    etc.). The S5 parity test needs per-organ realised lengths + per-rank
    stem latched lengths to localise drift to specific phytomers.

Output
------
* JSON  → tests/fixtures/oracle_fa_no_carbon_day130.json
* OBJ/MTL → tests/fixtures/oracle_fa_no_carbon_day130.obj (+ .mtl)

JSON schema (per plan §S1 step 2)
---------------------------------
{
  "meta": {...},
  "organs": {
    "<organ_id>": {
      "organ_type": int,            # 2=root, 3=stem, 4=leaf
      "subType":    int,
      "sp_lmax":    float,          # cm
      "realised_length": float,     # cm
      "insertion_z":   float,       # cm  (parent attachment z)
      "per_rank_lengths_for_stems": [float, ...]  # only on stems; [] otherwise
    },
    ...
  }
}

Usage (from /home/lukas/PHD/CPlantBox)
--------------------------------------
    cpbenv/bin/python3 dart/coupling/tests/baselines/capture_oracle_fa_no_carbon_day130.py
    cpbenv/bin/python3 dart/coupling/tests/baselines/capture_oracle_fa_no_carbon_day130.py --verify
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path

BASELINE_DIR = Path(__file__).resolve().parent
COUPLING_DIR = BASELINE_DIR.parent.parent
CPLANTBOX_ROOT = COUPLING_DIR.parent.parent
FIXTURES_DIR = COUPLING_DIR / "tests" / "fixtures"
sys.path.insert(0, str(CPLANTBOX_ROOT))

import plantbox as pb  # noqa: E402

from dart.coupling.growth.grow import grow_plant  # noqa: E402
from dart.coupling.geometry import (  # noqa: E402
    extract_organs_for_lofter,
    loft_organs,
)


XML_PATH = COUPLING_DIR / "data" / "maize_calibrated.xml"
OUT_JSON = FIXTURES_DIR / "oracle_fa_no_carbon_day130.json"
OUT_OBJ = FIXTURES_DIR / "oracle_fa_no_carbon_day130.obj"

SEED = 7
SIM_DAYS = 130


# ---------------------------------------------------------------------------
# Capture
# ---------------------------------------------------------------------------
def _insertion_z(o) -> float:
    """Z of the first node of organ o (i.e., its attachment point)."""
    nodes = list(o.getNodes())
    if not nodes:
        return 0.0
    return float(nodes[0].z)


def _per_rank_lengths(stem) -> list[float]:
    """Return per-rank latched length array for an FA-on stem.

    Pulled directly from MultiPhaseStemGrowth.per_organ_state[stem.id].
    Returns [] for non-FA stems (no per_organ_state entry) so the JSON
    schema stays uniform.
    """
    rp = stem.getOrganRandomParameter()
    f_gf = getattr(rp, "f_gf", None)
    state_map = getattr(f_gf, "per_organ_state", None)
    if state_map is None:
        return []
    sid = stem.getId()
    if sid not in state_map:
        return []
    raw = state_map[sid].length_per_n
    return [round(float(x), 6) for x in raw]


def capture_per_organ_snapshot(plant) -> dict:
    organs = plant.getOrgans()
    out: dict[str, dict] = {}
    for o in sorted(organs, key=lambda x: x.getId()):
        ot = int(o.organType())
        st = int(o.getParameter("subType"))
        sp_lmax = float(o.getParameter("lmax"))
        realised = float(o.getLength())
        insz = _insertion_z(o)
        per_rank = _per_rank_lengths(o) if ot == int(pb.OrganTypes.stem) else []
        out[str(o.getId())] = {
            "organ_type": ot,
            "subType": st,
            "sp_lmax": round(sp_lmax, 6),
            "realised_length": round(realised, 6),
            "insertion_z": round(insz, 6),
            "per_rank_lengths_for_stems": per_rank,
        }
    return out


def capture_aggregates(organs_map: dict) -> dict:
    """Compact aggregate signature for at-a-glance regression diffs."""
    by_type = {2: 0, 3: 0, 4: 0, 20: 0}  # root, stem, leaf, tassel-spike
    for v in organs_map.values():
        ot = v["organ_type"]
        st = v["subType"]
        if ot == 3 and st == 20:
            by_type[20] += 1
        else:
            by_type[ot] = by_type.get(ot, 0) + 1
    mainstem = next(
        (v for v in organs_map.values()
         if v["organ_type"] == 3 and v["subType"] == 1),
        None,
    )
    topmost_leaf_insz = max(
        (v["insertion_z"] for v in organs_map.values()
         if v["organ_type"] == 4),
        default=0.0,
    )

    # Geometry hash — sorted per-organ dump, identical across runs that
    # produce identical organs.
    h = hashlib.sha256()
    for oid in sorted(organs_map.keys(), key=int):
        v = organs_map[oid]
        h.update(struct.pack("<iidddd", v["organ_type"], v["subType"],
                             v["sp_lmax"], v["realised_length"],
                             v["insertion_z"], 0.0))
        for x in v["per_rank_lengths_for_stems"]:
            h.update(struct.pack("<d", x))
    return {
        "n_organs": len(organs_map),
        "n_roots": by_type[2],
        "n_stems": by_type[3],
        "n_leaves": by_type[4],
        "n_tassel_spikes": by_type[20],
        "mainstem_realised_length_cm": (
            round(mainstem["realised_length"], 4) if mainstem else 0.0
        ),
        "topmost_leaf_insertion_z_cm": round(topmost_leaf_insz, 4),
        "sha256": h.hexdigest(),
    }


def export_oracle_obj(plant, obj_path: Path) -> None:
    """Run the production lofter so the fixture mirrors what the with-carbon
    test will produce. Maize species so midrib/gutter defaults activate.
    """
    organs = extract_organs_for_lofter(plant, species="maize")
    mesh = loft_organs(organs, subdivide=False)
    obj_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.to_obj(str(obj_path), write_materials=True)


def grow_oracle():
    print(f"S5 oracle capture: {XML_PATH.name}, seed={SEED}, days={SIM_DAYS}")
    print(f"  Mode: FA-on, no-carbon (production grow_plant path, "
          f"no enable_cw_limited_growth)")
    plant = grow_plant(
        xml_path=str(XML_PATH),
        simulation_time=SIM_DAYS,
        seed=SEED,
        enable_photosynthesis=True,   # match diurnal carbon-mode bootstrap
        # daily_met=None → auto-loads juelich_2024_daily_met.csv
    )
    return plant


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------
def capture_mode():
    plant = grow_oracle()
    organs_map = capture_per_organ_snapshot(plant)
    agg = capture_aggregates(organs_map)
    print(f"  organs={agg['n_organs']} (roots={agg['n_roots']}, "
          f"stems={agg['n_stems']}, leaves={agg['n_leaves']}, "
          f"tassel_spikes={agg['n_tassel_spikes']})")
    print(f"  mainstem realised length: {agg['mainstem_realised_length_cm']:.2f} cm")
    print(f"  topmost leaf insertion z: {agg['topmost_leaf_insertion_z_cm']:.2f} cm")
    print(f"  geometry sha256: {agg['sha256']}")

    payload = {
        "meta": {
            "slug": "oracle_fa_no_carbon_day130",
            "xml": str(XML_PATH.relative_to(CPLANTBOX_ROOT)),
            "seed": SEED,
            "sim_days": SIM_DAYS,
            "fa_enabled": True,
            "carbon_feedback": False,
            **agg,
        },
        "organs": organs_map,
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with OUT_JSON.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"  -> {OUT_JSON.relative_to(CPLANTBOX_ROOT)}")

    print("  Exporting OBJ + MTL...")
    export_oracle_obj(plant, OUT_OBJ)
    print(f"  -> {OUT_OBJ.relative_to(CPLANTBOX_ROOT)}")


def verify_mode() -> int:
    if not OUT_JSON.exists():
        print(f"MISSING oracle at {OUT_JSON}")
        return 1
    with OUT_JSON.open() as f:
        baseline = json.load(f)
    plant = grow_oracle()
    organs_map = capture_per_organ_snapshot(plant)
    agg = capture_aggregates(organs_map)
    if agg["sha256"] != baseline["meta"]["sha256"]:
        print(f"  DIFF: expected {baseline['meta']['sha256']}")
        print(f"        got      {agg['sha256']}")
        return 1
    print(f"  OK (matches oracle); n_organs={agg['n_organs']}")
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true",
                        help="re-run and compare against stored oracle")
    args = parser.parse_args()
    if args.verify:
        sys.exit(verify_mode())
    else:
        capture_mode()


if __name__ == "__main__":
    main()
