#!/usr/bin/env python3
"""
Session 1: Verify Root + Shoot Co-Simulation

Confirms that the calibrated maize pipeline already grows roots alongside shoots
and that the full hydraulic solve works with real root water uptake.

Validation criteria (from COUPLING_STAGE2_AGROC_READINESS.md):
  - Root system has > 1000 segments at day 55
  - Shoot geometry matches Stage 1 within 5%
  - hm.solve() completes without errors
  - An_total within 20% of Stage 1 values

Stage 1 reference:
  - An_total (uniform PAR): 2830.49 mmol CO2/d
  - n_leaf_segments: 6202
  - n_leaf_organs: 11
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

# Add coupling package to path
COUPLING_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(COUPLING_DIR.parent.parent))

import plantbox as pb
from dart.coupling.config import DEFAULT_XML, get_hydraulics_json, get_photosynthesis_json
from dart.coupling.growth.grow import grow_plant, extract_g3_mesh, run_photosynthesis

# ---------------------------------------------------------------------------
# Stage 1 reference values
# ---------------------------------------------------------------------------
STAGE1_AN_UNIFORM_MMOL = 2830.49   # from coupled_results.json
STAGE1_N_LEAF_SEGS = 6202
STAGE1_N_LEAF_ORGANS = 11

# Validation thresholds
ROOT_SEG_THRESHOLD = 1000        # minimum root segments at day 55
SHOOT_TOLERANCE_PCT = 5.0        # shoot geometry match (%)
AN_TOLERANCE_PCT = 20.0          # An tolerance vs Stage 1 (%)

OUTPUT_DIR = COUPLING_DIR / "output" / "session1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def test_root_growth():
    """Task 1: Verify root growth in existing pipeline."""
    print("\n" + "=" * 60)
    print("TEST 1: Root Growth Verification")
    print("=" * 60)

    plant = grow_plant(
        xml_path=str(DEFAULT_XML),
        simulation_time=55,
        min_stem_nodes=100,
        min_leaf_nodes=40,
        enable_photosynthesis=True,
        seed=42,
    )

    organs = plant.getOrgans()
    n_roots = sum(1 for o in organs if o.organType() == pb.OrganTypes.root)
    n_leaves = sum(1 for o in organs if o.organType() == pb.OrganTypes.leaf)
    n_stems = sum(1 for o in organs if o.organType() == pb.OrganTypes.stem)

    # Count root segments
    root_organs = [o for o in organs if o.organType() == pb.OrganTypes.root]
    n_root_segs = sum(len(o.getNodes()) - 1 for o in root_organs)

    # Root depth
    root_nodes = []
    for o in root_organs:
        for n in o.getNodes():
            root_nodes.append(n.z)
    max_root_depth = abs(min(root_nodes)) if root_nodes else 0

    # Root length
    total_root_length = sum(o.getLength(False) for o in root_organs)

    # Stem height
    stem_organs = [o for o in organs if o.organType() == pb.OrganTypes.stem]
    stem_height = max(o.getLength(False) for o in stem_organs) if stem_organs else 0

    print(f"\n  Root organs:     {n_roots}")
    print(f"  Root segments:   {n_root_segs}")
    print(f"  Max root depth:  {max_root_depth:.1f} cm")
    print(f"  Total root len:  {total_root_length:.1f} cm")
    print(f"  Leaves:          {n_leaves}")
    print(f"  Stems:           {n_stems}")
    print(f"  Stem height:     {stem_height:.1f} cm")

    # Validation
    assert n_root_segs > ROOT_SEG_THRESHOLD, \
        f"FAIL: {n_root_segs} root segments < threshold {ROOT_SEG_THRESHOLD}"
    print(f"\n  PASS: {n_root_segs} root segments > {ROOT_SEG_THRESHOLD}")

    return plant, {
        "n_roots": n_roots,
        "n_root_segs": n_root_segs,
        "max_root_depth_cm": max_root_depth,
        "total_root_length_cm": total_root_length,
        "n_leaves": n_leaves,
        "n_stems": n_stems,
        "stem_height_cm": stem_height,
    }


def test_shoot_geometry(plant):
    """Task 3: Verify shoot geometry is unchanged."""
    print("\n" + "=" * 60)
    print("TEST 2: Shoot Geometry Verification")
    print("=" * 60)

    mesh, organ_dicts = extract_g3_mesh(
        plant, min_stem_nodes=100, min_leaf_nodes=40, stem_res=20
    )

    n_leaf_organs = sum(1 for o in organ_dicts if o['type'] == 'leaf')
    n_triangles = mesh.n_triangles

    # Count leaf segments from the plant (same way Stage 1 does)
    ot_arr = np.array(plant.organTypes)
    n_leaf_segs = int(np.sum(ot_arr == 4))

    # Leaf blade surface
    lbs = np.array(plant.leafBladeSurface)
    total_leaf_area_cm2 = np.sum(lbs)

    print(f"\n  Leaf organs:     {n_leaf_organs} (Stage 1: {STAGE1_N_LEAF_ORGANS})")
    print(f"  Leaf segments:   {n_leaf_segs} (Stage 1: {STAGE1_N_LEAF_SEGS})")
    print(f"  Mesh triangles:  {n_triangles}")
    print(f"  Leaf area:       {total_leaf_area_cm2:.1f} cm2 (one-sided)")

    # Validation
    leaf_seg_diff_pct = abs(n_leaf_segs - STAGE1_N_LEAF_SEGS) / STAGE1_N_LEAF_SEGS * 100
    print(f"\n  Leaf segment diff: {leaf_seg_diff_pct:.1f}% (threshold: {SHOOT_TOLERANCE_PCT}%)")

    assert n_leaf_organs == STAGE1_N_LEAF_ORGANS, \
        f"FAIL: {n_leaf_organs} leaf organs != expected {STAGE1_N_LEAF_ORGANS}"
    print(f"  PASS: leaf organ count = {n_leaf_organs}")

    assert leaf_seg_diff_pct < SHOOT_TOLERANCE_PCT, \
        f"FAIL: leaf segment diff {leaf_seg_diff_pct:.1f}% > {SHOOT_TOLERANCE_PCT}%"
    print(f"  PASS: leaf segment count within {SHOOT_TOLERANCE_PCT}%")

    return {
        "n_leaf_organs": n_leaf_organs,
        "n_leaf_segs": n_leaf_segs,
        "n_triangles": n_triangles,
        "total_leaf_area_cm2": float(total_leaf_area_cm2),
        "leaf_seg_diff_pct": leaf_seg_diff_pct,
    }


def test_hydraulics_photosynthesis(plant):
    """Task 2: Verify hydraulics with real root uptake."""
    print("\n" + "=" * 60)
    print("TEST 3: Hydraulics + Photosynthesis Solve")
    print("=" * 60)

    t0 = time.time()
    hm = run_photosynthesis(
        plant=plant,
        sim_time=55,
        output_prefix=str(OUTPUT_DIR / "session1_photosynthesis"),
        par_umol=1000.0,
        tair_c=25.0,
        rh=0.7,
        soil_psi_cm=-500.0,
    )
    solve_time = time.time() - t0

    if hm is None:
        print("  FAIL: hm.solve() returned None (error)")
        return None

    An_leaf = np.array(hm.get_net_assimilation())
    An_total_mmol = np.sum(An_leaf) * 1e3
    transp = np.sum(hm.get_transpiration()) / 18 * 1e3

    print(f"\n  An_total:       {An_total_mmol:.2f} mmol CO2/d")
    print(f"  Stage 1 ref:    {STAGE1_AN_UNIFORM_MMOL:.2f} mmol CO2/d")
    print(f"  Transpiration:  {transp:.2f} mmol H2O/d")
    print(f"  Solve time:     {solve_time:.1f} s")

    # Validation
    an_diff_pct = abs(An_total_mmol - STAGE1_AN_UNIFORM_MMOL) / STAGE1_AN_UNIFORM_MMOL * 100
    print(f"\n  An diff vs Stage 1: {an_diff_pct:.1f}% (threshold: {AN_TOLERANCE_PCT}%)")

    assert An_total_mmol > 0, "FAIL: An_total <= 0 (photosynthesis broken)"
    print(f"  PASS: An_total > 0")

    assert an_diff_pct < AN_TOLERANCE_PCT, \
        f"FAIL: An diff {an_diff_pct:.1f}% > {AN_TOLERANCE_PCT}%"
    print(f"  PASS: An within {AN_TOLERANCE_PCT}% of Stage 1")

    return {
        "An_total_mmol": An_total_mmol,
        "An_stage1_ref_mmol": STAGE1_AN_UNIFORM_MMOL,
        "An_diff_pct": an_diff_pct,
        "transpiration_mmol": transp,
        "solve_time_s": solve_time,
    }


def main():
    print("=" * 60)
    print("SESSION 1: Root + Shoot Co-Simulation Verification")
    print("=" * 60)
    print(f"  XML: {DEFAULT_XML}")
    print(f"  Output: {OUTPUT_DIR}")

    results = {"session": 1, "xml": str(DEFAULT_XML)}

    # Test 1: Root growth
    plant, root_results = test_root_growth()
    results["root_growth"] = root_results

    # Test 2: Shoot geometry
    shoot_results = test_shoot_geometry(plant)
    results["shoot_geometry"] = shoot_results

    # Test 3: Hydraulics + photosynthesis
    photo_results = test_hydraulics_photosynthesis(plant)
    results["photosynthesis"] = photo_results

    # Summary
    print("\n" + "=" * 60)
    print("SESSION 1 SUMMARY")
    print("=" * 60)
    all_pass = True
    tests = [
        ("Root segments > 1000", root_results["n_root_segs"] > ROOT_SEG_THRESHOLD),
        ("Leaf organs = 11", shoot_results["n_leaf_organs"] == STAGE1_N_LEAF_ORGANS),
        ("Leaf segs within 5%", shoot_results["leaf_seg_diff_pct"] < SHOOT_TOLERANCE_PCT),
        ("hm.solve() completed", photo_results is not None),
    ]
    if photo_results:
        tests.append(("An within 20% of Stage 1", photo_results["An_diff_pct"] < AN_TOLERANCE_PCT))

    for name, passed in tests:
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"  [{status}] {name}")

    # Save results
    results_path = OUTPUT_DIR / "session1_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved: {results_path}")

    if all_pass:
        print("\n  ALL TESTS PASSED")
    else:
        print("\n  SOME TESTS FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
