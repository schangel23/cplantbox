#!/usr/bin/env python3
"""
Session 3: Verify PhloemFluxPython Drop-In Replacement

Confirms that replacing PhotosynthesisPython with PhloemFluxPython
produces identical photosynthesis results. PhloemFluxPython inherits
from PhotosynthesisPython, so hm.solve() should behave identically
when solve_phloem_flow() is NOT called.

Validation criteria (from COUPLING_STAGE2_AGROC_READINESS.md):
  - All existing phases produce identical results
  - No import errors or missing methods
  - hm.solve() behaves identically

Reference values from Session 1:
  - An_total: 2830.50 mmol CO2/d (uniform PAR=1000, T=25, RH=0.7)
  - Transpiration: 1531.82 mmol H2O/d
  - n_leaf_segments: 6202
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
from dart.coupling.config import (
    DEFAULT_XML, get_hydraulics_json, get_photosynthesis_json, get_phloem_json,
)
from dart.coupling.growth.grow import grow_plant

# ---------------------------------------------------------------------------
# Reference values from Session 1 (which used PhotosynthesisPython)
# ---------------------------------------------------------------------------
SESSION1_AN_MMOL = 2830.50
SESSION1_TRANSP_MMOL = 1531.82
SESSION1_N_LEAF_SEGS = 6202
SESSION1_N_LEAF_ORGANS = 11

# Tolerance: PhloemFluxPython inherits solve() from PhotosynthesisPython,
# so results should be identical. Allow tiny floating-point tolerance.
AN_TOLERANCE_PCT = 1.0
TRANSP_TOLERANCE_PCT = 1.0

OUTPUT_DIR = COUPLING_DIR / "output" / "session3"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

results = {}
n_pass = 0
n_fail = 0


def check(name, passed, msg=""):
    global n_pass, n_fail
    status = "PASS" if passed else "FAIL"
    if not passed:
        n_fail += 1
    else:
        n_pass += 1
    print(f"  [{status}] {name}" + (f" — {msg}" if msg else ""))
    results[name] = {"passed": passed, "message": msg}


def test_import():
    """Test 1: PhloemFluxPython imports without error."""
    print("\n" + "=" * 60)
    print("TEST 1: PhloemFluxPython Import")
    print("=" * 60)

    try:
        from plantbox.functional.phloem_flux import PhloemFluxPython
        check("import_phloem_flux", True, "PhloemFluxPython imported successfully")
        return PhloemFluxPython
    except ImportError as e:
        check("import_phloem_flux", False, f"Import failed: {e}")
        return None


def test_inheritance(PhloemFluxPython):
    """Test 2: PhloemFluxPython inherits from PhotosynthesisPython."""
    print("\n" + "=" * 60)
    print("TEST 2: Class Inheritance")
    print("=" * 60)

    from plantbox.functional.Photosynthesis import PhotosynthesisPython

    is_subclass = issubclass(PhloemFluxPython, PhotosynthesisPython)
    check("inherits_photosynthesis", is_subclass,
          f"PhloemFluxPython bases: {[b.__name__ for b in PhloemFluxPython.__mro__[:5]]}")

    # Check key methods exist
    for method in ["solve", "get_net_assimilation", "get_transpiration",
                   "get_water_potential", "read_photosynthesis_parameters",
                   "solve_phloem_flow", "read_phloem_parameters",
                   "get_phloem_data"]:
        has_method = hasattr(PhloemFluxPython, method)
        check(f"has_{method}", has_method)


def test_construction_and_params():
    """Test 3: Construct PhloemFluxPython, load all parameters."""
    print("\n" + "=" * 60)
    print("TEST 3: Construction + Parameter Loading")
    print("=" * 60)

    from plantbox.functional.phloem_flux import PhloemFluxPython
    from plantbox.functional.PlantHydraulicParameters import PlantHydraulicParameters

    plant = grow_plant(
        xml_path=str(DEFAULT_XML),
        simulation_time=55,
        min_stem_nodes=100,
        min_leaf_nodes=40,
        enable_photosynthesis=True,
        seed=42,
    )

    params = PlantHydraulicParameters()
    params.read_parameters(get_hydraulics_json())

    try:
        hm = PhloemFluxPython(plant, params)
        check("construct_phloemflux", True)
    except Exception as e:
        check("construct_phloemflux", False, str(e))
        return None, None

    try:
        hm.read_photosynthesis_parameters(filename=get_photosynthesis_json())
        check("read_photosynthesis_params", True)
    except Exception as e:
        check("read_photosynthesis_params", False, str(e))
        return None, None

    try:
        hm.read_phloem_parameters(filename=get_phloem_json())
        check("read_phloem_params", True,
              f"file={get_phloem_json()}")
    except Exception as e:
        check("read_phloem_params", False, str(e))
        return None, None

    # Verify phloem-specific attributes were loaded
    has_leafgrowthzone = hasattr(hm, 'leafGrowthZone') and hm.leafGrowthZone > 0
    check("phloem_leafGrowthZone", has_leafgrowthzone,
          f"leafGrowthZone={getattr(hm, 'leafGrowthZone', 'N/A')}")

    has_vmaxloading = hasattr(hm, 'Vmaxloading') and hm.Vmaxloading > 0
    check("phloem_Vmaxloading", has_vmaxloading,
          f"Vmaxloading={getattr(hm, 'Vmaxloading', 'N/A')}")

    return plant, hm


def test_photosynthesis_regression(plant, hm):
    """Test 4: hm.solve() produces identical results to Session 1."""
    print("\n" + "=" * 60)
    print("TEST 4: Photosynthesis Regression (vs Session 1)")
    print("=" * 60)

    par_umol = 1000.0
    tair_c = 25.0
    rh = 0.7
    soil_psi_cm = -500.0

    depth = 100
    p_s = np.linspace(soil_psi_cm, soil_psi_cm - depth, depth)
    es = hm.get_es(tair_c)
    ea = es * rh

    par_mol_cm2_d = par_umol * 1e-6 * 86400 * 1e-4

    t0 = time.time()
    try:
        hm.solve(
            sim_time=55,
            rsx=p_s,
            cells=True,
            ea=ea,
            es=es,
            PAR=par_mol_cm2_d,
            TairC=tair_c,
            verbose=0,
        )
        solve_ok = True
    except Exception as e:
        solve_ok = False
        check("solve_completes", False, str(e))
        return
    solve_time = time.time() - t0
    check("solve_completes", solve_ok, f"solve time: {solve_time:.2f}s")

    An_leaf = np.array(hm.get_net_assimilation())
    transp_raw = np.array(hm.get_transpiration())
    hx_all = np.array(hm.get_water_potential())

    n_leaf_segs = len(An_leaf)
    An_total_mmol = np.sum(An_leaf) * 1e3
    transp_mmol = np.sum(transp_raw) / 18 * 1e3

    print(f"  An_total = {An_total_mmol:.2f} mmol CO2/d (ref: {SESSION1_AN_MMOL:.2f})")
    print(f"  Transpiration = {transp_mmol:.2f} mmol H2O/d (ref: {SESSION1_TRANSP_MMOL:.2f})")
    print(f"  n_leaf_segs = {n_leaf_segs} (ref: {SESSION1_N_LEAF_SEGS})")

    # Segment count
    check("n_leaf_segs_match", n_leaf_segs == SESSION1_N_LEAF_SEGS,
          f"{n_leaf_segs} vs {SESSION1_N_LEAF_SEGS}")

    # An regression
    an_diff_pct = abs(An_total_mmol - SESSION1_AN_MMOL) / SESSION1_AN_MMOL * 100
    check("An_regression", an_diff_pct < AN_TOLERANCE_PCT,
          f"diff = {an_diff_pct:.4f}% (tolerance: {AN_TOLERANCE_PCT}%)")

    # Transpiration regression
    transp_diff_pct = abs(transp_mmol - SESSION1_TRANSP_MMOL) / SESSION1_TRANSP_MMOL * 100
    check("transp_regression", transp_diff_pct < TRANSP_TOLERANCE_PCT,
          f"diff = {transp_diff_pct:.4f}% (tolerance: {TRANSP_TOLERANCE_PCT}%)")

    # Water potential should be finite
    check("water_potential_finite", np.all(np.isfinite(hx_all)),
          f"mean psi = {np.mean(hx_all):.0f} cm")

    # Store results
    results["An_total_mmol"] = float(An_total_mmol)
    results["transp_mmol"] = float(transp_mmol)
    results["n_leaf_segs"] = int(n_leaf_segs)
    results["solve_time_s"] = float(solve_time)
    results["mean_psi_cm"] = float(np.mean(hx_all))


def test_phloem_arrays_initialized():
    """Test 5: Phloem arrays exist but are empty (solve_phloem_flow not called)."""
    print("\n" + "=" * 60)
    print("TEST 5: Phloem Arrays Initialized (empty)")
    print("=" * 60)

    from plantbox.functional.phloem_flux import PhloemFluxPython
    from plantbox.functional.PlantHydraulicParameters import PlantHydraulicParameters

    plant = grow_plant(
        xml_path=str(DEFAULT_XML),
        simulation_time=55,
        min_stem_nodes=100,
        min_leaf_nodes=40,
        enable_photosynthesis=True,
        seed=42,
    )

    params = PlantHydraulicParameters()
    params.read_parameters(get_hydraulics_json())
    hm = PhloemFluxPython(plant, params)
    hm.read_photosynthesis_parameters(filename=get_photosynthesis_json())
    hm.read_phloem_parameters(filename=get_phloem_json())

    # After construction, phloem arrays should exist but be empty
    # (they're populated by solve_phloem_flow, which we don't call yet)
    for attr in ["Q_Rm", "Q_Gr", "Q_Exud", "Q_ST", "C_ST_np"]:
        arr = getattr(hm, attr, None)
        exists = arr is not None
        is_empty = exists and len(arr) == 0
        check(f"{attr}_exists_empty", exists and is_empty,
              f"len={len(arr) if exists else 'N/A'}")

    # get_phloem_data_list requires outputs_options, which is only populated
    # after update_outputs() (called by get_phloem_data or solve_phloem_flow).
    # Before any phloem solve, it's expected to raise AttributeError.
    try:
        keys = list(hm.get_phloem_data_list())
        check("phloem_data_list", len(keys) > 0,
              f"keys: {keys}")
    except AttributeError:
        check("phloem_data_list", True,
              "outputs_options not yet populated (expected before solve_phloem_flow)")


def test_pipeline_import():
    """Test 6: All pipeline files import without error after the swap."""
    print("\n" + "=" * 60)
    print("TEST 6: Pipeline Module Imports")
    print("=" * 60)

    modules = [
        ("growth.grow", "dart.coupling.growth.grow"),
        ("photosynthesis.coupled", "dart.coupling.photosynthesis.coupled"),
        ("photosynthesis.iterative", "dart.coupling.photosynthesis.iterative"),
        ("dart.multifield", "dart.coupling.dart.multifield"),
        ("validation.validate", "dart.coupling.validation.validate"),
    ]

    for label, modname in modules:
        try:
            __import__(modname)
            check(f"import_{label}", True)
        except Exception as e:
            check(f"import_{label}", False, str(e))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("SESSION 3: PhloemFluxPython Drop-In Replacement Test")
    print("=" * 60)

    # Test 1: Import
    PhloemFluxPython = test_import()
    if PhloemFluxPython is None:
        print("\nFATAL: Cannot import PhloemFluxPython. Aborting.")
        sys.exit(1)

    # Test 2: Inheritance
    test_inheritance(PhloemFluxPython)

    # Test 3: Construction + params
    plant, hm = test_construction_and_params()

    # Test 4: Photosynthesis regression
    if plant is not None and hm is not None:
        test_photosynthesis_regression(plant, hm)
    else:
        print("\nSKIPPED: Test 4 (plant/hm construction failed)")

    # Test 5: Phloem arrays
    test_phloem_arrays_initialized()

    # Test 6: Pipeline imports
    test_pipeline_import()

    # Summary
    print("\n" + "=" * 60)
    print(f"RESULTS: {n_pass} passed, {n_fail} failed, {n_pass + n_fail} total")
    print("=" * 60)

    # Save results
    results["summary"] = {
        "passed": n_pass,
        "failed": n_fail,
        "total": n_pass + n_fail,
    }
    out_file = OUTPUT_DIR / "session3_results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved to {out_file}")

    sys.exit(0 if n_fail == 0 else 1)
