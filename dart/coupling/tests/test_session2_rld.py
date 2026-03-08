#!/usr/bin/env python3
"""
Session 2: RLD Profile Extraction — Validation Test

Extracts root length density profiles at days 20, 35, 55 and validates:
  - RLD values in realistic range for maize (0.1-5.0 cm/cm3 near surface)
  - Profile deepens with plant age
  - Total root length increases monotonically
  - rrd.in format is correct (relative depth 0-1, density normalised)

Reference: COUPLING_STAGE2_AGROC_READINESS.md, Session 2
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
from dart.coupling.config import DEFAULT_XML, OUTPUT_DIR
from dart.coupling.growth import (
    grow_plant,
    extract_rld_profile,
    export_rld_csv,
    export_rrd_in,
    plot_rld_profile,
    plot_rld_growth_trajectory,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
TEST_DAYS = [20, 35, 55]
N_LAYERS = 20
DEPTH_CM = 100.0
ROW_SPACING_CM = 75.0
PLANT_SPACING_CM = 20.0

# Validation thresholds
RLD_SURFACE_MIN = 0.01   # cm/cm3 — at least some roots near surface
RLD_SURFACE_MAX = 10.0   # cm/cm3 — not unrealistically dense
RLD_MAX_REALISTIC = 15.0  # cm/cm3 — upper bound for any layer

SESSION_DIR = OUTPUT_DIR / "session2"
SESSION_DIR.mkdir(parents=True, exist_ok=True)


def test_single_day(day, seed=42):
    """Grow plant to `day`, extract and validate RLD profile."""
    print(f"\n{'='*60}")
    print(f"RLD EXTRACTION — Day {day}")
    print(f"{'='*60}")

    plant = grow_plant(
        xml_path=str(DEFAULT_XML),
        simulation_time=day,
        enable_photosynthesis=True,
        seed=seed,
    )

    t0 = time.time()
    profile = extract_rld_profile(
        plant,
        n_layers=N_LAYERS,
        depth_cm=DEPTH_CM,
        row_spacing_cm=ROW_SPACING_CM,
        plant_spacing_cm=PLANT_SPACING_CM,
    )
    extract_time = time.time() - t0

    # Print summary
    rld = profile["RLD_cm_per_cm3"]
    print(f"\n  --- RLD Profile Summary ---")
    print(f"  Layers: {N_LAYERS}, Depth: {DEPTH_CM} cm")
    print(f"  Ground area: {ROW_SPACING_CM} x {PLANT_SPACING_CM} = "
          f"{profile['ground_area_cm2']:.0f} cm2")
    print(f"  Total root length: {profile['total_root_length_cm']:.1f} cm")
    print(f"  Max root depth: {profile['max_root_depth_cm']:.1f} cm")
    print(f"  Root segments: {profile['n_root_segments']}")
    print(f"  Surface RLD (layer 0): {rld[0]:.4f} cm/cm3")
    print(f"  Max RLD: {np.max(rld):.4f} cm/cm3 (layer {np.argmax(rld)})")
    print(f"  Extraction time: {extract_time:.2f} s")

    # Print per-layer table
    print(f"\n  {'Layer':>5} {'Depth':>10} {'RootLen':>10} {'RLD':>10}")
    for i in range(N_LAYERS):
        d = profile["depth_mid_cm"][i]
        rl = profile["root_length_cm"][i]
        r = rld[i]
        if rl > 0.01:  # only print non-empty layers
            print(f"  {i:>5} {d:>8.1f} cm {rl:>8.2f} cm {r:>10.4f}")

    # Export
    csv_path = export_rld_csv(profile, SESSION_DIR / f"maize_day{day}_rld_profile.csv")
    rrd_path = export_rrd_in(profile, SESSION_DIR / f"maize_day{day}_rrd.in")
    plot_rld_profile(profile, SESSION_DIR / f"maize_day{day}_rld_profile.png", day=day)

    return profile


def validate_rrd_format(day):
    """Validate rrd.in file format."""
    rrd_path = SESSION_DIR / f"maize_day{day}_rrd.in"
    if not rrd_path.exists():
        return False, "rrd.in file not found"

    lines = rrd_path.read_text().strip().split("\n")
    n_rows = int(lines[0])

    if n_rows != len(lines) - 1:
        return False, f"Row count mismatch: header says {n_rows}, got {len(lines)-1} data lines"

    depths = []
    densities = []
    for line in lines[1:]:
        parts = line.strip().split()
        if len(parts) != 2:
            return False, f"Expected 2 columns, got {len(parts)}: '{line}'"
        d, r = float(parts[0]), float(parts[1])
        depths.append(d)
        densities.append(r)

    # Depths should be in [0, 1]
    if min(depths) < -0.001 or max(depths) > 1.001:
        return False, f"Depths out of range [0,1]: min={min(depths):.4f}, max={max(depths):.4f}"

    # Densities should be non-negative and sum ~1
    if min(densities) < -0.001:
        return False, f"Negative density: {min(densities):.4f}"

    density_sum = sum(densities)
    if abs(density_sum - 1.0) > 0.01:
        return False, f"Density sum = {density_sum:.4f} (expected ~1.0)"

    return True, f"OK: {n_rows} rows, depth [{min(depths):.3f}, {max(depths):.3f}], density sum={density_sum:.4f}"


def main():
    print("=" * 60)
    print("SESSION 2: RLD Profile Extraction — Validation")
    print("=" * 60)
    print(f"  XML: {DEFAULT_XML}")
    print(f"  Days: {TEST_DAYS}")
    print(f"  Output: {SESSION_DIR}")

    results = {
        "session": 2,
        "xml": str(DEFAULT_XML),
        "config": {
            "n_layers": N_LAYERS,
            "depth_cm": DEPTH_CM,
            "row_spacing_cm": ROW_SPACING_CM,
            "plant_spacing_cm": PLANT_SPACING_CM,
        },
        "profiles": {},
    }

    # --- Extract profiles for each day ---
    profiles = {}
    for day in TEST_DAYS:
        profile = test_single_day(day)
        profiles[day] = profile
        results["profiles"][day] = {
            "total_root_length_cm": profile["total_root_length_cm"],
            "max_root_depth_cm": profile["max_root_depth_cm"],
            "n_root_segments": profile["n_root_segments"],
            "surface_RLD": float(profile["RLD_cm_per_cm3"][0]),
            "max_RLD": float(np.max(profile["RLD_cm_per_cm3"])),
            "max_RLD_layer": int(np.argmax(profile["RLD_cm_per_cm3"])),
        }

    # --- Growth trajectory plot ---
    plot_rld_growth_trajectory(profiles, SESSION_DIR / "rld_growth_trajectory.png")

    # --- Validation ---
    print(f"\n{'='*60}")
    print("VALIDATION")
    print(f"{'='*60}")

    tests = []
    all_pass = True

    # Test 1: RLD values in realistic range at day 55
    rld_55 = profiles[55]["RLD_cm_per_cm3"]
    max_rld = float(np.max(rld_55))
    surface_rld = float(rld_55[0])

    t1_pass = RLD_SURFACE_MIN <= surface_rld <= RLD_SURFACE_MAX
    tests.append(("Surface RLD in range [0.01, 10.0] at day 55", t1_pass,
                   f"surface_RLD={surface_rld:.4f}"))
    if not t1_pass:
        all_pass = False

    t1b_pass = max_rld <= RLD_MAX_REALISTIC
    tests.append(("Max RLD < 15.0 cm/cm3", t1b_pass, f"max_RLD={max_rld:.4f}"))
    if not t1b_pass:
        all_pass = False

    # Test 2: Profile deepens with plant age
    max_depths = [profiles[d]["max_root_depth_cm"] for d in TEST_DAYS]
    t2_pass = all(max_depths[i] <= max_depths[i+1] for i in range(len(max_depths)-1))
    tests.append(("Root depth increases with age", t2_pass,
                   f"depths={[f'{d:.1f}' for d in max_depths]}"))
    if not t2_pass:
        all_pass = False

    # Test 3: Total root length increases monotonically
    total_lengths = [profiles[d]["total_root_length_cm"] for d in TEST_DAYS]
    t3_pass = all(total_lengths[i] < total_lengths[i+1] for i in range(len(total_lengths)-1))
    tests.append(("Total root length increases monotonically", t3_pass,
                   f"lengths={[f'{l:.0f}' for l in total_lengths]}"))
    if not t3_pass:
        all_pass = False

    # Test 4: rrd.in format valid for each day
    for day in TEST_DAYS:
        valid, msg = validate_rrd_format(day)
        tests.append((f"rrd.in format valid (day {day})", valid, msg))
        if not valid:
            all_pass = False

    # Test 5: RLD profile has expected shape (highest near surface, decays with depth)
    rld_55_top5 = np.mean(rld_55[:5])  # top 25 cm
    rld_55_bot5 = np.mean(rld_55[-5:])  # bottom 25 cm
    t5_pass = rld_55_top5 > rld_55_bot5
    tests.append(("RLD higher near surface than at depth (day 55)", t5_pass,
                   f"top25cm_mean={rld_55_top5:.4f}, bot25cm_mean={rld_55_bot5:.4f}"))
    if not t5_pass:
        all_pass = False

    # Print results
    for name, passed, detail in tests:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")
        print(f"         {detail}")

    results["tests"] = [
        {"name": name, "passed": passed, "detail": detail}
        for name, passed, detail in tests
    ]
    results["all_pass"] = all_pass

    # Save results
    results_path = SESSION_DIR / "session2_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved: {results_path}")

    if all_pass:
        print(f"\n  ALL {len(tests)} TESTS PASSED")
    else:
        n_fail = sum(1 for _, p, _ in tests if not p)
        print(f"\n  {n_fail} / {len(tests)} TESTS FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
