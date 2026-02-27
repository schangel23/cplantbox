#!/usr/bin/env python3
"""Prepare a day-55 multifield DART simulation for GUI inspection (no DART run).

9 unique plants (seeds 42-50), each with its own OBJ model.

Usage:
  cd /home/lukas/PHD/CPlantBox
  source cpbenv/bin/activate
  python -m dart.coupling.prepare_gui_simulation

Opens in DART GUI: user_data/simulations/cpb_multifield_day55_par
"""
from .dart.multifield import (
    step1_grow_plants,
    step2_export_meshes,
    step3_create_dart_simulation,
)

if __name__ == "__main__":
    print("=" * 70)
    print("Preparing MULTIFIELD DART simulation for GUI inspection (day 55)")
    print("  9 unique plants (seeds 42-50)")
    print("=" * 70)

    plants = step1_grow_plants()
    meshes, mappings, obj_paths, dart_obj_paths, mapping_json_paths = \
        step2_export_meshes(plants)
    simu = step3_create_dart_simulation(dart_obj_paths)

    print("\n" + "=" * 70)
    print("DONE — Multifield simulation ready for DART GUI")
    print("=" * 70)
    print(f"  Open in DART GUI: {simu.simu_dir}")
    print(f"  Simulation name:  cpb_multifield_day55_par")
    print()
    print("  Check in GUI:")
    print("    - Phase tab → irradiance mode, 6 PAR bands")
    print("    - Object_3d → 9 models (p0..p8), each with unique OBJ")
    print("    - Object_3d → field file: multifield_plant_field.txt")
    print("    - Coeff_diff → 11 PROSPECT leaf OPs + stem + ground")
