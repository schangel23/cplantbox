"""Production-path tassel verification.

Runs the real production pipeline (``extract_organs_for_lofter`` +
``loft_organs`` from ``dart.coupling.geometry``) end-to-end on a
VT-capped, day-88 plant. Writes a full-plant OBJ to
``blender_preview/stages/``.

Difference from production ``grow_plant``: skips ``apply_donor_cps`` to
dodge a known heap-corruption interaction with the uncommitted
``maize_calibrated.xml`` (LRP ``surface_cps`` setters double-modifying
baked XML CPs). Tracked separately.

The VT cap is now XML-native (``use_thermal_cessation=1, tt_cessation=1000``
on mainstem subType 1) and fires via real met forcing — no in-script
delayNG patch needed.

Run:
    cd /home/lukas/PHD/CPlantBox
    source cpbenv/bin/activate
    PYTHONPATH=. python dart/coupling/output/blender_preview/_verify_production_tassel.py
"""
from __future__ import annotations
from pathlib import Path
import sys

import numpy as np
import plantbox as pb

from dart.coupling.config import DEFAULT_XML
from dart.coupling.geometry import extract_organs_for_lofter, loft_organs
from dart.coupling.growth.grow import setup_successor_where

DAY = 88  # pre-cessation at Juelich rates; should not crash
SEED = 7
OUT = Path(__file__).resolve().parent / "stages" / \
    f"verify_production_tassel_day{DAY}.obj"


def main() -> None:
    from dart.coupling.carbon.dvs_partitioning import get_daily_met
    daily_met = get_daily_met()
    if daily_met is None:
        raise SystemExit("No daily met CSV found; aborting.")

    plant = pb.MappedPlant()
    plant.readParameters(str(DEFAULT_XML))
    plant.setSeed(SEED)
    setup_successor_where(plant)
    plant.initialize()

    for day_1b in range(1, DAY + 1):
        T = float(daily_met.get(day_1b, {}).get('T_mean_C', 25.0))
        plant.setAirTemperature(T)
        plant.simulate(1.0, verbose=False)

    print(f"  final TT: {plant.getAccumulatedTT():.1f} Cd", flush=True)
    print(f"  extracting organs...", flush=True)

    organs = extract_organs_for_lofter(plant, skip_roots=True)
    mesh = loft_organs(organs, stem_sides=8, use_nurbs_backend=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    mesh.to_obj(str(OUT), group_by_organ=True)
    print(f"\nWrote: {OUT}")

    # Sanity-report the mainstem / tassel joint
    name_by_id = {m['organ_id']: m['name'] for m in mesh.organ_meta}
    V = mesh.vertices
    faces = mesh.indices.reshape(-1, 3)
    oids = mesh.organ_ids[faces[:, 0]]

    leaf_bases = sorted(
        np.asarray(o['skeleton'])[0, 2]
        for o in organs if o['type'] == 'leaf'
    )
    max_leaf_base = leaf_bases[-1] if leaf_bases else float('nan')

    for role, prefix in [("mainstem", "stem_"), ("tassel_spike", "tassel_spike_")]:
        target_ids = [oid for oid, nm in name_by_id.items() if nm.startswith(prefix)]
        if not target_ids:
            continue
        mask = np.isin(oids, target_ids)
        if not mask.any():
            continue
        verts = np.unique(faces[mask].flatten())
        zs = V[verts, 2]
        print(f"  {role:14s} mesh z: {zs.min():7.2f} .. {zs.max():7.2f}")

    print(f"  max leaf base z: {max_leaf_base:7.2f}")


if __name__ == "__main__":
    sys.exit(main() or 0)
