"""Stage progression for Blender preview.

Six representative tassel-emergence stages on the production XML, using
the parametric leaf-shape path. Output: full-plant OBJ per stage.

Run:
    cd /home/lukas/PHD/CPlantBox
    source cpbenv/bin/activate
    PYTHONPATH=. python dart/coupling/output/blender_preview/_gen_tassel_stages_nodonor.py
"""
from __future__ import annotations
from pathlib import Path
import sys

import numpy as np
import plantbox as pb

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _gen_tassel_test import (  # type: ignore[import-not-found]
    DEFAULT_XML, SKEL_SEED,
    _build_tassel_organs, _merge_billboards_into_mesh,
    _find_main_stem_apex, _recenter,
)
from dart.coupling.geometry import extract_organs_for_lofter, loft_organs
from dart.coupling.growth.grow import setup_successor_where

OUT = Path(__file__).resolve().parent / "stages_nodonor"
OUT.mkdir(exist_ok=True)

TASSEL_RNG_SEED = 20260421
ANTHER_RNG_SEED = 20260422
VT_CAP_DAY = 55
DELAY_NG_END = 9999.0

STAGES = [
    ("day55_notassel",  55, None),
    ("day58_tassel",    58, 0.15),
    ("day60_tassel",    60, 0.65),
    ("day62_tassel",    62, 1.00),
    ("day65_tassel",    65, 1.00),
    ("day70_tassel",    70, 1.00),
]


def _grow_vt_capped(simulation_time: int, seed: int) -> "pb.MappedPlant":
    plant = pb.MappedPlant()
    plant.readParameters(str(DEFAULT_XML))
    plant.setSeed(seed)
    setup_successor_where(plant)
    main_stem = plant.getOrganRandomParameter(pb.stem, 1)
    main_stem.delayNGStart = float(VT_CAP_DAY)
    main_stem.delayNGEnd = DELAY_NG_END
    plant.initialize()

    dt = 1.0
    total = 0.0
    while total < simulation_time:
        step = min(dt, simulation_time - total)
        plant.setAirTemperature(25.0)
        try:
            plant.simulate(step, verbose=False)
            total += step
        except (IndexError, RuntimeError) as e:
            print(f"  sim error at day {total + step:.1f}: {e} — stopping early")
            try:
                plant.simulate(0.0)
            except Exception:
                pass
            break
    return plant


def main() -> None:
    for label, day, frac in STAGES:
        print(f"--- {label}  (day={day}, frac={frac}) ---")
        plant = _grow_vt_capped(simulation_time=day, seed=SKEL_SEED)
        base_organs = extract_organs_for_lofter(plant, skip_roots=True)
        apex = _find_main_stem_apex(base_organs)
        print(f"  apex @ ({apex[0]:.1f}, {apex[1]:.1f}, {apex[2]:.1f}) cm")

        if frac is None:
            mesh = loft_organs(base_organs, stem_sides=8, use_nurbs_backend=True)
            _recenter(mesh, plant)
            out = OUT / f"plant_{label}.obj"
            mesh.to_obj(str(out))
            print(f"  plant-only: {mesh.n_triangles} t  →  {out.name}\n")
            continue

        next_id = max((o.get("organ_id", 0) for o in base_organs), default=-1) + 1

        tassel_only = _build_tassel_organs(
            np.zeros(3), frac, 0, np.random.default_rng(TASSEL_RNG_SEED)
        )
        mesh_only = loft_organs(tassel_only, stem_sides=8, use_nurbs_backend=False)
        _merge_billboards_into_mesh(
            mesh_only, tassel_only, np.random.default_rng(ANTHER_RNG_SEED)
        )
        out_only = OUT / f"tassel_only_{label}.obj"
        mesh_only.to_obj(str(out_only))

        tassel = _build_tassel_organs(
            apex, frac, next_id, np.random.default_rng(TASSEL_RNG_SEED)
        )
        organs = base_organs + tassel
        mesh = loft_organs(organs, stem_sides=8, use_nurbs_backend=True)
        _merge_billboards_into_mesh(
            mesh, tassel, np.random.default_rng(ANTHER_RNG_SEED)
        )
        _recenter(mesh, plant)
        out = OUT / f"plant_{label}.obj"
        mesh.to_obj(str(out))
        print(f"  tassel_only = {mesh_only.n_triangles:>6d} t  →  {out_only.name}")
        print(f"  plant+tassel = {mesh.n_triangles:>6d} t  →  {out.name}\n")

    print(f"Wrote stage OBJs to {OUT}")


if __name__ == "__main__":
    sys.exit(main() or 0)
