"""Reality-aligned tassel progression: plant grown to each day, tassel sized
to its maturity on that day, with main-stem growth capped at VT (day 55).

Timeline (calendar days) — tassel extends over ~4-5 real days, not 10:
  day 55  —  pre-VT baseline                        (no tassel)
  day 58  —  tassel peeking from whorl              (frac 0.15, ~6 cm spike)
  day 60  —  VT onset, tassel mostly extended       (frac 0.65, ~26 cm)
  day 62  —  VT complete, anthers pre-emergent      (frac 1.00, 40 cm)
  day 65  —  anthesis active                        (frac 1.00)
  day 70  —  mature, anthers senescing              (frac 1.00)

Main-stem growth is capped at day 55 via ``delayNGStart=55, delayNGEnd=9999``
on subType 1 — reflects the biological reality that stem elongation halts at
VT. Applied in-memory before ``initialize()``, so the canonical calibrated
XML is untouched.

Run:
    cd /home/lukas/PHD/CPlantBox
    source cpbenv/bin/activate
    PYTHONPATH=. python dart/coupling/output/blender_preview/_gen_tassel_stages.py
"""
from __future__ import annotations
from pathlib import Path
import sys

import numpy as np
import plantbox as pb

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _gen_tassel_test import (  # type: ignore[import-not-found]
    DEFAULT_XML, SKEL_SEED, DONOR_SEED,
    _build_tassel_organs, _merge_billboards_into_mesh,
    _find_main_stem_apex, _recenter,
)
from dart.coupling.geometry import extract_organs_for_lofter, loft_organs
from dart.coupling.growth.grow import apply_donor_cps, setup_successor_where

OUT = Path(__file__).resolve().parent / "stages"
OUT.mkdir(exist_ok=True)

TASSEL_RNG_SEED = 20260421
ANTHER_RNG_SEED = 20260422
VT_CAP_DAY = 55          # main-stem growth freezes at this day (VT)
DELAY_NG_END = 9999.0    # effectively infinite — growth never resumes

# (label, plant_day, tassel_frac). frac=None → no tassel.
STAGES = [
    ("day55_notassel",  55, None),
    ("day58_tassel",    58, 0.15),
    ("day60_tassel",    60, 0.65),
    ("day62_tassel",    62, 1.00),
    ("day65_tassel",    65, 1.00),
    ("day70_tassel",    70, 1.00),
]


def _cap_main_stem_growth(plant: "pb.Plant") -> None:
    """Freeze main stem (subType 1) length after VT_CAP_DAY."""
    main_stem = plant.getOrganRandomParameter(pb.stem, 1)
    main_stem.delayNGStart = float(VT_CAP_DAY)
    main_stem.delayNGEnd = DELAY_NG_END


def _grow_vt_capped(simulation_time: int, seed: int) -> "pb.MappedPlant":
    """Mirror ``grow_plant`` but patch main-stem delayNG* before initialize()."""
    plant = pb.MappedPlant()
    plant.readParameters(str(DEFAULT_XML))
    plant.setSeed(seed)
    apply_donor_cps(plant, donor_seed=DONOR_SEED, mode="draw_coherent", verbose=False)
    setup_successor_where(plant)
    _cap_main_stem_growth(plant)
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
        print(f"--- {label}  (day={day}, frac={frac}, stem_cap={VT_CAP_DAY}) ---")
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
