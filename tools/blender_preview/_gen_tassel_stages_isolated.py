"""Run ``_gen_tassel_stages_nodonor.py`` stages, each in a fresh subprocess.

Works around the pybind11 destructor heap-corruption that hits after
~2-3 sequential ``MappedPlant`` lifecycles in one process.

Run:
    cd /home/lukas/PHD/CPlantBox
    source cpbenv/bin/activate
    PYTHONPATH=. python dart/coupling/output/blender_preview/_gen_tassel_stages_isolated.py
"""
from __future__ import annotations
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
OUT = HERE / "stages_nodonor"
OUT.mkdir(exist_ok=True)

STAGES = [
    ("day55_notassel",  55, "None"),
    ("day58_tassel",    58, "0.15"),
    ("day60_tassel",    60, "0.65"),
    ("day62_tassel",    62, "1.00"),
    ("day65_tassel",    65, "1.00"),
    ("day70_tassel",    70, "1.00"),
]

WORKER = r'''
import sys, os
from pathlib import Path
import numpy as np
import plantbox as pb

HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path("{here}")
sys.path.insert(0, str(Path("{here}")))
from _gen_tassel_test import (
    DEFAULT_XML, SKEL_SEED,
    _build_tassel_organs, _merge_billboards_into_mesh,
    _find_main_stem_apex, _recenter,
)
from dart.coupling.geometry import extract_organs_for_lofter, loft_organs
from dart.coupling.growth.grow import setup_successor_where

LABEL = "{label}"
DAY = {day}
FRAC = {frac}
OUT = Path("{out}")
VT_CAP_DAY = 55
TASSEL_RNG_SEED = 20260421
ANTHER_RNG_SEED = 20260422

plant = pb.MappedPlant()
plant.readParameters(str(DEFAULT_XML))
plant.setSeed(SKEL_SEED)
setup_successor_where(plant)
main_stem = plant.getOrganRandomParameter(pb.stem, 1)
main_stem.delayNGStart = float(VT_CAP_DAY)
main_stem.delayNGEnd = 9999.0
plant.initialize()
plant.setAirTemperature(25.0)

step = 1.0
total = 0.0
while total < DAY:
    s = min(step, DAY - total)
    try:
        plant.simulate(s, verbose=False)
        total += s
    except (IndexError, RuntimeError) as e:
        print(f"  sim error day {{total+s:.1f}}: {{e}}")
        try: plant.simulate(0.0)
        except: pass
        break

base_organs = extract_organs_for_lofter(plant, skip_roots=True)
apex = _find_main_stem_apex(base_organs)
print(f"  apex @ ({{apex[0]:.1f}}, {{apex[1]:.1f}}, {{apex[2]:.1f}}) cm")

if FRAC is None:
    mesh = loft_organs(base_organs, stem_sides=8, use_nurbs_backend=True)
    _recenter(mesh, plant)
    out = OUT / f"plant_{{LABEL}}.obj"
    mesh.to_obj(str(out))
    print(f"  plant-only {{mesh.n_triangles}} t -> {{out.name}}")
else:
    next_id = max((o.get("organ_id", 0) for o in base_organs), default=-1) + 1

    tassel_only = _build_tassel_organs(np.zeros(3), FRAC, 0, np.random.default_rng(TASSEL_RNG_SEED))
    mesh_only = loft_organs(tassel_only, stem_sides=8, use_nurbs_backend=False)
    _merge_billboards_into_mesh(mesh_only, tassel_only, np.random.default_rng(ANTHER_RNG_SEED))
    out_only = OUT / f"tassel_only_{{LABEL}}.obj"
    mesh_only.to_obj(str(out_only))

    tassel = _build_tassel_organs(apex, FRAC, next_id, np.random.default_rng(TASSEL_RNG_SEED))
    organs = base_organs + tassel
    mesh = loft_organs(organs, stem_sides=8, use_nurbs_backend=True)
    _merge_billboards_into_mesh(mesh, tassel, np.random.default_rng(ANTHER_RNG_SEED))
    _recenter(mesh, plant)
    out = OUT / f"plant_{{LABEL}}.obj"
    mesh.to_obj(str(out))
    print(f"  tassel_only {{mesh_only.n_triangles}} t -> {{out_only.name}}")
    print(f"  plant+tassel {{mesh.n_triangles}} t -> {{out.name}}")

# Defensive teardown: drop plant before interpreter exit to dodge destructor race.
del plant
import gc; gc.collect()
sys.exit(0)
'''


def main() -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO) + os.pathsep + env.get("PYTHONPATH", "")
    for label, day, frac in STAGES:
        print(f"--- {label}  (day={day}, frac={frac}) ---", flush=True)
        code = WORKER.format(here=str(HERE), label=label, day=day, frac=frac, out=str(OUT))
        res = subprocess.run(
            [sys.executable, "-c", code], env=env, cwd=str(REPO),
            capture_output=True, text=True, timeout=120,
        )
        sys.stdout.write(res.stdout)
        if res.returncode != 0:
            sys.stderr.write(f"[stage {label} exit={res.returncode}]\n")
            sys.stderr.write(res.stderr[-2000:])
        print(flush=True)

    print(f"Wrote stage OBJs to {OUT}")


if __name__ == "__main__":
    sys.exit(main() or 0)
