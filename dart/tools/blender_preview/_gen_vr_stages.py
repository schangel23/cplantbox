"""Grow maize at ~13 representative days spanning V1 through R-stage proxies,
under FA-on Juelich-met calibration, and export each as a textured OBJ+MTL
to ``output/vr_stages/``.

Filenames carry day + auto-stage label (V<n> / VT_emerging / VT_mature /
VT_senescent) via ``dart.coupling.growth.phenology.detect_v_stage`` — same
labels the production grow CLI auto-appends.

Run:
    cd /home/lukas/PHD/CPlantBox
    source cpbenv/bin/activate
    PYTHONPATH=. python dart/tools/blender_preview/_gen_vr_stages.py
"""
from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

from dart.coupling.config import DEFAULT_XML
from dart.coupling.geometry import extract_organs_for_lofter, loft_organs
from dart.coupling.growth.grow import grow_plant
from dart.coupling.growth.phenology import (
    count_visible_leaves,
    detect_v_stage,
)

# Output: canonical pipeline location (resolves regardless of CWD).
# Script lives at dart/tools/blender_preview/<name>.py → parents[3] = project root.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = PROJECT_ROOT / "dart" / "coupling" / "output" / "vr_stages"

SEED = 42

# Day picks under FA-on Juelich met (tassel emerges day ~125, mature day ~150
# under calibrated stem TT cessation). Vegetative spaced denser early, sparser
# later; R-stages proxied by post-VT calendar days since CPlantBox has no
# kernel/cob model.
DAYS = [
    15,    # ~V1
    25,    # ~V2-V3
    35,    # ~V3-V4
    45,    # ~V4-V5
    55,    # ~V5-V6
    70,    # ~V8-V9
    85,    # ~V11-V12
    100,   # ~V14
    115,   # ~V15-V16
    125,   # tassel onset (VT_emerging)
    135,   # ~VT_mature
    150,   # R1-R2 proxy (silking → blister)
    165,   # R3 proxy (milk)
    180,   # R5-R6 proxy (dent → maturity, full senescence)
]


def grow_one(day: int) -> dict | None:
    """Grow + loft + export one stage. Returns metadata dict on success."""
    t0 = time.time()
    print(f"\n=== Day {day:3d} (seed {SEED}) ===")

    plant = grow_plant(
        str(DEFAULT_XML),
        simulation_time=day,
        seed=SEED,
        enable_photosynthesis=False,   # geometry-only, much faster
    )

    label = detect_v_stage(plant)
    counts = count_visible_leaves(plant)
    tt = plant.getAccumulatedTT() if hasattr(plant, "getAccumulatedTT") else -1.0

    out_path = OUT_DIR / f"maize_day{day:03d}_{label}.obj"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    organs = extract_organs_for_lofter(plant, species="maize", skip_roots=True)
    mesh = loft_organs(organs, stem_sides=8, use_nurbs_backend=True)
    # write_materials=True ships the sidecar .mtl with default colours
    # (blade, blade_senescent, midrib, stem, tassel) — Blender picks it up
    # automatically on OBJ import.
    mesh.to_obj(str(out_path), group_by_organ=True, write_materials=True)

    elapsed = time.time() - t0
    print(
        f"  -> {out_path.name}  "
        f"TT={tt:6.1f}  coll={counts['collared']:2d}/{counts['total']:2d}  "
        f"tris={mesh.n_triangles}  ({elapsed:.1f}s)"
    )
    return {
        "day": day,
        "label": label,
        "tt": tt,
        "n_collared": counts["collared"],
        "n_total_leaves": counts["total"],
        "n_triangles": mesh.n_triangles,
        "obj": str(out_path),
        "elapsed_s": elapsed,
    }


def main() -> int:
    print(f"Output folder: {OUT_DIR}")
    print(f"Days to grow:  {DAYS}")
    print(f"Seed:          {SEED}")
    t_total = time.time()
    results = []
    for day in DAYS:
        try:
            results.append(grow_one(day))
        except Exception as e:
            print(f"  FAILED day {day}: {e!r}")
            traceback.print_exc()

    print("\n" + "=" * 60)
    print(f"Done — {len(results)}/{len(DAYS)} stages in {time.time() - t_total:.1f}s")
    print("=" * 60)
    print("\n day   label             TT     coll/total  triangles  file")
    for r in results:
        print(
            f" {r['day']:3d}   {r['label']:<16s}  {r['tt']:6.1f}   "
            f"{r['n_collared']:2d}/{r['n_total_leaves']:2d}        "
            f"{r['n_triangles']:6d}    {Path(r['obj']).name}"
        )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except BaseException as e:
        print(f"FAILED: {e!r}")
        traceback.print_exc()
        sys.exit(1)
