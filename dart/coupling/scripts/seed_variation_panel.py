"""Grow N seed-replicates of the calibrated maize plant at one phenology stage.

Each seed reseeds CPlantBox's RNG (`plant.setSeed`), so every Gaussian
parameter with non-zero ``dev`` in ``maize_calibrated.xml`` (``cultivar_height_factor``,
per-rank ``ln``, leaf ``lmax``/``r``/``theta``, stem ``r``/``lmax``/``theta``,
organ radii, etc.) is drawn afresh. The resulting plants share one cultivar
calibration but differ along the 23 stochastic dimensions.

Outputs (under ``--out-dir``):

    seed_NNN_<V-label>.obj      production-lofter geometry per seed
    seed_NNN_<V-label>.mtl      sidecar materials
    seed_NNN_<V-label>_dart.obj DART-routed group-named OBJ (tassel split)
    stats.csv                   one row per seed: ms_len, n_leaves, leaf_area, V-stage

Run:

    cd /home/lukas/PHD/CPlantBox
    cpbenv/bin/python -m dart.coupling.scripts.seed_variation_panel \\
        --seeds 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 \\
        --day 130 \\
        --out-dir dart/coupling/output/seed_variation_day130

Then assemble the .blend overview with ``blender_seed_variation_panel.py``.
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path

# Ensure the local CPlantBox repo is importable when run as a script.
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dart.coupling.growth.grow import grow_plant  # noqa: E402
from dart.coupling.growth.phenology import detect_v_stage, count_visible_leaves  # noqa: E402
from dart.coupling.geometry.cplantbox_adapter import extract_organs_for_lofter  # noqa: E402
from dart.coupling.geometry.g1_to_g3 import loft_organs  # noqa: E402

import plantbox as pb  # noqa: E402


def _mainstem_len_cm(plant) -> float:
    """Arc length of the longest stem axis (mainstem proxy)."""
    longest = 0.0
    for organ in plant.getOrgans(pb.stem):
        nodes = organ.getNodes()
        if len(nodes) < 2:
            continue
        arc = 0.0
        prev = nodes[0]
        for nd in nodes[1:]:
            arc += math.sqrt(
                (float(nd.x) - float(prev.x)) ** 2
                + (float(nd.y) - float(prev.y)) ** 2
                + (float(nd.z) - float(prev.z)) ** 2
            )
            prev = nd
        longest = max(longest, arc)
    return longest


def _topmost_leaf_z_cm(plant) -> float:
    z = -1e9
    for leaf in plant.getOrgans(pb.leaf):
        nodes = leaf.getNodes()
        if not nodes:
            continue
        z = max(z, float(nodes[0].z))
    return z if z > -1e8 else 0.0


def _total_leaf_area_cm2(mesh) -> float:
    """Sum of triangle areas tagged as leaf/midrib in the lofted mesh."""
    leaf_parts = {"blade", "leaf", "midrib"}
    leaf_organ_ids = {
        m["organ_id"] for m in (mesh.organ_meta or [])
        if m.get("part_type", m.get("type", "")) in leaf_parts
    }
    if not leaf_organ_ids:
        return 0.0
    verts = mesh.vertices
    total = 0.0
    for tri_idx in range(len(mesh.indices)):
        if int(mesh.organ_ids[tri_idx]) not in leaf_organ_ids:
            continue
        a, b, c = verts[mesh.indices[tri_idx]]
        ux, uy, uz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
        vx, vy, vz = c[0] - a[0], c[1] - a[1], c[2] - a[2]
        cx = uy * vz - uz * vy
        cy = uz * vx - ux * vz
        cz = ux * vy - uy * vx
        total += 0.5 * math.sqrt(cx * cx + cy * cy + cz * cz)
    return total


def grow_one_seed(xml_path: Path, day: int, seed: int, out_dir: Path,
                  cp_donor_seed: int | None = None,
                  cp_donor_mode: str = "draw_coherent",
                  cp_donor_smooth_alpha: float = 1.0) -> dict:
    t0 = time.time()
    plant = grow_plant(
        xml_path=str(xml_path),
        simulation_time=day,
        seed=seed,
        enable_photosynthesis=True,
        cp_donor_seed=cp_donor_seed,
        cp_donor_mode=cp_donor_mode,
        cp_donor_smooth_alpha=cp_donor_smooth_alpha,
    )
    label = detect_v_stage(plant)
    counts = count_visible_leaves(plant)
    ms_len = _mainstem_len_cm(plant)
    top_z = _topmost_leaf_z_cm(plant)

    organs = extract_organs_for_lofter(plant, species="maize")
    mesh = loft_organs(organs, subdivide=False)
    leaf_area = _total_leaf_area_cm2(mesh)

    stem = f"seed_{seed:03d}_{label}"
    obj_path = out_dir / f"{stem}.obj"
    mesh.to_obj(str(obj_path), write_materials=True)

    elapsed = time.time() - t0
    return {
        "seed": seed,
        "day": day,
        "label": label,
        "ms_len_cm": round(ms_len, 3),
        "topmost_leaf_z_cm": round(top_z, 3),
        "n_leaves_total": counts["total"],
        "n_leaves_collared": counts["collared"],
        "n_leaves_emerging": counts["emerging"],
        "leaf_area_cm2": round(leaf_area, 2),
        "obj": obj_path.name,
        "elapsed_s": round(elapsed, 1),
    }


def main() -> int:
    p = argparse.ArgumentParser(
        description="Grow N seed-replicates at one phenology stage and export OBJs.",
    )
    p.add_argument(
        "--xml",
        default=str(REPO_ROOT / "dart/coupling/data/maize_calibrated.xml"),
    )
    p.add_argument("--day", type=int, default=130)
    p.add_argument("--seeds", type=int, nargs="+", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument(
        "--cp-donor",
        choices=["off", "match", "draw", "draw_coherent"],
        default="match",
        help="Per-plant MF3D leaf-CP donor mode. 'match' (default) reuses "
             "the panel seed as cp_donor_seed in draw_coherent mode — one "
             "donor plant per panel slot, picked from the 5-plant pool "
             "{216, 374, 430, 446, 455} that survives _default_tip_bounds "
             "at all 14 positions (per RUNTIME_CP_SWAP_2026-04-20.md). "
             "'draw' draws independently per position. 'off' disables donor "
             "swap (XML CPs).",
    )
    p.add_argument(
        "--smooth-alpha",
        type=float,
        default=0.7,
        help="Smooth-CP blend weight (RUNTIME_CP_SWAP_2026-04-20 §187). "
             "1.0 = pure donor (legacy, asymmetric mid-bulges visible). "
             "0.7 (default) = 70%% donor + 30%% per-position median, "
             "damps asymmetric mid-bulges while keeping donor identity. "
             "0.0 = pure median (no plant-to-plant variation). Only "
             "engaged for draw / draw_coherent modes.",
    )
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    xml_path = Path(args.xml)

    if args.cp_donor == "off":
        donor_mode_arg = "draw_coherent"  # ignored when seed=None
        donor_seed_arg = None
    elif args.cp_donor == "match":
        donor_mode_arg = "draw_coherent"
        donor_seed_arg = "use_panel_seed"  # sentinel: reuse seed per slot
    else:
        donor_mode_arg = args.cp_donor
        donor_seed_arg = "use_panel_seed"

    rows: list[dict] = []
    for i, seed in enumerate(args.seeds, 1):
        print(f"\n=== [{i}/{len(args.seeds)}] seed={seed} day={args.day} "
              f"cp_donor={args.cp_donor} ===")
        try:
            row = grow_one_seed(
                xml_path, args.day, seed, out_dir,
                cp_donor_seed=(seed if donor_seed_arg == "use_panel_seed" else None),
                cp_donor_mode=donor_mode_arg,
                cp_donor_smooth_alpha=args.smooth_alpha,
            )
        except Exception as exc:
            print(f"  FAILED: {exc!r}")
            row = {
                "seed": seed, "day": args.day, "label": "FAILED",
                "ms_len_cm": 0, "topmost_leaf_z_cm": 0,
                "n_leaves_total": 0, "n_leaves_collared": 0,
                "n_leaves_emerging": 0, "leaf_area_cm2": 0,
                "obj": "", "elapsed_s": 0, "error": repr(exc),
            }
        rows.append(row)
        print(f"  -> {row}")

    csv_path = out_dir / "stats.csv"
    fieldnames = list(rows[0].keys())
    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {csv_path}")

    ok = [r for r in rows if r["label"] != "FAILED"]
    if ok:
        ms = [r["ms_len_cm"] for r in ok]
        mean_ms = sum(ms) / len(ms)
        var = sum((x - mean_ms) ** 2 for x in ms) / max(len(ms) - 1, 1)
        sigma = math.sqrt(var)
        labels = sorted({r["label"] for r in ok})
        print(f"\nSummary across {len(ok)} successful seeds:")
        print(f"  mainstem length: {mean_ms:.1f} ± {sigma:.1f} cm "
              f"(min {min(ms):.1f}, max {max(ms):.1f})")
        print(f"  V-stage labels seen: {labels}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
