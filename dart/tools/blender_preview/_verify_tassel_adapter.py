"""§2 + §3 verification: grow day-90 plant, adapter + lofter + billboards.

Expects:
  - §2: tassel_spike_* and tassel_branch_* groups propagate as organ names.
  - §3: anther cross-billboards appear on tassel organs (tagged
    segment_id=-1, same organ_id as their parent spike / branch).

Writes an OBJ so the output can be visually compared against §10's
approved preview silhouette.

Run (subprocess-isolated because multi-grow in one process segfaults):
    cd /home/lukas/PHD/CPlantBox
    source cpbenv/bin/activate
    PYTHONPATH=. python dart/coupling/output/blender_preview/_verify_tassel_adapter.py
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from dart.coupling.config import DEFAULT_XML
from dart.coupling.growth.grow import grow_plant


def _summarize(organs):
    import numpy as np
    counts = Counter()
    bbox = {}
    for o in organs:
        name = o["name"]
        prefix = name.rsplit("_", 1)[0]
        counts[prefix] += 1
        # skeleton may be absent on nurbs-backend leaves; fall back to CPs
        if "skeleton" in o:
            skel = np.asarray(o["skeleton"])
        elif "surface_cps_local" in o:
            cps = np.asarray(o["surface_cps_local"])
            skel = cps[:, cps.shape[1] // 2, :]
        else:
            continue
        widths = o.get("widths")
        bbox.setdefault(prefix, {
            "n": 0, "min_w": float("inf"), "max_w": 0.0,
            "min_z": float("inf"), "max_z": float("-inf"),
        })
        b = bbox[prefix]
        b["n"] += 1
        if widths is not None and len(widths):
            b["min_w"] = min(b["min_w"], float(np.min(widths)))
            b["max_w"] = max(b["max_w"], float(np.max(widths)))
        b["min_z"] = min(b["min_z"], float(skel[:, 2].min()))
        b["max_z"] = max(b["max_z"], float(skel[:, 2].max()))
    return counts, bbox


def main():
    DAY = 88
    print(f"=== Growing day-{DAY} plant (seed=42) ===")
    plant = grow_plant(str(DEFAULT_XML), simulation_time=DAY, seed=42,
                       enable_photosynthesis=False)

    # Deferred import: pulling the geometry package at module load time
    # interacts badly with CPlantBox's simulate() for late-stage plants (~day 90)
    # and segfaults. Importing after grow_plant returns avoids this.
    from dart.coupling.geometry import extract_organs_for_lofter, loft_organs

    print("\n=== Running adapter ===")
    organs = extract_organs_for_lofter(plant, min_stem_nodes=50,
                                       min_leaf_nodes=20, skip_roots=True)

    counts, bbox = _summarize(organs)
    print("\n  Organ groups by prefix:")
    for prefix, n in sorted(counts.items()):
        b = bbox[prefix]
        print(f"    {prefix:20s}  n={n:3d}  "
              f"w=[{b['min_w']:.3f},{b['max_w']:.3f}] cm  "
              f"z=[{b['min_z']:6.1f},{b['max_z']:6.1f}] cm")

    has_spike = any(o["name"].startswith("tassel_spike_") for o in organs)
    has_branch = any(o["name"].startswith("tassel_branch_") for o in organs)
    print(f"\n  has tassel_spike_ group: {has_spike}")
    print(f"  has tassel_branch_ group: {has_branch}")
    if not (has_spike and has_branch):
        raise SystemExit("FAIL: tassel group(s) missing")

    print("\n=== Lofting ===")
    mesh = loft_organs(organs, stem_sides=8)
    print(f"  mesh: {mesh.n_vertices} verts  {mesh.n_triangles} tris  "
          f"{len(mesh.organ_meta)} organ_meta entries")

    # Count triangles per named prefix; split billboard (segment_id=-1) tris
    # from lofted tube / ribbon tris so §3 coverage is visible.
    tri_counts = Counter()
    bb_counts = Counter()
    name_by_id = {m["organ_id"]: m["name"] for m in mesh.organ_meta if "organ_id" in m and "name" in m}
    for oid, seg in zip(mesh.organ_ids, mesh.segment_ids):
        nm = name_by_id.get(int(oid), f"?{oid}")
        prefix = nm.rsplit("_", 1)[0]
        if int(seg) < 0:
            bb_counts[prefix] += 1
        else:
            tri_counts[prefix] += 1
    print("\n  Triangles by prefix (lofted | billboard):")
    all_prefixes = sorted(set(tri_counts) | set(bb_counts))
    for prefix in all_prefixes:
        print(f"    {prefix:20s}  lofted={tri_counts.get(prefix, 0):6d}  "
              f"billboard={bb_counts.get(prefix, 0):6d}")

    # §3 assertion: tassel organs must contribute billboard tris
    bb_spike = bb_counts.get("tassel_spike", 0)
    bb_branch = bb_counts.get("tassel_branch", 0)
    print(f"\n  billboard tris on spike: {bb_spike}")
    print(f"  billboard tris on branches: {bb_branch}")
    if bb_spike == 0:
        raise SystemExit("FAIL: no anther billboards on tassel spike")

    out_dir = Path("dart/coupling/output/blender_preview/stages")
    out_dir.mkdir(parents=True, exist_ok=True)
    obj_path = out_dir / "verify_tassel_billboards_day88.obj"
    mesh.to_obj(str(obj_path), group_by_organ=True)
    print(f"\n  wrote: {obj_path}")


if __name__ == "__main__":
    main()
