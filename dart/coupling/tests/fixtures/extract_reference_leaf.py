#!/usr/bin/env python3
"""Extract one leaf from maize_stage_16.obj as a standalone OBJ.

The source OBJ has all leaves merged under a single `g leaf` group, but each
leaf is a disjoint mesh island (connected component). We pick one component
by insertion height (z) rank and write it back with re-indexed vertices.

Usage:
    python3 extract_reference_leaf.py [--rank N] [--src PATH] [--out PATH]

Default: rank=5 (mid-height leaf; has a clear sheath wrap for templating).
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path


def parse_obj(path: Path):
    """Return (vertices, leaf_faces). Coordinates stay in the source unit (metres)."""
    verts: list[tuple[float, float, float]] = []
    leaf_faces: list[list[int]] = []
    current_group = None
    with path.open() as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            tag = parts[0]
            if tag == "v" and len(parts) >= 4:
                verts.append((float(parts[1]), float(parts[2]), float(parts[3])))
            elif tag == "g" and len(parts) > 1:
                current_group = parts[1]
            elif tag == "f" and current_group == "leaf":
                face = [int(p.split("/")[0]) - 1 for p in parts[1:]]
                leaf_faces.append(face)
    return verts, leaf_faces


def connected_components(faces: list[list[int]]) -> list[set[int]]:
    """Union-find over vertices linked by shared faces."""
    parent: dict[int, int] = {}

    def find(x: int) -> int:
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for face in faces:
        for v in face[1:]:
            union(face[0], v)

    groups: dict[int, set[int]] = defaultdict(set)
    for face in faces:
        for v in face:
            groups[find(v)].add(v)
    return list(groups.values())


def write_obj(path: Path, verts, faces, comp_vids: set[int], mtl_name: str) -> None:
    old_to_new: dict[int, int] = {}
    out_verts = []
    for vid in sorted(comp_vids):
        old_to_new[vid] = len(out_verts) + 1  # OBJ indices are 1-based
        out_verts.append(verts[vid])

    out_faces = []
    comp_set = comp_vids
    for face in faces:
        if all(v in comp_set for v in face):
            out_faces.append([old_to_new[v] for v in face])

    with path.open("w") as f:
        f.write(f"# extracted leaf component (n_v={len(out_verts)}, n_f={len(out_faces)})\n")
        f.write(f"mtllib {mtl_name}.mtl\n")
        f.write("o reference_leaf\n")
        for v in out_verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        f.write("g leaf\nusemtl leaf\n")
        for face in out_faces:
            f.write("f " + " ".join(str(i) for i in face) + "\n")


def main() -> None:
    here = Path(__file__).parent
    default_src = Path("/home/lukas/Downloads/Maize/export/maize_stage_16.obj")

    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=default_src)
    ap.add_argument("--out-dir", type=Path, default=here)
    ap.add_argument("--rank", type=int, default=5,
                    help="0-based leaf index, ordered by insertion height (min-z within component).")
    ap.add_argument("--all", action="store_true",
                    help="Export every component as reference_leaf_<rank>.obj.")
    args = ap.parse_args()

    verts, leaf_faces = parse_obj(args.src)
    comps = connected_components(leaf_faces)
    print(f"parsed {len(verts)} verts, {len(leaf_faces)} leaf faces, {len(comps)} components")

    def min_z(c: set[int]) -> float:
        return min(verts[v][2] for v in c)

    def max_z(c: set[int]) -> float:
        return max(verts[v][2] for v in c)

    comps_sorted = sorted(comps, key=min_z)  # Blender -Z up in this file → lowest min_z = tip
    for i, c in enumerate(comps_sorted):
        bbox_z = max_z(c) - min_z(c)
        print(f"  rank {i}: n_v={len(c):4d} z_range={bbox_z:.2f} (min_z={min_z(c):.2f})")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    ranks = range(len(comps_sorted)) if args.all else [args.rank]
    for r in ranks:
        if not (0 <= r < len(comps_sorted)):
            raise SystemExit(f"rank {r} out of range (have {len(comps_sorted)} components)")
        comp = comps_sorted[r]
        out_path = args.out_dir / (f"reference_leaf_stage16_rank{r:02d}.obj"
                                   if args.all else f"reference_leaf_stage16.obj")
        write_obj(out_path, verts, leaf_faces, comp, out_path.stem)
        print(f"wrote {out_path}  ({len(comp)} verts)")


if __name__ == "__main__":
    main()
