#!/usr/bin/env python3
"""Match lofter output to OBJ reference vertex-by-vertex.

For each leaf at each stage:
1. Extract skeleton from OBJ → run production lofter → base mesh
2. Find nearest OBJ vertex for each lofter vertex → displacement
3. Apply displacement → exact match mesh
4. Export both (base + corrected) as OBJ for visual comparison
5. Analyze displacement field: what % is capturable by existing deformations

Usage:
    python3 match_vertices.py /path/to/Maize/export/ --output output/matched/ -v
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import KDTree

_CPLANTBOX_ROOT = str(Path(__file__).resolve().parent.parent.parent.parent)
sys.path.insert(0, _CPLANTBOX_ROOT)

from dart.coupling.geometry.g1_to_g3 import loft_organs
from dart.coupling.experimental.reverse_engineer_maize import (
    parse_obj, find_connected_components, track_leaves_across_stages,
    extract_leaf_g1, count_developed_leaves, vstage_to_day,
)


def load_all_stages(export_dir):
    """Load all stages with per-leaf tracking."""
    export_dir = Path(export_dir)
    manifest = json.loads((export_dir / "manifest.json").read_text())
    files = [(s["stage"], export_dir / s["file"]) for s in manifest["stages"]]

    all_verts, all_groups, all_comps = [], [], []
    for _, fpath in files:
        verts, groups = parse_obj(fpath)
        verts[:, 2] *= -1  # flip Z
        all_verts.append(verts)
        all_groups.append(groups)
        lf = []
        for gn, gf in groups.items():
            if "leaf" in gn.lower():
                lf.extend(gf)
        all_comps.append(find_connected_components(lf))

    canonical = track_leaves_across_stages(all_comps)
    sorted_lids = sorted(canonical.keys(),
                         key=lambda lid: np.mean(all_verts[0][list(
                             canonical[lid].get(0, canonical[lid][list(
                                 canonical[lid].keys())[0]]))][:, 2]))
    pos_map = {lid: p + 1 for p, lid in enumerate(sorted_lids)}

    return files, all_verts, all_groups, all_comps, canonical, pos_map


def process_leaf(verts, leaf_faces, comp, g1):
    """Run bare lofter on extracted skeleton, compute per-vertex displacement.

    Returns:
        base_verts: (M, 3) bare lofter output
        corrected_verts: (M, 3) lofter + displacement = matches OBJ
        displacements: (M, 3) per-vertex correction vectors
        obj_leaf_verts: (N, 3) original OBJ vertices for this leaf
        match_indices: (M,) index into obj_leaf_verts for each lofter vertex
        mesh_indices: (K, 3) triangle indices from lofter
    """
    skel = np.array(g1.skeleton)
    widths = np.array(g1.widths)
    if len(skel) < 3 or widths.max() < 0.1:
        return None

    organ = {
        "type": "leaf",
        "skeleton": skel,
        "widths": widths,
        "organ_id": g1.leaf_id,
        "name": f"leaf_{g1.position}",
        "node_ids": list(range(len(skel))),
    }

    try:
        mesh = loft_organs([organ], subdivide=True, smooth=False)
    except Exception:
        return None

    base_verts = mesh.vertices
    mesh_indices = mesh.indices

    # OBJ vertices for this leaf
    comp_ids = sorted(comp)
    obj_leaf_verts = verts[comp_ids]

    # Nearest-neighbor: for each lofter vert, find closest OBJ vert
    obj_tree = KDTree(obj_leaf_verts)
    dists, match_idx = obj_tree.query(base_verts)

    # Displacement = target - source
    displacements = obj_leaf_verts[match_idx] - base_verts
    corrected_verts = base_verts + displacements

    return {
        "base_verts": base_verts,
        "corrected_verts": corrected_verts,
        "displacements": displacements,
        "obj_leaf_verts": obj_leaf_verts,
        "match_indices": match_idx,
        "mesh_indices": mesh_indices,
        "match_dists": dists,
    }


def export_obj(path, verts, faces, comment=""):
    """Write vertices and faces to OBJ file."""
    with open(path, "w") as f:
        if comment:
            f.write(f"# {comment}\n")
        for v in verts:
            f.write(f"v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}\n")
        for tri in faces:
            f.write(f"f {tri[0]+1} {tri[1]+1} {tri[2]+1}\n")


def decompose_displacement_field(displacements, base_verts, skeleton, widths):
    """Analyze how much of the displacement field is capturable by
    existing lofter deformation types.

    Returns dict with per-type analysis.
    """
    n_skel = len(skeleton)
    if n_skel < 3:
        return {}

    # Local frame at each skeleton point
    tangents = np.zeros((n_skel, 3))
    tangents[0] = skeleton[1] - skeleton[0]
    tangents[-1] = skeleton[-1] - skeleton[-2]
    for i in range(1, n_skel - 1):
        tangents[i] = skeleton[i + 1] - skeleton[i - 1]
    norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    tangents = tangents / np.maximum(norms, 1e-8)

    up = np.array([0, 0, 1.0])
    binormals = np.cross(tangents, up)
    bn_norms = np.linalg.norm(binormals, axis=1, keepdims=True)
    binormals = binormals / np.maximum(bn_norms, 1e-8)
    normals = np.cross(binormals, tangents)

    # Assign each vertex to nearest skeleton point
    skel_tree = KDTree(skeleton)
    _, skel_idx = skel_tree.query(base_verts)

    # Decompose each displacement
    d_tang = np.zeros(len(displacements))
    d_norm = np.zeros(len(displacements))
    d_binorm = np.zeros(len(displacements))

    for i in range(len(displacements)):
        si = skel_idx[i]
        d = displacements[i]
        d_tang[i] = np.dot(d, tangents[si])
        d_norm[i] = np.dot(d, normals[si])
        d_binorm[i] = np.dot(d, binormals[si])

    total_energy = np.sum(np.linalg.norm(displacements, axis=1) ** 2)
    tang_energy = np.sum(d_tang ** 2)
    norm_energy = np.sum(d_norm ** 2)
    binorm_energy = np.sum(d_binorm ** 2)

    # Percentages
    if total_energy > 0:
        pct_tang = tang_energy / total_energy * 100
        pct_norm = norm_energy / total_energy * 100
        pct_binorm = binorm_energy / total_energy * 100
    else:
        pct_tang = pct_norm = pct_binorm = 0

    # Per-skeleton-point profile (mean displacement in each direction)
    tang_profile = np.zeros(n_skel)
    norm_profile = np.zeros(n_skel)
    binorm_profile = np.zeros(n_skel)
    counts = np.zeros(n_skel)
    for i in range(len(displacements)):
        si = skel_idx[i]
        tang_profile[si] += d_tang[i]
        norm_profile[si] += d_norm[i]
        binorm_profile[si] += d_binorm[i]
        counts[si] += 1
    counts = np.maximum(counts, 1)
    tang_profile /= counts
    norm_profile /= counts
    binorm_profile /= counts

    return {
        "total_displacement_rms": float(np.sqrt(np.mean(
            np.linalg.norm(displacements, axis=1) ** 2))),
        "tangent_pct": float(pct_tang),
        "normal_pct": float(pct_norm),
        "binormal_pct": float(pct_binorm),
        "tangent_profile": tang_profile.tolist(),
        "normal_profile": norm_profile.tolist(),
        "binormal_profile": binorm_profile.tolist(),
        "tangent_rms": float(np.sqrt(np.mean(d_tang ** 2))),
        "normal_rms": float(np.sqrt(np.mean(d_norm ** 2))),
        "binormal_rms": float(np.sqrt(np.mean(d_binorm ** 2))),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Match lofter vertices to OBJ reference 1:1")
    parser.add_argument("export_dir", help="Directory with maize_stage_*.obj")
    parser.add_argument("--output", "-o", default="output/matched")
    parser.add_argument("--stages", default="all",
                        help="Stages to process: 'all', 'last', or '1,8,14'")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("VERTEX MATCHING — Lofter → OBJ 1:1")
    print("=" * 70)

    # Load
    files, all_verts, all_groups, all_comps, canonical, pos_map = \
        load_all_stages(args.export_dir)

    # Select stages
    if args.stages == "all":
        stage_indices = list(range(len(files)))
    elif args.stages == "last":
        stage_indices = [len(files) - 1]
    else:
        stage_indices = [int(s) - 1 for s in args.stages.split(",")]

    all_results = []

    for sidx in stage_indices:
        stage_num = files[sidx][0]
        verts = all_verts[sidx]
        groups = all_groups[sidx]

        leaf_faces = []
        for gn, gf in groups.items():
            if "leaf" in gn.lower():
                leaf_faces.extend(gf)

        print(f"\n--- Stage {stage_num} ---")
        stage_dir = output_dir / f"stage_{stage_num:02d}"
        stage_dir.mkdir(exist_ok=True)

        stage_results = {"stage": stage_num, "leaves": []}

        for lid, stage_map in canonical.items():
            if sidx not in stage_map:
                continue
            pos = pos_map[lid]
            comp = stage_map[sidx]
            g1 = extract_leaf_g1(comp, verts, leaf_faces, lid, pos, n_samples=20)
            if g1.length < 3:
                continue

            result = process_leaf(verts, leaf_faces, comp, g1)
            if result is None:
                continue

            # Export OBJs
            export_obj(stage_dir / f"leaf{pos:02d}_base.obj",
                       result["base_verts"], result["mesh_indices"],
                       f"Bare lofter - leaf {pos} stage {stage_num}")
            export_obj(stage_dir / f"leaf{pos:02d}_corrected.obj",
                       result["corrected_verts"], result["mesh_indices"],
                       f"Corrected (1:1 match) - leaf {pos} stage {stage_num}")
            export_obj(stage_dir / f"leaf{pos:02d}_reference.obj",
                       result["obj_leaf_verts"],
                       [f for f in leaf_faces if all(v in comp for v in f)],
                       f"OBJ reference - leaf {pos} stage {stage_num}")

            # Displacement analysis
            decomp = decompose_displacement_field(
                result["displacements"], result["base_verts"],
                np.array(g1.skeleton), np.array(g1.widths))

            mean_match_dist = float(result["match_dists"].mean())
            max_match_dist = float(result["match_dists"].max())

            leaf_result = {
                "position": pos,
                "n_lofter_verts": len(result["base_verts"]),
                "n_obj_verts": len(result["obj_leaf_verts"]),
                "mean_nn_dist": mean_match_dist,
                "max_nn_dist": max_match_dist,
                **decomp,
            }
            stage_results["leaves"].append(leaf_result)

            if args.verbose:
                print(f"  Leaf {pos:>2}: {len(result['base_verts']):>4} lofter → "
                      f"{len(result['obj_leaf_verts']):>4} OBJ verts | "
                      f"disp={decomp['total_displacement_rms']:.2f}cm "
                      f"(T:{decomp['tangent_pct']:.0f}% "
                      f"N:{decomp['normal_pct']:.0f}% "
                      f"B:{decomp['binormal_pct']:.0f}%)")

        all_results.append(stage_results)

    # Save results
    (output_dir / "match_results.json").write_text(
        json.dumps(all_results, indent=2))

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    all_decomps = []
    for sr in all_results:
        for lr in sr["leaves"]:
            all_decomps.append(lr)

    if all_decomps:
        print(f"\n{'Leaf':>4} {'Loft':>5} {'OBJ':>5} {'Disp RMS':>9} "
              f"{'Tang%':>6} {'Norm%':>6} {'Bin%':>6} "
              f"{'Tang':>6} {'Norm':>6} {'Bin':>6}")
        print("-" * 75)
        for d in all_decomps:
            print(f"{d['position']:>4} {d['n_lofter_verts']:>5} "
                  f"{d['n_obj_verts']:>5} {d['total_displacement_rms']:>9.2f} "
                  f"{d['tangent_pct']:>6.0f} {d['normal_pct']:>6.0f} "
                  f"{d['binormal_pct']:>6.0f} "
                  f"{d['tangent_rms']:>6.2f} {d['normal_rms']:>6.2f} "
                  f"{d['binormal_rms']:>6.2f}")

        mean_disp = np.mean([d["total_displacement_rms"] for d in all_decomps])
        mean_tang = np.mean([d["tangent_pct"] for d in all_decomps])
        mean_norm = np.mean([d["normal_pct"] for d in all_decomps])
        mean_bin = np.mean([d["binormal_pct"] for d in all_decomps])

        print(f"\n  Mean displacement: {mean_disp:.2f}cm")
        print(f"  Energy split: tangent={mean_tang:.0f}% normal={mean_norm:.0f}% binormal={mean_bin:.0f}%")
        print(f"  → Tangent = skeleton error (CPlantBox)")
        print(f"  → Normal = gutter/wave/fold (lofter out-of-plane)")
        print(f"  → Binormal = width/curl/twist (lofter in-plane)")

    print(f"\n  Output: {output_dir}/")
    print(f"  Per-stage OBJs: stage_XX/leaf??_base.obj, leaf??_corrected.obj, leaf??_reference.obj")


if __name__ == "__main__":
    main()
