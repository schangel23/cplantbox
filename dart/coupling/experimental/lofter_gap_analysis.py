#!/usr/bin/env python3
"""Lofter gap analysis: compute exact per-vertex displacement from bare lofter to OBJ mesh.

Takes extracted skeletons (from reverse_engineer_maize.py), runs them through
the production lofter with no deformations, and compares against the original
OBJ mesh vertex-by-vertex via nearest-neighbor matching.

Decomposes displacements into the lofter's local frame:
  - tangent component → skeleton position error (length/curvature)
  - normal component → gutter/wave_normal/fold
  - binormal component → width/curl/wave_lateral/twist

Outputs per-leaf displacement maps + summary of what deformations are needed.

Usage:
    python3 lofter_gap_analysis.py /path/to/Maize/export/ \
        --output output/lofter_gaps/ -v
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import KDTree

# CPlantBox root for dart.coupling imports
_CPLANTBOX_ROOT = str(Path(__file__).resolve().parent.parent.parent.parent)
sys.path.insert(0, _CPLANTBOX_ROOT)

from dart.coupling.geometry.g1_to_g3 import loft_organs
from dart.coupling.experimental.reverse_engineer_maize import (
    parse_obj, find_connected_components, track_leaves_across_stages,
    extract_leaf_g1, count_developed_leaves, vstage_to_day,
)


def load_stage(export_dir, stage_idx=-1):
    """Load one OBJ stage and extract per-leaf data.

    Returns:
        verts: (N, 3) all vertices (cm, Z-flipped)
        leaf_components: dict[position -> set of vertex IDs]
        leaf_g1s: dict[position -> LeafG1]
        leaf_faces: list of face lists
    """
    export_dir = Path(export_dir)
    manifest = json.loads((export_dir / "manifest.json").read_text())
    files = [(s["stage"], export_dir / s["file"]) for s in manifest["stages"]]

    # Parse all for tracking, but only analyze one stage
    all_verts, all_groups, all_comps = [], [], []
    for _, fpath in files:
        verts, groups = parse_obj(fpath)
        verts[:, 2] *= -1
        all_verts.append(verts)
        all_groups.append(groups)
        lf = []
        for gn, gf in groups.items():
            if "leaf" in gn.lower():
                lf.extend(gf)
        all_comps.append(find_connected_components(lf))

    canonical = track_leaves_across_stages(all_comps)
    def sort_key(lid):
        for si in range(len(all_verts)):
            if si in canonical[lid]:
                return np.mean(all_verts[si][list(canonical[lid][si])][:, 2])
        return 0
    sorted_lids = sorted(canonical.keys(), key=sort_key)
    pos_map = {lid: p + 1 for p, lid in enumerate(sorted_lids)}

    # Target stage
    idx = stage_idx if stage_idx >= 0 else len(files) + stage_idx
    verts = all_verts[idx]
    groups = all_groups[idx]

    leaf_faces = []
    for gn, gf in groups.items():
        if "leaf" in gn.lower():
            leaf_faces.extend(gf)

    leaf_components = {}
    leaf_g1s = {}
    for lid, stage_map in canonical.items():
        if idx not in stage_map:
            continue
        pos = pos_map[lid]
        comp = stage_map[idx]
        leaf_components[pos] = comp
        g1 = extract_leaf_g1(comp, verts, leaf_faces, lid, pos, n_samples=20)
        if g1.length > 3:
            leaf_g1s[pos] = g1

    return verts, leaf_components, leaf_g1s, leaf_faces


def bare_loft_leaf(g1, n_cross=7):
    """Run production lofter on one leaf skeleton with NO deformations.

    Returns lofted vertices or None.
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
        return mesh.vertices
    except Exception as e:
        print(f"  Loft failed for leaf {g1.position}: {e}")
        return None


def compute_local_frame(skeleton):
    """Compute tangent, normal, binormal at each skeleton point."""
    n = len(skeleton)
    tangents = np.zeros((n, 3))
    tangents[0] = skeleton[1] - skeleton[0]
    tangents[-1] = skeleton[-1] - skeleton[-2]
    for i in range(1, n - 1):
        tangents[i] = skeleton[i + 1] - skeleton[i - 1]
    norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    tangents = tangents / norms

    up = np.array([0, 0, 1.0])
    binormals = np.cross(tangents, up)
    bn_norms = np.linalg.norm(binormals, axis=1, keepdims=True)
    bn_norms = np.maximum(bn_norms, 1e-8)
    binormals = binormals / bn_norms

    normals = np.cross(binormals, tangents)
    nn = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.maximum(nn, 1e-8)

    return tangents, normals, binormals


def analyze_leaf_displacements(g1, lofter_verts, obj_verts, obj_faces, comp):
    """Compute per-vertex displacements and decompose into local frame.

    Returns dict with displacement statistics.
    """
    skeleton = np.array(g1.skeleton)
    tangents, normals, binormals = compute_local_frame(skeleton)

    # OBJ vertices for this leaf
    comp_ids = sorted(comp)
    obj_leaf_verts = obj_verts[comp_ids]

    # Nearest-neighbor matching: for each OBJ vert, find closest lofter vert
    lofter_tree = KDTree(lofter_verts)
    dists_obj_to_loft, idx_obj_to_loft = lofter_tree.query(obj_leaf_verts)

    # For each lofter vert, find closest OBJ vert
    obj_tree = KDTree(obj_leaf_verts)
    dists_loft_to_obj, idx_loft_to_obj = obj_tree.query(lofter_verts)

    # Displacement vectors: OBJ - nearest_lofter
    displacements = obj_leaf_verts - lofter_verts[idx_obj_to_loft]

    # Find which skeleton point each OBJ vertex is closest to
    skel_tree = KDTree(skeleton)
    _, skel_idx = skel_tree.query(obj_leaf_verts)

    # Decompose displacements into local frame at nearest skeleton point
    tangent_comp = np.zeros(len(displacements))
    normal_comp = np.zeros(len(displacements))
    binormal_comp = np.zeros(len(displacements))

    for i, disp in enumerate(displacements):
        si = skel_idx[i]
        tangent_comp[i] = np.dot(disp, tangents[si])
        normal_comp[i] = np.dot(disp, normals[si])
        binormal_comp[i] = np.dot(disp, binormals[si])

    # Compute arc-length fraction for each OBJ vertex (for spatial profile)
    seg_lens = np.linalg.norm(np.diff(skeleton, axis=0), axis=1)
    cum_arc = np.concatenate([[0], np.cumsum(seg_lens)])
    total_arc = cum_arc[-1]
    if total_arc > 0:
        arc_fracs = cum_arc[skel_idx] / total_arc
    else:
        arc_fracs = np.zeros(len(skel_idx))

    # Compute cross-section fraction for each OBJ vertex
    # (signed distance from skeleton midline in binormal direction)
    widths = np.array(g1.widths)
    cross_fracs = np.zeros(len(displacements))
    for i in range(len(obj_leaf_verts)):
        si = skel_idx[i]
        to_vert = obj_leaf_verts[i] - skeleton[si]
        bn_dist = np.dot(to_vert, binormals[si])
        w = widths[si] if widths[si] > 0.01 else 1.0
        cross_fracs[i] = bn_dist / (w * 0.5)  # normalized to [-1, 1]

    # Summary statistics
    chamfer = float((dists_obj_to_loft.mean() + dists_loft_to_obj.mean()) / 2)
    total_disp_magnitude = float(np.linalg.norm(displacements, axis=1).mean())

    # Profile along leaf: bin by arc fraction
    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)
    tangent_profile = np.zeros(n_bins)
    normal_profile = np.zeros(n_bins)
    binormal_profile = np.zeros(n_bins)
    disp_magnitude_profile = np.zeros(n_bins)

    for b in range(n_bins):
        mask = (arc_fracs >= bin_edges[b]) & (arc_fracs < bin_edges[b + 1])
        if mask.sum() > 0:
            tangent_profile[b] = np.mean(np.abs(tangent_comp[mask]))
            normal_profile[b] = np.mean(np.abs(normal_comp[mask]))
            binormal_profile[b] = np.mean(np.abs(binormal_comp[mask]))
            disp_magnitude_profile[b] = np.mean(
                np.linalg.norm(displacements[mask], axis=1))

    # Interpret: what deformation types are needed?
    interpretations = []

    # Normal component = gutter/wave/fold
    mean_normal = float(np.abs(normal_comp).mean())
    if mean_normal > 0.3:
        # Check if it's consistent (gutter) or oscillating (wave)
        normal_std = float(np.std(normal_comp))
        if normal_std < mean_normal * 0.5:
            interpretations.append({
                "type": "gutter_depth",
                "magnitude_cm": mean_normal,
                "description": f"Consistent normal displacement {mean_normal:.1f}cm → leaf is curved (U/V shape)",
                "lofter_param": "gutter_depth",
                "profile": normal_profile.tolist(),
            })
        else:
            interpretations.append({
                "type": "wave_normal",
                "magnitude_cm": mean_normal,
                "description": f"Oscillating normal displacement {mean_normal:.1f}cm → vertical undulation",
                "lofter_param": "wave_normal_amp",
                "profile": normal_profile.tolist(),
            })

    # Binormal component = width error / curl / lateral wave
    mean_binormal = float(np.abs(binormal_comp).mean())
    if mean_binormal > 0.3:
        # Check if symmetric (width error) or asymmetric (curl)
        # Split by cross_frac sign
        left = binormal_comp[cross_fracs < -0.1]
        right = binormal_comp[cross_fracs > 0.1]
        if len(left) > 0 and len(right) > 0:
            asym = abs(left.mean() + right.mean())  # symmetric if ~0
            if asym > mean_binormal * 0.3:
                interpretations.append({
                    "type": "curl",
                    "magnitude_cm": mean_binormal,
                    "description": f"Asymmetric binormal {mean_binormal:.1f}cm → leaf curl",
                    "lofter_param": "curl_amp",
                    "profile": binormal_profile.tolist(),
                })
            else:
                interpretations.append({
                    "type": "width_error",
                    "magnitude_cm": mean_binormal,
                    "description": f"Symmetric binormal {mean_binormal:.1f}cm → width profile mismatch",
                    "lofter_param": "widths",
                    "profile": binormal_profile.tolist(),
                })

    # Tangent component = skeleton position error
    mean_tangent = float(np.abs(tangent_comp).mean())
    if mean_tangent > 0.5:
        interpretations.append({
            "type": "skeleton_shift",
            "magnitude_cm": mean_tangent,
            "description": f"Tangent displacement {mean_tangent:.1f}cm → skeleton length/curvature mismatch",
            "lofter_param": None,
            "note": "This is a CPlantBox skeleton issue, not lofter",
            "profile": tangent_profile.tolist(),
        })

    # Check for twist: does binormal_comp change sign across the leaf width?
    # At each arc position, compare left vs right binormal displacement
    twist_signal = []
    for b in range(n_bins):
        mask = (arc_fracs >= bin_edges[b]) & (arc_fracs < bin_edges[b + 1])
        if mask.sum() > 5:
            left_mask = mask & (cross_fracs < -0.2)
            right_mask = mask & (cross_fracs > 0.2)
            if left_mask.sum() > 0 and right_mask.sum() > 0:
                left_n = normal_comp[left_mask].mean()
                right_n = normal_comp[right_mask].mean()
                twist_signal.append(left_n - right_n)
    if twist_signal:
        twist_range = max(twist_signal) - min(twist_signal)
        if twist_range > 0.5:
            interpretations.append({
                "type": "twist",
                "magnitude_cm": float(twist_range),
                "description": f"Left/right normal difference varies {twist_range:.1f}cm along leaf → twist",
                "lofter_param": "twist_max",
                "profile": twist_signal,
            })

    return {
        "position": g1.position,
        "chamfer": chamfer,
        "total_displacement": total_disp_magnitude,
        "mean_tangent": mean_tangent,
        "mean_normal": mean_normal,
        "mean_binormal": mean_binormal,
        "n_obj_verts": len(obj_leaf_verts),
        "n_lofter_verts": len(lofter_verts),
        "tangent_profile": tangent_profile.tolist(),
        "normal_profile": normal_profile.tolist(),
        "binormal_profile": binormal_profile.tolist(),
        "displacement_profile": disp_magnitude_profile.tolist(),
        "interpretations": interpretations,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Lofter gap analysis: displacement decomposition")
    parser.add_argument("export_dir", help="Directory with maize_stage_*.obj")
    parser.add_argument("--output", "-o", default="output/lofter_gaps")
    parser.add_argument("--stage", type=int, default=-1,
                        help="Stage index to analyze (default: -1 = most mature)")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("LOFTER GAP ANALYSIS — Displacement Decomposition")
    print("=" * 70)

    # Load reference
    print("\n[1/3] Loading reference...")
    verts, leaf_comps, leaf_g1s, leaf_faces = load_stage(
        args.export_dir, args.stage)
    print(f"  {len(leaf_g1s)} developed leaves")

    # Analyze each leaf
    print("\n[2/3] Computing displacements per leaf...")
    results = []
    for pos in sorted(leaf_g1s.keys()):
        g1 = leaf_g1s[pos]
        comp = leaf_comps[pos]

        if args.verbose:
            print(f"\n  Leaf {pos}: {g1.length:.0f}cm, {len(comp)} OBJ verts")

        # Bare loft
        lofter_verts = bare_loft_leaf(g1)
        if lofter_verts is None:
            continue

        if args.verbose:
            print(f"    Lofter: {len(lofter_verts)} verts")

        # Displacement analysis
        result = analyze_leaf_displacements(
            g1, lofter_verts, verts, leaf_faces, comp)
        results.append(result)

        if args.verbose:
            print(f"    Chamfer: {result['chamfer']:.2f}cm")
            print(f"    Displacement: tangent={result['mean_tangent']:.2f}, "
                  f"normal={result['mean_normal']:.2f}, "
                  f"binormal={result['mean_binormal']:.2f}cm")
            for interp in result["interpretations"]:
                print(f"    → {interp['type']}: {interp['magnitude_cm']:.1f}cm "
                      f"({interp['description'][:80]})")

    # Summary
    print("\n[3/3] Summary")
    print("=" * 70)
    (output_dir / "displacement_results.json").write_text(
        json.dumps(results, indent=2))

    print(f"\n{'Leaf':>4} {'Chamfer':>8} {'Total':>7} {'Tangent':>8} "
          f"{'Normal':>8} {'Binormal':>9} {'Interpretations'}")
    print("-" * 80)
    for r in results:
        interp_str = ", ".join(f"{i['type']}({i['magnitude_cm']:.1f})"
                               for i in r["interpretations"])
        print(f"{r['position']:>4} {r['chamfer']:>8.2f} {r['total_displacement']:>7.2f} "
              f"{r['mean_tangent']:>8.2f} {r['mean_normal']:>8.2f} "
              f"{r['mean_binormal']:>9.2f}  {interp_str}")

    # Aggregate: what deformations are most needed?
    all_interps = {}
    for r in results:
        for interp in r["interpretations"]:
            t = interp["type"]
            if t not in all_interps:
                all_interps[t] = {"count": 0, "total_mag": 0, "leaves": []}
            all_interps[t]["count"] += 1
            all_interps[t]["total_mag"] += interp["magnitude_cm"]
            all_interps[t]["leaves"].append(r["position"])

    print(f"\n{'Deformation':>20} {'Leaves':>7} {'Mean Mag':>9} {'Positions'}")
    print("-" * 65)
    for t, info in sorted(all_interps.items(), key=lambda x: -x[1]["total_mag"]):
        mean_mag = info["total_mag"] / info["count"]
        positions = ",".join(str(p) for p in info["leaves"][:8])
        print(f"{t:>20} {info['count']:>7} {mean_mag:>9.2f}cm  {positions}")

    if results:
        mean_chamfer = np.mean([r["chamfer"] for r in results])
        print(f"\n  Overall mean Chamfer: {mean_chamfer:.2f}cm")
        print(f"  Output: {output_dir}/displacement_results.json")


if __name__ == "__main__":
    main()
