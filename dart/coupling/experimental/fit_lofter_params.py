#!/usr/bin/env python3
"""Fit production lofter deformation params to match OBJ mesh, measure residual.

For each leaf:
1. Extract skeleton from OBJ
2. Fit lofter deformation params (gutter, wave, curl, twist, ruffle, fold)
   to minimize displacement to OBJ mesh
3. Re-loft with fitted params → measure residual
4. Residual = what the lofter fundamentally CANNOT do (needs new capabilities)

Uses scipy.optimize.minimize (L-BFGS-B) — fast, handles bounds, 6-10 params.

Usage:
    python3 fit_lofter_params.py /path/to/Maize/export/ --output output/lofter_fit/ -v
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.spatial import KDTree

_CPLANTBOX_ROOT = str(Path(__file__).resolve().parent.parent.parent.parent)
sys.path.insert(0, _CPLANTBOX_ROOT)

from dart.coupling.geometry.g1_to_g3 import loft_organs
from dart.coupling.experimental.reverse_engineer_maize import (
    parse_obj, find_connected_components, track_leaves_across_stages,
    extract_leaf_g1,
)
from dart.coupling.experimental.match_vertices import (
    load_all_stages, export_obj, decompose_displacement_field,
)

# Deformation parameters to optimize: (name, low, high, default)
DEFORM_PARAMS = [
    ("wave_normal_amp",   0.0, 15.0, 0.0),
    ("wave_normal_freq",  0.5,  8.0, 2.5),
    ("wave_normal_phase", -3.14, 3.14, 0.0),
    ("wave_lateral_amp",  0.0,  5.0, 0.0),
    ("wave_lateral_freq", 0.5,  5.0, 1.5),
    ("wave_lateral_phase",-3.14, 3.14, 0.0),
    ("twist_max",         -1.0, 1.0, 0.0),  # radians
    ("curl_amp",          0.0, 10.0, 0.0),
    ("curl_freq",         0.5,  4.0, 1.0),
    ("curl_phase",        -3.14, 3.14, 0.0),
    ("edge_ruffle_amp",   0.0,  5.0, 0.0),
    ("edge_ruffle_freq",  2.0, 10.0, 7.0),
    ("fold_amp",          0.0,  5.0, 0.0),
    ("fold_freq",         1.0,  5.0, 2.0),
]

PARAM_NAMES = [p[0] for p in DEFORM_PARAMS]


def loft_with_params(skeleton, widths, leaf_id, position, param_vec):
    """Run production lofter with specific deformation params."""
    params = {}
    for i, (name, lo, hi, _) in enumerate(DEFORM_PARAMS):
        params[name] = float(np.clip(param_vec[i], lo, hi))

    organ = {
        "type": "leaf",
        "skeleton": skeleton,
        "widths": widths,
        "organ_id": leaf_id,
        "name": f"leaf_{position}",
        "node_ids": list(range(len(skeleton))),
        **params,
    }

    try:
        mesh = loft_organs([organ], subdivide=True, smooth=False)
        return mesh.vertices, mesh.indices
    except Exception:
        return None, None


def chamfer_to_obj(lofter_verts, obj_verts):
    """Bidirectional nearest-neighbor mean distance."""
    if len(lofter_verts) == 0 or len(obj_verts) == 0:
        return 999.0
    tree_obj = KDTree(obj_verts)
    tree_loft = KDTree(lofter_verts)
    d1, _ = tree_obj.query(lofter_verts)
    d2, _ = tree_loft.query(obj_verts)
    return float((d1.mean() + d2.mean()) / 2.0)


def fit_leaf_deformations(g1, obj_leaf_verts, verbose=False):
    """Fit deformation params for one leaf using L-BFGS-B.

    Returns: (best_params_dict, chamfer_bare, chamfer_fitted, chamfer_residual_profile)
    """
    skeleton = np.array(g1.skeleton)
    widths = np.array(g1.widths)

    # Bare loft (no deformations)
    bare_verts, bare_indices = loft_with_params(
        skeleton, widths, g1.leaf_id, g1.position,
        [p[3] for p in DEFORM_PARAMS])  # defaults = all zeros
    if bare_verts is None:
        return None, 999, 999, None

    chamfer_bare = chamfer_to_obj(bare_verts, obj_leaf_verts)

    # Objective: Chamfer distance
    n_eval = [0]

    def objective(x):
        n_eval[0] += 1
        verts, _ = loft_with_params(skeleton, widths, g1.leaf_id, g1.position, x)
        if verts is None:
            return 999.0
        return chamfer_to_obj(verts, obj_leaf_verts)

    # Initial guess: all defaults
    x0 = np.array([p[3] for p in DEFORM_PARAMS])
    bounds = [(p[1], p[2]) for p in DEFORM_PARAMS]

    # Multi-start: try a few initial conditions
    best_x = x0
    best_chamfer = chamfer_bare

    # Start 1: defaults
    result = minimize(objective, x0, method="L-BFGS-B", bounds=bounds,
                      options={"maxiter": 200, "ftol": 1e-4})
    if result.fun < best_chamfer:
        best_chamfer = result.fun
        best_x = result.x

    # Start 2: moderate deformations
    x1 = x0.copy()
    x1[0] = 3.0   # wave_normal_amp
    x1[6] = 0.3   # twist_max
    x1[7] = 2.0   # curl_amp
    result2 = minimize(objective, x1, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": 200, "ftol": 1e-4})
    if result2.fun < best_chamfer:
        best_chamfer = result2.fun
        best_x = result2.x

    # Start 3: strong deformations
    x2 = x0.copy()
    x2[0] = 8.0   # wave_normal_amp
    x2[6] = 0.5   # twist_max
    x2[7] = 5.0   # curl_amp
    x2[10] = 2.0  # edge_ruffle_amp
    x2[12] = 1.5  # fold_amp
    result3 = minimize(objective, x2, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": 200, "ftol": 1e-4})
    if result3.fun < best_chamfer:
        best_chamfer = result3.fun
        best_x = result3.x

    if verbose:
        print(f"    {n_eval[0]} evals, bare={chamfer_bare:.2f} → fitted={best_chamfer:.2f}cm")

    # Build params dict
    best_params = {}
    for i, (name, _, _, _) in enumerate(DEFORM_PARAMS):
        best_params[name] = float(best_x[i])

    # Get fitted mesh for residual analysis
    fitted_verts, fitted_indices = loft_with_params(
        skeleton, widths, g1.leaf_id, g1.position, best_x)

    return best_params, chamfer_bare, best_chamfer, fitted_verts, fitted_indices


def main():
    parser = argparse.ArgumentParser(
        description="Fit production lofter deformations to OBJ reference")
    parser.add_argument("export_dir", help="Directory with maize_stage_*.obj")
    parser.add_argument("--output", "-o", default="output/lofter_fit")
    parser.add_argument("--stage", type=int, default=-1,
                        help="Stage index (default: -1 = most mature)")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("FIT PRODUCTION LOFTER DEFORMATIONS → OBJ REFERENCE")
    print("=" * 70)

    # Load
    files, all_verts, all_groups, all_comps, canonical, pos_map = \
        load_all_stages(args.export_dir)

    sidx = args.stage if args.stage >= 0 else len(files) + args.stage
    stage_num = files[sidx][0]
    verts = all_verts[sidx]
    groups = all_groups[sidx]

    leaf_faces = []
    for gn, gf in groups.items():
        if "leaf" in gn.lower():
            leaf_faces.extend(gf)

    print(f"\n  Stage {stage_num}")

    results = []
    objs_dir = output_dir / f"stage_{stage_num:02d}"
    objs_dir.mkdir(exist_ok=True)

    for lid, stage_map in sorted(canonical.items(), key=lambda x: pos_map[x[0]]):
        if sidx not in stage_map:
            continue
        pos = pos_map[lid]
        comp = stage_map[sidx]
        g1 = extract_leaf_g1(comp, verts, leaf_faces, lid, pos, n_samples=20)
        if g1.length < 3:
            continue

        comp_ids = sorted(comp)
        obj_leaf_verts = verts[comp_ids]

        if args.verbose:
            print(f"\n  Leaf {pos} ({g1.length:.0f}cm, {len(obj_leaf_verts)} OBJ verts):")

        best_params, bare, fitted, fitted_verts, fitted_indices = \
            fit_leaf_deformations(g1, obj_leaf_verts, args.verbose)

        if best_params is None:
            continue

        # Residual decomposition
        skeleton = np.array(g1.skeleton)
        widths_arr = np.array(g1.widths)
        if fitted_verts is not None:
            obj_tree = KDTree(obj_leaf_verts)
            dists, match_idx = obj_tree.query(fitted_verts)
            residual_displacements = obj_leaf_verts[match_idx] - fitted_verts
            residual_decomp = decompose_displacement_field(
                residual_displacements, fitted_verts, skeleton, widths_arr)
        else:
            residual_decomp = {}

        # Export
        if fitted_verts is not None and fitted_indices is not None:
            export_obj(objs_dir / f"leaf{pos:02d}_fitted.obj",
                       fitted_verts, fitted_indices,
                       f"Fitted lofter - leaf {pos}")
            # Also corrected (1:1 via displacement)
            corrected = fitted_verts + residual_displacements
            export_obj(objs_dir / f"leaf{pos:02d}_corrected.obj",
                       corrected, fitted_indices,
                       f"Corrected (residual applied) - leaf {pos}")

        leaf_result = {
            "position": pos,
            "chamfer_bare": bare,
            "chamfer_fitted": fitted,
            "improvement_pct": (1 - fitted / max(bare, 0.01)) * 100,
            "params": best_params,
            "residual": residual_decomp,
        }
        results.append(leaf_result)

        if args.verbose:
            imp = leaf_result["improvement_pct"]
            print(f"    Bare: {bare:.2f}cm → Fitted: {fitted:.2f}cm "
                  f"({imp:.0f}% improvement)")
            # Show which params are active
            active = [(k, v) for k, v in best_params.items()
                      if abs(v) > 0.01 and "phase" not in k and "freq" not in k]
            if active:
                print(f"    Active: {', '.join(f'{k}={v:.2f}' for k, v in active)}")
            if residual_decomp:
                print(f"    Residual: {residual_decomp.get('total_displacement_rms', 0):.2f}cm "
                      f"(T:{residual_decomp.get('tangent_pct', 0):.0f}% "
                      f"N:{residual_decomp.get('normal_pct', 0):.0f}% "
                      f"B:{residual_decomp.get('binormal_pct', 0):.0f}%)")

    # Save
    (output_dir / "fit_results.json").write_text(json.dumps(results, indent=2))

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    if results:
        print(f"\n{'Leaf':>4} {'Bare':>7} {'Fitted':>8} {'Improv':>7} "
              f"{'Residual':>9} {'Active deformations'}")
        print("-" * 80)
        for r in results:
            active = [(k, v) for k, v in r["params"].items()
                      if abs(v) > 0.01 and "phase" not in k and "freq" not in k]
            active_str = ", ".join(f"{k}={v:.1f}" for k, v in active[:4])
            resid = r["residual"].get("total_displacement_rms", 0)
            print(f"{r['position']:>4} {r['chamfer_bare']:>7.2f} "
                  f"{r['chamfer_fitted']:>8.2f} {r['improvement_pct']:>6.0f}% "
                  f"{resid:>9.2f}  {active_str}")

        mean_bare = np.mean([r["chamfer_bare"] for r in results])
        mean_fitted = np.mean([r["chamfer_fitted"] for r in results])
        mean_resid = np.mean([r["residual"].get("total_displacement_rms", 0)
                              for r in results])
        improvement = (1 - mean_fitted / max(mean_bare, 0.01)) * 100

        print(f"\n  Mean bare:     {mean_bare:.2f}cm")
        print(f"  Mean fitted:   {mean_fitted:.2f}cm ({improvement:.0f}% improvement)")
        print(f"  Mean residual: {mean_resid:.2f}cm ← THIS IS THE LOFTER'S STRUCTURAL LIMIT")

        if mean_resid > 1.0:
            print(f"\n  ⚠ Residual {mean_resid:.1f}cm > 1cm = lofter needs NEW capabilities:")
            # Analyze what residual is made of
            all_resid_norm = [r["residual"].get("normal_pct", 0) for r in results]
            all_resid_bin = [r["residual"].get("binormal_pct", 0) for r in results]
            all_resid_tang = [r["residual"].get("tangent_pct", 0) for r in results]
            print(f"    Normal (out-of-plane):  {np.mean(all_resid_norm):.0f}% "
                  f"→ need per-node gutter/cross-section (not global)")
            print(f"    Binormal (across width): {np.mean(all_resid_bin):.0f}% "
                  f"→ need per-node width correction / asymmetry")
            print(f"    Tangent (along leaf):    {np.mean(all_resid_tang):.0f}% "
                  f"→ skeleton extraction accuracy")
        else:
            print(f"\n  ✓ Residual < 1cm = existing lofter capabilities are sufficient!")

    print(f"\n  Output: {output_dir}/")


if __name__ == "__main__":
    main()
