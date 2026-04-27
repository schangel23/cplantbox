#!/usr/bin/env python3
"""Fit production lofter deformation params to match a reference, two modes.

OBJ mode (original, default):
  For each leaf in an OBJ export:
  1. Extract skeleton from OBJ
  2. Fit lofter deformation params (gutter, wave, curl, twist, ruffle, fold)
     to minimize bidirectional Chamfer distance to the OBJ mesh
  3. Re-loft with fitted params → measure residual
  4. Residual = what the lofter fundamentally CANNOT do (needs new capabilities)

CP mode (``--target-cps``):
  Single-leaf CP-space L2 fit against one leaf from a canonical
  ``pheno4d_canonical_cps.json``-style file. The NURBS backend of
  ``loft_organs`` is used so that the predicted mesh exposes a canonical
  ``(N_U, N_V, 3)`` CP grid; the objective becomes
  ``cp_l2_loss({0: pred_cps}, {0: target_cps}, [(0, 0)], reduction="mean")``
  and replaces the bidirectional-Chamfer objective used in OBJ mode.

Uses scipy.optimize.minimize (L-BFGS-B) — fast, handles bounds, 6-10 params.

Usage (OBJ mode, backward-compat):
    python3 fit_lofter_params.py /path/to/Maize/export/ --output output/lofter_fit/ -v

Usage (CP mode, Phase 4a sanity check):
    python3 fit_lofter_params.py \\
        --target-cps Resources/Pheno4D/pheno4d_canonical_cps.json \\
        --target-leaf-id M01_0317:3 \\
        --output dart/coupling/experimental/output/cp_fit_smoke/ -v
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

from dart.coupling.geometry.canonical_cp_grid import N_U, N_V
from dart.coupling.geometry.g1_to_g3 import loft_organs
from dart.coupling.experimental.losses.cp_distance import cp_l2_loss
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


def loft_with_params(skeleton, widths, leaf_id, position, param_vec,
                     use_nurbs_backend=False):
    """Run production lofter with specific deformation params.

    Always returns a 3-tuple ``(vertices, indices, cps)``; ``cps`` is only
    populated (as a canonical ``(N_U, N_V, 3)`` float64 grid) when
    ``use_nurbs_backend=True``, otherwise it is ``None``.
    """
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
        mesh = loft_organs(
            [organ], subdivide=True, smooth=False,
            use_nurbs_backend=use_nurbs_backend,
        )
        cps = mesh.organ_cps.get(int(leaf_id)) if use_nurbs_backend else None
        return mesh.vertices, mesh.indices, cps
    except Exception:
        return None, None, None


def chamfer_to_obj(lofter_verts, obj_verts):
    """Bidirectional nearest-neighbor mean distance."""
    if len(lofter_verts) == 0 or len(obj_verts) == 0:
        return 999.0
    tree_obj = KDTree(obj_verts)
    tree_loft = KDTree(lofter_verts)
    d1, _ = tree_obj.query(lofter_verts)
    d2, _ = tree_loft.query(obj_verts)
    return float((d1.mean() + d2.mean()) / 2.0)


# ---------------------------------------------------------------------------
# CP-mode helpers (Phase 4a — single-leaf sanity check)
# ---------------------------------------------------------------------------
def load_cp_target(json_path, leaf_key):
    """Locate one leaf in ``pheno4d_canonical_cps.json`` and return its
    canonical CP grid plus a derived ``(skeleton, widths)`` pair.

    Args:
        json_path: Path to the CP JSON file.
        leaf_key: Leaf identifier of the form ``"{plant_id}_{date}:{label}"``
            (e.g. ``"M01_0317:3"``). ``label`` is the integer rank stored
            per-leaf in the JSON.

    Returns:
        (target_cps, skeleton, widths)
        target_cps: (N_U, N_V, 3) float64.
        skeleton:   (N_U, 3)       float64 — midrib CP column (v=2 of 5).
        widths:     (N_U,)         float64 — ||cps[:, 0] - cps[:, -1]||,
                    i.e. the full edge-to-edge width at each u station.

    Raises:
        KeyError: if the leaf cannot be found.
    """
    with open(json_path) as f:
        data = json.load(f)
    if "scans" not in data:
        raise KeyError(
            f"{json_path} has top-level keys {list(data.keys())}; expected "
            "a pheno4d_canonical_cps-style file with a 'scans' list."
        )

    try:
        scan_prefix, label_str = leaf_key.split(":")
        plant_id, date = scan_prefix.split("_", 1)
        label = int(label_str)
    except (ValueError, AttributeError) as exc:
        raise KeyError(
            f"leaf_key {leaf_key!r} must match format '<plant>_<date>:<label>'"
        ) from exc

    for scan in data["scans"]:
        if scan["plant_id"] != plant_id or scan["date"] != date:
            continue
        for leaf in scan["leaves"]:
            if int(leaf["label"]) != label:
                continue
            cps = np.asarray(leaf["cps_cm"], dtype=np.float64)
            if cps.shape != (N_U, N_V, 3):
                raise ValueError(
                    f"CP shape {cps.shape} in {leaf_key} does not match "
                    f"canonical ({N_U}, {N_V}, 3) — did the canonical "
                    "convention drift?"
                )
            mid_j = N_V // 2
            skeleton = cps[:, mid_j, :].copy()
            widths = np.linalg.norm(cps[:, 0, :] - cps[:, -1, :], axis=1)
            return cps, skeleton, widths
    raise KeyError(f"leaf {leaf_key} not found in {json_path}")


def fit_leaf_cp_mode(target_cps, skeleton, widths, leaf_id, verbose=False):
    """Fit deformation params in CP space against a canonical target.

    Objective is ``cp_l2_loss({0: pred_cps}, {0: target_cps}, [(0, 0)],
    reduction="mean")`` — the per-leaf averaged sum of squared distances
    between matched CPs. No Hungarian step (single leaf, trivial match).

    Returns:
        (best_params_dict, loss_bare, loss_fitted, pred_cps_fitted,
         pred_verts_fitted, pred_indices_fitted)
    """
    target_cps = np.asarray(target_cps, dtype=np.float64)
    target_dict = {0: target_cps}
    match = [(0, 0)]

    # Bare loft (no deformations).
    bare_verts, bare_idx, bare_cps = loft_with_params(
        skeleton, widths, leaf_id, 0,
        [p[3] for p in DEFORM_PARAMS],
        use_nurbs_backend=True,
    )
    if bare_cps is None:
        return None, 999.0, 999.0, None, None, None
    loss_bare = cp_l2_loss({0: bare_cps}, target_dict, match, reduction="mean")

    n_eval = [0]

    def objective(x):
        n_eval[0] += 1
        _, _, cps = loft_with_params(
            skeleton, widths, leaf_id, 0, x, use_nurbs_backend=True,
        )
        if cps is None:
            return 9.999e6
        return cp_l2_loss({0: cps}, target_dict, match, reduction="mean")

    x0 = np.array([p[3] for p in DEFORM_PARAMS])
    bounds = [(p[1], p[2]) for p in DEFORM_PARAMS]

    best_x = x0
    best_loss = loss_bare

    # Same multi-start schedule as OBJ mode — the deformation-space
    # multi-modality story is unchanged.
    result = minimize(objective, x0, method="L-BFGS-B", bounds=bounds,
                      options={"maxiter": 200, "ftol": 1e-4})
    if result.fun < best_loss:
        best_loss = float(result.fun)
        best_x = result.x

    x1 = x0.copy()
    x1[0] = 3.0   # wave_normal_amp
    x1[6] = 0.3   # twist_max
    x1[7] = 2.0   # curl_amp
    result2 = minimize(objective, x1, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": 200, "ftol": 1e-4})
    if result2.fun < best_loss:
        best_loss = float(result2.fun)
        best_x = result2.x

    x2 = x0.copy()
    x2[0] = 8.0   # wave_normal_amp
    x2[6] = 0.5   # twist_max
    x2[7] = 5.0   # curl_amp
    x2[10] = 2.0  # edge_ruffle_amp
    x2[12] = 1.5  # fold_amp
    result3 = minimize(objective, x2, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": 200, "ftol": 1e-4})
    if result3.fun < best_loss:
        best_loss = float(result3.fun)
        best_x = result3.x

    if verbose:
        print(f"    {n_eval[0]} evals, "
              f"bare CP-L2={loss_bare:.4f} cm² → "
              f"fitted CP-L2={best_loss:.4f} cm² "
              f"(ratio {loss_bare / max(best_loss, 1e-9):.2f}×)")

    best_params = {
        name: float(best_x[i]) for i, (name, _, _, _) in enumerate(DEFORM_PARAMS)
    }

    fit_verts, fit_idx, fit_cps = loft_with_params(
        skeleton, widths, leaf_id, 0, best_x, use_nurbs_backend=True,
    )
    return best_params, loss_bare, best_loss, fit_cps, fit_verts, fit_idx


def run_cp_mode(args):
    """Execute the CP-mode Phase 4a sanity check and write outputs."""
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("FIT LOFTER DEFORMATIONS → CANONICAL CP REFERENCE (Phase 4a)")
    print("=" * 70)
    print(f"\n  Target file: {args.target_cps}")
    print(f"  Target leaf: {args.target_leaf_id}")

    target_cps, skeleton, widths = load_cp_target(
        args.target_cps, args.target_leaf_id
    )

    if args.verbose:
        print(f"\n  Derived skeleton: {len(skeleton)} nodes, "
              f"span {skeleton.max(0) - skeleton.min(0)} cm")
        print(f"  Widths: min={widths.min():.2f} cm, "
              f"median={np.median(widths):.2f} cm, "
              f"max={widths.max():.2f} cm")

    best_params, loss_bare, loss_fitted, fit_cps, fit_verts, fit_idx = \
        fit_leaf_cp_mode(
            target_cps, skeleton, widths, leaf_id=1, verbose=args.verbose,
        )
    if best_params is None:
        print("  ERROR: bare loft failed — cannot compute baseline.")
        return

    improvement = (1.0 - loss_fitted / max(loss_bare, 1e-9)) * 100.0

    # Artifacts: fitted params + per-CP residuals + OBJ mesh.
    results = {
        "target_file": str(args.target_cps),
        "target_leaf_id": args.target_leaf_id,
        "loss_bare_cm2": loss_bare,
        "loss_fitted_cm2": loss_fitted,
        "improvement_pct": improvement,
        "ratio_bare_over_fitted": loss_bare / max(loss_fitted, 1e-9),
        "params": best_params,
    }
    if fit_cps is not None:
        per_cp = np.linalg.norm(fit_cps - target_cps, axis=-1)  # (N_U, N_V)
        results["per_cp_rms_cm"] = float(np.sqrt((per_cp ** 2).mean()))
        results["per_cp_max_cm"] = float(per_cp.max())
        results["per_cp_distances_cm"] = per_cp.tolist()
        # Save the reference + fit CPs side-by-side for downstream analysis.
        np.savez(
            output_dir / "cps.npz",
            target=target_cps, fitted=fit_cps, per_cp_distance=per_cp,
        )

    if fit_verts is not None and fit_idx is not None:
        export_obj(
            output_dir / "leaf_fitted.obj",
            fit_verts, fit_idx,
            f"CP-mode fitted - {args.target_leaf_id}",
        )

    (output_dir / "fit_results.json").write_text(
        json.dumps(results, indent=2, sort_keys=True)
    )

    # Summary
    print("\n" + "=" * 70)
    print("CP-MODE SUMMARY")
    print("=" * 70)
    print(f"\n  Bare CP-L2:    {loss_bare:.4f} cm²")
    print(f"  Fitted CP-L2:  {loss_fitted:.4f} cm²")
    print(f"  Improvement:   {improvement:.1f}%  "
          f"(ratio {results['ratio_bare_over_fitted']:.2f}×)")
    if fit_cps is not None:
        print(f"  Per-CP RMS:    {results['per_cp_rms_cm']:.3f} cm")
        print(f"  Per-CP max:    {results['per_cp_max_cm']:.3f} cm")
    print(f"\n  Artifacts: {output_dir}/")


def fit_leaf_deformations(g1, obj_leaf_verts, verbose=False):
    """Fit deformation params for one leaf using L-BFGS-B.

    Returns: (best_params_dict, chamfer_bare, chamfer_fitted, chamfer_residual_profile)
    """
    skeleton = np.array(g1.skeleton)
    widths = np.array(g1.widths)

    # Bare loft (no deformations)
    bare_verts, bare_indices, _ = loft_with_params(
        skeleton, widths, g1.leaf_id, g1.position,
        [p[3] for p in DEFORM_PARAMS])  # defaults = all zeros
    if bare_verts is None:
        return None, 999, 999, None, None

    chamfer_bare = chamfer_to_obj(bare_verts, obj_leaf_verts)

    # Objective: Chamfer distance
    n_eval = [0]

    def objective(x):
        n_eval[0] += 1
        verts, _, _ = loft_with_params(
            skeleton, widths, g1.leaf_id, g1.position, x)
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
    fitted_verts, fitted_indices, _ = loft_with_params(
        skeleton, widths, g1.leaf_id, g1.position, best_x)

    return best_params, chamfer_bare, best_chamfer, fitted_verts, fitted_indices


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Fit production lofter deformations to a reference. Default "
            "mode fits bidirectional Chamfer against an OBJ export; pass "
            "--target-cps to run Phase 4a single-leaf CP-space L2 fitting "
            "against a pheno4d_canonical_cps.json-style file instead."
        )
    )
    parser.add_argument("export_dir", nargs="?", default=None,
                        help="Directory with maize_stage_*.obj "
                             "(OBJ mode; ignored in CP mode).")
    parser.add_argument("--output", "-o", default="output/lofter_fit")
    parser.add_argument("--stage", type=int, default=-1,
                        help="Stage index (default: -1 = most mature). "
                             "OBJ mode only.")
    # --- CP mode (Phase 4a) ---
    parser.add_argument("--target-cps", default=None,
                        help="Path to a pheno4d_canonical_cps.json-style file. "
                             "Toggles CP-space L2 fitting against one leaf.")
    parser.add_argument("--target-leaf-id", default=None,
                        help="Leaf identifier for CP mode, format "
                             "'{plant_id}_{date}:{label}' (e.g. M01_0317:3).")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    # Dispatch: CP mode if both CP flags are provided.
    if args.target_cps is not None or args.target_leaf_id is not None:
        if args.target_cps is None or args.target_leaf_id is None:
            parser.error("--target-cps and --target-leaf-id must be given "
                         "together to enable CP mode.")
        run_cp_mode(args)
        return

    if args.export_dir is None:
        parser.error("export_dir is required in OBJ mode.")

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
