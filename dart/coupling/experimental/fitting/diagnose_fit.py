"""Diagnostic for hybrid fitting results: per-leaf Chamfer, spatial error maps.

Usage:
    python -m dart.coupling.experimental.fitting.diagnose_fit \
        /path/to/hybrid_spline_0001.json \
        /path/to/0001.stl
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

N_POSITIONS = 11


def _load_result(json_path):
    with open(json_path) as f:
        return json.load(f)


def _reconstruct_plant(result, day=60, template_xml=None):
    """Grow CPlantBox with the best XML params, return leaf organs."""
    from .hybrid_optimizer import _grow_and_extract
    xml_params = np.array(result['xml_params'])
    return _grow_and_extract(xml_params, day=day, template_xml=template_xml)


def _loft_with_deformations(leaf_organs, result):
    """Apply best deformation + width profile to each leaf, return per-leaf meshes."""
    import torch
    from ..diff_lofter.deformations import compute_deformations_spline, _interp_linear
    from ..diff_lofter.frames import compute_tangents, compute_binormal_field
    from ..diff_lofter.lofter import compute_arc_fracs, loft_leaf

    deform_params = result['deform_params']
    per_leaf_verts = []

    for i, organ in enumerate(leaf_organs):
        skeleton = torch.tensor(organ['skeleton'], dtype=torch.float32)
        widths = torch.tensor(organ['widths'], dtype=torch.float32)

        if skeleton.shape[0] < 3:
            per_leaf_verts.append(np.zeros((0, 3)))
            continue

        tangents = compute_tangents(skeleton)
        binormals = compute_binormal_field(skeleton, tangents)
        arc_fracs = compute_arc_fracs(skeleton)

        key = str(i)
        if key in deform_params:
            dp = deform_params[key]
            cp = {name: torch.tensor(vals, dtype=torch.float32)
                  for name, vals in dp['control_points'].items()}
            deforms = compute_deformations_spline(arc_fracs, cp)

            wp = torch.tensor(dp['width_profile'], dtype=torch.float32)
            w_mult = _interp_linear(arc_fracs, wp)
            widths = widths * w_mult
        else:
            from ..diff_lofter.deformations import make_spline_control_points
            cp = make_spline_control_points(requires_grad=False)
            deforms = compute_deformations_spline(arc_fracs, cp)

        verts = loft_leaf(skeleton, widths, deforms, tangents, binormals, n_cross=7)
        per_leaf_verts.append(verts.detach().numpy())

    return per_leaf_verts


def _loft_skeleton_only(leaf_organs):
    """Loft with zero deformations and unmodified widths — skeleton baseline."""
    import torch
    from ..diff_lofter.deformations import compute_deformations_spline, make_spline_control_points
    from ..diff_lofter.frames import compute_tangents, compute_binormal_field
    from ..diff_lofter.lofter import compute_arc_fracs, loft_leaf

    per_leaf_verts = []
    for organ in leaf_organs:
        skeleton = torch.tensor(organ['skeleton'], dtype=torch.float32)
        widths = torch.tensor(organ['widths'], dtype=torch.float32)

        if skeleton.shape[0] < 3:
            per_leaf_verts.append(np.zeros((0, 3)))
            continue

        tangents = compute_tangents(skeleton)
        binormals = compute_binormal_field(skeleton, tangents)
        arc_fracs = compute_arc_fracs(skeleton)

        cp = make_spline_control_points(requires_grad=False)
        deforms = compute_deformations_spline(arc_fracs, cp)
        verts = loft_leaf(skeleton, widths, deforms, tangents, binormals, n_cross=7)
        per_leaf_verts.append(verts.detach().numpy())

    return per_leaf_verts


def _per_leaf_chamfer(per_leaf_verts, target_pts):
    """Compute per-leaf Chamfer distance to target."""
    tree_target = cKDTree(target_pts)
    results = []
    for i, verts in enumerate(per_leaf_verts):
        if len(verts) == 0:
            results.append({'leaf': i, 'chamfer': float('nan'), 'n_verts': 0})
            continue
        d_gen_to_target, _ = tree_target.query(verts)
        tree_gen = cKDTree(verts)
        d_target_to_gen, _ = tree_gen.query(target_pts)
        chamfer = (d_gen_to_target.mean() + d_target_to_gen.mean()) / 2
        results.append({
            'leaf': i,
            'chamfer': float(chamfer),
            'mean_gen_to_target': float(d_gen_to_target.mean()),
            'max_gen_to_target': float(d_gen_to_target.max()),
            'p90_gen_to_target': float(np.percentile(d_gen_to_target, 90)),
            'n_verts': len(verts),
        })
    return results


def _spatial_error_by_height(per_leaf_verts, target_pts, n_bins=5):
    """Break down error by Z-height bins."""
    all_verts = np.concatenate([v for v in per_leaf_verts if len(v) > 0])
    tree_target = cKDTree(target_pts)
    dists, _ = tree_target.query(all_verts)

    z = all_verts[:, 2]
    z_min, z_max = z.min(), z.max()
    bins = np.linspace(z_min, z_max, n_bins + 1)
    results = []
    for j in range(n_bins):
        mask = (z >= bins[j]) & (z < bins[j + 1])
        if mask.sum() == 0:
            continue
        results.append({
            'z_range': f'{bins[j]:.1f}-{bins[j+1]:.1f} cm',
            'mean_dist': float(dists[mask].mean()),
            'p90_dist': float(np.percentile(dists[mask], 90)),
            'n_verts': int(mask.sum()),
        })
    return results


def _spatial_error_by_arc(per_leaf_verts, leaf_organs, target_pts):
    """Break down error by arc-length position (base/mid/tip)."""
    import torch
    from ..diff_lofter.lofter import compute_arc_fracs

    tree_target = cKDTree(target_pts)
    zones = {'base (0-0.33)': [], 'mid (0.33-0.66)': [], 'tip (0.66-1.0)': []}

    for i, (verts, organ) in enumerate(zip(per_leaf_verts, leaf_organs)):
        if len(verts) == 0:
            continue
        skeleton = torch.tensor(organ['skeleton'], dtype=torch.float32)
        arc_fracs = compute_arc_fracs(skeleton).numpy()
        n_cross = 7
        # Each skeleton node produces n_cross vertices
        arc_per_vert = np.repeat(arc_fracs, n_cross)
        if len(arc_per_vert) != len(verts):
            continue

        dists, _ = tree_target.query(verts)

        for label, lo, hi in [('base (0-0.33)', 0, 0.33),
                               ('mid (0.33-0.66)', 0.33, 0.66),
                               ('tip (0.66-1.0)', 0.66, 1.01)]:
            mask = (arc_per_vert >= lo) & (arc_per_vert < hi)
            if mask.sum() > 0:
                zones[label].extend(dists[mask].tolist())

    return {label: {
        'mean': float(np.mean(vals)) if vals else float('nan'),
        'p90': float(np.percentile(vals, 90)) if vals else float('nan'),
        'n': len(vals),
    } for label, vals in zones.items()}


def _deformation_analysis(result):
    """Analyze deformation param magnitudes."""
    deform_params = result['deform_params']
    summary = []
    for i in range(len(deform_params)):
        key = str(i)
        if key not in deform_params:
            continue
        dp = deform_params[key]
        cp = dp['control_points']
        wp = dp['width_profile']
        row = {'leaf': i}
        for name, vals in cp.items():
            arr = np.array(vals)
            row[f'{name}_max'] = float(np.abs(arr).max())
            row[f'{name}_mean'] = float(np.abs(arr).mean())
        row['width_min'] = float(min(wp))
        row['width_max'] = float(max(wp))
        row['width_negative'] = any(v < 0 for v in wp)
        summary.append(row)
    return summary


def _xml_params_table(result):
    """Pretty-print the per-leaf XML params."""
    names = result['xml_param_names']
    params = result['xml_params']
    n_per_leaf = len([n for n in names if n not in ('stem_ln', 'stem_tropismS', 'lnf')])

    rows = []
    for pos in range(N_POSITIONS):
        offset = pos * n_per_leaf
        row = {'position': pos + 2}  # subType = pos + 2
        for j, name in enumerate(names[:n_per_leaf]):
            row[name] = params[offset + j]
        rows.append(row)
    return rows, {n: params[-(len(names) - len(names[:n_per_leaf])) + i]
                  for i, n in enumerate(names[n_per_leaf:])}


def run_diagnostic(json_path, stl_path, day=60):
    print(f"Loading result: {json_path}")
    result = _load_result(json_path)
    print(f"  Best Chamfer: {result['best_loss']:.2f} cm")
    print(f"  Initial Chamfer: {result['initial_loss']:.2f} cm")
    print(f"  Evals: {result['n_evals']}")

    print(f"\nLoading target: {stl_path}")
    from ..targets.stl_loader import load_stl_as_pointcloud
    target_pts = load_stl_as_pointcloud(stl_path, n_points=10000)
    print(f"  Target points: {len(target_pts)}, extent: {np.ptp(target_pts, axis=0)}")

    print("\nGrowing CPlantBox plant with best XML params...")
    leaf_organs = _reconstruct_plant(result, day=day)
    if leaf_organs is None:
        print("ERROR: CPlantBox growth failed!")
        return
    print(f"  Leaves: {len(leaf_organs)}")

    # Rotation alignment (same as fitting)
    from dart.coupling.geometry.g1_to_g3 import loft_organs
    ref_mesh = loft_organs(leaf_organs)
    ref_pts = np.array(ref_mesh.vertices)
    if len(ref_pts) > 5000:
        ref_pts = ref_pts[np.random.RandomState(42).choice(len(ref_pts), 5000, replace=False)]
    from ..targets.pointcloud_loader import align_rotation_z
    target_pts, best_angle = align_rotation_z(target_pts, ref_pts, n_angles=72)
    print(f"  Rotation alignment: {best_angle:.0f} deg")

    # === 1. Per-leaf Chamfer (with deformations) ===
    print("\n=== PER-LEAF CHAMFER (with deformations) ===")
    verts_deformed = _loft_with_deformations(leaf_organs, result)
    per_leaf = _per_leaf_chamfer(verts_deformed, target_pts)
    per_leaf_sorted = sorted(per_leaf, key=lambda x: -x['chamfer'] if not np.isnan(x['chamfer']) else 0)
    for r in per_leaf_sorted:
        if np.isnan(r['chamfer']):
            continue
        print(f"  Leaf {r['leaf']:2d}: Chamfer={r['chamfer']:.2f} cm  "
              f"mean→target={r['mean_gen_to_target']:.2f}  "
              f"p90→target={r['p90_gen_to_target']:.2f}  "
              f"max→target={r['max_gen_to_target']:.2f}  "
              f"({r['n_verts']} verts)")

    # Total Chamfer
    all_deformed = np.concatenate([v for v in verts_deformed if len(v) > 0])
    tree_t = cKDTree(target_pts)
    tree_g = cKDTree(all_deformed)
    d1, _ = tree_t.query(all_deformed)
    d2, _ = tree_g.query(target_pts)
    print(f"  TOTAL: {(d1.mean() + d2.mean())/2:.2f} cm")

    # === 2. Per-leaf Chamfer (skeleton only, no deformations) ===
    print("\n=== PER-LEAF CHAMFER (skeleton only, no deformations) ===")
    verts_skel = _loft_skeleton_only(leaf_organs)
    per_leaf_skel = _per_leaf_chamfer(verts_skel, target_pts)
    per_leaf_skel_sorted = sorted(per_leaf_skel, key=lambda x: -x['chamfer'] if not np.isnan(x['chamfer']) else 0)
    for r in per_leaf_skel_sorted:
        if np.isnan(r['chamfer']):
            continue
        print(f"  Leaf {r['leaf']:2d}: Chamfer={r['chamfer']:.2f} cm  ({r['n_verts']} verts)")

    all_skel = np.concatenate([v for v in verts_skel if len(v) > 0])
    tree_gs = cKDTree(all_skel)
    d1s, _ = tree_t.query(all_skel)
    d2s, _ = tree_gs.query(target_pts)
    print(f"  TOTAL: {(d1s.mean() + d2s.mean())/2:.2f} cm")

    # === 3. Deformation impact per leaf ===
    print("\n=== DEFORMATION IMPACT (skeleton → deformed) ===")
    for i in range(len(leaf_organs)):
        ch_skel = next((r['chamfer'] for r in per_leaf_skel if r['leaf'] == i), float('nan'))
        ch_def = next((r['chamfer'] for r in per_leaf if r['leaf'] == i), float('nan'))
        if np.isnan(ch_skel):
            continue
        delta = ch_skel - ch_def
        pct = delta / ch_skel * 100 if ch_skel > 0 else 0
        print(f"  Leaf {i:2d}: {ch_skel:.2f} → {ch_def:.2f} cm  (Δ={delta:+.2f}, {pct:+.0f}%)")

    # === 4. Spatial error by height ===
    print("\n=== ERROR BY HEIGHT (deformed) ===")
    height_err = _spatial_error_by_height(verts_deformed, target_pts)
    for h in height_err:
        print(f"  {h['z_range']}: mean={h['mean_dist']:.2f} cm  p90={h['p90_dist']:.2f}  ({h['n_verts']} verts)")

    # === 5. Spatial error by arc position ===
    print("\n=== ERROR BY ARC POSITION (base/mid/tip, deformed) ===")
    arc_err = _spatial_error_by_arc(verts_deformed, leaf_organs, target_pts)
    for label, vals in arc_err.items():
        print(f"  {label}: mean={vals['mean']:.2f} cm  p90={vals['p90']:.2f}  ({vals['n']} verts)")

    # === 6. Deformation magnitude analysis ===
    print("\n=== DEFORMATION MAGNITUDES ===")
    deform_analysis = _deformation_analysis(result)
    for r in deform_analysis:
        neg = " *** NEGATIVE WIDTH ***" if r['width_negative'] else ""
        print(f"  Leaf {r['leaf']:2d}: "
              f"wave_n={r['wave_normal_max']:.1f}  "
              f"twist={r['twist_max']:.1f}  "
              f"curl={r['curl_max']:.1f}  "
              f"ruffle={r['edge_ruffle_max']:.1f}  "
              f"fold={r['fold_max']:.1f}  "
              f"w=[{r['width_min']:.1f},{r['width_max']:.1f}]{neg}")

    # === 7. XML params overview ===
    print("\n=== XML PARAMS (per leaf) ===")
    rows, globals_ = _xml_params_table(result)
    if rows:
        # Auto-detect columns from first row (skip 'position')
        col_names = [k for k in rows[0] if k != 'position']
        header = f"{'st':>3} " + " ".join(f"{n[:7]:>7}" for n in col_names)
        print(f"  {header}")
        for r in rows:
            vals = " ".join(f"{r[n]:7.3f}" for n in col_names)
            print(f"  {r['position']:3d} {vals}")
    print(f"  Globals: {globals_}")


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python -m dart.coupling.experimental.fitting.diagnose_fit <result.json> <target.stl>")
        sys.exit(1)
    run_diagnostic(sys.argv[1], sys.argv[2])
