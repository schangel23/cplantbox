"""Fit winter wheat 3D geometry from point cloud scans.

Usage:
    python -m dart.coupling.experimental.fitting.fit_wheat \\
        /path/to/wheat_1.txt [--leaf-evals 500] [--n-workers 64] [--day 55]

Loads a wheat point cloud (.txt with XYZ RGB), filters plant from background,
runs sequential CMA-ES fitting (stem 3D + 8 leaves 11D each), and exports
the fitted XML + deformation JSON.
"""

import argparse
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Fit wheat geometry to point cloud")
    parser.add_argument("target", help="Path to wheat point cloud (.txt XYZ RGB)")
    parser.add_argument("--stats", default=None,
                        help="Path to wheat_stats.json (default: data/wheat_stats.json)")
    parser.add_argument("--template-xml", default=None,
                        help="Path to wheat template XML (default: from species config)")
    parser.add_argument("--leaf-evals", type=int, default=500,
                        help="CMA-ES evaluations per leaf (default: 500)")
    parser.add_argument("--stem-evals", type=int, default=200,
                        help="CMA-ES evaluations for stem (default: 200)")
    parser.add_argument("--deform-steps", type=int, default=100,
                        help="Deformation optimization steps (default: 100)")
    parser.add_argument("--n-workers", type=int, default=16,
                        help="CPU workers for parallel CPlantBox growth (default: 16)")
    parser.add_argument("--day", type=int, default=55,
                        help="Simulation day (default: 55)")
    parser.add_argument("--n-points", type=int, default=10000,
                        help="Target subsample size (default: 10000)")
    parser.add_argument("--device", default="cuda",
                        help="Torch device (default: cuda)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tillers", type=int, default=3,
                        help="Number of tillers (default: 3, 0=single stem)")
    parser.add_argument("--xy-radius", type=float, default=0.0,
                        help="Cylindrical crop radius in cm (default: 0=disabled)")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (default: same as target)")
    args = parser.parse_args()

    import torch
    from ..targets.pointcloud_loader import load_pointcloud
    from .sequential_optimizer import fit_plant_sequential
    from .species_config import WHEAT

    # Resolve paths
    data_dir = Path(__file__).resolve().parents[2] / 'data'
    stats_path = args.stats or str(data_dir / 'wheat_stats.json')

    species = WHEAT
    if args.template_xml:
        species.template_xml = args.template_xml

    # Patch tiller count in template XML
    if args.tillers >= 0:
        import xml.etree.ElementTree as ET
        import tempfile
        tree = ET.parse(species.template_xml)
        for seed_el in tree.getroot().iter('seed'):
            for param in seed_el:
                name = param.get('name', '')
                if name == 'maxTil':
                    param.set('value', str(args.tillers))
                elif name == 'firstTil':
                    param.set('value', '5' if args.tillers > 0 else '1e+09')
                elif name == 'delayTil':
                    param.set('value', '3' if args.tillers > 0 else '1e+09')
        tmp = tempfile.NamedTemporaryFile(suffix='.xml', delete=False, dir=str(data_dir))
        tree.write(tmp.name)
        tmp.close()
        species.template_xml = tmp.name
        print(f"Tillers: {args.tillers} (patched XML)", file=sys.stderr)

    # Load and filter point cloud
    print(f"Loading: {args.target}", file=sys.stderr)
    target_pts, _colors = load_pointcloud(
        args.target, n_points=args.n_points, units='cm',
        xy_radius=args.xy_radius,
    )
    print(f"  Points after filtering: {len(target_pts)}", file=sys.stderr)
    print(f"  XYZ extent: X=[{target_pts[:,0].min():.1f}, {target_pts[:,0].max():.1f}] "
          f"Y=[{target_pts[:,1].min():.1f}, {target_pts[:,1].max():.1f}] "
          f"Z=[{target_pts[:,2].min():.1f}, {target_pts[:,2].max():.1f}]",
          file=sys.stderr)

    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
        print("  CUDA not available, using CPU", file=sys.stderr)

    # Run fitting
    result = fit_plant_sequential(
        target_pts,
        stats_path,
        stem_evals=args.stem_evals,
        leaf_evals=args.leaf_evals,
        deform_steps=args.deform_steps,
        day=args.day,
        device=device,
        seed=args.seed,
        n_workers=args.n_workers,
        species=species,
    )

    # Save results
    target_path = Path(args.target)
    out_dir = Path(args.output_dir) if args.output_dir else target_path.parent
    out_base = target_path.stem

    # Save fit result JSON
    fit_path = out_dir / f'{out_base}_wheat_fit.json'
    with open(fit_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved fit result: {fit_path}", file=sys.stderr)

    # Export fitted XML
    xml_path = out_dir / f'{out_base}_wheat_fitted.xml'
    _export_fitted_xml(result, species, xml_path)
    print(f"Saved fitted XML: {xml_path}", file=sys.stderr)

    # Save deformation JSON
    if result.get('deform_params'):
        deform_path = out_dir / f'{out_base}_wheat_fitted_deformations.json'
        with open(deform_path, 'w') as f:
            json.dump(result['deform_params'], f, indent=2)
        print(f"Saved deformations: {deform_path}", file=sys.stderr)

    # Export fitted mesh for visual verification
    _export_fitted_mesh(result, species, args.day, out_dir / f'{out_base}_wheat_fitted.obj')

    print(f"\nFinal Chamfer distance: {result['final_loss']:.2f} cm", file=sys.stderr)
    print(f"Per-leaf losses: {[f'{l:.2f}' for l in result['per_leaf_losses']]}", file=sys.stderr)


def _export_fitted_xml(result, species, out_path):
    """Write fitted parameters back into the wheat template XML."""
    import xml.etree.ElementTree as ET

    tree = ET.parse(species.template_xml)
    root = tree.getroot()

    stem_params = result['stem_params']
    leaf_params = result['leaf_params']
    param_names = result['leaf_param_names']

    for stem_el in root.iter('stem'):
        for elem in stem_el:
            name = elem.get('name', '')
            if name == 'ln':
                elem.set('value', str(stem_params['ln']))
            elif name == 'tropismS':
                elem.set('value', str(stem_params['tropismS']))
            elif name == 'lnf':
                elem.set('value', str(int(round(stem_params.get('lnf', 0)))))

    for leaf_el in root.iter('leaf'):
        sub = int(leaf_el.get('subType', '0'))
        pos = sub - species.subtype_offset
        if 0 <= pos < len(leaf_params):
            p = dict(zip(param_names, leaf_params[pos]))
            xml_map = {
                'lmax': p['lmax'],
                'Width_blade': p['Width_blade'],
                'theta': p['theta'],
                'tropismS': p['tropismS'],
                'tropismAge': p['tropismAge'],
                'r': p['r'],
                'areaMax': p['lmax'] * p['Width_blade'] * 2.0 * 0.73,
                'collarLength': p.get('collarLength', 3.0),
                'InitBeta': p.get('initBeta', 0.2),
            }
            to_remove = []
            for elem in leaf_el:
                name = elem.get('name', '')
                if name in xml_map:
                    elem.set('value', str(xml_map[name]))
                elif name == 'leafCurvature':
                    to_remove.append(elem)
            for elem in to_remove:
                leaf_el.remove(elem)

            curv = ET.SubElement(leaf_el, 'parameter')
            curv.set('name', 'leafCurvature')
            curv.set('phi', '0.0 0.5 1.0')
            curv.set('kappa', f"{p.get('kappa_base', 0)} {p.get('kappa_mid', 0)} {p.get('kappa_tip', 0)}")

    tree.write(str(out_path), xml_declaration=True, encoding='utf-8')


def _export_fitted_mesh(result, species, day, out_path):
    """Grow plant with fitted params and export OBJ mesh."""
    try:
        from .sequential_optimizer import _grow_single
        from dart.coupling.geometry.g1_to_g3 import loft_organs

        param_names = result['leaf_param_names']
        leaf_params_list = [
            dict(zip(param_names, lp)) for lp in result['leaf_params']
        ]
        organs = _grow_single(
            result['stem_params'], leaf_params_list,
            day=day, species=species,
        )
        if organs:
            mesh = loft_organs(organs)
            mesh.to_obj(str(out_path))
            print(f"Saved fitted mesh: {out_path} ({len(mesh.vertices)} verts)",
                  file=sys.stderr)
    except Exception as e:
        print(f"Warning: Could not export mesh: {e}", file=sys.stderr)


if __name__ == '__main__':
    main()
