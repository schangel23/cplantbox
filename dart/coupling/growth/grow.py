#!/usr/bin/env python3
"""
Grow a maize plant using Pheno4D-calibrated maize.xml, then extract high-quality G3 mesh.

This is Baker's approach:
  1. CPlantBox generates G1 skeleton from parametric growth model
  2. G1→G3 lofter adds realistic geometry (tubes + leaf surfaces)
  3. Export to OBJ with UV mapping for DART
  4. Render side-by-side G1 | G3 comparison PNG

No skeleton injection. Just parametric growth with calibrated parameters.

Usage:
  python grow_calibrated_plant.py \
      --xml maize_calibrated.xml \
      --days 30 \
      --output maize_day30 \
      --resolution fine
"""

import numpy as np
from pathlib import Path
import argparse

import plantbox as pb

from ..config import HYDRAULICS_PATH, DEFAULT_XML, get_hydraulics_json, get_photosynthesis_json, get_phloem_json
from ..geometry import loft_organs, G3Mesh, extract_organs_for_lofter
from ..prospect_params import get_chl_per_segment, vcmax25_from_cab


def setup_successor_where(plant):
    """Set deterministic per-position successorWhere on the mainstem.

    The XML contains leaf subtypes 2..N (one per position) but only a
    placeholder successor rule.  This function replaces it with per-position
    rules via the Python API so that linking node 0 gets subType 2,
    linking node 1 gets subType 3, etc.
    """
    # Discover which leaf subtypes exist (subType >= 2 with Width_blade > 0)
    leaf_subtypes = []
    for p in plant.getOrganRandomParameter(pb.leaf):
        if p.subType >= 2 and p.Width_blade > 0.01:
            leaf_subtypes.append(p.subType)
    leaf_subtypes.sort()

    if not leaf_subtypes:
        print("  No calibrated leaf subtypes found, skipping successorWhere")
        return

    # Set successorWhere on mainstem (subType=1) and enforce phyllotaxy
    import math
    for p in plant.getOrganRandomParameter(pb.stem):
        if p.subType == 1:
            p.successorST = [[st] for st in leaf_subtypes]
            p.successorOT = [[4] for _ in leaf_subtypes]  # organType=4 (leaf)
            p.successorP = [[1.0] for _ in leaf_subtypes]
            p.successorNo = [1] * len(leaf_subtypes)
            p.successorWhere = [[float(i)] for i in range(len(leaf_subtypes))]
            # Distichous phyllotaxy (180° alternating, two-ranked) — correct for maize.
            # BetaDev=0.22 (~12.6°) adds natural scatter so the plant doesn't look
            # like a perfectly flat ellipse from every angle.  Real maize deviates
            # 5-15° from perfect 180° due to mechanical interactions and growth.
            p.RotBeta = math.pi  # 180°
            p.BetaDev = 0.22
            plant.setOrganRandomParameter(p)
            print(f"  successorWhere: {len(leaf_subtypes)} rules "
                  f"(node 0->subType {leaf_subtypes[0]}, ..., "
                  f"node {len(leaf_subtypes)-1}->subType {leaf_subtypes[-1]})")
            print(f"  phyllotaxy: RotBeta={math.degrees(p.RotBeta):.0f} deg, BetaDev={p.BetaDev}")
            break


def init_plant(xml_path=None, seed=None, enable_photosynthesis=True):
    """Create and initialize a plant without growing. For carbon-limited mode.

    Same setup as grow_plant() but stops after initialize().
    Returns plant at day 0.

    Args:
        xml_path: Path to calibrated XML. Defaults to DEFAULT_XML.
        seed: Optional random seed for reproducibility.
        enable_photosynthesis: Enable soil grid for photosynthesis (default True).

    Returns:
        pb.MappedPlant at day 0, initialized and ready for simulate().
    """
    if xml_path is None:
        xml_path = str(DEFAULT_XML)

    plant = pb.MappedPlant()
    plant.readParameters(str(xml_path))

    if seed is not None:
        plant.setSeed(seed)

    setup_successor_where(plant)

    if enable_photosynthesis:
        depth = 100
        soil_domain = pb.SDF_PlantContainer(np.inf, np.inf, depth, True)
        plant.setGeometry(soil_domain)

        def _picker(_x, _y, z):
            return max(min(int(np.floor(-z)), depth - 1), -1)
        plant.setSoilGrid(_picker)

    plant.initialize()
    return plant


def grow_plant(xml_path, simulation_time, min_stem_nodes=50, min_leaf_nodes=20,
               enable_photosynthesis=False, seed=None):
    """Grow a CPlantBox plant from calibrated XML."""
    print(f"=== Growing Plant ===")
    print(f"  XML: {xml_path}")
    print(f"  Simulation time: {simulation_time} days")
    if seed is not None:
        print(f"  Seed: {seed}")
    if enable_photosynthesis:
        print(f"  Photosynthesis: ENABLED (soil grid active)")

    plant = pb.MappedPlant()
    plant.readParameters(str(xml_path))

    if seed is not None:
        plant.setSeed(seed)

    # Set per-position successor rules via Python API
    setup_successor_where(plant)

    # Soil geometry — must be set BEFORE plant.initialize() when using photosynthesis.
    # Roots are excluded from the G3 mesh (skip_roots=True in adapter) but kept in
    # the simulation for water uptake.
    if enable_photosynthesis:
        depth = 100  # cm — covers full maize root depth
        soil_domain = pb.SDF_PlantContainer(np.inf, np.inf, depth, True)
        plant.setGeometry(soil_domain)

        def _picker(_x, _y, z):
            """Map 3D position to soil cell index. Above-ground → -1.
            Clamp to [0, depth-1] to avoid out-of-bounds at exactly z=-depth."""
            return max(min(int(np.floor(-z)), depth - 1), -1)
        plant.setSoilGrid(_picker)

    plant.initialize()

    # Use incremental simulation with error recovery.
    # CPlantBox has a vector bounds bug with >8 leaf subtypes during initial
    # lateral creation. Incremental steps + catch allow partial growth.
    dt = 1.0  # 1-day steps
    total_simulated = 0.0
    while total_simulated < simulation_time:
        step = min(dt, simulation_time - total_simulated)
        try:
            plant.simulate(step, verbose=(total_simulated == 0))
            total_simulated += step
        except (IndexError, RuntimeError) as e:
            print(f"  Warning: simulate() error at day {total_simulated + step:.1f}: {e}")
            print(f"  Continuing with {total_simulated:.1f} days simulated")
            # Re-sync nodes after error
            try:
                plant.simulate(0.0)
            except Exception:
                pass
            break

    organs = plant.getOrgans()
    n_stems = sum(1 for o in organs if o.organType() == pb.OrganTypes.stem)
    n_leaves = sum(1 for o in organs if o.organType() == pb.OrganTypes.leaf)
    n_roots = sum(1 for o in organs if o.organType() == pb.OrganTypes.root)

    print(f"\n  Stems: {n_stems}, Leaves: {n_leaves}, Roots: {n_roots} (excluded from G3)")
    print(f"  Total nodes: {len(plant.getNodes())}")

    # Print per-leaf stats for verification
    leaf_organs = [o for o in organs if o.organType() == pb.OrganTypes.leaf]
    if leaf_organs:
        print(f"\n  Per-leaf summary:")
        print(f"  {'#':>3} {'SubType':>7} {'Length':>8} {'Nodes':>5}")
        for j, leaf in enumerate(leaf_organs):
            st = leaf.getParameter("subType")
            length = leaf.getLength(False)
            n_nodes = len(leaf.getNodes())
            print(f"  {j:>3} {st:>7.0f} {length:>8.1f} cm {n_nodes:>5}")

    return plant


def extract_g3_mesh(plant, min_stem_nodes=50, min_leaf_nodes=20, stem_res=16,
                    include_roots=False):
    """Extract G1 skeleton from CPlantBox and loft to G3 mesh.

    Args:
        include_roots: If True, include root geometry in the mesh.
                       Default False (shoot only, roots excluded for DART).
    """
    print(f"\n=== Extracting G3 Mesh ===")

    organ_dicts = extract_organs_for_lofter(
        plant,
        min_stem_nodes=min_stem_nodes,
        min_leaf_nodes=min_leaf_nodes,
        skip_roots=not include_roots,
    )

    label = "shoot + root" if include_roots else "shoot only"
    print(f"  Extracted {len(organ_dicts)} organs ({label})")

    mesh = loft_organs(organ_dicts, stem_sides=stem_res)

    print(f"  Vertices: {mesh.n_vertices}, Triangles: {mesh.n_triangles}")

    return mesh, organ_dicts


def extract_root_dicts(plant, min_root_nodes=20):
    """Extract root organ dicts for visualization."""
    root_dicts = extract_organs_for_lofter(
        plant,
        min_stem_nodes=min_root_nodes,
        min_leaf_nodes=min_root_nodes,
        skip_roots=False
    )
    # Keep only roots
    return [o for o in root_dicts if o['type'] == 'root']


def run_photosynthesis(plant, sim_time, output_prefix,
                       par_umol=1000.0, tair_c=25.0, rh=0.7,
                       soil_psi_cm=-500.0):
    """Set up hydraulics + C4 photosynthesis and run a single solve.

    Uses:
      - couvreur2012.json  : maize root hydraulics (Doussan 1998 via Couvreur 2012)
      - maize_C4_photosynthesis_parameters.json : PhotoType=1, alpha=0.05

    @param plant          pb.MappedPlant (grown, with soil grid)
    @param sim_time       days simulated (for age-dependent conductivities)
    @param output_prefix  path prefix for CSV output
    @param par_umol       PAR [umol photons m-2 s-1] — uniform over all leaves
    @param tair_c         Air / leaf temperature [°C]
    @param rh             Relative humidity [0–1]
    @param soil_psi_cm    Uniform soil water potential [cm] — -500 = well-watered
    """
    print(f"\n=== Photosynthesis Solve ===")
    print(f"  PAR={par_umol} umol m-2 s-1, T={tair_c}°C, RH={rh*100:.0f}%")
    print(f"  Soil psi={soil_psi_cm} cm  (hydraulics: couvreur2012 / C4 params)")

    from plantbox.functional.phloem_flux import PhloemFluxPython
    from plantbox.functional.PlantHydraulicParameters import PlantHydraulicParameters

    # --- Hydraulic parameters ---
    params = PlantHydraulicParameters()
    params.read_parameters(get_hydraulics_json())

    # --- Photosynthesis + phloem model ---
    hm = PhloemFluxPython(plant, params)
    hm.read_photosynthesis_parameters(filename=get_photosynthesis_json())
    hm.read_phloem_parameters(filename=get_phloem_json())

    # Per-segment Chl from LOPS per-position profiles
    chl_per_seg = get_chl_per_segment(sim_time, plant)
    seg_leaves_check = plant.getSegmentIds(4)
    if len(chl_per_seg) == len(seg_leaves_check):
        hm.Chl = chl_per_seg
        cab_min, cab_max = min(chl_per_seg), max(chl_per_seg)
        vcmax_range = f"[{vcmax25_from_cab(cab_min):.1f}, {vcmax25_from_cab(cab_max):.1f}]"
        print(f"  PhotoType={'C4' if hm.PhotoType == 1 else 'C3'}, "
              f"Vcmax range={vcmax_range} umol m-2 s-1 "
              f"(Cab range=[{cab_min:.1f}, {cab_max:.1f}] ug/cm2, "
              f"{len(chl_per_seg)} segs)")
    else:
        vcmax_umol = (hm.VcmaxrefChl1 * hm.Chl[0] + hm.VcmaxrefChl2)
        print(f"  PhotoType={'C4' if hm.PhotoType == 1 else 'C3'}, "
              f"Vcmax~{vcmax_umol:.1f} umol m-2 s-1 (Chl={hm.Chl[0]:.1f} ug/cm2)")

    # --- Soil water potential vector ---
    depth = 100  # must match setSoilGrid depth in grow_plant()
    p_s = np.linspace(soil_psi_cm, soil_psi_cm - depth, depth)

    # --- Weather ---
    es = hm.get_es(tair_c)
    ea = es * rh

    # PAR conversion: umol m-2 s-1  → mol cm-2 d-1
    par_mol_cm2_d = par_umol * 1e-6 * 86400 * 1e-4

    # --- Solve photosynthesis + hydraulics ---
    try:
        hm.solve(
            sim_time=sim_time,
            rsx=p_s,
            cells=True,
            ea=ea,
            es=es,
            PAR=par_mol_cm2_d,
            TairC=tair_c,
            verbose=0,
        )
    except Exception as e:
        print(f"  ERROR in hm.solve(): {e}")
        return None

    # --- Results ---
    # NB: get_net_assimilation() returns per-leaf-segment (size = n_leaf_segs),
    # NOT per-all-segments.  Indexed 0..n_leaf-1, matching seg_leaves_idx order.
    An_leaf  = np.array(hm.get_net_assimilation())       # mol CO2 d-1 per leaf seg
    An_per   = np.array(hm.get_net_assimilation_perleafBladeArea())  # mol CO2 cm-2 d-1
    hx_all   = np.array(hm.get_water_potential())
    transp   = np.sum(hm.get_transpiration()) / 18 * 1e3  # mmol H2O d-1

    An_total_mmol = np.sum(An_leaf) * 1e3  # mmol CO2 d-1 whole plant
    n_leaf_segs = len(An_leaf)

    # Convert An_per to umol m-2 s-1:  mol cm-2 d-1 * 1e4 cm2/m2 / 86400 s/d * 1e6
    An_per_umol = An_per * 1e4 / 86400 * 1e6  # umol CO2 m-2 s-1

    print(f"\n  --- Results ---")
    print(f"  Total net assimilation : {An_total_mmol:.3f} mmol CO2 d-1")
    print(f"  Total transpiration    : {transp:.3f} mmol H2O d-1")
    print(f"  Leaf-blade segments    : {n_leaf_segs}")
    if n_leaf_segs > 0:
        nonzero = An_per_umol[An_per_umol > 0]
        print(f"  Active segments        : {len(nonzero)} / {n_leaf_segs}")
        if len(nonzero) > 0:
            print(f"  Mean An (active)       : {np.mean(nonzero):.2f} umol CO2 m-2 s-1")
            print(f"  Min/Max An             : {np.min(nonzero):.2f} / {np.max(nonzero):.2f} umol m-2 s-1")
    print(f"  Mean xylem psi         : {np.mean(hx_all):.0f} cm")

    # --- Per-leaf organ summary ---
    # Use plant.organTypes and plant.subTypes arrays (aligned with getSegments())
    # to map An values to individual leaf organs.  get_net_assimilation() returns
    # one value per leaf segment, ordered by their position in getSegments()
    # filtered to organType==4.
    ot_arr = np.array(plant.organTypes)   # per-segment organ type
    st_arr = np.array(plant.subTypes)     # per-segment sub type
    leaf_mask = (ot_arr == 4)             # True for leaf segments
    leaf_global_indices = np.where(leaf_mask)[0]  # global seg indices of leaves
    # An arrays are indexed 0..n_leaf-1, same order as leaf_global_indices
    assert len(leaf_global_indices) == n_leaf_segs, \
        f"Mismatch: {len(leaf_global_indices)} vs {n_leaf_segs}"

    # Map global seg index -> An array index (for leaf segs only)
    global_to_an = {int(gi): ai for ai, gi in enumerate(leaf_global_indices)}

    organs = plant.getOrgans()
    leaf_organs = [o for o in organs if o.organType() == pb.OrganTypes.leaf]
    lbs = np.array(plant.leafBladeSurface)

    print(f"\n  Per-leaf An:")
    print(f"  {'#':>3} {'SubType':>7} {'Length':>8} {'Segs':>5} "
          f"{'An_mean_umol':>12} {'An_sum_mmol':>12}")

    organ_data = []
    for j, leaf in enumerate(leaf_organs):
        st = int(leaf.getParameter("subType"))
        length = leaf.getLength(False)
        width = leaf.getParameter("Width_blade")

        # Find An indices for this organ's segments via subType match
        organ_leaf_mask = leaf_mask & (st_arr == st)
        organ_global_indices = np.where(organ_leaf_mask)[0]
        an_indices = [global_to_an[int(gi)] for gi in organ_global_indices
                      if int(gi) in global_to_an]

        # Blade area for this organ
        blade_area = sum(lbs[gi] for gi in organ_global_indices if gi < len(lbs)) * 2

        if an_indices:
            An_org = An_per_umol[an_indices]
            An_mean = np.mean(An_org)
            An_sum  = np.sum(An_leaf[an_indices]) * 1e3
            psi_segs = [hx_all[gi] for gi in organ_global_indices
                        if 0 <= gi < len(hx_all)]
            psi_mean = np.mean(psi_segs) if psi_segs else 0.0
        else:
            An_mean = 0.0
            An_sum  = 0.0
            psi_mean = 0.0

        print(f"  {j:>3} {st:>7} {length:>8.1f} cm {len(an_indices):>5} "
              f"{An_mean:>12.2f} {An_sum:>12.4f}")

        organ_data.append({
            'index': j, 'subtype': st, 'length': length,
            'width': width, 'blade_area': blade_area,
            'An_sum_mmol': An_sum, 'psi_mean': psi_mean,
            'n_segs': len(an_indices),
        })

    # --- Save CSV (leaf segments only) ---
    if output_prefix is not None:
        csv_path = Path(output_prefix).with_suffix('.csv')
        header = "leaf_seg_idx,global_seg_idx,An_mol_d,An_umol_m2_s,psi_cm"
        rows = []
        for i in range(n_leaf_segs):
            gi = int(leaf_global_indices[i])
            psi = hx_all[gi] if 0 <= gi < len(hx_all) else 0.0
            rows.append(f"{i},{gi},{An_leaf[i]:.6e},{An_per_umol[i]:.4f},{psi:.2f}")
        csv_path.write_text(header + "\n" + "\n".join(rows))
        print(f"\n  CSV: {csv_path} ({len(rows)} leaf segments)")

        # --- Plot ---
        from .render import plot_photosynthesis
        plot_photosynthesis(
            organ_data=organ_data,
            An_per_umol=An_per_umol,
            hx_all=hx_all,
            seg_leaves_idx=list(leaf_global_indices),
            An_total_mmol=An_total_mmol,
            transp=transp,
            par_umol=par_umol,
            tair_c=tair_c,
            rh=rh,
            sim_time=sim_time,
            output_prefix=output_prefix,
        )

    return hm


def export_mesh(mesh, output_prefix):
    """Export G3 mesh to OBJ + JSON mapping files."""
    output_dir = Path(output_prefix).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    obj_path = Path(output_prefix).with_suffix('.obj')
    json_path = Path(output_prefix).with_suffix('.json')

    mesh.to_obj(str(obj_path), group_by_organ=True)
    mesh.to_mapping_json(str(json_path))

    print(f"\n=== Exported ===")
    print(f"  OBJ:  {obj_path} ({mesh.n_triangles} triangles)")
    print(f"  JSON: {json_path} ({len(mesh.organ_meta)} organs)")


def export_g1_skeleton(plant, output_prefix):
    """Export G1 skeleton as thin-tube OBJ."""
    organ_dicts = extract_organs_for_lofter(
        plant, min_stem_nodes=2, min_leaf_nodes=2, skip_roots=True,
    )

    for organ in organ_dicts:
        organ['widths'] = np.full(len(organ['skeleton']), 0.04)

    g1_mesh = loft_organs(organ_dicts, stem_sides=4)

    g1_path = Path(output_prefix).with_name(
        Path(output_prefix).stem + '_g1'
    ).with_suffix('.obj')
    g1_mesh.to_obj(str(g1_path), group_by_organ=True)

    print(f"  G1 OBJ: {g1_path} ({g1_mesh.n_triangles} triangles)")
    return g1_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description='Grow maize plant with calibrated parameters and extract G3 mesh'
    )
    parser.add_argument('--xml', required=True, help='Path to calibrated maize.xml')
    parser.add_argument('--days', type=float, default=30, help='Simulation time (days)')
    parser.add_argument('--output', required=True, help='Output prefix (e.g., maize_day30)')
    parser.add_argument('--resolution', choices=['coarse', 'medium', 'fine', 'ultra'],
                       default='fine', help='Mesh resolution')
    parser.add_argument('--export-g1', action='store_true',
                       help='Export G1 skeleton OBJ alongside G3 mesh')
    parser.add_argument('--no-png', action='store_true',
                       help='Skip PNG rendering (only export OBJ + JSON)')
    parser.add_argument('--svg', action='store_true',
                       help='Export SVG vector graphic (G1 skeleton | G3 mesh side-by-side)')
    parser.add_argument('--publication', action='store_true',
                       help='Export publication-quality SVG (white bg, scale bar, (a)/(b) labels)')
    parser.add_argument('--animate', action='store_true',
                       help='Export animated SVG showing growth from day 1 to --days')
    parser.add_argument('--animate-step', type=int, default=5,
                       help='Days between animation frames (default: 5)')
    parser.add_argument('--frame-dur', type=float, default=0.5,
                       help='Seconds per animation frame (default: 0.5)')
    parser.add_argument('--photosynthesis', action='store_true',
                       help='Run C4 photosynthesis solve after growth (requires soil grid)')
    parser.add_argument('--par', type=float, default=1000.0,
                       help='PAR for photosynthesis solve [umol m-2 s-1] (default: 1000)')
    parser.add_argument('--tair', type=float, default=25.0,
                       help='Air temperature for photosynthesis solve [°C] (default: 25)')
    parser.add_argument('--rh', type=float, default=0.7,
                       help='Relative humidity for photosynthesis solve [0-1] (default: 0.7)')
    parser.add_argument('--leuning', action='store_true',
                       help='(deprecated — PhloemFluxPython is now the default solver)')
    parser.add_argument('--include-roots-in-mesh', action='store_true',
                       help='Include root geometry in G3 mesh export (default: shoot only)')
    args = parser.parse_args()

    resolution_presets = {
        'coarse': {'min_stem_nodes': 30, 'min_leaf_nodes': 15, 'stem_res': 12},
        'medium': {'min_stem_nodes': 50, 'min_leaf_nodes': 20, 'stem_res': 16},
        'fine': {'min_stem_nodes': 100, 'min_leaf_nodes': 40, 'stem_res': 20},
        'ultra': {'min_stem_nodes': 200, 'min_leaf_nodes': 80, 'stem_res': 32}
    }

    preset = resolution_presets[args.resolution]

    print("=" * 60)
    print("CPlantBox -> G1 -> G3 Pipeline (Calibrated Growth)")
    print("=" * 60)

    # Grow plant
    plant = grow_plant(
        xml_path=args.xml,
        simulation_time=args.days,
        min_stem_nodes=preset['min_stem_nodes'],
        min_leaf_nodes=preset['min_leaf_nodes'],
        enable_photosynthesis=args.photosynthesis,
    )

    # Extract G3 mesh (also returns organ_dicts for rendering)
    mesh, organ_dicts = extract_g3_mesh(
        plant,
        min_stem_nodes=preset['min_stem_nodes'],
        min_leaf_nodes=preset['min_leaf_nodes'],
        stem_res=preset['stem_res'],
        include_roots=args.include_roots_in_mesh,
    )

    # Export OBJ + JSON
    export_mesh(mesh, args.output)

    # Optional: G1 skeleton OBJ
    if args.export_g1:
        export_g1_skeleton(plant, args.output)

    # Extract root dicts for visualization
    root_dicts = extract_root_dicts(plant, min_root_nodes=preset.get('min_stem_nodes', 50) // 2)
    print(f"  Roots extracted: {len(root_dicts)} organs")

    # Render G1 | G3 | Roots comparison PNG (default on)
    if not args.no_png:
        from .render import render_comparison_png
        render_comparison_png(organ_dicts, mesh, args.output, args.days,
                             root_dicts=root_dicts)

    # Optional: SVG vector graphic
    if args.svg:
        from .render import render_comparison_svg
        render_comparison_svg(organ_dicts, mesh, args.output, args.days,
                              root_dicts=root_dicts)

    # Optional: publication-quality SVG
    if args.publication:
        from .render import render_publication_svg
        render_publication_svg(organ_dicts, mesh, args.output, args.days,
                               root_dicts=root_dicts)

    # Optional: animated SVG (runs its own simulation loop)
    if args.animate:
        from .render import render_animated_svg
        render_animated_svg(
            xml_path=args.xml,
            max_days=args.days,
            output_prefix=args.output,
            preset=preset,
            day_step=args.animate_step,
            frame_dur=args.frame_dur,
        )

    # Optional: C4 photosynthesis solve
    if args.photosynthesis:
        photo_prefix = args.output + '_photosynthesis'
        run_photosynthesis(
            plant=plant,
            sim_time=args.days,
            output_prefix=photo_prefix,
            par_umol=args.par,
            tair_c=args.tair,
            rh=args.rh,
        )

    print("\n" + "=" * 60)
    print("Pipeline complete")
    print("=" * 60)


if __name__ == '__main__':
    main()
