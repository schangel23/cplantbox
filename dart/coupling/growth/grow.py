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

# ---------------------------------------------------------------------------
# Color palette (matching batch pipeline)
# ---------------------------------------------------------------------------
LEAF_GREENS = [
    (0.18, 0.55, 0.18),  # leaf 1 - forest green
    (0.30, 0.69, 0.31),  # leaf 2
    (0.46, 0.80, 0.46),  # leaf 3
    (0.56, 0.88, 0.56),  # leaf 4
    (0.60, 0.80, 0.20),  # leaf 5 - yellow-green
    (0.20, 0.65, 0.32),  # leaf 6
    (0.40, 0.75, 0.40),  # leaf 7
    (0.50, 0.85, 0.50),  # leaf 8
    (0.13, 0.55, 0.13),  # leaf 9 - dark green
    (0.24, 0.70, 0.44),  # leaf 10 - sea green
    (0.42, 0.76, 0.22),  # leaf 11 - olive-green
    (0.33, 0.65, 0.50),  # leaf 12 - medium sea green
    (0.52, 0.82, 0.32),  # leaf 13 - lime-green
    (0.22, 0.58, 0.28),  # leaf 14 - dark spring green
    (0.38, 0.72, 0.38),  # leaf 15
    (0.48, 0.78, 0.48),  # leaf 16
]
STEM_COLOR = (0.55, 0.27, 0.07)  # brown

PANEL_W, PANEL_H = 600, 800  # pixels per panel


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


# ---------------------------------------------------------------------------
# Root Length Density (RLD) profile extraction
# ---------------------------------------------------------------------------

def extract_rld_profile(plant, n_layers=20, depth_cm=100.0,
                        row_spacing_cm=75.0, plant_spacing_cm=20.0):
    """Extract root length density profile from a grown plant.

    Uses pb.SegmentAnalyser to compute vertical distribution of root length,
    then normalises to RLD [cm root / cm3 soil].

    Args:
        plant: pb.MappedPlant (grown, with roots)
        n_layers: number of depth bins
        depth_cm: maximum soil depth [cm]
        row_spacing_cm: inter-row spacing [cm] (maize default 75)
        plant_spacing_cm: intra-row spacing [cm] (maize default 20)

    Returns:
        dict with keys:
            depth_mid_cm: array of layer midpoint depths (positive downward) [cm]
            depth_top_cm: array of layer top depths [cm]
            depth_bot_cm: array of layer bottom depths [cm]
            root_length_cm: root length per layer [cm]
            RLD_cm_per_cm3: root length density [cm/cm3]
            total_root_length_cm: total root length [cm]
            max_root_depth_cm: deepest root [cm]
            n_root_segments: total root segment count
            layer_thickness_cm: thickness of each layer [cm]
            ground_area_cm2: ground area per plant [cm2]
    """
    ana = pb.SegmentAnalyser(plant)
    ana.filter("organType", pb.root)

    # Root length per layer [cm] — top=0, bot=-depth (CPlantBox z convention)
    root_length = np.array(
        ana.distribution("length", 0.0, -depth_cm, n_layers, True)
    )

    layer_thickness = depth_cm / n_layers
    ground_area = row_spacing_cm * plant_spacing_cm
    layer_volume = layer_thickness * ground_area  # cm3

    RLD = root_length / layer_volume  # cm root / cm3 soil

    # Depth arrays (positive = below surface)
    depth_top = np.linspace(0, depth_cm - layer_thickness, n_layers)
    depth_bot = depth_top + layer_thickness
    depth_mid = (depth_top + depth_bot) / 2.0

    # Root statistics
    total_root_length = float(np.sum(root_length))

    # Max root depth from actual node positions
    root_organs = [o for o in plant.getOrgans()
                   if o.organType() == pb.OrganTypes.root]
    n_root_segs = sum(len(o.getNodes()) - 1 for o in root_organs)
    root_z = []
    for o in root_organs:
        for n in o.getNodes():
            root_z.append(n.z)
    max_root_depth = abs(min(root_z)) if root_z else 0.0

    return {
        "depth_mid_cm": depth_mid,
        "depth_top_cm": depth_top,
        "depth_bot_cm": depth_bot,
        "root_length_cm": root_length,
        "RLD_cm_per_cm3": RLD,
        "total_root_length_cm": total_root_length,
        "max_root_depth_cm": max_root_depth,
        "n_root_segments": n_root_segs,
        "layer_thickness_cm": layer_thickness,
        "ground_area_cm2": ground_area,
    }


def export_rld_csv(rld_profile, output_path):
    """Export RLD profile to CSV.

    Columns: depth_top_cm, depth_bot_cm, depth_mid_cm,
             root_length_cm, RLD_cm_per_cm3
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    header = "depth_top_cm,depth_bot_cm,depth_mid_cm,root_length_cm,RLD_cm_per_cm3"
    rows = []
    for i in range(len(rld_profile["depth_mid_cm"])):
        rows.append(
            f"{rld_profile['depth_top_cm'][i]:.2f},"
            f"{rld_profile['depth_bot_cm'][i]:.2f},"
            f"{rld_profile['depth_mid_cm'][i]:.2f},"
            f"{rld_profile['root_length_cm'][i]:.4f},"
            f"{rld_profile['RLD_cm_per_cm3'][i]:.6f}"
        )
    output_path.write_text(header + "\n" + "\n".join(rows))
    print(f"  RLD CSV: {output_path} ({len(rows)} layers)")
    return output_path


def export_rrd_in(rld_profile, output_path):
    """Export RLD profile to AgroC rrd.in format.

    Format (read by plants.f90:2472-2524):
      N                         (number of rows)
      relative_depth  relative_density
      ...
    Depth: 0 = surface, 1 = max_root_depth.
    Density: normalised (AgroC re-normalises internally).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    max_depth = rld_profile["max_root_depth_cm"]
    if max_depth <= 0:
        print("  WARNING: No root depth — skipping rrd.in export")
        return None

    depths_rel = rld_profile["depth_mid_cm"] / max_depth
    # Clamp to [0, 1] — layers deeper than max_root_depth get 1.0
    depths_rel = np.clip(depths_rel, 0.0, 1.0)

    rld_values = rld_profile["RLD_cm_per_cm3"]
    rld_sum = np.sum(rld_values)
    if rld_sum > 0:
        density_rel = rld_values / rld_sum
    else:
        density_rel = np.zeros_like(rld_values)

    n_rows = len(depths_rel)
    lines = [f"{n_rows}"]
    for d, r in zip(depths_rel, density_rel):
        lines.append(f"{d:.4f} {r:.4f}")

    output_path.write_text("\n".join(lines) + "\n")
    print(f"  rrd.in: {output_path} ({n_rows} rows)")
    return output_path


def plot_rld_profile(rld_profile, output_path, day=None):
    """Plot RLD profile: depth (pointing down) vs RLD [cm/cm3]."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    depth = rld_profile["depth_mid_cm"]
    rld = rld_profile["RLD_cm_per_cm3"]
    total_len = rld_profile["total_root_length_cm"]
    max_depth = rld_profile["max_root_depth_cm"]
    n_segs = rld_profile["n_root_segments"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6), facecolor='white')

    # Panel 1: RLD vs depth (horizontal bars)
    ax1.barh(depth, rld, height=rld_profile["layer_thickness_cm"] * 0.9,
             color='#8B6914', edgecolor='#5a4510', linewidth=0.5, alpha=0.85)
    ax1.set_ylabel('Depth (cm)')
    ax1.set_xlabel('RLD (cm root / cm$^3$ soil)')
    ax1.invert_yaxis()
    ax1.set_ylim(rld_profile["depth_bot_cm"][-1], 0)
    title = 'Root Length Density Profile'
    if day is not None:
        title += f' — Day {day}'
    ax1.set_title(title)
    ax1.axhline(y=0, color='#666', linewidth=0.5, linestyle='--')

    # Panel 2: Cumulative root length vs depth
    cum_length = np.cumsum(rld_profile["root_length_cm"])
    ax2.plot(cum_length, depth, 'o-', color='#8B6914', markersize=3, linewidth=1.5)
    ax2.set_ylabel('Depth (cm)')
    ax2.set_xlabel('Cumulative Root Length (cm)')
    ax2.invert_yaxis()
    ax2.set_ylim(rld_profile["depth_bot_cm"][-1], 0)
    ax2.set_title('Cumulative Root Length')
    ax2.axhline(y=0, color='#666', linewidth=0.5, linestyle='--')

    # Summary annotation
    summary = (
        f"Total root length: {total_len:.0f} cm\n"
        f"Max root depth: {max_depth:.1f} cm\n"
        f"Root segments: {n_segs}\n"
        f"Surface RLD: {rld[0]:.3f} cm/cm$^3$\n"
        f"Ground area: {rld_profile['ground_area_cm2']:.0f} cm$^2$"
    )
    ax2.text(0.95, 0.05, summary, transform=ax2.transAxes, fontsize=9,
             verticalalignment='bottom', horizontalalignment='right',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='#f5f0e0',
                       edgecolor='#8B6914', alpha=0.9))

    plt.tight_layout()
    fig.savefig(str(output_path), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  RLD plot: {output_path}")
    return output_path


def plot_rld_growth_trajectory(profiles_by_day, output_path):
    """Plot RLD profiles across multiple growth stages on one figure.

    Args:
        profiles_by_day: dict mapping day -> rld_profile dict
        output_path: path for PNG output
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import cm

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    days = sorted(profiles_by_day.keys())
    n_days = len(days)
    cmap = cm.get_cmap('YlOrBr', max(n_days, 3))

    fig, axes = plt.subplots(1, 3, figsize=(14, 6), facecolor='white')

    # Panel 1: RLD profiles overlaid
    ax1 = axes[0]
    for i, day in enumerate(days):
        p = profiles_by_day[day]
        ax1.plot(p["RLD_cm_per_cm3"], p["depth_mid_cm"],
                 'o-', color=cmap(i / max(n_days - 1, 1)),
                 markersize=2, linewidth=1.5, label=f'Day {day}')
    ax1.set_xlabel('RLD (cm/cm$^3$)')
    ax1.set_ylabel('Depth (cm)')
    ax1.invert_yaxis()
    ax1.set_title('RLD Profiles by Growth Stage')
    ax1.legend(fontsize=9)
    ax1.axhline(y=0, color='#666', linewidth=0.5, linestyle='--')

    # Panel 2: Total root length over time
    ax2 = axes[1]
    total_lengths = [profiles_by_day[d]["total_root_length_cm"] for d in days]
    ax2.plot(days, total_lengths, 's-', color='#8B6914', markersize=6, linewidth=2)
    ax2.set_xlabel('Day')
    ax2.set_ylabel('Total Root Length (cm)')
    ax2.set_title('Root Length Growth')
    ax2.grid(alpha=0.3)

    # Panel 3: Max root depth over time
    ax3 = axes[2]
    max_depths = [profiles_by_day[d]["max_root_depth_cm"] for d in days]
    ax3.plot(days, max_depths, 'D-', color='#5a4510', markersize=6, linewidth=2)
    ax3.set_xlabel('Day')
    ax3.set_ylabel('Max Root Depth (cm)')
    ax3.set_title('Root Depth Progression')
    ax3.grid(alpha=0.3)

    fig.suptitle('Root System Development', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(str(output_path), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Growth trajectory plot: {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# Leaf Area Index (LAI) profile extraction
# ---------------------------------------------------------------------------

def extract_lai_profile(plant, n_bins=10, row_spacing_cm=75.0,
                        plant_spacing_cm=20.0):
    """Extract leaf area index profile from a grown plant.

    Computes LAI by binning one-sided leaf blade area into vertical height
    layers.  Does NOT use SegmentAnalyser.distribution("surface", ...) which
    gives cylindrical surface (2*pi*r*L), not blade area.

    Args:
        plant: pb.MappedPlant (grown)
        n_bins: number of vertical height bins
        row_spacing_cm: inter-row spacing [cm]
        plant_spacing_cm: intra-row spacing [cm]

    Returns:
        dict with LAI, total leaf area, vertical profile arrays, etc.
    """
    # Per-segment one-sided blade area [cm2]
    lbs = np.array(plant.leafBladeSurface)
    ot_arr = np.array(plant.organTypes)
    leaf_mask = (ot_arr == 4)

    # Segment z-midpoints
    nodes = plant.getNodes()
    segs = plant.getSegments()
    z_mids = np.zeros(len(segs))
    for i, seg in enumerate(segs):
        z0 = nodes[seg.x].z
        z1 = nodes[seg.y].z
        z_mids[i] = (z0 + z1) / 2.0

    # Only leaf segments above ground
    leaf_z = z_mids[leaf_mask]
    leaf_area = lbs[leaf_mask]

    # Plant height from all above-ground nodes
    all_z = np.array([n.z for n in nodes])
    plant_height = float(np.max(all_z)) if len(all_z) > 0 else 0.0

    # Leaf organ count
    organs = plant.getOrgans()
    n_leaf_organs = sum(1 for o in organs if o.organType() == pb.OrganTypes.leaf)
    n_leaf_segments = int(np.sum(leaf_mask))

    # Ground area and total LAI
    ground_area = row_spacing_cm * plant_spacing_cm  # cm2
    total_leaf_area_cm2 = float(np.sum(leaf_area))
    LAI = total_leaf_area_cm2 / ground_area

    # Vertical bins (from ground to plant height)
    bin_top = max(plant_height, 1.0)  # avoid zero-height edge case
    height_bot = np.linspace(0, bin_top - bin_top / n_bins, n_bins)
    height_top = height_bot + bin_top / n_bins
    height_mid = (height_bot + height_top) / 2.0

    # Bin leaf areas
    leaf_area_per_bin = np.zeros(n_bins)
    bin_edges = np.linspace(0, bin_top, n_bins + 1)
    for i in range(n_bins):
        mask = (leaf_z >= bin_edges[i]) & (leaf_z < bin_edges[i + 1])
        leaf_area_per_bin[i] = float(np.sum(leaf_area[mask]))
    # Top bin: include segments exactly at plant_height
    mask_top = (leaf_z >= bin_edges[-2]) & (leaf_z <= bin_edges[-1])
    leaf_area_per_bin[-1] = float(np.sum(leaf_area[mask_top]))

    # LAI per bin
    LAI_per_bin = leaf_area_per_bin / ground_area

    return {
        "LAI": LAI,
        "total_leaf_area_cm2": total_leaf_area_cm2,
        "total_leaf_area_m2": total_leaf_area_cm2 * 1e-4,
        "ground_area_cm2": ground_area,
        "plant_height_cm": plant_height,
        "n_leaf_organs": n_leaf_organs,
        "n_leaf_segments": n_leaf_segments,
        "height_mid_cm": height_mid,
        "height_top_cm": height_top,
        "height_bot_cm": height_bot,
        "leaf_area_per_bin_cm2": leaf_area_per_bin,
        "LAI_per_bin": LAI_per_bin,
    }


def export_lai_csv(lai_profile, output_path):
    """Export LAI profile to CSV.

    Columns: height_bot_cm, height_top_cm, height_mid_cm,
             leaf_area_cm2, LAI_per_bin
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    header = "height_bot_cm,height_top_cm,height_mid_cm,leaf_area_cm2,LAI_per_bin"
    rows = []
    for i in range(len(lai_profile["height_mid_cm"])):
        rows.append(
            f"{lai_profile['height_bot_cm'][i]:.2f},"
            f"{lai_profile['height_top_cm'][i]:.2f},"
            f"{lai_profile['height_mid_cm'][i]:.2f},"
            f"{lai_profile['leaf_area_per_bin_cm2'][i]:.4f},"
            f"{lai_profile['LAI_per_bin'][i]:.6f}"
        )
    output_path.write_text(header + "\n" + "\n".join(rows))
    print(f"  LAI CSV: {output_path} ({len(rows)} bins)")
    return output_path


def plot_lai_profile(lai_profile, output_path, day=None):
    """Plot LAI profile: height vs leaf area and cumulative LAI."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    height_mid = lai_profile["height_mid_cm"]
    leaf_area = lai_profile["leaf_area_per_bin_cm2"]
    lai_per_bin = lai_profile["LAI_per_bin"]
    bin_thickness = lai_profile["height_top_cm"][0] - lai_profile["height_bot_cm"][0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6), facecolor='white')

    # Panel 1: Leaf area per bin vs height (horizontal bars)
    ax1.barh(height_mid, leaf_area, height=bin_thickness * 0.9,
             color='#2e7d32', edgecolor='#1b5e20', linewidth=0.5, alpha=0.85)
    ax1.set_ylabel('Height (cm)')
    ax1.set_xlabel('Leaf area per bin (cm$^2$)')
    title = 'Leaf Area Profile'
    if day is not None:
        title += f' — Day {day}'
    ax1.set_title(title)
    ax1.set_ylim(0, lai_profile["plant_height_cm"] * 1.05)
    ax1.axhline(y=0, color='#666', linewidth=0.5, linestyle='--')

    # Panel 2: Cumulative LAI from ground up
    cum_lai = np.cumsum(lai_per_bin)
    ax2.plot(cum_lai, height_mid, 'o-', color='#2e7d32', markersize=3, linewidth=1.5)
    ax2.set_ylabel('Height (cm)')
    ax2.set_xlabel('Cumulative LAI')
    ax2.set_title('Cumulative LAI from Ground')
    ax2.set_ylim(0, lai_profile["plant_height_cm"] * 1.05)
    ax2.axhline(y=0, color='#666', linewidth=0.5, linestyle='--')

    # Summary annotation
    summary = (
        f"Total LAI: {lai_profile['LAI']:.2f}\n"
        f"Total leaf area: {lai_profile['total_leaf_area_cm2']:.0f} cm$^2$\n"
        f"Plant height: {lai_profile['plant_height_cm']:.1f} cm\n"
        f"Leaf organs: {lai_profile['n_leaf_organs']}\n"
        f"Ground area: {lai_profile['ground_area_cm2']:.0f} cm$^2$"
    )
    ax2.text(0.95, 0.05, summary, transform=ax2.transAxes, fontsize=9,
             verticalalignment='bottom', horizontalalignment='right',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='#e8f5e9',
                       edgecolor='#2e7d32', alpha=0.9))

    plt.tight_layout()
    fig.savefig(str(output_path), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  LAI plot: {output_path}")
    return output_path


def extract_plant_summary(plant, hm, carbon_result, lai_profile,
                          day, par_umol=1000.0, tair_c=25.0):
    """Assemble a unified plant summary dict from all session outputs.

    Args:
        plant: pb.MappedPlant (grown)
        hm: PhloemFluxPython (after solve) or None
        carbon_result: dict from solve_carbon_partitioning
        lai_profile: dict from extract_lai_profile
        day: simulation day
        par_umol: PAR used [umol m-2 s-1]
        tair_c: air temperature [C]

    Returns:
        dict suitable for JSON serialization.
    """
    from ..config import get_species_name

    # --- From LAI profile ---
    summary = {
        "species": get_species_name(),
        "day": int(day),
        "par_umol_m2_s": float(par_umol),
        "tair_C": float(tair_c),
        "LAI": float(lai_profile["LAI"]),
        "plant_height_cm": float(lai_profile["plant_height_cm"]),
        "total_leaf_area_cm2": float(lai_profile["total_leaf_area_cm2"]),
        "total_leaf_area_m2": float(lai_profile["total_leaf_area_m2"]),
        "n_leaf_organs": int(lai_profile["n_leaf_organs"]),
        "n_leaf_segments": int(lai_profile["n_leaf_segments"]),
        "ground_area_cm2": float(lai_profile["ground_area_cm2"]),
    }

    # --- From plant: root stats ---
    root_organs = [o for o in plant.getOrgans()
                   if o.organType() == pb.OrganTypes.root]
    n_root_segs = sum(len(o.getNodes()) - 1 for o in root_organs)
    root_z = []
    for o in root_organs:
        for n in o.getNodes():
            root_z.append(n.z)
    max_root_depth = abs(min(root_z)) if root_z else 0.0

    ana = pb.SegmentAnalyser(plant)
    ana.filter("organType", pb.root)
    total_root_length = float(ana.getSummed("length"))

    summary["total_root_length_cm"] = float(total_root_length)
    summary["max_root_depth_cm"] = float(max_root_depth)
    summary["n_root_segments"] = int(n_root_segs)

    # --- From photosynthesis model ---
    if hm is not None:
        An_leaf = np.array(hm.get_net_assimilation())
        An_total_mmol = float(np.sum(An_leaf)) * 1000.0
        transp_raw = np.sum(hm.get_transpiration())
        transp_mmol = float(transp_raw) / 18.0 * 1000.0
    else:
        An_total_mmol = 0.0
        transp_mmol = 0.0

    summary["An_total_mmol_CO2_d"] = float(An_total_mmol)
    summary["transpiration_mmol_H2O_d"] = float(transp_mmol)

    # --- From carbon partitioning ---
    if carbon_result is not None:
        summary["Rm_total_mmol"] = float(carbon_result.get("Rm_total_mmol", 0.0))
        summary["Rg_total_mmol"] = float(carbon_result.get("Rg_total_mmol", 0.0))
        summary["FR_leaf"] = float(carbon_result.get("FR_leaf", 0.0))
        summary["FR_stem"] = float(carbon_result.get("FR_stem", 0.0))
        summary["FR_root"] = float(carbon_result.get("FR_root", 0.0))
        summary["FR_storage"] = float(carbon_result.get("FR_storage", 0.0))
        summary["growth_mmol_d"] = float(carbon_result.get("growth_mmol_d", 0.0))
        summary["carbon_balance_error"] = float(carbon_result.get("carbon_balance_error", 0.0))
        summary["partitioning_source"] = str(carbon_result.get("partitioning_source", "unknown"))
    else:
        summary["Rm_total_mmol"] = 0.0
        summary["Rg_total_mmol"] = 0.0
        summary["FR_leaf"] = 0.0
        summary["FR_stem"] = 0.0
        summary["FR_root"] = 0.0
        summary["FR_storage"] = 0.0
        summary["growth_mmol_d"] = 0.0
        summary["carbon_balance_error"] = 0.0
        summary["partitioning_source"] = "none"

    return summary


def plot_growth_trajectory(summaries_by_day, output_path):
    """Plot growth trajectory across multiple days (2x2 panels).

    Args:
        summaries_by_day: dict mapping day (int) -> summary dict
        output_path: path for PNG output
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    days = sorted(summaries_by_day.keys())
    lais = [summaries_by_day[d]["LAI"] for d in days]
    heights = [summaries_by_day[d]["plant_height_cm"] for d in days]
    an_totals = [summaries_by_day[d]["An_total_mmol_CO2_d"] for d in days]
    root_depths = [summaries_by_day[d]["max_root_depth_cm"] for d in days]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), facecolor='white')

    # Panel 1: LAI vs day
    ax1 = axes[0, 0]
    ax1.plot(days, lais, 's-', color='#2e7d32', markersize=6, linewidth=2)
    ax1.set_xlabel('Day')
    ax1.set_ylabel('LAI')
    ax1.set_title('Leaf Area Index')
    ax1.grid(alpha=0.3)

    # Panel 2: Plant height vs day
    ax2 = axes[0, 1]
    ax2.plot(days, heights, 'D-', color='#1565C0', markersize=6, linewidth=2)
    ax2.set_xlabel('Day')
    ax2.set_ylabel('Plant Height (cm)')
    ax2.set_title('Plant Height')
    ax2.grid(alpha=0.3)

    # Panel 3: An total vs day
    ax3 = axes[1, 0]
    ax3.plot(days, an_totals, 'o-', color='#E65100', markersize=6, linewidth=2)
    ax3.set_xlabel('Day')
    ax3.set_ylabel('An total (mmol CO$_2$ d$^{-1}$)')
    ax3.set_title('Net Assimilation')
    ax3.grid(alpha=0.3)

    # Panel 4: Max root depth vs day
    ax4 = axes[1, 1]
    ax4.plot(days, root_depths, '^-', color='#8B6914', markersize=6, linewidth=2)
    ax4.set_xlabel('Day')
    ax4.set_ylabel('Max Root Depth (cm)')
    ax4.set_title('Root Depth')
    ax4.grid(alpha=0.3)

    species = summaries_by_day[days[0]].get("species", "maize")
    fig.suptitle(f'{species.capitalize()} Growth Trajectory',
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(str(output_path), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Growth trajectory plot: {output_path}")
    return output_path


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


def plot_photosynthesis(organ_data, An_per_umol, hx_all, seg_leaves_idx,
                        An_total_mmol, transp, par_umol, tair_c, rh,
                        sim_time, output_prefix):
    """Create a multi-panel photosynthesis summary figure."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    n_organs = len(organ_data)
    positions = [d['index'] for d in organ_data]
    subtypes = [d['subtype'] for d in organ_data]
    lengths = [d['length'] for d in organ_data]
    areas = [d['blade_area'] for d in organ_data]
    An_sums = [d['An_sum_mmol'] for d in organ_data]
    psi_means = [d['psi_mean'] for d in organ_data]

    fig = plt.figure(figsize=(14, 10), facecolor='white')
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.35,
                  left=0.07, right=0.97, top=0.90, bottom=0.08)

    colors = [LEAF_GREENS[i % len(LEAF_GREENS)] for i in range(n_organs)]
    labels = [f"L{d['subtype']}" for d in organ_data]

    # --- Panel 1: An per organ (bar) ---
    ax1 = fig.add_subplot(gs[0, 0])
    bars = ax1.bar(range(n_organs), An_sums, color=colors, edgecolor='#333', linewidth=0.5)
    ax1.set_xticks(range(n_organs))
    ax1.set_xticklabels(labels, fontsize=8)
    ax1.set_ylabel('An (mmol CO$_2$ d$^{-1}$)')
    ax1.set_title('Net Assimilation per Leaf')
    ax1.set_xlabel('Leaf (by subType)')
    for bar, val in zip(bars, An_sums):
        if val > 0:
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                     f'{val:.0f}', ha='center', va='bottom', fontsize=7)

    # --- Panel 2: Blade area per organ ---
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.bar(range(n_organs), areas, color=colors, edgecolor='#333', linewidth=0.5)
    ax2.set_xticks(range(n_organs))
    ax2.set_xticklabels(labels, fontsize=8)
    ax2.set_ylabel('Blade area (cm$^2$, both sides)')
    ax2.set_title('Leaf Blade Area')
    ax2.set_xlabel('Leaf (by subType)')

    # --- Panel 3: Leaf length + width ---
    ax3 = fig.add_subplot(gs[0, 2])
    widths_full = [d['width'] * 2 for d in organ_data]
    x = np.arange(n_organs)
    w = 0.35
    ax3.bar(x - w/2, lengths, w, color=colors, edgecolor='#333', linewidth=0.5,
            label='Length (cm)')
    ax3.bar(x + w/2, widths_full, w, color=[(c[0]*0.7, c[1]*0.7, c[2]*0.7) for c in colors],
            edgecolor='#333', linewidth=0.5, label='Width (cm)')
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels, fontsize=8)
    ax3.set_ylabel('cm')
    ax3.set_title('Leaf Dimensions')
    ax3.legend(fontsize=8)

    # --- Panel 4: Water potential per organ ---
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.bar(range(n_organs), psi_means, color=colors, edgecolor='#333', linewidth=0.5)
    ax4.set_xticks(range(n_organs))
    ax4.set_xticklabels(labels, fontsize=8)
    ax4.set_ylabel('$\\psi_{xylem}$ (cm)')
    ax4.set_title('Mean Xylem Water Potential')
    ax4.set_xlabel('Leaf (by subType)')

    # --- Panel 5: Water potential distribution (histogram) ---
    ax5 = fig.add_subplot(gs[1, 1])
    # Get psi for all leaf segments
    psi_leaves = []
    for li in range(len(An_per_umol)):
        gi = seg_leaves_idx[li] if li < len(seg_leaves_idx) else -1
        if 0 <= gi < len(hx_all):
            psi_leaves.append(hx_all[gi])
    if psi_leaves:
        ax5.hist(psi_leaves, bins=40, color='#4a90d9', edgecolor='#333',
                 linewidth=0.3, alpha=0.85)
    ax5.set_xlabel('$\\psi_{xylem}$ (cm)')
    ax5.set_ylabel('Count (leaf segments)')
    ax5.set_title('Xylem Potential Distribution')

    # --- Panel 6: Summary text ---
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    active = np.sum(An_per_umol > 0)
    total_area = sum(areas)
    An_mean = np.mean(An_per_umol[An_per_umol > 0]) if active > 0 else 0
    psi_mean_all = np.mean(psi_leaves) if psi_leaves else 0

    summary = (
        f"Day {sim_time:.0f} — C4 Maize\n"
        f"{'─' * 32}\n"
        f"PAR:          {par_umol:.0f} umol m$^{{-2}}$ s$^{{-1}}$\n"
        f"Temperature:  {tair_c:.1f} °C\n"
        f"RH:           {rh*100:.0f}%\n"
        f"Soil $\\psi$:    {-500:.0f} cm\n"
        f"{'─' * 32}\n"
        f"Leaves:       {n_organs}\n"
        f"Leaf segs:    {len(An_per_umol)} ({active} active)\n"
        f"Blade area:   {total_area:.0f} cm$^2$\n"
        f"{'─' * 32}\n"
        f"Total An:     {An_total_mmol:.1f} mmol d$^{{-1}}$\n"
        f"Mean An:      {An_mean:.1f} umol m$^{{-2}}$ s$^{{-1}}$\n"
        f"Transpir.:    {transp:.1f} mmol d$^{{-1}}$\n"
        f"Mean $\\psi$:    {psi_mean_all:.0f} cm\n"
    )
    ax6.text(0.05, 0.95, summary, transform=ax6.transAxes,
             fontsize=10, fontfamily='monospace', verticalalignment='top',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#f0f4f8', edgecolor='#888'))

    fig.suptitle(f'Photosynthesis Summary — Day {sim_time:.0f}',
                 fontsize=14, fontweight='bold', y=0.96)

    png_path = Path(output_prefix).with_name(
        Path(output_prefix).stem + '_summary'
    ).with_suffix('.png')
    fig.savefig(str(png_path), dpi=150)
    plt.close(fig)
    print(f"  Summary plot: {png_path}")


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
# VTK offscreen rendering (adapted from batch_pheno4d_pipeline_perpointnormals.py)
# ---------------------------------------------------------------------------
def _make_renderer(bg=(1, 1, 1)):
    import vtk
    renderer = vtk.vtkRenderer()
    renderer.SetBackground(*bg)
    win = vtk.vtkRenderWindow()
    win.SetSize(PANEL_W, PANEL_H)
    win.SetOffScreenRendering(1)
    win.AddRenderer(renderer)
    return renderer, win


def _set_camera(renderer, bounds, azimuth_deg=30, elevation_deg=20):
    """Set fixed camera from scene bounds."""
    cx = (bounds[0] + bounds[1]) / 2
    cy = (bounds[2] + bounds[3]) / 2
    cz = (bounds[4] + bounds[5]) / 2
    dx = bounds[1] - bounds[0]
    dy = bounds[3] - bounds[2]
    dz = bounds[5] - bounds[4]
    dist = max(dx, dy, dz, 1.0) * 2.2

    az = np.radians(azimuth_deg)
    el = np.radians(elevation_deg)
    cam_x = cx + dist * np.cos(el) * np.sin(az)
    cam_y = cy - dist * np.cos(el) * np.cos(az)
    cam_z = cz + dist * np.sin(el)

    camera = renderer.GetActiveCamera()
    camera.SetPosition(cam_x, cam_y, cam_z)
    camera.SetFocalPoint(cx, cy, cz)
    camera.SetViewUp(0, 0, 1)
    camera.SetParallelProjection(True)
    camera.SetParallelScale(max(dz, max(dx, dy)) * 0.55)
    renderer.ResetCameraClippingRange()


def _render_to_array(win):
    """Render and return as numpy uint8 array (H, W, 3)."""
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy
    win.Render()
    w2i = vtk.vtkWindowToImageFilter()
    w2i.SetInput(win)
    w2i.SetInputBufferTypeToRGB()
    w2i.Update()
    img = w2i.GetOutput()
    w, h, _ = img.GetDimensions()
    scalars = img.GetPointData().GetScalars()
    data = vtk_to_numpy(scalars).reshape(h, w, 3)
    return data[::-1].copy()


def _organ_color(organ_dict, leaf_idx):
    """Get color for an organ dict."""
    if organ_dict['type'] == 'stem':
        return STEM_COLOR
    return LEAF_GREENS[leaf_idx % len(LEAF_GREENS)]


def _compute_bounds(organ_dicts):
    """Compute bounding box across all organ skeletons."""
    all_pts = np.concatenate([o['skeleton'] for o in organ_dicts])
    mins = all_pts.min(axis=0)
    maxs = all_pts.max(axis=0)
    # Add padding
    pad = max(maxs - mins) * 0.05
    return [mins[0]-pad, maxs[0]+pad, mins[1]-pad, maxs[1]+pad, mins[2]-pad, maxs[2]+pad]


def render_g1_panel(organ_dicts, bounds, root_dicts=None):
    """Render G1 skeleton: tubes + node spheres, colored per organ."""
    import vtk

    renderer, win = _make_renderer()

    # Render roots first (behind shoots)
    if root_dicts:
        for idx, organ in enumerate(root_dicts):
            color = ROOT_COLORS[idx % len(ROOT_COLORS)]
            skel = organ['skeleton']
            n = len(skel)
            if n < 2:
                continue

            points = vtk.vtkPoints()
            for pt in skel:
                points.InsertNextPoint(pt)
            lines = vtk.vtkCellArray()
            for i in range(n - 1):
                line = vtk.vtkLine()
                line.GetPointIds().SetId(0, i)
                line.GetPointIds().SetId(1, i + 1)
                lines.InsertNextCell(line)
            polydata = vtk.vtkPolyData()
            polydata.SetPoints(points)
            polydata.SetLines(lines)

            mean_radius = max(np.mean(organ['widths']) / 2.0, 0.02)
            tube = vtk.vtkTubeFilter()
            tube.SetInputData(polydata)
            tube.SetRadius(mean_radius)
            tube.SetNumberOfSides(6)
            tube.Update()

            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(tube.GetOutputPort())
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(*color)
            actor.GetProperty().SetAmbient(0.3)
            actor.GetProperty().SetDiffuse(0.7)
            renderer.AddActor(actor)

    # Render shoot organs
    leaf_idx = 0
    for organ in organ_dicts:
        if organ['type'] == 'stem':
            color = STEM_COLOR
        else:
            color = LEAF_GREENS[leaf_idx % len(LEAF_GREENS)]
            leaf_idx += 1

        skel = organ['skeleton']
        n = len(skel)
        if n < 2:
            continue

        # Skeleton tube
        points = vtk.vtkPoints()
        for pt in skel:
            points.InsertNextPoint(pt)

        lines = vtk.vtkCellArray()
        for i in range(n - 1):
            line = vtk.vtkLine()
            line.GetPointIds().SetId(0, i)
            line.GetPointIds().SetId(1, i + 1)
            lines.InsertNextCell(line)

        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points)
        polydata.SetLines(lines)

        tube = vtk.vtkTubeFilter()
        tube.SetInputData(polydata)
        tube.SetRadius(0.08)
        tube.SetNumberOfSides(8)
        tube.Update()

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(tube.GetOutputPort())
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(*color)
        renderer.AddActor(actor)

        # Node spheres (every ~10th)
        step = max(1, n // 10)
        for i in range(0, n, step):
            sphere = vtk.vtkSphereSource()
            sphere.SetCenter(skel[i])
            sphere.SetRadius(0.12)
            sphere.SetThetaResolution(8)
            sphere.SetPhiResolution(8)
            m = vtk.vtkPolyDataMapper()
            m.SetInputConnection(sphere.GetOutputPort())
            a = vtk.vtkActor()
            a.SetMapper(m)
            a.GetProperty().SetColor(*color)
            renderer.AddActor(a)

        # Width indicators for leaves (every ~10th node)
        if organ['type'] == 'leaf':
            widths = organ['widths']
            step_w = max(1, n // 10)
            for i in range(0, n, step_w):
                if i == 0:
                    tangent = skel[min(1, n-1)] - skel[0]
                elif i == n - 1:
                    tangent = skel[-1] - skel[max(-2, -n)]
                else:
                    tangent = skel[min(i+1, n-1)] - skel[max(i-1, 0)]
                t_len = np.linalg.norm(tangent)
                if t_len < 1e-12:
                    continue
                tangent /= t_len

                binormal = np.cross(np.array([0, 0, 1.0]), tangent)
                bn_len = np.linalg.norm(binormal)
                if bn_len < 1e-6:
                    binormal = np.cross(np.array([1, 0, 0.0]), tangent)
                    bn_len = np.linalg.norm(binormal)
                binormal /= (bn_len + 1e-12)

                half_w = widths[i] / 2.0
                p1 = skel[i] - binormal * half_w
                p2 = skel[i] + binormal * half_w

                pts_w = vtk.vtkPoints()
                pts_w.InsertNextPoint(p1)
                pts_w.InsertNextPoint(p2)
                ln = vtk.vtkLine()
                ln.GetPointIds().SetId(0, 0)
                ln.GetPointIds().SetId(1, 1)
                lns = vtk.vtkCellArray()
                lns.InsertNextCell(ln)
                pd = vtk.vtkPolyData()
                pd.SetPoints(pts_w)
                pd.SetLines(lns)

                tb = vtk.vtkTubeFilter()
                tb.SetInputData(pd)
                tb.SetRadius(0.04)
                tb.SetNumberOfSides(6)
                tb.Update()

                m2 = vtk.vtkPolyDataMapper()
                m2.SetInputConnection(tb.GetOutputPort())
                a2 = vtk.vtkActor()
                a2.SetMapper(m2)
                a2.GetProperty().SetColor(*color)
                a2.GetProperty().SetOpacity(0.6)
                renderer.AddActor(a2)

    _set_camera(renderer, bounds)
    return _render_to_array(win)


def render_g3_panel(mesh, bounds):
    """Render G3 mesh with per-organ coloring and lighting."""
    import vtk

    renderer, win = _make_renderer()

    if mesh is None:
        _set_camera(renderer, bounds)
        return _render_to_array(win)

    polydata = mesh.to_vtk_polydata()

    # Per-cell colors from organ IDs
    n_cells = polydata.GetNumberOfCells()
    colors = vtk.vtkUnsignedCharArray()
    colors.SetNumberOfComponents(3)
    colors.SetName("OrganColors")

    organ_ids = mesh.organ_ids
    id_to_color = {}
    leaf_idx = 0
    for meta in mesh.organ_meta:
        oid = meta["organ_id"]
        if meta.get("type") == "stem":
            id_to_color[oid] = STEM_COLOR
        else:
            id_to_color[oid] = LEAF_GREENS[leaf_idx % len(LEAF_GREENS)]
            leaf_idx += 1

    for i in range(n_cells):
        oid = int(organ_ids[i])
        c = id_to_color.get(oid, (0.5, 0.5, 0.5))
        colors.InsertNextTuple3(int(c[0]*255), int(c[1]*255), int(c[2]*255))

    polydata.GetCellData().SetScalars(colors)

    # Smooth normals
    normals_filter = vtk.vtkPolyDataNormals()
    normals_filter.SetInputData(polydata)
    normals_filter.ComputePointNormalsOn()
    normals_filter.ComputeCellNormalsOn()
    normals_filter.AutoOrientNormalsOn()
    normals_filter.ConsistencyOn()
    normals_filter.SplittingOff()
    normals_filter.Update()

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(normals_filter.GetOutput())
    mapper.SetScalarModeToUseCellData()

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().BackfaceCullingOff()
    actor.GetProperty().SetAmbient(0.3)
    actor.GetProperty().SetDiffuse(0.7)
    renderer.AddActor(actor)

    _set_camera(renderer, bounds)
    return _render_to_array(win)


ROOT_COLOR = (0.72, 0.53, 0.04)  # dark gold / tan
ROOT_COLORS = [
    (0.72, 0.53, 0.04),  # root 1
    (0.82, 0.63, 0.14),  # root 2
    (0.65, 0.45, 0.02),  # root 3
    (0.75, 0.58, 0.10),  # root 4
    (0.60, 0.42, 0.08),  # root 5
]


def render_root_panel(root_dicts, bounds):
    """Render root skeleton as tubes, colored per root organ."""
    import vtk

    renderer, win = _make_renderer()

    if not root_dicts:
        _set_camera(renderer, bounds)
        return _render_to_array(win)

    for idx, organ in enumerate(root_dicts):
        color = ROOT_COLORS[idx % len(ROOT_COLORS)]
        skel = organ['skeleton']
        n = len(skel)
        if n < 2:
            continue

        points = vtk.vtkPoints()
        for pt in skel:
            points.InsertNextPoint(pt)

        lines = vtk.vtkCellArray()
        for i in range(n - 1):
            line = vtk.vtkLine()
            line.GetPointIds().SetId(0, i)
            line.GetPointIds().SetId(1, i + 1)
            lines.InsertNextCell(line)

        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points)
        polydata.SetLines(lines)

        # Use mean radius for the tube
        mean_radius = max(np.mean(organ['widths']) / 2.0, 0.02)

        tube = vtk.vtkTubeFilter()
        tube.SetInputData(polydata)
        tube.SetRadius(mean_radius)
        tube.SetNumberOfSides(6)
        tube.Update()

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(tube.GetOutputPort())
        mapper.ScalarVisibilityOff()
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(*color)
        actor.GetProperty().SetAmbient(0.3)
        actor.GetProperty().SetDiffuse(0.7)
        renderer.AddActor(actor)

    # Add a ground plane line at z=0
    ground = vtk.vtkLineSource()
    ground.SetPoint1(bounds[0], (bounds[2]+bounds[3])/2, 0)
    ground.SetPoint2(bounds[1], (bounds[2]+bounds[3])/2, 0)
    gm = vtk.vtkPolyDataMapper()
    gm.SetInputConnection(ground.GetOutputPort())
    ga = vtk.vtkActor()
    ga.SetMapper(gm)
    ga.GetProperty().SetColor(0.4, 0.4, 0.4)
    ga.GetProperty().SetLineWidth(2)
    renderer.AddActor(ga)

    _set_camera(renderer, bounds)
    return _render_to_array(win)


def render_comparison_png(organ_dicts, mesh, output_prefix, days, root_dicts=None):
    """Render side-by-side G1 (full plant) | G3 (shoot mesh) comparison PNG."""
    from PIL import Image, ImageDraw, ImageFont

    print(f"\n=== Rendering G1 Full Plant | G3 Shoot Mesh Comparison ===")

    # Compute bounds covering both shoot and root organs
    all_organ_lists = [organ_dicts]
    if root_dicts:
        all_organ_lists.append(root_dicts)
    all_pts = np.concatenate([o['skeleton'] for ol in all_organ_lists for o in ol])
    mins = all_pts.min(axis=0)
    maxs = all_pts.max(axis=0)
    pad = max(maxs - mins) * 0.05
    full_bounds = [mins[0]-pad, maxs[0]+pad, mins[1]-pad, maxs[1]+pad,
                   mins[2]-pad, maxs[2]+pad]

    g1_img = render_g1_panel(organ_dicts, full_bounds, root_dicts=root_dicts)
    # G3 mesh uses shoot-only bounds (no roots in mesh)
    shoot_bounds = _compute_bounds(organ_dicts)
    g3_img = render_g3_panel(mesh, shoot_bounds)

    # Concatenate: [G1 | G3]
    row = np.concatenate([g1_img, g3_img], axis=1)
    img = Image.fromarray(row)

    # Add header labels
    header_h = 35
    total_w = PANEL_W * 2
    total_h = PANEL_H + header_h
    canvas = Image.new("RGB", (total_w, total_h), (255, 255, 255))

    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/TTF/DejaVuSansMono.ttf", 16)
    except (OSError, IOError):
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 16)
        except (OSError, IOError):
            font = ImageFont.load_default()

    # Column headers
    draw.text((PANEL_W // 2 - 80, 8), "G1 Full Plant", fill=(30, 30, 30), font=font)
    draw.text((PANEL_W + PANEL_W // 2 - 60, 8), "G3 Shoot Mesh", fill=(30, 30, 30), font=font)

    # Title
    try:
        font_title = ImageFont.truetype("/usr/share/fonts/TTF/DejaVuSansMono-Bold.ttf", 14)
    except (OSError, IOError):
        font_title = font
    draw.text((5, 8), f"Day {days:.0f}", fill=(100, 100, 100), font=font_title)

    canvas.paste(img, (0, header_h))

    png_path = Path(output_prefix).with_suffix('.png')
    canvas.save(str(png_path), quality=95)
    print(f"  PNG: {png_path} ({total_w}x{total_h})")

    return png_path


# ---------------------------------------------------------------------------
# SVG export (vector graphics, front-view orthographic projection)
# ---------------------------------------------------------------------------

def _rgb_hex(r, g, b):
    """Convert float (0-1) RGB to hex string."""
    return f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}'


def render_comparison_svg(organ_dicts, mesh, output_prefix, days, root_dicts=None):
    """Render side-by-side G1 skeleton | G3 mesh as SVG.

    Front-view orthographic projection (XZ plane).
    G1 panel: polylines colored per organ.
    G3 panel: filled triangles with painter's algorithm and flat shading.
    """
    print(f"\n=== Rendering G1 | G3 Comparison SVG ===")

    # --- Layout constants ---
    panel_w, panel_h = 400.0, 600.0
    margin = 20.0
    header_h = 30.0
    gap = 20.0
    total_w = panel_w * 2 + gap
    total_h = panel_h + header_h

    # --- Compute bounds covering all visible geometry ---
    shoot_pts = np.concatenate([o['skeleton'] for o in organ_dicts])
    mesh_pts = mesh.vertices
    all_pts = [shoot_pts, mesh_pts]
    if root_dicts:
        root_pts = np.concatenate([o['skeleton'] for o in root_dicts])
        all_pts.append(root_pts)
    combined = np.concatenate(all_pts)

    x_min = combined[:, 0].min()
    x_max = combined[:, 0].max()
    z_min = combined[:, 2].min()
    z_max = combined[:, 2].max()

    data_w = max(x_max - x_min, 0.1)
    data_h = max(z_max - z_min, 0.1)

    usable_w = panel_w - 2 * margin
    usable_h = panel_h - 2 * margin
    scale = min(usable_w / data_w, usable_h / data_h)

    # Centering offsets
    off_x = (usable_w - data_w * scale) / 2
    off_y = (usable_h - data_h * scale) / 2

    def proj(x, z, panel_off=0.0):
        """Project world (x, z) -> SVG (sx, sy) in the given panel."""
        sx = margin + off_x + (x - x_min) * scale + panel_off
        sy = header_h + margin + off_y + (z_max - z) * scale
        return sx, sy

    # --- Build SVG as string buffer (efficient for 60k+ triangles) ---
    buf = []
    buf.append('<?xml version="1.0" encoding="UTF-8"?>')
    buf.append(f'<svg xmlns="http://www.w3.org/2000/svg" '
               f'viewBox="0 0 {total_w:.1f} {total_h:.1f}" '
               f'width="{total_w:.0f}" height="{total_h:.0f}">')

    # Background
    buf.append(f'<rect x="0" y="0" width="{total_w:.1f}" height="{total_h:.1f}" fill="white"/>')

    # Panel backgrounds
    for px in [0, panel_w + gap]:
        buf.append(f'<rect x="{px:.1f}" y="{header_h:.1f}" '
                   f'width="{panel_w:.1f}" height="{panel_h:.1f}" '
                   f'fill="#f8f8f8" stroke="#ccc" stroke-width="0.5"/>')

    # Headers
    buf.append(f'<text x="{panel_w/2:.1f}" y="20" text-anchor="middle" '
               f'font-family="Times New Roman, Times, serif" font-size="13" fill="#333">G1 Skeleton</text>')
    buf.append(f'<text x="{panel_w + gap + panel_w/2:.1f}" y="20" text-anchor="middle" '
               f'font-family="Times New Roman, Times, serif" font-size="13" fill="#333">G3 Mesh</text>')
    buf.append(f'<text x="5" y="20" font-family="Times New Roman, Times, serif" font-size="11" '
               f'fill="#999">Day {days:.0f}</text>')

    # --- G1 Panel: polylines ---
    buf.append('<g id="g1-panel">')

    # Ground line at z=0
    gx0, gy0 = proj(x_min, 0.0)
    gx1, gy1 = proj(x_max, 0.0)
    buf.append(f'<line x1="{gx0:.1f}" y1="{gy0:.1f}" x2="{gx1:.1f}" y2="{gy1:.1f}" '
               f'stroke="#aaa" stroke-width="0.5" stroke-dasharray="4,3"/>')

    # Roots (behind shoot)
    if root_dicts:
        for idx, organ in enumerate(root_dicts):
            skel = organ['skeleton']
            if len(skel) < 2:
                continue
            color = ROOT_COLORS[idx % len(ROOT_COLORS)]
            pts = ' '.join(f'{proj(p[0], p[2])[0]:.1f},{proj(p[0], p[2])[1]:.1f}'
                           for p in skel)
            buf.append(f'<polyline points="{pts}" stroke="{_rgb_hex(*color)}" '
                       f'stroke-width="1.2" fill="none" '
                       f'stroke-linecap="round" stroke-linejoin="round"/>')

    # Shoot organs
    leaf_idx = 0
    for organ in organ_dicts:
        skel = organ['skeleton']
        if len(skel) < 2:
            continue
        if organ['type'] == 'stem':
            color = STEM_COLOR
            sw = '3'
        else:
            color = LEAF_GREENS[leaf_idx % len(LEAF_GREENS)]
            leaf_idx += 1
            sw = '2'
        pts = ' '.join(f'{proj(p[0], p[2])[0]:.1f},{proj(p[0], p[2])[1]:.1f}'
                       for p in skel)
        buf.append(f'<polyline points="{pts}" stroke="{_rgb_hex(*color)}" '
                   f'stroke-width="{sw}" fill="none" '
                   f'stroke-linecap="round" stroke-linejoin="round"/>')

    buf.append('</g>')

    # --- G3 Panel: filled triangles with painter's algorithm ---
    buf.append('<g id="g3-panel">')

    # Ground line
    gx0, gy0 = proj(x_min, 0.0, panel_w + gap)
    gx1, gy1 = proj(x_max, 0.0, panel_w + gap)
    buf.append(f'<line x1="{gx0:.1f}" y1="{gy0:.1f}" x2="{gx1:.1f}" y2="{gy1:.1f}" '
               f'stroke="#aaa" stroke-width="0.5" stroke-dasharray="4,3"/>')

    verts = mesh.vertices
    tris = mesh.indices
    oids = mesh.organ_ids

    # Build organ_id -> base color
    id_to_color = {}
    leaf_idx = 0
    for organ in organ_dicts:
        oid = organ['organ_id']
        if organ['type'] == 'stem':
            id_to_color[oid] = STEM_COLOR
        else:
            id_to_color[oid] = LEAF_GREENS[leaf_idx % len(LEAF_GREENS)]
            leaf_idx += 1

    # Triangle vertices (vectorized)
    v0 = verts[tris[:, 0]]
    v1 = verts[tris[:, 1]]
    v2 = verts[tris[:, 2]]

    # Depth sort: front view looks from +Y toward -Y
    # Painter's algorithm: draw far (small Y) first, near (large Y) on top
    centroids_y = (v0[:, 1] + v1[:, 1] + v2[:, 1]) / 3.0
    order = np.argsort(centroids_y)

    # Flat shading with directional light
    e1 = v1 - v0
    e2 = v2 - v0
    normals = np.cross(e1, e2)
    nlen = np.linalg.norm(normals, axis=1, keepdims=True)
    nlen[nlen == 0] = 1
    normals /= nlen

    light = np.array([0.3, -0.8, 0.5])
    light /= np.linalg.norm(light)
    shade = np.abs(np.dot(normals, light))
    shade = 0.35 + 0.65 * shade  # ambient + diffuse

    panel_off = panel_w + gap
    n_tris = len(tris)
    print(f"  Writing {n_tris} triangles to SVG...")

    for tri_idx in order:
        tri = tris[tri_idx]
        oid = int(oids[tri_idx])
        bc = id_to_color.get(oid, (0.5, 0.5, 0.5))
        s = float(shade[tri_idx])

        r = min(1.0, bc[0] * s)
        g = min(1.0, bc[1] * s)
        b = min(1.0, bc[2] * s)
        fill = _rgb_hex(r, g, b)

        p0 = proj(verts[tri[0], 0], verts[tri[0], 2], panel_off)
        p1 = proj(verts[tri[1], 0], verts[tri[1], 2], panel_off)
        p2 = proj(verts[tri[2], 0], verts[tri[2], 2], panel_off)
        buf.append(f'<polygon points="{p0[0]:.1f},{p0[1]:.1f} '
                   f'{p1[0]:.1f},{p1[1]:.1f} '
                   f'{p2[0]:.1f},{p2[1]:.1f}" fill="{fill}"/>')

    buf.append('</g>')
    buf.append('</svg>')

    svg_path = Path(output_prefix).with_suffix('.svg')
    with open(svg_path, 'w') as f:
        f.write('\n'.join(buf))

    size_mb = svg_path.stat().st_size / (1024 * 1024)
    print(f"  SVG: {svg_path} ({size_mb:.1f} MB, {n_tris} triangles)")
    return svg_path


def render_publication_svg(organ_dicts, mesh, output_prefix, days, root_dicts=None):
    """Render publication-quality SVG: white background, (a)/(b) labels, scale bar.

    Clean layout suitable for journal figures. No panel borders or backgrounds.
    """
    print(f"\n=== Rendering Publication SVG ===")

    font = 'Times New Roman, Times, serif'

    # --- Compute bounds ---
    shoot_pts = np.concatenate([o['skeleton'] for o in organ_dicts])
    mesh_pts = mesh.vertices
    all_pts = [shoot_pts, mesh_pts]
    if root_dicts:
        root_pts = np.concatenate([o['skeleton'] for o in root_dicts])
        all_pts.append(root_pts)
    combined = np.concatenate(all_pts)

    x_min = float(combined[:, 0].min())
    x_max = float(combined[:, 0].max())
    z_min = float(combined[:, 2].min())
    z_max = float(combined[:, 2].max())

    # --- Layout ---
    panel_w, panel_h = 380.0, 620.0
    margin = 25.0
    header_h = 28.0
    gap = 40.0  # wider gap for clean separation
    footer_h = 30.0  # space for scale bar
    total_w = panel_w * 2 + gap
    total_h = panel_h + header_h + footer_h

    data_w = max(x_max - x_min, 0.1)
    data_h = max(z_max - z_min, 0.1)
    usable_w = panel_w - 2 * margin
    usable_h = panel_h - 2 * margin
    scale = min(usable_w / data_w, usable_h / data_h)
    off_x = (usable_w - data_w * scale) / 2
    off_y = (usable_h - data_h * scale) / 2

    def proj(x, z, panel_off=0.0):
        sx = margin + off_x + (x - x_min) * scale + panel_off
        sy = header_h + margin + off_y + (z_max - z) * scale
        return sx, sy

    # --- Determine scale bar ---
    # Pick a round number of cm that fills ~20-30% of panel width
    world_per_px = 1.0 / scale
    target_px = usable_w * 0.25
    target_cm = target_px * world_per_px
    # Round to nearest nice number
    nice = [1, 2, 5, 10, 20, 25, 50, 100]
    bar_cm = min(nice, key=lambda n: abs(n - target_cm))
    bar_px = bar_cm * scale

    # --- Build SVG ---
    buf = []
    buf.append('<?xml version="1.0" encoding="UTF-8"?>')
    buf.append(f'<svg xmlns="http://www.w3.org/2000/svg" '
               f'viewBox="0 0 {total_w:.1f} {total_h:.1f}" '
               f'width="{total_w:.0f}" height="{total_h:.0f}">')

    # Pure white background
    buf.append(f'<rect x="0" y="0" width="{total_w:.1f}" height="{total_h:.1f}" fill="white"/>')

    # --- Panel labels: (a) and (b) ---
    buf.append(f'<text x="{margin - 5:.1f}" y="20" '
               f'font-family="{font}" font-size="16" font-weight="bold" '
               f'fill="#000">(a)</text>')
    buf.append(f'<text x="{panel_w + gap + margin - 5:.1f}" y="20" '
               f'font-family="{font}" font-size="16" font-weight="bold" '
               f'fill="#000">(b)</text>')

    # Sub-labels (italic)
    buf.append(f'<text x="{margin + 25:.1f}" y="20" '
               f'font-family="{font}" font-size="12" font-style="italic" '
               f'fill="#555">G1 Skeleton</text>')
    buf.append(f'<text x="{panel_w + gap + margin + 25:.1f}" y="20" '
               f'font-family="{font}" font-size="12" font-style="italic" '
               f'fill="#555">G3 Triangle Mesh</text>')

    # --- G1 Panel ---
    buf.append('<g id="g1-panel">')

    # Subtle ground line at z=0
    gx0, gy0 = proj(x_min, 0.0)
    gx1, gy1 = proj(x_max, 0.0)
    buf.append(f'<line x1="{gx0:.1f}" y1="{gy0:.1f}" x2="{gx1:.1f}" y2="{gy1:.1f}" '
               f'stroke="#ccc" stroke-width="0.4" stroke-dasharray="3,2"/>')

    # Roots
    if root_dicts:
        for idx, organ in enumerate(root_dicts):
            skel = organ['skeleton']
            if len(skel) < 2:
                continue
            color = ROOT_COLORS[idx % len(ROOT_COLORS)]
            pts = ' '.join(f'{proj(p[0], p[2])[0]:.1f},{proj(p[0], p[2])[1]:.1f}'
                           for p in skel)
            buf.append(f'<polyline points="{pts}" stroke="{_rgb_hex(*color)}" '
                       f'stroke-width="1.0" fill="none" opacity="0.8" '
                       f'stroke-linecap="round" stroke-linejoin="round"/>')

    # Shoot organs
    leaf_idx = 0
    for organ in organ_dicts:
        skel = organ['skeleton']
        if len(skel) < 2:
            continue
        if organ['type'] == 'stem':
            color = STEM_COLOR
            sw = '2.5'
        else:
            color = LEAF_GREENS[leaf_idx % len(LEAF_GREENS)]
            leaf_idx += 1
            sw = '1.8'
        pts = ' '.join(f'{proj(p[0], p[2])[0]:.1f},{proj(p[0], p[2])[1]:.1f}'
                       for p in skel)
        buf.append(f'<polyline points="{pts}" stroke="{_rgb_hex(*color)}" '
                   f'stroke-width="{sw}" fill="none" '
                   f'stroke-linecap="round" stroke-linejoin="round"/>')

    buf.append('</g>')

    # --- G3 Panel ---
    buf.append('<g id="g3-panel">')

    panel_off = panel_w + gap

    # Ground line
    gx0, gy0 = proj(x_min, 0.0, panel_off)
    gx1, gy1 = proj(x_max, 0.0, panel_off)
    buf.append(f'<line x1="{gx0:.1f}" y1="{gy0:.1f}" x2="{gx1:.1f}" y2="{gy1:.1f}" '
               f'stroke="#ccc" stroke-width="0.4" stroke-dasharray="3,2"/>')

    verts = mesh.vertices
    tris = mesh.indices
    oids = mesh.organ_ids

    id_to_color = {}
    leaf_idx = 0
    for organ in organ_dicts:
        oid = organ['organ_id']
        if organ['type'] == 'stem':
            id_to_color[oid] = STEM_COLOR
        else:
            id_to_color[oid] = LEAF_GREENS[leaf_idx % len(LEAF_GREENS)]
            leaf_idx += 1

    v0 = verts[tris[:, 0]]
    v1 = verts[tris[:, 1]]
    v2 = verts[tris[:, 2]]

    centroids_y = (v0[:, 1] + v1[:, 1] + v2[:, 1]) / 3.0
    order = np.argsort(centroids_y)

    e1 = v1 - v0
    e2 = v2 - v0
    normals = np.cross(e1, e2)
    nlen = np.linalg.norm(normals, axis=1, keepdims=True)
    nlen[nlen == 0] = 1
    normals /= nlen

    light = np.array([0.3, -0.8, 0.5])
    light /= np.linalg.norm(light)
    shade = np.abs(np.dot(normals, light))
    shade = 0.35 + 0.65 * shade

    n_tris = len(tris)
    print(f"  Writing {n_tris} triangles...")

    for tri_idx in order:
        tri = tris[tri_idx]
        oid = int(oids[tri_idx])
        bc = id_to_color.get(oid, (0.5, 0.5, 0.5))
        s = float(shade[tri_idx])

        r = min(1.0, bc[0] * s)
        g = min(1.0, bc[1] * s)
        b = min(1.0, bc[2] * s)
        fill = _rgb_hex(r, g, b)

        p0 = proj(verts[tri[0], 0], verts[tri[0], 2], panel_off)
        p1 = proj(verts[tri[1], 0], verts[tri[1], 2], panel_off)
        p2 = proj(verts[tri[2], 0], verts[tri[2], 2], panel_off)
        buf.append(f'<polygon points="{p0[0]:.1f},{p0[1]:.1f} '
                   f'{p1[0]:.1f},{p1[1]:.1f} '
                   f'{p2[0]:.1f},{p2[1]:.1f}" fill="{fill}"/>')

    buf.append('</g>')

    # --- Scale bar (bottom center, spans both panels) ---
    bar_y = total_h - footer_h + 10
    bar_x = (total_w - bar_px) / 2
    buf.append(f'<line x1="{bar_x:.1f}" y1="{bar_y:.1f}" '
               f'x2="{bar_x + bar_px:.1f}" y2="{bar_y:.1f}" '
               f'stroke="#000" stroke-width="1.5"/>')
    # End caps
    cap_h = 4
    for bx in [bar_x, bar_x + bar_px]:
        buf.append(f'<line x1="{bx:.1f}" y1="{bar_y - cap_h:.1f}" '
                   f'x2="{bx:.1f}" y2="{bar_y + cap_h:.1f}" '
                   f'stroke="#000" stroke-width="1.5"/>')
    # Label
    buf.append(f'<text x="{bar_x + bar_px / 2:.1f}" y="{bar_y + 16:.1f}" '
               f'text-anchor="middle" font-family="{font}" font-size="11" '
               f'fill="#000">{bar_cm} cm</text>')

    # --- Thin separator line between panels ---
    sep_x = panel_w + gap / 2
    buf.append(f'<line x1="{sep_x:.1f}" y1="{header_h + 5:.1f}" '
               f'x2="{sep_x:.1f}" y2="{header_h + panel_h - 5:.1f}" '
               f'stroke="#e0e0e0" stroke-width="0.5"/>')

    buf.append('</svg>')

    svg_path = Path(output_prefix + '_publication').with_suffix('.svg')
    with open(svg_path, 'w') as f:
        f.write('\n'.join(buf))

    size_mb = svg_path.stat().st_size / (1024 * 1024)
    print(f"  Publication SVG: {svg_path} ({size_mb:.1f} MB, {n_tris} triangles)")
    print(f"  Scale bar: {bar_cm} cm = {bar_px:.0f} px")

    # --- Individual transparent SVGs ---
    _export_individual_svgs(organ_dicts, mesh, output_prefix, root_dicts,
                            x_min, x_max, z_min, z_max, scale, off_x, off_y,
                            margin, header_h, usable_w, usable_h,
                            bar_cm, bar_px, id_to_color, font)

    return svg_path


def _export_individual_svgs(organ_dicts, mesh, output_prefix, root_dicts,
                            x_min, x_max, z_min, z_max, scale, off_x, off_y,
                            margin, header_h, usable_w, usable_h,
                            bar_cm, bar_px, id_to_color, font):
    """Export G1 and G3 as individual SVGs with transparent backgrounds.

    Uses the same projection as the combined publication SVG so they
    overlay perfectly when composited.
    """
    panel_w = 2 * margin + usable_w
    panel_h = header_h + 2 * margin + usable_h
    footer_h = 30.0
    total_h = panel_h + footer_h

    def proj(x, z):
        sx = margin + off_x + (x - x_min) * scale
        sy = header_h + margin + off_y + (z_max - z) * scale
        return sx, sy

    # ---- G1 SVG ----
    buf = []
    buf.append('<?xml version="1.0" encoding="UTF-8"?>')
    buf.append(f'<svg xmlns="http://www.w3.org/2000/svg" '
               f'viewBox="0 0 {panel_w:.1f} {total_h:.1f}" '
               f'width="{panel_w:.0f}" height="{total_h:.0f}">')
    # No background rect → transparent

    # Ground line
    gx0, gy0 = proj(x_min, 0.0)
    gx1, gy1 = proj(x_max, 0.0)
    buf.append(f'<line x1="{gx0:.1f}" y1="{gy0:.1f}" x2="{gx1:.1f}" y2="{gy1:.1f}" '
               f'stroke="#ccc" stroke-width="0.4" stroke-dasharray="3,2"/>')

    # Roots
    if root_dicts:
        for idx, organ in enumerate(root_dicts):
            skel = organ['skeleton']
            if len(skel) < 2:
                continue
            color = ROOT_COLORS[idx % len(ROOT_COLORS)]
            pts = ' '.join(f'{proj(p[0], p[2])[0]:.1f},{proj(p[0], p[2])[1]:.1f}'
                           for p in skel)
            buf.append(f'<polyline points="{pts}" stroke="{_rgb_hex(*color)}" '
                       f'stroke-width="1.0" fill="none" opacity="0.8" '
                       f'stroke-linecap="round" stroke-linejoin="round"/>')

    # Shoot
    leaf_idx = 0
    for organ in organ_dicts:
        skel = organ['skeleton']
        if len(skel) < 2:
            continue
        if organ['type'] == 'stem':
            color = STEM_COLOR
            sw = '2.5'
        else:
            color = LEAF_GREENS[leaf_idx % len(LEAF_GREENS)]
            leaf_idx += 1
            sw = '1.8'
        pts = ' '.join(f'{proj(p[0], p[2])[0]:.1f},{proj(p[0], p[2])[1]:.1f}'
                       for p in skel)
        buf.append(f'<polyline points="{pts}" stroke="{_rgb_hex(*color)}" '
                   f'stroke-width="{sw}" fill="none" '
                   f'stroke-linecap="round" stroke-linejoin="round"/>')

    # Scale bar
    bar_y = total_h - footer_h + 10
    bar_x = (panel_w - bar_px) / 2
    buf.append(f'<line x1="{bar_x:.1f}" y1="{bar_y:.1f}" '
               f'x2="{bar_x + bar_px:.1f}" y2="{bar_y:.1f}" '
               f'stroke="#000" stroke-width="1.5"/>')
    for bx in [bar_x, bar_x + bar_px]:
        buf.append(f'<line x1="{bx:.1f}" y1="{bar_y - 4:.1f}" '
                   f'x2="{bx:.1f}" y2="{bar_y + 4:.1f}" '
                   f'stroke="#000" stroke-width="1.5"/>')
    buf.append(f'<text x="{bar_x + bar_px/2:.1f}" y="{bar_y + 16:.1f}" '
               f'text-anchor="middle" font-family="{font}" font-size="11" '
               f'fill="#000">{bar_cm} cm</text>')

    buf.append('</svg>')

    g1_path = Path(output_prefix + '_g1').with_suffix('.svg')
    with open(g1_path, 'w') as f:
        f.write('\n'.join(buf))
    size_kb = g1_path.stat().st_size / 1024
    print(f"  G1 SVG: {g1_path} ({size_kb:.0f} KB, transparent)")

    # ---- G3 SVG ----
    buf = []
    buf.append('<?xml version="1.0" encoding="UTF-8"?>')
    buf.append(f'<svg xmlns="http://www.w3.org/2000/svg" '
               f'viewBox="0 0 {panel_w:.1f} {total_h:.1f}" '
               f'width="{panel_w:.0f}" height="{total_h:.0f}">')

    # Ground line
    buf.append(f'<line x1="{gx0:.1f}" y1="{gy0:.1f}" x2="{gx1:.1f}" y2="{gy1:.1f}" '
               f'stroke="#ccc" stroke-width="0.4" stroke-dasharray="3,2"/>')

    verts = mesh.vertices
    tris = mesh.indices
    oids = mesh.organ_ids

    v0 = verts[tris[:, 0]]
    v1 = verts[tris[:, 1]]
    v2 = verts[tris[:, 2]]

    centroids_y = (v0[:, 1] + v1[:, 1] + v2[:, 1]) / 3.0
    order = np.argsort(centroids_y)

    e1 = v1 - v0
    e2 = v2 - v0
    normals = np.cross(e1, e2)
    nlen = np.linalg.norm(normals, axis=1, keepdims=True)
    nlen[nlen == 0] = 1
    normals /= nlen

    light = np.array([0.3, -0.8, 0.5])
    light /= np.linalg.norm(light)
    shade = np.abs(np.dot(normals, light))
    shade = 0.35 + 0.65 * shade

    n_tris = len(tris)
    for tri_idx in order:
        tri = tris[tri_idx]
        oid = int(oids[tri_idx])
        bc = id_to_color.get(oid, (0.5, 0.5, 0.5))
        s = float(shade[tri_idx])
        r = min(1.0, bc[0] * s)
        g = min(1.0, bc[1] * s)
        b = min(1.0, bc[2] * s)
        fill = _rgb_hex(r, g, b)

        p0 = proj(verts[tri[0], 0], verts[tri[0], 2])
        p1 = proj(verts[tri[1], 0], verts[tri[1], 2])
        p2 = proj(verts[tri[2], 0], verts[tri[2], 2])
        buf.append(f'<polygon points="{p0[0]:.1f},{p0[1]:.1f} '
                   f'{p1[0]:.1f},{p1[1]:.1f} '
                   f'{p2[0]:.1f},{p2[1]:.1f}" fill="{fill}"/>')

    # Scale bar
    buf.append(f'<line x1="{bar_x:.1f}" y1="{bar_y:.1f}" '
               f'x2="{bar_x + bar_px:.1f}" y2="{bar_y:.1f}" '
               f'stroke="#000" stroke-width="1.5"/>')
    for bx in [bar_x, bar_x + bar_px]:
        buf.append(f'<line x1="{bx:.1f}" y1="{bar_y - 4:.1f}" '
                   f'x2="{bx:.1f}" y2="{bar_y + 4:.1f}" '
                   f'stroke="#000" stroke-width="1.5"/>')
    buf.append(f'<text x="{bar_x + bar_px/2:.1f}" y="{bar_y + 16:.1f}" '
               f'text-anchor="middle" font-family="{font}" font-size="11" '
               f'fill="#000">{bar_cm} cm</text>')

    buf.append('</svg>')

    g3_path = Path(output_prefix + '_g3').with_suffix('.svg')
    with open(g3_path, 'w') as f:
        f.write('\n'.join(buf))
    size_mb = g3_path.stat().st_size / (1024 * 1024)
    print(f"  G3 SVG: {g3_path} ({size_mb:.1f} MB, {n_tris} triangles, transparent)")


# ---------------------------------------------------------------------------
# Animated SVG (SMIL, frame-by-frame growth sequence)
# ---------------------------------------------------------------------------

def _build_frame_svg(organ_dicts, mesh, root_dicts, proj_fn, panel_w, gap,
                     header_h, x_min, x_max, id_to_color_fn):
    """Build SVG elements for a single animation frame (G1 + G3 panels).

    Returns list of SVG line strings.
    """
    lines = []

    # --- G1 polylines ---
    # Ground line
    gx0, gy0 = proj_fn(x_min, 0.0)
    gx1, gy1 = proj_fn(x_max, 0.0)
    lines.append(f'<line x1="{gx0:.1f}" y1="{gy0:.1f}" x2="{gx1:.1f}" y2="{gy1:.1f}" '
                 f'stroke="#aaa" stroke-width="0.5" stroke-dasharray="4,3"/>')

    # Roots
    if root_dicts:
        for idx, organ in enumerate(root_dicts):
            skel = organ['skeleton']
            if len(skel) < 2:
                continue
            color = ROOT_COLORS[idx % len(ROOT_COLORS)]
            pts = ' '.join(f'{proj_fn(p[0], p[2])[0]:.1f},{proj_fn(p[0], p[2])[1]:.1f}'
                           for p in skel)
            lines.append(f'<polyline points="{pts}" stroke="{_rgb_hex(*color)}" '
                         f'stroke-width="1.2" fill="none" '
                         f'stroke-linecap="round" stroke-linejoin="round"/>')

    # Shoot organs
    leaf_idx = 0
    for organ in organ_dicts:
        skel = organ['skeleton']
        if len(skel) < 2:
            continue
        if organ['type'] == 'stem':
            color = STEM_COLOR
            sw = '3'
        else:
            color = LEAF_GREENS[leaf_idx % len(LEAF_GREENS)]
            leaf_idx += 1
            sw = '2'
        pts = ' '.join(f'{proj_fn(p[0], p[2])[0]:.1f},{proj_fn(p[0], p[2])[1]:.1f}'
                       for p in skel)
        lines.append(f'<polyline points="{pts}" stroke="{_rgb_hex(*color)}" '
                     f'stroke-width="{sw}" fill="none" '
                     f'stroke-linecap="round" stroke-linejoin="round"/>')

    # --- G3 filled triangles ---
    panel_off = panel_w + gap

    # Ground line in G3 panel
    gx0, gy0 = proj_fn(x_min, 0.0, panel_off)
    gx1, gy1 = proj_fn(x_max, 0.0, panel_off)
    lines.append(f'<line x1="{gx0:.1f}" y1="{gy0:.1f}" x2="{gx1:.1f}" y2="{gy1:.1f}" '
                 f'stroke="#aaa" stroke-width="0.5" stroke-dasharray="4,3"/>')

    verts = mesh.vertices
    tris = mesh.indices
    oids = mesh.organ_ids

    # Organ color map for this frame
    id_to_color = id_to_color_fn(organ_dicts)

    # Vectorized depth sort + shading
    v0 = verts[tris[:, 0]]
    v1 = verts[tris[:, 1]]
    v2 = verts[tris[:, 2]]

    centroids_y = (v0[:, 1] + v1[:, 1] + v2[:, 1]) / 3.0
    order = np.argsort(centroids_y)

    e1 = v1 - v0
    e2 = v2 - v0
    normals = np.cross(e1, e2)
    nlen = np.linalg.norm(normals, axis=1, keepdims=True)
    nlen[nlen == 0] = 1
    normals /= nlen

    light = np.array([0.3, -0.8, 0.5])
    light /= np.linalg.norm(light)
    shade = np.abs(np.dot(normals, light))
    shade = 0.35 + 0.65 * shade

    for tri_idx in order:
        tri = tris[tri_idx]
        oid = int(oids[tri_idx])
        bc = id_to_color.get(oid, (0.5, 0.5, 0.5))
        s = float(shade[tri_idx])

        r = min(1.0, bc[0] * s)
        g = min(1.0, bc[1] * s)
        b = min(1.0, bc[2] * s)
        fill = _rgb_hex(r, g, b)

        p0 = proj_fn(verts[tri[0], 0], verts[tri[0], 2], panel_off)
        p1 = proj_fn(verts[tri[1], 0], verts[tri[1], 2], panel_off)
        p2 = proj_fn(verts[tri[2], 0], verts[tri[2], 2], panel_off)
        lines.append(f'<polygon points="{p0[0]:.1f},{p0[1]:.1f} '
                     f'{p1[0]:.1f},{p1[1]:.1f} '
                     f'{p2[0]:.1f},{p2[1]:.1f}" fill="{fill}"/>')

    return lines


def render_animated_svg(xml_path, max_days, output_prefix, preset,
                        day_step=5, frame_dur=0.5):
    """Generate an animated SVG showing plant growth day by day.

    Uses SMIL <animate> with discrete visibility toggling per frame.
    Each frame contains both G1 skeleton and G3 mesh panels.

    Args:
        xml_path: Path to calibrated maize XML.
        max_days: Final simulation day.
        output_prefix: Output file prefix (writes {prefix}_animated.svg).
        preset: Resolution preset dict (min_stem_nodes, min_leaf_nodes, stem_res).
        day_step: Days between frames (default 5).
        frame_dur: Seconds per frame (default 0.5).
    """
    print(f"\n{'='*60}")
    print(f"Animated SVG: day 1 to {max_days}, step {day_step}")
    print(f"{'='*60}")

    max_days_int = int(max_days)
    frame_days = list(range(day_step, max_days_int + 1, day_step))
    if not frame_days or frame_days[-1] != max_days_int:
        frame_days.append(max_days_int)
    # Always include day 1 at the start
    if frame_days[0] > 1:
        frame_days.insert(0, max(1, day_step // 2))

    n_frames = len(frame_days)
    total_dur = n_frames * frame_dur
    print(f"  Frames: {n_frames} ({', '.join(f'd{d}' for d in frame_days)})")
    print(f"  Animation: {total_dur:.1f}s loop at {frame_dur}s/frame")

    # --- Collect per-frame data ---
    frames = []  # list of (day, organ_dicts, mesh, root_dicts)

    for i, day in enumerate(frame_days):
        print(f"\n  [{i+1}/{n_frames}] Simulating day {day}...", end=' ', flush=True)
        plant = grow_plant(
            xml_path=xml_path,
            simulation_time=day,
            min_stem_nodes=preset['min_stem_nodes'],
            min_leaf_nodes=preset['min_leaf_nodes'],
        )
        mesh, organ_dicts = extract_g3_mesh(
            plant,
            min_stem_nodes=preset['min_stem_nodes'],
            min_leaf_nodes=preset['min_leaf_nodes'],
            stem_res=preset['stem_res'],
        )
        root_dicts = extract_root_dicts(plant, min_root_nodes=preset.get('min_stem_nodes', 50) // 2)
        print(f"{len(organ_dicts)} organs, {mesh.n_triangles} tris, {len(root_dicts)} roots")
        frames.append((day, organ_dicts, mesh, root_dicts))

    # --- Compute global bounds across ALL frames ---
    all_pts_list = []
    for day, organ_dicts, mesh, root_dicts in frames:
        all_pts_list.append(np.concatenate([o['skeleton'] for o in organ_dicts]))
        all_pts_list.append(mesh.vertices)
        if root_dicts:
            all_pts_list.append(np.concatenate([o['skeleton'] for o in root_dicts]))
    combined = np.concatenate(all_pts_list)

    x_min = float(combined[:, 0].min())
    x_max = float(combined[:, 0].max())
    z_min = float(combined[:, 2].min())
    z_max = float(combined[:, 2].max())

    # --- Layout ---
    panel_w, panel_h = 400.0, 600.0
    margin = 20.0
    header_h = 30.0
    gap = 20.0
    counter_w = 80.0  # extra width for day counter
    total_w = panel_w * 2 + gap + counter_w
    total_h = panel_h + header_h

    data_w = max(x_max - x_min, 0.1)
    data_h = max(z_max - z_min, 0.1)
    usable_w = panel_w - 2 * margin
    usable_h = panel_h - 2 * margin
    scale = min(usable_w / data_w, usable_h / data_h)
    off_x = (usable_w - data_w * scale) / 2
    off_y = (usable_h - data_h * scale) / 2

    def proj(x, z, panel_off=0.0):
        sx = margin + off_x + (x - x_min) * scale + panel_off
        sy = header_h + margin + off_y + (z_max - z) * scale
        return sx, sy

    def make_color_map(organ_dicts):
        id_to_color = {}
        li = 0
        for organ in organ_dicts:
            oid = organ['organ_id']
            if organ['type'] == 'stem':
                id_to_color[oid] = STEM_COLOR
            else:
                id_to_color[oid] = LEAF_GREENS[li % len(LEAF_GREENS)]
                li += 1
        return id_to_color

    # --- Build animated SVG ---
    print(f"\n  Building animated SVG...")
    buf = []
    buf.append('<?xml version="1.0" encoding="UTF-8"?>')
    buf.append(f'<svg xmlns="http://www.w3.org/2000/svg" '
               f'viewBox="0 0 {total_w:.1f} {total_h:.1f}" '
               f'width="{total_w:.0f}" height="{total_h:.0f}">')

    # Background
    buf.append(f'<rect x="0" y="0" width="{total_w:.1f}" height="{total_h:.1f}" fill="white"/>')

    # Panel backgrounds (static)
    for px in [0, panel_w + gap]:
        buf.append(f'<rect x="{px:.1f}" y="{header_h:.1f}" '
                   f'width="{panel_w:.1f}" height="{panel_h:.1f}" '
                   f'fill="#f8f8f8" stroke="#ccc" stroke-width="0.5"/>')

    # Headers (static, Times New Roman)
    font = 'Times New Roman, Times, serif'
    buf.append(f'<text x="{panel_w/2:.1f}" y="20" text-anchor="middle" '
               f'font-family="{font}" font-size="14" fill="#333">G1 Skeleton</text>')
    buf.append(f'<text x="{panel_w + gap + panel_w/2:.1f}" y="20" text-anchor="middle" '
               f'font-family="{font}" font-size="14" fill="#333">G3 Mesh</text>')

    # Day counter area (right side)
    counter_x = panel_w * 2 + gap + counter_w / 2
    buf.append(f'<text x="{counter_x:.1f}" y="{header_h + 30:.1f}" text-anchor="middle" '
               f'font-family="{font}" font-size="12" fill="#999">Day</text>')

    # --- Animated day counter numbers ---
    for i, (day, _, _, _) in enumerate(frames):
        kt0 = i / n_frames
        kt1 = (i + 1) / n_frames
        if i == 0:
            vals = 'visible;hidden'
            kts = f'0;{kt1:.6f}'
        elif i == n_frames - 1:
            vals = 'hidden;visible'
            kts = f'0;{kt0:.6f}'
        else:
            vals = 'hidden;visible;hidden'
            kts = f'0;{kt0:.6f};{kt1:.6f}'

        buf.append(f'<text x="{counter_x:.1f}" y="{header_h + 70:.1f}" text-anchor="middle" '
                   f'font-family="{font}" font-size="32" font-weight="bold" '
                   f'fill="#222" visibility="hidden">{day}'
                   f'<animate attributeName="visibility" values="{vals}" '
                   f'keyTimes="{kts}" dur="{total_dur:.2f}s" '
                   f'repeatCount="indefinite" calcMode="discrete"/></text>')

    # Progress bar background
    bar_x = panel_w * 2 + gap + 10
    bar_w = counter_w - 20
    bar_y = header_h + 100
    bar_h = panel_h - 120
    buf.append(f'<rect x="{bar_x:.1f}" y="{bar_y:.1f}" '
               f'width="{bar_w:.1f}" height="{bar_h:.1f}" '
               f'fill="none" stroke="#ddd" stroke-width="0.5" rx="3"/>')

    # Animated progress bar fill (grows upward)
    for i, (day, _, _, _) in enumerate(frames):
        frac = day / max_days
        fill_h = bar_h * frac
        fill_y = bar_y + bar_h - fill_h

        kt0 = i / n_frames
        kt1 = (i + 1) / n_frames
        if i == 0:
            vals = 'visible;hidden'
            kts = f'0;{kt1:.6f}'
        elif i == n_frames - 1:
            vals = 'hidden;visible'
            kts = f'0;{kt0:.6f}'
        else:
            vals = 'hidden;visible;hidden'
            kts = f'0;{kt0:.6f};{kt1:.6f}'

        buf.append(f'<rect x="{bar_x:.1f}" y="{fill_y:.1f}" '
                   f'width="{bar_w:.1f}" height="{fill_h:.1f}" '
                   f'fill="#8bc34a" rx="3" visibility="hidden">'
                   f'<animate attributeName="visibility" values="{vals}" '
                   f'keyTimes="{kts}" dur="{total_dur:.2f}s" '
                   f'repeatCount="indefinite" calcMode="discrete"/></rect>')

    # --- Animated frame groups ---
    total_tris = 0
    for i, (day, organ_dicts, mesh, root_dicts) in enumerate(frames):
        print(f"  Frame {i+1}/{n_frames} (day {day}): {mesh.n_triangles} triangles...",
              flush=True)

        kt0 = i / n_frames
        kt1 = (i + 1) / n_frames
        if i == 0:
            vals = 'visible;hidden'
            kts = f'0;{kt1:.6f}'
        elif i == n_frames - 1:
            vals = 'hidden;visible'
            kts = f'0;{kt0:.6f}'
        else:
            vals = 'hidden;visible;hidden'
            kts = f'0;{kt0:.6f};{kt1:.6f}'

        buf.append(f'<g id="frame-{i}" visibility="hidden">')
        buf.append(f'<animate attributeName="visibility" values="{vals}" '
                   f'keyTimes="{kts}" dur="{total_dur:.2f}s" '
                   f'repeatCount="indefinite" calcMode="discrete"/>')

        frame_lines = _build_frame_svg(
            organ_dicts, mesh, root_dicts, proj, panel_w, gap,
            header_h, x_min, x_max, make_color_map)
        buf.extend(frame_lines)
        total_tris += mesh.n_triangles

        buf.append('</g>')

    buf.append('</svg>')

    svg_path = Path(output_prefix + '_animated').with_suffix('.svg')
    with open(svg_path, 'w') as f:
        f.write('\n'.join(buf))

    size_mb = svg_path.stat().st_size / (1024 * 1024)
    print(f"\n  Animated SVG: {svg_path}")
    print(f"  {n_frames} frames, {total_tris} total triangles, {size_mb:.1f} MB")
    return svg_path


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
        render_comparison_png(organ_dicts, mesh, args.output, args.days,
                             root_dicts=root_dicts)

    # Optional: SVG vector graphic
    if args.svg:
        render_comparison_svg(organ_dicts, mesh, args.output, args.days,
                              root_dicts=root_dicts)

    # Optional: publication-quality SVG
    if args.publication:
        render_publication_svg(organ_dicts, mesh, args.output, args.days,
                               root_dicts=root_dicts)

    # Optional: animated SVG (runs its own simulation loop)
    if args.animate:
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
