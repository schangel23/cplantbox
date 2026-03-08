"""RLD/LAI profile extraction, plant summary, and growth trajectory plots.

Extracted from grow.py — pure file reorganization, no logic changes.
"""

import numpy as np
from pathlib import Path

import plantbox as pb


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


# ---------------------------------------------------------------------------
# CLI entry points (called from __main__.py)
# ---------------------------------------------------------------------------

def main_rld(args):
    """CLI handler for the ``rld`` subcommand."""
    from .grow import grow_plant
    from ..config import DEFAULT_XML, OUTPUT_DIR

    rld_out = OUTPUT_DIR / "rld"
    rld_out.mkdir(parents=True, exist_ok=True)

    if args.multi_day:
        test_days = [20, 35, 55]
        profiles = {}
        for day in test_days:
            print(f"\n{'='*60}")
            print(f"RLD EXTRACTION — Day {day}")
            print(f"{'='*60}")
            plant = grow_plant(
                xml_path=str(DEFAULT_XML),
                simulation_time=day,
                enable_photosynthesis=True,
                seed=42,
            )
            prof = extract_rld_profile(
                plant, n_layers=args.layers, depth_cm=args.depth,
                row_spacing_cm=args.row_spacing,
                plant_spacing_cm=args.plant_spacing,
            )
            profiles[day] = prof
            export_rld_csv(prof, rld_out / f"maize_day{day}_rld_profile.csv")
            export_rrd_in(prof, rld_out / f"maize_day{day}_rrd.in")
            plot_rld_profile(prof, rld_out / f"maize_day{day}_rld_profile.png",
                             day=day)
        plot_rld_growth_trajectory(profiles,
                                   rld_out / "rld_growth_trajectory.png")
    else:
        plant = grow_plant(
            xml_path=str(DEFAULT_XML),
            simulation_time=args.day,
            enable_photosynthesis=True,
            seed=42,
        )
        prof = extract_rld_profile(
            plant, n_layers=args.layers, depth_cm=args.depth,
            row_spacing_cm=args.row_spacing,
            plant_spacing_cm=args.plant_spacing,
        )
        export_rld_csv(prof,
                       rld_out / f"maize_day{args.day}_rld_profile.csv")
        export_rrd_in(prof,
                      rld_out / f"maize_day{args.day}_rrd.in")
        plot_rld_profile(prof,
                         rld_out / f"maize_day{args.day}_rld_profile.png",
                         day=args.day)

        print(f"\n  Total root length: {prof['total_root_length_cm']:.0f} cm")
        print(f"  Max root depth: {prof['max_root_depth_cm']:.1f} cm")
        print(f"  Surface RLD: {prof['RLD_cm_per_cm3'][0]:.4f} cm/cm3")


def main_summary(args):
    """CLI handler for the ``summary`` subcommand."""
    import json
    from .grow import grow_plant, run_photosynthesis
    from ..carbon import solve_carbon_partitioning
    from ..config import DEFAULT_XML, OUTPUT_DIR, get_species_name

    summary_out = OUTPUT_DIR / "summaries"
    summary_out.mkdir(parents=True, exist_ok=True)
    species = get_species_name()

    def _run_single_day(day, par, tair, method, row_sp, plant_sp, bins,
                        out_dir):
        """Run full pipeline for one day, return summary dict."""
        print(f"\n{'='*60}")
        print(f"LAI + PLANT SUMMARY — {species} Day {day}")
        print(f"{'='*60}")

        # 1. Grow plant
        plant = grow_plant(
            xml_path=str(DEFAULT_XML),
            simulation_time=day,
            enable_photosynthesis=True,
            seed=42,
        )

        # 2. LAI extraction
        lai = extract_lai_profile(
            plant, n_bins=bins,
            row_spacing_cm=row_sp,
            plant_spacing_cm=plant_sp,
        )
        export_lai_csv(lai, out_dir / f"{species}_day{day}_lai_profile.csv")
        plot_lai_profile(lai, out_dir / f"{species}_day{day}_lai_profile.png",
                         day=day)

        print(f"\n  LAI: {lai['LAI']:.2f}")
        print(f"  Total leaf area: {lai['total_leaf_area_cm2']:.0f} cm2 "
              f"({lai['total_leaf_area_m2']:.4f} m2)")
        print(f"  Plant height: {lai['plant_height_cm']:.1f} cm")
        print(f"  Leaf organs: {lai['n_leaf_organs']}, "
              f"segments: {lai['n_leaf_segments']}")

        # 3. Photosynthesis
        prefix = str(out_dir / f"{species}_day{day}_photo")
        hm = run_photosynthesis(
            plant, sim_time=day, output_prefix=prefix,
            par_umol=par, tair_c=tair,
        )
        if hm is None:
            print(f"  WARNING: Photosynthesis failed for day {day}, "
                  f"skipping carbon partitioning")

        # 4. Carbon partitioning
        carbon_result = None
        if hm is not None:
            An_leaf = np.array(hm.get_net_assimilation())
            try:
                carbon_result = solve_carbon_partitioning(
                    plant, An_leaf, Tair_C=tair,
                    method=method, day=day,
                )
            except Exception as e:
                print(f"  WARNING: Carbon partitioning failed: {e}")

        # 5. Assemble summary
        summary = extract_plant_summary(
            plant, hm, carbon_result, lai, day,
            par_umol=par, tair_c=tair,
        )

        # 6. Save JSON
        json_path = out_dir / f"{species}_day{day}_summary.json"
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\n  Summary JSON: {json_path}")

        return summary

    if args.multi_day:
        test_days = [10, 20, 30, 40, 55]
        summaries = {}
        for day in test_days:
            summaries[day] = _run_single_day(
                day, args.par, args.tair, args.method,
                args.row_spacing, args.plant_spacing, args.bins,
                summary_out,
            )

        # Growth trajectory plot
        plot_growth_trajectory(
            summaries, summary_out / "growth_trajectory.png")

        # Combined results JSON
        combined_path = summary_out / "summary_results.json"
        with open(combined_path, 'w') as f:
            json.dump(
                {str(d): s for d, s in summaries.items()},
                f, indent=2,
            )
        print(f"\n  Combined results: {combined_path}")
    else:
        _run_single_day(
            args.day, args.par, args.tair, args.method,
            args.row_spacing, args.plant_spacing, args.bins,
            summary_out,
        )
