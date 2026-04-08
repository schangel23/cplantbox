#!/usr/bin/env python3
"""Fit CPlantBox XML parameters to match OBJ reference models.

Uses CMA-ES to optimize per-leaf growth parameters until CPlantBox-grown
skeletons and lofted meshes match the reference OBJ geometry. Outputs a
best-fit XML file.

Two-level comparison separates CPlantBox vs lofter blame:
  1. Skeleton Chamfer: CPlantBox skeleton vs OBJ-extracted skeleton
  2. Mesh Chamfer: lofted mesh vs OBJ mesh (includes lofter contribution)

When the skeleton error hits a floor despite optimization, that's a CPlantBox
model limitation. When skeleton matches but mesh doesn't, that's a lofter
limitation.

Usage (server):
    source /media/data/Lukas/CPlantBox/cpbenv/bin/activate
    cd /media/data/Lukas/CPlantBox
    python3 dart/coupling/experimental/fit_to_reference.py \\
        /media/data/Lukas/Maize/export/ \\
        --output dart/coupling/experimental/output/fit_result/ \\
        --workers 64 --evals 500 -v
"""

import argparse
import copy
import json
import math
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import cma
from scipy.spatial import KDTree

# Add coupling to path
_COUPLING_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_COUPLING_DIR.parent))

import plantbox as pb
from dart.coupling.geometry.g1_to_g3 import loft_organs
from dart.coupling.geometry.cplantbox_adapter import extract_organs_for_lofter
from dart.coupling.growth.grow import setup_successor_where

# Import skeleton extraction from reverse_engineer_maize
from dart.coupling.experimental.reverse_engineer_maize import (
    parse_obj, find_connected_components, track_leaves_across_stages,
    extract_leaf_g1, extract_stem_g1, count_developed_leaves, vstage_to_day,
    _chamfer_distance, _mesh_to_points,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MAX_WORKERS = int(os.environ.get("FIT_MAX_WORKERS", "64"))

# Parameter names (fixed structure — 12 leaf params, 4 stem params)
LEAF_PARAM_NAMES = [
    "lmax", "r", "theta", "tropismS", "tropismAge",
    "tropismExponent", "collarLength",
    "curv_k0", "curv_k1", "curv_k2", "curv_k3", "curv_k4",
]
STEM_PARAM_NAMES = ["lmax", "r", "ln", "lb"]
N_LEAF_PARAMS = len(LEAF_PARAM_NAMES)
N_STEM_PARAMS = len(STEM_PARAM_NAMES)

# Hard physical safety limits — CPlantBox crashes or produces nonsense beyond these
SAFETY_LIMITS = {
    "lmax": (1.0, 200.0),
    "r": (0.1, 20.0),
    "theta": (0.01, 1.5),
    "tropismS": (0.0, 0.2),
    "tropismAge": (0.5, 80.0),
    "tropismExponent": (0.5, 3.0),
    "collarLength": (0.0, 30.0),
    "curv_k": (0.0, 0.2),
}


# ---------------------------------------------------------------------------
# Reference-derived initialization helpers
# ---------------------------------------------------------------------------

def _fit_exponential_r(day_length_pairs, lmax):
    """Fit exponential growth rate from multi-stage (day, length) observations.

    Model: L(t) = lmax * (1 - exp(-r/lmax * t))
    Time is shifted so the first observation is t=0 (approximates emergence).
    """
    if len(day_length_pairs) < 2:
        return lmax / 30.0

    from scipy.optimize import curve_fit as _curve_fit
    days = np.array([d for d, _l in day_length_pairs])
    lengths = np.array([_l for _d, _l in day_length_pairs])

    # Shift to time-since-emergence (first observation ≈ emergence)
    t0 = days[0]
    t_rel = days - t0

    def model(t, r):
        return lmax * (1.0 - np.exp(-r / lmax * t))

    try:
        popt, _ = _curve_fit(model, t_rel, lengths, p0=[lmax / 15.0],
                             bounds=(0.01, lmax * 0.5), maxfev=1000)
        return float(popt[0])
    except Exception:
        # Fallback: solve from last observation
        t = t_rel[-1]
        L = min(lengths[-1], lmax * 0.95)
        if L > 0 and t > 0 and L < lmax:
            return float(-lmax / t * math.log(1.0 - L / lmax))
        return lmax / 15.0


def _fit_gompertz_kr(day_length_pairs):
    """Fit Gompertz K and r from multi-stage observations.

    Model: L(t) = K * exp(ln(L0/K) * exp(-r*t/K))
    Time is shifted so the first observation is t=0.
    """
    if len(day_length_pairs) < 2:
        max_l = max(l for _, l in day_length_pairs) if day_length_pairs else 50.0
        return max_l * 1.3, max_l * 1.3 / 15.0

    from scipy.optimize import curve_fit as _curve_fit
    days = np.array([d for d, _l in day_length_pairs])
    lengths = np.array([_l for _d, _l in day_length_pairs])

    # Shift to time-since-emergence
    t0 = days[0]
    t_rel = days - t0

    L0 = max(lengths[0], 0.5)
    max_L = float(max(lengths))

    def model(t, K, r):
        return K * np.exp(np.log(L0 / K) * np.exp(-r * t / K))

    try:
        K_init = max_L * 1.3
        r_init = K_init / 15.0
        popt, _ = _curve_fit(model, t_rel, lengths,
                             p0=[K_init, r_init],
                             bounds=([max_L, 0.1], [max_L * 3.0, max_L]),
                             maxfev=2000)
        return float(popt[0]), float(popt[1])
    except Exception:
        K = max_L * 1.3
        return K, K / 15.0


def _estimate_tropism_age(curvature_profile, length, r, lmax):
    """Estimate tropismAge from where curvature first exceeds background.

    CPlantBox grows the leaf straight for tropismAge days, then the tip bends.
    """
    if not curvature_profile or len(curvature_profile) < 3:
        return 20.0

    curv = np.array(curvature_profile)
    threshold = np.percentile(curv, 25) + 0.5 * np.std(curv)
    if threshold < 1e-4:
        threshold = 0.01

    above = np.where(curv > threshold)[0]
    frac = above[0] / len(curv) if len(above) > 0 else 0.9

    target = frac * length
    ratio = min(target / max(lmax, 1.0), 0.95)
    if ratio > 0 and r > 0 and lmax > 0:
        tropism_age = -lmax / r * math.log(1.0 - ratio)
        return max(2.0, min(tropism_age, 70.0))
    return 20.0


def _estimate_collar_length(curvature_profile, length):
    """Estimate collar length from distance to first curvature peak."""
    if not curvature_profile or len(curvature_profile) < 5:
        return 2.0

    curv = np.array(curvature_profile)
    half = len(curv) // 2
    threshold = np.percentile(curv, 50)
    above = np.where(curv[:half] > threshold)[0]

    if len(above) > 0:
        frac = above[0] / len(curv)
        return max(0.0, min(frac * length, length * 0.2))
    return 2.0


def compute_leaf_init_and_bounds(pos, ref_stages, gf=3):
    """Compute reference-derived x0 and bounds for one leaf position.

    Analyzes all stages where this leaf appears to derive data-driven
    initial values and tight bounds for CMA-ES.

    Returns:
        (x0, bounds) where x0 is a list of floats and bounds is
        [(name, lo, hi), ...] with len == N_LEAF_PARAMS.
    """
    observations = []
    for stage in ref_stages:
        leaf = next((l for l in stage["leaves"] if l.position == pos), None)
        if leaf and leaf.length > 2:
            observations.append((stage["day"], leaf))

    if not observations:
        bounds = [(n, 10.0, 100.0) if n == "lmax"
                  else (n, 0.5, 5.0) if n == "r"
                  else (n, 0.1, 1.2) if n == "theta"
                  else (n, 0.001, 0.1) if n == "tropismS"
                  else (n, 5.0, 50.0) if n == "tropismAge"
                  else (n, 0.5, 3.0) if n == "tropismExponent"
                  else (n, 0.0, 10.0) if n == "collarLength"
                  else (n, 0.0, 0.1)
                  for n in LEAF_PARAM_NAMES]
        x0 = [50.0, 2.0, 0.5, 0.03, 20.0, 1.5, 3.0,
              0.03, 0.03, 0.03, 0.03, 0.03]
        return x0, bounds

    _, mature_leaf = max(observations, key=lambda x: x[1].length)
    max_length = max(leaf.length for _, leaf in observations)
    curv = mature_leaf.curvature_profile
    day_len = [(d, l.length) for d, l in observations if l.length > 2]

    # --- lmax + r (coupled for Gompertz) ---
    # Bounds are wide: reference-derived x0 provides the starting point,
    # but CPlantBox params don't map 1:1 to observed geometry.
    if gf == 4:
        K, gomp_r = _fit_gompertz_kr(day_len)
        x0_lmax = K
        lmax_lo = max(max_length * 0.5, SAFETY_LIMITS["lmax"][0])
        lmax_hi = min(K * 3.0, SAFETY_LIMITS["lmax"][1])
        x0_r = gomp_r
        r_lo = max(x0_r * 0.15, SAFETY_LIMITS["r"][0])
        r_hi = min(x0_r * 5.0, SAFETY_LIMITS["r"][1])
    else:
        x0_lmax = max_length * 1.1
        lmax_lo = max(min(max_length * 0.5, 10.0), SAFETY_LIMITS["lmax"][0])
        lmax_hi = min(max_length * 2.5, SAFETY_LIMITS["lmax"][1])
        x0_r = _fit_exponential_r(day_len, x0_lmax)
        r_lo = max(min(x0_r * 0.15, 0.5), SAFETY_LIMITS["r"][0])
        r_hi = min(x0_r * 5.0, min(x0_lmax * 0.5, SAFETY_LIMITS["r"][1]))

    # --- theta ---
    x0_theta = mature_leaf.insertion_angle
    theta_lo = max(x0_theta * 0.3, SAFETY_LIMITS["theta"][0])
    theta_hi = min(x0_theta * 3.0, SAFETY_LIMITS["theta"][1])

    # --- tropismS ---
    if curv and len(curv) > 0:
        mean_curv = float(np.mean(curv))
        max_curv = float(np.max(curv))
        x0_tropS = mean_curv
        tropS_lo = max(mean_curv * 0.1, SAFETY_LIMITS["tropismS"][0])
        tropS_hi = min(max(max_curv * 3.0, 0.15), SAFETY_LIMITS["tropismS"][1])
    else:
        x0_tropS, tropS_lo, tropS_hi = 0.03, 0.001, 0.15

    # --- tropismAge ---
    x0_tropAge = _estimate_tropism_age(curv, mature_leaf.length, x0_r, x0_lmax)
    tropAge_lo = max(min(x0_tropAge * 0.2, 2.0), SAFETY_LIMITS["tropismAge"][0])
    tropAge_hi = min(x0_tropAge * 4.0, SAFETY_LIMITS["tropismAge"][1])

    # --- tropismExponent (hard to extract — keep broad) ---
    x0_tropExp = 1.5
    tropExp_lo, tropExp_hi = SAFETY_LIMITS["tropismExponent"]

    # --- collarLength ---
    x0_collar = _estimate_collar_length(curv, mature_leaf.length)
    collar_lo = 0.0
    collar_hi = min(max(x0_collar * 2.5, 3.0), SAFETY_LIMITS["collarLength"][1])

    # --- curvature knots ---
    if curv and len(curv) >= 5:
        indices = np.linspace(0, len(curv) - 1, 5, dtype=int)
        curv_arr = np.array(curv)
        curv_x0 = [float(curv_arr[i]) for i in indices]
        curv_max = min(float(np.max(curv_arr)) * 2.0, SAFETY_LIMITS["curv_k"][1])
    else:
        curv_x0 = [x0_tropS] * 5
        curv_max = min(0.1, SAFETY_LIMITS["curv_k"][1])

    x0 = [x0_lmax, x0_r, x0_theta, x0_tropS, x0_tropAge, x0_tropExp, x0_collar,
          *curv_x0]
    bounds = [
        ("lmax",           lmax_lo,   lmax_hi),
        ("r",              r_lo,      r_hi),
        ("theta",          theta_lo,  theta_hi),
        ("tropismS",       tropS_lo,  tropS_hi),
        ("tropismAge",     tropAge_lo, tropAge_hi),
        ("tropismExponent", tropExp_lo, tropExp_hi),
        ("collarLength",   collar_lo, collar_hi),
        ("curv_k0",        0.0,       curv_max),
        ("curv_k1",        0.0,       curv_max),
        ("curv_k2",        0.0,       curv_max),
        ("curv_k3",        0.0,       curv_max),
        ("curv_k4",        0.0,       curv_max),
    ]
    return x0, bounds


def compute_stem_init_and_bounds(ref_stages):
    """Compute reference-derived x0 and bounds for the stem.

    Returns:
        (x0, bounds) where bounds is [(name, lo, hi), ...] with len == N_STEM_PARAMS.
    """
    mature = ref_stages[-1]
    stem = mature["stem"]
    ref_height = stem.height if stem else 180.0
    last_day = mature["day"]

    # lmax
    x0_lmax = ref_height * 1.3
    lmax_lo = max(ref_height * 0.9, 50.0)
    lmax_hi = min(ref_height * 2.0, 300.0)

    # r: solve L(t) = lmax*(1-exp(-r/lmax*t)) = ref_height
    ratio = min(ref_height / x0_lmax, 0.95)
    x0_r = -x0_lmax / last_day * math.log(1.0 - ratio)
    r_lo = max(x0_r * 0.3, 0.5)
    r_hi = min(x0_r * 3.0, 15.0)

    # ln: from internode lengths
    n_leaves = len(set(l.position for l in mature["leaves"] if l.length > 3))
    if stem and stem.internode_lengths and len(stem.internode_lengths) > 0:
        internode_arr = np.array(stem.internode_lengths)
        x0_ln = float(np.mean(internode_arr))
        ln_lo = max(float(np.min(internode_arr)) * 0.5, 1.0)
        ln_hi = min(float(np.max(internode_arr)) * 1.5, 30.0)
    else:
        x0_ln = ref_height / max(n_leaves, 8)
        ln_lo = max(x0_ln * 0.5, 1.0)
        ln_hi = min(x0_ln * 2.0, 30.0)

    # lb: insertion height of lowest leaf across all stages
    lowest_h = float("inf")
    for stage in ref_stages:
        for leaf in stage["leaves"]:
            if leaf.length > 3 and leaf.insertion_height < lowest_h:
                lowest_h = leaf.insertion_height
    if lowest_h == float("inf"):
        lowest_h = 4.0
    x0_lb = max(lowest_h, 1.0)
    lb_lo = max(x0_lb * 0.3, 1.0)
    lb_hi = min(x0_lb * 2.0, 20.0)

    x0 = [x0_lmax, x0_r, x0_ln, x0_lb]
    bounds = [
        ("lmax", lmax_lo, lmax_hi),
        ("r",    r_lo,    r_hi),
        ("ln",   ln_lo,   ln_hi),
        ("lb",   lb_lo,   lb_hi),
    ]
    return x0, bounds


# ---------------------------------------------------------------------------
# Reference data loading
# ---------------------------------------------------------------------------

def load_reference(export_dir, n_samples=20, verbose=False):
    """Load all OBJ stages and extract reference skeletons + meshes.

    Returns:
        stages: list of dicts with keys:
            stage, file, vstage, day, leaves (list of LeafG1),
            stem, verts, leaf_faces, leaf_components
    """
    export_dir = Path(export_dir)
    manifest = json.loads((export_dir / "manifest.json").read_text())
    files = [(s["stage"], export_dir / s["file"]) for s in manifest["stages"]]

    # Parse all
    all_verts, all_groups, all_comps = [], [], []
    for stage_num, fpath in files:
        if verbose:
            print(f"  Parsing stage {stage_num}...")
        verts, groups = parse_obj(fpath)[:2]
        # Flip Z: OBJ has Z negative (plant grows down), CPlantBox has Z positive
        verts[:, 2] *= -1
        all_verts.append(verts)
        all_groups.append(groups)
        leaf_faces = []
        for gname, gfaces in groups.items():
            if "leaf" in gname.lower():
                leaf_faces.extend(gfaces)
        comps = find_connected_components(leaf_faces)
        all_comps.append(comps)

    # Track leaves
    canonical = track_leaves_across_stages(all_comps)

    def leaf_sort_key(lid):
        for sidx in range(len(all_verts)):
            if sidx in canonical[lid]:
                comp = canonical[lid][sidx]
                return -np.mean(all_verts[sidx][list(comp)][:, 2])
        return 0
    sorted_lids = sorted(canonical.keys(), key=leaf_sort_key)
    position_map = {lid: pos + 1 for pos, lid in enumerate(sorted_lids)}

    # Extract per-stage
    stages = []
    for idx, (stage_num, fpath) in enumerate(files):
        verts = all_verts[idx]
        groups = all_groups[idx]

        leaf_faces = []
        for gname, gfaces in groups.items():
            if "leaf" in gname.lower():
                leaf_faces.extend(gfaces)

        # Per-leaf G1
        leaves = []
        leaf_components = {}
        for lid, stage_map in canonical.items():
            if idx in stage_map:
                comp = stage_map[idx]
                g1 = extract_leaf_g1(comp, verts, leaf_faces, lid,
                                     position_map[lid], n_samples)
                leaves.append(g1)
                leaf_components[position_map[lid]] = comp
        leaves.sort(key=lambda l: l.position)

        # Stem
        stem_faces = []
        for gname, gfaces in groups.items():
            if "stem" in gname.lower():
                stem_faces.extend(gfaces)
        leaf_bases = [(l.insertion_height, l.position)
                      for l in leaves if l.length > 1.0]
        stem = extract_stem_g1(verts, stem_faces, leaf_bases)

        vstage = count_developed_leaves(leaves)
        day = vstage_to_day(vstage)

        # Per-leaf mesh points for mesh-level comparison
        leaf_mesh_points = {}
        for leaf in leaves:
            if leaf.length < 2:
                continue
            pos = leaf.position
            if pos in leaf_components:
                comp = leaf_components[pos]
                comp_faces = [f for f in leaf_faces if all(v in comp for v in f)]
                pts = _mesh_to_points(verts, comp_faces, n_samples=500)
                leaf_mesh_points[pos] = pts

        stages.append({
            "stage": stage_num,
            "file": fpath.name,
            "vstage": vstage,
            "day": day,
            "leaves": leaves,
            "stem": stem,
            "verts": verts,
            "leaf_faces": leaf_faces,
            "leaf_mesh_points": leaf_mesh_points,
        })

    if verbose:
        print(f"  {len(stages)} stages, V{stages[0]['vstage']}→V{stages[-1]['vstage']}")
        print(f"  Days: {stages[0]['day']:.0f}→{stages[-1]['day']:.0f}")
        n_pos = len(set(l.position for s in stages for l in s["leaves"]))
        print(f"  {n_pos} leaf positions tracked")

    return stages


# ---------------------------------------------------------------------------
# Template XML preparation
# ---------------------------------------------------------------------------

def ensure_xml_has_all_subtypes(xml_path, output_path, positions):
    """Ensure the XML has leaf subtypes for ALL reference positions.

    If the XML has 11 subtypes (2-12) but we need 16 (2-17), this clones
    the last existing subtype as a template for the missing ones. Also
    ensures the stem has enough ln spacing for all leaves.

    Args:
        xml_path: input XML path
        output_path: output XML path (can be same as input)
        positions: list of 1-indexed leaf positions needed

    Returns:
        output_path
    """
    import xml.etree.ElementTree as ET

    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Find existing leaf subtypes
    existing = {}
    for leaf_elem in root.findall("leaf"):
        st = int(leaf_elem.get("subType", "0"))
        existing[st] = leaf_elem

    # For each needed position, ensure subType exists
    needed_subtypes = {pos + 1 for pos in positions}  # pos 1 → subType 2
    missing = needed_subtypes - set(existing.keys())

    if missing:
        # Use highest existing subtype as template
        template_st = max(existing.keys())
        template_elem = existing[template_st]

        for st in sorted(missing):
            pos = st - 1  # subType back to position
            # Deep copy template
            new_elem = copy.deepcopy(template_elem)
            new_elem.set("subType", str(st))
            new_elem.set("name", f"maize_leaf_L{pos}")

            # Set ldelay proportional to position
            for param in new_elem.findall("parameter"):
                if param.get("name") == "ldelay":
                    param.set("value", str(pos * 3.0))  # 3-day phyllochron
            root.append(new_elem)

        # Update stem la to accommodate more leaves
        stem_elem = root.find(".//stem[@subType='1']")
        if stem_elem is not None:
            n_leaves = len(needed_subtypes)
            for param in stem_elem.findall("parameter"):
                if param.get("name") == "la":
                    lmax_val = 200.0
                    lb_val = 4.0
                    ln_val = 14.0
                    for p2 in stem_elem.findall("parameter"):
                        if p2.get("name") == "lmax":
                            lmax_val = float(p2.get("value", 200))
                        elif p2.get("name") == "lb":
                            lb_val = float(p2.get("value", 4))
                        elif p2.get("name") == "ln":
                            ln_val = float(p2.get("value", 14))
                    la_val = max(0.1, lmax_val - lb_val - (n_leaves - 1) * ln_val)
                    param.set("value", str(la_val))

        print(f"  Added {len(missing)} missing leaf subtypes: {sorted(missing)}")

    tree.write(output_path, encoding="utf-8", xml_declaration=True)
    return output_path


# ---------------------------------------------------------------------------
# CPlantBox growing + skeleton extraction
# ---------------------------------------------------------------------------

def _params_to_dict(params_vec, bounds):
    """Convert parameter vector to named dict, clamping to bounds."""
    d = {}
    for i, (name, lo, hi) in enumerate(bounds):
        d[name] = float(np.clip(params_vec[i], lo, hi))
    return d


def grow_and_extract(xml_path, day, leaf_params_by_position, stem_params):
    """Grow a CPlantBox plant and extract per-leaf skeletons.

    Args:
        xml_path: path to template XML
        day: simulation day
        leaf_params_by_position: dict[position -> param_dict]
        stem_params: dict of stem params

    Returns:
        dict[position -> skeleton_array (N,3)] or None on failure
    """
    try:
        plant = pb.Plant()
        plant.readParameters(str(xml_path))

        # Set stem params
        sp = plant.getOrganRandomParameter(3, 1)
        if stem_params:
            sp.lmax = stem_params["lmax"]
            sp.r = stem_params["r"]
            sp.ln = stem_params["ln"]
            sp.lb = stem_params["lb"]

        # Set per-leaf params
        for pos, params in leaf_params_by_position.items():
            sub_type = pos + 1  # position 1 → subType 2, etc.
            try:
                lp = plant.getOrganRandomParameter(4, sub_type)
            except Exception:
                continue

            # Set growth function type if specified (4 = Gompertz)
            gf = int(params.get("gf", lp.gf))
            if gf != lp.gf:
                lp.gf = gf
                lp.f_gf = plant.createGrowthFunction(gf)

            # Clamp to safe ranges to prevent CPlantBox crashes
            lp.lmax = max(1.0, params["lmax"])
            if gf == 4:
                # Gompertz r is sigmoid steepness, not initial rate — no lmax/2 cap
                lp.r = max(0.5, params["r"])
            else:
                lp.r = max(0.1, min(params["r"], params["lmax"] * 0.5))  # r < lmax/2
            lp.theta = max(0.01, min(params["theta"], 1.5))
            lp.tropismS = max(0.0, min(params["tropismS"], 0.15))
            lp.tropismAge = max(0.0, params["tropismAge"])
            lp.tropismExponent = max(0.3, min(params["tropismExponent"], 4.0))
            lp.collarLength = max(0.0, min(params["collarLength"], params["lmax"] * 0.3))

            # Curvature spline — clamp kappa to safe range
            phi = [0.0, 0.25, 0.5, 0.75, 1.0]
            kappa = [max(0.0, min(0.2, params[f"curv_k{i}"])) for i in range(5)]
            lp.leafCurvaturePhi = phi
            lp.leafCurvatureKappa = kappa

        setup_successor_where(plant)
        plant.initialize(False)
        plant.simulate(day)

        # Extract skeletons
        skeletons = {}
        leaves = plant.getOrgans(4)
        for leaf in leaves:
            st = leaf.getParameter("subType")
            pos = int(st) - 1  # subType 2 → position 1
            nodes = leaf.getNodes()
            if len(nodes) < 2:
                continue
            skel = np.array([[n.x, n.y, n.z] for n in nodes])
            skeletons[pos] = skel

        return skeletons

    except Exception as e:
        # Log failures with Gompertz context for debugging
        import traceback
        gomp_info = {pos: f"gf={p.get('gf','?')},lmax={p.get('lmax','?')},r={p.get('r','?')}"
                     for pos, p in leaf_params_by_position.items()
                     if p.get("gf") == 4}
        print(f"  [grow_and_extract FAILED] day={day}: {type(e).__name__}: {e}"
              f"{f' Gompertz leaves: {gomp_info}' if gomp_info else ''}",
              file=sys.stderr, flush=True)
        traceback.print_exc(file=sys.stderr)
        return None


def grow_and_loft(xml_path, day, leaf_params_by_position, stem_params):
    """Grow + loft → per-leaf mesh points for mesh-level comparison."""
    try:
        plant = pb.Plant()
        plant.readParameters(str(xml_path))

        sp = plant.getOrganRandomParameter(3, 1)
        if stem_params:
            sp.lmax = stem_params["lmax"]
            sp.r = stem_params["r"]
            sp.ln = stem_params["ln"]
            sp.lb = stem_params["lb"]

        for pos, params in leaf_params_by_position.items():
            sub_type = pos + 1
            try:
                lp = plant.getOrganRandomParameter(4, sub_type)
            except Exception:
                continue

            gf = int(params.get("gf", lp.gf))
            if gf != lp.gf:
                lp.gf = gf
                lp.f_gf = plant.createGrowthFunction(gf)

            lp.lmax = params["lmax"]
            lp.r = params["r"]
            lp.theta = params["theta"]
            lp.tropismS = params["tropismS"]
            lp.tropismAge = params["tropismAge"]
            lp.tropismExponent = params["tropismExponent"]
            lp.collarLength = params["collarLength"]
            phi = [0.0, 0.25, 0.5, 0.75, 1.0]
            kappa = [params["curv_k0"], params["curv_k1"], params["curv_k2"],
                     params["curv_k3"], params["curv_k4"]]
            lp.leafCurvaturePhi = phi
            lp.leafCurvatureKappa = kappa

        setup_successor_where(plant)
        plant.initialize(False)
        plant.simulate(day)

        organs = extract_organs_for_lofter(plant)
        mesh = loft_organs(organs, subdivide=True, smooth=True)

        return mesh

    except Exception:
        return None


# ---------------------------------------------------------------------------
# Objective function
# ---------------------------------------------------------------------------

def skeleton_chamfer(skel1, skel2):
    """Chamfer distance between two skeletons (polylines)."""
    if len(skel1) < 2 or len(skel2) < 2:
        return 50.0  # penalty
    tree1 = KDTree(skel1)
    tree2 = KDTree(skel2)
    d1, _ = tree2.query(skel1)
    d2, _ = tree1.query(skel2)
    return float((d1.mean() + d2.mean()) / 2.0)


def evaluate_leaf(params_vec, pos, ref_stages, xml_path, stem_params,
                  other_leaf_params, bounds, gf=3):
    """Objective for a single leaf: skeleton Chamfer across ALL stages."""
    params = _params_to_dict(params_vec, bounds)
    params["gf"] = gf
    all_leaf_params = dict(other_leaf_params)
    all_leaf_params[pos] = params

    total_error = 0.0
    n_compared = 0

    for stage in ref_stages:
        day = stage["day"]
        ref_leaf = next((l for l in stage["leaves"] if l.position == pos), None)
        if ref_leaf is None or ref_leaf.length < 3:
            continue

        ref_skel = np.array(ref_leaf.skeleton)

        skeletons = grow_and_extract(xml_path, day, all_leaf_params, stem_params)
        if skeletons is None or pos not in skeletons:
            total_error += 30.0
            n_compared += 1
            continue

        cpb_skel = skeletons[pos]
        chamfer = skeleton_chamfer(cpb_skel, ref_skel)
        total_error += chamfer
        n_compared += 1

    return total_error / max(n_compared, 1)


def evaluate_stem(params_vec, ref_stages, xml_path, leaf_params_by_position, bounds):
    """Objective for stem: height + internode pattern across stages."""
    params = _params_to_dict(params_vec, bounds)

    total_error = 0.0
    n_compared = 0

    for stage in ref_stages:
        day = stage["day"]
        ref_stem = stage["stem"]
        if ref_stem is None or ref_stem.height < 1:
            continue

        skeletons = grow_and_extract(xml_path, day, leaf_params_by_position, params)
        if skeletons is None:
            total_error += 100.0
            n_compared += 1
            continue

        # Compare stem height via growing plant
        try:
            plant = pb.Plant()
            plant.readParameters(str(xml_path))
            sp = plant.getOrganRandomParameter(3, 1)
            sp.lmax = params["lmax"]
            sp.r = params["r"]
            sp.ln = params["ln"]
            sp.lb = params["lb"]
            setup_successor_where(plant)
            plant.initialize(False)
            plant.simulate(day)
            stems = plant.getOrgans(3)
            if stems:
                stem_nodes = stems[0].getNodes()
                if len(stem_nodes) >= 2:
                    stem_skel = np.array([[n.x, n.y, n.z] for n in stem_nodes])
                    cpb_height = abs(stem_skel[:, 2].max() - stem_skel[:, 2].min())
                    height_err = abs(cpb_height - ref_stem.height)
                    total_error += height_err
                    n_compared += 1
        except Exception:
            total_error += 100.0
            n_compared += 1

    return total_error / max(n_compared, 1)


# ---------------------------------------------------------------------------
# CMA-ES fitting
# ---------------------------------------------------------------------------

def fit_stem(ref_stages, xml_path, leaf_params, n_evals=200, verbose=False):
    """Fit stem parameters with CMA-ES using reference-derived init/bounds."""
    x0, stem_bounds = compute_stem_init_and_bounds(ref_stages)

    if verbose:
        print("\n[Stem] Fitting stem parameters...")
        print(f"  Reference-derived x0:")
        for (name, lo, hi), val in zip(stem_bounds, x0):
            print(f"    {name:>6s}: x0={val:7.2f}  bounds=[{lo:.2f}, {hi:.2f}]")

    sigma0 = 0.3

    bounds_lo = [b[1] for b in stem_bounds]
    bounds_hi = [b[2] for b in stem_bounds]
    ranges = [hi - lo for lo, hi in zip(bounds_lo, bounds_hi)]

    x0_scaled = [np.clip((x - lo) / rng, 0.01, 0.99)
                 for x, lo, rng in zip(x0, bounds_lo, ranges)]

    def objective(x_scaled):
        x_real = [lo + x * rng for x, lo, rng in zip(x_scaled, bounds_lo, ranges)]
        return evaluate_stem(x_real, ref_stages, xml_path, leaf_params, stem_bounds)

    es = cma.CMAEvolutionStrategy(x0_scaled, sigma0, {
        "bounds": [0, 1],
        "maxfevals": n_evals,
        "verbose": -9,
        "seed": 42,
    })

    while not es.stop():
        solutions = es.ask()
        fitnesses = [objective(s) for s in solutions]
        es.tell(solutions, fitnesses)

    best_scaled = es.result.xbest
    best_real = [lo + x * rng for x, lo, rng in zip(best_scaled, bounds_lo, ranges)]
    best_params = _params_to_dict(best_real, stem_bounds)
    best_fitness = es.result.fbest

    if verbose:
        print(f"  Best stem: lmax={best_params['lmax']:.1f}, r={best_params['r']:.2f}, "
              f"ln={best_params['ln']:.1f}, lb={best_params['lb']:.1f}")
        print(f"  Height error: {best_fitness:.1f}cm")

    return best_params, best_fitness


def fit_leaf(pos, ref_stages, xml_path, stem_params, other_leaf_params,
             n_evals=300, verbose=False, n_parallel=1, gf=3,
             gompertz_init=None):
    """Fit one leaf position with CMA-ES using reference-derived init/bounds.

    Args:
        gf: growth function type (3=CWLimited/exponential, 4=Gompertz)
        gompertz_init: ignored (kept for API compat). Init is now computed
                       from reference data via compute_leaf_init_and_bounds.
    """
    # Check that this leaf exists in reference data
    has_ref = any(
        any(l.position == pos and l.length > 2 for l in s["leaves"])
        for s in ref_stages
    )
    if not has_ref:
        if verbose:
            print(f"\n[Leaf {pos}] Skipped (no reference leaf)")
        return None, 999.0

    # Compute init and bounds from reference data
    x0, bounds = compute_leaf_init_and_bounds(pos, ref_stages, gf=gf)

    if verbose:
        gf_name = "Gompertz" if gf == 4 else "exp"
        print(f"\n[Leaf {pos}] Fitting ({n_evals} evals, gf={gf} {gf_name})...")
        print(f"  Reference-derived x0:")
        for (name, lo, hi), val in zip(bounds, x0):
            print(f"    {name:>16s}: x0={val:8.3f}  bounds=[{lo:.3f}, {hi:.3f}]")

    # Scale to [0,1]
    bounds_lo = [b[1] for b in bounds]
    bounds_hi = [b[2] for b in bounds]
    ranges = [hi - lo for lo, hi in zip(bounds_lo, bounds_hi)]

    x0_scaled = [np.clip((x - lo) / rng, 0.02, 0.98)
                 for x, lo, rng in zip(x0, bounds_lo, ranges)]

    def objective(x_scaled):
        x_real = [lo + x * rng for x, lo, rng in zip(x_scaled, bounds_lo, ranges)]
        return evaluate_leaf(x_real, pos, ref_stages, xml_path, stem_params,
                             other_leaf_params, bounds, gf=gf)

    popsize = max(8, 2 * N_LEAF_PARAMS)
    es = cma.CMAEvolutionStrategy(x0_scaled, 0.15, {
        "bounds": [0, 1],
        "maxfevals": n_evals,
        "verbose": -9,
        "seed": 42 + pos,
        "popsize": popsize,
    })

    gen = 0
    best_fitness = float("inf")
    while not es.stop():
        solutions = es.ask()
        fitnesses = [objective(s) for s in solutions]
        es.tell(solutions, fitnesses)
        gen += 1
        if es.result.fbest < best_fitness:
            best_fitness = es.result.fbest
            if verbose and gen % 5 == 0:
                print(f"    Gen {gen}: best={best_fitness:.2f}cm")

    best_scaled = es.result.xbest
    best_real = [lo + x * rng for x, lo, rng in zip(best_scaled, bounds_lo, ranges)]
    best_params = _params_to_dict(best_real, bounds)
    best_params["gf"] = gf

    if verbose:
        print(f"  Result: lmax={best_params['lmax']:.1f}, r={best_params['r']:.2f}, "
              f"theta={math.degrees(best_params['theta']):.0f}\u00b0, "
              f"tropS={best_params['tropismS']:.4f}, "
              f"tropAge={best_params['tropismAge']:.1f}")
        print(f"  Skeleton Chamfer: {best_fitness:.2f}cm ({gen} generations)")

    return best_params, best_fitness


# ---------------------------------------------------------------------------
# XML export
# ---------------------------------------------------------------------------

def export_fitted_xml(xml_path, output_path, stem_params, leaf_params_by_position):
    """Write a new XML with the fitted parameters."""
    import xml.etree.ElementTree as ET

    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Update stem
    stem_elem = root.find(".//stem[@subType='1']")
    if stem_elem is not None and stem_params:
        for param_elem in stem_elem.findall("parameter"):
            name = param_elem.get("name")
            if name == "lmax":
                param_elem.set("value", str(stem_params["lmax"]))
            elif name == "r":
                param_elem.set("value", str(stem_params["r"]))
            elif name == "ln":
                param_elem.set("value", str(stem_params["ln"]))
            elif name == "lb":
                param_elem.set("value", str(stem_params["lb"]))

    # Update each leaf
    for leaf_elem in root.findall("leaf"):
        sub_type = int(leaf_elem.get("subType", "0"))
        pos = sub_type - 1  # subType 2 → position 1
        if pos not in leaf_params_by_position:
            continue

        params = leaf_params_by_position[pos]

        # Remove old curvature params
        for old in leaf_elem.findall(".//parameter[@name='leafCurvature']"):
            leaf_elem.remove(old)

        for param_elem in leaf_elem.findall("parameter"):
            name = param_elem.get("name")
            if name == "gf" and "gf" in params:
                param_elem.set("value", str(int(params["gf"])))
            elif name == "lmax":
                param_elem.set("value", str(params["lmax"]))
            elif name == "r":
                param_elem.set("value", str(params["r"]))
            elif name == "theta":
                param_elem.set("value", str(params["theta"]))
                param_elem.set("dev", str(params["theta"] * 0.1))
            elif name == "tropismS":
                param_elem.set("value", str(params["tropismS"]))
            elif name == "tropismAge":
                param_elem.set("value", str(params["tropismAge"]))
            elif name == "collarLength":
                param_elem.set("value", str(params["collarLength"]))

        # Set tropismExponent (may not exist as element)
        found_exp = False
        for param_elem in leaf_elem.findall("parameter"):
            if param_elem.get("name") == "tropismExponent":
                param_elem.set("value", str(params["tropismExponent"]))
                found_exp = True
        if not found_exp:
            pe = ET.SubElement(leaf_elem, "parameter")
            pe.set("name", "tropismExponent")
            pe.set("value", str(params["tropismExponent"]))

        # Add curvature spline
        phi = [0.0, 0.25, 0.5, 0.75, 1.0]
        kappa = [params["curv_k0"], params["curv_k1"], params["curv_k2"],
                 params["curv_k3"], params["curv_k4"]]
        for p, k in zip(phi, kappa):
            cp = ET.SubElement(leaf_elem, "parameter")
            cp.set("name", "leafCurvature")
            cp.set("phi", f"{p:.4f}")
            cp.set("kappa", f"{k:.6f}")

    tree.write(output_path, encoding="utf-8", xml_declaration=True)
    return output_path


# ---------------------------------------------------------------------------
# Deformation optimization (4 Optuna features via diff_lofter)
# ---------------------------------------------------------------------------

def optimize_deformations(ref_stages, xml_path, stem_params, leaf_params_by_position,
                          n_steps=200, verbose=False):
    """GPU deformation optimization using 4 extended features.

    For each stage, grows plant with fitted structural params, then optimizes
    baseline spline deformations + 4 extended features (out_of_plane_curv,
    asymmetry, edge_curl, cross_section_profile) via Adam gradient descent
    against the reference point cloud.

    Returns dict with per-stage deformation Chamfer and control points.
    """
    import torch
    from dart.coupling.experimental.fitting.multistage_optimizer import _optimize_deformations_single

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if verbose:
        print(f"\n  Deformation optimization: {n_steps} Adam steps, device={device}")

    deform_results = {"per_stage": [], "per_leaf_deform_chamfer": {}}

    for stage in ref_stages:
        day = stage["day"]

        # Build target point cloud from all reference leaf meshes
        ref_pts_list = []
        for leaf in stage["leaves"]:
            if leaf.position in stage.get("leaf_mesh_points", {}):
                ref_pts_list.append(stage["leaf_mesh_points"][leaf.position])
        if not ref_pts_list:
            continue

        target_np = np.concatenate(ref_pts_list, axis=0)
        target_pc = torch.tensor(target_np, dtype=torch.float32, device=device)

        # Grow plant and extract organs for diff_lofter
        organs = _grow_for_deformations(xml_path, day, leaf_params_by_position, stem_params)
        if not organs:
            continue

        # Run deformation optimization
        deform_chamfer, best_params = _optimize_deformations_single(
            organs, target_pc, device=device, n_steps=n_steps, lr=0.03)

        if verbose:
            print(f"    Stage {stage['stage']} (day {day:.0f}): "
                  f"deform_chamfer={deform_chamfer:.2f}cm")

        # Map leaf indices back to positions using organ metadata
        leaf_idx = 0
        position_keyed_params = {}
        for organ in organs:
            if organ.get('type') not in ('stem', 'root'):
                if leaf_idx in best_params:
                    pos = organ.get('position')
                    if pos is not None:
                        position_keyed_params[pos] = best_params[leaf_idx]
                leaf_idx += 1

        deform_results["per_stage"].append({
            "stage": stage["stage"], "day": day,
            "deform_chamfer": deform_chamfer,
            "deform_params": position_keyed_params,
        })

    # Build per-leaf average CPs across stages (for coupling pipeline use)
    leaf_cp_accum = {}  # pos -> list of extended_cp dicts
    for sr in deform_results["per_stage"]:
        for pos, params in sr["deform_params"].items():
            if "extended_cp" in params:
                if pos not in leaf_cp_accum:
                    leaf_cp_accum[pos] = []
                leaf_cp_accum[pos].append(params["extended_cp"])

    # Average across stages
    avg_cps = {}
    for pos, cp_list in leaf_cp_accum.items():
        avg = {}
        for feat_name in cp_list[0]:
            vals = [cp[feat_name] for cp in cp_list]
            avg[feat_name] = [float(np.mean([v[i] for v in vals]))
                              for i in range(len(vals[0]))]
        avg_cps[pos] = avg
    deform_results["per_leaf_avg_cps"] = avg_cps

    return deform_results


def _grow_for_deformations(xml_path, day, leaf_params_by_position, stem_params):
    """Grow plant and return organ dicts suitable for diff_lofter."""
    try:
        plant = pb.Plant()
        plant.readParameters(str(xml_path))

        sp = plant.getOrganRandomParameter(3, 1)
        if stem_params:
            sp.lmax = stem_params["lmax"]
            sp.r = stem_params["r"]
            sp.ln = stem_params["ln"]
            sp.lb = stem_params["lb"]

        for pos, params in leaf_params_by_position.items():
            sub_type = pos + 1
            try:
                lp = plant.getOrganRandomParameter(4, sub_type)
            except Exception:
                continue

            gf = int(params.get("gf", lp.gf))
            if gf != lp.gf:
                lp.gf = gf
                lp.f_gf = plant.createGrowthFunction(gf)

            lp.lmax = params["lmax"]
            lp.r = params["r"]
            lp.theta = params["theta"]
            lp.tropismS = params["tropismS"]
            lp.tropismAge = params["tropismAge"]
            lp.tropismExponent = params["tropismExponent"]
            lp.collarLength = params["collarLength"]
            phi = [0.0, 0.25, 0.5, 0.75, 1.0]
            kappa = [params[f"curv_k{i}"] for i in range(5)]
            lp.leafCurvaturePhi = phi
            lp.leafCurvatureKappa = kappa

        setup_successor_where(plant)
        plant.initialize(False)
        plant.simulate(day)

        organs = extract_organs_for_lofter(plant)

        # Convert to diff_lofter format, tracking leaf position for CP mapping
        organ_dicts = []
        leaf_counter = 0
        leaves = plant.getOrgans(4)
        for organ in organs:
            d = {
                'skeleton': np.array(organ['skeleton']),
                'widths': np.array(organ['widths']),
                'type': organ.get('type', 'leaf'),
            }
            if organ.get('type') not in ('stem', 'root'):
                # Map leaf to position via subType
                if leaf_counter < len(leaves):
                    st = int(leaves[leaf_counter].getParameter('subType'))
                    d['position'] = st - 1
                leaf_counter += 1
            organ_dicts.append(d)
        return organ_dicts

    except Exception as e:
        print(f"    [_grow_for_deformations FAILED] day={day}: {e}",
              file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# Post-fit analysis
# ---------------------------------------------------------------------------

def analyze_fit(ref_stages, xml_path, stem_params, leaf_params_by_position,
                export_dir, verbose=False):
    """Run final analysis: per-leaf skeleton + mesh Chamfer at each stage.

    Exports grown OBJs for visual comparison.
    """
    results = {"stages": [], "per_leaf_skeleton_chamfer": {}, "per_leaf_mesh_chamfer": {},
               "model_limitations": []}

    positions = sorted(leaf_params_by_position.keys())

    for stage in ref_stages:
        day = stage["day"]
        if verbose:
            print(f"\n  Analyzing stage {stage['stage']} (day {day:.0f})...")

        # Grow with fitted params
        skeletons = grow_and_extract(xml_path, day, leaf_params_by_position,
                                     stem_params)
        mesh = grow_and_loft(xml_path, day, leaf_params_by_position, stem_params)

        stage_result = {"stage": stage["stage"], "day": day,
                        "skeleton_chamfer": {}, "mesh_chamfer": {}}

        for pos in positions:
            ref_leaf = next((l for l in stage["leaves"] if l.position == pos), None)
            if ref_leaf is None or ref_leaf.length < 3:
                continue

            ref_skel = np.array(ref_leaf.skeleton)

            # Skeleton Chamfer (CPlantBox blame)
            if skeletons and pos in skeletons:
                sc = skeleton_chamfer(skeletons[pos], ref_skel)
                stage_result["skeleton_chamfer"][pos] = sc
                if pos not in results["per_leaf_skeleton_chamfer"]:
                    results["per_leaf_skeleton_chamfer"][pos] = []
                results["per_leaf_skeleton_chamfer"][pos].append(sc)

            # Mesh Chamfer (CPlantBox + lofter blame)
            if mesh and pos in stage.get("leaf_mesh_points", {}):
                ref_pts = stage["leaf_mesh_points"][pos]
                grown_pts = mesh.vertices  # whole mesh — approximate
                mc = _chamfer_distance(grown_pts, ref_pts)
                stage_result["mesh_chamfer"][pos] = mc
                if pos not in results["per_leaf_mesh_chamfer"]:
                    results["per_leaf_mesh_chamfer"][pos] = []
                results["per_leaf_mesh_chamfer"][pos].append(mc)

        results["stages"].append(stage_result)

        # Export grown OBJ
        if mesh and export_dir:
            obj_path = Path(export_dir) / f"fitted_stage_{stage['stage']:02d}.obj"
            with open(obj_path, "w") as f:
                f.write(f"# Fitted CPlantBox stage {stage['stage']} day {day:.0f}\n")
                for v in mesh.vertices:
                    f.write(f"v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}\n")
                for tri in mesh.indices:
                    f.write(f"f {tri[0]+1} {tri[1]+1} {tri[2]+1}\n")

    # Summarize
    if verbose:
        print("\n--- Fit Summary ---")
    for pos in positions:
        sc_list = results["per_leaf_skeleton_chamfer"].get(pos, [])
        mc_list = results["per_leaf_mesh_chamfer"].get(pos, [])
        mean_sc = np.mean(sc_list) if sc_list else 999
        mean_mc = np.mean(mc_list) if mc_list else 999
        if verbose:
            print(f"  Leaf {pos}: skeleton={mean_sc:.2f}cm, mesh={mean_mc:.2f}cm")

        if mean_sc > 3.0:
            results["model_limitations"].append({
                "type": "cplantbox",
                "leaf_position": pos,
                "skeleton_chamfer": mean_sc,
                "description": f"Leaf {pos}: skeleton Chamfer {mean_sc:.1f}cm despite optimization. "
                               f"CPlantBox growth model cannot reproduce this leaf shape.",
            })
        if mc_list and mean_mc > mean_sc + 1.0:
            results["model_limitations"].append({
                "type": "lofter",
                "leaf_position": pos,
                "mesh_chamfer": mean_mc,
                "skeleton_chamfer": mean_sc,
                "description": f"Leaf {pos}: mesh Chamfer {mean_mc:.1f}cm but skeleton only "
                               f"{mean_sc:.1f}cm. Lofter adds {mean_mc - mean_sc:.1f}cm error.",
            })

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Fit CPlantBox XML to match OBJ reference models via CMA-ES")
    parser.add_argument("export_dir",
                        help="Directory with maize_stage_*.obj files")
    parser.add_argument("--output", "-o", default="output/fit_result",
                        help="Output directory")
    parser.add_argument("--xml", default=None,
                        help="Template XML (default: dart/coupling/data/maize_calibrated.xml)")
    parser.add_argument("--evals", type=int, default=300,
                        help="CMA-ES evaluations per leaf (default: 300)")
    parser.add_argument("--stem-evals", type=int, default=200,
                        help="CMA-ES evaluations for stem (default: 200)")
    parser.add_argument("--workers", "-w", type=int, default=1,
                        help=f"Parallel workers for leaf fitting (default: 1)")
    parser.add_argument("--gompertz", nargs="*", type=int, default=None,
                        help="Leaf positions to fit with Gompertz (gf=4). "
                             "If flag given with no args, uses positions 5-10.")
    parser.add_argument("--gompertz-init", default=None,
                        help="(deprecated, ignored) Gompertz init is now computed "
                             "from reference data automatically")
    parser.add_argument("--with-deformations", action="store_true",
                        help="Run GPU deformation optimization (4 Optuna features) "
                             "after CMA-ES structural fitting")
    parser.add_argument("--deform-steps", type=int, default=200,
                        help="Adam steps per stage for deformation optimization (default: 200)")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    # Resolve Gompertz positions
    if args.gompertz is not None:
        gompertz_positions = set(args.gompertz) if args.gompertz else set(range(5, 11))
    else:
        gompertz_positions = set()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    xml_path = args.xml or str(_COUPLING_DIR / "data" / "maize_calibrated.xml")

    print("=" * 70)
    print("CPlantBox FIT-TO-REFERENCE (CMA-ES)")
    print("=" * 70)
    print(f"Reference: {args.export_dir}")
    print(f"Template XML: {xml_path}")
    print(f"Output: {args.output}")
    print(f"Evals/leaf: {args.evals}, Stem evals: {args.stem_evals}")
    if gompertz_positions:
        print(f"Gompertz (gf=4) positions: {sorted(gompertz_positions)}")

    # Step 1: Load reference
    print("\n[1/5] Loading reference OBJ models...")
    ref_stages = load_reference(args.export_dir, verbose=args.verbose)

    # Get all leaf positions
    all_positions = sorted(set(
        l.position for s in ref_stages for l in s["leaves"] if l.length > 5))
    print(f"  Fitting {len(all_positions)} leaf positions: {all_positions}")

    # Ensure XML has subtypes for all positions
    prep_xml = output_dir / "template_prepared.xml"
    print(f"\n[2/5] Preparing template XML for {len(all_positions)} leaves...")
    ensure_xml_has_all_subtypes(xml_path, str(prep_xml), all_positions)
    xml_path = str(prep_xml)

    # Sanity check: can CPlantBox grow with this XML?
    print("\n  Sanity check: growing test plant at day 40...")
    try:
        test_plant = pb.Plant()
        test_plant.readParameters(xml_path)
        setup_successor_where(test_plant)
        test_plant.initialize(False)
        test_plant.simulate(40)
        test_leaves = test_plant.getOrgans(4)
        test_stems = test_plant.getOrgans(3)
        print(f"  OK: {len(test_leaves)} leaves, {len(test_stems)} stems")
        for leaf in test_leaves[:3]:
            st = int(leaf.getParameter("subType"))
            nodes = leaf.getNodes()
            zs = [n.z for n in nodes]
            print(f"    subType={st} (pos {st-1}): {len(nodes)} nodes, "
                  f"Z=[{min(zs):.1f}, {max(zs):.1f}]")
        # Show all subtypes present
        all_sts = sorted(set(int(l.getParameter("subType")) for l in test_leaves))
        print(f"  SubTypes present: {all_sts}")
        print(f"  → Positions: {[st-1 for st in all_sts]}")
    except Exception as e:
        print(f"  FAILED: {e}")
        print("  Cannot grow plants with this XML. Fix the template first.")
        sys.exit(1)

    # Step 3: Fit stem
    print("\n[3/5] Fitting stem...")
    # Build init leaf params from reference for stem fitting context
    init_leaf_params = {}
    for pos in all_positions:
        gf = 4 if pos in gompertz_positions else 3
        x0, leaf_bounds = compute_leaf_init_and_bounds(pos, ref_stages, gf=gf)
        params = {name: val for (name, _lo, _hi), val in zip(leaf_bounds, x0)}
        if gf == 4:
            params["gf"] = 4
        init_leaf_params[pos] = params

    stem_params, stem_error = fit_stem(ref_stages, xml_path, init_leaf_params,
                                        n_evals=args.stem_evals, verbose=args.verbose)

    # Step 4: Fit leaves — all leaves in parallel (each CMA-ES runs sequentially)
    n_leaf_parallel = min(args.workers, len(all_positions))
    print(f"\n[4/5] Fitting {len(all_positions)} leaves "
          f"({args.evals} evals each, {n_leaf_parallel} in parallel)...")

    fitted_leaf_params = dict(init_leaf_params)  # start with initial guesses
    t0 = time.time()

    def _leaf_gf(pos):
        """Return gf for a given leaf position."""
        return 4 if pos in gompertz_positions else 3

    if n_leaf_parallel > 1:
        # Parallel across leaves
        with ProcessPoolExecutor(max_workers=n_leaf_parallel) as executor:
            futures = {}
            for pos in all_positions:
                other = {p: v for p, v in fitted_leaf_params.items() if p != pos}
                f = executor.submit(fit_leaf, pos, ref_stages, xml_path,
                                    stem_params, other, args.evals,
                                    args.verbose, gf=_leaf_gf(pos))
                futures[f] = pos

            for future in as_completed(futures):
                pos = futures[future]
                try:
                    params, fitness = future.result()
                    if params:
                        fitted_leaf_params[pos] = params
                        gf_label = " [Gompertz]" if pos in gompertz_positions else ""
                        print(f"  Leaf {pos}{gf_label}: Chamfer={fitness:.2f}cm")
                except Exception as e:
                    print(f"  Leaf {pos}: FAILED ({e})")
    else:
        # Sequential
        for pos in all_positions:
            other = {p: v for p, v in fitted_leaf_params.items() if p != pos}
            params, fitness = fit_leaf(pos, ref_stages, xml_path,
                                       stem_params, other, args.evals,
                                       args.verbose, gf=_leaf_gf(pos))
            if params:
                fitted_leaf_params[pos] = params

    elapsed = time.time() - t0
    print(f"  Fitting took {elapsed:.0f}s")

    # Step 5: Export XML + analysis
    n_steps = 6 if args.with_deformations else 5
    print(f"\n[5/{n_steps}] Exporting fitted XML and analyzing results...")
    fitted_xml = output_dir / "maize_fitted.xml"
    export_fitted_xml(xml_path, str(fitted_xml), stem_params, fitted_leaf_params)
    print(f"  XML: {fitted_xml}")

    # Save fitted params as JSON
    params_json = {
        "stem": stem_params,
        "leaves": {str(pos): params for pos, params in fitted_leaf_params.items()},
    }
    (output_dir / "fitted_params.json").write_text(
        json.dumps(params_json, indent=2))

    # Run analysis + export OBJs
    objs_dir = output_dir / "fitted_objs"
    objs_dir.mkdir(exist_ok=True)
    results = analyze_fit(ref_stages, str(fitted_xml), stem_params,
                          fitted_leaf_params, str(objs_dir), args.verbose)

    (output_dir / "fit_results.json").write_text(
        json.dumps(results, indent=2, default=str))

    # Step 6: Deformation optimization (optional)
    if args.with_deformations:
        print(f"\n[6/{n_steps}] Deformation optimization "
              f"(4 features, {args.deform_steps} Adam steps)...")
        t0_deform = time.time()
        deform_results = optimize_deformations(
            ref_stages, str(fitted_xml), stem_params, fitted_leaf_params,
            n_steps=args.deform_steps, verbose=args.verbose)
        deform_elapsed = time.time() - t0_deform
        print(f"  Deformation optimization took {deform_elapsed:.0f}s")

        # Save deformation results (per-stage Chamfer + per-leaf average CPs)
        deform_json = output_dir / "deformation_results.json"
        deform_save = {
            "per_stage": [
                {"stage": sr["stage"], "day": sr["day"],
                 "deform_chamfer": sr["deform_chamfer"]}
                for sr in deform_results["per_stage"]
            ],
            # Per-leaf average CPs for coupling pipeline use
            "per_leaf_avg_cps": {
                str(pos): cps
                for pos, cps in deform_results.get("per_leaf_avg_cps", {}).items()
            },
        }
        deform_json.write_text(json.dumps(deform_save, indent=2))
        print(f"  Saved deformation CPs: {deform_json}")

        # Print deformation summary
        deform_chamfers = [sr["deform_chamfer"] for sr in deform_results["per_stage"]]
        if deform_chamfers:
            print(f"  Deformation Chamfer: mean={np.mean(deform_chamfers):.2f}cm, "
                  f"best={min(deform_chamfers):.2f}cm")
        avg_cps = deform_results.get("per_leaf_avg_cps", {})
        if avg_cps:
            print(f"  Per-leaf average CPs: {len(avg_cps)} leaves "
                  f"({', '.join(sorted(str(p) for p in avg_cps))})")
        results["deformation_results"] = deform_save

    # Print summary
    print("\n" + "=" * 70)
    print("FIT COMPLETE")
    print("=" * 70)
    print(f"  Fitted XML: {fitted_xml}")
    print(f"  Fitted OBJs: {objs_dir}/")

    skel_errors = []
    for pos in sorted(fitted_leaf_params.keys()):
        sc = results["per_leaf_skeleton_chamfer"].get(pos, [])
        if sc:
            mean_sc = np.mean(sc)
            skel_errors.append(mean_sc)
            print(f"  Leaf {pos:>2}: skeleton Chamfer = {mean_sc:.2f}cm")

    if skel_errors:
        print(f"\n  Mean skeleton Chamfer: {np.mean(skel_errors):.2f}cm")
        print(f"  Best leaf: {np.min(skel_errors):.2f}cm")
        print(f"  Worst leaf: {np.max(skel_errors):.2f}cm")

    if results["model_limitations"]:
        print(f"\n  MODEL LIMITATIONS ({len(results['model_limitations'])}):")
        for lim in results["model_limitations"]:
            print(f"    [{lim['type']}] {lim['description']}")

    print(f"\n  Total time: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
