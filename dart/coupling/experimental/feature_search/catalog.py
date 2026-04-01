"""Feature catalog for geometry search.

Defines candidate geometry features that could improve CPlantBox mesh realism.
Each feature specifies:
- What deformation/annotation it adds
- How many spline control points it uses
- Parameter bounds for optimization
- Which layer it belongs to (G1 annotation, mesh gen logic, growth behavior)
"""

import math

# --- Feature Catalog ---
# Each entry: dict with keys:
#   n_cp:    number of spline control points (continuous params per leaf)
#   bounds:  (low, high) for each control point
#   layer:   'annotation' | 'mesh_logic' | 'growth'
#   desc:    human-readable description
#   ramp:    'linear' | 'quadratic' | 'none' — how deformation ramps from base to tip

FEATURE_CATALOG = {
    # --- Existing deformations (baseline, always on) ---
    # These are the 6 original deformation types from the diff lofter.
    # They are NOT part of the search — they're always active as the baseline.

    # --- NEW candidate features for search ---
    # NOTE: The baseline already handles: wave_normal, wave_lateral, twist,
    # curl, edge_ruffle, fold via gradient-optimized spline CPs.
    # These features are ADDITIONAL deformations the baseline can't express.

    # G1 annotation features: data carried on the skeleton
    "width_taper": {
        "n_cp": 5,
        "bounds": (0.2, 2.0),
        "layer": "annotation",
        "desc": "Per-node width multiplier profile (taper from base to tip)",
        "ramp": "none",  # width taper applies uniformly, no ramp needed
    },
    "blade_tilt": {
        "n_cp": 5,
        "bounds": (-math.radians(60), math.radians(60)),
        "layer": "annotation",
        "desc": "Cross-section V-angle relative to gravity (blade tilts left/right)",
        "ramp": "linear",
    },
    "midrib_depth": {
        "n_cp": 5,
        "bounds": (0.0, 3.0),
        "layer": "annotation",
        "desc": "Midrib channel/gutter depth varying along leaf (cm)",
        "ramp": "linear",
    },
    "asymmetry": {
        "n_cp": 5,
        "bounds": (-1.5, 1.5),
        "layer": "annotation",
        "desc": "Left/right width asymmetry: positive = right wider, negative = left wider (cm)",
        "ramp": "linear",
    },
    "out_of_plane_curv": {
        "n_cp": 5,
        "bounds": (-0.15, 0.15),
        "layer": "annotation",
        "desc": "Curvature in binormal direction, perpendicular to growth plane (1/cm)",
        "ramp": "linear",
    },
    "edge_curl": {
        "n_cp": 5,
        "bounds": (-math.radians(45), math.radians(45)),
        "layer": "annotation",
        "desc": "Margin deflection angle — edges curl up or down (radians)",
        "ramp": "linear",
    },

    # Mesh generator logic features
    "cross_section_profile": {
        "n_cp": 5,
        "bounds": (-2.0, 2.0),
        "layer": "mesh_logic",
        "desc": "Cross-section curvature profile: positive = concave (V/U), negative = convex",
        "ramp": "linear",
    },
    "tip_taper_onset": {
        "n_cp": 1,
        "bounds": (0.5, 0.95),
        "layer": "mesh_logic",
        "desc": "Where tip narrowing begins as fraction of leaf length",
        "ramp": "none",
    },
}

# All searchable feature names
SEARCH_FEATURE_NAMES = list(FEATURE_CATALOG.keys())

# Features that modify the skeleton itself (vs. cross-section)
SKELETON_FEATURES = {"out_of_plane_curv"}

# Features that modify width
WIDTH_FEATURES = {"width_taper", "asymmetry", "tip_taper_onset"}

# Features that modify the cross-section shape
CROSS_SECTION_FEATURES = {"blade_tilt", "midrib_depth", "cross_section_profile", "edge_curl"}


def get_active_feature_dims(active_features: set[str]) -> int:
    """Count total continuous parameters for a set of active features.

    This is per-leaf — multiply by n_leaves for total dims.
    """
    total = 0
    for name in active_features:
        if name in FEATURE_CATALOG:
            total += FEATURE_CATALOG[name]["n_cp"]
    return total


def describe_catalog() -> str:
    """Pretty-print the feature catalog."""
    lines = ["Feature Catalog for CPlantBox Geometry Search", "=" * 50]
    for name, spec in FEATURE_CATALOG.items():
        lines.append(
            f"  {name:25s}  {spec['n_cp']} CP  "
            f"[{spec['bounds'][0]:.2f}, {spec['bounds'][1]:.2f}]  "
            f"({spec['layer']})  — {spec['desc']}"
        )
    lines.append(f"\nTotal features: {len(FEATURE_CATALOG)}")
    lines.append(f"Max dims per leaf: {get_active_feature_dims(set(SEARCH_FEATURE_NAMES))}")
    return "\n".join(lines)
