"""Pheno4D morphology JSON adapter for G1-to-G3 lofting engine.

Reads Pheno4D morphology parameters (extracted from laser scans) and
converts them to organ dicts compatible with g1_to_g3.loft_organs().

Includes width sanity-checking based on area constraints and
skeleton smoothing to reduce lofting artifacts.
"""

import json
import warnings
import numpy as np
from scipy.ndimage import median_filter


def _smooth_skeleton(skeleton, window=5, passes=2):
    """Smooth skeleton with iterated moving average, preserving endpoints."""
    skel = np.array(skeleton, dtype=np.float64)
    n = len(skel)
    if n < window + 2:
        warnings.warn(f"Skeleton too short ({n} points) for smoothing window={window}, returning unsmoothed")
        return skel
    for _ in range(passes):
        pad = window // 2
        smoothed = skel.copy()
        for i in range(pad, n - pad):
            smoothed[i] = skel[i - pad:i + pad + 1].mean(axis=0)
        smoothed[0] = skel[0]
        smoothed[-1] = skel[-1]
        skel = smoothed
    return skel


def _smooth_radii(radii, median_window=7, avg_window=5):
    """Smooth stem radii: median filter removes outlier spikes, then average."""
    r = np.array(radii, dtype=np.float64)
    # Median filter to remove isolated bad values (e.g. 0.1 cm spikes)
    r = median_filter(r, size=median_window, mode='nearest')
    # Moving average for overall smoothness
    n = len(r)
    if n >= avg_window:
        pad = avg_window // 2
        smoothed = r.copy()
        for i in range(pad, n - pad):
            smoothed[i] = r[i - pad:i + pad + 1].mean()
        r = smoothed
    return r


def _cap_widths_by_area(widths, skeleton, target_area):
    """Scale down width profile if it implies more area than measured.

    Integrates the width profile along the skeleton arc length (trapezoidal
    rule) and scales all widths proportionally if the integrated area exceeds
    the target area from the point cloud.
    """
    widths = np.array(widths, dtype=np.float64)
    skel = np.array(skeleton, dtype=np.float64)

    seg_lengths = np.linalg.norm(np.diff(skel, axis=0), axis=1)

    integrated_area = 0.0
    for i in range(len(seg_lengths)):
        w_avg = (widths[i] + widths[min(i + 1, len(widths) - 1)]) / 2.0
        integrated_area += w_avg * seg_lengths[i]

    if integrated_area > target_area * 1.1 and integrated_area > 1e-6:
        scale = target_area / integrated_area
        widths *= scale

    return widths


def extract_organs_from_pheno4d(morphology_json_path: str) -> list[dict]:
    """Load Pheno4D morphology JSON and extract organ descriptions.

    Returns list of organ dicts with keys: type, skeleton, widths, organ_id, name
    """
    with open(morphology_json_path) as f:
        data = json.load(f)

    if "stem" not in data:
        raise ValueError(f"Morphology JSON missing 'stem' key: {morphology_json_path}")
    if "leaves" not in data or "per_leaf" not in data.get("leaves", {}):
        raise ValueError(f"Morphology JSON missing 'leaves.per_leaf': {morphology_json_path}")

    organs = []
    organ_id = 0

    # Stem
    stem = data["stem"]
    skeleton = np.array(stem["skeleton_points_cm"], dtype=np.float64)
    radii = np.array(stem["radii_per_segment_cm"], dtype=np.float64)

    # Smooth stem radii to remove outlier spikes from leaf insertion zones
    radii = _smooth_radii(radii)

    if len(radii) == 0:
        warnings.warn("Stem has no radii data, skipping")
        radii = np.array([0.2])

    # radii has N-1 values for N skeleton points; duplicate last to get N
    diameters = np.concatenate([radii, [radii[-1]]]) * 2.0

    organs.append({
        "type": "stem",
        "skeleton": skeleton,
        "widths": diameters,
        "organ_id": organ_id,
        "name": "stem",
        "node_ids": list(range(len(skeleton))),
    })
    organ_id += 1

    # Leaves
    for leaf in data["leaves"]["per_leaf"]:
        skeleton = np.array(leaf["skeleton_points_cm"], dtype=np.float64)
        widths = np.array(leaf["width_profile_cm"], dtype=np.float64)
        if len(widths) == 0:
            warnings.warn(f"Leaf '{leaf.get('name', '?')}' has no width data, skipping")
            continue
        # widths has N-1 values for N skeleton points; duplicate last to get N
        widths = np.concatenate([widths, [widths[-1]]])

        # Smooth skeleton to reduce sharp bends that cause lofting artifacts
        # Use 2 passes with window=9 for heavily curved leaves
        skeleton = _smooth_skeleton(skeleton, window=9, passes=2)

        # Cap widths if they imply more area than the point cloud measurement
        area = leaf.get("area_cm2", 0)
        if area > 0:
            widths = _cap_widths_by_area(widths, skeleton, area)

        # Per-point normals for twist-aware lofting (preferred over single normal)
        per_point_normals = leaf.get("per_point_normals")
        if per_point_normals is not None:
            per_point_normals = np.array(per_point_normals, dtype=np.float64)

        # Fallback: single global curvature plane normal
        curv_dir = leaf.get("curvature_direction", {})
        plane_normal = curv_dir.get("curvature_plane_normal")
        if plane_normal is not None:
            plane_normal = np.array(plane_normal, dtype=np.float64)

        # Cross-sectional gutter depths (if available from morphology)
        gutter_depths = leaf.get("gutter_depths_cm")
        if gutter_depths is not None:
            gutter_depths = np.array(gutter_depths, dtype=np.float64)
            # Pad to match skeleton length if needed
            if len(gutter_depths) < len(skeleton) - 1:
                gutter_depths = np.concatenate([
                    gutter_depths,
                    np.full(len(skeleton) - 1 - len(gutter_depths), gutter_depths[-1] if len(gutter_depths) > 0 else 0.0)
                ])

        organs.append({
            "type": "leaf",
            "skeleton": skeleton,
            "widths": widths,
            "organ_id": organ_id,
            "name": leaf["name"],
            "plane_normal": plane_normal,
            "per_point_normals": per_point_normals,
            "gutter_depths": gutter_depths,
            "node_ids": list(range(len(skeleton))),
        })
        organ_id += 1

    return organs
