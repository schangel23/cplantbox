"""Main orchestrator: Pheno4D .txt -> CPlantBox MappedSegments -> DART export."""

import os
import glob as globmod

import numpy as np

from .loader import load_pheno4d, load_unlabeled
from .segmenter import segment_maize
from .skeletonizer import (
    skeletonize_stem, skeletonize_leaf, resample_path,
    _skeletonize_stem_pca_binning,
)
from .g1_builder import build_mapped_segments


def _point_to_polyline_dist(pts, skeleton):
    """Distance from each point to nearest skeleton segment."""
    n_pts = len(pts)
    min_dists = np.full(n_pts, np.inf)
    nearest_seg = np.zeros(n_pts, dtype=int)
    for si in range(len(skeleton) - 1):
        p0, p1 = skeleton[si], skeleton[si + 1]
        seg_vec = p1 - p0
        seg_len = np.linalg.norm(seg_vec)
        if seg_len < 1e-8:
            continue
        seg_dir = seg_vec / seg_len
        vecs = pts - p0
        t = np.clip(vecs @ seg_dir, 0, seg_len)
        projs = p0 + np.outer(t, seg_dir)
        dists = np.linalg.norm(pts - projs, axis=1)
        closer = dists < min_dists
        min_dists[closer] = dists[closer]
        nearest_seg[closer] = si
    return min_dists, nearest_seg


def segment_plant(filepath, stem_radius_mult=3.0,
                  dbscan_eps=0.8, angle_weight=3.0, min_leaf_z_range=1.5,
                  min_leaf_points=100, soil_margin_cm=1.0):
    """Label-free geometric segmentation for monocot plants (maize).

    Uses a Z-axis skeleton approach robust to plant lean and pseudostem spread:
    1. Build a stem skeleton via XY-centroid binning along Z (iteratively
       refined to exclude outlier/leaf points).
    2. Estimate stem radius from the tightest section of the skeleton.
    3. Classify points by perpendicular distance from the skeleton.
    4. Cluster leaf candidates using angular position around the axis
       (DBSCAN on XYZ + angular features).

    Args:
        filepath: path to point cloud .txt file
        stem_radius_mult: multiplier on base stem radius for stem/leaf threshold
        dbscan_eps: DBSCAN epsilon for leaf clustering
        angle_weight: weight for angular features in clustering
        min_leaf_z_range: minimum Z extent (cm) for a valid leaf cluster
        min_leaf_points: minimum points for a valid leaf cluster
        soil_margin_cm: passed to load_unlabeled for ground filtering

    Returns:
        dict of organ_name -> np.array([N, 3]) in cm, same format as load_pheno4d()
    """
    from sklearn.cluster import DBSCAN as _DBSCAN

    # 1. Load raw points
    points = load_unlabeled(filepath, soil_margin_cm=soil_margin_cm)
    print(f"[segment] Loaded {len(points):,} plant points")

    # 2. Build stem skeleton: Z-binned XY centroids, iteratively refined
    z_min, z_max = points[:, 2].min(), points[:, 2].max()
    n_slices = 60
    z_edges = np.linspace(z_min, z_max, n_slices + 1)

    # First pass: XY centroids at each Z slice (all points)
    skeleton_pass1 = []
    for i in range(n_slices):
        zmask = (points[:, 2] >= z_edges[i]) & (points[:, 2] < z_edges[i + 1])
        if zmask.sum() >= 3:
            skeleton_pass1.append(points[zmask].mean(axis=0))
    skeleton_pass1 = np.array(skeleton_pass1)
    if skeleton_pass1[0, 2] > skeleton_pass1[-1, 2]:
        skeleton_pass1 = skeleton_pass1[::-1]

    # First pass distances
    dists_pass1, _ = _point_to_polyline_dist(points, skeleton_pass1)

    # Estimate base radius from the bottom 20% where the pseudostem is tightest
    z_20pct = z_min + (z_max - z_min) * 0.2
    base_mask = points[:, 2] < z_20pct
    if base_mask.sum() < 20:
        base_mask = points[:, 2] < (z_min + z_max) / 2
    base_radius = np.median(dists_pass1[base_mask])
    base_radius = max(base_radius, 0.1)  # safety floor

    # Second pass: recompute centroids excluding far points
    refine_thresh = max(base_radius * 4.0, 0.5)
    refined_centers = []
    for i in range(n_slices):
        zmask = (points[:, 2] >= z_edges[i]) & (points[:, 2] < z_edges[i + 1])
        close = zmask & (dists_pass1 < refine_thresh)
        if close.sum() >= 3:
            refined_centers.append(points[close].mean(axis=0))
        elif zmask.sum() >= 3:
            refined_centers.append(points[zmask].mean(axis=0))

    if len(refined_centers) < 2:
        raise ValueError("Could not build stem skeleton")
    stem_skeleton = np.array(refined_centers)
    if stem_skeleton[0, 2] > stem_skeleton[-1, 2]:
        stem_skeleton = stem_skeleton[::-1]
    print(f"[segment] Stem skeleton: {len(stem_skeleton)} nodes, "
          f"z=[{stem_skeleton[0, 2]:.1f}, {stem_skeleton[-1, 2]:.1f}], "
          f"base_radius={base_radius:.3f}")

    # 4. Distance from refined skeleton
    final_dists, nearest_seg = _point_to_polyline_dist(points, stem_skeleton)

    # 5. Classify: fixed threshold from base radius
    stem_threshold = max(base_radius * stem_radius_mult, 0.4)
    labels = np.zeros(len(points), dtype=int)
    labels[final_dists <= stem_threshold] = 1  # stem
    labels[final_dists > stem_threshold] = -1  # leaf candidate

    n_stem = (labels == 1).sum()
    n_cand = (labels == -1).sum()
    print(f"[segment] Threshold: {stem_threshold:.2f} cm → "
          f"stem={n_stem:,}, leaf_cand={n_cand:,}")

    # 6. Cluster leaf candidates with angular features
    leaf_mask = labels == -1
    if leaf_mask.sum() > min_leaf_points:
        leaf_pts = points[leaf_mask]
        leaf_indices = np.where(leaf_mask)[0]
        leaf_nearest = nearest_seg[leaf_mask]
        leaf_vecs = leaf_pts - stem_skeleton[leaf_nearest]
        angles = np.arctan2(leaf_vecs[:, 1], leaf_vecs[:, 0])

        features = np.column_stack([
            leaf_pts,
            np.sin(angles) * angle_weight,
            np.cos(angles) * angle_weight,
        ])

        clustering = _DBSCAN(eps=dbscan_eps, min_samples=30).fit(features)
        cl_labels = clustering.labels_
        unique_clusters = sorted(set(cl_labels) - {-1})

        leaf_count = 0
        for cl in unique_clusters:
            cl_mask = cl_labels == cl
            cl_pts = leaf_pts[cl_mask]
            z_range = cl_pts[:, 2].max() - cl_pts[:, 2].min()
            n_cl = cl_mask.sum()

            if z_range >= min_leaf_z_range and n_cl >= min_leaf_points:
                leaf_count += 1
                labels[leaf_indices[cl_mask]] = leaf_count + 1
            else:
                labels[leaf_indices[cl_mask]] = 1  # absorb into stem

        # DBSCAN noise → stem
        noise = cl_labels == -1
        labels[leaf_indices[noise]] = 1

    # 7. Build organs dict
    organs = {}
    organs['stem'] = points[labels == 1]
    leaf_labels = sorted(set(labels) - {0, 1})
    for i, lid in enumerate(leaf_labels):
        organs[f'leaf_{i + 1}'] = points[labels == lid]

    n_total = sum(len(v) for v in organs.values())
    n_leaves = len(leaf_labels)
    print(f"[segment] Result: {n_total:,} points → stem + {n_leaves} leaves")
    for name, pts in sorted(organs.items()):
        zr = pts[:, 2].max() - pts[:, 2].min()
        print(f"  {name}: {len(pts):,} pts, z=[{pts[:, 2].min():.1f}, "
              f"{pts[:, 2].max():.1f}], height={zr:.1f} cm")

    if 'stem' not in organs:
        raise ValueError("Segmentation failed: no stem identified")

    return organs


def pheno4d_to_cplantbox(filepath, dx=0.5, label_method='collar',
                          n_stem_bins=50, n_leaf_centers=30,
                          use_labels=True):
    """Complete pipeline: Pheno4D .txt -> pb.MappedSegments.

    Args:
        filepath: path to Pheno4D .txt file
        dx: target segment length in cm
        label_method: 'collar' or 'tip' for label column
        n_stem_bins: bins for stem skeletonization
        n_leaf_centers: sample points for leaf L1-medial skeleton
        use_labels: if True (default), use pre-labeled data via load_pheno4d();
                    if False, use label-free segmentation via segment_plant()

    Returns:
        pb.MappedSegments
    """
    organs = load_pheno4d(filepath, label_method) if use_labels else segment_maize(filepath)

    # Skeletonize stem
    stem_skel = None
    stem_points = None
    if 'stem' in organs:
        stem_points = organs['stem']
        raw_skel = skeletonize_stem(stem_points, n_bins=n_stem_bins)
        stem_skel = resample_path(raw_skel, dx=dx)
        print(f"[pipeline] Stem skeleton: {len(stem_skel)} nodes "
              f"(height: {stem_skel[:, 2].max() - stem_skel[:, 2].min():.1f} cm)")
    else:
        raise ValueError("No stem points found in point cloud")

    # Skeletonize leaves
    leaf_skeletons = {}
    leaf_points = {}
    for name, pts in sorted(organs.items()):
        if name.startswith('leaf_'):
            raw_skel = skeletonize_leaf(pts, n_centers=n_leaf_centers)
            leaf_skeletons[name] = resample_path(raw_skel, dx=dx)
            leaf_points[name] = pts
            print(f"[pipeline] {name} skeleton: {len(leaf_skeletons[name])} nodes "
                  f"(length: {_path_length(leaf_skeletons[name]):.1f} cm)")

    # Build MappedSegments
    ms = build_mapped_segments(stem_skel, leaf_skeletons, stem_points,
                               leaf_points, dx=dx)
    return ms


def pheno4d_to_dart(filepath, output_prefix, dx=0.5, label_method='collar'):
    """Full chain: Pheno4D .txt -> MappedSegments -> OBJ + DART mapping JSON.

    Args:
        filepath: path to Pheno4D .txt file
        output_prefix: path prefix for OBJ/JSON output files
        dx: target segment length in cm
        label_method: 'collar' or 'tip'

    Returns:
        dict with MappedSegments and DART export summary
    """
    from plantbox.visualisation.vtk_dart_export import export_vtk_plant_for_dart

    ms = pheno4d_to_cplantbox(filepath, dx=dx, label_method=label_method)

    out_dir = os.path.dirname(output_prefix)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    summary = export_vtk_plant_for_dart(ms, output_prefix)
    summary['mapped_segments'] = ms
    return summary


def process_time_series(plant_dir, output_dir, dx=0.5, label_method='collar'):
    """Process all timesteps for one Pheno4D plant.

    Args:
        plant_dir: directory containing .txt files (e.g. Maize01/)
        output_dir: directory for output files
        dx: segment spacing
        label_method: 'collar' or 'tip'

    Returns:
        list of (filename, MappedSegments) tuples
    """
    os.makedirs(output_dir, exist_ok=True)
    txt_files = sorted(globmod.glob(os.path.join(plant_dir, '*.txt')))

    results = []
    for txt_path in txt_files:
        name = os.path.splitext(os.path.basename(txt_path))[0]
        print(f"\n{'=' * 50}")
        print(f"Processing: {name}")
        print(f"{'=' * 50}")

        try:
            ms = pheno4d_to_cplantbox(txt_path, dx=dx, label_method=label_method)
            results.append((name, ms))
        except Exception as e:
            print(f"[pipeline] ERROR processing {name}: {e}")
            results.append((name, None))

    print(f"\n[pipeline] Processed {len(results)} timesteps, "
          f"{sum(1 for _, ms in results if ms is not None)} successful")
    return results


def _path_length(points):
    """Total arc length of an ordered point sequence."""
    if len(points) < 2:
        return 0.0
    return float(sum(
        ((points[i+1] - points[i]) ** 2).sum() ** 0.5
        for i in range(len(points) - 1)
    ))
