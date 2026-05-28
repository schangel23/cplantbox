"""Template-guided leaf candidate refinement for maize point clouds.

This module bridges graph/track proposals and downstream NURBS fitting.  It
uses a simple maize leaf prior: each leaf is represented by an insertion
height, an azimuth track around the stem axis, and a sheet-like PCA fit.  The
result is still a segmentation proposal, but it is suitable for the expensive
per-leaf NURBS fitter to accept or reject by fit quality.
"""

import numpy as np

try:
    from .segmenter import (
        estimate_virtual_axis,
        to_cylindrical,
        detect_blade_clusters_fullheight,
        identify_leaves,
        assign_all_points,
    )
    from .geodesic_assignment import (
        assign_points_geodesic,
        leaf_quality_gate,
    )
except ImportError:  # direct file loading in diagnostics
    from segmenter import (  # type: ignore
        estimate_virtual_axis,
        to_cylindrical,
        detect_blade_clusters_fullheight,
        identify_leaves,
        assign_all_points,
    )
    from geodesic_assignment import (  # type: ignore
        assign_points_geodesic,
        leaf_quality_gate,
    )


def segment_by_leaf_templates(points, axis=None, n_axis_slices=80,
                              inner_percentile=60, n_detect_slices=120,
                              dbscan_eps_rad=0.25, dbscan_min_samples=4,
                              link_tolerance_rad=0.45,
                              max_fragment_merge_gap_cm=10.0,
                              min_track_z_span=1.5,
                              min_track_points=60,
                              tight_tolerance_rad=0.45,
                              core_fraction=0.15,
                              min_leaf_points=80,
                              min_leaf_z_range=1.5,
                              assignment="geodesic",
                              knn_k=10,
                              knn_max_edge_cm=3.0,
                              knn_core_percentile=35,
                              knn_tip_reach_cm=10.0,
                              apply_quality_gate=True,
                              gate_max_zspan_frac=0.6,
                              gate_max_verticality=0.995,
                              gate_min_elongation=1.6,
                              return_debug=False):
    """Segment a maize crop by track-derived leaf templates.

    Defaults are tuned for cross-row-clipped FP4D crops (≈5 k pts / plant,
    cf. :func:`loader.crop_plant_window` with ``cross_row_window_cm``). On
    Plot04/230621 centre[9] these defaults recover 5 leaf candidates while
    retaining separate same-azimuth tracks at different insertion heights.
    The previous noisier preset was tuned for crops that included ~60 % of
    cross-row floor; pass ``n_detect_slices=50, dbscan_eps_rad=0.45,
    dbscan_min_samples=8, min_track_points=100, min_track_z_span=3.0,
    min_leaf_points=200, min_leaf_z_range=3.0`` to restore that behaviour.

    Args:
        points: ``(N, 3)`` point cloud in cm.
        axis: optional precomputed stem axis.  If omitted, it is estimated
            with :func:`segmenter.estimate_virtual_axis`.
        n_axis_slices: Z-bins for virtual-axis estimation and reassignment.
        inner_percentile: radial percentile used to estimate the pseudostem.
        n_detect_slices: Z-bins used for blade-track detection.
        dbscan_eps_rad: angular DBSCAN epsilon for blade candidates.
        dbscan_min_samples: angular DBSCAN minimum samples.
        link_tolerance_rad: angular tolerance for linking tracks across Z.
        max_fragment_merge_gap_cm: maximum height gap for merging same-leaf
            track fragments. Unlike the legacy global same-angle merge, this
            only joins fragments that are contiguous in height.
        min_track_z_span: minimum track height span in cm.
        min_track_points: minimum points in an initial blade track.
        tight_tolerance_rad: angular tolerance for assigning outer points.
        core_fraction: inner core fraction retained as stem/pseudostem.
        min_leaf_points: minimum assigned points for an accepted leaf.
        min_leaf_z_range: minimum assigned Z range for an accepted leaf.
        return_debug: if True, also return coordinates, tracks, labels, and
            per-leaf template diagnostics.

    Returns:
        ``organs`` dict in the same shape as ``load_pheno4d`` output.  If
        ``return_debug`` is True, returns ``(organs, debug)``.
    """
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must be an (N, 3) array")
    if len(points) == 0:
        raise ValueError("points must not be empty")

    if axis is None:
        axis = estimate_virtual_axis(
            points,
            n_slices=n_axis_slices,
            inner_percentile=inner_percentile,
        )

    r, theta, z_proj, nearest_seg = to_cylindrical(points, axis)
    tracks = detect_blade_clusters_fullheight(
        points,
        r,
        theta,
        z_proj,
        n_z_slices=n_detect_slices,
        dbscan_eps_rad=dbscan_eps_rad,
        dbscan_min_samples=dbscan_min_samples,
        link_tolerance_rad=link_tolerance_rad,
        max_fragment_merge_gap_cm=max_fragment_merge_gap_cm,
        min_leaf_z_span=min_track_z_span,
        min_leaf_points=min_track_points,
    )
    leaves = identify_leaves(tracks)

    if assignment == "geodesic":
        # MonGraphSeg Section 4 spirit: geodesic nearest-instance over a kNN
        # graph, seeded from the blade tracks + pseudostem core. Respects blade
        # connectivity, so a low leaf can no longer absorb the vertical column
        # above it (the failure mode of the slice-wise angular assignment).
        labels = assign_points_geodesic(
            points, tracks, r, z_proj,
            k=knn_k,
            max_edge_cm=knn_max_edge_cm,
            core_percentile=knn_core_percentile,
            tip_reach_cm=knn_tip_reach_cm,
        )
    elif assignment == "angular":
        labels = assign_all_points(
            points,
            r,
            theta,
            z_proj,
            leaves,
            n_z_slices=n_axis_slices,
            core_fraction=core_fraction,
            tight_tolerance_rad=tight_tolerance_rad,
            min_leaf_points=min_leaf_points,
            min_leaf_z_range=min_leaf_z_range,
        )
    else:
        raise ValueError(f"unknown assignment backend: {assignment!r}")

    labels = np.asarray(labels, dtype=int)
    plant_height = float(np.ptp(points[:, 2]))
    median_r = float(np.median(r))

    organs = {"stem": points[labels == 0]}
    active_labels = sorted(set(labels) - {0})
    diagnostics = []
    rejected = []
    out_id = 0
    for label in active_labels:
        leaf_points = points[labels == label]
        if apply_quality_gate:
            leaf_max_r = float(r[labels == label].max())
            keep, reason, metrics = leaf_quality_gate(
                leaf_points, plant_height,
                min_points=min_leaf_points,
                max_zspan_frac=gate_max_zspan_frac,
                max_verticality=gate_max_verticality,
                min_elongation=gate_min_elongation,
                leaf_max_r_cm=leaf_max_r,
                median_plant_r_cm=median_r,
            )
            if not keep:
                # rejected candidates fold back into the stem bucket
                organs["stem"] = np.vstack([organs["stem"], leaf_points])
                labels[labels == label] = 0
                rejected.append({"label": int(label), "reason": reason,
                                 "metrics": metrics})
                continue
        out_id += 1
        name = f"leaf_{out_id}"
        organs[name] = leaf_points
        diagnostics.append(_leaf_template_diagnostics(name, leaf_points))

    if not return_debug:
        return organs

    debug = {
        "axis": axis,
        "r": r,
        "theta": theta,
        "z_proj": z_proj,
        "nearest_seg": nearest_seg,
        "tracks": tracks,
        "leaves": leaves,
        "labels": labels,
        "diagnostics": diagnostics,
        "rejected": rejected,
        "assignment": assignment,
    }
    return organs, debug


def _leaf_template_diagnostics(name, points):
    """Return lightweight fit-quality metrics for a sheet-like leaf prior."""
    if len(points) == 0:
        return {
            "name": name,
            "n_points": 0,
            "length_cm": 0.0,
            "width_cm": 0.0,
            "thickness_rms_cm": 0.0,
            "elongation": 0.0,
            "z_span_cm": 0.0,
        }

    centered = points - points.mean(axis=0)
    cov = np.cov(centered, rowvar=False)
    values, vectors = np.linalg.eigh(cov)
    order = np.argsort(values)[::-1]
    scores = centered @ vectors[:, order]
    length = float(np.ptp(scores[:, 0]))
    width = float(np.ptp(scores[:, 1]))
    thickness_rms = float(np.sqrt(np.mean(scores[:, 2] ** 2)))

    return {
        "name": name,
        "n_points": int(len(points)),
        "length_cm": length,
        "width_cm": width,
        "thickness_rms_cm": thickness_rms,
        "elongation": length / max(width, 1e-8),
        "z_span_cm": float(np.ptp(points[:, 2])),
    }


def write_labelled_xyz(points, labels, output_path):
    """Write ``x y z label`` rows for hand-off to NURBS fitting/QC tools."""
    points = np.asarray(points, dtype=float)
    labels = np.asarray(labels, dtype=int)
    if points.shape[0] != labels.shape[0]:
        raise ValueError("points and labels must have the same length")
    data = np.column_stack([points, labels])
    np.savetxt(output_path, data, fmt="%.8f %.8f %.8f %d")
