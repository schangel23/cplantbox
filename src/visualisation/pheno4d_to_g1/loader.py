"""Load Pheno4D .txt point clouds and separate by organ label.

Also supports FieldPheno4D-style LAS field scans via ``load_las`` and
``separate_plants_along_row`` — the FP4D laser-triangulation robot scans
8 m row strips of multiple plants at sub-mm density (LAS 1.2 pf-3 with a
DEM-normalised ``height`` scalar). The helpers here bridge that format to
the (N, 3 cm) plant-frame array the rest of the pipeline expects.
"""

import numpy as np


LABEL_NAMES = {0: 'soil', 1: 'stem', 2: 'leaf_1', 3: 'leaf_2', 4: 'leaf_3',
               5: 'leaf_4', 6: 'leaf_5', 7: 'leaf_6', 8: 'leaf_7'}


def load_unlabeled(filepath, soil_margin_cm=0.5):
    """Load a point cloud file ignoring any label columns.

    Reads only XYZ coordinates, filters soil by Z threshold, centers on
    the lowest plant point, and converts mm to cm.

    Args:
        filepath: path to .txt file (at least 3 columns: X Y Z ...)
        soil_margin_cm: points below min(Z) + margin (in cm) are removed as soil

    Returns:
        np.array([N, 3]) plant points in cm, centered on base
    """
    data = np.loadtxt(filepath)
    coords = data[:, :3].copy()  # mm

    # Convert mm -> cm first for threshold comparison
    coords /= 10.0

    # Remove soil: points below min(Z) + margin
    z_min = coords[:, 2].min()
    plant_mask = coords[:, 2] >= z_min + soil_margin_cm
    coords = coords[plant_mask]

    if len(coords) == 0:
        raise ValueError(f"No plant points after soil removal in {filepath}")

    # Center on lowest remaining point (approximate stem base)
    base_idx = np.argmin(coords[:, 2])
    base_point = coords[base_idx].copy()
    coords[:, 0] -= base_point[0]
    coords[:, 1] -= base_point[1]
    coords[:, 2] -= base_point[2]

    print(f"[loader] Loaded unlabeled {filepath}: {len(coords):,} plant points")
    return coords


def load_las(filepath, height_lo_m=0.10, voxel_m=0.005):
    """Load a FieldPheno4D-style LAS scan as a row-scale plant cloud in cm.

    The scan is a multi-plant row strip (~2 x 8 m). Returns the full
    above-ground row in centimetres so callers can split into individual
    plants via ``separate_plants_along_row``. No centring is applied here —
    centring is per-plant after splitting.

    Args:
        filepath: path to .las
        height_lo_m: drop points with DEM-normalised height below this (m)
        voxel_m: voxel-downsample to this resolution (m); set to 0 to skip

    Returns:
        np.array([N, 3]) above-ground points in cm, in the LAS local frame
        with z centred on min-Z.
    """
    import laspy
    las = laspy.read(filepath)
    xyz = np.column_stack([np.asarray(las.x, float),
                           np.asarray(las.y, float),
                           np.asarray(las.z, float)])
    dims = set(las.point_format.dimension_names)
    if 'height' in dims:
        height = np.asarray(las.height, float)
    else:
        height = xyz[:, 2] - np.percentile(xyz[:, 2], 1)

    xyz = xyz[height > height_lo_m]
    if xyz.shape[0] == 0:
        raise ValueError(f"No above-ground points in {filepath} at h>{height_lo_m} m")

    # Recentre x, y on the median of the row (preserves intra-row geometry)
    # and z on min-Z so plant base sits near z=0.
    xyz[:, 0] -= float(np.median(xyz[:, 0]))
    xyz[:, 1] -= float(np.median(xyz[:, 1]))
    xyz[:, 2] -= float(xyz[:, 2].min())

    if voxel_m > 0:
        mn = xyz.min(0)
        vidx = np.floor((xyz - mn) / voxel_m).astype(np.int64)
        _, uniq = np.unique(vidx, axis=0, return_index=True)
        xyz = xyz[uniq]

    xyz_cm = xyz * 100.0
    print(f"[loader] LAS {filepath}: {xyz_cm.shape[0]:,} pts (h>{height_lo_m}m, voxel {voxel_m*1000:.0f}mm), cm-frame")
    return xyz_cm


def separate_plants_along_row(points_cm, smoothing_cm=2.0,
                              min_separation_cm=10.0, min_density=0.2,
                              base_height_cm=10.0):
    """Find plant centres along the long row axis from BASE points only.

    Density peak detection on the full canopy is unreliable for monocot rows:
    splayed leaf tips create false peaks 30-60 cm away from the actual stem
    they belong to. Stems are vertical and well-localised at the base, so we
    restrict the density profile to points within ``base_height_cm`` of the
    ground — that isolates pseudostems / stem columns.

    Returns ``(row_axis, centres_cm)``.
    """
    from scipy.ndimage import gaussian_filter1d
    extents = points_cm.max(0) - points_cm.min(0)
    row_axis = int(np.argmax(extents[:2]))

    base_mask = points_cm[:, 2] <= base_height_cm
    if base_mask.sum() < 20:
        # fall back to full canopy if no base points
        base_mask = np.ones(points_cm.shape[0], bool)
    coord = points_cm[base_mask, row_axis]

    lo, hi = coord.min(), coord.max()
    bin_size_cm = 0.5
    bins = np.arange(lo, hi + bin_size_cm, bin_size_cm)
    cnt, edges = np.histogram(coord, bins)
    smoothed = gaussian_filter1d(cnt.astype(float), smoothing_cm / bin_size_cm)
    centres_cm = 0.5 * (edges[:-1] + edges[1:])

    sep_bins = max(1, int(min_separation_cm / bin_size_cm))
    peaks = []
    thresh = float(smoothed.max()) * min_density
    for i in range(len(smoothed)):
        if smoothed[i] < thresh:
            continue
        lo_i = max(0, i - sep_bins)
        hi_i = min(len(smoothed), i + sep_bins + 1)
        if smoothed[i] == smoothed[lo_i:hi_i].max():
            peaks.append(centres_cm[i])
    centres = np.array(sorted(peaks))
    return row_axis, centres


def crop_plant_window(points_cm, row_axis, centre_cm, window_cm=20.0,
                      cross_row_window_cm=None,
                      cross_row_base_height_cm=5.0):
    """Crop a window around ``centre_cm`` along ``row_axis``, then centre on
    its base (XY at the densest low-Z column, Z at min).

    By default only the row axis is clipped. Pass ``cross_row_window_cm`` to
    also clip the cross-row axis to ``±cross_row_window_cm/2`` around the
    densest low-Z column — without that, ~150 cm of cross-row width (incl.
    adjacent rows / mulch) leaks into the crop and dominates the point count.
    The base column is taken as the median cross-row coordinate of points
    below ``cross_row_base_height_cm`` (rebased to the post-row-crop minimum
    Z, so it matches what survives the height_lo_m filter in load_las).
    """
    coord = points_cm[:, row_axis]
    sel = (coord >= centre_cm - window_cm / 2) & (coord <= centre_cm + window_cm / 2)
    P = points_cm[sel].copy()
    if P.shape[0] == 0:
        return P

    if cross_row_window_cm is not None and cross_row_window_cm > 0:
        cross = 1 - row_axis  # row_axis is 0 or 1
        z_floor = float(P[:, 2].min())
        base_mask = P[:, 2] <= z_floor + cross_row_base_height_cm
        if base_mask.sum() < 10:
            base_mask = P[:, 2] <= np.percentile(P[:, 2], 5)
        base_cross = float(np.median(P[base_mask, cross]))
        keep = np.abs(P[:, cross] - base_cross) <= cross_row_window_cm / 2
        P = P[keep]
        if P.shape[0] == 0:
            return P

    base_idx = int(np.argmin(P[:, 2]))
    P[:, 0] -= P[base_idx, 0]
    P[:, 1] -= P[base_idx, 1]
    P[:, 2] -= P[base_idx, 2]
    return P


def load_pheno4d(filepath, label_method='collar'):
    """Load a Pheno4D .txt file and separate points by organ.

    Args:
        filepath: path to .txt file (5 columns: X Y Z label_collar label_tip, in mm)
        label_method: 'collar' uses column 3, 'tip' uses column 4

    Returns:
        dict of organ_name -> np.array([N, 3]) in cm, centered on stem base
    """
    data = np.loadtxt(filepath)
    coords = data[:, :3]  # mm

    label_col = 3 if label_method == 'collar' else 4
    labels = data[:, label_col].astype(int)

    # Filter out soil (label 0)
    plant_mask = labels != 0
    coords = coords[plant_mask]
    labels = labels[plant_mask]

    # Find stem base: lowest Z point among stem points
    stem_mask = labels == 1
    if stem_mask.any():
        stem_points = coords[stem_mask]
        base_idx = np.argmin(stem_points[:, 2])
        base_point = stem_points[base_idx].copy()
        # Center XY on stem base, Z on ground level
        coords[:, 0] -= base_point[0]
        coords[:, 1] -= base_point[1]
        coords[:, 2] -= base_point[2]

    # Convert mm -> cm
    coords /= 10.0

    # Separate by organ
    organs = {}
    for label_id in np.unique(labels):
        name = LABEL_NAMES.get(label_id, f'label_{label_id}')
        mask = labels == label_id
        organs[name] = coords[mask]

    n_total = sum(len(v) for v in organs.values())
    print(f"[loader] Loaded {filepath}: {n_total:,} plant points")
    for name, pts in sorted(organs.items()):
        print(f"  {name}: {len(pts):,} points")

    return organs
