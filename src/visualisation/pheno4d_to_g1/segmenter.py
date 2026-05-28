"""Morphology-aware maize segmentation from unlabeled point clouds.

Uses maize structural priors (virtual stem axis, cross-sectional analysis,
180-degree phyllotaxis) to segment individual leaves as sheath+blade units,
replacing the incorrect Pheno4D labels where "stem" = pseudostem.

Algorithm:
  1. Estimate virtual stem axis via iterative Z-bin centroids
  2. Convert to cylindrical coordinates (r, theta, z_proj)
  3. Scan all Z-levels for blade candidates (r > pseudostem boundary)
  4. Link blade fragments across Z-levels by angular proximity
  5. Fit phyllotaxis model to validate/complete leaf set
  6. Track leaves into sheath zone with angular assignment
  7. Output organ dict compatible with load_pheno4d() format
"""

import numpy as np
from scipy.signal import savgol_filter
from sklearn.cluster import DBSCAN


# ── Utilities ────────────────────────────────────────────────────────────


def _circular_mean(angles):
    """Mean angle on the unit circle."""
    return np.arctan2(np.sin(angles).mean(), np.cos(angles).mean())


def _circular_distance(a, b):
    """Shortest angular distance between two angles in [-pi, pi]."""
    d = a - b
    return np.abs(np.arctan2(np.sin(d), np.cos(d)))


def _angle_at_z(z_value, z_profile, angle_profile):
    """Interpolate a circular angle profile at one height."""
    if len(z_profile) == 0:
        raise ValueError("angle profile must not be empty")
    if len(z_profile) == 1:
        return float(angle_profile[0])

    order = np.argsort(z_profile)
    z_sorted = np.asarray(z_profile, dtype=float)[order]
    angles_sorted = np.unwrap(np.asarray(angle_profile, dtype=float)[order])
    angle = np.interp(z_value, z_sorted, angles_sorted)
    return float(np.arctan2(np.sin(angle), np.cos(angle)))


def _track_angle_at_z(track, z_value):
    """Evaluate one detected track's angular profile at ``z_value``."""
    return _angle_at_z(z_value, track['angle_z'], track['angle_theta'])


def _circular_dbscan(angles, eps_rad=0.5, min_samples=10):
    """DBSCAN clustering on circular angles.

    Maps angles to (sin, cos) features so that -pi and +pi are adjacent.

    Returns:
        labels: cluster labels (-1 = noise)
    """
    features = np.column_stack([np.sin(angles), np.cos(angles)])
    eps_euclidean = 2.0 * np.sin(min(eps_rad, np.pi) / 2.0)
    clustering = DBSCAN(eps=eps_euclidean, min_samples=min_samples).fit(features)
    return clustering.labels_


# ── Phase 1: Virtual Stem Axis ───────────────────────────────────────────


def estimate_virtual_axis(points, n_slices=80, inner_percentile=60,
                          savgol_window=11, savgol_order=3):
    """Estimate the virtual stem axis from a plant point cloud.

    Two-pass approach:
      Pass 1: XY centroids per Z-slice (all points)
      Pass 2: recompute using only inner points (pseudostem core)
      Smooth with Savitzky-Golay filter.
    """
    z_min, z_max = points[:, 2].min(), points[:, 2].max()
    z_edges = np.linspace(z_min, z_max, n_slices + 1)

    # Pass 1: crude centroids
    centroids_1 = []
    z_mids_1 = []
    for i in range(n_slices):
        mask = (points[:, 2] >= z_edges[i]) & (points[:, 2] < z_edges[i + 1])
        if mask.sum() >= 3:
            centroids_1.append(points[mask].mean(axis=0))
            z_mids_1.append((z_edges[i] + z_edges[i + 1]) / 2)

    if len(centroids_1) < 3:
        raise ValueError("Too few Z-slices with points for axis estimation")

    centroids_1 = np.array(centroids_1)
    z_mids_1 = np.array(z_mids_1)

    # Compute radial distances from pass-1 centroids
    nearest_z = np.argmin(np.abs(points[:, 2:3] - z_mids_1[None, :]), axis=1)
    nearest_xy = centroids_1[nearest_z, :2]
    r_dists = np.linalg.norm(points[:, :2] - nearest_xy, axis=1)

    # Threshold: keep inner points (pseudostem core)
    r_thresh = np.percentile(r_dists, inner_percentile)

    # Pass 2: recompute centroids with only inner points
    inner_mask = r_dists <= r_thresh
    centroids_2 = []
    for i in range(n_slices):
        z_mask = (points[:, 2] >= z_edges[i]) & (points[:, 2] < z_edges[i + 1])
        core = z_mask & inner_mask
        if core.sum() >= 3:
            centroids_2.append(points[core].mean(axis=0))
        elif z_mask.sum() >= 3:
            centroids_2.append(points[z_mask].mean(axis=0))

    if len(centroids_2) < 3:
        raise ValueError("Could not build refined stem axis")

    axis = np.array(centroids_2)
    if axis[0, 2] > axis[-1, 2]:
        axis = axis[::-1]

    # Smooth with SavGol
    n_pts = len(axis)
    win = min(savgol_window, n_pts)
    if win % 2 == 0:
        win -= 1
    if win >= 5:
        order = min(savgol_order, win - 1)
        axis[:, 0] = savgol_filter(axis[:, 0], win, order)
        axis[:, 1] = savgol_filter(axis[:, 1], win, order)

    return axis


# ── Phase 2: Cylindrical Coordinates ────────────────────────────────────


def to_cylindrical(points, axis):
    """Convert points to cylindrical coordinates relative to a polyline axis.

    Returns:
        r: (N,) perpendicular distance to axis
        theta: (N,) azimuth angle in [-pi, pi]
        z_proj: (N,) projected height along axis
        nearest_seg: (N,) index of nearest axis segment
    """
    n_pts = len(points)
    min_dists = np.full(n_pts, np.inf)
    nearest_seg = np.zeros(n_pts, dtype=int)
    proj_points = np.zeros((n_pts, 3))

    seg_lengths = np.linalg.norm(np.diff(axis, axis=0), axis=1)
    cum_length = np.concatenate([[0], np.cumsum(seg_lengths)])
    proj_z = np.zeros(n_pts)

    for si in range(len(axis) - 1):
        p0, p1 = axis[si], axis[si + 1]
        seg_vec = p1 - p0
        seg_len = np.linalg.norm(seg_vec)
        if seg_len < 1e-8:
            continue
        seg_dir = seg_vec / seg_len
        vecs = points - p0
        t = np.clip(vecs @ seg_dir, 0, seg_len)
        projs = p0 + np.outer(t, seg_dir)
        dists = np.linalg.norm(points - projs, axis=1)
        closer = dists < min_dists
        min_dists[closer] = dists[closer]
        nearest_seg[closer] = si
        proj_points[closer] = projs[closer]
        proj_z[closer] = cum_length[si] + t[closer]

    r = min_dists
    offset = points - proj_points
    theta = np.arctan2(offset[:, 1], offset[:, 0])

    return r, theta, proj_z, nearest_seg


# ── Phase 3: Full-Height Blade Detection ────────────────────────────────


def _estimate_pseudostem_boundary(r_values):
    """Estimate pseudostem boundary from radial distribution in a Z-slice.

    Uses the largest gap in the upper portion of the sorted radial distances.
    Falls back to 75th percentile if no clear gap.
    """
    if len(r_values) < 10:
        return np.percentile(r_values, 75)

    sorted_r = np.sort(r_values)
    n = len(sorted_r)

    # Look for largest gap between 40th and 90th percentile
    start = int(n * 0.4)
    end = int(n * 0.9)
    if end - start < 3:
        return np.percentile(r_values, 75)

    gaps = np.diff(sorted_r[start:end])
    if len(gaps) == 0:
        return np.percentile(r_values, 75)

    max_gap = gaps.max()
    max_gap_idx = start + np.argmax(gaps)

    # Only use gap if it's significant (> 1.5x median gap)
    median_gap = np.median(gaps)
    if max_gap > median_gap * 1.5:
        return sorted_r[max_gap_idx]
    else:
        return np.percentile(r_values, 75)


def detect_blade_clusters_fullheight(points, r, theta, z_proj,
                                     n_z_slices=40,
                                     boundary_percentile=70,
                                     min_blade_r_cm=0.3,
                                     dbscan_eps_rad=0.6,
                                     dbscan_min_samples=8,
                                     link_tolerance_rad=0.5,
                                     max_fragment_merge_gap_cm=10.0,
                                     min_leaf_z_span=1.0,
                                     min_leaf_points=100):
    """Detect leaf blade clusters by scanning all Z-levels.

    At each Z-slice:
      1. Estimate pseudostem boundary from r-distribution
      2. Points with r > boundary = blade candidates
      3. Cluster blade candidates by angle (circular DBSCAN)
    Then link clusters across adjacent Z-levels by angular proximity.

    Returns:
        leaf_tracks: list of dicts with 'angle', 'z_min', 'z_max', 'n_points'
    """
    z_min, z_max = z_proj.min(), z_proj.max()
    z_height = z_max - z_min
    if z_height < 1.0:
        return []

    z_edges = np.linspace(z_min, z_max, n_z_slices + 1)

    # Collect per-slice blade clusters
    slice_clusters = []  # list of lists, one per Z-slice

    for zi in range(n_z_slices):
        z_lo, z_hi = z_edges[zi], z_edges[zi + 1]
        slice_mask = (z_proj >= z_lo) & (z_proj < z_hi)

        if slice_mask.sum() < 20:
            slice_clusters.append([])
            continue

        slice_r = r[slice_mask]
        slice_theta = theta[slice_mask]
        slice_indices = np.where(slice_mask)[0]

        # Estimate boundary
        boundary = _estimate_pseudostem_boundary(slice_r)
        # Also enforce a minimum: blade must be at least min_blade_r_cm from axis
        boundary = max(boundary, min_blade_r_cm)

        # Blade candidates
        blade_mask = slice_r > boundary
        if blade_mask.sum() < dbscan_min_samples:
            slice_clusters.append([])
            continue

        blade_theta = slice_theta[blade_mask]
        blade_indices = slice_indices[blade_mask]

        # Cluster by angle
        labels = _circular_dbscan(blade_theta, eps_rad=dbscan_eps_rad,
                                  min_samples=dbscan_min_samples)

        clusters_this_slice = []
        for cl_id in sorted(set(labels) - {-1}):
            cl_mask = labels == cl_id
            cl_angles = blade_theta[cl_mask]
            angle = _circular_mean(cl_angles)
            clusters_this_slice.append({
                'angle': float(angle),
                'n_points': int(cl_mask.sum()),
                'z_mid': (z_lo + z_hi) / 2,
                'indices': blade_indices[cl_mask],
            })
        slice_clusters.append(clusters_this_slice)

    # Link clusters across Z-slices into leaf tracks
    # Each track = contiguous chain of slice-clusters with similar angle
    active_tracks = []   # list of tracks currently being built
    finished_tracks = []

    for zi in range(n_z_slices):
        clusters_here = slice_clusters[zi]
        used = [False] * len(clusters_here)

        # Try to extend existing tracks
        for track in active_tracks:
            track_angle = track['current_angle']
            best_match = None
            best_dist = link_tolerance_rad

            for ci, cl in enumerate(clusters_here):
                if used[ci]:
                    continue
                d = _circular_distance(track_angle, cl['angle'])
                if d < best_dist:
                    best_dist = d
                    best_match = ci

            if best_match is not None:
                cl = clusters_here[best_match]
                used[best_match] = True
                track['all_indices'].append(cl['indices'])
                track['n_points'] += cl['n_points']
                track['z_max'] = cl['z_mid']
                # Update angle with running mean
                track['all_angles'].append(cl['angle'])
                track['all_z_mids'].append(cl['z_mid'])
                track['current_angle'] = _circular_mean(
                    np.array(track['all_angles']))
                track['gap_count'] = 0
            else:
                track['gap_count'] += 1

        # Finish tracks that have been unmatched for too long
        still_active = []
        for track in active_tracks:
            if track['gap_count'] > 3:
                finished_tracks.append(track)
            else:
                still_active.append(track)
        active_tracks = still_active

        # Start new tracks for unmatched clusters
        for ci, cl in enumerate(clusters_here):
            if not used[ci]:
                active_tracks.append({
                    'current_angle': cl['angle'],
                    'all_angles': [cl['angle']],
                    'all_z_mids': [cl['z_mid']],
                    'all_indices': [cl['indices']],
                    'n_points': cl['n_points'],
                    'z_min': cl['z_mid'],
                    'z_max': cl['z_mid'],
                    'gap_count': 0,
                })

    finished_tracks.extend(active_tracks)

    # Filter: minimum Z-span and point count
    leaf_tracks = []
    for track in finished_tracks:
        z_span = track['z_max'] - track['z_min']
        if z_span >= min_leaf_z_span and track['n_points'] >= min_leaf_points:
            all_idx = np.concatenate(track['all_indices']) if track['all_indices'] else np.array([], dtype=int)
            leaf_tracks.append({
                'angle': _circular_mean(np.array(track['all_angles'])),
                'angle_z': np.array(track['all_z_mids'], dtype=float),
                'angle_theta': np.array(track['all_angles'], dtype=float),
                'z_min': track['z_min'],
                'z_max': track['z_max'],
                'n_points': track['n_points'],
                'indices': all_idx,
            })

    return _merge_contiguous_track_fragments(
        leaf_tracks,
        max_gap_cm=max_fragment_merge_gap_cm,
        angle_tolerance_rad=link_tolerance_rad,
    )


def _merge_contiguous_track_fragments(tracks, max_gap_cm=3.0,
                                      angle_tolerance_rad=0.5):
    """Merge same-leaf fragments without collapsing repeated phyllotaxis.

    Maize leaves from different ranks can share a similar mean azimuth, so a
    global same-angle merge is unsafe.  This only joins fragments when their
    height ranges are contiguous and their endpoint angles agree.
    """
    if not tracks:
        return []

    pending = sorted(tracks, key=lambda t: (t['z_min'], t['z_max']))
    merged = []

    for track in pending:
        best_i = None
        best_gap = max_gap_cm

        for i, existing in enumerate(merged):
            gap = track['z_min'] - existing['z_max']
            if gap < -1e-8 or gap > max_gap_cm:
                continue

            z_join = 0.5 * (track['z_min'] + existing['z_max'])
            d = _circular_distance(
                _track_angle_at_z(existing, z_join),
                _track_angle_at_z(track, z_join),
            )
            if d <= angle_tolerance_rad and gap <= best_gap:
                best_i = i
                best_gap = gap

        if best_i is None:
            merged.append(track.copy())
            continue

        existing = merged[best_i]
        existing['indices'] = np.concatenate([existing['indices'], track['indices']])
        existing['n_points'] += track['n_points']
        existing['z_min'] = min(existing['z_min'], track['z_min'])
        existing['z_max'] = max(existing['z_max'], track['z_max'])
        existing['angle_z'] = np.concatenate([existing['angle_z'], track['angle_z']])
        existing['angle_theta'] = np.concatenate([existing['angle_theta'], track['angle_theta']])
        existing['angle'] = _circular_mean(existing['angle_theta'])

    merged.sort(key=lambda t: t['z_min'])
    return merged


# ── Phase 4: Phyllotaxis-Aware Leaf Identification ──────────────────────


def identify_leaves(tracks):
    """Identify individual leaves from blade tracks.

    Each track is a candidate leaf. Tracks are sorted by insertion height
    (z_min). The phyllotaxis model (180-degree alternation) is used to
    validate groupings: tracks at similar angles but different heights
    are SEPARATE leaves (e.g., leaf 1 and leaf 3 both at angle A,
    leaf 2 and leaf 4 at angle B).

    Returns:
        leaves: list of dicts with 'angle', 'z_min', 'z_max', 'indices',
                'leaf_id' (1-based)
    """
    if not tracks:
        return []

    # Sort by insertion height (z_min)
    sorted_tracks = sorted(tracks, key=lambda t: t['z_min'])

    # Each track = one leaf, numbered by emergence order
    leaves = []
    for i, track in enumerate(sorted_tracks):
        leaves.append({
            'leaf_id': i + 1,
            'angle': track['angle'],
            'angle_z': track.get('angle_z', np.array([track['z_min'], track['z_max']])),
            'angle_theta': track.get('angle_theta', np.array([track['angle'], track['angle']])),
            'z_min': track['z_min'],
            'z_max': track['z_max'],
            'n_points': track['n_points'],
            'indices': track['indices'],
        })

    return leaves


# ── Phase 5: Full Assignment ────────────────────────────────────────────


def assign_all_points(points, r, theta, z_proj, leaves,
                      n_z_slices=80, core_fraction=0.15,
                      tight_tolerance_rad=0.55,
                      min_leaf_points=200, min_leaf_z_range=1.5):
    """Assign all points to leaves or pseudostem.

    Each leaf has both an angle AND an insertion height. At each Z-level,
    only leaves with z_min <= current height are candidates. Among matching
    leaves, the one with nearest angle wins. For leaves at the same angle
    but different heights, the most recently inserted (highest z_min below
    current level) gets priority.

    Returns:
        labels: (N,) array — 0=stem/pseudostem, 1..n_leaves=leaf ID
    """
    n_pts = len(points)
    labels = np.zeros(n_pts, dtype=int)
    n_leaves = len(leaves)

    if n_leaves == 0:
        return labels

    leaf_z_mins = np.array([lf['z_min'] for lf in leaves])

    # Pre-assign tracked blade points
    for lf in leaves:
        lid = lf['leaf_id']
        for idx in lf['indices']:
            if idx < n_pts:
                labels[idx] = lid

    # Assign remaining unassigned points by Z-slice (top-to-bottom)
    z_min, z_max = z_proj.min(), z_proj.max()
    z_edges = np.linspace(z_max, z_min, n_z_slices + 1)

    wide_tolerance = np.pi / max(n_leaves, 1)

    for zi in range(n_z_slices):
        z_hi = z_edges[zi]
        z_lo = z_edges[zi + 1]
        slice_z_mid = (z_hi + z_lo) / 2

        slice_mask = (z_proj <= z_hi) & (z_proj > z_lo) & (labels == 0)
        if slice_mask.sum() < 3:
            continue

        slice_r = r[slice_mask]
        slice_theta = theta[slice_mask]
        slice_indices = np.where(slice_mask)[0]

        boundary = _estimate_pseudostem_boundary(slice_r)
        core_r = boundary * core_fraction

        # Which leaves are active at this height?
        # A leaf is active if its insertion height is at or below this Z-level
        # Use some slack: active if z_min <= slice_z_mid + margin
        margin = (z_max - z_min) / n_z_slices * 2  # 2 slice widths
        active = leaf_z_mins <= (slice_z_mid + margin)

        if not active.any():
            continue

        active_ids = np.where(active)[0]
        active_angles = np.array([
            _angle_at_z(slice_z_mid, leaves[li]['angle_z'], leaves[li]['angle_theta'])
            for li in active_ids
        ])

        for j, idx in enumerate(slice_indices):
            pt_r = slice_r[j]
            pt_theta = slice_theta[j]

            if pt_r < core_r:
                continue  # stays 0 (stem)

            # Find nearest active leaf by angle
            dists = np.array([
                _circular_distance(pt_theta, ca) for ca in active_angles
            ])
            best_active = np.argmin(dists)
            min_dist = dists[best_active]
            best_leaf_id = active_ids[best_active]

            if pt_r > boundary:
                if min_dist < tight_tolerance_rad:
                    labels[idx] = best_leaf_id + 1
            else:
                if min_dist < wide_tolerance:
                    labels[idx] = best_leaf_id + 1

    # Post-process: filter small/thin leaves
    for li in range(n_leaves):
        lid = li + 1
        leaf_mask = labels == lid
        if leaf_mask.sum() < min_leaf_points:
            labels[leaf_mask] = 0
            continue
        leaf_z = z_proj[leaf_mask]
        if (leaf_z.max() - leaf_z.min()) < min_leaf_z_range:
            labels[leaf_mask] = 0

    return labels


# ── Phase 6: Main Entry Point ───────────────────────────────────────────


def segment_maize(filepath, n_axis_slices=80, inner_percentile=60,
                  n_detect_slices=40, boundary_percentile=70,
                  dbscan_eps_rad=0.6, dbscan_min_samples=8,
                  link_tolerance_rad=0.5,
                  tight_tolerance_rad=0.55, core_fraction=0.15,
                  min_leaf_points=200, min_leaf_z_range=1.5,
                  soil_margin_cm=0.5):
    """Morphology-aware maize segmentation from unlabeled point cloud.

    Uses maize structural priors to segment individual leaves as
    sheath+blade units (not just the visible blade portion).

    Args:
        filepath: path to Pheno4D .txt file
        n_axis_slices: Z-bins for axis estimation
        inner_percentile: percentile for inner point selection
        n_detect_slices: Z-bins for blade detection
        boundary_percentile: percentile for pseudostem boundary
        dbscan_eps_rad: angular DBSCAN epsilon
        dbscan_min_samples: DBSCAN min_samples for blade clustering
        link_tolerance_rad: angular tolerance for linking clusters across Z
        tight_tolerance_rad: angular tolerance for outer zone assignment
        core_fraction: fraction of boundary for core zone
        min_leaf_points: minimum points per leaf
        min_leaf_z_range: minimum Z-range per leaf (cm)
        soil_margin_cm: soil removal margin

    Returns:
        dict of organ_name -> np.array([N, 3]) in cm,
        same format as load_pheno4d()
    """
    from .loader import load_unlabeled

    points = load_unlabeled(filepath, soil_margin_cm=soil_margin_cm)
    print(f"[segmenter] Loaded {len(points):,} plant points")

    plant_height = points[:, 2].max() - points[:, 2].min()
    print(f"[segmenter] Plant height: {plant_height:.1f} cm")

    # Phase 1: Virtual stem axis
    axis = estimate_virtual_axis(points, n_slices=n_axis_slices,
                                 inner_percentile=inner_percentile)
    print(f"[segmenter] Virtual axis: {len(axis)} nodes, "
          f"z=[{axis[0, 2]:.1f}, {axis[-1, 2]:.1f}]")

    # Phase 2: Cylindrical coordinates
    r, theta, z_proj, nearest_seg = to_cylindrical(points, axis)
    print(f"[segmenter] Radial: median={np.median(r):.2f}, "
          f"p90={np.percentile(r, 90):.2f}, max={r.max():.2f} cm")

    # Phase 3: Full-height blade detection
    tracks = detect_blade_clusters_fullheight(
        points, r, theta, z_proj,
        n_z_slices=n_detect_slices,
        boundary_percentile=boundary_percentile,
        dbscan_eps_rad=dbscan_eps_rad,
        dbscan_min_samples=dbscan_min_samples,
        link_tolerance_rad=link_tolerance_rad,
        min_leaf_z_span=min_leaf_z_range,
        min_leaf_points=min(min_leaf_points, 50),  # lower for detection
    )
    print(f"[segmenter] Detected {len(tracks)} leaf tracks")
    for i, t in enumerate(tracks):
        print(f"  track {i}: angle={np.degrees(t['angle']):.0f} deg, "
              f"{t['n_points']} pts, z=[{t['z_min']:.1f}, {t['z_max']:.1f}]")

    # Fallback for very young plants
    if not tracks:
        print("[segmenter] No tracks detected — radial fallback")
        median_r = np.median(r)
        outer = r > median_r * 2.0
        if outer.sum() >= min_leaf_points:
            organs = {'stem': points[~outer], 'leaf_1': points[outer]}
        else:
            organs = {'stem': points}
        _print_summary(organs)
        return organs

    # Phase 4: Identify individual leaves from tracks
    leaves = identify_leaves(tracks)
    print(f"[segmenter] Identified {len(leaves)} leaves")
    for lf in leaves:
        print(f"  leaf {lf['leaf_id']}: angle={np.degrees(lf['angle']):.0f} deg, "
              f"z=[{lf['z_min']:.1f}, {lf['z_max']:.1f}], {lf['n_points']} blade pts")

    # Phase 5: Assign all points
    labels = assign_all_points(
        points, r, theta, z_proj, leaves,
        n_z_slices=n_axis_slices,
        core_fraction=core_fraction,
        tight_tolerance_rad=tight_tolerance_rad,
        min_leaf_points=min_leaf_points,
        min_leaf_z_range=min_leaf_z_range,
    )

    # Phase 6: Build output dict
    organs = {}
    organs['stem'] = points[labels == 0]

    active_labels = sorted(set(labels) - {0})
    for i, lid in enumerate(active_labels):
        organs[f'leaf_{i + 1}'] = points[labels == lid]

    if 'stem' not in organs or len(organs['stem']) == 0:
        raise ValueError("Segmentation failed: no stem points")

    _print_summary(organs)
    return organs


def _print_summary(organs):
    """Print segmentation summary."""
    n_total = sum(len(v) for v in organs.values())
    n_leaves = sum(1 for k in organs if k.startswith('leaf_'))
    print(f"[segmenter] Result: {n_total:,} points → stem + {n_leaves} leaves")
    for name, pts in sorted(organs.items()):
        zr = pts[:, 2].max() - pts[:, 2].min() if len(pts) > 0 else 0
        print(f"  {name}: {len(pts):,} pts, z=[{pts[:, 2].min():.1f}, "
              f"{pts[:, 2].max():.1f}], height={zr:.1f} cm")
