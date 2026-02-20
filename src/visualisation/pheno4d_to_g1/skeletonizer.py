"""Hybrid skeleton extraction: PCA-axis binning (stem) + slice-and-walk (leaves)."""

import numpy as np
from collections import deque
from scipy.spatial import KDTree
from scipy.spatial.distance import pdist, squareform
from scipy.sparse import csr_matrix, diags, eye
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse.linalg import spsolve
from scipy.signal import savgol_filter


def refine_tip_nodes(skeleton, points, direction_window=3, max_lateral_dist=None):
    """Project terminal skeleton nodes to the point cloud boundary.

    Laplacian contraction and slice-and-walk systematically leave terminal
    nodes inside the cloud, underestimating organ length.  This projects
    the first and last skeleton nodes outward to the nearest cloud boundary
    along the local growth direction, constrained to stay close to the
    skeleton axis to avoid hooking onto outliers or adjacent organs.

    Args:
        skeleton: np.array([M, 3]) ordered skeleton points
        points: np.array([N, 3]) original point cloud
        direction_window: number of skeleton points used to estimate
            local growth direction at each end
        max_lateral_dist: maximum perpendicular distance from the skeleton
            axis for candidate points.  Default: median segment length * 2.

    Returns:
        np.array([M, 3]) skeleton with refined terminal nodes
    """
    if len(skeleton) < 2 or len(points) < 3:
        return skeleton

    skeleton = skeleton.copy()
    w = min(direction_window, len(skeleton))

    # Default lateral constraint from skeleton spacing
    if max_lateral_dist is None:
        seg_lengths = np.linalg.norm(np.diff(skeleton, axis=0), axis=1)
        max_lateral_dist = np.median(seg_lengths) * 2.0

    def _refine_end(skel_end, skel_ref, pts):
        """Refine one endpoint: find farthest point along growth direction
        that is within max_lateral_dist of the skeleton axis."""
        direction = skel_end - skel_ref
        dir_len = np.linalg.norm(direction)
        if dir_len < 1e-8:
            return skel_end
        direction /= dir_len

        vecs = pts - skel_end
        along = vecs @ direction
        # Only points ahead of the current endpoint
        forward_mask = along > 0
        if not forward_mask.any():
            return skel_end

        # Perpendicular distance from the axis
        proj_along = np.outer(along, direction)
        perp = vecs - proj_along
        perp_dist = np.linalg.norm(perp, axis=1)

        # Constrain: must be within lateral distance
        valid = forward_mask & (perp_dist < max_lateral_dist)
        if not valid.any():
            return skel_end

        # Among valid points, pick the farthest along the axis
        along_valid = along.copy()
        along_valid[~valid] = -np.inf
        best = np.argmax(along_valid)
        return pts[best]

    # --- refine tip (last node) ---
    skeleton[-1] = _refine_end(skeleton[-1], skeleton[-max(1, w)], points)

    # --- refine base (first node) ---
    skeleton[0] = _refine_end(skeleton[0], skeleton[min(w, len(skeleton) - 1)], points)

    return skeleton


def laplacian_contraction_skeleton(points, k_neighbors=10, iterations=5,
                                   contraction_weight=2.0, merge_threshold=None):
    """Extract a 1D skeleton from a point cloud via iterative Laplacian contraction.

    Builds a k-NN graph, constructs the uniform umbrella Laplacian, and
    iteratively contracts the point cloud toward its medial axis.  Contracted
    points are then merged into skeleton nodes and ordered along a path.

    Args:
        points: np.array([N, 3]) point cloud
        k_neighbors: number of nearest neighbors for graph construction
        iterations: number of contraction iterations
        contraction_weight: weight pulling contracted points toward their
            original positions (higher = less contraction per step)
        merge_threshold: distance below which contracted points are merged
            into a single skeleton node.  Default: median NN distance * 1.5

    Returns:
        np.array([M, 3]) ordered skeleton points from one end to the other
    """
    n = len(points)
    if n < 4:
        return points.copy()

    # Subsample large clouds for tractable sparse solve
    max_points = 1500
    if n > max_points:
        idx = np.random.choice(n, max_points, replace=False)
        points = points[idx]
        n = max_points

    k = min(k_neighbors, n - 1)

    # Build k-NN adjacency (symmetric)
    tree = KDTree(points)
    dists, indices = tree.query(points, k=k + 1)  # +1 because first neighbor is self

    if merge_threshold is None:
        # Median distance to nearest non-self neighbor
        merge_threshold = np.median(dists[:, 1]) * 1.5

    # Build sparse symmetric adjacency with uniform weights
    rows, cols = [], []
    for i in range(n):
        for j_idx in range(1, k + 1):  # skip self (index 0)
            j = indices[i, j_idx]
            rows.append(i)
            cols.append(j)
    rows = np.array(rows)
    cols = np.array(cols)

    # Symmetrize
    all_rows = np.concatenate([rows, cols])
    all_cols = np.concatenate([cols, rows])
    data = np.ones(len(all_rows))
    adj = csr_matrix((data, (all_rows, all_cols)), shape=(n, n))
    adj.data[:] = 1.0  # uniform weights after dedup

    # Umbrella Laplacian: L = D - A where D is the degree diagonal
    degree = np.array(adj.sum(axis=1)).flatten()
    degree[degree == 0] = 1  # avoid division by zero
    D = diags(degree)
    L = D - adj

    # Iterative contraction (Au et al. 2008):
    #   solve (sl * L^T L + wh * I) x = wh * p_original
    # sl (Laplacian weight) increases each iteration → stronger contraction
    # wh (position weight) stays fixed → anchors to original positions
    contracted = points.copy().astype(float)
    original = points.copy().astype(float)
    I_n = eye(n, format='csc')
    sl = 1.0  # Laplacian weight (increases)
    wh = contraction_weight  # position anchor weight (fixed)

    LtL = (L.T @ L).tocsc()

    for it in range(iterations):
        A_sys = (sl * LtL + wh * I_n).tocsc()
        result = np.empty_like(contracted)
        for dim in range(3):
            rhs = wh * original[:, dim]
            result[:, dim] = spsolve(A_sys, rhs)
        contracted = result
        # Increase Laplacian weight → more contraction each step
        sl *= 2.0

    # Merge contracted points that are very close together
    merge_tree = KDTree(contracted)
    visited = np.zeros(n, dtype=bool)
    skeleton_nodes = []

    for i in range(n):
        if visited[i]:
            continue
        cluster = merge_tree.query_ball_point(contracted[i], merge_threshold)
        visited[cluster] = True
        # Skeleton node = mean of original positions in cluster
        skeleton_nodes.append(original[cluster].mean(axis=0))

    skeleton_nodes = np.array(skeleton_nodes)

    if len(skeleton_nodes) < 3:
        return skeleton_nodes

    # Order nodes along a path using MST double-BFS
    return _order_points_mst(skeleton_nodes)


def _skeletonize_stem_pca_binning(points, n_bins=80):
    """PCA-axis binning stem skeletonization.

    Finds the principal axis of the stem point cloud via PCA, bins along
    that axis (handles tilted stems), and takes the centroid of each bin.
    Much smoother than Laplacian contraction for thick cylindrical organs.

    Args:
        points: np.array([N, 3]) stem points in cm
        n_bins: number of bins along the principal axis

    Returns:
        np.array([M, 3]) ordered skeleton points
    """
    if len(points) < 10:
        return points[:2].copy() if len(points) >= 2 else points.copy()

    # PCA to find stem principal axis
    centroid = points.mean(axis=0)
    centered = points - centroid
    cov = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    principal = eigenvectors[:, -1]  # longest axis

    # Ensure principal axis points upward (positive Z component)
    if principal[2] < 0:
        principal = -principal

    # Project all points onto principal axis
    proj = centered @ principal
    p_min, p_max = proj.min(), proj.max()
    if p_max - p_min < 0.1:
        return points[:2].copy()

    bin_edges = np.linspace(p_min, p_max, n_bins + 1)
    skeleton = []
    for i in range(n_bins):
        mask = (proj >= bin_edges[i]) & (proj < bin_edges[i + 1])
        if i == n_bins - 1:
            mask |= (proj == bin_edges[i + 1])
        if mask.sum() > 0:
            skeleton.append(points[mask].mean(axis=0))

    if not skeleton:
        return points[:2].copy()
    return np.array(skeleton)


def _savgol_smooth_skeleton(skeleton, window=None, polyorder=3):
    """Smooth a 3D skeleton using Savitzky-Golay filter.

    Preserves endpoints and macro curvature while removing high-frequency
    noise.  Window size is automatically chosen based on skeleton length
    if not provided.

    Args:
        skeleton: np.array([M, 3]) ordered skeleton points
        window: SavGol window size (odd integer). Default: auto from length.
        polyorder: polynomial order for local fit (default 3).

    Returns:
        np.array([M, 3]) smoothed skeleton with original endpoints.
    """
    n = len(skeleton)
    if n < 5:
        return skeleton.copy()

    if window is None:
        # Auto: ~15% of skeleton length, minimum 7, maximum 31
        window = max(7, min(31, int(n * 0.15) | 1))
    # Ensure window is odd and <= n
    if window % 2 == 0:
        window += 1
    window = min(window, n if n % 2 == 1 else n - 1)
    if window < polyorder + 2:
        return skeleton.copy()

    result = skeleton.copy()
    for dim in range(3):
        result[:, dim] = savgol_filter(skeleton[:, dim], window, polyorder)

    # Restore original endpoints (critical for injection pipeline)
    result[0] = skeleton[0]
    result[-1] = skeleton[-1]
    return result


def skeletonize_stem(points, n_bins=80, method='pca_binning'):
    """Extract stem skeleton from point cloud.

    Args:
        points: np.array([N, 3]) stem points in cm
        n_bins: number of bins along axis
        method: 'pca_binning' for PCA-axis binning + SavGol (default),
                'laplacian' for Laplacian contraction,
                'binning' for Z-axis binning (legacy)

    Returns:
        np.array([M, 3]) ordered skeleton points from base to tip
    """
    if method == 'pca_binning':
        skeleton = _skeletonize_stem_pca_binning(points, n_bins)
    elif method == 'laplacian':
        skeleton = laplacian_contraction_skeleton(points)
        if len(skeleton) < 3:
            skeleton = _skeletonize_stem_pca_binning(points, n_bins)
    else:
        skeleton = _skeletonize_stem_binning(points, n_bins)

    # Orient base-to-tip: lowest Z first
    if len(skeleton) >= 2 and skeleton[0, 2] > skeleton[-1, 2]:
        skeleton = skeleton[::-1]

    skeleton = refine_tip_nodes(skeleton, points)

    # SavGol post-smoothing to remove residual noise from bin boundaries
    skeleton = _savgol_smooth_skeleton(skeleton)

    return skeleton


def _skeletonize_stem_binning(points, n_bins=50):
    """Legacy Z-axis binning stem skeletonization.

    Args:
        points: np.array([N, 3]) stem points in cm
        n_bins: number of Z-axis bins

    Returns:
        np.array([M, 3]) ordered skeleton points
    """
    z_min, z_max = points[:, 2].min(), points[:, 2].max()
    bin_edges = np.linspace(z_min, z_max, n_bins + 1)

    skeleton = []
    for i in range(n_bins):
        mask = (points[:, 2] >= bin_edges[i]) & (points[:, 2] < bin_edges[i + 1])
        if i == n_bins - 1:  # include top edge in last bin
            mask |= (points[:, 2] == bin_edges[i + 1])
        if mask.sum() > 0:
            skeleton.append(points[mask].mean(axis=0))

    return np.array(skeleton) if skeleton else points[:1].copy()


def _filter_slab_bimodal(slab_pts, slab_idx, next_pos, direction, min_pts=10,
                          gap_ratio=5.0, min_gap_cm=0.2):
    """Filter bimodal slab distributions from overlapping leaf surfaces.

    When two leaves physically overlap, a slab may contain points from both
    surfaces. This detects bimodality in the cross-section thickness direction
    and keeps only the cluster consistent with the walk trajectory.
    """
    if len(slab_pts) < min_pts:
        return slab_pts, slab_idx

    # Project to cross-section plane (perpendicular to walk direction)
    vecs = slab_pts - next_pos
    along = np.outer(vecs @ direction, direction)
    cross_section = vecs - along

    # PCA of cross-section
    cs_centered = cross_section - cross_section.mean(axis=0)
    try:
        _, s_vals, vh = np.linalg.svd(cs_centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return slab_pts, slab_idx

    # PC2 = thickness direction. If variance ratio is small, leaf is thin -> no overlap
    if s_vals[0] < 1e-8 or s_vals[1] / s_vals[0] < 0.3:
        return slab_pts, slab_idx

    # Project onto PC2 (thickness axis)
    pc2 = vh[1]
    proj = cs_centered @ pc2

    # Gap detection: sort projections, find largest gap
    sorted_proj = np.sort(proj)
    gaps = np.diff(sorted_proj)
    if len(gaps) == 0:
        return slab_pts, slab_idx

    median_gap = np.median(gaps)
    if median_gap < 1e-8:
        return slab_pts, slab_idx

    max_gap_idx = np.argmax(gaps)
    max_gap = gaps[max_gap_idx]

    if max_gap < gap_ratio * median_gap or max_gap < min_gap_cm:
        return slab_pts, slab_idx

    # Bimodal: split at the gap
    threshold = (sorted_proj[max_gap_idx] + sorted_proj[max_gap_idx + 1]) / 2
    group_lo = proj <= threshold
    group_hi = proj > threshold

    # Pick group whose median is closer to next_pos
    if group_lo.sum() >= 3 and group_hi.sum() >= 3:
        center_lo = np.median(slab_pts[group_lo], axis=0)
        center_hi = np.median(slab_pts[group_hi], axis=0)
        dist_lo = np.linalg.norm(center_lo - next_pos)
        dist_hi = np.linalg.norm(center_hi - next_pos)
        keep = group_lo if dist_lo <= dist_hi else group_hi
        return slab_pts[keep], slab_idx[keep]

    return slab_pts, slab_idx


def skeletonize_leaf(points, step_size=0.3, slab_thickness=None):
    """Skeleton extraction for leaf point clouds via PCA-guided slice-and-walk.

    Uses PCA to find the leaf's principal axis, identifies the two
    endpoints (tips of the elongated shape), then walks from one end
    to the other taking perpendicular slabs and computing medians.

    Args:
        points: np.array([N, 3]) leaf points in cm
        step_size: advance distance per skeleton point (cm)
        slab_thickness: thickness of perpendicular slab (default: step_size * 1.5)

    Returns:
        np.array([M, 3]) ordered skeleton points from base to tip
    """
    if len(points) < 6:
        return points.copy()

    if slab_thickness is None:
        slab_thickness = step_size * 1.5

    centroid = points.mean(axis=0)

    # PCA to find the principal elongation axis
    centered = points - centroid
    cov = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    # Principal axis = eigenvector with largest eigenvalue (last column)
    principal_axis = eigenvectors[:, -1]

    # Project all points onto principal axis to find the two extreme ends
    projections = centered @ principal_axis
    end_a_idx = np.argmin(projections)
    end_b_idx = np.argmax(projections)

    # Start from the end that is lower (more likely to be the base for maize)
    if points[end_a_idx, 2] <= points[end_b_idx, 2]:
        start_pt = points[end_a_idx].copy()
        initial_dir = principal_axis.copy()
    else:
        start_pt = points[end_b_idx].copy()
        initial_dir = -principal_axis.copy()

    # Ensure initial direction points away from start toward the cloud
    if np.dot(initial_dir, centroid - start_pt) < 0:
        initial_dir = -initial_dir

    tree = KDTree(points)
    used = np.zeros(len(points), dtype=bool)
    skeleton = [start_pt.copy()]

    # Mark points near start as used
    near_start = tree.query_ball_point(start_pt, step_size)
    used[near_start] = True

    current = start_pt.copy()
    direction = initial_dir.copy()
    total_extent = projections.max() - projections.min()
    max_steps = int(np.ceil(total_extent * 3 / step_size)) + 10

    consecutive_small_advance = 0

    for _ in range(max_steps):
        next_pos = current + direction * step_size

        # Find points near next position (used AND unused — for slab centroid)
        search_radius = max(step_size * 3.0, slab_thickness * 2.5)
        candidate_idx = tree.query_ball_point(next_pos, search_radius)
        if not candidate_idx:
            break

        candidates = np.array(candidate_idx)
        candidate_pts = points[candidates]

        # Perpendicular slab: only points within slab_thickness of the plane
        vecs = candidate_pts - next_pos
        along = vecs @ direction
        slab_mask = np.abs(along) < slab_thickness

        # Among slab points, check if enough are unused (to know we're advancing)
        unused_in_slab = slab_mask & ~used[candidates]
        if unused_in_slab.sum() < 2:
            consecutive_small_advance += 1
            if consecutive_small_advance >= 3:
                break
            # Try advancing further before giving up
            continue
        consecutive_small_advance = 0

        if slab_mask.sum() < 3:
            break

        slab_pts = candidate_pts[slab_mask]
        slab_idx = candidates[slab_mask]

        # Filter bimodal slabs (overlapping leaf surfaces)
        slab_pts, slab_idx = _filter_slab_bimodal(
            slab_pts, slab_idx, next_pos, direction)
        if len(slab_pts) < 3:
            break

        # Use median instead of mean — robust to asymmetric density / outliers
        new_center = np.median(slab_pts, axis=0)

        # Check per-step turn angle: reject >80° single-step turns
        new_dir = new_center - current
        new_dir_len = np.linalg.norm(new_dir)
        if new_dir_len < 1e-8:
            break
        new_dir_unit = new_dir / new_dir_len
        cos_angle = np.dot(new_dir_unit, direction)
        if cos_angle < 0.17:  # >80° single-step turn — noise or cloud edge
            break

        skeleton.append(new_center)

        # Loop detection: if the walk re-enters a region it already passed
        # through, it has followed an emerging leaf back.  Truncate at the
        # farthest point from start (the true tip).
        if len(skeleton) > 15:
            skel_arr = np.array(skeleton)
            # Distance from new point to all earlier points (skip last 10)
            early = skel_arr[:-10]
            dists_to_early = np.linalg.norm(early - new_center, axis=1)
            if dists_to_early.min() < step_size * 3:
                # Walk looped — find the farthest point from start and truncate
                dists_from_start = np.linalg.norm(skel_arr - skel_arr[0], axis=1)
                cut = int(np.argmax(dists_from_start)) + 1
                skeleton = list(skel_arr[:cut])
                break

        # Mark unused slab points as used
        used[slab_idx[~used[slab_idx]]] = True

        # Update direction: smoothed to follow curvature
        direction = 0.7 * new_dir_unit + 0.3 * direction
        direction /= np.linalg.norm(direction)

        current = new_center

    if len(skeleton) < 2:
        return points[:2].copy()

    skeleton = np.array(skeleton)

    # Orient: lowest Z first (base for maize)
    if skeleton[0, 2] > skeleton[-1, 2]:
        skeleton = skeleton[::-1]

    return refine_tip_nodes(skeleton, points)


def _order_points_nn(points):
    """Order unordered points into a path via nearest-neighbor chain.

    Starts from the point with the most extreme coordinate value
    (likely a leaf tip or base), then greedily visits nearest unvisited.
    """
    n = len(points)
    if n <= 2:
        return points.copy()

    # Start from the point farthest from the centroid
    centroid = points.mean(axis=0)
    dists_to_center = np.linalg.norm(points - centroid, axis=1)
    start = np.argmax(dists_to_center)

    visited = np.zeros(n, dtype=bool)
    order = [start]
    visited[start] = True

    for _ in range(n - 1):
        current = order[-1]
        dists = np.linalg.norm(points - points[current], axis=1)
        dists[visited] = np.inf
        nearest = np.argmin(dists)
        order.append(nearest)
        visited[nearest] = True

    return points[order]


def _order_points_mst(points):
    """Order unordered points into a path via MST longest-path traversal.

    Builds a Minimum Spanning Tree of the point cloud, then finds the
    diameter (longest shortest-path) using double-BFS. This correctly
    follows curve topology even for highly curved leaves, unlike greedy
    nearest-neighbor which can create spirals/shortcuts.

    Falls back to _order_points_nn() if the MST is disconnected.
    """
    n = len(points)
    if n <= 2:
        return points.copy()

    # Build full pairwise distance matrix and MST
    dist_mat = squareform(pdist(points))
    mst = minimum_spanning_tree(csr_matrix(dist_mat))

    # Make symmetric for traversal
    mst_sym = mst + mst.T

    # Check connectivity: BFS from node 0
    def bfs_farthest(start):
        """BFS returning (farthest_node, parent_map)."""
        visited = np.full(n, False)
        dist = np.full(n, -1.0)
        parent = np.full(n, -1, dtype=int)
        queue = deque([start])
        visited[start] = True
        dist[start] = 0.0
        farthest = start
        max_dist = 0.0

        while queue:
            u = queue.popleft()
            row = mst_sym.getrow(u)
            for v, w in zip(row.indices, row.data):
                if not visited[v]:
                    visited[v] = True
                    dist[v] = dist[u] + w
                    parent[v] = u
                    queue.append(v)
                    if dist[v] > max_dist:
                        max_dist = dist[v]
                        farthest = v

        return farthest, parent, visited

    # Double-BFS to find diameter endpoints
    end_a, _, visited_a = bfs_farthest(0)
    if not visited_a.all():
        # MST is disconnected — fall back to NN
        return _order_points_nn(points)

    end_b, parent_b, _ = bfs_farthest(end_a)

    # Extract path from end_a to end_b via parent pointers
    path = []
    node = end_b
    while node != -1:
        path.append(node)
        node = parent_b[node]
    path.reverse()

    ordered = points[path]

    # Orient: start from the endpoint closest to centroid (likely base)
    centroid = points.mean(axis=0)
    d_start = np.linalg.norm(ordered[0] - centroid)
    d_end = np.linalg.norm(ordered[-1] - centroid)
    if d_start > d_end:
        ordered = ordered[::-1]

    return ordered


def resample_path(skeleton_points, dx=0.5):
    """Resample a skeleton path at uniform arc-length intervals.

    Args:
        skeleton_points: np.array([M, 3]) ordered path
        dx: target spacing in cm

    Returns:
        np.array([K, 3]) resampled path
    """
    if len(skeleton_points) < 2:
        return skeleton_points.copy()

    # Compute cumulative arc length
    diffs = np.diff(skeleton_points, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)
    cum_length = np.concatenate([[0], np.cumsum(seg_lengths)])
    total_length = cum_length[-1]

    if total_length < dx:
        return skeleton_points.copy()

    # Resample at uniform intervals
    n_samples = max(2, int(np.round(total_length / dx)) + 1)
    target_lengths = np.linspace(0, total_length, n_samples)

    resampled = np.empty((n_samples, 3))
    for dim in range(3):
        resampled[:, dim] = np.interp(target_lengths, cum_length,
                                       skeleton_points[:, dim])

    return resampled


def extract_stem_radius(points, skeleton, segment_idx):
    """Estimate stem radius at a segment from cross-section perpendicular distance.

    Args:
        points: np.array stem points
        skeleton: np.array skeleton points
        segment_idx: which segment (0 to len(skeleton)-2)

    Returns:
        radius in cm (median perpendicular distance)
    """
    p0 = skeleton[segment_idx]
    p1 = skeleton[segment_idx + 1]
    direction = p1 - p0
    seg_len = np.linalg.norm(direction)
    if seg_len < 1e-8:
        return 0.1  # fallback

    direction /= seg_len

    # Find points near this segment
    mid = (p0 + p1) / 2
    dists_to_mid = np.linalg.norm(points - mid, axis=1)
    nearby_mask = dists_to_mid < seg_len * 1.5

    if nearby_mask.sum() < 5:
        return 0.1

    nearby = points[nearby_mask]

    # Project onto segment axis, compute perpendicular distance
    vecs = nearby - p0
    along = vecs @ direction
    # Only points within the segment length
    valid = (along >= -seg_len * 0.2) & (along <= seg_len * 1.2)
    if valid.sum() < 3:
        return 0.1

    vecs_valid = vecs[valid]
    along_valid = along[valid]
    projections = np.outer(along_valid, direction)
    perp = vecs_valid - projections
    perp_dists = np.linalg.norm(perp, axis=1)

    return np.median(perp_dists)


def extract_leaf_width(points, skeleton, segment_idx):
    """Estimate leaf half-width at a segment (max perpendicular distance from midline).

    Args:
        points: np.array leaf points
        skeleton: np.array skeleton points
        segment_idx: which segment

    Returns:
        half-width in cm
    """
    p0 = skeleton[segment_idx]
    p1 = skeleton[segment_idx + 1]
    direction = p1 - p0
    seg_len = np.linalg.norm(direction)
    if seg_len < 1e-8:
        return 0.1

    direction /= seg_len
    mid = (p0 + p1) / 2

    # Points near this segment
    dists_to_mid = np.linalg.norm(points - mid, axis=1)
    nearby_mask = dists_to_mid < seg_len * 2.0
    if nearby_mask.sum() < 3:
        return 0.1

    nearby = points[nearby_mask]
    vecs = nearby - p0
    along = vecs @ direction
    valid = (along >= -seg_len * 0.3) & (along <= seg_len * 1.3)
    if valid.sum() < 3:
        return 0.1

    vecs_valid = vecs[valid]
    along_valid = along[valid]
    projections = np.outer(along_valid, direction)
    perp = vecs_valid - projections
    perp_dists = np.linalg.norm(perp, axis=1)

    # Use 90th percentile to be robust against outliers
    return np.percentile(perp_dists, 90)
