"""Geodesic nearest-instance point assignment (MonGraphSeg Section 4 spirit).

The original MonGraphSeg (Tobies et al., 2025) assigns every point to the
nearest *panoptic instance* — i.e. the nearest stem/leaf skeleton curve —
after the graph-refinement phases produce a parametric model.  The earlier
Python port replaced that step with a slice-wise cylindrical-angle heuristic
(:func:`segmenter.assign_all_points`).  That heuristic marks a leaf "active"
at *every* height above its insertion and then assigns by nearest azimuth, so
the lowest leaf at a given azimuth absorbs the whole vertical column at that
azimuth — turning each maize blade into a full-height vertical wedge.

This module restores the MonGraphSeg behaviour without needing the Bézier
panoptic model and without any labelled training data: it grows labels
*geodesically* over a kNN graph of the point cloud, seeded from the reliable
blade tracks (and a pseudostem core).  Geodesic distance respects blade
connectivity — a point at the top of the plant is far (along the cloud) from a
low leaf's seed, so it cannot be grabbed across empty space.

Public API:
    assign_points_geodesic(points, tracks, r, ...) -> labels  (0 = stem)
    leaf_quality_gate(points_of_leaf, plant_height_cm, ...)   -> (keep, reason, metrics)
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra
from sklearn.neighbors import NearestNeighbors


# ── kNN graph ────────────────────────────────────────────────────────────


def build_knn_graph(points, k=10, max_edge_cm=3.0):
    """Symmetric kNN graph with Euclidean edge weights, long edges dropped.

    Dropping edges longer than ``max_edge_cm`` keeps geodesics from jumping
    across the gaps between separate blades while still bridging the small
    holes left by voxel downsampling / occlusion.

    Returns:
        scipy.sparse.csr_matrix (N, N) — symmetric distance graph.
    """
    points = np.asarray(points, float)
    n = len(points)
    k_eff = min(k + 1, n)  # +1 because the first neighbour is the point itself
    nn = NearestNeighbors(n_neighbors=k_eff).fit(points)
    dist, idx = nn.kneighbors(points)

    rows = np.repeat(np.arange(n), k_eff)
    cols = idx.ravel()
    w = dist.ravel()

    keep = (rows != cols) & (w <= max_edge_cm) & (w > 0)
    g = sp.csr_matrix((w[keep], (rows[keep], cols[keep])), shape=(n, n))
    # Symmetrise (kNN is not symmetric); keep the shorter of the two directions.
    g = g.maximum(g.T)
    return g


# ── pseudostem core seeding ──────────────────────────────────────────────


def estimate_stem_core_mask(r, z_proj, n_z_slices=40, core_percentile=35,
                            min_core_cm=0.0):
    """Mark axis-near (pseudostem core) points per height slice as stem seeds.

    A per-slice radial percentile is more robust than a single global radius
    because the pseudostem tapers and the canopy radius grows with height.
    """
    r = np.asarray(r, float)
    z_proj = np.asarray(z_proj, float)
    core = np.zeros(len(r), bool)
    z_lo, z_hi = z_proj.min(), z_proj.max()
    if z_hi - z_lo < 1e-6:
        return r <= np.percentile(r, core_percentile)
    edges = np.linspace(z_lo, z_hi, n_z_slices + 1)
    for i in range(n_z_slices):
        m = (z_proj >= edges[i]) & (z_proj < edges[i + 1])
        if m.sum() < 5:
            continue
        thr = max(np.percentile(r[m], core_percentile), min_core_cm)
        core[m] = r[m] <= thr
    return core


# ── geodesic multi-source assignment ─────────────────────────────────────


def assign_points_geodesic(points, tracks, r, z_proj,
                           k=10, max_edge_cm=3.0,
                           core_percentile=35,
                           core_n_z_slices=40,
                           tip_reach_cm=None,
                           return_debug=False):
    """Assign every point to the nearest seed instance by geodesic distance.

    Seeds:
      * label 0 = stem  -> pseudostem-core points (axis-near, per-slice).
      * label i = leaf  -> the i-th track's detected blade points
        (tracks are sorted by ``z_min`` so labels run bottom-to-top).

    Args:
        points: (N, 3) cloud in cm.
        tracks: list of track dicts from
            :func:`segmenter.detect_blade_clusters_fullheight` — each must
            carry ``indices`` (point indices of the detected blade).
        r, z_proj: cylindrical radius / projected height from
            :func:`segmenter.to_cylindrical`.
        k: neighbours per node in the kNN graph.
        max_edge_cm: drop graph edges longer than this (gap guard).
        core_percentile: per-slice radial percentile defining the stem core.
        tip_reach_cm: if set, a leaf may only claim points within
            ``[track.z_min - tip_reach, track.z_max + tip_reach]`` (a cheap
            extra guard on top of the geodesic metric). ``None`` disables it.

    Returns:
        labels: (N,) int — 0 = stem, 1..K = leaf id (track order).
        If ``return_debug``: also a dict with the kNN graph and seed masks.
    """
    points = np.asarray(points, float)
    n = len(points)
    labels = np.zeros(n, dtype=int)
    if n == 0:
        return (labels, {}) if return_debug else labels

    tracks_sorted = sorted(tracks, key=lambda t: t['z_min'])

    g = build_knn_graph(points, k=k, max_edge_cm=max_edge_cm)

    # Seed masks: label 0 stem core, labels 1..K leaf tracks.
    stem_core = estimate_stem_core_mask(
        r, z_proj, n_z_slices=core_n_z_slices, core_percentile=core_percentile)
    seed_lists = [np.where(stem_core)[0]]
    for t in tracks_sorted:
        seed_lists.append(np.asarray(t['indices'], dtype=int))

    n_lab = len(seed_lists)  # = 1 + n_leaves

    # Build an augmented graph with one zero-cost super-source per label.
    # Super-source node for label L lives at index n + L.
    aug = g.tolil()
    aug.resize((n + n_lab, n + n_lab))
    for lab, seeds in enumerate(seed_lists):
        src = n + lab
        for s in seeds:
            if 0 <= s < n:
                aug[src, s] = 1e-9  # ~0 cost link source -> its seed points
    aug = aug.tocsr()
    aug = aug.maximum(aug.T)  # undirected

    src_nodes = [n + lab for lab in range(n_lab)]
    dmat = dijkstra(aug, directed=False, indices=src_nodes)  # (n_lab, n+n_lab)
    dmat = dmat[:, :n]  # keep real points only

    reachable = np.isfinite(dmat).any(axis=0)
    best = np.full(n, -1, dtype=int)
    best[reachable] = np.argmin(dmat[:, reachable], axis=0)

    # Optional height-band guard on top of the geodesic metric.
    if tip_reach_cm is not None:
        # leaf_bands[lab-1] = (lo, hi) for leaf label lab in 1..K
        leaf_bands = [(t['z_min'] - tip_reach_cm, t['z_max'] + tip_reach_cm)
                      for t in tracks_sorted]

        def in_band(lab, zv):
            lo, hi = leaf_bands[lab - 1]
            return lo <= zv <= hi

        for i in np.where(best >= 1)[0]:
            if in_band(best[i], z_proj[i]):
                continue
            # fall back to nearest *allowed* label (stem always allowed)
            best[i] = 0
            for lab in np.argsort(dmat[:, i]):
                if lab == 0 or np.isinf(dmat[lab, i]):
                    continue
                if in_band(lab, z_proj[i]):
                    best[i] = lab
                    break

    labels[reachable] = best[reachable]
    # Unreachable points (isolated voxels) default to stem (0).

    if return_debug:
        debug = {
            "knn_graph": g,
            "stem_core_mask": stem_core,
            "seed_lists": seed_lists,
            "geodesic_dist": dmat,
        }
        return labels, debug
    return labels


# ── label-free quality gate ──────────────────────────────────────────────


def leaf_quality_gate(leaf_points, plant_height_cm,
                      min_points=120,
                      max_zspan_frac=0.6,
                      max_verticality=0.995,
                      min_elongation=1.6,
                      min_length_cm=4.0,
                      leaf_max_r_cm=None,
                      median_plant_r_cm=None,
                      min_radial_reach=1.3):
    """Geometric accept/reject for a leaf candidate. No labels required.

    Catches the failure modes a NURBS-RMS gate misses — in particular the
    full-height vertical wedge, which fits a smooth surface (low RMS) yet is
    not a leaf.

    Note on ``max_verticality``: young monocot leaves are genuinely *erect*
    (their principal axis is near-vertical), so verticality alone is NOT a
    wedge discriminator and the default is effectively off (0.995). The two
    discriminators that work label-free are (a) z-span as a fraction of plant
    height — a real blade never spans most of the plant — and (b) radial reach
    — a real blade tip extends outward past the pseudostem, a stem sliver does
    not (needs ``leaf_max_r_cm`` + ``median_plant_r_cm``).

    Returns:
        (keep: bool, reason: str, metrics: dict)
    """
    pts = np.asarray(leaf_points, float)
    n = len(pts)
    if n < min_points:
        return False, f"too few points ({n} < {min_points})", {"n_points": n}

    centred = pts - pts.mean(axis=0)
    vals, vecs = np.linalg.eigh(np.cov(centred, rowvar=False))
    order = np.argsort(vals)[::-1]
    vecs = vecs[:, order]
    scores = centred @ vecs
    length = float(np.ptp(scores[:, 0]))
    width = float(np.ptp(scores[:, 1]))
    elong = length / max(width, 1e-8)
    verticality = float(abs(vecs[2, 0]))   # |z-component| of principal axis
    z_span = float(np.ptp(pts[:, 2]))
    z_frac = z_span / max(plant_height_cm, 1e-8)

    metrics = {
        "n_points": n, "length_cm": length, "width_cm": width,
        "elongation": elong, "verticality": verticality,
        "z_span_cm": z_span, "z_span_frac": z_frac,
    }
    if leaf_max_r_cm is not None and median_plant_r_cm is not None:
        metrics["radial_reach"] = leaf_max_r_cm / max(median_plant_r_cm, 1e-8)

    if z_frac > max_zspan_frac:
        return False, (f"z-span {z_span:.1f} cm = {z_frac:.0%} of plant height "
                       f"(> {max_zspan_frac:.0%}) — vertical wedge"), metrics
    if elong < min_elongation:
        return False, (f"not ribbon-like (elongation {elong:.2f} "
                       f"< {min_elongation})"), metrics
    if length < min_length_cm:
        return False, f"too short ({length:.1f} cm < {min_length_cm})", metrics
    if verticality > max_verticality:
        return False, (f"principal axis perfectly vertical "
                       f"(|z|={verticality:.3f} > {max_verticality})"), metrics
    if "radial_reach" in metrics and metrics["radial_reach"] < min_radial_reach:
        return False, (f"tip does not reach past pseudostem "
                       f"(radial reach {metrics['radial_reach']:.2f} "
                       f"< {min_radial_reach}) — stem sliver"), metrics
    return True, "ok", metrics
