"""Build CPlantBox MappedSegments from organ skeletons."""

import numpy as np
import plantbox as pb

from .skeletonizer import extract_stem_radius, extract_leaf_width


def build_mapped_segments(stem_skeleton, leaf_skeletons, stem_points,
                          leaf_points_dict, dx=0.5):
    """Construct pb.MappedSegments from organ skeletons.

    Args:
        stem_skeleton: np.array([M, 3]) resampled stem skeleton (base to tip)
        leaf_skeletons: dict of leaf_name -> np.array([K, 3]) resampled skeleton
        stem_points: np.array original stem point cloud (for radius estimation)
        leaf_points_dict: dict of leaf_name -> np.array original leaf points
        dx: segment spacing (used for creationTime estimation)

    Returns:
        pb.MappedSegments with nodes, nodeCTs, segments, radii, subTypes, organTypes
    """
    nodes = []       # Vector3d
    nodeCTs = []     # float (creation time in days)
    segments = []    # Vector2i
    radii = []       # float
    subTypes = []    # int
    organTypes = []  # int

    node_idx = 0

    # --- Stem (organType=3, subType=0) ---
    stem_node_start = node_idx
    for i, pt in enumerate(stem_skeleton):
        nodes.append(pb.Vector3d(pt[0], pt[1], pt[2]))
        # Linear creationTime: base=0, tip=max_days
        # Estimate: stem grows ~3 cm/day for maize
        ct = i * dx / 3.0
        nodeCTs.append(ct)
        node_idx += 1

    for i in range(len(stem_skeleton) - 1):
        seg_idx = stem_node_start + i
        segments.append(pb.Vector2i(seg_idx, seg_idx + 1))
        r = extract_stem_radius(stem_points, stem_skeleton, i)
        radii.append(r)
        subTypes.append(0)
        organTypes.append(3)  # stem

    # --- Leaves (organType=4, subType=leaf_rank) ---
    stem_tree = None
    if len(stem_skeleton) > 0:
        from scipy.spatial import KDTree
        stem_tree = KDTree(stem_skeleton)

    for leaf_rank, (leaf_name, leaf_skel) in enumerate(sorted(leaf_skeletons.items()), start=1):
        if len(leaf_skel) < 2:
            continue

        leaf_pts = leaf_points_dict.get(leaf_name)

        # Find attachment point: closest stem node to leaf base
        # Leaf base = the skeleton endpoint closest to the stem
        if stem_tree is not None:
            d0, idx0 = stem_tree.query(leaf_skel[0])
            d1, idx1 = stem_tree.query(leaf_skel[-1])
            if d1 < d0:
                leaf_skel = leaf_skel[::-1]  # flip so [0] is base
                d0, idx0 = d1, idx1
            attach_node = stem_node_start + idx0
        else:
            attach_node = 0

        # Add leaf nodes (skip first point; connect to stem attachment)
        leaf_node_start = node_idx
        for i, pt in enumerate(leaf_skel):
            nodes.append(pb.Vector3d(pt[0], pt[1], pt[2]))
            # CreationTime based on phyllochron: ~3 days between leaves,
            # then linear along leaf
            base_ct = nodeCTs[attach_node] if attach_node < len(nodeCTs) else 0
            leaf_ct = base_ct + i * dx / 5.0  # leaf extends ~5 cm/day
            nodeCTs.append(leaf_ct)
            node_idx += 1

        # First leaf segment connects to stem
        segments.append(pb.Vector2i(attach_node, leaf_node_start))
        if leaf_pts is not None and len(leaf_skel) > 1:
            w = extract_leaf_width(leaf_pts, leaf_skel, 0)
        else:
            w = 0.2
        radii.append(w)
        subTypes.append(leaf_rank)
        organTypes.append(4)  # leaf

        # Remaining leaf segments
        for i in range(len(leaf_skel) - 1):
            if i == 0:
                continue  # already added first segment above
            seg_from = leaf_node_start + i - 1
            seg_to = leaf_node_start + i
            segments.append(pb.Vector2i(seg_from, seg_to))
            if leaf_pts is not None:
                w = extract_leaf_width(leaf_pts, leaf_skel, i)
            else:
                w = 0.2
            radii.append(w)
            subTypes.append(leaf_rank)
            organTypes.append(4)

    # Build MappedSegments
    ms = pb.MappedSegments(nodes, nodeCTs, segments, radii, subTypes, organTypes)

    n_stem = sum(1 for ot in organTypes if ot == 3)
    n_leaf = sum(1 for ot in organTypes if ot == 4)
    print(f"[g1_builder] Built MappedSegments: {len(nodes)} nodes, "
          f"{len(segments)} segments ({n_stem} stem, {n_leaf} leaf)")

    return ms
