"""Graph-based overlapping leaf resolution for unlabeled point clouds.

Inspired by MonGraphSeg (Tobies et al. 2025) step 3c: splitOverlappingLeaves,
and the broader graph refinement pipeline (3a-3f). Operates on skeleton points
produced by Laplacian contraction or similar methods, building a NetworkX graph
and resolving cycles caused by overlapping/touching leaves.

Usage:
    from plantbox.visualisation.pheno4d_to_g1.graph_refinement import (
        build_skeleton_graph,
        remove_ground_nodes,
        split_overlapping_leaves,
        identify_stem_and_leaves,
        prune_spurious_branches,
        segment_point_cloud,
    )

    G = build_skeleton_graph(skeleton_points, k=3)
    G, start = remove_ground_nodes(G, angle_threshold_deg=50)
    G = split_overlapping_leaves(G)
    stem_path, leaf_dict = identify_stem_and_leaves(G, start)
    stem_path, leaf_dict = prune_spurious_branches(G, stem_path, leaf_dict, min_points=10)
    labels = segment_point_cloud(points, skeleton_points, stem_path, leaf_dict)

Dependencies: numpy, scipy, networkx
"""

import numpy as np
from scipy.spatial import KDTree
import networkx as nx


# ---------------------------------------------------------------------------
# 1. Initial graph construction (MonGraphSeg 3a)
# ---------------------------------------------------------------------------

def build_skeleton_graph(skeleton_points, k=3):
    """Build initial graph from skeleton points with k-nearest-neighbor edges.

    Each skeleton point becomes a node with a ``pos`` attribute (np.array of
    shape (3,)).  Edges connect each node to its *k* nearest neighbours,
    weighted by Euclidean distance.

    Args:
        skeleton_points: np.array([N, 3]) skeleton node positions (cm).
        k: number of nearest neighbours to connect per node.

    Returns:
        G: ``nx.Graph`` with node attribute ``pos`` and edge attribute ``weight``.
    """
    skeleton_points = np.asarray(skeleton_points, dtype=float)
    n = len(skeleton_points)
    if n == 0:
        return nx.Graph()

    G = nx.Graph()
    for i, pos in enumerate(skeleton_points):
        G.add_node(i, pos=pos.copy())

    tree = KDTree(skeleton_points)
    # query k+1 because the first hit is the point itself
    distances, indices = tree.query(skeleton_points, k=min(k + 1, n))

    for i in range(n):
        for j_idx in range(1, indices.shape[1]):
            j = indices[i, j_idx]
            dist = distances[i, j_idx]
            if not G.has_edge(i, j):
                G.add_edge(i, j, weight=dist)

    return G


# ---------------------------------------------------------------------------
# 2. Ground removal (MonGraphSeg 3b)
# ---------------------------------------------------------------------------

def remove_ground_nodes(G, angle_threshold_deg=50):
    """Remove ground nodes and identify the plant starting node.

    The algorithm finds the highest node in the graph, then evaluates paths
    from every node to that highest node.  The lowest node whose upward path
    fulfils a verticality criterion (angle from vertical < *angle_threshold_deg*)
    is chosen as the plant base.  All nodes below this base are removed.

    Args:
        G: ``nx.Graph`` from :func:`build_skeleton_graph`.
        angle_threshold_deg: maximum angle from vertical (degrees) to accept
            an edge as "vertical growth".

    Returns:
        G_plant: copy of *G* with ground nodes removed.
        start_node: node ID of the plant base.
    """
    if G.number_of_nodes() == 0:
        return G.copy(), None

    positions = nx.get_node_attributes(G, "pos")
    highest_node = max(positions, key=lambda n: positions[n][2])

    angle_thresh_rad = np.radians(angle_threshold_deg)
    start_candidates = []

    for node in G.nodes():
        if node == highest_node:
            continue
        try:
            path = nx.shortest_path(G, node, highest_node, weight="weight")
        except nx.NetworkXNoPath:
            continue

        # Walk upward along path; first edge satisfying verticality marks start
        for i in range(len(path) - 1):
            p0 = positions[path[i]]
            p1 = positions[path[i + 1]]
            direction = p1 - p0
            length = np.linalg.norm(direction)
            if length < 1e-8:
                continue
            cos_angle = direction[2] / length  # cos(angle from vertical)
            # clamp for numerical safety
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.arccos(cos_angle)
            if angle < angle_thresh_rad:
                start_candidates.append(path[i])
                break

    if not start_candidates:
        # Fallback: use the lowest node overall
        start_node = min(positions, key=lambda n: positions[n][2])
    else:
        start_node = min(start_candidates, key=lambda n: positions[n][2])

    start_z = positions[start_node][2]
    ground_nodes = [n for n in G.nodes() if positions[n][2] < start_z]

    G_plant = G.copy()
    G_plant.remove_nodes_from(ground_nodes)

    # Ensure start_node is still in the graph
    if start_node not in G_plant:
        if G_plant.number_of_nodes() == 0:
            return G_plant, None
        start_node = min(G_plant.nodes(), key=lambda n: positions[n][2])

    # FP4D skeleton graphs can remain disconnected after low ground nodes are
    # removed.  In that case, the first vertical edge may belong to a small
    # basal side component; start stem tracing in the component with the
    # largest vertical span instead.
    components = list(nx.connected_components(G_plant))
    if len(components) > 1:
        def z_span(component):
            z_vals = [positions[n][2] for n in component]
            return max(z_vals) - min(z_vals)

        main_component = max(components, key=z_span)
        current_component = next(
            component for component in components if start_node in component
        )
        if z_span(main_component) > z_span(current_component) + 1e-6:
            start_node = min(main_component, key=lambda n: positions[n][2])

    return G_plant, start_node


# ---------------------------------------------------------------------------
# 3. Overlapping leaf splitting (MonGraphSeg 3c)
# ---------------------------------------------------------------------------

def split_overlapping_leaves(G, min_cycle_length=None, junction_merge_distance=5.0):
    """Resolve cycles created by overlapping / touching leaves.

    When two leaves overlap in 3D space, the k-NN graph may connect them,
    creating a cycle.  This function detects such cycles and splits them at
    the point farthest from the central (vertical) stem axis, which is where
    the two leaves are most likely merely spatially close rather than
    structurally connected.

    Short cycles (< *min_cycle_length* edges) are treated as leaf-stem
    junction artefacts and are broken by removing the longest edge.

    Args:
        G: ``nx.Graph`` (modified in-place).
        min_cycle_length: cycles shorter than this are treated as junction
            artefacts (edge count).  If ``None``, use an adaptive threshold
            that keeps the original Pheno4D scale at roughly 50 nodes while
            allowing compact FP4D skeletons to split at 10-15 node cycles.
        junction_merge_distance: if two cycle-breaking candidate nodes are
            within this distance (cm), merge the operations.

    Returns:
        G: the same graph object, with cycle-creating edges removed.
    """
    positions = nx.get_node_attributes(G, "pos")

    if min_cycle_length is None:
        min_cycle_length = max(8, min(50, int(round(0.15 * G.number_of_nodes()))))

    # Estimate central stem axis as the vertical line through the XY centroid
    all_pos = np.array([positions[n] for n in G.nodes()])
    stem_xy = all_pos[:, :2].mean(axis=0)  # XY centroid

    max_iterations = 200  # safety bound
    for _ in range(max_iterations):
        try:
            cycle = nx.find_cycle(G)
        except nx.NetworkXNoCycle:
            break

        cycle_nodes = list(dict.fromkeys(n for e in cycle for n in e[:2]))

        if len(cycle_nodes) < min_cycle_length:
            # Short cycle: remove the longest edge (likely spurious connection)
            longest_edge = max(
                cycle,
                key=lambda e: G.edges[e[0], e[1]].get("weight", 0.0),
            )
            G.remove_edge(longest_edge[0], longest_edge[1])
        else:
            # Long cycle (overlapping leaves): split at the node farthest
            # from the stem axis in XY.
            max_dist = -1.0
            split_node = cycle_nodes[0]
            for n in cycle_nodes:
                xy_dist = np.linalg.norm(positions[n][:2] - stem_xy)
                if xy_dist > max_dist:
                    max_dist = xy_dist
                    split_node = n

            # Remove the edge incident to split_node that has the longest weight
            # among the cycle edges touching it.
            incident_cycle_edges = [
                (u, v)
                for (u, v, *_) in cycle
                if u == split_node or v == split_node
            ]
            if incident_cycle_edges:
                edge_to_remove = max(
                    incident_cycle_edges,
                    key=lambda e: G.edges[e[0], e[1]].get("weight", 0.0),
                )
                G.remove_edge(edge_to_remove[0], edge_to_remove[1])
            else:
                # Fallback: remove longest edge in the whole cycle
                longest_edge = max(
                    cycle,
                    key=lambda e: G.edges[e[0], e[1]].get("weight", 0.0),
                )
                G.remove_edge(longest_edge[0], longest_edge[1])

    return G


# ---------------------------------------------------------------------------
# 4. Stem and leaf identification (MonGraphSeg 3d + 3e)
# ---------------------------------------------------------------------------

def identify_stem_and_leaves(G, start_node):
    """Identify stem path and individual leaf instances.

    For unbranched monocots, the stem is the path from *start_node* to the
    **highest** terminal node (maximum Z coordinate).  This exploits the
    biological prior that the stem grows vertically and reaches the highest
    point, while leaves branch off laterally at lower heights.

    Tie-breaking uses path length (prefer longer paths) when multiple
    terminals share similar Z height.

    Args:
        G: ``nx.Graph`` (acyclic after :func:`split_overlapping_leaves`).
        start_node: plant base node ID.

    Returns:
        stem_path: list of node IDs from base to tip.
        leaf_dict: ``{leaf_id: [node_ids]}`` where leaf_id starts at 1.
    """
    if start_node is None or start_node not in G:
        return [], {}

    positions = nx.get_node_attributes(G, "pos")

    # Find terminal nodes (degree 1, excluding start if it has degree 1)
    terminal_nodes = [
        n for n in G.nodes()
        if G.degree(n) == 1 and n != start_node
    ]
    # If start_node itself is degree-1, it is still the start
    if G.degree(start_node) == 1 and not terminal_nodes:
        return [start_node], {}

    # Evaluate every path from start_node to a terminal
    best_path = []
    best_terminal_z = -float("inf")
    best_path_len = 0.0

    for terminal in terminal_nodes:
        try:
            path = nx.shortest_path(G, start_node, terminal, weight="weight")
        except nx.NetworkXNoPath:
            continue

        terminal_z = positions[terminal][2]
        path_len = sum(
            G.edges[path[i], path[i + 1]].get("weight", 1.0)
            for i in range(len(path) - 1)
        )

        # Primary: highest terminal Z (monocot stem = tallest)
        # Secondary: longest path (tie-break)
        if terminal_z > best_terminal_z + 0.5 or (
            abs(terminal_z - best_terminal_z) <= 0.5
            and path_len > best_path_len
        ):
            best_terminal_z = terminal_z
            best_path_len = path_len
            best_path = path

    stem_path = best_path
    stem_set = set(stem_path)

    # Remove stem edges to isolate leaf subgraphs
    G_leaves = G.copy()
    for i in range(len(stem_path) - 1):
        if G_leaves.has_edge(stem_path[i], stem_path[i + 1]):
            G_leaves.remove_edge(stem_path[i], stem_path[i + 1])

    # Each connected component that is NOT purely stem nodes is a leaf
    leaf_dict = {}
    leaf_id = 1
    for comp in nx.connected_components(G_leaves):
        # Skip components that are entirely on the stem
        non_stem = comp - stem_set
        if not non_stem:
            continue

        # Include the stem junction node for later attachment
        leaf_nodes = list(comp)

        # Order nodes: from stem junction outward (BFS from junction)
        junctions = comp & stem_set
        if junctions:
            root = next(iter(junctions))
        else:
            # Pick the node closest to any stem node
            positions = nx.get_node_attributes(G, "pos")
            stem_positions = np.array([positions[n] for n in stem_path])
            root = min(
                leaf_nodes,
                key=lambda n: np.min(
                    np.linalg.norm(stem_positions - positions[n], axis=1)
                ),
            )

        ordered = _bfs_order(G_leaves, root, leaf_nodes)
        leaf_dict[leaf_id] = ordered
        leaf_id += 1

    return stem_path, leaf_dict


def _bfs_order(G, root, node_subset):
    """BFS ordering of *node_subset* starting from *root*."""
    visited = set()
    order = []
    queue = [root]
    subset = set(node_subset)

    while queue:
        node = queue.pop(0)
        if node in visited or node not in subset:
            continue
        visited.add(node)
        order.append(node)
        for neighbor in G.neighbors(node):
            if neighbor not in visited and neighbor in subset:
                queue.append(neighbor)

    return order


# ---------------------------------------------------------------------------
# 5. Pruning (MonGraphSeg 3f)
# ---------------------------------------------------------------------------

def prune_spurious_branches(G, stem_path, leaf_dict, min_points=10):
    """Remove leaves with fewer than *min_points* skeleton nodes.

    Also trims each leaf to its longest path from the stem junction, removing
    secondary branches within the leaf subgraph (which are likely noise).

    Args:
        G: ``nx.Graph``.
        stem_path: list of node IDs (from :func:`identify_stem_and_leaves`).
        leaf_dict: ``{leaf_id: [node_ids]}``.
        min_points: minimum skeleton nodes for a valid leaf.

    Returns:
        stem_path: unchanged.
        pruned_dict: ``{leaf_id: [node_ids]}`` with short leaves removed and
            each leaf trimmed to its longest branch.
    """
    positions = nx.get_node_attributes(G, "pos")
    stem_set = set(stem_path)
    pruned = {}

    for leaf_id, nodes in leaf_dict.items():
        if len(nodes) < min_points:
            continue

        # Build subgraph for this leaf
        sub = G.subgraph(nodes).copy()

        # Find junction node (on stem) and terminal nodes
        junctions = [n for n in nodes if n in stem_set]
        if junctions:
            root = junctions[0]
        else:
            # Pick node closest to stem
            stem_positions = np.array([positions[n] for n in stem_path])
            root = min(
                nodes,
                key=lambda n: np.min(
                    np.linalg.norm(stem_positions - positions[n], axis=1)
                ),
            )

        terminals = [n for n in sub.nodes() if sub.degree(n) == 1 and n != root]
        if not terminals:
            pruned[leaf_id] = nodes
            continue

        # Keep only the longest path from root to any terminal
        best_path = []
        best_length = 0.0

        for t in terminals:
            try:
                path = nx.shortest_path(sub, root, t, weight="weight")
            except nx.NetworkXNoPath:
                continue

            path_length = sum(
                sub.edges[path[i], path[i + 1]].get("weight", 1.0)
                for i in range(len(path) - 1)
            )
            if path_length > best_length:
                best_length = path_length
                best_path = path

        if best_path:
            pruned[leaf_id] = best_path
        else:
            pruned[leaf_id] = nodes

    return stem_path, pruned


# ---------------------------------------------------------------------------
# 6. Point cloud segmentation (MonGraphSeg section 4)
# ---------------------------------------------------------------------------

def segment_point_cloud(points, skeleton_points, stem_path, leaf_dict):
    """Assign each point to the nearest skeleton segment.

    Labels:
        0 = unassigned / ground
        1 = stem
        2, 3, ... = leaf instances (matching keys of *leaf_dict*)

    Args:
        points: np.array([N, 3]) original point cloud (cm).
        skeleton_points: np.array([M, 3]) all skeleton node positions (cm),
            indexed consistently with the node IDs in *stem_path* and
            *leaf_dict*.
        stem_path: list of node IDs belonging to the stem.
        leaf_dict: ``{leaf_id: [node_ids]}``.

    Returns:
        labels: np.array([N,], dtype=int) per-point labels.
    """
    points = np.asarray(points, dtype=float)
    skeleton_points = np.asarray(skeleton_points, dtype=float)
    n_points = len(points)

    if n_points == 0:
        return np.zeros(0, dtype=int)

    # Build a label map: skeleton node ID -> label
    node_label = {}
    for nid in stem_path:
        node_label[nid] = 1

    for leaf_id, nodes in leaf_dict.items():
        for nid in nodes:
            if nid not in node_label:  # stem junction stays as stem
                node_label[nid] = leaf_id + 1  # leaf IDs offset by 1

    # Collect labelled skeleton positions
    labeled_ids = sorted(node_label.keys())
    labeled_positions = skeleton_points[labeled_ids]
    labeled_labels = np.array([node_label[nid] for nid in labeled_ids])

    if len(labeled_positions) == 0:
        return np.zeros(n_points, dtype=int)

    # Nearest-neighbour assignment
    tree = KDTree(labeled_positions)
    _, nearest_idx = tree.query(points)
    labels = labeled_labels[nearest_idx]

    return labels
