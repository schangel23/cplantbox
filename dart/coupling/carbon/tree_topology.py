"""Build tree graph from CPlantBox segments for phloem transport.

Extracts parent-child relationships, topological ordering, and per-segment
properties (organ type, volume, blade length, etc.) needed by the
quasi-steady phloem solver.
"""

from dataclasses import dataclass, field
from collections import deque
from typing import List

import numpy as np
import plantbox as pb


@dataclass
class VascularTree:
    """Tree graph of the vascular network extracted from a CPlantBox plant."""
    n_nodes: int
    n_segments: int
    parent_of: np.ndarray           # [N], parent_of[0] = -1 (root collar)
    children: List[List[int]]       # children[node] -> [child node IDs]
    topo_order: np.ndarray          # BFS from root collar (root-first)
    reverse_topo_order: np.ndarray  # tips first (for forward sweep)
    seg_for_node: np.ndarray        # segment index connecting node to parent (-1 for root collar)

    # Per-segment arrays (length = n_segments)
    seg_length: np.ndarray
    organ_type: np.ndarray          # 2=root, 3=stem, 4=leaf
    sub_type: np.ndarray
    seg_vol: np.ndarray
    blade_length: np.ndarray
    leaf_surface: np.ndarray
    node_z: np.ndarray              # z coordinate of each node
    is_leaf_node: np.ndarray        # True if node is a leaf-blade segment endpoint
    is_root_below: np.ndarray       # True if segment is below ground (seg2cell >= 0)
    seg_radius: np.ndarray          # segment radius [cm]


def build_tree(plant: pb.MappedPlant) -> VascularTree:
    """Build a VascularTree from a CPlantBox MappedPlant.

    The tree is rooted at the root collar (node 0). Each segment connects
    a parent node (.x) to a child node (.y). This gives exactly N-1 segments
    for N nodes (tree guarantee, asserted in PiafMunch runPM.cpp:355).

    Args:
        plant: A grown MappedPlant with segments, nodes, and soil grid.

    Returns:
        VascularTree with all topology and per-segment properties.
    """
    segments = plant.getSegments()
    nodes = plant.getNodes()
    n_nodes = len(nodes)
    n_segments = len(segments)

    assert n_segments == n_nodes - 1, (
        f"Tree guarantee violated: {n_segments} segments vs {n_nodes} nodes "
        f"(expected {n_nodes - 1})"
    )

    # Per-segment arrays from CPlantBox
    ot_arr = np.array(plant.organTypes, dtype=np.int32)
    st_arr = np.array(plant.subTypes, dtype=np.int32)
    seg_len = np.array(plant.segLength(), dtype=np.float64)
    radii = np.array(plant.radii, dtype=np.float64)
    blade_len = np.array(plant.bladeLength, dtype=np.float64)
    leaf_surf = np.array(plant.leafBladeSurface, dtype=np.float64)
    # seg2cell is a dict {seg_idx: cell_idx}, not a list
    seg2cell_dict = plant.seg2cell
    seg2cell = np.full(n_segments, -1, dtype=np.int32)
    for si, ci in seg2cell_dict.items():
        if 0 <= si < n_segments:
            seg2cell[si] = ci
    # Tissue volume per segment. CPlantBox's segVol (C++) is not exposed
    # in the Python API. For roots and stems, pi*r^2*L is correct (cylindrical).
    # For leaves, CPlantBox's r=0.04 cm is the midrib radius — far smaller than
    # the blade. Use leafBladeSurface * leaf_thickness for realistic tissue mass.
    LEAF_THICKNESS = 0.025  # cm — maize blade thickness (Colbert & Rhoades 1993)
    seg_vol = np.pi * radii**2 * seg_len  # default: cylindrical
    leaf_mask = (ot_arr == 4) & (leaf_surf > 0)
    seg_vol[leaf_mask] = leaf_surf[leaf_mask] * LEAF_THICKNESS

    # Node z-coordinates
    node_z = np.array([n.z for n in nodes], dtype=np.float64)

    # Build parent_of and children arrays from segment parent-child pairs
    parent_of = np.full(n_nodes, -1, dtype=np.int32)
    children = [[] for _ in range(n_nodes)]
    seg_for_node = np.full(n_nodes, -1, dtype=np.int32)

    for seg_idx in range(n_segments):
        seg = segments[seg_idx]
        parent_node = seg.x
        child_node = seg.y
        parent_of[child_node] = parent_node
        children[parent_node].append(child_node)
        seg_for_node[child_node] = seg_idx

    # BFS from root collar (node 0) for topological order
    topo_order = []
    visited = np.zeros(n_nodes, dtype=bool)
    queue = deque([0])
    visited[0] = True
    while queue:
        node = queue.popleft()
        topo_order.append(node)
        for child in children[node]:
            if not visited[child]:
                visited[child] = True
                queue.append(child)

    topo_order = np.array(topo_order, dtype=np.int32)
    reverse_topo_order = topo_order[::-1].copy()

    # Per-node boolean flags
    is_leaf_node = np.zeros(n_nodes, dtype=bool)
    is_root_below = np.zeros(n_segments, dtype=bool)
    for seg_idx in range(n_segments):
        child_node = segments[seg_idx].y
        if blade_len[seg_idx] > 0:
            is_leaf_node[child_node] = True
        # Use z-coordinate to determine below-ground status (z < 0).
        # More robust than seg2cell which requires mapSegments() to have
        # been called (seg2cell = INT_MIN for unmapped segments).
        if ot_arr[seg_idx] == 2 and node_z[child_node] < 0:
            is_root_below[seg_idx] = True

    return VascularTree(
        n_nodes=n_nodes,
        n_segments=n_segments,
        parent_of=parent_of,
        children=children,
        topo_order=topo_order,
        reverse_topo_order=reverse_topo_order,
        seg_for_node=seg_for_node,
        seg_length=seg_len,
        organ_type=ot_arr,
        sub_type=st_arr,
        seg_vol=seg_vol,
        blade_length=blade_len,
        leaf_surface=leaf_surf,
        node_z=node_z,
        is_leaf_node=is_leaf_node,
        is_root_below=is_root_below,
        seg_radius=radii,
    )
