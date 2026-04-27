"""CPlantBox plant → OpenAlea MTG converter.

Converts CPlantBox's segment-based plant representation to an OpenAlea MTG
(Multi-scale Tree Graph) data structure.

MTG Scale Hierarchy:
    0: $ (Scene, implicit root)
    1: P (Plant)
    2: A (Axis / main stem)
    3: M (Metamer / Phytomer)
    4: I/S/B/L (Organ: internode, sheath, blade, or monolithic leaf)

Usage:
    import plantbox as pb
    from dart.coupling.geometry.cplantbox_to_mtg import cplantbox_to_mtg

    plant = pb.MappedPlant()
    # ... grow plant ...
    g = cplantbox_to_mtg(plant, decompose_phytomer=True)
"""

import pickle
from pathlib import Path

import numpy as np
from openalea.mtg import MTG

try:
    import plantbox as pb
except ImportError:
    pb = None

# Class-at-scale mapping for MTG serialization
CLASS_AT_SCALE = {'P': 1, 'A': 2, 'M': 3, 'I': 4, 'S': 4, 'B': 4, 'L': 4}


def cplantbox_to_mtg(plant, decompose_phytomer=True):
    """Convert a CPlantBox MappedPlant to an OpenAlea MTG.

    Args:
        plant: A grown pb.MappedPlant with segments and nodes.
        decompose_phytomer: If True, leaf organs are split into sheath (S) +
            blade (B) based on subType parity (even=sheath, odd=blade).
            If False, each leaf becomes a monolithic L vertex.

    Returns:
        openalea.mtg.MTG with scale hierarchy 0-4.
    """
    g = MTG()

    # Scale 1: Plant
    plant_vid = g.add_component(g.root, label='P')

    # Scale 2: Axis (main stem)
    axis_vid = g.add_component(plant_vid, label='A')

    # Get the main stem organ
    stems = plant.getOrgans(pb.stem)
    if not stems:
        return g
    main_stem = stems[0]

    # Group leaf organs into phytomers
    phytomers = _group_phytomers(plant, main_stem, decompose_phytomer)
    if not phytomers:
        return g

    # Split stem into per-phytomer internodes
    attachment_nids = [p['attachment_nid'] for p in phytomers]
    stem_nids = list(main_stem.getNodeIds())
    internodes = _split_stem_internodes(main_stem, attachment_nids, stem_nids)

    # Scale 3+4: Metamers and organs
    prev_metamer = None
    for phytomer, internode_data in zip(phytomers, internodes):
        # Create Metamer vertex (scale 3)
        if prev_metamer is None:
            metamer_vid = g.add_component(axis_vid, label='M')
        else:
            metamer_vid = g.add_child(prev_metamer, edge_type='<', label='M')

        # Create Internode vertex (scale 4) — decomposition from metamer
        internode_vid = g.add_component(metamer_vid, label='I')
        _set_vertex_props(g, internode_vid, {
            'organ_type': 'internode',
            'sub_type': int(main_stem.getParameter("subType")),
            'skeleton': internode_data['skeleton'],
            'widths': internode_data['widths'],
            'length': float(np.sum(np.linalg.norm(
                np.diff(internode_data['skeleton'], axis=0), axis=1))),
            'position_x': float(internode_data['skeleton'][0, 0]),
            'position_y': float(internode_data['skeleton'][0, 1]),
            'position_z': float(internode_data['skeleton'][0, 2]),
            'node_ids': internode_data['node_ids'],
            'age': None,
            'aPAR': None, 'Tleaf': None, 'An': None, 'gs': None,
        })

        if decompose_phytomer:
            # Sheath + Blade as + children of internode
            sheath = phytomer.get('sheath')
            blade = phytomer.get('blade')
            if sheath is not None:
                sheath_vid = g.add_child(
                    internode_vid, edge_type='+', label='S')
                _set_vertex_props(g, sheath_vid, _organ_properties(sheath, 'sheath'))
            if blade is not None:
                blade_vid = g.add_child(
                    internode_vid, edge_type='+', label='B')
                _set_vertex_props(g, blade_vid, _organ_properties(blade, 'blade'))
        else:
            # Monolithic leaf as + child of internode
            leaf = phytomer.get('leaf')
            if leaf is not None:
                leaf_vid = g.add_child(
                    internode_vid, edge_type='+', label='L')
                _set_vertex_props(g, leaf_vid, _organ_properties(leaf, 'leaf'))

        prev_metamer = metamer_vid

    return g


def _group_phytomers(plant, main_stem, decompose_phytomer):
    """Infer phytomers from CPlantBox's flat organ list.

    A leaf organ's first node (organ.getNodeIds()[0]) is the shared stem node
    where it attaches. Group leaf organs by attachment stem node, sort base-to-tip.

    Args:
        plant: MappedPlant.
        main_stem: The primary stem organ.
        decompose_phytomer: If True, expect sheath (even) + blade (odd) pairs.

    Returns:
        List of dicts, each with 'attachment_nid' and either
        'sheath'+'blade' (phytomer mode) or 'leaf' (monolithic mode).
    """
    stem_nids = [int(nid) for nid in main_stem.getNodeIds()]
    stem_nid_set = set(stem_nids)
    stem_nid_order = {nid: idx for idx, nid in enumerate(stem_nids)}

    # Build stem node positions for fallback matching
    stem_nodes = main_stem.getNodes()
    stem_nodes_pos = np.array([[n.x, n.y, n.z] for n in stem_nodes])

    leaf_organs = [o for o in plant.getOrgans(pb.leaf)
                   if len(o.getNodes()) > 1]

    # Group by attachment node
    groups = {}  # attachment_nid → list of organs
    for organ in leaf_organs:
        lrp = organ.getLeafRandomParameter()
        is_pseudostem = organ.getParameter("isPseudostem") == 1
        # Skip broken leaf subtypes but keep sheaths (Width_blade=0 is normal)
        if not is_pseudostem and lrp.Width_blade < 0.01:
            continue
        attach_nid = _find_stem_attachment_node(
            organ, stem_nid_set, stem_nodes_pos, stem_nids)
        if attach_nid is None:
            continue
        groups.setdefault(attach_nid, []).append(organ)

    # Sort groups by stem position (base-to-tip)
    sorted_nids = sorted(groups.keys(), key=lambda nid: stem_nid_order.get(nid, 999))

    phytomers = []
    for nid in sorted_nids:
        organs_at_node = groups[nid]
        entry = {'attachment_nid': nid}

        if decompose_phytomer:
            sheaths = [o for o in organs_at_node
                       if int(o.getParameter("subType")) % 2 == 0]
            blades = [o for o in organs_at_node
                      if int(o.getParameter("subType")) % 2 == 1]
            entry['sheath'] = sheaths[0] if sheaths else None
            entry['blade'] = blades[0] if blades else None
        else:
            entry['leaf'] = organs_at_node[0] if organs_at_node else None

        phytomers.append(entry)

    return phytomers


def _split_stem_internodes(stem_organ, attachment_nids, stem_nids):
    """Split a continuous stem organ into per-phytomer internodes.

    Internode i = stem nodes from attachment_nid[i] to attachment_nid[i+1].
    Basal segment (below first attachment) included in first phytomer.
    Apical segment (above last attachment) included in last phytomer.

    Args:
        stem_organ: The stem pb.Organ.
        attachment_nids: Ordered list of attachment node IDs (base-to-tip).
        stem_nids: Ordered list of all stem node IDs.

    Returns:
        List of dicts with 'skeleton' (ndarray[N,3]), 'widths' (ndarray[N]),
        'node_ids' (list[int]).
    """
    stem_nodes = stem_organ.getNodes()
    stem_pos = np.array([[n.x, n.y, n.z] for n in stem_nodes])
    radius = stem_organ.getParameter("a")
    stem_widths = np.full(len(stem_nodes), 2.0 * radius)

    nid_to_idx = {int(nid): idx for idx, nid in enumerate(stem_nids)}

    # Get stem indices for each attachment point
    attach_indices = []
    for nid in attachment_nids:
        idx = nid_to_idx.get(nid)
        if idx is not None:
            attach_indices.append(idx)

    if not attach_indices:
        # Fallback: one big internode for the whole stem
        return [{
            'skeleton': stem_pos,
            'widths': stem_widths,
            'node_ids': [int(nid) for nid in stem_nids],
        }]

    internodes = []
    n = len(attach_indices)
    for i in range(n):
        # Start: base of stem for first phytomer, attachment point otherwise
        s = 0 if i == 0 else attach_indices[i]
        # End: next attachment point, or stem tip for last phytomer
        e = attach_indices[i + 1] if i + 1 < n else len(stem_pos) - 1
        # Ensure at least 2 points (adjacent or terminal attachments)
        if e <= s:
            e = min(s + 1, len(stem_pos) - 1)
        if e == s and s > 0:
            s = s - 1
        internodes.append({
            'skeleton': stem_pos[s:e + 1].copy(),
            'widths': stem_widths[s:e + 1].copy(),
            'node_ids': [int(stem_nids[j]) for j in range(s, e + 1)],
        })

    return internodes


def _find_stem_attachment_node(organ, stem_nid_set, stem_nodes_pos, stem_nids):
    """Find the stem node where a leaf organ attaches.

    Primary: exact match of organ's first node ID in stem node set.
    Fallback: nearest Euclidean match within 1.0 cm tolerance.

    Args:
        organ: A leaf pb.Organ.
        stem_nid_set: Set of stem node IDs.
        stem_nodes_pos: ndarray[N,3] of stem node positions.
        stem_nids: Ordered list of stem node IDs.

    Returns:
        int node ID or None if no match found.
    """
    organ_nids = organ.getNodeIds()
    first_nid = int(organ_nids[0])

    # Primary: exact ID match
    if first_nid in stem_nid_set:
        return first_nid

    # Fallback: geometric nearest-neighbor
    first_node = organ.getNodes()[0]
    pos = np.array([first_node.x, first_node.y, first_node.z])
    dists = np.linalg.norm(stem_nodes_pos - pos, axis=1)
    min_idx = np.argmin(dists)
    if dists[min_idx] < 1.0:
        return int(stem_nids[min_idx])

    return None


def _organ_properties(organ, organ_type_str):
    """Extract properties from a CPlantBox organ for MTG vertex.

    Args:
        organ: A pb.Organ (leaf, sheath, or blade).
        organ_type_str: One of 'sheath', 'blade', 'leaf'.

    Returns:
        Dict of properties.
    """
    nodes = organ.getNodes()
    skeleton = np.array([[n.x, n.y, n.z] for n in nodes])
    node_ids = [int(nid) for nid in organ.getNodeIds()]

    # Width computation (reuses cplantbox_adapter.py logic)
    lrp = organ.getLeafRandomParameter()
    is_pseudostem = organ.getParameter("isPseudostem") == 1

    if is_pseudostem:
        # Sheaths are cylindrical — width from radius parameter
        radius = organ.getParameter("a")
        widths = np.full(len(nodes), 2.0 * radius)
    else:
        width_blade = lrp.Width_blade
        phi = np.array(lrp.leafGeometryPhi)
        x = np.array(lrp.leafGeometryX)

        if len(phi) > 0 and len(x) > 0:
            diffs = np.diff(skeleton, axis=0)
            seg_lengths = np.linalg.norm(diffs, axis=1)
            cumulative = np.concatenate([[0.0], np.cumsum(seg_lengths)])
            total_length = cumulative[-1]
            if total_length > 1e-12:
                fracs = cumulative / total_length
                phi_min, phi_max = phi.min(), phi.max()
                node_phi = phi_min + fracs * (phi_max - phi_min)
                rel_widths = np.interp(node_phi, phi, x)
                widths = rel_widths * width_blade * 2.0
            else:
                widths = np.full(len(nodes), width_blade * 2.0)
        else:
            widths = np.full(len(nodes), width_blade * 2.0)

    seg_lengths = np.linalg.norm(np.diff(skeleton, axis=0), axis=1)
    length = float(np.sum(seg_lengths))

    return {
        'organ_type': organ_type_str,
        'sub_type': int(organ.getParameter("subType")),
        'skeleton': skeleton,
        'widths': widths,
        'length': length,
        'width': float(np.max(widths)) if len(widths) > 0 else 0.0,
        'age': float(organ.getAge()),
        'position_x': float(skeleton[0, 0]),
        'position_y': float(skeleton[0, 1]),
        'position_z': float(skeleton[0, 2]),
        'node_ids': node_ids,
        'aPAR': None, 'Tleaf': None, 'An': None, 'gs': None,
    }


def _set_vertex_props(g, vid, props):
    """Set multiple properties on an MTG vertex."""
    for key, val in props.items():
        g.property(key)[vid] = val


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------
# NOTE: openalea.mtg.io.write_mtg has a bug with 4-scale MTGs — it only
# serializes the first metamer's sub-tree. We use pickle instead, which
# preserves the full topology and all property types including ndarrays.

def write_mtg_file(g, filepath):
    """Save an MTG to disk.

    Writes two files:
      - {filepath}.pkl  — full MTG with topology and all properties (pickle)
      - {filepath}.npz  — array properties for efficient direct access

    Args:
        g: openalea.mtg.MTG instance.
        filepath: Base path (without extension).
    """
    filepath = Path(filepath)

    # Pickle the full MTG
    with open(filepath.with_suffix('.pkl'), 'wb') as f:
        pickle.dump(g, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Save array properties to npz for efficient access
    arrays = {}
    array_prop_names = ('skeleton', 'widths', 'node_ids')
    for prop_name in array_prop_names:
        prop_dict = g.property(prop_name)
        for vid, val in prop_dict.items():
            if val is not None:
                key = f"{prop_name}_v{vid}"
                if isinstance(val, np.ndarray):
                    arrays[key] = val
                elif isinstance(val, list):
                    arrays[key] = np.array(val)
    if arrays:
        np.savez_compressed(filepath.with_suffix('.npz'), **arrays)


def read_mtg_with_arrays(filepath):
    """Read an MTG from disk.

    Loads the pickle file and optionally merges companion npz arrays.

    Args:
        filepath: Base path (without extension) or path to .pkl file.

    Returns:
        openalea.mtg.MTG instance.
    """
    filepath = Path(filepath)
    pkl_path = filepath.with_suffix('.pkl') if filepath.suffix != '.pkl' else filepath

    with open(pkl_path, 'rb') as f:
        g = pickle.load(f)

    # Merge npz arrays if present (they're already in the pickle, but
    # this allows updating arrays without re-pickling)
    npz_path = pkl_path.with_suffix('.npz')
    if npz_path.exists():
        data = np.load(npz_path, allow_pickle=True)
        for key in data.files:
            parts = key.rsplit('_v', 1)
            if len(parts) == 2:
                prop_name = parts[0]
                vid = int(parts[1])
                g.property(prop_name)[vid] = data[key]

    return g
