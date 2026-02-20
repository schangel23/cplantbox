"""
VTK-based plant export with node ID tracking for DART coupling.

Uses the high-quality VTK rendering pipeline (tubes + leaf quads)
while preserving triangle-to-segment mapping needed for DART radiative
transfer feedback (e.g. Baleno per-triangle temperatures mapped back
to CPlantBox segments).

Usage:
    import plantbox as pb
    from plantbox.visualisation.vtk_dart_export import export_vtk_plant_for_dart

    plant = pb.MappedPlant()
    # ... setup and simulate ...
    export_vtk_plant_for_dart(plant, 'results/wheat_dart')
"""

import plantbox as pb
from plantbox.visualisation.vtk_plot import segs_to_polydata, create_leaf_
from plantbox.visualisation.vtk_tools import np_convert, vtk_data

import vtk
import numpy as np
import json
import os


def export_vtk_plant_for_dart(plant, output_prefix, sim_days=None):
    """Export VTK-quality plant geometry as OBJ with DART coupling mapping.

    Produces:
        {output_prefix}.obj  -- OBJ mesh (tubes for stems/roots, quads for leaves)
        {output_prefix}.mtl  -- material file
        {output_prefix}_mapping.json  -- sidecar mapping: cell index -> CPlantBox node ID

    Args:
        plant: a pb.Plant, pb.MappedPlant, or pb.Organism
        output_prefix: path prefix for output files (e.g. 'results/wheat_dart')
        sim_days: optional metadata (simulation age in days)

    Returns:
        dict with summary statistics
    """
    # --- build SegmentAnalyser ---
    if isinstance(plant, pb.Organism):
        ana = pb.SegmentAnalyser(plant)
    elif isinstance(plant, pb.MappedSegments):
        ana = pb.SegmentAnalyser(plant)
    elif isinstance(plant, pb.SegmentAnalyser):
        ana = plant
    else:
        ana = pb.SegmentAnalyser(plant)

    segments = ana.segments  # Vector2i list
    nodes = ana.nodes        # Vector3d list

    # Build segment -> nodeIdY lookup (the end-node global ID per segment)
    seg_node_y = np.array([seg.y for seg in segments], dtype=np.int64)

    # --- Tube geometry for stems/roots ---
    pd = segs_to_polydata(ana, 1., ["radius", "organType", "creationTime"])

    # Add nodeIdY as cell data BEFORE tube filter so it propagates
    node_y_arr = vtk_data(seg_node_y.astype(np.float64))
    node_y_arr.SetName("nodeIdY")
    pd.GetCellData().AddArray(node_y_arr)

    # Apply tube filter (same as plot_roots)
    pd.GetPointData().SetActiveScalars("radius")
    tube_filter = vtk.vtkTubeFilter()
    tube_filter.SetInputData(pd)
    tube_filter.SetNumberOfSides(9)
    tube_filter.SetVaryRadiusToVaryRadiusByAbsoluteScalar()
    tube_filter.Update()
    tube_pd = tube_filter.GetOutput()

    # Extract the propagated nodeIdY from tube output cells
    tube_node_y_data = tube_pd.GetCellData().GetArray("nodeIdY")
    n_tube_cells = tube_pd.GetNumberOfCells()
    tube_cell_mapping = []
    for i in range(n_tube_cells):
        nid_y = int(tube_node_y_data.GetTuple1(i))
        tube_cell_mapping.append(nid_y)

    # Also extract organType per tube cell
    tube_organ_type_data = tube_pd.GetCellData().GetArray("organType")
    tube_organ_types = []
    if tube_organ_type_data:
        for i in range(n_tube_cells):
            tube_organ_types.append(int(tube_organ_type_data.GetTuple1(i)))

    # --- Leaf geometry ---
    leaf_points = vtk.vtkPoints()
    leaf_polys = vtk.vtkCellArray()
    globalIdx_y = []
    leaves = []
    if hasattr(plant, 'getOrgans'):
        leaves = plant.getOrgans(ot=pb.leaf)
        for l in leaves:
            globalIdx_y = globalIdx_y + create_leaf_(l, leaf_points, leaf_polys)
    globalIdx_y = np.array(globalIdx_y, dtype=np.int64)

    leaf_pd = vtk.vtkPolyData()
    leaf_pd.SetPoints(leaf_points)
    leaf_pd.SetPolys(leaf_polys)
    n_leaf_cells = leaf_pd.GetNumberOfCells()

    # --- Set up render scene for OBJ export ---
    colors = vtk.vtkNamedColors()
    ren = vtk.vtkRenderer()
    ren.SetBackground(1, 1, 1)

    # Tube actor
    tube_mapper = vtk.vtkPolyDataMapper()
    tube_mapper.SetInputData(tube_pd)
    tube_mapper.ScalarVisibilityOff()
    tube_actor = vtk.vtkActor()
    tube_actor.SetMapper(tube_mapper)
    tube_actor.GetProperty().SetColor(0.55, 0.27, 0.07)  # brown for stems/roots
    ren.AddActor(tube_actor)

    # Leaf actor
    if n_leaf_cells > 0:
        leaf_mapper = vtk.vtkPolyDataMapper()
        leaf_mapper.SetInputData(leaf_pd)
        leaf_mapper.ScalarVisibilityOff()
        leaf_actor = vtk.vtkActor()
        leaf_actor.SetMapper(leaf_mapper)
        leaf_actor.GetProperty().SetColor(0.13, 0.55, 0.13)  # green for leaves
        ren.AddActor(leaf_actor)

    ren_win = vtk.vtkRenderWindow()
    ren_win.SetOffScreenRendering(1)
    ren_win.AddRenderer(ren)
    ren_win.SetSize(1, 1)  # minimal — we only need the scene for export
    ren_win.Render()

    # --- Export OBJ ---
    out_dir = os.path.dirname(output_prefix)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)

    exporter = vtk.vtkOBJExporter()
    exporter.SetRenderWindow(ren_win)
    exporter.SetFilePrefix(output_prefix)
    exporter.Write()

    # --- Build the mapping JSON ---
    # The OBJ exporter writes actors in order. First actor = tubes, second = leaves.
    # Within each actor, faces correspond to VTK cells in order.
    mapping = {
        "metadata": {
            "description": "CPlantBox VTK-to-DART cell mapping",
            "sim_days": sim_days,
            "n_segments": len(segments),
            "n_nodes": len(nodes),
            "n_leaves": len(leaves),
            "n_tube_cells": n_tube_cells,
            "n_leaf_cells": n_leaf_cells,
            "total_obj_faces": n_tube_cells + n_leaf_cells,
        },
        "tubes": {
            "description": "Tube geometry cells (stems, roots). cell_index -> global node ID (segment end-node).",
            "face_offset": 0,
            "n_faces": n_tube_cells,
            "cell_to_nodeIdY": tube_cell_mapping,
        },
        "leaves": {
            "description": "Leaf quad cells. cell_index -> global node ID.",
            "face_offset": n_tube_cells,
            "n_faces": n_leaf_cells,
            "cell_to_nodeIdY": globalIdx_y.tolist(),
        },
        "node_coordinates": {
            "description": "Global node ID -> [x, y, z] in cm. Only nodes referenced by segments.",
            "nodes": {},
        },
        "segment_lookup": {
            "description": "Global nodeIdY -> {node_x, node_y, organ_type} for reverse lookup.",
            "segments": {},
        },
    }

    # Add organ type to tubes if available
    if tube_organ_types:
        mapping["tubes"]["cell_to_organType"] = tube_organ_types

    # Populate node coordinates for all referenced nodes
    referenced_node_ids = set(tube_cell_mapping) | set(globalIdx_y.tolist())
    for seg in segments:
        referenced_node_ids.add(seg.x)
        referenced_node_ids.add(seg.y)

    for nid in sorted(referenced_node_ids):
        if 0 <= nid < len(nodes):
            n = nodes[nid]
            mapping["node_coordinates"]["nodes"][str(nid)] = [n.x, n.y, n.z]

    # Populate segment lookup: nodeIdY -> segment info
    organ_types = np.array(ana.getParameter("organType"))
    for i, seg in enumerate(segments):
        ot = int(organ_types[i]) if i < len(organ_types) else -1
        mapping["segment_lookup"]["segments"][str(seg.y)] = {
            "node_x": seg.x,
            "node_y": seg.y,
            "organ_type": ot,
        }

    # Write JSON sidecar
    json_path = output_prefix + "_mapping.json"
    with open(json_path, "w") as f:
        json.dump(mapping, f, indent=2)

    summary = {
        "obj_file": output_prefix + ".obj",
        "mtl_file": output_prefix + ".mtl",
        "mapping_file": json_path,
        "n_tube_cells": n_tube_cells,
        "n_leaf_cells": n_leaf_cells,
        "n_segments": len(segments),
        "n_leaves": len(leaves),
    }

    print(f"[vtk_dart_export] Exported OBJ: {output_prefix}.obj")
    print(f"[vtk_dart_export] Mapping JSON: {json_path}")
    print(f"[vtk_dart_export]   Tube cells: {n_tube_cells}, Leaf cells: {n_leaf_cells}")
    print(f"[vtk_dart_export]   Segments: {len(segments)}, Leaves: {len(leaves)}")

    return summary
