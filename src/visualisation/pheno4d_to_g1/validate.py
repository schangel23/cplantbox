"""Validation metrics and VTK visualization for G1 pipelines."""

import numpy as np


def validate_g1(mapped_segments, organ_points, verbose=True):
    """Validate a MappedSegments object against the original point cloud.

    Args:
        mapped_segments: pb.MappedSegments
        organ_points: dict of organ_name -> np.array([N, 3]) original points
        verbose: print detailed report

    Returns:
        dict of validation metrics
    """
    nodes = mapped_segments.nodes
    segments = mapped_segments.segments
    organ_types = mapped_segments.organTypes
    radii = mapped_segments.radii

    node_coords = np.array([[n.x, n.y, n.z] for n in nodes])
    n_nodes = len(nodes)
    n_segs = len(segments)

    n_stem_segs = sum(1 for ot in organ_types if ot == 3)
    n_leaf_segs = sum(1 for ot in organ_types if ot == 4)

    # Topology check
    topology_ok = (n_segs == n_nodes - 1) or (n_segs < n_nodes)

    # Segment lengths
    seg_lengths = []
    for seg in segments:
        p0 = node_coords[seg.x]
        p1 = node_coords[seg.y]
        seg_lengths.append(np.linalg.norm(p1 - p0))
    seg_lengths = np.array(seg_lengths)

    # Stem height
    stem_node_mask = np.array([ot == 3 for ot in organ_types])
    if stem_node_mask.any():
        stem_seg_indices = np.where(stem_node_mask)[0]
        stem_node_ids = set()
        for idx in stem_seg_indices:
            seg = segments[idx]
            stem_node_ids.add(seg.x)
            stem_node_ids.add(seg.y)
        stem_z = [node_coords[i, 2] for i in stem_node_ids]
        stem_height = max(stem_z) - min(stem_z)
    else:
        stem_height = 0

    # Skeleton-to-cloud distances per organ
    from scipy.spatial import KDTree

    cloud_distances = {}
    all_cloud_pts = np.concatenate(list(organ_points.values()))
    all_node_tree = KDTree(node_coords)
    dists_all, _ = all_node_tree.query(all_cloud_pts)
    cloud_distances['all'] = {
        'mean': float(np.mean(dists_all)),
        'median': float(np.median(dists_all)),
        'max': float(np.max(dists_all)),
        'p95': float(np.percentile(dists_all, 95)),
    }

    # Per-organ distance (cloud point to nearest skeleton node)
    for name, pts in organ_points.items():
        dists, _ = all_node_tree.query(pts)
        cloud_distances[name] = {
            'mean': float(np.mean(dists)),
            'median': float(np.median(dists)),
            'max': float(np.max(dists)),
            'p95': float(np.percentile(dists, 95)),
        }

    # Leaf count from organ_points
    n_leaves_data = sum(1 for k in organ_points if k.startswith('leaf_'))
    # Leaf count from segments
    leaf_subtypes = set()
    for i, ot in enumerate(organ_types):
        if ot == 4:
            leaf_subtypes.add(mapped_segments.subTypes[i])
    n_leaves_g1 = len(leaf_subtypes)

    # Point cloud Z range
    if 'stem' in organ_points:
        cloud_stem_z = organ_points['stem'][:, 2]
        cloud_height = cloud_stem_z.max() - cloud_stem_z.min()
    else:
        cloud_height = all_cloud_pts[:, 2].max() - all_cloud_pts[:, 2].min()

    metrics = {
        'n_nodes': n_nodes,
        'n_segments': n_segs,
        'n_stem_segments': n_stem_segs,
        'n_leaf_segments': n_leaf_segs,
        'n_leaves_data': n_leaves_data,
        'n_leaves_g1': n_leaves_g1,
        'topology_ok': topology_ok,
        'stem_height_cm': stem_height,
        'cloud_height_cm': cloud_height,
        'height_error_cm': abs(stem_height - cloud_height),
        'seg_length_mean': float(seg_lengths.mean()),
        'seg_length_std': float(seg_lengths.std()),
        'radii_mean': float(np.mean(radii)),
        'cloud_distances': cloud_distances,
    }

    if verbose:
        print("\n" + "=" * 50)
        print("G1 Validation Report")
        print("=" * 50)
        print(f"  Nodes: {n_nodes}, Segments: {n_segs} "
              f"(stem: {n_stem_segs}, leaf: {n_leaf_segs})")
        print(f"  Topology OK: {topology_ok}")
        print(f"  Leaves: {n_leaves_g1} (data: {n_leaves_data})")
        print(f"  Stem height: {stem_height:.1f} cm "
              f"(cloud: {cloud_height:.1f} cm, error: {metrics['height_error_cm']:.1f} cm)")
        print(f"  Segment length: {seg_lengths.mean():.2f} +/- {seg_lengths.std():.2f} cm")
        print(f"  Mean radius: {np.mean(radii):.3f} cm")
        print(f"\n  Cloud-to-skeleton distances (cm):")
        for name, d in sorted(cloud_distances.items()):
            print(f"    {name}: mean={d['mean']:.2f}, "
                  f"median={d['median']:.2f}, p95={d['p95']:.2f}, max={d['max']:.2f}")

    return metrics


def visualize_skeleton(mapped_segments, organ_points=None, output_path=None):
    """Render skeleton overlaid on point cloud using VTK.

    Args:
        mapped_segments: pb.MappedSegments
        organ_points: optional dict of organ_name -> np.array for point cloud overlay
        output_path: if given, save screenshot (otherwise display interactively)
    """
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk

    nodes = mapped_segments.nodes
    segments = mapped_segments.segments
    organ_types = mapped_segments.organTypes

    node_coords = np.array([[n.x, n.y, n.z] for n in nodes])

    renderer = vtk.vtkRenderer()
    renderer.SetBackground(0.95, 0.95, 0.95)

    # --- Draw skeleton as colored lines ---
    colors_map = {3: (0.6, 0.3, 0.0), 4: (0.0, 0.7, 0.0)}  # stem=brown, leaf=green

    points_vtk = vtk.vtkPoints()
    points_vtk.SetData(numpy_to_vtk(node_coords))

    lines = vtk.vtkCellArray()
    line_colors = vtk.vtkUnsignedCharArray()
    line_colors.SetNumberOfComponents(3)
    line_colors.SetName("Colors")

    for i, seg in enumerate(segments):
        line = vtk.vtkLine()
        line.GetPointIds().SetId(0, seg.x)
        line.GetPointIds().SetId(1, seg.y)
        lines.InsertNextCell(line)
        ot = organ_types[i] if i < len(organ_types) else 3
        c = colors_map.get(ot, (0.5, 0.5, 0.5))
        line_colors.InsertNextTuple3(int(c[0]*255), int(c[1]*255), int(c[2]*255))

    skel_pd = vtk.vtkPolyData()
    skel_pd.SetPoints(points_vtk)
    skel_pd.SetLines(lines)
    skel_pd.GetCellData().SetScalars(line_colors)

    # Thicken skeleton lines with tubes
    tube = vtk.vtkTubeFilter()
    tube.SetInputData(skel_pd)
    tube.SetRadius(0.08)
    tube.SetNumberOfSides(6)
    tube.Update()

    skel_mapper = vtk.vtkPolyDataMapper()
    skel_mapper.SetInputData(tube.GetOutput())
    skel_actor = vtk.vtkActor()
    skel_actor.SetMapper(skel_mapper)
    renderer.AddActor(skel_actor)

    # --- Draw skeleton nodes as spheres ---
    sphere_src = vtk.vtkSphereSource()
    sphere_src.SetRadius(0.12)
    sphere_src.SetThetaResolution(8)
    sphere_src.SetPhiResolution(8)

    for i, pt in enumerate(node_coords):
        glyph = vtk.vtkGlyph3D()
        pts_single = vtk.vtkPoints()
        pts_single.InsertNextPoint(pt)
        pd_single = vtk.vtkPolyData()
        pd_single.SetPoints(pts_single)
        glyph.SetInputData(pd_single)
        glyph.SetSourceConnection(sphere_src.GetOutputPort())
        glyph.Update()

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(glyph.GetOutput())
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(1.0, 0.2, 0.2)
        renderer.AddActor(actor)

    # --- Draw point cloud if provided ---
    if organ_points:
        pc_colors = {
            'stem': (0.5, 0.8, 0.5, 0.15),
            'leaf_1': (0.3, 0.6, 0.9, 0.1),
            'leaf_2': (0.9, 0.6, 0.3, 0.1),
            'leaf_3': (0.6, 0.3, 0.9, 0.1),
        }
        for name, pts in organ_points.items():
            # Downsample for rendering
            step = max(1, len(pts) // 20000)
            pts_ds = pts[::step]

            vtk_pts = vtk.vtkPoints()
            vtk_pts.SetData(numpy_to_vtk(pts_ds.astype(np.float64)))
            pc_pd = vtk.vtkPolyData()
            pc_pd.SetPoints(vtk_pts)
            verts = vtk.vtkCellArray()
            for j in range(len(pts_ds)):
                verts.InsertNextCell(1)
                verts.InsertCellPoint(j)
            pc_pd.SetVerts(verts)

            pc_mapper = vtk.vtkPolyDataMapper()
            pc_mapper.SetInputData(pc_pd)
            pc_actor = vtk.vtkActor()
            pc_actor.SetMapper(pc_mapper)
            c = pc_colors.get(name, (0.5, 0.5, 0.5, 0.1))
            pc_actor.GetProperty().SetColor(c[0], c[1], c[2])
            pc_actor.GetProperty().SetOpacity(c[3])
            pc_actor.GetProperty().SetPointSize(1)
            renderer.AddActor(pc_actor)

    renderer.ResetCamera()
    cam = renderer.GetActiveCamera()
    cam.Elevation(15)
    cam.Azimuth(30)

    ren_win = vtk.vtkRenderWindow()
    ren_win.SetSize(1920, 1440)
    ren_win.AddRenderer(renderer)

    if output_path:
        ren_win.SetOffScreenRendering(1)
        ren_win.Render()
        w2i = vtk.vtkWindowToImageFilter()
        w2i.SetInput(ren_win)
        w2i.Update()
        writer = vtk.vtkPNGWriter()
        writer.SetFileName(output_path)
        writer.SetInputConnection(w2i.GetOutputPort())
        writer.Write()
        print(f"[validate] Screenshot saved: {output_path}")
    else:
        iren = vtk.vtkRenderWindowInteractor()
        iren.SetRenderWindow(ren_win)
        ren_win.Render()
        iren.Start()
