"""3D Plant Viewer tab — interactive Plotly visualization of G1 skeletons and G3 meshes."""

from __future__ import annotations

import json
from pathlib import Path

import dash_bootstrap_components as dbc
import numpy as np
from dash import Input, Output, State, dcc, html


def layout() -> dbc.Container:
    return dbc.Container(
        [
            dbc.Row(
                [
                    dbc.Col(
                        [dbc.Label("View"), dcc.Dropdown(
                            id="v3d-view",
                            options=[
                                {"label": "G3 Mesh (shoot)", "value": "g3_shoot"},
                                {"label": "G3 Mesh + Roots", "value": "g3_roots"},
                                {"label": "G1 Skeleton + Roots", "value": "g1_roots"},
                            ],
                            value="g3_shoot",
                        )],
                        width=2,
                    ),
                    dbc.Col(
                        [dbc.Label("Result Mode"), dcc.Dropdown(
                            id="v3d-mode",
                            options=[
                                {"label": "Diurnal (3D)", "value": "diurnal"},
                                {"label": "Uniform", "value": "diurnal_uniform"},
                            ],
                            value="diurnal",
                        )],
                        width=2,
                    ),
                    dbc.Col(
                        [dbc.Label("Day"), dcc.Dropdown(id="v3d-day", options=[], value=None)],
                        width=2,
                    ),
                    dbc.Col(
                        [dbc.Label("Plant"), dcc.Dropdown(id="v3d-plant", options=[], value=None)],
                        width=2,
                    ),
                    dbc.Col(
                        [dbc.Label("Color by"), dcc.Dropdown(
                            id="v3d-color",
                            options=[
                                {"label": "Organ Type", "value": "organ_type"},
                                {"label": "Organ ID", "value": "organ_id"},
                                {"label": "Segment Index", "value": "segment_idx"},
                                {"label": "APAR (umol/m2/s)", "value": "APAR_umol_m2_s"},
                                {"label": "Tleaf (C)", "value": "Tleaf_C"},
                                {"label": "An (umol CO2/m2/s)", "value": "An_umol_CO2_m2_s"},
                            ],
                            value="organ_type",
                        )],
                        width=2,
                    ),
                ],
                className="mb-2",
            ),
            dbc.Row(
                dbc.Col(
                    dbc.Button("Load", id="v3d-load-btn", color="primary"),
                    width="auto",
                ),
                className="mb-3",
            ),
            dbc.Alert(id="v3d-alert", is_open=False),
            dcc.Loading(
                html.Div(id="v3d-plot-container"),
                type="circle",
            ),
        ],
        fluid=True,
        className="py-3",
    )


# ---------------------------------------------------------------------------
# Organ-type color palette (shared across all view modes)
# ---------------------------------------------------------------------------

_ORGAN_COLORS = {
    "stem": "#8B4513",   # saddle brown
    "leaf": "#228B22",   # forest green
    "root": "#D2691E",   # chocolate
}
_ORGAN_TYPE_INT = {"stem": 0, "leaf": 1, "root": 2}


# ---------------------------------------------------------------------------
# OBJ parser (for G3 from file)
# ---------------------------------------------------------------------------

def _parse_obj(filepath: Path):
    """Parse OBJ file -> vertices (N,3), faces (K,3), face_groups (K,) int."""
    verts = []
    faces = []
    face_groups = []
    current_group = 0
    group_names = {}

    with open(filepath) as f:
        for line in f:
            if line.startswith("v "):
                parts = line.split()
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith("g "):
                gname = line.strip().split(None, 1)[1]
                if gname not in group_names:
                    group_names[gname] = len(group_names)
                current_group = group_names[gname]
            elif line.startswith("f "):
                parts = line.split()[1:]
                idxs = [int(p.split("/")[0]) - 1 for p in parts]
                for i in range(1, len(idxs) - 1):
                    faces.append([idxs[0], idxs[i], idxs[i + 1]])
                    face_groups.append(current_group)

    vertices = np.array(verts, dtype=np.float64)
    indices = np.array(faces, dtype=np.int32)
    groups = np.array(face_groups, dtype=np.int32)
    return vertices, indices, groups, group_names


# ---------------------------------------------------------------------------
# Face color builder (for G3 views)
# ---------------------------------------------------------------------------

def _build_face_colors(mapping_path, baleno_path, n_faces, color_by):
    """Build per-face scalar array for coloring.

    Returns (values, colorscale, title) or None.
    """
    if color_by in ("organ_type", "organ_id", "segment_idx") and mapping_path and mapping_path.exists():
        mapping = json.loads(mapping_path.read_text())
        values = np.full(n_faces, -1, dtype=np.float64)

        for organ in mapping.get("organs", []):
            oid = organ["organ_id"]
            otype = _ORGAN_TYPE_INT.get(organ["type"], 3)
            for seg in organ.get("segments", []):
                for tri_idx in seg.get("triangle_indices", []):
                    if tri_idx < n_faces:
                        if color_by == "organ_type":
                            values[tri_idx] = otype
                        elif color_by == "organ_id":
                            values[tri_idx] = oid
                        elif color_by == "segment_idx":
                            values[tri_idx] = seg["segment_idx"]

        if color_by == "organ_type":
            return values, "Viridis", "Organ Type"
        elif color_by == "organ_id":
            return values, "Turbo", "Organ ID"
        else:
            return values, "Viridis", "Segment"

    if color_by in ("APAR_umol_m2_s", "Tleaf_C", "An_umol_CO2_m2_s"):
        if not baleno_path or not baleno_path.exists():
            return None
        if not mapping_path or not mapping_path.exists():
            return None

        import pandas as pd
        df = pd.read_csv(baleno_path)
        mapping = json.loads(mapping_path.read_text())

        seg_values = {}
        for _, row in df.iterrows():
            key = (str(row["organ"]), int(row["segment_idx"]))
            seg_values[key] = float(row[color_by])

        values = np.full(n_faces, np.nan, dtype=np.float64)
        for organ in mapping.get("organs", []):
            organ_name = organ.get("name", f"{organ['type']}_{organ['organ_id']}")
            for seg in organ.get("segments", []):
                key = (organ_name, seg["segment_idx"])
                if key not in seg_values:
                    key = (f"{organ['type']}_{organ['organ_id']}", seg["segment_idx"])
                val = seg_values.get(key, np.nan)
                for tri_idx in seg.get("triangle_indices", []):
                    if tri_idx < n_faces:
                        values[tri_idx] = val

        titles = {
            "APAR_umol_m2_s": "aPAR (umol/m2/s)",
            "Tleaf_C": "Tleaf (C)",
            "An_umol_CO2_m2_s": "An (umol CO2/m2/s)",
        }
        return values, "RdYlGn", titles.get(color_by, color_by)

    return None


# ---------------------------------------------------------------------------
# Live plant growth (for G1 / G3+roots views)
# ---------------------------------------------------------------------------

def _grow_plant_for_view(day: int, seed: int = 42):
    """Grow a CPlantBox plant and return it. Cached per (day, seed)."""
    from dart.coupling.growth.grow import grow_plant
    from dart.coupling.config import DEFAULT_XML
    plant = grow_plant(
        xml_path=str(DEFAULT_XML),
        simulation_time=day,
        min_stem_nodes=50,
        min_leaf_nodes=20,
        enable_photosynthesis=True,
        seed=seed,
    )
    return plant


def _extract_g1_data(plant):
    """Extract G1 skeleton as lists of polylines, one per organ.

    Returns list of dicts:
        {type, organ_id, nodes (N,3), radii (N,)}
    """
    import plantbox as pb

    result = []
    organ_counter = 0

    for otype, type_name in [(pb.stem, "stem"), (pb.leaf, "leaf"), (pb.root, "root")]:
        for organ in plant.getOrgans(otype):
            nodes = organ.getNodes()
            if len(nodes) < 2:
                continue
            pts = np.array([[n.x, n.y, n.z] for n in nodes])
            radius = organ.getParameter("a")
            radii = np.full(len(nodes), radius)
            result.append({
                "type": type_name,
                "organ_id": organ_counter,
                "nodes": pts,
                "radii": radii,
                "subtype": int(organ.getParameter("subType")),
            })
            organ_counter += 1

    return result


def _extract_g3_with_roots(plant):
    """Loft G3 mesh including root geometry. Returns G3Mesh."""
    from dart.coupling.growth.grow import extract_g3_mesh
    mesh, organ_dicts = extract_g3_mesh(plant, include_roots=True)
    return mesh, organ_dicts


# ---------------------------------------------------------------------------
# Plotly figure builders
# ---------------------------------------------------------------------------

def _scene_layout(all_points):
    """Common 3D scene layout."""
    return dict(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        zaxis=dict(visible=False),
        aspectmode="data",
        camera=dict(
            eye=dict(x=1.5, y=1.5, z=0.8),
            center=dict(x=0, y=0, z=0),
            up=dict(x=0, y=0, z=1),
        ),
    )


def _build_g3_figure(vertices, indices, color_data, color_by, title):
    """Build a Plotly figure for a G3 triangle mesh."""
    import plotly.graph_objects as go

    fig = go.Figure()

    if color_data is not None:
        values, colorscale, cbar_title = color_data
        if color_by == "organ_type":
            int_to_color = {
                _ORGAN_TYPE_INT["stem"]: _ORGAN_COLORS["stem"],
                _ORGAN_TYPE_INT["leaf"]: _ORGAN_COLORS["leaf"],
                _ORGAN_TYPE_INT["root"]: _ORGAN_COLORS["root"],
            }
            facecolor = [int_to_color.get(int(v), "#888888") for v in values]
            fig.add_trace(go.Mesh3d(
                x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
                i=indices[:, 0], j=indices[:, 1], k=indices[:, 2],
                facecolor=facecolor,
                flatshading=True,
                lighting=dict(ambient=0.4, diffuse=0.6, specular=0.2),
                lightposition=dict(x=100, y=200, z=300),
                hoverinfo="skip",
            ))
        else:
            fig.add_trace(go.Mesh3d(
                x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
                i=indices[:, 0], j=indices[:, 1], k=indices[:, 2],
                intensity=values,
                intensitymode="cell",
                colorscale=colorscale,
                colorbar=dict(title=cbar_title),
                flatshading=True,
                lighting=dict(ambient=0.4, diffuse=0.6, specular=0.2),
                lightposition=dict(x=100, y=200, z=300),
                hoverinfo="skip",
            ))
    else:
        fig.add_trace(go.Mesh3d(
            x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
            i=indices[:, 0], j=indices[:, 1], k=indices[:, 2],
            color="#228B22",
            flatshading=True,
            lighting=dict(ambient=0.4, diffuse=0.6, specular=0.2),
            lightposition=dict(x=100, y=200, z=300),
            hoverinfo="skip",
        ))

    fig.update_layout(
        scene=_scene_layout(vertices),
        margin=dict(l=0, r=0, t=30, b=0),
        height=700,
        title=title,
    )
    return fig


def _build_g1_figure(g1_organs, title):
    """Build a Plotly figure for G1 skeleton (lines per organ, colored by type)."""
    import plotly.graph_objects as go

    fig = go.Figure()

    # Group organs by type for legend
    type_added = set()

    for organ in g1_organs:
        otype = organ["type"]
        color = _ORGAN_COLORS.get(otype, "#888888")
        pts = organ["nodes"]
        show_legend = otype not in type_added
        type_added.add(otype)

        # Line width based on mean radius (scaled for visibility)
        mean_r = float(organ["radii"].mean())
        lw = max(1.5, min(8.0, mean_r * 10))

        fig.add_trace(go.Scatter3d(
            x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
            mode="lines",
            line=dict(color=color, width=lw),
            name=otype if show_legend else None,
            legendgroup=otype,
            showlegend=show_legend,
            hovertext=f"{organ['type']} id={organ['organ_id']} st={organ['subtype']}",
            hoverinfo="text",
        ))

    # Collect all points for scene bounds
    all_pts = np.concatenate([o["nodes"] for o in g1_organs]) if g1_organs else np.zeros((1, 3))

    fig.update_layout(
        scene=_scene_layout(all_pts),
        margin=dict(l=0, r=0, t=30, b=0),
        height=700,
        title=title,
        legend=dict(itemsizing="constant"),
    )
    return fig


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def _find_days(output_dir: Path, subdir: str) -> list[int]:
    d = output_dir / subdir
    if not d.exists():
        return []
    days = []
    for p in sorted(d.iterdir()):
        if p.is_dir() and p.name.startswith("day"):
            try:
                days.append(int(p.name[3:]))
            except ValueError:
                pass
    return days


def _find_plants(output_dir: Path, subdir: str, day: int) -> list[str]:
    d = output_dir / subdir / f"day{day}"
    if not d.exists():
        return []
    plants = []
    for p in sorted(d.glob("*.obj")):
        if "_dart" in p.name:
            continue
        plants.append(p.stem)
    return plants


def _find_obj_and_mapping(output_dir: Path, subdir: str, day: int, plant: str | None):
    day_dir = output_dir / subdir / f"day{day}"

    if plant:
        obj_path = day_dir / f"{plant}.obj"
        mapping_path = day_dir / f"{plant}_mapping.json"
    else:
        objs = [p for p in day_dir.glob("*.obj") if "_dart" not in p.name]
        if not objs:
            return None, None, None
        obj_path = objs[0]
        mapping_path = day_dir / f"{obj_path.stem}_mapping.json"

    baleno_path = None
    for candidate in [
        day_dir / f"{obj_path.stem}_baleno_segments.csv",
        day_dir / "baleno_segments.csv",
        output_dir / f"multifield_day{day}_baleno_segments.csv",
        output_dir / f"maize_day{day}_baleno_segments.csv",
    ]:
        if candidate.exists():
            baleno_path = candidate
            break

    return obj_path, mapping_path, baleno_path


def _seed_from_plant_name(plant_name: str | None) -> int:
    """Extract seed from plant name like 'maize_day30_p3' -> 42+3. Default 42."""
    if plant_name:
        parts = plant_name.split("_")
        for p in parts:
            if p.startswith("p") and p[1:].isdigit():
                return 42 + int(p[1:])
    return 42


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

def register_callbacks(app):
    @app.callback(
        Output("v3d-day", "options"),
        Output("v3d-day", "value"),
        Input("v3d-mode", "value"),
        State("pipeline-config-store", "data"),
    )
    def scan_days(mode, store):
        from dart.coupling.config import OUTPUT_DIR
        output_dir = Path(store.get("output_dir", str(OUTPUT_DIR))) if store else OUTPUT_DIR
        days = _find_days(output_dir, mode)
        options = [{"label": f"Day {d}", "value": d} for d in days]
        value = days[-1] if days else None
        return options, value

    @app.callback(
        Output("v3d-plant", "options"),
        Output("v3d-plant", "value"),
        Input("v3d-day", "value"),
        State("v3d-mode", "value"),
        State("pipeline-config-store", "data"),
    )
    def scan_plants(day, mode, store):
        if day is None:
            return [], None
        from dart.coupling.config import OUTPUT_DIR
        output_dir = Path(store.get("output_dir", str(OUTPUT_DIR))) if store else OUTPUT_DIR
        plants = _find_plants(output_dir, mode, day)
        options = [{"label": p, "value": p} for p in plants]
        value = plants[0] if plants else None
        return options, value

    @app.callback(
        Output("v3d-plot-container", "children"),
        Output("v3d-alert", "children"),
        Output("v3d-alert", "color"),
        Output("v3d-alert", "is_open"),
        Input("v3d-load-btn", "n_clicks"),
        State("v3d-view", "value"),
        State("v3d-day", "value"),
        State("v3d-plant", "value"),
        State("v3d-mode", "value"),
        State("v3d-color", "value"),
        State("pipeline-config-store", "data"),
        prevent_initial_call=True,
    )
    def load_view(n_clicks, view, day, plant, mode, color_by, store):
        import plotly.graph_objects as go

        if day is None:
            return html.P("Select a day."), "No day selected.", "warning", True

        from dart.coupling.config import OUTPUT_DIR
        output_dir = Path(store.get("output_dir", str(OUTPUT_DIR))) if store else OUTPUT_DIR

        try:
            if view == "g3_shoot":
                return _load_g3_from_file(output_dir, mode, day, plant, color_by)
            elif view == "g3_roots":
                return _load_g3_with_roots(day, plant, color_by)
            elif view == "g1_roots":
                return _load_g1_skeleton(day, plant)
            else:
                return html.P("Unknown view."), "", "secondary", False
        except Exception as e:
            import traceback
            return html.P(f"Error: {e}"), traceback.format_exc(), "danger", True


def _load_g3_from_file(output_dir, mode, day, plant, color_by):
    """Load G3 mesh from existing OBJ file (shoot only)."""
    obj_path, mapping_path, baleno_path = _find_obj_and_mapping(
        output_dir, mode, day, plant,
    )
    if not obj_path or not obj_path.exists():
        return html.P("No mesh found."), f"OBJ not found for day {day}", "danger", True

    vertices, indices, groups, group_names = _parse_obj(obj_path)
    n_faces = len(indices)
    color_data = _build_face_colors(mapping_path, baleno_path, n_faces, color_by)

    title = f"{obj_path.stem} -- {color_by} ({n_faces:,} triangles)"
    fig = _build_g3_figure(vertices, indices, color_data, color_by, title)

    info = f"Loaded {obj_path.name}: {len(vertices):,} vertices, {n_faces:,} faces"
    return dcc.Graph(figure=fig, style={"height": "700px"}), info, "success", True


def _load_g3_with_roots(day, plant_name, color_by):
    """Grow plant live and loft G3 mesh with roots included."""
    seed = _seed_from_plant_name(plant_name)
    plant = _grow_plant_for_view(day, seed=seed)
    mesh, organ_dicts = _extract_g3_with_roots(plant)

    vertices = mesh.vertices
    indices = mesh.indices
    n_faces = len(indices)

    # Build organ-type face colors from mesh metadata
    color_data = None
    if color_by == "organ_type":
        values = np.full(n_faces, -1, dtype=np.float64)
        for tri_idx in range(n_faces):
            oid = int(mesh.organ_ids[tri_idx])
            # Find organ type from organ_meta
            for meta in mesh.organ_meta:
                if meta["organ_id"] == oid:
                    values[tri_idx] = _ORGAN_TYPE_INT.get(meta["type"], 3)
                    break
        color_data = (values, "Viridis", "Organ Type")
    elif color_by == "organ_id":
        values = np.array(mesh.organ_ids, dtype=np.float64)
        color_data = (values, "Turbo", "Organ ID")
    elif color_by == "segment_idx":
        values = np.array(mesh.segment_ids, dtype=np.float64)
        color_data = (values, "Viridis", "Segment")

    n_root_tris = sum(
        1 for i in range(n_faces)
        if any(m["organ_id"] == int(mesh.organ_ids[i]) and m["type"] == "root"
               for m in mesh.organ_meta)
    )
    n_shoot_tris = n_faces - n_root_tris

    title = f"Day {day} (seed {seed}) -- G3+Roots -- {color_by} ({n_faces:,} tris)"
    fig = _build_g3_figure(vertices, indices, color_data, color_by, title)

    info = (f"Live G3+Roots: {len(vertices):,} verts, {n_faces:,} faces "
            f"(shoot: {n_shoot_tris:,}, root: {n_root_tris:,})")
    return dcc.Graph(figure=fig, style={"height": "700px"}), info, "success", True


def _load_g1_skeleton(day, plant_name):
    """Grow plant live and render G1 skeleton with roots."""
    seed = _seed_from_plant_name(plant_name)
    plant = _grow_plant_for_view(day, seed=seed)
    g1_organs = _extract_g1_data(plant)

    n_stems = sum(1 for o in g1_organs if o["type"] == "stem")
    n_leaves = sum(1 for o in g1_organs if o["type"] == "leaf")
    n_roots = sum(1 for o in g1_organs if o["type"] == "root")
    n_total_pts = sum(len(o["nodes"]) for o in g1_organs)

    title = f"Day {day} (seed {seed}) -- G1 Skeleton ({n_stems}S {n_leaves}L {n_roots}R)"
    fig = _build_g1_figure(g1_organs, title)

    info = (f"G1 Skeleton: {n_stems} stems, {n_leaves} leaves, {n_roots} roots, "
            f"{n_total_pts:,} nodes")
    return dcc.Graph(figure=fig, style={"height": "700px"}), info, "success", True
