"""Simulation Config tab — mode, species, scene, physics, PROSPECT, carbon, DART tuning."""

from __future__ import annotations

import json
from dataclasses import asdict

import dash_bootstrap_components as dbc
from dash import Input, Output, State, dcc, html, dash_table

from ..state import config_from_store, config_to_store

_MODES = [
    {"label": "Full Production — parametric growth (3D DART)", "value": "full_production"},
    {"label": "Full Production — carbon-feedback growth (3D DART)", "value": "carbon_feedback"},
    {"label": "Uniform Baseline (no DART)", "value": "uniform_baseline"},
    {"label": "Single Day", "value": "single_day"},
]

_SPECIES = [
    {"label": "Maize (C4)", "value": "maize"},
    {"label": "Wheat (C3)", "value": "wheat"},
]

_CARBON_METHODS = [
    {"label": "Auto (phloem if available)", "value": "auto"},
    {"label": "Phloem (PiafMunch-style)", "value": "phloem"},
    {"label": "DVS (WOFOST-style)", "value": "dvs"},
]

# PROSPECT table columns — Stage/Days are read-only, parameters are editable
_PROSPECT_COLUMNS = [
    {"name": "Stage", "id": "label", "editable": False},
    {"name": "Days", "id": "days", "editable": False},
    {"name": "Cab", "id": "Cab", "type": "numeric"},
    {"name": "Car", "id": "Car", "type": "numeric"},
    {"name": "Cw", "id": "Cw", "type": "numeric"},
    {"name": "Cm", "id": "Cm", "type": "numeric"},
    {"name": "N", "id": "N", "type": "numeric"},
    {"name": "CBrown", "id": "CBrown", "type": "numeric"},
    {"name": "Anth", "id": "anthocyanin", "type": "numeric"},
]


def _stages_to_table_data(stages: list[dict]) -> list[dict]:
    """Convert prospect_params stage list to DataTable rows."""
    rows = []
    for s in stages:
        lo, hi = s["day_range"]
        rows.append({
            "label": s["label"],
            "days": f"{lo}–{hi}" if hi < 9999 else f"{lo}+",
            "day_from": lo,
            "day_to": hi,
            "Cab": s["Cab"],
            "Car": s["Car"],
            "Cw": s["Cw"],
            "Cm": s["Cm"],
            "N": s["N"],
            "CBrown": s["CBrown"],
            "anthocyanin": s["anthocyanin"],
        })
    return rows


def _get_default_table_data(species: str = "maize") -> list[dict]:
    """Get default PROSPECT table data for a species."""
    from dart.coupling.prospect_params import _PROSPECT_STAGES
    stages = _PROSPECT_STAGES.get(species, _PROSPECT_STAGES["maize"])
    return _stages_to_table_data(stages)


def _table_data_to_stages(rows: list[dict]) -> list[dict]:
    """Convert DataTable rows back to prospect_params stage format."""
    stages = []
    for r in rows:
        stages.append({
            "day_range": [r["day_from"], r["day_to"]],
            "label": r["label"],
            "Cab": float(r["Cab"]),
            "Car": float(r["Car"]),
            "Cw": float(r["Cw"]),
            "Cm": float(r["Cm"]),
            "N": float(r["N"]),
            "CBrown": float(r["CBrown"]),
            "anthocyanin": float(r["anthocyanin"]),
        })
    return stages


def layout() -> dbc.Container:
    return dbc.Container(
        [
            # --- Run Mode ---
            dbc.Card(
                [
                    dbc.CardHeader("Run Mode"),
                    dbc.CardBody(
                        [
                            dbc.Row(
                                [
                                    dbc.Col([dbc.Label("Mode"), dcc.Dropdown(id="sim-mode", options=_MODES, value="full_production")], width=5),
                                    dbc.Col([dbc.Label("Species"), dcc.Dropdown(id="sim-species", options=_SPECIES, value="maize")], width=3),
                                    dbc.Col([dbc.Label("Single Day"), dbc.Input(id="sim-single-day", type="number", value=55, disabled=True)], width=2),
                                ],
                                className="mb-2",
                            ),
                            dbc.Row(
                                [
                                    dbc.Col([dbc.Label("Growth Days (comma-separated)"), dbc.Input(id="sim-growth-days", type="text", value="10,14,18,22,26,30,35,40,45,50,55,58")], width=6),
                                    dbc.Col([dbc.Label("Timestep (min)"), dbc.Input(id="sim-timestep", type="number", value=60, min=10, max=240)], width=2),
                                ],
                            ),
                        ]
                    ),
                ]
            ),
            # --- Scene Geometry ---
            dbc.Card(
                [
                    dbc.CardHeader("Scene & Field Layout"),
                    dbc.CardBody(
                        [
                            dbc.Row(
                                [
                                    dbc.Col([dbc.Label("Latitude"), dbc.Input(id="sim-lat", type="number", value=50.92, step=0.01)], width=2),
                                    dbc.Col([dbc.Label("Longitude"), dbc.Input(id="sim-lon", type="number", value=6.36, step=0.01)], width=2),
                                    dbc.Col([dbc.Label("Sowing Date"), dbc.Input(id="sim-sowing-date", type="text", value="2025-05-01")], width=2),
                                ],
                                className="mb-2",
                            ),
                            dbc.Row(
                                [
                                    dbc.Col([dbc.Label("Rows (NX)"), dbc.Input(id="sim-grid-nx", type="number", value=3, min=1, max=10)], width=2),
                                    dbc.Col([dbc.Label("Plants/row (NY)"), dbc.Input(id="sim-grid-ny", type="number", value=5, min=1, max=15)], width=2),
                                    dbc.Col([dbc.Label("Row spacing (m)"), dbc.Input(id="sim-spacing-x", type="number", value=0.75, step=0.05, min=0.1)], width=2),
                                    dbc.Col([dbc.Label("Plant spacing (m)"), dbc.Input(id="sim-spacing-y", type="number", value=0.15, step=0.01, min=0.05)], width=2),
                                ],
                                className="mb-2",
                            ),
                            dbc.Row(
                                [
                                    dbc.Col([dbc.Label("Scene size X (m)"), dbc.Input(id="sim-scene-x", type="number", value=4.0, step=0.5, min=1.0)], width=2),
                                    dbc.Col([dbc.Label("Scene size Y (m)"), dbc.Input(id="sim-scene-y", type="number", value=2.25, step=0.25, min=1.0)], width=2),
                                ],
                            ),
                            html.P(
                                "Each plant is a unique realization (different random seed). All plants get individual "
                                "aPAR, Tleaf, and An. Scene must be large enough to contain the grid (grid extent + margin).",
                                className="text-muted small mt-2 mb-0",
                            ),
                        ]
                    ),
                ],
                className="mt-3",
            ),
            # --- Soil / Water ---
            dbc.Card(
                [
                    dbc.CardHeader("Soil & Water"),
                    dbc.CardBody(
                        dbc.Row(
                            [
                                dbc.Col([
                                    dbc.Label("Soil water potential (cm)"),
                                    dbc.Input(id="sim-soil-psi", type="number", value=-500.0, step=50, max=0),
                                ], width=3),
                                dbc.Col(
                                    html.P(
                                        "-500 = well-watered, -5000 = severe drought, -15000 = wilting point. "
                                        "Fixed uniform value (no 3D Richards solver in current pipeline).",
                                        className="text-muted small mt-4",
                                    ),
                                    width=6,
                                ),
                            ],
                        ),
                    ),
                ],
                className="mt-3",
            ),
            # --- Physics ---
            dbc.Card(
                [
                    dbc.CardHeader("Physics"),
                    dbc.CardBody(
                        [
                            dbc.Row(
                                [
                                    dbc.Col(dbc.Checklist(
                                        id="sim-physics-checks",
                                        options=[
                                            {"label": " Enable Baleno EB", "value": "baleno"},
                                            {"label": " Iterate gs (Tuzet)", "value": "iterate_gs"},
                                            {"label": " Resume from checkpoint", "value": "resume"},
                                        ],
                                        value=["baleno", "iterate_gs", "resume"],
                                        inline=True,
                                    ), width=12),
                                ],
                                className="mb-2",
                            ),
                            dbc.Collapse(
                                dbc.Row(
                                    [
                                        dbc.Col([dbc.Label("Max gs iterations"), dbc.Input(id="sim-gs-max-iter", type="number", value=6, min=1, max=20)], width=2),
                                        dbc.Col([dbc.Label("gs tolerance"), dbc.Input(id="sim-gs-tol", type="number", value=0.05, step=0.01, min=0.01, max=0.5)], width=2),
                                        dbc.Col([dbc.Label("gs damping alpha"), dbc.Input(id="sim-gs-damping", type="number", value=0.6, step=0.1, min=0.1, max=1.0)], width=2),
                                    ],
                                ),
                                id="sim-gs-collapse",
                                is_open=True,
                            ),
                        ]
                    ),
                ],
                className="mt-3",
            ),
            # --- PROSPECT Leaf Optics ---
            dbc.Card(
                [
                    dbc.CardHeader("PROSPECT Leaf Optics"),
                    dbc.CardBody(
                        [
                            dbc.Row(
                                dbc.Col(dbc.Checklist(
                                    id="sim-prospect-checks",
                                    options=[
                                        {"label": " Override PROSPECT (edit table below)", "value": "override"},
                                    ],
                                    value=[],
                                    inline=True,
                                ), width=12),
                                className="mb-2",
                            ),
                            dash_table.DataTable(
                                id="sim-prospect-table",
                                columns=_PROSPECT_COLUMNS,
                                data=_get_default_table_data("maize"),
                                editable=False,
                                style_table={"overflowX": "auto"},
                                style_cell={"textAlign": "center", "padding": "4px 8px", "fontSize": "13px"},
                                style_header={"fontWeight": "bold", "backgroundColor": "#f0f0f0"},
                                style_data_conditional=[
                                    {"if": {"column_id": "label"}, "fontWeight": "bold", "textAlign": "left"},
                                    {"if": {"column_id": "days"}, "color": "#666", "textAlign": "left"},
                                ],
                            ),
                            html.Hr(),
                            dbc.Row(
                                [
                                    dbc.Col([dbc.Label("Vcmax-Chl slope"), dbc.Input(id="sim-vcmax-chl1", type="number", value=0.64, step=0.01)], width=2),
                                    dbc.Col([dbc.Label("Vcmax-Chl intercept"), dbc.Input(id="sim-vcmax-chl2", type="number", value=4.165, step=0.1)], width=2),
                                    dbc.Col(
                                        html.Div(id="sim-prospect-vcmax-display", className="mt-4 fw-bold"),
                                        width=6,
                                    ),
                                ],
                            ),
                            html.P(
                                "Table shows per-growth-stage PROSPECT parameters. "
                                "Enable override to edit values. Vcmax = slope * Cab + intercept [umol/m2/s].",
                                className="text-muted small mt-2 mb-0",
                            ),
                        ]
                    ),
                ],
                className="mt-3",
            ),
            # --- Carbon ---
            dbc.Card(
                [
                    dbc.CardHeader("Carbon"),
                    dbc.CardBody(
                        dbc.Row(
                            [
                                dbc.Col(dbc.Checklist(
                                    id="sim-carbon-checks",
                                    options=[
                                        {"label": " With carbon partitioning", "value": "with_carbon"},
                                        {"label": " With AgroC", "value": "with_agroc"},
                                    ],
                                    value=["with_carbon"],
                                    inline=True,
                                ), width=6),
                                dbc.Col([dbc.Label("Method"), dcc.Dropdown(id="sim-carbon-method", options=_CARBON_METHODS, value="auto")], width=3),
                            ],
                        ),
                    ),
                ],
                className="mt-3",
            ),
            # --- SIF / Fluorescence ---
            dbc.Card(
                [
                    dbc.CardHeader("SIF / Fluorescence"),
                    dbc.CardBody(
                        [
                            dbc.Row(
                                dbc.Col(dbc.Checklist(
                                    id="sim-sif-checks",
                                    options=[
                                        {"label": " Enable SIF emission (per-segment eta)", "value": "with_sif"},
                                        {"label": " DART-F TOC radiance (Level 2)", "value": "with_dart_f"},
                                        {"label": " Per-triangle SIF CSVs (large files)", "value": "sif_triangles"},
                                    ],
                                    value=[],
                                    inline=True,
                                ), width=12),
                            ),
                            html.P(
                                "SIF requires iterate_gs (Tuzet) enabled. Level 2 (DART-F) adds ~106-band "
                                "fluorescence RT per timestep — significantly slower.",
                                className="text-muted small mt-2 mb-0",
                            ),
                        ]
                    ),
                ],
                className="mt-3",
            ),
            # --- DART Tuning ---
            dbc.Card(
                [
                    dbc.CardHeader("DART Tuning"),
                    dbc.CardBody(
                        dbc.Row(
                            [
                                dbc.Col([dbc.Label("Threads"), dbc.Input(id="sim-threads", type="number", value=8, min=1, max=256)], width=2),
                                dbc.Col([dbc.Label("Ray density/pixel"), dbc.Input(id="sim-ray-density", type="number", value=50, min=10, max=5000)], width=3),
                                dbc.Col([dbc.Label("Max render time (0=inf)"), dbc.Input(id="sim-max-render", type="number", value=0, min=0)], width=3),
                            ],
                        ),
                    ),
                ],
                className="mt-3",
            ),
            # --- I/O ---
            dbc.Card(
                [
                    dbc.CardHeader("I/O"),
                    dbc.CardBody(
                        dbc.Row(
                            [
                                dbc.Col([
                                    dbc.Label("Log file"),
                                    dbc.Input(id="sim-log-file", type="text", value="",
                                              placeholder=".dashboard_run.log (in output dir)"),
                                ], width=6),
                            ],
                        ),
                    ),
                ],
                className="mt-3",
            ),
            # --- Load / Preview ---
            dbc.Row(
                [
                    dbc.Col(
                        dcc.Upload(
                            dbc.Button("Load Config JSON", color="secondary", outline=True),
                            id="sim-upload-config",
                        ),
                        width="auto",
                    ),
                ],
                className="mt-3",
            ),
            dbc.Card(
                [
                    dbc.CardHeader("Config Preview"),
                    dbc.CardBody(
                        dcc.Textarea(
                            id="sim-config-preview",
                            className="config-preview",
                            style={"width": "100%", "height": "250px"},
                            readOnly=True,
                        ),
                    ),
                ],
                className="mt-3",
            ),
        ],
        fluid=True,
        className="py-3",
    )


# All scalar input IDs used in build_config (order matters for callback signature).
# The PROSPECT table data is a separate Input (not in this list).
_BUILD_INPUTS = [
    "sim-mode", "sim-species", "sim-growth-days", "sim-single-day", "sim-timestep",
    "sim-lat", "sim-lon", "sim-sowing-date",
    "sim-grid-nx", "sim-grid-ny", "sim-spacing-x", "sim-spacing-y",
    "sim-scene-x", "sim-scene-y", "sim-soil-psi",
    "sim-physics-checks", "sim-gs-max-iter", "sim-gs-tol", "sim-gs-damping",
    "sim-prospect-checks", "sim-vcmax-chl1", "sim-vcmax-chl2",
    "sim-carbon-checks", "sim-carbon-method",
    "sim-sif-checks",
    "sim-threads", "sim-ray-density", "sim-max-render",
    "sim-log-file",
]


def register_callbacks(app):
    @app.callback(
        Output("sim-single-day", "disabled"),
        Output("sim-gs-collapse", "is_open"),
        Output("sim-prospect-table", "editable"),
        Input("sim-mode", "value"),
        Input("sim-physics-checks", "value"),
        Input("sim-prospect-checks", "value"),
    )
    def toggle_fields(mode, physics, prospect_checks):
        single_disabled = mode != "single_day"
        gs_open = "iterate_gs" in (physics or [])
        prospect_editable = "override" in (prospect_checks or [])
        return single_disabled, gs_open, prospect_editable

    @app.callback(
        Output("sim-prospect-vcmax-display", "children"),
        Input("sim-prospect-table", "data"),
        Input("sim-vcmax-chl1", "value"),
        Input("sim-vcmax-chl2", "value"),
    )
    def update_vcmax_display(table_data, chl1, chl2):
        """Show Vcmax for each stage's Cab."""
        try:
            chl1 = float(chl1 or 0)
            chl2 = float(chl2 or 0)
            parts = []
            for row in (table_data or []):
                cab = float(row.get("Cab", 0))
                vcmax = chl1 * cab + chl2
                parts.append(f"{row['label']}: Cab={cab:.0f} -> Vcmax={vcmax:.1f}")
            return " | ".join(parts) + " [umol/m2/s]"
        except (TypeError, ValueError):
            return "—"

    @app.callback(
        Output("sim-prospect-table", "data", allow_duplicate=True),
        Output("sim-vcmax-chl1", "value", allow_duplicate=True),
        Output("sim-vcmax-chl2", "value", allow_duplicate=True),
        Input("sim-species", "value"),
        prevent_initial_call=True,
    )
    def update_prospect_defaults(species):
        """Reset PROSPECT table and Vcmax coefficients when species changes."""
        from dart.coupling.config import SPECIES_REGISTRY
        sp_name = (species or "maize").lower()
        sp = SPECIES_REGISTRY.get(sp_name, SPECIES_REGISTRY["maize"])
        return _get_default_table_data(sp_name), sp["vcmax_chl1"], sp["vcmax_chl2"]

    @app.callback(
        Output("pipeline-config-store", "data"),
        Output("sim-config-preview", "value"),
        [Input(id_, "value") for id_ in _BUILD_INPUTS] + [Input("sim-prospect-table", "data")],
        State("pipeline-config-store", "data"),
    )
    def build_config(*args):
        # Unpack: _BUILD_INPUTS values + table data + trailing state
        n_scalar = len(_BUILD_INPUTS)
        scalar_args = args[:n_scalar]
        prospect_table_data = args[n_scalar]
        current_store = args[n_scalar + 1]

        (mode, species, growth_days_str, single_day, timestep,
         lat, lon, sowing_date,
         grid_nx, grid_ny, spacing_x, spacing_y,
         scene_x, scene_y, soil_psi,
         physics, gs_max, gs_tol, gs_damp,
         prospect_checks, vcmax_chl1, vcmax_chl2,
         carbon_checks, carbon_method,
         sif_checks,
         threads, ray_density, max_render,
         log_file) = scalar_args

        physics = physics or []
        prospect_checks = prospect_checks or []
        carbon_checks = carbon_checks or []
        sif_checks = sif_checks or []

        # Parse growth days
        try:
            growth_days = [int(d.strip()) for d in (growth_days_str or "").split(",") if d.strip()]
        except ValueError:
            growth_days = [10, 14, 18, 22, 26, 30, 35, 40, 45, 50, 55, 58]

        # Preserve met_csv / met_daily_csv from store (set by Meteorology tab)
        met_csv = (current_store or {}).get("met_csv")
        met_daily_csv = (current_store or {}).get("met_daily_csv")

        # PROSPECT overrides — per-stage from table
        prospect_override = "override" in prospect_checks
        stages_override = None
        vcmax1_ov = None
        vcmax2_ov = None
        if prospect_override and prospect_table_data:
            stages_override = _table_data_to_stages(prospect_table_data)
            vcmax1_ov = float(vcmax_chl1) if vcmax_chl1 is not None else None
            vcmax2_ov = float(vcmax_chl2) if vcmax_chl2 is not None else None

        from dart.coupling.pipeline import PipelineConfig
        config = PipelineConfig(
            mode=mode or "full_production",
            species=species or "maize",
            growth_days=growth_days,
            single_day=int(single_day) if single_day else None,
            timestep_min=int(timestep or 60),
            lat=float(lat or 50.92),
            lon=float(lon or 6.36),
            sowing_date=sowing_date or "2025-05-01",
            grid_nx=int(grid_nx or 3),
            grid_ny=int(grid_ny or 5),
            grid_spacing_x=float(spacing_x or 0.75),
            grid_spacing_y=float(spacing_y or 0.15),
            scene_size_x=float(scene_x or 4.0),
            scene_size_y=float(scene_y or 4.0),
            soil_psi_cm=float(soil_psi if soil_psi is not None else -500.0),
            enable_baleno="baleno" in physics,
            iterate_gs="iterate_gs" in physics,
            gs_max_iterations=int(gs_max or 6),
            gs_tolerance=float(gs_tol or 0.05),
            gs_damping_alpha=float(gs_damp or 0.6),
            prospect_stages_override=stages_override,
            vcmax_chl1_override=vcmax1_ov,
            vcmax_chl2_override=vcmax2_ov,
            with_carbon="with_carbon" in carbon_checks,
            carbon_method=carbon_method or "auto",
            with_agroc="with_agroc" in carbon_checks,
            with_sif="with_sif" in sif_checks,
            with_dart_f="with_dart_f" in sif_checks,
            sif_triangles="sif_triangles" in sif_checks,
            threads=int(threads or 8),
            dart_ray_density=int(ray_density or 50),
            dart_max_rendering_time=int(max_render or 0),
            resume="resume" in physics,
            log_file=log_file or "",
            met_csv=met_csv,
            met_daily_csv=met_daily_csv,
        )
        store = config_to_store(config)
        preview = json.dumps(store, indent=2)
        return store, preview

    @app.callback(
        Output("sim-mode", "value"),
        Output("sim-species", "value"),
        Output("sim-growth-days", "value"),
        Output("sim-single-day", "value"),
        Output("sim-timestep", "value"),
        Output("sim-lat", "value"),
        Output("sim-lon", "value"),
        Output("sim-sowing-date", "value"),
        Output("sim-grid-nx", "value"),
        Output("sim-grid-ny", "value"),
        Output("sim-spacing-x", "value"),
        Output("sim-spacing-y", "value"),
        Output("sim-scene-x", "value"),
        Output("sim-scene-y", "value"),
        Output("sim-soil-psi", "value"),
        Output("sim-physics-checks", "value"),
        Output("sim-prospect-checks", "value"),
        Output("sim-prospect-table", "data"),
        Output("sim-vcmax-chl1", "value"),
        Output("sim-vcmax-chl2", "value"),
        Output("sim-carbon-checks", "value"),
        Output("sim-carbon-method", "value"),
        Output("sim-sif-checks", "value"),
        Output("sim-threads", "value"),
        Output("sim-ray-density", "value"),
        Output("sim-max-render", "value"),
        Output("sim-gs-max-iter", "value"),
        Output("sim-gs-tol", "value"),
        Output("sim-gs-damping", "value"),
        Output("sim-log-file", "value"),
        Input("sim-upload-config", "contents"),
        prevent_initial_call=True,
    )
    def load_config(contents):
        n_outputs = 30
        if not contents:
            from dash import no_update
            return (no_update,) * n_outputs
        import base64
        _, content_string = contents.split(",")
        raw = json.loads(base64.b64decode(content_string))

        physics = []
        if raw.get("enable_baleno", True):
            physics.append("baleno")
        if raw.get("iterate_gs", True):
            physics.append("iterate_gs")
        if raw.get("resume", True):
            physics.append("resume")

        # PROSPECT
        prospect_checks = []
        stages_ov = raw.get("prospect_stages_override")
        if stages_ov is not None:
            prospect_checks.append("override")
            # Convert saved stages to table data
            table_data = _stages_to_table_data(stages_ov)
        else:
            sp = (raw.get("species") or "maize").lower()
            table_data = _get_default_table_data(sp)

        carbon = []
        if raw.get("with_carbon"):
            carbon.append("with_carbon")
        if raw.get("with_agroc"):
            carbon.append("with_agroc")

        sif = []
        if raw.get("with_sif"):
            sif.append("with_sif")
        if raw.get("with_dart_f"):
            sif.append("with_dart_f")
        if raw.get("sif_triangles"):
            sif.append("sif_triangles")

        gd = raw.get("growth_days", [10, 14, 18, 22, 26, 30, 35, 40, 45, 50, 55, 58])

        return (
            raw.get("mode", "full_production"),
            raw.get("species", "maize"),
            ",".join(str(d) for d in gd),
            raw.get("single_day", 55),
            raw.get("timestep_min", 60),
            raw.get("lat", 50.92),
            raw.get("lon", 6.36),
            raw.get("sowing_date", "2025-05-01"),
            raw.get("grid_nx", 3),
            raw.get("grid_ny", 5),
            raw.get("grid_spacing_x", 0.75),
            raw.get("grid_spacing_y", 0.15),
            raw.get("scene_size_x", 4.0),
            raw.get("scene_size_y", 4.0),
            raw.get("soil_psi_cm", -500.0),
            physics,
            prospect_checks,
            table_data,
            raw.get("vcmax_chl1_override") or 0.64,
            raw.get("vcmax_chl2_override") or 4.165,
            carbon,
            raw.get("carbon_method", "auto"),
            sif,
            raw.get("threads", 8),
            raw.get("dart_ray_density", 50),
            raw.get("dart_max_rendering_time", 0),
            raw.get("gs_max_iterations", 6),
            raw.get("gs_tolerance", 0.05),
            raw.get("gs_damping_alpha", 0.6),
            raw.get("log_file", ""),
        )
