"""Simulation Config tab — mode, species, scene, physics, carbon, DART tuning."""

from __future__ import annotations

import json
from dataclasses import asdict

import dash_bootstrap_components as dbc
from dash import Input, Output, State, dcc, html

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
                                    dbc.Col([dbc.Label("Grid NX (cols)"), dbc.Input(id="sim-grid-nx", type="number", value=3, min=1, max=10)], width=2),
                                    dbc.Col([dbc.Label("Grid NY (rows)"), dbc.Input(id="sim-grid-ny", type="number", value=3, min=1, max=10)], width=2),
                                    dbc.Col([dbc.Label("Row spacing (m)"), dbc.Input(id="sim-spacing-x", type="number", value=0.75, step=0.05, min=0.1)], width=2),
                                    dbc.Col([dbc.Label("Plant spacing (m)"), dbc.Input(id="sim-spacing-y", type="number", value=0.25, step=0.05, min=0.05)], width=2),
                                ],
                                className="mb-2",
                            ),
                            dbc.Row(
                                [
                                    dbc.Col([dbc.Label("Scene size X (m)"), dbc.Input(id="sim-scene-x", type="number", value=4.0, step=0.5, min=1.0)], width=2),
                                    dbc.Col([dbc.Label("Scene size Y (m)"), dbc.Input(id="sim-scene-y", type="number", value=4.0, step=0.5, min=1.0)], width=2),
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
            # --- DART Tuning ---
            dbc.Card(
                [
                    dbc.CardHeader("DART Tuning"),
                    dbc.CardBody(
                        dbc.Row(
                            [
                                dbc.Col([dbc.Label("Threads"), dbc.Input(id="sim-threads", type="number", value=8, min=1, max=256)], width=2),
                                dbc.Col([dbc.Label("Ray density/pixel"), dbc.Input(id="sim-ray-density", type="number", value=500, min=10, max=5000)], width=3),
                                dbc.Col([dbc.Label("Max render time (0=inf)"), dbc.Input(id="sim-max-render", type="number", value=0, min=0)], width=3),
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


# All input IDs used in build_config (order matters for callback signature)
_BUILD_INPUTS = [
    "sim-mode", "sim-species", "sim-growth-days", "sim-single-day", "sim-timestep",
    "sim-lat", "sim-lon", "sim-sowing-date",
    "sim-grid-nx", "sim-grid-ny", "sim-spacing-x", "sim-spacing-y",
    "sim-scene-x", "sim-scene-y", "sim-soil-psi",
    "sim-physics-checks", "sim-gs-max-iter", "sim-gs-tol", "sim-gs-damping",
    "sim-carbon-checks", "sim-carbon-method",
    "sim-threads", "sim-ray-density", "sim-max-render",
]


def register_callbacks(app):
    @app.callback(
        Output("sim-single-day", "disabled"),
        Output("sim-gs-collapse", "is_open"),
        Input("sim-mode", "value"),
        Input("sim-physics-checks", "value"),
    )
    def toggle_fields(mode, physics):
        single_disabled = mode != "single_day"
        gs_open = "iterate_gs" in (physics or [])
        return single_disabled, gs_open

    @app.callback(
        Output("pipeline-config-store", "data"),
        Output("sim-config-preview", "value"),
        [Input(id_, "value") for id_ in _BUILD_INPUTS],
        State("pipeline-config-store", "data"),
    )
    def build_config(*args):
        # Unpack in same order as _BUILD_INPUTS + trailing state
        (mode, species, growth_days_str, single_day, timestep,
         lat, lon, sowing_date,
         grid_nx, grid_ny, spacing_x, spacing_y,
         scene_x, scene_y, soil_psi,
         physics, gs_max, gs_tol, gs_damp,
         carbon_checks, carbon_method,
         threads, ray_density, max_render,
         current_store) = args

        physics = physics or []
        carbon_checks = carbon_checks or []

        # Parse growth days
        try:
            growth_days = [int(d.strip()) for d in (growth_days_str or "").split(",") if d.strip()]
        except ValueError:
            growth_days = [10, 14, 18, 22, 26, 30, 35, 40, 45, 50, 55, 58]

        # Preserve met_csv / met_daily_csv from store (set by Meteorology tab)
        met_csv = (current_store or {}).get("met_csv")
        met_daily_csv = (current_store or {}).get("met_daily_csv")

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
            grid_ny=int(grid_ny or 3),
            grid_spacing_x=float(spacing_x or 0.75),
            grid_spacing_y=float(spacing_y or 0.25),
            scene_size_x=float(scene_x or 4.0),
            scene_size_y=float(scene_y or 4.0),
            soil_psi_cm=float(soil_psi if soil_psi is not None else -500.0),
            enable_baleno="baleno" in physics,
            iterate_gs="iterate_gs" in physics,
            gs_max_iterations=int(gs_max or 6),
            gs_tolerance=float(gs_tol or 0.05),
            gs_damping_alpha=float(gs_damp or 0.6),
            with_carbon="with_carbon" in carbon_checks,
            carbon_method=carbon_method or "auto",
            with_agroc="with_agroc" in carbon_checks,
            threads=int(threads or 8),
            dart_ray_density=int(ray_density or 500),
            dart_max_rendering_time=int(max_render or 0),
            resume="resume" in physics,
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
        Output("sim-carbon-checks", "value"),
        Output("sim-carbon-method", "value"),
        Output("sim-threads", "value"),
        Output("sim-ray-density", "value"),
        Output("sim-max-render", "value"),
        Output("sim-gs-max-iter", "value"),
        Output("sim-gs-tol", "value"),
        Output("sim-gs-damping", "value"),
        Input("sim-upload-config", "contents"),
        prevent_initial_call=True,
    )
    def load_config(contents):
        n_outputs = 24
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

        carbon = []
        if raw.get("with_carbon"):
            carbon.append("with_carbon")
        if raw.get("with_agroc"):
            carbon.append("with_agroc")

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
            raw.get("grid_ny", 3),
            raw.get("grid_spacing_x", 0.75),
            raw.get("grid_spacing_y", 0.25),
            raw.get("scene_size_x", 4.0),
            raw.get("scene_size_y", 4.0),
            raw.get("soil_psi_cm", -500.0),
            physics,
            carbon,
            raw.get("carbon_method", "auto"),
            raw.get("threads", 8),
            raw.get("dart_ray_density", 500),
            raw.get("dart_max_rendering_time", 0),
            raw.get("gs_max_iterations", 6),
            raw.get("gs_tolerance", 0.05),
            raw.get("gs_damping_alpha", 0.6),
        )
