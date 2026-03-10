"""Dash app factory — creates and wires the dashboard."""

from __future__ import annotations

import dash
import dash_bootstrap_components as dbc
from dash import dcc, html

from .pages import system, simulation, meteorology, runner, outputs, viewer3d


def create_app() -> dash.Dash:
    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.SANDSTONE],
        suppress_callback_exceptions=True,
    )

    app.layout = dbc.Container(
        [
            # Title row
            dbc.Row(
                dbc.Col(
                    html.H5("CPlantBox-DART Coupling Dashboard", className="dashboard-title my-3"),
                ),
            ),
            # Shared stores
            dcc.Store(id="pipeline-config-store", storage_type="session"),
            dcc.Store(id="validation-store"),
            # Tabs
            dcc.Tabs(
                id="main-tabs",
                value="tab-system",
                children=[
                    dcc.Tab(label="System Setup", value="tab-system"),
                    dcc.Tab(label="Simulation", value="tab-simulation"),
                    dcc.Tab(label="Meteorology", value="tab-meteo"),
                    dcc.Tab(label="Run", value="tab-runner"),
                    dcc.Tab(label="Results", value="tab-outputs"),
                    dcc.Tab(label="3D Viewer", value="tab-3d"),
                ],
            ),
            html.Div(id="tab-content", className="mt-3"),
            # Institutional logos — fixed bottom-left
            html.Div(
                className="logoContainer",
                children=[
                    html.A(
                        html.Img(src=app.get_asset_url("cplantbox.png"), className="logo"),
                        href="https://github.com/Plant-Root-Soil-Interactions-Modelling/CPlantBox",
                        target="_blank",
                    ),
                    html.A(
                        html.Img(src=app.get_asset_url("fzj.png"), className="logo"),
                        href="https://www.fz-juelich.de/de",
                        target="_blank",
                    ),
                    html.A(
                        html.Img(src=app.get_asset_url("logo_dart.png"), className="logo"),
                        href="https://dart.omp.eu/#/",
                        target="_blank",
                    ),
                    html.A(
                        html.Img(src=app.get_asset_url("agroc.png"), className="logo"),
                        href="https://agroc.io/",
                        target="_blank",
                    ),
                ],
            ),
        ],
        fluid=True,
    )

    # Tab routing
    @app.callback(
        dash.Output("tab-content", "children"),
        dash.Input("main-tabs", "value"),
    )
    def render_tab(tab):
        if tab == "tab-system":
            return system.layout()
        elif tab == "tab-simulation":
            return simulation.layout()
        elif tab == "tab-meteo":
            return meteorology.layout()
        elif tab == "tab-runner":
            return runner.layout()
        elif tab == "tab-outputs":
            return outputs.layout()
        elif tab == "tab-3d":
            return viewer3d.layout()
        return html.P("Select a tab.")

    # Register all page callbacks
    system.register_callbacks(app)
    simulation.register_callbacks(app)
    meteorology.register_callbacks(app)
    runner.register_callbacks(app)
    outputs.register_callbacks(app)
    viewer3d.register_callbacks(app)

    return app
