"""Dash app factory — creates and wires the dashboard."""

from __future__ import annotations

import dash
import dash_bootstrap_components as dbc
from dash import dcc, html

from .pages import system, simulation, meteorology, runner, outputs


def create_app() -> dash.Dash:
    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.FLATLY],
        suppress_callback_exceptions=True,
    )

    app.layout = dbc.Container(
        [
            # Title row
            dbc.Row(
                dbc.Col(
                    html.H3("CPlantBox-DART Coupling Dashboard", className="dashboard-title my-3"),
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
                ],
            ),
            html.Div(id="tab-content", className="mt-3"),
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
        return html.P("Select a tab.")

    # Register all page callbacks
    system.register_callbacks(app)
    simulation.register_callbacks(app)
    meteorology.register_callbacks(app)
    runner.register_callbacks(app)
    outputs.register_callbacks(app)

    return app
