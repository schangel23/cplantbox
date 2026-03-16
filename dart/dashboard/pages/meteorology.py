"""Meteorology CSV tab — default Jülich data, upload, preview, plot."""

from __future__ import annotations

import base64
import io
from pathlib import Path

import dash_bootstrap_components as dbc
from dash import Input, Output, State, dash_table, dcc, html

HOURLY_REQUIRED = {"datetime_utc", "T_air_C", "RH", "wind_ms"}
HOURLY_OPTIONAL = {"ea_hPa", "PAR_umol"}
DAILY_REQUIRED = {"sim_day", "T_min_C", "T_max_C", "T_mean_C"}
DAILY_OPTIONAL = {"date", "RH_min", "RH_max", "RH_mean", "wind_mean_kmh", "wind_max_kmh"}

_DEFAULT_DAILY_MET = Path(__file__).resolve().parents[2] / "coupling" / "data" / "juelich_2024_daily_met.csv"


def layout() -> dbc.Container:
    return dbc.Container(
        [
            dbc.Card(
                [
                    dbc.CardHeader("Current Weather Source"),
                    dbc.CardBody(
                        [
                            dbc.Badge("Default: Jülich 2024", id="met-source-badge",
                                      color="info", className="mb-2", style={"fontSize": "0.9rem"}),
                            html.P(id="met-source-text", className="text-muted mb-0"),
                        ]
                    ),
                ]
            ),
            dbc.Card(
                [
                    dbc.CardHeader("Upload Custom Met CSV"),
                    dbc.CardBody(
                        [
                            dcc.Upload(
                                id="met-upload",
                                children=html.Div([
                                    "Drag & drop or ",
                                    html.A("click to select", style={"fontWeight": "bold"}),
                                    " a CSV file",
                                ]),
                                className="upload-zone",
                                multiple=False,
                            ),
                            dbc.Alert(id="met-alert", is_open=False, className="mt-2"),
                        ]
                    ),
                ],
                className="mt-3",
            ),
            dbc.Card(
                [
                    dbc.CardHeader("FLUXNET Hourly Data"),
                    dbc.CardBody(
                        [
                            html.P(
                                "Point to a FLUXNET FULLSET hourly CSV on disk. "
                                "The pipeline will extract the selected year and convert "
                                "to its internal format automatically.",
                                className="text-muted small mb-2",
                            ),
                            dbc.Row(
                                [
                                    dbc.Col([
                                        dbc.Label("FLUXNET CSV path (on server)"),
                                        dbc.Input(id="met-fluxnet-csv", type="text",
                                                  placeholder="/media/data/Lukas/CPlantBox/dart/coupling/data/AMF_US-Ne1_...HR...csv",
                                                  value=""),
                                    ], width=8),
                                    dbc.Col([
                                        dbc.Label("Year"),
                                        dbc.Input(id="met-fluxnet-year", type="number",
                                                  value=2002, min=1990, max=2030),
                                    ], width=2),
                                    dbc.Col([
                                        dbc.Label("\u00a0"),  # spacer
                                        dbc.Button("Apply", id="met-fluxnet-apply",
                                                   color="primary", className="d-block"),
                                    ], width=2),
                                ],
                                className="mb-2",
                            ),
                            dbc.Alert(id="met-fluxnet-alert", is_open=False, className="mt-2"),
                        ]
                    ),
                ],
                className="mt-3",
            ),
            dbc.Card(
                [
                    dbc.CardHeader("Accepted Formats"),
                    dbc.CardBody(
                        [
                            html.H6("Option 1: Hourly (drives diurnal loop directly)"),
                            html.Pre(
                                "datetime_utc, T_air_C, RH, wind_ms [, ea_hPa, PAR_umol]\n"
                                "2024-06-25 06:00, 20.3, 0.72, 1.5\n"
                                "2024-06-25 07:00, 22.1, 0.65, 1.8\n"
                                "...",
                                className="config-preview",
                                style={"fontSize": "0.8rem", "background": "#f8f9fa", "padding": "0.5rem"},
                            ),
                            html.P([
                                html.Strong("Required: "), "datetime_utc, T_air_C, RH (0-1), wind_ms. ",
                                html.Strong("Optional: "), "ea_hPa, PAR_umol. ",
                                "Should span sunrise-sunset for each simulation day. "
                                "Daily stats are auto-derived for GDD calculation.",
                            ], className="text-muted small"),
                            html.Hr(),
                            html.H6("Option 2: Daily (generates sinusoidal hourly profiles)"),
                            html.Pre(
                                "sim_day, date, T_min_C, T_max_C, T_mean_C [, RH_min, RH_max, wind_mean_kmh, wind_max_kmh]\n"
                                "1, 2024-05-01, 12.2, 24.8, 18.0, 60, 94, 10.1, 15.3\n"
                                "2, 2024-05-02, 12.4, 23.3, 16.8, 56, 93, 11.3, 24.8\n"
                                "...",
                                className="config-preview",
                                style={"fontSize": "0.8rem", "background": "#f8f9fa", "padding": "0.5rem"},
                            ),
                            html.P([
                                html.Strong("Required: "), "sim_day, T_min_C, T_max_C, T_mean_C. ",
                                html.Strong("Optional: "), "date, RH_min/max, wind_mean/max_kmh. ",
                                "sim_day = days since sowing (day 1 = sowing). "
                                "T_min/T_max drive sinusoidal hourly shape; T_mean drives GDD for development stage.",
                            ], className="text-muted small"),
                            html.Hr(),
                            html.P([
                                "The default Jülich 2024 file is ", html.Strong("daily format"),
                                " (153 days, May-Sep). Both the hourly diurnal loop and the "
                                "DVS carbon partitioning use the same weather source.",
                            ], className="small"),
                        ]
                    ),
                ],
                className="mt-3",
            ),
            dbc.Card(
                [
                    dbc.CardHeader("Preview (first 20 rows)"),
                    dbc.CardBody(
                        dash_table.DataTable(
                            id="met-table",
                            page_size=20,
                            style_table={"overflowX": "auto"},
                            style_cell={"fontSize": "0.85rem"},
                        ),
                    ),
                ],
                className="mt-3",
            ),
            dbc.Card(
                [
                    dbc.CardHeader("Temperature"),
                    dbc.CardBody(dcc.Graph(id="met-plot", style={"height": "350px"})),
                ],
                className="mt-3",
            ),
        ],
        fluid=True,
        className="py-3",
    )


def _load_default_preview():
    """Load the default Jülich daily met CSV for preview."""
    import pandas as pd
    import plotly.graph_objects as go

    if not _DEFAULT_DAILY_MET.exists():
        return [], [], go.Figure()

    df = pd.read_csv(_DEFAULT_DAILY_MET)
    table_data = df.head(20).to_dict("records")
    columns = [{"name": c, "id": c} for c in df.columns]

    fig = go.Figure()
    x = df["date"] if "date" in df.columns else df["sim_day"]
    fig.add_trace(go.Scatter(x=x, y=df["T_max_C"], name="T_max", line=dict(color="#e74c3c")))
    fig.add_trace(go.Scatter(x=x, y=df["T_mean_C"], name="T_mean", line=dict(color="#f39c12")))
    fig.add_trace(go.Scatter(x=x, y=df["T_min_C"], name="T_min", line=dict(color="#3498db")))
    fig.update_layout(
        margin=dict(l=50, r=50, t=30, b=30),
        yaxis=dict(title="Temperature (C)"),
        legend=dict(x=0, y=1.1, orientation="h"),
    )
    return table_data, columns, fig


def _detect_format(df):
    """Detect whether a CSV is hourly or daily format. Returns 'hourly', 'daily', or None."""
    cols = set(df.columns)
    if "datetime_utc" in cols and "T_air_C" in cols:
        return "hourly"
    if "sim_day" in cols and "T_min_C" in cols and "T_max_C" in cols and "T_mean_C" in cols:
        return "daily"
    return None


def register_callbacks(app):
    # Load default on mount
    @app.callback(
        Output("met-table", "data"),
        Output("met-table", "columns"),
        Output("met-plot", "figure"),
        Output("met-source-text", "children"),
        Input("met-source-badge", "id"),  # fires once on mount
    )
    def load_default(_):
        table_data, columns, fig = _load_default_preview()
        n_days = len(table_data)
        source = f"Using {_DEFAULT_DAILY_MET.name} — {153} days, daily format (T_min/T_max → sinusoidal hourly)."
        if not table_data:
            source = "Default Jülich file not found. Upload a custom CSV."
        return table_data, columns, fig, source

    @app.callback(
        Output("met-table", "data", allow_duplicate=True),
        Output("met-table", "columns", allow_duplicate=True),
        Output("met-plot", "figure", allow_duplicate=True),
        Output("met-alert", "children"),
        Output("met-alert", "color"),
        Output("met-alert", "is_open"),
        Output("met-source-badge", "children"),
        Output("met-source-badge", "color"),
        Output("met-source-text", "children", allow_duplicate=True),
        Output("pipeline-config-store", "data", allow_duplicate=True),
        Input("met-upload", "contents"),
        State("met-upload", "filename"),
        State("pipeline-config-store", "data"),
        prevent_initial_call=True,
    )
    def process_upload(contents, filename, config_store):
        import pandas as pd
        import plotly.graph_objects as go

        empty_fig = go.Figure()
        no_change = [], [], empty_fig, "", "secondary", False, "Default: Jülich 2024", "info", "", config_store
        if not contents:
            return no_change

        try:
            _, content_string = contents.split(",")
            decoded = base64.b64decode(content_string)
            df = pd.read_csv(io.StringIO(decoded.decode("utf-8")))
        except Exception as e:
            return ([], [], empty_fig, f"Failed to parse CSV: {e}", "danger", True,
                    "Default: Jülich 2024", "info", "Upload failed, still using default.",
                    config_store)

        fmt = _detect_format(df)
        if fmt is None:
            msg = (
                "Unrecognized format. Need either:\n"
                "  Hourly: datetime_utc, T_air_C, RH, wind_ms\n"
                "  Daily: sim_day, T_min_C, T_max_C, T_mean_C"
            )
            return ([], [], empty_fig, msg, "danger", True,
                    "Default: Jülich 2024", "info", "Upload rejected, still using default.",
                    config_store)

        if fmt == "hourly":
            missing = HOURLY_REQUIRED - set(df.columns)
            if missing:
                return ([], [], empty_fig,
                        f"Missing required columns: {', '.join(sorted(missing))}",
                        "danger", True, "Default: Jülich 2024", "info",
                        "Upload rejected.", config_store)

        # Write to output dir
        from dart.coupling.config import OUTPUT_DIR
        if fmt == "hourly":
            out_path = OUTPUT_DIR / "met_custom.csv"
        else:
            out_path = OUTPUT_DIR / "met_custom_daily.csv"
        df.to_csv(out_path, index=False)

        # Update config store
        if config_store is None:
            config_store = {}
        if fmt == "hourly":
            config_store["met_csv"] = str(out_path)
        else:
            # Daily format: no met_csv (pipeline generates hourly from daily),
            # but inject into DVS cache at run time
            config_store["met_csv"] = None
            config_store["met_daily_csv"] = str(out_path)

        # Table data
        table_data = df.head(20).to_dict("records")
        columns = [{"name": c, "id": c} for c in df.columns]

        # Plot
        fig = go.Figure()
        if fmt == "hourly":
            fig.add_trace(go.Scatter(
                x=df["datetime_utc"], y=df["T_air_C"],
                name="T_air (C)", line=dict(color="#e74c3c"),
            ))
            if "RH" in df.columns:
                rh_pct = df["RH"] * 100 if df["RH"].max() <= 1 else df["RH"]
                fig.add_trace(go.Scatter(
                    x=df["datetime_utc"], y=rh_pct,
                    name="RH (%)", yaxis="y2", line=dict(color="#3498db", dash="dot"),
                ))
            fig.update_layout(
                margin=dict(l=50, r=50, t=30, b=30),
                yaxis=dict(title="Temperature (C)"),
                yaxis2=dict(title="RH (%)", overlaying="y", side="right", range=[0, 100]),
                legend=dict(x=0, y=1.1, orientation="h"),
            )
        else:
            x = df["date"] if "date" in df.columns else df["sim_day"]
            fig.add_trace(go.Scatter(x=x, y=df["T_max_C"], name="T_max", line=dict(color="#e74c3c")))
            fig.add_trace(go.Scatter(x=x, y=df["T_mean_C"], name="T_mean", line=dict(color="#f39c12")))
            fig.add_trace(go.Scatter(x=x, y=df["T_min_C"], name="T_min", line=dict(color="#3498db")))
            fig.update_layout(
                margin=dict(l=50, r=50, t=30, b=30),
                yaxis=dict(title="Temperature (C)"),
                legend=dict(x=0, y=1.1, orientation="h"),
            )

        badge_text = f"Custom: {filename}"
        source_text = (
            f"Loaded {filename}: {len(df)} rows, {fmt} format. "
            f"{'Hourly data drives diurnal loop + daily stats auto-derived for GDD.' if fmt == 'hourly' else 'Daily data drives sinusoidal hourly profiles + GDD calculation.'}"
        )
        alert = f"Loaded {filename}: {len(df)} rows ({fmt} format), saved to {out_path.name}"
        return (table_data, columns, fig, alert, "success", True,
                badge_text, "success", source_text, config_store)

    # FLUXNET CSV — load from server path
    @app.callback(
        Output("met-table", "data", allow_duplicate=True),
        Output("met-table", "columns", allow_duplicate=True),
        Output("met-plot", "figure", allow_duplicate=True),
        Output("met-fluxnet-alert", "children"),
        Output("met-fluxnet-alert", "color"),
        Output("met-fluxnet-alert", "is_open"),
        Output("met-source-badge", "children", allow_duplicate=True),
        Output("met-source-badge", "color", allow_duplicate=True),
        Output("met-source-text", "children", allow_duplicate=True),
        Output("pipeline-config-store", "data", allow_duplicate=True),
        Input("met-fluxnet-apply", "n_clicks"),
        State("met-fluxnet-csv", "value"),
        State("met-fluxnet-year", "value"),
        State("pipeline-config-store", "data"),
        prevent_initial_call=True,
    )
    def apply_fluxnet(n_clicks, csv_path, year, config_store):
        import plotly.graph_objects as go
        empty_fig = go.Figure()

        if not csv_path or not csv_path.strip():
            return ([], [], empty_fig, "Enter a FLUXNET CSV path.", "warning", True,
                    "Default", "info", "", config_store)

        csv_path = csv_path.strip()
        year = int(year or 2002)

        try:
            from dart.coupling.utils.met_forcing import load_fluxnet_csv
            df = load_fluxnet_csv(csv_path, year)
        except Exception as e:
            return ([], [], empty_fig, f"Failed to load FLUXNET: {e}", "danger", True,
                    "Default", "info", "FLUXNET load failed.", config_store)

        # Preview table (growing season only)
        import pandas as pd
        grow = df[(df.index.month >= 5) & (df.index.month <= 10)]
        # Downsample to daily for table preview
        daily = grow.resample('D').agg({
            'T_air_C': ['min', 'max', 'mean'],
            'PAR_umol': 'max',
            'wind_ms': 'mean',
            'RH': 'mean',
        })
        daily.columns = ['T_min_C', 'T_max_C', 'T_mean_C', 'PAR_peak', 'wind_mean', 'RH_mean']
        daily = daily.reset_index()
        daily['date'] = daily['datetime_utc'].dt.strftime('%Y-%m-%d')
        daily = daily.drop(columns=['datetime_utc'])
        for c in ['T_min_C', 'T_max_C', 'T_mean_C', 'PAR_peak', 'wind_mean', 'RH_mean']:
            daily[c] = daily[c].round(1)

        table_data = daily.head(20).to_dict("records")
        columns = [{"name": c, "id": c} for c in daily.columns]

        # Plot growing season temperature
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=daily["date"], y=daily["T_max_C"],
                                  name="T_max", line=dict(color="#e74c3c")))
        fig.add_trace(go.Scatter(x=daily["date"], y=daily["T_mean_C"],
                                  name="T_mean", line=dict(color="#f39c12")))
        fig.add_trace(go.Scatter(x=daily["date"], y=daily["T_min_C"],
                                  name="T_min", line=dict(color="#3498db")))
        fig.update_layout(
            margin=dict(l=50, r=50, t=30, b=30),
            yaxis=dict(title="Temperature (C)"),
            legend=dict(x=0, y=1.1, orientation="h"),
        )

        # Update config store
        if config_store is None:
            config_store = {}
        config_store["fluxnet_csv"] = csv_path
        config_store["fluxnet_year"] = year
        config_store["met_csv"] = None       # FLUXNET overrides manual met_csv
        config_store["met_daily_csv"] = None

        badge = f"FLUXNET {year}"
        source = (
            f"FLUXNET: {len(df)} hourly rows for {year} "
            f"(T: {df['T_air_C'].min():.0f}-{df['T_air_C'].max():.0f} C, "
            f"PAR peak: {df['PAR_umol'].max():.0f} umol/m2/s). "
            f"Pipeline will auto-convert at run time."
        )
        alert = f"Loaded FLUXNET {year}: {len(df)} hourly rows from {csv_path.split('/')[-1]}"

        return (table_data, columns, fig, alert, "success", True,
                badge, "success", source, config_store)
