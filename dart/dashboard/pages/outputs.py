"""Results browser tab — plots for diurnal, comparison, carbon."""

from __future__ import annotations

import json
from pathlib import Path

import dash_bootstrap_components as dbc
from dash import Input, Output, State, dcc, html


_MODES = [
    {"label": "Diurnal (3D)", "value": "diurnal"},
    {"label": "Uniform", "value": "diurnal_uniform"},
    {"label": "Compare", "value": "compare"},
]


def layout() -> dbc.Container:
    return dbc.Container(
        [
            dbc.Row(
                [
                    dbc.Col(
                        [dbc.Label("Result Mode"), dcc.Dropdown(id="out-mode", options=_MODES, value="diurnal")],
                        width=3,
                    ),
                    dbc.Col(
                        [dbc.Label("Day"), dcc.Dropdown(id="out-day", options=[], value=None)],
                        width=2,
                    ),
                    dbc.Col(
                        dbc.Button("Refresh", id="out-refresh-btn", color="secondary", outline=True, className="mt-4"),
                        width="auto",
                    ),
                ],
                className="mb-3",
            ),
            dbc.Alert(id="out-alert", is_open=False),
            dcc.Tabs(
                id="out-tabs",
                value="tab-an",
                children=[
                    dcc.Tab(label="Diurnal An", value="tab-an"),
                    dcc.Tab(label="Per-plant An", value="tab-per-plant"),
                    dcc.Tab(label="Tleaf vs Tair", value="tab-tleaf"),
                    dcc.Tab(label="Carbon Partitioning", value="tab-carbon"),
                    dcc.Tab(label="Growth Trajectory", value="tab-growth"),
                    dcc.Tab(label="SIF / Fluorescence", value="tab-sif"),
                ],
            ),
            html.Div(id="out-plot-container", className="mt-3"),
        ],
        fluid=True,
        className="py-3",
    )


def _find_days(output_dir: Path, subdir: str) -> list[int]:
    """Scan output/{subdir}/day* for available days."""
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


def _read_hourly(output_dir: Path, subdir: str, day: int):
    """Read hourly_results.csv for a given day. Returns DataFrame or None."""
    import pandas as pd
    csv = output_dir / subdir / f"day{day}" / "hourly_results.csv"
    if not csv.exists():
        return None
    return pd.read_csv(csv)


def _read_daily(output_dir: Path, subdir: str, day: int) -> dict | None:
    """Read daily_summary.json for a given day."""
    p = output_dir / subdir / f"day{day}" / "daily_summary.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def register_callbacks(app):
    @app.callback(
        Output("out-day", "options"),
        Output("out-day", "value"),
        Input("out-mode", "value"),
        Input("out-refresh-btn", "n_clicks"),
        State("pipeline-config-store", "data"),
    )
    def scan_days(mode, _, store):
        from dart.coupling.config import OUTPUT_DIR
        output_dir = Path(store.get("output_dir", str(OUTPUT_DIR))) if store else OUTPUT_DIR

        if mode == "compare":
            days_3d = set(_find_days(output_dir, "diurnal"))
            days_uni = set(_find_days(output_dir, "diurnal_uniform"))
            days = sorted(days_3d & days_uni)
        else:
            days = _find_days(output_dir, mode)

        options = [{"label": f"Day {d}", "value": d} for d in days]
        value = days[-1] if days else None
        return options, value

    @app.callback(
        Output("out-plot-container", "children"),
        Output("out-alert", "children"),
        Output("out-alert", "color"),
        Output("out-alert", "is_open"),
        Input("out-day", "value"),
        Input("out-tabs", "value"),
        State("out-mode", "value"),
        State("pipeline-config-store", "data"),
    )
    def render_plot(day, tab, mode, store):
        import plotly.graph_objects as go

        from dart.coupling.config import OUTPUT_DIR
        output_dir = Path(store.get("output_dir", str(OUTPUT_DIR))) if store else OUTPUT_DIR

        if day is None:
            return html.P("No day selected."), "Select a day to view.", "info", True

        try:
            if tab == "tab-an":
                return _plot_diurnal_an(output_dir, mode, day), "", "secondary", False
            elif tab == "tab-per-plant":
                return _plot_per_plant(output_dir, mode, day), "", "secondary", False
            elif tab == "tab-tleaf":
                return _plot_tleaf(output_dir, mode, day), "", "secondary", False
            elif tab == "tab-carbon":
                return _plot_carbon(output_dir, mode, day), "", "secondary", False
            elif tab == "tab-growth":
                return _plot_growth(output_dir, mode), "", "secondary", False
            elif tab == "tab-sif":
                return _plot_sif(output_dir, mode, day), "", "secondary", False
        except Exception as e:
            return html.P(f"Error: {e}"), str(e), "danger", True

        return html.P("Unknown tab"), "", "secondary", False


def _plot_diurnal_an(output_dir: Path, mode: str, day: int):
    import plotly.graph_objects as go

    fig = go.Figure()

    if mode == "compare":
        for subdir, name, dash in [("diurnal", "3D", "solid"), ("diurnal_uniform", "Uniform", "dash")]:
            df = _read_hourly(output_dir, subdir, day)
            if df is not None:
                fig.add_trace(go.Scatter(
                    x=df["time_utc"], y=df["An_total_mmol_d"],
                    name=name, line=dict(dash=dash),
                ))
    else:
        df = _read_hourly(output_dir, mode, day)
        if df is not None:
            fig.add_trace(go.Scatter(
                x=df["time_utc"], y=df["An_total_mmol_d"],
                name="An total", line=dict(color="#27ae60"),
            ))

    fig.update_layout(
        title=f"Day {day} — Diurnal Net Assimilation",
        xaxis_title="Time (UTC)", yaxis_title="An (mmol d⁻¹)",
        margin=dict(l=50, r=30, t=40, b=30),
    )
    return dcc.Graph(figure=fig)


def _plot_per_plant(output_dir: Path, mode: str, day: int):
    import plotly.graph_objects as go

    subdir = "diurnal" if mode == "compare" else mode
    df = _read_hourly(output_dir, subdir, day)
    if df is None:
        return html.P("No data available.")

    plant_cols = [c for c in df.columns if c.startswith("An_p")]
    if not plant_cols:
        return html.P("Per-plant columns (An_p0..An_pN) not found in hourly results.")

    fig = go.Figure()
    for col in plant_cols:
        fig.add_trace(go.Box(y=df[col], name=col.replace("An_", "Plant ")))

    fig.update_layout(
        title=f"Day {day} — Per-plant An Distribution",
        yaxis_title="An (mmol d⁻¹)",
        margin=dict(l=50, r=30, t=40, b=30),
    )
    return dcc.Graph(figure=fig)


def _plot_tleaf(output_dir: Path, mode: str, day: int):
    import plotly.graph_objects as go

    fig = go.Figure()

    if mode == "compare":
        for subdir, name, dash in [("diurnal", "Tleaf 3D", "solid"), ("diurnal_uniform", "Tleaf Uniform", "dash")]:
            df = _read_hourly(output_dir, subdir, day)
            if df is not None:
                if "mean_tleaf_C" in df.columns:
                    fig.add_trace(go.Scatter(x=df["time_utc"], y=df["mean_tleaf_C"], name=name, line=dict(dash=dash)))
        # Add Tair from either
        for subdir in ["diurnal", "diurnal_uniform"]:
            df = _read_hourly(output_dir, subdir, day)
            if df is not None and "T_air_C" in df.columns:
                fig.add_trace(go.Scatter(x=df["time_utc"], y=df["T_air_C"], name="Tair", line=dict(color="gray", dash="dot")))
                break
    else:
        df = _read_hourly(output_dir, mode, day)
        if df is not None:
            if "mean_tleaf_C" in df.columns:
                fig.add_trace(go.Scatter(x=df["time_utc"], y=df["mean_tleaf_C"], name="Tleaf", line=dict(color="#e74c3c")))
            if "T_air_C" in df.columns:
                fig.add_trace(go.Scatter(x=df["time_utc"], y=df["T_air_C"], name="Tair", line=dict(color="gray", dash="dot")))

    fig.update_layout(
        title=f"Day {day} — Leaf vs Air Temperature",
        xaxis_title="Time (UTC)", yaxis_title="Temperature (C)",
        margin=dict(l=50, r=30, t=40, b=30),
    )
    return dcc.Graph(figure=fig)


def _plot_carbon(output_dir: Path, mode: str, day: int):
    import plotly.graph_objects as go

    subdir = "diurnal" if mode == "compare" else mode
    summary = _read_daily(output_dir, subdir, day)
    if not summary:
        return html.P("No daily summary found.")

    # Try carbon fraction keys
    labels, values = [], []
    for key, label in [("FR_leaf", "Leaf"), ("FR_stem", "Stem"), ("FR_root", "Root"), ("FR_storage", "Storage")]:
        if key in summary:
            labels.append(label)
            values.append(summary[key])

    if not values:
        return html.P("No carbon partitioning data in daily summary.")

    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.35)])
    fig.update_layout(
        title=f"Day {day} — Carbon Partitioning",
        margin=dict(l=30, r=30, t=40, b=30),
    )
    return dcc.Graph(figure=fig)


def _plot_growth(output_dir: Path, mode: str):
    import plotly.graph_objects as go

    fig = go.Figure()

    subdirs = ["diurnal", "diurnal_uniform"] if mode == "compare" else [mode if mode != "compare" else "diurnal"]
    for subdir in subdirs:
        days = _find_days(output_dir, subdir)
        an_values = []
        for d in days:
            s = _read_daily(output_dir, subdir, d)
            if s and "daily_An_mmol" in s:
                an_values.append((d, s["daily_An_mmol"]))

        if an_values:
            ds, ans = zip(*an_values)
            name = "3D" if subdir == "diurnal" else ("Uniform" if subdir == "diurnal_uniform" else subdir)
            dash = "solid" if subdir == "diurnal" else "dash"
            fig.add_trace(go.Scatter(x=list(ds), y=list(ans), name=name, line=dict(dash=dash), mode="lines+markers"))

    fig.update_layout(
        title="Growth Trajectory — Daily An across Growth Days",
        xaxis_title="Growth Day", yaxis_title="Daily An (mmol)",
        margin=dict(l=50, r=30, t=40, b=30),
    )
    return dcc.Graph(figure=fig)


def _plot_sif(output_dir: Path, mode: str, day: int):
    """2x2 SIF / Fluorescence plots."""
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    subdir = "diurnal" if mode == "compare" else mode
    df = _read_hourly(output_dir, subdir, day)
    if df is None:
        return html.P("No data available.")

    if "SIF_canopy_W_m2" not in df.columns or df["SIF_canopy_W_m2"].isna().all():
        return html.P("No SIF data. Run with --with-sif to generate fluorescence output.")

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Canopy SIF Emission",
            "Sunlit / Shaded Fraction",
            "Fluorescence Yield (eta)",
            "SIF vs An",
        ),
    )

    # Panel 1: Canopy SIF diurnal curve
    fig.add_trace(
        go.Scatter(x=df["time_utc"], y=df["SIF_canopy_W_m2"],
                   name="SIF canopy", line=dict(color="#8e44ad")),
        row=1, col=1,
    )
    # Add TOC SIF if available (Level 2)
    if "SIF_760_Wm2sr" in df.columns and not df["SIF_760_Wm2sr"].isna().all():
        fig.add_trace(
            go.Scatter(x=df["time_utc"], y=df["SIF_760_Wm2sr"],
                       name="TOC SIF 760nm", line=dict(color="#e74c3c", dash="dash")),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(x=df["time_utc"], y=df["SIF_687_Wm2sr"],
                       name="TOC SIF 687nm", line=dict(color="#3498db", dash="dash")),
            row=1, col=1,
        )
    # Compare overlay
    if mode == "compare":
        df_uni = _read_hourly(output_dir, "diurnal_uniform", day)
        if df_uni is not None and "SIF_canopy_W_m2" in df_uni.columns:
            fig.add_trace(
                go.Scatter(x=df_uni["time_utc"], y=df_uni["SIF_canopy_W_m2"],
                           name="SIF uniform", line=dict(color="#8e44ad", dash="dot")),
                row=1, col=1,
            )

    # Panel 2: Sunlit/shaded fraction area chart
    if "f_sunlit_area" in df.columns:
        fig.add_trace(
            go.Scatter(x=df["time_utc"], y=df["f_sunlit_area"],
                       name="Sunlit frac", fill="tozeroy",
                       line=dict(color="#f39c12")),
            row=1, col=2,
        )

    # Panel 3: Fluorescence yield (eta) — sunlit vs shaded
    if "mean_eta_sunlit" in df.columns:
        fig.add_trace(
            go.Scatter(x=df["time_utc"], y=df["mean_eta_sunlit"],
                       name="eta sunlit", line=dict(color="#e67e22")),
            row=2, col=1,
        )
    if "mean_eta_shaded" in df.columns:
        fig.add_trace(
            go.Scatter(x=df["time_utc"], y=df["mean_eta_shaded"],
                       name="eta shaded", line=dict(color="#2c3e50", dash="dash")),
            row=2, col=1,
        )

    # Panel 4: SIF vs An scatter (or TOC vs canopy if Level 2 available)
    has_toc = ("SIF_760_Wm2sr" in df.columns
               and not df["SIF_760_Wm2sr"].isna().all())
    if has_toc:
        fig.add_trace(
            go.Scatter(x=df["SIF_canopy_W_m2"], y=df["SIF_total_Wm2sr"],
                       mode="markers", name="TOC vs canopy SIF",
                       marker=dict(color=list(range(len(df))),
                                   colorscale="Viridis", size=8)),
            row=2, col=2,
        )
        fig.update_xaxes(title_text="Canopy SIF (W/m2)", row=2, col=2)
        fig.update_yaxes(title_text="TOC SIF (W/m2/sr)", row=2, col=2)
    elif "An_field_mean_mmol_d" in df.columns:
        fig.add_trace(
            go.Scatter(x=df["An_field_mean_mmol_d"], y=df["SIF_canopy_W_m2"],
                       mode="markers", name="SIF vs An",
                       marker=dict(color=list(range(len(df))),
                                   colorscale="Viridis", size=8)),
            row=2, col=2,
        )
        fig.update_xaxes(title_text="An (mmol/d)", row=2, col=2)
        fig.update_yaxes(title_text="Canopy SIF (W/m2)", row=2, col=2)

    fig.update_xaxes(title_text="Time (UTC)", row=1, col=1)
    fig.update_yaxes(title_text="SIF (W/m2)", row=1, col=1)
    fig.update_xaxes(title_text="Time (UTC)", row=1, col=2)
    fig.update_yaxes(title_text="Fraction", row=1, col=2)
    fig.update_xaxes(title_text="Time (UTC)", row=2, col=1)
    fig.update_yaxes(title_text="eta (dimensionless)", row=2, col=1)

    fig.update_layout(
        title=f"Day {day} — SIF / Fluorescence",
        height=700,
        margin=dict(l=50, r=30, t=60, b=30),
        showlegend=True,
    )
    return dcc.Graph(figure=fig)
