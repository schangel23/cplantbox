"""System Setup tab — path configuration and validation."""

from __future__ import annotations

import os

import dash_bootstrap_components as dbc
from dash import Input, Output, State, callback_context, dcc, html


def _path_row(label: str, input_id: str, placeholder: str = "") -> dbc.Row:
    return dbc.Row(
        [
            dbc.Col(dbc.Label(label, className="fw-bold"), width=3),
            dbc.Col(dbc.Input(id=input_id, type="text", placeholder=placeholder), width=7),
            dbc.Col(dbc.Badge("?", id=f"{input_id}-badge", color="secondary", className="badge-ok"), width=2),
        ],
        className="mb-2 align-items-center",
    )


def layout() -> dbc.Container:
    return dbc.Container(
        [
            dbc.Card(
                [
                    dbc.CardHeader("External Tool Paths"),
                    dbc.CardBody(
                        [
                            _path_row("DART Home", "sys-dart-home", "/home/lukas/DART"),
                            _path_row("DART License (.dartrc)", "sys-dartrc", "~/.dartrcv1457"),
                            _path_row("Baleno Python", "sys-baleno-python", "darteb_venv/bin/python3.12"),
                            _path_row("CPlantBox Root", "sys-cplantbox-root", "CPlantBox/"),
                            _path_row("AgroC Source", "sys-agroc-src", "agroC_20250327_1511/src"),
                            _path_row("Output Directory", "sys-output-dir", "coupling/output"),
                        ]
                    ),
                ]
            ),
            dbc.Card(
                [
                    dbc.CardHeader("Python Modules"),
                    dbc.CardBody(
                        [
                            dbc.Row(
                                [
                                    dbc.Col(dbc.Label("plantbox", className="fw-bold"), width=3),
                                    dbc.Col(dbc.Badge("?", id="sys-plantbox-badge", color="secondary"), width=2),
                                ],
                                className="mb-2 align-items-center",
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(dbc.Label("pytools4dart", className="fw-bold"), width=3),
                                    dbc.Col(dbc.Badge("?", id="sys-pytools4dart-badge", color="secondary"), width=2),
                                ],
                                className="mb-2 align-items-center",
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(dbc.Label("pvlib", className="fw-bold"), width=3),
                                    dbc.Col(dbc.Badge("?", id="sys-pvlib-badge", color="secondary"), width=2),
                                ],
                                className="mb-2 align-items-center",
                            ),
                        ]
                    ),
                ]
            ),
            dbc.Row(
                dbc.Col(
                    dbc.Button("Validate System", id="sys-validate-btn", color="primary", className="mt-3"),
                    width="auto",
                ),
            ),
            dbc.Alert(id="sys-validation-alert", is_open=False, className="mt-3"),
        ],
        fluid=True,
        className="py-3",
    )


def register_callbacks(app):
    @app.callback(
        Output("sys-dart-home", "value"),
        Output("sys-dartrc", "value"),
        Output("sys-baleno-python", "value"),
        Output("sys-cplantbox-root", "value"),
        Output("sys-agroc-src", "value"),
        Output("sys-output-dir", "value"),
        Input("sys-dart-home", "id"),  # fires once on mount
    )
    def load_defaults(_):
        from dart.coupling.config import DART_HOME, DARTRC, BALENO_PYTHON, CPLANTBOX_ROOT, AGROC_SRC, OUTPUT_DIR
        return str(DART_HOME), str(DARTRC), str(BALENO_PYTHON), str(CPLANTBOX_ROOT), str(AGROC_SRC), str(OUTPUT_DIR)

    @app.callback(
        Output("sys-dart-home-badge", "children"),
        Output("sys-dart-home-badge", "color"),
        Output("sys-dartrc-badge", "children"),
        Output("sys-dartrc-badge", "color"),
        Output("sys-baleno-python-badge", "children"),
        Output("sys-baleno-python-badge", "color"),
        Output("sys-cplantbox-root-badge", "children"),
        Output("sys-cplantbox-root-badge", "color"),
        Output("sys-agroc-src-badge", "children"),
        Output("sys-agroc-src-badge", "color"),
        Output("sys-output-dir-badge", "children"),
        Output("sys-output-dir-badge", "color"),
        Output("sys-plantbox-badge", "children"),
        Output("sys-plantbox-badge", "color"),
        Output("sys-pytools4dart-badge", "children"),
        Output("sys-pytools4dart-badge", "color"),
        Output("sys-pvlib-badge", "children"),
        Output("sys-pvlib-badge", "color"),
        Output("sys-validation-alert", "children"),
        Output("sys-validation-alert", "color"),
        Output("sys-validation-alert", "is_open"),
        Output("validation-store", "data"),
        Input("sys-validate-btn", "n_clicks"),
        State("sys-dart-home", "value"),
        State("sys-dartrc", "value"),
        State("sys-baleno-python", "value"),
        State("sys-cplantbox-root", "value"),
        State("sys-agroc-src", "value"),
        State("sys-output-dir", "value"),
        prevent_initial_call=True,
    )
    def validate_system(n_clicks, dart_home, dartrc, baleno_python, cplantbox_root, agroc_src, output_dir):
        from dart.coupling.pipeline import PipelineConfig, PipelineRunner

        # Save/restore env vars around validate_system (it calls _apply_env)
        saved_env = {k: os.environ.get(k) for k in [
            "COUPLING_SPECIES", "DART_THREADS", "DART_RAY_DENSITY",
            "DART_MAX_RENDERING_TIME", "DART_HOME", "DARTRC",
            "BALENO_PYTHON", "CPLANTBOX_ROOT", "AGROC_SRC", "GROWTH_MODE",
        ]}

        try:
            config = PipelineConfig(
                dart_home=dart_home or "",
                dartrc=dartrc or "",
                baleno_python=baleno_python or "",
                cplantbox_root=cplantbox_root or "",
                agroc_src=agroc_src or "",
                output_dir=output_dir or "",
            )
            runner = PipelineRunner(config)
            results = runner.validate_system()
        finally:
            for k, v in saved_env.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

        outputs = []

        # Path badges
        path_keys = [
            ("dart_binary", "sys-dart-home"),
            ("dart_license", "sys-dartrc"),
            ("baleno_python", "sys-baleno-python"),
        ]
        for key, _ in path_keys:
            info = results.get(key, {"ok": False, "error": "not checked"})
            outputs.extend(["OK" if info["ok"] else "FAIL", "success" if info["ok"] else "danger"])

        # CPlantBox root — check species_xml as proxy
        sp_xml = results.get("species_xml", {"ok": False})
        outputs.extend(["OK" if sp_xml["ok"] else "FAIL", "success" if sp_xml["ok"] else "danger"])

        # AgroC source — check binary exists
        agroc_info = results.get("agroc_src", {"ok": False})
        outputs.extend(["OK" if agroc_info["ok"] else "FAIL", "success" if agroc_info["ok"] else "warning"])

        # Output dir — always OK if path is set
        outputs.extend(["OK" if output_dir else "FAIL", "success" if output_dir else "danger"])

        # Module badges
        for mod in ["plantbox", "pytools4dart", "pvlib"]:
            info = results.get(mod, {"ok": False})
            outputs.extend(["OK" if info["ok"] else "FAIL", "success" if info["ok"] else "danger"])

        # Summary alert
        n_ok = sum(1 for v in results.values() if v.get("ok"))
        n_total = len(results)
        all_ok = all(v.get("ok") for v in results.values())
        if all_ok:
            alert_msg = f"All {n_total} checks passed."
            alert_color = "success"
        else:
            failures = [f"{k}: {v.get('error', 'unknown')}" for k, v in results.items() if not v.get("ok")]
            alert_msg = f"{n_ok}/{n_total} passed. Failures: " + "; ".join(failures)
            alert_color = "warning"

        outputs.extend([alert_msg, alert_color, True])
        outputs.append(results)  # validation-store

        return tuple(outputs)
