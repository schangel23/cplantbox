"""Run + Progress tab — start/stop pipeline, poll checkpoint, show log."""

from __future__ import annotations

import json
import multiprocessing
import os
import signal
import sys
import time
from pathlib import Path

import dash_bootstrap_components as dbc
from dash import Input, Output, State, dcc, html

from ..state import (
    clear_error,
    clear_pid,
    config_from_store,
    is_pipeline_running,
    read_checkpoint,
    read_error,
    stop_pipeline,
    write_pid,
)

# Log file for capturing subprocess stdout
_LOG_FILE = ".dashboard_run.log"


def _run_pipeline_subprocess(config_dict: dict, output_dir: str):
    """Target for multiprocessing.Process. Runs pipeline in isolated process."""
    import io

    log_path = Path(output_dir) / _LOG_FILE

    # Redirect stdout/stderr to log file so the dashboard can read it
    log_fh = open(log_path, "w", buffering=1)  # line-buffered
    sys.stdout = log_fh
    sys.stderr = log_fh

    from dart.coupling.pipeline import PipelineConfig, PipelineRunner

    valid_fields = {f.name for f in PipelineConfig.__dataclass_fields__.values()}
    filtered = {k: v for k, v in config_dict.items() if k in valid_fields}
    config = PipelineConfig(**filtered)

    runner = PipelineRunner(config)
    try:
        runner.run()
        print("\n=== Pipeline completed successfully ===")
    except Exception as e:
        print(f"\n=== Pipeline FAILED: {e} ===")
        Path(output_dir, ".dashboard_run_error.txt").write_text(str(e))
    finally:
        log_fh.flush()
        log_fh.close()
        Path(output_dir, ".dashboard_run.pid").unlink(missing_ok=True)


def layout() -> dbc.Container:
    return dbc.Container(
        [
            dbc.Card(
                [
                    dbc.CardHeader("Config Summary"),
                    dbc.CardBody(html.Pre(id="run-config-summary", className="config-preview")),
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Button("Start Pipeline", id="run-start-btn", color="success", className="me-2"),
                        width="auto",
                    ),
                    dbc.Col(
                        dbc.Button("Stop Pipeline", id="run-stop-btn", color="danger", outline=True),
                        width="auto",
                    ),
                ],
                className="my-3",
            ),
            dbc.Alert(id="run-alert", is_open=False, className="mb-3"),
            dbc.Card(
                [
                    dbc.CardHeader("Progress"),
                    dbc.CardBody(
                        [
                            dbc.Progress(id="run-progress", value=0, striped=True, animated=True, className="mb-2"),
                            html.P(id="run-progress-text", className="text-muted"),
                        ]
                    ),
                ]
            ),
            dbc.Card(
                [
                    dbc.CardHeader("Log"),
                    dbc.CardBody(
                        html.Pre(id="run-log", className="log-area", children="Waiting to start..."),
                    ),
                ],
                className="mt-3",
            ),
            dcc.Interval(id="run-interval", interval=3000, disabled=True),
        ],
        fluid=True,
        className="py-3",
    )


def _read_log_tail(output_dir: str, n_lines: int = 80) -> str:
    """Read the last N lines of the subprocess log file."""
    log_path = Path(output_dir) / _LOG_FILE
    if not log_path.exists():
        return ""
    try:
        text = log_path.read_text(errors="replace")
        lines = text.splitlines()
        return "\n".join(lines[-n_lines:])
    except OSError:
        return ""


def register_callbacks(app):
    @app.callback(
        Output("run-config-summary", "children"),
        Input("pipeline-config-store", "data"),
    )
    def show_summary(store):
        if not store:
            return "No config set. Configure in Simulation tab."
        lines = []
        for key in ["mode", "species", "growth_days", "grid_nx", "grid_ny",
                     "timestep_min", "enable_baleno", "iterate_gs",
                     "with_carbon", "resume"]:
            if key in store:
                lines.append(f"{key}: {store[key]}")
        return "\n".join(lines)

    @app.callback(
        Output("run-alert", "children", allow_duplicate=True),
        Output("run-alert", "color", allow_duplicate=True),
        Output("run-alert", "is_open", allow_duplicate=True),
        Output("run-interval", "disabled", allow_duplicate=True),
        Output("run-progress", "value", allow_duplicate=True),
        Output("run-log", "children", allow_duplicate=True),
        Input("run-start-btn", "n_clicks"),
        State("pipeline-config-store", "data"),
        prevent_initial_call=True,
    )
    def start_pipeline(n_clicks, store):
        if not store:
            return "No config — set up Simulation tab first.", "warning", True, True, 0, "No config."

        config = config_from_store(store)
        output_dir = config.output_dir

        # Ensure output dir exists
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        running, pid = is_pipeline_running(output_dir)
        if running:
            return (
                f"Pipeline already running (PID {pid}). Stop it first.",
                "warning", True, False, 0, _read_log_tail(output_dir) or f"Running PID {pid}",
            )

        # Clear previous error and log
        clear_error(output_dir)
        log_path = Path(output_dir) / _LOG_FILE
        log_path.unlink(missing_ok=True)

        # Spawn subprocess
        proc = multiprocessing.Process(
            target=_run_pipeline_subprocess,
            args=(store, output_dir),
            daemon=False,  # non-daemon so it survives if Dash restarts
        )
        proc.start()
        write_pid(output_dir, proc.pid)

        return (
            f"Pipeline started (PID {proc.pid}).",
            "success", True, False, 0, f"Started PID {proc.pid}, waiting for output...\n",
        )

    @app.callback(
        Output("run-alert", "children", allow_duplicate=True),
        Output("run-alert", "color", allow_duplicate=True),
        Output("run-alert", "is_open", allow_duplicate=True),
        Output("run-interval", "disabled", allow_duplicate=True),
        Input("run-stop-btn", "n_clicks"),
        State("pipeline-config-store", "data"),
        prevent_initial_call=True,
    )
    def stop_btn(n_clicks, store):
        if not store:
            return "No config.", "warning", True, True

        config = config_from_store(store)
        output_dir = config.output_dir

        running, pid = is_pipeline_running(output_dir)
        if not running:
            return "No running pipeline to stop.", "secondary", True, True

        # Kill the process tree (pipeline may spawn DART subprocesses)
        try:
            # First try SIGTERM for graceful shutdown
            os.kill(pid, signal.SIGTERM)
            # Give it a moment, then force-kill if still alive
            time.sleep(1)
            try:
                os.kill(pid, 0)  # check if still alive
                os.kill(pid, signal.SIGKILL)  # force kill
            except OSError:
                pass  # already dead, good
        except OSError:
            pass

        clear_pid(output_dir)
        return f"Pipeline stopped (PID {pid}).", "info", True, True

    @app.callback(
        Output("run-progress", "value"),
        Output("run-progress-text", "children"),
        Output("run-log", "children"),
        Output("run-interval", "disabled"),
        Output("run-alert", "children"),
        Output("run-alert", "color"),
        Output("run-alert", "is_open"),
        Input("run-interval", "n_intervals"),
        State("pipeline-config-store", "data"),
    )
    def poll_progress(n, store):
        if not store:
            return 0, "", "Waiting to start...", True, "", "secondary", False

        config = config_from_store(store)
        output_dir = config.output_dir
        running, pid = is_pipeline_running(output_dir)

        # Read live log from subprocess stdout
        log_text = _read_log_tail(output_dir)

        # Determine checkpoint path based on mode
        if config.mode == "uniform_baseline":
            cp_dir = Path(output_dir) / "diurnal_uniform"
        elif config.mode == "carbon_feedback":
            cp_dir = Path(output_dir) / "diurnal_carbon"
        else:
            cp_dir = Path(output_dir) / "diurnal"

        cp = read_checkpoint(cp_dir / "production_checkpoint.json")

        # Progress from checkpoint
        total = len(config.growth_days) if config.growth_days else 1
        completed_days = cp.get("completed_days", []) if cp else []
        completed = len(completed_days)
        pct = min(100, int(completed / total * 100)) if total > 0 else 0
        ptext = f"{completed}/{total} growth days completed"
        if completed_days:
            ptext += f" (days: {', '.join(str(d) for d in completed_days)})"

        if not running:
            err = read_error(output_dir)
            if err:
                if log_text:
                    log_text += f"\n\n=== ERROR: {err} ==="
                else:
                    log_text = f"ERROR: {err}"
                return (
                    pct, ptext, log_text,
                    True, f"Pipeline failed: {err[:200]}", "danger", True,
                )
            if completed >= total and total > 0:
                return (
                    100, f"Complete: {completed}/{total} days",
                    log_text or "Complete.",
                    True, "Pipeline finished successfully!", "success", True,
                )
            # Stopped or never started
            return (
                pct, ptext, log_text or "Waiting to start...",
                True, "", "secondary", False,
            )

        # Still running
        return (
            pct, ptext, log_text or "Running, waiting for output...",
            False, f"Running (PID {pid})", "info", True,
        )

    @app.callback(
        Output("run-interval", "disabled", allow_duplicate=True),
        Input("run-interval", "id"),  # fires on mount
        State("pipeline-config-store", "data"),
        prevent_initial_call="initial_duplicate",
    )
    def resume_on_load(_, store):
        """Detect running pipeline on page load and resume polling."""
        if not store:
            return True
        config = config_from_store(store)
        running, _ = is_pipeline_running(config.output_dir)
        return not running
