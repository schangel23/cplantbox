"""Shared state helpers for the dashboard."""

from __future__ import annotations

import json
import os
import signal
from dataclasses import asdict
from pathlib import Path


def config_to_store(config) -> dict:
    """Serialize PipelineConfig → dict for dcc.Store."""
    return asdict(config)


def config_from_store(data: dict | None):
    """Deserialize dcc.Store dict → PipelineConfig."""
    from dart.coupling.pipeline import PipelineConfig

    if not data:
        return PipelineConfig()
    valid_fields = {f.name for f in PipelineConfig.__dataclass_fields__.values()}
    filtered = {k: v for k, v in data.items() if k in valid_fields}
    return PipelineConfig(**filtered)


def read_checkpoint(path: str | Path) -> dict | None:
    """Safely read a checkpoint JSON, return None if missing/corrupt."""
    p = Path(path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _pid_file(output_dir: str) -> Path:
    return Path(output_dir) / ".dashboard_run.pid"


def _error_file(output_dir: str) -> Path:
    return Path(output_dir) / ".dashboard_run_error.txt"


def is_pipeline_running(output_dir: str) -> tuple[bool, int | None]:
    """Check PID file + os.kill(pid, 0). Returns (running, pid)."""
    pf = _pid_file(output_dir)
    if not pf.exists():
        return False, None
    try:
        pid = int(pf.read_text().strip())
        os.kill(pid, 0)  # signal 0 = existence check
        return True, pid
    except (ValueError, OSError):
        # Stale PID file
        pf.unlink(missing_ok=True)
        return False, None


def write_pid(output_dir: str, pid: int) -> None:
    _pid_file(output_dir).write_text(str(pid))


def clear_pid(output_dir: str) -> None:
    _pid_file(output_dir).unlink(missing_ok=True)


def read_error(output_dir: str) -> str | None:
    ef = _error_file(output_dir)
    if ef.exists():
        try:
            return ef.read_text().strip()
        except OSError:
            return None
    return None


def clear_error(output_dir: str) -> None:
    _error_file(output_dir).unlink(missing_ok=True)


def stop_pipeline(output_dir: str) -> bool:
    """Kill pipeline + all child processes (DART, Baleno). Returns True if killed."""
    running, pid = is_pipeline_running(output_dir)
    if running and pid is not None:
        try:
            pgid = os.getpgid(pid)
            os.killpg(pgid, signal.SIGTERM)
            clear_pid(output_dir)
            return True
        except OSError:
            clear_pid(output_dir)
    return False
