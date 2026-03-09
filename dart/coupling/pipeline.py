"""Programmatic pipeline entry point: PipelineConfig + PipelineRunner.

Enables config-file-driven runs and direct instantiation from dashboards.
All submodule imports are deferred to run() — env vars must be set first
since config.py reads them at import time.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


@dataclass
class PipelineConfig:
    """Configuration for a coupling pipeline run.

    Covers all user-facing knobs: run mode, scene geometry, physics,
    carbon, DART tuning, external paths, and I/O.
    """

    # Run mode
    mode: str = "full_production"  # full_production | uniform_baseline | carbon_feedback | single_day
    species: str = "maize"

    # Temporal
    growth_days: list = field(default_factory=lambda: [10, 14, 18, 22, 26, 30, 35, 40, 45, 50, 55, 58])
    single_day: int | None = None  # only for mode="single_day"
    timestep_min: int = 60

    # Scene geometry
    lat: float = 50.92         # Jülich
    lon: float = 6.36
    sowing_date: str = "2025-05-01"
    scene_size_x: float = 4.0  # DART scene [m]
    scene_size_y: float = 4.0
    grid_nx: int = 3           # plants per row
    grid_ny: int = 3           # rows
    grid_spacing_x: float = 0.75  # inter-row [m]
    grid_spacing_y: float = 0.25  # intra-row [m]

    # Soil / water
    soil_psi_cm: float = -500.0  # collar water potential [cm] (-500 = well-watered)

    # Physics
    enable_baleno: bool = True
    iterate_gs: bool = True
    gs_max_iterations: int = 6
    gs_tolerance: float = 0.05
    gs_damping_alpha: float = 0.6

    # Carbon
    with_carbon: bool = True
    carbon_method: str = "auto"  # auto | phloem | dvs
    with_agroc: bool = False

    # SIF / Fluorescence
    with_sif: bool = False
    with_dart_f: bool = False
    sif_triangles: bool = False

    # DART tuning
    threads: int = 8
    dart_ray_density: int = 500
    dart_max_rendering_time: int = 0

    # External paths (empty = read from config.py defaults)
    dart_home: str = ""
    dartrc: str = ""
    baleno_python: str = ""
    cplantbox_root: str = ""

    # I/O
    output_dir: str = ""
    log_file: str = ""  # log file path (default: {output_dir}/.dashboard_run.log)
    met_csv: str | None = None
    met_daily_csv: str | None = None  # daily format override for DVS GDD
    resume: bool = True

    def __post_init__(self):
        """Fill empty path fields from config.py defaults."""
        from .config import DART_HOME, DARTRC, BALENO_PYTHON, CPLANTBOX_ROOT, OUTPUT_DIR
        if not self.dart_home:
            self.dart_home = str(DART_HOME)
        if not self.dartrc:
            self.dartrc = str(DARTRC)
        if not self.baleno_python:
            self.baleno_python = str(BALENO_PYTHON)
        if not self.cplantbox_root:
            self.cplantbox_root = str(CPLANTBOX_ROOT)
        if not self.output_dir:
            self.output_dir = str(OUTPUT_DIR)

    def save(self, path: str | Path) -> None:
        """Save config to JSON file."""
        d = asdict(self)
        Path(path).write_text(json.dumps(d, indent=2) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> PipelineConfig:
        """Load config from JSON file. Unknown keys are ignored."""
        raw = json.loads(Path(path).read_text())
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in raw.items() if k in valid_fields}
        return cls(**filtered)

    @classmethod
    def from_args(cls, args) -> PipelineConfig:
        """Create from an argparse namespace (backward-compat bridge).

        Maps the diurnal main() CLI args to PipelineConfig fields.
        """
        # Determine mode
        if getattr(args, 'uniform', False):
            mode = "uniform_baseline"
        elif getattr(args, 'growth_mode', 'parametric') == 'carbon' and getattr(args, 'with_carbon', False):
            mode = "carbon_feedback"
        elif getattr(args, 'days', None) is not None:
            mode = "single_day"
        else:
            mode = "full_production"

        # Parse growth_days
        if getattr(args, 'growth_days', None):
            growth_days = [int(d.strip()) for d in args.growth_days.split(',')]
        else:
            growth_days = [10, 14, 18, 22, 26, 30, 35, 40, 45, 50, 55, 58]

        return cls(
            mode=mode,
            species=os.environ.get("COUPLING_SPECIES", "maize"),
            growth_days=growth_days,
            single_day=getattr(args, 'days', None),
            timestep_min=getattr(args, 'timestep_min', 60),
            enable_baleno=not getattr(args, 'no_baleno', False),
            iterate_gs=getattr(args, 'iterate_gs', False),
            gs_max_iterations=getattr(args, 'gs_max_iter', 6),
            gs_tolerance=getattr(args, 'gs_tolerance', 0.05),
            gs_damping_alpha=getattr(args, 'gs_damping', 0.6),
            with_carbon=getattr(args, 'with_carbon', False),
            carbon_method=getattr(args, 'carbon_method', 'auto'),
            with_agroc=getattr(args, 'with_agroc', False),
            with_sif=getattr(args, 'with_sif', False),
            with_dart_f=getattr(args, 'with_dart_f', False),
            sif_triangles=getattr(args, 'sif_triangles', False),
            met_csv=getattr(args, 'met_csv', None),
            resume=getattr(args, 'resume', False),
        )


class PipelineRunner:
    """Unified pipeline runner — drives the coupling pipeline from a config.

    Usage::

        config = PipelineConfig.load("pipeline_config.json")
        runner = PipelineRunner(config)
        status = runner.validate_system()
        result = runner.run()
    """

    def __init__(self, config: PipelineConfig, on_progress=None):
        self.config = config
        self.on_progress = on_progress  # callable({'phase': str, 'message': str, ...})

    def _notify(self, phase: str, message: str, **extra):
        """Fire progress callback if registered."""
        if self.on_progress is not None:
            self.on_progress({'phase': phase, 'message': message, **extra})

    def _apply_env(self):
        """Set env vars from config BEFORE importing submodules.

        config.py reads env vars at import time, so these must be set first.
        """
        c = self.config
        os.environ["COUPLING_SPECIES"] = c.species.lower()
        os.environ["DART_THREADS"] = str(c.threads)
        os.environ["DART_RAY_DENSITY"] = str(c.dart_ray_density)
        os.environ["DART_MAX_RENDERING_TIME"] = str(c.dart_max_rendering_time)

        if c.dart_home:
            os.environ["DART_HOME"] = c.dart_home
        if c.dartrc:
            os.environ["DARTRC"] = c.dartrc
        if c.baleno_python:
            os.environ["BALENO_PYTHON"] = c.baleno_python
        if c.cplantbox_root:
            os.environ["CPLANTBOX_ROOT"] = c.cplantbox_root

        # Growth mode: carbon_feedback → "carbon", else "parametric"
        if c.mode == "carbon_feedback":
            os.environ["GROWTH_MODE"] = "carbon"
        else:
            os.environ["GROWTH_MODE"] = "parametric"

    def validate_system(self) -> dict[str, dict[str, Any]]:
        """Check external tools and data files.

        Returns dict of {component: {ok: bool, path: str, error: str|None}}.
        Does NOT raise — caller decides what to do with failures.
        """
        self._apply_env()
        c = self.config

        results = {}

        # 1. DART binary (may be "dart" or "dart.exe")
        dart_bin = Path(c.dart_home) / "bin" / "dart"
        dart_bin_exe = Path(c.dart_home) / "bin" / "dart.exe"
        found = dart_bin.exists() or dart_bin_exe.exists()
        found_path = str(dart_bin_exe) if dart_bin_exe.exists() else str(dart_bin)
        results["dart_binary"] = {
            "ok": found,
            "path": found_path,
            "error": None if found else "DART binary not found",
        }

        # 2. DART license
        dartrc = Path(c.dartrc)
        results["dart_license"] = {
            "ok": dartrc.exists(),
            "path": str(dartrc),
            "error": None if dartrc.exists() else "DART license file not found",
        }

        # 3. Baleno Python
        baleno = Path(c.baleno_python)
        results["baleno_python"] = {
            "ok": baleno.exists() or shutil.which(str(baleno)) is not None,
            "path": str(baleno),
            "error": None if (baleno.exists() or shutil.which(str(baleno))) else "Baleno Python not found",
        }

        # 4. CPlantBox import
        try:
            import plantbox  # noqa: F401
            results["plantbox"] = {"ok": True, "path": "", "error": None}
        except ImportError as e:
            results["plantbox"] = {"ok": False, "path": "", "error": str(e)}

        # 5. pytools4dart import
        try:
            import pytools4dart  # noqa: F401
            results["pytools4dart"] = {"ok": True, "path": "", "error": None}
        except ImportError as e:
            results["pytools4dart"] = {"ok": False, "path": "", "error": str(e)}

        # 6. pvlib import
        try:
            import pvlib  # noqa: F401
            results["pvlib"] = {"ok": True, "path": "", "error": None}
        except ImportError as e:
            results["pvlib"] = {"ok": False, "path": "", "error": str(e)}

        # 7. Species data files
        from .config import DATA_DIR, SPECIES_REGISTRY
        sp_name = c.species.lower()
        if sp_name in SPECIES_REGISTRY:
            sp = SPECIES_REGISTRY[sp_name]
            for key in ("hydraulics", "photosynthesis", "phloem"):
                json_path = DATA_DIR / (sp[key] + ".json")
                results[f"species_{key}"] = {
                    "ok": json_path.exists(),
                    "path": str(json_path),
                    "error": None if json_path.exists() else f"{key} JSON not found",
                }
            # Calibrated XML (maize-specific)
            xml_path = DATA_DIR / f"{sp_name}_calibrated.xml"
            results["species_xml"] = {
                "ok": xml_path.exists(),
                "path": str(xml_path),
                "error": None if xml_path.exists() else "Calibrated XML not found",
            }
        else:
            results["species_config"] = {
                "ok": False, "path": "",
                "error": f"Unknown species '{sp_name}'",
            }

        # 8. Met CSV (if specified)
        if c.met_csv:
            met_path = Path(c.met_csv)
            results["met_csv"] = {
                "ok": met_path.exists(),
                "path": str(met_path),
                "error": None if met_path.exists() else "Met CSV not found",
            }

        return results

    @property
    def checkpoint_path(self) -> Path:
        """Return path to the checkpoint JSON for the current config mode."""
        out = Path(self.config.output_dir)
        if self.config.mode == "uniform_baseline":
            return out / "diurnal_uniform" / "production_checkpoint.json"
        elif self.config.mode == "carbon_feedback":
            return out / "diurnal_carbon" / "production_checkpoint.json"
        else:
            return out / "diurnal" / "production_checkpoint.json"

    def run(self) -> dict:
        """Execute pipeline per config.mode. Returns result dict."""
        c = self.config
        self._apply_env()

        # Check critical components
        validation = self.validate_system()
        critical = ["plantbox"]
        if c.mode not in ("uniform_baseline",):
            critical += ["dart_binary", "dart_license", "baleno_python"]
        critical += ["pytools4dart", "pvlib"]

        failures = [name for name in critical
                    if name in validation and not validation[name]["ok"]]
        if failures:
            msgs = [f"  {n}: {validation[n]['error']}" for n in failures]
            raise RuntimeError(
                "System validation failed for critical components:\n"
                + "\n".join(msgs)
            )

        # Inject custom daily met into DVS cache if provided
        if c.met_daily_csv:
            from .carbon.dvs_partitioning import load_daily_met
            import dart.coupling.carbon.dvs_partitioning as dvs_mod
            custom_daily = load_daily_met(c.met_daily_csv)
            existing = dvs_mod._DEFAULT_DAILY_MET or {}
            merged = dict(existing)
            merged.update(custom_daily)
            dvs_mod._DEFAULT_DAILY_MET = merged

        # Lazy import — env vars are already set
        from .photosynthesis import diurnal as _diurnal_mod
        from .photosynthesis.diurnal import (
            run_single_day,
            run_single_day_with_carbon,
            run_production_series,
            run_production_series_carbon,
        )

        # Patch scene geometry and location from config
        _diurnal_mod.LAT = c.lat
        _diurnal_mod.LON = c.lon
        _diurnal_mod.SOWING_DATE = c.sowing_date
        _diurnal_mod.SCENE_SIZE = [c.scene_size_x, c.scene_size_y]
        _diurnal_mod.GRID_NX = c.grid_nx
        _diurnal_mod.GRID_NY = c.grid_ny
        _diurnal_mod.GRID_SPACING_X = c.grid_spacing_x
        _diurnal_mod.GRID_SPACING_Y = c.grid_spacing_y
        _diurnal_mod.N_PLANTS = c.grid_nx * c.grid_ny
        _diurnal_mod.CENTER_PLANT_IDX = (c.grid_nx * c.grid_ny) // 2

        self._notify("start", f"Pipeline starting: mode={c.mode}")

        if c.mode == "single_day":
            if c.single_day is None:
                raise ValueError("mode='single_day' requires single_day to be set")
            if c.with_carbon:
                result = run_single_day_with_carbon(
                    c.single_day, use_dart=True,
                    timestep_min=c.timestep_min,
                    enable_baleno=c.enable_baleno,
                    met_csv=c.met_csv,
                    iterate_gs=c.iterate_gs,
                    gs_max_iterations=c.gs_max_iterations,
                    gs_tolerance=c.gs_tolerance,
                    gs_damping_alpha=c.gs_damping_alpha,
                    carbon_method=c.carbon_method,
                    with_sif=c.with_sif,
                    with_dart_f=c.with_dart_f,
                    sif_triangles=c.sif_triangles,
                )
            else:
                result = run_single_day(
                    c.single_day, use_dart=True,
                    timestep_min=c.timestep_min,
                    enable_baleno=c.enable_baleno,
                    met_csv=c.met_csv,
                    iterate_gs=c.iterate_gs,
                    gs_max_iterations=c.gs_max_iterations,
                    gs_tolerance=c.gs_tolerance,
                    gs_damping_alpha=c.gs_damping_alpha,
                    with_sif=c.with_sif,
                    with_dart_f=c.with_dart_f,
                    sif_triangles=c.sif_triangles,
                )

        elif c.mode == "uniform_baseline":
            if c.single_day is not None and not c.growth_days:
                result = run_single_day(
                    c.single_day, use_dart=False,
                    timestep_min=c.timestep_min,
                    met_csv=c.met_csv,
                    with_sif=c.with_sif,
                    with_dart_f=c.with_dart_f,
                    sif_triangles=c.sif_triangles,
                )
            else:
                result = run_production_series(
                    c.growth_days, use_dart=False,
                    timestep_min=c.timestep_min,
                    carbon_method=c.carbon_method,
                    run_agroc_fortran=c.with_agroc,
                    resume=c.resume,
                    with_sif=c.with_sif,
                    with_dart_f=c.with_dart_f,
                    sif_triangles=c.sif_triangles,
                )

        elif c.mode == "carbon_feedback":
            result = run_production_series_carbon(
                c.growth_days,
                timestep_min=c.timestep_min,
                enable_baleno=c.enable_baleno,
                iterate_gs=c.iterate_gs,
                gs_max_iterations=c.gs_max_iterations,
                gs_tolerance=c.gs_tolerance,
                gs_damping_alpha=c.gs_damping_alpha,
                carbon_method=c.carbon_method,
                run_agroc_fortran=c.with_agroc,
                resume=c.resume,
                with_sif=c.with_sif,
                with_dart_f=c.with_dart_f,
                sif_triangles=c.sif_triangles,
            )

        elif c.mode == "full_production":
            result = run_production_series(
                c.growth_days, use_dart=True,
                timestep_min=c.timestep_min,
                enable_baleno=c.enable_baleno,
                iterate_gs=c.iterate_gs,
                gs_max_iterations=c.gs_max_iterations,
                gs_tolerance=c.gs_tolerance,
                gs_damping_alpha=c.gs_damping_alpha,
                carbon_method=c.carbon_method,
                run_agroc_fortran=c.with_agroc,
                resume=c.resume,
                with_sif=c.with_sif,
                with_dart_f=c.with_dart_f,
                sif_triangles=c.sif_triangles,
            )

        else:
            raise ValueError(f"Unknown pipeline mode: {c.mode!r}")

        self._notify("done", "Pipeline complete")
        return result
