# CPlantBox-DART Coupling Pipeline

Couples [CPlantBox](https://plant-root-soil-interactions-modelling.github.io/CPlantBox/) (3D functional-structural plant model) with [DART](https://dart.omp.eu/) (discrete anisotropic radiative transfer) for spatially-resolved photosynthesis simulation. Per-triangle absorbed PAR and leaf temperature drive FvCB carbon assimilation and Munch-transport carbon partitioning. Supports maize (C4) and wheat (C3). Developed for PhenoRob Core Project 6, University of Bonn.

## Architecture

```
                    CPlantBox                                 DART
              ┌─────────────────┐                    ┌──────────────────┐
              │  1. Growth      │                    │  3. RT           │
              │  Parametric XML │                    │  5-band PAR      │
              │  → G1 skeleton  │                    │  3x3 plant grid  │
              └───────┬─────────┘                    │  → per-tri aPAR  │
                      │                              └────────┬─────────┘
              ┌───────▼─────────┐                             │
              │  2. Geometry    │         OBJ                 │
              │  G1 → G3 mesh  ├──────────────────────────────┘
              │  Baker lofting  │
              └───────┬─────────┘
                      │                              ┌──────────────────┐
                      │                              │  4. Baleno EB    │
                      │                              │  per-tri Tleaf   │
                      │                              │  Rn, LE, H       │
                      │                              └────────┬─────────┘
              ┌───────▼─────────────────────────────────────────▼────────┐
              │  5. Photosynthesis                                      │
              │  FvCB + 3D aPAR + Tleaf → per-segment An               │
              ├─────────────────────────────────────────────────────────┤
              │  6. Iterative gs (Tuzet ↔ Baleno, ~5 iterations)       │
              └───────┬─────────────────────────────────────────────────┘
                      │
              ┌───────▼─────────┐              ┌──────────────────┐
              │  7. Carbon      │              │  8. AgroC        │
              │  Quasi-steady   │──────────────│  One-way GPP/RLD │
              │  Münch phloem   │  coupling.csv│  → soil C pools  │
              └─────────────────┘              └──────────────────┘
```

1. **Growth** -- CPlantBox grows a plant from calibrated XML parameters (MaizeField3D-derived), producing a G1 skeleton with per-organ widths and ages.
2. **Geometry** -- Baker quad-lofting converts G1 skeletons to G3 triangle meshes (OBJ) with UV-mapped organ IDs, realistic blade deformation (undulation, twist, curl), and pointed tips.
3. **DART RT** -- Radiative transfer over a 3x3 plant grid in 5 PAR bands. Per-triangle aPAR extracted via DART ObjectFields, mapped back to CPlantBox segments.
4. **Baleno EB** -- Triangle-level energy balance computes Tleaf, net radiation, latent/sensible heat. Replaces the Tleaf=Tair assumption.
5. **Photosynthesis** -- FvCB model (C4 Bonan 2019 or C3 Farquhar) with spatially-resolved aPAR and Tleaf per leaf segment.
6. **Iterative gs** -- Tuzet stomatal conductance iterated with Baleno: gs affects Tleaf via energy balance, Tleaf affects gs via photosynthesis. Converges in ~5 iterations (alpha=0.6 damping).
7. **Carbon** -- Quasi-steady Münch phloem solver allocates assimilate to leaf, stem, and root growth sinks with DVS-dependent partitioning.
8. **AgroC** -- One-way export of GPP, root length density, root respiration, and exudation to AgroC Fortran for soil biogeochemistry (RothC pools, NEE). No feedback to CPlantBox.

## Quick Start

### Prerequisites

- Python 3.12+
- `plantbox` (CPlantBox Python module)
- DART installation with valid license
- `pytools4dart`, `pvlib`, `numpy`, `scipy`, `pandas`
- Baleno venv (Python 3.12, separate from CPlantBox env)

### Setup

```bash
source /path/to/CPlantBox/cpbenv/bin/activate
cd /path/to/CPlantBox
```

### Verify installation

```bash
python3 -m dart.coupling run pipeline_config.json --validate-only
```

### First run (no DART needed)

```bash
python3 -m dart.coupling diurnal --days 55 --uniform --with-carbon
```

### First run with full 3D radiative transfer

```bash
python3 -m dart.coupling diurnal --days 55 --iterate-gs --with-carbon
```

## CLI Reference

Invoke via `python3 -m dart.coupling <subcommand>`. Global options: `--species {maize,wheat}`, `--threads N`.

### Core Pipeline

| Subcommand | Description |
|------------|-------------|
| `simulation --day N` | DART radiative transfer for a single growth day |
| `baleno` | Standalone Baleno energy balance |
| `photosynthesis` | Single-step coupled photosynthesis |
| `validate` | Coupling validation checks |
| `diurnal` | Diurnal coupling loop (main production entry point) |

### Growth

| Subcommand | Description |
|------------|-------------|
| `calibrate` | Calibrate maize XML from MaizeField3D data |
| `grow` | Grow calibrated plant to specified day |

### Analysis

| Subcommand | Description |
|------------|-------------|
| `rld --day N` | Root length density profiles |
| `carbon --day N` | Carbon partitioning analysis |
| `summary --day N` | LAI + whole-plant summary |

### Soil Coupling

| Subcommand | Description |
|------------|-------------|
| `agroc-export --day N` | Generate AgroC coupling CSV and profiles |
| `agroc-run --coupling-csv PATH` | Run AgroC Fortran with ExternalPlantMode |

### Testing

| Subcommand | Description |
|------------|-------------|
| `integration-test --day N` | Full pipeline integration test (6 tasks) |

### Config-Driven

| Subcommand | Description |
|------------|-------------|
| `create-config [PATH]` | Generate default pipeline config JSON |
| `run CONFIG.json` | Run pipeline from config file |

### Dashboard

| Subcommand | Description |
|------------|-------------|
| `dashboard --port 8050` | Launch web dashboard |

### Key `diurnal` Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--days N` | -- | Single growth day |
| `--growth-days "10,14,...,58"` | -- | Multi-day production series |
| `--timestep-min N` | 30 | Timestep in minutes |
| `--iterate-gs` | off | Enable Tuzet-Baleno gs iteration |
| `--with-carbon` | off | Enable carbon partitioning |
| `--with-agroc` | off | Enable AgroC Fortran coupling |
| `--uniform` | off | Skip DART/Baleno (clearsky PAR + Tleaf=Tair) |
| `--resume` | off | Resume from checkpoint |
| `--no-baleno` | off | Skip Baleno energy balance |
| `--with-sif` | off | Enable SIF computation |
| `--carbon-method` | auto | Partitioning method: auto, phloem, dvs |
| `--growth-mode` | parametric | Growth mode: parametric or carbon |
| `--met-csv PATH` | -- | Custom meteorological forcing CSV |

## Common Workflows

### 1. Single-Day Exploration

```bash
python3 -m dart.coupling calibrate
python3 -m dart.coupling grow --day 55
python3 -m dart.coupling simulation --day 55
python3 -m dart.coupling photosynthesis
python3 -m dart.coupling summary --day 55
```

### 2. Production 3D Multi-Day

```bash
python3 -m dart.coupling diurnal \
    --growth-days "10,14,18,22,26,30,35,40,45,50,55,58" \
    --iterate-gs --with-carbon --resume
```

### 3. Uniform Baseline (Chapter 1 comparison)

```bash
python3 -m dart.coupling diurnal \
    --growth-days "10,14,18,22,26,30,35,40,45,50,55,58" \
    --with-carbon --uniform --resume
```

The difference between workflows 2 and 3 quantifies the value of 3D radiative transfer over a uniform PAR assumption.

## Configuration

### CLI-Driven vs Config-Driven

The CLI is sufficient for most runs. For reproducible experiments or dashboard-launched runs, use config files:

```bash
python3 -m dart.coupling create-config my_run.json   # generate template
# edit my_run.json
python3 -m dart.coupling run my_run.json              # execute
python3 -m dart.coupling run my_run.json --validate-only  # check setup
```

### PipelineConfig Categories

| Category | Key fields |
|----------|------------|
| Run mode | `mode` (full_production, uniform_baseline, carbon_feedback, single_day) |
| Temporal | `growth_days`, `single_day`, `timestep_min` |
| Scene | `lat`, `lon`, `sowing_date`, `scene_size_x/y`, `grid_nx/ny`, `grid_spacing_x/y` |
| Soil/water | `soil_psi_cm` (default -500 = well-watered) |
| Physics | `enable_baleno`, `iterate_gs`, `gs_max_iterations`, `gs_tolerance`, `gs_damping_alpha` |
| Carbon | `with_carbon`, `carbon_method`, `with_agroc` |
| SIF | `with_sif`, `with_dart_f`, `sif_triangles` |
| DART tuning | `threads`, `dart_ray_density`, `dart_max_rendering_time` |
| Paths | `dart_home`, `dartrc`, `baleno_python`, `cplantbox_root`, `output_dir` |
| PROSPECT | `prospect_stages_override`, `vcmax_chl1_override`, `vcmax_chl2_override` |

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DART_HOME` | `/home/lukas/DART` | DART installation directory |
| `DARTRC` | `~/.dartrcv1457` | DART license file |
| `BALENO_PYTHON` | `darteb_venv/bin/python3.12` | Python for Baleno subprocess |
| `CPLANTBOX_ROOT` | auto-detected | CPlantBox repository root |
| `COUPLING_SPECIES` | `maize` | Active species (maize or wheat) |
| `DART_THREADS` | `8` | DART thread count |
| `DART_RAY_DENSITY` | `50` | LuxCore ray density per pixel |
| `DART_MAX_RENDERING_TIME` | `0` | LuxCore render timeout (0 = unlimited) |
| `AGROC_SRC` | auto-detected | AgroC source directory |
| `GROWTH_MODE` | `parametric` | Growth mode (parametric or carbon) |

## Directory Structure

```
coupling/
├── __main__.py              # CLI entry point (17 subcommands)
├── config.py                # Environment config, species registry, path resolution
├── pipeline.py              # PipelineConfig dataclass + PipelineRunner
├── prospect_params.py       # PROSPECT biochemistry (Cab → Vcmax bridge)
│
├── growth/                  # Plant growth
│   ├── calibrate.py         #   MaizeField3D → calibrated XML
│   ├── grow.py              #   Run CPlantBox growth simulation
│   ├── carbon_growth.py     #   Carbon-feedback growth mode
│   ├── profiles.py          #   RLD extraction, LAI summary
│   └── render.py            #   Growth visualization
│
├── geometry/                # G1 → G3 mesh conversion
│   ├── g1_to_g3.py          #   Baker quad-lofting with blade deformation
│   ├── cplantbox_adapter.py #   CPlantBox plant → organ dicts
│   ├── pheno4d_adapter.py   #   Pheno4D morphology → organ dicts
│   └── obj_dart_converter.py#   OBJ ↔ DART format conversion
│
├── dart/                    # DART simulation interface
│   ├── simulation.py        #   RT setup, aPAR extraction, multifield
│   ├── baleno.py            #   Baleno energy balance integration
│   ├── baleno_standalone.py #   Standalone Baleno runner
│   ├── dart_f.py            #   DART fluorescence support
│   ├── parsers.py           #   DART output file parsers
│   └── diagnose_tleaf.py    #   Tleaf diagnostic utilities
│
├── photosynthesis/          # Carbon assimilation
│   ├── coupled.py           #   Single-step coupled photosynthesis
│   ├── diurnal.py           #   Diurnal loop (main production driver)
│   └── iterative.py         #   Tuzet-Baleno gs iteration
│
├── carbon/                  # Carbon partitioning
│   ├── phloem_steady.py     #   Quasi-steady Münch phloem solver
│   ├── dvs_partitioning.py  #   DVS-based partitioning (WOFOST-style)
│   ├── tree_topology.py     #   Plant graph topology utilities
│   └── cli.py               #   Carbon CLI handler
│
├── agroc/                   # AgroC soil coupling
│   ├── export.py            #   Generate coupling.csv + profiles
│   ├── run.py               #   Run AgroC Fortran executable
│   ├── profiles.py          #   Soil profile extraction
│   └── unit_conversions.py  #   Unit conversion helpers
│
├── sif/                     # Solar-induced fluorescence
│   ├── sif_analysis.py      #   SIF analysis utilities
│   └── sif_writer.py        #   SIF output writers
│
├── validation/              # Pipeline validation
│   ├── validate.py          #   Coupling validation suite
│   └── verify_alignment.py  #   Geometry alignment checks
│
├── utils/                   # Shared utilities
│   ├── met_forcing.py       #   Meteorological data loading
│   └── solar_position.py    #   Solar geometry calculations
│
├── tests/                   # Test suite
│   ├── test_session8_integration.py  # Full integration test
│   ├── test_session1_root_shoot.py
│   ├── test_session2_rld.py
│   └── test_session3_phloem_swap.py
│
├── data/                    # Species parameters and calibration data
│   ├── maize_calibrated.xml
│   ├── maize_C4_photosynthesis_parameters.json
│   ├── maize_couvreur2012_hydraulics.json
│   ├── phloem_parameters_maize2026.json
│   ├── wheat_C3_photosynthesis_parameters.json
│   ├── wheat_Giraud2023adapted.json
│   ├── wheat_phloem_parameters.json
│   ├── maizefield3d_stats.json
│   ├── maizefield3d_blade_deformation.json
│   ├── maizefield3d_stem_profile.json
│   ├── lops_prospect_profiles.json
│   └── juelich_2024_daily_met.csv
│
└── output/                  # Pipeline outputs (generated)
    ├── diurnal/             #   3D production results
    ├── diurnal_uniform/     #   Uniform baseline results
    └── diurnal_carbon/      #   Carbon-feedback results
```

## Output Layout

Each `diurnal/` or `diurnal_uniform/` directory contains:

```
diurnal/
├── production_checkpoint.json    # Resume state (completed days + timesteps)
├── day_010/
│   ├── growth/                   # Plant OBJ meshes, skeleton data
│   ├── dart/                     # DART simulation files, aPAR results
│   ├── baleno/                   # Energy balance outputs (Tleaf, fluxes)
│   ├── photosynthesis/           # Per-segment An, gs time series
│   ├── carbon/                   # Partitioning results, phloem flows
│   └── summary/                  # Aggregated metrics (LAI, GPP, etc.)
├── day_014/
│   └── ...
└── summaries/                    # Cross-day comparison plots and CSVs
```

## Dashboard

```bash
python3 -m dart.dashboard --port 8050
```

Six pages: **System** (component status), **Simulation** (DART config), **Meteorology** (forcing data), **Runner** (launch/monitor runs), **Outputs** (results browser), **Viewer3D** (interactive mesh viewer).

## Deployment: Local vs Server

| | Local | Server |
|---|---|---|
| **Path** | `/home/lukas/PHD/CPlantBox/` | `/media/data/Lukas/CPlantBox/` |
| **Python** | 3.14 | 3.12 |
| **DART** | `/home/lukas/DART` | `/media/data/Lukas/DART/` |
| **Purpose** | Development | Production runs |
| **pytools4dart** | Repo clone | pip install |

Override paths via environment variables (see table above) or config JSON. `config.py` resolves defaults automatically.

## Testing

```bash
# Smoke test (no DART, ~30s)
python3 -m dart.coupling integration-test --day 55 --skip-dart --skip-agroc

# Full validation (requires DART + Baleno, ~10 min)
python3 -m dart.coupling integration-test --day 55

# Validate system components only
python3 -m dart.coupling run pipeline_config.json --validate-only
```

## Species Support

| Species | Photosynthesis | Status | Selection |
|---------|---------------|--------|-----------|
| Maize | C4 (Bonan 2019) | Primary, fully calibrated | `--species maize` (default) |
| Wheat | C3 (Farquhar, Giraud 2023) | Secondary | `--species wheat` |

Species selection applies globally: calibrated XML, hydraulics, photosynthesis parameters, phloem transport, and PROSPECT Cab-Vcmax bridge.

## Troubleshooting

| Issue | Fix |
|-------|-----|
| DART license error | Set `DARTRC` to point to your `.dartrc` file |
| All-triangle Tleaf collapses to ~21C | Increase `DART_RAY_DENSITY` to 500 (default 50 is too low for energy balance) |
| LuxCore uses all CPU cores | Use `--threads N` to set CPU affinity (LuxCore ignores XML nbThreads) |
| `libreadline` conflict | Rename `DART/bin/python/lib/libreadline*` to `.DISABLED` |
| An = 0 everywhere | Check leaf kr in hydraulics JSON (must be > 0, e.g. 3.83e-05) |
| Interrupted run | Use `--resume` to continue from checkpoint |
