"""Environment configuration for CPlantBox-DART coupling.

Centralises all external paths and default file references.
Override via environment variables for portability.

Species selection:
  Set COUPLING_SPECIES env var (default "maize") or use --species CLI flag.
  get_species() returns the active species parameter dict.
"""

import os
from pathlib import Path

# Package directories
PACKAGE_DIR = Path(__file__).resolve().parent
DATA_DIR = PACKAGE_DIR / "data"
OUTPUT_DIR = PACKAGE_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# External tool paths — override via environment variables
DART_HOME = Path(os.environ.get("DART_HOME", "/home/lukas/DART"))
DART_EB_DIR = DART_HOME / "bin" / "python_script" / "dart-eb-main"
DARTRC = Path(os.environ.get("DARTRC", str(Path.home() / ".dartrcv1457")))
BALENO_PYTHON = Path(os.environ.get(
    "BALENO_PYTHON", "/home/lukas/PHD/darteb_venv/bin/python3.12"
))

# DART thread count — override via DART_THREADS env var or --threads CLI flag.
# Default 8 is conservative; set higher on multi-core servers (e.g. 64, 128).
from multiprocessing import cpu_count as _cpu_count
DART_THREADS = min(int(os.environ.get("DART_THREADS", "8")), _cpu_count())

# CPlantBox paths
CPLANTBOX_ROOT = Path(os.environ.get(
    "CPLANTBOX_ROOT", str(PACKAGE_DIR.parent.parent.parent)
))
HYDRAULICS_PATH = str(CPLANTBOX_ROOT / "modelparameter" / "functional" / "plant_hydraulics") + "/"

# ---------------------------------------------------------------------------
# Species registry — photosynthesis type, parameters, file references
# ---------------------------------------------------------------------------
SPECIES_REGISTRY = {
    "maize": {
        "photo_type": "C4",
        "photo_type_code": 1,        # CPlantBox PhotoType enum
        "rd_per_vcmax25": 0.025,     # Rd/Vcmax25 (C4 Bonan 2019)
        "ci_ca_ratio": 0.4,          # typical Ci/Ca for C4
        "alpha": 0.05,               # quantum yield (C4)
        "a1": 4.0,                   # Tuzet stomatal slope
        "a3": 10.0,                  # Jmax/Vcmax ratio (not used for C4)
        "theta": 0.9,                # light response curvature (not used for C4)
        "vcmax_chl1": 0.64,          # Vcmax-Chl slope
        "vcmax_chl2": 4.165,         # Vcmax-Chl intercept
        "hydraulics": "maize_couvreur2012_hydraulics",
        "photosynthesis": "maize_C4_photosynthesis_parameters",
    },
    "wheat": {
        "photo_type": "C3",
        "photo_type_code": 0,
        "rd_per_vcmax25": 0.015,     # Rd/Vcmax25 (C3 Bonan 2019)
        "ci_ca_ratio": 0.7,          # typical Ci/Ca for C3
        "alpha": 0.4,                # quantum yield (C3, Giraud2023)
        "a1": 0.5,                   # Tuzet stomatal slope (Giraud2023)
        "a3": 1.5,                   # Jmax/Vcmax ratio (Giraud2023)
        "theta": 0.6,                # light response curvature (Giraud2023)
        "vcmax_chl1": 1.28,          # Vcmax-Chl slope (Giraud2023)
        "vcmax_chl2": 8.33,          # Vcmax-Chl intercept (Giraud2023)
        "hydraulics": "wheat_Giraud2023adapted",
        "photosynthesis": "wheat_C3_photosynthesis_parameters",
    },
}


def get_species() -> dict:
    """Return active species config dict from SPECIES_REGISTRY.

    Reads COUPLING_SPECIES env var (default "maize").
    """
    name = os.environ.get("COUPLING_SPECIES", "maize").lower()
    if name not in SPECIES_REGISTRY:
        raise ValueError(
            f"Unknown species '{name}'. Available: {list(SPECIES_REGISTRY.keys())}"
        )
    return SPECIES_REGISTRY[name]


def get_species_name() -> str:
    """Return the active species name string."""
    return os.environ.get("COUPLING_SPECIES", "maize").lower()


# ---------------------------------------------------------------------------
# Default plant config files (bundled in data/)
# Computed dynamically from active species.
# ---------------------------------------------------------------------------
def _species_hydraulics_json() -> str:
    sp = get_species()
    return str(DATA_DIR / sp["hydraulics"])  # no .json (CPlantBox appends)


def _species_photosynthesis_json() -> str:
    sp = get_species()
    return str(DATA_DIR / sp["photosynthesis"])  # no .json (CPlantBox appends)


DEFAULT_XML = DATA_DIR / "maize_calibrated.xml"
DEFAULT_TEMPLATE_XML = CPLANTBOX_ROOT / "modelparameter" / "structural" / "plant" / "maize.xml"
PHOTO_PATH = str(DATA_DIR) + "/"  # trailing slash for CPlantBox API compatibility


def get_hydraulics_json() -> str:
    """Return path to active species' hydraulics JSON (no .json extension)."""
    return _species_hydraulics_json()


def get_photosynthesis_json() -> str:
    """Return path to active species' photosynthesis JSON (no .json extension)."""
    return _species_photosynthesis_json()


# Module-level variables — evaluated at import time.
# Since __main__.py sets COUPLING_SPECIES before importing subcommands,
# these reflect the correct species when used via the CLI.
HYDRAULICS_JSON = _species_hydraulics_json()
PHOTOSYNTHESIS_JSON = _species_photosynthesis_json()

# MaizeField3D data
MAIZEFIELD3D_STATS = DATA_DIR / "maizefield3d_stats.json"
MAIZEFIELD3D_DEFORMATION = DATA_DIR / "maizefield3d_blade_deformation.json"
MAIZEFIELD3D_STEM_PROFILE = DATA_DIR / "maizefield3d_stem_profile.json"
