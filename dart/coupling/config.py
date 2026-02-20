"""Environment configuration for CPlantBox-DART coupling.

Centralises all external paths and default file references.
Override via environment variables for portability.
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

# CPlantBox paths
CPLANTBOX_ROOT = Path(os.environ.get(
    "CPLANTBOX_ROOT", str(PACKAGE_DIR.parent.parent.parent)
))
HYDRAULICS_PATH = str(CPLANTBOX_ROOT / "modelparameter" / "functional" / "plant_hydraulics") + "/"

# Default plant config files (bundled in data/)
DEFAULT_XML = DATA_DIR / "maize_calibrated.xml"
DEFAULT_TEMPLATE_XML = CPLANTBOX_ROOT / "modelparameter" / "structural" / "plant" / "maize.xml"
HYDRAULICS_JSON = str(DATA_DIR / "maize_couvreur2012_hydraulics")   # no .json (CPlantBox appends)
PHOTOSYNTHESIS_JSON = str(DATA_DIR / "maize_C4_photosynthesis_parameters")  # no .json
PHOTO_PATH = str(DATA_DIR) + "/"  # trailing slash for CPlantBox API compatibility

# MaizeField3D data
MAIZEFIELD3D_STATS = DATA_DIR / "maizefield3d_stats.json"
MAIZEFIELD3D_DEFORMATION = DATA_DIR / "maizefield3d_blade_deformation.json"
MAIZEFIELD3D_STEM_PROFILE = DATA_DIR / "maizefield3d_stem_profile.json"
