"""Growth subpackage: plant growth and XML calibration."""

from .grow import (
    grow_plant, init_plant, extract_g3_mesh, run_photosynthesis,
    extract_lai_profile, export_lai_csv, plot_lai_profile,
    extract_plant_summary, plot_growth_trajectory,
)
from .calibrate import main as calibrate_main
from .carbon_growth import (
    enable_cw_limited_growth, inject_cw_gr, step_plant_carbon,
)
