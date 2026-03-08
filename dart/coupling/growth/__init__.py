"""Growth subpackage: plant growth, profiles, and rendering."""

from .grow import (
    grow_plant, init_plant, extract_g3_mesh, extract_root_dicts,
    setup_successor_where, run_photosynthesis,
    export_mesh, export_g1_skeleton,
)
from .profiles import (
    extract_rld_profile, export_rld_csv, export_rrd_in,
    plot_rld_profile, plot_rld_growth_trajectory,
    extract_lai_profile, export_lai_csv, plot_lai_profile,
    extract_plant_summary, plot_growth_trajectory,
    main_rld, main_summary,
)
from .render import (
    render_comparison_png, render_comparison_svg,
    render_publication_svg, render_animated_svg,
    plot_photosynthesis,
    LEAF_GREENS, STEM_COLOR,
)
from .calibrate import main as calibrate_main
from .carbon_growth import (
    enable_cw_limited_growth, inject_cw_gr, step_plant_carbon,
)
