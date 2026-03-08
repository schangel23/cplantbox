"""Carbon partitioning subpackage: quasi-steady phloem transport and DVS fallback."""

from .phloem_steady import QuasiSteadyPhloem, solve_carbon_partitioning
from .dvs_partitioning import (
    partition_carbon_dvs, compute_gdd_day, accumulate_gdd, dvs_from_gdd,
    dvs_for_day, gdd_at_day, get_daily_met, load_daily_met,
)
from .cli import main_carbon
