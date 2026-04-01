"""Species-specific configuration for the fitting pipeline.

Each species defines leaf count, subtype mapping, parameter bounds,
and position-dependent defaults. The optimizer code is species-agnostic —
it reads everything from the config.
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


LEAF_PARAMS = [
    'lmax', 'Width_blade', 'theta', 'tropismS', 'tropismAge',
    'r', 'collarLength', 'initBeta',
    'kappa_base', 'kappa_mid', 'kappa_tip',
]


@dataclass
class SpeciesConfig:
    """Configuration for a plant species."""

    name: str
    n_positions: int                  # number of leaf positions
    subtype_offset: int               # leaf subType = position + offset
    template_xml: str | None = None   # CPlantBox XML template path

    # Stem defaults
    stem_ln: float = 14.5
    stem_tropismS: float = 0.002
    stem_lnf: float = 0.0

    # Stem bounds
    stem_ln_bounds: tuple = (5.0, 25.0)
    stem_tropismS_bounds: tuple = (0.0, 0.02)
    stem_lnf_bounds: tuple = (0.0, 5.0)

    # Theta bounds (species-specific)
    theta_lo: float = 0.2
    theta_hi: float = 0.85

    # Position-dependent defaults (callables or fixed values)
    # theta: lower→upper gradient
    theta_lower: float = 0.6         # theta at lowest leaf
    theta_upper: float = 0.4         # theta at highest leaf

    # tropismAge: lower→upper gradient
    tropismAge_lower: float = 3.0
    tropismAge_upper: float = 8.0

    # Default scalar values
    default_collarLength: float = 10.0
    default_initBeta: float = 0.2
    default_tropismS: float = 0.03
    default_r: float = 3.0

    # Leaf size bounds (multipliers on per-position stats)
    lmax_lo_mult: float = 0.5
    lmax_hi_mult: float = 1.8
    lmax_min: float = 5.0             # absolute minimum lmax (cm)
    width_lo_mult: float = 0.3
    width_hi_mult: float = 2.5
    width_min: float = 0.3            # absolute minimum width (cm)
    r_lo_mult: float = 0.3
    r_hi_mult: float = 3.0
    r_min: float = 0.3

    # Tropism bounds
    tropismS_bounds: tuple = (0.001, 0.1)
    tropismAge_max_mult: float = 2.0
    tropismAge_min: float = 1.0
    tropismAge_max_abs: float = 15.0

    # Collar bounds
    collarLength_bounds: tuple = (0.0, 30.0)

    # Curvature spline bounds
    kappa_base_bounds: tuple = (0.0, 0.05)
    kappa_mid_bounds: tuple = (0.0, 0.15)
    kappa_tip_bounds: tuple = (0.0, 0.25)

    def default_leaf_params(self, stats_pos: dict, position: int = 0) -> dict:
        """Position-dependent default params for one leaf."""
        pos_frac = position / max(self.n_positions - 1, 1)

        theta = self.theta_lower + (self.theta_upper - self.theta_lower) * pos_frac
        trop_age = self.tropismAge_lower + (self.tropismAge_upper - self.tropismAge_lower) * pos_frac

        return {
            'lmax': float(stats_pos.get('lmax', 30.0)),
            'Width_blade': float(stats_pos.get('Width_blade', 2.0)),
            'theta': theta,
            'tropismS': float(stats_pos.get('tropismS', self.default_tropismS)),
            'tropismAge': trop_age,
            'r': float(stats_pos.get('r', self.default_r)),
            'collarLength': self.default_collarLength,
            'initBeta': self.default_initBeta,
            'kappa_base': 0.0,
            'kappa_mid': 0.0,
            'kappa_tip': 0.0,
        }

    def leaf_bounds(self, stats_pos: dict) -> tuple:
        """Return (x0, lo, hi) arrays for one leaf's CMA-ES."""
        default = self.default_leaf_params(stats_pos)
        x0 = [default[k] for k in LEAF_PARAMS]

        lmax = default['lmax']
        width = default['Width_blade']
        r = default['r']
        tage = default['tropismAge']

        lo = [
            max(lmax * self.lmax_lo_mult, self.lmax_min),
            max(width * self.width_lo_mult, self.width_min),
            self.theta_lo,
            self.tropismS_bounds[0],
            self.tropismAge_min,
            max(r * self.r_lo_mult, self.r_min),
            self.collarLength_bounds[0],
            -3.14,  # initBeta
            self.kappa_base_bounds[0],
            self.kappa_mid_bounds[0],
            self.kappa_tip_bounds[0],
        ]
        hi = [
            lmax * self.lmax_hi_mult,
            width * self.width_hi_mult,
            self.theta_hi,
            self.tropismS_bounds[1],
            min(max(tage * self.tropismAge_max_mult, self.tropismAge_max_abs), 30.0),
            r * self.r_hi_mult,
            self.collarLength_bounds[1],
            3.14,  # initBeta
            self.kappa_base_bounds[1],
            self.kappa_mid_bounds[1],
            self.kappa_tip_bounds[1],
        ]

        return np.array(x0), np.array(lo), np.array(hi)


# ============ Pre-built species configs ============

MAIZE = SpeciesConfig(
    name='maize',
    n_positions=11,
    subtype_offset=2,       # subType = position + 2
    stem_ln=14.5,
    stem_tropismS=0.002,
    theta_lo=0.2,
    theta_hi=0.85,
    theta_lower=0.6,        # lower leaves wider
    theta_upper=0.4,        # upper leaves more vertical
    tropismAge_lower=3.0,
    tropismAge_upper=8.0,
    default_collarLength=10.0,
    default_r=3.0,
    lmax_min=20.0,
    width_min=1.0,
    collarLength_bounds=(0.0, 30.0),
)

WHEAT = SpeciesConfig(
    name='wheat',
    n_positions=8,
    subtype_offset=2,       # subType = position + 2 (subtypes 2-9)
    template_xml=str(Path(__file__).resolve().parents[2] / 'data' / 'wheat_calibrated.xml'),
    stem_ln=7.5,            # (63 - 2) / 8 internodes
    stem_tropismS=0.01,     # upright stem
    stem_lnf=0.0,
    stem_ln_bounds=(4.0, 12.0),
    stem_tropismS_bounds=(0.0, 0.03),
    theta_lo=0.08,          # very erect
    theta_hi=0.5,           # max ~30°
    theta_lower=0.35,       # lower leaves slightly wider
    theta_upper=0.15,       # upper leaves very erect
    tropismAge_lower=5.0,
    tropismAge_upper=12.0,  # wheat leaves stay straight longer
    default_collarLength=3.0,  # shorter collar
    default_tropismS=0.015,    # less droop
    default_r=2.0,             # slower growth
    lmax_min=5.0,              # wheat leaves can be short
    width_min=0.3,             # narrow leaves
    width_lo_mult=0.3,
    width_hi_mult=3.0,
    collarLength_bounds=(0.0, 10.0),
    kappa_tip_bounds=(0.0, 0.15),  # less tip droop
)

# Registry
SPECIES = {
    'maize': MAIZE,
    'wheat': WHEAT,
}


def get_species(name: str) -> SpeciesConfig:
    """Get species config by name."""
    if name not in SPECIES:
        raise ValueError(f"Unknown species '{name}'. Available: {list(SPECIES.keys())}")
    return SPECIES[name]
