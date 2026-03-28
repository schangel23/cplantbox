"""Latin Hypercube sampling of CPlantBox XML parameter space.

Loads MaizeField3D per-position statistics and samples parameter sets
for training data generation.  Bounds are centred on the measured medians
with +/- 50 % range for lengths/widths and +/- 50 % of the measured
inter-position range for angles.
"""

import json
import math
from pathlib import Path

import numpy as np
from scipy.stats.qmc import LatinHypercube

# Parameters sampled per leaf position (12 per position).
LEAF_PARAM_NAMES = [
    "lmax",
    "Width_blade",
    "theta",
    "tropismS",
    "tropismAge",
    "r",
    "areaMax",
    "wave_normal_amp",
    "twist_max",
    "curl_amp",
    "edge_ruffle_amp",
    "fold_amp",
]

# Global stem parameter.
STEM_PARAM_NAMES = ["stem_ln"]

# Total dimension of the flat param vector:
# 11 positions * 12 leaf params + 1 stem param = 133.
N_POSITIONS = 11
N_LEAF_PARAMS = len(LEAF_PARAM_NAMES)
N_PARAMS = N_POSITIONS * N_LEAF_PARAMS + len(STEM_PARAM_NAMES)


def load_prior_bounds(stats_path: str) -> dict:
    """Load MaizeField3D stats and compute per-position parameter bounds.

    Returns a dict mapping ``position`` (0-10) to
    ``{param_name: (low, high)}``, plus a top-level ``"stem_ln"`` entry.
    """
    with open(stats_path, "r") as f:
        data = json.load(f)

    per_pos = data["per_position"]

    # Collect thetas across positions for range-based angle bounds.
    all_theta = [p["theta"] for p in per_pos if p is not None]
    theta_range = max(all_theta) - min(all_theta) if len(all_theta) > 1 else 0.3

    bounds: dict = {}

    for entry in per_pos:
        if entry is None:
            continue
        pos = entry["position"]
        pb: dict = {}

        # --- Lengths / widths / areas: median * [0.5, 1.5] ---
        for key in ("lmax", "Width_blade", "areaMax"):
            med = entry[key]
            pb[key] = (med * 0.5, med * 1.5)

        # --- Growth rate: default 4.0 if not measured ---
        r_med = entry.get("r", 4.0)
        pb["r"] = (r_med * 0.5, r_med * 1.5)

        # --- tropismAge: median +/- 50 % ---
        ta_med = entry.get("tropismAge", 5.0)
        pb["tropismAge"] = (max(ta_med * 0.5, 1.0), ta_med * 1.5)

        # --- Angles: median +/- 50 % of cross-position range ---
        theta_med = entry["theta"]
        half_range = theta_range * 0.5
        pb["theta"] = (max(theta_med - half_range, 0.05), theta_med + half_range)

        # --- tropismS: median +/- 50 % ---
        ts_med = entry.get("tropismS", 0.15)
        pb["tropismS"] = (max(ts_med * 0.5, 0.01), ts_med * 1.5)

        # --- Deformation amplitudes (hand-tuned ranges) ---
        # wave_normal_amp: fraction of leaf length
        lmax = entry["lmax"]
        pb["wave_normal_amp"] = (lmax * 0.003, lmax * 0.02)

        # twist_max: radians
        pb["twist_max"] = (math.radians(5), math.radians(45))

        # curl_amp: cm
        pb["curl_amp"] = (0.2, 2.0)

        # edge_ruffle_amp: cm
        pb["edge_ruffle_amp"] = (0.3, 2.5)

        # fold_amp: cm
        pb["fold_amp"] = (0.1, 1.5)

        bounds[pos] = pb

    # Stem internode length
    bounds["stem_ln"] = (10.0, 20.0)

    return bounds


def _flatten_params(sample: dict) -> np.ndarray:
    """Flatten a single sample dict into a (N_PARAMS,) vector.

    Layout: [pos0_lmax, pos0_Width_blade, ..., pos0_fold_amp,
             pos1_lmax, ..., pos10_fold_amp,
             stem_ln]
    """
    vec = np.empty(N_PARAMS, dtype=np.float32)
    idx = 0
    for pos in range(N_POSITIONS):
        for pname in LEAF_PARAM_NAMES:
            vec[idx] = sample[pos][pname]
            idx += 1
    vec[idx] = sample["stem_ln"]
    return vec


def sample_params(
    n_samples: int,
    bounds: dict,
    seed: int = 42,
) -> list[dict]:
    """Latin Hypercube sample *n_samples* parameter sets.

    Each sample is a dict mapping position indices (0-10) to
    ``{param_name: value}`` dicts, plus a ``"stem_ln"`` float.

    Uses :class:`scipy.stats.qmc.LatinHypercube` for space-filling
    coverage of the parameter space.
    """
    sampler = LatinHypercube(d=N_PARAMS, seed=seed)
    unit_samples = sampler.random(n=n_samples)  # (n_samples, N_PARAMS) in [0, 1]

    samples: list[dict] = []

    for i in range(n_samples):
        sample: dict = {}
        idx = 0
        for pos in range(N_POSITIONS):
            pos_bounds = bounds[pos]
            pos_vals: dict = {}
            for pname in LEAF_PARAM_NAMES:
                lo, hi = pos_bounds[pname]
                pos_vals[pname] = float(lo + unit_samples[i, idx] * (hi - lo))
                idx += 1
            sample[pos] = pos_vals
        lo, hi = bounds["stem_ln"]
        sample["stem_ln"] = float(lo + unit_samples[i, idx] * (hi - lo))
        samples.append(sample)

    return samples
