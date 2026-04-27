"""Experimental fitting losses.

Two families of loss are exposed:

  * :mod:`.chamfer` — bidirectional Chamfer distance on triangle point
    clouds. Torch-native, GPU-compatible. Used by the original quad-ribbon
    fitter (``fit_lofter_params.py``, ``fit_to_reference.py``).
  * :mod:`.cp_distance` — CP-space L2 loss in the canonical NURBS grid
    (:mod:`dart.coupling.geometry.canonical_cp_grid`). Pure NumPy; direct
    leaf-pair comparison with a Hungarian matcher on centroid + arc +
    rank features. Used by the NURBS-parametric fitting pipeline.
"""

from .chamfer import chamfer_distance, chamfer_distance_batch
from .cp_distance import (
    cp_l2_loss,
    hungarian_leaf_match,
    leaf_arc_length,
    leaf_centroid,
    per_cp_distance,
)

__all__ = [
    "chamfer_distance",
    "chamfer_distance_batch",
    "cp_l2_loss",
    "hungarian_leaf_match",
    "leaf_arc_length",
    "leaf_centroid",
    "per_cp_distance",
]
