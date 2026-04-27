"""Canonical NURBS control-point grid for leaf representation.

Single source of truth for the shared leaf representation used by:
  - the NurbsPatch lofter backend (`nurbs_blade.py`)
  - the MaizeField3D canonical resampler (`maizefield3d_nurbs_reader.py`)
  - the Pheno4D LSQ fitter (`ply_to_nurbs.py`)
  - the CP-space L2 fitting loss (`losses/cp_distance.py`)

Canonical grid
--------------
Shape: ``(N_U=11, N_V=5, 3)``
  - U (axis 0) = arc-length along the leaf midrib: u=0 at collar, u=1 at tip.
  - V (axis 1) = across the leaf width: v=0/v=1 are edges, v=0.5 is the midrib.

Degrees
-------
  - DEG_U = 3 (cubic along midrib, continuous 2nd derivative — smooth curvature)
  - DEG_V = 2 (quadratic across width — matches MaizeField3D `.dat` deg=2 once
    axis names are aligned; see "Axis gotcha" below).

Knot vectors
------------
Clamped uniform knots: ``[0]*(deg+1)`` + evenly spaced interior knots in (0, 1) +
``[1]*(deg+1)``. Length = ``n_ctrl + deg + 1``. A 5×11 grid therefore has:
  - knots_u (11 CPs, deg 3): length 15, 7 interior knots
  - knots_v (5 CPs, deg 2): length 8, 2 interior knots at 1/3 and 2/3

Orientation convention
----------------------
To disambiguate the ``v=0 ↔ v=1`` mirror, we fix the rule:

  *v=0 edge is the one whose centroid has a larger signed projection onto
  the +x axis of the leaf-local frame.*

In practice the leaf-local frame is the SVD frame of the CP grid with +x the
principal in-plane direction orthogonal to the midrib. The helper
`enforce_orientation` computes the projection and flips the v-axis if needed.

Axis gotcha (MaizeField3D)
--------------------------
MaizeField3D `.dat` files store control points as ``(n_v=6, n_u=3, 3)`` with
``u`` as the **lateral** (across-width) direction and ``v`` as the
**longitudinal** (along-leaf) direction. Our canonical convention inverts
this: axis 0 is along-leaf (U, arc-length), axis 1 is across-width (V).
Callers converting MaizeField3D data must swap axes before feeding into the
canonical adapters (see `resample_to_canonical` in
`maizefield3d_nurbs_reader.py`).

PlantGL LD_LIBRARY_PATH mitigation
----------------------------------
PlantGL's C++ shared libraries are installed next to the Python package but
are not on the system linker's search path. `ensure_plantgl_loaded()` uses
ctypes to `RTLD_GLOBAL`-preload the four required ``.so`` files before the
first scenegraph import. Importing this module triggers that preload
automatically.
"""

from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path
import numpy as np

# ---------------------------------------------------------------------------
# Canonical dimensions and degrees
# ---------------------------------------------------------------------------
N_U: int = 11
N_V: int = 5
DEG_U: int = 3
DEG_V: int = 2

U_COLLAR: float = 0.0
U_TIP: float = 1.0
V_MIDRIB: float = 0.5


# ---------------------------------------------------------------------------
# PlantGL loader (ctypes preload; avoids LD_LIBRARY_PATH dependency)
# ---------------------------------------------------------------------------
_PLANTGL_LOADED = False
_PLANTGL_CANDIDATE_LIBDIRS = [
    # Local (python 3.14)
    "/home/lukas/PHD/CPlantBox/cpbenv/lib/python3.14/site-packages/lib",
    # Server (python 3.12) — whichever is present at import time
    "/media/data/Lukas/CPlantBox/cpbenv/lib/python3.12/site-packages/lib",
]
# Dependency order: tool → math → sg → algo. gui is skipped (broken/irrelevant).
_PLANTGL_LIB_NAMES = ["libpgltool.so", "libpglmath.so", "libpglsg.so", "libpglalgo.so"]


def _discover_plantgl_libdir() -> Path | None:
    """Locate the PlantGL shared-library directory."""
    env_override = os.environ.get("PLANTGL_LD_LIBRARY_PATH")
    if env_override:
        p = Path(env_override)
        if p.is_dir():
            return p
    for cand in _PLANTGL_CANDIDATE_LIBDIRS:
        p = Path(cand)
        if p.is_dir() and (p / _PLANTGL_LIB_NAMES[0]).exists():
            return p
    # Last resort: inspect sys.prefix for a site-packages/lib directory
    for site in (Path(sys.prefix), Path(sys.base_prefix)):
        for sp in site.rglob("site-packages/lib"):
            if (sp / _PLANTGL_LIB_NAMES[0]).exists():
                return sp
    return None


def ensure_plantgl_loaded() -> None:
    """Preload PlantGL shared libraries so `from openalea.plantgl.scenegraph
    import NurbsPatch` succeeds without setting LD_LIBRARY_PATH externally.

    Idempotent. Safe to call repeatedly.
    """
    global _PLANTGL_LOADED
    if _PLANTGL_LOADED:
        return
    libdir = _discover_plantgl_libdir()
    if libdir is None:
        # Let the import raise; the caller will see a clearer error.
        _PLANTGL_LOADED = True
        return
    for name in _PLANTGL_LIB_NAMES:
        lib_path = libdir / name
        if not lib_path.exists():
            continue
        try:
            ctypes.CDLL(str(lib_path), mode=ctypes.RTLD_GLOBAL)
        except OSError:
            # Already loaded or dependency issue — fall back silently.
            pass
    _PLANTGL_LOADED = True


# Preload on import so downstream `from openalea.plantgl.scenegraph import ...`
# calls work without per-call setup.
ensure_plantgl_loaded()


# ---------------------------------------------------------------------------
# Knot-vector builder
# ---------------------------------------------------------------------------
def build_uniform_knotvector(n_ctrl: int, degree: int) -> np.ndarray:
    """Clamped uniform knot vector of length ``n_ctrl + degree + 1``.

    Args:
        n_ctrl: number of control points along this axis.
        degree: polynomial degree.

    Returns:
        1-D ``float64`` array with ``degree+1`` zeros, ``n_ctrl-degree-1``
        evenly spaced interior knots in (0, 1), and ``degree+1`` ones.
    """
    if n_ctrl <= degree:
        raise ValueError(
            f"Need n_ctrl > degree; got n_ctrl={n_ctrl}, degree={degree}"
        )
    n_interior = n_ctrl - degree - 1
    if n_interior > 0:
        interior = np.arange(1, n_interior + 1, dtype=np.float64) / (n_interior + 1)
    else:
        interior = np.empty(0, dtype=np.float64)
    return np.concatenate(
        [np.zeros(degree + 1, dtype=np.float64), interior, np.ones(degree + 1, dtype=np.float64)]
    )


def canonical_knots() -> tuple[np.ndarray, np.ndarray]:
    """Return the canonical ``(knots_u, knots_v)`` pair for the N_U×N_V grid."""
    return build_uniform_knotvector(N_U, DEG_U), build_uniform_knotvector(N_V, DEG_V)


# ---------------------------------------------------------------------------
# Orientation convention
# ---------------------------------------------------------------------------
# Gravity reference for the leaf-local frame. The leaf-local +x axis is
# defined as ``tangent × UP`` where ``tangent`` is the collar-to-tip midrib
# direction; this gives a deterministic sign regardless of SVD ambiguity.
_UP = np.array([0.0, 0.0, 1.0])


def _leaf_local_x(cps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(x_local, centroid)`` where `x_local` is ``tangent × UP`` for
    the CP grid.

    ``tangent`` = (mean of tip CPs) − (mean of collar CPs), normalised.
    If ``tangent`` is nearly parallel to ``UP`` (leaf growing straight down
    or up), we fall back to the SVD in-plane secondary direction to avoid
    a degenerate cross product — still deterministic per grid, just relies
    on SVD for the fallback axis.
    """
    flat = cps.reshape(-1, 3).astype(np.float64)
    centroid = flat.mean(axis=0)

    collar_mean = cps[0, :, :].mean(axis=0)
    tip_mean = cps[-1, :, :].mean(axis=0)
    tangent = tip_mean - collar_mean
    t_len = np.linalg.norm(tangent)
    if t_len > 1e-9:
        tangent /= t_len

    x_local = np.cross(tangent, _UP)
    x_len = np.linalg.norm(x_local)
    if x_len < 1e-6:
        # Tangent ~parallel to UP: use SVD in-plane secondary axis, then
        # pick a deterministic sign by requiring positive dot product with
        # the +y world axis.
        centered = flat - centroid
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        x_local = vh[1].astype(np.float64)
        if x_local[1] < 0:
            x_local = -x_local
    else:
        x_local /= x_len
    return x_local, centroid


def enforce_orientation(cps: np.ndarray) -> np.ndarray:
    """Return `cps` with the canonical v-edge orientation.

    Convention: with ``x_local = tangent × gravity_up``, the ``v=0`` edge
    centroid must have the **larger** signed projection onto ``+x_local``
    than the ``v=1`` edge centroid. If violated, the v-axis is flipped
    via ``cps[:, ::-1, :]``.

    Geometrically: looking down the leaf from collar to tip with gravity
    pointing down, ``+x_local`` is on your **left**. So the canonical
    rule is "v=0 is the left edge, v=1 is the right edge". This matches
    the flat-rectangle synthetic test case where v runs −y → +y for a
    leaf pointing along +x.

    Args:
        cps: ``(N_U, N_V, 3)`` array.

    Returns:
        Oriented ``(N_U, N_V, 3)`` array (may be the same buffer if no flip
        was needed).
    """
    cps = np.asarray(cps, dtype=np.float64)
    if cps.ndim != 3 or cps.shape[-1] != 3:
        raise ValueError(f"cps must be (N_U, N_V, 3); got {cps.shape}")

    x_local, centroid = _leaf_local_x(cps)

    edge_v0 = cps[:, 0, :].mean(axis=0) - centroid
    edge_v1 = cps[:, -1, :].mean(axis=0) - centroid

    proj_v0 = float(np.dot(edge_v0, x_local))
    proj_v1 = float(np.dot(edge_v1, x_local))

    if proj_v0 < proj_v1:
        # v=0 edge is on the −x_local side → flip to match convention.
        return cps[:, ::-1, :].copy()
    return cps


# ---------------------------------------------------------------------------
# PlantGL / geomdl adapters
# ---------------------------------------------------------------------------
def cp_grid_to_plantgl_patch(cps: np.ndarray):
    """Build a PlantGL ``NurbsPatch`` from a canonical CP grid.

    Args:
        cps: ``(N_U, N_V, 3)`` control points in world coordinates (cm).

    Returns:
        ``openalea.plantgl.scenegraph.NurbsPatch`` with canonical degrees and
        clamped-uniform knot vectors. w=1 (no rational weights).
    """
    cps = np.asarray(cps, dtype=np.float64)
    if cps.shape != (N_U, N_V, 3):
        raise ValueError(
            f"Expected CP shape ({N_U}, {N_V}, 3); got {cps.shape}"
        )

    ensure_plantgl_loaded()
    from openalea.plantgl.scenegraph import NurbsPatch, Point4Matrix, RealArray

    # PlantGL indexes Point4Matrix as [u][v]; the constructor expects a
    # list of lists with outer index = U and inner index = V. That matches
    # our canonical (N_U, N_V, 3) layout directly.
    rows = []
    for i in range(N_U):
        row = []
        for j in range(N_V):
            row.append(
                (float(cps[i, j, 0]), float(cps[i, j, 1]), float(cps[i, j, 2]), 1.0)
            )
        rows.append(row)
    pmat = Point4Matrix(rows)

    knots_u, knots_v = canonical_knots()
    patch = NurbsPatch(
        pmat,
        DEG_U,
        DEG_V,
        RealArray(knots_u.tolist()),
        RealArray(knots_v.tolist()),
    )
    return patch


def cp_grid_to_plantgl_patch_general(
    cps: np.ndarray,
    deg_u: int = DEG_U,
    deg_v: int = DEG_V,
):
    """Build a PlantGL ``NurbsPatch`` from an arbitrary-shaped CP grid.

    Same as :func:`cp_grid_to_plantgl_patch` but without the canonical
    ``(N_U, N_V)`` shape check. Used by the compound sheath+blade path where
    the grid has extra u-rows and a denser v-direction.
    """
    cps = np.asarray(cps, dtype=np.float64)
    if cps.ndim != 3 or cps.shape[-1] != 3:
        raise ValueError(f"Expected (n_u, n_v, 3); got {cps.shape}")
    n_u, n_v, _ = cps.shape
    if n_u <= deg_u or n_v <= deg_v:
        raise ValueError(
            f"Need n_u>deg_u and n_v>deg_v; got n_u={n_u}, n_v={n_v}, "
            f"deg_u={deg_u}, deg_v={deg_v}"
        )

    ensure_plantgl_loaded()
    from openalea.plantgl.scenegraph import NurbsPatch, Point4Matrix, RealArray

    rows = [
        [(float(cps[i, j, 0]), float(cps[i, j, 1]), float(cps[i, j, 2]), 1.0)
         for j in range(n_v)]
        for i in range(n_u)
    ]
    knots_u = build_uniform_knotvector(n_u, deg_u)
    knots_v = build_uniform_knotvector(n_v, deg_v)
    return NurbsPatch(
        Point4Matrix(rows), deg_u, deg_v,
        RealArray(knots_u.tolist()), RealArray(knots_v.tolist()),
    )


def cp_grid_to_geomdl_surface(cps: np.ndarray):
    """Build a ``geomdl.BSpline.Surface`` from a canonical CP grid.

    Note on axis order
    ------------------
    geomdl's ``set_ctrlpts(flat, size_u, size_v)`` expects the flat list to
    iterate **v-first within u** (the doc calls this "u fixed, v varies").
    Our canonical layout is ``(N_U, N_V, 3)`` with u varying along axis 0,
    which *is* the layout geomdl wants when we flatten in C order — each
    u-row of length N_V is contiguous. See unit test for the verified
    round-trip.

    Args:
        cps: ``(N_U, N_V, 3)`` control points.

    Returns:
        ``geomdl.BSpline.Surface`` with matching degrees and knot vectors.
    """
    cps = np.asarray(cps, dtype=np.float64)
    if cps.shape != (N_U, N_V, 3):
        raise ValueError(
            f"Expected CP shape ({N_U}, {N_V}, 3); got {cps.shape}"
        )

    from geomdl import BSpline

    surf = BSpline.Surface()
    surf.degree_u = DEG_U
    surf.degree_v = DEG_V

    flat = cps.reshape(-1, 3).tolist()  # (N_U * N_V, 3), u-major
    surf.set_ctrlpts(flat, N_U, N_V)

    knots_u, knots_v = canonical_knots()
    surf.knotvector_u = knots_u.tolist()
    surf.knotvector_v = knots_v.tolist()
    return surf


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def eval_grid(patch, n_u: int = 30, n_v: int = 7) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate a PlantGL ``NurbsPatch`` at a uniform ``(n_u × n_v)`` grid.

    Args:
        patch: ``NurbsPatch`` (e.g. built by `cp_grid_to_plantgl_patch`).
        n_u: number of u samples (inclusive of endpoints).
        n_v: number of v samples (inclusive of endpoints).

    Returns:
        ``(vertices, normals)`` with shapes ``(n_u * n_v, 3)`` each, vertices
        in the same units as the CPs, analytic normals unit-length. The
        flat ordering is **u-major** (row i, column j → index ``i * n_v + j``)
        so a consumer can reshape to ``(n_u, n_v, 3)`` directly.
    """
    us = np.linspace(0.0, 1.0, n_u, dtype=np.float64)
    vs = np.linspace(0.0, 1.0, n_v, dtype=np.float64)

    verts = np.empty((n_u * n_v, 3), dtype=np.float64)
    norms = np.empty((n_u * n_v, 3), dtype=np.float64)
    for i, u in enumerate(us):
        for j, v in enumerate(vs):
            # PlantGL clamps evaluations inside the parametric domain;
            # u, v in [0, 1] are safe with clamped knot vectors.
            p = patch.getPointAt(float(u), float(v))
            n = patch.getNormalAt(float(u), float(v))
            k = i * n_v + j
            verts[k] = (p.x, p.y, p.z)
            norms[k] = (n.x, n.y, n.z)
    # Normalize (PlantGL may return unnormalized normals depending on build)
    lens = np.linalg.norm(norms, axis=1, keepdims=True)
    lens = np.maximum(lens, 1e-12)
    norms /= lens
    return verts, norms


__all__ = [
    "N_U", "N_V", "DEG_U", "DEG_V", "U_COLLAR", "U_TIP", "V_MIDRIB",
    "ensure_plantgl_loaded",
    "build_uniform_knotvector",
    "canonical_knots",
    "enforce_orientation",
    "cp_grid_to_plantgl_patch",
    "cp_grid_to_plantgl_patch_general",
    "cp_grid_to_geomdl_surface",
    "eval_grid",
]
