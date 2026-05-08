"""Leaf-local-frame canonical CP library (Phase B).

Re-aggregates per-plant canonical (11, 5, 3) NURBS control points from
``Resources/MaizeField3d/maizefield3d_canonical_cps.json`` into a leaf-local
frame and groups by leaf position along the stem. The resulting per-position
median CP grids form a *library*: one canonical leaf shape per phytomer,
expressed in a collar-anchored local frame ready to be placed on any stem
insertion point.

Frame convention
----------------
Given a leaf CP grid ``cps`` of shape ``(N_U, N_V, 3)``:

- Collar centroid: mean of ``cps[0, :, :]`` (u=0 edge).
- Tangent: ``tip_centroid - collar_centroid``, normalized.
- Leaf-local +z: tangent.
- Leaf-local +x: ``tangent × UP`` normalized (UP = world +z).
- Leaf-local +y: ``+z × +x`` (right-handed).

The transform is rigid (rotation + translation) — length and shape are
preserved. The library's per-position CPs therefore retain the raw lengths
seen in MaizeField3D; at runtime the lofter length-scales by the ratio
``current_length / mature_length``.

The orientation convention (``enforce_orientation``) is applied *before*
transforming so every local grid has the same v-axis sign.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

from .canonical_cp_grid import (
    DEG_U,
    DEG_V,
    N_U,
    N_V,
    enforce_orientation,
)


_UP = np.array([0.0, 0.0, 1.0], dtype=np.float64)


def _default_tip_bounds(pos: int) -> tuple[float, float, float, float]:
    """Position-aware (min_tip_z, max_tip_z, min_tip_y, min_world_droop).

    First three: local-frame arc-normalised tip-shape bounds. Local-frame
    ``+z`` is the collar→tip axis, so local tip_z_frac measures blade
    straightness, not world droop.

    Fourth: world-frame droop fraction. Tried as a "must look droopy in
    world frame" filter but proved unreliable — MF3D scans include
    lodged/tilted plants where world-up ≠ plant-up, so a visually-droopy
    cane can read as ``world_tip_z > world_collar_z``. Left as a very
    loose sanity floor (``-inf`` effectively) until we have a per-plant
    orientation correction.

    Policy: keep the **original strict** local bounds on positions 0-10
    (proven visually good) and relax only pos 11-13 where coverage is
    sparse and emerging-whorl shapes are valid.
    """
    if pos <= 10:
        return (-0.30, 0.75, 0.20, -10.0)   # stock strict filter, proven good
    return (-0.60, 0.99, 0.10, -10.0)       # pos 11-13: relax for sparse, whorl-OK upper


# ---------------------------------------------------------------------------
# Local-frame construction
# ---------------------------------------------------------------------------
def _build_local_frame(cps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(R, collar)`` for a canonical CP grid.

    ``R`` is a world-from-local rotation matrix (columns = local +x, +y, +z in
    world coordinates). ``collar`` is the world position of the u=0 centroid.

    The returned frame satisfies:
      - ``R @ [0, 0, 1] = tangent`` where tangent is the **collar-local**
        direction (``u=1`` centroid minus ``u=0`` centroid). This matches
        CPlantBox's ``Organ::getiHeading0()`` at runtime — the insertion
        heading, not the mature (tip-collar) chord. A drooping mature leaf
        has a shallow tip-collar chord but a steep collar-local tangent;
        using the latter keeps library and runtime frames consistent.
      - ``R @ [1, 0, 0] = leaf-local +x``, defined as ``tangent × UP`` (or an
        SVD fallback when tangent is parallel to UP).
    """
    cps = np.asarray(cps, dtype=np.float64)
    if cps.shape != (N_U, N_V, 3):
        raise ValueError(f"expected {(N_U, N_V, 3)}, got {cps.shape}")

    collar = cps[0, :, :].mean(axis=0)
    near_collar = cps[1, :, :].mean(axis=0)
    tangent = near_collar - collar
    t_len = float(np.linalg.norm(tangent))
    if t_len < 1e-9:
        # Degenerate collar-local segment; fall back to tip-collar chord.
        tip = cps[-1, :, :].mean(axis=0)
        tangent = tip - collar
        t_len = float(np.linalg.norm(tangent))
        if t_len < 1e-9:
            raise ValueError("degenerate leaf: tip == collar")
    tangent = tangent / t_len

    x_local = np.cross(tangent, _UP)
    x_len = float(np.linalg.norm(x_local))
    if x_len < 1e-6:
        # Tangent is (anti)parallel to UP. Fall back to SVD in-plane secondary
        # axis, deterministic by requiring positive +y component.
        flat = cps.reshape(-1, 3)
        centered = flat - flat.mean(axis=0)
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        x_local = vh[1].astype(np.float64)
        if x_local[1] < 0:
            x_local = -x_local
    else:
        x_local = x_local / x_len

    y_local = np.cross(tangent, x_local)
    y_local /= max(float(np.linalg.norm(y_local)), 1e-12)

    # R columns are local axes in world coords, so R @ local = world.
    R = np.column_stack([x_local, y_local, tangent])
    return R, collar


def to_local_frame(
    cps_world: np.ndarray,
    normalize_arc: bool = False,
    tip_canonical_rotate: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Transform a world-frame CP grid into leaf-local coordinates.

    Args:
        cps_world: ``(N_U, N_V, 3)`` control points in world coordinates.
        normalize_arc: if True, divide all local CPs by the midrib arc
            length so the grid has unit midrib length. Enables size
            consistency across library plants whose raw leaves span
            14–110 cm — the lofter then re-scales via ``mature_length``.
        tip_canonical_rotate: if True (default), rotate each leaf about
            the +z midrib axis so its tip lies in the canonical
            ``(+y, +z)`` half-plane. Set to False to reproduce the
            ``canonical_leaf_library.npz`` build (the baked
            ``maize_calibrated.xml`` surface_cps live in this NPZ-compat
            frame). Mixing rotated and non-rotated CPs on the same plant
            produces frame mismatch and crumbled meshes.

    Returns:
        Tuple ``(cps_local, R, collar)`` where ``cps_local`` has the same
        shape (collar at origin, tangent along +z, leaf-local +x along
        ``tangent × UP``), ``R`` is the world-from-local rotation, and
        ``collar`` is the world-space collar centroid. Applying
        ``R @ cps_local[i, j] + collar`` reproduces the original world CPs
        only when ``normalize_arc`` is False.

    With ``tip_canonical_rotate=True``, after building the collar-local
    frame the CPs are additionally rotated about +z so the **tip** lies in
    the canonical ``(+y, +z)`` half-plane. Without this step (NPZ-compat
    mode), individual real leaves keep their plant-specific droop
    azimuth — medianing 520 plants hides the scatter, but picking a
    single plant via ``draw`` / ``draw_coherent`` carries it through.
    """
    cps_world = enforce_orientation(np.asarray(cps_world, dtype=np.float64))
    R, collar = _build_local_frame(cps_world)
    R_inv = R.T
    shifted = cps_world - collar[None, None, :]
    cps_local = np.einsum("ij,uvj->uvi", R_inv, shifted)

    if tip_canonical_rotate:
        tip = cps_local[-1, :, :].mean(axis=0)
        tip_xy = float(np.hypot(tip[0], tip[1]))
        if tip_xy > 1e-9:
            cos_a = tip[1] / tip_xy
            sin_a = tip[0] / tip_xy
            Rz = np.array([[cos_a, -sin_a, 0.0],
                           [sin_a,  cos_a, 0.0],
                           [0.0,    0.0,   1.0]])
            cps_local = np.einsum("ij,uvj->uvi", Rz, cps_local)
            R = R @ Rz.T

    if normalize_arc:
        midrib = cps_local[:, N_V // 2, :]
        arc = float(np.sum(np.linalg.norm(np.diff(midrib, axis=0), axis=1)))
        if arc > 1e-9:
            cps_local = cps_local / arc
    return cps_local, R, collar


def from_local_frame(
    cps_local: np.ndarray,
    collar_pos: np.ndarray,
    tangent: np.ndarray,
    up: np.ndarray | None = None,
) -> np.ndarray:
    """Inverse transform: place a local-frame CP grid at an insertion frame.

    Args:
        cps_local: ``(N_U, N_V, 3)`` control points in leaf-local coordinates.
        collar_pos: world position (3,) of the leaf collar.
        tangent: initial tangent direction (3,) at the collar; will be
            normalised. The library CPs are rotated so their +z aligns with
            this vector.
        up: world up axis used to orient leaf-local +x. Defaults to +z world.

    Returns:
        World-frame ``(N_U, N_V, 3)`` control points.
    """
    cps_local = np.asarray(cps_local, dtype=np.float64)
    if cps_local.ndim != 3 or cps_local.shape[-1] != 3:
        raise ValueError(f"expected (n_u, n_v, 3), got {cps_local.shape}")

    collar_pos = np.asarray(collar_pos, dtype=np.float64).reshape(3)
    tangent = np.asarray(tangent, dtype=np.float64).reshape(3)
    t_len = float(np.linalg.norm(tangent))
    if t_len < 1e-9:
        raise ValueError("tangent is zero; cannot orient library leaf")
    tangent = tangent / t_len

    up_vec = _UP if up is None else np.asarray(up, dtype=np.float64).reshape(3)

    x_local = np.cross(tangent, up_vec)
    x_len = float(np.linalg.norm(x_local))
    if x_len < 1e-6:
        # tangent parallel to up: pick any orthogonal axis deterministically.
        alt = np.array([1.0, 0.0, 0.0]) if abs(tangent[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        x_local = np.cross(tangent, alt)
        x_len = float(np.linalg.norm(x_local))
    x_local = x_local / max(x_len, 1e-12)

    y_local = np.cross(tangent, x_local)
    y_local /= max(float(np.linalg.norm(y_local)), 1e-12)

    R = np.column_stack([x_local, y_local, tangent])
    rotated = np.einsum("ij,uvj->uvi", R, cps_local)
    return rotated + collar_pos[None, None, :]


# ---------------------------------------------------------------------------
# Library aggregation
# ---------------------------------------------------------------------------
def aggregate_library(
    per_position_cps: dict[int, list[np.ndarray]],
    reducer: str = "median",
    draw_seed: int | None = None,
    per_position_plant_ids: dict[int, list[int]] | None = None,
    per_position_metrics: dict[int, list[tuple[float, float]]] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Collapse per-plant local-frame CPs into one grid per position.

    Args:
        per_position_cps: mapping ``{position: [local (N_U, N_V, 3) grids]}``.
        reducer: one of

          - ``"median"`` (default): element-wise median across plants.
          - ``"mean"``: element-wise mean.
          - ``"draw"``: independent per-position draw. Each position's
            CP grid comes from a different random plant — preserves
            per-leaf correlations (droop, edge-roll) that median smooths,
            at the cost of plant coherence (sizes won't match across
            positions of the same plant).
          - ``"draw_coherent"``: single plant ID drawn once, then all
            positions pulled from that plant. Requires
            ``per_position_plant_ids``. Preserves both per-leaf
            correlations *and* plant-level proportions (tall plant has
            long leaves at every position, short plant has short leaves).

        draw_seed: integer seed for ``"draw"`` / ``"draw_coherent"``;
            required when either draw mode is selected.
        per_position_plant_ids: parallel mapping
            ``{position: [plant_id, ...]}`` for ``"draw_coherent"``.
            Plant IDs are integers; a plant only contributes to a
            position if it has a leaf there.
        per_position_metrics: parallel mapping
            ``{position: [(lmax_cm, max_width_cm), ...]}``. When
            provided, the returned ``chosen_metrics`` array carries
            per-position ``(lmax_cm, max_width_cm)`` for the reducer's
            chosen leaf (for ``"median"``/``"mean"`` it's the per-position
            median; for ``"draw"``/``"draw_coherent"`` it's the chosen
            plant's actual values). Used by calibrate to emit
            plant-specific lmax and Width_blade alongside the CP grid
            so small plants don't get stretched to median length.

    Returns:
        Tuple ``(cps, counts, chosen_metrics)``. ``cps`` is
        ``(n_positions, N_U, N_V, 3)``. ``counts`` reports the donor
        pool size per position. ``chosen_metrics`` is
        ``(n_positions, 2)`` with columns ``(lmax_cm, max_width_cm)``
        when ``per_position_metrics`` is supplied, else ``None``.
    """
    valid = ("median", "mean", "draw", "draw_coherent")
    if reducer not in valid:
        raise ValueError(f"unknown reducer {reducer!r}")
    if reducer in ("draw", "draw_coherent") and draw_seed is None:
        raise ValueError(f"reducer={reducer!r} requires draw_seed")
    if reducer == "draw_coherent" and per_position_plant_ids is None:
        raise ValueError("reducer='draw_coherent' requires per_position_plant_ids")

    positions = sorted(per_position_cps.keys())
    if not positions:
        raise ValueError("per_position_cps is empty")

    rng = (
        np.random.default_rng(int(draw_seed))
        if reducer in ("draw", "draw_coherent") and draw_seed is not None
        else None
    )

    n_positions = len(positions)
    out = np.zeros((n_positions, N_U, N_V, 3), dtype=np.float64)
    counts = np.zeros((n_positions,), dtype=np.int64)
    chosen_metrics: np.ndarray | None = (
        np.zeros((n_positions, 2), dtype=np.float64)
        if per_position_metrics is not None
        else None
    )

    def _metrics_for_index(pos: int, k: int) -> tuple[float, float]:
        assert per_position_metrics is not None
        lm, mw = per_position_metrics[pos][k]
        return float(lm), float(mw)

    def _metrics_reduce(pos: int, indices: list[int] | None = None) -> tuple[float, float]:
        assert per_position_metrics is not None
        arr = np.asarray(per_position_metrics[pos], dtype=np.float64)
        if indices is not None:
            arr = arr[indices]
        if reducer == "mean":
            lm = float(np.mean(arr[:, 0])); mw = float(np.mean(arr[:, 1]))
        else:
            lm = float(np.median(arr[:, 0])); mw = float(np.median(arr[:, 1]))
        return lm, mw

    if reducer == "draw_coherent":
        assert per_position_plant_ids is not None
        assert rng is not None
        id_sets = [set(per_position_plant_ids[p]) for p in positions]
        common = set.intersection(*id_sets) if id_sets else set()
        if not common:
            raise ValueError(
                f"no single plant covers all {n_positions} positions; "
                f"reduce the position range or use reducer='draw'"
            )
        chosen = int(rng.choice(sorted(common)))
        for idx, pos in enumerate(positions):
            ids = per_position_plant_ids[pos]
            k = ids.index(chosen)
            out[idx] = per_position_cps[pos][k]
            counts[idx] = len(per_position_cps[pos])
            if chosen_metrics is not None:
                chosen_metrics[idx] = _metrics_for_index(pos, k)
        return out, counts, chosen_metrics

    for idx, pos in enumerate(positions):
        stack = np.stack(per_position_cps[pos], axis=0)
        if reducer == "median":
            out[idx] = np.median(stack, axis=0)
            if chosen_metrics is not None:
                chosen_metrics[idx] = _metrics_reduce(pos)
        elif reducer == "mean":
            out[idx] = np.mean(stack, axis=0)
            if chosen_metrics is not None:
                chosen_metrics[idx] = _metrics_reduce(pos)
        else:  # "draw" (independent per-position)
            assert rng is not None
            k = int(rng.integers(0, stack.shape[0]))
            out[idx] = stack[k]
            if chosen_metrics is not None:
                chosen_metrics[idx] = _metrics_for_index(pos, k)
        counts[idx] = stack.shape[0]
    return out, counts, chosen_metrics


# ---------------------------------------------------------------------------
# MaizeField3D loader → local-frame → aggregate
# ---------------------------------------------------------------------------
def build_from_maizefield3d(
    canonical_json_path: Path,
    reducer: str = "median",
    draw_seed: int | None = None,
    normalize_arc: bool = True,
    tip_bounds: Callable[[int], tuple[float, float, float, float]] | None = None,
    tip_canonical_rotate: bool = True,
) -> dict:
    """Read the canonical per-plant CP JSON and build a local-frame library.

    Args:
        canonical_json_path: path to ``maizefield3d_canonical_cps.json``.
        reducer: aggregation mode; see :func:`aggregate_library`.
        draw_seed: seed for ``reducer='draw'`` / ``'draw_coherent'``.
        normalize_arc: if True (default), normalise each plant's CPs to
            unit midrib-arc before aggregation. Downstream consumers
            should re-scale via ``mature_length`` at loft time — the
            returned dict flags this via ``normalized=True``.
        tip_canonical_rotate: forwarded to :func:`to_local_frame`. Set to
            False to match the frame the baked
            ``canonical_leaf_library.npz`` (and the NPZ-derived
            ``maize_calibrated.xml`` surface_cps) live in. ``cp_swap``
            uses this for runtime donor injection so swapped CPs share a
            frame with the XML's CPs.

    Returns a dict with keys:
      - ``cps_local``: ``(n_positions, N_U, N_V, 3)`` aggregated local CPs
      - ``positions``: ``(n_positions,)`` int array of position indices
      - ``counts``: ``(n_positions,)`` sample counts
      - ``n_u``, ``n_v``, ``deg_u``, ``deg_v``: canonical grid metadata
      - ``reducer``: the aggregation function applied
      - ``draw_seed``: the seed used for draw modes, else ``None``
      - ``normalized``: True when CPs are in unit-arc space
      - ``source``: provenance string
    """
    canonical_json_path = Path(canonical_json_path)
    data = json.loads(canonical_json_path.read_text())
    bounds_fn = tip_bounds if tip_bounds is not None else _default_tip_bounds

    meta_n_u = int(data.get("n_u", N_U))
    meta_n_v = int(data.get("n_v", N_V))
    if meta_n_u != N_U or meta_n_v != N_V:
        raise ValueError(
            f"library expects ({N_U}, {N_V}) grid; JSON has ({meta_n_u}, {meta_n_v})"
        )

    per_position: dict[int, list[np.ndarray]] = {}
    per_position_ids: dict[int, list[int]] = {}
    per_position_metrics: dict[int, list[tuple[float, float]]] = {}
    n_rejected_shape = 0
    plants = data.get("plants") or []
    for plant_idx, plant_record in enumerate(plants):
        for leaf in plant_record.get("leaves", []):
            pos = int(leaf["position"])
            cps_world = np.asarray(leaf["cps_cm"], dtype=np.float64)
            if cps_world.shape != (N_U, N_V, 3):
                continue
            # World-frame metrics (unaffected by local-frame normalization).
            world_midrib = cps_world[:, N_V // 2, :]
            world_lmax_cm = float(
                np.sum(np.linalg.norm(np.diff(world_midrib, axis=0), axis=1))
            )
            # Max full width = widest V-span across all U rows.
            v_span = np.linalg.norm(
                cps_world[:, -1, :] - cps_world[:, 0, :], axis=1
            )
            world_max_width_cm = float(np.max(v_span))
            # World-frame droop: how far below collar the tip sits, as a
            # fraction of arc. Positive = drooping, negative = tip above
            # collar (whorl). Filter on this to reject flat-horizontal
            # donors — local tip_z can't tell a droopy cane from a flag.
            world_collar_z = float(cps_world[0, :, 2].mean())
            world_tip_z = float(cps_world[-1, :, 2].mean())
            world_droop_frac = (
                (world_collar_z - world_tip_z) / world_lmax_cm
                if world_lmax_cm > 1e-9 else 0.0
            )
            try:
                cps_local, _, _ = to_local_frame(
                    cps_world,
                    normalize_arc=normalize_arc,
                    tip_canonical_rotate=tip_canonical_rotate,
                )
            except ValueError:
                continue
            midrib = cps_local[:, N_V // 2, :]
            arc = float(np.sum(np.linalg.norm(np.diff(midrib, axis=0), axis=1)))
            if arc < 1e-9:
                continue
            tip = midrib[-1]
            tip_z_frac = float(tip[2]) / arc
            tip_y_frac = float(tip[1]) / arc
            mn_z, mx_z, mn_y, mn_droop = bounds_fn(pos)
            if (tip_z_frac > mx_z
                    or tip_z_frac < mn_z
                    or tip_y_frac < mn_y
                    or world_droop_frac < mn_droop):
                n_rejected_shape += 1
                continue
            per_position.setdefault(pos, []).append(cps_local)
            per_position_ids.setdefault(pos, []).append(plant_idx)
            per_position_metrics.setdefault(pos, []).append(
                (world_lmax_cm, world_max_width_cm)
            )

    if not per_position:
        raise ValueError(f"no valid leaves parsed from {canonical_json_path}")

    cps_stacked, counts, chosen_metrics = aggregate_library(
        per_position,
        reducer=reducer,
        draw_seed=draw_seed,
        per_position_plant_ids=(
            per_position_ids if reducer == "draw_coherent" else None
        ),
        per_position_metrics=per_position_metrics,
    )
    positions = np.asarray(sorted(per_position.keys()), dtype=np.int64)

    return {
        "cps_local": cps_stacked,
        "positions": positions,
        "counts": counts,
        "n_u": N_U,
        "n_v": N_V,
        "deg_u": DEG_U,
        "deg_v": DEG_V,
        "reducer": reducer,
        "draw_seed": draw_seed if reducer in ("draw", "draw_coherent") else None,
        "normalized": bool(normalize_arc),
        "shape_filter": {
            "bounds_fn": (
                "default_position_aware"
                if tip_bounds is None
                else getattr(tip_bounds, "__name__", "custom")
            ),
            "per_position_bounds": {
                int(p): tuple(bounds_fn(int(p))) for p in positions
            },
            "n_rejected": int(n_rejected_shape),
        },
        # Per-position (lmax_cm, max_width_cm) for the reducer's chosen leaf:
        # - median/mean: per-position median/mean across the filtered pool
        # - draw: chosen plant's actual size, per position
        # - draw_coherent: the single chosen plant's actual sizes
        "chosen_metrics_cm": chosen_metrics,
        "source": str(canonical_json_path),
    }


def augment_with_sheath(
    cps_local: np.ndarray,
    *,
    stem_radius_cm: float,
    sheath_length_cm: float,
    wrap_deg: float = 272.0,
    n_sheath_rows: int = 3,
) -> np.ndarray:
    """Prepend sheath rows to a blade-only leaf-local CP grid.

    The plan (Phase E.2) treats the sheath as a u-region of the same CP grid
    whose CPs are placed *on an arc around the parent-stem axis* so the NURBS
    surface naturally wraps the internode. No change to grid topology is
    required on the blade side — we simply extend the grid in -u by
    ``n_sheath_rows`` rows and re-parameterise ``u`` so the new rows cover
    ``[0, sheath_length_cm]`` and the existing blade covers
    ``[sheath_length_cm, sheath_length_cm + blade_arc]``.

    Frame convention (leaf-local, collar at origin):
      - +z = midrib tangent at the collar (blade direction)
      - +x = ``tangent x UP`` in world frame (lateral)
      - +y = ``+z x +x``

    The sheath axis is **opposite** to the blade tangent: sheath wraps the
    stem *below* the collar. So sheath CPs live at ``-z`` (below origin). The
    wrap is placed in the local +x/+y plane at a distance ``stem_radius_cm``
    from the axis. The arc spans ``wrap_deg`` degrees centred on the
    blade-facing side of the stem (seam on the back of the stem).

    Args:
        cps_local: ``(N_U, N_V, 3)`` blade-only CP grid in leaf-local frame.
        stem_radius_cm: parent stem radius in cm.
        sheath_length_cm: sheath length along -z in cm.
        wrap_deg: angular extent of the wrap around the stem axis (default
            272° matches external-maize observations).
        n_sheath_rows: number of sheath u-rows prepended (default 3; total
            grid becomes ``(N_U + n_sheath_rows, N_V, 3)``).

    Returns:
        Augmented CP grid of shape ``(N_U + n_sheath_rows, N_V, 3)``.
    """
    cps_local = np.asarray(cps_local, dtype=np.float64)
    if cps_local.ndim != 3 or cps_local.shape[2] != 3:
        raise ValueError(f"expected (n_u, n_v, 3), got {cps_local.shape}")
    _, n_v, _ = cps_local.shape
    if n_sheath_rows < 1:
        raise ValueError("n_sheath_rows must be >= 1")
    if stem_radius_cm <= 0:
        raise ValueError("stem_radius_cm must be > 0")
    if sheath_length_cm <= 0:
        raise ValueError("sheath_length_cm must be > 0")

    half_wrap = 0.5 * np.deg2rad(wrap_deg)
    # v coordinates along the existing grid run linearly 0..1; mirror that
    # onto the sheath wrap arc so v=0 and v=N_V-1 sit at the seam.
    angles = np.linspace(-half_wrap, +half_wrap, n_v, dtype=np.float64)

    # z positions for the prepended rows: u=0 at the furthest sheath point
    # (tip of sheath, below collar), stepping up to z = 0 at the collar.
    # We keep a *separate* collar row (first row of the existing blade CPs)
    # by prepending n_sheath_rows rows strictly at z < 0.
    z_sheath = np.linspace(-sheath_length_cm, 0.0, n_sheath_rows + 1)[:-1]

    sheath = np.empty((n_sheath_rows, n_v, 3), dtype=np.float64)
    for i_u, z in enumerate(z_sheath):
        for i_v, phi in enumerate(angles):
            # +y is the blade-facing direction; sheath wraps around +y axis
            # with seam at -y.
            sheath[i_u, i_v, 0] = stem_radius_cm * np.sin(phi)
            sheath[i_u, i_v, 1] = stem_radius_cm * np.cos(phi)
            sheath[i_u, i_v, 2] = z

    return np.concatenate([sheath, cps_local], axis=0)


def upsample_v(cps: np.ndarray, new_n_v: int) -> np.ndarray:
    """Linearly interpolate the v-axis of a CP grid to ``new_n_v`` samples.

    Cheap first-order upsample. The resulting grid represents a slightly
    different NURBS surface than the original (not an exact knot-insertion
    reparameterisation), but the blade region is smooth enough that the
    visual difference is negligible. Needed to increase N_V from 5 (current
    library) to 9 (the smallest grid that lets us represent a closed
    sheath tube without excessive chord-collapse).
    """
    cps = np.asarray(cps, dtype=np.float64)
    if cps.ndim != 3 or cps.shape[-1] != 3:
        raise ValueError(f"expected (n_u, n_v, 3), got {cps.shape}")
    n_u, old_n_v, _ = cps.shape
    if new_n_v == old_n_v:
        return cps.copy()
    if new_n_v < 2:
        raise ValueError(f"new_n_v must be >= 2; got {new_n_v}")
    old_t = np.linspace(0.0, 1.0, old_n_v)
    new_t = np.linspace(0.0, 1.0, new_n_v)
    out = np.empty((n_u, new_n_v, 3), dtype=np.float64)
    for i in range(n_u):
        for d in range(3):
            out[i, :, d] = np.interp(new_t, old_t, cps[i, :, d])
    return out


def build_compound_leaf_cps(
    blade_cps_local: np.ndarray,
    *,
    stem_radius_cm: float,
    sheath_length_cm: float,
    stem_axis: np.ndarray | None = None,
    stem_radius_at_z: Callable[[float], float] | None = None,
    max_sheath_length_cm: float | None = None,
    wrap_deg: float = 360.0,
    bulge: float = 0.18,
    base_clearance: float = 0.03,
    n_cup: int = 5,
    n_morph: int = 3,
    n_v: int = 13,
    ligule_tilt_frac: float = 0.35,
) -> np.ndarray:
    """Compound sheath-ring + blade CP grid (single NURBS patch).

    The sheath is a short stack of closed 360° rings around the stem
    with a light uniform bulge; the blade's lowest ``n_morph`` rows
    smoothly un-curl out of that ring into the flat blade ribbon.
    Topology matches the stage-16 maize reference: a short collar
    hugging the stem, with the blade emerging from its front side.

    Rows, stacked along ``u``:

    1. **Ring cup** — rows ``0..n_cup-1``. ``n_cup`` closed rings
       stacked from ``z = -L_rendered`` (cup bottom) to ``z = 0``
       (ligule crest at the midrib column). Each ring has radius
       ``stem_r(z) · (1 + base_clearance + bulge)`` measured from
       the stem central axis. CPlantBox attaches a leaf's node 0 to
       the parent stem node, which lies on the stem skeleton, so
       the leaf-local origin IS the stem-axis point and
       ``stem_center(z) = axis · z`` — rings wrap the stem
       symmetrically with no lateral bias. ``v = 0`` and ``v = n_v − 1``
       both sit at the back seam (θ = ±π), so every ring is a true
       360° wrap. The top of the cup tilts upward on the blade-
       emergence side and downward at the back (ligule asymmetry).

    2. **Ring → blade transition** — rows ``n_cup..n_cup+n_morph-1``.
       The first ``n_morph`` blade rows smoothstep-blend from a ring
       at the blade's native ``z`` (plus a fading ligule offset) into
       the flat blade ribbon ``blade_up[i]``. Bulge and ligule offset
       both fade to zero across the transition; at ``i = n_morph − 1``
       the blend equals ``blade_up[n_morph − 1]`` exactly. This is
       where the blade's left/right edges (``v = 0``, ``v = n_v − 1``)
       un-curl from the back seam and swing forward to the blade's
       flat edges.

    3. **Blade** — rows ``n_cup+n_morph..end``. Copied verbatim from
       ``blade_up[n_morph..]``.

    Ligule asymmetry (ring still uniform-in-θ at each row, but z tilts):

        ``ligule_z(θ) = -ligule_tilt_frac · L_rendered · sin²(θ/2)``
        ``z_cup(i, θ) = z_base(i) + t_i · ligule_z(θ)``

    where ``t_i = i/(n_cup-1)`` ramps 0→1 from cup bottom to cup top.
    At θ = 0 (midrib) the offset is 0; at θ = ±π (back seam) it
    reaches its full magnitude. The bottom row (``t = 0``) is a
    perfectly horizontal ring; the top row (``t = 1``) is tilted —
    front at ``z = 0``, back at ``z = -ligule_tilt_frac · L``.

    **Stem-aware radius.** If ``stem_radius_at_z`` is provided the
    cup queries it at each per-column ``z`` offset so the sheath
    tracks a tapering stem.

    **Uniform 360° protrusion.** ``bulge`` is applied uniformly at
    every v-column — the ring puffs outward by the same radial
    amount around the full circumference. No azimuthal wedge.

    The rendered sheath length defaults to
    ``min(sheath_length_cm, 2.5 · stem_radius_cm)`` when
    ``max_sheath_length_cm`` is ``None`` — this produces the short
    visual collar that matches the reference. Pass an explicit
    positive ``max_sheath_length_cm`` (e.g. ``math.inf``) to render
    the full botanical sheath.

    Frame convention (leaf-local, collar at origin):
      - ``+z`` : leaf tangent at collar (points toward blade tip)
      - ``+x`` : ``tangent × UP`` (lateral blade-width direction)
      - ``+y`` : ``+z × +x``
      - ``stem_axis`` : parent stem growth direction in the same frame.

    v-parametrization:
      ``θ(j) = θ_half − wrap_rad · j/(n_v − 1)`` with
      ``θ_half = wrap_rad / 2``. For the default ``wrap_deg = 360``
      this runs from θ = +π at ``j = 0`` to θ = −π at ``j = n_v − 1``,
      which coincide: the patch has a degenerate back seam. Triangles
      along that seam are filtered by the lofter's degenerate-area
      pass.

    Args:
        blade_cps_local: ``(N_U_blade, N_V_blade, 3)`` blade CPs already
            length-scaled into the leaf-local frame.
        stem_radius_cm: parent stem radius at collar (cm).
        sheath_length_cm: full botanical sheath length (cm).
        stem_axis: unit vector giving stem growth direction in the
            same frame as ``blade_cps_local``. Defaults to leaf-local
            ``+z`` (erect-leaf fallback).
        stem_radius_at_z: optional callable ``z_local_cm → stem_r_cm``.
        max_sheath_length_cm: visual cap on rendered sheath length.
            ``None`` (default) uses ``2.5 · stem_radius_cm``.
        wrap_deg: arc coverage; defaults to 360 for a fully closed
            ring. Values in ``(0, 360)`` leave a back slit.
        bulge: uniform radial bulge fraction (default 0.18).
        base_clearance: constant radial offset as a fraction of the
            local stem radius (default 0.03).
        n_cup: number of closed-ring cup rows (default 5). Must be ≥ 2.
        n_morph: number of blade rows that transition ring → flat
            (default 3). Must be ≥ 2.
        n_v: v-direction CP count (default 13).
        ligule_tilt_frac: fraction of ``L_rendered`` by which the back
            of the cup top sits below the front (default 0.35).

    Returns:
        Compound CP grid of shape ``(n_cup + N_U_blade, n_v, 3)``.
    """
    blade_cps_local = np.asarray(blade_cps_local, dtype=np.float64)
    if blade_cps_local.ndim != 3 or blade_cps_local.shape[-1] != 3:
        raise ValueError(
            f"blade_cps_local must be (n_u, n_v, 3); got {blade_cps_local.shape}"
        )
    if n_cup < 2:
        raise ValueError("n_cup must be >= 2")
    if n_morph < 2:
        raise ValueError("n_morph must be >= 2")
    if n_v < 5:
        raise ValueError("n_v must be >= 5 for a smooth ring")
    if stem_radius_cm <= 0:
        raise ValueError("stem_radius_cm must be > 0")
    if sheath_length_cm <= 0:
        raise ValueError("sheath_length_cm must be > 0")
    if not (0.0 < wrap_deg <= 360.0):
        raise ValueError("wrap_deg must be in (0, 360]")
    if bulge < 0.0:
        raise ValueError("bulge must be >= 0")
    if base_clearance < 0.0:
        raise ValueError("base_clearance must be >= 0")
    if not (0.0 <= ligule_tilt_frac < 1.0):
        raise ValueError("ligule_tilt_frac must be in [0, 1)")

    # Stem axis in leaf-local frame.
    if stem_axis is None:
        axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    else:
        axis = np.asarray(stem_axis, dtype=np.float64).reshape(3)
        a_len = float(np.linalg.norm(axis))
        if a_len < 1e-9:
            raise ValueError("stem_axis must be non-zero")
        axis = axis / a_len

    # "Front" = projection of blade tangent (+z_local) onto the stem-
    # perpendicular plane.
    blade_dir_local = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    front = blade_dir_local - float(np.dot(blade_dir_local, axis)) * axis
    f_len = float(np.linalg.norm(front))
    if f_len < 1e-6:
        tmp = (
            np.array([1.0, 0.0, 0.0])
            if abs(axis[0]) < 0.9
            else np.array([0.0, 1.0, 0.0])
        )
        front = tmp - float(np.dot(tmp, axis)) * axis
        f_len = float(np.linalg.norm(front))
    front = front / max(f_len, 1e-12)
    side = np.cross(axis, front)

    blade_up = upsample_v(blade_cps_local, n_v)
    n_u_blade = blade_up.shape[0]
    if n_morph > n_u_blade:
        raise ValueError(
            f"n_morph ({n_morph}) cannot exceed blade u-rows ({n_u_blade})"
        )

    # Rendered sheath length. Default cap keeps the cup short — the
    # reference maize ring is ~1–1.5× stem diameter tall.
    if max_sheath_length_cm is None:
        cap = 2.5 * float(stem_radius_cm)
    else:
        if max_sheath_length_cm <= 0:
            raise ValueError("max_sheath_length_cm must be > 0 when provided")
        cap = float(max_sheath_length_cm)
    L_rendered = min(float(sheath_length_cm), cap)
    stem_r_fallback = float(stem_radius_cm)

    # v-parametrization. At wrap_deg = 360, θ runs +π..−π so j=0 and
    # j=n_v-1 coincide at the back seam → closed ring, degenerate seam
    # triangles filtered by the lofter downstream.
    wrap_rad = float(wrap_deg) * np.pi / 180.0
    theta_half = wrap_rad / 2.0
    j_idx = np.arange(n_v)
    theta = theta_half - wrap_rad * (j_idx / (n_v - 1))
    arc_dir = (
        np.cos(theta)[:, None] * front[None, :]
        + np.sin(theta)[:, None] * side[None, :]
    )  # (n_v, 3)

    # Ligule tilt: 0 at midrib (θ=0), -ligule_tilt_frac·L at back (θ=±π).
    ligule_z = -ligule_tilt_frac * L_rendered * np.sin(theta / 2.0) ** 2  # (n_v,)

    def _stem_r_at(z_off: float) -> float:
        if stem_radius_at_z is None:
            return stem_r_fallback
        try:
            r = float(stem_radius_at_z(z_off))
        except Exception:
            return stem_r_fallback
        return r if r > 0.0 else stem_r_fallback

    def _ring_row(z_j: np.ndarray, bulge_scale: float) -> np.ndarray:
        """Closed ring at per-column z values (n_v,). Returns (n_v, 3).

        The leaf-local origin sits on the stem CENTRAL AXIS (CPlantBox
        places a leaf's node 0 at the parent stem node, which lies on
        the stem skeleton — not on the stem surface). So the stem
        centre at z-offset ``z`` is simply ``axis · z`` — no lateral
        offset. Rings around that centre wrap the stem symmetrically.
        """
        row = np.empty((n_v, 3), dtype=np.float64)
        for j in range(n_v):
            z = float(z_j[j])
            stem_r_z = _stem_r_at(z)
            stem_center = axis * z
            R = stem_r_z * (1.0 + base_clearance + bulge * bulge_scale)
            row[j] = stem_center + R * arc_dir[j]
        return row

    n_u_total = n_cup + n_u_blade
    out = np.zeros((n_u_total, n_v, 3), dtype=np.float64)

    # --- Ring cup: closed rings with ligule tilt ramped 0→1 bottom→top ---
    # Cup bulge tapers along u: bottom row (t_i = 0) hugs the stem
    # (bulge_scale = 0 → R = stem_r · (1 + base_clearance)), top row
    # (t_i = 1) keeps the full bulge so the collar wrap is unchanged.
    # Smoothstep ramp keeps the cup C¹ and joins the morph rows
    # (bulge_scale = asym = 1 at the cup-top boundary) without a jump.
    for i in range(n_cup):
        t_i = i / max(n_cup - 1, 1)
        z_base = -L_rendered + t_i * L_rendered  # -L at bottom, 0 at top
        z_j = z_base + t_i * ligule_z
        cup_bulge_scale = t_i * t_i * (3.0 - 2.0 * t_i)  # smoothstep(t_i)
        out[i] = _ring_row(z_j, bulge_scale=cup_bulge_scale)

    # --- Transition: first n_morph blade rows blend ring → flat blade ---
    # frac = 0 at i=0 (pure ring at blade_up[0].z + ligule) matches the
    # last cup row (z = 0 + ligule = ligule_z); frac = 1 at i=n_morph-1
    # lands exactly on blade_up[n_morph-1], so the next verbatim blade
    # row joins without a jump.
    for i in range(n_morph):
        frac = i / (n_morph - 1)
        smooth = frac * frac * (3.0 - 2.0 * frac)
        asym = 1.0 - smooth
        flat = blade_up[i]
        z_j = flat[:, 2] + asym * ligule_z
        ring_pt = _ring_row(z_j, bulge_scale=asym)
        out[n_cup + i] = (1.0 - smooth) * ring_pt + smooth * flat

    # --- Remaining blade rows verbatim ---
    if n_morph < n_u_blade:
        out[n_cup + n_morph:] = blade_up[n_morph:]

    return out


# ---------------------------------------------------------------------------
# Pheno4D young-stage library — maturity-bucketed, arc-normalised
# ---------------------------------------------------------------------------
def _midrib_arc(cps: np.ndarray) -> float:
    """Return the midrib (v=N_V//2) arc length of a (N_U, N_V, 3) grid."""
    n_v = cps.shape[1]
    midrib = cps[:, n_v // 2, :]
    return float(np.sum(np.linalg.norm(np.diff(midrib, axis=0), axis=1)))


def _planarise_pheno4d_fit(
    cps_local: np.ndarray,
    *,
    max_wind_deg: float = 60.0,
    tip_z_min: float = 0.85,
    y_range_max: float = 0.20,
    x_range_max: float = 0.25,
) -> np.ndarray | None:
    """Strip sensor-state (droop, twist, whorl-wrap) from a Pheno4D fit.

    Pheno4D NURBS fits are snapshots of the leaf's physical state: blade
    shape *plus* gravity droop, twist, and one-sided whorl-wrap. MF3D fits
    are near-planar idealised templates. A raw linear blend adds the
    state terms onto the blade and amplifies them through the lofter's
    length-scale, producing visually mangled young leaves (documented
    2026-04-19, see Known Gap #5 in ``NATIVE_SURFACE_CPS_IMPLEMENTATION``).

    This function applies the five-step de-stating pass agreed in the
    follow-up plan:

      1. **Midrib planarisation.** PCA-project the midrib polyline
         (``cps[:, n_v // 2, :]``) onto its best-fit 2D plane and translate
         every u-row by the midrib correction, preserving the v-cross-
         section shape.
      2. **Plane-contains-+z rotation.** Rigidly rotate the blade so the
         midrib plane contains the leaf-local +z axis (no off-plane
         droop) and the plane normal lies on the ±y axis (midrib is then
         canonically in the xz-plane).
      3. **Cross-row symmetrisation.** Per u-row, enforce reflection
         symmetry across ``v = n_v // 2`` using the symmetry plane
         containing the midrib tangent and the (post-rotation) plane
         normal. Cancels twist and one-sided whorl-wrap.
      4. **Whorl-wrap reject (early gate).** Fits whose midrib winds more
         than ``max_wind_deg`` of azimuth around +z (unwrapped atan2 of
         midrib xy positions, measured only where xy-radius exceeds a
         small noise floor) are rejected outright.
      5. **Post-filter QA.** After steps 1-3 the midrib is arc-normalised
         in-place and must satisfy ``tip_z > tip_z_min``,
         ``x_range < x_range_max``, ``y_range < y_range_max``. Failures
         return ``None``.

    The returned grid is still in un-normalised leaf-local coordinates —
    the caller (:func:`build_young_library_from_pheno4d`) applies arc
    normalisation afterwards.

    Args:
        cps_local: ``(n_u, n_v, 3)`` leaf-local fit to clean.
        max_wind_deg: rejection threshold for azimuthal winding.
        tip_z_min: QA lower bound on ``(midrib[-1].z - midrib[0].z) / arc``.
        y_range_max: QA upper bound on midrib y extent / arc.
        x_range_max: QA upper bound on midrib x extent / arc.

    Returns:
        Cleaned ``(n_u, n_v, 3)`` CP grid, or ``None`` if the fit cannot
        be planarised.
    """
    cps = np.asarray(cps_local, dtype=np.float64).copy()
    if cps.ndim != 3 or cps.shape[-1] != 3:
        raise ValueError(f"expected (n_u, n_v, 3), got {cps.shape}")
    n_u, n_v, _ = cps.shape
    if n_u < 3 or n_v < 3:
        return None
    mid_j = n_v // 2

    # --- Step 4 (early gate): whorl-wrap detection --------------------
    midrib = cps[:, mid_j, :]
    base = midrib[0].copy()
    arc0 = float(np.sum(np.linalg.norm(np.diff(midrib, axis=0), axis=1)))
    if arc0 < 1e-6:
        return None
    mid_xy = midrib[:, :2] - base[:2]
    r_xy = np.linalg.norm(mid_xy, axis=1)
    r_floor = 0.05 * arc0
    if np.max(r_xy) > r_floor:
        mask = r_xy > 0.02 * arc0
        if int(mask.sum()) >= 2:
            phi = np.arctan2(mid_xy[mask, 1], mid_xy[mask, 0])
            phi_unwrapped = np.unwrap(phi)
            winding_deg = float(np.rad2deg(
                phi_unwrapped.max() - phi_unwrapped.min()
            ))
            if winding_deg > max_wind_deg:
                return None

    # --- Step 1: midrib planarisation (PCA) ---------------------------
    centroid = midrib.mean(axis=0)
    centred = midrib - centroid
    # SVD gives principal axes in vh (rows, descending singular value).
    _, _, vh = np.linalg.svd(centred, full_matrices=False)
    plane_normal = vh[2].astype(np.float64)
    # Project midrib onto best-fit plane; translate each u-row by the
    # midrib delta so the v-cross-section shape is preserved.
    off_plane = (centred @ plane_normal)[:, None] * plane_normal[None, :]
    midrib_proj = midrib - off_plane
    delta = midrib_proj - midrib
    cps = cps + delta[:, None, :]

    # --- Step 2: rotate plane normal perpendicular to +z --------------
    z_axis = np.array([0.0, 0.0, 1.0])
    # Choose plane_normal sign so the rotation to target_normal is minimal
    # (dot product ≥ 0 after perp construction).
    proj_on_z = float(plane_normal @ z_axis)
    perp = plane_normal - proj_on_z * z_axis
    perp_len = float(np.linalg.norm(perp))
    if perp_len < 1e-9:
        # Plane normal parallel to +z → midrib nearly horizontal.
        # These leaves fail QA anyway; let the post-filter reject them.
        target_normal = plane_normal
    else:
        target_normal = perp / perp_len
        cps, plane_normal = _rotate_vector_to_vector(
            cps, plane_normal, target_normal
        )

    # Sub-step: rotate around +z so plane_normal aligns with +y so the
    # midrib sits canonically in the xz-plane (y_range check is then a
    # direct out-of-plane residual measurement).
    pn_xy = np.array([plane_normal[0], plane_normal[1], 0.0])
    pn_xy_len = float(np.linalg.norm(pn_xy))
    if pn_xy_len > 1e-9:
        pn_xy = pn_xy / pn_xy_len
        # Rotation about +z by angle theta mapping pn_xy → +y. For a unit
        # pn_xy = (sin θ, cos θ, 0) solves R_z(θ) · pn_xy = (0, 1, 0).
        theta = np.arctan2(pn_xy[0], pn_xy[1])
        c = np.cos(theta)
        s = np.sin(theta)
        Rz = np.array([
            [c, -s, 0.0],
            [s,  c, 0.0],
            [0.0, 0.0, 1.0],
        ])
        cps = cps @ Rz.T
        plane_normal = Rz @ plane_normal

    # --- Step 3: cross-row symmetrisation -----------------------------
    midrib = cps[:, mid_j, :]
    tangents = np.zeros_like(midrib)
    tangents[1:-1] = midrib[2:] - midrib[:-2]
    tangents[0] = midrib[1] - midrib[0]
    tangents[-1] = midrib[-1] - midrib[-2]
    t_lens = np.linalg.norm(tangents, axis=1, keepdims=True)
    t_lens = np.where(t_lens < 1e-9, 1e-9, t_lens)
    tangents = tangents / t_lens
    sym_normal = np.cross(tangents, plane_normal[None, :])
    sn_lens = np.linalg.norm(sym_normal, axis=1, keepdims=True)
    sn_lens = np.where(sn_lens < 1e-9, 1e-9, sn_lens)
    sym_normal = sym_normal / sn_lens  # (n_u, 3)

    cps_sym = cps.copy()
    for i in range(n_u):
        M = midrib[i]
        n_sym = sym_normal[i]
        for j in range(mid_j):
            j_m = n_v - 1 - j
            p_j = cps[i, j] - M
            p_m = cps[i, j_m] - M
            refl_j = p_j - 2.0 * float(p_j @ n_sym) * n_sym
            refl_m = p_m - 2.0 * float(p_m @ n_sym) * n_sym
            cps_sym[i, j] = M + 0.5 * (p_j + refl_m)
            cps_sym[i, j_m] = M + 0.5 * (p_m + refl_j)
    cps = cps_sym

    # --- Step 5: post-filter QA --------------------------------------
    midrib = cps[:, mid_j, :]
    arc = float(np.sum(np.linalg.norm(np.diff(midrib, axis=0), axis=1)))
    if arc < 1e-6:
        return None
    base = midrib[0]
    tip_z = float((midrib[-1, 2] - base[2]) / arc)
    x_range = float((midrib[:, 0].max() - midrib[:, 0].min()) / arc)
    y_range = float((midrib[:, 1].max() - midrib[:, 1].min()) / arc)
    if tip_z < tip_z_min or x_range > x_range_max or y_range > y_range_max:
        return None
    return cps


def _rotate_vector_to_vector(
    cps: np.ndarray, from_vec: np.ndarray, to_vec: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Rotate ``cps`` (applied to every point) so ``from_vec`` → ``to_vec``.

    Uses Rodrigues' formula. Returns ``(rotated_cps, rotated_from_vec)``
    (the rotated from_vec equals ``to_vec`` up to numerical noise).
    """
    f = np.asarray(from_vec, dtype=np.float64)
    t = np.asarray(to_vec, dtype=np.float64)
    f = f / max(float(np.linalg.norm(f)), 1e-12)
    t = t / max(float(np.linalg.norm(t)), 1e-12)
    v = np.cross(f, t)
    s = float(np.linalg.norm(v))
    c = float(f @ t)
    if s < 1e-12:
        if c > 0:
            return cps, f
        # 180° flip: pick any axis perpendicular to f.
        alt = np.array([1.0, 0.0, 0.0]) if abs(f[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        axis = np.cross(f, alt)
        axis = axis / float(np.linalg.norm(axis))
        K = np.array([
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ])
        R = np.eye(3) + 2.0 * (K @ K)  # 180° rotation: R = I + 2 K²
        return cps @ R.T, R @ f
    v = v / s
    K = np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0],
    ])
    R = np.eye(3) + s * K + (1.0 - c) * (K @ K)
    return cps @ R.T, R @ f


def build_young_library_from_pheno4d(
    pheno4d_json_path: Path,
    *,
    n_buckets: int = 10,
    reducer: str = "median",
    min_samples_per_bucket: int = 3,
    planarise: bool = True,
    max_wind_deg: float = 60.0,
    tip_z_min: float = 0.85,
    y_range_max: float = 0.20,
    x_range_max: float = 0.25,
) -> dict:
    """Aggregate Pheno4D annotated fits into a maturity-bucketed young library.

    Each fitted leaf is transformed into its leaf-local frame, normalised so
    its midrib arc equals 1.0 (width scales proportionally), then pooled
    across all (plant_id, label) combinations and bucketed by per-leaf
    maturity.

    Maturity proxy
    --------------
    For each ``(plant_id, label)`` chain of scans the largest observed
    midrib arc is the in-dataset lmax; each scan's maturity is
    ``arc / lmax_chain``. Maturity therefore lies in (0, 1] per chain, and
    pooling across chains builds up a per-bucket distribution spanning the
    observed growth-curve states. Cultivar mismatch with MaizeField3D is
    absorbed by the arc-normalisation: shape is cultivar-agnostic, scale
    is re-imposed downstream by the blending adapter using the production
    cultivar's lmax.

    Buckets
    -------
    Bucket ``i`` covers maturity in ``[i/n_buckets, (i+1)/n_buckets]`` with
    the upper edge included in the final bucket. Empty buckets (no chain
    reached that maturity band) are omitted from the output; callers
    interpolate over the retained ``bucket_centers`` at blend time.

    Args:
        pheno4d_json_path: path to ``pheno4d_canonical_cps.json`` produced
            by ``fit_all_ply_to_nurbs.py``.
        n_buckets: number of equal-width maturity bins.
        reducer: ``"median"`` or ``"mean"`` for per-bucket aggregation.
        min_samples_per_bucket: drop buckets with fewer samples than this.

    Returns:
        dict with keys:
          - ``cps_normalised`` : ``(n_kept_buckets, N_U, N_V, 3)`` arc-normalised CPs
          - ``bucket_centers`` : ``(n_kept_buckets,)`` center-of-bucket maturities
          - ``counts``         : ``(n_kept_buckets,)`` per-bucket sample counts
          - ``n_u``, ``n_v``, ``deg_u``, ``deg_v``, ``reducer``, ``source``,
            ``n_buckets``      : metadata
    """
    pheno4d_json_path = Path(pheno4d_json_path)
    if not pheno4d_json_path.exists():
        raise FileNotFoundError(f"Pheno4D fit JSON missing: {pheno4d_json_path}")
    if reducer not in ("median", "mean"):
        raise ValueError(f"unknown reducer {reducer!r}")
    if n_buckets < 2:
        raise ValueError("n_buckets must be >= 2")

    data = json.loads(pheno4d_json_path.read_text())

    # Pass 1: collect per-(plant, label) max arc so maturity is chain-relative.
    chain_max: dict[tuple[str, int], float] = {}
    raw: list[tuple[str, int, np.ndarray, float]] = []
    for scan in data.get("scans", []):
        plant = scan.get("plant_id")
        for leaf in scan.get("leaves", []):
            cps = leaf.get("cps_cm")
            label = leaf.get("label")
            if cps is None or label is None:
                continue
            cps = np.asarray(cps, dtype=np.float64)
            if cps.shape != (N_U, N_V, 3):
                continue
            arc = _midrib_arc(cps)
            if arc <= 1e-6:
                continue
            key = (plant, int(label))
            chain_max[key] = max(chain_max.get(key, 0.0), arc)
            raw.append((plant, int(label), cps, arc))

    if not raw:
        raise ValueError(f"no valid Pheno4D leaves in {pheno4d_json_path}")

    # Pass 2: per-leaf maturity + planarised + arc-normalised local-frame CPs.
    bucket_samples: dict[int, list[np.ndarray]] = {}
    bucket_seen: dict[int, int] = {}
    bucket_rejected: dict[int, int] = {}
    for plant, label, cps_world, arc in raw:
        lmax_chain = chain_max[(plant, label)]
        if lmax_chain <= 1e-6:
            continue
        maturity = min(arc / lmax_chain, 1.0)
        # Bucket index: floor(maturity * n_buckets), with upper edge clamped
        # into the final bucket so maturity == 1.0 doesn't overflow.
        b = int(min(maturity * n_buckets, n_buckets - 1))
        bucket_seen[b] = bucket_seen.get(b, 0) + 1
        try:
            cps_local, _, _ = to_local_frame(cps_world)
        except ValueError:
            bucket_rejected[b] = bucket_rejected.get(b, 0) + 1
            continue
        if planarise:
            cleaned = _planarise_pheno4d_fit(
                cps_local,
                max_wind_deg=max_wind_deg,
                tip_z_min=tip_z_min,
                y_range_max=y_range_max,
                x_range_max=x_range_max,
            )
            if cleaned is None:
                bucket_rejected[b] = bucket_rejected.get(b, 0) + 1
                continue
            cps_local = cleaned
        local_arc = _midrib_arc(cps_local)
        if local_arc <= 1e-6:
            bucket_rejected[b] = bucket_rejected.get(b, 0) + 1
            continue
        cps_norm = cps_local * (1.0 / local_arc)
        bucket_samples.setdefault(b, []).append(cps_norm)

    kept_buckets = [
        b for b in sorted(bucket_samples.keys())
        if len(bucket_samples[b]) >= min_samples_per_bucket
    ]
    if not kept_buckets:
        raise ValueError(
            f"no maturity bucket reached min_samples_per_bucket={min_samples_per_bucket}"
        )

    cps_out = np.zeros((len(kept_buckets), N_U, N_V, 3), dtype=np.float64)
    counts = np.zeros((len(kept_buckets),), dtype=np.int64)
    centers = np.zeros((len(kept_buckets),), dtype=np.float64)
    for i, b in enumerate(kept_buckets):
        stack = np.stack(bucket_samples[b], axis=0)
        agg = np.median(stack, axis=0) if reducer == "median" else np.mean(stack, axis=0)
        # Pointwise median/mean of arc-normalised grids shortens the midrib
        # by a few percent (curve-space median ≠ point-space median). Re-
        # normalise so downstream ``young * mature_length`` gives an
        # arc-accurate grid for blending.
        post_arc = _midrib_arc(agg)
        if post_arc > 1e-6:
            agg = agg * (1.0 / post_arc)
        cps_out[i] = agg
        counts[i] = stack.shape[0]
        centers[i] = (b + 0.5) / n_buckets

    stats = {
        "seen_per_bucket": dict(sorted(bucket_seen.items())),
        "rejected_per_bucket": dict(sorted(bucket_rejected.items())),
        "kept_per_bucket": {b: len(bucket_samples.get(b, [])) for b in sorted(bucket_seen)},
        "planarise": bool(planarise),
        "max_wind_deg": float(max_wind_deg),
        "tip_z_min": float(tip_z_min),
        "y_range_max": float(y_range_max),
        "x_range_max": float(x_range_max),
    }

    return {
        "cps_normalised": cps_out,
        "bucket_centers": centers,
        "counts": counts,
        "n_u": N_U,
        "n_v": N_V,
        "deg_u": DEG_U,
        "deg_v": DEG_V,
        "reducer": reducer,
        "source": str(pheno4d_json_path),
        "n_buckets": int(n_buckets),
        "stats": stats,
    }


def save_young_library(library: dict, out_path: Path) -> None:
    """Persist a young-stage library dict to ``.npz``."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        cps_normalised=library["cps_normalised"],
        bucket_centers=library["bucket_centers"],
        counts=library["counts"],
        n_u=np.int64(library["n_u"]),
        n_v=np.int64(library["n_v"]),
        deg_u=np.int64(library["deg_u"]),
        deg_v=np.int64(library["deg_v"]),
        n_buckets=np.int64(library["n_buckets"]),
        reducer=np.array(library["reducer"]),
        source=np.array(library["source"]),
    )


def load_young_library(path: Path) -> dict:
    """Reverse of :func:`save_young_library`."""
    data = np.load(Path(path), allow_pickle=False)
    return {
        "cps_normalised": data["cps_normalised"],
        "bucket_centers": data["bucket_centers"],
        "counts": data["counts"],
        "n_u": int(data["n_u"]),
        "n_v": int(data["n_v"]),
        "deg_u": int(data["deg_u"]),
        "deg_v": int(data["deg_v"]),
        "n_buckets": int(data["n_buckets"]),
        "reducer": str(data["reducer"]),
        "source": str(data["source"]),
    }


def _smoothstep(x: float) -> float:
    """Hermite smoothstep on [0, 1]: 0→0, 1→1, zero 1st derivative at edges."""
    t = float(np.clip(x, 0.0, 1.0))
    return t * t * (3.0 - 2.0 * t)


def select_young_cps(young_lib: dict, maturity: float) -> np.ndarray:
    """Pick the young-library CP grid closest to ``maturity``.

    The library is bucketed at discrete maturities; this helper does a
    nearest-center lookup. For maturity outside the library's observed
    range the nearest edge bucket is returned — the caller is responsible
    for down-weighting the contribution via cross-fade (usually the
    library is only consulted when ``maturity`` is below the mature
    cross-fade threshold).

    Args:
        young_lib: dict from :func:`load_young_library`.
        maturity: per-leaf maturity in [0, 1].

    Returns:
        ``(N_U, N_V, 3)`` arc-normalised CP grid.
    """
    centers = np.asarray(young_lib["bucket_centers"], dtype=np.float64)
    cps = np.asarray(young_lib["cps_normalised"], dtype=np.float64)
    if centers.size == 0:
        raise ValueError("young library has no buckets")
    idx = int(np.argmin(np.abs(centers - float(maturity))))
    return cps[idx].copy()


def blend_young_mature_cps(
    young_cps_normalised: np.ndarray,
    mature_cps_local: np.ndarray,
    *,
    maturity: float,
    mature_length: float,
    fade_start: float = 0.4,
    fade_end: float = 0.8,
) -> np.ndarray:
    """Cross-fade a young-library CP grid into a mature one.

    Both grids live in the leaf-local frame. ``young_cps_normalised`` has
    midrib arc = 1.0 (per-leaf normalisation from
    :func:`build_young_library_from_pheno4d`); ``mature_cps_local`` has
    midrib arc = ``mature_length`` (the MaizeField3D library as emitted
    by ``calibrate.py --surface-cps``).

    Returns a grid in the same frame with midrib arc = ``mature_length``
    so the downstream lofter pipeline (``scale = current / mature``) is
    unchanged. The fade is a smoothstep over ``[fade_start, fade_end]``
    on maturity: below ``fade_start`` the output is pure young-shape
    (scaled by ``mature_length``); above ``fade_end`` it is pure mature.
    """
    young = np.asarray(young_cps_normalised, dtype=np.float64)
    mature = np.asarray(mature_cps_local, dtype=np.float64)
    if young.shape != mature.shape:
        raise ValueError(
            f"shape mismatch: young {young.shape} vs mature {mature.shape}"
        )
    if fade_end <= fade_start:
        raise ValueError("fade_end must exceed fade_start")
    if mature_length <= 0:
        raise ValueError("mature_length must be > 0")

    young_scaled = young * float(mature_length)
    fade = _smoothstep((float(maturity) - fade_start) / (fade_end - fade_start))
    blended = (1.0 - fade) * young_scaled + fade * mature

    # A pointwise linear combination of two curves has arc length ≤ the
    # input arcs (chord vs arc). Re-scale so the blended midrib matches
    # ``mature_length`` — the lofter assumes
    # ``scale = current_length / mature_length`` when placing the grid,
    # so any shrinkage here would leak into the rendered leaf length.
    arc = _midrib_arc(blended)
    if arc > 1e-6:
        blended = blended * (float(mature_length) / arc)
    return blended


def save_library(library: dict, out_path: Path) -> None:
    """Persist a library dict to ``.npz``."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        cps_local=library["cps_local"],
        positions=library["positions"],
        counts=library["counts"],
        n_u=np.int64(library["n_u"]),
        n_v=np.int64(library["n_v"]),
        deg_u=np.int64(library["deg_u"]),
        deg_v=np.int64(library["deg_v"]),
        reducer=np.array(library["reducer"]),
        source=np.array(library["source"]),
    )


def load_library(path: Path) -> dict:
    """Reverse of :func:`save_library`."""
    data = np.load(Path(path), allow_pickle=False)
    return {
        "cps_local": data["cps_local"],
        "positions": data["positions"],
        "counts": data["counts"],
        "n_u": int(data["n_u"]),
        "n_v": int(data["n_v"]),
        "deg_u": int(data["deg_u"]),
        "deg_v": int(data["deg_v"]),
        "reducer": str(data["reducer"]),
        "source": str(data["source"]),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _default_canonical_json() -> Path:
    return Path("/home/lukas/PHD/Resources/MaizeField3d/maizefield3d_canonical_cps.json")


def _default_out_path() -> Path:
    # Co-locate with the rest of the per-species calibration data so the
    # lofter / calibrate.py can pick it up by default.
    return Path(__file__).resolve().parents[1] / "data" / "canonical_leaf_library.npz"


def _default_pheno4d_json() -> Path:
    return Path("/home/lukas/PHD/Resources/Pheno4D/pheno4d_canonical_cps.json")


def _default_young_out_path() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "pheno4d_young_library.npz"


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Build leaf-local-frame canonical CP libraries. "
        "Defaults build the MaizeField3D mature library; use "
        "--young-from-pheno4d to build the Pheno4D maturity-bucketed "
        "young-stage library instead.",
    )
    ap.add_argument(
        "--canonical-json",
        type=Path,
        default=_default_canonical_json(),
        help="Path to maizefield3d_canonical_cps.json",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output .npz path (defaults: mature→coupling/data/canonical_leaf_library.npz, "
             "young→coupling/data/pheno4d_young_library.npz)",
    )
    ap.add_argument(
        "--reducer",
        choices=("median", "mean", "draw"),
        default="median",
        help="Aggregation across plants. 'draw' picks one plant per "
             "position via --draw-seed instead of averaging — preserves "
             "plant-to-plant variance (droop/arch) that median smooths.",
    )
    ap.add_argument(
        "--draw-seed",
        type=int,
        default=None,
        help="Seed for --reducer draw. Required when draw mode is selected.",
    )
    ap.add_argument(
        "--young-from-pheno4d",
        type=Path,
        nargs="?",
        const=_default_pheno4d_json(),
        default=None,
        help="Build the young-stage library from a Pheno4D fit JSON. "
             "Omit the path to use the default pheno4d_canonical_cps.json.",
    )
    ap.add_argument(
        "--n-buckets",
        type=int,
        default=10,
        help="Number of maturity buckets for --young-from-pheno4d (default 10).",
    )
    ap.add_argument(
        "--min-samples-per-bucket",
        type=int,
        default=3,
        help="Drop young-library buckets with fewer samples than this "
             "(default 3 — matches the Known Gap #5 spec).",
    )
    ap.add_argument(
        "--no-planarise",
        action="store_true",
        help="Disable the 5-step Pheno4D planariser (raw fits). "
             "Only use for before/after diagnostics — raw fits produce "
             "mangled young leaves at render time (2026-04-19).",
    )
    ap.add_argument("--max-wind-deg", type=float, default=60.0)
    ap.add_argument("--tip-z-min", type=float, default=0.85)
    ap.add_argument("--y-range-max", type=float, default=0.20)
    ap.add_argument("--x-range-max", type=float, default=0.25)
    args = ap.parse_args(list(argv) if argv is not None else None)

    if args.young_from_pheno4d is not None:
        out = args.out if args.out is not None else _default_young_out_path()
        lib = build_young_library_from_pheno4d(
            args.young_from_pheno4d,
            n_buckets=args.n_buckets,
            reducer=args.reducer,
            min_samples_per_bucket=args.min_samples_per_bucket,
            planarise=not args.no_planarise,
            max_wind_deg=args.max_wind_deg,
            tip_z_min=args.tip_z_min,
            y_range_max=args.y_range_max,
            x_range_max=args.x_range_max,
        )
        save_young_library(lib, out)
        print(f"wrote {out}")
        print(f"  buckets kept:    {lib['cps_normalised'].shape[0]} / {lib['n_buckets']}")
        print(f"  bucket centers:  {['%.2f' % c for c in lib['bucket_centers'].tolist()]}")
        print(f"  counts:          {lib['counts'].tolist()}")
        print(f"  cps shape:       {lib['cps_normalised'].shape}")
        stats = lib.get("stats", {})
        if stats:
            print(f"  planarise:       {stats.get('planarise')}")
            print(f"  seen / bucket:     {stats.get('seen_per_bucket')}")
            print(f"  rejected / bucket: {stats.get('rejected_per_bucket')}")
            print(f"  kept / bucket:     {stats.get('kept_per_bucket')}")
            total_seen = sum(stats.get('seen_per_bucket', {}).values())
            total_rej = sum(stats.get('rejected_per_bucket', {}).values())
            if total_seen:
                print(
                    f"  rejection rate:    {total_rej}/{total_seen} "
                    f"= {100.0 * total_rej / total_seen:.1f}%"
                )
        return 0

    out = args.out if args.out is not None else _default_out_path()
    lib = build_from_maizefield3d(args.canonical_json, reducer=args.reducer)
    save_library(lib, out)

    print(f"wrote {out}")
    print(f"  positions:  {lib['positions'].tolist()}")
    print(f"  counts:     {lib['counts'].tolist()}")
    print(f"  cps shape:  {lib['cps_local'].shape}")
    # Sanity: tip z-coords per position (should be ~lmax in cm)
    tip_z = lib["cps_local"][:, -1, :, 2].mean(axis=1)
    print(f"  tip +z (≈lmax):  {['%.1f' % z for z in tip_z.tolist()]}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "to_local_frame",
    "from_local_frame",
    "aggregate_library",
    "augment_with_sheath",
    "upsample_v",
    "build_compound_leaf_cps",
    "build_from_maizefield3d",
    "save_library",
    "load_library",
    "build_young_library_from_pheno4d",
    "save_young_library",
    "load_young_library",
    "select_young_cps",
    "blend_young_mature_cps",
]
