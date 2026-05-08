"""Runtime swap of per-plant leaf surface_cps from the MF3D library.

Lets each plant in a canopy draw its own donor from the 520-plant
MaizeField3D pool without regenerating the XML. Mutates
``LeafRandomParameter`` in memory after ``readParameters()`` and before
``initialize()`` — CPlantBox re-reads those params on every organ
creation, so downstream growth, lofting, DART export, and photosynthesis
all pick up the donor's CPs automatically.
"""
from __future__ import annotations

import numpy as np
import plantbox as pb

_PROFILE_SHAPE_FACTOR = 0.73  # trapezoidal integral of typical maize profile

# Bottom-leaf taper to keep drooping tips above ground (mirrors calibrate.py).
# Pos 0-1 insert close to the collar; a raw 60-80 cm donor blade would
# place its tip underground. Apply a multiplicative shrink plus an absolute
# cap. areaMax scales with lmax; Width_blade is unchanged (a shorter blade
# of the same max-width is geometrically valid).
_LMAX_TAPER = {0: 0.60, 1: 0.75, 2: 0.85, 3: 0.90}
_LMAX_CAP_CM = {0: 32.0, 1: 45.0, 2: 50.0, 3: 50.0}


def _apply_bottom_taper(pos: int, lmax: float, max_w: float) -> tuple[float, float]:
    """Return (lmax', areaMax') after bottom-leaf taper + cap."""
    factor = _LMAX_TAPER.get(pos)
    if factor is None:
        return lmax, lmax * max_w * _PROFILE_SHAPE_FACTOR
    cap = _LMAX_CAP_CM.get(pos, float("inf"))
    new_lmax = min(lmax * factor, cap)
    if new_lmax >= lmax:
        return lmax, lmax * max_w * _PROFILE_SHAPE_FACTOR
    area_max = new_lmax * max_w * _PROFILE_SHAPE_FACTOR
    return new_lmax, area_max


def _blade_position_for_subtype(lrp, *, phytomer_mode: bool) -> int | None:
    """Map a leaf RandomParameter to its MF3D position index.

    Returns None when the leaf is not a blade (e.g. sheath/pseudostem)
    or has a subType outside the normal range.
    """
    try:
        if int(lrp.getParameter("isPseudostem")) == 1:
            return None  # sheath — no surface_cps
    except Exception:
        pass
    st = int(lrp.subType)
    if phytomer_mode:
        # blade_st = 2 * pos + 1 → pos = (st - 1) // 2
        if st < 3 or st % 2 == 0:
            return None
        return (st - 1) // 2
    # monolithic: leaf_st = pos + 2 → pos = st - 2
    if st < 2:
        return None
    return st - 2


def _detect_phytomer_mode(plant) -> bool:
    """Decompose-phytomer mode is on when any leaf has isPseudostem=1."""
    for lrp in plant.getOrganRandomParameter(pb.leaf):
        try:
            if int(lrp.getParameter("isPseudostem")) == 1:
                return True
        except Exception:
            continue
    return False


def apply_donor_cps(
    plant,
    *,
    donor_seed: int,
    mode: str = "draw_coherent",
    resize_blades: bool = True,
    smooth_alpha: float = 1.0,
    verbose: bool = False,
) -> dict:
    """Overwrite each blade's ``surface_cps`` with the donor plant's CPs.

    Call AFTER ``plant.readParameters(xml)`` and BEFORE ``plant.initialize()``.

    Args:
        plant: CPlantBox MappedPlant with parameters loaded.
        donor_seed: RNG seed selecting which MF3D donor to use.
        mode: ``"draw_coherent"`` (default; one plant for all positions),
            ``"draw"`` (independent per-position draws), or ``"median"``
            (no variation — useful for smoke tests).
        resize_blades: if True, also set ``lmax``, ``Width_blade``, and
            ``areaMax`` from the donor's per-position metrics so the
            blade's absolute scale matches its shape.
        smooth_alpha: Median-smoothing weight in ``[0, 1]``. Implements the
            "smooth-CP option" from RUNTIME_CP_SWAP_2026-04-20.md §187 —
            blends the picked donor's per-position CPs with the per-position
            median of the entire filtered pool:
            ``cps = smooth_alpha * cps_donor + (1 - smooth_alpha) * cps_median``.
            Damps asymmetric mid-bulges that real-leaf scans carry without
            erasing donor identity. ``1.0`` (default) is the legacy
            no-smoothing behaviour. ``0.7`` is the recommended starting
            point — keeps donor character, removes the worst bulges. Only
            engaged for ``mode in {"draw", "draw_coherent"}`` (median /
            mean reducers are already smooth across the pool).
        verbose: print a one-line summary of swapped subtypes.

    Returns:
        dict with ``donor_seed``, ``mode``, ``positions``, and a list of
        modified subtypes — suitable for logging in a canopy snapshot.
    """
    from dart.coupling.geometry.canonical_library import (
        build_from_maizefield3d,
        _default_canonical_json,
    )

    # NPZ-compat: tip_canonical_rotate=False matches the frame the baked
    # canonical_leaf_library.npz (and maize_calibrated.xml's surface_cps)
    # live in. Without this, runtime CPs land in a +y-aligned canonical
    # frame while the XML's CPs keep their plant-natural azimuth — mixing
    # the two on the same plant produces frame mismatch and crumbled
    # meshes. The default _default_tip_bounds thresholds were tuned for
    # the rotated frame, so they reject most plants under the natural
    # azimuth (in particular emptying the draw_coherent intersection).
    # Disable the filter to recover the 520-plant pool the NPZ was built
    # against — calibrate also bakes the XML from this unfiltered pool.
    npz_compat_filter = lambda _pos: (-1e9, 1e9, -1e9, -1e9)
    lib = build_from_maizefield3d(
        _default_canonical_json(),
        reducer=mode,
        draw_seed=int(donor_seed) if mode in ("draw", "draw_coherent") else None,
        tip_canonical_rotate=False,
        tip_bounds=npz_compat_filter,
    )
    cps_by_pos = lib["cps_local"]          # (n_pos, N_U, N_V, 3)
    metrics = lib["chosen_metrics_cm"]     # (n_pos, 2) → (lmax, max_w)
    positions = np.asarray(lib["positions"], dtype=int)
    pos_to_idx = {int(p): i for i, p in enumerate(positions)}
    n_u, n_v = int(lib["n_u"]), int(lib["n_v"])
    deg_u, deg_v = int(lib["deg_u"]), int(lib["deg_v"])

    # Smooth-CP option (RUNTIME_CP_SWAP_2026-04-20 §187): blend the picked
    # donor against per-position median to damp asymmetric mid-bulges that
    # real-leaf scans carry. Only build the median library when the picked
    # donor mode is non-deterministic across plants — median/mean reducers
    # already average bulges out.
    smooth_alpha = float(min(max(smooth_alpha, 0.0), 1.0))
    median_by_pos = None
    median_metrics = None
    median_pos_to_idx = None
    if smooth_alpha < 1.0 and mode in ("draw", "draw_coherent"):
        median_lib = build_from_maizefield3d(
            _default_canonical_json(), reducer="median",
            tip_canonical_rotate=False, tip_bounds=npz_compat_filter,
        )
        median_by_pos = median_lib["cps_local"]
        median_metrics = median_lib["chosen_metrics_cm"]
        median_pos_to_idx = {
            int(p): i for i, p in enumerate(median_lib["positions"])
        }

    phytomer_mode = _detect_phytomer_mode(plant)
    modified: list[tuple[int, int]] = []  # (subType, pos)

    for lrp in plant.getOrganRandomParameter(pb.leaf):
        pos = _blade_position_for_subtype(lrp, phytomer_mode=phytomer_mode)
        if pos is None or pos not in pos_to_idx:
            continue
        # Respect the baseline XML's per-position lofter choice. Positions
        # that ship WITHOUT surface_cps are rendered by the quad-ribbon
        # lofter (pointed-tip taper, leafGeometry-driven); overwriting them
        # would force NURBS and lose the pointed bottom-leaf tips. Only
        # donor-swap positions that already opted into NURBS.
        if not list(lrp.surface_cps):
            continue
        idx = pos_to_idx[pos]
        raw_lmax = float(metrics[idx][0])
        max_w = float(metrics[idx][1])

        # Damp per-position size noise. MaizeField3D scans carry per-plant
        # spike-noise on lmax/Width_blade (occluded leaves, partial captures
        # of upper leaves, lodged plants with mis-numbered positions). A
        # single donor's per-position curve can have e.g. pos 8=30.9 cm
        # next to pos 9=47.7 cm. The CP shape blend above damps shape
        # noise; mirror it here so the size profile also looks plausible.
        # Plant-to-plant variation is preserved across seeds because the
        # blend keeps a fraction of the donor's actual size — heights
        # vary, but each plant's position-curve smooths.
        if (median_metrics is not None
                and median_pos_to_idx is not None
                and pos in median_pos_to_idx):
            m_idx = median_pos_to_idx[pos]
            median_lmax = float(median_metrics[m_idx][0])
            median_max_w = float(median_metrics[m_idx][1])
            raw_lmax = smooth_alpha * raw_lmax + (1.0 - smooth_alpha) * median_lmax
            max_w = smooth_alpha * max_w + (1.0 - smooth_alpha) * median_max_w

        if resize_blades:
            new_lmax, new_area = _apply_bottom_taper(pos, raw_lmax, max_w)
        else:
            new_lmax = float(lrp.lmax)
            new_area = float(lrp.areaMax)

        # Rescale library CPs so their midrib arc length equals the target
        # ``lmax`` in cm. Mirrors calibrate.py (XML-bake path): the C++
        # ``Leaf::updateNodesFromSurfaceCPs`` and the Python NURBS lofter
        # both apply ``scale = current_length / lmax`` at runtime, so the CPs
        # must be in absolute cm with ``midrib_arc == lmax``. Library CPs
        # are normalised (arc=1); skipping this rescale produced NaN leaf
        # nodes from the C++ midrib re-projection.
        grid = np.asarray(cps_by_pos[idx], dtype=np.float64)

        # Smooth-CP blend (per-position median anti-bulge). Applied in
        # normalised local-frame space BEFORE axial rescale so both grids
        # are on the same arc=1 scale. Position must exist in median lib.
        if (median_by_pos is not None
                and median_pos_to_idx is not None
                and pos in median_pos_to_idx):
            median_grid = np.asarray(
                median_by_pos[median_pos_to_idx[pos]], dtype=np.float64,
            )
            grid = smooth_alpha * grid + (1.0 - smooth_alpha) * median_grid

        midrib = grid[:, n_v // 2, :]
        lib_arc = float(np.sum(np.linalg.norm(np.diff(midrib, axis=0), axis=1)))
        if lib_arc > 1e-9 and new_lmax > 1e-9:
            grid = grid * (new_lmax / lib_arc)

        # Lateral rescale so CP-encoded area matches the target ``areaMax``.
        # The donor library inherits the same flawed lateral convention as
        # the original XML CPs (peak ≈ Width_blade vs the calibrate.py-asked
        # Width_blade × 2). The 2026-05-08 surface_cp lateral-rescale fix
        # (scripts/fix_surface_cp_lateral.py) only touched the XML; mirror
        # it here so library-injected blades render with correct width too.
        # SHAPE_FACTOR = 0.73 mirrors fix_surface_cp_lateral.py.
        if resize_blades and new_area > 1e-9:
            mid_xyz = grid[:, n_v // 2, :]
            mid_arc = float(np.sum(np.linalg.norm(np.diff(mid_xyz, axis=0), axis=1)))
            peak_x = float(grid[:, :, 0].max() - grid[:, :, 0].min())
            cp_area = mid_arc * peak_x * 0.73
            if cp_area > 1e-9:
                grid[:, :, 0] *= (new_area / cp_area)

        # Setter expects List[Vector3d] flattened row-major (i_u*n_v + i_v).
        flat = [pb.Vector3d(float(grid[iu, iv, 0]),
                            float(grid[iu, iv, 1]),
                            float(grid[iu, iv, 2]))
                for iu in range(n_u) for iv in range(n_v)]
        lrp.surface_cps = flat
        lrp.surface_n_u = n_u
        lrp.surface_n_v = n_v
        lrp.surface_deg_u = deg_u
        lrp.surface_deg_v = deg_v

        if resize_blades:
            lrp.lmax = new_lmax
            lrp.Width_blade = max_w / 2.0
            lrp.areaMax = new_area

        plant.setOrganRandomParameter(lrp)
        modified.append((int(lrp.subType), int(pos)))

    if verbose:
        subtypes = ", ".join(f"st{st}->pos{p}" for st, p in modified)
        print(f"  apply_donor_cps(mode={mode}, seed={donor_seed}): "
              f"{len(modified)} blades swapped [{subtypes}]")

    return {
        "donor_seed": int(donor_seed),
        "mode": mode,
        "positions": positions.tolist(),
        "modified_subtypes": modified,
    }
