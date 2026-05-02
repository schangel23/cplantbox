"""NURBS-patch leaf lofter.

Drop-in replacement for the quad-ribbon ``_loft_leaf`` in ``g1_to_g3.py``.

Pipeline
--------
1. Resample the input skeleton to ``N_U=11`` arc-length stations.
2. Build a smooth Darboux-like frame at each station (tangent → plane normal
   via gravity-referenced SVD fallback → binormal).
3. Place ``N_V=5`` control points across the width at ``v ∈ {0, 0.25, 0.5,
   0.75, 1}``, placing ``v=0.5`` on the midrib and ``v=0``/``v=1`` at ±half
   width along the binormal.
4. Apply leaf-blade deformations as control-point displacements:
   - gutter depression (normal-offset on the midrib CP)
   - vertical undulation (normal-offset on the whole cross-section)
   - axial twist (rotate the cross-section around the tangent)
   - asymmetric edge curl (signed normal-offset on the two edge CPs)
5. Build a ``plantbox.NurbsPatch`` via
   ``canonical_cp_grid.cp_grid_to_plantgl_patch`` and tessellate at
   ``n_u_eval × n_v_eval`` uniform samples.
6. Emit a 9-tuple: ``(vertices, indices, normals, uvs, organ_ids,
   segment_ids, quad_indices, quad_organ_ids, cps)`` — the first eight
   elements match ``_loft_leaf``'s tuple; the ninth is the canonical
   ``(N_U, N_V, 3)`` CP grid after deformations, retained so downstream
   fitters can compute CP-space L2 losses against a target.

Segment-ID tracking maps each tessellation row's u-value back to the
original ``_orig_segment_map`` built by ``_subdivide_skeleton``.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np

from .canonical_cp_grid import (
    N_U, N_V,
    cp_grid_to_plantgl_patch, cp_grid_to_plantgl_patch_general, eval_grid,
)

# Sentinel for "no deformation applied to this CP column".
_V_PARAMS = np.linspace(0.0, 1.0, N_V)  # (5,) canonical v positions


def _compute_arc_lengths(skeleton: np.ndarray) -> np.ndarray:
    diffs = np.diff(skeleton, axis=0)
    seg = np.linalg.norm(diffs, axis=1)
    return np.concatenate([[0.0], np.cumsum(seg)])


def _resample_skeleton(
    skeleton: np.ndarray, widths: np.ndarray, n: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resample skeleton+widths to ``n`` uniform arc-length stations.

    Returns ``(skel_n, widths_n, arc_frac)`` where ``arc_frac`` ∈ [0, 1]
    is the per-station arc-length fraction (0 at collar, 1 at tip).
    """
    cum = _compute_arc_lengths(skeleton)
    total = float(cum[-1])
    if total <= 1e-9:
        raise ValueError("degenerate skeleton: zero arc length")
    target_arc = np.linspace(0.0, total, n)
    skel_n = np.column_stack([
        np.interp(target_arc, cum, skeleton[:, d]) for d in range(3)
    ])
    widths_n = np.interp(target_arc, cum, widths)
    arc_frac = target_arc / total
    return skel_n, widths_n, arc_frac


def _compute_tangents(skeleton: np.ndarray) -> np.ndarray:
    """Centered finite-difference tangents (unit-length)."""
    n = len(skeleton)
    tangents = np.empty_like(skeleton)
    for i in range(n):
        if i == 0:
            t = skeleton[1] - skeleton[0]
        elif i == n - 1:
            t = skeleton[-1] - skeleton[-2]
        else:
            t = skeleton[i + 1] - skeleton[i - 1]
        tl = np.linalg.norm(t)
        tangents[i] = t / tl if tl > 1e-9 else np.array([1.0, 0.0, 0.0])
    return tangents


def _darboux_frames(skeleton: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(tangents, normals, binormals)`` for the skeleton.

    Strategy: use the SVD plane of the skeleton as a stable reference — maize
    leaves curve essentially in a plane. Normal = plane normal; binormal =
    ``normal × tangent`` (points across the width). If the leaf is nearly
    straight or not well-planar, fall back to gravity-referenced normals.
    """
    n = len(skeleton)
    tangents = _compute_tangents(skeleton)

    # SVD plane reference
    centroid = skeleton.mean(axis=0)
    centered = skeleton - centroid
    _, svals, vh = np.linalg.svd(centered, full_matrices=False)
    plane_ok = bool(svals[1] > 1e-8 and svals[2] < 0.5 * svals[1])
    svd_normal = vh[2] if plane_ok else None

    up = np.array([0.0, 0.0, 1.0])
    normals = np.empty_like(skeleton)
    binormals = np.empty_like(skeleton)
    for i in range(n):
        t = tangents[i]
        # Primary: gravity-referenced face normal (blade stays roughly horizontal)
        face = up - np.dot(up, t) * t
        fl = np.linalg.norm(face)
        if fl > 0.3:
            face /= fl
            bn = np.cross(t, face)
        elif svd_normal is not None:
            bn = svd_normal - np.dot(svd_normal, t) * t
            bnl = np.linalg.norm(bn)
            if bnl > 1e-6:
                bn /= bnl
                face = np.cross(bn, t)
            else:
                face = up
                bn = np.cross(t, face)
        else:
            face = up if fl > 1e-6 else np.array([1.0, 0.0, 0.0])
            bn = np.cross(t, face)

        bl = np.linalg.norm(bn)
        if bl > 1e-9:
            bn /= bl
        # Re-orthogonalise face for numerical cleanliness.
        face = np.cross(bn, t)
        fl2 = np.linalg.norm(face)
        if fl2 > 1e-9:
            face /= fl2
        normals[i] = face
        binormals[i] = bn
    return tangents, normals, binormals


def _build_cp_grid(
    skeleton_u: np.ndarray,
    widths_u: np.ndarray,
    binormals_u: np.ndarray,
) -> np.ndarray:
    """Place the canonical 11×5 CP grid using skeleton + frames.

    CP layout across the width:
        v=0.0 → -w/2 * binormal
        v=0.25 → -w/4 * binormal
        v=0.5 → midrib (on skeleton)
        v=0.75 → +w/4 * binormal
        v=1.0 → +w/2 * binormal

    For a quadratic B-spline with 5 uniform CPs and clamped knots the
    cross-section curve is very close to the linear ribbon these CPs
    define; interior curve points are influenced smoothly by adjacent CPs
    but the overall width footprint (from v=0 to v=1) is preserved.
    """
    cps = np.empty((N_U, N_V, 3), dtype=np.float64)
    for i in range(N_U):
        hw = 0.5 * widths_u[i]  # half-width in cm
        offsets = (_V_PARAMS - 0.5) * 2.0 * hw  # -hw, -hw/2, 0, +hw/2, +hw
        for j, off in enumerate(offsets):
            cps[i, j] = skeleton_u[i] + off * binormals_u[i]
    return cps


def _apply_deformations(
    cps: np.ndarray,
    arc_frac: np.ndarray,
    tangents: np.ndarray,
    normals: np.ndarray,
    organ: dict[str, Any],
) -> np.ndarray:
    """Apply leaf-blade deformations as per-CP displacements.

    Supported effects (all scaled by a quadratic ramp from ``ramp_onset``
    to 1.0, matching ``g1_to_g3._loft_leaf``):

    - ``gutter_depths`` (N,) depress the midrib CP along -normal.
    - ``wave_normal_amp`` / ``wave_normal_freq`` / ``wave_normal_phase``:
        bulk normal-offset applied to every CP in the cross-section.
    - ``twist_max`` rotates each cross-section around its tangent by a
        linearly-ramped angle (radians).
    - ``curl_amp`` / ``curl_freq`` / ``curl_phase`` signed normal-offset
        applied ONLY to the two edge CPs (v=0 and v=1), with opposite
        signs so one edge curls up while the other curls down.

    All deformations respect the canonical orientation convention — they
    are defined in the leaf-local ``(tangent, normal, binormal)`` frame,
    so they rotate correctly with the plant.
    """
    ramp_onset = float(organ.get("ramp_onset", 0.15))
    ramp = np.clip((arc_frac - ramp_onset) / max(1e-6, 1.0 - ramp_onset), 0.0, 1.0)
    ramp = ramp * ramp  # quadratic: smooth start

    # Maturity scaling (young leaves unfurl smoothly).
    maturity = float(organ.get("maturity_fraction", 1.0))
    maturity = max(0.0, min(1.0, maturity))
    unfurl = maturity ** 0.6

    # 1. Gutter depression on the midrib CP.
    gutter_depths = organ.get("gutter_depths")
    if gutter_depths is not None:
        gd = np.asarray(gutter_depths, dtype=np.float64)
        if len(gd) != N_U:
            # Interpolate to the 11 CP stations.
            gd = np.interp(arc_frac, np.linspace(0, 1, len(gd)), gd)
        gd = gd * ramp * unfurl
        for i in range(N_U):
            cps[i, 2] += (-gd[i]) * normals[i]

    # 1b. Geometric midrib ridge: removed. Earlier versions lifted the
    # v=0.5 CP along +normal so the rib stood above the gutter floor,
    # but on curved/drooping leaves that 3D ridge reads as a protruding
    # fold rather than paint. The painted stripe is now carried by a
    # material tag on the original surface, so the cross-section keeps a
    # clean gutter U with no centerline relief and the rib reads as a
    # flat painted feature. ``midrib_amps_cm`` is still consumed
    # downstream as the trigger / gate for the optical-stripe mask.

    # 2. Wave normal (bulk vertical undulation).
    wave_amp = float(organ.get("wave_normal_amp", 0.0)) * unfurl
    if wave_amp != 0.0:
        freq = float(organ.get("wave_normal_freq", 3.5))
        phase = float(organ.get("wave_normal_phase", 0.0))
        offsets = wave_amp * ramp * np.sin(2.0 * np.pi * freq * arc_frac + phase)
        for i in range(N_U):
            cps[i, :, :] += offsets[i] * normals[i]

    # 3. Axial twist.
    twist_max = float(organ.get("twist_max", 0.0)) * unfurl
    if abs(twist_max) > 1e-6:
        twist_angles = twist_max * ramp  # radians
        for i in range(N_U):
            ang = twist_angles[i]
            if abs(ang) < 1e-9:
                continue
            # Rotate each CP around the (tangent[i]) axis through skeleton[i].
            # skeleton[i] is the midrib CP (cps[i, 2]). Use Rodrigues.
            axis = tangents[i]
            ca, sa = np.cos(ang), np.sin(ang)
            center = cps[i, 2].copy()
            for j in range(N_V):
                if j == 2:
                    continue
                v = cps[i, j] - center
                rot = v * ca + np.cross(axis, v) * sa + axis * np.dot(axis, v) * (1 - ca)
                cps[i, j] = center + rot

    # 4. Asymmetric edge curl (v=0 and v=1 move in opposite directions).
    curl_amp = float(organ.get("curl_amp", 0.0)) * unfurl
    if curl_amp != 0.0:
        freq = float(organ.get("curl_freq", 2.0))
        phase = float(organ.get("curl_phase", 0.0))
        curl_onset = float(organ.get("curl_onset", ramp_onset))
        curl_ramp = np.clip((arc_frac - curl_onset) / max(1e-6, 1.0 - curl_onset),
                            0.0, 1.0)
        curl_ramp = curl_ramp ** 2
        factors = curl_amp * curl_ramp * np.sin(2.0 * np.pi * freq * arc_frac + phase)
        for i in range(N_U):
            cps[i, 0] += (-factors[i]) * normals[i]
            cps[i, -1] += (+factors[i]) * normals[i]

    return cps


def _build_segment_ids(
    arc_frac_eval: np.ndarray,
    orig_segment_map: np.ndarray,
    skeleton_cum_frac: np.ndarray,
) -> np.ndarray:
    """Map each eval u-station to an original skeleton segment index.

    ``skeleton_cum_frac`` has length ``len(skeleton) = len(orig_segment_map)+1``
    (per-node arc fraction). ``orig_segment_map[i]`` is the original segment
    index for the i-th post-subdivision segment. For a tessellation row at
    parameter u, find segment k such that u ∈ [skeleton_cum_frac[k],
    skeleton_cum_frac[k+1]), then return ``orig_segment_map[k]``.
    """
    n_eval = len(arc_frac_eval)
    seg_idx = np.empty(n_eval, dtype=np.int32)
    # Map each eval midpoint (between rows i and i+1) to original segment.
    for ei in range(n_eval):
        u = arc_frac_eval[ei]
        k = np.searchsorted(skeleton_cum_frac, u, side="right") - 1
        k = int(np.clip(k, 0, len(orig_segment_map) - 1))
        seg_idx[ei] = int(orig_segment_map[k])
    return seg_idx


def loft_leaf_nurbs(
    organ: dict[str, Any],
    n_u_eval: int = 30,
    n_v_eval: int = 7,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
           np.ndarray, np.ndarray,
           np.ndarray | None, np.ndarray | None,
           np.ndarray, np.ndarray]:
    """Loft a leaf organ via the canonical 11×5 NURBS CP grid.

    Extends ``g1_to_g3._loft_leaf``'s contract with a ninth return value:
    the canonical ``(N_U, N_V, 3)`` CP grid after deformations. Callers
    that just want the mesh data can slice ``result[:8]``; CP-space fitters
    read ``result[8]``.

    Args:
        organ: organ dict with ``skeleton`` (N, 3), ``widths`` (N,),
            ``organ_id`` (int), optional deformation params, optional
            ``_orig_segment_map``.
        n_u_eval: tessellation rows along the leaf (default 30).
        n_v_eval: tessellation columns across the leaf (default 7).

    Returns:
        9-tuple with float64 vertices/normals/uvs, int32 indices/organ_ids/
        segment_ids/quad_indices/quad_organ_ids, and a float64 ``(N_U, N_V, 3)``
        CP grid.
    """
    organ_id = int(organ["organ_id"])

    # Library-CP path (Phase C): organ carries a pre-fitted local-frame CP
    # grid plus collar pose. Scale by current/mature length, transform into
    # world coordinates, then skip the quad-ribbon build + deformation block.
    cps_local = organ.get("surface_cps_local")
    if cps_local is not None:
        from .canonical_library import from_local_frame, build_compound_leaf_cps
        cps_local = np.asarray(cps_local, dtype=np.float64)
        if cps_local.shape != (N_U, N_V, 3):
            raise ValueError(
                f"surface_cps_local must be {(N_U, N_V, 3)}; got {cps_local.shape}"
            )
        mature_length = float(organ.get("mature_length", 1.0))
        current_length = float(organ.get("current_length", mature_length))
        if organ.get("surface_cps_normalized", False):
            scale = max(current_length, 0.0)
        else:
            scale = max(current_length, 0.0) / max(mature_length, 1e-9)
        # Width matures ~2x faster than length (young maize blades are
        # wider relative to length than mature ones — the V2-V3 blade
        # reaches near-mature width by ~50% of final length, then keeps
        # elongating). Without decoupling, a V3 leaf at scale=0.3 renders
        # as a 30%-width hairline and misses the Nielsen reference where
        # young blades are visibly broader. Width saturates at scale=0.5;
        # length keeps scaling linearly. Mature endpoint (scale=1.0) is
        # bit-for-bit unchanged. Floor at 0.15 so very-young emerging
        # leaves have a visible but narrower blade.
        width_maturity = max(0.15, min(1.0, scale / 0.50))
        cps_local = cps_local.copy()
        cps_local[..., 0] *= width_maturity   # lateral (blade width)
        cps_local[..., 1] *= scale            # droop axis
        cps_local[..., 2] *= scale            # midrib-along-tangent

        # Tip taper (mirror of the legacy path in cplantbox_adapter.py:991):
        # library-path CPs inherit the scanned plant's tip profile directly.
        # Plants whose scan CPs don't narrow to a point produce blunt-tip
        # renders (e.g. seed=42 pos=4 has near-constant edge offsets in
        # rows 5-8, so the back half of the blade stays wide then cuts off
        # abruptly). Enforce an envelope on the per-row edge offset,
        # shrinking any row that exceeds it. Rows already below the
        # envelope are left alone so we never widen an existing narrow
        # region.
        mid_col = cps_local.shape[1] // 2
        midrib_col = cps_local[:, mid_col:mid_col + 1, :].copy()
        offset_from_mid = (cps_local - midrib_col).copy()
        edge_off = np.linalg.norm(offset_from_mid[:, -1, :], axis=1)
        w_max = float(edge_off.max()) if edge_off.size else 0.0
        if w_max > 1e-6:
            # Position 0 (V1 seedling leaf, Nielsen "rounded leaf #1") gets
            # a gentler envelope + active widening of narrow CPs and skips
            # the hard midrib pinch so the tip stays blunt/oval rather than
            # pointed. All other positions keep the production tapered-to-
            # point shape.
            rounded_tip = bool(organ.get("rounded_tip", False))
            if rounded_tip:
                taper_start = 0.85
                taper_end = 0.55
            else:
                taper_start = 0.35
                taper_end = 0.01
            u_rows = np.linspace(0.0, 1.0, cps_local.shape[0])
            for i, u in enumerate(u_rows):
                if u <= taper_start:
                    continue
                t = (u - taper_start) / (1.0 - taper_start)
                envelope = 1.0 - (t * t) * (1.0 - taper_end)
                target_edge = envelope * w_max
                current_edge = edge_off[i]
                if current_edge < 1e-9:
                    # Degenerate row (collapsed to midrib in the source CP
                    # grid). For rounded-tip, reconstruct symmetric outward
                    # offsets from the peak row scaled to the target width.
                    if rounded_tip:
                        peak_row = int(np.argmax(edge_off))
                        peak_edge = float(edge_off[peak_row])
                        if peak_edge > 1e-6:
                            scale_ratio = target_edge / peak_edge
                            cps_local[i] = (
                                midrib_col[i]
                                + offset_from_mid[peak_row] * scale_ratio
                            )
                    continue
                if rounded_tip and current_edge < target_edge:
                    # Narrow-tip CP grid (pointed-leaf scan): widen outward
                    # to match the rounded envelope.
                    widen = target_edge / current_edge
                    cps_local[i] = (
                        midrib_col[i] + offset_from_mid[i] * widen
                    )
                    continue
                if current_edge <= target_edge:
                    continue
                shrink = target_edge / current_edge
                cps_local[i] = midrib_col[i] + offset_from_mid[i] * shrink
            # Hard pinch: collapse the final U-row to the midrib so the NURBS
            # surface terminates in a true point regardless of donor shape.
            # Skipped for rounded-tip (pos 0) leaves so the blade ends in
            # an oval blunt instead of a point.
            if not rounded_tip:
                cps_local[-1, :, :] = midrib_col[-1, 0, :]

        # Muted procedural deformations in leaf-local frame. The base shape
        # already carries the data-driven blade curvature; this pass adds
        # per-leaf ruffle / twist / wave so aggregated libraries don't look
        # artificially smooth. In leaf-local frame the midrib tangent is +z
        # and the blade-normal is +y, so we pass constant tangents/normals.
        arc_frac_u = np.linspace(0.0, 1.0, N_U)
        tangents_local = np.tile(np.array([0.0, 0.0, 1.0]), (N_U, 1))
        normals_local = np.tile(np.array([0.0, 1.0, 0.0]), (N_U, 1))
        cps_local = _apply_deformations(
            cps_local.copy(), arc_frac_u, tangents_local, normals_local, organ
        )
        # Re-pinch after deformations: edge-curl is strongest at the tip and
        # would splay the last CP row back open. Collapse again to guarantee
        # the NURBS surface terminates in a point — except for rounded-tip
        # (pos 0, V1 seedling leaf) where the blunt oval must be preserved.
        if not bool(organ.get("rounded_tip", False)):
            cps_local[-1, :, :] = cps_local[-1, N_V // 2, :]

        # Compound sheath+blade path: when stem_radius and sheath_length are
        # present, wrap the blade CPs with closed-tube sheath rows + transition
        # rows that peel the seam open. This produces a single NURBS surface
        # representing the whole leaf (sheath + blade), matching the maize
        # biology where the two are one organ.
        _sr = organ.get("stem_radius_cm")
        _sl = organ.get("sheath_length_cm")
        stem_radius_cm = float(_sr) if _sr is not None else 0.0
        sheath_length_cm = float(_sl) if _sl is not None else 0.0
        # The adapter hands us the MATURE median sheath length from the
        # MaizeField3D reference (organ-independent of current age). In real
        # maize the sheath reaches near-mature length ~3x faster than the
        # blade — by the time a leaf's blade is 30 % of mature length, its
        # sheath is already at full length, forming the telescoping whorl
        # column that wraps the stem on V2–V5 plants. Scaling by the blade's
        # current/mature ratio would leave young plants with tiny sheaths
        # and a visible bare stem between collars, contradicting the V3
        # reference (Nielsen 2004). Saturating curve: sheath fraction
        # reaches 1.0 at blade scale = 0.3, clamped to 0.05 floor so
        # very-young emerging leaves still have a visible sheath stub.
        # Mature endpoint (scale=1.0) unchanged (→ sheath_maturity=1.0).
        sheath_maturity = max(0.05, min(1.0, scale / 0.30))
        sheath_length_cm *= sheath_maturity
        use_compound = stem_radius_cm > 0.0 and sheath_length_cm > 0.0

        collar_pos = np.asarray(organ["collar_pos"], dtype=np.float64)
        tangent = np.asarray(organ["collar_tangent"], dtype=np.float64)
        if use_compound:
            # The sheath must wrap the parent STEM, not extend along the
            # blade's tangent: for drooping/horizontal leaves these two
            # directions differ wildly. Use the parent stem's local tangent
            # at the collar if available (``parent_tangent`` carries that
            # from the adapter); otherwise fall back to world +z.
            stem_axis_world = np.asarray(
                organ.get("parent_tangent", (0.0, 0.0, 1.0)),
                dtype=np.float64,
            ).reshape(3)
            sa_len = float(np.linalg.norm(stem_axis_world))
            if sa_len < 1e-9:
                stem_axis_world = np.array([0.0, 0.0, 1.0], dtype=np.float64)
            else:
                stem_axis_world = stem_axis_world / sa_len

            # Build the leaf-local rotation matching ``from_local_frame``
            # so we can transform the world stem axis into the leaf-local
            # frame that ``build_compound_leaf_cps`` operates in.
            t = tangent / max(float(np.linalg.norm(tangent)), 1e-12)
            up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
            x_local_w = np.cross(t, up)
            x_len = float(np.linalg.norm(x_local_w))
            if x_len < 1e-6:
                # Blade tangent parallel to world up: any perpendicular axis
                # works; pick world +x as the tie-breaker to match
                # ``from_local_frame``'s fallback branch.
                alt = (
                    np.array([1.0, 0.0, 0.0])
                    if abs(t[0]) < 0.9
                    else np.array([0.0, 1.0, 0.0])
                )
                x_local_w = np.cross(t, alt)
                x_len = float(np.linalg.norm(x_local_w))
            x_local_w = x_local_w / max(x_len, 1e-12)
            y_local_w = np.cross(t, x_local_w)
            y_local_w = y_local_w / max(float(np.linalg.norm(y_local_w)), 1e-12)
            R_leaf = np.column_stack([x_local_w, y_local_w, t])
            stem_axis_local = R_leaf.T @ stem_axis_world

            # Optional stem-taper awareness: adapter may supply a
            # callable ``parent_stem_radius_at_z_cm(z_local_cm) -> cm``
            # that returns the stem radius at a given leaf-local z
            # (negative below collar). When present, the sheath cup
            # tracks the stem's taper instead of assuming a constant
            # radius across the wrapped height.
            _raw_stem_r = organ.get("parent_stem_radius_at_z_cm")
            stem_r_at_z: Callable[[float], float] | None
            if callable(_raw_stem_r):
                stem_r_at_z = _raw_stem_r  # type: ignore[assignment]
            else:
                stem_r_at_z = None

            # Sheath-cap gate (young-plant whorl fix, 2026-04-22; refined
            # 2026-05-02). When the sheath length is sourced from Vidal 2021
            # SupData1 (sheath_provenance="vidal_per_rank"), the value is
            # cultivar-calibrated and trustworthy at every maturity — render
            # full botanical length so the sheath wraps the stem and forms
            # the telescoping whorl column. Otherwise (MF3D medians or
            # missing data, e.g. non-maize XMLs) keep the conservative
            # young_frac blend so master renders stay bit-for-bit identical
            # for those code paths.
            sheath_provenance = organ.get("sheath_provenance")
            if sheath_provenance == "vidal_per_rank":
                max_sheath_cap = float(sheath_length_cm)
            else:
                young_frac = max(0.0, min(1.0, 1.0 - scale / 0.50))
                default_cap = 2.5 * float(stem_radius_cm)
                max_sheath_cap = (
                    young_frac * float(sheath_length_cm)
                    + (1.0 - young_frac) * default_cap
                )
            explicit_cup_cap = organ.get("sheath_cup_max_length_cm")
            if explicit_cup_cap is not None:
                max_sheath_cap = min(max_sheath_cap, float(explicit_cup_cap))
            cps_local = build_compound_leaf_cps(
                cps_local,
                stem_radius_cm=stem_radius_cm,
                sheath_length_cm=sheath_length_cm,
                stem_axis=stem_axis_local,
                stem_radius_at_z=stem_r_at_z,
                max_sheath_length_cm=max_sheath_cap,
            )

        cps = from_local_frame(cps_local, collar_pos, tangent)
        # Use the CPlantBox node positions as the reference skeleton for
        # per-triangle segment-ID mapping (arc_frac along this polyline).
        skeleton = np.asarray(organ["skeleton"], dtype=np.float64)
        if len(skeleton) < 2:
            # Fallback: derive midrib from library CPs. Use v=N_V//2 (midrib
            # column) on the blade portion of the compound grid.
            n_v_used = cps.shape[1]
            cps_blade_only = cps if not use_compound else cps[-N_U:]
            skeleton = cps_blade_only[:, n_v_used // 2, :].copy()
    else:
        skeleton = np.asarray(organ["skeleton"], dtype=np.float64)
        widths = np.asarray(organ["widths"], dtype=np.float64)
        if len(skeleton) < 3:
            raise ValueError(
                f"NURBS lofter needs ≥3 skeleton points; got {len(skeleton)}"
            )

        # --- 1. Resample skeleton to N_U=11 uniform arc-length stations ---
        skel_u, widths_u, arc_frac_u = _resample_skeleton(skeleton, widths, N_U)

        # --- 2. Local frames at each station ---
        tangents_u, normals_u, binormals_u = _darboux_frames(skel_u)

        # --- 3. Canonical 11×5 CP grid (placed via binormal only) ---
        cps = _build_cp_grid(skel_u, widths_u, binormals_u)

        # --- 4. Deformations (need tangent+normal for twist/curl/wave) ---
        cps = _apply_deformations(
            cps, arc_frac_u, tangents_u, normals_u, organ
        )

    # --- 5. PlantGL patch + tessellation ---
    # Use the general-shape patch builder when the compound path has been
    # taken (cps shape differs from (N_U, N_V)); otherwise use the canonical
    # (11, 5) patch builder for backward compatibility.
    if cps.shape == (N_U, N_V, 3):
        patch = cp_grid_to_plantgl_patch(cps)
    else:
        patch = cp_grid_to_plantgl_patch_general(cps)
    verts, norms = eval_grid(patch, n_u=n_u_eval, n_v=n_v_eval)
    # verts layout: row-major (i * n_v_eval + j). Reshape to (n_u_eval, n_v_eval, 3)
    # for face building.
    n_v = n_v_eval
    n_u = n_u_eval

    # UVs uniform on [0,1]^2 matching tessellation.
    us = np.linspace(0.0, 1.0, n_u)
    vs = np.linspace(0.0, 1.0, n_v)
    uv_grid = np.empty((n_u, n_v, 2), dtype=np.float64)
    for i, u in enumerate(us):
        for j, v in enumerate(vs):
            uv_grid[i, j] = (u, v)
    uvs = uv_grid.reshape(-1, 2)

    # Triangles: 2 per quad cell in an (n_u-1) × (n_v-1) grid.
    n_cells = (n_u - 1) * (n_v - 1)
    indices = np.empty((2 * n_cells, 3), dtype=np.int32)
    quads = np.empty((n_cells, 4), dtype=np.int32)
    # Midrib triangle mask: True for cells in the central v-band (Stage B
    # optical routing). Active only when the organ carries non-zero
    # midrib_amps_cm. Real maize: rib is ~15 % of width near the sheath
    # junction, narrowing to ~5 % at the tip — matches the per-u band
    # taper computed by the adapter. Coarser tessellations fall back to
    # the nearest strip pair around v=0.5.
    midrib_amps = organ.get("midrib_amps_cm")
    has_midrib_active = (midrib_amps is not None
                         and float(np.max(np.abs(np.asarray(midrib_amps)))) > 1e-6)
    midrib_band_v_input = organ.get("midrib_band_v_frac", 0.025)
    if np.isscalar(midrib_band_v_input):
        midrib_band_v_per_u = np.full(n_u, float(midrib_band_v_input))
    else:
        _arr = np.asarray(midrib_band_v_input, dtype=np.float64)
        if len(_arr) != n_u:
            _arr = np.interp(np.linspace(0, 1, n_u),
                             np.linspace(0, 1, len(_arr)), _arr)
        midrib_band_v_per_u = _arr
    is_midrib_tri = np.zeros(2 * n_cells, dtype=bool)
    # Per-u amplitude gate: the adapter's basal ramp drives midrib_amps
    # to ~0 near the collar; without this gate the optical tagging would
    # paint the full arc despite the geometric ridge fading out. Skip
    # u-rows whose local amplitude is below 1 % of the organ max so the
    # painted stripe matches the geometric ridge.
    if has_midrib_active:
        _amp_arr = np.asarray(midrib_amps, dtype=np.float64)
        if len(_amp_arr) != n_u:
            midrib_amps_per_u = np.interp(np.linspace(0, 1, n_u),
                                          np.linspace(0, 1, len(_amp_arr)),
                                          _amp_arr)
        else:
            midrib_amps_per_u = _amp_arr
        amp_threshold = 0.01 * float(np.max(np.abs(midrib_amps_per_u)))
    else:
        midrib_amps_per_u = None
        amp_threshold = 0.0
    c = 0
    for i in range(n_u - 1):
        if midrib_amps_per_u is not None:
            _row_amp = 0.5 * (abs(midrib_amps_per_u[i])
                              + abs(midrib_amps_per_u[i + 1]))
            _row_active = _row_amp > amp_threshold
        else:
            _row_active = False
        for j in range(n_v - 1):
            v00 = i * n_v + j
            v10 = (i + 1) * n_v + j
            v11 = (i + 1) * n_v + (j + 1)
            v01 = i * n_v + (j + 1)
            indices[2 * c] = (v00, v10, v11)
            indices[2 * c + 1] = (v00, v11, v01)
            quads[c] = (v00, v10, v11, v01)
            if _row_active:
                # vs[j] / vs[j+1] are the v-fractions of this strip's rails.
                # Mark midrib when the mean |v - 0.5| sits inside the band
                # for THIS u-row (the band tapers along the leaf length).
                mean_off = 0.5 * (abs(vs[j] - 0.5) + abs(vs[j + 1] - 0.5))
                band_i = 0.5 * (midrib_band_v_per_u[i]
                                + midrib_band_v_per_u[i + 1])
                if mean_off <= band_i:
                    is_midrib_tri[2 * c] = True
                    is_midrib_tri[2 * c + 1] = True
            c += 1

    organ_ids = np.full(len(indices), organ_id, dtype=np.int32)
    quad_organ_ids = np.full(n_cells, organ_id, dtype=np.int32)

    # --- 6. Per-triangle segment IDs ---
    orig_segment_map = organ.get("_orig_segment_map")
    segment_ids = np.full(len(indices), -1, dtype=np.int32)
    if orig_segment_map is not None and len(orig_segment_map) > 0:
        orig_segment_map = np.asarray(orig_segment_map, dtype=np.int32)
        # Arc-length fraction for each node in the (post-subdivision)
        # skeleton — the same skeleton that orig_segment_map indexes.
        cum = _compute_arc_lengths(skeleton)
        total = float(cum[-1])
        if total > 1e-9:
            skel_cum_frac = cum / total
            # Per-row midpoint u-value maps to original segment.
            u_mids = (us[:-1] + us[1:]) / 2.0  # (n_u-1,) midpoints per row-pair
            row_seg = _build_segment_ids(u_mids, orig_segment_map, skel_cum_frac)
            # Each row-pair contains 2*(n_v-1) triangles.
            tris_per_row = 2 * (n_v - 1)
            for i in range(n_u - 1):
                segment_ids[i * tris_per_row:(i + 1) * tris_per_row] = row_seg[i]

    return (verts, indices, norms, uvs, organ_ids, segment_ids,
            quads, quad_organ_ids, cps, is_midrib_tri)


__all__ = ["loft_leaf_nurbs"]
