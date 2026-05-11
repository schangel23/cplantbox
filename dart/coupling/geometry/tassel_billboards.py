"""Anther (spikelet) cross-billboards for tassel organs.

Ported from ``dart/coupling/output/blender_preview/_gen_tassel_test.py`` to
the production pipeline. The visible fuzz on a tasseling maize plant is
thousands of narrow cream-coloured anthers hanging from the spike and
primary branches. Modelling each as full 3D geometry is prohibitive for
DART ray tracing, so each anther is instead represented by a cross-billboard
(two perpendicular narrow quads) whose silhouette drives APAR changes at
the VT stage.

Billboards are appended into an existing :class:`G3Mesh` in place, tagged
with the tassel organ's ``organ_id`` for DART group routing and
``segment_id = -1`` (they are not CPlantBox skeleton segments).

Tassel organs are detected by ``name`` prefix: ``tassel_spike_`` and
``tassel_branch_``. Non-tassel organs are ignored.
"""
from __future__ import annotations

import numpy as np


SPIKELET_SPACING_CM = 0.35
SPIKELET_LEN_CM = 0.35
SPIKELET_WIDTH_CM = 0.08
SPIKELET_TIP_SCALE = 0.85
SPIKELET_AZIMUTHS = 3
SPIKELETS_PER_AZIMUTH = 2
# Spikelet direction: bias along the local stem tangent so the anthers
# read as a feathery plume following the branch curve, with a small
# radial kick to spread the 3 azimuthal copies around the stem.
SPIKELET_ALONG_FRAC = 0.70
SPIKELET_OUTWARD_FRAC = 0.30
SPIKELET_HANG_FRAC = 0.0
SPIKELET_JITTER_DEG = 12.0
SPIKELET_PAIR_OFFSET_CM = 0.08
SPIKELET_CROSS_BILLBOARD = True

# Leave the peduncle (lower 30 % of the central spike) bare — real maize
# tassels have a smooth glabrous peduncle below the lowest branch.
SPIKE_SPIKELET_START_FRAC = 0.30
BRANCH_SPIKELET_START_FRAC = 0.10

SPIKE_PREFIX = "tassel_spike_"
BRANCH_PREFIX = "tassel_branch_"


def spikelet_billboards(skeleton: np.ndarray, start_frac: float,
                        rng: np.random.Generator,
                        start_arc_cm: float | None = None,
                        ) -> tuple[np.ndarray, np.ndarray]:
    """Generate anther-like cross-billboards along a skeleton.

    ``start_arc_cm`` overrides ``start_frac * total_len`` when provided —
    used for the central spike, where the peduncle should stay bare up to
    the first branch attachment regardless of total spike length.

    Returns ``(verts, tris)``; ``verts`` is ``(V, 3)``, ``tris`` is
    ``(T, 3)`` with vertex indices local to the returned ``verts`` array.
    """
    skel = np.asarray(skeleton, dtype=np.float64)
    if len(skel) < 2:
        return np.empty((0, 3)), np.empty((0, 3), dtype=np.int64)

    seg_len = np.linalg.norm(np.diff(skel, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seg_len)])
    total_len = arc[-1]
    if total_len < 2 * SPIKELET_SPACING_CM:
        return np.empty((0, 3)), np.empty((0, 3), dtype=np.int64)

    start_arc = (start_arc_cm if start_arc_cm is not None
                 else start_frac * total_len)
    if start_arc >= total_len * 0.98:
        return np.empty((0, 3)), np.empty((0, 3), dtype=np.int64)
    arcs = np.arange(start_arc, total_len * 0.98, SPIKELET_SPACING_CM)
    verts: list[np.ndarray] = []
    tris: list[list[int]] = []
    gravity = np.array([0.0, 0.0, -1.0])

    for s in arcs:
        idx = int(np.searchsorted(arc, s)) - 1
        idx = max(0, min(idx, len(skel) - 2))
        denom = arc[idx + 1] - arc[idx] + 1e-12
        frac = (s - arc[idx]) / denom
        pos = skel[idx] + frac * (skel[idx + 1] - skel[idx])

        tan = skel[idx + 1] - skel[idx]
        tan = tan / (np.linalg.norm(tan) + 1e-12)

        up = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(tan, up)) > 0.95:
            up = np.array([1.0, 0.0, 0.0])
        bi = np.cross(tan, up); bi /= np.linalg.norm(bi) + 1e-12
        no = np.cross(tan, bi); no /= np.linalg.norm(no) + 1e-12

        taper = 1.0 - (1.0 - SPIKELET_TIP_SCALE) * (s / total_len)
        L = SPIKELET_LEN_CM * taper
        W = SPIKELET_WIDTH_CM * taper

        az_jitter = rng.uniform(-np.pi / (3 * SPIKELET_AZIMUTHS),
                                np.pi / (3 * SPIKELET_AZIMUTHS))

        for k in range(SPIKELET_AZIMUTHS):
            phi = 2 * np.pi * k / SPIKELET_AZIMUTHS + az_jitter
            radial = np.cos(phi) * bi + np.sin(phi) * no

            # Bias along the local stem tangent so anthers lay along the
            # branch curve (feathery plume), with a small outward kick to
            # spread the 3 azimuthal copies around the stem. Gravity term
            # is opt-in (default 0).
            base_dir = (SPIKELET_ALONG_FRAC * tan
                        + SPIKELET_OUTWARD_FRAC * radial
                        + SPIKELET_HANG_FRAC * gravity)
            base_dir /= np.linalg.norm(base_dir) + 1e-12

            pair_axis = np.cross(base_dir, radial)
            n = np.linalg.norm(pair_axis)
            pair_axis = pair_axis / n if n > 1e-9 else bi.copy()

            for p in range(SPIKELETS_PER_AZIMUTH):
                if SPIKELETS_PER_AZIMUTH > 1:
                    off_frac = p - (SPIKELETS_PER_AZIMUTH - 1) / 2.0
                    offset = off_frac * SPIKELET_PAIR_OFFSET_CM * pair_axis
                else:
                    offset = np.zeros(3)
                p_base = pos + offset

                jitter = rng.normal(scale=np.radians(SPIKELET_JITTER_DEG),
                                    size=3)
                jitter[2] *= 0.3  # keep downward bias
                spike_dir = base_dir + jitter
                spike_dir /= np.linalg.norm(spike_dir) + 1e-12

                wa = np.cross(spike_dir, np.array([0.0, 0.0, 1.0]))
                if np.linalg.norm(wa) < 1e-6:
                    wa = np.cross(spike_dir, np.array([1.0, 0.0, 0.0]))
                wa /= np.linalg.norm(wa) + 1e-12
                wb = np.cross(spike_dir, wa); wb /= np.linalg.norm(wb) + 1e-12

                width_axes = [wa, wb] if SPIKELET_CROSS_BILLBOARD else [wa]

                # Six perimeter vertices on an ellipse with major axis L
                # along spike_dir, minor axis W along width_dir.  Fan-
                # triangulated from the base vertex (v0).  Reads as an
                # oval silhouette from any angle when paired with the
                # perpendicular cross billboard.
                phis = (np.pi, 2.0 * np.pi / 3.0, np.pi / 3.0,
                        0.0, -np.pi / 3.0, -2.0 * np.pi / 3.0)
                for width_dir in width_axes:
                    center = p_base + 0.5 * L * spike_dir
                    v0 = len(verts)
                    for phi in phis:
                        verts.append(center
                                     + 0.5 * L * np.cos(phi) * spike_dir
                                     + 0.5 * W * np.sin(phi) * width_dir)
                    for k in range(1, 5):
                        tris.append([v0, v0 + k, v0 + k + 1])

    return (np.asarray(verts, dtype=np.float64),
            np.asarray(tris, dtype=np.int64))


def append_tassel_billboards(mesh, organ_dicts, seed: int | None = 42,
                             verbose: bool = False):
    """Append anther cross-billboards to tassel organs in ``mesh``.

    Detects tassel organs in ``organ_dicts`` via ``name`` prefix
    (``tassel_spike_`` → spike schedule, ``tassel_branch_`` → branch
    schedule). Non-tassel organs and organs with no ``skeleton`` key
    (e.g. NURBS leaves with only ``surface_cps_local``) are skipped.

    Modifies ``mesh`` in place and returns it.

    Args:
        mesh: G3Mesh to augment.
        organ_dicts: Organ dicts as returned by
            ``extract_organs_for_lofter``.
        seed: RNG seed for per-run determinism. Use ``None`` for a fresh
            nondeterministic draw.
        verbose: If True, print a summary of billboards added.
    """
    rng = np.random.default_rng(seed)

    extra_verts: list[np.ndarray] = []
    extra_tris: list[np.ndarray] = []
    extra_normals: list[np.ndarray] = []
    extra_uvs: list[np.ndarray] = []
    extra_organ_ids: list[np.ndarray] = []
    vertex_offset = len(mesh.vertices)
    n_spike = n_branch = 0

    # Project each branch's base point onto the spike skeleton to find the
    # arc-length where it attaches. The highest such arc is the top of the
    # branch whorl — below that, the central rachis stays bare (peduncle +
    # branch zone), and only the spike above the whorl carries spikelets.
    spike_organs = [o for o in organ_dicts
                    if o.get("name", "").startswith(SPIKE_PREFIX)
                    and o.get("skeleton") is not None]
    branch_organs = [o for o in organ_dicts
                     if o.get("name", "").startswith(BRANCH_PREFIX)
                     and o.get("skeleton") is not None]
    spike_start_arcs: dict[int, float] = {}
    for spike in spike_organs:
        spike_skel = np.asarray(spike["skeleton"], dtype=np.float64)
        if len(spike_skel) < 2 or not branch_organs:
            continue
        spike_seg_len = np.linalg.norm(np.diff(spike_skel, axis=0), axis=1)
        spike_arc = np.concatenate([[0.0], np.cumsum(spike_seg_len)])
        max_arc = 0.0
        for branch in branch_organs:
            base = np.asarray(branch["skeleton"], dtype=np.float64)[0]
            d2 = np.sum((spike_skel - base) ** 2, axis=1)
            idx = int(np.argmin(d2))
            arc_at = float(spike_arc[idx])
            if arc_at > max_arc:
                max_arc = arc_at
        if max_arc > 0.0:
            spike_start_arcs[int(spike["organ_id"])] = max_arc

    for organ in organ_dicts:
        name = organ.get("name", "")
        start_arc_cm: float | None = None
        if name.startswith(SPIKE_PREFIX):
            start_frac = SPIKE_SPIKELET_START_FRAC
            start_arc_cm = spike_start_arcs.get(int(organ["organ_id"]))
            n_spike += 1
        elif name.startswith(BRANCH_PREFIX):
            start_frac = BRANCH_SPIKELET_START_FRAC
            n_branch += 1
        else:
            continue

        skeleton = organ.get("skeleton")
        if skeleton is None:
            continue

        v, t = spikelet_billboards(np.asarray(skeleton), start_frac, rng,
                                   start_arc_cm=start_arc_cm)
        if len(v) == 0:
            continue

        extra_verts.append(v)
        extra_tris.append(t + vertex_offset)
        extra_normals.append(np.tile([0.0, 0.0, 1.0], (len(v), 1)))
        extra_uvs.append(np.zeros((len(v), 2)))
        extra_organ_ids.append(np.full(len(t), organ["organ_id"],
                                       dtype=np.int32))
        vertex_offset += len(v)

    if not extra_verts:
        if verbose:
            print(f"  Tassel billboards: 0 (no tassel organs matched)")
        return mesh

    n_new_verts = sum(len(v) for v in extra_verts)
    n_new_tris = sum(len(t) for t in extra_tris)

    mesh.vertices = np.concatenate([mesh.vertices, *extra_verts])
    mesh.indices = np.concatenate([mesh.indices, *extra_tris]).astype(np.int32)
    mesh.normals = np.concatenate([mesh.normals, *extra_normals])
    mesh.uvs = np.concatenate([mesh.uvs, *extra_uvs])
    mesh.organ_ids = np.concatenate([mesh.organ_ids,
                                     *extra_organ_ids]).astype(np.int32)
    mesh.segment_ids = np.concatenate([
        mesh.segment_ids,
        np.full(n_new_tris, -1, dtype=np.int32),
    ])
    # Tassel anther billboards are not midrib geometry — extend with False.
    if hasattr(mesh, "is_midrib"):
        mesh.is_midrib = np.concatenate([
            mesh.is_midrib,
            np.zeros(n_new_tris, dtype=bool),
        ])
    # Billboards are not end caps either — extend with False so the
    # degenerate-triangle filter still culls any over-thin anther quads.
    if hasattr(mesh, "is_cap"):
        mesh.is_cap = np.concatenate([
            mesh.is_cap,
            np.zeros(n_new_tris, dtype=bool),
        ])

    if verbose:
        print(f"  Tassel billboards: +{n_new_verts} verts, +{n_new_tris} tris "
              f"({n_spike} spike + {n_branch} branch organs)")

    return mesh
