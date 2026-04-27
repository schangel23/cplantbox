"""Preview: synthetic tassel geometry attached to a day-55 maize plant.

v2 (2026-04-21): matched to real maize tassel photo reference —
  - 6-8 erect branches (was 18 spoke-like), minimal droop
  - insertion in tight basal whorl (first ~15 % of spike)
  - insertion angle 25-35° from vertical (was 30-55° drooping)
  - spikelet billboards along spike AND branches (critical for silhouette)

Uses the existing g1_to_g3 lofter for the naked axis tubes; spikelets are
appended directly into the returned G3Mesh as extra triangles (billboard
quads) so we don't need a new lofter code path.

Run:
    cd /home/lukas/PHD/CPlantBox
    source cpbenv/bin/activate
    PYTHONPATH=. python dart/coupling/output/blender_preview/_gen_tassel_test.py
"""
from __future__ import annotations
from pathlib import Path
import sys

import numpy as np
import plantbox as pb

from dart.coupling.config import DEFAULT_XML
from dart.coupling.growth.grow import grow_plant
from dart.coupling.geometry import extract_organs_for_lofter, loft_organs

OUT = Path(__file__).resolve().parent
SKEL_SEED = 7
DONOR_SEED = 4
GROW_DAY = 55


# ---------- tassel geometry parameters ----------
# Mature tassel total height bounded to ~25 cm (spike) + 15 cm branches,
# giving a silhouette ≤ 40 cm tall from base of spike to branch tip.
SPIKE_LEN_MATURE_CM = 40.0              # target total height 40 cm
SPIKE_DIAM_BASE = 0.50        # full-width (diameter), cm
SPIKE_DIAM_TIP = 0.07         # strongly tapered tip
SPIKE_N_NODES = 40

BRANCH_LEN_MATURE_CM = 15.0             # proportional to taller spike
BRANCH_DIAM_BASE = 0.22
BRANCH_DIAM_TIP = 0.03        # strongly tapered
BRANCH_N_NODES = 18
N_BRANCHES = 7
BRANCH_INSERTION_ANGLE_BASAL = 38.0     # from vertical, lowest branch (mature)
BRANCH_INSERTION_ANGLE_DISTAL = 26.0    # from vertical, highest branch (mature)
BRANCH_INSERTION_FRAC_LOW = 0.35        # lowest branch at 35% up spike (was 20%)
BRANCH_INSERTION_FRAC_HIGH = 0.55       # highest branch at 55% up spike (was 40%)
BRANCH_STRAIGHTNESS = 0.01              # very slight outward arc (no droop)
# Young branches hug the spike: insertion angles scale with length_frac so
# emerging tassels are tightly packed (nearly vertical) and splay out as
# they mature. effective_angle = mature_angle * (YOUNG_ANGLE_FLOOR + (1-floor)*frac)
YOUNG_ANGLE_FLOOR = 0.15      # at frac=0, branches ~15% of mature splay → near-vertical

# ---------- anther (spikelet) billboard parameters ----------
# Model: thin rod-like billboards hanging mostly DOWNWARD from each node,
# with a slight outward component. Matches the close-up photo where
# anthers dangle as narrow cream-coloured rods.
SPIKELET_SPACING_CM = 0.35              # denser than v2 (was 0.45)
SPIKELET_LEN_CM = 0.65                  # shorter rods (was 0.9 ribbons)
SPIKELET_WIDTH_CM = 0.06                # much narrower (was 0.18)
SPIKELET_TIP_SCALE = 0.75               # less taper — anthers are fairly uniform
SPIKELET_AZIMUTHS = 3                   # 3 radial positions per cluster
SPIKELETS_PER_AZIMUTH = 2               # paired per azimuth → 6 per node
SPIKELET_HANG_FRAC = 0.80               # 0=horizontal radial, 1=full down
SPIKELET_OUTWARD_FRAC = 0.30            # outward component
SPIKELET_JITTER_DEG = 12.0              # per-anther direction jitter
SPIKELET_PAIR_OFFSET_CM = 0.08          # lateral offset between paired anthers
SPIKELET_CROSS_BILLBOARD = True         # 2 perpendicular quads per anther for volume
SPIKE_SPIKELET_START_FRAC = 0.12
BRANCH_SPIKELET_START_FRAC = 0.08

# Scenarios
SCENARIOS = [
    ("day60_emerging", 0.70),   # 28 cm tall (0.70 × 40)
    ("day65_mature",   1.00),   # 40 cm tall
]


def _find_main_stem_apex(organs) -> np.ndarray:
    for o in organs:
        if o.get("type") == "stem" and o.get("name", "").startswith("stem_0"):
            skel = np.asarray(o["skeleton"])
            return skel[np.argmax(skel[:, 2])].copy()
    best = None
    for o in organs:
        skel = np.asarray(o["skeleton"])
        top = skel[np.argmax(skel[:, 2])]
        if best is None or top[2] > best[2]:
            best = top
    return best.copy()


STEM_OVERLAP_CM = 3.0  # tassel base dips into stem tube to hide smoothing cap-shrink


def _make_spike_skeleton(apex_xyz: np.ndarray, length_cm: float, rng) -> np.ndarray:
    n = SPIKE_N_NODES
    lean_dir = rng.normal(size=2); lean_dir /= np.linalg.norm(lean_dir) + 1e-12
    lean_amp = np.tan(np.radians(rng.uniform(0.0, 4.0)))
    dz = np.linspace(-STEM_OVERLAP_CM, length_cm - STEM_OVERLAP_CM, n)
    dx = lean_amp * dz * lean_dir[0]
    dy = lean_amp * dz * lean_dir[1]
    skel = np.stack([apex_xyz[0] + dx, apex_xyz[1] + dy, apex_xyz[2] + dz], axis=1)
    return skel


def _make_branch_skeleton(insertion_xyz: np.ndarray, azimuth_rad: float,
                          length_cm: float, insertion_angle_deg: float,
                          rng) -> np.ndarray:
    """Nearly-straight, erect branch; very slight outward arc, no droop."""
    n = BRANCH_N_NODES
    theta = np.radians(insertion_angle_deg)
    t = np.array([
        np.sin(theta) * np.cos(azimuth_rad),
        np.sin(theta) * np.sin(azimuth_rad),
        np.cos(theta),
    ])
    ds = length_cm / (n - 1)
    skel = [insertion_xyz.copy()]
    pos = insertion_xyz.copy()
    # small outward bending (away from vertical, not toward ground)
    radial = np.array([np.cos(azimuth_rad), np.sin(azimuth_rad), 0.0])
    for _ in range(n - 1):
        pos = pos + ds * t
        skel.append(pos.copy())
        t = t + BRANCH_STRAIGHTNESS * ds * radial
        t /= np.linalg.norm(t) + 1e-12
    return np.array(skel)


def _tapered_widths(base_cm: float, tip_cm: float, n: int) -> np.ndarray:
    return np.linspace(base_cm, tip_cm, n)


def _build_tassel_organs(apex_xyz: np.ndarray, length_frac: float,
                         next_organ_id: int, rng) -> list[dict]:
    organs = []
    spike_len = SPIKE_LEN_MATURE_CM * length_frac
    branch_len = BRANCH_LEN_MATURE_CM * length_frac

    spike_skel = _make_spike_skeleton(apex_xyz, spike_len, rng)
    organs.append({
        "type": "stem",
        "skeleton": spike_skel,
        "widths": _tapered_widths(SPIKE_DIAM_BASE, SPIKE_DIAM_TIP, SPIKE_N_NODES),
        "organ_id": next_organ_id,
        "name": f"tassel_spike_{next_organ_id}",
        "_tassel_role": "spike",
    })
    next_organ_id += 1

    azimuths = np.linspace(0, 2 * np.pi, N_BRANCHES, endpoint=False)
    azimuths += rng.uniform(-np.pi / (2 * N_BRANCHES),
                            np.pi / (2 * N_BRANCHES), N_BRANCHES)

    # Young tassels: branches splay less (tightly packed around spike)
    angle_scale = YOUNG_ANGLE_FLOOR + (1.0 - YOUNG_ANGLE_FLOOR) * length_frac
    for k, az in enumerate(azimuths):
        ang_frac_k = k / max(N_BRANCHES - 1, 1)
        frac_up = (BRANCH_INSERTION_FRAC_LOW
                   + ang_frac_k * (BRANCH_INSERTION_FRAC_HIGH - BRANCH_INSERTION_FRAC_LOW))
        node_idx = int(frac_up * (SPIKE_N_NODES - 1))
        insertion = spike_skel[node_idx].copy()
        insert_deg = angle_scale * (BRANCH_INSERTION_ANGLE_BASAL
                      + ang_frac_k * (BRANCH_INSERTION_ANGLE_DISTAL - BRANCH_INSERTION_ANGLE_BASAL))
        branch_skel = _make_branch_skeleton(insertion, az, branch_len, insert_deg, rng)
        organs.append({
            "type": "stem",
            "skeleton": branch_skel,
            "widths": _tapered_widths(BRANCH_DIAM_BASE, BRANCH_DIAM_TIP, BRANCH_N_NODES),
            "organ_id": next_organ_id,
            "name": f"tassel_branch_{next_organ_id}",
            "_tassel_role": "branch",
        })
        next_organ_id += 1

    return organs


# ---------- spikelet billboard generator ----------

def _spikelet_billboards(skeleton: np.ndarray, start_frac: float,
                         rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Generate anther-like billboards along a skeleton. Returns (verts, tris).

    At each arc interval, ``SPIKELET_AZIMUTHS`` radial positions are sampled;
    at each, ``SPIKELETS_PER_AZIMUTH`` paired anthers hang in directions
    dominated by gravity (``-z``) with a small outward component. Each
    anther optionally becomes a cross-billboard (two perpendicular thin
    quads) to avoid the flat-cardboard look at grazing angles.
    """
    skel = skeleton
    seg_len = np.linalg.norm(np.diff(skel, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seg_len)])
    total_len = arc[-1]
    if total_len < 2 * SPIKELET_SPACING_CM:
        return np.empty((0, 3)), np.empty((0, 3), dtype=np.int64)

    arcs = np.arange(start_frac * total_len, total_len * 0.98, SPIKELET_SPACING_CM)
    verts = []
    tris = []
    gravity = np.array([0.0, 0.0, -1.0])

    for s in arcs:
        idx = int(np.searchsorted(arc, s)) - 1
        idx = max(0, min(idx, len(skel) - 2))
        denom = arc[idx + 1] - arc[idx] + 1e-12
        frac = (s - arc[idx]) / denom
        pos = skel[idx] + frac * (skel[idx + 1] - skel[idx])

        tan = skel[idx + 1] - skel[idx]
        tan = tan / (np.linalg.norm(tan) + 1e-12)

        # Radial frame around the axis
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

            # Base anther direction: outward component + downward (gravity)
            base_dir = SPIKELET_OUTWARD_FRAC * radial + SPIKELET_HANG_FRAC * gravity
            base_dir /= np.linalg.norm(base_dir) + 1e-12

            # Offset axis for paired anthers: perpendicular to both tan and base_dir
            pair_axis = np.cross(base_dir, radial)
            n = np.linalg.norm(pair_axis)
            pair_axis = pair_axis / n if n > 1e-9 else bi.copy()

            for p in range(SPIKELETS_PER_AZIMUTH):
                # Offset along pair_axis so paired anthers emerge side-by-side
                if SPIKELETS_PER_AZIMUTH > 1:
                    off_frac = (p - (SPIKELETS_PER_AZIMUTH - 1) / 2.0)
                    offset = off_frac * SPIKELET_PAIR_OFFSET_CM * pair_axis
                else:
                    offset = np.zeros(3)
                p_base = pos + offset

                # Per-anther directional jitter
                jitter = rng.normal(scale=np.radians(SPIKELET_JITTER_DEG), size=3)
                jitter[2] *= 0.3   # keep downward bias strong
                spike_dir = base_dir + jitter
                spike_dir /= np.linalg.norm(spike_dir) + 1e-12

                # Two orthogonal width axes for (optional) cross-billboard
                wa = np.cross(spike_dir, np.array([0.0, 0.0, 1.0]))
                if np.linalg.norm(wa) < 1e-6:
                    wa = np.cross(spike_dir, np.array([1.0, 0.0, 0.0]))
                wa /= np.linalg.norm(wa) + 1e-12
                wb = np.cross(spike_dir, wa); wb /= np.linalg.norm(wb) + 1e-12

                width_axes = [wa, wb] if SPIKELET_CROSS_BILLBOARD else [wa]

                for width_dir in width_axes:
                    base_l = p_base - 0.5 * W * width_dir
                    base_r = p_base + 0.5 * W * width_dir
                    tip_l  = p_base + L * spike_dir - 0.35 * W * width_dir
                    tip_r  = p_base + L * spike_dir + 0.35 * W * width_dir
                    v0 = len(verts)
                    verts.extend([base_l, base_r, tip_r, tip_l])
                    tris.append([v0, v0 + 1, v0 + 2])
                    tris.append([v0, v0 + 2, v0 + 3])

    return np.asarray(verts, dtype=np.float64), np.asarray(tris, dtype=np.int64)


def _merge_billboards_into_mesh(mesh, tassel_organs, rng: np.random.Generator) -> None:
    """Append anther billboards to ``mesh`` in place."""
    extra_verts = []
    extra_tris = []
    extra_normals = []
    extra_uvs = []
    extra_organ_ids = []
    vertex_offset = len(mesh.vertices)

    for organ in tassel_organs:
        role = organ.get("_tassel_role")
        if role not in ("spike", "branch"):
            continue
        start_frac = SPIKE_SPIKELET_START_FRAC if role == "spike" else BRANCH_SPIKELET_START_FRAC
        v, t = _spikelet_billboards(np.asarray(organ["skeleton"]), start_frac, rng)
        if len(v) == 0:
            continue
        extra_verts.append(v)
        t_shifted = t + vertex_offset
        extra_tris.append(t_shifted)
        # per-vertex: dummy normal (points out along branch tangent average)
        extra_normals.append(np.tile([0.0, 0.0, 1.0], (len(v), 1)))
        extra_uvs.append(np.zeros((len(v), 2)))
        extra_organ_ids.append(np.full(len(t), organ["organ_id"], dtype=np.int32))
        vertex_offset += len(v)

    if not extra_verts:
        return

    mesh.vertices = np.concatenate([mesh.vertices, *extra_verts])
    mesh.indices = np.concatenate([mesh.indices, *extra_tris]).astype(np.int32)
    mesh.normals = np.concatenate([mesh.normals, *extra_normals])
    mesh.uvs = np.concatenate([mesh.uvs, *extra_uvs])
    mesh.organ_ids = np.concatenate([mesh.organ_ids, *extra_organ_ids]).astype(np.int32)
    # extend segment_ids with -1 (billboards are not from CPlantBox skeleton segments)
    n_new_tris = sum(len(t) for t in extra_tris)
    mesh.segment_ids = np.concatenate([
        mesh.segment_ids, np.full(n_new_tris, -1, dtype=np.int32)
    ])


def _recenter(mesh, plant) -> None:
    stems = plant.getOrgans(pb.stem)
    if not stems:
        return
    seed_node = stems[0].getNodes()[0]
    mesh.vertices[:, 0] -= float(seed_node.x)
    mesh.vertices[:, 1] -= float(seed_node.y)


def main() -> None:
    print(f"Growing plant to day {GROW_DAY} (seed={SKEL_SEED}, donor={DONOR_SEED})")
    plant = grow_plant(
        str(DEFAULT_XML), simulation_time=GROW_DAY, seed=SKEL_SEED,
        cp_donor_seed=DONOR_SEED, cp_donor_mode="draw_coherent",
    )
    base_organs = extract_organs_for_lofter(plant, skip_roots=True)
    apex = _find_main_stem_apex(base_organs)
    next_id = max((o.get("organ_id", 0) for o in base_organs), default=-1) + 1
    print(f"  main-stem apex @ ({apex[0]:.1f}, {apex[1]:.1f}, {apex[2]:.1f}) cm")

    # Tassel-only (no plant) — sanity check of shape + spikelet density
    rng = np.random.default_rng(123)
    tassel_only = _build_tassel_organs(np.zeros(3), 1.0, 0, rng)
    mesh = loft_organs(tassel_only, stem_sides=8,
                       use_nurbs_backend=False)
    _merge_billboards_into_mesh(mesh, tassel_only, np.random.default_rng(456))
    out = OUT / "tassel_only_mature_v2.obj"
    mesh.to_obj(str(out))
    print(f"  wrote {out.name} ({mesh.n_vertices} v, {mesh.n_triangles} t)")

    # Plant + tassel
    for label, frac in SCENARIOS:
        rng = np.random.default_rng(hash(label) & 0xFFFFFFFF)
        tassel = _build_tassel_organs(apex, frac, next_id, rng)
        organs = base_organs + tassel
        mesh = loft_organs(organs, stem_sides=8, use_nurbs_backend=True)
        _merge_billboards_into_mesh(mesh, tassel, np.random.default_rng(hash(label + "bb") & 0xFFFFFFFF))
        _recenter(mesh, plant)
        out = OUT / f"plant55_tassel_{label}_v2.obj"
        mesh.to_obj(str(out))
        print(f"  [{label}] frac={frac:.2f}  "
              f"{mesh.n_vertices} v, {mesh.n_triangles} t (+{len(tassel)} tassel organs)")


if __name__ == "__main__":
    sys.exit(main() or 0)
