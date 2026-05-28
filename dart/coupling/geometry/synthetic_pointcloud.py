"""Synthetic point-cloud generator for sim-to-real leaf instance + rank segmentation.

Takes a grown CPlantBox plant, lofts it to a triangle mesh, samples points
uniformly by triangle area, and propagates per-triangle organ identity to
each point.  Output: a labelled point cloud with three per-point fields ready
to train a segmentation network on:

* ``organ_id``  — global organ index (instance label).
* ``organ_type`` — coarse category: 0=stem, 1=leaf-blade, 2=leaf-midrib, 3=tassel.
* ``rank``      — leaf rank from the plant base (1..N for leaves; -1 otherwise).

The companion CLI lives at
``dart/coupling/scripts/generate_synthetic_segmentation_data.py``.
"""
from __future__ import annotations
import numpy as np

ORGAN_TYPE_CODES = {"stem": 0, "leaf_blade": 1, "leaf_midrib": 2, "tassel": 3, "root": 4}


def sample_mesh_uniform(vertices, indices, n_points, rng=None):
    """Area-weighted uniform sample of triangles -> (n_points, 3) plus the
    triangle index each point came from."""
    rng = np.random.default_rng(rng)
    tris = vertices[indices]                                # (T, 3, 3)
    e1 = tris[:, 1] - tris[:, 0]
    e2 = tris[:, 2] - tris[:, 0]
    areas = 0.5 * np.linalg.norm(np.cross(e1, e2), axis=1)
    total = float(areas.sum())
    if total <= 0:
        raise ValueError("Mesh has zero total triangle area.")
    probs = areas / total
    tri_idx = rng.choice(len(indices), size=n_points, p=probs)
    u = rng.random(n_points)
    v = rng.random(n_points)
    mask = u + v > 1.0
    u[mask] = 1.0 - u[mask]
    v[mask] = 1.0 - v[mask]
    A = tris[tri_idx, 0]; B = tris[tri_idx, 1]; C = tris[tri_idx, 2]
    pts = A + u[:, None] * (B - A) + v[:, None] * (C - A)
    return pts, tri_idx, total


def assign_leaf_ranks(organs):
    """Sort leaves by base-z (= stem-attachment height) and number them 1..N.

    Acropetal emergence in maize means base-z order matches biological rank.
    Returns ``{organ_id: rank}`` for leaves; non-leaves are omitted.
    """
    leaves = [o for o in organs if o.get("type") == "leaf"]
    leaves_sorted = sorted(leaves, key=lambda o: float(o["skeleton"][0, 2]))
    return {int(o["organ_id"]): rank + 1 for rank, o in enumerate(leaves_sorted)}


def _organ_type_for(organ):
    """Coarse organ category for labelling (per-organ; the midrib split is
    applied later per-triangle via ``mesh.is_midrib``)."""
    if organ["type"] == "leaf":
        return ORGAN_TYPE_CODES["leaf_blade"]
    if organ["type"] == "stem":
        name = organ.get("name", "")
        if name.startswith(("tassel_spike_", "tassel_branch_")):
            return ORGAN_TYPE_CODES["tassel"]
        return ORGAN_TYPE_CODES["stem"]
    if organ["type"] == "root":
        return ORGAN_TYPE_CODES["root"]
    return ORGAN_TYPE_CODES["stem"]


def labels_per_point(mesh, organs, tri_idx):
    """Per-point (organ_id, organ_type, rank) from a sampled mesh.

    organ_type is taken from the per-organ category, with the
    ``mesh.is_midrib`` triangle mask flipping leaf points to ``leaf_midrib``.
    """
    organ_type_by_id = {int(o["organ_id"]): _organ_type_for(o) for o in organs}
    rank_by_id = assign_leaf_ranks(organs)

    organ_ids_tri = np.asarray(mesh.organ_ids, dtype=np.int32)
    organ_id_p = organ_ids_tri[tri_idx]

    type_lut = np.full(int(organ_ids_tri.max()) + 1, -1, dtype=np.int8)
    rank_lut = np.full(int(organ_ids_tri.max()) + 1, -1, dtype=np.int16)
    for oid, t in organ_type_by_id.items():
        type_lut[oid] = t
    for oid, r in rank_by_id.items():
        rank_lut[oid] = r

    organ_type_p = type_lut[organ_id_p]
    rank_p = rank_lut[organ_id_p]

    # leaf triangles flagged as midrib -> override type
    is_midrib_tri = np.asarray(mesh.is_midrib, dtype=bool)
    midrib_mask_p = is_midrib_tri[tri_idx] & (organ_type_p == ORGAN_TYPE_CODES["leaf_blade"])
    organ_type_p = organ_type_p.astype(np.int8)
    organ_type_p[midrib_mask_p] = ORGAN_TYPE_CODES["leaf_midrib"]

    return organ_id_p.astype(np.int32), organ_type_p, rank_p.astype(np.int16)


def generate_synthetic_pointcloud(xml_path, simulation_time, seed,
                                  n_points=120_000, noise_sigma_m=0.0005,
                                  loft_kwargs=None, extract_kwargs=None,
                                  rng=None):
    """End-to-end: grow -> loft -> sample -> label.

    Returns a dict with ``points`` (N,3 in metres), ``organ_id``, ``organ_type``,
    ``rank``, plus metadata.
    """
    from dart.coupling.growth.grow import grow_plant
    from dart.coupling.geometry.cplantbox_adapter import extract_organs_for_lofter
    from dart.coupling.geometry.g1_to_g3 import loft_organs

    rng = np.random.default_rng(rng)
    plant = grow_plant(xml_path, simulation_time=simulation_time, seed=seed)
    organs = extract_organs_for_lofter(plant, **(extract_kwargs or {}))
    mesh = loft_organs(organs, **(loft_kwargs or {}))

    pts_cm, tri_idx, total_area_cm2 = sample_mesh_uniform(
        mesh.vertices, mesh.indices, n_points, rng=rng)
    organ_id_p, organ_type_p, rank_p = labels_per_point(mesh, organs, tri_idx)

    pts_m = pts_cm.astype(np.float64) / 100.0   # CPlantBox cm -> metres
    if noise_sigma_m > 0:
        pts_m += rng.normal(0.0, noise_sigma_m, size=pts_m.shape)

    n_leaves = sum(1 for o in organs if o["type"] == "leaf")
    return {
        "points": pts_m,
        "organ_id": organ_id_p,
        "organ_type": organ_type_p,
        "rank": rank_p,
        "meta": {
            "xml": str(xml_path),
            "sim_time_d": int(simulation_time),
            "seed": int(seed),
            "n_leaves": int(n_leaves),
            "n_points": int(pts_m.shape[0]),
            "total_surface_area_m2": float(total_area_cm2) / 1e4,
            "noise_sigma_m": float(noise_sigma_m),
            "organ_type_codes": ORGAN_TYPE_CODES,
        },
    }


def save_npz(out_path, data):
    np.savez_compressed(
        out_path,
        points=data["points"].astype(np.float32),
        organ_id=data["organ_id"],
        organ_type=data["organ_type"],
        rank=data["rank"],
        meta=np.array([repr(data["meta"])]),
    )


def save_las(out_path, data):
    """Write the labelled cloud to LAS, packing labels into available extra dims.

    Uses three ExtraBytes scalar fields ``organ_id`` (uint16),
    ``organ_type`` (uint8), ``rank`` (int16) so the file opens in CloudCompare
    side-by-side with the real FP4D / MuST-C clouds.
    """
    import laspy
    pts = data["points"]
    header = laspy.LasHeader(version="1.4", point_format=6)
    header.scales = np.array([1e-4, 1e-4, 1e-4])
    header.offsets = pts.min(axis=0)
    header.add_extra_dim(laspy.ExtraBytesParams(name="organ_id", type=np.uint16))
    header.add_extra_dim(laspy.ExtraBytesParams(name="organ_type", type=np.uint8))
    header.add_extra_dim(laspy.ExtraBytesParams(name="rank", type=np.int16))
    las = laspy.LasData(header)
    las.x = pts[:, 0]; las.y = pts[:, 1]; las.z = pts[:, 2]
    las.organ_id = data["organ_id"].astype(np.uint16)
    las.organ_type = data["organ_type"].astype(np.uint8)
    las.rank = np.maximum(data["rank"], -1).astype(np.int16)
    las.write(str(out_path))
