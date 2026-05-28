"""Field-realistic synthetic scan simulation for sim-to-real segmentation.

Wraps the per-plant generator (synthetic_pointcloud.generate_synthetic_pointcloud)
with three pieces of domain realism the bare generator skips:

* **Row layout** — translate N plants along a row, offset their organ_ids so
  instance labels stay globally unique, and tag each point with ``plant_id``.
* **Terrestrial-scanner visibility** — Katz et al. (2007) hidden-point removal
  from a sequence of scanner viewpoints along the row. Only points actually
  reachable by at least one viewpoint survive (mimics the occlusion pattern
  of a UGV-mounted line scanner like FieldPheno4D's robot).
* **Anisotropic range noise** — Gaussian noise dominantly along the
  scanner-to-point ray direction, matching laser-triangulation error.

Background soil clutter (with ``organ_type=255``) is layered in optionally.
"""
from __future__ import annotations
import numpy as np


# -----------------------------------------------------------------------------
# Row layout
# -----------------------------------------------------------------------------
def compose_row(plants, spacing_m, row_axis=1):
    """Concatenate plants into a row along ``row_axis``.

    Each input dict is the output of generate_synthetic_pointcloud. ``organ_id``
    is offset so instance labels stay unique across the row, and a new
    ``plant_id`` array (0..N-1) is added.
    """
    P, oid, otp, rk, pid = [], [], [], [], []
    organ_offset = 0
    for k, p in enumerate(plants):
        pts = p["points"].copy()
        pts[:, row_axis] += k * spacing_m
        P.append(pts)
        oid.append(p["organ_id"].astype(np.int32) + organ_offset)
        otp.append(p["organ_type"])
        rk.append(p["rank"])
        pid.append(np.full(pts.shape[0], k, dtype=np.int16))
        organ_offset += int(p["organ_id"].max()) + 1
    return {
        "points": np.vstack(P),
        "organ_id": np.concatenate(oid),
        "organ_type": np.concatenate(otp),
        "rank": np.concatenate(rk),
        "plant_id": np.concatenate(pid),
    }


# -----------------------------------------------------------------------------
# Angular z-buffer visibility (terrestrial-scanner model)
# -----------------------------------------------------------------------------
def zbuffer_visible_mask(points, viewpoint, ang_res_mrad=1.0):
    """Visibility mask from ``viewpoint`` via spherical-angular z-buffer.

    Each point is projected to (azimuth, elevation, range) relative to
    ``viewpoint``; bins of width ``ang_res_mrad`` keep the nearest range per
    bin. Approximates a true laser scanner: every ray sees only its
    closest-hit surface, dense scans stay dense.

    At ``ang_res_mrad=1.0`` and 1 m range, bin width ~1 mm — matches the
    sub-mm precision of FieldPheno4D's laser triangulation.
    """
    p = points - viewpoint
    r = np.linalg.norm(p, axis=1)
    finite = r > 1e-9
    az = np.arctan2(p[finite, 0], p[finite, 1])
    el = np.arctan2(p[finite, 2], np.linalg.norm(p[finite, :2], axis=1))
    b = ang_res_mrad / 1000.0
    az_i = np.floor(az / b).astype(np.int64)
    el_i = np.floor(el / b).astype(np.int64)
    az_i -= az_i.min(); el_i -= el_i.min()
    n_el = int(el_i.max()) + 1
    key = az_i * n_el + el_i
    r_v = r[finite]
    order = np.lexsort((r_v, key))                  # primary=key, secondary=r
    sorted_keys = key[order]
    keep_first = np.concatenate([[True], sorted_keys[1:] != sorted_keys[:-1]])
    vis_local = order[keep_first]
    mask = np.zeros(len(points), bool)
    finite_idx = np.where(finite)[0]
    mask[finite_idx[vis_local]] = True
    return mask


def simulate_terrestrial_scanner(points, row_extent_y,
                                 scanner_height_m=0.6,
                                 scanner_x_offsets=(-0.5, 0.5),
                                 n_poses_per_side=12,
                                 plant_row_x=0.0,
                                 ang_res_mrad=1.0):
    """Visibility from a UGV-mounted line scanner sweeping along the row.

    Two scanners (left + right of the row by default), each visiting
    ``n_poses_per_side`` poses evenly spaced along the row. A point is
    visible if the angular z-buffer keeps it at any pose.
    """
    y_lo, y_hi = row_extent_y
    ys = np.linspace(y_lo, y_hi, n_poses_per_side)
    visible = np.zeros(len(points), bool)
    poses = []
    for dx in scanner_x_offsets:
        for y in ys:
            vp = np.array([plant_row_x + dx, y, scanner_height_m])
            poses.append(vp)
            visible |= zbuffer_visible_mask(points, vp, ang_res_mrad=ang_res_mrad)
    return visible, np.array(poses)


# -----------------------------------------------------------------------------
# Anisotropic range noise + soil clutter
# -----------------------------------------------------------------------------
def add_anisotropic_noise(points, scanner_positions,
                          range_sigma_mm=0.5, angular_sigma_mrad=0.3, rng=None):
    """Per-point Gaussian noise along the nearest-scanner ray direction.

    Range noise (along the ray) + angular noise (perpendicular). Each point
    is associated with its nearest scanner pose.
    """
    rng = np.random.default_rng(rng)
    sp = np.asarray(scanner_positions)
    # nearest pose per point
    d = np.linalg.norm(points[:, None, :] - sp[None, :, :], axis=2)
    nearest = d.argmin(axis=1)
    rays = points - sp[nearest]
    ranges = np.linalg.norm(rays, axis=1) + 1e-9
    ray_hat = rays / ranges[:, None]
    range_noise = rng.normal(0.0, range_sigma_mm / 1000.0, len(points))
    tangent_sigma = ranges * (angular_sigma_mrad / 1000.0)
    tangent_noise = rng.normal(0.0, 1.0, (len(points), 3)) * tangent_sigma[:, None]
    # remove the radial component from the tangent noise
    radial = (tangent_noise * ray_hat).sum(axis=1)
    tangent_noise -= radial[:, None] * ray_hat
    return points + range_noise[:, None] * ray_hat + tangent_noise


def add_ground_clutter(extent_xy, n_points, z0=0.0, z_noise_mm=5.0, rng=None):
    """Sparse soil points: ``organ_type=background (5)``, ``organ_id=-1``, ``rank=-1``."""
    from dart.coupling.geometry.synthetic_pointcloud import ORGAN_TYPE_CODES
    rng = np.random.default_rng(rng)
    (x0, x1), (y0, y1) = extent_xy
    pts = np.column_stack([
        rng.uniform(x0, x1, n_points),
        rng.uniform(y0, y1, n_points),
        z0 + rng.normal(0.0, z_noise_mm / 1000.0, n_points),
    ])
    return {
        "points": pts,
        "organ_id": np.full(n_points, -1, np.int32),
        "organ_type": np.full(n_points, ORGAN_TYPE_CODES["background"], np.int8),
        "rank": np.full(n_points, -1, np.int16),
        "plant_id": np.full(n_points, -1, np.int16),
    }


# -----------------------------------------------------------------------------
# End-to-end
# -----------------------------------------------------------------------------
def simulate_field_scan(xml_path, seeds, sim_time, spacing_m=0.20,
                        points_per_plant=200_000, plant_row_x=0.0,
                        scanner_height_m=0.6, scanner_x_offsets=(-0.5, 0.5),
                        n_poses_per_side=12, ang_res_mrad=1.0,
                        range_sigma_mm=0.5, angular_sigma_mrad=0.3,
                        ground_density_per_m2=200, rng=None,
                        gen_kwargs=None):
    """Compose row -> apply scanner occlusion -> add noise + soil -> return."""
    from dart.coupling.geometry.synthetic_pointcloud import generate_synthetic_pointcloud
    rng = np.random.default_rng(rng)
    gk = dict(gen_kwargs or {})
    plants = [generate_synthetic_pointcloud(
        xml_path=xml_path, simulation_time=sim_time, seed=int(s),
        n_points=points_per_plant, noise_sigma_m=0.0, **gk) for s in seeds]
    row = compose_row(plants, spacing_m=spacing_m)

    y_pts = row["points"][:, 1]
    visible, poses = simulate_terrestrial_scanner(
        row["points"], row_extent_y=(y_pts.min() - 0.1, y_pts.max() + 0.1),
        scanner_height_m=scanner_height_m,
        scanner_x_offsets=scanner_x_offsets,
        n_poses_per_side=n_poses_per_side,
        plant_row_x=plant_row_x,
        ang_res_mrad=ang_res_mrad,
    )
    for k in ("points", "organ_id", "organ_type", "rank", "plant_id"):
        row[k] = row[k][visible]

    noisy_pts = add_anisotropic_noise(
        row["points"], poses,
        range_sigma_mm=range_sigma_mm,
        angular_sigma_mrad=angular_sigma_mrad, rng=rng)
    row["points"] = noisy_pts

    if ground_density_per_m2 > 0:
        x_pts = row["points"][:, 0]
        ext_x = (x_pts.min() - 0.2, x_pts.max() + 0.2)
        ext_y = (y_pts.min() - 0.2, y_pts.max() + 0.2)
        area = (ext_x[1] - ext_x[0]) * (ext_y[1] - ext_y[0])
        n_ground = int(ground_density_per_m2 * area)
        ground = add_ground_clutter((ext_x, ext_y), n_ground, rng=rng)
        for k in ("points", "organ_id", "organ_type", "rank", "plant_id"):
            row[k] = np.concatenate([row[k], ground[k]])

    row["scanner_poses"] = poses
    row["meta"] = {
        "xml": str(xml_path), "sim_time_d": int(sim_time), "seeds": list(map(int, seeds)),
        "spacing_m": spacing_m, "n_plants": len(seeds),
        "n_points_after_occlusion": int(row["points"].shape[0]),
        "scanner_poses": int(len(poses)),
        "range_sigma_mm": range_sigma_mm, "angular_sigma_mrad": angular_sigma_mrad,
        "ground_density_per_m2": int(ground_density_per_m2),
    }
    return row
