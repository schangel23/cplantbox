#!/usr/bin/env python3
"""
Visualise the 3×3 multifield in G1 (skeleton) and G3 (mesh) as SVG.

Reads existing multifield_p{0..8}.obj + mapping JSONs from the output directory,
re-grows the 9 plants (seeds 42–50) for the G1 skeletons, then produces a
two-panel SVG with an isometric orthographic projection:

    [ G1 — skeleton polylines ]  |  [ G3 — flat-shaded mesh triangles ]

Both panels share the same camera (azimuth / elevation configurable).

Usage:
    cd /home/lukas/PHD
    source CPlantBox/cpbenv/bin/activate
    python CPlantBox/dart/coupling/visualize_multifield_svg.py
    python CPlantBox/dart/coupling/visualize_multifield_svg.py --no-g1 --subsample 3
    python CPlantBox/dart/coupling/visualize_multifield_svg.py --az 35 --el 20

Options:
    --output PATH       Output SVG path (default: output/multifield_field_view.svg)
    --no-g1             Skip G1 panel (faster; still renders G3)
    --subsample N       Keep only every N-th triangle (default 1 = all)
    --az DEGREES        Camera azimuth (default 40)
    --el DEGREES        Camera elevation (default 25)
"""

import sys
import re
import json
import argparse
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants — must match multifield.py
# ---------------------------------------------------------------------------
FIELD_SEED = 42
N_PLANTS = 9
SIMULATION_DAYS = 55

# Color palette (matching grow.py)
LEAF_GREENS = [
    (0.18, 0.55, 0.18),
    (0.30, 0.69, 0.31),
    (0.46, 0.80, 0.46),
    (0.56, 0.88, 0.56),
    (0.60, 0.80, 0.20),
    (0.20, 0.65, 0.32),
    (0.40, 0.75, 0.40),
    (0.50, 0.85, 0.50),
    (0.13, 0.55, 0.13),
    (0.24, 0.70, 0.44),
    (0.42, 0.76, 0.22),
    (0.33, 0.65, 0.50),
    (0.52, 0.82, 0.32),
    (0.22, 0.58, 0.28),
    (0.38, 0.72, 0.38),
    (0.48, 0.78, 0.48),
]
STEM_COLOR = (0.55, 0.27, 0.07)  # brown

# Directional light vector (world space) for G3 shading
LIGHT_DIR = np.array([0.4, -0.6, 0.7], dtype=np.float64)
LIGHT_DIR /= np.linalg.norm(LIGHT_DIR)

# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _rgb_hex(r, g, b):
    return f'#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}'


def _clamp01(v):
    return max(0.0, min(1.0, v))


# ---------------------------------------------------------------------------
# OBJ + mapping JSON loader
# ---------------------------------------------------------------------------

def load_plant_geometry(obj_path, mapping_path):
    """Parse one plant's OBJ + mapping JSON.

    Returns:
        verts      : (M, 3) float64  — vertex positions in local plant coords (cm)
        faces      : (N, 3) int32    — 0-based vertex indices
        face_colors: (N, 3) float64  — base RGB (pre-shading) from organ type/index
    """
    # --- Mapping JSON: organ_id → type, leaf_index ---
    with open(mapping_path) as f:
        mapping = json.load(f)

    organ_id_to_type = {}   # int → 'stem' | 'leaf'
    organ_id_to_leaf_idx = {}  # int → int (leaf counter, -1 for stem)
    leaf_counter = 0
    for org in mapping['organs']:
        oid = org['organ_id']
        otype = org.get('type', 'leaf')
        organ_id_to_type[oid] = otype
        if otype == 'leaf':
            organ_id_to_leaf_idx[oid] = leaf_counter
            leaf_counter += 1
        else:
            organ_id_to_leaf_idx[oid] = -1

    # --- OBJ: vertices, faces, group membership ---
    verts = []
    faces = []
    group_faces = {}   # group_name → [face_idx, ...]
    current_group = 'default'

    with open(obj_path) as f:
        for line in f:
            if line.startswith('v '):
                verts.append(list(map(float, line.split()[1:4])))
            elif line.startswith('g '):
                parts = line.split()
                current_group = parts[1] if len(parts) > 1 else 'default'
                group_faces.setdefault(current_group, [])
            elif line.startswith('f '):
                idx = [int(p.split('/')[0]) - 1 for p in line.split()[1:]]
                if len(idx) == 3:
                    group_faces.setdefault(current_group, []).append(len(faces))
                    faces.append(idx)
                elif len(idx) == 4:
                    group_faces.setdefault(current_group, []).append(len(faces))
                    faces.append([idx[0], idx[1], idx[2]])
                    group_faces.setdefault(current_group, []).append(len(faces))
                    faces.append([idx[0], idx[2], idx[3]])

    verts = np.array(verts, dtype=np.float64)
    faces = np.array(faces, dtype=np.int32) if faces else np.zeros((0, 3), dtype=np.int32)
    n_faces = len(faces)

    # --- Per-face base color from group → organ_id → type/leaf_idx ---
    face_colors = np.full((n_faces, 3), 0.5, dtype=np.float64)  # default grey

    for gname, fidxs in group_faces.items():
        if not fidxs:
            continue
        # Group name: p{i}_organ_{N}  → organ_id = N
        m = re.search(r'organ_(\d+)', gname)
        if m:
            oid = int(m.group(1))
            otype = organ_id_to_type.get(oid, 'leaf')
            if otype == 'stem':
                bc = STEM_COLOR
            else:
                li = organ_id_to_leaf_idx.get(oid, 0)
                bc = LEAF_GREENS[li % len(LEAF_GREENS)]
        else:
            bc = STEM_COLOR

        fidxs_arr = np.array([fi for fi in fidxs if fi < n_faces], dtype=np.int32)
        if len(fidxs_arr):
            face_colors[fidxs_arr] = bc

    return verts, faces, face_colors


# ---------------------------------------------------------------------------
# Isometric orthographic projector
# ---------------------------------------------------------------------------

class IsoProjector:
    """Orthographic projection using azimuth + elevation, matching _set_camera()
    convention in grow.py:
        cam_right = (cos(az),  sin(az),  0)
        cam_up    = (-sin(az)*sin(el),  cos(az)*sin(el),  cos(el))
        cam_fwd   = (-sin(az)*cos(el),  cos(az)*cos(el), -sin(el))   [into scene]
    """

    def __init__(self, az_deg, el_deg, bounds, panel_w, panel_h, margin):
        az = np.radians(az_deg)
        el = np.radians(el_deg)

        self.cam_right = np.array([np.cos(az), np.sin(az), 0.0])
        self.cam_up = np.array([-np.sin(az) * np.sin(el),
                                  np.cos(az) * np.sin(el),
                                  np.cos(el)])
        # cam_fwd points INTO the scene; larger depth = farther from camera
        self.cam_fwd = np.array([-np.sin(az) * np.cos(el),
                                   np.cos(az) * np.cos(el),
                                  -np.sin(el)])

        # Project all 8 bounding-box corners to find projected extents
        x0, x1, y0, y1, z0, z1 = (bounds[0], bounds[1], bounds[2],
                                    bounds[3], bounds[4], bounds[5])
        corners = np.array([
            [x0, y0, z0], [x1, y0, z0], [x0, y1, z0], [x1, y1, z0],
            [x0, y0, z1], [x1, y0, z1], [x0, y1, z1], [x1, y1, z1],
        ], dtype=np.float64)

        px = corners @ self.cam_right
        py = corners @ self.cam_up

        self.px_min, self.px_max = px.min(), px.max()
        self.py_min, self.py_max = py.min(), py.max()

        data_w = max(self.px_max - self.px_min, 1.0)
        data_h = max(self.py_max - self.py_min, 1.0)

        usable_w = panel_w - 2 * margin
        usable_h = panel_h - 2 * margin
        self.scale = min(usable_w / data_w, usable_h / data_h)

        # Centering offsets (local panel coords, before adding panel offset)
        self.off_x = margin + (usable_w - data_w * self.scale) / 2.0
        self.off_y = margin + (usable_h - data_h * self.scale) / 2.0

    def project(self, pts, panel_ox=0.0, panel_oy=0.0):
        """Project (N, 3) world → (N, 2) SVG coords in the given panel."""
        px = pts @ self.cam_right   # (N,)
        py = pts @ self.cam_up      # (N,)
        sx = panel_ox + self.off_x + (px - self.px_min) * self.scale
        sy = panel_oy + self.off_y + (self.py_max - py) * self.scale  # flip Y
        return np.stack([sx, sy], axis=1)

    def depth(self, pts):
        """Depth along camera-forward axis (N,). Larger = farther from camera."""
        return pts @ self.cam_fwd


# ---------------------------------------------------------------------------
# SVG ground grid helper
# ---------------------------------------------------------------------------

def _ground_grid_lines(proj, bounds, panel_ox, panel_oy):
    """Return SVG lines for a ground-plane perimeter at z=0."""
    x0, x1 = bounds[0], bounds[1]
    y0, y1 = bounds[2], bounds[3]
    edges = [
        ([x0, y0, 0.0], [x1, y0, 0.0]),
        ([x1, y0, 0.0], [x1, y1, 0.0]),
        ([x1, y1, 0.0], [x0, y1, 0.0]),
        ([x0, y1, 0.0], [x0, y0, 0.0]),
    ]
    lines = []
    for a, b in edges:
        pa = proj.project(np.array([a]), panel_ox, panel_oy)[0]
        pb = proj.project(np.array([b]), panel_ox, panel_oy)[0]
        lines.append(
            f'<line x1="{pa[0]:.1f}" y1="{pa[1]:.1f}" '
            f'x2="{pb[0]:.1f}" y2="{pb[1]:.1f}" '
            f'stroke="#bbb" stroke-width="0.5" stroke-dasharray="5,3"/>'
        )
    return lines


# ---------------------------------------------------------------------------
# Main SVG render
# ---------------------------------------------------------------------------

def render_field_svg(grid_info, all_organ_dicts, obj_paths, mapping_paths,
                     output_path, az_deg=40.0, el_deg=25.0, subsample=1):
    """Write two-panel field-view SVG.

    Parameters
    ----------
    all_organ_dicts : list[list[dict]] or None
        G1 organ skeletons (re-grown plants). If None, G1 panel is skipped.
    obj_paths       : list[Path]   — 9 OBJ files (local plant coords, cm)
    mapping_paths   : list[Path]   — 9 mapping JSON files
    subsample       : int          — keep every N-th triangle (1 = all)
    """
    positions_m = [np.array(p, dtype=np.float64) for p in grid_info['positions_m']]
    positions_cm = [p * 100.0 for p in positions_m]   # m → cm

    # ------------------------------------------------------------------
    # 1. Gather all world-space points to compute global bounds
    # ------------------------------------------------------------------
    print("  Computing global bounds from OBJ files...")
    world_verts_all = []

    parsed_plants = []   # cache: (verts_local, faces, face_colors)
    for i, (obj_path, map_path) in enumerate(zip(obj_paths, mapping_paths)):
        v, f, fc = load_plant_geometry(obj_path, map_path)
        v_world = v.copy()
        v_world[:, 0] += positions_cm[i][0]
        v_world[:, 1] += positions_cm[i][1]
        world_verts_all.append(v_world)
        parsed_plants.append((v_world, f, fc))

    if all_organ_dicts is not None:
        for i, organs in enumerate(all_organ_dicts):
            for org in organs:
                s = org['skeleton'].copy()
                s[:, 0] += positions_cm[i][0]
                s[:, 1] += positions_cm[i][1]
                world_verts_all.append(s)

    combined = np.concatenate(world_verts_all, axis=0)
    pad = max(combined.max(axis=0) - combined.min(axis=0)) * 0.04
    bounds = [
        combined[:, 0].min() - pad, combined[:, 0].max() + pad,
        combined[:, 1].min() - pad, combined[:, 1].max() + pad,
        combined[:, 2].min() - pad, combined[:, 2].max() + pad,
    ]
    print(f"    X: {bounds[0]:.0f}…{bounds[1]:.0f}  "
          f"Y: {bounds[2]:.0f}…{bounds[3]:.0f}  "
          f"Z: {bounds[4]:.0f}…{bounds[5]:.0f}  cm")

    # ------------------------------------------------------------------
    # 2. Layout
    # ------------------------------------------------------------------
    header_h = 36.0
    margin = 22.0
    gap = 28.0
    panel_w = 720.0
    panel_h = 760.0
    total_w = panel_w * 2 + gap
    total_h = panel_h + header_h

    do_g1 = all_organ_dicts is not None

    if not do_g1:
        # Single G3 panel, centred
        total_w = panel_w
        g3_panel_ox = 0.0
    else:
        g3_panel_ox = panel_w + gap

    proj = IsoProjector(az_deg, el_deg, bounds, panel_w, panel_h, margin)

    # ------------------------------------------------------------------
    # 3. SVG header
    # ------------------------------------------------------------------
    buf = []
    buf.append('<?xml version="1.0" encoding="UTF-8"?>')
    buf.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {total_w:.1f} {total_h:.1f}" '
        f'width="{int(total_w)}" height="{int(total_h)}">'
    )
    buf.append(
        f'<rect x="0" y="0" width="{total_w:.1f}" height="{total_h:.1f}" fill="white"/>'
    )

    # Panel backgrounds + labels
    panels = []
    if do_g1:
        panels.append((0.0,           'G1  Skeleton  Field'))
    panels.append((g3_panel_ox, 'G3  Mesh  Field'))

    for px_off, label in panels:
        buf.append(
            f'<rect x="{px_off:.1f}" y="{header_h:.1f}" '
            f'width="{panel_w:.1f}" height="{panel_h:.1f}" '
            f'fill="#f8f8f8" stroke="#cccccc" stroke-width="0.5"/>'
        )
        cx = px_off + panel_w / 2
        buf.append(
            f'<text x="{cx:.1f}" y="24" text-anchor="middle" '
            f'font-family="Times New Roman,Times,serif" font-size="15" '
            f'font-weight="bold" fill="#222">{label}</text>'
        )

    buf.append(
        f'<text x="8" y="24" font-family="Times New Roman,Times,serif" '
        f'font-size="11" fill="#888">Day {SIMULATION_DAYS}  '
        f'az={az_deg:.0f}°  el={el_deg:.0f}°</text>'
    )

    py_off = header_h   # panel top-left Y in SVG coords

    # ------------------------------------------------------------------
    # 4. G1 Panel
    # ------------------------------------------------------------------
    if do_g1:
        print("  Rendering G1 panel...")
        buf.append('<g id="g1-panel">')
        buf.extend(_ground_grid_lines(proj, bounds, panel_ox=0.0, panel_oy=py_off))

        # Sort plants back-to-front (painter's: large depth → draw first)
        plant_centers = np.array([
            [positions_cm[i][0], positions_cm[i][1],
             (bounds[4] + bounds[5]) / 2.0]
            for i in range(N_PLANTS)
        ])
        plant_depths = proj.depth(plant_centers)
        draw_order = np.argsort(-plant_depths)   # large depth (far) first

        total_polylines = 0
        for pi in draw_order:
            organs = all_organ_dicts[pi]
            pos = positions_cm[pi]
            leaf_idx = 0

            for organ in organs:
                skel = organ['skeleton'].copy()
                skel[:, 0] += pos[0]
                skel[:, 1] += pos[1]

                if len(skel) < 2:
                    continue

                if organ['type'] == 'stem':
                    color = STEM_COLOR
                    sw = '2.8'
                else:
                    color = LEAF_GREENS[leaf_idx % len(LEAF_GREENS)]
                    leaf_idx += 1
                    sw = '1.6'

                pts_svg = proj.project(skel, panel_ox=0.0, panel_oy=py_off)
                pts_str = ' '.join(f'{p[0]:.1f},{p[1]:.1f}' for p in pts_svg)
                buf.append(
                    f'<polyline points="{pts_str}" stroke="{_rgb_hex(*color)}" '
                    f'stroke-width="{sw}" fill="none" '
                    f'stroke-linecap="round" stroke-linejoin="round"/>'
                )
                total_polylines += 1

        buf.append('</g>')
        print(f"    {total_polylines} polylines written")

    # ------------------------------------------------------------------
    # 5. G3 Panel — collect all triangles globally, sort, render
    # ------------------------------------------------------------------
    print(f"  Rendering G3 panel (subsample={subsample})...")
    buf.append('<g id="g3-panel">')
    buf.extend(_ground_grid_lines(proj, bounds, panel_ox=g3_panel_ox, panel_oy=py_off))

    # Accumulate per-plant arrays then sort globally
    all_depths_list = []
    all_colors_list = []
    all_pts2d_list = []

    total_tris_input = 0
    for i, (verts_w, faces, face_colors_base) in enumerate(parsed_plants):
        if len(faces) == 0:
            print(f"    Plant {i}: no faces, skipping")
            continue

        # Optional subsampling (every N-th face)
        if subsample > 1:
            keep = np.arange(0, len(faces), subsample)
            faces = faces[keep]
            face_colors_base = face_colors_base[keep]

        total_tris_input += len(faces)

        # Per-triangle flat-shading: normal · light
        v0 = verts_w[faces[:, 0]]
        v1 = verts_w[faces[:, 1]]
        v2 = verts_w[faces[:, 2]]

        e1 = v1 - v0
        e2 = v2 - v0
        normals = np.cross(e1, e2)
        nlen = np.linalg.norm(normals, axis=1, keepdims=True)
        nlen = np.where(nlen < 1e-12, 1.0, nlen)
        normals /= nlen

        # Double-sided shading (leaves are thin → both faces visible)
        shade = 0.30 + 0.70 * np.abs(normals @ LIGHT_DIR)   # (N,)

        # Apply shading and convert to uint8
        shaded = np.clip(face_colors_base * shade[:, np.newaxis], 0.0, 1.0)
        colors_u8 = (shaded * 255.0).astype(np.uint8)    # (N, 3)

        # Project all vertices once
        pts2d = proj.project(verts_w, panel_ox=g3_panel_ox, panel_oy=py_off)  # (M, 2)

        # Triangle 2-D corners, packed as (N, 6): x0,y0,x1,y1,x2,y2
        p0 = pts2d[faces[:, 0]]
        p1 = pts2d[faces[:, 1]]
        p2 = pts2d[faces[:, 2]]
        pts_flat = np.concatenate([p0, p1, p2], axis=1).astype(np.float32)  # (N, 6)

        # Centroid depth (back-to-front order, larger depth = farther)
        depths = proj.depth(verts_w)
        tri_depths = (depths[faces[:, 0]] + depths[faces[:, 1]] + depths[faces[:, 2]]) / 3.0

        all_depths_list.append(tri_depths)
        all_colors_list.append(colors_u8)
        all_pts2d_list.append(pts_flat)

        print(f"    Plant {i}: {len(faces)} triangles")

    # Global depth sort (back = large depth drawn first)
    all_depths = np.concatenate(all_depths_list)
    all_colors = np.concatenate(all_colors_list)
    all_pts2d = np.concatenate(all_pts2d_list)

    print(f"    Sorting {len(all_depths)} triangles globally...")
    order = np.argsort(-all_depths)   # descending: far (large depth) first
    all_colors = all_colors[order]
    all_pts2d = all_pts2d[order]

    # Write polygon elements
    print(f"    Writing {len(order)} triangles to SVG buffer...")
    for ti in range(len(order)):
        x0, y0, x1, y1, x2, y2 = all_pts2d[ti]
        r, g, b = all_colors[ti]
        fill = f'#{r:02x}{g:02x}{b:02x}'
        buf.append(
            f'<polygon points="{x0:.1f},{y0:.1f} {x1:.1f},{y1:.1f} {x2:.1f},{y2:.1f}" '
            f'fill="{fill}"/>'
        )

    buf.append('</g>')

    # ------------------------------------------------------------------
    # 6. Scale bar (50 cm) in G3 panel, bottom-left corner
    # ------------------------------------------------------------------
    bar_world_a = np.array([[bounds[0] + 5, bounds[2] + 5, bounds[4]]])
    bar_world_b = np.array([[bounds[0] + 55, bounds[2] + 5, bounds[4]]])
    ba = proj.project(bar_world_a, panel_ox=g3_panel_ox, panel_oy=py_off)[0]
    bb = proj.project(bar_world_b, panel_ox=g3_panel_ox, panel_oy=py_off)[0]
    bar_len_svg = np.sqrt((bb[0] - ba[0]) ** 2 + (bb[1] - ba[1]) ** 2)

    # Place horizontally at bottom of panel
    bar_y_svg = py_off + panel_h - 14
    bar_x0_svg = g3_panel_ox + margin
    bar_x1_svg = bar_x0_svg + bar_len_svg
    bar_mid = (bar_x0_svg + bar_x1_svg) / 2

    buf.append(
        f'<line x1="{bar_x0_svg:.1f}" y1="{bar_y_svg:.1f}" '
        f'x2="{bar_x1_svg:.1f}" y2="{bar_y_svg:.1f}" '
        f'stroke="#333" stroke-width="2"/>'
    )
    # End ticks
    for bx in [bar_x0_svg, bar_x1_svg]:
        buf.append(
            f'<line x1="{bx:.1f}" y1="{bar_y_svg - 4:.1f}" '
            f'x2="{bx:.1f}" y2="{bar_y_svg + 4:.1f}" '
            f'stroke="#333" stroke-width="2"/>'
        )
    buf.append(
        f'<text x="{bar_mid:.1f}" y="{bar_y_svg - 6:.1f}" text-anchor="middle" '
        f'font-family="Times New Roman,Times,serif" font-size="11" fill="#333">50 cm</text>'
    )

    buf.append('</svg>')

    # ------------------------------------------------------------------
    # 7. Write file
    # ------------------------------------------------------------------
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"  Joining {len(buf)} SVG elements...")
    svg_text = '\n'.join(buf)

    print(f"  Writing to {output_path} ...")
    output_path.write_text(svg_text)

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  Done: {output_path}  ({size_mb:.1f} MB, "
          f"{total_tris_input} triangles rendered)")
    return output_path


# ---------------------------------------------------------------------------
# Standalone single-panel SVGs (transparent background)
# ---------------------------------------------------------------------------

def render_g1_standalone_svg(grid_info, all_organ_dicts, output_path,
                              az_deg=40.0, el_deg=25.0):
    """Write a single G1-skeleton SVG with transparent background.

    No white rect, no panel box, no header — suitable for overlaying in
    Inkscape / presentations.
    """
    positions_cm = [np.array(p) * 100.0 for p in grid_info['positions_m']]

    # Collect all skeleton points for bounds
    all_pts = []
    for i, organs in enumerate(all_organ_dicts):
        for org in organs:
            s = org['skeleton'].copy()
            s[:, 0] += positions_cm[i][0]
            s[:, 1] += positions_cm[i][1]
            all_pts.append(s)

    combined = np.concatenate(all_pts, axis=0)
    pad = max(combined.max(axis=0) - combined.min(axis=0)) * 0.05
    bounds = [combined[:, 0].min() - pad, combined[:, 0].max() + pad,
              combined[:, 1].min() - pad, combined[:, 1].max() + pad,
              combined[:, 2].min() - pad, combined[:, 2].max() + pad]

    margin = 18.0
    panel_w = 720.0
    panel_h = 760.0
    proj = IsoProjector(az_deg, el_deg, bounds, panel_w, panel_h, margin)

    buf = []
    buf.append('<?xml version="1.0" encoding="UTF-8"?>')
    buf.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {panel_w:.1f} {panel_h:.1f}" '
        f'width="{int(panel_w)}" height="{int(panel_h)}">'
    )
    # No background rect → transparent

    buf.extend(_ground_grid_lines(proj, bounds, panel_ox=0.0, panel_oy=0.0))

    # Plants back-to-front
    plant_centers = np.array([
        [positions_cm[i][0], positions_cm[i][1],
         (bounds[4] + bounds[5]) / 2.0]
        for i in range(N_PLANTS)
    ])
    order = np.argsort(-proj.depth(plant_centers))

    for pi in order:
        organs = all_organ_dicts[pi]
        pos = positions_cm[pi]
        leaf_idx = 0
        for organ in organs:
            skel = organ['skeleton'].copy()
            skel[:, 0] += pos[0]
            skel[:, 1] += pos[1]
            if len(skel) < 2:
                continue
            if organ['type'] == 'stem':
                color = STEM_COLOR
                sw = '2.8'
            else:
                color = LEAF_GREENS[leaf_idx % len(LEAF_GREENS)]
                leaf_idx += 1
                sw = '1.6'
            pts_svg = proj.project(skel, panel_ox=0.0, panel_oy=0.0)
            pts_str = ' '.join(f'{p[0]:.1f},{p[1]:.1f}' for p in pts_svg)
            buf.append(
                f'<polyline points="{pts_str}" stroke="{_rgb_hex(*color)}" '
                f'stroke-width="{sw}" fill="none" '
                f'stroke-linecap="round" stroke-linejoin="round"/>'
            )

    buf.append('</svg>')

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text('\n'.join(buf))
    size_kb = output_path.stat().st_size / 1024
    print(f"  G1 standalone: {output_path}  ({size_kb:.0f} KB)")
    return output_path


def render_g3_standalone_svg(grid_info, obj_paths, mapping_paths, output_path,
                              az_deg=40.0, el_deg=25.0, subsample=1):
    """Write a single G3-mesh SVG with transparent background."""
    positions_cm = [np.array(p) * 100.0 for p in grid_info['positions_m']]

    # Load + translate all plants
    print("  Loading G3 geometry...")
    parsed = []
    all_verts = []
    for i, (obj_p, map_p) in enumerate(zip(obj_paths, mapping_paths)):
        v, f, fc = load_plant_geometry(obj_p, map_p)
        v[:, 0] += positions_cm[i][0]
        v[:, 1] += positions_cm[i][1]
        parsed.append((v, f, fc))
        all_verts.append(v)

    combined = np.concatenate(all_verts, axis=0)
    pad = max(combined.max(axis=0) - combined.min(axis=0)) * 0.04
    bounds = [combined[:, 0].min() - pad, combined[:, 0].max() + pad,
              combined[:, 1].min() - pad, combined[:, 1].max() + pad,
              combined[:, 2].min() - pad, combined[:, 2].max() + pad]

    margin = 18.0
    panel_w = 720.0
    panel_h = 760.0
    proj = IsoProjector(az_deg, el_deg, bounds, panel_w, panel_h, margin)

    buf = []
    buf.append('<?xml version="1.0" encoding="UTF-8"?>')
    buf.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {panel_w:.1f} {panel_h:.1f}" '
        f'width="{int(panel_w)}" height="{int(panel_h)}">'
    )
    # No background rect → transparent

    buf.extend(_ground_grid_lines(proj, bounds, panel_ox=0.0, panel_oy=0.0))

    # Collect triangles
    all_depths_list, all_colors_list, all_pts2d_list = [], [], []
    total_tris = 0
    for i, (verts_w, faces, face_colors_base) in enumerate(parsed):
        if len(faces) == 0:
            continue
        if subsample > 1:
            keep = np.arange(0, len(faces), subsample)
            faces = faces[keep]
            face_colors_base = face_colors_base[keep]
        total_tris += len(faces)

        v0 = verts_w[faces[:, 0]]
        v1 = verts_w[faces[:, 1]]
        v2 = verts_w[faces[:, 2]]
        e1, e2 = v1 - v0, v2 - v0
        normals = np.cross(e1, e2)
        nlen = np.linalg.norm(normals, axis=1, keepdims=True)
        normals /= np.where(nlen < 1e-12, 1.0, nlen)

        shade = 0.30 + 0.70 * np.abs(normals @ LIGHT_DIR)
        colors_u8 = (np.clip(face_colors_base * shade[:, np.newaxis], 0, 1) * 255).astype(np.uint8)

        pts2d = proj.project(verts_w, panel_ox=0.0, panel_oy=0.0)
        pts_flat = np.concatenate(
            [pts2d[faces[:, 0]], pts2d[faces[:, 1]], pts2d[faces[:, 2]]], axis=1
        ).astype(np.float32)

        depths = proj.depth(verts_w)
        tri_depths = (depths[faces[:, 0]] + depths[faces[:, 1]] + depths[faces[:, 2]]) / 3.0

        all_depths_list.append(tri_depths)
        all_colors_list.append(colors_u8)
        all_pts2d_list.append(pts_flat)

    all_depths = np.concatenate(all_depths_list)
    all_colors = np.concatenate(all_colors_list)
    all_pts2d = np.concatenate(all_pts2d_list)

    print(f"  Sorting {len(all_depths)} triangles...")
    order = np.argsort(-all_depths)
    all_colors = all_colors[order]
    all_pts2d = all_pts2d[order]

    print(f"  Writing {len(order)} triangles...")
    for ti in range(len(order)):
        x0, y0, x1, y1, x2, y2 = all_pts2d[ti]
        r, g, b = all_colors[ti]
        buf.append(
            f'<polygon points="{x0:.1f},{y0:.1f} {x1:.1f},{y1:.1f} {x2:.1f},{y2:.1f}" '
            f'fill="#{r:02x}{g:02x}{b:02x}"/>'
        )

    # Scale bar
    bar_world_a = np.array([[bounds[0] + 5, bounds[2] + 5, bounds[4]]])
    bar_world_b = np.array([[bounds[0] + 55, bounds[2] + 5, bounds[4]]])
    ba = proj.project(bar_world_a, 0.0, 0.0)[0]
    bb = proj.project(bar_world_b, 0.0, 0.0)[0]
    bar_len = np.sqrt((bb[0] - ba[0]) ** 2 + (bb[1] - ba[1]) ** 2)
    bar_y = panel_h - 14
    bar_x0 = margin
    bar_x1 = bar_x0 + bar_len
    bar_mid = (bar_x0 + bar_x1) / 2
    buf.append(
        f'<line x1="{bar_x0:.1f}" y1="{bar_y:.1f}" x2="{bar_x1:.1f}" y2="{bar_y:.1f}" '
        f'stroke="#333" stroke-width="2"/>'
    )
    for bx in [bar_x0, bar_x1]:
        buf.append(
            f'<line x1="{bx:.1f}" y1="{bar_y - 4:.1f}" x2="{bx:.1f}" y2="{bar_y + 4:.1f}" '
            f'stroke="#333" stroke-width="2"/>'
        )
    buf.append(
        f'<text x="{bar_mid:.1f}" y="{bar_y - 6:.1f}" text-anchor="middle" '
        f'font-family="Times New Roman,Times,serif" font-size="11" fill="#333">50 cm</text>'
    )

    buf.append('</svg>')

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text('\n'.join(buf))
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  G3 standalone: {output_path}  ({size_mb:.1f} MB, {total_tris} triangles)")
    return output_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Render multifield G1|G3 field view as SVG",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--output', default=None,
                        help='Output SVG path (default: output/multifield_field_view.svg)')
    parser.add_argument('--no-g1', action='store_true',
                        help='Skip G1 panel (faster; no re-growing plants)')
    parser.add_argument('--subsample', type=int, default=1, metavar='N',
                        help='Keep every N-th G3 triangle (default 1 = all; '
                             'try 3 for a lighter file)')
    parser.add_argument('--az', type=float, default=40.0,
                        help='Camera azimuth in degrees (default 40)')
    parser.add_argument('--el', type=float, default=25.0,
                        help='Camera elevation in degrees (default 25)')
    parser.add_argument('--individual', action='store_true',
                        help='Also write standalone G1-only and G3-only SVGs '
                             '(transparent background, no panel decorations)')
    args = parser.parse_args()

    # Locate coupling directory and outputs
    coupling_dir = Path(__file__).resolve().parent
    output_dir = coupling_dir / 'output'

    if args.output is None:
        args.output = str(output_dir / 'multifield_field_view.svg')

    print("=" * 70)
    print("Multifield G1 | G3 Field View SVG")
    print("=" * 70)
    print(f"  Azimuth={args.az}°  Elevation={args.el}°  Subsample={args.subsample}")

    # Load grid info
    grid_info_path = output_dir / 'multifield_grid_info.json'
    if not grid_info_path.exists():
        print(f"ERROR: {grid_info_path} not found. "
              f"Run 'python -m coupling multifield' first.")
        sys.exit(1)
    with open(grid_info_path) as f:
        grid_info = json.load(f)

    # Locate OBJ + mapping files
    obj_paths, mapping_paths = [], []
    for i in range(N_PLANTS):
        obj_p = output_dir / f'multifield_p{i}.obj'
        map_p = output_dir / f'multifield_p{i}_mapping.json'
        if not obj_p.exists():
            print(f"ERROR: {obj_p} not found.")
            sys.exit(1)
        if not map_p.exists():
            print(f"ERROR: {map_p} not found.")
            sys.exit(1)
        obj_paths.append(obj_p)
        mapping_paths.append(map_p)

    print(f"  Found {N_PLANTS} OBJ + mapping files in {output_dir}")

    # ------------------------------------------------------------------
    # G1 skeletons: re-grow 9 plants with the same seeds
    # ------------------------------------------------------------------
    all_organ_dicts = None
    if not args.no_g1:
        print("\n--- Re-growing G1 skeletons (seeds 42–50) ---")

        # Python adds the script's own dir (dart/coupling/) to sys.path[0] when
        # run as a standalone script — this breaks relative imports inside the
        # coupling package.  Replace it with the CPlantBox repo root so that
        # 'dart.coupling.*' is importable with working relative imports.
        cplantbox_root = str(coupling_dir.parent.parent)  # .../CPlantBox
        sys.path = [p for p in sys.path
                    if p not in (str(coupling_dir), '')]
        if cplantbox_root not in sys.path:
            sys.path.insert(0, cplantbox_root)

        try:
            import plantbox as pb  # noqa: F401
            from dart.coupling.growth import grow_plant
            from dart.coupling.geometry import extract_organs_for_lofter
            from dart.coupling.geometry.cplantbox_adapter import (
                get_plantsim_feature_kwargs_from_env,
            )
            from dart.coupling.config import DEFAULT_XML
        except ImportError as e:
            print(f"  Import error: {e}")
            print("  Falling back to --no-g1 mode (G3 panel only).")
            all_organ_dicts = None
        else:
            xml_path = str(DEFAULT_XML)
            all_organ_dicts = []
            feature_kwargs = get_plantsim_feature_kwargs_from_env()
            for i in range(N_PLANTS):
                seed = FIELD_SEED + i
                print(f"  Plant {i} (seed={seed})...", flush=True)
                plant = grow_plant(xml_path, SIMULATION_DAYS, seed=seed)
                organs = extract_organs_for_lofter(
                    plant,
                    min_stem_nodes=50,
                    min_leaf_nodes=20,
                    name_prefix=f'p{i}_',
                    **feature_kwargs,
                )
                all_organ_dicts.append(organs)

            n_organs_total = sum(len(o) for o in all_organ_dicts)
            print(f"  G1 ready: {n_organs_total} organs across {N_PLANTS} plants")

    # ------------------------------------------------------------------
    # Render combined panel
    # ------------------------------------------------------------------
    print("\n--- Rendering combined SVG ---")
    out = render_field_svg(
        grid_info=grid_info,
        all_organ_dicts=all_organ_dicts,
        obj_paths=obj_paths,
        mapping_paths=mapping_paths,
        output_path=args.output,
        az_deg=args.az,
        el_deg=args.el,
        subsample=args.subsample,
    )

    # ------------------------------------------------------------------
    # Individual standalone SVGs (transparent background)
    # ------------------------------------------------------------------
    if args.individual:
        stem = Path(args.output).with_suffix('')
        print("\n--- Rendering standalone G1 SVG (transparent) ---")
        if all_organ_dicts is not None:
            render_g1_standalone_svg(
                grid_info=grid_info,
                all_organ_dicts=all_organ_dicts,
                output_path=str(stem) + '_g1_standalone.svg',
                az_deg=args.az,
                el_deg=args.el,
            )
        else:
            print("  Skipped (--no-g1 was set)")

        print("\n--- Rendering standalone G3 SVG (transparent) ---")
        render_g3_standalone_svg(
            grid_info=grid_info,
            obj_paths=obj_paths,
            mapping_paths=mapping_paths,
            output_path=str(stem) + '_g3_standalone.svg',
            az_deg=args.az,
            el_deg=args.el,
            subsample=args.subsample,
        )

    print(f"\nDone: {out}")


if __name__ == '__main__':
    main()
