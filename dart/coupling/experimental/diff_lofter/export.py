"""Non-differentiable mesh export for the diff_lofter.

Adds triangle index generation and OBJ file writing on top of the
diff_lofter's vertex-only output. Kept separate from lofter.py
because that module is pure autograd (no indices needed for Chamfer loss).

Triangle index pattern matches the production lofter (g1_to_g3.py:808-821).
"""

import numpy as np
import torch

from .frames import compute_binormal_field, compute_tangents
from .lofter import loft_leaf, resample_skeleton


def build_triangle_indices(
    n_rows: int,
    n_cross: int,
    offset: int = 0,
) -> np.ndarray:
    """Build triangle indices for an N x C vertex grid.

    For each quad (i, j), two triangles:
        bl = n_cross * i + j + offset
        br = bl + 1
        tl = bl + n_cross
        tr = tl + 1
        tri1 = [bl, tl, br]    (bottom-left, top-left, bottom-right)
        tri2 = [br, tl, tr]    (bottom-right, top-left, top-right)

    Args:
        n_rows: Number of skeleton nodes (rows in vertex grid).
        n_cross: Number of cross-section vertices per row.
        offset: Global vertex index offset (for multi-organ assembly).

    Returns:
        ((n_rows-1) * 2 * (n_cross-1), 3) int32 triangle indices.
    """
    n_segs = n_rows - 1
    n_quads = n_cross - 1
    n_tris = n_segs * 2 * n_quads
    indices = np.empty((n_tris, 3), dtype=np.int32)

    idx = 0
    for i in range(n_segs):
        for j in range(n_quads):
            bl = n_cross * i + j + offset
            br = bl + 1
            tl = bl + n_cross
            tr = tl + 1
            indices[idx] = [bl, tl, br]
            indices[idx + 1] = [br, tl, tr]
            idx += 2

    return indices


def compute_vertex_normals(
    vertices: np.ndarray,
    triangles: np.ndarray,
) -> np.ndarray:
    """Compute per-vertex normals by averaging adjacent face normals.

    Args:
        vertices: (V, 3) vertex positions.
        triangles: (T, 3) triangle vertex indices.

    Returns:
        (V, 3) unit normals per vertex.
    """
    v0 = vertices[triangles[:, 0]]
    v1 = vertices[triangles[:, 1]]
    v2 = vertices[triangles[:, 2]]

    face_normals = np.cross(v1 - v0, v2 - v0)

    vertex_normals = np.zeros_like(vertices)
    for k in range(3):
        np.add.at(vertex_normals, triangles[:, k], face_normals)

    norms = np.linalg.norm(vertex_normals, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return vertex_normals / norms


def export_obj(
    filepath: str,
    organs: list[dict],
    write_normals: bool = True,
) -> None:
    """Export multi-organ mesh to Wavefront OBJ.

    Each organ dict must have:
        'name': str (e.g., 'leaf_0')
        'vertices': (V, 3) numpy array
        'n_rows': int (skeleton node count after resampling)
        'n_cross': int (cross-section vertex count)

    Args:
        filepath: Output .obj file path.
        organs: List of organ dicts.
        write_normals: Whether to compute and write vertex normals.
    """
    with open(filepath, 'w') as f:
        f.write("# Wheat reconstruction mesh\n")

        # First pass: write all vertices (and normals)
        all_normals = []
        for organ in organs:
            verts = organ['vertices']
            for v in verts:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

            if write_normals:
                tris = build_triangle_indices(organ['n_rows'], organ['n_cross'])
                normals = compute_vertex_normals(verts, tris)
                all_normals.append(normals)

        if write_normals:
            for normals in all_normals:
                for n in normals:
                    f.write(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n")

        # Second pass: write faces per organ group
        v_offset = 0
        for organ in organs:
            n_verts = organ['vertices'].shape[0]
            f.write(f"g {organ['name']}\n")

            tris = build_triangle_indices(
                organ['n_rows'], organ['n_cross'], offset=0)

            for tri in tris:
                # OBJ is 1-indexed
                a = tri[0] + v_offset + 1
                b = tri[1] + v_offset + 1
                c = tri[2] + v_offset + 1
                if write_normals:
                    f.write(f"f {a}//{a} {b}//{b} {c}//{c}\n")
                else:
                    f.write(f"f {a} {b} {c}\n")

            v_offset += n_verts


def _make_zero_deformations(n: int, device: str = 'cpu') -> dict[str, torch.Tensor]:
    """Create zero-valued deformation dict for n skeleton nodes."""
    keys = ['wave_normal', 'wave_lateral', 'twist', 'curl', 'edge_ruffle', 'fold']
    return {k: torch.zeros(n, device=device) for k in keys}


def loft_and_export(
    skeletons: list[np.ndarray],
    half_widths: list[np.ndarray],
    names: list[str],
    filepath: str,
    n_cross: int = 11,
    gutter_depth: float = 0.1,
    target_spacing: float = 0.2,
) -> dict:
    """Loft multiple organs through the diff_lofter and export to OBJ.

    Convenience function that handles torch conversion, frame computation,
    zero deformations, lofting, and OBJ export.

    Args:
        skeletons: List of (N_i, 3) numpy skeleton arrays.
        half_widths: List of (N_i,) numpy half-width arrays.
        names: List of organ names (e.g., ['leaf_0', 'leaf_1', ...]).
        filepath: Output OBJ path.
        n_cross: Cross-section vertex count.
        gutter_depth: Midrib V-fold depth in cm.
        target_spacing: Skeleton resampling spacing in cm.

    Returns:
        Summary dict with vertex/triangle counts.
    """
    organs = []
    total_verts = 0
    total_tris = 0

    for skel_np, w_np, name in zip(skeletons, half_widths, names):
        # Convert to torch
        skel_t = torch.tensor(skel_np, dtype=torch.float32)
        w_t = torch.tensor(w_np, dtype=torch.float32)

        # Resample
        skel_t, w_t = resample_skeleton(skel_t, w_t, target_spacing=target_spacing)
        n_rows = skel_t.shape[0]

        if n_rows < 2:
            continue

        # Compute frames
        tangents = compute_tangents(skel_t)
        binormals = compute_binormal_field(skel_t, tangents)

        # Zero deformations
        deforms = _make_zero_deformations(n_rows)

        # Loft
        verts = loft_leaf(
            skel_t, w_t, deforms, tangents, binormals,
            n_cross=n_cross, gutter_depth=gutter_depth,
        )
        verts_np = verts.detach().numpy()

        n_tris = (n_rows - 1) * 2 * (n_cross - 1)
        organs.append({
            'name': name,
            'vertices': verts_np,
            'n_rows': n_rows,
            'n_cross': n_cross,
        })
        total_verts += verts_np.shape[0]
        total_tris += n_tris

    export_obj(filepath, organs)

    return {
        'n_organs': len(organs),
        'total_vertices': total_verts,
        'total_triangles': total_tris,
    }
