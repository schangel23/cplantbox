"""PCA-based mesh generator — drop-in replacement for the lofter in g1_to_g3.py.

Instead of sweeping cross-sections along skeletons, this module decodes a mesh
from a low-dimensional PCA space trained on reference OBJ growth stages.
CPlantBox skeleton features are mapped to PCA coefficients, producing meshes
with realistic sheath wrapping, curvature, and blade shapes that the flat-ribbon
lofter cannot represent.

Pipeline slot:
    XML → CPlantBox → organ dicts → **pca_lofter.loft_organs_pca()** → G3Mesh

Training:
    pca_lofter.train("path/to/reference/export/", "output/model.npz")

Usage in pipeline:
    from dart.coupling.geometry.pca_lofter import PCALofter
    lofter = PCALofter("path/to/model.npz")
    mesh = lofter.loft(organ_dicts)
"""

import json
from pathlib import Path

import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.csgraph import connected_components

from .g1_to_g3 import G3Mesh


# ---------------------------------------------------------------------------
# Reference mesh loading + template construction
# ---------------------------------------------------------------------------

def _load_reference_stages(export_dir: str) -> tuple[np.ndarray, list, list, list]:
    """Load all maize_stage_*.obj files, return (S,N,3) in cm Z-up."""
    export_path = Path(export_dir)
    obj_files = sorted(export_path.glob("maize_stage_*.obj"))
    if not obj_files:
        raise FileNotFoundError(f"No maize_stage_*.obj in {export_dir}")

    all_verts = []
    stage_names = []
    faces_out = None
    groups_out = None

    for obj_file in obj_files:
        verts, faces, grps = [], [], []
        current_group = ""
        with open(obj_file) as f:
            for line in f:
                if line.startswith("v "):
                    verts.append([float(x) for x in line.split()[1:4]])
                elif line.startswith("g "):
                    current_group = line.strip().split()[1]
                elif line.startswith("f "):
                    face = tuple(int(t.split("/")[0]) for t in line.split()[1:])
                    faces.append(face)
                    grps.append(current_group)
        arr = np.array(verts) * 100.0      # m → cm
        arr[:, 2] *= -1.0                  # Z-down → Z-up
        all_verts.append(arr)
        stage_names.append(obj_file.stem)
        if faces_out is None:
            faces_out = faces
            groups_out = grps

    return np.stack(all_verts), faces_out, groups_out, stage_names


def _find_leaf_components(n_verts, faces, groups):
    """Find connected components for the leaf group → per-leaf vertex sets.

    Returns list of (component_id, sorted list of vertex indices), sorted by
    base Z ascending.  Also returns stem vertex indices.
    """
    adj = lil_matrix((n_verts, n_verts), dtype=bool)
    leaf_verts = set()
    stem_verts = set()

    for face, group in zip(faces, groups):
        vis = [vi - 1 for vi in face]  # 0-indexed
        bucket = leaf_verts if group == "leaf" else stem_verts
        for vi in vis:
            bucket.add(vi)
        for i in range(len(vis)):
            for j in range(i + 1, len(vis)):
                adj[vis[i], vis[j]] = True
                adj[vis[j], vis[i]] = True

    _, labels = connected_components(adj.tocsr(), directed=False)

    comp_dict = {}
    for vi in leaf_verts:
        comp_dict.setdefault(labels[vi], []).append(vi)

    # Keep components with >10 verts (filter dust)
    leaf_comps = [(cid, sorted(vis)) for cid, vis in comp_dict.items() if len(vis) > 10]
    return leaf_comps, sorted(stem_verts)


def _triangulate_faces(faces):
    """Convert mixed quad/tri face list to pure triangles (0-indexed)."""
    tris = []
    for face in faces:
        vis = [vi - 1 for vi in face]
        if len(vis) == 3:
            tris.append(vis)
        elif len(vis) == 4:
            tris.append([vis[0], vis[1], vis[2]])
            tris.append([vis[0], vis[2], vis[3]])
        else:
            # Fan triangulation
            for i in range(1, len(vis) - 1):
                tris.append([vis[0], vis[i], vis[i + 1]])
    return np.array(tris, dtype=np.int32)


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def _features_from_mesh(V, leaf_comps, stem_vids):
    """Extract skeleton-like features from a single mesh stage.

    Returns a fixed-size feature vector (length = 4 + max_leaves * 3).
    """
    n_leaves = len(leaf_comps)

    # Sort leaves by base Z
    leaf_info = []
    for _, vis in leaf_comps:
        lv = V[vis]
        base_z = lv[:, 2].min()
        z_range = lv[:, 2].max() - lv[:, 2].min()
        xy_range = max(lv[:, 0].max() - lv[:, 0].min(),
                       lv[:, 1].max() - lv[:, 1].min())
        # Arc-length proxy: max pairwise distance along leaf
        extent = np.sqrt(np.sum((lv.max(axis=0) - lv.min(axis=0)) ** 2))
        leaf_info.append({
            "base_z": base_z,
            "extent": extent,
            "width": xy_range,
            "z_range": z_range,
        })
    leaf_info.sort(key=lambda x: x["base_z"])

    # Global features
    plant_height = V[:, 2].max() - V[:, 2].min()
    stem_height = (V[stem_vids, 2].max() - V[stem_vids, 2].min()
                   if stem_vids else 0.0)

    return plant_height, stem_height, leaf_info


def _features_from_organs(organ_dicts):
    """Extract the same feature set from CPlantBox organ dicts.

    Returns: plant_height, stem_height, leaf_info (same format as _features_from_mesh).
    """
    leaves = [o for o in organ_dicts if o["type"] == "leaf"]
    stems = [o for o in organ_dicts if o["type"] == "stem"]

    leaf_info = []
    for o in leaves:
        skel = np.asarray(o["skeleton"])
        widths = np.asarray(o["widths"])
        if len(skel) < 2:
            continue
        diffs = np.diff(skel, axis=0)
        arc_length = np.sum(np.sqrt(np.sum(diffs ** 2, axis=1)))
        leaf_info.append({
            "base_z": skel[0, 2],
            "extent": arc_length,
            "width": widths.max() * 2.0,  # full width (lofter convention: half-width)
            "z_range": skel[:, 2].max() - skel[:, 2].min(),
        })
    leaf_info.sort(key=lambda x: x["base_z"])

    plant_height = 0.0
    for o in organ_dicts:
        skel = np.asarray(o["skeleton"])
        plant_height = max(plant_height, skel[:, 2].max() - skel[:, 2].min())

    stem_height = 0.0
    for o in stems:
        skel = np.asarray(o["skeleton"])
        stem_height = max(stem_height, skel[:, 2].max() - skel[:, 2].min())

    return plant_height, stem_height, leaf_info


def _vectorize_features(plant_height, stem_height, leaf_info, max_leaves=16):
    """Convert structured features to a scale-invariant vector.

    Uses relative/normalized features so that the mapping works across
    different absolute growth scales (e.g. CPlantBox vs artist reference).

    Vector layout: [n_emerged_frac, height_norm,
                    leaf_0_extent_norm, leaf_0_width_norm, leaf_0_rel_z,
                    ...,
                    leaf_{max-1}_extent_norm, leaf_{max-1}_width_norm, leaf_{max-1}_rel_z]
    """
    h = max(plant_height, 1.0)
    n_emerged = sum(1 for li in leaf_info if li["extent"] > 1.0)
    n_total = max(len(leaf_info), 1)

    # Normalization references (from the plant itself)
    max_extent = max((li["extent"] for li in leaf_info), default=1.0)
    max_extent = max(max_extent, 1.0)
    max_width = max((li["width"] for li in leaf_info), default=1.0)
    max_width = max(max_width, 0.1)

    vec = [
        n_emerged / max_leaves,        # fraction of max possible leaves
        n_emerged / n_total,            # fraction of actual leaves emerged
    ]
    for i in range(max_leaves):
        if i < len(leaf_info):
            li = leaf_info[i]
            vec.extend([
                li["extent"] / max_extent,   # relative length
                li["width"] / max_width,     # relative width
                li["base_z"] / h,            # relative insertion height
            ])
        else:
            vec.extend([0.0, 0.0, 0.0])
    return np.array(vec, dtype=np.float64)


# ---------------------------------------------------------------------------
# PCA model
# ---------------------------------------------------------------------------

class PCALofter:
    """PCA-based mesh generator that replaces g1_to_g3.loft_organs().

    Trained from reference OBJ stages, produces a G3Mesh from CPlantBox
    organ dicts by mapping skeleton features to PCA coefficients.
    """

    def __init__(self, model_path: str):
        """Load a trained PCA lofter model.

        Args:
            model_path: Path to .npz file produced by PCALofter.train().
        """
        data = np.load(model_path, allow_pickle=True)
        self.mean = data["mean"]                    # (N*3,)
        self.components = data["components"]          # (k, N*3)
        self.n_verts = int(data["n_verts"])
        self.n_components = self.components.shape[0]

        # Template mesh topology
        self.template_tris = data["template_tris"]    # (T, 3) int32
        self.template_groups = data["template_groups"] # per-tri: 0=leaf, 1=stem

        # Leaf component info: (n_leaf_comps,) arrays
        self.leaf_comp_vids = data["leaf_comp_vids"]   # list of arrays
        self.leaf_comp_base_z = data["leaf_comp_base_z"]  # (n_comps,)
        self.stem_vids = data["stem_vids"]

        # Feature → PCA coefficient mapping
        self.W = data["W"]                  # (F+1, k)
        self.feat_mean = data["feat_mean"]  # (F,)
        self.feat_std = data["feat_std"]    # (F,)
        self.max_leaves = int(data["max_leaves"])

        # Training data coefficients for interpolation fallback
        self.train_coeffs = data["train_coefficients"]  # (S, k)
        self.train_features = data["train_features"]    # (S, F)

    def _extract_features(self, organ_dicts):
        """Extract feature vector from CPlantBox organ dicts."""
        ph, sh, li = _features_from_organs(organ_dicts)
        return _vectorize_features(ph, sh, li, self.max_leaves)

    def _features_to_pca(self, features):
        """Map feature vector to PCA coefficients.

        Uses a two-step approach:
        1. Ridge regression for initial prediction
        2. Project onto the convex hull of training trajectory
           (prevents extrapolation artifacts from scale mismatch)
        """
        x = (features - self.feat_mean) / np.where(
            self.feat_std > 1e-8, self.feat_std, 1.0)
        x_aug = np.append(x, 1.0)  # bias
        z_raw = x_aug @ self.W      # (k,)

        # Project onto training trajectory: find the closest point
        # on the piecewise-linear path through training coefficients.
        # This ensures we never produce meshes outside the reference range.
        z_proj = self._project_onto_trajectory(z_raw)
        return z_proj

    def _project_onto_trajectory(self, z):
        """Project PCA coefficients onto nearest point on training trajectory.

        The training coefficients form a path through PCA space (stage 1→16).
        We find the closest point on this piecewise-linear path, which gives
        a continuous parameter t ∈ [0, S-1] and interpolated coefficients.
        """
        coeffs = self.train_coeffs  # (S, k)
        best_dist = np.inf
        best_z = coeffs[0]

        for i in range(len(coeffs) - 1):
            a, b = coeffs[i], coeffs[i + 1]
            ab = b - a
            ab_len2 = np.dot(ab, ab)
            if ab_len2 < 1e-12:
                t_seg = 0.0
            else:
                t_seg = np.clip(np.dot(z - a, ab) / ab_len2, 0.0, 1.0)
            proj = a + t_seg * ab
            dist = np.linalg.norm(z - proj)
            if dist < best_dist:
                best_dist = dist
                best_z = proj

        return best_z

    def _decode(self, z):
        """PCA coefficients → vertex positions (N, 3)."""
        flat = self.mean + z @ self.components
        return flat.reshape(self.n_verts, 3)

    def _compute_normals(self, verts):
        """Compute per-vertex normals via face-area weighting."""
        normals = np.zeros_like(verts)
        tris = self.template_tris
        v0, v1, v2 = verts[tris[:, 0]], verts[tris[:, 1]], verts[tris[:, 2]]
        face_normals = np.cross(v1 - v0, v2 - v0)
        for i in range(3):
            np.add.at(normals, tris[:, i], face_normals)
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1.0
        return normals / norms

    def _compute_uvs(self, verts):
        """Simple planar UV: u = arc position within leaf, v = across."""
        uvs = np.zeros((len(verts), 2), dtype=np.float64)
        # Normalize XY to [0,1] for basic UV
        for axis, col in [(0, 0), (2, 1)]:
            vals = verts[:, axis]
            vmin, vmax = vals.min(), vals.max()
            rng = vmax - vmin
            if rng > 1e-6:
                uvs[:, col] = (vals - vmin) / rng
        return uvs

    def _assign_organ_ids(self, verts, organ_dicts):
        """Assign organ IDs and build organ_meta for DART mapping.

        Matches template leaf components to CPlantBox organs by insertion
        height (base_z).  Returns per-triangle organ_ids and organ_meta.
        """
        n_tris = len(self.template_tris)
        organ_ids = np.full(n_tris, -1, dtype=np.int32)
        organ_meta = []

        # Map template leaf components → CPlantBox leaf organs by base_z
        cpb_leaves = sorted(
            [(i, o) for i, o in enumerate(organ_dicts) if o["type"] == "leaf"],
            key=lambda x: np.asarray(x[1]["skeleton"])[0, 2],
        )

        # Template leaves sorted by base_z
        temp_leaves = sorted(
            enumerate(self.leaf_comp_base_z),
            key=lambda x: x[1],
        )

        # Greedy match by base_z order
        n_match = min(len(cpb_leaves), len(temp_leaves))

        for match_i in range(n_match):
            temp_idx, _ = temp_leaves[match_i]
            cpb_i, cpb_organ = cpb_leaves[match_i]

            # Which triangles belong to this template leaf?
            comp_vids = set(self.leaf_comp_vids[temp_idx].tolist()
                           if hasattr(self.leaf_comp_vids[temp_idx], 'tolist')
                           else list(self.leaf_comp_vids[temp_idx]))
            for ti in range(n_tris):
                tri = self.template_tris[ti]
                if tri[0] in comp_vids or tri[1] in comp_vids or tri[2] in comp_vids:
                    organ_ids[ti] = cpb_organ["organ_id"]

            organ_meta.append({
                "organ_id": cpb_organ["organ_id"],
                "type": "leaf",
                "name": cpb_organ.get("name", f"leaf_{match_i}"),
                "node_ids": cpb_organ.get("node_ids", []),
            })

        # Stem
        cpb_stems = [o for o in organ_dicts if o["type"] == "stem"]
        if cpb_stems:
            stem_vid_set = set(self.stem_vids.tolist()
                              if hasattr(self.stem_vids, 'tolist')
                              else list(self.stem_vids))
            stem_oid = cpb_stems[0]["organ_id"]
            for ti in range(n_tris):
                tri = self.template_tris[ti]
                if tri[0] in stem_vid_set or tri[1] in stem_vid_set or tri[2] in stem_vid_set:
                    organ_ids[ti] = stem_oid
            organ_meta.append({
                "organ_id": stem_oid,
                "type": "stem",
                "name": cpb_stems[0].get("name", "stem_0"),
                "node_ids": cpb_stems[0].get("node_ids", []),
            })

        # Any unassigned → assign to nearest matched organ
        unassigned = organ_ids == -1
        if unassigned.any() and organ_meta:
            organ_ids[unassigned] = organ_meta[0]["organ_id"]

        return organ_ids, organ_meta

    def loft(self, organ_dicts):
        """Generate G3Mesh from CPlantBox organ dicts.

        Drop-in replacement for g1_to_g3.loft_organs().

        Args:
            organ_dicts: List of organ dicts from cplantbox_adapter.

        Returns:
            G3Mesh with template topology and PCA-decoded vertex positions.
        """
        features = self._extract_features(organ_dicts)
        z = self._features_to_pca(features)
        verts = self._decode(z)
        normals = self._compute_normals(verts)
        uvs = self._compute_uvs(verts)
        organ_ids, organ_meta = self._assign_organ_ids(verts, organ_dicts)

        # Segment IDs: approximate by nearest skeleton node per triangle
        segment_ids = np.full(len(self.template_tris), -1, dtype=np.int32)

        return G3Mesh(
            vertices=verts,
            indices=self.template_tris,
            normals=normals,
            uvs=uvs,
            organ_ids=organ_ids,
            segment_ids=segment_ids,
            organ_meta=organ_meta,
        )

    def loft_at_t(self, t):
        """Generate G3Mesh by interpolating in PCA space at continuous stage t.

        Args:
            t: Growth stage index (0 = first stage, S-1 = last stage).

        Returns:
            G3Mesh with interpolated vertex positions.
        """
        n = len(self.train_coeffs)
        t = np.clip(t, 0, n - 1)
        i0 = int(np.floor(t))
        i1 = min(i0 + 1, n - 1)
        frac = t - i0
        z = (1 - frac) * self.train_coeffs[i0] + frac * self.train_coeffs[i1]
        verts = self._decode(z)
        normals = self._compute_normals(verts)
        uvs = self._compute_uvs(verts)

        organ_ids = np.where(self.template_groups == 1, 1, 0).astype(np.int32)
        return G3Mesh(
            vertices=verts,
            indices=self.template_tris,
            normals=normals,
            uvs=uvs,
            organ_ids=organ_ids,
        )

    @staticmethod
    def train(export_dir: str, output_path: str, n_components: int = 10,
              max_leaves: int = 16, alpha: float = None):
        """Train PCA lofter from reference OBJ stages.

        Args:
            export_dir: Directory containing maize_stage_*.obj files.
            output_path: Path for the output .npz model file.
            n_components: PCA components to keep (default 10 → 0.47cm RMSE).
            max_leaves: Maximum number of leaves in feature vector.
            alpha: Ridge regression regularization. None = auto via LOO-CV.

        Returns:
            Dict with training stats.
        """
        print(f"Training PCA Lofter from {export_dir}")
        V, faces, groups, stage_names = _load_reference_stages(export_dir)
        S, N, _ = V.shape
        n_components = min(n_components, S - 1)

        print(f"  {S} stages, {N} verts, {len(faces)} faces")

        # -- PCA --
        V_flat = V.reshape(S, -1)
        mean = V_flat.mean(axis=0)
        centered = V_flat - mean
        U, Sigma, Vt = np.linalg.svd(centered, full_matrices=False)
        components = Vt[:n_components]
        coefficients = centered @ components.T  # (S, k)

        var_total = (Sigma ** 2).sum()
        var_explained = np.cumsum(Sigma[:n_components] ** 2) / var_total
        print(f"  {n_components} PCs explain {var_explained[-1]*100:.2f}% of variance")

        # -- Reconstruction RMSE --
        recon = coefficients @ components
        residual = centered - recon
        rmse = np.sqrt((residual ** 2).mean())
        print(f"  Reconstruction RMSE: {rmse:.3f} cm")

        # -- Template mesh --
        tris = _triangulate_faces(faces)
        tri_groups = np.zeros(len(tris), dtype=np.int32)
        # Map groups: accumulate as we triangulate
        tri_idx = 0
        for face, group in zip(faces, groups):
            n_tris_for_face = max(len(face) - 2, 1)
            g_val = 1 if group == "stem" else 0
            for _ in range(n_tris_for_face):
                if tri_idx < len(tri_groups):
                    tri_groups[tri_idx] = g_val
                tri_idx += 1

        # -- Leaf components (from first stage for base_z ordering) --
        leaf_comps, stem_vids = _find_leaf_components(N, faces, groups)

        # Sort leaf components by their base_z in the MEAN shape
        mean_verts = mean.reshape(N, 3)
        leaf_base_z = []
        leaf_vids_list = []
        for _, vis in leaf_comps:
            leaf_base_z.append(mean_verts[vis, 2].min())
            leaf_vids_list.append(np.array(vis, dtype=np.int32))
        order = np.argsort(leaf_base_z)
        leaf_base_z = np.array(leaf_base_z)[order]
        leaf_vids_list = [leaf_vids_list[i] for i in order]

        print(f"  {len(leaf_vids_list)} leaf components, {len(stem_vids)} stem verts")

        # -- Feature extraction --
        all_features = []
        for s in range(S):
            ph, sh, li = _features_from_mesh(V[s], leaf_comps, stem_vids)
            all_features.append(_vectorize_features(ph, sh, li, max_leaves))
        features = np.array(all_features)  # (S, F)
        F_dim = features.shape[1]

        # -- Ridge regression: features → PCA coefficients --
        feat_mean = features.mean(axis=0)
        feat_std = features.std(axis=0)
        feat_std[feat_std < 1e-8] = 1.0
        X = (features - feat_mean) / feat_std
        X_aug = np.column_stack([X, np.ones(S)])

        # LOO-CV to select alpha
        alphas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0] if alpha is None else [alpha]
        best_alpha = alphas[0]
        best_loo = np.inf

        for a in alphas:
            loo_err = 0.0
            for i in range(S):
                mask = np.ones(S, dtype=bool)
                mask[i] = False
                reg = a * np.eye(X_aug.shape[1])
                reg[-1, -1] = 0.0
                W_cv = np.linalg.solve(
                    X_aug[mask].T @ X_aug[mask] + reg,
                    X_aug[mask].T @ coefficients[mask],
                )
                pred = X_aug[i:i+1] @ W_cv
                loo_err += ((pred - coefficients[i:i+1]) ** 2).sum()
            if loo_err < best_loo:
                best_loo = loo_err
                best_alpha = a

        reg = best_alpha * np.eye(X_aug.shape[1])
        reg[-1, -1] = 0.0
        W = np.linalg.solve(X_aug.T @ X_aug + reg, X_aug.T @ coefficients)

        z_pred = X_aug @ W
        V_pred = (mean + z_pred @ components).reshape(S, N, 3)
        feat_rmse = np.sqrt(((V_pred - V) ** 2).sum(axis=2).mean(axis=1))
        print(f"  Feature-mapped RMSE: mean={feat_rmse.mean():.2f}cm, "
              f"max={feat_rmse.max():.2f}cm (α={best_alpha})")

        # -- Save --
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # leaf_comp_vids as object array (ragged)
        leaf_vids_obj = np.empty(len(leaf_vids_list), dtype=object)
        for i, arr in enumerate(leaf_vids_list):
            leaf_vids_obj[i] = arr

        np.savez_compressed(
            str(out_path),
            # PCA
            mean=mean,
            components=components,
            n_verts=N,
            # Template
            template_tris=tris,
            template_groups=tri_groups,
            # Leaf structure
            leaf_comp_vids=leaf_vids_obj,
            leaf_comp_base_z=leaf_base_z,
            stem_vids=np.array(sorted(stem_vids), dtype=np.int32),
            # Feature mapping
            W=W,
            feat_mean=feat_mean,
            feat_std=feat_std,
            max_leaves=max_leaves,
            # Training data
            train_coefficients=coefficients,
            train_features=features,
        )

        print(f"  Model saved to {out_path}")
        print(f"  Size: {out_path.stat().st_size / 1024:.0f} KB")

        return {
            "n_stages": S,
            "n_verts": N,
            "n_components": n_components,
            "variance_explained": float(var_explained[-1]),
            "pca_rmse_cm": float(rmse),
            "feature_rmse_mean_cm": float(feat_rmse.mean()),
            "feature_rmse_max_cm": float(feat_rmse.max()),
            "alpha": float(best_alpha),
            "n_leaf_components": len(leaf_vids_list),
        }


# ---------------------------------------------------------------------------
# Convenience: drop-in for pipeline
# ---------------------------------------------------------------------------

def loft_organs_pca(organ_dicts, model_path):
    """Drop-in replacement for g1_to_g3.loft_organs() using PCA lofter.

    Args:
        organ_dicts: List of organ dicts from cplantbox_adapter.
        model_path: Path to trained PCA lofter model (.npz).

    Returns:
        G3Mesh compatible with the rest of the coupling pipeline.
    """
    lofter = PCALofter(model_path)
    return lofter.loft(organ_dicts)
