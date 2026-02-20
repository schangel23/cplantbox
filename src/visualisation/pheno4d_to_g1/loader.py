"""Load Pheno4D .txt point clouds and separate by organ label."""

import numpy as np


LABEL_NAMES = {0: 'soil', 1: 'stem', 2: 'leaf_1', 3: 'leaf_2', 4: 'leaf_3',
               5: 'leaf_4', 6: 'leaf_5', 7: 'leaf_6', 8: 'leaf_7'}


def load_unlabeled(filepath, soil_margin_cm=0.5):
    """Load a point cloud file ignoring any label columns.

    Reads only XYZ coordinates, filters soil by Z threshold, centers on
    the lowest plant point, and converts mm to cm.

    Args:
        filepath: path to .txt file (at least 3 columns: X Y Z ...)
        soil_margin_cm: points below min(Z) + margin (in cm) are removed as soil

    Returns:
        np.array([N, 3]) plant points in cm, centered on base
    """
    data = np.loadtxt(filepath)
    coords = data[:, :3].copy()  # mm

    # Convert mm -> cm first for threshold comparison
    coords /= 10.0

    # Remove soil: points below min(Z) + margin
    z_min = coords[:, 2].min()
    plant_mask = coords[:, 2] >= z_min + soil_margin_cm
    coords = coords[plant_mask]

    if len(coords) == 0:
        raise ValueError(f"No plant points after soil removal in {filepath}")

    # Center on lowest remaining point (approximate stem base)
    base_idx = np.argmin(coords[:, 2])
    base_point = coords[base_idx].copy()
    coords[:, 0] -= base_point[0]
    coords[:, 1] -= base_point[1]
    coords[:, 2] -= base_point[2]

    print(f"[loader] Loaded unlabeled {filepath}: {len(coords):,} plant points")
    return coords


def load_pheno4d(filepath, label_method='collar'):
    """Load a Pheno4D .txt file and separate points by organ.

    Args:
        filepath: path to .txt file (5 columns: X Y Z label_collar label_tip, in mm)
        label_method: 'collar' uses column 3, 'tip' uses column 4

    Returns:
        dict of organ_name -> np.array([N, 3]) in cm, centered on stem base
    """
    data = np.loadtxt(filepath)
    coords = data[:, :3]  # mm

    label_col = 3 if label_method == 'collar' else 4
    labels = data[:, label_col].astype(int)

    # Filter out soil (label 0)
    plant_mask = labels != 0
    coords = coords[plant_mask]
    labels = labels[plant_mask]

    # Find stem base: lowest Z point among stem points
    stem_mask = labels == 1
    if stem_mask.any():
        stem_points = coords[stem_mask]
        base_idx = np.argmin(stem_points[:, 2])
        base_point = stem_points[base_idx].copy()
        # Center XY on stem base, Z on ground level
        coords[:, 0] -= base_point[0]
        coords[:, 1] -= base_point[1]
        coords[:, 2] -= base_point[2]

    # Convert mm -> cm
    coords /= 10.0

    # Separate by organ
    organs = {}
    for label_id in np.unique(labels):
        name = LABEL_NAMES.get(label_id, f'label_{label_id}')
        mask = labels == label_id
        organs[name] = coords[mask]

    n_total = sum(len(v) for v in organs.values())
    print(f"[loader] Loaded {filepath}: {n_total:,} plant points")
    for name, pts in sorted(organs.items()):
        print(f"  {name}: {len(pts):,} points")

    return organs
