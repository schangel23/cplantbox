"""Shared DART output parsers and grid utilities.

Functions extracted from simulation.py and baleno.py to avoid duplication.
"""

import json
import re
import numpy as np
from pathlib import Path


# ---------------------------------------------------------------------------
# Grid constants (5×3 plant field — 5 along row, 3 rows)
# ---------------------------------------------------------------------------
GRID_NX, GRID_NY = 3, 5         # 3 rows (x) × 5 along-row (y)
GRID_SPACING_X = 0.75            # meters (between-row spacing, typical maize)
GRID_SPACING_Y = 0.15            # meters (within-row spacing, ~89k pl/ha)
SCENE_SIZE = [4.0, 2.25]         # meters (≥0.75 m border each side)
PLANT_POS = (SCENE_SIZE[0] / 2, SCENE_SIZE[1] / 2)  # scene center


def compute_plant_positions(seed=42):
    """Compute grid positions with per-plant random azimuthal rotation.

    Returns list of (x, y, yrot) tuples. yrot is uniform 0-360° —
    maize has no preferred heading relative to row direction.
    """
    import numpy as np
    rng = np.random.default_rng(seed)
    positions = []
    for iy in range(GRID_NY):
        for ix in range(GRID_NX):
            x = PLANT_POS[0] + (ix - (GRID_NX - 1) / 2) * GRID_SPACING_X
            y = PLANT_POS[1] + (iy - (GRID_NY - 1) / 2) * GRID_SPACING_Y
            yrot = float(rng.uniform(0, 360))
            positions.append((x, y, yrot))
    return positions


# ---------------------------------------------------------------------------
# DART radiative budget parsers
# ---------------------------------------------------------------------------

def parse_radiative_budget_txt(filepath):
    """Parse RadiativeBudgetFigures.txt into per-object arrays.

    Format:
      Header line (tab-separated column names)
      ObjectName object0_0_0
      <triangle data rows>
      ObjectName object1_0_0
      <triangle data rows>
      ...

    Returns dict with 'header' and 'per_object' (object_key -> numpy array).
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()

    if not lines:
        return None

    # Parse header
    header_line = lines[0].strip('* \n\t')
    header = re.split(r'[\t;]+', header_line)
    header = [h.strip() for h in header if h.strip()]

    per_object = {}
    current_key = None

    for line in lines[1:]:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith('ObjectName'):
            current_key = stripped
            per_object[current_key] = []
        elif current_key is not None:
            # Parse data row
            values = re.split(r'[\t ]+', stripped)
            try:
                row = [float(v) for v in values if v]
                per_object[current_key].append(row)
            except ValueError:
                continue  # skip malformed lines

    # Convert to numpy
    for key in per_object:
        if per_object[key]:
            per_object[key] = np.array(per_object[key])
        else:
            per_object[key] = np.empty((0, len(header)))

    return {'header': header, 'per_object': per_object}


def parse_maket_scn(simu_path):
    """Parse maket.scn to map budget object IDs to (instance, group) pairs.

    DART ObjectFields assigns each field instance + group a unique budget
    object.  The maket.scn file contains lines like::

        scene.objects.objectN_0_0.dartNameId=fo0_moX_goY

    where *N* is the budget object index, *X* is the field instance index,
    and *Y* is the group index (alphabetically ordered, matching .ori
    file numbering).

    Returns:
        dict mapping ``{budget_obj_idx: (instance_idx, group_idx)}``,
        or ``None`` if maket.scn is missing or contains no ObjectField entries.
    """
    maket_path = Path(simu_path) / 'output' / 'maket.scn'
    if not maket_path.exists():
        return None

    mapping = {}
    with open(maket_path) as f:
        for line in f:
            if 'dartNameId=fo' not in line:
                continue
            m = re.match(
                r'scene\.objects\.object(\d+)_0_0\.dartNameId='
                r'fo\d+_mo(\d+)_go(\d+)',
                line.strip(),
            )
            if m:
                budget_idx = int(m.group(1))
                instance = int(m.group(2))
                group = int(m.group(3))
                mapping[budget_idx] = (instance, group)

    return mapping if mapping else None


# ---------------------------------------------------------------------------
# Baleno CSV utilities
# ---------------------------------------------------------------------------

def detect_delimiter(filepath):
    """Detect CSV delimiter from header line."""
    with open(filepath) as f:
        header = f.readline().strip()
    if ';' in header:
        return ';'
    if '\t' in header:
        return '\t'
    return ','


def read_baleno_csv(filepath, delimiter=';'):
    """Read a Baleno output CSV with header."""
    with open(filepath) as f:
        header_line = f.readline().strip()
    header = [h.strip() for h in header_line.split(delimiter)]

    # Try to read as numeric, falling back to string for scene columns
    try:
        data = np.genfromtxt(
            str(filepath), skip_header=1, delimiter=delimiter,
            dtype=float, filling_values=np.nan,
        )
    except ValueError:
        # Scene file may have string columns (DART_NAME)
        data = np.genfromtxt(
            str(filepath), skip_header=1, delimiter=delimiter,
            dtype=str,
        )

    return header, data


def write_json5(path, data):
    """Write a JSON5-compatible file (JSON with comments support)."""
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"    Created: {path.name}")
