"""
OBJ-to-DART coordinate converter.

Converts standard OBJ files (v x y z) to DART convention (v y z x)
with optional cm-to-m scaling and zero-padded group names.

DART coordinate convention (from pytools4dart docs):
  DART considers the object to be in an X-forward, Y-up system.
  Vertex coordinates must be written as: v y z x

CPlantBox coordinate system: X=East, Y=North, Z=Up
DART scene: X=East, Y=North, Z=Up (same horizontal, same vertical)

So the OBJ mapping is: v CPlantBox_Y CPlantBox_Z CPlantBox_X
"""

import re
from pathlib import Path


def convert_obj_to_dart(input_path, output_path, scale=0.01, zero_pad_groups=True):
    """Convert standard OBJ to DART coordinate convention.

    Transformations applied:
      1. Vertex swizzle: v x y z  ->  v y z x
      2. Normal swizzle: vn x y z -> vn y z x
      3. Scale all coordinates by `scale` (default: 0.01 for cm->m)
      4. Zero-pad group names: organ_0 -> organ_00

    Face definitions (f), texture coordinates (vt), and comments are preserved.

    Args:
        input_path: Path to input OBJ file (standard v x y z).
        output_path: Path to output DART OBJ file.
        scale: Coordinate scale factor (default 0.01 for cm to meters).
        zero_pad_groups: If True, zero-pad group names for alphabetical ordering.

    Returns:
        dict with conversion stats: n_vertices, n_normals, n_faces, n_groups, groups
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    stats = {'n_vertices': 0, 'n_normals': 0, 'n_faces': 0, 'n_groups': 0, 'groups': []}

    # Pattern for group names like organ_0, organ_1, ..., organ_11
    # Also handles plant-prefixed names like p0_organ_0, p1_organ_11
    group_pat = re.compile(r'^((?:p\d+_)?organ_)(\d+)$')

    lines_out = []
    with open(input_path, 'r') as f:
        for line in f:
            stripped = line.strip()

            if stripped.startswith('v ') and not stripped.startswith('vt') and not stripped.startswith('vn'):
                # Vertex: v x y z -> v y z x (with scaling)
                parts = stripped.split()
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                # Apply scale and swizzle
                lines_out.append(f"v {y*scale:.6f} {z*scale:.6f} {x*scale:.6f}\n")
                stats['n_vertices'] += 1

            elif stripped.startswith('vn '):
                # Normal: vn nx ny nz -> vn ny nz nx (no scaling)
                parts = stripped.split()
                nx, ny, nz = float(parts[1]), float(parts[2]), float(parts[3])
                lines_out.append(f"vn {ny:.6f} {nz:.6f} {nx:.6f}\n")
                stats['n_normals'] += 1

            elif stripped.startswith('f '):
                lines_out.append(line)
                stats['n_faces'] += 1

            elif stripped.startswith('g '):
                group_name = stripped[2:].strip()
                if zero_pad_groups:
                    m = group_pat.match(group_name)
                    if m:
                        group_name = f"{m.group(1)}{int(m.group(2)):02d}"
                lines_out.append(f"g {group_name}\n")
                stats['n_groups'] += 1
                stats['groups'].append(group_name)

            else:
                # Pass through comments, vt, mtllib, usemtl, etc.
                lines_out.append(line)

    with open(output_path, 'w') as f:
        f.writelines(lines_out)

    return stats


def convert_mapping_json_groups(mapping_json_path, output_path=None, zero_pad=True):
    """Update organ names in mapping JSON to match zero-padded OBJ group names.

    Args:
        mapping_json_path: Path to original mapping JSON.
        output_path: Path to write updated JSON. If None, overwrites input.
        zero_pad: If True, zero-pad organ names.

    Returns:
        Updated mapping dict.
    """
    import json

    with open(mapping_json_path, 'r') as f:
        mapping = json.load(f)

    if zero_pad:
        group_pat = re.compile(r'^((?:p\d+_)?organ_)(\d+)$')
        for organ in mapping.get('organs', []):
            name = organ.get('name', '')
            m = group_pat.match(name)
            if m:
                organ['name'] = f"{m.group(1)}{int(m.group(2)):02d}"

    out = output_path or mapping_json_path
    with open(out, 'w') as f:
        json.dump(mapping, f, indent=2)

    return mapping
