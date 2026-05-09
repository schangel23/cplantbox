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


def convert_obj_to_dart(input_path, output_path, scale=0.01,
                        zero_pad_groups=True, xy_offset_cm=(0.0, 0.0)):
    """Convert standard OBJ to DART coordinate convention.

    Transformations applied:
      1. Subtract ``xy_offset_cm`` from each vertex's (x, y) (z untouched).
      2. Vertex swizzle: v x y z  ->  v y z x
      3. Normal swizzle: vn x y z -> vn y z x
      4. Scale all coordinates by `scale` (default: 0.01 for cm->m)
      5. Zero-pad group names: organ_0 -> organ_00

    Face definitions (f), texture coordinates (vt), and comments are preserved.

    Args:
        input_path: Path to input OBJ file (standard v x y z).
        output_path: Path to output DART OBJ file.
        scale: Coordinate scale factor (default 0.01 for cm to meters).
        zero_pad_groups: If True, zero-pad group names for alphabetical ordering.
        xy_offset_cm: ``(ox, oy)`` translation in **cm** subtracted from
            each vertex before scale+swizzle. Z is left untouched (root
            depth below ground / shoot above is the DART expectation).
            Set to ``(seedPos.x, seedPos.y)`` from the plant to put the
            plant at the scene origin in DART meters; the field-position
            file (``plant_field.txt``) then provides the per-plant
            placement on top. Default ``(0.0, 0.0)`` is bit-identical
            with pre-2026-05-09 behaviour (suitable for XMLs whose
            seedPos is at the origin).

            Why this exists: Phase 3.5 (2026-05-07) shifted
            ``maize_calibrated.xml``'s seedPos from ``(0, 0, -3)`` to
            ``(200, 200, -3)`` to satisfy DuMux ``setRectangularGrid``
            mapping. Without an offset subtraction here, the plant
            geometry lands at scene-coord ``(2.0, 2.0) m`` *before*
            ``plant_field.txt`` adds the per-plant grid offset, putting
            every plant out of the 4×2.25 m DART scene bounds; DART
            maket discards them all and writes no ``.ori`` reindex
            tables, breaking the entire downstream photosynthesis loop.

    Returns:
        dict with conversion stats: n_vertices, n_normals, n_faces, n_groups, groups
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    stats = {'n_vertices': 0, 'n_normals': 0, 'n_faces': 0, 'n_groups': 0, 'groups': []}
    ox_cm, oy_cm = float(xy_offset_cm[0]), float(xy_offset_cm[1])

    # Pattern for group names like organ_N, tassel_spike_N, tassel_branch_N.
    # Also handles plant-prefixed names like p0_organ_0, p1_tassel_spike_15.
    group_pat = re.compile(
        r'^((?:p\d+_)?(?:organ|tassel_spike|tassel_branch)_)(\d+)$'
    )

    lines_out = []
    with open(input_path, 'r') as f:
        for line in f:
            stripped = line.strip()

            if stripped.startswith('v ') and not stripped.startswith('vt') and not stripped.startswith('vn'):
                # Vertex: v x y z -> v y z x (with optional XY offset
                # subtraction in cm, then scale).
                parts = stripped.split()
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                x_cm = x - ox_cm
                y_cm = y - oy_cm
                lines_out.append(
                    f"v {y_cm*scale:.6f} {z*scale:.6f} {x_cm*scale:.6f}\n")
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
        group_pat = re.compile(
            r'^((?:p\d+_)?(?:organ|tassel_spike|tassel_branch)_)(\d+)$'
        )
        for organ in mapping.get('organs', []):
            name = organ.get('name', '')
            m = group_pat.match(name)
            if m:
                organ['name'] = f"{m.group(1)}{int(m.group(2)):02d}"

    out = output_path or mapping_json_path
    with open(out, 'w') as f:
        json.dump(mapping, f, indent=2)

    return mapping
