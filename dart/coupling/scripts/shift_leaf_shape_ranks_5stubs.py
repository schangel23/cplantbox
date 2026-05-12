"""Shift per-rank fields in maize_leaf_shape_distribution.json so the MF3D
bell peak (currently at JSON rank 6 / XML slot 6 / z=13 cm) lands at the
ear-bearing phytomer (XML slot 9 / z=91 cm).

Naming reconciliation:
  - The user's "5 stubs at the bottom" refers to NPZ-side stub count.
  - Current NPZ already has 2 stubs of MF3D pos 0 prepended, so the JSON's
    rank 0 corresponds to that stub. To go from 2 → 5 NPZ stubs the
    JSON-side shift is +3 (= 5 − 2).
  - JSON rank K's intercept becomes new rank K + 3.

Per-rank fields shifted:
  - intercepts (33-vec spline coeffs per rank)
  - asym_residual_grids_cm (11, 5, 3) per rank
  - lmax_intercept_cm, max_w_xml_cm, lmax_xml_cm
  - leaf_names
  - donors_per_position (informational only)

Unchanged: covariance, cholesky_factor, pca_truncation, schema metadata.

Rank mapping:
  new_rank K -> source rank max(0, K - 3)

  K = 0..2  -> rank 0 (3 stub slots at the bottom — adds to the 2 NPZ-side
                       stubs already in rank 0, total 5)
  K = 3     -> rank 0 (was at slot 3 already in current NPZ)
  K = 4     -> rank 1 (was at slot 4)
  ...
  K = 9     -> rank 6 (MF3D pos 4, the 78.83 cm peak — NOW AT SLOT 9 / z=91 cm)
  ...
  K = 14    -> rank 11 (was at slot 14 indirectly: old slot 11 had MF3D pos 9)

Old ranks 12..14 are DROPPED. Their donor coverage was the sparse tail
(62 / 27 / unspecified). The new top-of-canopy gets rank 11 (136 donors,
better data than the old top-of-canopy values).
"""
from __future__ import annotations
import json
import shutil
from pathlib import Path

JSON_PATH = Path("/home/lukas/PHD/CPlantBox/dart/coupling/data/maize_leaf_shape_distribution.json")
SHIFT = 3
N_RANKS = 15


def shift_dict(d: dict, n_ranks: int = N_RANKS, shift: int = SHIFT) -> dict:
    """new_rank[K] = old_rank[max(0, K - shift)]"""
    out = {}
    for K in range(n_ranks):
        src = max(0, K - shift)
        key_src = str(src)
        if key_src not in d:
            # informational fields (donors_per_position) may not cover all ranks
            continue
        out[str(K)] = d[key_src]
    return out


def main():
    backup = JSON_PATH.with_suffix(".json.bak_2026-05-12_pre-5stub")
    if not backup.exists():
        shutil.copy2(JSON_PATH, backup)
        print(f"backed up: {backup}")

    with open(JSON_PATH) as f:
        d = json.load(f)

    fields_to_shift = [
        "intercepts",
        "asym_residual_grids_cm",
        "lmax_intercept_cm",
        "max_w_xml_cm",
        "lmax_xml_cm",
        "leaf_names",
    ]

    print("\nBefore shift:")
    print(f"  lmax_intercept_cm peak = rank {max(d['lmax_intercept_cm'], key=lambda k: d['lmax_intercept_cm'][k])} "
          f"({d['lmax_intercept_cm'][max(d['lmax_intercept_cm'], key=lambda k: d['lmax_intercept_cm'][k])]:.2f} cm)")

    for field in fields_to_shift:
        if field not in d:
            print(f"  WARN: {field} not in JSON")
            continue
        d[field] = shift_dict(d[field])
        print(f"  shifted: {field}")

    # donors_per_position has only 14 entries (0..13); shift carefully
    if "donors_per_position" in d:
        d["donors_per_position"] = shift_dict(d["donors_per_position"])
        print(f"  shifted: donors_per_position (informational)")

    # Update leaf_names to reflect new rank labels — keep "maize_leaf_L{K}"
    # consistent with the XML name (XML names already match by slot index).
    if "leaf_names" in d:
        d["leaf_names"] = {str(K): f"maize_leaf_L{K}" for K in range(N_RANKS)}

    # Record the shift in the metadata
    d.setdefault("shift_history", []).append({
        "date": "2026-05-12",
        "shift": SHIFT,
        "reason": "JSON +3 shift (NPZ-side: 5 stubs at bottom). MF3D bell "
                  "peak (rank 6, 78.83 cm) moves from XML slot 6 (z=13 cm, "
                  "basal) to XML slot 9 (z=91 cm, ear-bearing phytomer). "
                  "Old ranks 12..14 dropped (MF3D sparse tail, 62/27/-- "
                  "donors). See DIAG_MAIZE_LEAF_BOTTOM_HEAVY_2026-05-12.",
    })

    print("\nAfter shift:")
    peak_K = max(d["lmax_intercept_cm"], key=lambda k: d["lmax_intercept_cm"][k])
    print(f"  lmax_intercept_cm peak = rank {peak_K} "
          f"({d['lmax_intercept_cm'][peak_K]:.2f} cm)")
    print(f"  full bell:")
    for K in range(N_RANKS):
        print(f"    rank {K:>2}: lmax={d['lmax_intercept_cm'][str(K)]:6.2f} cm,"
              f" max_w={d['max_w_xml_cm'][str(K)]:5.2f} cm")

    with open(JSON_PATH, "w") as f:
        json.dump(d, f, indent=2)
    print(f"\nWrote {JSON_PATH}")


if __name__ == "__main__":
    main()
