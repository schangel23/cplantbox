#!/usr/bin/env python3
"""Standalone V21d Babst-size diagnostic for maize geometry."""

from __future__ import annotations

import inspect
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

import plantbox as pb  # noqa: E402

from dart.coupling.config import DEFAULT_XML  # noqa: E402
from dart.coupling.growth.grow import grow_plant  # noqa: E402
from dart.coupling.scripts.pm_v3_babst_comparison import BABST_MET  # noqa: E402


ROOT = 2
STEM = 3
LEAF = 4
MAINSTEM_SUBTYPES = {0, 1}


def _organ_subtype(organ) -> int | None:
    try:
        return int(organ.getParameter("subType"))
    except Exception:
        return None


def _organ_length(organ) -> float:
    for with_laterals in (False, True):
        try:
            return float(organ.getLength(with_laterals))
        except TypeError:
            continue
    return float(organ.getLength())


def _organ_radius(organ) -> float:
    for name in ("a", "radius"):
        try:
            return float(organ.getParameter(name))
        except Exception:
            continue
    return float("nan")


def _segment_node_ids(segment) -> tuple[int, int] | None:
    for a_name, b_name in (("x", "y"), ("i", "j")):
        if hasattr(segment, a_name) and hasattr(segment, b_name):
            return int(getattr(segment, a_name)), int(getattr(segment, b_name))
    try:
        return int(segment[0]), int(segment[1])
    except Exception:
        return None


def _mainstem_segment_indices(plant) -> list[int]:
    organ_types = [int(v) for v in getattr(plant, "organTypes", [])]
    sub_types = [int(v) for v in getattr(plant, "subTypes", [])]
    return [
        i for i, ot in enumerate(organ_types)
        if ot == STEM and i < len(sub_types) and sub_types[i] in MAINSTEM_SUBTYPES
    ]


def _mainstem_height_cm(plant) -> float:
    lengths = [float(v) for v in plant.segLength()]
    indices = _mainstem_segment_indices(plant)
    if indices:
        return sum(lengths[i] for i in indices if i < len(lengths))

    mainstem_organs = [
        o for o in plant.getOrgans()
        if int(o.organType()) == STEM and _organ_subtype(o) in MAINSTEM_SUBTYPES
    ]
    return sum(_organ_length(o) for o in mainstem_organs)


def _basal_mainstem_internode(plant) -> tuple[float, float]:
    nodes = plant.getNodes()
    segments = plant.getSegments()
    lengths = [float(v) for v in plant.segLength()]
    radii = [float(v) for v in getattr(plant, "radii", [])]
    candidates = []

    for i in _mainstem_segment_indices(plant):
        if i >= len(segments) or i >= len(lengths):
            continue
        node_ids = _segment_node_ids(segments[i])
        if node_ids is None:
            continue
        a, b = node_ids
        if a >= len(nodes) or b >= len(nodes):
            continue
        z_mid = 0.5 * (float(nodes[a].z) + float(nodes[b].z))
        if z_mid >= -1e-6:
            radius = radii[i] if i < len(radii) else float("nan")
            candidates.append((z_mid, lengths[i], radius))

    if candidates:
        _, length_cm, radius_cm = min(candidates, key=lambda item: item[0])
        return length_cm, radius_cm

    mainstem_organs = [
        o for o in plant.getOrgans()
        if int(o.organType()) == STEM and _organ_subtype(o) in MAINSTEM_SUBTYPES
    ]
    if not mainstem_organs:
        return 0.0, float("nan")
    basal = min(
        mainstem_organs,
        key=lambda o: min((float(n.z) for n in o.getNodes()), default=float("inf")),
    )
    return _organ_length(basal), _organ_radius(basal)


def _manual_visible_leaf_count(plant) -> int:
    collared = 0
    for organ in plant.getOrgans(pb.leaf):
        nodes = organ.getNodes()
        if len(nodes) < 2:
            continue
        try:
            lmax = max(float(organ.getLeafRandomParameter().lmax), 1e-9)
        except Exception:
            continue
        cur = 0.0
        prev = nodes[0]
        for node in nodes[1:]:
            dx = float(node.x) - float(prev.x)
            dy = float(node.y) - float(prev.y)
            dz = float(node.z) - float(prev.z)
            cur += math.sqrt(dx * dx + dy * dy + dz * dz)
            prev = node
        if min(cur / lmax, 1.0) >= 0.45:
            collared += 1
    return collared


def _leaf_count(plant) -> int:
    try:
        from dart.coupling.growth.phenology import count_visible_leaves

        counts = count_visible_leaves(plant)
        return int(counts["collared"])
    except Exception:
        return _manual_visible_leaf_count(plant)


def _grow_kwargs() -> dict:
    kwargs = {
        "xml_path": str(DEFAULT_XML),
        "simulation_time": 21,
        "min_stem_nodes": 10,
        "min_leaf_nodes": 4,
        "enable_photosynthesis": False,
        "seed": 42,
        "daily_met": BABST_MET,
        "T_air_default": 20.75,
    }
    if "use_fa" in inspect.signature(grow_plant).parameters:
        kwargs["use_fa"] = True
    return kwargs


def main() -> None:
    plant = grow_plant(**_grow_kwargs())
    organs = plant.getOrgans()

    enum_values = (
        int(pb.OrganTypes.root),
        int(pb.OrganTypes.stem),
        int(pb.OrganTypes.leaf),
    )
    if enum_values != (ROOT, STEM, LEAF):
        raise RuntimeError(f"Unexpected CPlantBox organType enum values: {enum_values}")
    first_organ_type = int(organs[0].organType()) if organs else -1

    n_leaves = sum(1 for o in organs if int(o.organType()) == LEAF)
    n_stems = sum(1 for o in organs if int(o.organType()) == STEM)
    n_roots = sum(1 for o in organs if int(o.organType()) == ROOT)
    height_cm = _mainstem_height_cm(plant)
    leaf_count = _leaf_count(plant)
    basal_length_cm, basal_radius_cm = _basal_mainstem_internode(plant)
    root_length_cm = sum(_organ_length(o) for o in organs if int(o.organType()) == ROOT)

    print("\nPlant geometry table")
    print(f"Enum check — first organType={first_organ_type}; root=2 stem=3 leaf=4")
    print(f"Mainstem height [cm] — {height_cm:.2f}")
    print(f"Leaf count — {leaf_count}")
    print(f"Total node count — {len(plant.getNodes())}")
    print(
        "Total organ count — "
        f"{len(organs)} (leaves / stems / roots: {n_leaves} / {n_stems} / {n_roots})"
    )
    print(
        "Basal stem internode — "
        f"length [cm]: {basal_length_cm:.2f}; radius [cm]: {basal_radius_cm:.4f}"
    )
    print(f"Total root system length [cm] — {root_length_cm:.2f}")

    if 15 <= height_cm <= 100 and 2 <= leaf_count <= 8:
        print("SIZE: roughly Babst-compatible (verdict a)")
    elif height_cm > 100 or leaf_count > 10:
        print("SIZE: much larger than Babst V3 (verdict b) — L mismatch likely explains v shortfall")
    else:
        print("SIZE: smaller or atypical vs Babst V3 (verdict c) — check growth settings")


if __name__ == "__main__":
    main()
