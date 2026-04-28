#!/usr/bin/env python3
"""Capture pre-S0 cross-species baselines for the leaf-kinematics ADR (S0.8).

Plan: ADR_LEAF_KINEMATICS_2026-04-28.md §S0.8 + §D8 (non-maize regression as
a hard validation gate).

Loads every XML in gui/cplantbox/params/, simulates 30 days deterministically
(seed=42, no met forcing, no setup_successor_where — raw native CPlantBox
semantics), and records a topology fingerprint per XML. After S0 ships
the MultiPhaseStemGrowth refactor, re-run with --verify; every native-XML
case must remain bit-for-bit identical (CLAUDE.md principle 6).

The fingerprint follows the d0 convention: sha256 over packed xyz bytes,
segregated by organ type. Topology counts (n_organs, n_stems, n_leaves,
n_roots, n_segments_total) are surfaced for Lock #7 (topology-exact gates
alongside cm-tolerance gates).

Pre-S0 contract: native (non-FA) growth-function dispatch only. None of these
XMLs sets use_fournier_andrieu_kinetics; all use gft_negexp by default.
After S0 the gf_dispatch path swap must leave them bit-identical.

Usage (from /home/lukas/PHD/CPlantBox):
    cpbenv/bin/python3 dart/coupling/tests/baselines/capture_cross_species_baseline.py
    cpbenv/bin/python3 dart/coupling/tests/baselines/capture_cross_species_baseline.py --verify
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path

BASELINE_DIR = Path(__file__).resolve().parent
COUPLING_DIR = BASELINE_DIR.parent.parent             # .../CPlantBox/dart/coupling
CPLANTBOX_ROOT = COUPLING_DIR.parent.parent           # .../CPlantBox
sys.path.insert(0, str(CPLANTBOX_ROOT))

import plantbox as pb  # noqa: E402

SEED = 42
SIM_DAYS = 30
PARAM_DIR = CPLANTBOX_ROOT / "gui" / "cplantbox" / "params"
OUTPUT_JSON = BASELINE_DIR / "cross_species_baseline_pre_s0.json"


@dataclass(frozen=True)
class Case:
    slug: str
    xml: Path


def discover_cases() -> list[Case]:
    cases = []
    for xml_path in sorted(PARAM_DIR.glob("*.xml")):
        slug = xml_path.stem
        cases.append(Case(slug=slug, xml=xml_path))
    return cases


def _pack_xyz(nodes) -> bytes:
    buf = bytearray()
    for n in nodes:
        buf += struct.pack("<ddd", float(n.x), float(n.y), float(n.z))
    return bytes(buf)


def grow_deterministic(xml_path: Path, sim_days: int, seed: int):
    """Pure native-CPlantBox path: ctor seed → readParameters → initialize → simulate.

    No setup_successor_where, no met forcing, no FA opt-in. This is what an
    external user calling `pb.MappedPlant(seed).readParameters(...).simulate(...)`
    actually gets, and is the canonical surface S0 must preserve."""
    plant = pb.MappedPlant(seed)
    plant.readParameters(str(xml_path))
    plant.setSeed(seed)  # defensive; setSeed bug only affects post-init RNG draws
    plant.initialize()
    plant.simulate(float(sim_days), False)
    return plant


def capture_signature(plant, case: Case) -> dict:
    organs = plant.getOrgans()
    stems = [o for o in organs if o.organType() == pb.OrganTypes.stem]
    leaves = [o for o in organs if o.organType() == pb.OrganTypes.leaf]
    roots = [o for o in organs if o.organType() == pb.OrganTypes.root]

    stem_bytes = b""
    leaf_bytes = b""
    root_bytes = b""
    n_stem_nodes = 0
    n_leaf_nodes = 0
    n_root_nodes = 0
    n_segments_total = 0

    for s in sorted(stems, key=lambda o: o.getId()):
        nodes = list(s.getNodes())
        stem_bytes += _pack_xyz(nodes)
        n_stem_nodes += len(nodes)
    for lf in sorted(leaves, key=lambda o: o.getId()):
        nodes = list(lf.getNodes())
        leaf_bytes += _pack_xyz(nodes)
        n_leaf_nodes += len(nodes)
    for r in sorted(roots, key=lambda o: o.getId()):
        nodes = list(r.getNodes())
        root_bytes += _pack_xyz(nodes)
        n_root_nodes += len(nodes)
    for o in organs:
        n_segments_total += max(0, len(o.getNodes()) - 1)

    h = hashlib.sha256()
    h.update(b"stems:")
    h.update(stem_bytes)
    h.update(b"|leaves:")
    h.update(leaf_bytes)
    h.update(b"|roots:")
    h.update(root_bytes)
    h.update(b"|nseg:")
    h.update(struct.pack("<q", n_segments_total))
    return {
        "slug": case.slug,
        "xml": str(case.xml.relative_to(CPLANTBOX_ROOT)),
        "seed": SEED,
        "sim_days": SIM_DAYS,
        "n_organs": len(organs),
        "n_stems": len(stems),
        "n_leaves": len(leaves),
        "n_roots": len(roots),
        "n_stem_nodes": n_stem_nodes,
        "n_leaf_nodes": n_leaf_nodes,
        "n_root_nodes": n_root_nodes,
        "n_segments_total": n_segments_total,
        "sha256": h.hexdigest(),
    }


def run_case(case: Case) -> dict:
    print(f"[{case.slug}]  xml={case.xml.name}  days={SIM_DAYS}", flush=True)
    try:
        plant = grow_deterministic(case.xml, SIM_DAYS, SEED)
    except Exception as e:
        print(f"  GROW FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()
        return {
            "slug": case.slug,
            "xml": str(case.xml.relative_to(CPLANTBOX_ROOT)),
            "seed": SEED,
            "sim_days": SIM_DAYS,
            "error": f"{type(e).__name__}: {e}",
        }
    sig = capture_signature(plant, case)
    print(f"  sha256={sig['sha256']}  organs={sig['n_organs']}  "
          f"stems={sig['n_stems']} leaves={sig['n_leaves']} roots={sig['n_roots']}  "
          f"nseg={sig['n_segments_total']}")
    return sig


def capture_mode():
    cases = discover_cases()
    print(f"Capturing pre-S0 baseline: {len(cases)} XMLs, seed={SEED}, days={SIM_DAYS}\n")
    payload = {
        "comment": (
            "Pre-S0 cross-species baseline for ADR_LEAF_KINEMATICS_2026-04-28 §S0.8. "
            "Native CPlantBox path only (no FA, no setup_successor_where, no met "
            "forcing). After S0 (MultiPhaseStemGrowth refactor) ships, re-run with "
            "--verify; every entry must remain bit-for-bit identical."
        ),
        "seed": SEED,
        "sim_days": SIM_DAYS,
        "n_cases": len(cases),
        "cases": {},
    }
    for case in cases:
        sig = run_case(case)
        payload["cases"][case.slug] = sig
        print()
    with OUTPUT_JSON.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"Baseline written to {OUTPUT_JSON.relative_to(CPLANTBOX_ROOT)}")


def verify_mode() -> int:
    if not OUTPUT_JSON.exists():
        print(f"BASELINE MISSING: {OUTPUT_JSON}")
        return 1
    with OUTPUT_JSON.open() as f:
        baseline = json.load(f)
    cases = discover_cases()
    failures = []
    new_xmls = []
    for case in cases:
        if case.slug not in baseline["cases"]:
            new_xmls.append(case.slug)
            continue
        sig = run_case(case)
        ref = baseline["cases"][case.slug]
        if "error" in ref or "error" in sig:
            if ref.get("error") != sig.get("error"):
                print(f"  ERROR-DIFF: baseline={ref.get('error')!r} now={sig.get('error')!r}")
                failures.append(case.slug)
            else:
                print(f"  OK (both errored identically)")
            continue
        if sig["sha256"] != ref["sha256"]:
            print(f"  DIFF: expected {ref['sha256']} got {sig['sha256']}")
            for k in ("n_organs", "n_stems", "n_leaves", "n_roots",
                      "n_stem_nodes", "n_leaf_nodes", "n_root_nodes",
                      "n_segments_total"):
                if sig.get(k) != ref.get(k):
                    print(f"    {k}: baseline={ref.get(k)} now={sig.get(k)}")
            failures.append(case.slug)
        else:
            print(f"  OK (matches baseline)")
        print()
    if new_xmls:
        print(f"NOTE: {len(new_xmls)} XML(s) added since baseline (not validated): {new_xmls}")
    if failures:
        print(f"\nFAILED: {len(failures)} case(s) diverged from baseline: {failures}")
        return 1
    print(f"\nPASSED: all {len(cases) - len(new_xmls)} case(s) bit-for-bit identical.")
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true",
                        help="re-run all cases and compare against stored baseline")
    args = parser.parse_args()
    if args.verify:
        sys.exit(verify_mode())
    else:
        capture_mode()


if __name__ == "__main__":
    main()
