#!/usr/bin/env python3
"""
S0.7 — Bake runtime calibration into ``maize_calibrated.xml``.

ADR_LEAF_KINEMATICS_2026-04-28 §S0.7 (Lock #3 Half A): the runtime contract
becomes pure XML + C++ — ``pb.MappedPlant("maize_calibrated.xml").initialize()
.simulate(130)`` reproduces today's ``grow_plant(...)`` output without any
Python configuration step between ``readParameters`` and ``initialize``.

What this script does:

  1. Loads the master XML.
  2. Calls ``setup_successor_where(plant)`` — bakes per-position successorST/OT/P/No/Where
     plus ``RotBeta=π`` and ``BetaDev=0.22`` on stem subType=1.
  3. Calls ``enable_fa_on_mainstem(plant)`` — bakes
     ``use_fournier_andrieu_kinetics=1``, ``internode_v_n``, ``internode_D_n``,
     ``internode_IL_final`` (with basal-zero overrides) from
     ``data/phase_III_per_rank.json``.
  4. Calls ``enable_fa_on_leaves(plant)`` — bakes ``use_fa_kinetics=1``,
     ``tau_extension_n`` and ``sigma_extension_n`` per leaf LRP. (Lock #3 Half B
     deferred: the formulas ``tau = tt_em + 2·phyll`` and ``sigma = phyll`` will
     move into ``MultiPhaseLeafGrowth`` C++ in S2 — until then the bake script
     evaluates them, but only at bake time, not at runtime.)
  5. Dumps the mutated state via ``plant.writeParameters(out_xml)``.

The bake is **idempotent**: the three helpers all overwrite (rather than
append) target fields, so re-baking the already-baked XML produces an
identical file. This is the property that lets us use the same path as both
input and output.

Usage:

    python -m dart.coupling.scripts.bake_calibration_to_xml \
        [--input dart/coupling/data/maize_calibrated.xml] \
        [--output dart/coupling/data/maize_calibrated.xml]

CLI defaults are the canonical maize XML in either direction (in-place bake).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import plantbox as pb

from ..config import DEFAULT_XML
from ..growth.grow import (
    enable_fa_on_leaves,
    enable_fa_on_mainstem,
    setup_successor_where,
)


DEFAULT_INPUT_XML = Path(DEFAULT_XML)
DEFAULT_OUTPUT_XML = Path(DEFAULT_XML)


def bake(input_xml: Path, output_xml: Path, *, verbose: bool = True) -> None:
    """Bake runtime Python configuration into ``output_xml``.

    Args:
        input_xml: master XML to read (typically ``maize_calibrated.xml``).
        output_xml: target XML to write (defaults to ``input_xml`` for
            in-place bake; safe because the helpers below are idempotent).
        verbose: forwarded to the helpers for one-line summaries.
    """
    if verbose:
        print(f"=== S0.7 bake: {input_xml} → {output_xml} ===")

    plant = pb.MappedPlant()
    plant.readParameters(str(input_xml))

    # Order matches grow_plant() pre-init contract.
    setup_successor_where(plant)
    enable_fa_on_mainstem(plant, verbose=verbose)
    enable_fa_on_leaves(plant, verbose=verbose)

    output_xml.parent.mkdir(parents=True, exist_ok=True)
    plant.writeParameters(str(output_xml))

    if verbose:
        srp = plant.getOrganRandomParameter(pb.OrganTypes.stem, 1)
        n_v = len(list(getattr(srp, "internode_v_n", [])))
        fa_on = int(getattr(srp, "use_fournier_andrieu_kinetics", 0))
        n_leaf_fa = sum(
            1 for lrp in plant.getOrganRandomParameter(pb.OrganTypes.leaf)
            if int(getattr(lrp, "use_fa_kinetics", 0)) == 1
        )
        print(f"  baked: stem FA={fa_on} (v_n len={n_v}); leaf FA on {n_leaf_fa} LRPs")
        print(f"  wrote {output_xml}")


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="bake_calibration_to_xml",
        description=(
            "S0.7: bake setup_successor_where + enable_fa_on_mainstem + "
            "enable_fa_on_leaves into the maize XML so the runtime contract "
            "becomes pure XML + C++ (D6 closure)."
        ),
    )
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT_XML,
                   help=f"master XML to read (default: {DEFAULT_INPUT_XML})")
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_XML,
                   help=f"target XML to write (default: {DEFAULT_OUTPUT_XML}; in-place)")
    p.add_argument("--quiet", action="store_true",
                   help="suppress per-step prints")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    bake(args.input, args.output, verbose=not args.quiet)
    return 0


if __name__ == "__main__":
    sys.exit(main())
