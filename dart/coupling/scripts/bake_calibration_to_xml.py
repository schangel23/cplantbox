#!/usr/bin/env python3
"""
S0.7 + S2.D — Bake runtime calibration into ``maize_calibrated.xml``.

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
  4. Calls ``enable_fa_on_leaves(plant)`` — legacy logistic helper (writes
     ``use_fa_kinetics=1``, ``tau_extension_n`` and ``sigma_extension_n``
     per leaf LRP).  Post-S2.C the C++ no longer reads these fields, but the
     bake keeps writing them for one cycle so external callers (Pheno4D
     scripts, regression captures) can migrate.  Will be deleted in S3.
  5. **S2.D:** Calls ``enable_andrieu_on_leaves(plant)`` — bakes
     ``gf=6`` plus the Andrieu/Hillier/Birch piecewise primitives
     (``R1_n``, ``R2_n``, ``lag_exp_n``, ``D_lin_n``, ``T0_n``, ``L_min``)
     per leaf LRP from ``data/phase_III_per_rank_LEAF.json``.  Mirrors
     stem-side per-rank baking on the leaf-side now that S2.A's
     ``MultiPhaseLeafGrowth`` GF is live and S2.C's dispatch wiring routes
     gf=6 leaves through ``f_gf->getLength``.  Gated juvenile ranks (1-3
     per Andrieu 2006 §3.A) are left at ``gf=1`` (ExponentialGrowth)
     because they have no published kinetic fit; their LRP arrays stay
     zero so any accidental ``gf=6`` mint at runtime falls through to
     ExponentialGrowth via the Lock #6 empty-array guard.
  6. Dumps the mutated state via ``plant.writeParameters(out_xml)``.

The bake is **idempotent**: every helper overwrites (rather than appends)
target fields, so re-baking the already-baked XML produces an identical
file.  This is the property that lets us use the same path as both input
and output.

Usage:

    python -m dart.coupling.scripts.bake_calibration_to_xml \
        [--input dart/coupling/data/maize_calibrated.xml] \
        [--output dart/coupling/data/maize_calibrated.xml]

CLI defaults are the canonical maize XML in either direction (in-place bake).
"""

from __future__ import annotations

import argparse
import json
import math
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
LEAF_KINETICS_JSON = Path(DEFAULT_XML).parent / "phase_III_per_rank_LEAF.json"

# MultiPhaseLeafGrowth enum value (mirrors Plant.h ``gft_multi_phase_leaf``).
# Reasserted via ``int(pb.GrowthFunctionType.multi_phase_leaf)`` at call
# time so a future enum-value renumber surfaces here as a drop-in fix.
GFT_MULTI_PHASE_LEAF = 6
GFT_EXPONENTIAL = 1


def enable_andrieu_on_leaves(
    plant,
    json_path: Path = LEAF_KINETICS_JSON,
    *,
    verbose: bool = False,
) -> int:
    """Bake Andrieu/Hillier/Birch (2006) per-rank piecewise primitives onto
    every leaf LRP whose subType maps to a non-gated Déa rank.

    ADR_LEAF_KINEMATICS_2026-04-28 §S2.D + §C4.  After S2.A shipped
    ``MultiPhaseLeafGrowth`` and S2.C wired ``Leaf::simulate`` to dispatch
    through ``f_gf->getLength``, this helper makes the calibrated maize
    XML actually USE the new GF — until this commit, every leaf in
    ``maize_calibrated.xml`` carried ``gf=1`` and the Andrieu fields
    were absent or zero, so the runtime fell back to ExponentialGrowth.

    Mapping (per ADR §C4):
      - Leaf subType N in maize_calibrated.xml  ⇄  Déa rank N (the JSON
        ``leaf_subtype_to_rank_map`` is identity 2→2, 3→3, ..., 17→17).
      - Ranks 1-3 are gated juvenile (no Andrieu fit) → leave at
        ``gf=1`` + zero kinetic fields → falls through to
        ExponentialGrowth at runtime.
      - Ranks 4-17 are fitted (figure-read from Figs 6/7/9/10 of Andrieu
        2006 + MF3D L_fin endpoint anchor via C¹ continuity) → set
        ``gf=6`` + write the per-rank primitives.

    R2 source: this helper writes ``R2_cm_per_Cd_rescaled`` (the C¹-anchored
    rescaled value) rather than ``R2_cm_per_Cd_published``.  Per the JSON
    ``unit_caveat_R2_published``, the published values appear to carry a
    units mismatch (mm/°Cd printed as cm/°Cd); the rescaled column lands
    in the kinematic literature range (0.47-0.91 cm/°Cd) and honours
    MF3D ``lmax_n`` at the per-rank endpoint.

    Idempotent: every call overwrites the same scalar fields with the
    same values from the same JSON.

    Args:
        plant: pre-initialise() ``pb.MappedPlant`` with maize-shaped LRPs.
        json_path: path to ``phase_III_per_rank_LEAF.json`` (defaults to
            the in-tree calibration data alongside ``maize_calibrated.xml``).
        verbose: print one line per LRP touched.

    Returns:
        int — count of LRPs that received the Andrieu primitives (does
        NOT count gated subTypes left on ``gf=1``).
    """
    if not json_path.exists():
        raise FileNotFoundError(
            f"phase_III_per_rank_LEAF.json missing at {json_path}; "
            "S2.D bake cannot proceed without the leaf calibration JSON."
        )
    with json_path.open() as f:
        kin = json.load(f)
    rank_by_n = {int(row["n"]): row for row in kin["ranks"]}
    subtype_to_rank = {int(k): int(v) for k, v in
                       kin["_meta"]["leaf_subtype_to_rank_map"].items()}
    L_min_default = float(kin["_meta"].get("L_min_cm", 0.025))

    # Confirm enum is what we expect — defensive guard against an upstream
    # renumber that would silently mis-mint the GF on every leaf.
    expected_enum = int(pb.GrowthFunctionType.multi_phase_leaf)
    assert expected_enum == GFT_MULTI_PHASE_LEAF, (
        f"pb.GrowthFunctionType.multi_phase_leaf={expected_enum}; "
        f"S2.D was written assuming {GFT_MULTI_PHASE_LEAF}. Check "
        "Plant.h GrowthFunctionTypes ordering and update GFT_MULTI_PHASE_LEAF."
    )

    leaf_lrps = list(plant.getOrganRandomParameter(pb.OrganTypes.leaf))
    n_andrieu = 0
    n_gated_left_alone = 0
    for lrp in leaf_lrps:
        sub = int(lrp.subType)
        rank = subtype_to_rank.get(sub)
        if rank is None:
            # Subtype not in the published mapping — leave at gf=1 (legacy
            # ExponentialGrowth).  Maize XML's tassel subType=21 is the
            # canonical example; tassels are not leaves under Andrieu.
            if verbose:
                print(f"    leaf subType={sub}: no rank mapping, skipped")
            continue
        row = rank_by_n.get(rank)
        if row is None:
            if verbose:
                print(f"    leaf subType={sub}: rank {rank} missing in JSON, skipped")
            continue
        if bool(row.get("gated", False)):
            # Gated juvenile (Andrieu §3.A no-fit). Leave at gf=1.
            # Zero out Andrieu fields so a downstream caller flipping
            # gf to 6 still gets the empty-array fallback rather than
            # stale data.
            lrp.gf = GFT_EXPONENTIAL
            lrp.R1_n = 0.0
            lrp.R2_n = 0.0
            lrp.lag_exp_n = 0.0
            lrp.D_lin_n = 0.0
            lrp.T0_n = float(row.get("T0_Cd", 0.0))
            lrp.L_min = L_min_default
            n_gated_left_alone += 1
            if verbose:
                print(f"    leaf subType={sub}: rank {rank} gated, gf=1 zero-kinetics")
            continue
        # Non-gated fitted rank → opt-in to MultiPhaseLeafGrowth.
        # S1.B fix: derive R2_n and lag_exp_n live from the XML's actual
        # `lmax` instead of trusting JSON's pre-computed `R2_cm_per_Cd_rescaled`
        # / `lag_exp_Cd_implied`. The JSON's L_fin_target_cm was generated
        # against an MF3D snapshot that has since drifted for ranks 4-5
        # (JSON 53.2/64.8 cm vs current XML 45.2/50.0 cm). Live derivation
        # against `lrp.lmax` keeps the C¹ rescaling self-consistent under
        # future lmax updates and matches ADR §D6's "XML is the calibration
        # source of truth" principle. JSON's R1_Cd_inv and D_lin_Cd remain
        # the figure-read primitives. T0_Cd stays the published Phase-E
        # origin (S1 plastochron-stepped). The published `lag_exp_Cd` was
        # the original bug — it ended the exp phase at L_min·exp(R1·lag)
        # << L1, so the linear phase over D_lin couldn't reach lmax (high
        # ranks 15-17 plateaued at 24-56 % of lmax under the published value).
        R1 = float(row["R1_Cd_inv"])
        Dlin = float(row["D_lin_Cd"])
        lmax_xml = float(lrp.lmax)
        L1 = lmax_xml / (1.0 + R1 * Dlin)
        R2_rescaled = R1 * L1
        lag_implied = math.log(L1 / L_min_default) / R1
        lrp.gf = GFT_MULTI_PHASE_LEAF
        lrp.R1_n = R1
        lrp.R2_n = R2_rescaled
        lrp.lag_exp_n = lag_implied
        lrp.D_lin_n = Dlin
        lrp.T0_n = float(row["T0_Cd"])
        lrp.L_min = L_min_default
        # Empirical collar emergence (Vidal 2021 SupData3, M40+M52 averaged).
        # Setting t_col_emp_Cd >= 0 makes MultiPhaseStemGrowth::calcLengthPerPhytomer
        # anchor stem internode init_tt on the empirical event instead of computing
        # it from the leaf curve fits (decouples stem timing from leaf C¹ rescaling).
        # Default -1.0 preserves bit-identical fallback for ranks not in SupData3.
        lrp.t_col_emp_Cd = float(row.get("t_col_emp_Cd", -1.0))
        n_andrieu += 1
        if verbose:
            print(
                f"    leaf subType={sub}: rank {rank} → gf=6 "
                f"R1={lrp.R1_n:.4f} R2={lrp.R2_n:.4f} lag={lrp.lag_exp_n:.0f} "
                f"D={lrp.D_lin_n:.0f} T0={lrp.T0_n:.0f}"
            )
    if verbose:
        print(
            f"  Andrieu kinetics: enabled on {n_andrieu} leaf LRPs "
            f"(plus {n_gated_left_alone} gated juveniles left on gf=1)"
        )
    return n_andrieu


def bake(input_xml: Path, output_xml: Path, *, verbose: bool = True) -> None:
    """Bake runtime Python configuration into ``output_xml``.

    Args:
        input_xml: master XML to read (typically ``maize_calibrated.xml``).
        output_xml: target XML to write (defaults to ``input_xml`` for
            in-place bake; safe because the helpers below are idempotent).
        verbose: forwarded to the helpers for one-line summaries.
    """
    if verbose:
        print(f"=== bake: {input_xml} → {output_xml} ===")

    plant = pb.MappedPlant()
    plant.readParameters(str(input_xml))

    # Order matches grow_plant() pre-init contract.
    setup_successor_where(plant)
    enable_fa_on_mainstem(plant, verbose=verbose)
    enable_fa_on_leaves(plant, verbose=verbose)
    enable_andrieu_on_leaves(plant, verbose=verbose)

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
        n_leaf_andrieu = sum(
            1 for lrp in plant.getOrganRandomParameter(pb.OrganTypes.leaf)
            if int(getattr(lrp, "gf", 1)) == GFT_MULTI_PHASE_LEAF
        )
        print(
            f"  baked: stem FA={fa_on} (v_n len={n_v}); "
            f"leaf FA(legacy)={n_leaf_fa} LRPs; "
            f"leaf Andrieu(gf=6)={n_leaf_andrieu} LRPs"
        )
        print(f"  wrote {output_xml}")


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="bake_calibration_to_xml",
        description=(
            "Bake setup_successor_where + enable_fa_on_mainstem + "
            "enable_fa_on_leaves + enable_andrieu_on_leaves into the "
            "maize XML so the runtime contract becomes pure XML + C++ "
            "(D6 closure)."
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
