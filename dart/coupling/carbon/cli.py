"""CLI entry point for the ``carbon`` subcommand."""

import json
import sys

import numpy as np

from ..growth import grow_plant, run_photosynthesis
from .phloem_steady import solve_carbon_partitioning
from ..config import DEFAULT_XML, OUTPUT_DIR


def main_carbon(args):
    """CLI handler for the ``carbon`` subcommand."""
    carbon_out = OUTPUT_DIR / "carbon"
    carbon_out.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"CARBON PARTITIONING — Day {args.day}, method={args.method}")
    print(f"{'='*60}")

    # 1. Grow plant
    plant = grow_plant(
        xml_path=str(DEFAULT_XML),
        simulation_time=args.day,
        enable_photosynthesis=True,
        seed=42,
    )

    # 2. Run photosynthesis to get An per leaf segment
    prefix = str(carbon_out / f"day{args.day}_photo")
    hm = run_photosynthesis(
        plant, sim_time=args.day, output_prefix=prefix,
        par_umol=args.par, tair_c=args.tair,
    )
    if hm is None:
        print("ERROR: Photosynthesis solve failed.")
        sys.exit(1)

    An_leaf = np.array(hm.get_net_assimilation())  # mol CO2/d per leaf seg
    An_total_mmol = float(np.sum(An_leaf)) * 1000.0

    # 3. Solve carbon partitioning
    result = solve_carbon_partitioning(
        plant, An_leaf, Tair_C=args.tair,
        method=args.method, day=args.day,
    )

    # 4. Print results
    print(f"\n{'='*60}")
    print(f"CARBON PARTITIONING RESULTS ({result['partitioning_source']})")
    print(f"{'='*60}")
    print(f"  An total (input)     : {An_total_mmol:.1f} mmol CO2/d")
    print(f"  Rm total             : {result['Rm_total_mmol']:.1f} mmol/d")
    print(f"    Rm leaf            : {result['Rm_leaf']:.1f}")
    print(f"    Rm stem            : {result['Rm_stem']:.1f}")
    print(f"    Rm root            : {result['Rm_root']:.1f}")
    print(f"    Rm storage         : {result['Rm_storage']:.1f}")
    print(f"  Rg total             : {result['Rg_total_mmol']:.1f} mmol/d")
    print(f"  Growth               : {result['growth_mmol_d']:.1f} mmol/d")
    if 'stem_storage_mmol' in result:
        print(f"  Stem storage         : {result['stem_storage_mmol']:.1f} mmol/d")
    if result.get('seed_reserve_mmol', 0) > 0:
        print(f"  Seed reserve         : {result['seed_reserve_mmol']:.1f} mmol/d")
    print(f"  Partitioning fractions:")
    print(f"    FR_leaf            : {result['FR_leaf']:.3f}")
    print(f"    FR_stem            : {result['FR_stem']:.3f}")
    print(f"    FR_root            : {result['FR_root']:.3f}")
    print(f"    FR_storage         : {result['FR_storage']:.3f}")
    print(f"    Sum                : {result['FR_leaf']+result['FR_stem']+result['FR_root']+result['FR_storage']:.3f}")
    print(f"  Carbon balance error : {result['carbon_balance_error']:.2%}")
    if 'C_ST_mean' in result and not np.isnan(result.get('C_ST_mean', np.nan)):
        print(f"  C_ST mean            : {result['C_ST_mean']:.3f} mmol/cm3")
        print(f"  C_ST range           : [{result.get('C_ST_min', 0):.3f}, {result.get('C_ST_max', 0):.3f}]")
        print(f"  Picard iterations    : {result.get('n_iterations', 'N/A')}")
        print(f"  Converged            : {result.get('converged', 'N/A')}")

    # 5. Save results
    out_path = carbon_out / f"day{args.day}_{args.method}_carbon.json"
    # Convert numpy types for JSON serialization
    serializable = {}
    for k, v in result.items():
        if isinstance(v, np.ndarray):
            serializable[k] = v.tolist()
        elif isinstance(v, (np.floating, np.integer)):
            serializable[k] = float(v)
        else:
            serializable[k] = v
    with open(out_path, 'w') as f:
        json.dump(serializable, f, indent=2)
    print(f"\n  Results saved: {out_path}")
