"""End-to-end Phase 2 integration smoke test.

Grows a maize plant, runs `run_photosynthesis_solve` via three paths, and
asserts:
  1. Legacy fall-through (no provider, soil_psi_cm=-500) produces the same
     output as the explicit FixedSoilPsi(-500) provider — proves the
     rewired code path doesn't drift from the bit-identical regression.
  2. DumuxSoilPsi end-to-end runs without crashing.

This catches integration bugs that the unit tests on get_profile alone
can't see (sim_time threading, t_days plumbing, wrong array unit, etc.).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from dart.coupling.growth.grow import grow_plant
from dart.coupling.hydraulics.soil_psi import DumuxSoilPsi, FixedSoilPsi
from dart.coupling.photosynthesis.coupled import run_photosynthesis_solve

XML_PATH = Path(__file__).resolve().parents[1] / "data" / "maize_calibrated.xml"
SIM_DAY = 30


def _summarise(label: str, result: dict) -> dict:
    summary = {
        "An_total_mmol": float(result["An_total_mmol"]),
        "transp_mmol": float(result.get("transp_mmol", float("nan"))),
        "psi_collar": float(np.asarray(result.get("psi_xyl"))[0])
        if result.get("psi_xyl") is not None else float("nan"),
    }
    print(f"  [{label}]")
    print(f"    An_total = {summary['An_total_mmol']:>10.4f} mmol/d")
    print(f"    transp   = {summary['transp_mmol']:>10.4f} mmol/d")
    print(f"    psi_collar = {summary['psi_collar']:>10.4f} cm")
    return summary


def main() -> int:
    print(f"Growing maize to day {SIM_DAY}...")
    plant = grow_plant(str(XML_PATH), simulation_time=SIM_DAY, seed=42)
    print(f"  plant: {len(plant.getSegments())} segments")

    print("\n[1] Legacy fall-through: soil_psi_cm=-500, no provider")
    legacy = run_photosynthesis_solve(
        plant, SIM_DAY, par=1500.0, tleaf=25.0,
        label="legacy_fallthrough", rh=0.7, soil_psi_cm=-500.0,
    )
    s_legacy = _summarise("legacy", legacy)

    print("\n[2] Explicit FixedSoilPsi(-500): should match [1] bit-for-bit")
    fixed_explicit = run_photosynthesis_solve(
        plant, SIM_DAY, par=1500.0, tleaf=25.0,
        label="fixed_explicit", rh=0.7, soil_psi_cm=-999.0,  # ignored
        soil_psi_provider=FixedSoilPsi(psi_cm=-500.0),
    )
    s_fixed = _summarise("fixed-explicit", fixed_explicit)

    def _eq_nan(a, b):
        if np.isnan(a) and np.isnan(b):
            return True
        return a == b

    legacy_match = (
        _eq_nan(s_legacy["An_total_mmol"], s_fixed["An_total_mmol"])
        and _eq_nan(s_legacy["transp_mmol"], s_fixed["transp_mmol"])
        and _eq_nan(s_legacy["psi_collar"], s_fixed["psi_collar"])
    )
    print(f"\n  Bit-identical legacy ≡ FixedSoilPsi: {legacy_match}")

    print("\n[3] DumuxSoilPsi(psi_init=-500): end-to-end through hm.solve")
    dumux_provider = DumuxSoilPsi(
        depth_cm=100, n_cells_z=100, psi_init_cm=-500.0, verbose=False,
    )
    dumux_result = run_photosynthesis_solve(
        plant, SIM_DAY, par=1500.0, tleaf=25.0,
        label="dumux_psi500", rh=0.7,
        soil_psi_provider=dumux_provider,
    )
    s_dumux = _summarise("dumux", dumux_result)

    print("\n=== Phase 2 integration smoke ===")
    if not legacy_match:
        print("  FAIL: legacy fall-through != explicit FixedSoilPsi")
        return 1
    if not (np.isfinite(s_dumux["An_total_mmol"])
            and s_dumux["An_total_mmol"] != 0):
        print("  FAIL: DumuxSoilPsi produced non-finite or zero An")
        return 2
    print("  PASS: legacy ≡ FixedSoilPsi, DumuxSoilPsi end-to-end clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
