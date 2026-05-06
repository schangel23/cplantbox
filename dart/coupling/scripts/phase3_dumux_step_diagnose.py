"""Diagnose the DuMux d01 convergence failure surfaced by phase3_drying_smoke.

phase3_drying_smoke.py shows:
    DRY_d00:  An=668.4, transp=537.2, ψ_leaf=-0.077..-0.030 MPa  ✓
    DRY_d01:  ERROR in hm.solve(): photosynthesis::solve: did not reach convergence

G1 (day-0 An ≡ FixedSoilPsi) passes, so the matric/total convention is fine.
The failure is in the *next* step: after DuMux integrates 1 day under the
queued RWU sink. This script isolates that single step and prints:

    * rsx profile range before vs after the DuMux solve
    * sink dict size + per-cell magnitude
    * total mass balance (Σsink ≈ -transp)
    * post-step rsx full profile (top 20 cells + 5 deepest)
    * hm.solve verbose=1 retry on the post-step profile

Outcome should reveal whether:
    A. Sink is being mis-applied (wrong sign / units / cell mapping) →
       DuMux falls into severe negative head in 1-2 cells.
    B. Sink concentrates in few cells →
       rsx heterogeneity confuses photosynthesis Newton iteration.
    C. Something else (DuMux numerical, hm pathological at this sim_time…).

Usage:
    cpbenv/bin/python -m dart.coupling.scripts.phase3_dumux_step_diagnose
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from dart.coupling.growth.grow import grow_plant
from dart.coupling.hydraulics.soil_psi import DumuxSoilPsi
from dart.coupling.photosynthesis.coupled import run_photosynthesis_solve

XML = Path(__file__).resolve().parents[1] / "data" / "maize_calibrated.xml"
START_DAY = 55
PSI_INIT = -300.0
PAR = 1500.0
T = 25.0
RH = 0.7
DEPTH = 100


def _print_profile(label: str, profile: np.ndarray) -> None:
    print(f"\n  {label}:")
    print(f"    min={profile.min():.2f}, max={profile.max():.2f}, "
          f"mean={profile.mean():.2f}, std={profile.std():.2f}")
    print(f"    top20 (idx 0..19): "
          f"{np.array2string(profile[:20], precision=1, suppress_small=True)}")
    print(f"    bot5 (idx 95..99): "
          f"{np.array2string(profile[-5:], precision=1, suppress_small=True)}")


def main() -> int:
    print(f"Growing maize to day {START_DAY}…")
    plant = grow_plant(str(XML), simulation_time=START_DAY, seed=42,
                       enable_photosynthesis=True)
    print(f"  {len(plant.getSegments())} segments")

    provider = DumuxSoilPsi(depth_cm=DEPTH, n_cells_z=DEPTH,
                            psi_init_cm=PSI_INIT, verbose=False)
    provider._t_last_days = float(START_DAY)

    # --- d00: should match FixedSoilPsi(-300) bit-identically ---
    print("\n=== Step 1: DRY_d00 (sim_time = 55) ===")
    profile_d00 = provider.get_profile(START_DAY, DEPTH)
    _print_profile("rsx d00 (DuMux IC)", profile_d00)

    res_d00 = run_photosynthesis_solve(
        plant, START_DAY, par=PAR, tleaf=T, label="DRY_d00", rh=RH,
        soil_psi_provider=provider,
    )
    if res_d00 is None:
        print("  d00 solve failed — bail")
        return 1
    print(f"\n  d00: An={res_d00['An_total_mmol']:.2f} mmol/d, "
          f"transp={res_d00['transp_mmol']:.2f} mmol/d")

    # Inspect the queued sink before DuMux applies it.
    queued = provider._pending_sink or {}
    if queued:
        vals = np.array(list(queued.values()))
        print(f"\n  Pending sink (DuMux native ordering):")
        print(f"    n_cells_with_sink = {len(queued)}")
        print(f"    sum = {vals.sum():.4g} cm³/d  (negative = uptake)")
        print(f"    min (most negative) = {vals.min():.4g}")
        print(f"    max = {vals.max():.4g}")
        print(f"    transp_cm3_d ≈ {res_d00['transp_mmol'] * 18 * 1e-3:.4g}")
        print(f"    |Σsink| / transp_cm3 = "
              f"{abs(vals.sum()) / (res_d00['transp_mmol'] * 18 * 1e-3):.4f}")
        # Top 10 most-aggressive sink cells
        order = np.argsort(vals)
        print(f"    top10 most-negative cells (native_idx, value):")
        for k in order[:10]:
            cell_id = list(queued.keys())[k]
            print(f"       cell {cell_id}: {vals[k]:.4g} cm³/d")
    else:
        print("  Pending sink: EMPTY (push_rwu_sink_to_provider didn't run "
              "or aggregated to {})")

    # --- Step DuMux to d01 manually (don't go through run_photosynthesis_solve). ---
    print(f"\n=== Step 2: advance DuMux 1 day (55 → 56) under queued sink ===")
    profile_d01 = provider.get_profile(START_DAY + 1, DEPTH)
    _print_profile("rsx d01 (post 1-day DuMux solve with sink)", profile_d01)

    delta = profile_d01 - profile_d00
    print(f"\n  Δrsx (d01 - d00):")
    print(f"    min (most-decreased) = {delta.min():.2f} cm")
    print(f"    max (most-increased) = {delta.max():.2f} cm")
    n_changed = int(np.sum(np.abs(delta) > 0.5))
    print(f"    cells with |Δ| > 0.5 cm: {n_changed} / {DEPTH}")

    # --- Step 3: try hm.solve with the post-step profile ---
    print(f"\n=== Step 3: hm.solve(rsx=d01_profile, cells=True) — "
          f"the failing call ===")

    # Re-create a fresh hm by going through run_photosynthesis_solve. The
    # provider state is already advanced, so get_profile(t=56) is idempotent
    # (dt_days = 0).
    res_d01 = run_photosynthesis_solve(
        plant, START_DAY + 1, par=PAR, tleaf=T, label="DRY_d01_retry", rh=RH,
        soil_psi_provider=provider,
    )
    if res_d01 is None:
        print("  → CONFIRMED: hm.solve fails on the d01 rsx profile.")
        print("  → Inspect 'rsx d01' stats above for diagnosis.")
        return 2

    print(f"\n  d01: An={res_d01['An_total_mmol']:.2f} mmol/d, "
          f"transp={res_d01['transp_mmol']:.2f} mmol/d")
    print(f"\n  No convergence failure on retry — flake?")
    return 0


if __name__ == "__main__":
    sys.exit(main())
