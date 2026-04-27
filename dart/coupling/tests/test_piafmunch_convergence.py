#!/usr/bin/env python3
"""
Standalone test: Can PiafMunch (full dynamic ODE solver) converge
with the corrected maize2026 parameters on a day-55 plant?

This test does NOT modify any existing code. It:
1. Grows a day-55 maize plant (same as production pipeline)
2. Runs photosynthesis to get An per segment
3. Attempts PiafMunch startPM() with corrected parameters
4. Reports convergence/stiffness outcome

Expected outcome: PiafMunch may fail on the ~1700-segment tree
due to ODE stiffness. This test documents whether it does.
"""

import signal
import sys
import time
import traceback
from pathlib import Path

import numpy as np

# Setup path
COUPLING_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(COUPLING_DIR.parent.parent))

import plantbox as pb
from dart.coupling.config import (
    DEFAULT_XML, get_hydraulics_json, get_photosynthesis_json, get_phloem_json,
)
from dart.coupling.growth import grow_plant


class TimeoutError(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutError("PiafMunch solver timed out")


def main():
    print("=" * 70)
    print("PiafMunch Convergence Test (read-only, no code modifications)")
    print("=" * 70)

    # --- Step 1: Grow plant ---
    print("\n[1/4] Growing day-55 maize plant...")
    t0 = time.time()
    plant = grow_plant(
        xml_path=str(DEFAULT_XML),
        simulation_time=55,
        min_stem_nodes=100,
        min_leaf_nodes=40,
        enable_photosynthesis=True,
        seed=42,
    )
    n_segs = len(plant.getNodes()) - 1
    organ_types = np.array(plant.organTypes)
    n_root = np.sum(organ_types == 2)
    n_stem = np.sum(organ_types == 3)
    n_leaf = np.sum(organ_types == 4)
    print(f"  Plant: {n_segs} segments (root={n_root}, stem={n_stem}, leaf={n_leaf})")
    print(f"  Time: {time.time() - t0:.1f}s")

    # --- Step 2: Create PhloemFluxPython and load params ---
    print("\n[2/4] Creating PhloemFluxPython with corrected maize2026 params...")
    from plantbox.functional.phloem_flux import PhloemFluxPython
    from plantbox.functional.PlantHydraulicParameters import PlantHydraulicParameters

    params = PlantHydraulicParameters()
    params.read_parameters(get_hydraulics_json())

    hm = PhloemFluxPython(plant, params, psiXylInit=-500, ciInit=350e-6)
    hm.read_photosynthesis_parameters(filename=get_photosynthesis_json())
    hm.read_phloem_parameters(filename=get_phloem_json())

    print(f"  Vmaxloading = {hm.Vmaxloading}")
    print(f"  beta_loading = {hm.beta_loading}")
    print(f"  KMfu = {hm.KMfu}")
    print(f"  Gr_Y = {hm.Gr_Y}")
    print(f"  leafGrowthZone = {hm.leafGrowthZone}")
    print(f"  atol = {hm.atol}, rtol = {hm.rtol}")

    # Relax tolerances (default 1e-17/1e-23 is extremely tight)
    hm.atol = 1e-6
    hm.rtol = 1e-4
    print(f"  Relaxed tolerances: atol={hm.atol}, rtol={hm.rtol}")

    # --- Step 3: Run photosynthesis ---
    print("\n[3/4] Running photosynthesis (uniform PAR=1000, T=25°C)...")
    par_umol = 1000.0
    tair_c = 25.0
    rh = 0.7
    soil_psi_cm = -500.0

    depth = 100
    p_s = np.linspace(soil_psi_cm, soil_psi_cm - depth, depth)
    es = hm.get_es(tair_c)
    ea = es * rh
    par_mol_cm2_d = par_umol * 1e-6 * 86400 * 1e-4

    t0 = time.time()
    hm.solve(
        sim_time=55,
        rsx=p_s,
        cells=True,
        ea=ea,
        es=es,
        PAR=par_mol_cm2_d,
        TairC=tair_c,
        verbose=0,
    )
    An_leaf = np.array(hm.get_net_assimilation())
    An_total = np.sum(An_leaf) * 1e3
    print(f"  An_total = {An_total:.1f} mmol CO2/d ({len(An_leaf)} leaf segments)")
    print(f"  Time: {time.time() - t0:.1f}s")

    # --- Step 4: Attempt PiafMunch ---
    print("\n[4/4] Attempting PiafMunch startPM() ...")
    print(f"  Solver will run for dt=0.04167 days (1 hour)")
    print(f"  Timeout: 120 seconds")

    # Set timeout
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(120)  # 2 minute timeout

    dt = 1.0 / 24.0  # 1 hour
    start_time = 55.0
    end_time = start_time + dt
    tair_k = tair_c + 273.15

    # Suppress verbose C++ output to see actual result
    import io
    import os

    t0 = time.time()
    try:
        # Redirect stdout/stderr to suppress waterLimitedGrowth spam
        old_stdout_fd = os.dup(1)
        old_stderr_fd = os.dup(2)
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)

        result = hm.startPM(
            start_time,
            end_time,
            1,          # output step
            tair_k,
            False,      # verbose=False
            "piafmunch_test_output.txt",
        )

        # Restore stdout/stderr
        os.dup2(old_stdout_fd, 1)
        os.dup2(old_stderr_fd, 2)
        os.close(devnull)
        os.close(old_stdout_fd)
        os.close(old_stderr_fd)
        signal.alarm(0)  # Cancel timeout
        elapsed = time.time() - t0

        print(f"\n  >>> PiafMunch CONVERGED in {elapsed:.1f}s <<<")
        print(f"  Return code: {result}")

        # Extract results
        Nt = len(plant.getNodes())
        Q_out = np.array(hm.Q_out)
        Q_ST = Q_out[0:Nt]
        Q_meso = Q_out[Nt:Nt*2]
        Q_Rm = Q_out[Nt*2:Nt*3]
        Q_Exud = Q_out[Nt*3:Nt*4]
        Q_Gr = Q_out[Nt*4:Nt*5]

        C_ST = np.array(hm.C_ST)

        print(f"\n  Results:")
        print(f"    C_ST: mean={np.mean(C_ST):.4f}, min={np.min(C_ST):.4f}, max={np.max(C_ST):.4f} mmol/cm³")
        print(f"    Q_Rm total: {np.sum(Q_Rm):.4f} mmol Suc")
        print(f"    Q_Gr total: {np.sum(Q_Gr):.4f} mmol Suc")
        print(f"    Q_Exud total: {np.sum(Q_Exud):.4f} mmol Suc")

        # Sanity checks
        c_st_ok = np.all(np.isfinite(C_ST)) and np.max(C_ST) < 10.0
        balance_ok = np.sum(Q_Rm) + np.sum(Q_Gr) + np.sum(Q_Exud) > 0
        print(f"\n  Sanity: C_ST finite & < 10? {c_st_ok}")
        print(f"  Sanity: positive usage? {balance_ok}")

        if c_st_ok and balance_ok:
            print("\n  VERDICT: PiafMunch works with corrected params!")
        else:
            print("\n  VERDICT: PiafMunch ran but produced suspicious values")

    except TimeoutError:
        signal.alarm(0)
        # Restore stdout/stderr
        try:
            os.dup2(old_stdout_fd, 1)
            os.dup2(old_stderr_fd, 2)
            os.close(devnull)
            os.close(old_stdout_fd)
            os.close(old_stderr_fd)
        except Exception:
            pass
        elapsed = time.time() - t0
        print(f"\n  >>> PiafMunch TIMED OUT after {elapsed:.0f}s <<<")
        print("  VERDICT: ODE solver is stiff / non-convergent for this plant size")

    except Exception as e:
        signal.alarm(0)
        # Restore stdout/stderr
        try:
            os.dup2(old_stdout_fd, 1)
            os.dup2(old_stderr_fd, 2)
            os.close(devnull)
            os.close(old_stdout_fd)
            os.close(old_stderr_fd)
        except Exception:
            pass
        elapsed = time.time() - t0
        print(f"\n  >>> PiafMunch FAILED after {elapsed:.1f}s <<<")
        print(f"  Error: {e}")
        traceback.print_exc()
        print("  VERDICT: PiafMunch crashes with corrected params")

    # Cleanup test output file
    test_output = Path("piafmunch_test_output.txt")
    if test_output.exists():
        test_output.unlink()

    print("\n" + "=" * 70)
    print("Test complete. No existing code was modified.")
    print("=" * 70)


if __name__ == "__main__":
    main()
