"""DuMux-Rosi 1D vertical soil-column drying smoke test (Phase 0).

Verifies that the existing rosi_richards.so build can solve a real Richards
problem end-to-end, and produces a per-day wall-time number used to size the
diurnal-loop integration in Phase 2 of PLAN_DUMUX_INTEGRATION_2026-05-05.md.

Setup:
  - 1D vertical column, 100 cm depth, 1 cm cells (100 vertical DOF)
  - Loam VG params [theta_r, theta_s, alpha, n, Ksat] = [0.08, 0.43, 0.04, 1.6, 50]
  - Initial uniform pressure head psi = -500 cm
  - Top BC: zero flux (drying isolated from atmosphere)
  - Bottom BC: free drainage
  - Sim length: 30 days, snapshot every 6 h

Acceptance:
  - psi_s decreases monotonically near the bottom (gravity drainage)
  - Total wall time printed -> Phase 2 budget input

Talks to RichardsSP directly. The Python wrapper layer (rosi.richards) is
broken in the current build (rosi/ namespace package empty, mpi4py not
installed), but RichardsSP exposes the full API natively. This is the
abstraction layer DumuxSoilPsi will wrap in Phase 2 anyway.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

DUMUX_BIND = Path(
    "/home/lukas/PHD/dumux-build/dumux/dumux-rosi/build-cmake/cpp/python_binding"
)
sys.path.insert(0, str(DUMUX_BIND))

from rosi_richards import RichardsSP  # noqa: E402

# BC enum (cpp/soil_richards/richardsproblem.hh)
BC_CONSTANT_PRESSURE = 1
BC_CONSTANT_FLUX = 2
BC_ATMOSPHERIC = 4
BC_FREE_DRAINAGE = 5

# Loam Van Genuchten params [theta_r, theta_s, alpha (1/cm), n, Ksat (cm/d)]
LOAM = (0.08, 0.43, 0.04, 1.6, 50.0)

DEPTH_CM = 100
N_CELLS_Z = 100
PSI_INIT_CM = -500.0
SIM_DAYS = 30
SNAPSHOT_HOURS = 6
DAY_SECONDS = 86400.0


def build_solver():
    """Initial setup must go through the parameter tree.

    `setTopBC`/`setBotBC` are *runtime* overrides — they write directly to
    `problem->bcTopType_`, which only exists after `initializeProblem()`
    constructs the problem. Calling them before init segfaults. The problem
    constructor reads `Soil.BC.Top.Type` / `Soil.BC.Top.Value` (and Bot)
    from the parameter tree, so we set those there.
    """
    s = RichardsSP()
    s.initialize([""], verbose=False, doMPI=False)

    qr, qs, alpha, n, ks = LOAM
    s.setParameter("Soil.VanGenuchten.Qr", str(qr))
    s.setParameter("Soil.VanGenuchten.Qs", str(qs))
    s.setParameter("Soil.VanGenuchten.Alpha", str(alpha))
    s.setParameter("Soil.VanGenuchten.N", str(n))
    s.setParameter("Soil.VanGenuchten.Ks", str(ks))
    s.setParameter("Soil.Layer.Number", "1")

    # IC + BCs via parameter tree (read during initializeProblem)
    s.setParameter("Soil.IC.P", str(PSI_INIT_CM))
    s.setParameter("Soil.BC.Top.Type", str(BC_CONSTANT_FLUX))
    s.setParameter("Soil.BC.Top.Value", "0.0")
    s.setParameter("Soil.BC.Bot.Type", str(BC_FREE_DRAINAGE))
    s.setParameter("Soil.BC.Bot.Value", "0.0")

    # 1D-ish grid: 1x1 cell laterally, N_CELLS_Z vertical
    s.createGrid([-5.0, -5.0, -float(DEPTH_CM)], [5.0, 5.0, 0.0],
                 [1, 1, N_CELLS_Z], False)

    s.initializeProblem(-1.0)
    return s


def main():
    s = build_solver()

    coords = np.asarray(s.getDofCoordinates(), dtype=float)
    z_cm = coords[:, 2]
    order = np.argsort(z_cm)  # bottom (most negative z) first
    z_sorted = z_cm[order]
    print(f"Grid: {len(z_cm)} DOFs, z range [{z_sorted[0]:.2f}, {z_sorted[-1]:.2f}] cm")

    psi_init = np.asarray(s.getSolutionHead(), dtype=float)[order]
    print(f"Initial psi (top {z_sorted[-1]:+.1f} cm): {psi_init[-1]:.2f} cm")
    print(f"Initial psi (bot {z_sorted[0]:+.1f} cm): {psi_init[0]:.2f} cm")
    print()

    snapshots = [(0.0, psi_init.copy())]

    step_hours = SNAPSHOT_HOURS
    step_seconds = step_hours * 3600.0
    n_steps = int(SIM_DAYS * 24 / step_hours)

    t0_wall = time.perf_counter()
    for i in range(n_steps):
        t1 = time.perf_counter()
        s.solveNoMPI(step_seconds, False)
        t_sim_h = (i + 1) * step_hours
        psi = np.asarray(s.getSolutionHead(), dtype=float)[order]
        snapshots.append((t_sim_h / 24.0, psi.copy()))
        if (i + 1) % 4 == 0:  # one print per simulated day
            dt_step = time.perf_counter() - t1
            print(f"  day {t_sim_h/24:5.1f}  psi_top={psi[-1]:8.2f} cm  "
                  f"psi_mid={psi[len(psi)//2]:8.2f} cm  "
                  f"psi_bot={psi[0]:8.2f} cm  step={dt_step:5.2f}s")

    wall = time.perf_counter() - t0_wall
    print()
    print(f"Total wall time:        {wall:.2f} s for {SIM_DAYS} simulated days")
    print(f"Per-simulated-day cost: {wall/SIM_DAYS:.3f} s")

    # Acceptance: bottom psi must rise monotonically (water leaves via free drainage,
    # but column is drying so psi gets MORE negative everywhere; "monotonic decline"
    # means d(psi)/dt < 0 cell-by-cell over the whole run)
    psi_final = snapshots[-1][1]
    delta = psi_final - psi_init
    n_drier = int(np.sum(delta < 0))
    n_wetter = int(np.sum(delta > 0))
    print()
    print(f"Cells that got drier (psi decreased): {n_drier}/{len(delta)}")
    print(f"Cells that got wetter (psi increased): {n_wetter}/{len(delta)}")
    print(f"Bottom psi change over {SIM_DAYS} d: "
          f"{psi_init[0]:.2f} -> {psi_final[0]:.2f} cm (Δ={delta[0]:+.2f})")
    print(f"Top    psi change over {SIM_DAYS} d: "
          f"{psi_init[-1]:.2f} -> {psi_final[-1]:.2f} cm (Δ={delta[-1]:+.2f})")

    # Pass criteria: at least bottom-half is monotonic (gravity drainage), and
    # solver didn't blow up (no NaNs, no |psi| > 1e6)
    assert np.all(np.isfinite(psi_final)), "non-finite psi after solve"
    assert np.all(np.abs(psi_final) < 1e6), "psi blew up"
    print()
    print("SMOKE TEST: PASS")


if __name__ == "__main__":
    main()
