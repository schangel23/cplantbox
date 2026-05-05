"""pm_maize55_24h.py — GATE B: 24-hour PiafMunch transient on day-55 maize.

Regenerated from the lost /tmp/_pm_maize55_24h.py (now tracked).

Purpose: at the corrected Q_Rmmax = 16 mmol Suc/d (not 385), does the
cuse-gate trap still close (C_ST_mean pinned at CSTimin, Rg = 0)?
- If trap closes anyway: the diagnosis was correct in spirit but the
  magnitude was wrong; (alpha') JSON Krm1 trim toward WOFOST may still
  help, but we should investigate Vmaxloading + beta_loading first.
- If trap is gone: prior cuse-gate diagnosis was an artifact of the
  385 misreading. (alpha') is unnecessary.

Reads hm.C_ST and integrates Q_Rm / Q_Gr / Q_Exud hourly to track rates.
"""

import os
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from dart.coupling.config import (  # noqa: E402
    DEFAULT_XML, get_hydraulics_json, get_photosynthesis_json, get_phloem_json,
)
from dart.coupling.growth import grow_plant  # noqa: E402

N_DAYS    = 1
DT_HOURS  = 1.0
N_STEPS   = int(N_DAYS * 24 / DT_HOURS)


def _suppress():
    o1 = os.dup(1); o2 = os.dup(2); dn = os.open(os.devnull, os.O_WRONLY)
    os.dup2(dn, 1); os.dup2(dn, 2)
    return o1, o2, dn


def _restore(o1, o2, dn):
    os.dup2(o1, 1); os.dup2(o2, 2)
    os.close(dn); os.close(o1); os.close(o2)


def main():
    print("=" * 78)
    print(f"PiafMunch maize day-55 24h transient (GATE B)")
    print("=" * 78)

    age = 55
    Tair_C = 25.0
    plant = grow_plant(
        xml_path=str(DEFAULT_XML),
        simulation_time=age,
        min_stem_nodes=100,
        min_leaf_nodes=40,
        enable_photosynthesis=True,
        seed=42,
    )

    from plantbox.functional.phloem_flux import PhloemFluxPython
    from plantbox.functional.PlantHydraulicParameters import PlantHydraulicParameters

    params = PlantHydraulicParameters()
    params.read_parameters(get_hydraulics_json())

    hm = PhloemFluxPython(plant, params, psiXylInit=-500, ciInit=350e-6)
    hm.read_photosynthesis_parameters(filename=get_photosynthesis_json())
    hm.read_phloem_parameters(filename=get_phloem_json())
    hm.atol = 1e-6; hm.rtol = 1e-4; hm.solver = 32
    hm.useCWGr = False  # avoid Gr_Y assert during diagnostic-only run

    # Photosynthesis (saturating PAR, T=25)
    rh = 0.7; psi_soil = -500.0
    p_s = np.linspace(psi_soil, psi_soil - 100, 100)
    es = hm.get_es(Tair_C); ea = es * rh
    par = 1000.0 * 1e-6 * 86400 * 1e-4
    hm.solve(sim_time=age, rsx=p_s, cells=True, ea=ea, es=es,
             PAR=par, TairC=Tair_C, verbose=0)
    An_total = float(np.sum(np.array(hm.get_net_assimilation()))) * 1e3
    print(f"An_total = {An_total:.1f} mmol CO2/d  ({Tair_C} C, PAR=1000 umol/m2/s)")

    Nt = len(plant.getNodes())
    Tair_K = Tair_C + 273.15
    dt_days = DT_HOURS / 24.0

    print(f"\n{'h':>4} {'Rm_h':>10} {'Rg_h':>10} {'Exud_h':>10} "
          f"{'C_ST_mean':>10} {'C_ST_max':>10} {'wall':>6}")
    print(f"{'':>4} {'mmolSuc':>10} {'mmolSuc':>10} {'mmolSuc':>10} "
          f"{'m/cm3':>10} {'m/cm3':>10} {'s':>6}")
    print("-" * 78)

    prev_QRm = 0.0; prev_QGr = 0.0; prev_QExud = 0.0
    QRm = QGr = QExud = 0.0
    cmean = 0.0; cmax = 0.0

    for step in range(1, N_STEPS + 1):
        if step > 1:
            hm.withInitVal = False  # continue from prior state

        t_start = float(age) + (step - 1) * dt_days
        t_end   = t_start + dt_days

        t0 = time.time()
        fdpair = _suppress()
        try:
            ret = hm.startPM(t_start, t_end, 1, Tair_K, False,
                             str(REPO_ROOT / "dart/coupling/scripts/_pm_24h.txt"))
        finally:
            _restore(*fdpair)
        wall = time.time() - t0
        if ret != 1:
            print(f"step {step}: solver returned {ret} (treated as failure, breaking)")
            break

        Q = np.array(hm.Q_out)
        Q_Rm   = Q[Nt*2:Nt*3]
        Q_Exud = Q[Nt*3:Nt*4]
        Q_Gr   = Q[Nt*4:Nt*5]
        QRm   = float(np.sum(Q_Rm))
        QGr   = float(np.sum(Q_Gr))
        QExud = float(np.sum(Q_Exud))

        Rm_h   = QRm   - prev_QRm
        Rg_h   = QGr   - prev_QGr
        Exud_h = QExud - prev_QExud
        prev_QRm = QRm; prev_QGr = QGr; prev_QExud = QExud

        C_ST = np.array(hm.C_ST)
        cmean = float(np.mean(C_ST)); cmax = float(np.max(C_ST))

        if step in (1, 2, 3, 4, 5, 6, 12, 18, 24):
            print(f"{step:>4} {Rm_h:>10.3f} {Rg_h:>10.3f} {Exud_h:>10.3f} "
                  f"{cmean:>10.4f} {cmax:>10.4f} {wall:>6.2f}")

    # Final summary
    print("-" * 78)
    print(f"\n  Final C_ST_mean = {cmean:.4f}  (CSTimin = 0.20)")
    print(f"  Final cumulative: Rm={QRm:.2f} Gr={QGr:.2f} Exud={QExud:.2f} mmol Suc")
    print(f"  Daily-rate equiv: Rm={QRm/N_DAYS:.2f} Gr={QGr/N_DAYS:.2f} mmol Suc/d")

    # Verdict
    print("\n" + "=" * 78)
    print("CUSE-GATE VERDICT")
    print("=" * 78)
    pinned = abs(cmean - 0.20) < 0.01
    rg_zero = QGr / max(N_DAYS, 0.001) < 1.0
    if pinned and rg_zero:
        print("  -> C_ST_mean PINNED at CSTimin AND Rg ~= 0.")
        print("  -> Cuse-gate trap is real and remains closed at the corrected")
        print("     Q_Rmmax level. The trap is from a DIFFERENT mechanism than")
        print("     the prior (false) S/D inequality; investigate Vmaxloading,")
        print("     initialization, or per-segment loading distribution.")
    elif pinned and not rg_zero:
        print(f"  -> C_ST_mean ~ CSTimin BUT Rg = {QGr/max(N_DAYS,0.001):.2f} mmol/d nonzero.")
        print("  -> Loading/usage stoichiometry is balanced at the gate, not trapped.")
    elif not pinned:
        print(f"  -> C_ST_mean = {cmean:.3f} > CSTimin. Trap is OPEN.")
        print("  -> Prior cuse-gate diagnosis was driven by 385-mmol/d misreading.")
        print("  -> (alpha') maintenance recalibration is UNNECESSARY for trap closure.")
        print("  -> Krm1 vs WOFOST overshoot is now a literature-fidelity issue,")
        print("     not a Ch1/Ch2 blocker.")

    p = REPO_ROOT / "dart/coupling/scripts/_pm_24h.txt"
    if p.exists():
        p.unlink()


if __name__ == "__main__":
    main()
