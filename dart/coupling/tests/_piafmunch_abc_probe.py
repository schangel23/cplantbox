#!/usr/bin/env python3
"""Parametrised PiafMunch convergence probe — variants A/B/C.

Usage:
  python -u _piafmunch_abc_probe.py --age 55 --phloem maize2026 --dt-hours 1.0
  python -u _piafmunch_abc_probe.py --age 55 --phloem wheat2025 --dt-hours 1.0
  python -u _piafmunch_abc_probe.py --age 20 --phloem maize2026 --dt-hours 1.0
  python -u _piafmunch_abc_probe.py --age 55 --phloem maize2026 --dt-hours 0.0833

Wall-clock budget for startPM is enforced by the outer `timeout` shell wrapper.
"""
import argparse, sys, time
from pathlib import Path
import numpy as np

COUPLING_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(COUPLING_DIR.parent.parent))

import plantbox as pb
from dart.coupling.config import (
    DEFAULT_XML, DATA_DIR, get_hydraulics_json, get_photosynthesis_json,
)
from dart.coupling.growth import grow_plant


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--age", type=int, required=True)
    p.add_argument("--phloem", choices=["maize2026", "wheat2025"], required=True)
    p.add_argument("--dt-hours", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--solver", type=int, default=1, help="PiafMunch solver index (1=SPFGMR default, 32=KLU, 28/30=BAND)")
    args = p.parse_args()

    phloem_map = {
        "maize2026": str(DATA_DIR / "phloem_parameters_maize2026"),
        "wheat2025": str(DATA_DIR / "wheat_phloem_parameters"),
    }
    phloem_path = phloem_map[args.phloem]

    print(f"=== variant: age={args.age}d, phloem={args.phloem}, dt={args.dt_hours:.4f}h ===", flush=True)

    t0 = time.time()
    plant = grow_plant(
        xml_path=str(DEFAULT_XML),
        simulation_time=args.age,
        min_stem_nodes=20,
        min_leaf_nodes=5,
        enable_photosynthesis=True,
        seed=args.seed,
    )
    n_segs = len(plant.getNodes()) - 1
    ot = np.array(plant.organTypes)
    n_root = int(np.sum(ot == 2)); n_stem = int(np.sum(ot == 3)); n_leaf = int(np.sum(ot == 4))
    print(f"[grow {time.time()-t0:.1f}s] segs={n_segs} root={n_root} stem={n_stem} leaf={n_leaf}", flush=True)

    from plantbox.functional.phloem_flux import PhloemFluxPython
    from plantbox.functional.PlantHydraulicParameters import PlantHydraulicParameters

    params = PlantHydraulicParameters()
    params.read_parameters(get_hydraulics_json())

    hm = PhloemFluxPython(plant, params, psiXylInit=-500, ciInit=350e-6)
    hm.read_photosynthesis_parameters(filename=get_photosynthesis_json())
    hm.read_phloem_parameters(filename=phloem_path)
    hm.atol = 1e-6
    hm.rtol = 1e-4
    hm.solver = args.solver
    print(f"[setup] Vmaxloading={hm.Vmaxloading} beta={hm.beta_loading} KMfu={hm.KMfu} solver={hm.solver}", flush=True)

    par_umol = 1000.0; tair_c = 25.0; rh = 0.7; soil_psi = -500.0
    depth = 100
    p_s = np.linspace(soil_psi, soil_psi - depth, depth)
    es = hm.get_es(tair_c); ea = es * rh
    par = par_umol * 1e-6 * 86400 * 1e-4

    t0 = time.time()
    hm.solve(sim_time=args.age, rsx=p_s, cells=True, ea=ea, es=es, PAR=par, TairC=tair_c, verbose=0)
    An = np.array(hm.get_net_assimilation())
    print(f"[photo {time.time()-t0:.2f}s] An_total={float(np.sum(An)*1e3):.1f} mmol/d ({len(An)} leaf segs)", flush=True)

    dt_days = args.dt_hours / 24.0
    start_t = float(args.age)
    end_t = start_t + dt_days
    tair_k = tair_c + 273.15

    print(f"[startPM] t={start_t} → {end_t:.5f}d (dt={args.dt_hours}h)…", flush=True)
    sys.stdout.flush()
    t0 = time.time()
    try:
        ret = hm.startPM(start_t, end_t, 1, tair_k, True, "/tmp/_pm_probe.txt")
        elapsed = time.time() - t0
        Nt = len(plant.getNodes())
        Q = np.array(hm.Q_out)
        Q_Rm = Q[Nt*2:Nt*3]; Q_Gr = Q[Nt*4:Nt*5]; Q_Exud = Q[Nt*3:Nt*4]
        C_ST = np.array(hm.C_ST)
        c_ok = bool(np.all(np.isfinite(C_ST)) and np.max(C_ST) < 10.0)
        bal = float(np.sum(Q_Rm) + np.sum(Q_Gr) + np.sum(Q_Exud))
        print(f"[CONVERGED {elapsed:.1f}s] ret={ret} C_ST max={np.max(C_ST):.3f} mean={np.mean(C_ST):.3f} sane={c_ok} sumQ={bal:.4f}", flush=True)
    except Exception as e:
        elapsed = time.time() - t0
        print(f"[FAILED {elapsed:.1f}s] {type(e).__name__}: {e}", flush=True)


if __name__ == "__main__":
    main()
