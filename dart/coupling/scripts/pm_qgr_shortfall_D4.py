"""pm_qgr_shortfall_D4.py — instrument wheat-tutorial step-1 to localize
the 10× Q_Gr shortfall (notebook 0.00725 vs ours 0.000756 mmol Suc).

Decomposes Q_Gr_dot via:
  Q_Gtot_dot[i] = max(min(Fu_lim - Q_Rm_dot[i], Q_Grmax[i]), 0)
  Fu_lim        = (Q_Rmmax + Q_Grmax) * CSTi/(CSTi + KMfu)
  Q_Rm_dot      = min(Fu_lim, Q_Rmmax)

Exposed via pybind: hm.Q_Grmax, hm.Q_Rmmax, hm.Fl, hm.C_ST. KMfu /
CSTimin / Csoil from configure block. Per-segment ot/st from plant.

Reports Σ over organ class (root/stem/leaf) for substep 1 only.

Run:
  cpbenv/bin/python -u dart/coupling/scripts/pm_qgr_shortfall_D4.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "modelparameter" / "functional"))
sys.path.insert(0, str(REPO_ROOT / "dart/coupling/scripts"))

WHEAT_XML = REPO_ROOT / "modelparameter/structural/plant/Triticum_aestivum_adapted_2023.xml"


def _suppress():
    o1 = os.dup(1); o2 = os.dup(2); dn = os.open(os.devnull, os.O_WRONLY)
    os.dup2(dn, 1); os.dup2(dn, 2)
    return o1, o2, dn


def _restore(o1, o2, dn):
    os.dup2(o1, 1); os.dup2(o2, 2)
    os.close(dn); os.close(o1); os.close(o2)


def node_organ_types(plant):
    """Return per-node organ types (1=seed/0, 2=root, 3=stem, 4=leaf).
    Uses segment ot — each non-root node maps to its parent segment's ot."""
    n_nodes = len(plant.getNodes())
    seg_ot = np.array(plant.organTypes, dtype=int)
    node_ot = np.zeros(n_nodes, dtype=int)
    # node 0 is the seed node; downstream node indices come from segments.
    segments = np.array([(int(s.x), int(s.y)) for s in plant.segments],
                        dtype=int)
    # The "downstream" node of each segment carries the segment's organ type.
    for i, (a, b) in enumerate(segments):
        node_ot[b] = seg_ot[i]
    return node_ot


def main():
    print("=" * 100)
    print("D4: wheat-tutorial step-1 Q_Gr decomposition by organ class")
    print("=" * 100)

    import plantbox as pb
    from climate.dummyWeather import weather

    sim_init = 7.0
    dt = 2.0 / 24.0
    depth = 60.0

    plant = pb.MappedPlant(seednum=2)
    plant.readParameters(str(WHEAT_XML))
    plant.setGeometry(pb.SDF_PlantBox(np.inf, np.inf, depth))
    plant.initialize(False)
    plant.simulate(sim_init, False)
    picker = lambda x, y, z: max(int(np.floor(-z)), -1)
    plant.setSoilGrid(picker)

    n_nodes = len(plant.getNodes())
    n_segs = len(plant.segments)
    n_organs = len(plant.getOrgans(-1, True))
    seg_ot = np.array(plant.organTypes, dtype=int)
    node_ot = node_organ_types(plant)
    print(f"\nWheat plant at sim={sim_init}: nodes={n_nodes} segs={n_segs} "
          f"organs={n_organs}")
    print(f"  Notebook reference reports 936 nodes (D5 will compare).")
    print(f"  Per-organ-type segment count: "
          f"root={int(np.sum(seg_ot==2))} stem={int(np.sum(seg_ot==3))} "
          f"leaf={int(np.sum(seg_ot==4))}")
    print(f"  Per-organ-type node count: "
          f"seed/0={int(np.sum(node_ot==0))} root={int(np.sum(node_ot==2))} "
          f"stem={int(np.sum(node_ot==3))} leaf={int(np.sum(node_ot==4))}")

    # Build PM, configure to wheat tutorial.
    from functional.phloem_flux import PhloemFluxPython
    from pm_vs_s5_wheat_tutorial import tutorial_hydraulic_params_for_weather

    weather_init = weather(sim_init)
    p_mean = weather_init["p_mean"]; p_top = p_mean - depth/2
    hyd = tutorial_hydraulic_params_for_weather(weather_init)
    hm = PhloemFluxPython(plant, hyd, psiXylInit=p_top,
                          ciInit=weather_init["cs"] * 0.5)

    from plant_photosynthesis.wheat_FcVB_Giraud2023adapted import (
        setPhotosynthesisParameters,
    )
    from plant_sucrose.wheat_phloem_Giraud2023adapted import setKrKx_phloem
    setPhotosynthesisParameters(hm, weather_init)
    setKrKx_phloem(hm)
    hm.setKrm2([[2e-5]])
    hm.setKrm1([[10e-2]])
    hm.setRhoSucrose([[0.51], [0.65], [0.56]])
    hm.setRmax_st([[14.4, 9.0, 6.0, 14.4], [5.0, 5.0], [15.0]])
    hm.KMfu = 0.11
    hm.beta_loading = 0.6
    hm.Vmaxloading = 0.05
    hm.Mloading = 0.2
    hm.Gr_Y = 0.8
    hm.CSTimin = 0.4
    hm.Csoil = 1e-4
    hm.update_viscosity = True
    hm.atol = 1e-12
    hm.rtol = 1e-8

    # Run substep 1.
    p_bot = p_mean + depth/2
    sx = np.linspace(p_top, p_bot, int(depth))
    wx = weather(sim_init)
    hm.Qlight = [float(wx["Qlight"])]
    hm.cs = [float(wx["cs"])]
    TleafK = [wx["TairC"] + 273.15]
    fd = _suppress()
    try:
        hm.solve_photosynthesis(sim_time=sim_init, sxx=sx, ea=wx["ea"],
                                es=wx["es"], TleafK=TleafK, cells=True,
                                doLog=False, verbose=False)
    finally:
        _restore(*fd)

    fd = _suppress()
    try:
        ret = hm.startPM(sim_init, sim_init + dt, 1, wx["TairC"] + 273.15,
                         True, str(REPO_ROOT / "dart/coupling/scripts/_pm_d4.txt"))
    finally:
        _restore(*fd)
    print(f"\nstartPM ret: {ret}, dt={dt:.4f}")

    # Pull per-node arrays.
    Q_Grmax = np.array(hm.Q_Grmax)
    Q_Rmmax = np.array(hm.Q_Rmmax)
    Fl      = np.array(hm.Fl)
    C_ST    = np.array(hm.C_ST)
    nt = n_nodes

    # Per-node Fu_lim (mirrors PiafMunch2.cpp:208).
    KMfu = float(hm.KMfu)
    CSTimin = float(hm.CSTimin)
    # CSTi in PiafMunch: CSTi = max(0, C_ST - CSTimin).
    CSTi = np.clip(C_ST - CSTimin, 0.0, None)
    # Q10 normalisation at TairK=295.9 (notebook day-7 TairC≈22.75; but Q_Rmmax
    # already includes the Q10 scaling from runPM). For diagnostic the
    # un-scaled quantity hm.Q_Rmmax is what's exposed; we use it as-is.
    Fu_lim = (Q_Rmmax + Q_Grmax) * (CSTi / (CSTi + KMfu + 1e-30))
    Q_Rm_dot = np.minimum(Fu_lim, Q_Rmmax)
    Q_Gr_dot = np.maximum(np.minimum(Fu_lim - Q_Rm_dot, Q_Grmax), 0.0)
    # Q_Gr cumulative over dt: trapezoidal-like integral isn't exposed,
    # but the per-substep snapshot dot * dt is the linearised proxy.

    # Q_out has 5*Nt entries: ST, meso, Rm, Exud, Gr.
    Q_out = np.array(hm.Q_out)
    cum_Q_Gr = Q_out[4*nt:5*nt]

    print()
    print(f"  CSTimin={CSTimin}  KMfu={KMfu}")
    print(f"  C_ST  mean={C_ST.mean():.4f}  min={C_ST.min():.4f}  max={C_ST.max():.4f}")
    print(f"  CSTi  mean={CSTi.mean():.4f}  min={CSTi.min():.4f}  max={CSTi.max():.4f}")
    print()
    print("  Per organ class (sums over nodes):")
    print(f"    {'class':<10s}{'n_nodes':>10s}{'Σ Q_Grmax':>14s}"
          f"{'Σ Q_Rmmax':>14s}{'Σ Fl':>14s}{'Σ Fu_lim':>14s}"
          f"{'Σ Q_Rm_dot':>14s}{'Σ Q_Gr_dot':>14s}{'Σ cum Q_Gr':>14s}")
    print("  " + "-" * 110)
    for ot, name in [(2, "root"), (3, "stem"), (4, "leaf"), (0, "seed/0")]:
        mask = node_ot == ot
        nm = int(mask.sum())
        if nm == 0: continue
        print(f"    {name:<10s}{nm:>10d}"
              f"{Q_Grmax[mask].sum():>14.4e}{Q_Rmmax[mask].sum():>14.4e}"
              f"{Fl[mask].sum():>14.4e}{Fu_lim[mask].sum():>14.4e}"
              f"{Q_Rm_dot[mask].sum():>14.4e}{Q_Gr_dot[mask].sum():>14.4e}"
              f"{cum_Q_Gr[mask].sum():>14.4e}")
    # Total row
    print("  " + "-" * 110)
    print(f"    {'TOTAL':<10s}{nt:>10d}"
          f"{Q_Grmax.sum():>14.4e}{Q_Rmmax.sum():>14.4e}"
          f"{Fl.sum():>14.4e}{Fu_lim.sum():>14.4e}"
          f"{Q_Rm_dot.sum():>14.4e}{Q_Gr_dot.sum():>14.4e}"
          f"{cum_Q_Gr.sum():>14.4e}")
    print()
    print(f"  Notebook reference (day-7 dt=0.083d): cum Q_Gr = 0.00725 mmol Suc")
    print(f"  Our cum Q_Gr (after first startPM): {cum_Q_Gr.sum():.4e} mmol Suc")
    ratio = cum_Q_Gr.sum() / 0.00725 if cum_Q_Gr.sum() > 0 else 0
    print(f"  Ratio: {ratio:.3f}  (notebook×{1/ratio:.2f} larger)" if ratio > 0 else
          f"  Ratio: 0")

    # Diagnostic deductions.
    print()
    print("  Decomposition checks:")
    if Q_Grmax.sum() > 0:
        clamp_to_grmax = np.sum(np.minimum(Fu_lim - Q_Rm_dot, Q_Grmax) >=
                                Q_Grmax) / np.sum(Q_Grmax > 0)
        print(f"    Fraction of nodes Fu_lim-Q_Rm_dot >= Q_Grmax (Q_Grmax-clamped): "
              f"{clamp_to_grmax:.3f}")
    cuse_zero = float(np.sum(CSTi <= 1e-12)) / nt
    print(f"    Fraction of nodes with CSTi<=0 (cuse-gate closed): {cuse_zero:.3f}")


if __name__ == "__main__":
    main()
