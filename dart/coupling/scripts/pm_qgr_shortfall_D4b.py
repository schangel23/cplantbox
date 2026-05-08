"""pm_qgr_shortfall_D4b.py — drill-down into D4 finding (stem+leaf
Q_Grmax = 0 in wheat-tutorial).

Reproduces D4 setup, then *after startPM* iterates organs printing:
  - organ id, ot, st, age, length, lmax, num_growing_nodes
  - deltaSucOrgNode_ entries that landed on its nodes
  - the f_gf class and what getLength returns at age and age+dt

Goal: identify whether the zero-Q_Grmax-on-stem/leaf is due to
  (i)   getGrowingNodes returning empty for stems/leaves
  (ii)  f_gf->getLength(age+dt) - f_gf->getLength(age) ≈ 0
  (iii) Fpsi == 0 on the relevant nodes (water-stress gate)
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


def main():
    print("=" * 100)
    print("D4b: per-organ deltaSucOrgNode + getLength delta on wheat-tutorial step-1")
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
                         True, str(REPO_ROOT / "dart/coupling/scripts/_pm_d4b.txt"))
    finally:
        _restore(*fd)
    print(f"\nstartPM ret: {ret}")

    # deltaSucOrgNode is exposed; type is std::vector<std::map<int,double>>.
    # Each entry: per-node map[orgID -> deltaSuc] with [-1] as total.
    dsn = hm.deltaSucOrgNode

    print("\nPer-organ summary:")
    print(f"  {'oid':>4} {'ot':>3} {'st':>3} {'age':>7} {'len':>7} {'lmax':>7} "
          f"{'k':>7} {'r':>6} {'fgf':>16} {'getLen(age)':>13} "
          f"{'getLen(age+dt)':>15} {'deltaL':>9} {'Σ deltaSuc':>12}")
    print("  " + "-" * 130)
    for ot in (2, 3, 4):
        for org in plant.getOrgans(ot, True):
            oid = org.getId()
            age = org.getAge()
            length = org.getLength(False)
            param = org.getOrganRandomParameter()
            fgf = type(getattr(param, "f_gf", None)).__name__
            r = float(org.getParameter("r"))
            k = float(org.getParameter("k"))
            try:
                lmax = float(org.getParameter("lmax"))
            except Exception:
                lmax = float("nan")
            # getLength on the GF: needs an organ ptr.
            try:
                gf = param.f_gf
                # GF::getLength(t, r, k, o)
                lAge = gf.getLength(age, r, k, org)
                lAgePlus = gf.getLength(age + dt, r, k, org)
                deltaL = max(0.0, lAgePlus - lAge)
            except Exception as e:
                lAge = lAgePlus = float("nan")
                deltaL = float("nan")

            # Sum deltaSucOrgNode over all nodes for this orgID.
            sum_ds = 0.0
            for nmap in dsn:
                if oid in nmap:
                    sum_ds += nmap[oid]

            print(f"  {oid:>4} {ot:>3d} {int(org.getParameter('subType')):>3d} "
                  f"{age:>7.3f} {length:>7.3f} {lmax:>7.3f} "
                  f"{k:>7.3f} {r:>6.3f} {fgf:>16s} {lAge:>13.4f} "
                  f"{lAgePlus:>15.4f} {deltaL:>9.4e} {sum_ds:>12.4e}")

    # Probe Fpsi distribution.
    Q_Grmax = np.array(hm.Q_Grmax)
    print(f"\nQ_Grmax stats:")
    print(f"  total non-zero nodes: {int((Q_Grmax > 0).sum())}/{len(Q_Grmax)}")
    print(f"  max: {Q_Grmax.max():.4e}, sum: {Q_Grmax.sum():.4e}")


if __name__ == "__main__":
    main()
