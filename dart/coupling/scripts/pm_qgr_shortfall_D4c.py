"""pm_qgr_shortfall_D4c — verify Fpsi vs psiXyl is responsible for
zero Q_Grmax on stem+leaves in our wheat-tutorial run.
"""
from __future__ import annotations
import os, sys
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "modelparameter" / "functional"))
sys.path.insert(0, str(REPO_ROOT / "dart/coupling/scripts"))
WHEAT_XML = REPO_ROOT / "modelparameter/structural/plant/Triticum_aestivum_adapted_2023.xml"


def _suppress():
    o1 = os.dup(1); o2 = os.dup(2); dn = os.open(os.devnull, os.O_WRONLY)
    os.dup2(dn, 1); os.dup2(dn, 2); return o1, o2, dn

def _restore(o1, o2, dn):
    os.dup2(o1, 1); os.dup2(o2, 2); os.close(dn); os.close(o1); os.close(o2)

def main():
    print("=" * 100)
    print("D4c: Fpsi vs psiXyl per organ class on wheat tutorial step-1")
    print("=" * 100)

    import plantbox as pb
    from climate.dummyWeather import weather

    sim_init = 7.0; dt = 2.0/24.0; depth = 60.0
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
                          ciInit=weather_init["cs"]*0.5)
    from plant_photosynthesis.wheat_FcVB_Giraud2023adapted import setPhotosynthesisParameters
    from plant_sucrose.wheat_phloem_Giraud2023adapted import setKrKx_phloem
    setPhotosynthesisParameters(hm, weather_init); setKrKx_phloem(hm)
    hm.setKrm2([[2e-5]]); hm.setKrm1([[10e-2]])
    hm.setRhoSucrose([[0.51], [0.65], [0.56]])
    hm.setRmax_st([[14.4, 9.0, 6.0, 14.4], [5.0, 5.0], [15.0]])
    hm.KMfu = 0.11; hm.beta_loading = 0.6; hm.Vmaxloading = 0.05
    hm.Mloading = 0.2; hm.Gr_Y = 0.8; hm.CSTimin = 0.4; hm.Csoil = 1e-4
    hm.update_viscosity = True; hm.atol = 1e-12; hm.rtol = 1e-8

    p_bot = p_mean + depth/2
    sx = np.linspace(p_top, p_bot, int(depth))
    wx = weather(sim_init)
    hm.Qlight = [float(wx["Qlight"])]; hm.cs = [float(wx["cs"])]
    fd = _suppress()
    try:
        hm.solve_photosynthesis(sim_time=sim_init, sxx=sx, ea=wx["ea"],
                                es=wx["es"], TleafK=[wx["TairC"]+273.15],
                                cells=True, doLog=False, verbose=False)
    finally: _restore(*fd)
    fd = _suppress()
    try:
        hm.startPM(sim_init, sim_init+dt, 1, wx["TairC"]+273.15, True,
                   str(REPO_ROOT/"dart/coupling/scripts/_pm_d4c.txt"))
    finally: _restore(*fd)

    psiXyl = np.array(hm.psiXyl)
    Fpsi = np.array(hm.Fpsi)
    psi_osmo_proto = float(hm.psi_osmo_proto)
    psiMin = float(hm.psiMin)
    print(f"  psi_osmo_proto = {psi_osmo_proto:.2f} cm")
    print(f"  psiMin         = {psiMin:.2f} cm")
    print(f"  hyd water-pot regime")
    print(f"  psiXyl: mean={psiXyl.mean():.1f}, min={psiXyl.min():.1f}, "
          f"max={psiXyl.max():.1f}")
    print(f"  Fpsi: mean={Fpsi.mean():.4f}, min={Fpsi.min():.4f}, "
          f"max={Fpsi.max():.4f}")
    print(f"  Fpsi==0 nodes: {int((Fpsi==0).sum())}/{len(Fpsi)}")
    print(f"  Fpsi>0 nodes:  {int((Fpsi>0).sum())}/{len(Fpsi)}")

    # Per organ class
    seg_ot = np.array(plant.organTypes, dtype=int)
    n_nodes = len(plant.getNodes())
    node_ot = np.zeros(n_nodes, dtype=int)
    for i, s in enumerate(plant.segments):
        node_ot[int(s.y)] = seg_ot[i]
    print()
    print("  Per organ class:")
    print(f"    {'class':<10s}{'n':>6s}{'psiXyl mean':>14s}"
          f"{'psiXyl min':>14s}{'Fpsi mean':>12s}{'Fpsi>0':>10s}")
    print("  " + "-" * 75)
    for ot, name in [(2,"root"), (3,"stem"), (4,"leaf"), (0,"seed/0")]:
        m = node_ot == ot
        n = int(m.sum())
        if n == 0: continue
        print(f"    {name:<10s}{n:>6d}{psiXyl[m].mean():>14.1f}"
              f"{psiXyl[m].min():>14.1f}{Fpsi[m].mean():>12.4f}"
              f"{int((Fpsi[m]>0).sum()):>10d}")

    print()
    print(f"  Q_Grmax per node:")
    Q_Grmax = np.array(hm.Q_Grmax)
    for ot, name in [(2,"root"), (3,"stem"), (4,"leaf"), (0,"seed/0")]:
        m = node_ot == ot
        n = int(m.sum())
        if n == 0: continue
        print(f"    {name:<10s}: nonzero={int((Q_Grmax[m]>0).sum())}/{n}, "
              f"sum={Q_Grmax[m].sum():.4e}")

if __name__ == "__main__":
    main()
