"""pm_grow_plant_compat_D2.py — confirm wheat tutorial f_gf is
CWLimitedGrowth (gf=3) on all organ types, hence the assertion never
fires. Mirrors D1 but on the notebook's wheat plant.

Run:
  cpbenv/bin/python -u dart/coupling/scripts/pm_grow_plant_compat_D2.py
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


def cw_audit(plant, label):
    by_class = {}
    tripping = 0
    for ot in (2, 3, 4):
        for o in plant.getOrgans(ot, True):
            param = o.getOrganRandomParameter()
            f_gf = getattr(param, "f_gf", None)
            if f_gf is None:
                continue
            cls = type(f_gf).__name__
            d = by_class.setdefault(cls, dict(n=0, n_active=0,
                                              n_missing=0, n_pos=0, n_neg=0))
            d["n"] += 1
            if o.isActive():
                d["n_active"] += 1
            cw = f_gf.CW_Gr
            oid = o.getId()
            if oid not in cw:
                d["n_missing"] += 1
            else:
                v = cw[oid]
                if v >= 0:
                    d["n_pos"] += 1
                    if o.isActive():
                        tripping += 1
                else:
                    d["n_neg"] += 1
    print(f"  [{label}] CW_Gr by f_gf class:")
    for cls, d in by_class.items():
        print(f"    {cls:32s} n={d['n']:4d} active={d['n_active']:4d} "
              f"pos={d['n_pos']:4d} neg={d['n_neg']:4d} missing={d['n_missing']:4d}")
    print(f"  [{label}] tripping count: {tripping}")
    return tripping


def main():
    print("=" * 100)
    print("D2: Wheat tutorial CW_Gr lifecycle — does it ever trip?")
    print("=" * 100)

    import plantbox as pb
    from climate.dummyWeather import weather

    sim_init = 7.0
    sim_max = 8.0
    dt = 2.0 / 24.0
    depth = 60.0

    plant = pb.MappedPlant(seednum=2)
    plant.readParameters(str(WHEAT_XML))
    plant.setGeometry(pb.SDF_PlantBox(np.inf, np.inf, depth))
    plant.initialize(False)
    plant.simulate(sim_init, False)
    picker = lambda x, y, z: max(int(np.floor(-z)), -1)
    plant.setSoilGrid(picker)

    print("\n--- After raw MappedPlant.simulate(7) (BEFORE PM) ---")
    cw_audit(plant, "post-grow")
    print("\n  XML gf inventory:")
    for ot in (2, 3, 4):
        gfs = set()
        for p in plant.getOrganRandomParameter(ot):
            if p is None: continue
            gfs.add((int(getattr(p, "gf", -1)),
                     type(getattr(p, "f_gf", None)).__name__))
        print(f"    ot={ot}: {sorted(gfs)}")

    # Build PM and run substep 1.
    from functional.phloem_flux import PhloemFluxPython
    from pm_vs_s5_wheat_tutorial import tutorial_hydraulic_params_for_weather

    weather_init = weather(sim_init)
    p_mean = weather_init["p_mean"]
    p_top = p_mean - depth/2

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

    print(f"\nuseCWGr (default): {bool(hm.useCWGr)}")

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

    print("\n--- Substep 1 startPM ---")
    fd = _suppress()
    try:
        ret = hm.startPM(sim_init, sim_init + dt, 1, wx["TairC"] + 273.15,
                         True, str(REPO_ROOT / "dart/coupling/scripts/_pm_d2.txt"))
    finally:
        _restore(*fd)
    print(f"  startPM ret: {ret}")

    print("\n--- AFTER startPM (PM has filled CW_Gr) ---")
    cw_audit(plant, "after-startPM-sub1")

    print("\n--- plant.simulate(dt) ---")
    fd = _suppress()
    try:
        plant.simulate(dt, False)
    finally:
        _restore(*fd)

    print("\n--- AFTER plant.simulate ---")
    tripping = cw_audit(plant, "after-plant-simulate")

    print()
    print("=" * 100)
    if tripping == 0:
        print("VERDICT: wheat tutorial — CWLimitedGrowth consumes CW_Gr → assertion never trips.")
        print("This explains why the FSPM 2023 notebook works without intervention.")
    else:
        print(f"VERDICT: {tripping} organs would trip — wheat is NOT immune. Investigate.")


if __name__ == "__main__":
    main()
