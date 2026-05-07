"""pm_notebook_loop.py — mirror the FSPM 2023 notebook loop pattern
(useCWGr=True, plant.simulate(dt) between PM substeps) and check
whether the entire 'PiafMunch broken on mature plants' diagnostic
arc collapses to a useCWGr=False harness artifact.

The notebook (`tutorial/jupyter/fspm_2023/3_sucrose_flux.ipynb`) loops:

    while sim < simMax:
        weather  = weather(sim)
        r.solve_photosynthesis(...)
        r.startPM(sim, sim+dt, 1, TairK, True, filename)
        # ... inspect Q_out ...
        r.plant.simulate(dt, verbose)         # <-- consumes CW_Gr
        sim += dt

Three targets selectable by --case:

  wheat_tutorial : Reproduce the published Lacointe-Giraud day-7.3
                   regime (BINARY CHECK -- if our PM run matches the
                   notebook reference numbers, the entire
                   useCWGr=False diagnostic chain was a harness bug).

  v3_maize       : V3 (21d) maize under Babst chamber-proxy met,
                   useCWGr=True. Looking for Rg > 0, load_eff > 30%,
                   C_ST_mean > CSTimin (i.e., gate doesn't pin).

  day55_maize    : Day-55 mature maize at saturating PAR. Looking for
                   physical Rg/Rm/Exud at the production scale.

Notebook reference (extracted from notebook output at day 7 0h, dt=0.083d):
  C_ST_mean  : 1.78 mmol/cm^3  (range 1.64 - 1.87)
  Rm         : 0.0104 mmol Suc cumulative over dt
  Gr         : 0.00725 mmol Suc
  Exud       : 0.0379 mmol Suc
  Sink split : Rm 18.8% / Gr 13.0% / Exud 68.2%
  AnSum      : 0.102 mmol Suc

NO useCWGr=False line in this script.  NO solver-side workarounds.
"""

import argparse
import os
import sys
import json
import time as _time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "modelparameter" / "functional"))

from dart.coupling.config import (  # noqa: E402
    DEFAULT_XML, get_hydraulics_json, get_photosynthesis_json, get_phloem_json,
)
from dart.coupling.growth import grow_plant  # noqa: E402

WHEAT_XML = REPO_ROOT / "modelparameter/structural/plant/Triticum_aestivum_adapted_2023.xml"

# Notebook reference (day 7.0 -> 7.083 substep, dt=2/24)
NOTEBOOK_REF = {
    "C_ST_mean":  1.78,
    "C_ST_min":   1.64,
    "C_ST_max":   1.87,
    "Rm":         0.0104,
    "Gr":         0.00725,
    "Exud":       0.0379,
    "AnSum":      0.102,
    "Rm_pct":     18.8,
    "Gr_pct":     13.0,
    "Exud_pct":   68.2,
}

BABST_MET = {
    d: {"T_mean_C": 20.75, "T_min_C": 19.0, "T_max_C": 22.0,
        "PAR_MJ_m2_d": 30.0 * 0.219, "VPD_kPa": 1.0,
        "RH_pct": 60.0, "Wind_m_s": 0.5}
    for d in range(1, 60)
}


def _suppress():
    o1 = os.dup(1); o2 = os.dup(2); dn = os.open(os.devnull, os.O_WRONLY)
    os.dup2(dn, 1); os.dup2(dn, 2)
    return o1, o2, dn


def _restore(o1, o2, dn):
    os.dup2(o1, 1); os.dup2(o2, 2)
    os.close(dn); os.close(o1); os.close(o2)


def build_wheat_tutorial():
    import plantbox as pb
    from climate.dummyWeather import weather

    sim_init = 7.0
    sim_max  = 8.0
    dt       = 2.0 / 24.0
    depth    = 60.0
    weather_init = weather(sim_init)

    plant = pb.MappedPlant(seednum=2)
    plant.readParameters(str(WHEAT_XML))
    plant.setGeometry(pb.SDF_PlantBox(np.inf, np.inf, depth))
    plant.initialize(False)
    plant.simulate(sim_init, False)
    picker = lambda x, y, z: max(int(np.floor(-z)), -1)
    plant.setSoilGrid(picker)
    return plant, dict(sim_init=sim_init, sim_max=sim_max, dt=dt,
                       depth=depth, weather_init=weather_init)


def build_maize(age_days):
    Tair_C = 20.75
    plant = grow_plant(
        xml_path=str(DEFAULT_XML),
        simulation_time=age_days,
        min_stem_nodes=10,
        min_leaf_nodes=4,
        enable_photosynthesis=True,
        seed=42,
        daily_met=BABST_MET,
        T_air_default=Tair_C,
    )
    return plant, dict(sim_init=float(age_days),
                       sim_max=float(age_days) + 1.0,
                       dt=1.0 / 24.0, Tair_C=Tair_C)


def configure_wheat_tutorial(hm, weather_init):
    """Mirror the notebook's r.* configuration (cell 11)."""
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
    return hm


def configure_maize(hm, **overrides):
    hm.read_phloem_parameters(filename=get_phloem_json())
    hm.atol = 1e-6
    hm.rtol = 1e-4
    for k, v in overrides.items():
        setattr(hm, k, v)
    return hm


def run_loop(plant, sim_init, sim_max, dt, hm,
             solve_photosynthesis_fn, Tair_C_at,
             label="run", verbose=False):
    """Mirror the notebook loop. Returns per-substep results."""
    Q_Rmbu   = np.zeros(0)
    Q_Grbu   = np.zeros(0)
    Q_Exudbu = np.zeros(0)
    AnSum = 0.0
    rows = []

    sim = float(sim_init)
    pm_filename = str(REPO_ROOT / "dart/coupling/scripts/_pm_nb_loop.txt")

    while sim <= sim_max + 1e-9:
        nt = len(plant.getNodes())
        Tair_C = float(Tair_C_at(sim))
        Tair_K = Tair_C + 273.15

        # 1) Photosynthesis update.
        solve_photosynthesis_fn(hm, sim, Tair_C)

        # cumulative An via Ag4Phloem * dt (notebook line: AnSum += np.sum(r.Ag4Phloem)*dt)
        Ag = np.array(hm.Ag4Phloem)
        AnSum += float(np.sum(Ag)) * dt

        # 2) PiafMunch substep.
        fdpair = _suppress()
        try:
            ret = hm.startPM(sim, sim + dt, 1, Tair_K, True, pm_filename)
        finally:
            _restore(*fdpair)
        if ret != 1:
            print(f"  startPM returned {ret} at t={sim:.4f}")
            return rows

        # 3) Read state (per notebook cell 13).
        Q_ST   = np.array(hm.Q_out[0:nt])
        Q_meso = np.array(hm.Q_out[nt:(nt*2)])
        Q_Rm   = np.array(hm.Q_out[(nt*2):(nt*3)])
        Q_Exud = np.array(hm.Q_out[(nt*3):(nt*4)])
        Q_Gr   = np.array(hm.Q_out[(nt*4):(nt*5)])
        C_ST   = np.array(hm.C_ST)

        # Per-step deltas
        Q_ST_i   = Q_ST  # not differenced; absolute since AnSum is also absolute
        if Q_Rmbu.size != Q_Rm.size:
            Q_Rmbu   = np.zeros_like(Q_Rm)
            Q_Grbu   = np.zeros_like(Q_Gr)
            Q_Exudbu = np.zeros_like(Q_Exud)
        d_Rm   = float(np.sum(Q_Rm   - Q_Rmbu))
        d_Gr   = float(np.sum(Q_Gr   - Q_Grbu))
        d_Exud = float(np.sum(Q_Exud - Q_Exudbu))
        d_out  = d_Rm + d_Gr + d_Exud

        rows.append(dict(
            sim=sim, dt=dt, AnSum=AnSum,
            sum_Q_ST=float(np.sum(Q_ST)),
            sum_Q_meso=float(np.sum(Q_meso)),
            cum_Q_Rm=float(np.sum(Q_Rm)),
            cum_Q_Gr=float(np.sum(Q_Gr)),
            cum_Q_Exud=float(np.sum(Q_Exud)),
            d_Rm=d_Rm, d_Gr=d_Gr, d_Exud=d_Exud,
            Rm_pct=100*d_Rm / d_out if d_out > 0 else 0,
            Gr_pct=100*d_Gr / d_out if d_out > 0 else 0,
            Exud_pct=100*d_Exud / d_out if d_out > 0 else 0,
            C_ST_mean=float(np.mean(C_ST)),
            C_ST_min=float(np.min(C_ST)),
            C_ST_max=float(np.max(C_ST)),
        ))

        if verbose:
            r = rows[-1]
            print(f"  t={sim:.4f}  An_cum={AnSum:.4e}  "
                  f"Q_Rm={r['cum_Q_Rm']:.4e}  Q_Gr={r['cum_Q_Gr']:.4e}  "
                  f"Q_Exud={r['cum_Q_Exud']:.4e}  C_ST mean/max={r['C_ST_mean']:.3f}/{r['C_ST_max']:.3f}  "
                  f"split Rm/Gr/Exud {r['Rm_pct']:.1f}/{r['Gr_pct']:.1f}/{r['Exud_pct']:.1f}")

        Q_Rmbu   = Q_Rm.copy()
        Q_Grbu   = Q_Gr.copy()
        Q_Exudbu = Q_Exud.copy()

        # 4) Plant grows -- this is the step that consumes CW_Gr and lets useCWGr=True work.
        fdpair = _suppress()
        try:
            plant.simulate(dt, False)
        finally:
            _restore(*fdpair)

        sim += dt

    p = Path(pm_filename)
    if p.exists():
        p.unlink()
    return rows


def case_wheat_tutorial(verbose):
    print("=" * 100)
    print("Wheat day-7.3 notebook reproduction (binary check)")
    print("=" * 100)
    plant, cfg = build_wheat_tutorial()
    weather_init = cfg["weather_init"]

    from functional.phloem_flux import PhloemFluxPython
    sys.path.insert(0, str(REPO_ROOT / "dart/coupling/scripts"))
    from pm_vs_s5_wheat_tutorial import tutorial_hydraulic_params_for_weather

    p_mean = weather_init["p_mean"]; depth = cfg["depth"]
    p_top = p_mean - depth/2

    hyd = tutorial_hydraulic_params_for_weather(weather_init)
    hm = PhloemFluxPython(plant, hyd,
                          psiXylInit=p_top,
                          ciInit=weather_init["cs"] * 0.5)
    configure_wheat_tutorial(hm, weather_init)
    # Notebook does NOT set useCWGr -> default True per runPM.h
    print(f"  useCWGr (default): {bool(hm.useCWGr)}")
    print(f"  Loop: simInit={cfg['sim_init']}, simMax={cfg['sim_max']}, dt={cfg['dt']:.4f}")
    print()

    from climate.dummyWeather import weather
    # tutorial_hydraulic_params_for_weather uses the new
    # PlantHydraulicParameters API (setKr/setKx are no longer on hm
    # in the current build — the wheat helper relies on the old API).
    from pm_vs_s5_wheat_tutorial import tutorial_hydraulic_params_for_weather

    p_bot = p_mean + depth/2
    sx = np.linspace(p_top, p_bot, int(depth))

    def solve_photo(hm_, sim_, _Tair_C):
        wx = weather(sim_)
        hm_.Qlight = [float(wx["Qlight"])]
        hm_.cs = [float(wx["cs"])]
        TleafK = [wx["TairC"] + 273.15]
        fdpair = _suppress()
        try:
            hm_.solve_photosynthesis(
                sim_time=sim_, sxx=sx,
                ea=wx["ea"], es=wx["es"],
                TleafK=TleafK,
                cells=True, doLog=False, verbose=False,
            )
        finally:
            _restore(*fdpair)

    Tair_C_at = lambda sim_: weather(sim_)["TairC"]

    t0 = _time.time()
    rows = run_loop(plant, cfg["sim_init"], cfg["sim_max"], cfg["dt"],
                    hm, solve_photo, Tair_C_at, label="wheat", verbose=verbose)
    wall = _time.time() - t0
    print(f"\nLoop wall: {wall:.1f}s, {len(rows)} substeps")

    if not rows:
        print("Loop produced no rows -- likely solver failure.")
        return None

    first = rows[0]   # day 7.0 -> 7.083 substep
    print()
    print("=" * 100)
    print("FIRST-SUBSTEP COMPARISON TO NOTEBOOK REFERENCE")
    print("=" * 100)
    print(f"{'metric':<22} {'model':>14} {'notebook':>14} {'ratio':>10} {'verdict':>8}")
    print("-" * 70)
    pairs = [
        ("AnSum",      "AnSum",       NOTEBOOK_REF["AnSum"]),
        ("C_ST_mean",  "C_ST_mean",   NOTEBOOK_REF["C_ST_mean"]),
        ("C_ST_min",   "C_ST_min",    NOTEBOOK_REF["C_ST_min"]),
        ("C_ST_max",   "C_ST_max",    NOTEBOOK_REF["C_ST_max"]),
        ("cum Rm",     "cum_Q_Rm",    NOTEBOOK_REF["Rm"]),
        ("cum Gr",     "cum_Q_Gr",    NOTEBOOK_REF["Gr"]),
        ("cum Exud",   "cum_Q_Exud",  NOTEBOOK_REF["Exud"]),
        ("split Rm %", "Rm_pct",      NOTEBOOK_REF["Rm_pct"]),
        ("split Gr %", "Gr_pct",      NOTEBOOK_REF["Gr_pct"]),
        ("split Exud%","Exud_pct",    NOTEBOOK_REF["Exud_pct"]),
    ]
    n_pass = 0
    n_total = 0
    for label, key, ref in pairs:
        val = first[key]
        if ref == 0:
            ratio = float('inf') if val != 0 else 1.0
        else:
            ratio = val / ref
        within = (0.5 <= ratio <= 2.0) if ref != 0 else (val == 0)
        verdict = "PASS" if within else "FAIL"
        n_total += 1
        if within: n_pass += 1
        print(f"{label:<22} {val:>14.4g} {ref:>14.4g} {ratio:>10.3f} {verdict:>8}")

    print("-" * 70)
    print(f"  {n_pass}/{n_total} metrics within 2x of notebook reference.")
    print()
    if n_pass >= 7:
        print(">>> BINARY CHECK PASS: PM with useCWGr=True+plant.simulate matches notebook.")
        print("    Multi-session 'PiafMunch broken' diagnosis was a harness artifact.")
    elif n_pass >= 4:
        print(">>> BINARY CHECK PARTIAL: some metrics match, some don't. Investigate per-row.")
    else:
        print(">>> BINARY CHECK FAIL: PM diverges from notebook even in correct mode.")
        print("    Real model issue persists -- harness fix alone insufficient.")
    return rows


def case_maize(age_days, verbose):
    print("=" * 100)
    print(f"Maize day-{age_days} with useCWGr=True + plant.simulate loop")
    print("=" * 100)
    plant, cfg = build_maize(age_days)
    Tair_C = cfg["Tair_C"]

    from functional.phloem_flux import PhloemFluxPython
    from plantbox.functional.PlantHydraulicParameters import PlantHydraulicParameters
    params_h = PlantHydraulicParameters()
    params_h.read_parameters(get_hydraulics_json())

    hm = PhloemFluxPython(plant, params_h, psiXylInit=-500, ciInit=350e-6)
    hm.read_photosynthesis_parameters(filename=get_photosynthesis_json())
    configure_maize(hm, Vmaxloading=0.20, beta_loading=2.0, solver=32)
    print(f"  useCWGr (default): {bool(hm.useCWGr)}  Vmax=0.20  beta=2.0")
    print(f"  Loop: from {cfg['sim_init']:.1f} to {cfg['sim_max']:.1f}, dt={cfg['dt']*24:.1f}h")

    p_s = np.linspace(-500, -700, 200)

    def solve_photo(hm_, sim_, T):
        es = hm_.get_es(T); ea = es * 0.6
        par = 600.0 * 1e-6 * 86400 * 1e-4
        fdpair = _suppress()
        try:
            hm_.solve(sim_time=sim_, rsx=p_s, cells=True, ea=ea, es=es,
                      PAR=par, TairC=T, verbose=0)
        finally:
            _restore(*fdpair)

    Tair_C_at = lambda sim_: Tair_C
    t0 = _time.time()
    rows = run_loop(plant, cfg["sim_init"], cfg["sim_max"], cfg["dt"],
                    hm, solve_photo, Tair_C_at, label=f"maize_d{age_days}", verbose=verbose)
    wall = _time.time() - t0
    print(f"\nLoop wall: {wall:.1f}s, {len(rows)} substeps")

    if not rows:
        return None

    last = rows[-1]
    total_sinks = last["cum_Q_Rm"] + last["cum_Q_Gr"] + last["cum_Q_Exud"]
    load_eff = total_sinks / last["AnSum"] if last["AnSum"] > 0 else 0
    print()
    print(f"24h cumulative (mmol Suc):")
    print(f"  AnSum   : {last['AnSum']:.3f}")
    print(f"  Rm      : {last['cum_Q_Rm']:.3f}  ({100*last['cum_Q_Rm']/last['AnSum']:.1f}% of An)" if last['AnSum']>0 else "  Rm     : 0")
    print(f"  Gr      : {last['cum_Q_Gr']:.3f}  ({100*last['cum_Q_Gr']/last['AnSum']:.1f}%)" if last['AnSum']>0 else "  Gr     : 0")
    print(f"  Exud    : {last['cum_Q_Exud']:.3f}  ({100*last['cum_Q_Exud']/last['AnSum']:.1f}%)" if last['AnSum']>0 else "  Exud   : 0")
    print(f"  Σsinks/An (load_eff proxy) : {load_eff:.3f}")
    print(f"  C_ST mean/min/max          : {last['C_ST_mean']:.3f} / {last['C_ST_min']:.3f} / {last['C_ST_max']:.3f}")
    print()
    if last["cum_Q_Gr"] > 0.01 * last["AnSum"]:
        print(">>> Rg > 1% of An -- carbon-limited growth path is ACTIVE in this run.")
    else:
        print(">>> Rg still negligible. Issue may extend beyond useCWGr toggle.")
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--case", choices=("wheat_tutorial", "v3_maize", "day55_maize", "all"),
                   default="wheat_tutorial")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    out = {}
    if args.case in ("wheat_tutorial", "all"):
        out["wheat_tutorial"] = case_wheat_tutorial(args.verbose)
    if args.case in ("v3_maize", "all"):
        print()
        out["v3_maize"] = case_maize(21, args.verbose)
    if args.case in ("day55_maize", "all"):
        print()
        out["day55_maize"] = case_maize(55, args.verbose)

    out_json = REPO_ROOT / f"dart/coupling/scripts/_pm_notebook_loop_{args.case}.json"
    with open(out_json, 'w') as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nJSON dump: {out_json}")


if __name__ == "__main__":
    main()
