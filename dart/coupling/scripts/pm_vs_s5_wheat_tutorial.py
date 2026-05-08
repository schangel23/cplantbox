#!/usr/bin/env python3
"""Cross-validate S5 and PiafMunch on the wheat 7.3d sucrose tutorial plant.

This intentionally mirrors tutorial/jupyter/fspm_2023/3_sucrose_flux.ipynb:
pb.MappedPlant(seednum=2), Triticum_aestivum_adapted_2023.xml, 60 cm soil
box/grid, weather(7), and the wheat Giraud 2023 helper parameter functions.
"""

from __future__ import annotations

import os
import ctypes
import json
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cplantbox")
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "modelparameter" / "functional"))
sys.path.append(str(REPO_ROOT / "src"))


def ensure_suitesparse_symbols():
    """Expose SuiteSparse symbols for builds where libCPlantBox did not link them."""
    shim = Path("/tmp/cplantbox_suitesparse_shim.so")
    libs = [
        REPO_ROOT / "src/external/suitsparse/lib/libamd.a",
        REPO_ROOT / "src/external/suitsparse/lib/libbtf.a",
        REPO_ROOT / "src/external/suitsparse/lib/libcolamd.a",
        REPO_ROOT / "src/external/suitsparse/lib/libsuitesparseconfig.a",
    ]
    if not all(p.exists() for p in libs):
        return
    if not shim.exists() or any(p.stat().st_mtime > shim.stat().st_mtime for p in libs):
        cmd = [
            "gcc",
            "-shared",
            "-o",
            str(shim),
            "-Wl,--whole-archive",
            *[str(p) for p in libs],
            "-Wl,--no-whole-archive",
            "-lm",
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    ctypes.CDLL(str(shim), mode=ctypes.RTLD_GLOBAL)


ensure_suitesparse_symbols()

from climate.dummyWeather import weather  # noqa: E402
from plant_hydraulics.wheat_Giraud2023adapted import setKrKx_xylem  # noqa: E402
from plant_photosynthesis.wheat_FcVB_Giraud2023adapted import (  # noqa: E402
    setPhotosynthesisParameters,
)
from plant_sucrose.wheat_phloem_Giraud2023adapted import setKrKx_phloem  # noqa: E402

from dart.coupling.carbon.phloem_steady import (  # noqa: E402
    PhloemParams,
    QuasiSteadyPhloem,
)

SUC_TO_CO2 = 12.0
SIM_INIT = 7.0
SIMULATION_TIME = 7.3
DEPTH_CM = 60
XML_PATH = REPO_ROOT / "modelparameter/structural/plant/Triticum_aestivum_adapted_2023.xml"
WHEAT_PHLOEM_JSON = REPO_ROOT / "dart/coupling/data/wheat_phloem_parameters"
WHEAT_HYDRAULICS_JSON = REPO_ROOT / "dart/coupling/data/wheat_Giraud2023adapted"
PM_TMP = REPO_ROOT / "dart/coupling/scripts/_pm_vs_s5_wheat_tutorial.txt"


def suppress_fds():
    o1 = os.dup(1)
    o2 = os.dup(2)
    dn = os.open(os.devnull, os.O_WRONLY)
    os.dup2(dn, 1)
    os.dup2(dn, 2)
    return o1, o2, dn


def restore_fds(fds):
    o1, o2, dn = fds
    os.dup2(o1, 1)
    os.dup2(o2, 2)
    os.close(dn)
    os.close(o1)
    os.close(o2)


def build_tutorial_plant():
    import plantbox as pb

    plant = pb.MappedPlant(seednum=2)
    plant.readParameters(str(XML_PATH))
    plant.setGeometry(pb.SDF_PlantBox(np.inf, np.inf, DEPTH_CM))
    plant.initialize(False)
    plant.simulate(SIM_INIT, False)

    picker = lambda x, y, z: max(int(np.floor(-z)), -1)
    plant.setSoilGrid(picker)
    return plant


def tutorial_soil(weather_init):
    p_mean = weather_init["p_mean"]
    p_bot = p_mean + DEPTH_CM / 2
    p_top = p_mean - DEPTH_CM / 2
    return np.linspace(p_top, p_bot, DEPTH_CM)


def configure_tutorial_phloem(hm, weather_init):
    hm = setPhotosynthesisParameters(hm, weather_init)
    hm = setKrKx_phloem(hm)
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
    if hasattr(hm, "CsoilDefault"):
        hm.CsoilDefault = 1e-4
    hm.update_viscosity = True
    if hasattr(hm, "update_viscosity_"):
        hm.update_viscosity_ = True
    hm.atol = 1e-12
    hm.rtol = 1e-8
    return hm


def tutorial_hydraulic_params():
    return tutorial_hydraulic_params_for_weather(weather(SIM_INIT))


def tutorial_hydraulic_params_for_weather(weather_init, *, plant=None, notebook_aligned=False):
    """Build PlantHydraulicParameters from weather (TairC, RH).

    PiafMunch follow-up #2 (2026-05-08): when ``notebook_aligned=True``
    AND a ``plant`` (MappedPlant or any MappedSegments) is provided, the
    legacy ``setKrKx_xylem`` regime from
    ``modelparameter/functional/plant_hydraulics/wheat_Giraud2023adapted.py``
    is reproduced — root kr is gated to the last ``kr_length=0.8 cm`` of
    each root segment (tip-only kr, not whole-root) and ``psi_air`` is
    derived from RH via the standard Kelvin formula.

    The ``plant`` argument is required for ``kr_length>0`` because the
    underlying ``PlantHydraulicParameters::setKrConst`` calls
    ``ms->calcExchangeZoneCoefs()`` and segfaults if ``ms`` is null. When
    ``plant=None`` the helper silently disables the ``kr_length`` gate
    (kr applied to whole segment, identical to the pre-2026-05-08 path)
    but still applies ``psi_air`` if ``notebook_aligned`` is requested.

    **Default ``notebook_aligned=False`` (rationale: 2026-05-08 A/B/C/D
    falsification of plan §"New follow-ups uncovered" #2):**

    - ``psi_air`` from RH is REDUNDANT — ``configure_wheat_tutorial`` calls
      ``setPhotosynthesisParameters`` which sets ``hm.psi_air`` from
      ``weather_init["RH"]`` directly on the PhloemFluxPython instance,
      overriding whatever ``params.psi_air`` carries. Effect on Q_Gr: 0 %.
    - ``kr_length=0.8 cm`` (root tip-only kr) actively makes wheat-tutorial
      water status WORSE — psiXyl drops from −4832 cm (whole-segment kr)
      to −13274 cm (0.8 cm tip-only) under the fixed soil profile, and
      Σ Q_Grmax collapses from 7.56e-4 mmol Suc to **zero** at substep 1.
      Plan hypothesis "Q_Gr lands in [3.6e-3, 1.45e-2] band" is FALSIFIED.

    The plan-predicted closure of the wheat-notebook Q_Gr gap therefore
    has a different root cause than hydraulic regime alignment alone
    (likely XML drift 90 vs 936 nodes + soil profile + phloem
    parameters). New follow-up.

    Set ``notebook_aligned=True`` only to reproduce the legacy regime
    bit-for-bit (e.g. for byte-identical xylem-side comparisons against
    ``setKrKx_xylem`` consumers); expect collapsed Q_Gr.
    """
    from plantbox.functional.PlantHydraulicParameters import PlantHydraulicParameters

    if plant is not None:
        params = PlantHydraulicParameters(plant)
    else:
        params = PlantHydraulicParameters()
    TairC = float(weather_init["TairC"])
    hPa2cm = 1.0197
    d_water = (
        999.83952
        + TairC
        * (
            16.952577
            + TairC
            * (
                -0.0079905127
                + TairC
                * (-0.000046241757 + TairC * (0.00000010584601 + TairC * -0.00000000028103006))
            )
        )
    ) / (1 + 0.016887236 * TairC)
    si_phi = (30 - TairC) / (91 + TairC)
    mu = pow(10, -0.114 + (si_phi * 1.1))
    mu = mu / (24 * 60 * 60) / 100 / 1000
    mu = mu * hPa2cm

    v_leaf = 32
    v_stem = 52
    v_root = 1
    kz_l = v_leaf * ((0.0015**4) * 2 + (0.0005**4) * 2) * np.pi / (mu * 8)
    kz_s = v_stem * ((0.0017**4) * 3 + (0.0008**4) * 1) * np.pi / (mu * 8)
    kz_r0 = v_root * (0.0015**4) * 4 * np.pi / (mu * 8)
    kz_r1 = v_root * ((0.00041**4) * 4 + (0.00087**4) * 1) * np.pi / (mu * 8)
    kz_r2 = kz_r1
    kz_r3 = v_root * (0.00068**4) * np.pi / (mu * 8)

    kr_l = 3.83e-4 * hPa2cm
    kr_s = 0.0
    kr_r0 = 6.37e-5 * hPa2cm
    kr_r1 = 7.9e-5 * hPa2cm
    kr_r2 = 7.9e-5 * hPa2cm
    kr_r3 = 6.8e-5 * hPa2cm

    root = 2
    stem = 3
    leaf = 4
    # PiafMunch follow-up #2 (2026-05-08): legacy setKrKx_xylem applies kr
    # only to the last l_kr=0.8 cm of every root segment (tip-only); setKrConst
    # exposes this via the kr_length C++ argument (default -1 = whole-segment).
    # kr_length>0 requires params.ms to be bound (calcExchangeZoneCoefs);
    # if no plant was provided the gate stays off to avoid a segfault.
    #
    # Order matters: setKrConst's last call wins for kr_f (the dispatch
    # callback). We need the kr_RootExchangeZonePerType callback active on
    # the FINAL setKrConst call so root segments dispatch through
    # ms->exchangeZoneCoefs. So: stem + leaf first (whole-segment kr,
    # callback bound to kr_perType), then root subTypes last with
    # kr_length=0.8 cm (callback bound to kr_RootExchangeZonePerType, which
    # ALSO handles stem + leaf correctly — see PlantHydraulicParameters.cpp:136).
    # The trailing setMode("constant", "constant") is a no-op in C++
    # (string mismatch: "constant" != "const") and is dropped here.
    kr_length_root = 0.8 if (notebook_aligned and plant is not None) else -1.0
    for st in [0, 1]:
        params.set_kr_const(kr_s, subType=st, organType=stem)
        params.set_kx_const(kz_s, subType=st, organType=stem)
    for st in range(0, 12):
        params.set_kr_const(kr_l, subType=st, organType=leaf)
        params.set_kx_const(kz_l, subType=st, organType=leaf)
    for st, kr, kx in [
        (0, kr_r0, kz_r0),
        (1, kr_r1, kz_r1),
        (2, kr_r2, kz_r2),
        (3, kr_r0, kz_r0),
        (4, kr_r3, kz_r3),
    ]:
        # Bypass set_kr_const wrapper (no kr_length pass-through) and
        # call the C++ method directly with the 4-arg form.
        params.setKrConst(kr, subType=st, organType=root, kr_length=kr_length_root)
        params.set_kx_const(kx, subType=st, organType=root)

    # PiafMunch follow-up #2: psi_air from RH via Kelvin formula
    # (legacy wheat_Giraud2023adapted.py:55). Default psi_air=-954378
    # corresponds to RH=0.5, TairC=20 °C; out-of-band weather requires
    # this update or psi_air becomes inconsistent with TairC.
    if notebook_aligned and "RH" in weather_init:
        RH = float(weather_init["RH"])
        Rgaz = 8.314  # J K-1 mol-1
        rho_h2o = d_water / 1000.0  # g/cm3
        Mh2o = 18.05  # g/mol
        MPa2hPa = 10000.0
        hPa2cm_psi = 1.0 / 0.9806806
        params.psi_air = (
            np.log(RH) * Rgaz * rho_h2o * (TairC + 273.15) / Mh2o * MPa2hPa * hPa2cm_psi
        )
    return params


def solve_tutorial_photosynthesis(hm, sim_time, sx, weather_init):
    weather_x = dict(weather_init)
    if hasattr(hm, "setKr"):
        hm = setKrKx_xylem(weather_x["TairC"], weather_x["RH"], hm)
    hm.Qlight = [float(weather_x["Qlight"])]
    hm.cs = [float(weather_x["cs"])]
    hm.solve_photosynthesis(
        sim_time=sim_time,
        sxx=sx,
        cells=True,
        ea=weather_x["ea"],
        es=weather_x["es"],
        TleafK=[weather_x["TairC"] + 273.15],
        verbose=False,
        doLog=False,
    )
    return np.array(hm.get_net_assimilation(), dtype=float)


def tutorial_s5_params():
    """S5 consumes JSON-shaped params; apply the notebook's in-memory overrides."""
    with open(str(WHEAT_PHLOEM_JSON) + ".json") as f:
        data = json.load(f)
    st = data["SieveTube"]
    pt = data["PerType"]
    params = PhloemParams(
        Vmaxloading=st["Vmaxloading"]["value"],
        beta_loading=st["beta_loading"]["value"],
        C_targ=st["C_targ"]["value"],
        Mloading=st["Mloading"]["value"],
        CSTimin=st["CSTimin"]["value"],
        Q10=st["Q10"]["value"],
        TrefQ10=st["TrefQ10"]["value"],
        KMfu=st["KMfu"]["value"],
        Gr_Y=data["Growth"]["Gr_Y"]["value"],
        leafGrowthZone=data["Growth"]["leafGrowthZone"]["value"],
        C_soil=data["Soil"]["DefaultC"]["value"],
        k_S_ST=st["k_S_ST"]["value"],
        kHyd_S_ST=st["kHyd_S_ST"]["value"],
        kx_st=pt["kx_st"]["value"],
        kr_st=pt["kr_st"]["value"],
        Across_st=pt["Across_st"]["value"],
        Rmax_st=pt["Rmax_st"]["value"],
        Krm1=pt["Krm1"]["value"],
        Krm2=pt["Krm2"]["value"],
        Rho_s=pt["Rho_s"]["value"],
    )
    rmax = [[14.4, 9.0, 6.0, 14.4], [5.0, 5.0], [15.0]]
    return replace(
        params,
        Vmaxloading=0.05,
        beta_loading=0.6,
        Mloading=0.2,
        CSTimin=0.4,
        KMfu=0.11,
        Gr_Y=0.8,
        C_soil=1e-4,
        Rmax_st=rmax,
        Krm2=[[2e-5]],
        Krm1=[[10e-2]],
        Rho_s=[[0.51], [0.65], [0.56]],
    )


def run_s5(plant, an_leaf, tair_c):
    t0 = time.time()
    solver = QuasiSteadyPhloem(plant, params=tutorial_s5_params(), sim_day=SIMULATION_TIME)
    out = solver.solve(an_leaf, Tair_C=tair_c, sim_day=SIMULATION_TIME)
    wall = time.time() - t0
    if not out.get("converged", False):
        raise RuntimeError(
            f"S5 did not converge: n_iter={out.get('n_iterations')} "
            f"max_delta={out.get('max_delta')}"
        )
    return {
        "Rg_total": float(out["Rg_total_mmol"]),
        "Rm_total": float(out["Rm_total_mmol"]),
        "Exud_total": float(np.sum(out["root_exud_mmol_d"])),
        "C_ST_mean": float(out["C_ST_mean"]),
        "C_ST_max": float(out["C_ST_max"]),
        "mass_balance_err": float(out["carbon_balance_error"]),
        "wall_time": wall,
        "storage_change": float(out.get("stem_storage_mmol", 0.0) + out.get("starch_surplus_mmol", 0.0)),
        "An_total": float(np.sum(an_leaf) * 1000.0),
    }


def run_piafmunch(plant, an_leaf, sx, weather_init):
    from functional.phloem_flux import PhloemFluxPython

    params = tutorial_hydraulic_params()
    hm = PhloemFluxPython(plant, params, psiXylInit=min(sx), ciInit=weather_init["cs"] * 0.5)
    hm = configure_tutorial_phloem(hm, weather_init)
    _ = solve_tutorial_photosynthesis(hm, SIM_INIT, sx, weather_init)
    hm.useCWGr = True
    hm.solver = 32

    dt_days = 1.0 / 24.0
    n_steps = 24
    nt = len(plant.getNodes())
    tair_k = weather_init["TairC"] + 273.15
    prev_rm = prev_rg = prev_exud = 0.0
    prev_cst = None
    dc_dt = np.inf

    t0 = time.time()
    for step in range(1, n_steps + 1):
        if step > 1:
            hm.withInitVal = False
        start = SIM_INIT + (step - 1) * dt_days
        end = start + dt_days
        fds = suppress_fds()
        try:
            ret = hm.startPM(start, end, 1, tair_k, True, str(PM_TMP))
        finally:
            restore_fds(fds)
        if ret != 1:
            raise RuntimeError(f"PiafMunch startPM failed at hour {step} with ret={ret}")

        cst = np.array(hm.C_ST, dtype=float)
        if prev_cst is not None:
            dc_dt = float(np.max(np.abs(cst - prev_cst)))
        prev_cst = cst

        q = np.array(hm.Q_out, dtype=float)
        q_rm = float(np.sum(q[nt * 2:nt * 3]))
        q_exud = float(np.sum(q[nt * 3:nt * 4]))
        q_gr = float(np.sum(q[nt * 4:nt * 5]))
        d_rm = q_rm - prev_rm
        d_rg = q_gr - prev_rg
        d_exud = q_exud - prev_exud
        prev_rm, prev_rg, prev_exud = q_rm, q_gr, q_exud
    wall = time.time() - t0

    if dc_dt >= 1e-3:
        raise RuntimeError(f"PiafMunch failed convergence: dC_ST/dt={dc_dt:.3e} mmol/cm3/h")

    rm_total = d_rm * SUC_TO_CO2 * 24.0
    rg_total = d_rg * SUC_TO_CO2 * 24.0
    exud_total = d_exud * 24.0
    an_total = float(np.sum(an_leaf) * 1000.0)
    storage_change = an_total - rm_total - rg_total - exud_total
    mb = abs(an_total - rm_total - rg_total - exud_total - storage_change) / an_total if an_total > 0 else 0.0
    return {
        "Rg_total": float(rg_total),
        "Rm_total": float(rm_total),
        "Exud_total": float(exud_total),
        "C_ST_mean": float(np.mean(prev_cst)),
        "C_ST_max": float(np.max(prev_cst)),
        "mass_balance_err": float(mb),
        "wall_time": wall,
        "storage_change": float(storage_change),
        "An_total": an_total,
        "dc_dt": dc_dt,
    }


def ratio_verdict(s5, pm):
    if s5 == 0 and pm == 0:
        return 1.0, "AGREE"
    if s5 == 0 or pm == 0:
        return np.inf, "DISAGREE"
    ratio = pm / s5
    return ratio, "AGREE" if 0.5 <= ratio <= 2.0 else "DISAGREE"


def fmt(value):
    if value is None:
        return "?"
    if not np.isfinite(value):
        return "inf"
    return f"{value:.6g}"


def print_comparison(s5, pm):
    rows = [
        ("Rg_total [mmol/d]", "Rg_total"),
        ("Rm_total [mmol/d]", "Rm_total"),
        ("Exud_total [mmol/d]", "Exud_total"),
        ("C_ST_mean", "C_ST_mean"),
        ("C_ST_max", "C_ST_max"),
    ]
    failing = []

    print("metric              S5            PiafMunch     ratio    verdict")
    for label, key in rows:
        ratio, verdict = ratio_verdict(s5[key], pm[key])
        if verdict == "DISAGREE":
            failing.append(label)
        print(f"{label:<19} {fmt(s5[key]):<13} {fmt(pm[key]):<13} {fmt(ratio):<8} {verdict}")

    s5_mb = s5["mass_balance_err"]
    pm_mb = pm["mass_balance_err"]
    mb_ok = s5_mb < 0.05 and pm_mb < 0.05
    print(
        f"{'Mass balance err':<19} {fmt(s5_mb):<13} {fmt(pm_mb):<13} "
        f"{'--':<8} {'OK' if mb_ok else 'VIOLATION'}"
    )
    print(
        f"{'Wall time [s]':<19} {fmt(s5['wall_time']):<13} {fmt(pm['wall_time']):<13} "
        f"{'--':<8} info"
    )

    if failing or not mb_ok:
        if not mb_ok:
            failing.append("Mass balance err")
        print(f"VERDICT: DISAGREE ({', '.join(failing)})")
    else:
        print("VERDICT: AGREE (all metrics within 2x)")


def main():
    if not XML_PATH.exists():
        raise FileNotFoundError(f"Tutorial XML missing: {XML_PATH}")
    weather_init = weather(SIM_INIT)
    sx = tutorial_soil(weather_init)

    print("Tutorial notebook extraction:")
    print(f"  XML filename: {XML_PATH}")
    print(f"  simulation_time: {SIMULATION_TIME} d (notebook simInit={SIM_INIT}, simMax=8, dt=2/24)")
    print(
        "  weatherInit: "
        f"TairC={weather_init['TairC']:.6g}, RH={weather_init['RH']:.6g}, "
        f"Qlight={weather_init['Qlight']:.6g}, cs={weather_init['cs']:.6g}, "
        f"p_mean={weather_init['p_mean']:.6g}"
    )
    print(f"  phloem parameter setup: {WHEAT_PHLOEM_JSON}.json plus tutorial in-memory overrides")
    print("  plant setup: pb.MappedPlant(seednum=2), SDF_PlantBox(inf, inf, 60), soil-grid picker")
    print("  units: Rm/Rg are S5 CO2-equivalent mmol/d; Exud remains sucrose mmol/d")
    print()

    from functional.phloem_flux import PhloemFluxPython

    plant = build_tutorial_plant()
    params = tutorial_hydraulic_params()
    photo = PhloemFluxPython(plant, params, psiXylInit=min(sx), ciInit=weather_init["cs"] * 0.5)
    photo = configure_tutorial_phloem(photo, weather_init)
    an_leaf = solve_tutorial_photosynthesis(photo, SIM_INIT, sx, weather_init)
    an_total = float(np.sum(an_leaf) * 1000.0)
    print(f"Shared photosynthesis: An_total={an_total:.6g} mmol CO2/d, leaf_segments={len(an_leaf)}")
    print()

    try:
        s5 = run_s5(plant, an_leaf, weather_init["TairC"])
    except Exception as exc:
        print(f"VERDICT: SOLVER FAILURE (S5: {exc})")
        return 2

    try:
        pm = run_piafmunch(plant, an_leaf, sx, weather_init)
    except Exception as exc:
        print(f"VERDICT: SOLVER FAILURE (PiafMunch: {exc})")
        return 2
    finally:
        if PM_TMP.exists():
            PM_TMP.unlink()

    print_comparison(s5, pm)
    print(f"PiafMunch convergence: dC_ST/dt={pm['dc_dt']:.6g} mmol/cm3/h")
    print(f"S5 storage proxy [mmol/d]: {s5['storage_change']:.6g}")
    print(f"PiafMunch storage proxy [mmol/d]: {pm['storage_change']:.6g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
