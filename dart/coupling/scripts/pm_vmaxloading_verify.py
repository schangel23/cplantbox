"""pm_vmaxloading_verify.py — sweep Vmaxloading against V3 Babst windows.

Vmaxloading is overridden in-memory after read_phloem_parameters, mirroring
the CSTimin override pattern in pm_cstimin_verify.py. NO JSON edit.

Sweep target:
  V3 maize, 21 d, Babst 2022 growth-chamber proxy
  T=20.75 C, DLI≈30 mol m-2 d-1, 14:10 PP proxy, well-watered

If the sweep finds a 3/3 PASS, the highest passing Vmaxloading is re-run
through:
  GATE C — wheat tutorial day-7.3 Lacointe/Giraud regression
  GATE B — maize day-55 24h transient
"""

import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

import plantbox as pb  # noqa: E402
from dart.coupling.config import (  # noqa: E402
    DEFAULT_XML, get_hydraulics_json, get_photosynthesis_json, get_phloem_json,
)
from dart.coupling.growth import grow_plant  # noqa: E402

VMAX_GRID = [0.02, 0.05, 0.10, 0.20, 0.50]
N_SUBSTEPS = 6

BABST_MET = {
    d: {
        "T_mean_C": 20.75,
        "T_min_C": 19.0,
        "T_max_C": 22.0,
        "PAR_MJ_m2_d": 30.0 * 0.219,
        "VPD_kPa": 1.0,
        "RH_pct": 60.0,
        "Wind_m_s": 0.5,
    }
    for d in range(1, 31)
}

T_K = 20.75 + 273.15
RT_MPa_per_mmol_cm3 = 8.314 * T_K * 1e-3
CMH2O_TO_MPA = 9.80665e-5

DP_WINDOW = (0.51, 2.27)
V_WINDOW = (0.55, 1.35)
CST_WINDOW = (0.105, 0.465)


def _suppress():
    o1 = os.dup(1); o2 = os.dup(2); dn = os.open(os.devnull, os.O_WRONLY)
    os.dup2(dn, 1); os.dup2(dn, 2)
    return o1, o2, dn


def _restore(o1, o2, dn):
    os.dup2(o1, 1); os.dup2(o2, 2)
    os.close(dn); os.close(o1); os.close(o2)


def _pad_wheat_hydraulics(target_n=30):
    """Pad wheat hydraulics JSON to maxSubTypes=24 (30 for safety)."""
    src = REPO_ROOT / "modelparameter/functional/plant_hydraulics/wheat_Giraud2023adapted.json"
    with open(src) as f:
        d = json.load(f)
    for key in ("kx_ages", "kx_values", "kr_ages", "kr_values"):
        for ot in list(d[key].keys()):
            arr = d[key][ot]
            while len(arr) < target_n:
                arr.append(list(arr[-1]))
            d[key][ot] = arr
    out = Path(tempfile.gettempdir()) / "wheat_hyd_padded.json"
    with open(out, "w") as f:
        json.dump(d, f)
    return str(out.with_suffix(""))


def _make_v3_plant(age=21, Tair_C=20.75):
    return grow_plant(
        xml_path=str(DEFAULT_XML),
        simulation_time=age,
        min_stem_nodes=10,
        min_leaf_nodes=4,
        enable_photosynthesis=True,
        seed=42,
        daily_met=BABST_MET,
        T_air_default=Tair_C,
    )


def run_v3_babst(vmaxloading, age=21, Tair_C=20.75):
    """Run the V3 Babst comparison, overriding hm.Vmaxloading in-memory."""
    plant = _make_v3_plant(age=age, Tair_C=Tair_C)
    organ_types = np.array(plant.organTypes, dtype=np.int32)
    sub_types = np.array(plant.subTypes, dtype=np.int32)
    n_segs = len(plant.getSegments())
    n_nodes = len(plant.getNodes())
    n_organs = len(plant.getOrgans())

    from plantbox.functional.phloem_flux import PhloemFluxPython
    from plantbox.functional.PlantHydraulicParameters import PlantHydraulicParameters

    params = PlantHydraulicParameters()
    params.read_parameters(get_hydraulics_json())

    hm = PhloemFluxPython(plant, params, psiXylInit=-500, ciInit=350e-6)
    hm.read_photosynthesis_parameters(filename=get_photosynthesis_json())
    hm.read_phloem_parameters(filename=get_phloem_json())
    hm.atol = 1e-6; hm.rtol = 1e-4; hm.solver = 32
    hm.useCWGr = False
    hm.Vmaxloading = float(vmaxloading)

    rh = 0.6; psi_soil = -500.0
    p_s = np.linspace(psi_soil, psi_soil - 100, 100)
    es = hm.get_es(Tair_C); ea = es * rh
    par = 600.0 * 1e-6 * 86400 * 1e-4
    hm.solve(sim_time=age, rsx=p_s, cells=True, ea=ea, es=es,
             PAR=par, TairC=Tair_C, verbose=0)

    Tair_K = Tair_C + 273.15
    dt_days = 1.0 / 24.0
    last_ret = -1
    failures = 0
    fdpair = _suppress()
    try:
        for step in range(1, N_SUBSTEPS + 1):
            if step > 1:
                hm.withInitVal = False
            t_start = float(age) + (step - 1) * dt_days
            t_end = t_start + dt_days
            last_ret = hm.startPM(t_start, t_end, 1, Tair_K, False,
                                  str(REPO_ROOT / "dart/coupling/scripts/_pm_vmax.txt"))
            if last_ret != 1:
                failures += 1
                break
    finally:
        _restore(*fdpair)

    C_ST = np.array(hm.C_ST)
    psi_xyl = np.array(hm.psiXyl)
    JW_ST = np.array(hm.JW_ST)

    seg_node = np.zeros(n_nodes, dtype=np.int32)
    seg_st = np.zeros(n_nodes, dtype=np.int32)
    for si in range(n_segs):
        nodeID = si + 1
        if nodeID < n_nodes:
            seg_node[nodeID] = int(organ_types[si])
            seg_st[nodeID] = int(sub_types[si])

    leaf_mask = seg_node == 4
    stem_mask = seg_node == 3
    root_mask = seg_node == 2
    src_mask = leaf_mask & (C_ST > 0.21)

    if src_mask.any():
        c_st_src = float(np.percentile(C_ST[src_mask], 75))
        psi_src = float(np.mean(psi_xyl[src_mask]))
    else:
        c_st_src = float(np.max(C_ST[leaf_mask])) if leaf_mask.any() else 0.0
        psi_src = float(np.mean(psi_xyl[leaf_mask])) if leaf_mask.any() else 0.0

    nodes = plant.getNodes()
    node_z = np.array([n.z for n in nodes], dtype=np.float64)
    if root_mask.any():
        deep_idx = np.argsort(node_z[root_mask])[:max(5, root_mask.sum() // 50)]
        root_node_ids = np.where(root_mask)[0][deep_idx]
        c_st_sink = float(np.mean(C_ST[root_node_ids]))
        psi_sink = float(np.mean(psi_xyl[root_node_ids]))
    else:
        c_st_sink = float(np.min(C_ST))
        psi_sink = float(np.min(psi_xyl))

    P_src = psi_src * CMH2O_TO_MPA + RT_MPa_per_mmol_cm3 * c_st_src
    P_sink = psi_sink * CMH2O_TO_MPA + RT_MPa_per_mmol_cm3 * c_st_sink
    delta_P = P_src - P_sink

    if stem_mask.any():
        stem_ids = np.where(stem_mask)[0]
        basal_id = stem_ids[np.argmin(np.abs(node_z[stem_ids]))]
        Across_stem = hm.Across_st[1][0]
        flux_cm3_h = abs(float(JW_ST[basal_id]))
        v_m_per_hr = (flux_cm3_h / Across_stem) * 1e-2 if Across_stem > 0 else 0.0
    else:
        v_m_per_hr = 0.0

    dp_pass = DP_WINDOW[0] <= delta_P <= DP_WINDOW[1]
    v_pass = V_WINDOW[0] <= v_m_per_hr <= V_WINDOW[1]
    c_pass = CST_WINDOW[0] <= c_st_src <= CST_WINDOW[1]
    n_pass = int(dp_pass) + int(v_pass) + int(c_pass)
    verdict = "PASS" if n_pass == 3 else "FAIL"
    if c_st_src > CST_WINDOW[1]:
        verdict = "FAIL over-loading"

    return dict(
        Vmaxloading=float(vmaxloading),
        ret=int(last_ret),
        failures=int(failures),
        delta_P_MPa=float(delta_P),
        v_m_per_hr=float(v_m_per_hr),
        c_st_src=float(c_st_src),
        c_st_max=float(np.max(C_ST)),
        n_pass=n_pass,
        verdict=verdict,
        n_organs=int(n_organs),
        n_nodes=int(n_nodes),
    )


def gate_c_wheat73(vmaxloading):
    print(f"\n[GATE C] Wheat day-7.3 at Vmaxloading={vmaxloading:.2f}")
    WHEAT_XML = REPO_ROOT / "modelparameter/structural/plant/Triticum_aestivum_test_2021_shapeType2.xml"
    WHEAT_PHLOEM = str(REPO_ROOT / "modelparameter/functional/plant_sucrose/phloem_parameters2025")
    WHEAT_HYD = _pad_wheat_hydraulics(target_n=30)

    plant = pb.MappedPlant(seednum=2)
    plant.readParameters(str(WHEAT_XML))
    plant.setGeometry(pb.SDF_PlantBox(np.inf, np.inf, 60.))
    plant.initialize(False)
    plant.simulate(7.3, False)
    n_nodes = plant.getNumberOfNodes()
    print(f"  Wheat 7.3 days: {n_nodes} nodes")

    from plantbox.functional.phloem_flux import PhloemFluxPython
    from plantbox.functional.PlantHydraulicParameters import PlantHydraulicParameters

    params = PlantHydraulicParameters()
    params.read_parameters(WHEAT_HYD)

    hm = PhloemFluxPython(plant, params, psiXylInit=-500, ciInit=350e-6)
    hm.read_photosynthesis_parameters(filename=get_photosynthesis_json())
    hm.read_phloem_parameters(filename=WHEAT_PHLOEM)
    hm.atol = 1e-6; hm.rtol = 1e-4; hm.solver = 32
    hm.useCWGr = False
    hm.Vmaxloading = float(vmaxloading)

    Tair_C = 25.0; rh = 0.7; psi_soil = -500.0
    p_s = np.linspace(psi_soil, psi_soil - 60, 60)
    es = hm.get_es(Tair_C); ea = es * rh
    par = 1000.0 * 1e-6 * 86400 * 1e-4
    hm.solve(sim_time=7.3, rsx=p_s, cells=True, ea=ea, es=es,
             PAR=par, TairC=Tair_C, verbose=0)

    Nt = len(plant.getNodes())
    Tair_K = Tair_C + 273.15
    fdpair = _suppress()
    try:
        ret = hm.startPM(7.3, 7.3 + 1.0 / 24.0, 1, Tair_K, False,
                         str(REPO_ROOT / "dart/coupling/scripts/_pm_vmax_gate_c.txt"))
    finally:
        _restore(*fdpair)

    Q = np.array(hm.Q_out)
    QRm = float(np.sum(Q[Nt*2:Nt*3]))
    QGr = float(np.sum(Q[Nt*4:Nt*5]))
    C_ST = np.array(hm.C_ST)
    cmean = float(np.mean(C_ST)); cmax = float(np.max(C_ST))
    failures = 0 if ret == 1 and np.isfinite(cmax) else 1
    accepted = failures == 0 and cmax < 5.0
    print(f"  ret={ret}  Q_Rm_sum={QRm:.6f}  Q_Gr_sum={QGr:.6f}")
    print(f"  C_ST mean={cmean:.6f}  max={cmax:.6f}  accepted={accepted}")
    return dict(ret=int(ret), Q_Rm=QRm, Q_Gr=QGr, C_ST_mean=cmean,
                C_ST_max=cmax, solver_failures=failures, accepted=accepted)


def gate_b_maize55(vmaxloading, n_substeps=24):
    print(f"\n[GATE B] Day-55 maize 24h at Vmaxloading={vmaxloading:.2f}")
    Tair_C = 25.0
    age = 55
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
    hm.useCWGr = False
    hm.Vmaxloading = float(vmaxloading)

    rh = 0.7; psi_soil = -500.0
    p_s = np.linspace(psi_soil, psi_soil - 100, 100)
    es = hm.get_es(Tair_C); ea = es * rh
    par = 1000.0 * 1e-6 * 86400 * 1e-4
    hm.solve(sim_time=age, rsx=p_s, cells=True, ea=ea, es=es,
             PAR=par, TairC=Tair_C, verbose=0)
    An_total = float(np.sum(np.array(hm.get_net_assimilation()))) * 1e3
    print(f"  An={An_total:.1f} mmol CO2/d  Vmaxloading={hm.Vmaxloading}")

    Nt = len(plant.getNodes())
    Tair_K = Tair_C + 273.15
    dt_days = 1.0 / 24.0
    prev_QRm = prev_QGr = 0.0
    QRm = QGr = 0.0
    cmean = cmax = 0.0
    failures = 0

    print(f"  {'h':>3} {'Rm_h':>10} {'Rg_h':>10} {'C_ST_mean':>10} {'C_ST_max':>10} "
          f"{'wall':>6} {'ret':>4}")
    print("  " + "-" * 64)
    for step in range(1, n_substeps + 1):
        if step > 1:
            hm.withInitVal = False
        t_start = float(age) + (step - 1) * dt_days
        t_end = t_start + dt_days
        t0 = time.time()
        fdpair = _suppress()
        try:
            ret = hm.startPM(t_start, t_end, 1, Tair_K, False,
                             str(REPO_ROOT / "dart/coupling/scripts/_pm_vmax_gate_b.txt"))
        finally:
            _restore(*fdpair)
        wall = time.time() - t0

        Q = np.array(hm.Q_out)
        QRm = float(np.sum(Q[Nt*2:Nt*3]))
        QGr = float(np.sum(Q[Nt*4:Nt*5]))
        C_ST = np.array(hm.C_ST)
        cmean = float(np.mean(C_ST)); cmax = float(np.max(C_ST))
        Rm_h = QRm - prev_QRm; Rg_h = QGr - prev_QGr
        prev_QRm = QRm; prev_QGr = QGr

        if step in (1, 2, 6, 12, 18, 24):
            print(f"  {step:>3} {Rm_h:>10.4f} {Rg_h:>10.4f} {cmean:>10.4f} "
                  f"{cmax:>10.4f} {wall:>6.2f} {ret:>4}")
        if ret != 1 or not np.isfinite(cmax) or cmax > 1e6:
            failures += 1
            break

    Q_Rmmax_sum = float(np.sum(np.array(hm.Q_Rmmax)))
    accepted = failures == 0 and abs(Q_Rmmax_sum - 16.0) <= 5.0 and cmax < 5.0
    print("  " + "-" * 64)
    print(f"  Final: Rm 24h cumulative={QRm:.3f}  Rg cumulative={QGr:.3f} mmol Suc")
    print(f"  Q_Rmmax={Q_Rmmax_sum:.3f} mmol/d  C_ST mean={cmean:.4f}  max={cmax:.4f}")
    print(f"  Solver failures (ret≠1 or NaN): {failures}/{n_substeps}  accepted={accepted}")
    return dict(QRm=QRm, QGr=QGr, Q_Rmmax=Q_Rmmax_sum, C_ST_mean=cmean,
                C_ST_max=cmax, solver_failures=failures, An=An_total,
                accepted=accepted)


def _pass_text(r):
    return f"{r['n_pass']}/3"


def main():
    print("=" * 100)
    print("Vmaxloading sweep on V3 maize vs Babst 2022 Table A1")
    print("=" * 100)
    print("Override: hm.Vmaxloading set in-memory after read_phloem_parameters.")
    print(f"Babst 2σ windows: ΔP {DP_WINDOW[0]:.2f}-{DP_WINDOW[1]:.2f} MPa; "
          f"v {V_WINDOW[0]:.2f}-{V_WINDOW[1]:.2f} m/hr; "
          f"C_ST {CST_WINDOW[0]:.3f}-{CST_WINDOW[1]:.3f} mmol/cm³")

    results = []
    for vmax in VMAX_GRID:
        print(f"\n--- Vmaxloading = {vmax:.2f} ---")
        r = run_v3_babst(vmax)
        results.append(r)
        print(f"  ret={r['ret']}  n_organs={r['n_organs']}  n_nodes={r['n_nodes']}")
        print(f"  ΔP={r['delta_P_MPa']:.3f} MPa  v={r['v_m_per_hr']:.4f} m/hr  "
              f"C_ST_src={r['c_st_src']:.4f}  C_ST_max={r['c_st_max']:.4f}  "
              f"{_pass_text(r)} {r['verdict']}")

    print("\n" + "=" * 100)
    print("SWEEP SUMMARY")
    print("=" * 100)
    print(f"{'Vmaxloading':>11} {'ΔP [MPa]':>10} {'v [m/hr]':>10} "
          f"{'C_ST_src':>10} {'n_PASS/3':>8} {'ret':>5} {'n_org':>7} {'n_nodes':>8} {'verdict':>18}")
    print("-" * 100)
    for r in results:
        print(f"{r['Vmaxloading']:>11.2f} {r['delta_P_MPa']:>10.3f} "
              f"{r['v_m_per_hr']:>10.4f} {r['c_st_src']:>10.4f} "
              f"{_pass_text(r):>8} {r['ret']:>5} {r['n_organs']:>7} "
              f"{r['n_nodes']:>8} {r['verdict']:>18}")

    passing = [r for r in results if r["n_pass"] == 3 and r["c_st_src"] <= CST_WINDOW[1]]
    gate_c = gate_b = None
    if passing:
        rec = max(passing, key=lambda r: r["Vmaxloading"])
        print(f"\nRecommended Vmaxloading: {rec['Vmaxloading']:.2f} "
              "(highest 3/3 PASS with C_ST_source <= 0.465)")
        gate_c = gate_c_wheat73(rec["Vmaxloading"])
        gate_b = gate_b_maize55(rec["Vmaxloading"])
    else:
        rec = max(results, key=lambda r: (r["n_pass"], -max(0.0, r["c_st_src"] - CST_WINDOW[1])))
        print(f"\nNo 3/3 found. Best partial: Vmaxloading={rec['Vmaxloading']:.2f} "
              f"with {_pass_text(rec)}, C_ST_source={rec['c_st_src']:.4f}.")
        print("Next-step recommendation: sweep beta_loading to reduce source over-loading "
              "without sacrificing ΔP/v.")

    print("\n" + "=" * 100)
    print("JSON SUMMARY")
    print("=" * 100)
    print(json.dumps({
        "sweep": results,
        "recommended_Vmaxloading": rec["Vmaxloading"] if passing else None,
        "best_partial_Vmaxloading": None if passing else rec["Vmaxloading"],
        "gate_c": gate_c,
        "gate_b": gate_b,
    }, indent=2, sort_keys=True))

    for name in ("_pm_vmax.txt", "_pm_vmax_gate_b.txt", "_pm_vmax_gate_c.txt"):
        p = REPO_ROOT / "dart/coupling/scripts" / name
        if p.exists():
            p.unlink()


if __name__ == "__main__":
    main()
