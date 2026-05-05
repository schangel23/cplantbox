"""pm_cstimin_sweep.py — sweep hm.CSTimin on V21 maize vs Babst Table A1.

Grows the V21 plant once under Babst chamber-proxy met, then for each
CSTimin in {0.20, 0.10, 0.05, 0.02} rebuilds PhloemFluxPython, overrides
hm.CSTimin AFTER reading the JSON, runs 6 hourly substeps, and reports
the 3-metric Babst comparison.

Acceptance: smallest reduction that lifts ΔP and v into Babst's 2σ
window while keeping C_ST source PASS. NO JSON edit (override is in-
memory).
"""

import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from dart.coupling.config import (  # noqa: E402
    DEFAULT_XML, get_hydraulics_json, get_photosynthesis_json, get_phloem_json,
)
from dart.coupling.growth import grow_plant  # noqa: E402

CSTIMIN_GRID = [0.20, 0.10, 0.05, 0.02]
N_SUBSTEPS = 6  # hourly

BABST_MET = {
    d: {"T_mean_C": 20.75, "T_min_C": 19.0, "T_max_C": 22.0,
        "PAR_MJ_m2_d": 30.0 * 0.219, "VPD_kPa": 1.0,
        "RH_pct": 60.0, "Wind_m_s": 0.5}
    for d in range(1, 31)
}

BABST = {
    "delta_p_MPa":   (1.39, 0.44),
    "v_m_per_hr":    (0.95, 0.20),
    "c_st_mmol_cm3": (0.285, 0.090),
}
TOL_SIGMA = 2.0

T_K = 20.75 + 273.15
RT_MPa_per_mmol_cm3 = 8.314 * T_K * 1e-3   # ≈ 2.443 MPa per mmol/cm³
CMH2O_TO_MPA = 9.80665e-5


def _suppress():
    o1 = os.dup(1); o2 = os.dup(2); dn = os.open(os.devnull, os.O_WRONLY)
    os.dup2(dn, 1); os.dup2(dn, 2)
    return o1, o2, dn


def _restore(o1, o2, dn):
    os.dup2(o1, 1); os.dup2(o2, 2)
    os.close(dn); os.close(o1); os.close(o2)


def run_one(cstimin_override, age=21, Tair_C=20.75):
    """Grow fresh + run PhloemFlux. Re-grow per iteration to avoid
    accumulated plant-state contamination across sweep iterations
    (each grow_plant gives an identical seeded plant; only CSTimin varies)."""
    plant = grow_plant(
        xml_path=str(DEFAULT_XML),
        simulation_time=age,
        min_stem_nodes=10,
        min_leaf_nodes=4,
        enable_photosynthesis=True,
        seed=42,
        daily_met=BABST_MET,
        T_air_default=Tair_C,
    )
    organ_types = np.array(plant.organTypes, dtype=np.int32)

    from plantbox.functional.phloem_flux import PhloemFluxPython
    from plantbox.functional.PlantHydraulicParameters import PlantHydraulicParameters

    params = PlantHydraulicParameters()
    params.read_parameters(get_hydraulics_json())

    hm = PhloemFluxPython(plant, params, psiXylInit=-500, ciInit=350e-6)
    hm.read_photosynthesis_parameters(filename=get_photosynthesis_json())
    hm.read_phloem_parameters(filename=get_phloem_json())
    hm.atol = 1e-6; hm.rtol = 1e-4; hm.solver = 32
    hm.useCWGr = False
    # OVERRIDE — happens after JSON read so it sticks
    hm.CSTimin = float(cstimin_override)

    rh = 0.6; psi_soil = -500.0
    p_s = np.linspace(psi_soil, psi_soil - 100, 100)
    es = hm.get_es(Tair_C); ea = es * rh
    par = 600.0 * 1e-6 * 86400 * 1e-4
    hm.solve(sim_time=age, rsx=p_s, cells=True, ea=ea, es=es,
             PAR=par, TairC=Tair_C, verbose=0)

    Tair_K = Tair_C + 273.15
    dt_days = 1.0 / 24.0
    Nt = len(plant.getNodes())

    fdpair = _suppress()
    last_ret = -1
    try:
        for step in range(1, N_SUBSTEPS + 1):
            if step > 1:
                hm.withInitVal = False
            t_start = float(age) + (step - 1) * dt_days
            t_end   = t_start + dt_days
            last_ret = hm.startPM(t_start, t_end, 1, Tair_K, False,
                                  str(REPO_ROOT / "dart/coupling/scripts/_pm_sweep.txt"))
            if last_ret != 1:
                break
    finally:
        _restore(*fdpair)

    C_ST    = np.array(hm.C_ST)
    psi_xyl = np.array(hm.psiXyl)
    JW_ST   = np.array(hm.JW_ST)
    Q_total = np.array(hm.Q_out)

    QRm   = float(np.sum(Q_total[Nt*2:Nt*3]))
    QGr   = float(np.sum(Q_total[Nt*4:Nt*5]))
    QExud = float(np.sum(Q_total[Nt*3:Nt*4]))

    seg_ot = np.zeros(Nt, dtype=np.int32)
    for si in range(len(organ_types)):
        nodeID = si + 1
        if nodeID < Nt:
            seg_ot[nodeID] = int(organ_types[si])

    leaf_mask = seg_ot == 4
    stem_mask = seg_ot == 3
    root_mask = seg_ot == 2

    # Source = leaf nodes with C_ST above CSTimin (active loaders)
    src_mask = leaf_mask & (C_ST > cstimin_override + 0.01)
    if src_mask.any():
        c_st_src = float(np.percentile(C_ST[src_mask], 75))
        psi_src  = float(np.mean(psi_xyl[src_mask]))
    else:
        c_st_src = float(np.max(C_ST[leaf_mask])) if leaf_mask.any() else 0.0
        psi_src  = float(np.mean(psi_xyl[leaf_mask])) if leaf_mask.any() else 0.0

    nodes = plant.getNodes()
    node_z = np.array([n.z for n in nodes], dtype=np.float64)
    if root_mask.any():
        deep_idx = np.argsort(node_z[root_mask])[:max(5, root_mask.sum() // 50)]
        root_node_ids = np.where(root_mask)[0][deep_idx]
        c_st_sink = float(np.mean(C_ST[root_node_ids]))
        psi_sink  = float(np.mean(psi_xyl[root_node_ids]))
    else:
        c_st_sink = float(np.min(C_ST))
        psi_sink  = float(np.min(psi_xyl))

    P_src  = psi_src  * CMH2O_TO_MPA + RT_MPa_per_mmol_cm3 * c_st_src
    P_sink = psi_sink * CMH2O_TO_MPA + RT_MPa_per_mmol_cm3 * c_st_sink
    delta_P = P_src - P_sink

    if stem_mask.any():
        stem_ids = np.where(stem_mask)[0]
        basal_id = stem_ids[np.argmin(np.abs(node_z[stem_ids]))]
        Across_stem = hm.Across_st[1][0]
        flux_cm3_d = abs(float(JW_ST[basal_id]))
        v_m_per_hr = (flux_cm3_d / Across_stem) * 1e-2 / 24.0 if Across_stem > 0 else 0.0
    else:
        v_m_per_hr = 0.0

    return dict(
        cstimin=cstimin_override,
        ret=last_ret,
        c_st_src=c_st_src,
        c_st_sink=c_st_sink,
        c_st_leaf_mean=float(np.mean(C_ST[leaf_mask])) if leaf_mask.any() else 0.0,
        c_st_root_mean=float(np.mean(C_ST[root_mask])) if root_mask.any() else 0.0,
        c_st_max=float(np.max(C_ST)),
        delta_P_MPa=delta_P,
        v_m_per_hr=v_m_per_hr,
        QRm_6h=QRm, QGr_6h=QGr, QExud_6h=QExud,
        n_src_nodes=int(src_mask.sum()),
    )


def main():
    print("=" * 100)
    print(f"CSTimin sweep on V21 maize (Babst chamber-proxy met)")
    print(f"  RT[{T_K-273.15:.1f}C] = {RT_MPa_per_mmol_cm3:.3f} MPa per mmol/cm³  "
          f"(needs ~0.6 mmol/cm³ gradient for Babst ΔP=1.39 MPa)")
    print("=" * 100)

    age = 21
    print(f"Plant grown FRESH per iteration (seed=42, deterministic) to avoid "
          "state contamination.")

    results = []
    for cs in CSTIMIN_GRID:
        print(f"\n--- CSTimin = {cs:.2f} ---")
        r = run_one(cs, age=age)
        results.append(r)
        print(f"  ret={r['ret']}  C_ST src/sink/leaf-mean/root-mean/max = "
              f"{r['c_st_src']:.3f} / {r['c_st_sink']:.3f} / "
              f"{r['c_st_leaf_mean']:.3f} / {r['c_st_root_mean']:.3f} / "
              f"{r['c_st_max']:.3f}")
        print(f"  ΔP={r['delta_P_MPa']:.3f} MPa  v_basal={r['v_m_per_hr']:.4f} m/hr  "
              f"src_nodes={r['n_src_nodes']}")
        print(f"  6h: Rm={r['QRm_6h']:.3f}  Gr={r['QGr_6h']:.3f}  "
              f"Exud={r['QExud_6h']:.3f} mmol Suc")

    # Summary table
    print("\n" + "=" * 100)
    print("SWEEP SUMMARY (PASS within 2σ Babst Table A1)")
    print("=" * 100)
    hdr = f"{'CSTimin':>9} {'ΔP':>8} {'(verdict)':>10} {'v[m/hr]':>9} " \
          f"{'(verdict)':>10} {'C_ST_src':>10} {'(verdict)':>10} {'pass':>5}"
    print(hdr)
    print("-" * 100)
    for r in results:
        dp = r['delta_P_MPa']; vp = r['v_m_per_hr']; cs = r['c_st_src']
        dp_v = "PASS" if abs(dp - BABST["delta_p_MPa"][0]) <= TOL_SIGMA*BABST["delta_p_MPa"][1] else "FAIL"
        v_v  = "PASS" if abs(vp - BABST["v_m_per_hr"][0])  <= TOL_SIGMA*BABST["v_m_per_hr"][1]  else "FAIL"
        c_v  = "PASS" if abs(cs - BABST["c_st_mmol_cm3"][0]) <= TOL_SIGMA*BABST["c_st_mmol_cm3"][1] else "FAIL"
        n_pass = sum(v == "PASS" for v in (dp_v, v_v, c_v))
        print(f"{r['cstimin']:>9.2f} {dp:>8.3f} {dp_v:>10} {vp:>9.4f} "
              f"{v_v:>10} {cs:>10.3f} {c_v:>10} {n_pass:>5}/3")

    print("\nBabst targets (2σ window):")
    print(f"  ΔP            : {BABST['delta_p_MPa'][0]:.3f} ± {2*BABST['delta_p_MPa'][1]:.3f} MPa")
    print(f"  v sap         : {BABST['v_m_per_hr'][0]:.3f} ± {2*BABST['v_m_per_hr'][1]:.3f} m/hr")
    print(f"  C_ST source   : {BABST['c_st_mmol_cm3'][0]:.3f} ± {2*BABST['c_st_mmol_cm3'][1]:.3f} mmol/cm³")

    p = REPO_ROOT / "dart/coupling/scripts/_pm_sweep.txt"
    if p.exists():
        p.unlink()


if __name__ == "__main__":
    main()
