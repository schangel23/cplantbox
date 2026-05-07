"""pm_starch_restore_sweep.py — restore mesophyll-starch dynamics
in-memory and sweep k_S_Mesophyll on V21 maize for 24h.

Background
----------
The 2026-05-07 mass-balance audit (pm_v3_mass_balance_and_v.py /
pm_v3_mass_balance_24h.py) showed:
  - 86% (after 6h) / 79% (after 24h) of An accumulates in mesophyll
    cytosolic Q_Mesophyll
  - Q_S_Mesophyll (starch buffer) stays at zero for 22h, only
    activating once Cmeso crosses C_targMesophyll = 0.8 mmol/cm^3
  - Loading efficiency Q_Fl/An rises from 4.8% to 25.1% but does not
    reach the ~50% Stitt 2012 implies for real maize source leaves

JSON-source description annotations:
  k_S_Mesophyll: 1.0 (annotated "Matches k_S_ST (was 20.0)")
  C_targMesophyll: 0.8 (annotated "raised from 0.4")
  Vmax_S_Mesophyll: 0.0

So someone deliberately deneutered the buffer (k 20 -> 1, C_targ 0.4 -> 0.8).
This sweep restores the pre-change defaults and tests four k_S_Mesophyll
levels to find the value that gives:
  - physical equilibrium Cmeso (~0.5-2 mmol/cm^3, NOT 20)
  - ~50% loading efficiency (Stitt 2012)
  - active Q_S_Mesophyll storage flux

If the proper starch buffer alone closes the loading-efficiency gap,
the v repair is one parameter edit. If loading stays low even with
restored starch, the bottleneck is downstream (Vmaxloading, beta_loading
saturation, or len_leaf coverage).

Grid
----
  k_S_Mesophyll  in {1.0, 5.0, 20.0, 50.0}
  C_targMesophyll = 0.4  (Lacointe pre-change default)
  Vmax_S_Mesophyll = 0.0 (keep MM kinetic off until needed)
  Vmaxloading = 0.20, beta_loading = 2.0  (JSON-default best cell)
  24 substeps per cell, ~2 min wall-clock each.

NO JSON edit. All overrides are hm.* attribute writes after
read_phloem_parameters.
"""

import json
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

K_GRID = [1.0, 5.0, 20.0, 50.0]
C_TARG_RESTORED = 0.4
VMAX_S_MESO_RESTORED = 0.0    # keep MM off; only target-driven term active
N_SUBSTEPS = 24

VMAX_LOADING = 0.20
BETA_LOADING = 2.0

BABST_MET = {
    d: {"T_mean_C": 20.75, "T_min_C": 19.0, "T_max_C": 22.0,
        "PAR_MJ_m2_d": 30.0 * 0.219, "VPD_kPa": 1.0,
        "RH_pct": 60.0, "Wind_m_s": 0.5}
    for d in range(1, 31)
}

V_BABST_WIN = (0.55, 1.35)
JENSEN_FACTOR_MID = 0.40

T_K = 20.75 + 273.15
RT_MPa_per_mmol_cm3 = 8.314 * T_K * 1e-3
CMH2O_TO_MPA = 9.80665e-5

SLOT = {
    "Q_ST":             0,
    "Q_Mesophyll":      1,
    "Q_RespMaint":      2,
    "Q_Exudation":      3,
    "Q_Growthtot":      4,
    "Q_RespMaintmax":   5,
    "Q_Growthtotmax":   6,
    "Q_S_Mesophyll":    7,
    "Q_S_ST":           8,
    "Q_Mucil":          9,
}


def _suppress():
    o1 = os.dup(1); o2 = os.dup(2); dn = os.open(os.devnull, os.O_WRONLY)
    os.dup2(dn, 1); os.dup2(dn, 2)
    return o1, o2, dn


def _restore(o1, o2, dn):
    os.dup2(o1, 1); os.dup2(o2, 2)
    os.close(dn); os.close(o1); os.close(o2)


def slot_sum(Q_out, name, Nt):
    i = SLOT[name]
    return float(np.sum(Q_out[i*Nt:(i+1)*Nt]))


def run_one(k_meso, age=21, Tair_C=20.75):
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
    n_segs = len(plant.getSegments())

    from plantbox.functional.phloem_flux import PhloemFluxPython
    from plantbox.functional.PlantHydraulicParameters import PlantHydraulicParameters

    params = PlantHydraulicParameters()
    params.read_parameters(get_hydraulics_json())

    hm = PhloemFluxPython(plant, params, psiXylInit=-500, ciInit=350e-6)
    hm.read_photosynthesis_parameters(filename=get_photosynthesis_json())
    hm.read_phloem_parameters(filename=get_phloem_json())
    hm.atol = 1e-6; hm.rtol = 1e-4; hm.solver = 32
    hm.useCWGr = False
    hm.Vmaxloading  = VMAX_LOADING
    hm.beta_loading = BETA_LOADING
    # Starch restoration:
    hm.k_S_Mesophyll    = float(k_meso)
    hm.C_targMesophyll  = float(C_TARG_RESTORED)
    hm.Vmax_S_Mesophyll = float(VMAX_S_MESO_RESTORED)

    rh = 0.6; psi_soil = -500.0
    p_s = np.linspace(psi_soil, psi_soil - 200, 200)
    es = hm.get_es(Tair_C); ea = es * rh
    par = 600.0 * 1e-6 * 86400 * 1e-4
    hm.solve(sim_time=age, rsx=p_s, cells=True, ea=ea, es=es,
             PAR=par, TairC=Tair_C, verbose=0)

    An = np.array(hm.get_net_assimilation())
    SUC_PER_CO2 = 1.0 / 12.0
    An_total_mmol_suc_per_d = float(np.sum(An)) * 1e3 * SUC_PER_CO2
    dt_days = 1.0 / 24.0
    An_per_step = An_total_mmol_suc_per_d * dt_days

    Tair_K = Tair_C + 273.15
    Nt = len(plant.getNodes())
    nodes = plant.getNodes()
    node_z = np.array([n.z for n in nodes], dtype=np.float64)

    fdpair = _suppress()
    cum_first = None
    cum_last = None
    JW_basal_last = 0.0
    c_st_src_last = 0.0
    c_meso_last = 0.0
    last_ret = -1
    try:
        for step in range(1, N_SUBSTEPS + 1):
            if step > 1:
                hm.withInitVal = False
            t_start = float(age) + (step - 1) * dt_days
            t_end = t_start + dt_days
            last_ret = hm.startPM(t_start, t_end, 1, Tair_K, False,
                                  str(REPO_ROOT / "dart/coupling/scripts/_pm_starch.txt"))
            if last_ret != 1:
                break
            Q_out = np.array(hm.Q_out)
            cum = {k: slot_sum(Q_out, k, Nt) for k in SLOT}
            if cum_first is None:
                cum_first = {k: 0.0 for k in SLOT}  # zero-init reference
            cum_last = cum

            if step == N_SUBSTEPS:
                C_ST = np.array(hm.C_ST)
                JW_ST = np.array(hm.JW_ST)
                seg_ot = np.zeros(Nt, dtype=np.int32)
                for si in range(n_segs):
                    nodeID = si + 1
                    if nodeID < Nt:
                        seg_ot[nodeID] = int(organ_types[si])
                stem_mask = seg_ot == 3
                leaf_mask = seg_ot == 4
                if stem_mask.any():
                    stem_ids = np.where(stem_mask)[0]
                    basal_id = stem_ids[np.argmin(np.abs(node_z[stem_ids]))]
                    JW_basal_last = abs(float(JW_ST[basal_id]))
                src = leaf_mask & (C_ST > 0.21)
                if src.any():
                    c_st_src_last = float(np.percentile(C_ST[src], 75))
                else:
                    c_st_src_last = float(np.max(C_ST[leaf_mask])) if leaf_mask.any() else 0.0
                # Rough Cmeso estimate at end: Q_Mesophyll_total / vol_total_meso
                # vol_ParApo isn't directly exposed; approximate via Q_Mesophyll/leaf_node_count*0.0005
                Q_meso_total = cum["Q_Mesophyll"]
                # Use mean per-leaf-node volume as 0.0005 cm^3 (V21 ballpark)
                est_vol_total = max(1, leaf_mask.sum()) * 0.0005
                c_meso_last = Q_meso_total / est_vol_total if est_vol_total > 0 else 0.0
    finally:
        _restore(*fdpair)

    Across_anat = float(hm.Across_st[1][0])

    # Mass balance over 24h
    sums = {k: cum_last[k] for k in SLOT}
    sum_An = An_per_step * N_SUBSTEPS
    sum_dMeso  = sums["Q_Mesophyll"]
    sum_dSMeso = sums["Q_S_Mesophyll"]
    sum_dST    = sums["Q_ST"]
    sum_dRm    = sums["Q_RespMaint"]
    sum_dGr    = sums["Q_Growthtot"]
    sum_dExud  = sums["Q_Exudation"]
    sum_QFl = sum_An - sum_dMeso - sum_dSMeso

    load_eff   = sum_QFl / sum_An if sum_An > 0 else 0
    storage    = (sum_dMeso + sum_dSMeso) / sum_An if sum_An > 0 else 0
    starch_eff = sum_dSMeso / sum_An if sum_An > 0 else 0

    # v probes (cm^3/h interpretation, anatomical and Jensen 0.4)
    v_anat = (JW_basal_last / Across_anat) * 1e-2 if Across_anat > 0 else 0.0
    v_jensen = v_anat / JENSEN_FACTOR_MID if Across_anat > 0 else 0.0
    in_babst_anat = V_BABST_WIN[0] <= v_anat <= V_BABST_WIN[1]
    in_babst_jensen = V_BABST_WIN[0] <= v_jensen <= V_BABST_WIN[1]

    return dict(
        k_S_Mesophyll=float(k_meso),
        ret=int(last_ret),
        An_total=sum_An,
        dQ_Meso=sum_dMeso,
        dQ_S_Meso=sum_dSMeso,
        dQ_ST=sum_dST,
        dQ_Rm=sum_dRm,
        dQ_Gr=sum_dGr,
        dQ_Exud=sum_dExud,
        Q_Fl=sum_QFl,
        load_eff=load_eff,
        storage_frac=storage,
        starch_frac=starch_eff,
        c_meso_estimate=c_meso_last,
        c_st_src=c_st_src_last,
        JW_basal=JW_basal_last,
        v_anat_m_per_hr=v_anat,
        v_jensen04_m_per_hr=v_jensen,
        in_babst_anat=in_babst_anat,
        in_babst_jensen=in_babst_jensen,
    )


def main():
    print("=" * 100)
    print("Starch-buffer restoration sweep (V21 maize, 24h, JSON-default loading)")
    print("=" * 100)
    print(f"  Restored constants: C_targMesophyll = {C_TARG_RESTORED}")
    print(f"                      Vmax_S_Mesophyll = {VMAX_S_MESO_RESTORED}")
    print(f"  Sweep:              k_S_Mesophyll in {K_GRID}")
    print(f"  Loading regime:     Vmaxloading = {VMAX_LOADING}  beta_loading = {BETA_LOADING}")
    print(f"  Babst v window:     {V_BABST_WIN[0]:.2f} - {V_BABST_WIN[1]:.2f} m/hr")
    print()

    results = []
    for k in K_GRID:
        print(f"--- k_S_Mesophyll = {k:.1f} ---")
        try:
            r = run_one(k)
        except Exception as exc:
            print(f"  FAILED: {exc}")
            continue
        results.append(r)
        print(f"  ret={r['ret']}")
        print(f"  An: {r['An_total']:.2f} mmol Suc  (24h cumulative)")
        print(f"  Where it goes:  cyto={r['dQ_Meso']:.2f} ({100*r['dQ_Meso']/r['An_total']:.1f}%) "
              f"starch={r['dQ_S_Meso']:.2f} ({100*r['starch_frac']:.1f}%) "
              f"phloem-loaded={r['Q_Fl']:.2f} ({100*r['load_eff']:.1f}%)")
        print(f"  Cmeso estimate (final): {r['c_meso_estimate']:.2f} mmol/cm^3")
        print(f"  C_ST_source (final):    {r['c_st_src']:.3f} mmol/cm^3")
        print(f"  JW_basal:               {r['JW_basal']:.4e}")
        print(f"  v anat:                 {r['v_anat_m_per_hr']:.4f} m/hr  "
              f"{'PASS' if r['in_babst_anat'] else 'FAIL'}")
        print(f"  v Jensen-mid (0.4):     {r['v_jensen04_m_per_hr']:.4f} m/hr  "
              f"{'PASS' if r['in_babst_jensen'] else 'FAIL'}")
        print()

    # Summary grid
    print("=" * 100)
    print("SUMMARY")
    print("=" * 100)
    hdr = (f"{'k_S_M':>8} {'load_eff':>9} {'storage':>9} {'starch':>9} "
           f"{'Cmeso~':>9} {'C_ST_src':>9} {'v_anat':>10} {'v_J04':>10} "
           f"{'PASS':>5}")
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        n_pass = int(r["in_babst_jensen"]) + int(r["in_babst_anat"])
        print(f"{r['k_S_Mesophyll']:>8.1f} "
              f"{100*r['load_eff']:>8.1f}% "
              f"{100*r['storage_frac']:>8.1f}% "
              f"{100*r['starch_frac']:>8.1f}% "
              f"{r['c_meso_estimate']:>9.3f} "
              f"{r['c_st_src']:>9.3f} "
              f"{r['v_anat_m_per_hr']:>10.4f} "
              f"{r['v_jensen04_m_per_hr']:>10.4f} "
              f"{('JENSEN' if r['in_babst_jensen'] else ('ANAT' if r['in_babst_anat'] else '-')):>5}")

    print()
    print("Comparators:")
    print(f"  Stitt 2012 maize source leaf: ~30-50% to starch, ~30-50% to phloem")
    print(f"  Real cytosolic Cmeso physical: ~0.1-0.5 mmol/cm^3")
    print(f"  Babst 2022 v window:           {V_BABST_WIN[0]:.2f} - {V_BABST_WIN[1]:.2f} m/hr")
    print()

    # JSON dump
    out = REPO_ROOT / "dart/coupling/scripts/_pm_starch_sweep.json"
    with open(out, 'w') as f:
        json.dump({
            "config": {
                "C_targMesophyll": C_TARG_RESTORED,
                "Vmax_S_Mesophyll": VMAX_S_MESO_RESTORED,
                "Vmaxloading": VMAX_LOADING,
                "beta_loading": BETA_LOADING,
                "n_substeps": N_SUBSTEPS,
            },
            "k_grid": K_GRID,
            "babst_v_window": list(V_BABST_WIN),
            "jensen_factor": JENSEN_FACTOR_MID,
            "results": results,
        }, f, indent=2)
    print(f"JSON dump: {out}")

    p = REPO_ROOT / "dart/coupling/scripts/_pm_starch.txt"
    if p.exists():
        p.unlink()


if __name__ == "__main__":
    main()
