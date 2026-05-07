"""pm_v3_mass_balance_and_v.py — Steps (1) + (2) of the v-shortfall
repair plan from 2026-05-07.

Step (1): apply the Jensen 2012 / Thompson & Holbrook 2003 "effective
transport area" correction to the velocity probe. The maize JSON header
already cites this factor (beta_plate = 0.5) but only applies it to
kx_st (resistance), not to the v-output probe. The probe currently
divides JW_ST by anatomical Across_st, which over-counts inactive
sieve elements and plate-blocked area. Babst's 11C tracer measures
travel time through *active* SE only -- so to compare apples-to-apples
we need v = JW_ST / (frac_active * Across_anatomical) where frac_active
is in [0.3, 0.5] per Jensen 2012. Defaults: 0.4 midpoint.

Step (2): per-substep plant-level mass balance to localise where An is
going. PiafMunch exposes cumulative Q_out per state at end-of-step.
By differencing consecutive substeps we recover the per-step rates of:
  - An delivered (from get_net_assimilation, mmol Suc/d * dt)
  - delta Q_Mesophyll  (cytosolic sucrose change)
  - delta Q_S_Mesophyll (mesophyll starch change)
  - delta Q_ST         (phloem sieve tube content change)
  - delta Q_Rm         (cumulative respiration delta)
  - delta Q_Gr         (cumulative growth-resp delta)
  - delta Q_Exud       (cumulative exudation delta)
  - Q_Fl implied       = An_step - delta Q_Meso - delta Q_S_Meso

These let us read off:
  loading_efficiency = Q_Fl / An_step
  storage_fraction   = (delta Q_Meso + delta Q_S_Meso) / An_step
  phloem_sink_split  = (delta Q_Rm, delta Q_Gr, delta Q_Exud) / Q_Fl

Why we expect this to be useful: in our V3 maize Vmax=0.20 / beta=2.0
sweep cell (which is 2/3 PASS Babst), the v shortfall is 6x. Possible
decompositions:

  - loading_efficiency low -> phloem under-supplied -> v low
  - storage_fraction high  -> An accumulates locally instead of flowing
  - phloem split off       -> sucrose loaded but consumed locally to
                              maintenance, not transported

This script puts numbers on each.

Run this AT the JSON default best-2/3 cell so we measure the model
in its publishable regime, not at an extreme parameter point. No
overrides except Vmaxloading=0.20 (the prior session's best cell).

Babst windows (Table A1, x=0.5 case):
  delta_P  : [0.51, 2.27] MPa
  v        : [0.55, 1.35] m/hr
  C_ST_src : [0.105, 0.465] mmol/cm^3
"""

import os
import sys
import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from dart.coupling.config import (  # noqa: E402
    DEFAULT_XML, get_hydraulics_json, get_photosynthesis_json, get_phloem_json,
)
from dart.coupling.growth import grow_plant  # noqa: E402

# Step (1) constants
JENSEN_FACTOR_LOW  = 0.30
JENSEN_FACTOR_MID  = 0.40
JENSEN_FACTOR_HIGH = 0.50

VMAX_OVERRIDE = 0.20   # JSON default best-2/3 cell from joint sweep
BETA_OVERRIDE = 2.0    # JSON default
N_SUBSTEPS = 6         # hourly

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
DP_WIN = (BABST["delta_p_MPa"][0]   - 2*BABST["delta_p_MPa"][1],
          BABST["delta_p_MPa"][0]   + 2*BABST["delta_p_MPa"][1])
V_WIN  = (BABST["v_m_per_hr"][0]    - 2*BABST["v_m_per_hr"][1],
          BABST["v_m_per_hr"][0]    + 2*BABST["v_m_per_hr"][1])
CST_WIN = (BABST["c_st_mmol_cm3"][0] - 2*BABST["c_st_mmol_cm3"][1],
           BABST["c_st_mmol_cm3"][0] + 2*BABST["c_st_mmol_cm3"][1])

T_K = 20.75 + 273.15
RT_MPa_per_mmol_cm3 = 8.314 * T_K * 1e-3
CMH2O_TO_MPA = 9.80665e-5


def _suppress():
    o1 = os.dup(1); o2 = os.dup(2); dn = os.open(os.devnull, os.O_WRONLY)
    os.dup2(dn, 1); os.dup2(dn, 2)
    return o1, o2, dn


def _restore(o1, o2, dn):
    os.dup2(o1, 1); os.dup2(o2, 2)
    os.close(dn); os.close(o1); os.close(o2)


def main():
    age = 21
    Tair_C = 20.75
    Tair_K = Tair_C + 273.15
    dt_days = 1.0 / 24.0

    print("=" * 100)
    print("V3 maize: mass balance + Jensen-corrected v probe")
    print("=" * 100)
    print(f"  Cell: Vmaxloading={VMAX_OVERRIDE}, beta_loading={BETA_OVERRIDE} (JSON default best-2/3)")
    print(f"  Babst window v: [{V_WIN[0]:.2f}, {V_WIN[1]:.2f}] m/hr")
    print()

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
    n_root = int(np.sum(organ_types == 2))
    n_stem = int(np.sum(organ_types == 3))
    n_leaf = int(np.sum(organ_types == 4))
    print(f"V{age} plant: segs root={n_root} stem={n_stem} leaf={n_leaf}  (total {n_segs})")

    from plantbox.functional.phloem_flux import PhloemFluxPython
    from plantbox.functional.PlantHydraulicParameters import PlantHydraulicParameters

    params = PlantHydraulicParameters()
    params.read_parameters(get_hydraulics_json())

    hm = PhloemFluxPython(plant, params, psiXylInit=-500, ciInit=350e-6)
    hm.read_photosynthesis_parameters(filename=get_photosynthesis_json())
    hm.read_phloem_parameters(filename=get_phloem_json())
    hm.atol = 1e-6; hm.rtol = 1e-4; hm.solver = 32
    hm.useCWGr = False
    hm.Vmaxloading  = VMAX_OVERRIDE
    hm.beta_loading = BETA_OVERRIDE

    rh = 0.6; psi_soil = -500.0
    p_s = np.linspace(psi_soil, psi_soil - 200, 200)
    es = hm.get_es(Tair_C); ea = es * rh
    par = 600.0 * 1e-6 * 86400 * 1e-4
    hm.solve(sim_time=age, rsx=p_s, cells=True, ea=ea, es=es,
             PAR=par, TairC=Tair_C, verbose=0)

    An = np.array(hm.get_net_assimilation())
    An_total_mmol_co2_per_d = float(np.sum(An)) * 1e3
    SUC_PER_CO2 = 1.0 / 12.0
    An_total_mmol_suc_per_d = An_total_mmol_co2_per_d * SUC_PER_CO2
    print(f"An total: {An_total_mmol_co2_per_d:.2f} mmol CO2/d  "
          f"(= {An_total_mmol_suc_per_d:.2f} mmol Suc/d)")
    print()

    Nt = len(plant.getNodes())
    nodes = plant.getNodes()
    node_z = np.array([n.z for n in nodes], dtype=np.float64)

    # Q_out layout per solve.cpp:195-207, slot index i -> Q_out[i*Nt:(i+1)*Nt]
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

    def slot_sum(Q_out, name):
        i = SLOT[name]
        return float(np.sum(Q_out[i*Nt:(i+1)*Nt]))

    # Run substeps and capture state after each
    print(f"Running {N_SUBSTEPS} substeps of {dt_days*24:.1f} h each...")
    cumulative = []     # list of dicts of total Q values
    JW_basal_per_step = []
    C_ST_src_per_step = []
    delta_P_per_step  = []

    fdpair = _suppress()
    try:
        for step in range(1, N_SUBSTEPS + 1):
            if step > 1:
                hm.withInitVal = False
            t_start = float(age) + (step - 1) * dt_days
            t_end   = t_start + dt_days
            ret = hm.startPM(t_start, t_end, 1, Tair_K, False,
                             str(REPO_ROOT / "dart/coupling/scripts/_pm_mb.txt"))
            Q_out = np.array(hm.Q_out)
            cumulative.append({k: slot_sum(Q_out, k) for k in SLOT})
            cumulative[-1]["ret"] = int(ret)

            # capture v / Cst for this substep
            C_ST  = np.array(hm.C_ST)
            JW_ST = np.array(hm.JW_ST)
            psi_xyl = np.array(hm.psiXyl)

            seg_ot = np.zeros(Nt, dtype=np.int32)
            for si in range(n_segs):
                nodeID = si + 1
                if nodeID < Nt:
                    seg_ot[nodeID] = int(organ_types[si])
            leaf_mask = seg_ot == 4
            stem_mask = seg_ot == 3
            root_mask = seg_ot == 2
            src_mask = leaf_mask & (C_ST > 0.21)

            if src_mask.any():
                c_st_src = float(np.percentile(C_ST[src_mask], 75))
                psi_src  = float(np.mean(psi_xyl[src_mask]))
            else:
                c_st_src = float(np.max(C_ST[leaf_mask])) if leaf_mask.any() else 0.0
                psi_src  = float(np.mean(psi_xyl[leaf_mask])) if leaf_mask.any() else 0.0

            if root_mask.any():
                deep_idx = np.argsort(node_z[root_mask])[:max(5, root_mask.sum() // 50)]
                rids = np.where(root_mask)[0][deep_idx]
                c_st_sink = float(np.mean(C_ST[rids]))
                psi_sink  = float(np.mean(psi_xyl[rids]))
            else:
                c_st_sink = float(np.min(C_ST))
                psi_sink  = float(np.min(psi_xyl))

            P_src  = psi_src  * CMH2O_TO_MPA + RT_MPa_per_mmol_cm3 * c_st_src
            P_sink = psi_sink * CMH2O_TO_MPA + RT_MPa_per_mmol_cm3 * c_st_sink
            delta_P_per_step.append(P_src - P_sink)

            # basal stem JW_ST
            if stem_mask.any():
                stem_ids = np.where(stem_mask)[0]
                basal_id = stem_ids[np.argmin(np.abs(node_z[stem_ids]))]
                JW_basal_per_step.append(abs(float(JW_ST[basal_id])))
            else:
                JW_basal_per_step.append(0.0)

            C_ST_src_per_step.append(c_st_src)

            if ret != 1:
                break
    finally:
        _restore(*fdpair)

    Across_anat = float(hm.Across_st[1][0])      # cm^2
    print(f"Across_st (anatomical, hm.Across_st[1][0]): {Across_anat:.4e} cm^2")
    print()

    # ---------------------------------------------------------------
    # Step 2: Mass balance per substep
    # ---------------------------------------------------------------
    print("=" * 100)
    print(f"PER-SUBSTEP MASS BALANCE (units: mmol Suc per {dt_days*24:.1f} h substep)")
    print("=" * 100)

    # An delivered per step (constant): An_total_mmol_suc_per_d * dt_days
    An_per_step = An_total_mmol_suc_per_d * dt_days

    hdr = (f"{'h':>3} {'An':>8} {'dQ_Meso':>9} {'dQ_S_Meso':>10} {'dQ_ST':>8} "
           f"{'dQ_Rm':>8} {'dQ_Gr':>8} {'dQ_Exud':>9} {'Q_Fl':>8} "
           f"{'load_eff':>9} {'storage_frac':>13}")
    print(hdr)
    print("-" * len(hdr))
    prev = {k: 0.0 for k in SLOT}
    rows = []
    for h, c in enumerate(cumulative, start=1):
        d_Meso   = c["Q_Mesophyll"]   - prev["Q_Mesophyll"]
        d_SMeso  = c["Q_S_Mesophyll"] - prev["Q_S_Mesophyll"]
        d_ST     = c["Q_ST"]          - prev["Q_ST"]
        d_Rm     = c["Q_RespMaint"]   - prev["Q_RespMaint"]
        d_Gr     = c["Q_Growthtot"]   - prev["Q_Growthtot"]
        d_Exud   = c["Q_Exudation"]   - prev["Q_Exudation"]
        # Q_Fl per Q_Mesophyll_dot = Ag - Q_Fl - Q_S_Meso_dot
        # =>  Q_Fl = An - dQ_Meso - dQ_S_Meso
        Q_Fl     = An_per_step - d_Meso - d_SMeso
        load_eff = Q_Fl / An_per_step if An_per_step > 0 else 0.0
        store_frac = (d_Meso + d_SMeso) / An_per_step if An_per_step > 0 else 0.0
        print(f"{h:>3} {An_per_step:>8.4f} {d_Meso:>9.4f} {d_SMeso:>10.4f} "
              f"{d_ST:>8.4f} {d_Rm:>8.4f} {d_Gr:>8.4f} {d_Exud:>9.4f} "
              f"{Q_Fl:>8.4f} {load_eff:>9.3f} {store_frac:>13.3f}")
        rows.append(dict(h=h, An=An_per_step, d_Meso=d_Meso, d_SMeso=d_SMeso,
                         d_ST=d_ST, d_Rm=d_Rm, d_Gr=d_Gr, d_Exud=d_Exud,
                         Q_Fl=Q_Fl, load_eff=load_eff, store_frac=store_frac))
        prev = c

    # Totals
    print("-" * len(hdr))
    sum_An      = sum(r["An"]      for r in rows)
    sum_dMeso   = sum(r["d_Meso"]  for r in rows)
    sum_dSMeso  = sum(r["d_SMeso"] for r in rows)
    sum_dST     = sum(r["d_ST"]    for r in rows)
    sum_dRm     = sum(r["d_Rm"]    for r in rows)
    sum_dGr     = sum(r["d_Gr"]    for r in rows)
    sum_dExud   = sum(r["d_Exud"]  for r in rows)
    sum_Q_Fl    = sum_An - sum_dMeso - sum_dSMeso
    print(f"{'TOT':>3} {sum_An:>8.4f} {sum_dMeso:>9.4f} {sum_dSMeso:>10.4f} "
          f"{sum_dST:>8.4f} {sum_dRm:>8.4f} {sum_dGr:>8.4f} {sum_dExud:>9.4f} "
          f"{sum_Q_Fl:>8.4f} "
          f"{(sum_Q_Fl/sum_An if sum_An > 0 else 0):>9.3f} "
          f"{((sum_dMeso + sum_dSMeso)/sum_An if sum_An > 0 else 0):>13.3f}")

    print()
    print("Carbon-budget closure check:")
    bal_meso = sum_An - sum_dMeso - sum_dSMeso - sum_Q_Fl
    bal_phloem = sum_Q_Fl - sum_dST - sum_dRm - sum_dGr - sum_dExud
    print(f"  An - dQ_Meso - dQ_S_Meso - Q_Fl     = {bal_meso:.6f} mmol Suc  (mesophyll node)")
    print(f"  Q_Fl - dQ_ST - dQ_Rm - dQ_Gr - dQ_Exud = {bal_phloem:.6f} mmol Suc  (phloem node)")
    print("  (mesophyll closure is identity by construction; phloem closure tests")
    print("   whether dQ_ST + dQ_Rm + dQ_Gr + dQ_Exud accounts for all loaded sucrose)")

    # ---------------------------------------------------------------
    # Step 1: v probe with Jensen 2012 effective area
    # ---------------------------------------------------------------
    print()
    print("=" * 100)
    print(f"V PROBE — anatomical vs Jensen-corrected (last substep, h={N_SUBSTEPS})")
    print("=" * 100)

    JW_basal_last = JW_basal_per_step[-1]
    # JW_ST in cm^3/d (per the unit-chain analysis: R=83.14 hPa cm^3/(K mmol),
    # mu in hPa d, so r_ST in hPa d / cm^3 -> JW in cm^3/d).
    # Convert to m/hr: cm^3/d / cm^2 = cm/d ; cm/d -> m/hr is /2400.
    print(f"JW_ST[basal] (cm^3/d, units per unit-chain analysis): {JW_basal_last:.4e}")
    print(f"  Note: prior diagnostic scripts treated this as cm^3/h (legacy")
    print(f"  PiafMunch GUI comment); empirical unit check on a junction segment")
    print(f"  was inconclusive. Treat conservative.")
    print()

    def v_for(area_factor, time_unit):
        A_eff = area_factor * Across_anat
        if time_unit == 'd':
            v_cm_per_d = JW_basal_last / A_eff
            return v_cm_per_d / 2400.0    # cm/d -> m/h
        if time_unit == 'h':
            v_cm_per_h = JW_basal_last / A_eff
            return v_cm_per_h * 1e-2      # cm/h -> m/h
        raise ValueError

    print(f"{'config':<45} {'v [m/hr]':>10} {'in Babst window?':>20}")
    print("-" * 80)
    rows_v = []
    for desc, factor in [
        ("anatomical (legacy)", 1.0),
        (f"Jensen low (frac={JENSEN_FACTOR_LOW})", JENSEN_FACTOR_LOW),
        (f"Jensen mid (frac={JENSEN_FACTOR_MID})", JENSEN_FACTOR_MID),
        (f"Jensen high (frac={JENSEN_FACTOR_HIGH})", JENSEN_FACTOR_HIGH),
    ]:
        for tu in ('h', 'd'):
            v = v_for(factor, tu)
            in_win = V_WIN[0] <= v <= V_WIN[1]
            label = f"{desc}  (JW assumed cm^3/{tu})"
            print(f"{label:<45} {v:>10.4f} {('PASS' if in_win else 'FAIL'):>20}")
            rows_v.append(dict(desc=desc, time_unit=tu, factor=factor,
                               v_m_per_hr=v, in_babst=bool(in_win)))

    print()
    print("=" * 100)
    print("DIAGNOSIS")
    print("=" * 100)

    # Storage fraction
    if sum_An > 0:
        store_pct = 100.0 * (sum_dMeso + sum_dSMeso) / sum_An
        load_pct = 100.0 * sum_Q_Fl / sum_An
        rm_pct = 100.0 * sum_dRm / sum_An
        gr_pct = 100.0 * sum_dGr / sum_An
        exud_pct = 100.0 * sum_dExud / sum_An
        st_pct = 100.0 * sum_dST / sum_An
        print(f"  Where An is going (over {N_SUBSTEPS}h):")
        print(f"    Mesophyll storage (cyto + starch):  {store_pct:>6.1f}%")
        print(f"    Phloem loaded (Q_Fl):                {load_pct:>6.1f}%")
        print(f"      ... of which:")
        print(f"        retained in sieve tube (dQ_ST): {st_pct:>6.1f}% of An")
        print(f"        respired (Rm + Gr):             {rm_pct + gr_pct:>6.1f}% of An")
        print(f"        exuded:                         {exud_pct:>6.1f}% of An")
        print()
        if store_pct > 60:
            print(f"  --> Storage dominates. {store_pct:.0f}% of An accumulates in mesophyll")
            print( "       instead of entering the phloem. Loading is the bottleneck.")
        elif load_pct < 30:
            print(f"  --> Loading capacity bottleneck. Only {load_pct:.0f}% of An enters phloem.")
            print( "       Investigate len_leaf coverage or Mloading saturation.")
        else:
            print(f"  --> Loading efficiency {load_pct:.0f}% is in the realistic range.")
            print( "       v shortfall is downstream of loading; investigate flow side.")

    p = REPO_ROOT / "dart/coupling/scripts/_pm_mb.txt"
    if p.exists():
        p.unlink()

    # JSON dump for downstream use
    out = REPO_ROOT / "dart/coupling/scripts/_pm_mass_balance_v3.json"
    with open(out, 'w') as f:
        json.dump({
            "config": {"Vmaxloading": VMAX_OVERRIDE, "beta_loading": BETA_OVERRIDE,
                       "n_substeps": N_SUBSTEPS, "age_days": age},
            "An_total_mmol_suc_per_d": An_total_mmol_suc_per_d,
            "Across_anatomical_cm2": Across_anat,
            "per_step": rows,
            "v_probes": rows_v,
            "totals": {
                "sum_An": sum_An,
                "sum_dQ_Meso": sum_dMeso,
                "sum_dQ_S_Meso": sum_dSMeso,
                "sum_dQ_ST": sum_dST,
                "sum_dQ_Rm": sum_dRm,
                "sum_dQ_Gr": sum_dGr,
                "sum_dQ_Exud": sum_dExud,
                "sum_Q_Fl_implied": sum_Q_Fl,
                "loading_efficiency": sum_Q_Fl / sum_An if sum_An > 0 else 0,
                "storage_fraction":  (sum_dMeso + sum_dSMeso) / sum_An if sum_An > 0 else 0,
            },
        }, f, indent=2)
    print(f"\nJSON dump: {out}")


if __name__ == "__main__":
    main()
