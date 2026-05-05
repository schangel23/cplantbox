"""pm_v3_babst_comparison.py — Step -1: V3 maize vs Babst 2022 Table A1.

Runs PiafMunch on a 21-day maize plant under Babst growth-chamber proxy
forcing (T_mean=20.75 C, 14:10 PP, 600 umol/m2/s PPFD) and compares
modeled source-phloem [Suc], turgor pressure, and sap velocity to
Babst Table A1 at 2-sigma tolerance.

Measurement targets (memory: reference_babst_2022_phloem.md, x=0.5 case):
  Source ΔP  : 1.39 ± 0.44 MPa
  Sap v       : 0.95 ± 0.20 m/hr  (Darcy mid-point of measured 0.81-1.16)
  C_ST source : 0.285 ± 0.090 mmol/cm³

Caveats:
- Babst plants were 3 weeks old (V3); we approximate by simulating 21 d
  under chamber proxy met. Real Babst plants were grown in 75% Promix +
  25% sand + Osmocote — no exact rooting analogue here.
- ΔP in PiafMunch is computed as turgor difference source-vs-sink:
      P = psi_xyl + R*T*C_ST
  for nodes with C_ST > 0. ΔP reported = P_max(leaf) - P_min(root_tip).
- Sap velocity = JW_ST[stem] / Across_ST[stem] at the stem-base node.
"""

import os
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from dart.coupling.config import (  # noqa: E402
    DEFAULT_XML, get_hydraulics_json, get_photosynthesis_json, get_phloem_json,
)
from dart.coupling.growth import grow_plant  # noqa: E402

# --- Babst 2022 chamber proxy met (constant 21 d at 20.75 C, DLI ~30 mol/m2/d)
BABST_MET = {
    d: {
        "T_mean_C": 20.75,
        "T_min_C": 19.0,
        "T_max_C": 22.0,
        "PAR_MJ_m2_d": 30.0 * 0.219,   # 30 mol/m²/d → ~6.57 MJ/m²/d
        "VPD_kPa": 1.0,
        "RH_pct": 60.0,
        "Wind_m_s": 0.5,
    } for d in range(1, 31)
}

# Babst Table A1 (x=0.5; matches XRF K+ ≈ 297 mol/m³)
BABST = {
    "delta_p_MPa":     (1.39, 0.44),
    "v_m_per_hr":      (0.95, 0.20),
    "c_st_mmol_cm3":   (0.285, 0.090),
}
TOL_SIGMA = 2.0

# Constants
R_GAS = 8.314e-6   # MPa·cm³ / (mmol·K)  =  J/(mol·K) * 1e-6 MPa/Pa * cm³/mol scaling
T_K   = 20.75 + 273.15
RT    = R_GAS * T_K  # ~2.44e-3 MPa·cm³/mmol  ?? let's recompute carefully
# R = 8.314 J/(mol·K). RT[J/mol] = 8.314 × 293.9 = 2444 J/mol.
# 1 mol Suc/m³ × RT [J/mol] = 2444 J/m³ = 2444 Pa = 0.002444 MPa.
# So 1 mmol Suc/cm³ = 1000 mol/m³ → osmotic Π = 2.44 MPa.
RT_MPa_per_mmol_cm3 = 8.314 * T_K * 1e-3  # = 2.443 MPa per mmol/cm³  [verified]
# Note: psi_xyl in CPlantBox is [cm pressure head]. 1 cm H2O = 9.81e-5 MPa.
CMH2O_TO_MPA = 9.80665e-5


def _suppress():
    o1 = os.dup(1); o2 = os.dup(2); dn = os.open(os.devnull, os.O_WRONLY)
    os.dup2(dn, 1); os.dup2(dn, 2)
    return o1, o2, dn


def _restore(o1, o2, dn):
    os.dup2(o1, 1); os.dup2(o2, 2)
    os.close(dn); os.close(o1); os.close(o2)


def main():
    print("=" * 78)
    print("Step -1: V3 maize PiafMunch vs Babst 2022 Table A1")
    print("=" * 78)
    print(f"  RT (osmotic-to-MPa coeff at {T_K-273.15:.1f}C): "
          f"{RT_MPa_per_mmol_cm3:.4f} MPa per (mmol/cm³)")

    age = 21
    Tair_C = 20.75

    # 1) Grow V3 plant under Babst chamber-proxy met
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
    sub_types   = np.array(plant.subTypes,   dtype=np.int32)
    n_root = int(np.sum(organ_types == 2))
    n_stem = int(np.sum(organ_types == 3))
    n_leaf = int(np.sum(organ_types == 4))
    n_segs = len(plant.getSegments())
    n_nodes = len(plant.getNodes())
    print(f"\nPlant V{age}d: segs root={n_root} stem={n_stem} leaf={n_leaf} "
          f"({n_segs} segs / {n_nodes} nodes)")

    # 2) Build PhloemFluxPython
    from plantbox.functional.phloem_flux import PhloemFluxPython
    from plantbox.functional.PlantHydraulicParameters import PlantHydraulicParameters

    params = PlantHydraulicParameters()
    params.read_parameters(get_hydraulics_json())

    hm = PhloemFluxPython(plant, params, psiXylInit=-500, ciInit=350e-6)
    hm.read_photosynthesis_parameters(filename=get_photosynthesis_json())
    hm.read_phloem_parameters(filename=get_phloem_json())
    hm.atol = 1e-6; hm.rtol = 1e-4; hm.solver = 32
    hm.useCWGr = False

    # 3) Photosynthesis under chamber PAR (600 umol/m2/s)
    rh = 0.6; psi_soil = -500.0
    p_s = np.linspace(psi_soil, psi_soil - 100, 100)
    es = hm.get_es(Tair_C); ea = es * rh
    par = 600.0 * 1e-6 * 86400 * 1e-4
    hm.solve(sim_time=age, rsx=p_s, cells=True, ea=ea, es=es,
             PAR=par, TairC=Tair_C, verbose=0)
    An = np.array(hm.get_net_assimilation())
    An_total = float(np.sum(An)) * 1e3
    print(f"An_total (V{age}, PAR=600) = {An_total:.1f} mmol CO2/d  ({len(An)} leaf segs)")

    # 4) Run startPM for 6 hourly substeps to let the transient settle
    Tair_K = Tair_C + 273.15
    dt_days = 1.0 / 24.0
    Nt = len(plant.getNodes())
    n_substeps = 6

    fdpair = _suppress()
    try:
        for step in range(1, n_substeps + 1):
            if step > 1:
                hm.withInitVal = False
            t_start = float(age) + (step - 1) * dt_days
            t_end   = t_start + dt_days
            ret = hm.startPM(t_start, t_end, 1, Tair_K, False,
                             str(REPO_ROOT / "dart/coupling/scripts/_pm_v3.txt"))
            if ret != 1:
                break
    finally:
        _restore(*fdpair)

    # 5) Extract per-node arrays
    C_ST    = np.array(hm.C_ST)        # mmol Suc / cm³
    psi_xyl = np.array(hm.psiXyl)      # cm pressure head
    JW_ST   = np.array(hm.JW_ST)       # cm³ / d  (volumetric phloem flux)
    Q_Rmmax = np.array(hm.Q_Rmmax)     # mmol Suc / d (per node, base rate)
    Q_Grmax = np.array(hm.Q_Grmax)
    Q_total = np.array(hm.Q_out)
    QRm   = float(np.sum(Q_total[Nt*2:Nt*3]))
    QGr   = float(np.sum(Q_total[Nt*4:Nt*5]))
    QExud = float(np.sum(Q_total[Nt*3:Nt*4]))
    print(f"Cumulative {n_substeps}h: Rm={QRm:.3f}  Gr={QGr:.3f}  Exud={QExud:.3f} mmol Suc")
    print(f"Daily-rate totals    : Q_Rmmax={float(Q_Rmmax.sum()):.3f}  "
          f"Q_Grmax={float(Q_Grmax.sum()):.3f} mmol/d")

    # 6) Map node -> organ class via segments
    seg_node = np.zeros(Nt, dtype=np.int32)  # 0=unmapped, else organ_type
    seg_st   = np.zeros(Nt, dtype=np.int32)
    for si in range(n_segs):
        nodeID = si + 1
        if nodeID < Nt:
            seg_node[nodeID] = int(organ_types[si])
            seg_st[nodeID]   = int(sub_types[si])

    leaf_mask = seg_node == 4
    stem_mask = seg_node == 3
    root_mask = seg_node == 2
    src_mask  = leaf_mask & (C_ST > 0.21)  # source nodes with active loading

    if not src_mask.any():
        print("\n!! No source-leaf nodes with C_ST > 0.21 — model is fully gate-locked.")
        c_st_src = float(np.max(C_ST[leaf_mask])) if leaf_mask.any() else 0.0
        psi_src  = float(np.mean(psi_xyl[leaf_mask])) if leaf_mask.any() else 0.0
    else:
        c_st_src = float(np.percentile(C_ST[src_mask], 75))   # representative source
        psi_src  = float(np.mean(psi_xyl[src_mask]))

    # Sink: pick deepest root nodes (z most negative)
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

    # 7) ΔP source-to-sink = (psi_xyl + RT*C_ST) source - sink
    P_src  = psi_src  * CMH2O_TO_MPA + RT_MPa_per_mmol_cm3 * c_st_src
    P_sink = psi_sink * CMH2O_TO_MPA + RT_MPa_per_mmol_cm3 * c_st_sink
    delta_P = P_src - P_sink

    # 8) Sap velocity at the basal stem node
    if stem_mask.any():
        # Pick the stem node closest to the collar (smallest |z|)
        stem_ids = np.where(stem_mask)[0]
        basal_id = stem_ids[np.argmin(np.abs(node_z[stem_ids]))]
        Across_stem = hm.Across_st[1][0]   # PerType[stem][mainstem subtype]
        flux_cm3_d  = abs(float(JW_ST[basal_id]))
        v_cm_per_d  = flux_cm3_d / Across_stem if Across_stem > 0 else 0.0
        v_m_per_hr  = v_cm_per_d * 1e-2 / 24.0
    else:
        v_m_per_hr  = 0.0
        Across_stem = 0.0
        flux_cm3_d  = 0.0

    # 9) Comparison table
    print("\n" + "-" * 78)
    print(f"{'metric':<22} {'model':>12} {'Babst µ':>10} {'±σ':>8} {'verdict':>8}")
    print("-" * 78)
    rows = [
        ("ΔP source-sink [MPa]", delta_P,         *BABST["delta_p_MPa"]),
        ("v sap basal [m/hr]",   v_m_per_hr,      *BABST["v_m_per_hr"]),
        ("C_ST source [mmol/cm³]", c_st_src,      *BABST["c_st_mmol_cm3"]),
    ]
    pass_count = 0
    for name, val, mu, sd in rows:
        within = abs(val - mu) <= TOL_SIGMA * sd
        verdict = "PASS" if within else "FAIL"
        pass_count += int(within)
        print(f"{name:<22} {val:>12.3f} {mu:>10.3f} {sd:>8.3f} {verdict:>8}")
    print("-" * 78)
    print(f"  C_ST_max (any node)  : {float(np.max(C_ST)):.3f} mmol/cm³")
    print(f"  C_ST_mean leaf       : {float(np.mean(C_ST[leaf_mask])) if leaf_mask.any() else 0.0:.3f}")
    print(f"  C_ST_mean root       : {float(np.mean(C_ST[root_mask])) if root_mask.any() else 0.0:.3f}")
    print(f"  Source nodes (>0.21) : {int(src_mask.sum())} / {int(leaf_mask.sum())} leaf nodes")
    print(f"  basal stem JW_ST     : {flux_cm3_d:.3e} cm³/d")
    print(f"  basal stem Across    : {Across_stem:.3e} cm²")

    # 10) Verdict
    print("\n" + "=" * 78)
    if pass_count == 3:
        print("VERDICT: 3/3 PASS — V3 model IS within Babst's 2σ window.")
        print("  Architecture/regime is calibrated correctly at the validated scale.")
        print("  Day-55 cuse-gate behaviour is extrapolation-related rather than")
        print("  parameter regime; coarsening could be useful if it returns.")
    elif pass_count == 0:
        print("VERDICT: 0/3 FAIL — V3 model is NOT in Babst's 2σ window.")
        print("  Parameter regime is wrong even at validated scale.")
        print("  Loading + maintenance recalibration warranted (β' + α').")
    else:
        print(f"VERDICT: {pass_count}/3 PASS — partial.")
        print("  Narrow recalibration to failing arm only; see plan note for branching.")

    p = REPO_ROOT / "dart/coupling/scripts/_pm_v3.txt"
    if p.exists():
        p.unlink()


if __name__ == "__main__":
    main()
