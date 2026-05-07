"""pm_jwst_unit_check.py — empirically determine the time unit of JW_ST.

Two competing readings:

  (A) ml/h (per the legacy comment at solve.cpp:78 inherited from the
      original Lacointe/Minchin PiafMunch GUI code)
  (B) cm^3/d (per the unit chain implied by the CPlantBox port:
      R=83.14 in hPa cm^3/(K mmol), mu in hPa d, r_ST = mu * (l/kx_st),
      kx_st in cm^4 (or cm^3 day^-1 hPa^-1 depending on which comment
      you trust), JW_ST = dP/r_ST -> cm^3/d.)

Reading (B) implies our diagnostic scripts under-report v by 24x,
in which case the V3 sweep already passes Babst. Reading (A) implies
the v shortfall is real and structural.

Strategy
--------
Run a fresh V3 startPM call, extract:
  - stem-segment ID and its node IDs
  - kx_st(stem) in cm^4 (per maize_phloem_2026.py header comment)
  - segment length l in cm
  - upstream and downstream P_ST values
  - C_amont (upstream sucrose conc.) at that segment
  - r_ST and JW_ST as the C++ engine reports them

Then compute, for that one segment, what JW_ST SHOULD be in
both candidate time bases:

  pred_cm3_per_h  = (P_up - P_down) [hPa] * kx_st [cm^4] / l [cm] / mu [hPa h]
  pred_cm3_per_d  = (P_up - P_down) [hPa] * kx_st [cm^4] / l [cm] / mu [hPa d]

If reported JW_ST matches pred_cm3_per_h  -> reading (A) wins
If reported JW_ST matches pred_cm3_per_d  -> reading (B) wins

mu(C_amont) is computed via Mathlouthi & Genotelle 1995 (the same
formula used in solve.cpp:172-185) so it's apples-to-apples.

If the V3 sweep was 24x off, this script will say so unambiguously.
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

BABST_MET = {
    d: {"T_mean_C": 20.75, "T_min_C": 19.0, "T_max_C": 22.0,
        "PAR_MJ_m2_d": 30.0 * 0.219, "VPD_kPa": 1.0,
        "RH_pct": 60.0, "Wind_m_s": 0.5}
    for d in range(1, 31)
}


def _suppress():
    o1 = os.dup(1); o2 = os.dup(2); dn = os.open(os.devnull, os.O_WRONLY)
    os.dup2(dn, 1); os.dup2(dn, 2)
    return o1, o2, dn


def _restore(o1, o2, dn):
    os.dup2(o1, 1); os.dup2(o2, 2)
    os.close(dn); os.close(o1); os.close(o2)


def mathlouthi_mu_hPa_unit(C_mmol_cm3, T_C, time_unit):
    """Replicate solve.cpp:172-185 viscosity formula.
    time_unit ∈ {'h', 'd'} controls the final mPa s -> hPa·time conversion.
    """
    TdC = T_C
    dEauPure = (999.83952 + TdC * (16.952577 + TdC * (-0.0079905127
                + TdC * (-0.000046241757 + TdC * (0.00000010584601
                + TdC * (-0.00000000028103006)))))) / (1 + 0.016887236 * TdC)
    siPhi = (30 - TdC) / (91 + TdC)
    C = max(0.0, C_mmol_cm3)
    PartMolalVol_ = 0.0
    d = C * 342.3 + (1 - C * PartMolalVol_) * dEauPure
    siEnne = (100 * 342.30 * C) / d
    siEnne /= 1900 - 18 * siEnne
    mu_mPa_s = 10 ** ((22.46 * siEnne) - 0.114 + (siPhi * (1.1 + 43.1 * siEnne ** 1.25)))
    # mPa s -> hPa second:  1 mPa s = 1e-3 Pa s = 1e-5 hPa s
    mu_hPa_s = mu_mPa_s * 1e-5
    if time_unit == 'h':
        return mu_hPa_s / 3600.0      # hPa h
    if time_unit == 'd':
        return mu_hPa_s / 86400.0     # hPa d
    raise ValueError(time_unit)


def main():
    age = 21
    Tair_C = 20.75
    Tair_K = Tair_C + 273.15

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
    hm.Vmaxloading = 0.20
    hm.beta_loading = 2.0  # JSON default

    rh = 0.6; psi_soil = -500.0
    p_s = np.linspace(psi_soil, psi_soil - 200, 200)
    es = hm.get_es(Tair_C); ea = es * rh
    par = 600.0 * 1e-6 * 86400 * 1e-4
    hm.solve(sim_time=age, rsx=p_s, cells=True, ea=ea, es=es,
             PAR=par, TairC=Tair_C, verbose=0)

    # Run 6 hourly substeps to settle
    dt_days = 1.0 / 24.0
    fdpair = _suppress()
    try:
        for step in range(1, 7):
            if step > 1:
                hm.withInitVal = False
            t_start = float(age) + (step - 1) * dt_days
            t_end = t_start + dt_days
            hm.startPM(t_start, t_end, 1, Tair_K, False,
                       str(REPO_ROOT / "dart/coupling/scripts/_pm_unitcheck.txt"))
    finally:
        _restore(*fdpair)

    # Extract per-segment arrays
    JW_ST   = np.array(hm.JW_ST)        # mystery time base
    r_ST    = np.array(hm.r_ST)         # whatever runPM reports
    r_ST_ref = np.array(hm.r_ST_ref)
    C_ST    = np.array(hm.C_ST)         # mmol/cm3
    psi_xyl = np.array(hm.psiXyl)       # cm pressure head

    # Segment topology
    segs = plant.getSegments()
    nodes = plant.getNodes()
    node_z = np.array([n.z for n in nodes], dtype=np.float64)
    Nt = len(nodes)

    # P_ST in hPa per the unit chain we're testing:
    # P_ST = C * RT + Psi_Xyl_hPa, where R=83.14 hPa cm3 / (K mmol)
    R_HPA = 83.14
    P_ST_hPa = C_ST * R_HPA * Tair_K + psi_xyl * 0.980638  # cmH2O -> hPa

    # Pick a basal mainstem segment (organ_type==3, smallest |z|)
    stem_seg_ids = np.where(organ_types == 3)[0]
    if stem_seg_ids.size == 0:
        print("ERROR: no stem segments found")
        return
    # Pick the segment whose downstream node has smallest |z|
    seg_choice = min(stem_seg_ids, key=lambda si: abs(node_z[segs[si].y]))
    s = segs[seg_choice]
    node_up = s.x
    node_dn = s.y
    seg_idx_in_PM = seg_choice + 1   # PiafMunch uses 1-based segment indices

    # Segment length l (cm)
    l_seg = float(np.linalg.norm(
        np.array([nodes[node_dn].x - nodes[node_up].x,
                  nodes[node_dn].y - nodes[node_up].y,
                  nodes[node_dn].z - nodes[node_up].z])
    ))

    # kx_st(stem, mainstem-subtype) from the model — read off the C++ binding
    kx_st = float(hm.kx_st[1][0])      # PerType[stem][mainstem]
    A_st  = float(hm.Across_st[1][0])
    print("=" * 90)
    print("JW_ST time-base unit cross-check on basal mainstem segment")
    print("=" * 90)
    print(f"  Segment idx (C++ 1-based) : {seg_idx_in_PM}")
    print(f"  Up node idx               : {node_up}, z={nodes[node_up].z:.2f} cm")
    print(f"  Dn node idx               : {node_dn}, z={nodes[node_dn].z:.2f} cm")
    print(f"  Length l                  : {l_seg:.4f} cm")
    print(f"  kx_st (stem, mainstem)    : {kx_st:.4e}  (cm^4 if maize_phloem_2026.py header is right;")
    print(f"                              cm^3/(hPa*d) if JSON description is right)")
    print(f"  Across_st                 : {A_st:.4e} cm^2")

    # P_ST values at the two endpoint nodes
    # (Fortran 1-based -> Python 0-based: node_up/dn are already 0-based plant nodes,
    #  but PiafMunch puts NodeIDs at 1..Nt with Nt-1 actual nodes; check off-by-one.)
    # The simplest robust thing: probe both ways and take the larger gradient.
    P_up_a = P_ST_hPa[node_up] if node_up < len(P_ST_hPa) else float('nan')
    P_dn_a = P_ST_hPa[node_dn] if node_dn < len(P_ST_hPa) else float('nan')
    P_up_b = P_ST_hPa[node_up + 1] if (node_up + 1) < len(P_ST_hPa) else float('nan')
    P_dn_b = P_ST_hPa[node_dn + 1] if (node_dn + 1) < len(P_ST_hPa) else float('nan')

    dP_a = P_up_a - P_dn_a
    dP_b = P_up_b - P_dn_b
    use_one_based = abs(dP_b) > abs(dP_a)
    P_up = P_up_b if use_one_based else P_up_a
    P_dn = P_dn_b if use_one_based else P_dn_a
    dP   = P_up - P_dn
    C_up = C_ST[node_up + 1] if use_one_based else C_ST[node_up]
    C_dn = C_ST[node_dn + 1] if use_one_based else C_ST[node_dn]

    print(f"\n  Indexing (0-based vs 1-based): using "
          f"{'1-based' if use_one_based else '0-based'} for P_ST/C_ST lookup")
    print(f"  P_ST upstream  : {P_up:.3e} hPa")
    print(f"  P_ST downstrm  : {P_dn:.3e} hPa")
    print(f"  Delta P        : {dP:.3e} hPa  ({dP/1e4:.4f} MPa)")
    print(f"  C upstream     : {C_up:.4f} mmol/cm^3")
    print(f"  C downstream   : {C_dn:.4f} mmol/cm^3")

    C_amont = C_up if abs(dP) >= 0 else C_dn  # JW positive convention -> upstream
    mu_h = mathlouthi_mu_hPa_unit(C_amont, Tair_C, 'h')
    mu_d = mathlouthi_mu_hPa_unit(C_amont, Tair_C, 'd')
    print(f"\n  mu(C={C_amont:.3f}, T={Tair_C} C) :")
    print(f"    if hPa h: {mu_h:.3e}")
    print(f"    if hPa d: {mu_d:.3e}")

    # Predicted JW_ST in each candidate time base (same kx_st as cm^4)
    # JW = dP / r_ST = dP / (mu * l / kx_st) = dP * kx_st / (mu * l)
    pred_h = dP * kx_st / (mu_h * l_seg) if mu_h > 0 and l_seg > 0 else float('nan')
    pred_d = dP * kx_st / (mu_d * l_seg) if mu_d > 0 and l_seg > 0 else float('nan')
    print(f"\n  HP-formula prediction for JW at this segment:")
    print(f"    if mu time-base = hour: JW = {pred_h:.4e}  (cm^3/h)")
    print(f"    if mu time-base = day : JW = {pred_d:.4e}  (cm^3/d)")
    print(f"    (these differ by a factor of 24, by construction)")

    # Reported JW_ST at this segment
    if seg_idx_in_PM < len(JW_ST):
        JW_reported = float(JW_ST[seg_idx_in_PM])
    elif seg_idx_in_PM - 1 < len(JW_ST):
        JW_reported = float(JW_ST[seg_idx_in_PM - 1])
    else:
        JW_reported = float('nan')

    print(f"\n  REPORTED JW_ST[seg]      : {JW_reported:.4e}")
    print(f"  Reported r_ST[seg]       : {r_ST[seg_idx_in_PM]:.4e}"
          if seg_idx_in_PM < len(r_ST) else "  Reported r_ST[seg]       : (out of bounds)")
    print(f"  Reported r_ST_ref[seg]   : {r_ST_ref[seg_idx_in_PM]:.4e}"
          if seg_idx_in_PM < len(r_ST_ref) else "  Reported r_ST_ref[seg]   : (out of bounds)")

    print("\n" + "=" * 90)
    print("VERDICT")
    print("=" * 90)

    if not (np.isfinite(pred_h) and np.isfinite(pred_d) and np.isfinite(JW_reported)):
        print("  Unable to verify — NaN somewhere in inputs.")
        return

    ratio_h = abs(JW_reported) / abs(pred_h) if pred_h else float('nan')
    ratio_d = abs(JW_reported) / abs(pred_d) if pred_d else float('nan')
    print(f"  |JW_reported| / |pred_h| = {ratio_h:.3f}  (expect ~1 if JW is cm^3/h)")
    print(f"  |JW_reported| / |pred_d| = {ratio_d:.3f}  (expect ~1 if JW is cm^3/d)")
    if abs(np.log(max(ratio_d, 1e-30))) < abs(np.log(max(ratio_h, 1e-30))):
        print("\n  --> JW_ST is in cm^3/DAY. Diagnostic scripts that treat it as ml/h")
        print("      are under-reporting v by 24x. Re-evaluate the joint sweep with")
        print("      v_corrected = v_reported * 24.")
    else:
        print("\n  --> JW_ST is in ml/h (= cm^3/h). The legacy comment is correct;")
        print("      diagnostic scripts have been right all along; v shortfall is real.")

    p = REPO_ROOT / "dart/coupling/scripts/_pm_unitcheck.txt"
    if p.exists():
        p.unlink()


if __name__ == "__main__":
    main()
