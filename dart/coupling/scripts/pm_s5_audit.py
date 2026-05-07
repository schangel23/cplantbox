"""pm_s5_audit.py — audit S5 (QuasiSteadyPhloem) the same way we
audited PiafMunch. Prior sessions assumed S5 was 'proven' and used it
as a reference for PiafMunch — but never verified S5's outputs are
physical. User flagged 2026-05-07 that BOTH are not functioning.

Test: V21 maize at JSON-default phloem params, saturating PAR (Babst
chamber proxy), well-watered. Report:
  - An_total          [mmol CO2/d, mmol Suc/d]
  - total_loading     (S5's Q_Fl equivalent, mmol CO2/d but mmol Suc
                       interpretation if /SUC_TO_CO2)
  - Rm split by class
  - Rg split by class
  - root exudation
  - starch surplus
  - C_ST mean/min/max
  - mass-balance closure (carbon_balance_error)
  - implied loading efficiency Q_Fl/An
  - implied storage fraction

Compare to PiafMunch on same plant + JSON params:
  - PiafMunch (24h, k_S_M restored): load_eff 20%, storage 80%, starch 40%
  - S5 should give a comparable picture if both backends are physical.

Outputs the comparison side-by-side so we can see if S5 is self-consistent
(not whether it agrees with PiafMunch — they are different model paradigms).
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


def main():
    age = 21
    Tair_C = 20.75
    print("=" * 100)
    print("S5 (QuasiSteadyPhloem) audit on V21 maize")
    print("=" * 100)

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
    n_root = int(np.sum(organ_types == 2))
    n_stem = int(np.sum(organ_types == 3))
    n_leaf = int(np.sum(organ_types == 4))
    print(f"V{age}: organs={len(plant.getOrgans())}  segs={len(plant.getSegments())} "
          f"(root={n_root} stem={n_stem} leaf={n_leaf})")

    # Get An via PhloemFluxPython (same harness as PiafMunch audit)
    from plantbox.functional.phloem_flux import PhloemFluxPython
    from plantbox.functional.PlantHydraulicParameters import PlantHydraulicParameters

    params_h = PlantHydraulicParameters()
    params_h.read_parameters(get_hydraulics_json())
    hm = PhloemFluxPython(plant, params_h, psiXylInit=-500, ciInit=350e-6)
    hm.read_photosynthesis_parameters(filename=get_photosynthesis_json())

    rh = 0.6; psi_soil = -500.0
    p_s = np.linspace(psi_soil, psi_soil - 200, 200)
    es = hm.get_es(Tair_C); ea = es * rh
    par = 600.0 * 1e-6 * 86400 * 1e-4
    fdpair = _suppress()
    try:
        hm.solve(sim_time=age, rsx=p_s, cells=True, ea=ea, es=es,
                 PAR=par, TairC=Tair_C, verbose=0)
    finally:
        _restore(*fdpair)

    An_per_leaf_seg = np.array(hm.get_net_assimilation())   # mol CO2/d per leaf seg
    An_total_mmol_co2 = float(np.sum(An_per_leaf_seg)) * 1e3
    SUC_PER_CO2 = 1.0 / 12.0
    An_total_mmol_suc = An_total_mmol_co2 * SUC_PER_CO2
    print(f"An: {An_total_mmol_co2:.2f} mmol CO2/d  (= {An_total_mmol_suc:.2f} mmol Suc/d)")
    print(f"   = saturating PAR baseline at chamber-proxy met (T=20.75 C, DLI=30)")
    print()

    # Run S5
    from dart.coupling.carbon.phloem_steady import (
        QuasiSteadyPhloem, load_phloem_params,
    )
    s5_params = load_phloem_params("maize")
    solver = QuasiSteadyPhloem(plant, params=s5_params, sim_day=age)

    print("Running S5 ...")
    fdpair = _suppress()
    try:
        out = solver.solve(An_per_leaf_seg, Tair_C=Tair_C, sim_day=age)
    finally:
        _restore(*fdpair)

    print()
    if not out.get("converged", False):
        print(f"!! S5 DID NOT CONVERGE: n_iter={out.get('n_iterations')} "
              f"max_delta={out.get('max_delta')}")
    else:
        print(f"S5 converged in {out.get('n_iterations')} iters, "
              f"max_delta={out.get('max_delta'):.3e}, "
              f"carbon_balance_error={out.get('carbon_balance_error'):.4f}")
    print()

    # Convert S5's CO2 outputs to Suc for comparison
    Rm_total_co2 = out["Rm_total_mmol"]   # mmol CO2/d (per line 1040)
    Rg_total_co2 = out["Rg_total_mmol"]
    Rm_total_suc = Rm_total_co2 * SUC_PER_CO2
    Rg_total_suc = Rg_total_co2 * SUC_PER_CO2
    Exud_suc = float(np.sum(out["root_exud_mmol_d"]))    # already in Suc per line 1052
    starch_suc = out.get("starch_surplus_mmol", 0.0) * SUC_PER_CO2
    storage_suc = out.get("stem_storage_mmol", 0.0) * SUC_PER_CO2
    growth_co2 = out.get("growth_mmol_d", 0.0)            # CO2
    loading_total_co2 = out.get("total_loading_mmol", 0.0)
    loading_total_suc = loading_total_co2 * SUC_PER_CO2

    print("=" * 100)
    print("S5 outputs (carbon balance)")
    print("=" * 100)
    print(f"  An_total                    : {An_total_mmol_suc:>10.2f} mmol Suc/d  "
          f"({An_total_mmol_co2:>8.1f} mmol CO2/d)")
    print(f"  total_loading (Q_Fl-equiv)  : {loading_total_suc:>10.2f} mmol Suc/d  "
          f"({loading_total_co2:>8.1f} mmol CO2/d)")
    print(f"  Rm_total                    : {Rm_total_suc:>10.2f} mmol Suc/d  "
          f"({Rm_total_co2:>8.1f} mmol CO2/d)")
    print(f"    Rm_leaf                   : {out['Rm_leaf'] * SUC_PER_CO2:>10.2f} mmol Suc/d  "
          f"({out['Rm_leaf']:>8.1f} mmol CO2/d)")
    print(f"    Rm_stem                   : {out['Rm_stem'] * SUC_PER_CO2:>10.2f} mmol Suc/d  "
          f"({out['Rm_stem']:>8.1f} mmol CO2/d)")
    print(f"    Rm_root                   : {out['Rm_root'] * SUC_PER_CO2:>10.2f} mmol Suc/d  "
          f"({out['Rm_root']:>8.1f} mmol CO2/d)")
    print(f"  Rg_total                    : {Rg_total_suc:>10.2f} mmol Suc/d  "
          f"({Rg_total_co2:>8.1f} mmol CO2/d)")
    print(f"  Exudation (already Suc)     : {Exud_suc:>10.2f} mmol Suc/d")
    print(f"  Starch surplus              : {starch_suc:>10.2f} mmol Suc/d  "
          f"({out.get('starch_surplus_mmol', 0.0):>8.1f} mmol CO2/d)")
    print(f"  C_ST mean/min/max           : "
          f"{out['C_ST_mean']:.3f}/{out['C_ST_min']:.3f}/{out['C_ST_max']:.3f} mmol/cm^3")
    print()

    total_sinks_suc = Rm_total_suc + Rg_total_suc + Exud_suc + starch_suc
    print(f"  total sinks (Rm+Rg+Exud+Starch)  = {total_sinks_suc:.2f} mmol Suc/d")
    print(f"  An_total                          = {An_total_mmol_suc:.2f} mmol Suc/d")
    print(f"  ratio total_sinks / An_total      = {total_sinks_suc / An_total_mmol_suc:.3f}")
    print(f"  carbon_balance_error reported     = {out.get('carbon_balance_error', 0):.4f}")
    print()

    # Sanity checks
    print("=" * 100)
    print("PHYSICAL SANITY CHECKS")
    print("=" * 100)
    issues = []
    if Rm_total_suc + Rg_total_suc > An_total_mmol_suc * 1.2:
        issues.append(f"Rm+Rg = {Rm_total_suc + Rg_total_suc:.1f} > 120% of An "
                      f"(impossible at saturating PAR)")
    if Rm_total_suc / max(An_total_mmol_suc, 1e-9) > 0.6:
        issues.append(f"Rm fraction {100*Rm_total_suc/An_total_mmol_suc:.0f}% > 60% "
                      f"(Stitt 2012 says ~30% in maize source)")
    if Rg_total_suc / max(An_total_mmol_suc, 1e-9) > 0.6:
        issues.append(f"Rg fraction {100*Rg_total_suc/An_total_mmol_suc:.0f}% > 60%")
    cstm = out['C_ST_mean']
    if cstm < 0.05 or cstm > 2.0:
        issues.append(f"C_ST_mean = {cstm:.3f} outside physical range [0.05, 2.0]")
    if out.get('carbon_balance_error', 0) > 0.05:
        issues.append(f"carbon_balance_error {out['carbon_balance_error']:.3f} > 5%")
    if abs(total_sinks_suc - An_total_mmol_suc) / max(An_total_mmol_suc, 1e-9) > 0.10:
        issues.append(f"Σsinks vs An off by "
                      f"{100*(total_sinks_suc - An_total_mmol_suc) / An_total_mmol_suc:.1f}%")

    if not issues:
        print("  All checks PASS — S5 outputs are physically defensible at this regime.")
    else:
        print("  ISSUES FOUND:")
        for issue in issues:
            print(f"    - {issue}")

    print()
    print("=" * 100)
    print("COMPARISON TO PIAFMUNCH (24h, JSON-default best cell, k_S_M=20 restored)")
    print("=" * 100)
    pm_load_eff = 0.20
    pm_storage  = 0.80
    pm_starch   = 0.376
    pm_Rm_total_suc = 3.41    # 24h cumulative
    pm_Rg_total_suc = 0.14
    pm_Exud_suc     = 1.30
    print(f"{'metric':<32} {'S5':>14} {'PiafMunch (24h)':>18}")
    print("-" * 70)
    print(f"{'load_eff (Q_Fl/An)':<32} {loading_total_suc / max(An_total_mmol_suc, 1e-9):>14.3f} {pm_load_eff:>18.3f}")
    print(f"{'storage frac':<32} {(starch_suc + storage_suc) / max(An_total_mmol_suc, 1e-9):>14.3f} {pm_storage:>18.3f}")
    print(f"{'Rm_total mmol Suc/d':<32} {Rm_total_suc:>14.3f} {pm_Rm_total_suc:>18.3f}")
    print(f"{'Rg_total mmol Suc/d':<32} {Rg_total_suc:>14.3f} {pm_Rg_total_suc:>18.3f}")
    print(f"{'Exud mmol Suc/d':<32} {Exud_suc:>14.3f} {pm_Exud_suc:>18.3f}")
    print(f"{'C_ST_mean':<32} {out['C_ST_mean']:>14.3f} {0.482:>18.3f}")

    out_json = REPO_ROOT / "dart/coupling/scripts/_pm_s5_audit.json"
    with open(out_json, 'w') as f:
        json.dump({
            "An_total_mmol_suc": An_total_mmol_suc,
            "An_total_mmol_co2": An_total_mmol_co2,
            "s5": {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                   for k, v in out.items() if k != "root_resp_profile_mmol_d"},
            "issues": issues,
        }, f, indent=2, default=str)
    print(f"\nJSON: {out_json}")


if __name__ == "__main__":
    main()
