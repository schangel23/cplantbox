"""pm_vmax_beta_joint_sweep.py — joint (Vmaxloading × beta_loading) sweep.

Tests the self-inhibition hypothesis on V3 maize against Babst 2022 Table A1.

Background
----------
Single-knob Vmaxloading sweep (pm_vmaxloading_verify.py) capped at 2/3 PASS:
v_basal stayed at ~0.07 m/hr even at Vmaxloading=0.50 (25× JSON default).
Diagnosis: the loading equation contains an explicit self-shutoff term

    Q_Fl[i] = Vmaxloading * len_leaf[i] * Cmeso/(Mloading+Cmeso)
              * exp(-CSTi * beta_loading)        (PiafMunch2.cpp:202)

With beta_loading=2.0 (current maize JSON), loading rate at C_ST=0.85
(Lohaus 2000 target) is exp(-1.7) = 18% of Vmax. The source phloem
literally cannot build up to literature concentrations because the
loading equation throttles itself before it gets there.

This sweep adds the beta_loading axis so we can disambiguate two
hypotheses for the v shortfall in the prior single-knob sweep:

  H1: Vmaxloading is just too low. v -> Babst window with high enough
      Vmaxloading regardless of beta. Single-knob sweep should have shown
      this; it didn't, but maybe the grid (max 0.50) was too narrow.

  H2: exp(-CSTi*beta_loading) self-inhibition prevents Munch buildup.
      Setting beta=0 unlocks the source-side concentration; v reaches
      Babst window at moderate Vmaxloading. Single-knob with beta=2.0
      could never reach this regime.

Grid choice
-----------
Vmaxloading [mmol cm-1 d-1]: {0.02, 0.20, 1.0, 5.0}
  - 0.02 = current JSON ("Q4: start low" placeholder)
  - 0.20 = best 2/3 in single-knob sweep
  - 1.0  = 50x JSON, midway to literature
  - 5.0  = approaching Giaquinta 1983 tissue-level anchor (200 mmol/cm2/s
           x ~1% loading-active vein area / typical leaf width ~4 cm
           => ~10 mmol/cm/d). Stop at 5.0 to bound runtime; if v reaches
           Babst at 5.0 we know it can.

beta_loading [-]: {0.0, 0.6, 2.0}
  - 0.0  = self-inhibition off (test for H2)
  - 0.6  = Lacointe & Giraud 2019/2023 wheat default
  - 2.0  = current maize JSON ("Q5: strong self-regulation")

12 cells total. Each cell ~5 min wall (V3 plant + 6h transient at solver=32
KLU). Total ~1 h.

Acceptance
----------
  - Find any cell with 3/3 PASS against Babst 2sigma window
  - If only beta=0 cells pass: structural finding, beta_loading is wrong
  - If only beta>0 cells pass: H2 falsified, calibration is the only gap
  - If no cells pass at any (V, beta): the loading equation form itself
    cannot reach Babst regime; structural rebase is the next move

NO JSON edit (overrides are in-memory).
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

VMAX_GRID = [0.02, 0.20, 1.0, 5.0]
BETA_GRID = [0.0, 0.6, 2.0]
N_SUBSTEPS = 6  # hourly

BABST_MET = {
    d: {"T_mean_C": 20.75, "T_min_C": 19.0, "T_max_C": 22.0,
        "PAR_MJ_m2_d": 30.0 * 0.219, "VPD_kPa": 1.0,
        "RH_pct": 60.0, "Wind_m_s": 0.5}
    for d in range(1, 31)
}

# Babst 2022 Table A1, x=0.5 case (matches XRF [K+] ~ 297 mol/m3)
BABST = {
    "delta_p_MPa":   (1.39, 0.44),
    "v_m_per_hr":    (0.95, 0.20),
    "c_st_mmol_cm3": (0.285, 0.090),
}
TOL_SIGMA = 2.0

# Window edges (mu +- 2*sigma)
DP_WIN  = (BABST["delta_p_MPa"][0]   - 2*BABST["delta_p_MPa"][1],
           BABST["delta_p_MPa"][0]   + 2*BABST["delta_p_MPa"][1])
V_WIN   = (BABST["v_m_per_hr"][0]    - 2*BABST["v_m_per_hr"][1],
           BABST["v_m_per_hr"][0]    + 2*BABST["v_m_per_hr"][1])
CST_WIN = (BABST["c_st_mmol_cm3"][0] - 2*BABST["c_st_mmol_cm3"][1],
           BABST["c_st_mmol_cm3"][0] + 2*BABST["c_st_mmol_cm3"][1])

T_K = 20.75 + 273.15
RT_MPa_per_mmol_cm3 = 8.314 * T_K * 1e-3   # ~2.443 MPa per mmol/cm3
CMH2O_TO_MPA = 9.80665e-5


def _suppress():
    o1 = os.dup(1); o2 = os.dup(2); dn = os.open(os.devnull, os.O_WRONLY)
    os.dup2(dn, 1); os.dup2(dn, 2)
    return o1, o2, dn


def _restore(o1, o2, dn):
    os.dup2(o1, 1); os.dup2(o2, 2)
    os.close(dn); os.close(o1); os.close(o2)


def run_one(vmax, beta, age=21, Tair_C=20.75):
    """Grow fresh V3 + run PhloemFlux with (Vmaxloading, beta_loading) override."""
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
    # In-memory overrides — must come after read_phloem_parameters.
    hm.Vmaxloading = float(vmax)
    hm.beta_loading = float(beta)

    rh = 0.6; psi_soil = -500.0
    p_s = np.linspace(psi_soil, psi_soil - 100, 100)
    es = hm.get_es(Tair_C); ea = es * rh
    par = 600.0 * 1e-6 * 86400 * 1e-4
    hm.solve(sim_time=age, rsx=p_s, cells=True, ea=ea, es=es,
             PAR=par, TairC=Tair_C, verbose=0)

    Tair_K = Tair_C + 273.15
    dt_days = 1.0 / 24.0
    Nt = len(plant.getNodes())
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
                                  str(REPO_ROOT / "dart/coupling/scripts/_pm_joint.txt"))
            if last_ret != 1:
                failures += 1
                break
    finally:
        _restore(*fdpair)

    C_ST    = np.array(hm.C_ST)
    psi_xyl = np.array(hm.psiXyl)
    JW_ST   = np.array(hm.JW_ST)
    Q_total = np.array(hm.Q_out)
    QRm   = float(np.sum(Q_total[Nt*2:Nt*3]))
    QGr   = float(np.sum(Q_total[Nt*4:Nt*5]))

    seg_ot = np.zeros(Nt, dtype=np.int32)
    for si in range(n_segs):
        nodeID = si + 1
        if nodeID < Nt:
            seg_ot[nodeID] = int(organ_types[si])
    leaf_mask = seg_ot == 4
    stem_mask = seg_ot == 3
    root_mask = seg_ot == 2

    # Source: leaf nodes above (CSTimin + 0.01) so we keep working when
    # CSTimin is the default 0.20 OR overridden. Fall back to argmax leaf.
    cstimin_eff = 0.20
    src_mask = leaf_mask & (C_ST > cstimin_eff + 0.01)
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
        # JW_ST units = ml/h per runPM.cpp:93
        flux_cm3_h = abs(float(JW_ST[basal_id]))
        v_m_per_hr = (flux_cm3_h / Across_stem) * 1e-2 if Across_stem > 0 else 0.0
    else:
        v_m_per_hr = 0.0

    dp_pass = DP_WIN[0]  <= delta_P    <= DP_WIN[1]
    v_pass  = V_WIN[0]   <= v_m_per_hr <= V_WIN[1]
    c_pass  = CST_WIN[0] <= c_st_src   <= CST_WIN[1]
    n_pass  = int(dp_pass) + int(v_pass) + int(c_pass)

    return dict(
        Vmaxloading=float(vmax),
        beta_loading=float(beta),
        ret=int(last_ret),
        failures=int(failures),
        delta_P_MPa=float(delta_P),
        v_m_per_hr=float(v_m_per_hr),
        c_st_src=float(c_st_src),
        c_st_sink=float(c_st_sink),
        c_st_max=float(np.max(C_ST)),
        c_st_leaf_mean=float(np.mean(C_ST[leaf_mask])) if leaf_mask.any() else 0.0,
        QRm_6h=QRm, QGr_6h=QGr,
        n_src_nodes=int(src_mask.sum()),
        dp_pass=bool(dp_pass), v_pass=bool(v_pass), c_pass=bool(c_pass),
        n_pass=int(n_pass),
    )


def main():
    print("=" * 100)
    print("Joint (Vmaxloading x beta_loading) sweep on V21 maize vs Babst 2022")
    print("=" * 100)
    print(f"  Babst 2sigma windows:")
    print(f"    delta_P  : [{DP_WIN[0]:.3f}, {DP_WIN[1]:.3f}] MPa")
    print(f"    v        : [{V_WIN[0]:.3f}, {V_WIN[1]:.3f}] m/hr")
    print(f"    C_ST_src : [{CST_WIN[0]:.3f}, {CST_WIN[1]:.3f}] mmol/cm3")
    print(f"  Vmaxloading grid: {VMAX_GRID}")
    print(f"  beta_loading grid: {BETA_GRID}")
    print(f"  Total cells: {len(VMAX_GRID) * len(BETA_GRID)}, ~5 min wall each")
    print()

    results = []
    for vmax in VMAX_GRID:
        for beta in BETA_GRID:
            print(f"--- Vmaxloading={vmax:.2f}  beta_loading={beta:.2f} ---")
            try:
                r = run_one(vmax, beta)
            except Exception as exc:
                print(f"  FAILED: {exc}")
                r = dict(Vmaxloading=vmax, beta_loading=beta,
                         ret=-99, failures=1, delta_P_MPa=float("nan"),
                         v_m_per_hr=float("nan"), c_st_src=float("nan"),
                         c_st_max=float("nan"), c_st_sink=float("nan"),
                         c_st_leaf_mean=float("nan"),
                         QRm_6h=0.0, QGr_6h=0.0, n_src_nodes=0,
                         dp_pass=False, v_pass=False, c_pass=False, n_pass=0)
            results.append(r)
            print(f"  ret={r['ret']}  dP={r['delta_P_MPa']:.3f} MPa  "
                  f"v={r['v_m_per_hr']:.4f} m/hr  "
                  f"C_ST src={r['c_st_src']:.3f}  max={r['c_st_max']:.3f}  "
                  f"PASS={r['n_pass']}/3")

    # Summary grid
    print("\n" + "=" * 100)
    print("SWEEP GRID — n_PASS / 3 per (Vmaxloading, beta_loading)")
    print("=" * 100)
    hdr = "Vmax \\ beta " + "".join(f"{b:>10.2f}" for b in BETA_GRID)
    print(hdr)
    print("-" * len(hdr))
    for vmax in VMAX_GRID:
        row = f"{vmax:>10.2f} "
        for beta in BETA_GRID:
            r = next(x for x in results
                     if x["Vmaxloading"] == vmax and x["beta_loading"] == beta)
            row += f"{r['n_pass']}/3      "[:10]
        print(row)

    # Detailed per-metric grids
    for label, key, fmt in [
        ("delta_P [MPa]",   "delta_P_MPa", "{:>10.3f}"),
        ("v [m/hr]",        "v_m_per_hr",  "{:>10.4f}"),
        ("C_ST_source",     "c_st_src",    "{:>10.3f}"),
        ("C_ST_max (any)",  "c_st_max",    "{:>10.3f}"),
    ]:
        print(f"\n{label}")
        print("-" * len(hdr))
        print(hdr)
        for vmax in VMAX_GRID:
            row = f"{vmax:>10.2f} "
            for beta in BETA_GRID:
                r = next(x for x in results
                         if x["Vmaxloading"] == vmax and x["beta_loading"] == beta)
                v = r[key]
                row += fmt.format(v) if np.isfinite(v) else "      NaN "
            print(row)

    # Verdict logic
    print("\n" + "=" * 100)
    print("VERDICT")
    print("=" * 100)
    full_passes = [r for r in results if r["n_pass"] == 3]
    beta_zero = [r for r in results if r["beta_loading"] == 0.0]
    beta_pos  = [r for r in results if r["beta_loading"] > 0.0]
    bz_pass = [r for r in beta_zero if r["n_pass"] == 3]
    bp_pass = [r for r in beta_pos  if r["n_pass"] == 3]

    if full_passes:
        # Find minimum-Vmax 3/3 to be conservative, plus highest-beta 3/3
        min_vmax = min(full_passes, key=lambda r: r["Vmaxloading"])
        max_beta = max(full_passes, key=lambda r: r["beta_loading"])
        print(f"  3/3 PASS cells found: {len(full_passes)} of {len(results)}")
        print(f"    Lowest Vmax 3/3: V={min_vmax['Vmaxloading']}, beta={min_vmax['beta_loading']}")
        print(f"    Highest beta 3/3: V={max_beta['Vmaxloading']}, beta={max_beta['beta_loading']}")
        if bz_pass and not bp_pass:
            print("  --> Only beta=0 cells pass. STRUCTURAL: exp(-CSTi*beta) is the bottleneck.")
            print("      Recommend: drop beta_loading toward 0 in maize JSON, refit Vmaxloading.")
        elif bp_pass and not bz_pass:
            print("  --> Only beta>0 cells pass. H2 falsified; pure calibration gap.")
            print("      Recommend: refit Vmaxloading at current beta_loading.")
        else:
            print("  --> Multiple regimes pass. Calibration is the only gap.")
    else:
        n_close = sum(1 for r in results if r["n_pass"] == 2)
        print(f"  No 3/3 PASS in any cell. {n_close} cells at 2/3.")
        if any(r["v_pass"] for r in results):
            print("  v reached Babst window in some cell -- the loading form CAN drive Munch flow.")
            print("  Failing arm is likely C_ST_src (over-loading) or delta_P. Narrow further.")
        else:
            print("  v never reached Babst window in ANY cell at any (V, beta).")
            print("  STRUCTURAL: loading equation cannot drive Munch flow on this V3 plant.")
            print("  Recommend: rebase to upstream auxin_master CPB_to_PM, or reach out to Giraud.")

    # JSON dump for downstream programmatic use
    print("\n" + "=" * 100)
    print("JSON SUMMARY")
    print("=" * 100)
    print(json.dumps({
        "grid": {"Vmaxloading": VMAX_GRID, "beta_loading": BETA_GRID},
        "babst_windows": {"delta_p_MPa": list(DP_WIN), "v_m_per_hr": list(V_WIN),
                          "c_st_mmol_cm3": list(CST_WIN)},
        "results": results,
    }, indent=2, sort_keys=True))

    p = REPO_ROOT / "dart/coupling/scripts/_pm_joint.txt"
    if p.exists():
        p.unlink()


if __name__ == "__main__":
    main()
