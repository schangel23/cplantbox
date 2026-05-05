"""Across_st runtime multiplier sweep for V3 maize Babst comparison.

This is intentionally a runtime-only diagnostic:
  - no tracked JSON edits
  - Vmaxloading is forced to 0.20 in the temporary phloem JSON
  - kx_st is read from the existing maize2026 JSON without re-multiplication
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
    DEFAULT_XML,
    get_hydraulics_json,
    get_photosynthesis_json,
    get_phloem_json,
)


ACROSS_GRID = [0.25, 0.5, 1.0, 2.0, 4.0]
VMAXLOADING = 0.20
N_SUBSTEPS = 6

DP_WINDOW = (0.51, 2.27)
V_WINDOW = (0.55, 1.35)
CST_WINDOW = (0.105, 0.465)

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
RT_MPA_PER_MMOL_CM3 = 8.314 * T_K * 1e-3
CMH2O_TO_MPA = 9.80665e-5


def grow_plant(xml_path, simulation_time, enable_photosynthesis=False, seed=None,
               daily_met=None, T_air_default=25.0, **_unused):
    """Minimal local grow helper to avoid optional G3/OpenAlea dependencies."""
    print(f"=== Growing Plant ===")
    print(f"  XML: {xml_path}")
    print(f"  Simulation time: {simulation_time} days")
    if seed is not None:
        print(f"  Seed: {seed}")
    if enable_photosynthesis:
        print("  Photosynthesis: ENABLED (soil grid active)")

    plant = pb.MappedPlant()
    plant.readParameters(str(xml_path))
    if seed is not None:
        plant.setSeed(seed)
    if enable_photosynthesis:
        depth = 100
        plant.setGeometry(pb.SDF_PlantContainer(np.inf, np.inf, depth, True))

        def _picker(_x, _y, z):
            return max(min(int(np.floor(-z)), depth - 1), -1)

        plant.setSoilGrid(_picker)
    plant.initialize()

    dt = 1.0
    total_simulated = 0.0
    while total_simulated < simulation_time:
        step = min(dt, simulation_time - total_simulated)
        sim_day = int(total_simulated) + 1
        day_met = daily_met.get(sim_day) if daily_met is not None else None
        T_air = float(day_met["T_mean_C"]) if day_met else float(T_air_default)
        if hasattr(plant, "setAirTemperature"):
            plant.setAirTemperature(T_air)
        plant.simulate(step, verbose=(total_simulated == 0))
        total_simulated += step

    organs = plant.getOrgans()
    n_stems = sum(1 for o in organs if o.organType() == pb.OrganTypes.stem)
    n_leaves = sum(1 for o in organs if o.organType() == pb.OrganTypes.leaf)
    n_roots = sum(1 for o in organs if o.organType() == pb.OrganTypes.root)
    print(f"\n  Stems: {n_stems}, Leaves: {n_leaves}, Roots: {n_roots}")
    print(f"  Total nodes: {len(plant.getNodes())}")
    return plant


def _suppress():
    o1 = os.dup(1)
    o2 = os.dup(2)
    dn = os.open(os.devnull, os.O_WRONLY)
    os.dup2(dn, 1)
    os.dup2(dn, 2)
    return o1, o2, dn


def _restore(o1, o2, dn):
    os.dup2(o1, 1)
    os.dup2(o2, 2)
    os.close(dn)
    os.close(o1)
    os.close(o2)


def _scale_nested(values, multiplier):
    if isinstance(values, list):
        return [_scale_nested(v, multiplier) for v in values]
    return float(values) * float(multiplier)


def _runtime_phloem_path(source_stem, across_multiplier):
    """Return a temporary phloem JSON stem after dict-level runtime mutation."""
    src = Path(f"{source_stem}.json")
    with src.open() as f:
        data = json.load(f)
    data["SieveTube"]["Vmaxloading"]["value"] = float(VMAXLOADING)
    data["PerType"]["Across_st"]["value"] = _scale_nested(
        data["PerType"]["Across_st"]["value"], across_multiplier
    )

    tmp = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".json",
        prefix=f"pm_acrosst_{across_multiplier:g}_",
        delete=False,
    )
    with tmp:
        json.dump(data, tmp)
    return str(Path(tmp.name).with_suffix(""))


def _remove_runtime_phloem(path_stem):
    path = Path(f"{path_stem}.json")
    if path.exists():
        path.unlink()


def _pad_wheat_hydraulics(target_n=30):
    src = REPO_ROOT / "modelparameter/functional/plant_hydraulics/wheat_Giraud2023adapted.json"
    with src.open() as f:
        data = json.load(f)
    for key in ("kx_ages", "kx_values", "kr_ages", "kr_values"):
        for ot in list(data[key].keys()):
            arr = data[key][ot]
            while len(arr) < target_n:
                arr.append(list(arr[-1]))
            data[key][ot] = arr
    out = Path(tempfile.gettempdir()) / "wheat_hyd_padded_acrosst.json"
    with out.open("w") as f:
        json.dump(data, f)
    return str(out.with_suffix(""))


def _setup_hm(plant, phloem_stem, hyd_stem=None):
    from plantbox.functional.PlantHydraulicParameters import PlantHydraulicParameters
    from plantbox.functional.phloem_flux import PhloemFluxPython

    params = PlantHydraulicParameters()
    params.read_parameters(hyd_stem or get_hydraulics_json())
    hm = PhloemFluxPython(plant, params, psiXylInit=-500, ciInit=350e-6)
    hm.read_photosynthesis_parameters(filename=get_photosynthesis_json())
    hm.read_phloem_parameters(filename=phloem_stem)
    hm.atol = 1e-6
    hm.rtol = 1e-4
    hm.solver = 32
    hm.useCWGr = False
    return hm


def _map_node_organs(plant):
    organ_types = np.array(plant.organTypes, dtype=np.int32)
    sub_types = np.array(plant.subTypes, dtype=np.int32)
    n_segs = len(plant.getSegments())
    n_nodes = len(plant.getNodes())
    seg_node = np.zeros(n_nodes, dtype=np.int32)
    seg_st = np.zeros(n_nodes, dtype=np.int32)
    for si in range(n_segs):
        node_id = si + 1
        if node_id < n_nodes:
            seg_node[node_id] = int(organ_types[si])
            seg_st[node_id] = int(sub_types[si])
    return seg_node, seg_st


def run_v3_babst(across_multiplier):
    age = 21
    Tair_C = 20.75
    phloem_stem = _runtime_phloem_path(get_phloem_json(), across_multiplier)
    try:
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
        hm = _setup_hm(plant, phloem_stem)

        rh = 0.6
        psi_soil = -500.0
        p_s = np.linspace(psi_soil, psi_soil - 100, 100)
        es = hm.get_es(Tair_C)
        ea = es * rh
        par = 600.0 * 1e-6 * 86400 * 1e-4
        hm.solve(sim_time=age, rsx=p_s, cells=True, ea=ea, es=es,
                 PAR=par, TairC=Tair_C, verbose=0)

        Tair_K = Tair_C + 273.15
        dt_days = 1.0 / 24.0
        ret = -1
        failures = 0
        fdpair = _suppress()
        try:
            for step in range(1, N_SUBSTEPS + 1):
                if step > 1:
                    hm.withInitVal = False
                t_start = float(age) + (step - 1) * dt_days
                ret = hm.startPM(
                    t_start,
                    t_start + dt_days,
                    1,
                    Tair_K,
                    False,
                    str(REPO_ROOT / "dart/coupling/scripts/_pm_acrosst.txt"),
                )
                if ret != 1:
                    failures += 1
                    break
        finally:
            _restore(*fdpair)

        C_ST = np.array(hm.C_ST)
        psi_xyl = np.array(hm.psiXyl)
        JW_ST = np.array(hm.JW_ST)
        seg_node, _ = _map_node_organs(plant)
        nodes = plant.getNodes()
        node_z = np.array([n.z for n in nodes], dtype=np.float64)

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

        if root_mask.any():
            deep_idx = np.argsort(node_z[root_mask])[:max(5, root_mask.sum() // 50)]
            root_node_ids = np.where(root_mask)[0][deep_idx]
            c_st_sink = float(np.mean(C_ST[root_node_ids]))
            psi_sink = float(np.mean(psi_xyl[root_node_ids]))
        else:
            c_st_sink = float(np.min(C_ST))
            psi_sink = float(np.min(psi_xyl))

        p_src = psi_src * CMH2O_TO_MPA + RT_MPA_PER_MMOL_CM3 * c_st_src
        p_sink = psi_sink * CMH2O_TO_MPA + RT_MPA_PER_MMOL_CM3 * c_st_sink
        delta_p = p_src - p_sink

        if stem_mask.any():
            stem_ids = np.where(stem_mask)[0]
            basal_id = stem_ids[np.argmin(np.abs(node_z[stem_ids]))]
            across_stem = hm.Across_st[1][0]
            flux_cm3_h = abs(float(JW_ST[basal_id]))
            v_m_per_hr = (flux_cm3_h / across_stem) * 1e-2 if across_stem > 0 else 0.0
        else:
            v_m_per_hr = 0.0

        passes = int(DP_WINDOW[0] <= delta_p <= DP_WINDOW[1])
        passes += int(V_WINDOW[0] <= v_m_per_hr <= V_WINDOW[1])
        passes += int(CST_WINDOW[0] <= c_st_src <= CST_WINDOW[1])
        return {
            "multiplier": float(across_multiplier),
            "ret": int(ret),
            "failures": int(failures),
            "delta_P": float(delta_p),
            "v": float(v_m_per_hr),
            "C_ST_src": float(c_st_src),
            "C_ST_max": float(np.max(C_ST)),
            "passes": passes,
            "n_nodes": len(nodes),
        }
    finally:
        _remove_runtime_phloem(phloem_stem)


def gate_c_wheat73(across_multiplier):
    print(f"\n[GATE C] Wheat day-7.3, Vmaxloading={VMAXLOADING:.2f}")
    wheat_xml = REPO_ROOT / "modelparameter/structural/plant/Triticum_aestivum_test_2021_shapeType2.xml"
    wheat_phloem_src = str(REPO_ROOT / "modelparameter/functional/plant_sucrose/phloem_parameters2025")
    wheat_hyd = _pad_wheat_hydraulics(target_n=30)
    phloem_stem = _runtime_phloem_path(wheat_phloem_src, across_multiplier)
    try:
        plant = pb.MappedPlant(seednum=2)
        plant.readParameters(str(wheat_xml))
        plant.setGeometry(pb.SDF_PlantBox(np.inf, np.inf, 60.0))
        plant.initialize(False)
        plant.simulate(7.3, False)

        hm = _setup_hm(plant, phloem_stem, hyd_stem=wheat_hyd)
        Tair_C = 25.0
        rh = 0.7
        psi_soil = -500.0
        p_s = np.linspace(psi_soil, psi_soil - 60, 60)
        es = hm.get_es(Tair_C)
        ea = es * rh
        par = 1000.0 * 1e-6 * 86400 * 1e-4
        hm.solve(sim_time=7.3, rsx=p_s, cells=True, ea=ea, es=es,
                 PAR=par, TairC=Tair_C, verbose=0)
        Nt = len(plant.getNodes())
        fdpair = _suppress()
        try:
            ret = hm.startPM(
                7.3,
                7.3 + 1.0 / 24.0,
                1,
                Tair_C + 273.15,
                False,
                str(REPO_ROOT / "dart/coupling/scripts/_pm_acrosst_gate_c.txt"),
            )
        finally:
            _restore(*fdpair)

        Q = np.array(hm.Q_out)
        C_ST = np.array(hm.C_ST)
        cmax = float(np.max(C_ST))
        accepted = int(ret) == 1 and np.isfinite(cmax) and cmax < 5.0
        return {
            "ret": int(ret),
            "Q_Rm": float(np.sum(Q[Nt * 2:Nt * 3])),
            "Q_Gr": float(np.sum(Q[Nt * 4:Nt * 5])),
            "C_ST_mean": float(np.mean(C_ST)),
            "C_ST_max": cmax,
            "accepted": bool(accepted),
        }
    finally:
        _remove_runtime_phloem(phloem_stem)


def gate_b_maize55(across_multiplier, n_substeps=24):
    print(f"\n[GATE B] Maize day-55 24h, Vmaxloading={VMAXLOADING:.2f}, "
          f"Across_st x{across_multiplier:g}")
    phloem_stem = _runtime_phloem_path(get_phloem_json(), across_multiplier)
    try:
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
        hm = _setup_hm(plant, phloem_stem)
        rh = 0.7
        psi_soil = -500.0
        p_s = np.linspace(psi_soil, psi_soil - 100, 100)
        es = hm.get_es(Tair_C)
        ea = es * rh
        par = 1000.0 * 1e-6 * 86400 * 1e-4
        hm.solve(sim_time=age, rsx=p_s, cells=True, ea=ea, es=es,
                 PAR=par, TairC=Tair_C, verbose=0)
        An_total = float(np.sum(np.array(hm.get_net_assimilation()))) * 1e3

        Nt = len(plant.getNodes())
        Tair_K = Tair_C + 273.15
        dt_days = 1.0 / 24.0
        QRm = QGr = cmean = cmax = 0.0
        ret = -1
        failures = 0
        for step in range(1, n_substeps + 1):
            if step > 1:
                hm.withInitVal = False
            t_start = float(age) + (step - 1) * dt_days
            t0 = time.time()
            fdpair = _suppress()
            try:
                try:
                    ret = hm.startPM(
                        t_start,
                        t_start + dt_days,
                        1,
                        Tair_K,
                        False,
                        str(REPO_ROOT / "dart/coupling/scripts/_pm_acrosst_gate_b.txt"),
                    )
                    error = ""
                except Exception as exc:
                    ret = -1
                    error = str(exc)
            finally:
                _restore(*fdpair)
            Q = np.array(hm.Q_out)
            C_ST = np.array(hm.C_ST)
            QRm = float(np.sum(Q[Nt * 2:Nt * 3]))
            QGr = float(np.sum(Q[Nt * 4:Nt * 5]))
            cmean = float(np.mean(C_ST)) if C_ST.size else float("nan")
            cmax = float(np.max(C_ST)) if C_ST.size else float("inf")
            if step in (1, 2, 6, 12, 18, 24):
                print(f"  h={step:02d} ret={ret} C_ST_mean={cmean:.4f} "
                      f"C_ST_max={cmax:.4f} wall={time.time() - t0:.2f}s")
            if ret != 1 or not np.isfinite(cmax) or cmax > 1e6:
                if error:
                    print(f"  startPM error: {error}")
                failures += 1
                break

        Q_Rmmax = float(np.sum(np.array(hm.Q_Rmmax)))
        accepted = (
            failures == 0
            and int(ret) == 1
            and abs(Q_Rmmax - 16.0) <= 5.0
            and cmax < 5.0
        )
        return {
            "ret": int(ret),
            "QRm_24h": QRm,
            "QGr_24h": QGr,
            "Q_Rmmax": Q_Rmmax,
            "C_ST_mean": cmean,
            "C_ST_max": cmax,
            "solver_failures": failures,
            "An": An_total,
            "accepted": bool(accepted),
        }
    finally:
        _remove_runtime_phloem(phloem_stem)


def _passes_text(r):
    return f"{r['passes']}/3"


def main():
    print("=" * 100)
    print("Across_st multiplier sweep on V3 maize vs Babst 2022 Table A1")
    print("=" * 100)
    print("Runtime dict mutation: Vmaxloading=0.20 and PerType.Across_st scaled "
          "in a temporary phloem JSON.")
    print(f"Babst windows: ΔP {DP_WINDOW[0]:.2f}-{DP_WINDOW[1]:.2f} MPa; "
          f"v {V_WINDOW[0]:.2f}-{V_WINDOW[1]:.2f} m/hr; "
          f"C_ST_source {CST_WINDOW[0]:.3f}-{CST_WINDOW[1]:.3f} mmol/cm3")

    results = []
    for multiplier in ACROSS_GRID:
        print(f"\n--- Across_st x{multiplier:g} ---")
        r = run_v3_babst(multiplier)
        results.append(r)
        print(f"  ret={r['ret']} nodes={r['n_nodes']} ΔP={r['delta_P']:.3f} "
              f"v={r['v']:.4f} C_ST_src={r['C_ST_src']:.4f} "
              f"C_ST_max={r['C_ST_max']:.4f} {_passes_text(r)}")

    print("\nSWEEP TABLE")
    print(f"{'Across_st multiplier':>22} {'DeltaP':>9} {'v':>9} "
          f"{'C_ST_src':>10} {'passes':>8}")
    print("-" * 64)
    for r in results:
        print(f"{r['multiplier']:>22.2g} {r['delta_P']:>9.3f} "
              f"{r['v']:>9.4f} {r['C_ST_src']:>10.4f} {_passes_text(r):>8}")

    passing = [r for r in results if r["passes"] == 3]
    gate_c = gate_b = None
    recommended = None
    if passing:
        recommended = max(passing, key=lambda r: r["multiplier"])
        print(f"\nRecommended Across_st multiplier: {recommended['multiplier']:g}")
        gate_c = gate_c_wheat73(recommended["multiplier"])
        gate_b = gate_b_maize55(recommended["multiplier"])
        print("\nGATE C TABLE")
        print("ret Q_Rm Q_Gr C_ST_mean C_ST_max accepted")
        print(f"{gate_c['ret']} {gate_c['Q_Rm']:.6f} {gate_c['Q_Gr']:.6f} "
              f"{gate_c['C_ST_mean']:.6f} {gate_c['C_ST_max']:.6f} "
              f"{gate_c['accepted']}")
        print("\nGATE B TABLE")
        print("ret Q_Rmmax QRm_24h QGr_24h C_ST_mean C_ST_max accepted")
        print(f"{gate_b['ret']} {gate_b['Q_Rmmax']:.3f} {gate_b['QRm_24h']:.3f} "
              f"{gate_b['QGr_24h']:.3f} {gate_b['C_ST_mean']:.4f} "
              f"{gate_b['C_ST_max']:.4f} {gate_b['accepted']}")
        verdict = "PASS" if gate_c["accepted"] and gate_b["accepted"] else "FAIL"
        print(f"\nGATE VERDICT: {verdict}")
    else:
        best = max(results, key=lambda r: r["passes"])
        print(f"\nNo 3/3 found. Best partial: x{best['multiplier']:g} "
              f"with {_passes_text(best)}.")

    print("\nJSON SUMMARY")
    print(json.dumps({
        "Vmaxloading": VMAXLOADING,
        "sweep": results,
        "recommended_multiplier": recommended["multiplier"] if recommended else None,
        "gate_c": gate_c,
        "gate_b": gate_b,
    }, indent=2, sort_keys=True))

    for name in ("_pm_acrosst.txt", "_pm_acrosst_gate_b.txt", "_pm_acrosst_gate_c.txt"):
        path = REPO_ROOT / "dart/coupling/scripts" / name
        if path.exists():
            path.unlink()


if __name__ == "__main__":
    main()
