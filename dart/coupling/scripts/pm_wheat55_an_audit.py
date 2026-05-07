"""pm_wheat55_an_audit.py — audit wheat day-55 photosynthesis (the
An=2 mmol Suc/d anomaly that confounded the wheat-55 phloem diagnostic).

Background
----------
The 2026-05-04 wheat-55 diagnostic reported An_total = 24 mmol CO2/d
(~2 mmol Suc/d) for a 154 cm-leaf-length, 16 958-node mature wheat
plant. The plan flagged this as open-question §3:

  > Why is An so low on wheat-55? 24 mmol CO2/d for a 154 cm-leaf-length
  > plant seems undersized.

Before treating wheat-55 as evidence of a structural mature-plant
PiafMunch problem, this script disambiguates whether wheat-55 An is
actually broken at the photosynthesis layer — independent of phloem.

Three things can produce an unrealistically low whole-plant An:

  (A) Few or small leaves -- geometry/leaf-area issue from the wheat
      XML at age 55. Could be insufficient simulated mass / poor mature
      calibration (the wheat XML is calibrated by Lacointe-Giraud for
      day-7 tutorial validation, not mature plants).
  (B) Per-leaf An rate too low -- FvCB parameter or chlorophyll/Vcmax
      values from setPhotosynthesisParameters() are off, OR PAR isn't
      reaching the leaves (canopy / leaf orientation issue).
  (C) Solver issue / gs collapse -- stomatal conductance pinned closed
      because of psi_xyl coupling under saturating PAR.

Comparison anchors
------------------
  - Maize day-55 same harness:    An ~ 80 mmol Suc/d (964 mmol CO2/d)
                                  per pm_maize55_24h.py
  - Mature wheat literature:      An ~ 30-50 mmol Suc/d
      (Lawlor 2002 Trends Plant Sci 7:217 -- single-leaf An ~25 umol/m2/s
       at saturating PAR, midday, vegetative wheat)
  - Single mature wheat leaf:     ~10-30 cm2 area, 25 umol CO2/m2/s
                                  -> 0.6-1.8 mmol CO2/d per leaf
  - Mature wheat plant (~10 leaves * ~30 cm2): ~10 mmol CO2/d at min,
                                  more like 30-50 mmol CO2/d if you
                                  assume well-developed flag leaf

If wheat-55 reports per-leaf An ~ 25 umol/m2/s: the FvCB layer is fine
and the issue is geometry/leaf-area (case A).

If wheat-55 reports per-leaf An < 5 umol/m2/s: the FvCB layer is
broken (case B/C) and we need to track gs and Vcmax explicitly.

In either case, the wheat-55 result is NOT clean evidence for a
mature-plant phloem failure mode -- it's confounded with an upstream
photosynthesis issue.

Output
------
  - Plant counts (organs, leaves, stem segments, root organs)
  - Total leaf area [cm^2]
  - Per-leaf-segment An [umol CO2 m-2 s-1, mean / max / min / 25/50/75 pct]
  - Whole-plant An_total [mmol CO2/d, mmol Suc/d]
  - Stomatal conductance gs [mol m-2 s-1, mean / max / min]
  - Verdict: which case (A/B/C) the data supports

NO PiafMunch run, NO phloem solve. This is a PURE photosynthesis
audit at saturating PAR / well-watered.
"""

import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "modelparameter" / "functional"))

import plantbox as pb  # noqa: E402

WHEAT_XML = REPO_ROOT / "modelparameter/structural/plant/Triticum_aestivum_test_2021_shapeType2.xml"
WHEAT_HYDRAULICS_JSON = REPO_ROOT / "modelparameter/functional/plant_hydraulics/wheat_Giraud2023adapted.json"

# Photosynthesis test conditions
PAR_UMOL = 1500.0     # umol/m2/s -- saturating PAR (above maize-55's 1000)
TAIR_C   = 25.0       # vegetative wheat optimum
RH_FRAC  = 0.6
PSI_SOIL = -500.0     # well-watered

# Sanity windows
LITERATURE_PER_LEAF_AN_UMOL = (15.0, 35.0)   # umol CO2 m-2 s-1 at saturating PAR
LITERATURE_TOTAL_AN_MMOL    = (10.0, 80.0)   # mmol CO2/d for mature wheat

SUC_TO_CO2 = 12.0


def _suppress():
    o1 = os.dup(1); o2 = os.dup(2); dn = os.open(os.devnull, os.O_WRONLY)
    os.dup2(dn, 1); os.dup2(dn, 2)
    return o1, o2, dn


def _restore(o1, o2, dn):
    os.dup2(o1, 1); os.dup2(o2, 2)
    os.close(dn); os.close(o1); os.close(o2)


def build_wheat55(seednum=2, depth_cm=60.0, age_days=55):
    """Replicate the wheat-55 build that the original diagnostic used.

    NOTE: this is RAW MappedPlant + .simulate() per the plan reproducer,
    not grow_plant -- because grow_plant has no wheat tuning, and we
    are auditing the same wheat geometry the diagnostic saw.
    """
    plant = pb.MappedPlant(seednum=seednum)
    plant.readParameters(str(WHEAT_XML))
    plant.setGeometry(pb.SDF_PlantBox(np.inf, np.inf, depth_cm))
    plant.initialize(False)
    plant.simulate(age_days, False)
    return plant


def main():
    print("=" * 100)
    print("Wheat day-55 An audit (PURE photosynthesis, no PiafMunch)")
    print("=" * 100)
    print(f"  Wheat XML        : {WHEAT_XML.name}")
    print(f"  Hydraulics JSON  : {WHEAT_HYDRAULICS_JSON.name}")
    print(f"  PAR              : {PAR_UMOL:.0f} umol/m2/s (saturating)")
    print(f"  T_air            : {TAIR_C:.1f} C")
    print(f"  psi_soil         : {PSI_SOIL:.0f} cm (well-watered)")
    print(f"  Literature target: per-leaf An "
          f"{LITERATURE_PER_LEAF_AN_UMOL[0]:.0f}-{LITERATURE_PER_LEAF_AN_UMOL[1]:.0f} umol/m2/s; "
          f"total {LITERATURE_TOTAL_AN_MMOL[0]:.0f}-{LITERATURE_TOTAL_AN_MMOL[1]:.0f} mmol CO2/d")
    print()

    # Build wheat plant the same way the wheat-55 diagnostic did
    plant = build_wheat55()
    organ_types = np.array(plant.organTypes, dtype=np.int32)
    n_organs = len(plant.getOrgans())
    n_segs   = len(plant.getSegments())
    n_nodes  = plant.getNumberOfNodes()
    n_root   = int(np.sum(organ_types == 2))
    n_stem   = int(np.sum(organ_types == 3))
    n_leaf   = int(np.sum(organ_types == 4))

    print(f"Plant counts: organs={n_organs}  segs={n_segs}  nodes={n_nodes}")
    print(f"  by organ type: root_segs={n_root}  stem_segs={n_stem}  leaf_segs={n_leaf}")

    # Per-leaf-segment area (cm2). plant.segLength[i] * 2*radius[i] is the
    # crude rectangular approximation; for shape-2D leaves CPlantBox has
    # leafBladeSurface but it's per-organ, not per-segment.
    seg_lengths = np.array(plant.segLength, dtype=np.float64) if hasattr(plant, "segLength") else None
    if seg_lengths is None:
        # Fallback: compute from node positions
        nodes = plant.getNodes()
        node_arr = np.array([[n.x, n.y, n.z] for n in nodes])
        segs = plant.getSegments()
        seg_lengths = np.array([
            float(np.linalg.norm(node_arr[s.y] - node_arr[s.x])) for s in segs
        ])

    leaf_seg_idx = np.where(organ_types == 4)[0]
    leaf_seg_len = seg_lengths[leaf_seg_idx]
    print(f"  total leaf segment length: {leaf_seg_len.sum():.1f} cm "
          f"(was 154 cm in 2026-05-04 diagnostic)")

    # Try to get per-leaf-organ areas via getLeafArea() / getLeaf*
    organs = plant.getOrgans()
    leaf_organs = [o for o in organs if o.organType() == 4]
    print(f"  leaf organs: {len(leaf_organs)}")
    leaf_org_areas = []
    leaf_org_lens  = []
    leaf_org_lmax  = []
    for o in leaf_organs:
        try:
            lmax = float(o.getParameter("lmax"))
        except Exception:
            lmax = float("nan")
        try:
            blade_w = float(o.getParameter("Width_blade"))
        except Exception:
            blade_w = float("nan")
        leaf_org_lmax.append(lmax)
        L = float(o.getLength(False))
        leaf_org_lens.append(L)
        # rough rectangular area, refine if leaf has a 2D shape function
        leaf_org_areas.append(L * blade_w if np.isfinite(blade_w) else float("nan"))

    leaf_org_lens = np.array(leaf_org_lens)
    leaf_org_lmax = np.array(leaf_org_lmax)
    leaf_org_areas = np.array(leaf_org_areas)
    if np.isfinite(leaf_org_areas).any():
        total_leaf_area_cm2 = float(np.nansum(leaf_org_areas))
    else:
        total_leaf_area_cm2 = float("nan")
    print(f"  leaf lengths      mean/max/min: "
          f"{leaf_org_lens.mean():.1f} / {leaf_org_lens.max():.1f} / {leaf_org_lens.min():.1f} cm")
    if np.isfinite(leaf_org_lmax).any():
        print(f"  leaf lmax         mean/max/min: "
              f"{np.nanmean(leaf_org_lmax):.1f} / {np.nanmax(leaf_org_lmax):.1f} / "
              f"{np.nanmin(leaf_org_lmax):.1f} cm")
    print(f"  total leaf area (length*width est): {total_leaf_area_cm2:.1f} cm^2")

    # Build PhloemFluxPython for photosynthesis only -- mirror the wheat
    # tutorial harness from pm_vs_s5_wheat_tutorial.py.
    from plantbox.functional.phloem_flux import PhloemFluxPython
    from plantbox.functional.PlantHydraulicParameters import PlantHydraulicParameters
    from plant_photosynthesis.wheat_FcVB_Giraud2023adapted import (
        setPhotosynthesisParameters,
    )
    from plant_hydraulics.wheat_Giraud2023adapted import setKrKx_xylem

    params = PlantHydraulicParameters()
    # Pad wheat hydraulics to enough subtypes for mature wheat
    import json, tempfile
    with open(WHEAT_HYDRAULICS_JSON) as f:
        d = json.load(f)
    for key in ("kx_ages", "kx_values", "kr_ages", "kr_values"):
        for ot in list(d[key].keys()):
            arr = d[key][ot]
            while len(arr) < 30:
                arr.append(list(arr[-1]))
            d[key][ot] = arr
    padded = Path(tempfile.gettempdir()) / "wheat_hyd_padded_an_audit.json"
    with open(padded, "w") as f:
        json.dump(d, f)
    params.read_parameters(str(padded.with_suffix("")))

    # Provide a per-element kr/kx via setKrKx_xylem (Giraud's helper)
    es = 6.112 * np.exp((17.67 * TAIR_C) / (TAIR_C + 243.5))   # kPa
    ea = es * RH_FRAC
    weather_init = {
        "TairC": TAIR_C,
        "RH": RH_FRAC,
        "Qlight": PAR_UMOL,
        "cs": 350e-6,
        "es": es * 100.0,         # Pa
        "ea": ea * 100.0,         # Pa
        "p_mean": -150.0,
        "vg": [0.045, 0.43, 0.04, 1.6, 50.0],
    }

    hm = PhloemFluxPython(plant, params, psiXylInit=PSI_SOIL,
                          ciInit=weather_init["cs"] * 0.5)
    hm = setPhotosynthesisParameters(hm, weather_init)
    if hasattr(hm, "setKr"):
        hm = setKrKx_xylem(weather_init["TairC"], weather_init["RH"], hm)

    # solve_photosynthesis with verbose to surface FvCB diagnostics
    Tair_K = TAIR_C + 273.15
    p_s = np.linspace(PSI_SOIL, PSI_SOIL - 60, 60)
    hm.Qlight = [float(weather_init["Qlight"])]
    hm.cs     = [float(weather_init["cs"])]

    print("\nRunning solve_photosynthesis at PAR=1500, T=25, well-watered...")
    fdpair = _suppress()
    try:
        hm.solve_photosynthesis(
            sim_time=55.0,
            sxx=p_s,
            cells=True,
            ea=weather_init["ea"],
            es=weather_init["es"],
            TleafK=[Tair_K],
            verbose=False,
            doLog=False,
        )
    finally:
        _restore(*fdpair)

    An = np.array(hm.get_net_assimilation(), dtype=float)   # mol CO2 m-2 s-1
    # Documentation in pm_v3_babst_comparison.py:134 multiplies sum(An)*1e3
    # to get mmol CO2/d total. Same convention here.
    An_total_mmol_co2 = float(np.sum(An)) * 1e3
    An_total_mmol_suc = An_total_mmol_co2 / SUC_TO_CO2
    An_per_leaf_umol = An * 1e6 if len(An) > 0 else np.array([])

    if len(An) > 0:
        print(f"\nAn output array length: {len(An)} (matches leaf segs = {n_leaf}: "
              f"{'yes' if len(An) == n_leaf else 'NO MISMATCH'})")
        print(f"An values are: mean={An.mean()*1e6:.3f}, "
              f"max={An.max()*1e6:.3f}, "
              f"min={An.min()*1e6:.3f} umol/m2/s")
        print(f"Quantiles: 25%={np.percentile(An_per_leaf_umol, 25):.3f}, "
              f"50%={np.percentile(An_per_leaf_umol, 50):.3f}, "
              f"75%={np.percentile(An_per_leaf_umol, 75):.3f}")
        active = int((An_per_leaf_umol > 1.0).sum())
        zero   = int((An_per_leaf_umol < 0.1).sum())
        print(f"Active leaf segs (An>1 umol/m2/s) : {active}/{len(An)}")
        print(f"Inactive leaf segs (An<0.1)       : {zero}/{len(An)}")

    print(f"\nWhole-plant An total: {An_total_mmol_co2:.2f} mmol CO2/d")
    print(f"                    : {An_total_mmol_suc:.2f} mmol Suc/d")
    print(f"Maize-55 reference  : ~80 mmol Suc/d (~964 mmol CO2/d)")
    print(f"Wheat-55 prior      : 24 mmol CO2/d (the anomalous value)")

    # gs probe if available
    gs_arr = np.array([], dtype=float)
    for attr in ("get_gs", "gs", "stomatal_conductance"):
        if hasattr(hm, attr):
            try:
                v = getattr(hm, attr)
                raw = v() if callable(v) else v
                gs_arr = np.array(raw, dtype=float)
                if gs_arr.size > 0:
                    break
            except Exception:
                gs_arr = np.array([], dtype=float)
    if gs_arr.size > 0:
        print(f"\nStomatal conductance gs [mol m-2 s-1]: "
              f"mean={gs_arr.mean():.4f}  max={gs_arr.max():.4f}  "
              f"min={gs_arr.min():.4f}")
        if gs_arr.mean() < 0.05:
            print("  -> gs is collapsed; case (C) stomatal closure")
    else:
        print("\nStomatal conductance: probe attribute not found on hm")

    # Verdict
    print("\n" + "=" * 100)
    print("VERDICT")
    print("=" * 100)
    in_lit_total = LITERATURE_TOTAL_AN_MMOL[0] <= An_total_mmol_co2 <= LITERATURE_TOTAL_AN_MMOL[1]

    if len(An) > 0:
        median_per_leaf = float(np.percentile(An_per_leaf_umol, 50))
        in_lit_per_leaf = (LITERATURE_PER_LEAF_AN_UMOL[0] <= median_per_leaf
                           <= LITERATURE_PER_LEAF_AN_UMOL[1])
    else:
        median_per_leaf = 0.0
        in_lit_per_leaf = False

    if in_lit_total:
        print(f"  Total An_total in literature window "
              f"[{LITERATURE_TOTAL_AN_MMOL[0]:.0f}-{LITERATURE_TOTAL_AN_MMOL[1]:.0f}]: PASS")
        print("  --> Wheat-55 photosynthesis was a TRANSIENT issue, not structurally broken.")
        print("      The 2026-05-04 An=24 result was probably under non-saturating PAR")
        print("      or some transient solve state. Wheat-55 phloem result needs re-running")
        print("      under conditions matching this audit before drawing conclusions.")
    else:
        if in_lit_per_leaf:
            print(f"  Per-leaf An={median_per_leaf:.1f} umol/m2/s in literature window: PASS")
            print(f"  Total An={An_total_mmol_co2:.1f} mmol CO2/d below literature: FAIL")
            print("  --> CASE (A): per-leaf rate is fine; total is low because TOTAL LEAF AREA is low.")
            print(f"      Wheat XML at day 55 produces leaf area = {total_leaf_area_cm2:.0f} cm^2")
            print("      which is geometrically too small. The wheat-55 phloem result is")
            print("      CONFOUNDED by undersized geometry, NOT a structural mature-plant")
            print("      phloem failure. Wheat XML calibration at age 55 is the upstream issue.")
        else:
            print(f"  Per-leaf An={median_per_leaf:.1f} umol/m2/s below literature window: FAIL")
            print(f"  Total An={An_total_mmol_co2:.1f} mmol CO2/d below literature: FAIL")
            print("  --> CASE (B/C): the FvCB layer or stomatal layer is producing low")
            print("      per-leaf assimilation. The wheat photosynthesis helper")
            print("      (wheat_FcVB_Giraud2023adapted.setPhotosynthesisParameters) was")
            print("      calibrated for day-7 wheat tutorial regime, not mature.")
            print("      The wheat-55 phloem result is CONFOUNDED by FvCB calibration drift,")
            print("      NOT a structural mature-plant phloem failure.")

    print("\n" + "=" * 100)
    print("Conclusion for thesis framing")
    print("=" * 100)
    print("If the verdict is PASS: wheat-55 is an honest mature-plant test --")
    print("  the prior diagnostic chain stands.")
    print("If the verdict is CASE (A/B/C): wheat-55 was confounded by upstream")
    print("  photosynthesis calibration, and 'PiafMunch broken on mature plants'")
    print("  rests on maize-55 evidence alone. That is still defensible (Babst")
    print("  comparison + cuse-gate trap), but it's a per-species claim, not")
    print("  a cross-species architectural one.")


if __name__ == "__main__":
    main()
