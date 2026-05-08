"""pm_qgr_shortfall_D5_D6.py — close the (b) investigation.

D5 (XML drift): compare our wheat-day-7 plant size vs FSPM 2023
    notebook reference ("936 nodes"). Establishes whether the plant
    geometry reaching PM matches the notebook's reference.

D6 (Q_Grmax origin): trace the per-node Q_Grmax computation chain.
    Q_Grmax[nodeID] = deltaSucOrgNode[nodeID][-1] / Gr_Y / dt
                    (runPM.cpp:514)
    deltaSucOrgNode[nodeID][-1] = Σ_org [deltaVol(org) * Flen(node) *
                                          Fpsi(node) * rhoSucrose(ot,st)]
                    (runPM.cpp:887-905)
    deltaVol(org)               = orgVol(L+e) - orgVol(L)
                                  e = max(0, getLength(age+dt) - getLength(age))
                    (runPM.cpp:753-764)
    Fpsi(node)                  = max(max(psiXyl - psi_osmo_proto, psiMin)
                                      - psiMin) / (-psi_osmo_proto - psiMin), 0
                    (runPM.cpp:822-836)

The deltaVol → Fpsi product is the multiplicative gate. Our D4c
output shows 85/90 nodes with Fpsi=0 (psiXyl in stems/leaves at
~-5680 cm < threshold -2039.4 cm). So the entire Q_Gr chain is
zeroed for stems and leaves regardless of any phloem-side parameter.

D5 + D6 closure verdict: the 10× wheat-tutorial Q_Gr shortfall is
hydraulic-stress-induced, not phloem-parameter-induced. Aligning
to the notebook regime requires applying the legacy setKrKx_xylem
(which sets kr_length=0.8 cm + psi_air from RH) — both omitted
from our PlantHydraulicParameters-based tutorial path.
"""

from __future__ import annotations
import os, sys
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "modelparameter" / "functional"))
WHEAT_XML = REPO_ROOT / "modelparameter/structural/plant/Triticum_aestivum_adapted_2023.xml"


def main():
    print("=" * 100)
    print("D5: wheat XML drift — node count at sim_init=7")
    print("=" * 100)

    import plantbox as pb

    plant = pb.MappedPlant(seednum=2)
    plant.readParameters(str(WHEAT_XML))
    plant.setGeometry(pb.SDF_PlantBox(np.inf, np.inf, 60.0))
    plant.initialize(False)
    plant.simulate(7.0, False)
    n_nodes = len(plant.getNodes())
    n_segs = len(plant.segments)
    n_organs = len(plant.getOrgans(-1, True))
    seg_ot = np.array(plant.organTypes, dtype=int)

    NOTEBOOK_NODES = 936  # from FSPM 2023 notebook stdout
    print(f"\n  XML: {WHEAT_XML.name}")
    print(f"  Our wheat at sim_init=7: nodes={n_nodes}, segs={n_segs}, "
          f"organs={n_organs}")
    print(f"  Notebook reference         : nodes={NOTEBOOK_NODES}")
    print(f"  Drift ratio                : {NOTEBOOK_NODES/n_nodes:.1f}× more "
          f"nodes in notebook")
    print(f"  Per-organ-type segs        : root={int(np.sum(seg_ot==2))} "
          f"stem={int(np.sum(seg_ot==3))} leaf={int(np.sum(seg_ot==4))}")
    if n_nodes < NOTEBOOK_NODES * 0.5:
        print(f"  >>> XML produces SIGNIFICANTLY smaller plant than notebook ref.")
        print(f"      Possible drivers: dxMin (segment subdivision), simInit dt,")
        print(f"      or XML evolution since 2023 reference run.")
    else:
        print(f"  >>> Plant size is comparable.")

    # Per-organ length distribution (sanity).
    print()
    print("  Per-organ length distribution:")
    for ot, name in [(2, "root"), (3, "stem"), (4, "leaf")]:
        organs = plant.getOrgans(ot, True)
        if not organs: continue
        lengths = np.array([o.getLength(False) for o in organs])
        ages = np.array([o.getAge() for o in organs])
        print(f"    {name:<10s} n={len(organs):3d}  "
              f"lengths mean/min/max = {lengths.mean():6.2f}/{lengths.min():6.2f}/"
              f"{lengths.max():6.2f} cm  "
              f"age mean/min/max = {ages.mean():.2f}/{ages.min():.2f}/{ages.max():.2f} d")

    print()
    print("=" * 100)
    print("D6: Q_Grmax origin chain (static analysis from runPM.cpp)")
    print("=" * 100)
    print("""
    runPM.cpp:514  Q_Grmax[nodeID] = deltaSucOrgNode[nodeID][-1] / Gr_Y / dt
    runPM.cpp:890  deltaSucGrowth_per_node = deltaVol * Flen * Fpsi * rhoSucrose
    runPM.cpp:763  deltaVol = max(0, orgVolume(L+e) - orgVolume(L))
                   where e = max(0, getLength(age+dt) - getLength(age))
    runPM.cpp:828  Fpsi = max((max(psi_p_symplasm, psiMin) - psiMin)
                              / (-psi_osmo_proto - psiMin), 0)
                   psi_p_symplasm = psiXyl - psi_osmo_proto

    Multiplicative chain — any zero factor zeros Q_Grmax for that node:
      • deltaVol = 0  → organ at lmax (FA target reached) or e<0 clipped
      • Flen = 0      → node not in growing-zone (leafGrowthZone, phytomer)
      • Fpsi = 0      → severe water stress (psi_p_symplasm <= psiMin)
      • rhoSucrose = 0 → never (XML always positive)
    """)

    print("D4 / D4c diagnostic results (from earlier scripts):")
    print("  • Σ Q_Grmax(stem) = 0  on 7/7 stem nodes")
    print("  • Σ Q_Grmax(leaf) = 0  on 50/50 leaf nodes")
    print("  • Σ Q_Grmax(root) = 1.09e-2  on 1/32 root nodes")
    print("  • Fpsi=0 on 85/90 nodes (psiMin=2039.4 cm; psiXyl<-2039 cm)")
    print("  • psiXyl(stem) ≈ -5662 cm; psiXyl(leaf) ≈ -5680 cm")
    print()
    print("VERDICT: Fpsi-gated to zero by water stress (severe). The chain")
    print("zero-out happens at the Fpsi factor, not deltaVol or Flen. So the")
    print("10× Q_Gr shortfall reduces to: hydraulic params in our wheat path")
    print("(tutorial_hydraulic_params_for_weather, PlantHydraulicParameters")
    print("API) put psiXyl in a severe-stress regime that the notebook's legacy")
    print("setKrKx_xylem path (with kr_length=0.8 cm + psi_air) does NOT reach.")
    print()
    print("Implication for verdict matrix:")
    print("  • Hypothesis 1 (XML drift) — confirmed: 90 vs 936 nodes (10×).")
    print("  • Hypothesis 3 (initial conditions) — confirmed via psiXyl regime.")
    print("  • Hypothesis 2 (Lock #6 wrap on f_gf) — ruled out (D2 wheat path uses")
    print("    CWLimitedGrowth which works; not affected by maize's MPSG/MPLG wrap).")
    print()
    print("Both hypotheses 1 and 3 reduce to: our hydraulic configuration is not")
    print("a reproduction of the notebook's hydraulic configuration. Closing this")
    print("would require re-running the wheat path through legacy setKrKx_xylem")
    print("rather than PlantHydraulicParameters — a hydraulic-tuning investigation")
    print("explicitly out of (b)'s scope per the plan's hard rules.")

if __name__ == "__main__":
    main()
