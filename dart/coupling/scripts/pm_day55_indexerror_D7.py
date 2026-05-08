"""pm_day55_indexerror_D7.py — diagnose the day-55 maize PM IndexError
that surfaced after follow-up #1 (Stem::getEffectiveLn) unblocked the
prior runtime_error throw.

Behaviour:

  1. Build the day-55 maize plant via the same `pm_notebook_loop.build_maize`
     code path used in production diagnostics.
  2. Print every per-leaf-subtype array length in the maize phloem JSON.
  3. Print every leaf subType present on the plant + its mapped
     `st2newst` index.  The error fires when any new_st >= JSON array size.
  4. Run `hm.startPM` once and report the IndexError.

Root cause confirmed by this script: the maize XML defines 15 leaf
subtypes (2..16, mainstem positions 0..14) since the 2026-04-22 16-leaf
extension, but `phloem_parameters_maize2026.json` was last calibrated
when there were 11 leaf subtypes.  Across_st / kr_st / kx_st / Rmax_st
each have 11 entries for organType=4, indexed 0..10.  Day-55 leaves
with subTypes 13 and 14 get mapped to new_st 11 and 12, which trip
`std::vector::_M_range_check: __n (which is 11) >= this->size()
(which is 11)` inside `rhoSucrose_perType` / `Rmax_st_perType` etc.

Fix: pad the four per-leaf-subtype arrays in the JSON to 15 entries
to match the XML.  Tail values replicate the last calibrated index 10
value (biologically reasonable since the new entries are upper-rank
stubs that barely grow at day-55: subType 13 = 0.4 cm, subType 14 =
0.0 cm).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from dart.coupling.config import get_phloem_json  # noqa: E402
from dart.coupling.scripts.pm_notebook_loop import build_maize  # noqa: E402


def main() -> int:
    # 1) Plant
    plant, _cfg = build_maize(55)
    print()
    print("=" * 80)
    print("Day-55 maize PM IndexError diagnosis (follow-up #3)")
    print("=" * 80)

    # 2) JSON array sizes
    pj = json.loads(Path(get_phloem_json() + ".json").read_text())
    pertypes = pj.get("PerType", {})
    print("\nJSON `PerType` array sizes (root, stem, leaf):")
    for k in ("Across_st", "kr_st", "kx_st", "Rmax_st"):
        v = pertypes.get(k, {}).get("value", [])
        sizes = [len(x) for x in v]
        print(f"  {k:<10} sizes = {sizes}")

    # 3) Leaf subTypes on the plant + new_st mapping
    leaf_subtypes_on_plant = set()
    for org in plant.getOrgans(-1, True):
        if org.organType() == 4:
            leaf_subtypes_on_plant.add(int(org.getParameter("subType")))
    print(f"\nLeaf subTypes present on plant: {sorted(leaf_subtypes_on_plant)}")

    # st2newst is built in C++ via mapSubTypes: iterates organParam[ot]
    # from index 1 onward, assigning sequential new_st 0..N-1 to
    # non-NULL entries.  Reproduce here from the Python-visible
    # organRandomParameter list.
    leaf_params = plant.getOrganRandomParameter(4)
    new_st_map = {}
    new_st = 0
    # organParam is 1-indexed for actual subTypes; index 0 is the
    # default/dummy entry that mapSubTypes skips.
    for p in leaf_params:
        if p is None:
            continue
        if int(p.subType) == 0:
            continue
        new_st_map[int(p.subType)] = new_st
        new_st += 1
    print("\nLeaf subType -> new_st mapping (mapSubTypes order):")
    leaf_arr_size = len(pertypes.get("Across_st", {}).get("value", [[], [], []])[2])
    for st in sorted(new_st_map.keys()):
        in_range = "OK" if new_st_map[st] < leaf_arr_size else "OUT-OF-RANGE"
        print(f"  subType={st:2d} -> new_st={new_st_map[st]:2d}  ({in_range})")

    print(f"\nLeaf array size in JSON: {leaf_arr_size}")
    print(f"Max new_st on plant     : {max(new_st_map.values())}")
    if max(new_st_map.values()) >= leaf_arr_size:
        n_oob = sum(1 for v in new_st_map.values() if v >= leaf_arr_size)
        print(f"=> {n_oob} leaf subType(s) exceed JSON array size: IndexError will fire.")

    # 4) Trigger startPM to confirm
    print("\nReproducing IndexError via hm.startPM ...")
    import numpy as np
    sys.path.insert(0, str(REPO_ROOT / "modelparameter" / "functional"))
    from functional.phloem_flux import PhloemFluxPython
    from plantbox.functional.PlantHydraulicParameters import PlantHydraulicParameters
    from dart.coupling.config import get_hydraulics_json, get_photosynthesis_json
    from dart.coupling.scripts.pm_notebook_loop import configure_maize

    params_h = PlantHydraulicParameters()
    params_h.read_parameters(get_hydraulics_json())
    hm = PhloemFluxPython(plant, params_h, psiXylInit=-500, ciInit=350e-6)
    hm.read_photosynthesis_parameters(filename=get_photosynthesis_json())
    configure_maize(hm, Vmaxloading=0.20, beta_loading=2.0, solver=32)

    p_s = np.linspace(-500, -700, 200)
    T = 20.75
    es = hm.get_es(T); ea = es * 0.6
    par = 600.0 * 1e-6 * 86400 * 1e-4
    sim = 55.0
    hm.solve(sim_time=sim, rsx=p_s, cells=True, ea=ea, es=es,
             PAR=par, TairC=T, verbose=0)
    try:
        hm.startPM(sim, sim + 1.0 / 24.0, 1, T + 273.15, True,
                   "/tmp/pm_d7_repro.txt")
        print("startPM completed without error -- fix is live.")
        return 0
    except IndexError as e:
        print(f"IndexError reproduced: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
