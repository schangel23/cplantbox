"""pm_maize55_diagnostic.py — runtime cross-check of the Step -0.5 audit.

Re-runs the prior-session diagnostic to verify whether the C++ runtime
Q_Rmmax matches the Python audit's prediction.

If audit and runtime AGREE around 16 mmol Suc/d on day-55 maize, the
prior session's "385 mmol Suc/d" was a misreading and the cuse-gate trap
is dominated by ~9.5x JSON-Krm1 inflation (parameter regime) — not a
hidden C++ unit factor.

If runtime confirms 385 mmol Suc/d, there is a hidden ~24x multiplier
in the C++ pipeline that the pure-Python audit (which replicates only
runPM.cpp:506-509) misses. That would re-open the unit-chain bug branch.

This script regenerates what the (lost) /tmp/_pm_maize55_diagnostic.py
did. NO C++ edit. NO JSON edit. Reads Q_Rmmax / Q_Grmax directly off
the Python-bound PhloemFluxPython after a short startPM call.
"""

import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from dart.coupling.config import (  # noqa: E402
    DEFAULT_XML, get_hydraulics_json, get_photosynthesis_json, get_phloem_json,
)
from dart.coupling.growth import grow_plant  # noqa: E402

CLASS_NAME = {2: "root", 3: "stem", 4: "leaf"}


def _suppress_cpp_stdout():
    """Return (old_stdout_fd, devnull_fd) and redirect FDs 1+2 to /dev/null."""
    old_stdout_fd = os.dup(1)
    old_stderr_fd = os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 1)
    os.dup2(devnull, 2)
    return old_stdout_fd, old_stderr_fd, devnull


def _restore_cpp_stdout(old_stdout_fd, old_stderr_fd, devnull):
    os.dup2(old_stdout_fd, 1)
    os.dup2(old_stderr_fd, 2)
    os.close(devnull)
    os.close(old_stdout_fd)
    os.close(old_stderr_fd)


def main():
    print("=" * 78)
    print("PiafMunch maize day-55 runtime diagnostic")
    print("(cross-check vs Python audit at pm_unit_audit.py)")
    print("=" * 78)

    Tair_C = 25.0
    age    = 55

    # 1) Grow plant
    plant = grow_plant(
        xml_path=str(DEFAULT_XML),
        simulation_time=age,
        min_stem_nodes=100,
        min_leaf_nodes=40,
        enable_photosynthesis=True,
        seed=42,
    )
    organ_types = np.array(plant.organTypes, dtype=np.int32)
    n_root = int(np.sum(organ_types == 2))
    n_stem = int(np.sum(organ_types == 3))
    n_leaf = int(np.sum(organ_types == 4))
    n_segs = len(plant.getSegments())
    n_nodes = len(plant.getNodes())
    print(f"Plant: segs root={n_root} stem={n_stem} leaf={n_leaf} "
          f"(total {n_segs} segs / {n_nodes} nodes)")

    # 2) Build PhloemFluxPython and load JSON params
    from plantbox.functional.phloem_flux import PhloemFluxPython
    from plantbox.functional.PlantHydraulicParameters import PlantHydraulicParameters

    params = PlantHydraulicParameters()
    params.read_parameters(get_hydraulics_json())

    hm = PhloemFluxPython(plant, params, psiXylInit=-500, ciInit=350e-6)
    hm.read_photosynthesis_parameters(filename=get_photosynthesis_json())
    hm.read_phloem_parameters(filename=get_phloem_json())
    hm.atol = 1e-6
    hm.rtol = 1e-4
    hm.solver = 32
    hm.useCWGr = False  # avoid CWGr Gr_Y assert during diagnostic-only run

    # 3) Photosynthesis (gives An / sets up xylem state)
    rh = 0.7
    psi_soil = -500.0
    p_s = np.linspace(psi_soil, psi_soil - 100, 100)
    es = hm.get_es(Tair_C); ea = es * rh
    par = 1000.0 * 1e-6 * 86400 * 1e-4
    hm.solve(sim_time=age, rsx=p_s, cells=True, ea=ea, es=es,
             PAR=par, TairC=Tair_C, verbose=0)
    An = np.array(hm.get_net_assimilation())
    An_total_mmol = float(np.sum(An)) * 1e3
    print(f"An_total = {An_total_mmol:.1f} mmol CO2/d  ({len(An)} leaf segs)")

    # 4) Run startPM for 1 hour to populate hm.Q_Rmmax / Q_Grmax internals
    Tair_K = Tair_C + 273.15
    dt_days = 1.0 / 24.0
    t_start = float(age)
    t_end   = t_start + dt_days

    t0 = time.time()
    fdpair = _suppress_cpp_stdout()
    try:
        ret = hm.startPM(t_start, t_end, 1, Tair_K, False,
                         str(REPO_ROOT / "dart/coupling/scripts/_pm_diag.txt"))
    finally:
        _restore_cpp_stdout(*fdpair)
    elapsed = time.time() - t0
    print(f"startPM returned {ret} in {elapsed:.1f}s")
    if ret != 1:
        print("startPM did not converge cleanly; per-node arrays may still be "
              "valid since initializePM_ ran first. Continuing.")

    # 5) Read per-node arrays from the Python binding
    Q_Rmmax = np.array(hm.Q_Rmmax)   # base maintenance rate, mmol Suc/d
    Q_Grmax = np.array(hm.Q_Grmax)   # base growth rate,      mmol Suc/d
    print(f"\nArrays exposed: Q_Rmmax len={len(Q_Rmmax)}, Q_Grmax len={len(Q_Grmax)}")

    # PiafMunch indexes nodes 1..N (Fortran-style); node 0 is unused (seed/dummy).
    # Map segment -> child_node = seg_idx+1 (matches runPM.cpp:498 nodeID = k).
    per_class = defaultdict(lambda: dict(qrm=0.0, qgr=0.0, n_nodes=0))
    for si in range(n_segs):
        nodeID = si + 1
        if nodeID >= len(Q_Rmmax):
            continue
        cls = CLASS_NAME.get(int(organ_types[si]), f"ot{int(organ_types[si])}")
        per_class[cls]["qrm"] += float(Q_Rmmax[nodeID])
        per_class[cls]["qgr"] += float(Q_Grmax[nodeID])
        per_class[cls]["n_nodes"] += 1

    print("\nC++ runtime per-class breakdown:")
    print(f"{'class':<6} {'n_nodes':>9} {'sum Q_Rmmax':>14} {'sum Q_Grmax':>14}")
    print(f"{'':<6} {'':>9} {'mmol Suc/d':>14} {'mmol Suc/d':>14}")
    print("-" * 60)
    qrm_total = 0.0; qgr_total = 0.0
    for cls in ("root", "stem", "leaf"):
        r = per_class[cls]
        print(f"{cls:<6} {r['n_nodes']:>9} {r['qrm']:>14.3f} {r['qgr']:>14.3f}")
        qrm_total += r["qrm"]; qgr_total += r["qgr"]
    print("-" * 60)
    print(f"{'TOTAL':<6} {'':>9} {qrm_total:>14.3f} {qgr_total:>14.3f}")

    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    print(f"  C++ runtime Q_Rmmax_total: {qrm_total:>10.2f} mmol Suc/d")
    print(f"  Python audit Q_Rmmax_total: ~16 mmol Suc/d (at this seed/age)")
    print(f"  Prior-session claim:        385 mmol Suc/d")
    if qrm_total < 25:
        print("  -> Audit and runtime AGREE around ~16 mmol Suc/d.")
        print("  -> Prior 385 figure was a MISREADING (likely cumulative or post-T-scaled).")
        print("  -> Excess vs Amthor (~1.7 mmol/d for 34 g DM) is ~9.5x = JSON Krm1 inflation.")
        print("  -> Cause: parameter regime, not unit-chain bug.")
    elif qrm_total > 200:
        print("  -> Runtime CONFIRMS 385-class magnitude. Hidden ~24x multiplier in C++.")
        print("  -> Re-open unit-chain bug branch (cause 1).")
    else:
        print(f"  -> Intermediate ({qrm_total:.1f}). Investigate.")

    # Cleanup .txt
    p = REPO_ROOT / "dart/coupling/scripts/_pm_diag.txt"
    if p.exists():
        p.unlink()


if __name__ == "__main__":
    main()
