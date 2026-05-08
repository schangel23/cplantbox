"""pm_grow_plant_compat_D1.py — instrument CW_Gr lifecycle around the
substep-2 ``assertUsedCReserves`` failure on V3 maize via grow_plant.

Reproduces the failure, snapshotting:

  * ``param.f_gf`` class for each (ot, subType)        — hypothesis 3 check
  * ``param.f_gf.CW_Gr`` (size, sample entries)        — hypothesis 1 check
  * per-organ ``dl_backlog`` (size, max)               — hypothesis 2 check
  * per-organ ``isActive``                             — assertion gate
  * ``param.gf`` (XML int)                             — gf=3 vs other

Run:
  cpbenv/bin/python -u dart/coupling/scripts/pm_grow_plant_compat_D1.py

The assertion only triggers when (CW_Gr non-empty) ∧ (entry for orgID exists)
∧ (entry value ≥ 0) ∧ (organ.isActive()) ∧ (useCWGr=True).
Substep 1 fills CW_Gr → if plant.simulate's f_gf does not consume the entry
(i.e., set it to <0) the substep-2 assertion in initializePM_'s
waterLimitedGrowth fires.

What this script ESTABLISHES (vs the plan's three hypotheses):

  H1 (CW_Gr stale): print pre/post-substep CW_Gr sample sizes per organ
                    type to confirm whether plant.simulate consumed entries
  H2 (dl_backlog):  print sum of dl_backlog over organs after substep 1
  H3 (gf=3 wrap):   print f_gf class — if MPSG/MPLG (not CWLimited),
                    the consumption code path is missing
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "modelparameter" / "functional"))

from dart.coupling.config import (  # noqa: E402
    DEFAULT_XML, get_hydraulics_json, get_photosynthesis_json, get_phloem_json,
)
from dart.coupling.growth import grow_plant  # noqa: E402

BABST_MET = {
    d: {"T_mean_C": 20.75, "T_min_C": 19.0, "T_max_C": 22.0,
        "PAR_MJ_m2_d": 30.0 * 0.219, "VPD_kPa": 1.0,
        "RH_pct": 60.0, "Wind_m_s": 0.5}
    for d in range(1, 60)
}


def _suppress():
    o1 = os.dup(1); o2 = os.dup(2); dn = os.open(os.devnull, os.O_WRONLY)
    os.dup2(dn, 1); os.dup2(dn, 2)
    return o1, o2, dn


def _restore(o1, o2, dn):
    os.dup2(o1, 1); os.dup2(o2, 2)
    os.close(dn); os.close(o1); os.close(o2)


def gf_summary(plant):
    """Per (ot, subType) snapshot of f_gf class and CW_Gr state."""
    rows = []
    for ot in (2, 3, 4):
        for param in plant.getOrganRandomParameter(ot):
            if param is None:
                continue
            f_gf = getattr(param, "f_gf", None)
            cls = type(f_gf).__name__ if f_gf is not None else "None"
            cw = getattr(f_gf, "CW_Gr", None)
            cw_size = len(cw) if cw is not None else -1
            sample_pos = sample_neg = 0
            sample_keys = []
            if cw is not None and len(cw) > 0:
                items = list(cw.items())
                for k, v in items:
                    if v >= 0:
                        sample_pos += 1
                    else:
                        sample_neg += 1
                sample_keys = [k for k, _ in items[:3]]
            rows.append(dict(
                ot=ot, st=int(getattr(param, "subType", 0)),
                gf_xml=int(getattr(param, "gf", -1)),
                f_gf_cls=cls,
                cw_size=cw_size, cw_pos=sample_pos, cw_neg=sample_neg,
                cw_sample_keys=sample_keys,
            ))
    return rows


def organ_active_summary(plant):
    """Per organ-type: count active, count with non-empty dl_backlog."""
    rows = []
    for ot in (2, 3, 4):
        organs = plant.getOrgans(ot, True)
        n_total = len(organs)
        n_active = sum(1 for o in organs if o.isActive())
        max_dlb = 0.0
        n_dlb = 0
        for o in organs:
            dlb = getattr(o, "dl_backlog", 0.0)
            if dlb and dlb > 1e-12:
                n_dlb += 1
                max_dlb = max(max_dlb, float(dlb))
        rows.append(dict(ot=ot, n_total=n_total, n_active=n_active,
                         n_dlb=n_dlb, max_dlb=max_dlb))
    return rows


def cw_gr_audit_against_organs(plant, label):
    """Walk every organ; for each, evaluate the assertion predicate.
    Reports any organ that *would trip* the assertion."""
    tripping = []
    by_class = {}
    for ot in (2, 3, 4):
        organs = plant.getOrgans(ot, True)
        for o in organs:
            param = o.getOrganRandomParameter()
            f_gf = getattr(param, "f_gf", None)
            if f_gf is None:
                continue
            cw = f_gf.CW_Gr
            oid = o.getId()
            cls = type(f_gf).__name__
            by_class.setdefault(cls, dict(n=0, n_pos=0, n_neg=0,
                                           n_missing=0, n_active=0))
            by_class[cls]["n"] += 1
            if o.isActive():
                by_class[cls]["n_active"] += 1
            if len(cw) == 0:
                by_class[cls]["n_missing"] += 1
            elif oid not in cw:
                by_class[cls]["n_missing"] += 1
            else:
                v = cw[oid]
                if v >= 0:
                    by_class[cls]["n_pos"] += 1
                    if o.isActive():
                        # would trip the assertion
                        tripping.append(dict(
                            ot=ot, oid=oid, gf_cls=cls, val=float(v),
                            length=float(o.getLength(False)),
                            age=float(o.getAge()),
                        ))
                else:
                    by_class[cls]["n_neg"] += 1
    print(f"  [{label}] CW_Gr state by f_gf class:")
    for cls, d in by_class.items():
        print(f"    {cls:32s} n={d['n']:4d} active={d['n_active']:4d} "
              f"pos={d['n_pos']:4d} neg={d['n_neg']:4d} missing={d['n_missing']:4d}")
    if tripping:
        print(f"  [{label}] {len(tripping)} organ(s) would TRIP assertUsedCReserves:")
        for r in tripping[:10]:
            print(f"    ot={r['ot']} oid={r['oid']:5d} gf={r['gf_cls']} "
                  f"val={r['val']:.4e} len={r['length']:.3f} age={r['age']:.3f}")
        if len(tripping) > 10:
            print(f"    ... and {len(tripping) - 10} more")
    else:
        print(f"  [{label}] no organ would trip the assertion")
    return tripping, by_class


def main():
    print("=" * 100)
    print("D1: V3 maize CW_Gr lifecycle around substep-2 assertion")
    print("=" * 100)

    plant = grow_plant(
        xml_path=str(DEFAULT_XML), simulation_time=21,
        min_stem_nodes=10, min_leaf_nodes=4,
        enable_photosynthesis=True, seed=42,
        daily_met=BABST_MET, T_air_default=20.75,
    )

    print("\n--- After grow_plant (BEFORE any PM substep) ---")
    print("\nf_gf summary by (ot, subType):")
    for r in gf_summary(plant):
        print(f"  ot={r['ot']} st={r['st']} gf_xml={r['gf_xml']} "
              f"f_gf_cls={r['f_gf_cls']:30s} cw_size={r['cw_size']}")

    print("\nOrgan summary:")
    for r in organ_active_summary(plant):
        print(f"  ot={r['ot']}  n_total={r['n_total']:5d}  n_active={r['n_active']:5d}  "
              f"n_dlb={r['n_dlb']:4d}  max_dlb={r['max_dlb']:.4e}")

    cw_gr_audit_against_organs(plant, "post-grow_plant")

    # Build PM with default useCWGr=True.
    from functional.phloem_flux import PhloemFluxPython
    from plantbox.functional.PlantHydraulicParameters import PlantHydraulicParameters

    params_h = PlantHydraulicParameters()
    params_h.read_parameters(get_hydraulics_json())

    hm = PhloemFluxPython(plant, params_h, psiXylInit=-500, ciInit=350e-6)
    hm.read_photosynthesis_parameters(filename=get_photosynthesis_json())
    hm.read_phloem_parameters(filename=get_phloem_json())
    hm.atol = 1e-6
    hm.rtol = 1e-4
    hm.Vmaxloading = 0.20
    hm.beta_loading = 2.0
    hm.solver = 32
    print(f"\nuseCWGr (default): {bool(hm.useCWGr)}")

    p_s = np.linspace(-500, -700, 200)
    Tair_C = 20.75
    sim = 21.0
    dt = 1.0 / 24.0

    def solve_photo(sim_):
        es = hm.get_es(Tair_C); ea = es * 0.6
        par = 600.0 * 1e-6 * 86400 * 1e-4
        fdpair = _suppress()
        try:
            hm.solve(sim_time=sim_, rsx=p_s, cells=True, ea=ea, es=es,
                     PAR=par, TairC=Tair_C, verbose=0)
        finally:
            _restore(*fdpair)

    # Substep 1.
    print("\n--- Substep 1: photosynthesis + startPM ---")
    solve_photo(sim)
    fd = _suppress()
    try:
        ret = hm.startPM(sim, sim + dt, 1, Tair_C + 273.15, True,
                         str(REPO_ROOT / "dart/coupling/scripts/_pm_d1_loop.txt"))
    finally:
        _restore(*fd)
    print(f"  startPM returned: {ret}")

    print("\n--- AFTER substep 1 startPM (PM has filled CW_Gr) ---")
    cw_gr_audit_against_organs(plant, "after-startPM-sub1")

    # plant.simulate(dt) — should consume CW_Gr entries on CWLimitedGrowth f_gf.
    print(f"\n--- Calling plant.simulate(dt={dt:.4f}) ---")
    fd = _suppress()
    try:
        plant.simulate(dt, False)
    finally:
        _restore(*fd)

    print("\n--- AFTER plant.simulate (CWLimitedGrowth should have consumed entries) ---")
    tripping, _ = cw_gr_audit_against_organs(plant, "after-plant-simulate")

    if tripping:
        print()
        print("=" * 100)
        print(f"VERDICT: assertion WILL fire on next startPM ({len(tripping)} "
              f"tripping organs).")
        print("Hypothesis 1 confirmed: plant.simulate did not consume CW_Gr.")
        print(f"Tripping f_gf classes: "
              f"{sorted(set(r['gf_cls'] for r in tripping))}")
        print()
        print("This means f_gf classes other than CWLimitedGrowth do not")
        print("consume CW_Gr entries; the consumption logic only lives in")
        print("CWLimitedGrowth::getLength (sets entry to -1.0).")
        print()
        print("Fix path (D3): wrap MPSG/MPLG with CWLimitedGrowth(demand=existing)")
        print("via enable_cw_limited_growth(plant) before the PM loop. This is")
        print("the production diurnal.py path.")
    else:
        print()
        print("=" * 100)
        print("VERDICT: no organ would trip — but assertion is observed in pm_notebook_loop.")
        print("Need to investigate further (per-rank CW_Gr_per_n maybe?).")


if __name__ == "__main__":
    main()
