"""pm_unit_audit.py — Step -0.5: Trace Krm1 -> Q_Rmmax unit chain on day-55 maize.

Replicates the C++ unit chain from runPM.cpp:506-509 in pure Python:
    StructSucrose[node] = rhoSucrose_f(st,ot) * vol_Seg[node]   # mmol Suc
    Q_Rmmax[node]       = krm1_f(st,ot) * StructSucrose[node]   # mmol Suc / d

For Q_Rmmax to come out in mmol Suc/d, krm1 MUST be interpreted as [1/d]
(despite C++ verbose log labeling it "-"). This script confirms that math
and localizes any unit/Rho_s/StructSucrose anomaly without running the PiafMunch
ODE integrator.

Decision tree at end of output:
- Sigma StructSucrose >> ~50 g sucrose-equivalent  -> volume / Rho_s inflation (cause 2)
- Implied DM >> 50 g                                -> upstream geometry inflation
- Sigma Q_Rmmax matches expected per-class totals from literature anchors
  but is still ~150x Amthor at total -> JSON Krm1 is the literal cause (parameter regime)
- Math checks out and Sigma Q_Rmmax = 385 mmol/d    -> architectural overrun (cause 3)

Anchors:
- Amthor 2000 m_R: ~600 mmol CO2 / kg DM / d at 20 C -> 50 mmol Suc / kg / d -> 2.5 mmol Suc/d at 50 g DM
- WOFOST Krm1 (g CH2O / g DW / d, equivalent to [1/d]): leaf 0.030, stem 0.015, root 0.010-0.015

NO JSON edit. NO PiafMunch integration. Pure Python audit.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

import plantbox as pb  # noqa: E402
from dart.coupling.growth.grow import grow_plant  # noqa: E402

XML_PATH  = REPO_ROOT / "dart/coupling/data/maize_calibrated.xml"
JSON_PATH = REPO_ROOT / "dart/coupling/data/phloem_parameters_maize2026.json"

# Sucrose molar mass — for cross-checking implied DM
M_SUC_g_per_mmol = 0.342  # g / mmol  (sucrose MW 342 g/mol)

# Literature anchors
AMTHOR_TARGET_PER_50G  = 2.5    # mmol Suc / d, midpoint Amthor 2000 (range 1.4-4.2)
TYPICAL_DM_DENSITY_GCM3 = 0.20  # g DM / cm3, vegetative maize parenchyma + leaves
WOFOST_KRM1 = {"root": 0.012, "stem": 0.015, "leaf": 0.030}  # g CH2O/g DW/d == [1/d]


def lookup_pertype(array, organ_type, sub_type):
    """Replica of phloem_steady._pertype_lookup (PerType st2newst remapping)."""
    ot_idx = organ_type - 2
    if ot_idx < 0 or ot_idx >= len(array):
        return 0.0
    organ_array = array[ot_idx]
    if not organ_array:
        return 0.0
    if len(organ_array) == 1:
        return organ_array[0]
    if organ_type == 2:
        idx = sub_type - 1
    elif organ_type == 3:
        idx = sub_type - 1
    elif organ_type == 4:
        idx = sub_type - 2
    else:
        return 0.0
    idx = max(0, min(idx, len(organ_array) - 1))
    return organ_array[idx]


CLASS_NAME = {2: "root", 3: "stem", 4: "leaf"}


def main():
    print("=" * 78)
    print("PiafMunch unit-chain audit (Step -0.5)")
    print("=" * 78)

    # 1) Load JSON params
    with open(JSON_PATH) as f:
        d = json.load(f)
    pt = d["PerType"]
    krm1   = pt["Krm1"]["value"]    # [[root],[stem],[leaf]] each len 1 in current JSON
    krm2   = pt["Krm2"]["value"]
    rho_s  = pt["Rho_s"]["value"]
    print("\nJSON PerType (raw):")
    print(f"  Krm1   : {krm1}")
    print(f"  Krm2   : {krm2}")
    print(f"  Rho_s  : {rho_s}    [units in JSON: mmol Suc cm-3]")

    # 2) Grow the plant via canonical entrypoint
    print("\nGrowing maize via grow_plant(simulation_time=55, seed=42, FA on)...")
    # FA kinetics are wired in by grow_plant by default (see memory:
    # feedback_fa_always_on / FA always enabled, 2026-04-24).
    plant = grow_plant(
        str(XML_PATH),
        simulation_time=55,
        seed=42,
        enable_photosynthesis=True,
    )

    n_organs = len(plant.getOrgans())
    n_nodes  = len(plant.getNodes())
    n_segs   = len(plant.getSegments())
    print(f"Plant: {n_organs} organs, {n_nodes} nodes, {n_segs} segments")

    # 3) Walk segments — use python-side fields that mirror what
    #    plant->segVol does in C++:
    #      stems/roots: pi * r^2 * L
    #      leaves (shape_2D): leafBladeSurface * a   (a = radius/thickness param)
    ot_arr   = np.array(plant.organTypes, dtype=np.int32)
    st_arr   = np.array(plant.subTypes,   dtype=np.int32)
    seg_len  = np.array(plant.segLength(), dtype=np.float64)
    radii    = np.array(plant.radii,       dtype=np.float64)
    blade_surf = np.array(plant.leafBladeSurface, dtype=np.float64)

    # Cylindrical default for stems/roots and any "petiole" segments
    seg_vol = np.pi * radii ** 2 * seg_len
    # Leaf override: shape_2D uses bladeArea * a (radius == half-thickness)
    leaf_blade_mask = (ot_arr == 4) & (blade_surf > 0)
    seg_vol[leaf_blade_mask] = blade_surf[leaf_blade_mask] * radii[leaf_blade_mask]
    # Note: this is a faithful Python replica of C++ plant->segVol for
    # shape_2D leaves with withPetiole=True. Sheath/petiole segments retain
    # the cylindrical default.

    # 4) Per-class aggregation
    rows = defaultdict(lambda: dict(
        n_segs=0, sum_vol=0.0, sum_struct_suc=0.0, sum_qrmmax=0.0
    ))

    # Cache krm1/rho_s per (ot, st) to mirror runPM.cpp:487 / :506
    cache_k = {}
    cache_r = {}

    for si in range(n_segs):
        ot = int(ot_arr[si]); st = int(st_arr[si])
        key = (ot, st)
        if key not in cache_k:
            cache_k[key] = lookup_pertype(krm1,  ot, st)
            cache_r[key] = lookup_pertype(rho_s, ot, st)
        k1 = cache_k[key]
        rs = cache_r[key]

        v  = seg_vol[si]
        ss = rs * v             # mmol Suc per segment (StructSucrose)
        qr = k1 * ss            # mmol Suc / d         (Q_Rmmax)

        cls = CLASS_NAME.get(ot, f"ot{ot}")
        r = rows[cls]
        r["n_segs"] += 1
        r["sum_vol"] += v
        r["sum_struct_suc"] += ss
        r["sum_qrmmax"] += qr

    # 5) Print per-class table
    print("\nPer-class breakdown (Python replica of runPM.cpp:506-509):")
    print(f"{'class':<6} {'n_seg':>6} {'Krm1[1/d]':>11} {'Rho_s':>9} "
          f"{'Sigma Vol':>11} {'Sigma StSuc':>13} {'Sigma Q_Rmmax':>15}")
    print(f"{'':<6} {'':>6} {'':>11} {'mmol/cm3':>9} {'cm3':>11} "
          f"{'mmol Suc':>13} {'mmol Suc/d':>15}")
    print("-" * 78)
    grand_qrmmax = 0.0
    grand_struct = 0.0
    grand_vol    = 0.0
    for cls in ("root", "stem", "leaf"):
        r = rows[cls]
        # Pull the single per-class value (since current JSON has 1 entry per ot)
        ot = {"root": 2, "stem": 3, "leaf": 4}[cls]
        k1_repr = lookup_pertype(krm1,  ot, 1 if ot != 4 else 2)
        rs_repr = lookup_pertype(rho_s, ot, 1 if ot != 4 else 2)
        print(f"{cls:<6} {r['n_segs']:>6} {k1_repr:>11.4f} {rs_repr:>9.3f} "
              f"{r['sum_vol']:>11.3f} {r['sum_struct_suc']:>13.3f} "
              f"{r['sum_qrmmax']:>15.3f}")
        grand_qrmmax += r["sum_qrmmax"]
        grand_struct += r["sum_struct_suc"]
        grand_vol    += r["sum_vol"]
    print("-" * 78)
    print(f"{'TOTAL':<6} {sum(r['n_segs'] for r in rows.values()):>6} "
          f"{'':>11} {'':>9} "
          f"{grand_vol:>11.3f} {grand_struct:>13.3f} {grand_qrmmax:>15.3f}")

    # 6) Cross-checks
    implied_dm_g = grand_vol * TYPICAL_DM_DENSITY_GCM3
    implied_struct_dm_g = grand_struct * M_SUC_g_per_mmol  # treating StSuc as Suc-eq mass
    print("\nCross-checks (literature anchors):")
    print(f"  Plant total volume                 : {grand_vol:>10.2f} cm3")
    print(f"  Implied DM at 0.20 g/cm3           : {implied_dm_g:>10.2f} g  "
          f"(target ~50 g for V10 maize)")
    print(f"  Sigma StructSucrose * 0.342        : {implied_struct_dm_g:>10.2f} g "
          f"(StructSucrose expressed as Suc mass)")
    print(f"  Amthor 2000 anchor for 50 g plant  : {AMTHOR_TARGET_PER_50G:>10.2f} mmol Suc/d "
          f"(range 1.4-4.2)")
    print(f"  Amthor anchor scaled to implied DM : "
          f"{AMTHOR_TARGET_PER_50G * implied_dm_g / 50.0:>10.2f} mmol Suc/d")
    print(f"  Audit Q_Rmmax (this run)           : {grand_qrmmax:>10.2f} mmol Suc/d")
    if grand_qrmmax > 0 and implied_dm_g > 0:
        ratio_to_amthor = grand_qrmmax / (AMTHOR_TARGET_PER_50G * implied_dm_g / 50.0)
        print(f"  Ratio audit / Amthor anchor        : {ratio_to_amthor:>10.1f}x")

    # 7) WOFOST counter-factual: what would per-class sums look like with WOFOST Krm1?
    print("\nCounter-factual: SAME volumes/Rho_s but WOFOST Krm1 (literature anchor):")
    cf_total = 0.0
    print(f"{'class':<6} {'Krm1 WOFOST':>13} {'Sigma Q_Rmmax':>15}")
    print("-" * 40)
    for cls in ("root", "stem", "leaf"):
        ot = {"root": 2, "stem": 3, "leaf": 4}[cls]
        wk = WOFOST_KRM1[cls]
        cf_qr = wk * rows[cls]["sum_struct_suc"]
        cf_total += cf_qr
        print(f"{cls:<6} {wk:>13.4f} {cf_qr:>15.3f}")
    print("-" * 40)
    print(f"{'TOTAL':<6} {'':>13} {cf_total:>15.3f} mmol Suc/d "
          f"(this is what JSON would yield if it actually used WOFOST values)")

    # 8) Decision tree
    print("\n" + "=" * 78)
    print("DECISION TREE")
    print("=" * 78)
    print("Cause (1) Unit-chain bug: rejected if total has expected magnitude. "
          f"Q_Rmmax = Krm1*Rho_s*Vol = {grand_qrmmax:.1f} mmol/d via clean Python "
          f"replica of runPM.cpp:506-509. The C++ math is the same multiplication.")
    print("Cause (2) Rho_s / volume inflation: check 'Implied DM' line above.")
    if implied_dm_g > 100:
        print(f"  --> implied DM = {implied_dm_g:.0f} g >> 50 g target. "
              f"Either Rho_s or seg_vol is INFLATED.")
    elif implied_dm_g > 70:
        print(f"  --> implied DM = {implied_dm_g:.0f} g modestly above 50 g target. "
              f"Possible mild inflation; not the dominant cause.")
    else:
        print(f"  --> implied DM = {implied_dm_g:.0f} g (~50 g target). "
              f"NOT the dominant cause.")
    print("Cause (3) Architectural overrun: dominant if math is clean and DM is "
          "physical-but-Krm1-is-WOFOST-anchored.")
    print(f"  --> WOFOST counter-factual total: {cf_total:.2f} mmol Suc/d "
          f"(vs Amthor target {AMTHOR_TARGET_PER_50G:.1f}).")
    if cf_total <= 5.0:
        print("  --> WOFOST values would land near Amthor target. "
              "Therefore the 385 mmol/d Q_Rmmax is from JSON Krm1 being "
              "8-10x WOFOST despite the JSON description claiming 'WOFOST'. "
              "PARAMETER REGIME (the JSON Krm1 is the bug). Step -1 not needed.")
    else:
        print(f"  --> WOFOST counter-factual ({cf_total:.1f}) still above Amthor target "
              f"({AMTHOR_TARGET_PER_50G:.1f}). Even literal WOFOST Krm1 would "
              f"over-shoot at the per-organ-additive scale of {n_organs} organs.")
        print("  --> ARCHITECTURAL overrun confirmed. Step -1 (V3 Babst) "
              "becomes load-bearing.")

    print("\nNumeric summary for plan-note logging:")
    print(json.dumps({
        "n_organs": int(n_organs),
        "n_nodes": int(n_nodes),
        "n_segs": int(n_segs),
        "total_volume_cm3": float(grand_vol),
        "implied_dm_g": float(implied_dm_g),
        "sum_struct_suc_mmol": float(grand_struct),
        "sum_struct_suc_as_g_suc": float(implied_struct_dm_g),
        "sum_q_rmmax_mmol_per_d": float(grand_qrmmax),
        "wofost_counter_factual_mmol_per_d": float(cf_total),
        "amthor_target_50g": AMTHOR_TARGET_PER_50G,
        "per_class": {
            cls: dict(
                n_segs=int(rows[cls]["n_segs"]),
                sum_vol_cm3=float(rows[cls]["sum_vol"]),
                sum_struct_suc=float(rows[cls]["sum_struct_suc"]),
                sum_qrmmax_mmol_per_d=float(rows[cls]["sum_qrmmax"]),
            ) for cls in ("root", "stem", "leaf")
        },
    }, indent=2))


if __name__ == "__main__":
    main()
