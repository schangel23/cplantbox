#!/usr/bin/env python3
"""Ch2 diagnostic: locate the An↔Rm gap surfaced by Gate Ch1.PMDM.5 G5.1.

The v4 nile smoke (PLAN_PIAFMUNCH_DUMUX_COUPLING_2026-05-09 line 1410-1416)
landed G5.1 worst-plant residuals at 146-163 %: PM-calculated Rm demand
exceeds Baleno-integrated diurnal An at V3 by ~1.5×. This script runs
three orthogonal probes on the same V3 maize plant to decouple the
candidate causes:

  Probe 1 — Krm1 sweep
      Sweep Krm1 multiplier ∈ {0.25, 0.5, 1.0, 1.5, 2.0}× WOFOST default
      (root=0.012, stem=0.015, leaf=0.030 d⁻¹). Report Rm/An, locate the
      crossover where Rm ≈ An. If crossover lands at multiplier < 1
      under representative An, WOFOST Krm1 is over-calibrated at V3
      cumulative-GDD scale.

  Probe 2 — Baleno aggregation cross-check (3 rows after 2026-05-13)
      Three PM calls on the same V3 plant:
        (a) inject_an_target=False, par_umol=600  → PM's native FvCB
            at constant peak PAR drives Ag4Phloem. Reference for the
            "PM constant-PAR over-production" magnitude (173× at V3
            day-21 per 2026-05-13 nile measurement).
        (b) inject_an_target=True, par_umol=600   → Ag4Phloem rescaled
            so AnSum_suc matches a Baleno-shaped diurnal An target.
            Asserts wiring correctness of the inject path.
        (c) inject_an_target=False, par_umol=120  → PM native FvCB at
            Baleno-representative diurnal-mean PAR. Disambiguation row:
            if An(120) ≈ 0.2 × An(600) → PM is linear in PAR, the over-
            count is constant-PAR-vs-diurnal duration. If An(120) ≈
            An(600) → PM is light-saturated below 600, the over-count
            is from somewhere else (FvCB constants, leaf-area, 24-h
            vs daylight duration).
      Compares An_total_mmol_pm_internal vs the diurnal-realistic
      target AND the (c)/(a) ratio to disambiguate cause.

  Probe 3 — ψ_init sweep
      DumuxSoilPsi at ψ_init ∈ {-100, -200, -300, -500, -1000} cm,
      same V3 plant, default Krm1. Report Rm, An, ψ_leaf_min, Rm/An.
      If the gap closes (Rm/An → 1) at ψ_init = -100 cm, the diurnal
      smoke's --soil-psi-cm -300 is over-stressing early-V plants
      enough to depress An below realistic field rates.

Each probe writes a JSON sidecar to tests/fixtures/pm_an_rm_gap_<probe>.json
and prints a unified comparison table.

Usage (from /home/lukas/PHD/CPlantBox)::

    cpbenv/bin/python dart/coupling/scripts/pm_an_rm_gap_probe.py
    cpbenv/bin/python dart/coupling/scripts/pm_an_rm_gap_probe.py --probe krm1
    cpbenv/bin/python dart/coupling/scripts/pm_an_rm_gap_probe.py --probe psi_init

On the local box without rosi_richards, --probe psi_init skips
automatically (gracefully reports "rosi_richards unavailable").
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
COUPLING_DIR = SCRIPT_DIR.parent
CPLANTBOX_ROOT = COUPLING_DIR.parent.parent
FIXTURES_DIR = COUPLING_DIR / "tests" / "fixtures"
sys.path.insert(0, str(CPLANTBOX_ROOT))

from dart.coupling.config import DEFAULT_XML  # noqa: E402
from dart.coupling.growth.grow import grow_plant  # noqa: E402
from dart.coupling.carbon.pm_substep import solve_carbon_partitioning_pm  # noqa: E402
from dart.coupling.hydraulics.soil_psi import (  # noqa: E402
    FixedSoilPsi,
    make_provider,
)


V3_DAY = 21
SEED = 42
TAIR_C = 20.75  # matches test_g5_acceptance.BABST_MET T_mean
# Constant met fixture mirroring test_g5_acceptance — same V3 plant state.
BABST_MET = {
    d: {"T_mean_C": 20.75, "T_min_C": 19.0, "T_max_C": 22.0,
        "PAR_MJ_m2_d": 30.0 * 0.219, "VPD_kPa": 1.0,
        "RH_pct": 60.0, "Wind_m_s": 0.5}
    for d in range(1, 60)
}

# Diurnal-realistic daily An anchor for Probe 2. The v4 smoke logged
# day-20 mean An ≈ 0.001742 mol CO2/plant/day under T_mean=17 °C
# (PLAN_PIAFMUNCH_DUMUX_COUPLING_2026-05-09 line 1414). Use that as a
# representative Baleno-integrated daily An target.
BALENO_DIURNAL_AN_MOL = 0.001742  # mol CO2/plant/day


def _fresh_v3_plant():
    """Fresh V3 maize plant, default Krm1, no CW wrapping yet."""
    return grow_plant(
        xml_path=str(DEFAULT_XML),
        simulation_time=V3_DAY,
        min_stem_nodes=10,
        min_leaf_nodes=4,
        enable_photosynthesis=True,
        seed=SEED,
        daily_met=BABST_MET,
        T_air_default=TAIR_C,
    )


def _synth_an_per_leaf(plant, an_total_mol: float) -> np.ndarray:
    """Uniform per-leaf-segment An vector summing to an_total_mol."""
    n_leaf_segs = len(plant.getSegmentIds(4))
    if n_leaf_segs == 0:
        return np.array([], dtype=float)
    return np.full(n_leaf_segs, an_total_mol / n_leaf_segs, dtype=float)


def _summarise(label: str, result: Optional[dict]) -> dict:
    """Extract the headline carbon-balance fields from a PM result."""
    if result is None:
        return {
            "label": label,
            "ok": False,
            "An_total_mmol_internal": None,
            "An_total_mmol_target": None,
            "Rm_total_mmol": None,
            "Rg_total_mmol": None,
            "Rm_over_An": None,
            "psi_leaf_min_cm": None,
            "mass_balance_residual_pct": None,
        }
    an_int = float(result["An_total_mmol"])
    an_tgt = float(result["An_total_mmol_target"])
    rm = float(result["Rm_total_mmol"])
    rg = float(result["Rg_total_mmol"])
    rm_over_an = rm / an_int if abs(an_int) > 1e-9 else None
    return {
        "label": label,
        "ok": True,
        "An_total_mmol_internal": an_int,
        "An_total_mmol_target": an_tgt,
        "Rm_total_mmol": rm,
        "Rg_total_mmol": rg,
        "Rm_over_An": rm_over_an,
        "psi_leaf_min_cm": result.get("psi_leaf_min_cm"),
        "mass_balance_residual_pct": float(result["mass_balance_residual_pct"]),
    }


# ---------------------------------------------------------------------------
# Probe 1 — Krm1 sweep
# ---------------------------------------------------------------------------

def probe_krm1(multipliers=None) -> dict:
    if multipliers is None:
        multipliers = [0.25, 0.5, 1.0, 1.5, 2.0]
    rows = []
    print()
    print("=" * 78)
    print("Probe 1 — Krm1 sensitivity sweep")
    print("=" * 78)
    print(f"  V3 plant, day={V3_DAY}, Tair={TAIR_C}°C, "
          f"FixedSoilPsi(-300 cm), advance_plant=True")
    print(f"  Multipliers (× WOFOST root=0.012, stem=0.015, leaf=0.030): "
          f"{multipliers}")
    print()
    for m in multipliers:
        plant = _fresh_v3_plant()
        An = _synth_an_per_leaf(plant, BALENO_DIURNAL_AN_MOL)
        t0 = time.time()
        result = solve_carbon_partitioning_pm(
            plant,
            An,
            Tair_C=TAIR_C,
            day=V3_DAY,
            n_substeps=24,
            advance_plant=True,
            soil_psi_provider=FixedSoilPsi(psi_cm=-300.0, n_cells=150),
            inject_an_target=False,
            krm1_multiplier=m,
        )
        dt = time.time() - t0
        row = _summarise(f"krm1×{m}", result)
        row["wall_s"] = round(dt, 1)
        row["multiplier"] = m
        rows.append(row)
        if result is not None:
            print(f"  Krm1×{m:.2f}  An={row['An_total_mmol_internal']:8.3f}  "
                  f"Rm={row['Rm_total_mmol']:8.3f}  Rg={row['Rg_total_mmol']:7.3f}  "
                  f"Rm/An={row['Rm_over_An']:6.3f}  mb={row['mass_balance_residual_pct']:5.2f}%  "
                  f"({dt:.0f}s)")
        else:
            print(f"  Krm1×{m:.2f}  PM solver FAILED  ({dt:.0f}s)")

    # Linearity check: if Rm scales linearly with Krm1, doubling the
    # multiplier should ~double Rm. Crossover-locate Rm ≈ An: at what
    # multiplier does Rm_over_An ≈ 1?
    rm_vals = [r["Rm_total_mmol"] for r in rows if r["Rm_total_mmol"] is not None]
    multipliers_ok = [r["multiplier"] for r in rows if r["Rm_total_mmol"] is not None]
    if len(rm_vals) >= 2:
        slope = (rm_vals[-1] - rm_vals[0]) / (multipliers_ok[-1] - multipliers_ok[0])
        intercept = rm_vals[0] - slope * multipliers_ok[0]
        print()
        print(f"  Linear fit: Rm ≈ {slope:.3f} × multiplier + {intercept:.3f} mmol CO2/d")
    crossover = None
    for r in rows:
        if r["Rm_over_An"] is None:
            continue
        if r["Rm_over_An"] <= 1.0 and crossover is None:
            crossover = r["multiplier"]
    if crossover is not None:
        print(f"  Rm/An ≤ 1 first crossed at Krm1×{crossover:.2f}")
    else:
        print("  Rm/An > 1 across the entire swept range — Krm1 cannot close "
              "the gap alone under this An target")
    return {"probe": "krm1", "rows": rows}


# ---------------------------------------------------------------------------
# Probe 2 — Baleno aggregation cross-check
# ---------------------------------------------------------------------------

def probe_baleno() -> dict:
    print()
    print("=" * 78)
    print("Probe 2 — Baleno aggregation cross-check (constant-PAR vs diurnal target)")
    print("=" * 78)
    print(f"  V3 plant, day={V3_DAY}, FixedSoilPsi(-300 cm), advance_plant=True")
    print(f"  Diurnal An anchor: {BALENO_DIURNAL_AN_MOL} mol CO2/plant/day "
          f"(v4 smoke day-20 mean)")
    print()
    rows = []

    # (a) PM-internal FvCB at constant par_umol=600 — bare smokeline.
    plant_a = _fresh_v3_plant()
    An_a = _synth_an_per_leaf(plant_a, BALENO_DIURNAL_AN_MOL)  # used only as target marker
    t0 = time.time()
    res_a = solve_carbon_partitioning_pm(
        plant_a, An_a, Tair_C=TAIR_C, day=V3_DAY, n_substeps=24,
        advance_plant=True,
        soil_psi_provider=FixedSoilPsi(psi_cm=-300.0, n_cells=150),
        inject_an_target=False,
        par_umol=600.0,
    )
    dt_a = time.time() - t0
    row_a = _summarise("constant_par600_native_fvcb", res_a)
    row_a["wall_s"] = round(dt_a, 1)
    row_a["par_umol"] = 600.0
    row_a["inject_an_target"] = False
    rows.append(row_a)
    if res_a:
        print(f"  (a) PM native FvCB @ PAR=600  An_internal={row_a['An_total_mmol_internal']:8.3f}  "
              f"An_target={row_a['An_total_mmol_target']:8.3f}  "
              f"Rm/An={row_a['Rm_over_An']:6.3f}  ({dt_a:.0f}s)")

    # (b) Inject diurnal-target rescaling.
    plant_b = _fresh_v3_plant()
    An_b = _synth_an_per_leaf(plant_b, BALENO_DIURNAL_AN_MOL)
    t0 = time.time()
    res_b = solve_carbon_partitioning_pm(
        plant_b, An_b, Tair_C=TAIR_C, day=V3_DAY, n_substeps=24,
        advance_plant=True,
        soil_psi_provider=FixedSoilPsi(psi_cm=-300.0, n_cells=150),
        inject_an_target=True,
        par_umol=600.0,
    )
    dt_b = time.time() - t0
    row_b = _summarise("inject_target_baleno_diurnal", res_b)
    row_b["wall_s"] = round(dt_b, 1)
    row_b["par_umol"] = 600.0
    row_b["inject_an_target"] = True
    rows.append(row_b)
    if res_b:
        print(f"  (b) Inject diurnal target      An_internal={row_b['An_total_mmol_internal']:8.3f}  "
              f"An_target={row_b['An_total_mmol_target']:8.3f}  "
              f"Rm/An={row_b['Rm_over_An']:6.3f}  ({dt_b:.0f}s)")

    # (c) PM-internal FvCB at low PAR=120 — disambiguation row.
    # Tests the assumption that the (a) over-production is a constant-PAR
    # convention issue (PM stays at peak 600 µmol/m²/s for 24 h) and that
    # a Baleno-representative *mean* PAR (~120 µmol/m²/s, the typical
    # diurnal average over daylight hours) would give an An roughly
    # one-fifth of (a)'s 302 mmol/day if PM's FvCB is linear in PAR below
    # saturation. If An stays close to (a) → PM is light-saturated below
    # 600 and the over-count comes from a different mechanism (FvCB
    # constants, leaf-area integration, duration over a 24-h window with
    # no dark period subtracted).
    plant_c = _fresh_v3_plant()
    An_c = _synth_an_per_leaf(plant_c, BALENO_DIURNAL_AN_MOL)
    t0 = time.time()
    res_c = solve_carbon_partitioning_pm(
        plant_c, An_c, Tair_C=TAIR_C, day=V3_DAY, n_substeps=24,
        advance_plant=True,
        soil_psi_provider=FixedSoilPsi(psi_cm=-300.0, n_cells=150),
        inject_an_target=False,
        par_umol=120.0,
    )
    dt_c = time.time() - t0
    row_c = _summarise("constant_par120_native_fvcb", res_c)
    row_c["wall_s"] = round(dt_c, 1)
    row_c["par_umol"] = 120.0
    row_c["inject_an_target"] = False
    rows.append(row_c)
    if res_c:
        print(f"  (c) PM native FvCB @ PAR=120  An_internal={row_c['An_total_mmol_internal']:8.3f}  "
              f"An_target={row_c['An_total_mmol_target']:8.3f}  "
              f"Rm/An={row_c['Rm_over_An']:6.3f}  ({dt_c:.0f}s)")

    if res_a and res_b:
        # The smoking gun: PM-internal An (constant PAR=600 for 24 h) /
        # diurnal-shape target. A ratio ≫ 1 quantifies the documented
        # over-production at constant peak PAR.
        ratio = (row_a["An_total_mmol_internal"]
                 / row_b["An_total_mmol_target"])
        print()
        print(f"  PM-internal-An / Baleno-diurnal-target = {ratio:.3f}")
        print(f"  → constant-PAR PM over-produces An by {ratio:.1f}× relative "
              f"to a Baleno-integrated diurnal day.")
        print("  (V3 day-21 baseline: 173× per 2026-05-13 nile measurement;"
              " v3 ~25× was day-130 leaf-area-scaled.)")

    if res_a and res_c:
        # PAR-sensitivity disambiguation. The expected linear-regime ratio
        # is 120/600 = 0.20 — i.e. An at PAR=120 should be ~one-fifth of
        # An at PAR=600 if PM's FvCB is unsaturated at PAR=600.
        par_ratio = (row_c["An_total_mmol_internal"]
                     / row_a["An_total_mmol_internal"])
        print()
        print(f"  An(PAR=120) / An(PAR=600) = {par_ratio:.3f}  "
              f"(linear expectation: 0.20)")
        if par_ratio < 0.40:
            print("  → PM FvCB IS roughly linear in PAR at V3 day-21; the "
                  "173× over-production comes from the constant-PAR=600 "
                  "convention. Baleno's hourly-mean PAR (~120) is the "
                  "physically correct driver; PM's constant-peak-600 "
                  "convention over-shoots by ~5× from light intensity "
                  "alone, plus another ~35× from the 24-h-vs-daylight "
                  "duration mismatch.")
        elif par_ratio > 0.80:
            print("  → PM FvCB is LIGHT-SATURATED below PAR=600 at V3 day-21; "
                  "An is insensitive to PAR in this regime, so the 173× "
                  "over-production lives somewhere other than light "
                  "intensity (FvCB Vcmax/Jmax constants, leaf-area "
                  "integration, or the 24-h duration with no dark "
                  "period subtracted). Baleno hourly-aggregation cross-"
                  "check needed: compare PM-internal An vs Σ(Baleno hourly An).")
        else:
            print("  → PM FvCB is partially saturated at PAR=120; the gap "
                  "is a mix of PAR sensitivity and FvCB constants. Reduce "
                  "PAR further (~30) to fully de-light or audit Vcmax/Jmax.")

    return {"probe": "baleno", "rows": rows}


# ---------------------------------------------------------------------------
# Probe 3 — ψ_init sweep
# ---------------------------------------------------------------------------

def probe_psi_init(psi_values=None) -> Optional[dict]:
    if psi_values is None:
        psi_values = [-100.0, -200.0, -300.0, -500.0, -1000.0]
    try:
        from dart.coupling.hydraulics.soil_psi import DumuxSoilPsi  # noqa: F401
        # Probe binding availability.
        DumuxSoilPsi(
            min_b=(-1, -1, -3), max_b=(1, 1, 0),
            cell_number=(1, 1, 3), psi_init_cm=-300.0,
        )
    except Exception as e:
        print()
        print("=" * 78)
        print("Probe 3 — ψ_init sweep SKIPPED (rosi_richards binding unavailable)")
        print(f"  {e}")
        print("=" * 78)
        return None

    rows = []
    print()
    print("=" * 78)
    print("Probe 3 — ψ_init sweep (DumuxSoilPsi 1×1×100 column)")
    print("=" * 78)
    print(f"  V3 plant, day={V3_DAY}, default Krm1, advance_plant=True, "
          f"inject_an_target=False")
    print(f"  ψ_init values [cm]: {psi_values}")
    print()
    for psi in psi_values:
        plant = _fresh_v3_plant()
        An = _synth_an_per_leaf(plant, BALENO_DIURNAL_AN_MOL)
        # Match plant default soil grid (grow.py DEFAULT_SOIL_*) so the
        # seg→cell mapping range fits inside the provider profile.
        provider = make_provider(
            "dumux",
            soil_psi_cm=psi,
            min_b=(-50.0, -50.0, -150.0),
            max_b=(50.0, 50.0, 0.0),
            cell_number=(1, 1, 150),
        )
        setattr(provider, "_t_last_days", float(V3_DAY))
        t0 = time.time()
        result = solve_carbon_partitioning_pm(
            plant, An, Tair_C=TAIR_C, day=V3_DAY, n_substeps=24,
            advance_plant=True,
            soil_psi_provider=provider,
            inject_an_target=False,
        )
        dt = time.time() - t0
        row = _summarise(f"psi_init={psi}", result)
        row["wall_s"] = round(dt, 1)
        row["psi_init_cm"] = psi
        rows.append(row)
        if result is not None:
            psi_min = row["psi_leaf_min_cm"]
            psi_min_str = f"{psi_min:7.1f}" if psi_min is not None else "  None "
            print(f"  ψ_init={psi:7.1f}  An={row['An_total_mmol_internal']:8.3f}  "
                  f"Rm={row['Rm_total_mmol']:8.3f}  Rg={row['Rg_total_mmol']:7.3f}  "
                  f"Rm/An={row['Rm_over_An']:6.3f}  "
                  f"ψ_leaf_min={psi_min_str}  ({dt:.0f}s)")
        else:
            print(f"  ψ_init={psi:7.1f}  PM solver FAILED  ({dt:.0f}s)")

    finite = [r for r in rows if r["Rm_over_An"] is not None]
    if finite:
        best = min(finite, key=lambda r: r["Rm_over_An"])
        print()
        print(f"  Best Rm/An = {best['Rm_over_An']:.3f} at ψ_init = "
              f"{best['psi_init_cm']:.1f} cm")
        if best["Rm_over_An"] < 1.0:
            print(f"  → ψ_init can close the gap; the -300 cm default in v4 "
                  f"smoke is over-stressing V3 plants.")
        else:
            print("  → Gap persists across the swept ψ_init range; the gap "
                  "is not just an over-stress artifact.")
    return {"probe": "psi_init", "rows": rows}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def _write_sidecar(result: dict):
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    name = f"pm_an_rm_gap_{result['probe']}.json"
    path = FIXTURES_DIR / name
    with path.open("w") as f:
        json.dump(result, f, indent=2)
    print(f"  → wrote {path}")


def main():
    parser = argparse.ArgumentParser(description="Ch2 An↔Rm gap probes")
    parser.add_argument("--probe", choices=("all", "krm1", "baleno", "psi_init"),
                        default="all")
    parser.add_argument("--no-sidecar", action="store_true",
                        help="skip JSON sidecar writes (console-only output)")
    args = parser.parse_args()

    results = []
    if args.probe in ("all", "krm1"):
        results.append(probe_krm1())
    if args.probe in ("all", "baleno"):
        results.append(probe_baleno())
    if args.probe in ("all", "psi_init"):
        r = probe_psi_init()
        if r is not None:
            results.append(r)

    if not args.no_sidecar:
        print()
        print("Writing JSON sidecars:")
        for r in results:
            _write_sidecar(r)

    print()
    print("=" * 78)
    print("Summary table")
    print("=" * 78)
    for r in results:
        print(f"  {r['probe']}: {len(r['rows'])} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
