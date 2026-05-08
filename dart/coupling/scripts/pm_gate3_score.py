"""pm_gate3_score.py — Gate Ch1.PM.3 mass-balance closure scorecard.

Extends `pm_gate2_score.py` (renamed in place — Gate 2 band scoring is
unchanged) with mass-balance closure: Q_S_Mesophyll (the mesophyll starch
buffer, hm.Q_out block 7 per src/external/PiafMunch/solve.cpp:191-207) is
now part of dStorage, so the residual

    |dAn - (dRm + dGr + dExud + dStorage_full)| / dAn          (target < 1 %)

with

    dStorage_full = (Q_ST + Q_meso + Q_S_Mesophyll)_last
                  - (Q_ST + Q_meso + Q_S_Mesophyll)_first

closes the An accounting that the Gate 2 scorer missed on day-55
(4.75 mmol Suc/d, 2.35 %) and day-130 (7.25 mmol Suc/d, 2.50 %). V3 was
already within 1 % via Q_ST + Q_meso alone (0.59 %); Gate 3 confirms
whether V3's residual was numeric or whether Q_S_Mesophyll engages even
at small plant size.

Bands (unchanged from Gate 2):

  * Rm/An ratio    : 5-25 % vegetative,  10-35 % maturity
  * Gr/An ratio    : 20-50 % vegetative,  0-10 % cessation-latched maturity
  * Exud/An ratio  : 1-10 % across all stages
  * C_ST_mean      : 0.30-0.90 mmol/cm^3 across the 24 h

Stage assignment (unchanged):

  * V3 (21 d)        : vegetative
  * day-55           : late-vegetative — FA cessation has latched on the
                       mainstem so Gr is near zero. Apply vegetative bands;
                       FLAG if Gr/An drops below the 20 % vegetative floor.
  * day-130          : maturity — cessation latched everywhere.

Anchors: Amthor 2000 (respiration), Babst 2022 (loading + transport),
Lohaus 2000 (phloem [Suc]). See
[[reference_amthor_2000_respiration]], [[reference_babst_2022_phloem]].

Usage:
    cpbenv/bin/python dart/coupling/scripts/pm_gate3_score.py
"""

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SCRIPTS = REPO / "dart" / "coupling" / "scripts"

CASES = [
    # (json key, label, stage)
    ("v3_maize",     "V3 (21 d)",     "vegetative"),
    ("day55_maize",  "day-55",        "late-vegetative"),
    ("day130_maize", "day-130",       "maturity"),
]

BANDS = {
    "vegetative":      {"Rm_An": (5, 25),  "Gr_An": (20, 50), "Exud_An": (1, 10)},
    "late-vegetative": {"Rm_An": (5, 25),  "Gr_An": (20, 50), "Exud_An": (1, 10)},
    "maturity":        {"Rm_An": (10, 35), "Gr_An": (0, 10),  "Exud_An": (1, 10)},
}
C_ST_BAND = (0.30, 0.90)
MASS_BAL_THRESH = 0.01


def _load(case_key):
    path = SCRIPTS / f"_pm_notebook_loop_{case_key}.json"
    if not path.exists():
        raise SystemExit(
            f"Missing {path} — run `pm_notebook_loop.py --case {case_key}` first."
        )
    with open(path) as f:
        data = json.load(f)
    return data[case_key]


def _verdict(val, lo, hi):
    return "PASS" if (lo <= val <= hi) else "FAIL"


def _score_case(case_key, label, stage, rows):
    first = rows[0]
    last = rows[-1]

    # Sinks: cumulative since simulation start, so use last - first to get
    # the integrated flux over the 24 h covered by rows[0]..rows[-1] (the
    # first substep is excluded because Q_out is read AFTER startPM, so
    # rows[0] already contains 1 substep of integral).
    dAn   = last["AnSum"]      - first["AnSum"]
    dRm   = last["cum_Q_Rm"]   - first["cum_Q_Rm"]
    dGr   = last["cum_Q_Gr"]   - first["cum_Q_Gr"]
    dExud = last["cum_Q_Exud"] - first["cum_Q_Exud"]
    dQ_ST     = last["sum_Q_ST"]     - first["sum_Q_ST"]
    dQ_meso   = last["sum_Q_meso"]   - first["sum_Q_meso"]
    # Gate Ch1.PM.3: dStorage now includes Q_S_Mesophyll (block 7 of
    # hm.Q_out); pre-Gate-3 dumps without sum_Q_S_meso fall back to the
    # Gate 2 storage definition (Q_ST + Q_meso) and the residual will
    # match the Gate 2 scorer.
    if "sum_Q_S_meso" in last and "sum_Q_S_meso" in first:
        dQ_S_meso = last["sum_Q_S_meso"] - first["sum_Q_S_meso"]
    else:
        dQ_S_meso = 0.0
    dStor = dQ_ST + dQ_meso + dQ_S_meso

    Rm_An_pct   = 100.0 * dRm   / dAn if dAn > 0 else 0.0
    Gr_An_pct   = 100.0 * dGr   / dAn if dAn > 0 else 0.0
    Exud_An_pct = 100.0 * dExud / dAn if dAn > 0 else 0.0

    # 24 h time-mean of C_ST_mean across substeps (per-substep mean is
    # already a node-mean; we want the temporal mean across the diurnal).
    cst_means = [r["C_ST_mean"] for r in rows]
    cst_24h_mean = sum(cst_means) / len(cst_means)
    cst_24h_min  = min(cst_means)
    cst_24h_max  = max(cst_means)

    # Mass-balance residual.
    mb_residual_abs = abs(dAn - (dRm + dGr + dExud + dStor))
    mb_residual_rel = mb_residual_abs / dAn if dAn > 0 else 0.0

    # Bands.
    bands = BANDS[stage]
    Rm_v   = _verdict(Rm_An_pct,   *bands["Rm_An"])
    Gr_v   = _verdict(Gr_An_pct,   *bands["Gr_An"])
    Exud_v = _verdict(Exud_An_pct, *bands["Exud_An"])
    cst_v  = _verdict(cst_24h_mean, *C_ST_BAND)
    mb_v   = "PASS" if mb_residual_rel < MASS_BAL_THRESH else "FAIL"

    # Stage-specific footnotes.
    notes = []
    if case_key == "day55_maize" and Gr_v == "FAIL" and Gr_An_pct < bands["Gr_An"][0]:
        notes.append(
            "Gr/An below vegetative floor; FA cessation has latched at "
            "day-55 so Gr -> 0 is expected (flag, not a calibration miss)."
        )
    if Rm_v == "FAIL" and Rm_An_pct < bands["Rm_An"][0]:
        notes.append(
            "Rm/An below band lower bound; literature ratio band assumes "
            "supply/demand balance, our regime is surplus (S/D ~ 27 at "
            "saturating PAR). Check absolute Rm against Amthor 2000 anchor."
        )
    if case_key == "v3_maize" and Exud_v == "FAIL" and Exud_An_pct > bands["Exud_An"][1]:
        notes.append(
            "V3 high Exud/An ratio is partly an artefact of the small "
            "AnSum (~24 mmol/d) dividing into a few mmol/d of exudation."
        )

    return {
        "case_key":      case_key,
        "label":         label,
        "stage":         stage,
        "n_substeps":    len(rows),
        "dAn":           dAn,
        "dRm":           dRm,
        "dGr":           dGr,
        "dExud":         dExud,
        "dQ_ST":         dQ_ST,
        "dQ_meso":       dQ_meso,
        "dQ_S_meso":     dQ_S_meso,
        "dStor":         dStor,
        "Rm_An_pct":     Rm_An_pct,
        "Gr_An_pct":     Gr_An_pct,
        "Exud_An_pct":   Exud_An_pct,
        "cst_24h_mean":  cst_24h_mean,
        "cst_24h_min":   cst_24h_min,
        "cst_24h_max":   cst_24h_max,
        "mb_residual_abs": mb_residual_abs,
        "mb_residual_rel": mb_residual_rel,
        "verdict":       {"Rm_An": Rm_v, "Gr_An": Gr_v, "Exud_An": Exud_v,
                          "C_ST_mean": cst_v, "mass_balance": mb_v},
        "notes":         notes,
    }


def _print_table(results):
    # Header.
    print("=" * 100)
    print("Gate Ch1.PM.3 — mass-balance closure scorecard (Gate 2 bands + Q_S_Mesophyll)")
    print("=" * 100)
    print(
        f"{'case':<14}{'stage':<18}{'Rm/An [%]':>14}{'Gr/An [%]':>14}"
        f"{'Exud/An [%]':>14}{'C_ST mean':>12}{'mass-bal':>12}"
    )
    print("-" * 100)
    for r in results:
        v = r["verdict"]
        print(
            f"{r['case_key']:<14}{r['stage']:<18}"
            f"{r['Rm_An_pct']:>9.2f} {v['Rm_An']:<3}"
            f"{r['Gr_An_pct']:>9.2f} {v['Gr_An']:<3}"
            f"{r['Exud_An_pct']:>9.2f} {v['Exud_An']:<3}"
            f"{r['cst_24h_mean']:>7.3f} {v['C_ST_mean']:<3}"
            f"{r['mb_residual_rel']*100:>7.2f}%{v['mass_balance']:>4}"
        )
    print("-" * 100)
    print()
    # Per-case detail block.
    for r in results:
        print(f"  {r['case_key']}  ({r['label']}, {r['n_substeps']} substeps, stage = {r['stage']})")
        print(
            f"    24h cumulative (mmol Suc): "
            f"dAn={r['dAn']:.3f}, dRm={r['dRm']:.3f}, dGr={r['dGr']:.4f}, "
            f"dExud={r['dExud']:.3f}"
        )
        print(
            f"    Storage breakdown (mmol Suc): "
            f"dQ_ST={r['dQ_ST']:.3f}, dQ_meso={r['dQ_meso']:.3f}, "
            f"dQ_S_meso={r['dQ_S_meso']:.3f}, dStorage_full={r['dStor']:.3f}"
        )
        print(
            f"    C_ST_mean over 24 h: mean={r['cst_24h_mean']:.3f} "
            f"(min={r['cst_24h_min']:.3f}, max={r['cst_24h_max']:.3f})"
        )
        print(
            f"    Mass-balance residual: |dAn - (dRm+dGr+dExud+dStorage_full)| = "
            f"{r['mb_residual_abs']:.3f} mmol Suc/d "
            f"({r['mb_residual_rel']*100:.2f} % of dAn) -> {r['verdict']['mass_balance']}"
        )
        for note in r["notes"]:
            print(f"    NOTE: {note}")
        print()

    # Summary.
    n_pass = sum(
        1 for r in results
        for k in ("Rm_An", "Gr_An", "Exud_An", "C_ST_mean")
        if r["verdict"][k] == "PASS"
    )
    n_total = 4 * len(results)
    n_mb_pass = sum(1 for r in results if r["verdict"]["mass_balance"] == "PASS")
    print(
        f"Summary: {n_pass}/{n_total} band-PASS across {len(results)} cases "
        f"(4 bands per case).  Mass balance: {n_mb_pass}/{len(results)} PASS."
    )


def main():
    results = []
    for case_key, label, stage in CASES:
        rows = _load(case_key)
        results.append(_score_case(case_key, label, stage, rows))

    _print_table(results)

    out_path = SCRIPTS / "_pm_gate3_score.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nJSON dump: {out_path}")


if __name__ == "__main__":
    main()
