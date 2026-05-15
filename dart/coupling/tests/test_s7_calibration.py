"""test_s7_calibration.py — Plan §S7 acceptance gate for buffered-carbon
calibration.

Three fixtures, each ticks a distinct hard rule from
PLAN_BUFFERED_CARBON_GROWTH_2026-05-15.md §S7 + §9:

  1. ``test_s7_mass_balance_day130_closed_loop`` — cumulative Liebig
     residual ≤ 1 % across the 130-day PM+DuMux closed loop using the
     currently-baked ``maize_calibrated.xml``.  Σ An ≈ Σ Rm + Σ Rg
     + Δreserve + ΔΣ local_C + storage_loss + remob_loss + exudation.

  2. ``test_s7_realised_fa_fraction`` — total realised cumulative organ
     length at day-130 divided by the FA-no-carbon oracle is inside
     [0.4, 0.9].  Below 0.4 ⇒ closed-loop is supply-starved (calibration
     too lean / capacity too small).  Above 0.9 ⇒ buffer barely engaged
     (effectively FA-target — no Liebig signal).

  3. ``test_s7_no_beta_prime`` — source-code regression guard.  Asserts
     the β' CW_Gr clearing block (deleted in S0 commit ``6e320940``,
     superseded by S4 daily-batched extension) does not reappear in
     ``pm_substep.py``.

The two live-simulation fixtures are gated behind ``@pytest.mark.slow_s7``
and require ``RUN_S7_LIVE=1``.  Without that, the live tests are
skipped — but ``test_s7_no_beta_prime`` always runs as a cheap source
check.

If a calibration sweep CSV (default
``out_calibration_s7_day130.csv`` at the repo root) is present, the
fixtures will additionally summarise the best-row outcome so a developer
who is iterating on the sweep can see whether the latest grid contains a
combo that meets the acceptance band.

Run (default — fast source-only check):

    cpbenv/bin/python -m pytest dart/coupling/tests/test_s7_calibration.py -v

Run with live 130-day PM+DuMux (server, ~4–6 h):

    RUN_S7_LIVE=1 cpbenv/bin/python -m pytest \\
        dart/coupling/tests/test_s7_calibration.py -m slow_s7 -v
"""
from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

PM_SUBSTEP_SRC = (REPO_ROOT / "dart" / "coupling" / "carbon" / "pm_substep.py")
CALIBRATION_CSV_CANDIDATES = [
    REPO_ROOT / "out_calibration_s7_day130.csv",
    REPO_ROOT / "dart" / "coupling" / "out_calibration_s7_day130.csv",
    REPO_ROOT / "dart" / "coupling" / "scripts" / "out_calibration_s7_day130.csv",
]

MB_RESIDUAL_MAX_PCT = 1.0
REALISED_FA_LOW = 0.4
REALISED_FA_HIGH = 0.9


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------

def _find_calibration_csv() -> Path | None:
    for cand in CALIBRATION_CSV_CANDIDATES:
        if cand.exists():
            return cand
    return None


def _read_best_row(csv_path: Path) -> dict | None:
    """Pick the row whose realised-FA fraction is closest to the midpoint
    of [REALISED_FA_LOW, REALISED_FA_HIGH] AND has cum_mb_residual_pct ≤
    MB_RESIDUAL_MAX_PCT.  Returns None when the CSV is empty or no row
    satisfies the MB band.
    """
    target = 0.5 * (REALISED_FA_LOW + REALISED_FA_HIGH)
    best = None
    best_d = None
    with csv_path.open() as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            if row.get("status") != "OK":
                continue
            try:
                mb = float(row["cum_mb_residual_pct"])
                fa = float(row["realised_fa_fraction"])
            except (KeyError, ValueError):
                continue
            if mb > MB_RESIDUAL_MAX_PCT:
                continue
            d = abs(fa - target)
            if best_d is None or d < best_d:
                best, best_d = row, d
    return best


def _run_live_day130(seed: int = 7,
                     bootstrap_day: int = 30,
                     sim_days: int = 130,
                     soil_mode: str = "dumux",
                     soil_psi_cm: float = -300.0,
                     krm1_mult: float = 0.01,
                     kmfu_mult: float = 0.1) -> dict:
    """Execute one PM+DuMux 130-day closed-loop run at the *currently
    baked* maize_calibrated.xml.  Returns the calibrate_c_cost script's
    row schema."""
    script_dir = REPO_ROOT / "dart" / "coupling" / "scripts"
    sys.path.insert(0, str(script_dir))
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "calibrate_c_cost_s7",
        script_dir / "calibrate_c_cost_per_cm_2026-05-15.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # Read currently-baked XML values to keep the live run faithful to
    # the on-disk state rather than the C++ defaults.
    import plantbox as pb
    p_probe = pb.Plant()
    p_probe.readParameters(str(mod.MAIZE_XML))
    leaf_rps = [rp for rp in p_probe.getOrganRandomParameter(4)
                if rp is not None and int(rp.subType) >= 2]
    stem_rps = [rp for rp in p_probe.getOrganRandomParameter(3)
                if rp is not None and int(rp.subType) >= 2]
    root_rps = [rp for rp in p_probe.getOrganRandomParameter(2)
                if rp is not None and int(rp.subType) >= 2]
    seed_rp = p_probe.getOrganRandomParameter(1)[0]
    knobs = {
        "c_cost_leaf": float(leaf_rps[0].c_cost_per_cm) if leaf_rps else 0.35,
        "c_cost_stem": float(stem_rps[0].c_cost_per_cm) if stem_rps else 0.55,
        "c_cost_root": float(root_rps[0].c_cost_per_cm) if root_rps else 0.20,
        "local_cap_factor": (float(leaf_rps[0].local_C_pool_capacity_factor)
                             if leaf_rps else 0.5),
        "local_cap_factor_root": (float(root_rps[0].local_C_pool_capacity_factor)
                                  if root_rps else 0.0),
        "reserve_cap_factor": float(seed_rp.reserve_capacity_factor),
        "starch_remob_rate": float(seed_rp.starch_remob_rate),
        "starch_storage_eff": float(seed_rp.starch_storage_efficiency),
        "starch_remob_eff": float(seed_rp.starch_remob_efficiency),
    }
    return mod.run_one_combo(
        knobs, seed=seed, bootstrap_day=bootstrap_day,
        sim_days=sim_days, soil_mode=soil_mode, soil_psi_cm=soil_psi_cm,
        krm1_mult=krm1_mult, kmfu_mult=kmfu_mult, verbose=True,
    )


# ----------------------------------------------------------------------
# Fixture #1 — cumulative Liebig closure ≤ 1 %
# ----------------------------------------------------------------------

@pytest.mark.slow_s7
@pytest.mark.skipif(
    os.environ.get("RUN_S7_LIVE", "0") != "1",
    reason="Set RUN_S7_LIVE=1 to run the live 130-day PM+DuMux closure "
           "(server-only, ~4-6 h).",
)
def test_s7_mass_balance_day130_closed_loop():
    row = _run_live_day130()
    assert row["status"] == "OK", f"Live run failed: {row.get('error')}"
    mb = float(row["cum_mb_residual_pct"])
    msg = (f"S7.1 Liebig closure: cumulative residual {mb:.3f}% over "
           f"{row['sim_days']}-day PM+DuMux (target ≤ "
           f"{MB_RESIDUAL_MAX_PCT}%). An={row['cum_an_mmol']:.2f} "
           f"used={row['cum_used_mmol']:.2f} mmol. "
           f"PM={row['n_pm_calls']} (fail={row['n_pm_fail']}).")
    assert mb <= MB_RESIDUAL_MAX_PCT, msg
    print(msg)


# ----------------------------------------------------------------------
# Fixture #2 — realised vs FA oracle fraction ∈ [0.4, 0.9]
# ----------------------------------------------------------------------

@pytest.mark.slow_s7
@pytest.mark.skipif(
    os.environ.get("RUN_S7_LIVE", "0") != "1",
    reason="Set RUN_S7_LIVE=1 to run the live 130-day PM+DuMux realised-FA "
           "check (server-only, ~4-6 h).",
)
def test_s7_realised_fa_fraction():
    row = _run_live_day130()
    assert row["status"] == "OK", f"Live run failed: {row.get('error')}"
    fa = float(row["realised_fa_fraction"])
    msg = (f"S7.2 realised-FA fraction: total realised/oracle = {fa:.3f} "
           f"at day {row['sim_days']} (target ∈ ["
           f"{REALISED_FA_LOW:.2f}, {REALISED_FA_HIGH:.2f}]). "
           f"mainstem={row['mainstem_fraction']:.3f}, "
           f"leaf={row['leaf_fraction']:.3f}, "
           f"root={row['root_fraction']:.3f}.")
    assert REALISED_FA_LOW <= fa <= REALISED_FA_HIGH, msg
    print(msg)


# ----------------------------------------------------------------------
# Fixture #3 — no β' regression
# ----------------------------------------------------------------------

def test_s7_no_beta_prime():
    """Source-level guarantee that the β' CW_Gr clearing block does NOT
    reappear in pm_substep.py.  Plan §12.6 + §13 keep this as a permanent
    regression guard."""
    src = PM_SUBSTEP_SRC.read_text()
    # The original β' block (S0 deletion 2026-05-15) cleared the CW_Gr
    # dict for every CWLimitedGrowth RP at the start of each substep
    # iteration.  Two signatures we explicitly forbid:
    forbidden_signatures = [
        # The literal `.f_gf.CW_Gr = {}` assignment pattern.
        ".CW_Gr = {}",
        # The "Step β'" header that historically labelled the block.
        "Step β'",
    ]
    matches = [sig for sig in forbidden_signatures if sig in src]
    assert not matches, (
        "β' regression detected in pm_substep.py — patterns "
        f"{matches} should have been deleted in S0 (commit 6e320940) and "
        "must not be re-introduced.  See PLAN_BUFFERED_CARBON_GROWTH "
        "§12.6 + §13.")


# ----------------------------------------------------------------------
# Fixture #4 — sweep CSV summary (no-op when CSV missing, advisory only)
# ----------------------------------------------------------------------

def test_s7_sweep_csv_summary():
    """Advisory: if a calibration sweep CSV is on disk, surface the best
    row so devs see whether the current grid contains an in-band combo.
    The fixture only fails when the CSV exists AND no row meets the
    cumulative MB ≤ 1% rule — which would mean we have data but it's
    still off-target."""
    csv_path = _find_calibration_csv()
    if csv_path is None:
        pytest.skip("No calibration CSV present yet — sweep not run.")
    best = _read_best_row(csv_path)
    assert best is not None, (
        f"Sweep CSV present at {csv_path} but no row meets the cumulative "
        f"MB ≤ {MB_RESIDUAL_MAX_PCT}% rule.  Widen the sweep or rerun "
        "with smaller knobs.")
    fa = float(best["realised_fa_fraction"])
    mb = float(best["cum_mb_residual_pct"])
    msg = (f"Best sweep combo: c_cost_leaf={best['c_cost_leaf']}, "
           f"c_cost_stem={best['c_cost_stem']}, "
           f"cap={best['local_cap_factor']}, "
           f"seed={best['seed']} → MB={mb:.3f}%, FA-frac={fa:.3f}.")
    print(msg)
    assert REALISED_FA_LOW <= fa <= REALISED_FA_HIGH, (
        f"Best in-MB-band row has FA-fraction {fa:.3f} outside "
        f"[{REALISED_FA_LOW}, {REALISED_FA_HIGH}].  Calibration sweep "
        "has not landed an in-band combo yet.")
