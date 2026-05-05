"""Matric/total convention diagnosis for SoilPsiProvider.

Question: when `hm.solve(rsx=p_s, cells=True)` reads `p_s` per-cell, does the
linear gradient in the legacy `np.linspace(psi, psi - depth, depth)` matter,
or is it ignored / equivalent to a constant?

Why it matters: ``DumuxSoilPsi`` returns matric ψ uniform-ish at t=0
(`np.full(100, psi_init)`-like), while the legacy expression is
`np.linspace(psi, psi - depth, depth)` (linear gradient -1 cm/cell).
If these produce different An at non-trivial soil ψ, ``DumuxSoilPsi`` needs a
+z gravity correction inside ``get_profile`` so it lands on the same
interpretation the production pipeline has been using.

Test design: grow a well-developed maize (day 60), then run
``run_photosynthesis_solve`` through three rsx forms × three soil-ψ levels:

  Forms:
    A. legacy_linspace   = np.linspace(psi, psi - depth, depth)
    B. constant_matric   = np.full(depth, psi)
    C. dumux_provider    = DumuxSoilPsi(psi_init=psi).get_profile(0, depth)

  Levels:
    - psi = -500   (well-watered, baseline)
    - psi = -5000  (moderate stress, ψ_leaf should drop)
    - psi = -15000 (severe drought / near wilting)

Acceptance:
  - If An_A ≈ An_B ≈ An_C across all psi -> gradient is irrelevant, no
    DumuxSoilPsi correction needed.
  - If An_A != An_B at the same psi -> the linspace gradient affects
    Doussan in a way that constant-matric doesn't reproduce.
  - If An_C != An_B at psi=-500 -> DumuxSoilPsi at t=0 is producing
    something other than uniform matric (e.g. solver step distorted IC).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from dart.coupling.growth.grow import grow_plant
from dart.coupling.hydraulics.soil_psi import DumuxSoilPsi
from dart.coupling.photosynthesis.coupled import run_photosynthesis_solve

XML_PATH = Path(__file__).resolve().parents[1] / "data" / "maize_calibrated.xml"
SIM_DAY = 60
DEPTH = 100
PSI_LEVELS = [-500.0, -5000.0, -15000.0]


class _ConstantProvider:
    """rsx[i] = psi for all i; no gravity gradient."""

    def __init__(self, psi: float):
        self._psi = float(psi)

    def get_profile(self, t_days, depth_cm):
        return np.full(depth_cm, self._psi, dtype=float)

    def update(self, t_days, sink_per_cell):
        return


class _LegacyProvider:
    """Reproduces the historical `np.linspace(psi, psi - depth, depth)`."""

    def __init__(self, psi: float):
        self._psi = float(psi)

    def get_profile(self, t_days, depth_cm):
        return np.linspace(self._psi, self._psi - depth_cm, depth_cm)

    def update(self, t_days, sink_per_cell):
        return


def _solve(plant, provider, label):
    return run_photosynthesis_solve(
        plant, SIM_DAY, par=1500.0, tleaf=25.0,
        label=label, rh=0.5,
        soil_psi_provider=provider,
    )


def main() -> int:
    print(f"Growing maize to day {SIM_DAY}...")
    # NB: enable_photosynthesis=True is REQUIRED — without it grow_plant
    # skips setSoilGrid() so seg2cell stays empty and Photosynthesis.cpp
    # silently substitutes psi_s=0 for every root segment regardless of rsx.
    # That bug invalidated the previous diagnosis (Codex rescue 2026-05-05).
    plant = grow_plant(str(XML_PATH), simulation_time=SIM_DAY, seed=42,
                       enable_photosynthesis=True)
    print(f"  plant: {len(plant.getSegments())} segments")
    print(f"  leaf segments: {len(plant.getSegmentIds(4))}")

    rows = []
    for psi in PSI_LEVELS:
        print(f"\n=== psi = {psi:+.1f} cm "
              f"({psi / 1.0197e4:+.3f} MPa) ===")
        legacy = _solve(plant, _LegacyProvider(psi), f"legacy_psi{int(psi)}")
        constant = _solve(plant, _ConstantProvider(psi), f"constant_psi{int(psi)}")
        # DumuxSoilPsi at t=0 should return ~constant matric == psi
        dumux = DumuxSoilPsi(depth_cm=DEPTH, n_cells_z=DEPTH,
                             psi_init_cm=psi, verbose=False)
        dumux_result = _solve(plant, dumux, f"dumux_psi{int(psi)}")

        # Probe what each provider actually returned
        p_legacy = _LegacyProvider(psi).get_profile(0.0, DEPTH)
        p_constant = _ConstantProvider(psi).get_profile(0.0, DEPTH)
        p_dumux = dumux.get_profile(0.0, DEPTH)

        print(f"  rsx[0] (top):   legacy={p_legacy[0]:+.2f}  "
              f"constant={p_constant[0]:+.2f}  dumux={p_dumux[0]:+.2f}")
        print(f"  rsx[50] (mid):  legacy={p_legacy[50]:+.2f}  "
              f"constant={p_constant[50]:+.2f}  dumux={p_dumux[50]:+.2f}")
        print(f"  rsx[99] (bot):  legacy={p_legacy[99]:+.2f}  "
              f"constant={p_constant[99]:+.2f}  dumux={p_dumux[99]:+.2f}")

        an_legacy = legacy['An_total_mmol'] if legacy else float('nan')
        an_constant = constant['An_total_mmol'] if constant else float('nan')
        an_dumux = dumux_result['An_total_mmol'] if dumux_result else float('nan')
        rows.append((psi, an_legacy, an_constant, an_dumux))

    print("\n\n=== SUMMARY ===")
    print(f"{'psi (cm)':>10}  {'legacy':>12}  {'constant':>12}  {'dumux':>12}  "
          f"{'L-C diff':>10}  {'L-D diff':>10}")
    for psi, l, c, d in rows:
        ldiff = l - c
        ddiff = l - d
        print(f"{psi:>10.0f}  {l:>12.4f}  {c:>12.4f}  {d:>12.4f}  "
              f"{ldiff:>+10.4f}  {ddiff:>+10.4f}")

    # Verdict
    print("\n=== VERDICT ===")
    max_lc = max(abs(r[1] - r[2]) for r in rows)
    max_ld = max(abs(r[1] - r[3]) for r in rows)
    print(f"  max |An_legacy - An_constant| = {max_lc:.4f} mmol/d")
    print(f"  max |An_legacy - An_dumux|    = {max_ld:.4f} mmol/d")

    if max_lc < 1e-3 and max_ld < 1e-3:
        print("  -> Gradient is irrelevant. DumuxSoilPsi needs no correction.")
        return 0
    if max_lc < 1e-3 and max_ld >= 1e-3:
        print("  -> Constant ≡ legacy, but DumuxSoilPsi differs. Solver step")
        print("     at t=0 perturbing IC, or numerical noise. Investigate.")
        return 2
    print("  -> Linear gradient in legacy linspace AFFECTS An. ")
    print("     DumuxSoilPsi (constant-matric) likely needs a +z gravity ")
    print("     correction in get_profile to land on the legacy interpretation.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
