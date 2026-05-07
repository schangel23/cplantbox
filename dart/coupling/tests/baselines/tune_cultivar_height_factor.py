#!/usr/bin/env python3
"""S4 tuning probe — empirically sweep H + dev to land MF3D's 14-16-leaf
tall subset target (mean stem-only top ≈ 235 cm, σ ≈ 17–18 cm).

Plan: PLAN_CULTIVAR_HEIGHT_FACTOR_2026-05-07.md §S4.

Runs 20 seeds for each (H, dev) combo with cultivar_height_factor baked
via a tmp XML overlay (does NOT mutate the production XML). Reports per-
combo mean / σ of mainstem.getLength(True) and z_max so the user can pick
the value to bake into maize_calibrated.xml.

Usage (from CPlantBox repo root):
    source cpbenv/bin/activate
    python3 dart/coupling/tests/baselines/tune_cultivar_height_factor.py
"""
from __future__ import annotations

import re
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

from dart.coupling.growth.grow import grow_plant  # noqa: E402

XML = REPO_ROOT / "dart" / "coupling" / "data" / "maize_calibrated.xml"
SIM_DAYS = 130
N_SEEDS = 20

# Plan §D4 baseline: H=1.32, dev=0.07. Sweep around it lightly.
COMBOS = [
    (1.32, 0.00),
    (1.32, 0.07),
    (1.30, 0.07),
    (1.34, 0.07),
    (1.36, 0.07),
    (1.32, 0.10),
]

TARGET_MEAN = 235.0
TARGET_SD = 18.0


def _inject_h(xml_text: str, h: float, dev: float) -> str:
    """Insert <parameter name="cultivar_height_factor" .../> into mainstem
    subType=1 block immediately after the opening <stem ...> tag."""
    m = re.search(r'(<stem[^>]*subType="1"[^>]*>)', xml_text)
    if not m:
        raise RuntimeError("mainstem subType=1 block not found in XML")
    inj = f'\n        <parameter name="cultivar_height_factor" value="{h}" dev="{dev}"/>'
    return xml_text[: m.end()] + inj + xml_text[m.end():]


def _stats(values: list[float]) -> tuple[float, float]:
    n = len(values)
    mu = sum(values) / n
    var = sum((v - mu) ** 2 for v in values) / max(n - 1, 1)
    return mu, var ** 0.5


def main() -> int:
    base_text = XML.read_text()
    print(f"=== S4 cultivar_height_factor sweep ({N_SEEDS} seeds × {len(COMBOS)} combos) ===")
    print(f"target: mean ms_len ≈ {TARGET_MEAN} cm, σ ≈ {TARGET_SD} cm")
    print(f"\n{'H':>5} {'dev':>5} {'mean(ms)':>9} {'σ(ms)':>7} {'mean(z)':>8} {'σ(z)':>7}  status")

    best = None
    for h, dev in COMBOS:
        text = _inject_h(base_text, h, dev)
        tmp = Path(tempfile.mkdtemp()) / "maize_tuned.xml"
        tmp.write_text(text)

        ms_lens: list[float] = []
        zs: list[float] = []
        for seed in range(1, N_SEEDS + 1):
            plant = grow_plant(
                xml_path=str(tmp),
                simulation_time=SIM_DAYS,
                seed=seed,
                enable_photosynthesis=True,
            )
            ms = next(
                o for o in plant.getOrgans(-1, True)
                if int(o.organType()) == 3 and int(o.getParameter("subType")) == 1
            )
            ms_lens.append(float(ms.getLength(True)))
            zs.append(max(float(n.z) for n in plant.getNodes()))

        mu_ms, sd_ms = _stats(ms_lens)
        mu_z, sd_z = _stats(zs)
        # Fitness: how close to target mean and target sd
        fit = abs(mu_ms - TARGET_MEAN) + 2.0 * abs(sd_ms - TARGET_SD)
        flag = ""
        if 230.0 <= mu_ms <= 245.0 and 13.0 <= sd_ms <= 23.0:
            flag = "G4+G5 PASS"
        elif 230.0 <= mu_ms <= 245.0:
            flag = "G4 only"
        else:
            flag = "out-of-band"
        print(
            f"{h:>5.2f} {dev:>5.2f} {mu_ms:>9.2f} {sd_ms:>7.2f} {mu_z:>8.2f} {sd_z:>7.2f}  {flag}"
        )
        if best is None or fit < best[0]:
            best = (fit, h, dev, mu_ms, sd_ms)

    if best is not None:
        print(f"\nbest combo: H={best[1]:.2f} dev={best[2]:.2f}  "
              f"mean(ms)={best[3]:.2f} σ(ms)={best[4]:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
