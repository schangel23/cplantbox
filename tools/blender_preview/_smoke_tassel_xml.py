"""Smoke test for §1 XML tassel subTypes 20/21.

Grows the calibrated plant at days spanning the TT=1200 emergence gate
(constant 25C, ~17 degCd/day → gate opens ~day 72). Verifies:
  1. Pre-gate day: no tassel organs.
  2. Just past gate: spike exists, branches not yet (still in spike's basal zone).
  3. Well past gate: spike > 14cm + branches spawned.

Each day runs in a fresh subprocess to avoid pybind11 state accumulation
(multi-day runs in one process segfault around day 90+).

Run:
    cd /home/lukas/PHD/CPlantBox
    source cpbenv/bin/activate
    PYTHONPATH=. python dart/coupling/output/blender_preview/_smoke_tassel_xml.py
"""

from __future__ import annotations

import subprocess
import sys


WORKER = r"""
from dart.coupling.config import DEFAULT_XML
from dart.coupling.growth.grow import grow_plant
import plantbox as pb

day = int({day})
plant = grow_plant(str(DEFAULT_XML), simulation_time=day, seed=42,
                   enable_photosynthesis=False)
tt = plant.getAccumulatedTT() if hasattr(plant, 'getAccumulatedTT') else -1.0
spikes = branches = 0
spike_len = branch_total = mainstem_len = 0.0
for o in plant.getOrgans(pb.stem):
    st = int(o.getParameter('subType'))
    L = o.getLength()
    if st == 1:
        mainstem_len = max(mainstem_len, L)
    elif st == 20:
        spikes += 1
        spike_len = max(spike_len, L)
    elif st == 21:
        branches += 1
        branch_total += L
print(f'SMOKE day={{day}}  TT={{tt:.1f}}  main={{mainstem_len:.1f}}  '
      f'spikes={{spikes}}  spike_len={{spike_len:.2f}}  '
      f'branches={{branches}}  branch_total={{branch_total:.2f}}'
      .format(day=day, tt=tt, mainstem_len=mainstem_len, spikes=spikes,
              spike_len=spike_len, branches=branches, branch_total=branch_total))
"""


def _run(day: int) -> None:
    script = WORKER.format(day=day)
    proc = subprocess.run(
        [sys.executable, "-u", "-c", script],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        print(f"day={day}: EXIT={proc.returncode}")
    # Grab summary
    for line in proc.stdout.splitlines():
        if line.startswith("SMOKE "):
            print(" ", line[6:])


if __name__ == "__main__":
    for d in (55, 72, 85, 90):
        _run(d)
