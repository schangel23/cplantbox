"""V-stage detector: map calendar days under Juelich met to V-stage.

Uses the collar-release threshold shipped in cplantbox_adapter.py
(blade_maturity >= 0.45 = collared) to count visible collars per day.
Reports the first day each V-stage (V1..V8) is reached and exports
OBJs at those days so we have a phenology-aligned validation set
instead of fixed calendar targets.

Run:
    cd /home/lukas/PHD/CPlantBox
    source cpbenv/bin/activate
    PYTHONPATH=. python dart/coupling/output/blender_preview/_vstage_detector.py
"""
from __future__ import annotations
from pathlib import Path
import sys
import traceback

import plantbox as pb

from dart.coupling.config import DEFAULT_XML
from dart.coupling.geometry import extract_organs_for_lofter, loft_organs
from dart.coupling.growth.grow import setup_successor_where

COLLAR_RELEASE = 0.45   # matches cplantbox_adapter.py collar-threshold gate
COLLAR_THRESHOLD = 0.30
SEED = 7
STAGES = Path(__file__).resolve().parent / "stages"
V_TARGETS = (1, 2, 3, 4, 5, 6)
MAX_DAYS = 130
# Optional extra days to render beyond V-stage hits (mature regression).
EXTRA_DAYS = (55, 130)


def count_stages(plant: "pb.MappedPlant") -> tuple[int, int, int, int]:
    """Return (n_leaves_total, n_collared, n_emerging, n_whorl_tip).

    Emulates the maturity classification the NURBS adapter uses at
    render time.
    """
    n_total = 0; n_coll = 0; n_emrg = 0; n_whrl = 0
    for organ in plant.getOrgans(pb.leaf):
        nodes = organ.getNodes()
        if len(nodes) < 2:
            continue
        lrp = organ.getLeafRandomParameter()
        lmax = max(float(lrp.lmax), 1e-9)
        # current length = cumulative arc length along the leaf's own nodes
        cur = 0.0
        pts = [(float(n.x), float(n.y), float(n.z)) for n in nodes]
        for i in range(1, len(pts)):
            dx = pts[i][0] - pts[i - 1][0]
            dy = pts[i][1] - pts[i - 1][1]
            dz = pts[i][2] - pts[i - 1][2]
            cur += (dx * dx + dy * dy + dz * dz) ** 0.5
        m = min(cur / lmax, 1.0)
        n_total += 1
        if m >= COLLAR_RELEASE:
            n_coll += 1
        elif m >= COLLAR_THRESHOLD:
            n_emrg += 1
        else:
            n_whrl += 1
    return n_total, n_coll, n_emrg, n_whrl


def export_stage(plant, label: str, day: int, tt: float) -> Path:
    organs = extract_organs_for_lofter(plant, skip_roots=True)
    mesh = loft_organs(organs, stem_sides=8, use_nurbs_backend=True)
    out = STAGES / f"vstage_{label}_day{day:03d}.obj"
    out.parent.mkdir(parents=True, exist_ok=True)
    mesh.to_obj(str(out), group_by_organ=True)
    print(f"  -> exported {label} (day {day}, TT={tt:.1f}) to {out}")
    return out


def main() -> None:
    from dart.coupling.carbon.dvs_partitioning import get_daily_met
    daily_met = get_daily_met()
    if daily_met is None:
        raise SystemExit("No daily met CSV found; aborting.")

    plant = pb.MappedPlant()
    plant.readParameters(str(DEFAULT_XML))
    plant.setSeed(SEED)
    setup_successor_where(plant)
    plant.initialize()

    targets_remaining = list(V_TARGETS)
    per_day_log: list[tuple[int, float, int, int, int, int]] = []

    for day in range(1, MAX_DAYS + 1):
        T = float(daily_met.get(day, {}).get("T_mean_C", 25.0))
        plant.setAirTemperature(T)
        plant.simulate(1.0, verbose=False)

        tt = plant.getAccumulatedTT() if hasattr(plant, "getAccumulatedTT") else -1.0
        n_total, n_coll, n_emrg, n_whrl = count_stages(plant)
        per_day_log.append((day, tt, n_total, n_coll, n_emrg, n_whrl))

        # Export at first day each V-stage is reached — Nielsen-compatible
        # form: V-n needs n collared AND at least 2 non-collared (emerging
        # + whorl-tip) above them, matching the agronomic textbook V3 where
        # you see leaves #4 (blade emerged, collar hidden) and #5 (whorl
        # tip). This sidesteps the transient window after leaf N collars
        # but before leaf N+2 initiates.
        while targets_remaining and n_coll >= targets_remaining[0] \
                and (n_emrg + n_whrl) >= 2:
            v = targets_remaining.pop(0)
            print(f"V{v} reached at day {day:3d}  TT={tt:6.1f}  "
                  f"total={n_total} collared={n_coll} emerging={n_emrg} whorl={n_whrl}")
            export_stage(plant, f"V{v}", day, tt)

        if day in EXTRA_DAYS:
            export_stage(plant, f"mature_day{day:03d}", day, tt)

        if not targets_remaining and day >= max(EXTRA_DAYS):
            break

    # Also export a final mature snapshot for mature regression check.
    # Restart to day 55 is unnecessary since we were running day-by-day;
    # we already have the final state at whatever day we stopped.
    final = per_day_log[-1] if per_day_log else (0, 0.0, 0, 0, 0, 0)
    print("\nFinal day -> day {0} TT {1:.1f}  total={2}".format(*final[:3]))

    # Print compact daily table
    print("\n day   TT   total  coll  emrg  whorl")
    for rec in per_day_log:
        print(" {0:3d}  {1:6.1f}   {2:3d}   {3:3d}   {4:3d}   {5:3d}".format(*rec))


if __name__ == "__main__":
    try:
        sys.exit(main() or 0)
    except BaseException as e:
        print(f"FAILED: {e!r}")
        traceback.print_exc()
        sys.exit(1)
