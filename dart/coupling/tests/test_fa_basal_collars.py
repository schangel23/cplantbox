#!/usr/bin/env python3
"""S3b.7 — plastochron-driven rank initiation acceptance tests.

Plan: PLAN_FULL_B35_PER_PHYTOMER_BOOKKEEPING_2026-04-23.md §E.b.

Before S3b.7, FA-on maize fired every lateral at once when the mainstem
crossed `p.lb`: all 16 leaves + 1 tassel attached to the same apex node,
so a V3 plant had 5 initiated ranks stacked at a single z. S3b.7 decouples
node creation from leaf emergence by putting initiation on a plant
thermal-time clock (plastochron ≈ 23 °Cd on the Andrieu Tb=9.8 axis) —
each rank gets its own basal_internode_cm-sized internode.

Acceptance gate:
  V3 FA-on (day 33 under Juelich 2024 met): ≥5 distinct leaf-insertion z
    positions with minimum pairwise spacing ≥ 0.3 cm.
  V6 FA-on (day 57): ≥8 distinct leaf-insertion z positions with minimum
    pairwise spacing ≥ 0.3 cm.

Both tests fail under mutations:
  - Revert the S3b.7 branching-zone block to the scalar burst → all laterals
    share the p.lb node → 1 distinct z.
  - Zero basal_internode_cm → ranks spawn at dxMin offsets → collars
    indistinguishable at test tolerance.

Usage:
    cpbenv/bin/python3 -m pytest dart/coupling/tests/test_fa_basal_collars.py -v
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

COUPLING_DIR = Path(__file__).resolve().parent.parent
CPLANTBOX_ROOT = COUPLING_DIR.parent.parent
sys.path.insert(0, str(CPLANTBOX_ROOT))

import plantbox as pb  # noqa: E402

from dart.coupling.carbon.dvs_partitioning import get_daily_met  # noqa: E402
from dart.coupling.growth.grow import setup_successor_where  # noqa: E402


XML_FA = COUPLING_DIR / "data" / "maize_calibrated.xml"
KINETICS_PATH = COUPLING_DIR / "data" / "phase_III_per_rank.json"
SEED = 7
N_RANKS = 16
MIN_SPACING_CM = 0.3
Z_TOLERANCE_CM = 1e-4  # coincidence threshold for "distinct" z values


def _fill_per_rank(tab: dict, default: float) -> list[float]:
    arr = [default] * N_RANKS
    numeric_keys = [int(k) for k in tab.keys() if k.isdigit()]
    for n in range(1, N_RANKS + 1):
        if str(n) in tab:
            arr[n - 1] = float(tab[str(n)])
        else:
            near = min(numeric_keys, key=lambda k: abs(k - n))
            arr[n - 1] = float(tab[str(near)])
    return arr


def _configure_fa_mainstem(plant: "pb.MappedPlant") -> None:
    srp = plant.getOrganRandomParameter(pb.OrganTypes.stem, 1)
    srp.use_fournier_andrieu_kinetics = True
    k = json.loads(KINETICS_PATH.read_text())
    srp.internode_v_n = _fill_per_rank(k["v_n_cm_per_degCd"]["expt_1B_primary"], 0.2)
    srp.internode_D_n = _fill_per_rank(k["D_n_degCd"]["values"], 50.0)
    srp.internode_IL_final = _fill_per_rank(k["IL_final_cross_check_cm"]["values"], 16.0)


def _grow_to_day(day_end: int) -> "pb.MappedPlant":
    plant = pb.MappedPlant(SEED)
    plant.readParameters(str(XML_FA))
    setup_successor_where(plant)
    _configure_fa_mainstem(plant)
    plant.initialize(False)
    met = get_daily_met()
    for day in range(1, day_end + 1):
        m = met.get(day) if met else None
        if m is not None:
            plant.setAirTemperature(float(m["T_mean_C"]))
        plant.simulate(1.0, False)
    return plant


def _collar_zs(plant: "pb.MappedPlant") -> list[float]:
    """Return leaf-insertion z positions (collars) on the mainstem, ascending."""
    stems = [s for s in plant.getOrgans(pb.OrganTypes.stem)
             if int(s.getParameter("subType")) == 1]
    assert len(stems) == 1, f"expected 1 mainstem, got {len(stems)}"
    ms_id = stems[0].getId()
    leaves = [lf for lf in plant.getOrgans(pb.OrganTypes.leaf)
              if lf.getParent() and lf.getParent().getId() == ms_id]
    zs = []
    for lf in leaves:
        nodes = list(lf.getNodes())
        if nodes:
            zs.append(float(nodes[0].z))
    return sorted(zs)


def _distinct_zs(zs: list[float], tol: float = Z_TOLERANCE_CM) -> list[float]:
    """Collapse z values within `tol` cm into a single representative."""
    distinct: list[float] = []
    for z in sorted(zs):
        if not distinct or (z - distinct[-1]) > tol:
            distinct.append(z)
    return distinct


def test_v3_faon_has_distinct_collars():
    """Day 33 (~V3 calendar under Juelich 2024): ≥5 distinct collars, ≥0.3 cm apart."""
    plant = _grow_to_day(33)
    zs = _collar_zs(plant)
    assert len(zs) >= 5, (
        f"expected ≥5 leaves on mainstem at day 33, got {len(zs)}; "
        "plastochron-driven initiation should have created at least 5 ranks by "
        "AndrieuTT ≈ 5*plastochron ≈ 115 °Cd (well below day-33 AndrieuTT ≈ 170)"
    )
    distinct = _distinct_zs(zs[:5])
    assert len(distinct) == 5, (
        f"V3 collars collapse to {len(distinct)} distinct z (of 5 initiated ranks); "
        f"values: {zs[:5]}. Scalar-burst fallback would collapse all 5 to one z; "
        f"basal_internode_cm=0 would make spacings < {Z_TOLERANCE_CM} cm."
    )
    spacings = [distinct[i + 1] - distinct[i] for i in range(len(distinct) - 1)]
    assert min(spacings) >= MIN_SPACING_CM - 1e-9, (
        f"V3 min collar spacing {min(spacings):.4f} cm < {MIN_SPACING_CM} cm; "
        f"spacings: {[round(s, 4) for s in spacings]}"
    )


def test_v6_faon_has_distinct_collars():
    """Day 57 (~V6 calendar): ≥8 distinct collars, ≥0.3 cm apart."""
    plant = _grow_to_day(57)
    zs = _collar_zs(plant)
    assert len(zs) >= 8, (
        f"expected ≥8 leaves on mainstem at day 57, got {len(zs)}; "
        "plastochron should have initiated ≥8 ranks by AndrieuTT ≈ 184 °Cd "
        "(reached early in the calendar; day 57 AndrieuTT ≳ 300)"
    )
    distinct = _distinct_zs(zs[:8])
    assert len(distinct) == 8, (
        f"V6 collars collapse to {len(distinct)} distinct z (of 8 initiated ranks); "
        f"values: {zs[:8]}"
    )
    spacings = [distinct[i + 1] - distinct[i] for i in range(len(distinct) - 1)]
    assert min(spacings) >= MIN_SPACING_CM - 1e-9, (
        f"V6 min collar spacing {min(spacings):.4f} cm < {MIN_SPACING_CM} cm; "
        f"spacings: {[round(s, 4) for s in spacings]}"
    )


def test_initiation_andrieu_tt_recorded():
    """Rank n's initiation_andrieu_tt_per_n[n] ≈ n * plastochron_andrieu ± one-step slack.

    The plastochron clock fires at the end of the simulate(dt) step in which
    plant_andrieu_tt first exceeds n*plastochron. With dt=1 d and AndrieuTT
    accumulating ~10 °Cd/day, the recorded value can exceed n*plastochron by
    up to one day's AndrieuTT increment.
    """
    plant = _grow_to_day(57)
    stems = [s for s in plant.getOrgans(pb.OrganTypes.stem)
             if int(s.getParameter("subType")) == 1]
    ms = stems[0]
    srp = plant.getOrganRandomParameter(pb.OrganTypes.stem, 1)
    plastochron = srp.plastochron_andrieu
    itt = list(ms.initiation_andrieu_tt_per_n)
    spawned = list(ms.lateral_spawned_per_n)
    for n in range(1, min(len(itt), 9)):
        if not ord(spawned[n]):
            continue
        init_tt = itt[n]
        target = n * plastochron
        assert init_tt >= target - 1e-9, (
            f"rank {n} initiated at AndrieuTT={init_tt:.2f} < target {target:.2f}; "
            "plastochron gate should fire only AFTER the birthday crosses"
        )
        # Upper bound: one day's worth of AndrieuTT accumulation past the birthday.
        # Under Juelich 2024 May–June, daily AndrieuTT ≈ 5–12 °Cd.
        assert init_tt - target <= 20.0, (
            f"rank {n} initiated {init_tt - target:.2f} °Cd past its birthday; "
            f"expected ≤ 20 °Cd (one simulate-step slack)"
        )


def test_basal_ranks_pinned_at_basal_internode_cm():
    """S3b.8 acceptance gate: ranks 1-4 stay pinned at basal_internode_cm at day 130.

    Pre-S3b.8, internodalGrowth's equal-share distribution inflated
    basal_zero_ranks to ~p.ln ≈ 10 cm each, masking V-stage anatomy and
    requiring a render-time z-compression shim in cplantbox_adapter.py.
    S3b.8 teaches internodalGrowth to skip basal_zero_ranks, so ranks 1-4
    stay at the basal_internode_cm seed placed by the plastochron loop.

    Mutation that breaks this: remove the `rank_is_basal_zero(phytomerId+1)`
    check in Stem::internodalGrowth → ranks 1-4 grow back up to p.ln ≈ 10 cm.
    """
    plant = _grow_to_day(130)
    ms = [s for s in plant.getOrgans(pb.OrganTypes.stem)
          if int(s.getParameter("subType")) == 1][0]
    srp = plant.getOrganRandomParameter(pb.OrganTypes.stem, 1)
    basal_step = srp.basal_internode_cm
    basal_zero = list(srp.basal_zero_ranks)
    lpn = list(ms.length_per_n)
    cap_cm = 1.5 * basal_step  # plan §F tolerance
    for n in basal_zero:
        assert n < len(lpn), f"length_per_n[{n}] not populated (size={len(lpn)})"
        assert lpn[n] <= cap_cm, (
            f"basal rank {n} length {lpn[n]:.4f} cm exceeds "
            f"{cap_cm:.4f} cm (= 1.5 × basal_internode_cm={basal_step}). "
            "Pre-S3b.8 internodalGrowth inflates ranks 1-4 to ~10 cm "
            "via equal-share distribution; the basal_zero_ranks gate "
            "should keep them pinned at basal_internode_cm."
        )
