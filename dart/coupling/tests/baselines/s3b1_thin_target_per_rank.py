#!/usr/bin/env python3
"""S3b.1 pre-work — per-rank FA target-length trajectory under thin-B.3.5.

Plan: PLAN_FULL_B35_PER_PHYTOMER_BOOKKEEPING_2026-04-23.md §Pre-work (S3b.1).

Captures, per simulated day, the *kinetic target* length of each mainstem
internode under the shipped thin-B.3.5 Fournier-Andrieu build:

    target_per_rank[n-1] = stem.calcLengthPerPhytomer(n)    for n = 1..MAX_RANK

These are the lengths FA kinetics *prescribe*, independent of how they land
geometrically on the segment topology. Under thin-B.3.5 `Stem::simulate` sums
them into a single apex target (`targetlength = max(p.lb + Σ IL, calcLength(age))`)
and the basal/branching/apical allocation loop distributes nodes at the apex,
so per-rank *achieved* lengths are NOT meaningful under thin. The per-rank
curves captured here are the reference against which S3b.5 (per-phytomer
achieved lengths) is validated.

Output:
  * s3b1_thin_target_per_rank.json  — daily per-rank targets + leaf emergence schedule
  * s3b1_thin_target_per_rank.png   — per-rank H_n(TT_A) + cumulative Σ IL overlay

Run (from /home/lukas/PHD/CPlantBox):
    cpbenv/bin/python3 dart/coupling/tests/baselines/s3b1_thin_target_per_rank.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

BASELINE_DIR = Path(__file__).resolve().parent
COUPLING_DIR = BASELINE_DIR.parent.parent
CPLANTBOX_ROOT = COUPLING_DIR.parent.parent
sys.path.insert(0, str(CPLANTBOX_ROOT))

import plantbox as pb  # noqa: E402

from dart.coupling.carbon.dvs_partitioning import get_daily_met  # noqa: E402
from dart.coupling.growth.grow import setup_successor_where  # noqa: E402

XML_PATH = COUPLING_DIR / "data" / "maize_calibrated.xml"
KINETICS_PATH = COUPLING_DIR / "data" / "phase_III_per_rank.json"
SEED = 7
MAX_DAYS = 130
MAX_RANK = 16


def load_fa_kinetics(n_ranks: int):
    data = json.loads(KINETICS_PATH.read_text())
    v_table = data["v_n_cm_per_degCd"]["expt_1B_primary"]
    d_table = data["D_n_degCd"]["values"]
    il_table = data["IL_final_cross_check_cm"]["values"]
    v_n, D_n, IL = [0.0] * n_ranks, [0.0] * n_ranks, [0.0] * n_ranks
    for n in range(1, n_ranks + 1):
        k = str(n)
        v_n[n - 1] = float(v_table.get(k, v_table.get("15", 0.18)))
        D_n[n - 1] = float(d_table.get(k, d_table.get("15", 79)))
        IL[n - 1] = float(il_table.get(k, il_table.get("15", 16)))
    return v_n, D_n, IL


def enable_fa(plant):
    srp = plant.getOrganRandomParameter(pb.OrganTypes.stem, 1)
    srp.use_fournier_andrieu_kinetics = True
    v_n, D_n, IL = load_fa_kinetics(MAX_RANK)
    srp.internode_v_n = v_n
    srp.internode_D_n = D_n
    srp.internode_IL_final = IL
    return v_n, D_n, IL


def extract_mainstem(plant):
    for o in plant.getOrgans():
        if o.organType() == pb.OrganTypes.stem and int(o.getParameter("subType")) == 1:
            return o
    return None


def extract_mainstem_leaf_emergences(plant):
    mainstem = extract_mainstem(plant)
    if mainstem is None:
        return []
    mainstem_id = mainstem.getId()
    mainstem_leaves = [lf for lf in plant.getOrgans()
                       if lf.organType() == pb.OrganTypes.leaf
                       and lf.getParent() is not None
                       and lf.getParent().getId() == mainstem_id]
    mainstem_leaves.sort(key=lambda lf: lf.parentNI)
    out = []
    for rank, lf in enumerate(mainstem_leaves, start=1):
        em = float(lf.getEmergenceAndrieuTT()) if hasattr(lf, "getEmergenceAndrieuTT") else -1.0
        out.append({
            "rank": rank,
            "subType": int(lf.getParameter("subType")),
            "parentNI": int(lf.parentNI),
            "emergence_andrieu_tt": em,
        })
    return out


def main():
    print(f"S3b.1 per-rank FA target capture under thin-B.3.5")
    print(f"  seed={SEED}, XML={XML_PATH.name}, {MAX_DAYS} d, Juelich 2024 met, ranks 1..{MAX_RANK}")
    plant = pb.MappedPlant(SEED)
    plant.readParameters(str(XML_PATH))
    plant.setSeed(SEED)
    setup_successor_where(plant)
    v_n, D_n, IL_final = enable_fa(plant)
    plant.initialize()

    # One-shot: stem.p.lb (the bootstrap constant added to Σ IL in thin-B.3.5)
    mainstem = extract_mainstem(plant)
    p_lb = float(mainstem.getParameter("lb")) if mainstem is not None else 0.0
    print(f"  p.lb = {p_lb:.3f} cm (added to Σ IL in thin targetlength)")

    met = get_daily_met(daily_met=None)
    trajectory = []
    for day in range(1, MAX_DAYS + 1):
        T = float(met.get(day, {}).get("T_mean_C", 25.0)) if met else 25.0
        plant.setAirTemperature(T)
        try:
            plant.simulate(1.0, False)
        except (IndexError, RuntimeError) as e:
            print(f"  simulate() error at day {day}: {e}")
            break
        tt_tb8 = plant.getAccumulatedTT() if hasattr(plant, "getAccumulatedTT") else -1.0
        tt_a = plant.getAccumulatedAndrieuTT() if hasattr(plant, "getAccumulatedAndrieuTT") else -1.0
        stem = extract_mainstem(plant)
        if stem is None:
            continue

        # Per-rank kinetic target. calcLengthPerPhytomer returns 0 for basal-zero
        # ranks (1..4) and for ranks whose leaf primordium hasn't initiated yet.
        target_per_rank = [float(stem.calcLengthPerPhytomer(n)) for n in range(1, MAX_RANK + 1)]
        sigma_IL = sum(target_per_rank)
        achieved_length = float(stem.getLength())
        top_z = max(float(nd.z) for nd in stem.getNodes()) if stem.getNodes() else 0.0

        trajectory.append({
            "day": day,
            "T_mean_C": T,
            "tt_tb8": tt_tb8,
            "tt_andrieu": tt_a,
            "target_per_rank_cm": target_per_rank,
            "sigma_IL_cm": sigma_IL,
            "H_thin_cm": p_lb + sigma_IL,
            "achieved_length_cm": achieved_length,
            "top_z_cm": top_z,
        })
        if day % 10 == 0:
            nonzero = sum(1 for x in target_per_rank if x > 0.0)
            print(f"  d={day:3d} T={T:5.1f}°C  TT_A={tt_a:6.1f}  "
                  f"ΣIL={sigma_IL:6.2f} cm  H_thin={p_lb + sigma_IL:6.2f} cm  "
                  f"achieved={achieved_length:6.2f} cm  ranks>0: {nonzero}/{MAX_RANK}")

    leaf_emergences = extract_mainstem_leaf_emergences(plant)

    out_json = BASELINE_DIR / "s3b1_thin_target_per_rank.json"
    out_json.write_text(json.dumps({
        "seed": SEED,
        "xml": XML_PATH.name,
        "max_days": MAX_DAYS,
        "n_ranks": MAX_RANK,
        "p_lb_cm": p_lb,
        "fa_kinetics": {
            "v_n_cm_per_degCd": v_n,
            "D_n_degCd": D_n,
            "IL_final_cm": IL_final,
        },
        "trajectory": trajectory,
        "leaf_emergences_final": leaf_emergences,
    }, indent=2, default=float))
    print(f"\nSaved {len(trajectory)} daily samples to {out_json}")

    # Render PNG: per-rank H_n(TT_A) curves + cumulative Σ IL overlay.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib unavailable — skipping PNG render")
        return 0

    tt_series = [d["tt_andrieu"] for d in trajectory]
    per_rank_series = [[d["target_per_rank_cm"][n - 1] for d in trajectory] for n in range(1, MAX_RANK + 1)]
    sigma_series = [d["sigma_IL_cm"] for d in trajectory]

    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    cmap = plt.get_cmap("viridis")
    colors = [cmap(i / max(1, MAX_RANK - 1)) for i in range(MAX_RANK)]
    for n in range(1, MAX_RANK + 1):
        ax1.plot(tt_series, per_rank_series[n - 1], color=colors[n - 1],
                 lw=1.2, label=f"rank {n}" if n % 2 == 1 else None)
    ax1.set_xlabel("TT on Andrieu axis (Tb = 9.8 °C)  [°Cd]")
    ax1.set_ylabel("Per-rank target internode length  IL_n  [cm]")
    ax1.set_title("FA kinetic target per rank (thin-B.3.5)\n"
                  "ranks 1–4 basal-zero; ranks 5–16 Phase I–IV curves")
    ax1.legend(loc="upper left", fontsize=8, ncol=2)
    ax1.grid(alpha=0.3)

    ax2.plot(tt_series, sigma_series, "k-", lw=2, label="Σ IL_n (kinetic target)")
    ax2.plot(tt_series, [d["H_thin_cm"] for d in trajectory], "b--", lw=1.2,
             label=f"p.lb + Σ IL  ({p_lb:.1f} cm + …)")
    ax2.plot(tt_series, [d["achieved_length_cm"] for d in trajectory], "r:", lw=1.2,
             label="achieved mainstem length")
    ax2.set_xlabel("TT on Andrieu axis (Tb = 9.8 °C)  [°Cd]")
    ax2.set_ylabel("Cumulative mainstem length  [cm]")
    ax2.set_title("Cumulative target vs achieved  (thin-B.3.5)\n"
                  "divergence during bootstrap regime is expected")
    ax2.legend(loc="lower right", fontsize=9)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    out_png = BASELINE_DIR / "s3b1_thin_target_per_rank.png"
    plt.savefig(out_png, dpi=130)
    print(f"Saved plot to {out_png}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
