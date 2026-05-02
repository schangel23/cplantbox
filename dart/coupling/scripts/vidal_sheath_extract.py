"""Extract per-rank mature sheath length from Vidal 2021 SupData1 sheet 2.sheath_des.

Reads /home/lukas/Downloads/plaa072_suppl_supplementary_data_1.xlsx,
averages M40+M52 cultivars, takes mean of top 10% measurements per rank
(mature plateau), emits JSON consumed by the lofter.

Output: dart/coupling/data/vidal_per_rank_sheath_cm.json

Usage:
    cd /home/lukas/PHD/CPlantBox
    source cpbenv/bin/activate
    python dart/coupling/scripts/vidal_sheath_extract.py
"""
from __future__ import annotations

import hashlib
import json
import statistics
from pathlib import Path

import openpyxl

SOURCE_XLSX = Path("/home/lukas/Downloads/plaa072_suppl_supplementary_data_1.xlsx")
OUT_JSON = Path(__file__).resolve().parents[1] / "data" / "vidal_per_rank_sheath_cm.json"

CULTIVARS = ["M40", "M52"]
TOP_FRAC = 0.10  # mean of top 10% = mature plateau


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    if not SOURCE_XLSX.exists():
        raise FileNotFoundError(f"Vidal SupData1 xlsx not found at {SOURCE_XLSX}")

    wb = openpyxl.load_workbook(SOURCE_XLSX, read_only=True, data_only=True)
    ws = wb["2.sheath_des"]
    rows = list(ws.iter_rows(values_only=True))
    header = rows[0]

    rank_cols = [(i, c) for i, c in enumerate(header) if isinstance(c, str) and c.startswith("S")]
    rank_names = [c for _, c in rank_cols]

    data = {}  # cultivar -> rank_name -> list of values
    for r in rows[1:]:
        cv = r[2]
        if cv not in CULTIVARS:
            continue
        for i, name in rank_cols:
            v = r[i]
            if v is None:
                continue
            try:
                v = float(v)
            except (TypeError, ValueError):
                continue
            if v <= 0:
                continue
            data.setdefault(cv, {}).setdefault(name, []).append(v)

    per_rank = []
    for name in rank_names:
        rank_idx = int(name[1:])  # 'S0' -> 0
        cultivar_means = {}
        n_obs = {}
        for cv in CULTIVARS:
            vals = sorted(data.get(cv, {}).get(name, []))
            n_obs[cv] = len(vals)
            if not vals:
                cultivar_means[cv] = None
                continue
            topn = max(1, int(round(len(vals) * TOP_FRAC)))
            cultivar_means[cv] = statistics.mean(vals[-topn:])

        m40 = cultivar_means.get("M40")
        m52 = cultivar_means.get("M52")
        if m40 is None and m52 is None:
            continue
        if m40 is not None and m52 is not None:
            avg = (m40 + m52) / 2.0
        else:
            avg = m40 if m40 is not None else m52

        per_rank.append({
            "rank": rank_idx,
            "sheath_length_cm": round(avg, 3),
            "M40_top10pct_mean_cm": round(m40, 3) if m40 is not None else None,
            "M52_top10pct_mean_cm": round(m52, 3) if m52 is not None else None,
            "n_obs_M40": n_obs["M40"],
            "n_obs_M52": n_obs["M52"],
        })

    out = {
        "_meta": {
            "source": str(SOURCE_XLSX),
            "source_sha256": file_sha256(SOURCE_XLSX),
            "sheet": "2.sheath_des",
            "cultivars": CULTIVARS,
            "averaging": f"top {int(TOP_FRAC*100)}% per cultivar, then M40+M52 mean",
            "extracted_by": "dart/coupling/scripts/vidal_sheath_extract.py",
            "citation": "Vidal T., Andrieu B. et al. 2021 'Two maize cultivars of contrasting leaf size...' AoB PLANTS 13(1) plaa072",
            "note": "rank index is the 1-indexed leaf position (S0=basal/primordial; S1 maps to maize_calibrated.xml leaf subType=2 internal position 0)",
        },
        "per_rank": per_rank,
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {OUT_JSON}")
    print(f"  {len(per_rank)} ranks, S{per_rank[0]['rank']}..S{per_rank[-1]['rank']}")
    for e in per_rank:
        print(f"  S{e['rank']:>2}: {e['sheath_length_cm']:>5.1f} cm  "
              f"(M40 n={e['n_obs_M40']}, M52 n={e['n_obs_M52']})")


if __name__ == "__main__":
    main()
