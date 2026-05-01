"""Refit Vidal et al. 2021 per-rank 3-phase blade-extension model.

Source:
    Vidal T., Andrieu B., et al. (2021).
    "Two maize cultivars of contrasting leaf size show different leaf
     elongation rates with identical patterns of extension dynamics
     and coordination."
    AoB PLANTS 13(1), plaa072.  doi:10.1093/aobpla/plaa072
    PMC7877697

Input:
    SupData1 = ``plaa072_suppl_supplementary_data_1.xlsx`` (sheet
    ``1.blade_des``) — destructive blade lengths Bn at TTc °Cd, M40 + M52,
    5 plants per cultivar, ~25 sampling dates, ranks 1..20.

Model (Vidal Eq. 2 + 2b, C¹ continuity at exp/linear transition):
    L(t) = L_min · exp(R1·(t − T0))            t ∈ [T0, T1]
         = L1 + R2·(t − T1)                    t ∈ [T1, T2]
         = L_fin = L1 + R2·D_lin               t ≥ T2
    where  L1 = L_min · exp(R1·lag),  lag = T1 − T0,  D_lin = T2 − T1,
           R2 = R1 · L1   (Eq. 2b slope continuity).

The free parameters are (T0, R1, T1, T2). L_min is fixed at 0.025 cm to
match ``LeafRandomParameter.L_min`` and the bake script convention.

Per-rank fits are computed independently for cultivars M40 and M52, then
cultivar-averaged for the calibration JSON. Parameter standard errors come
from the curve_fit covariance matrix.

CLI::
    cpbenv/bin/python -m dart.coupling.scripts.vidal_blade_refit \\
        --xlsx /path/to/plaa072_suppl_supplementary_data_1.xlsx \\
        --json-out vidal_refit.json

Reference checksum (SupData1, downloaded 2026-05-01):
    sha256: c37d6f21024887fc4692bfae308bca981c0996d1ea51aadd608229cc7b769e90
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit

L_MIN = 0.025  # cm — must match LeafRandomParameter.L_min and bake L_min_default


def vidal_3phase(t, T0, R1, T1, T2, L_min: float = L_MIN):
    """Vidal 2021 Eq. 2 + 2b. t in °Cd. Free parameters: T0, R1, T1, T2."""
    t = np.asarray(t, dtype=float)
    L1 = L_min * np.exp(R1 * (T1 - T0))
    R2 = R1 * L1
    L_fin = L1 + R2 * (T2 - T1)
    out = np.full_like(t, L_min, dtype=float)
    expE = (t >= T0) & (t < T1)
    linL = (t >= T1) & (t < T2)
    plat = t >= T2
    out[expE] = L_min * np.exp(np.clip(R1 * (t[expE] - T0), -20, 20))
    out[linL] = L1 + R2 * (t[linL] - T1)
    out[plat] = L_fin
    return out


def load_blade_des(xlsx: Path) -> dict:
    import openpyxl
    wb = openpyxl.load_workbook(xlsx, data_only=True, read_only=True)
    ws = wb["1.blade_des"]
    rows = list(ws.iter_rows(values_only=True))
    wb.close()
    header = rows[0]
    idx_ttc = header.index("TTc")
    idx_cult = header.index("cultivar")
    rank_cols = {}
    for col, name in enumerate(header):
        if isinstance(name, str) and name.startswith("B"):
            try:
                rank_cols[int(name[1:])] = col
            except ValueError:
                pass
    out: dict[tuple[str, int], list[tuple[float, float]]] = {}
    for r in rows[1:]:
        if r[idx_ttc] is None:
            continue
        ttc = float(r[idx_ttc])  # type: ignore[arg-type]
        cult = r[idx_cult]
        if not isinstance(cult, str):
            continue
        for rank, col in rank_cols.items():
            v = r[col]
            if v is None:
                continue
            try:
                vf = float(v)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                continue
            if vf <= 0:
                continue
            out.setdefault((cult, rank), []).append((ttc, vf))
    return {k: (np.array([x[0] for x in vs]), np.array([x[1] for x in vs]))
            for k, vs in out.items()}


def fit_one(ttc: np.ndarray, length: np.ndarray) -> dict | None:
    """Fit one (cultivar, rank). Returns None if too few points or trivially small."""
    if len(ttc) < 6:
        return None
    order = np.argsort(ttc)
    ttc = ttc[order]
    length = length[order]
    if float(length.max()) <= L_MIN * 5:
        return None
    t_lo, t_hi = float(ttc.min()), float(ttc.max())
    span = max(t_hi - t_lo, 50.0)
    p0 = [t_lo - 50.0, 0.04, t_lo + 0.3 * span, t_lo + 0.7 * span]
    bounds_lo = [-300.0, 1e-3, t_lo - 100.0, t_lo - 50.0]
    bounds_hi = [t_hi + 50.0, 0.20, t_hi + 200.0, t_hi + 400.0]
    try:
        popt, pcov = curve_fit(
            vidal_3phase, ttc, length,
            p0=p0, bounds=(bounds_lo, bounds_hi), maxfev=8000,
        )
    except Exception as exc:
        return {"error": str(exc), "n_obs": len(ttc)}
    T0, R1, T1, T2 = popt
    perr = np.sqrt(np.diag(pcov)) if np.all(np.isfinite(pcov)) else np.full(4, np.nan)
    L1 = L_MIN * math.exp(R1 * (T1 - T0))
    R2 = R1 * L1
    L_fin = L1 + R2 * (T2 - T1)
    pred = vidal_3phase(ttc, *popt)
    rmse = float(np.sqrt(np.mean((pred - length) ** 2)))
    return {
        "n_obs": int(len(ttc)),
        "T0": float(T0), "T0_se": float(perr[0]),
        "R1": float(R1), "R1_se": float(perr[1]),
        "T1": float(T1), "T1_se": float(perr[2]),
        "T2": float(T2), "T2_se": float(perr[3]),
        "lag": float(T1 - T0),
        "D_lin": float(T2 - T1),
        "L1": float(L1),
        "R2": float(R2),
        "L_fin": float(L_fin),
        "rmse_cm": rmse,
    }


def fit_all(data: dict) -> dict:
    cultivars = sorted({c for c, _ in data})
    ranks = sorted({n for _, n in data})
    out: dict[str, dict] = {c: {} for c in cultivars}
    for c in cultivars:
        for n in ranks:
            obs = data.get((c, n))
            if obs is None:
                continue
            f = fit_one(obs[0], obs[1])
            if f is not None:
                out[c][n] = f
    out["averaged"] = {}
    for n in ranks:
        a = out.get("M40", {}).get(n)
        b = out.get("M52", {}).get(n)
        if not a or "error" in a or not b or "error" in b:
            continue
        avg = {
            "R1": 0.5 * (a["R1"] + b["R1"]),
            "lag": 0.5 * (a["lag"] + b["lag"]),
            "D_lin": 0.5 * (a["D_lin"] + b["D_lin"]),
            "R2": 0.5 * (a["R2"] + b["R2"]),
            "L_fin": 0.5 * (a["L_fin"] + b["L_fin"]),
            "T0": 0.5 * (a["T0"] + b["T0"]),
            # propagate worst-case SE percentage from the two cultivars
            "R1_se_pct": max(
                100 * a["R1_se"] / max(a["R1"], 1e-12),
                100 * b["R1_se"] / max(b["R1"], 1e-12),
            ),
            "D_lin_se_pct": max(
                100 * a["T2_se"] / max(a["D_lin"], 1e-12),
                100 * b["T2_se"] / max(b["D_lin"], 1e-12),
            ),
            "rmse_cm_max": max(a["rmse_cm"], b["rmse_cm"]),
        }
        out["averaged"][n] = avg
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--xlsx", required=True, type=Path,
                    help="Path to plaa072_suppl_supplementary_data_1.xlsx")
    ap.add_argument("--json-out", type=Path, default=None,
                    help="Optional path for machine-readable JSON output")
    args = ap.parse_args(argv)

    data = load_blade_des(args.xlsx)
    fits = fit_all(data)

    # Pretty print
    print(f"Cultivars: {sorted(fits.keys() - {'averaged'})}")
    for c in sorted(fits.keys() - {"averaged"}):
        print(f"\n=== {c} per-rank fits ===")
        print(f"{'rk':>3} {'n':>3} {'T0':>7} {'R1':>7} {'lag':>6} {'D_lin':>6} "
              f"{'L1':>6} {'R2':>6} {'L_fin':>7} {'RMSE':>5} {'R1±SE%':>8}")
        for n in sorted(fits[c]):
            f = fits[c][n]
            if "error" in f:
                continue
            r1pct = 100 * f["R1_se"] / max(f["R1"], 1e-12)
            print(f"{n:>3} {f['n_obs']:>3} {f['T0']:>7.1f} {f['R1']:>7.4f} "
                  f"{f['lag']:>6.1f} {f['D_lin']:>6.1f} {f['L1']:>6.2f} "
                  f"{f['R2']:>6.3f} {f['L_fin']:>7.2f} {f['rmse_cm']:>5.2f} "
                  f"{r1pct:>7.1f}%")
    print("\n=== M40+M52 averaged ===")
    print(f"{'rk':>3} {'R1':>7} {'lag':>6} {'D_lin':>6} {'R2':>6} {'L_fin':>7} {'R1_SE%':>7}")
    for n in sorted(fits["averaged"]):
        a = fits["averaged"][n]
        print(f"{n:>3} {a['R1']:>7.4f} {a['lag']:>6.1f} {a['D_lin']:>6.1f} "
              f"{a['R2']:>6.3f} {a['L_fin']:>7.2f} {a['R1_se_pct']:>6.1f}%")

    if args.json_out:
        args.json_out.write_text(json.dumps(fits, indent=2))
        print(f"\nWrote {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
