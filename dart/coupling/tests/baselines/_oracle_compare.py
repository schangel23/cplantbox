"""Shared per-organ snapshot + oracle comparator for FA reproduction gates.

Extracted from ``run_g3_with_carbon_parity.py`` so the §G3 (S5 synthetic
supply) and §G6 (PM+DuMux real substep dispatch) gates share one
comparator. Parametric tolerances make it reusable for shorter
intermediate horizons (e.g. day-60 local smoke).

The oracle JSON schema is what ``capture_oracle_fa_no_carbon_day130.py``
writes: ``{"organs": {"<id>": {"organ_type", "subType",
"realised_length", "insertion_z", ...}}}``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple


def per_organ_snapshot(plant) -> dict:
    """Snapshot every spawned organ (including pre-emergence len=0 leaves)
    to a JSON-serialisable dict keyed by organ id.

    Mirrors ``capture_oracle_fa_no_carbon_day130.per_organ_snapshot`` so the
    snapshot dict aligns key-for-key with the oracle JSON.
    """
    out = {}
    for o in plant.getOrgans(-1, True):
        nodes = list(o.getNodes())
        out[str(o.getId())] = {
            "organ_type": int(o.organType()),
            "subType": int(o.getParameter("subType")),
            "realised_length": round(float(o.getLength()), 6),
            "insertion_z": round(float(nodes[0].z) if nodes else 0.0, 6),
            "dl_backlog": round(float(getattr(o, "dl_backlog", 0.0)), 6),
        }
    return out


def _by_subtype(snapshot):
    """Collapse organ-id-keyed snapshot to subType-keyed: keep the longest
    organ per subType. Mirror of run_g3_with_carbon_parity._by_subtype."""
    out = {}
    for v in snapshot.values():
        if v["organ_type"] != 4:
            continue
        st = v["subType"]
        if st not in out or v["realised_length"] > out[st]["realised_length"]:
            out[st] = v
    return out


def compare_against_oracle(
    snap: dict,
    oracle_path: Path,
    *,
    tol_leaf_pct: float = 1.0,
    tol_mainstem_cm: float = 0.5,
    skip_leaves_shorter_than_cm: float = 0.0,
) -> Tuple[bool, list[str]]:
    """Compare a per-organ snapshot against a captured FA-no-carbon oracle.

    Args:
        snap: Output of ``per_organ_snapshot``.
        oracle_path: Path to oracle JSON
            (``capture_oracle_fa_no_carbon_day130.py`` output).
        tol_leaf_pct: Per-leaf realised-length tolerance [%].
            Default 1.0 (§G3 contract); §G6 spec passes 2.0.
        tol_mainstem_cm: Mainstem realised-length tolerance [cm].
            Default 0.5 (§G3 + §G6 share this threshold).
        skip_leaves_shorter_than_cm: Filter out oracle leaves below this
            cutoff. Set > 0 for intermediate horizons (day-60, day-80)
            where late-emerging leaves carry tiny lengths and high
            relative noise; keep 0.0 for the day-130 full-canopy gate.

    Returns:
        (ok, lines) — ``ok`` is True if every metric passed; ``lines``
        is the printable per-organ comparison + a trailing summary.
    """
    with oracle_path.open() as f:
        oracle = json.load(f)

    failures: list[str] = []
    notes: list[str] = []

    # Mainstem realised length (organ_type=3, subType=1).
    oracle_mainstem = next(
        (v for v in oracle["organs"].values()
         if v["organ_type"] == 3 and v["subType"] == 1),
        None,
    )
    snap_mainstem = next(
        (v for v in snap.values()
         if v["organ_type"] == 3 and v["subType"] == 1),
        None,
    )
    if oracle_mainstem and snap_mainstem:
        delta = snap_mainstem["realised_length"] - oracle_mainstem["realised_length"]
        line = (f"mainstem realised: oracle={oracle_mainstem['realised_length']:.4f} cm, "
                f"with-carbon={snap_mainstem['realised_length']:.4f} cm, "
                f"Δ={delta:+.4f} cm")
        notes.append(line)
        if abs(delta) > tol_mainstem_cm:
            failures.append(f"  FAIL: |mainstem delta| {abs(delta):.4f} > {tol_mainstem_cm} cm")

    # Per-leaf (subType) realised-length drift.
    oracle_by_st = _by_subtype(oracle["organs"])
    snap_by_st = _by_subtype(snap)
    n_leaf_pass = 0
    n_leaf_fail = 0
    n_leaf_skip = 0
    for st, ov in sorted(oracle_by_st.items()):
        ol = ov["realised_length"]
        if ol <= 0:
            continue
        if ol < skip_leaves_shorter_than_cm:
            n_leaf_skip += 1
            continue
        sv = snap_by_st.get(st)
        if sv is None:
            failures.append(
                f"  FAIL: leaf subType={st} missing from with-carbon snapshot "
                f"(oracle had length {ol:.2f} cm)")
            n_leaf_fail += 1
            continue
        sl = sv["realised_length"]
        pct = 100.0 * abs(sl - ol) / ol
        line = (f"  leaf st={st}: oracle={ol:.4f} cm, with-carbon={sl:.4f} cm, "
                f"drift={pct:.2f}%")
        if pct > tol_leaf_pct:
            failures.append(line + f" > {tol_leaf_pct}%  FAIL")
            n_leaf_fail += 1
        else:
            notes.append(line)
            n_leaf_pass += 1

    summary = (f"per-leaf parity: {n_leaf_pass} pass, {n_leaf_fail} fail "
               f"(tol={tol_leaf_pct}%, mainstem tol={tol_mainstem_cm} cm")
    if skip_leaves_shorter_than_cm > 0:
        summary += f", skipped {n_leaf_skip} leaves shorter than {skip_leaves_shorter_than_cm} cm"
    summary += ")"
    notes.append(summary)

    return (len(failures) == 0), notes + failures
