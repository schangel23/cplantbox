"""Audit maize phloem Across_st values against anatomy-derived areas.

No simulation and no parameter-file edits.  The maize JSON stores Across_st as
total sieve-tube cross-sectional area [cm2] per organ subtype:

    N_bundles * numSE * pi * r_SE**2

The source anatomy constants are read from maize_phloem_2026.py so the audit
stays tied to the same primary inputs used by the maize parameter generator.
"""

import ast
import json
import math
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
MAIZE_PY = REPO_ROOT / "modelparameter/functional/plant_sucrose/maize_phloem_2026.py"
MAIZE_JSON = REPO_ROOT / "dart/coupling/data/phloem_parameters_maize2026.json"
DRIFT_OVER = 2.0
DRIFT_UNDER = 0.5


def _eval_expr(node, vals):
    """Evaluate numeric/list/dict constants and simple arithmetic from source."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        return vals[node.id]
    if isinstance(node, ast.Dict):
        return {
            _eval_expr(k, vals): _eval_expr(v, vals)
            for k, v in zip(node.keys, node.values)
        }
    if isinstance(node, ast.List):
        return [_eval_expr(v, vals) for v in node.elts]
    if isinstance(node, ast.Tuple):
        return tuple(_eval_expr(v, vals) for v in node.elts)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_eval_expr(node.operand, vals)
    if isinstance(node, ast.BinOp):
        left = _eval_expr(node.left, vals)
        right = _eval_expr(node.right, vals)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        if isinstance(node.op, ast.Pow):
            return left ** right
    raise ValueError(f"Unsupported expression in {MAIZE_PY}: {ast.dump(node)}")


def _read_primary_inputs():
    wanted = {
        "LEAF_WIDTHS",
        "REF_WIDTH",
        "VascBundle_leaf_ref",
        "VascBundle_stem",
        "VascBundle_taproot",
        "VascBundle_lateral",
        "VascBundle_nodal",
        "VascBundle_shootborne",
        "numSE_leaf",
        "numSE_stem",
        "numSE_root_large",
        "numSE_root_small",
        "r_SE_leaf",
        "r_SE_stem",
        "r_SE_root_large",
        "r_SE_root_small",
    }
    tree = ast.parse(MAIZE_PY.read_text(), filename=str(MAIZE_PY))
    vals = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id in wanted:
                vals[target.id] = _eval_expr(node.value, vals)
    missing = sorted(wanted - vals.keys())
    if missing:
        raise RuntimeError(f"Missing primary input(s) in {MAIZE_PY}: {missing}")
    return vals


def _across(n_bundles, n_se, r_se):
    return n_bundles * n_se * math.pi * r_se ** 2


def _literal_across(vals):
    leaf_widths = vals["LEAF_WIDTHS"]
    ref_width = vals["REF_WIDTH"]
    rows = []

    root_specs = [
        ("Across_st_root_taproot", vals["VascBundle_taproot"],
         vals["numSE_root_large"], vals["r_SE_root_large"]),
        ("Across_st_root_lateral1", vals["VascBundle_lateral"],
         vals["numSE_root_small"], vals["r_SE_root_small"]),
        ("Across_st_root_lateral2", vals["VascBundle_lateral"] + 2,
         vals["numSE_root_small"], vals["r_SE_root_small"]),
        ("Across_st_root_nodal", vals["VascBundle_nodal"],
         vals["numSE_root_large"], vals["r_SE_root_large"]),
        ("Across_st_root_shootborne", vals["VascBundle_shootborne"],
         vals["numSE_root_large"], vals["r_SE_root_large"] - 0.5e-4),
    ]
    for name, n_bundles, n_se, r_se in root_specs:
        rows.append((name, _across(n_bundles, n_se, r_se)))

    rows.append((
        "Across_st_stem",
        _across(vals["VascBundle_stem"], vals["numSE_stem"], vals["r_SE_stem"]),
    ))

    for subtype in sorted(leaf_widths):
        n_bundles = vals["VascBundle_leaf_ref"] * (leaf_widths[subtype] / ref_width)
        rows.append((
            f"Across_st_leaf_L{subtype}",
            _across(n_bundles, vals["numSE_leaf"], vals["r_SE_leaf"]),
        ))

    return rows


def _json_across():
    with MAIZE_JSON.open() as f:
        data = json.load(f)
    per_type = data["PerType"]["Across_st"]["value"]
    root, stem, leaf = per_type
    names = [
        "Across_st_root_taproot",
        "Across_st_root_lateral1",
        "Across_st_root_lateral2",
        "Across_st_root_nodal",
        "Across_st_root_shootborne",
    ]
    rows = [(name, float(value)) for name, value in zip(names, root)]
    rows.append(("Across_st_stem", float(stem[0])))
    rows.extend((f"Across_st_leaf_L{subtype}", float(value))
                for subtype, value in zip(range(2, 13), leaf))
    return dict(rows)


def _verdict(json_value, literal_calc):
    ratio = json_value / literal_calc if literal_calc else math.inf
    if ratio > DRIFT_OVER:
        return ratio, "DRIFT-OVER"
    if ratio < DRIFT_UNDER:
        return ratio, "DRIFT-UNDER"
    return ratio, "OK"


def main():
    vals = _read_primary_inputs()
    literal_rows = _literal_across(vals)
    json_rows = _json_across()

    print("=" * 96)
    print("Across_st audit: maize_phloem_2026.py anatomy vs maize2026 JSON")
    print("=" * 96)
    print(f"Maize source: {MAIZE_PY}")
    print(f"Maize JSON:   {MAIZE_JSON}")
    print("\n| parameter | literal_calc | json_value | ratio | verdict |")
    print("|---|---:|---:|---:|---|")

    over = []
    under = []
    for parameter, literal_calc in literal_rows:
        json_value = json_rows[parameter]
        ratio, verdict = _verdict(json_value, literal_calc)
        if verdict == "DRIFT-OVER":
            over.append((parameter, literal_calc, json_value, ratio))
        elif verdict == "DRIFT-UNDER":
            under.append((parameter, literal_calc, json_value, ratio))
        print(f"| {parameter} | {literal_calc:.6e} | {json_value:.6e} | "
              f"{ratio:.6g} | {verdict} |")

    print()
    if not over and not under:
        print("AUDIT CLEAN")
    else:
        if over:
            organs = ", ".join(row[0] for row in over)
            print(f"AUDIT: DRIFT-OVER on {organs}")
        if under:
            organs = ", ".join(row[0] for row in under)
            print(f"AUDIT: DRIFT-UNDER on {organs}")

    if over:
        print("\nDRIFT-OVER details")
        for parameter, literal_calc, json_value, ratio in over:
            print(f"  {parameter}: literal={literal_calc:.6e}, "
                  f"json={json_value:.6e}, ratio={ratio:.6g}")
        print("Recommend: fix Across_st in maize_phloem_2026.py using same pattern "
              "as kx_st commit 74b1ac2c, regen JSON, then re-run V3 at "
              "corrected baseline. Do NOT auto-fix.")
    if under:
        print("\nDRIFT-UNDER details")
        for parameter, literal_calc, json_value, ratio in under:
            print(f"  {parameter}: literal={literal_calc:.6e}, "
                  f"json={json_value:.6e}, ratio={ratio:.6g}")
        print("Fixing would lower v (wrong direction). Log as separate ticket. "
              "Sweep against current buggy JSON value.")

    return 0 if not over and not under else 1


if __name__ == "__main__":
    raise SystemExit(main())
