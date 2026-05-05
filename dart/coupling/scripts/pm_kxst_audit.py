"""Audit maize phloem kx_st values against anatomy-derived expectations.

No simulation and no parameter-file edits.  The maize JSON stores kx_st as raw
geometry terms [cm4]; CPlantBox applies viscosity internally.  For the Babst
conductivity comparison this script also prints the hydraulic conversion using
eta_ref = 1 mPa s = 1 hPa day / 864.
"""

import ast
import json
import math
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
MAIZE_PY = REPO_ROOT / "modelparameter/functional/plant_sucrose/maize_phloem_2026.py"
MAIZE_JSON = REPO_ROOT / "modelparameter/functional/plant_sucrose/phloem_parameters_maize2026.json"
WHEAT_PY = REPO_ROOT / "modelparameter/functional/plant_sucrose/wheat_phloem_Giraud2023adapted.py"

ETA_REF_HPA_DAY = 1.0 / 864.0
DRIFT_FACTOR = 2.0


def hp_kx(n_bundles, n_se, r_cm, beta):
    """Hagen-Poiseuille geometry term [cm4], before viscosity division."""
    return n_bundles * n_se * (math.pi / 8.0) * r_cm ** 4 * beta


def babst_kx_raw(k_um2, n_bundles, n_se, r_cm):
    """Babst specific conductivity k [um2] -> raw geometry term [cm4]."""
    k_cm2 = k_um2 * 1e-8
    lumen_area = math.pi * r_cm ** 2
    return k_cm2 * lumen_area * n_bundles * n_se


def hydraulic_from_raw(kx_cm4):
    """Convert raw [cm4] to [cm3 hPa-1 day-1] at eta_ref."""
    return kx_cm4 / ETA_REF_HPA_DAY


def ratio(a, b):
    if b == 0:
        return math.inf
    return max(a, b) / min(a, b)


def line_of(path, needle):
    for i, line in enumerate(path.read_text().splitlines(), start=1):
        if needle in line:
            return i
    return None


def wheat_values():
    tree = ast.parse(WHEAT_PY.read_text(), filename=str(WHEAT_PY))
    vals = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in {"kz_l", "kz_s"}:
                    try:
                        vals[target.id] = ast.literal_eval(node.value)
                    except (ValueError, TypeError):
                        pass
    if not {"kz_l", "kz_s"} <= set(vals):
        # ast.literal_eval cannot evaluate the arithmetic expressions in this
        # old parameter file, so mirror the local formula explicitly.
        beta = 0.9
        vals["kz_l"] = hp_kx(32, 18, 0.00025, beta)
        vals["kz_s"] = hp_kx(52, 21, 0.00019, beta)
    return vals["kz_l"], vals["kz_s"]


def fmt(value):
    if isinstance(value, str):
        return value
    return f"{value:.6e}"


def main():
    print("=" * 100)
    print("kx_st audit: PiafMunch maize phloem coupling")
    print("=" * 100)
    print(f"Maize source: {MAIZE_PY}")
    print(f"Maize JSON:   {MAIZE_JSON}")
    print(f"Wheat sanity: {WHEAT_PY}")

    # Requested primary inputs.
    vasc_leaf_ref = 35
    vasc_stem = 175
    num_se_leaf = 2
    num_se_stem = 3
    r_se_leaf = 4.5e-4
    r_se_stem = 6.25e-4
    beta = 0.9

    literal_leaf = hp_kx(vasc_leaf_ref, num_se_leaf, r_se_leaf, beta)
    literal_stem = hp_kx(vasc_stem, num_se_stem, r_se_stem, beta)

    babst_leaf = babst_kx_raw(0.23, vasc_leaf_ref, num_se_leaf, r_se_leaf)
    babst_stem = babst_kx_raw(0.91, vasc_stem, num_se_stem, r_se_stem)

    with open(MAIZE_JSON) as f:
        maize_json = json.load(f)
    kx_json = maize_json["PerType"]["kx_st"]["value"]
    json_stem = float(kx_json[1][0])
    json_leaf_values = [float(v) for v in kx_json[2]]
    json_leaf_min = min(json_leaf_values)
    json_leaf_max = max(json_leaf_values)
    json_leaf_mid = json_leaf_values[4]  # L4, widest leaf in maize_phloem_2026.py summary.

    wheat_leaf, wheat_stem = wheat_values()

    print("\nConversion note")
    print("  Raw HP/Babst geometry is [cm4], matching PerType.kx_st.unit in maize JSON.")
    print("  Hydraulic kx at eta_ref uses: kx_organ = k[um2] * 1e-8 * "
          "pi*r_cm^2 * N_bundles * numSE / (1/864).")
    print(f"  Babst leaf raw={babst_leaf:.6e} cm4 -> {hydraulic_from_raw(babst_leaf):.6e} cm3/hPa/day")
    print(f"  Babst stem raw={babst_stem:.6e} cm4 -> {hydraulic_from_raw(babst_stem):.6e} cm3/hPa/day")

    leaf_ratio = ratio(json_leaf_mid, literal_leaf)
    stem_ratio = ratio(json_stem, literal_stem)
    rows = [
        {
            "parameter": "kx_st_leaf [cm4 raw]",
            "literal": literal_leaf,
            "json": f"{json_leaf_min:.6e}..{json_leaf_max:.6e} (L4={json_leaf_mid:.6e})",
            "json_compare": json_leaf_mid,
            "babst": babst_leaf,
            "wheat_ratio": json_leaf_mid / wheat_leaf,
            "ratio": leaf_ratio,
        },
        {
            "parameter": "kx_st_stem [cm4 raw]",
            "literal": literal_stem,
            "json": json_stem,
            "json_compare": json_stem,
            "babst": babst_stem,
            "wheat_ratio": json_stem / wheat_stem,
            "ratio": stem_ratio,
        },
    ]

    print("\n| parameter | literal calc | JSON value | Babst-derived | wheat ratio | verdict |")
    print("|---|---:|---:|---:|---:|---|")
    bug_rows = []
    for row in rows:
        verdict = "DRIFT" if row["ratio"] > DRIFT_FACTOR else "OK"
        if verdict == "DRIFT":
            bug_rows.append(row)
        print(f"| {row['parameter']} | {fmt(row['literal'])} | {fmt(row['json'])} | "
              f"{fmt(row['babst'])} | {row['wheat_ratio']:.3f}x | {verdict} |")

    print("\nSanity reference")
    print(f"  wheat kz_l={wheat_leaf:.6e} cm4 at {WHEAT_PY}:{line_of(WHEAT_PY, 'kz_l')}")
    print(f"  wheat kz_s={wheat_stem:.6e} cm4 at {WHEAT_PY}:{line_of(WHEAT_PY, 'kz_s')}")

    if bug_rows:
        print("\nCandidate bug(s)")
        for row in bug_rows:
            if "stem" in row["parameter"]:
                print("  kx_st_stem JSON is >2x below the HP(beta=0.9) anatomy literal "
                      f"({row['ratio']:.2f}x discrepancy).")
                print(f"  Citation: HP requested inputs at {MAIZE_PY}:98, {MAIZE_PY}:108-117; "
                      f"JSON value at {MAIZE_JSON}:227-228.")
                print(f"  Apparent cause: source code intentionally uses Babst stem k at "
                      f"{MAIZE_PY}:{line_of(MAIZE_PY, 'k_stem_cm2')} and "
                      f"{MAIZE_PY}:{line_of(MAIZE_PY, 'kz_stem = _babst_kz')}, "
                      "not the beta=0.9 HP literal.")
            else:
                print(f"  {row['parameter']} is >2x from the reference comparison "
                      f"({row['ratio']:.2f}x discrepancy).")
        print("\nVERDICT: BUG FOUND - shipped stem kx_st follows Babst-specific conductivity, "
              "but the requested primary-input HP(beta=0.9) audit expects a larger stem value.")
    else:
        print("\nVERDICT: CLEAN")


if __name__ == "__main__":
    main()
