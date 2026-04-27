"""Regression test for the Pheno4D emergence-time override path in
``dart.coupling.growth.calibrate``.

Checks three invariants:
  1. Calibrate without --pheno4d-phyllochron produces a ``ldelay`` block
     byte-identical to the shipped ``maize_calibrated.xml`` (fallback
     path unchanged; required by "grow --day 55 identical" regression
     guarantee).
  2. Calibrate with --pheno4d-phyllochron assigns ldelay from Pheno4D
     for covered positions (0,1,2) and from a mean-phyllochron
     extrapolation anchored at the highest covered position for the
     upper positions (3..10). This smooth continuation is required so
     the emergence sequence stays monotonic across the whole plant.
  3. The full ldelay sequence (baseline-coincident pos 0 through the
     last extrapolated position) is monotonically non-decreasing —
     later leaves emerge later (biological plausibility).

The first invariant uses the extracted emergence JSON produced by
``Resources/Pheno4D/extract_emergence_timeseries.py``. If that JSON is
missing, the test regenerates it from ``pheno4d_canonical_cps.json`` so
the test is self-contained.
"""

from __future__ import annotations

import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
CPBENV_PY = REPO_ROOT / "cpbenv" / "bin" / "python"
COUPLING_DATA = REPO_ROOT / "dart" / "coupling" / "data"
SHIPPED_XML = COUPLING_DATA / "maize_calibrated.xml"
MF3D_STATS = COUPLING_DATA / "maizefield3d_stats.json"

PHENO4D_ROOT = Path("/home/lukas/PHD/Resources/Pheno4D")
EMERGENCE_JSON = PHENO4D_ROOT / "pheno4d_emergence_timeseries.json"
EXTRACTOR = PHENO4D_ROOT / "extract_emergence_timeseries.py"


def _ensure_emergence_json():
    if EMERGENCE_JSON.exists():
        return
    if not EXTRACTOR.exists():
        pytest.skip(f"Pheno4D extractor not available: {EXTRACTOR}")
    subprocess.run(
        [sys.executable, str(EXTRACTOR)],
        check=True, capture_output=True,
    )


def _run_calibrate(out_path: Path, *, with_pheno4d: bool, with_stem_r: bool = False):
    cmd = [
        str(CPBENV_PY), "-m", "dart.coupling", "calibrate",
        "--maizefield3d", str(MF3D_STATS),
        "--template", str(SHIPPED_XML),
        "--output", str(out_path),
        "--surface-cps",
    ]
    if with_pheno4d:
        cmd += ["--pheno4d-phyllochron", str(EMERGENCE_JSON)]
    if with_stem_r:
        cmd += ["--pheno4d-stem-r"]
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT), capture_output=True)


def _stem_params(xml_path: Path) -> dict:
    """Return stem subType=1 ln and r values from the XML."""
    tree = ET.parse(xml_path)
    out = {}
    stem = tree.getroot().find(".//stem[@subType='1']")
    if stem is None:
        return out
    for name in ('ln', 'r', 'lmax'):
        p = stem.find(f".//parameter[@name='{name}']")
        if p is not None:
            v = p.get('value')
            if v is not None:
                out[name] = float(v)
    return out


def _ldelay_by_subtype(xml_path: Path) -> dict[int, float]:
    """Return {subType: ldelay} for every <leaf> element."""
    tree = ET.parse(xml_path)
    out: dict[int, float] = {}
    for leaf in tree.getroot().findall("leaf"):
        sub_type_attr = leaf.get("subType")
        if sub_type_attr is None:
            continue
        sub_type = int(sub_type_attr)
        ld_param = leaf.find(".//parameter[@name='ldelay']")
        if ld_param is None:
            continue
        value_attr = ld_param.get("value")
        if value_attr is None:
            continue
        out[sub_type] = float(value_attr)
    return out


@pytest.fixture(scope="module")
def calibrate_outputs(tmp_path_factory):
    if not CPBENV_PY.exists():
        pytest.skip(f"cpbenv python not available at {CPBENV_PY}")
    if not SHIPPED_XML.exists() or not MF3D_STATS.exists():
        pytest.skip("Coupling data not present; calibrate preconditions unmet.")
    _ensure_emergence_json()

    tmp = tmp_path_factory.mktemp("pheno4d_override")
    baseline = tmp / "baseline.xml"
    with_p4d = tmp / "with_pheno4d.xml"
    with_p4d_r = tmp / "with_pheno4d_stem_r.xml"
    _run_calibrate(baseline, with_pheno4d=False)
    _run_calibrate(with_p4d, with_pheno4d=True)
    _run_calibrate(with_p4d_r, with_pheno4d=True, with_stem_r=True)
    return baseline, with_p4d, with_p4d_r


def test_baseline_matches_shipped(calibrate_outputs):
    """Without --pheno4d-phyllochron, regenerated XML equals the shipped one
    (byte-for-byte) — guarantees ``grow --day 55`` stays deterministic."""
    baseline, _, _ = calibrate_outputs
    shipped_bytes = SHIPPED_XML.read_bytes()
    regen_bytes = baseline.read_bytes()
    assert regen_bytes == shipped_bytes, (
        "baseline calibrate diverged from shipped maize_calibrated.xml — "
        "fallback path is no longer a no-op"
    )


def test_override_covers_and_extrapolates(calibrate_outputs):
    """Covered positions (0,1,2) take Pheno4D ldelay directly; upper
    positions (3..10) are filled by mean-phyllochron extrapolation —
    both should differ from the 3.0-d/pos heuristic."""
    baseline, with_p4d, _ = calibrate_outputs
    ld_base = _ldelay_by_subtype(baseline)
    ld_p4d = _ldelay_by_subtype(with_p4d)
    shared = sorted(set(ld_base) & set(ld_p4d) & set(range(2, 13)))
    # subType 2 (pos 0): heuristic 0*3=0; Pheno4D pos 0 emergence 0
    # — same value (anchoring invariant, not a bug).
    assert ld_p4d[2] == pytest.approx(0.0)
    # subType 3 (pos 1): heuristic 3.0 → Pheno4D mean 4.0 d
    assert ld_p4d[3] == pytest.approx(4.0, abs=1e-6)
    # subType 4 (pos 2): heuristic 6.0 → Pheno4D mean ~9.17 d (6 plants)
    assert ld_p4d[4] == pytest.approx(9.1666666, abs=1e-3)
    # subType 5+ (pos 3+) extrapolated from the mean per-prev phyllochron of
    # the covered positions: 0 anchor + (4.0, 5.17) → mean ~4.58 d; anchored
    # at pos 2 = 9.17 d. So pos 3 ≈ 13.75 d, pos 4 ≈ 18.33 d, ...
    assert ld_p4d[5] > ld_base[5], "subType 5 should be extrapolated above heuristic"
    for st in range(5, 13):
        assert ld_p4d[st] != ld_base[st], (
            f"subType {st} should be filled by extrapolation, not heuristic"
        )
    # Sanity: every shared subType exists in both maps.
    assert len(shared) == 11


def test_override_monotonic_and_plausible(calibrate_outputs):
    """Overridden ldelay is monotonically non-decreasing with subType, and
    all override values are non-negative."""
    _, with_p4d, _ = calibrate_outputs
    ld = _ldelay_by_subtype(with_p4d)
    ordered = [ld[st] for st in sorted(ld) if st >= 2]
    for prev, nxt in zip(ordered, ordered[1:]):
        assert nxt >= prev - 1e-9, (
            f"ldelay sequence is not monotonic: {ordered}"
        )
    assert all(v >= 0 for v in ordered), f"negative ldelay in {ordered}"


def test_stem_ln_unchanged_by_phyllochron_flag(calibrate_outputs):
    """Regression guard against the backed-out ln override:
    --pheno4d-phyllochron must NOT touch stem ln. `ln` is CPlantBox's
    mature-internode target; Pheno4D's early-V-stage ~5 cm snapshot is
    not a mature target, so substituting it permanently stunts the stem
    (~54 cm at day 55 vs real maize 150-200 cm). Stem compression needs
    delayNGStart/delayNGEnd, not ln. Both baseline and Pheno4D variants
    keep MF3D's ln=14.5 cm."""
    baseline, with_p4d, with_p4d_r = calibrate_outputs
    base = _stem_params(baseline)
    p4d = _stem_params(with_p4d)
    p4d_r = _stem_params(with_p4d_r)
    assert base['ln'] == pytest.approx(14.5, abs=0.2), (
        f"baseline stem ln drifted from MF3D default: {base}"
    )
    assert p4d['ln'] == pytest.approx(14.5, abs=0.2), (
        f"--pheno4d-phyllochron must not touch stem ln: {p4d}"
    )
    assert p4d_r['ln'] == pytest.approx(14.5, abs=0.2), (
        f"--pheno4d-stem-r must not touch stem ln: {p4d_r}"
    )


def test_stem_r_override_opt_in(calibrate_outputs):
    """--pheno4d-stem-r (opt-in) flips stem r to the Pheno4D top-collar
    elongation fit (~0.94 cm/day). Without it, r stays at the default
    2.5 cm/day. Opt-in because the fit is an early-V-stage rate that
    extrapolates badly to the late-stage stem."""
    _, with_p4d, with_p4d_r = calibrate_outputs
    p4d = _stem_params(with_p4d)
    p4d_r = _stem_params(with_p4d_r)
    assert p4d['r'] == pytest.approx(2.5, abs=0.01), (
        f"--pheno4d-phyllochron alone must not touch stem r: {p4d}"
    )
    assert p4d_r['r'] == pytest.approx(0.938, abs=0.05), (
        f"Pheno4D stem r override not applied: {p4d_r}"
    )


def test_stem_lmax_never_overridden(calibrate_outputs):
    """Pheno4D stem elongation fit lmax hits the 200-cm upper bound
    (early-window data can't identify the mature ceiling), so lmax must
    never propagate. Expect MF3D's lmax=180 across all three variants."""
    baseline, with_p4d, with_p4d_r = calibrate_outputs
    for path in (baseline, with_p4d, with_p4d_r):
        params = _stem_params(path)
        assert params['lmax'] == pytest.approx(180.0, abs=0.01), (
            f"stem lmax drifted in {path.name}: {params}"
        )
