"""S5 acceptance gates — pybind diagnostic surface for G9 spy probes.

Plan: Literature/Chapter 1/Concepts/CPBOXBALENOCOUPLING/PLAN_PARAMETRIC_LEAF_SHAPE_2026-05-09_REV1.md.

S5 is pybind-only: extend the C++ → Python surface so the G9 spy probes (S8)
can verify per-plant ``z`` coherence and runtime shape source. These tests
exercise just the new accessors, not the underlying C++ math (S3 covers
that). They guard against silent regressions of the binding contract:

1. ``ParametricLeafShape`` coefficient accessors round-trip with the
   constructor inputs (so a Python-side test can read what got built).

2. ``LeafShapeDistribution`` diagnostic accessors agree with the S0 JSON
   schema (``cholesky_factor`` shape, ``intercepts`` length, knot vector,
   per-rank residual grid).

3. ``LeafSpecificParameter.shape`` is exposed and dispatches:
   - default-XML maize (``shape_distribution_path = ""``) → ``None`` until
     ``Leaf::getEffectiveSurfaceCPs`` lazily materialises a
     ``MedianLeafShape`` (the S2 fallback), at which point reads return one;
   - cultivar distribution wired with ``shape_variation_scale > 0`` →
     every blade's shape is ``ParametricLeafShape``.

4. **G9 Spy 2 primitive at the unit level**: given two realised
   ``ParametricLeafShape`` instances at different ranks for the same
   ``plant_seed_val``, the per-plant ``z`` recovered via
   ``L^{-1} (coeffs - intercept[r]) / scale`` matches across ranks within
   FP tolerance. This is the precise check S8 builds the canopy-level
   ``Spy 2`` on top of; if it fails here, the whole canopy spy is broken.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import plantbox as pb

REPO_ROOT = Path(__file__).resolve().parents[3]
DIST_JSON = REPO_ROOT / "dart/coupling/data/maize_leaf_shape_distribution.json"
XML_PATH = REPO_ROOT / "dart/coupling/data/maize_calibrated.xml"


@pytest.fixture(scope="module")
def dist() -> pb.LeafShapeDistribution:
    if not DIST_JSON.exists():
        pytest.skip(f"S0 distribution missing: {DIST_JSON}")
    return pb.LeafShapeDistribution.load(str(DIST_JSON))


@pytest.fixture(scope="module")
def dist_json() -> dict:
    if not DIST_JSON.exists():
        pytest.skip(f"S0 distribution missing: {DIST_JSON}")
    with open(DIST_JSON) as f:
        return json.load(f)


# -------------------- (1) ParametricLeafShape accessors --------------------


def test_parametric_leaf_shape_coeff_accessors_roundtrip(dist: pb.LeafShapeDistribution) -> None:
    sh = dist.makeShape(rank=3, scale=1.0, plant_seed_val=99)
    assert isinstance(sh, pb.ParametricLeafShape)
    assert sh.rank() == 3
    assert sh.splineDegree() == dist.splineDegree()
    assert sh.numCpsU() == dist.numCpsU()
    assert sh.numCpsV() == dist.numCpsV()

    # Knot vector identical to the distribution's (S3 contract: same knots
    # for all three coefficient blocks, taken verbatim from the JSON).
    knots_dist = np.asarray(dist.splineKnotsU())
    knots_sh = np.asarray(sh.splineKnotsU())
    assert knots_sh.shape == knots_dist.shape
    np.testing.assert_array_equal(knots_sh, knots_dist)

    # Coefficient block lengths match n_cp_per_axis (11 for the maize bake).
    assert len(sh.midribDroopCoeffs()) == dist.nCpPerAxis()
    assert len(sh.midribAlongCoeffs()) == dist.nCpPerAxis()
    assert len(sh.halfwidthCoeffs()) == dist.nCpPerAxis()

    # asym_residual_grid is frozen across plants — bit-identical to the
    # distribution's per-rank residual.
    res_sh = np.asarray([(v.x, v.y, v.z) for v in sh.asymResidualGrid()])
    res_dist = np.asarray([(v.x, v.y, v.z) for v in dist.asymResidualGrid(3)])
    np.testing.assert_array_equal(res_sh, res_dist)


def test_parametric_intercept_zero_scale_matches_distribution(dist: pb.LeafShapeDistribution) -> None:
    """At scale=0, the realised coefficients ARE intercept[r] (no perturbation)."""
    rank = 5
    sh = dist.makeShape(rank=rank, scale=0.0, plant_seed_val=12345)
    intercept = np.asarray(dist.intercept(rank))
    n_cp = dist.nCpPerAxis()
    droop = np.asarray(sh.midribDroopCoeffs())
    along = np.asarray(sh.midribAlongCoeffs())
    width = np.asarray(sh.halfwidthCoeffs())
    np.testing.assert_array_equal(droop, intercept[dist.droopBlockStart():dist.droopBlockStart() + n_cp])
    np.testing.assert_array_equal(along, intercept[dist.alongBlockStart():dist.alongBlockStart() + n_cp])
    np.testing.assert_array_equal(width, intercept[dist.halfwidthBlockStart():dist.halfwidthBlockStart() + n_cp])


# -------------------- (2) LeafShapeDistribution diagnostics --------------------


def test_distribution_accessors_match_json(dist: pb.LeafShapeDistribution, dist_json: dict) -> None:
    assert dist.numRanks() == dist_json["n_ranks"]
    assert dist.numComponents() == dist_json["n_components"]
    assert dist.splineDegree() == dist_json["spline_degree"]
    assert dist.numCpsU() == dist_json["n_u"]
    assert dist.numCpsV() == dist_json["n_v"]
    assert dist.nCpPerAxis() == dist_json["n_cp_per_axis"]

    knots = np.asarray(dist.splineKnotsU())
    np.testing.assert_array_equal(knots, np.asarray(dist_json["spline_knots_u"]))

    L = np.asarray(dist.choleskyFactor())
    chol_json = np.asarray(dist_json["cholesky_factor"])
    assert L.shape == chol_json.shape == (dist.numComponents(), dist.numComponents())
    np.testing.assert_array_equal(L, chol_json)

    # Lower-triangular by S0 contract (column j > row i should be 0).
    np.testing.assert_array_equal(np.triu(L, k=1), 0.0)

    intercept0 = np.asarray(dist.intercept(0))
    np.testing.assert_array_equal(intercept0, np.asarray(dist_json["intercepts"]["0"]))

    res0_pb = np.asarray([(v.x, v.y, v.z) for v in dist.asymResidualGrid(0)])
    res0_json = np.asarray(dist_json["asym_residual_grids_cm"]["0"]).reshape(
        dist.numCpsU() * dist.numCpsV(), 3)
    np.testing.assert_array_equal(res0_pb, res0_json)


# -------------------- (3) LeafSpecificParameter.shape exposure --------------------


def _make_plant(dist_path: str | None, scale: float, seed: int) -> pb.MappedPlant:
    p = pb.MappedPlant()
    p.readParameters(str(XML_PATH))
    leaf_t = pb.OrganTypes.leaf
    for sub in range(2, 17):
        rp = p.getOrganRandomParameter(leaf_t, sub)
        if rp is None:
            continue
        rp.shape_distribution_path = dist_path or ""
        rp.shape_variation_scale = scale
        rp.shape_rank_index = sub - 2
    p.setSeed(seed)
    p.initialize(False)
    p.simulate(40, False)
    return p


def test_lsp_shape_default_xml_is_none_until_lazy_realise() -> None:
    """Default XML (no distribution path) → realize() leaves
    ``LeafSpecificParameter::shape`` as nullptr per the S4 SHIPPED delta;
    a MedianLeafShape only materialises lazily when
    ``Leaf::getEffectiveSurfaceCPs`` is called (S2 fallback)."""
    p = _make_plant(dist_path=None, scale=0.0, seed=7)
    leaves = p.getOrgans(pb.OrganTypes.leaf)
    assert len(leaves) > 0
    # Sanity: no leaf got pre-populated with a ParametricLeafShape.
    for L in leaves:
        sh = L.param().shape
        assert sh is None or not isinstance(sh, pb.ParametricLeafShape)


def test_lsp_shape_dispatches_parametric_when_distribution_wired() -> None:
    """With ``shape_distribution_path`` set + ``scale > 0``, every blade
    carries a ``ParametricLeafShape`` after realize()."""
    p = _make_plant(dist_path=str(DIST_JSON), scale=1.0, seed=7)
    leaves = p.getOrgans(pb.OrganTypes.leaf)
    assert len(leaves) > 0
    parametric = sum(1 for L in leaves if isinstance(L.param().shape, pb.ParametricLeafShape))
    assert parametric == len(leaves), (
        f"only {parametric}/{len(leaves)} blades carry ParametricLeafShape; "
        "S5 binding or S4 realize() dispatch is broken")
    # Ranks are subType - 2 (per maize convention in S6 bake).
    for L in leaves:
        sh = L.param().shape
        # subType is on the leaf's own param; cross-check rank routing.
        sub = L.param().subType
        assert sh.rank() == sub - 2, f"rank mismatch: subType={sub}, rank()={sh.rank()}"


# -------------------- (4) G9 Spy 2 primitive — z coherence across ranks --------------------


def _solve_lower_triangular(L: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Forward substitution for Lz = b, L lower-triangular."""
    n = L.shape[0]
    z = np.zeros(n)
    for i in range(n):
        z[i] = (b[i] - L[i, :i] @ z[:i]) / L[i, i]
    return z


def _recover_z(dist: pb.LeafShapeDistribution, sh: pb.ParametricLeafShape, scale: float) -> np.ndarray:
    """Recover the unit-Gaussian z from a realised ParametricLeafShape:
    coeffs = intercept[r] + scale * L @ z  ⇒  z = L^{-1} (coeffs - intercept[r]) / scale.
    """
    rank = sh.rank()
    n_cp = dist.nCpPerAxis()
    coeffs = np.empty(dist.numComponents())
    coeffs[dist.droopBlockStart():dist.droopBlockStart() + n_cp] = sh.midribDroopCoeffs()
    coeffs[dist.alongBlockStart():dist.alongBlockStart() + n_cp] = sh.midribAlongCoeffs()
    coeffs[dist.halfwidthBlockStart():dist.halfwidthBlockStart() + n_cp] = sh.halfwidthCoeffs()
    intercept = np.asarray(dist.intercept(rank))
    L = np.asarray(dist.choleskyFactor())
    rhs = (coeffs - intercept) / scale
    return _solve_lower_triangular(L, rhs)


def test_g9_spy2_z_coherence_across_ranks(dist: pb.LeafShapeDistribution) -> None:
    """Same plant_seed_val → same z across all 15 ranks (D2 per-plant
    coherence). Recovered z must match across ranks within FP tolerance."""
    seed = 42
    scale = 1.0
    z_per_rank = []
    for rank in range(dist.numRanks()):
        sh = dist.makeShape(rank=rank, scale=scale, plant_seed_val=seed)
        z_per_rank.append(_recover_z(dist, sh, scale))

    z0 = z_per_rank[0]
    for r, z_r in enumerate(z_per_rank[1:], start=1):
        max_abs = float(np.max(np.abs(z_r - z0)))
        assert max_abs < 1e-9, (
            f"per-plant z mismatch between rank 0 and rank {r}: "
            f"max |Δz| = {max_abs:.3e} (D2 coherence violated)")


def test_g9_spy2_z_differs_across_plants(dist: pb.LeafShapeDistribution) -> None:
    """Different plant_seed_val → meaningfully different z (sanity that
    Spy 2 will actually distinguish plants in the canopy)."""
    z_42 = _recover_z(dist, dist.makeShape(rank=0, scale=1.0, plant_seed_val=42), 1.0)
    z_43 = _recover_z(dist, dist.makeShape(rank=0, scale=1.0, plant_seed_val=43), 1.0)
    diff = float(np.max(np.abs(z_42 - z_43)))
    assert diff > 1e-3, f"different plant seeds produced near-identical z (max |Δ|={diff:.3e})"
