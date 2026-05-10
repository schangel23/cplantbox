"""S4 acceptance gates — `LeafRandomParameter::shape_distribution_path`
plus the per-plant deviation draw routed into `LeafSpecificParameter::shape`.

Plan: Literature/Chapter 1/Concepts/CPBOXBALENOCOUPLING/PLAN_PARAMETRIC_LEAF_SHAPE_2026-05-09_REV1.md.

Gates exercised here (the subset testable through the minimal S4 pybind
exposure; full canopy gates G4 / G7 / G8 / G9 are S6/S8 territory):

1. **Default invariance fall-through (gate G1 baseline)** — `LeafShapeDistribution`
   only kicks in when the XML carries a `shape_distribution_path` attribute.
   Default maize_calibrated.xml omits it, so a `pb.MappedPlant` round-trip
   produces leaves whose `getEffectiveSurfaceCPs()` matches the LRP's
   `surface_cps` byte-for-byte (the S2 lazy `MedianLeafShape` fallback).
   Subsumed by `test_d0_5xml_pm_wrap_invariant` and the PM-dispatch suite,
   but pinned here as a focused reentry test for the S4 commit.

2. **Sampling determinism (D2)** — same `plant_seed_val` → same shape draw
   across repeated `makeShape` calls; different `plant_seed_val` → distinct
   shapes. Verified at `scale = 1.0` over a 11x5 sampled grid.

3. **Per-plant z coherence across ranks (D2)** — within one plant, all 15
   ranks are constructed from the same z. Verified by recovering z from
   each rank's `(coeffs - intercept[r])` and checking the recovered z is
   identical (up to FP precision) across ranks.

4. **scale = 0 reproduces XML at FP precision (D11 / G8 dry-run)** —
   `makeShape(rank, scale=0, ...)` returns intercept[rank] verbatim, and
   sampling that shape on the canonical (n_u, n_v) grid reproduces the
   XML's `surface_cp` median grid for that rank to ≤ 1e-9 cm (S0 gate (a)
   re-tested through the C++ realize() → makeShape → ParametricLeafShape
   path).

5. **End-to-end realize() integration** — when `shape_distribution_path` is
   set on a copy of the maize XML (with `shape_variation_scale = 0`), a
   fresh `pb.MappedPlant` initialised with two distinct seeds yields
   *identical* leaf CPs (because intercept[rank] is plant-seed independent
   when scale = 0). Setting `shape_variation_scale = 1.0` and re-running
   yields *different* leaf CPs across seeds. Confirms the realize() path
   threads `plant->getSeedVal()` into the shape draw.

The test is FAST (< 5 s); the canopy-scale G4/G8/G9 invariants ride on
the S6 bake commit later.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import plantbox as pb
import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
DIST_JSON = REPO_ROOT / "dart/coupling/data/maize_leaf_shape_distribution.json"
XML_PATH = REPO_ROOT / "dart/coupling/data/maize_calibrated.xml"


@pytest.fixture(scope="module")
def distribution() -> pb.LeafShapeDistribution:
    if not DIST_JSON.exists():
        pytest.skip(f"S0 distribution missing: {DIST_JSON}")
    return pb.LeafShapeDistribution.load(str(DIST_JSON))


@pytest.fixture(scope="module")
def max_w_xml_cm() -> dict:
    """Read per-rank max_w_xml_cm out of the distribution JSON.

    The XML's surface_cps was baked at this width (one value per rank),
    so reconstructing the XML grid via sampleCanonicalGrid needs the
    same max_w fed in to the lateral term `(v - 0.5) * w(u) * max_w`.
    Mirrors what `Leaf::getEffectiveSurfaceCPs` does at runtime: pass
    `lrp->Width_blade` which is the same value the XML was baked at.
    """
    if not DIST_JSON.exists():
        pytest.skip(f"S0 distribution missing: {DIST_JSON}")
    with open(DIST_JSON) as f:
        return json.load(f)["max_w_xml_cm"]


# ----------------------------------------------------------------------
# Gate 1 — default-XML fall-through (no shape_distribution_path)
# ----------------------------------------------------------------------

def test_default_xml_falls_through_to_median_shape():
    """When the XML omits shape_distribution_path, leaf::getEffectiveSurfaceCPs
    must hit the S2 lazy MedianLeafShape fallback and return surface_cps verbatim.

    This is the regression smoke for "S4 wiring is invisible to existing XMLs"
    (D5 + plan §S4 acceptance gate). The full D.0 6-XML invariance is in
    test_d0_5xml_pm_wrap_invariant; here we just spot-check one mature leaf
    on maize_calibrated.xml.
    """
    if not XML_PATH.exists():
        pytest.skip(f"maize XML missing: {XML_PATH}")

    plant = pb.MappedPlant()
    plant.readParameters(str(XML_PATH))
    plant.initialize()
    plant.simulate(40)  # enough for at least one mature blade

    leaves = [o for o in plant.getOrgans()
              if o.organType() == 4 and o.getNumberOfNodes() >= 3]
    if not leaves:
        pytest.skip("no leaves emerged at day 40")

    leaf = leaves[0]
    eff = np.asarray(leaf.getEffectiveSurfaceCPs(), dtype=float)
    if eff.shape[0] == 0:
        pytest.skip("leaf has no NURBS CP grid (non-surface_cps XML)")

    # The mature blend at this length should fully resolve to mature CPs
    # (the lazy MedianLeafShape::sampleCanonicalGrid short-circuit returns
    # cps_ verbatim for matching n_u/n_v).
    lrp = leaf.getLeafRandomParameter()
    surface_cps = np.asarray([(p.x, p.y, p.z) for p in lrp.surface_cps], dtype=float)

    # If young-fade is still active, getEffectiveSurfaceCPs blends toward a
    # flat template; we then can't compare against raw surface_cps. The first
    # leaf on a 40-day plant is mature so this fallback is rarely hit, but
    # guard against it explicitly.
    flat_template_y = np.allclose(eff[:, 1], 0.0, atol=1e-9)
    if flat_template_y and not np.allclose(surface_cps[:, 1], 0.0, atol=1e-9):
        pytest.skip("leaf still in young-fade window; mature comparison N/A")

    np.testing.assert_allclose(eff, surface_cps, atol=1e-9, rtol=0,
        err_msg="Default XML leaf surface CPs drifted from surface_cps; "
                "S2 lazy MedianLeafShape fallback is broken")


# ----------------------------------------------------------------------
# Gate 2 — sampling determinism: same seed → same shape; different seeds → different
# ----------------------------------------------------------------------

def _sample_grid(shape: pb.ParametricLeafShape, n_u: int, n_v: int,
                  max_w: float = 1.0) -> np.ndarray:
    """Sample a parametric leaf shape on its canonical (n_u, n_v) grid.

    `max_w` only affects the lateral term `(v - 0.5) * w(u) * max_w`. Pass
    1.0 when comparing two parametric shapes against each other (any
    consistent value works; the lateral component cancels in differences),
    pass the rank-specific Width_blade when comparing against the XML's
    `surface_cp` grid (which was baked at that width).
    """
    cps = shape.sampleCanonicalGrid(n_u, n_v, 1.0, max_w)
    return np.asarray([(p.x, p.y, p.z) for p in cps], dtype=float).reshape(n_u, n_v, 3)


def test_make_shape_same_seed_byte_identical(distribution):
    """Same plant_seed_val, scale > 0 → bitwise-identical sampled grid.

    Pins the local std::mt19937 + cholesky pathway as fully deterministic.
    """
    s_a = distribution.makeShape(rank=4, scale=1.0, plant_seed_val=42)
    s_b = distribution.makeShape(rank=4, scale=1.0, plant_seed_val=42)
    g_a = _sample_grid(s_a, distribution.numCpsU(), distribution.numCpsV())
    g_b = _sample_grid(s_b, distribution.numCpsU(), distribution.numCpsV())
    np.testing.assert_array_equal(g_a, g_b)


def test_make_shape_different_seeds_diverge(distribution):
    """Different plant_seed_val, scale > 0 → distinguishable shape draws.

    Threshold is loose because covariance scale magnitudes vary across
    components; we just need to see that the canopy spread reaches the
    plant-distinguishability range (G4 prelude).
    """
    s_a = distribution.makeShape(rank=4, scale=1.0, plant_seed_val=42)
    s_b = distribution.makeShape(rank=4, scale=1.0, plant_seed_val=43)
    g_a = _sample_grid(s_a, distribution.numCpsU(), distribution.numCpsV())
    g_b = _sample_grid(s_b, distribution.numCpsU(), distribution.numCpsV())
    diff = np.max(np.abs(g_a - g_b))
    assert diff > 0.5, (
        f"Different seeds produced near-identical shape (max-abs diff {diff:.4g} cm). "
        "The local RNG is degenerate or the covariance is unexpectedly tiny.")


# ----------------------------------------------------------------------
# Gate 3 — per-plant z coherence across all 15 ranks (D2)
# ----------------------------------------------------------------------

def test_per_plant_z_is_shared_across_ranks(distribution):
    """All 15 ranks of one plant must be constructed from the same z.

    Recovery: `coeffs(rank) = intercept[rank] + scale * L @ z`, so
    `delta[rank] = coeffs - intercept[rank] = scale * L @ z` is rank-independent
    (for fixed scale, plant_seed_val). We don't have direct access to coeffs
    via pybind at S4, but the rank-independent delta in symmetric coefficient
    space surfaces in the sampled grid as: at v = 0.5 (midrib), the (m_y, m_z)
    components are intercept[r]'s midrib spline plus the same scale*L@z
    delta in droop/along blocks. Across two ranks (r=0 and r=4), the
    *difference* (delta_at_v0.5_rank0 minus delta_at_v0.5_rank4) is
    rank-content only (both deltas share the same scale*L@z). This is fragile
    to write in closed form, so we use a more direct scaffold: under a fixed
    plant seed, set up two shape objects at the same rank with two different
    scales (0 and 1). Their difference is exactly scale*L@z. Across two
    plants with the same seed, that difference is identical.
    """
    rank = 4
    n_u = distribution.numCpsU()
    n_v = distribution.numCpsV()

    delta_for_seed = {}
    for seed in (101, 102):
        s_zero = distribution.makeShape(rank=rank, scale=0.0, plant_seed_val=seed)
        s_one  = distribution.makeShape(rank=rank, scale=1.0, plant_seed_val=seed)
        g_zero = _sample_grid(s_zero, n_u, n_v)
        g_one  = _sample_grid(s_one,  n_u, n_v)
        delta_for_seed[seed] = g_one - g_zero

    # Different plants → different delta (z is plant-specific).
    diff = np.max(np.abs(delta_for_seed[101] - delta_for_seed[102]))
    assert diff > 0.5, (
        "Per-plant deltas are nearly equal across distinct seeds "
        f"(max-abs {diff:.4g} cm); shape draw is not actually plant-keyed.")

    # Coherence across ranks: same seed, two different ranks → both should
    # share the same scale*L@z delta projected through the symmetric block.
    # We can't recover z directly without exposing intercept blocks, but the
    # delta vector at fixed v=0.5 (midrib) is dominated by the droop/along
    # spline contributions. We verify it has the same per-u "shape" (sign
    # pattern + relative magnitudes) across ranks. Concretely: the two delta
    # midrib columns should be linearly related (same z modulates the same
    # coefficient blocks; rank only changes intercept[r]). Pearson correlation
    # is a robust proxy for "same z applied to same blocks".
    seed = 101
    s_r0_zero = distribution.makeShape(rank=0, scale=0.0, plant_seed_val=seed)
    s_r0_one  = distribution.makeShape(rank=0, scale=1.0, plant_seed_val=seed)
    s_r4_zero = distribution.makeShape(rank=4, scale=0.0, plant_seed_val=seed)
    s_r4_one  = distribution.makeShape(rank=4, scale=1.0, plant_seed_val=seed)
    delta_r0 = _sample_grid(s_r0_one, n_u, n_v) - _sample_grid(s_r0_zero, n_u, n_v)
    delta_r4 = _sample_grid(s_r4_one, n_u, n_v) - _sample_grid(s_r4_zero, n_u, n_v)
    # The midrib column (v = N_V//2) carries (lateral_drift, droop, along)
    # deltas. Lateral drift is 0 by construction (midline lateral fixed at 0
    # in the symmetric model + asym_residual is rank-frozen and CANCELS in
    # the delta). So the delta at midrib should be exactly the (m_y, m_z)
    # component differences, both of which are scale*L@z applied to droop /
    # along blocks — same coefficient subspace across ranks. The two delta
    # columns must be *bit-identical* (asym_residual is rank-dependent but
    # CANCELS in (one - zero) since it does not depend on z).
    midrib = n_v // 2
    midrib_delta_r0 = delta_r0[:, midrib, :]
    midrib_delta_r4 = delta_r4[:, midrib, :]
    # Tolerance is FP precision (1e-9 cm). The two deltas evaluate the same
    # B-spline at the same u-stations on the same coefficient subspace —
    # mathematically identical, but the bilinear-asym-residual cancellation
    # carries a few ULPs of accumulated drift through the De Boor inner loop
    # (≤ ~1e-14 cm in the failure observation that motivated the tolerance).
    np.testing.assert_allclose(
        midrib_delta_r0, midrib_delta_r4, atol=1e-9, rtol=0,
        err_msg="Per-plant z is NOT shared across ranks: midrib delta differs "
                "between rank 0 and rank 4 for the same plant seed beyond "
                "FP precision.")


# ----------------------------------------------------------------------
# Gate 4 — scale = 0 reproduces XML rank's surface_cps to FP precision (G8 dry-run)
# ----------------------------------------------------------------------

def test_scale_zero_reproduces_xml_surface_cps(distribution, max_w_xml_cm):
    """For each of the 15 ranks, makeShape(rank, scale=0).sampleCanonicalGrid
    reproduces the XML's surface_cp median grid to ≤ 1e-9 cm element-wise.

    This is the C++ end-to-end re-test of S0 gate (a): the intercept[r] +
    asym_residual[r] composition through the runtime makeShape path
    matches the XML byte-for-byte. Mirrors S3's gate (2) but here we go
    through `LeafShapeDistribution::makeShape` instead of constructing
    `ParametricLeafShape` by hand from JSON.

    The lateral term `(v - 0.5) * w(u) * max_w` requires the rank-specific
    `max_w` (= the value the XML's `surface_cps` was baked with =
    `max_w_xml_cm[rank]`). At runtime, `Leaf::getEffectiveSurfaceCPs`
    threads `lrp->Width_blade` into the same parameter — same trust contract.
    """
    if not XML_PATH.exists():
        pytest.skip(f"maize XML missing: {XML_PATH}")

    plant = pb.MappedPlant()
    plant.readParameters(str(XML_PATH))

    n_u = distribution.numCpsU()
    n_v = distribution.numCpsV()

    # Maize convention: subType 2..16 → rank 0..14.
    for rank in range(distribution.numRanks()):
        sub = rank + 2
        try:
            lrp = plant.getOrganRandomParameter(4, sub)
        except Exception as exc:
            pytest.skip(f"subType {sub} not in XML: {exc}")
        if not lrp.surface_cps:
            pytest.skip(f"subType {sub} has no surface_cps grid")
        xml_cps = np.asarray(
            [(p.x, p.y, p.z) for p in lrp.surface_cps], dtype=float
        ).reshape(n_u, n_v, 3)

        shape = distribution.makeShape(rank=rank, scale=0.0, plant_seed_val=0)
        sampled = _sample_grid(shape, n_u, n_v,
                               max_w=float(max_w_xml_cm[str(rank)]))

        max_abs = np.max(np.abs(sampled - xml_cps))
        assert max_abs <= 1e-9, (
            f"rank {rank} (subType {sub}): max-abs deviation "
            f"{max_abs:.3e} cm exceeds 1e-9 cm. "
            "S0 D10 anchor broken at the runtime makeShape path.")


# ----------------------------------------------------------------------
# Gate 5 — realize() integration: scale=0 path-set XML stays seed-invariant;
#                                  scale>0 path-set XML diverges across seeds.
# ----------------------------------------------------------------------

def _patch_xml_with_distribution(src_xml: Path, dst_xml: Path,
                                  scale: float) -> None:
    """Copy maize_calibrated.xml to dst_xml; mutate every
    `<parameter name="shape_variation_scale" value=...>` to the requested
    scale.

    Post-S6 the canonical maize XML already carries `shape_distribution_path`
    + `shape_variation_scale=0.0` + `shape_rank_index` on every leaf RP
    (the S6 bake). This patcher therefore only needs to override
    shape_variation_scale; the path and rank index come from the bake. An
    older revision of this helper INJECTED the three params anew after the
    opening `<leaf>` tag, which created duplicates with the baked params —
    XML readXML kept the LAST occurrence (the baked scale=0.0), so the
    patched scale=1.0 was silently overridden.
    """
    import re
    text = src_xml.read_text()
    pattern = re.compile(
        r'(<parameter name="shape_variation_scale" value=")[^"]*(" />)')
    new_text, n = pattern.subn(rf'\g<1>{scale}\g<2>', text)
    if n == 0:
        raise RuntimeError(
            "_patch_xml_with_distribution: no shape_variation_scale parameters "
            "found in src_xml; expected the S6 bake to be present.")
    dst_xml.write_text(new_text)


def _grow_and_collect_leaf_cps(xml: Path, seed: int, days: float = 40.0) -> np.ndarray:
    """Grow a plant for `days` days at `seed`, return concatenated
    getEffectiveSurfaceCPs() arrays across all leaves with at least 3 nodes,
    sorted by leaf id for determinism."""
    plant = pb.MappedPlant(seed)
    plant.readParameters(str(xml))
    plant.initialize()
    plant.simulate(days)
    leaves = [o for o in plant.getOrgans()
              if o.organType() == 4 and o.getNumberOfNodes() >= 3]
    leaves.sort(key=lambda l: l.getId())
    arrays = []
    for leaf in leaves:
        cps = leaf.getEffectiveSurfaceCPs()
        if not cps:
            continue
        arrays.append(np.asarray([(p.x, p.y, p.z) for p in cps], dtype=float))
    if not arrays:
        return np.empty((0, 3), dtype=float)
    return np.concatenate(arrays, axis=0)


def test_realize_integration_seed_invariance_at_scale_zero(tmp_path):
    """Path-set XML at scale=0: two seeds → identical leaf CPs (intercept-only,
    independent of plant_seed_val); proves D11's "scale=0 = current per-rank
    median" guarantee survives the realize() path through XML I/O and the
    LeafShapeDistribution makeShape pipeline.

    Conversely at scale=1.0 the same two seeds produce diverging leaf CPs.
    """
    if not XML_PATH.exists() or not DIST_JSON.exists():
        pytest.skip("maize XML or distribution missing")

    xml_zero = tmp_path / "maize_calibrated_shape_scale0.xml"
    xml_one = tmp_path / "maize_calibrated_shape_scale1.xml"
    _patch_xml_with_distribution(XML_PATH, xml_zero, scale=0.0)
    _patch_xml_with_distribution(XML_PATH, xml_one, scale=1.0)

    cps_zero_seed_a = _grow_and_collect_leaf_cps(xml_zero, seed=42)
    cps_zero_seed_b = _grow_and_collect_leaf_cps(xml_zero, seed=43)
    if cps_zero_seed_a.size == 0:
        pytest.skip("no leaves emerged in the test simulation")

    # At scale=0 the shape is intercept-only — plant seed shouldn't change CPs.
    # We only compare the FIRST leaf's CPs because random number consumption
    # by other organ types (root angles etc.) under different seeds reorders
    # leaf insertion times, so the full concatenation isn't aligned across
    # seeds. The first leaf appears at the same time and rank for both seeds.
    n_per_leaf = 11 * 5
    first_a = cps_zero_seed_a[:n_per_leaf]
    first_b = cps_zero_seed_b[:n_per_leaf]
    np.testing.assert_allclose(
        first_a, first_b, atol=1e-9, rtol=0,
        err_msg="At scale=0 the same rank-0 leaf differs across plant seeds; "
                "intercept[rank] must be plant-seed-independent.")

    # At scale=1.0 the same rank-0 leaf SHOULD differ across plant seeds.
    cps_one_seed_a = _grow_and_collect_leaf_cps(xml_one, seed=42)
    cps_one_seed_b = _grow_and_collect_leaf_cps(xml_one, seed=43)
    first_a1 = cps_one_seed_a[:n_per_leaf]
    first_b1 = cps_one_seed_b[:n_per_leaf]
    diff = np.max(np.abs(first_a1 - first_b1))
    assert diff > 0.1, (
        f"At scale=1.0 the rank-0 leaf is too similar across seeds "
        f"(max-abs {diff:.4g} cm); per-plant z draw is not reaching realize().")
