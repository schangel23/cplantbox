// -*- mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-
#ifndef LEAFSHAPE_DISTRIBUTION_H_
#define LEAFSHAPE_DISTRIBUTION_H_

#include "leafshape.h"
#include "mymath.h"

#include <memory>
#include <string>
#include <vector>

namespace CPlantBox {

/**
 * Cultivar-level parametric leaf shape distribution
 * (PLAN_PARAMETRIC_LEAF_SHAPE_2026-05-09_REV1, S0 + S4).
 *
 * Loaded once per cultivar JSON file (process-wide cache, see
 * load(path)). One instance is shared across all leaf subTypes belonging
 * to the same cultivar XML; LeafRandomParameter::realize() then asks for
 * a per-plant ParametricLeafShape via makeShape(rank, scale, plant_seed_val).
 *
 * JSON schema consumed (subset of the S0 fitter output):
 *   {
 *     "n_components": 33,
 *     "n_cp_per_axis": 11,
 *     "spline_degree": 4,
 *     "spline_knots_u": [16 doubles],
 *     "n_u": 11,
 *     "n_v": 5,
 *     "n_ranks": 15,
 *     "intercepts":          { "0": [33 doubles], ..., "14": [...] },
 *     "asym_residual_grids_cm": { "0": [[ [3] x n_v ] x n_u], ... },
 *     "covariance":          [33 x 33 doubles],
 *     "cholesky_factor":     [33 x 33 doubles] (lower-triangular L of cov),
 *     "coeffs_block_layout": { "droop":[0,11], "along":[11,22], "halfwidth_norm":[22,33] }
 *   }
 *
 * Other fields (frame_convention, fit_residual_summary, gates, ...) are
 * read by the fitter / acceptance scripts and skipped here.
 */
class LeafShapeDistribution {
public:
    /**
     * Load (or fetch from process-wide cache) a distribution JSON.
     *
     * @param path Filesystem path to the JSON file. Resolved verbatim;
     *             callers are responsible for absolute-path resolution
     *             (LeafRandomParameter::realize uses the path as written
     *             in XML).
     * @return     Shared pointer to the parsed distribution. Subsequent
     *             load() calls with the same path return the same
     *             instance until the last user drops their reference.
     * @throws     std::runtime_error if the file cannot be opened or the
     *             JSON cannot be parsed; std::invalid_argument if the
     *             schema is missing required fields. There is no silent
     *             fallback — see project memory
     *             "No auto-fallback for explicit env flags" for the
     *             rationale (an explicit user-set distribution path that
     *             fails to load should crash early so the configuration
     *             drift is visible, not be papered over with the
     *             MedianLeafShape default).
     */
    static std::shared_ptr<LeafShapeDistribution> load(const std::string& path);

    /**
     * Construct a per-plant ParametricLeafShape for one rank.
     *
     * @param rank             Rank index in [0, n_ranks). Picks
     *                         intercept[rank] and asym_residual_grids[rank].
     * @param scale            shape_variation_scale (D11). 0 → return the
     *                         population intercept (deterministic, same
     *                         across all plants of this cultivar). >0 →
     *                         draw one per-plant deviation z and add
     *                         scale * (L @ z) on top of the intercept.
     * @param plant_seed_val   Plant-level RNG seed (Organism::getSeedVal()).
     *                         Same seed → same z across all 15 ranks of
     *                         the same plant (D2 per-plant coherence).
     *                         Drawn from a local std::mt19937 keyed by
     *                         (plant_seed_val ^ shape_seed_salt_) so the
     *                         master RNG stream is not perturbed and
     *                         D.0 6-XML invariance is preserved when
     *                         shape_distribution_path is unset on other
     *                         organs of the same plant.
     */
    std::shared_ptr<ParametricLeafShape> makeShape(int rank, double scale,
        unsigned int plant_seed_val) const;

    /// Number of ranks in the distribution (15 for the maize bake).
    int numRanks() const { return n_ranks_; }
    /// Number of symmetric coefficients per rank (33 for maize: 11+11+11).
    int numComponents() const { return n_components_; }
    int splineDegree() const { return spline_degree_; }
    int numCpsU() const { return n_u_; }
    int numCpsV() const { return n_v_; }
    const std::string& path() const { return path_; }
    const std::vector<double>& splineKnotsU() const { return spline_knots_u_; }
    /// Read-only access to per-rank intercept (length n_components_).
    const std::vector<double>& intercept(int rank) const { return intercepts_.at(rank); }
    /// Read-only access to per-rank asym residual grid (length n_u_ * n_v_).
    const std::vector<Vector3d>& asymResidualGrid(int rank) const { return asym_residuals_.at(rank); }

private:
    // Construction is private; use load() so the cache is consistent.
    LeafShapeDistribution() = default;

    std::string path_;
    int n_components_ = 0;
    int n_cp_per_axis_ = 0;
    int spline_degree_ = 0;
    int n_u_ = 0;
    int n_v_ = 0;
    int n_ranks_ = 0;
    int droop_block_start_ = 0;     ///< coeffs[droop] = [start, start + n_cp]
    int along_block_start_ = 0;
    int halfwidth_block_start_ = 0;
    std::vector<double> spline_knots_u_;
    std::vector<std::vector<double>> intercepts_;          ///< [n_ranks_][n_components_]
    std::vector<std::vector<Vector3d>> asym_residuals_;    ///< [n_ranks_][n_u_ * n_v_]
    std::vector<std::vector<double>> cholesky_factor_;     ///< [n_components_][n_components_]
    /// Salt mixed into Organism::getSeedVal() for shape draws.
    /// Decouples the shape-z stream from any other plant-level draws so
    /// that running with shape_variation_scale > 0 does not perturb
    /// (and therefore does not need to be matched in) any other RNG-fed
    /// downstream quantity.
    static constexpr unsigned int shape_seed_salt_ = 0x5EAFC0A1u;
};

} // namespace CPlantBox

#endif // LEAFSHAPE_DISTRIBUTION_H_
