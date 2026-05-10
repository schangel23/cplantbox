// -*- mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-
#ifndef LEAFSHAPE_H_
#define LEAFSHAPE_H_

#include "mymath.h"

#include <cstddef>
#include <stdexcept>
#include <vector>

namespace CPlantBox {

/**
 * Leaf shape evaluator (parametric leaf shape, S1 of
 * Literature/Chapter 1/Concepts/CPBOXBALENOCOUPLING/PLAN_PARAMETRIC_LEAF_SHAPE_2026-05-09_REV1.md).
 *
 * Returns a position in the canonical leaf-intrinsic frame
 * (canonical_library convention):
 *     +x_local = lateral (across-blade horizontal at the collar)
 *     +y_local = OOP / in-plane droop axis at the collar
 *     +z_local = along-midrib axis at the collar
 * Caller rotates into the target insertion pose. The base class is the
 * leaf analogue of GrowthFunction / Tropism: a small dispatch interface
 * with multiple subclasses, mounted on LeafSpecificParameter (added in S4).
 *
 * Phase 1 hierarchy:
 *   - MedianLeafShape : bit-identity wrapper around the LRP's surface_cps
 *                       grid. Default for every existing XML; preserves
 *                       D.0 6-XML invariance.
 *   - ParametricLeafShape : symmetric-spline reconstruction +
 *                       per-rank asym_residual grid, sampled per plant
 *                       at realize() time. Stub at S1; implemented in S3.
 */
class LeafShape
{
public:
	virtual ~LeafShape() = default;

	/**
	 * Evaluate the intrinsic surface at parametric (u, v) in [0, 1]^2.
	 *
	 * @param u along-midrib parameter, 0 = collar, 1 = tip.
	 * @param v across-blade parameter, 0 = one margin, 1 = the other,
	 *          0.5 = midrib.
	 * @param lmax mature leaf length [cm]. Median path ignores this
	 *          (the stored CPs are already at mature size); the
	 *          parametric path scales coefficients by it.
	 * @param max_w mature leaf max blade half-width [cm]. Same idea.
	 * @return position in the canonical leaf-intrinsic frame [cm].
	 */
	virtual Vector3d evaluate(double u, double v,
		double lmax, double max_w) const = 0;

	/**
	 * Sample the surface on the canonical (n_u, n_v) grid (row-major,
	 * index = i_u * n_v + i_v). Used by Leaf::getEffectiveSurfaceCPs
	 * (S2 of PLAN_PARAMETRIC_LEAF_SHAPE_2026-05-09_REV1) to materialise
	 * the mature CP grid that the maturity-fade blend sits on top of.
	 *
	 * Default implementation calls evaluate(u_i, v_j) at u_i = i/(n_u-1),
	 * v_j = j/(n_v-1). MedianLeafShape overrides this to short-circuit to a
	 * direct copy of its stored CPs when (n_u, n_v) match — that override is
	 * what gives the S2 D.0 6-XML byte-identity guarantee. Without it, the
	 * default bilinear path drifts by ~1e-15 cm at non-FP-exact canonical
	 * coordinates (e.g. u_i = 0.7 for n_u = 11) and the byte-identity gate
	 * would fail.
	 */
	virtual std::vector<Vector3d> sampleCanonicalGrid(int n_u, int n_v,
		double lmax, double max_w) const;
};

/**
 * Wraps a flat (n_u * n_v) surface_cps grid (row-major, index = i_u * n_v + i_v).
 *
 * evaluate() uses bilinear interpolation in the parametric domain (u, v) in
 * [0, 1]^2 with the standard mapping fu = u * (n_u - 1), fv = v * (n_v - 1).
 * At canonical grid coordinates u_i = i / (n_u - 1), v_j = j / (n_v - 1) this
 * returns cps_[i * n_v + j] exactly (FP precision), giving the S1 byte-identity
 * guarantee against the existing surface_cps consumption path.
 *
 * Out-of-range (u, v) are clamped to [0, 1]; the canonical Leaf::* callers
 * never pass values outside this range, but the clamp keeps misuse safe.
 *
 * The CP grid is in mature leaf-local frame; lmax and max_w are intentionally
 * unused. Downstream consumers (Leaf::updateNodesFromSurfaceCPs) apply the
 * current_length / mature_length scale themselves.
 */
class MedianLeafShape : public LeafShape
{
public:
	MedianLeafShape(std::vector<Vector3d> cps, int n_u, int n_v);

	Vector3d evaluate(double u, double v,
		double lmax, double max_w) const override;

	/// Direct copy of cps_ when (n_u, n_v) match (the S2 byte-identity path).
	/// If the grid sizes differ we fall back to the base-class default
	/// (bilinear sampling via evaluate); that case never fires today but
	/// keeps the contract complete for future callers.
	std::vector<Vector3d> sampleCanonicalGrid(int n_u, int n_v,
		double lmax, double max_w) const override;

	int numCpsU() const { return n_u_; }
	int numCpsV() const { return n_v_; }
	const std::vector<Vector3d>& cps() const { return cps_; }

private:
	std::vector<Vector3d> cps_;   ///< flat (n_u * n_v); row-major i_u * n_v + i_v
	int n_u_;
	int n_v_;
};

/**
 * Symmetric-spline + per-rank asymmetric residual reconstruction
 * (PLAN_PARAMETRIC_LEAF_SHAPE_2026-05-09_REV1, S0 SHIPPED, S3 SHIPPED).
 *
 * Field layout pinned by S0 outputs:
 *   - n_components = 33 = 11 droop + 11 along + 11 halfwidth-norm (n_cp = 11,
 *     deg = 4, exact-interpolation regime per S0 deltas).
 *   - asym_residual_grid_ : flat (n_u * n_v) = 11 * 5 = 55 Vector3d entries,
 *     same row-major index convention as MedianLeafShape::cps_. Frozen across
 *     plants (only the symmetric block deviates at runtime); added on top of
 *     the symmetric reconstruction so intercept[r] reproduces XML rank r at
 *     FP precision. Without this the S0 (a) anchor gate cannot pass.
 *   - spline_knots_u_ : clamped B-spline knot vector of length n_cp + degree +
 *     1 = 16 for n_cp = 11, degree = 4. Identical for all three coefficient
 *     blocks because they share the same u-station abscissas. Stored verbatim
 *     in the S0 distribution JSON (`spline_knots_u`) so the C++ evaluator
 *     reproduces scipy `make_interp_spline(linspace(0,1,11), y, k=4)` exactly.
 *
 * Reconstruction (canonical_library leaf-intrinsic frame, +x_local lateral,
 * +y_local OOP/droop, +z_local along-midrib):
 *
 *     P(u, v) = ( (v - 0.5) * w(u) * max_w,
 *                 m_y(u),
 *                 m_z(u) ) + asym_residual(u, v)
 *
 * where m_y/m_z/w are degree-4 B-splines evaluated at u via De Boor's
 * algorithm; asym_residual(u, v) is bilinear interpolation of the (n_u, n_v)
 * residual grid using the same u_i = i / (n_u - 1), v_j = j / (n_v - 1)
 * mapping as MedianLeafShape::evaluate. lmax is currently unused (the spline
 * coefficients carry physical droop/along positions in cm directly).
 */
class ParametricLeafShape : public LeafShape
{
public:
	ParametricLeafShape() = default;

	ParametricLeafShape(int rank,
		std::vector<double> spline_knots_u,
		int spline_degree,
		std::vector<double> midrib_droop_coeffs,
		std::vector<double> midrib_along_coeffs,
		std::vector<double> halfwidth_coeffs,
		std::vector<Vector3d> asym_residual_grid,
		int n_u, int n_v,
		double max_w_intercept);

	Vector3d evaluate(double u, double v,
		double lmax, double max_w) const override;

	int rank() const { return rank_; }
	int splineDegree() const { return spline_degree_; }
	int numCpsU() const { return n_u_; }
	int numCpsV() const { return n_v_; }
	/// Per-rank peak half-width (cm) the fitter normalised against (= the
	/// XML grid's grid-derived max half-width for this rank). evaluate()
	/// uses this constant for the lateral term (v - 0.5) * w(u) * max_w
	/// instead of the runtime-passed `max_w` arg, so the parametric path
	/// reproduces the XML surface_cps grid bit-for-bit at scale = 0
	/// (S6 D11 baseline guarantee). The runtime `max_w` arg of evaluate()
	/// is therefore ignored on the parametric path; per-plant width
	/// variation comes from the halfwidth coefficient block.
	double maxWIntercept() const { return max_w_intercept_; }
	const std::vector<double>& splineKnotsU()       const { return spline_knots_u_; }
	const std::vector<double>& midribDroopCoeffs()  const { return midrib_droop_coeffs_; }
	const std::vector<double>& midribAlongCoeffs()  const { return midrib_along_coeffs_; }
	const std::vector<double>& halfwidthCoeffs()    const { return halfwidth_coeffs_; }
	const std::vector<Vector3d>& asymResidualGrid() const { return asym_residual_grid_; }

private:
	int rank_ = -1;
	int spline_degree_ = 0;
	std::vector<double> spline_knots_u_;        ///< length n_cp + degree + 1
	std::vector<double> midrib_droop_coeffs_;   ///< m_y(u) spline coefficients
	std::vector<double> midrib_along_coeffs_;   ///< m_z(u) spline coefficients
	std::vector<double> halfwidth_coeffs_;      ///< w(u) along +x_local, normalized
	std::vector<Vector3d> asym_residual_grid_;  ///< (n_u * n_v) frozen residual; S0 delta #2
	int n_u_ = 0;
	int n_v_ = 0;
	double max_w_intercept_ = 0.0;              ///< per-rank fit-time max_w (cm); S6 max_w bake
};

} // namespace CPlantBox

#endif // LEAFSHAPE_H_
