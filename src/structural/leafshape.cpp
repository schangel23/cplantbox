// -*- mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-
#include "leafshape.h"

#include <algorithm>
#include <cmath>

namespace CPlantBox {

// ============================================================
// LeafShape (base default for sampleCanonicalGrid)
// ============================================================

std::vector<Vector3d> LeafShape::sampleCanonicalGrid(
	int n_u, int n_v, double lmax, double max_w) const
{
	if (n_u < 2 || n_v < 1) {
		throw std::invalid_argument(
			"LeafShape::sampleCanonicalGrid: n_u >= 2 and n_v >= 1 required");
	}
	std::vector<Vector3d> out;
	out.reserve(static_cast<size_t>(n_u) * static_cast<size_t>(n_v));
	const double du = 1.0 / static_cast<double>(n_u - 1);
	const double dv = (n_v > 1) ? 1.0 / static_cast<double>(n_v - 1) : 0.0;
	for (int iu = 0; iu < n_u; ++iu) {
		const double u = static_cast<double>(iu) * du;
		for (int iv = 0; iv < n_v; ++iv) {
			const double v = (n_v > 1) ? static_cast<double>(iv) * dv : 0.5;
			out.push_back(evaluate(u, v, lmax, max_w));
		}
	}
	return out;
}

// ============================================================
// MedianLeafShape
// ============================================================

MedianLeafShape::MedianLeafShape(
	std::vector<Vector3d> cps, int n_u, int n_v)
	: cps_(std::move(cps)), n_u_(n_u), n_v_(n_v)
{
	if (n_u_ < 2 || n_v_ < 2) {
		throw std::invalid_argument(
			"MedianLeafShape: n_u and n_v must be >= 2");
	}
	if (static_cast<int>(cps_.size()) != n_u_ * n_v_) {
		throw std::invalid_argument(
			"MedianLeafShape: cps.size() must equal n_u * n_v");
	}
}

Vector3d MedianLeafShape::evaluate(
	double u, double v, double /*lmax*/, double /*max_w*/) const
{
	// Clamp parametric coordinates into [0, 1]; canonical callers stay in range,
	// the clamp just keeps misuse safe.
	const double uc = std::min(std::max(u, 0.0), 1.0);
	const double vc = std::min(std::max(v, 0.0), 1.0);

	const double fu = uc * static_cast<double>(n_u_ - 1);
	const double fv = vc * static_cast<double>(n_v_ - 1);

	int iu = static_cast<int>(std::floor(fu));
	int iv = static_cast<int>(std::floor(fv));

	// Pin the upper edge to the last cell so iu+1 / iv+1 stay in-bounds.
	if (iu >= n_u_ - 1) { iu = n_u_ - 2; }
	if (iv >= n_v_ - 1) { iv = n_v_ - 2; }

	const double a = fu - static_cast<double>(iu);
	const double b = fv - static_cast<double>(iv);

	const Vector3d& p00 = cps_[iu       * n_v_ + iv      ];
	const Vector3d& p10 = cps_[(iu + 1) * n_v_ + iv      ];
	const Vector3d& p01 = cps_[iu       * n_v_ + (iv + 1)];
	const Vector3d& p11 = cps_[(iu + 1) * n_v_ + (iv + 1)];

	const double w00 = (1.0 - a) * (1.0 - b);
	const double w10 = a         * (1.0 - b);
	const double w01 = (1.0 - a) * b;
	const double w11 = a         * b;

	return Vector3d(
		w00 * p00.x + w10 * p10.x + w01 * p01.x + w11 * p11.x,
		w00 * p00.y + w10 * p10.y + w01 * p01.y + w11 * p11.y,
		w00 * p00.z + w10 * p10.z + w01 * p01.z + w11 * p11.z);
}

std::vector<Vector3d> MedianLeafShape::sampleCanonicalGrid(
	int n_u, int n_v, double lmax, double max_w) const
{
	// Byte-identity fast path: caller asks for the grid this MedianLeafShape
	// was built on → return a direct copy of cps_. This is the S2 D.0 6-XML
	// invariance guarantee — the default-fallback dispatch reproduces the
	// previous direct-surface_cps consumption byte-for-byte.
	if (n_u == n_u_ && n_v == n_v_) {
		return cps_;
	}
	// Fallback: bilinear via evaluate (base-class default). Not exercised
	// today but kept for any future caller that resamples on a different grid.
	return LeafShape::sampleCanonicalGrid(n_u, n_v, lmax, max_w);
}

// ============================================================
// ParametricLeafShape (S3 — symmetric splines + frozen asym residual)
// ============================================================

namespace {

/**
 * Find the knot span containing u: the largest index `i` with knots[i] <= u
 * (and knots[i] < knots[i+1] except at the right endpoint). Mirrors the
 * standard "FindSpan" used in De Boor's algorithm and scipy's BSpline._evaluate.
 *
 * For a clamped knot vector with degree+1 repeated zeros at the start and ones
 * at the end, the search is restricted to [degree, n_cp - 1] so the De Boor
 * recurrence below indexes valid coefficients only.
 */
int findKnotSpan(const std::vector<double>& knots, int n_cp, int degree, double u)
{
	if (u >= knots[n_cp]) return n_cp - 1;          // right endpoint
	if (u <= knots[degree]) return degree;          // left endpoint
	int lo = degree;
	int hi = n_cp;
	int mid = (lo + hi) / 2;
	while (u < knots[mid] || u >= knots[mid + 1]) {
		if (u < knots[mid]) hi = mid; else lo = mid;
		mid = (lo + hi) / 2;
	}
	return mid;
}

/**
 * Evaluate a B-spline curve at u via De Boor's algorithm.
 *
 * @param knots  knot vector of length n_cp + degree + 1 (clamped at both ends).
 * @param coeffs control points, length n_cp.
 * @param degree spline degree (k = 4 in the maize distribution).
 * @param u      evaluation point (caller pre-clamps to the knot range).
 *
 * Numerically equivalent to scipy `BSpline(knots, coeffs, degree)(u)`. The
 * S3 anchor gate compares the maize-distribution intercepts against XML CPs
 * via this routine and expects bit-equivalence at the canonical u-stations
 * (where scipy interpolates exactly per S0 delta #1).
 */
double evalBSpline(const std::vector<double>& knots,
	const std::vector<double>& coeffs, int degree, double u)
{
	const int n_cp = static_cast<int>(coeffs.size());
	const int span = findKnotSpan(knots, n_cp, degree, u);
	// Local copy of the (degree + 1) relevant control points.
	std::vector<double> d(degree + 1);
	for (int i = 0; i <= degree; ++i) {
		d[i] = coeffs[span - degree + i];
	}
	for (int r = 1; r <= degree; ++r) {
		for (int j = degree; j >= r; --j) {
			const double t_left  = knots[span - degree + j];
			const double t_right = knots[span + 1 + j - r];
			const double denom = t_right - t_left;
			const double alpha = (denom > 0.0) ? (u - t_left) / denom : 0.0;
			d[j] = (1.0 - alpha) * d[j - 1] + alpha * d[j];
		}
	}
	return d[degree];
}

} // anonymous

ParametricLeafShape::ParametricLeafShape(int rank,
	std::vector<double> spline_knots_u,
	int spline_degree,
	std::vector<double> midrib_droop_coeffs,
	std::vector<double> midrib_along_coeffs,
	std::vector<double> halfwidth_coeffs,
	std::vector<Vector3d> asym_residual_grid,
	int n_u, int n_v,
	double max_w_intercept)
	: rank_(rank)
	, spline_degree_(spline_degree)
	, spline_knots_u_(std::move(spline_knots_u))
	, midrib_droop_coeffs_(std::move(midrib_droop_coeffs))
	, midrib_along_coeffs_(std::move(midrib_along_coeffs))
	, halfwidth_coeffs_(std::move(halfwidth_coeffs))
	, asym_residual_grid_(std::move(asym_residual_grid))
	, n_u_(n_u)
	, n_v_(n_v)
	, max_w_intercept_(max_w_intercept)
{
	if (n_u_ < 2 || n_v_ < 2) {
		throw std::invalid_argument(
			"ParametricLeafShape: n_u and n_v must be >= 2");
	}
	if (spline_degree_ < 1) {
		throw std::invalid_argument(
			"ParametricLeafShape: spline_degree must be >= 1");
	}
	const int n_cp = static_cast<int>(midrib_droop_coeffs_.size());
	if (n_cp < spline_degree_ + 1) {
		throw std::invalid_argument(
			"ParametricLeafShape: n_cp must be >= spline_degree + 1");
	}
	if (static_cast<int>(midrib_along_coeffs_.size()) != n_cp ||
		static_cast<int>(halfwidth_coeffs_.size()) != n_cp) {
		throw std::invalid_argument(
			"ParametricLeafShape: all coefficient blocks must share length n_cp");
	}
	if (static_cast<int>(spline_knots_u_.size()) != n_cp + spline_degree_ + 1) {
		throw std::invalid_argument(
			"ParametricLeafShape: knot vector length must be n_cp + degree + 1");
	}
	if (static_cast<int>(asym_residual_grid_.size()) != n_u_ * n_v_) {
		throw std::invalid_argument(
			"ParametricLeafShape: asym_residual_grid size must equal n_u * n_v");
	}
}

Vector3d ParametricLeafShape::evaluate(
	double u, double v, double /*lmax*/, double /*max_w*/) const
{
	// Clamp parametric coordinates into [0, 1]; canonical callers stay in range.
	const double uc = std::min(std::max(u, 0.0), 1.0);
	const double vc = std::min(std::max(v, 0.0), 1.0);

	// 1) Symmetric component — three independent B-splines at u.
	const double m_y = evalBSpline(spline_knots_u_, midrib_droop_coeffs_,
		spline_degree_, uc);
	const double m_z = evalBSpline(spline_knots_u_, midrib_along_coeffs_,
		spline_degree_, uc);
	const double w   = evalBSpline(spline_knots_u_, halfwidth_coeffs_,
		spline_degree_, uc);

	// S6 max_w bake: lateral term uses the per-rank fit-time max_w that the
	// fitter normalised against (= XML grid's grid-derived peak half-width
	// for this rank), NOT the runtime `max_w` arg. lrp->Width_blade differs
	// from the XML-grid peak by ~7% on maize (Width_blade is the named
	// scalar used by area/volume math, not the grid normaliser). Without
	// this bake, scale=0 reconstruction drifts ~3 mm from XML surface_cps
	// and the S6 D11 baseline guarantee fails.
	const double sym_x = (vc - 0.5) * w * max_w_intercept_;
	const double sym_y = m_y;
	const double sym_z = m_z;

	// 2) Frozen asymmetric residual — bilinear interpolation over the
	//    (n_u_, n_v_) grid using the same row-major convention and clamping
	//    behaviour as MedianLeafShape::evaluate, so canonical callers landing
	//    on (u_i, v_j) = (i/(n_u-1), j/(n_v-1)) get the residual exactly.
	const double fu = uc * static_cast<double>(n_u_ - 1);
	const double fv = vc * static_cast<double>(n_v_ - 1);

	int iu = static_cast<int>(std::floor(fu));
	int iv = static_cast<int>(std::floor(fv));
	if (iu >= n_u_ - 1) { iu = n_u_ - 2; }
	if (iv >= n_v_ - 1) { iv = n_v_ - 2; }

	const double a = fu - static_cast<double>(iu);
	const double b = fv - static_cast<double>(iv);

	const Vector3d& r00 = asym_residual_grid_[iu       * n_v_ + iv      ];
	const Vector3d& r10 = asym_residual_grid_[(iu + 1) * n_v_ + iv      ];
	const Vector3d& r01 = asym_residual_grid_[iu       * n_v_ + (iv + 1)];
	const Vector3d& r11 = asym_residual_grid_[(iu + 1) * n_v_ + (iv + 1)];

	const double w00 = (1.0 - a) * (1.0 - b);
	const double w10 = a         * (1.0 - b);
	const double w01 = (1.0 - a) * b;
	const double w11 = a         * b;

	const double asym_x = w00 * r00.x + w10 * r10.x + w01 * r01.x + w11 * r11.x;
	const double asym_y = w00 * r00.y + w10 * r10.y + w01 * r01.y + w11 * r11.y;
	const double asym_z = w00 * r00.z + w10 * r10.z + w01 * r01.z + w11 * r11.z;

	return Vector3d(sym_x + asym_x, sym_y + asym_y, sym_z + asym_z);
}

} // namespace CPlantBox
