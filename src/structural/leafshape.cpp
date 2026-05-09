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
// ParametricLeafShape (S1 stub; implementation lands at S3)
// ============================================================

ParametricLeafShape::ParametricLeafShape(int rank,
	std::vector<double> midrib_droop_coeffs,
	std::vector<double> midrib_along_coeffs,
	std::vector<double> halfwidth_coeffs,
	std::vector<Vector3d> asym_residual_grid,
	int n_u, int n_v)
	: rank_(rank)
	, midrib_droop_coeffs_(std::move(midrib_droop_coeffs))
	, midrib_along_coeffs_(std::move(midrib_along_coeffs))
	, halfwidth_coeffs_(std::move(halfwidth_coeffs))
	, asym_residual_grid_(std::move(asym_residual_grid))
	, n_u_(n_u)
	, n_v_(n_v)
{
}

Vector3d ParametricLeafShape::evaluate(
	double /*u*/, double /*v*/, double /*lmax*/, double /*max_w*/) const
{
	throw std::runtime_error(
		"ParametricLeafShape::evaluate is a stub at S1; "
		"spline + asym_residual reconstruction lands at S3 of "
		"PLAN_PARAMETRIC_LEAF_SHAPE_2026-05-09_REV1.md.");
}

} // namespace CPlantBox
