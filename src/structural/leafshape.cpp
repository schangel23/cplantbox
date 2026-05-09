// -*- mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-
#include "leafshape.h"

#include <algorithm>
#include <cmath>

namespace CPlantBox {

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
