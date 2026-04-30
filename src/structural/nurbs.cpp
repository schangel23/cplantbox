// -*- mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-
#include "nurbs.h"

#include <algorithm>
#include <stdexcept>

namespace CPlantBox {

NurbsPatch::NurbsPatch(const std::vector<std::vector<Vector3d>>& cps,
                       int deg_u, int deg_v,
                       const std::vector<double>& knots_u,
                       const std::vector<double>& knots_v)
    : cps_(cps), deg_u_(deg_u), deg_v_(deg_v),
      knots_u_(knots_u), knots_v_(knots_v)
{
    if (cps_.empty() || cps_[0].empty()) {
        throw std::invalid_argument("NurbsPatch: empty control point grid");
    }
    n_u_ = static_cast<int>(cps_.size());
    n_v_ = static_cast<int>(cps_[0].size());
    for (const auto& row : cps_) {
        if (static_cast<int>(row.size()) != n_v_) {
            throw std::invalid_argument(
                "NurbsPatch: control point grid is not rectangular");
        }
    }
    if (deg_u_ < 1 || deg_v_ < 1) {
        throw std::invalid_argument("NurbsPatch: degrees must be >= 1");
    }
    if (n_u_ <= deg_u_ || n_v_ <= deg_v_) {
        throw std::invalid_argument(
            "NurbsPatch: need n_cps > degree in each parametric direction");
    }
    if (static_cast<int>(knots_u_.size()) != n_u_ + deg_u_ + 1) {
        throw std::invalid_argument(
            "NurbsPatch: knots_u length must equal n_u + deg_u + 1");
    }
    if (static_cast<int>(knots_v_.size()) != n_v_ + deg_v_ + 1) {
        throw std::invalid_argument(
            "NurbsPatch: knots_v length must equal n_v + deg_v + 1");
    }
    u_min_ = knots_u_[deg_u_];
    u_max_ = knots_u_[n_u_];
    v_min_ = knots_v_[deg_v_];
    v_max_ = knots_v_[n_v_];
    if (!(u_max_ > u_min_) || !(v_max_ > v_min_)) {
        throw std::invalid_argument(
            "NurbsPatch: knot vectors must be strictly increasing across the domain");
    }
}

int NurbsPatch::findSpan(double t,
                         const std::vector<double>& knots,
                         int degree,
                         int n_cps)
{
    const int n = n_cps - 1;
    if (t >= knots[n + 1]) return n;
    if (t <= knots[degree]) return degree;
    int low = degree;
    int high = n + 1;
    int mid = (low + high) / 2;
    while (t < knots[mid] || t >= knots[mid + 1]) {
        if (t < knots[mid]) high = mid; else low = mid;
        mid = (low + high) / 2;
    }
    return mid;
}

void NurbsPatch::basisFuns(double t, int span, int degree,
                           const std::vector<double>& knots,
                           std::vector<double>& N)
{
    N.assign(degree + 1, 0.0);
    N[0] = 1.0;
    std::vector<double> left(degree + 1, 0.0);
    std::vector<double> right(degree + 1, 0.0);
    for (int j = 1; j <= degree; ++j) {
        left[j] = t - knots[span + 1 - j];
        right[j] = knots[span + j] - t;
        double saved = 0.0;
        for (int r = 0; r < j; ++r) {
            const double temp = N[r] / (right[r + 1] + left[j - r]);
            N[r] = saved + right[r + 1] * temp;
            saved = left[j - r] * temp;
        }
        N[j] = saved;
    }
}

void NurbsPatch::basisFunsDeriv1(double t, int span, int degree,
                                 const std::vector<double>& knots,
                                 std::vector<std::vector<double>>& ders)
{
    const int p = degree;
    ders.assign(2, std::vector<double>(p + 1, 0.0));

    // ndu stores the basis-function values (upper triangle) and the knot
    // differences (lower triangle), per Piegl & Tiller A2.3.
    std::vector<std::vector<double>> ndu(p + 1, std::vector<double>(p + 1, 0.0));
    std::vector<double> left(p + 1, 0.0);
    std::vector<double> right(p + 1, 0.0);
    ndu[0][0] = 1.0;
    for (int j = 1; j <= p; ++j) {
        left[j] = t - knots[span + 1 - j];
        right[j] = knots[span + j] - t;
        double saved = 0.0;
        for (int r = 0; r < j; ++r) {
            ndu[j][r] = right[r + 1] + left[j - r];
            const double temp = ndu[r][j - 1] / ndu[j][r];
            ndu[r][j] = saved + right[r + 1] * temp;
            saved = left[j - r] * temp;
        }
        ndu[j][j] = saved;
    }
    for (int j = 0; j <= p; ++j) ders[0][j] = ndu[j][p];

    // First-derivative loop (k = 1 only of the general A2.3 algorithm).
    std::vector<std::vector<double>> a(2, std::vector<double>(p + 1, 0.0));
    for (int r = 0; r <= p; ++r) {
        const int s1 = 0;
        const int s2 = 1;
        a[0][0] = 1.0;
        const int k = 1;
        double d = 0.0;
        const int rk = r - k;
        const int pk = p - k;
        if (r >= k) {
            a[s2][0] = a[s1][0] / ndu[pk + 1][rk];
            d = a[s2][0] * ndu[rk][pk];
        }
        const int j1 = (rk >= -1) ? 1 : -rk;
        const int j2 = (r - 1 <= pk) ? k - 1 : p - r;
        for (int j = j1; j <= j2; ++j) {
            a[s2][j] = (a[s1][j] - a[s1][j - 1]) / ndu[pk + 1][rk + j];
            d += a[s2][j] * ndu[rk + j][pk];
        }
        if (r <= pk) {
            a[s2][k] = -a[s1][k - 1] / ndu[pk + 1][r];
            d += a[s2][k] * ndu[r][pk];
        }
        ders[1][r] = d;
    }
    for (int j = 0; j <= p; ++j) ders[1][j] *= p;
}

void NurbsPatch::evaluatePointAndDerivs(double u, double v,
                                        Vector3d& point,
                                        Vector3d& dS_du,
                                        Vector3d& dS_dv) const
{
    const double uu = std::min(std::max(u, u_min_), u_max_);
    const double vv = std::min(std::max(v, v_min_), v_max_);

    const int span_u = findSpan(uu, knots_u_, deg_u_, n_u_);
    const int span_v = findSpan(vv, knots_v_, deg_v_, n_v_);

    std::vector<std::vector<double>> ders_u, ders_v;
    basisFunsDeriv1(uu, span_u, deg_u_, knots_u_, ders_u);
    basisFunsDeriv1(vv, span_v, deg_v_, knots_v_, ders_v);

    point = Vector3d(0.0, 0.0, 0.0);
    dS_du = Vector3d(0.0, 0.0, 0.0);
    dS_dv = Vector3d(0.0, 0.0, 0.0);

    for (int i = 0; i <= deg_u_; ++i) {
        const int ui = span_u - deg_u_ + i;
        const double Nu  = ders_u[0][i];
        const double dNu = ders_u[1][i];
        for (int j = 0; j <= deg_v_; ++j) {
            const int vj = span_v - deg_v_ + j;
            const Vector3d& P = cps_[ui][vj];
            const double Nv  = ders_v[0][j];
            const double dNv = ders_v[1][j];

            const double w_p  = Nu  * Nv;
            const double w_du = dNu * Nv;
            const double w_dv = Nu  * dNv;

            point.x += w_p  * P.x;
            point.y += w_p  * P.y;
            point.z += w_p  * P.z;

            dS_du.x += w_du * P.x;
            dS_du.y += w_du * P.y;
            dS_du.z += w_du * P.z;

            dS_dv.x += w_dv * P.x;
            dS_dv.y += w_dv * P.y;
            dS_dv.z += w_dv * P.z;
        }
    }
}

Vector3d NurbsPatch::getPointAt(double u, double v) const
{
    Vector3d p, du, dv;
    evaluatePointAndDerivs(u, v, p, du, dv);
    return p;
}

Vector3d NurbsPatch::getNormalAt(double u, double v) const
{
    Vector3d p, du, dv;
    evaluatePointAndDerivs(u, v, p, du, dv);
    Vector3d n = du.cross(dv);
    const double len = n.length();
    if (len > 1e-12) {
        return Vector3d(n.x / len, n.y / len, n.z / len);
    }
    // Degenerate isoparametric line (e.g. at a pole where both partials are
    // collinear). Caller is expected to handle the rare case; return a
    // deterministic up-vector rather than NaN.
    return Vector3d(0.0, 0.0, 1.0);
}

std::vector<Vector3d> NurbsPatch::getPointsAtGrid(int n_u, int n_v) const
{
    if (n_u < 2 || n_v < 2) {
        throw std::invalid_argument(
            "NurbsPatch::getPointsAtGrid: need n_u >= 2 and n_v >= 2");
    }
    std::vector<Vector3d> out;
    out.reserve(static_cast<std::size_t>(n_u) * static_cast<std::size_t>(n_v));
    for (int i = 0; i < n_u; ++i) {
        const double u = u_min_ + (u_max_ - u_min_) *
                                  static_cast<double>(i) / (n_u - 1);
        for (int j = 0; j < n_v; ++j) {
            const double v = v_min_ + (v_max_ - v_min_) *
                                      static_cast<double>(j) / (n_v - 1);
            out.push_back(getPointAt(u, v));
        }
    }
    return out;
}

std::vector<Vector3d> NurbsPatch::getNormalsAtGrid(int n_u, int n_v) const
{
    if (n_u < 2 || n_v < 2) {
        throw std::invalid_argument(
            "NurbsPatch::getNormalsAtGrid: need n_u >= 2 and n_v >= 2");
    }
    std::vector<Vector3d> out;
    out.reserve(static_cast<std::size_t>(n_u) * static_cast<std::size_t>(n_v));
    for (int i = 0; i < n_u; ++i) {
        const double u = u_min_ + (u_max_ - u_min_) *
                                  static_cast<double>(i) / (n_u - 1);
        for (int j = 0; j < n_v; ++j) {
            const double v = v_min_ + (v_max_ - v_min_) *
                                      static_cast<double>(j) / (n_v - 1);
            out.push_back(getNormalAt(u, v));
        }
    }
    return out;
}

} // namespace CPlantBox
