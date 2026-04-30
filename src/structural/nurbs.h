// -*- mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-
#ifndef NURBS_H
#define NURBS_H

#include "mymath.h"

#include <vector>

namespace CPlantBox {

/**
 * Non-rational tensor-product B-spline (NURBS-with-w=1) surface.
 *
 * Represents a smooth surface S(u, v) in R^3 from an (n_u x n_v) grid of
 * 3D control points and clamped (or arbitrary) knot vectors in u and v.
 * Evaluation follows Piegl & Tiller, "The NURBS Book" (1997):
 * findSpan = A2.1, basisFuns = A2.2, basisFunsDeriv1 = specialised A2.3,
 * surface point + first partials = tensor-product summation (A3.6 with k=1).
 *
 * Weights are implicit 1; rational support is intentionally deferred —
 * every current call site in the CPlantBox lofter pipeline passes w=1.
 *
 * Designed as a self-contained drop-in replacement for the two PlantGL
 * methods used by the lofter (`getPointAt`, `getNormalAt`), eliminating
 * the openalea.plantgl runtime dependency. See
 * Literature/Chapter 1/Concepts/CPBOXBALENOCOUPLING/PLAN_REPLACE_PLANTGL_DEPENDENCY_2026-04-30.md.
 */
class NurbsPatch
{
public:
    /**
     * @param cps     (n_u x n_v) control point grid; outer index = u, inner = v
     * @param deg_u   polynomial degree in u (typically 3 for cubic)
     * @param deg_v   polynomial degree in v
     * @param knots_u knot vector in u; length must equal n_u + deg_u + 1
     * @param knots_v knot vector in v; length must equal n_v + deg_v + 1
     */
    NurbsPatch(const std::vector<std::vector<Vector3d>>& cps,
               int deg_u, int deg_v,
               const std::vector<double>& knots_u,
               const std::vector<double>& knots_v);

    Vector3d getPointAt(double u, double v) const;   ///< surface point
    Vector3d getNormalAt(double u, double v) const;  ///< unit normal

    /// Evaluate at a uniform (n_u x n_v) grid spanning the parametric domain;
    /// flat output, u-major (index = i * n_v + j).
    std::vector<Vector3d> getPointsAtGrid(int n_u, int n_v) const;
    std::vector<Vector3d> getNormalsAtGrid(int n_u, int n_v) const;

    int degreeU() const { return deg_u_; }
    int degreeV() const { return deg_v_; }
    int numCpsU() const { return n_u_; }
    int numCpsV() const { return n_v_; }
    const std::vector<double>& knotsU() const { return knots_u_; }
    const std::vector<double>& knotsV() const { return knots_v_; }

private:
    std::vector<std::vector<Vector3d>> cps_;
    int deg_u_, deg_v_;
    int n_u_, n_v_;
    std::vector<double> knots_u_, knots_v_;
    double u_min_, u_max_, v_min_, v_max_;

    /// Surface point + first partial derivatives wrt u and v.
    void evaluatePointAndDerivs(double u, double v,
                                Vector3d& point,
                                Vector3d& dS_du,
                                Vector3d& dS_dv) const;

    /// Piegl & Tiller A2.1: knot span index containing t.
    static int findSpan(double t,
                        const std::vector<double>& knots,
                        int degree,
                        int n_cps);

    /// Piegl & Tiller A2.2: non-zero basis functions at t (length = degree+1).
    static void basisFuns(double t, int span, int degree,
                          const std::vector<double>& knots,
                          std::vector<double>& N);

    /// First-derivative variant of A2.3:
    ///   ders[0][j] = N_{span-degree+j, degree}(t)
    ///   ders[1][j] = derivative of the above with respect to t
    static void basisFunsDeriv1(double t, int span, int degree,
                                const std::vector<double>& knots,
                                std::vector<std::vector<double>>& ders);
};

} // namespace CPlantBox

#endif // NURBS_H
