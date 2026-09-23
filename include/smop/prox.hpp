//======================================================================
// prox.hpp -- proximal operators and the KKT residual (header-only).
//
//   proj_inf(x, lambda) : projection onto the l_infinity ball
//                         y = max(-lambda, min(x, lambda))
//   proxL1(x, lambda)   : prox of lambda*||.||_1,  y = x - proj_inf(x, lambda)
//   kkt_residual        : eta_vec = g + proj_inf(x - g, lambda)
//                         eta     = ||eta_vec|| / (1 + ||g|| + ||x||)
// This is the standard lasso optimality condition
//     x = prox_{lambda ||.||_1}(x - A^T(Ax - b))
// rewritten as g + proj_inf(x - g, lambda) = 0 with g = A^T(Ax - b).
//======================================================================
#pragma once

#include <algorithm>
#include <cmath>

#include "types.hpp"

namespace smop {

// y = max(-lambda, min(x, lambda));  rr(i) = (y(i) == x(i)) marks the
// coordinates strictly inside the ball (==0 gradient of the dual block).
inline void proj_inf(const Vec& x, double lambda, Vec& y, Vec& rr)
{
    const Index n = x.size();
    y.resize(n);
    rr.resize(n);
    if (lambda <= 0.0) {
        y.setZero();
        rr.setZero();
        return;
    }
    for (Index i = 0; i < n; ++i) {
        const double xi = x(i);
        double yi = xi;
        if      (xi >  lambda) yi =  lambda;
        else if (xi < -lambda) yi = -lambda;
        y(i)  = yi;
        rr(i) = (yi == xi) ? 1.0 : 0.0;
    }
}

inline Vec proj_inf(const Vec& x, double lambda)
{
    Vec y, rr;
    proj_inf(x, lambda, y, rr);
    return y;
}

// prox of lambda * ||.||_1
inline Vec proxL1(const Vec& x, double lambda)
{
    return x - proj_inf(x, lambda);
}

// eta_vec = g + proj_inf(x - g, lambda); returns the relative residual.
inline double kkt_residual(const Vec& x, const Vec& g, double lambda,
                           Vec* eta_vec_out = nullptr)
{
    Vec pg = proj_inf(x - g, lambda);
    Vec eta_vec = g + pg;
    double normg = g.norm();
    double normx = x.norm();
    double eta = eta_vec.norm() / (1.0 + normg + normx);
    if (eta_vec_out) *eta_vec_out = std::move(eta_vec);
    return eta;
}

// nonzero support of x
inline std::vector<Index> nonzero_support(const Vec& x, double tol = 0.0)
{
    std::vector<Index> idx;
    idx.reserve(static_cast<size_t>(x.size()) / 10 + 1);
    for (Index i = 0; i < x.size(); ++i)
        if (std::abs(x(i)) > tol) idx.push_back(i);
    return idx;
}

} // namespace smop
