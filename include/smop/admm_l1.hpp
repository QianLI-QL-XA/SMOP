//======================================================================
// admm_l1.hpp -- ADMM solver for the Lasso subproblem (robust fallback)
//
//     min_x  (1/2)||A x - b||_2^2 + lambda ||x||_1
//
// Port of the reference `admmL1.m` (ClassicLasso folder).  Variables:
//   x    primal iterate,
//   xi   residual block (b - A x split),  y   dual variable with
//        ||y||_inf <= lambda;  the augmented-Lagrangian iteration is
//          xi <- (I + sigma A A^T)^{-1} ( -(Ax-b) - sigma A y )   [CG]
//          y  <- proj_inf( -A^T xi - x/sigma, lambda )
//          x  <- x + gamma*sigma*(A^T xi + y)                      [gamma=1.618]
//   sigma is adapted by the primal/dual winning counters.
//
// This solver is intentionally simple and robust: it is used when the
// smoothing-Newton solver fails to converge, and for large active sets
// inside the adaptive-sieving loop.
//======================================================================
#pragma once

#include <algorithm>
#include <cmath>

#include "linsolve.hpp"
#include "operators.hpp"
#include "prox.hpp"
#include "types.hpp"

namespace smop {

struct AdmmResult {
    Vec  x;
    Vec  xi;
    Vec  grad;
    double obj      = 0.0;
    double eta      = 0.0;
    double primfeas = 0.0;
    double dualfeas = 0.0;
    int    iter     = 0;
    int    termcode = 0;   // 0 ok, 1 max iter, 2 time limit, 3 failure
    double time     = 0.0;
    int    cg_iters = 0;
};

// sigma is re-tuned at iterations 3, 6, 12, 25, 50, 100, 200, ... (as in admmL1.m)
inline bool feasratio_trigger(int iter)
{
    if (iter < 100) {
        const int t[] = {3, 6, 12, 25, 50, 100};
        for (int k : t)
            if (iter == k) return true;
        return false;
    }
    return (iter % 100 == 0);
}

inline AdmmResult admm_l1(const LinearOperator& A, const Vec& b,
                          double lambda, const LassoOptions& opt,
                          const Vec& x_in)
{
    AdmmResult res;
    const double t0 = wall_time_seconds();
    const Index n = A.cols();
    const int    maxiter = std::max(1, opt.maxiter_admm);
    const double stoptol = opt.stoptol;
    const double gamma   = 1.618;

    const Index m = A.rows();

    // diagonal preconditioner for the xi subproblem (I + sigma A A^T):
    // M_ii = 1 + sigma * ||a_i||_2^2.  Preconditioning does not change the
    // solution of CG, only its iteration count, and on ill-conditioned real
    // data it cuts the xi CG iterations by an order of magnitude.  The row
    // norms are cached once per solve.
    const Vec row2 = A.rowNorm2();

    Vec x = (x_in.size() == n) ? x_in : Vec::Zero(n);
    Vec Ax = A.apply(x);
    Vec xi = b - Ax;
    Vec y  = Vec::Zero(n);
    Vec Atxi = A.applyT(xi);
    const double normb = 1.0 + b.norm();

    double sigma = std::max(1e-4, std::min(1.0, lambda));
    const double sigmamax = 1e6, sigmamin = 1e-4;
    int prim_win = 0, dual_win = 0;
    int cg_iters = 0;
    int termcode = 0;
    double primfeas = 0.0, dualfeas = 0.0, eta = 1.0;

    int iter;
    for (iter = 1; iter <= maxiter; ++iter) {
        //---- xi subproblem: (I + sigma A A^T) xi = -(Ax-b) - sigma A y ----
        Vec Ay = A.apply(y);
        Vec rhsxi = -(Ax - b) - sigma * Ay;
        Vec xinew;
        {
            // CG with warm start and diagonal preconditioner (M changes with
            // sigma); the reference admmL1.m uses plain CG -- the
            // preconditioner is an implementation-level acceleration that
            // leaves the converged solution unchanged
            Vec xinew0 = xi;
            auto matvec = [&](const Vec& v, Vec& y) {
                y = v + sigma * A.apply(A.applyT(v));
            };
            Vec diag(m);
            for (Index i = 0; i < m; ++i) diag(i) = 1.0 + sigma * row2(i);
            int cgflag = 0;
            cg_iters += cg_precond(matvec, diag, rhsxi, xinew0, 1e-6,
                                   std::max(100, 2 * static_cast<int>(A.rows())),
                                   std::max(opt.time_limit - (wall_time_seconds() - t0), 1.0),
                                   &cgflag);
            xinew = std::move(xinew0);
        }
        xi = std::move(xinew);
        Atxi = A.applyT(xi);

        //---- y update: projection onto the l_infinity ball ----
        Vec yinput = -Atxi - x / sigma;
        Vec rr;
        proj_inf(yinput, lambda, y, rr);

        //---- multiplier update ----
        Vec Rd = Atxi + y;
        x.noalias() += gamma * sigma * Rd;
        Ax = A.apply(x);

        //---- residuals ----
        Vec Rp = Ax - b + xi;
        primfeas = Rp.norm() / normb;
        dualfeas = Rd.norm() / (1.0 + y.norm());
        const double maxfeas = std::max(primfeas, dualfeas);

        if (maxfeas < 1e2 * stoptol) {
            Vec grad = A.applyT(Ax - b);
            eta = kkt_residual(x, grad, lambda);
            if (eta < stoptol) { termcode = 0; break; }
        }

        //---- adaptive sigma: the reference winning-counter schedule
        // (3/6/12/25/50 x 1.25).  Per-iteration rules (win counters or a
        // Boyd residual-ratio rule) were both tried on real data and both
        // de-stabilise small ill-posed subproblems (T5 single-Lasso runs
        // out of iterations); the sparse trigger schedule keeps sigma
        // nearly fixed where the subproblem is well-behaved and only
        // re-tunes it when the residuals clearly separate. ----
        if (feasratio_trigger(iter)) {
            if (primfeas < dualfeas) { prim_win++; } else { dual_win++; }
            if (prim_win > std::max(1, static_cast<int>(1.2 * dual_win))) {
                prim_win = 0;
                sigma = std::min(sigmamax, sigma * 1.25);
            } else if (dual_win > std::max(1, static_cast<int>(1.2 * prim_win))) {
                dual_win = 0;
                sigma = std::max(sigmamin, sigma / 1.25);
            }
        }

        if (wall_time_seconds() - t0 > opt.time_limit) { termcode = 2; break; }
    }
    if (iter > maxiter) { termcode = 1; }

    Vec grad = A.applyT(Ax - b);
    double obj = 0.5 * (Ax - b).squaredNorm() + lambda * x.lpNorm<1>();

    res.x        = std::move(x);
    res.xi       = std::move(xi);
    res.grad     = std::move(grad);
    res.obj      = obj;
    res.eta      = eta;
    res.primfeas = primfeas;
    res.dualfeas = dualfeas;
    res.iter     = iter - 1;
    res.termcode = termcode;
    res.time     = wall_time_seconds() - t0;
    res.cg_iters = cg_iters;
    return res;
}

} // namespace smop
