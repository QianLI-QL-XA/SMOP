//======================================================================
// adaptive_sieving.hpp -- adaptive sieving for the Lasso subproblem
//
//     min_x  (1/2)||A x - b||_2^2 + lambda ||x||_1
//
// Port of the reference `AdaptSieving_lasso_solve.m` (LevelsetLasso
// folder): solve a sequence of reduced problems on the active set
//   idx = support(x) U {coordinates violating the KKT condition},
// and stop when the relative KKT residual
//   eta = ||g + proj_inf(x - g, lambda)|| / (1 + ||g|| + ||x||)
// drops below `stoptolas`.  The number of indices added per round is
// capped by `control_addpram * n` (auto-tuned by n, as in the reference
// code); the reduced operator is a data-free column view (SubsetOperator).
//
// Solver dispatch:
//   * tall subproblem (m > nr) -> smoothing Newton (accurate), with a
//     30 s budget cap so it cannot stall a whole AS round
//   * high-dimensional / large active sets -> SSNAL (semismooth Newton
//     ALM); its dense LDLT / PCG linear systems are cheap enough after
//     the nested subset fix
//   * ADMM is strictly the last-resort fallback: it runs only when both
//     smoothing Newton and SSNAL fail (or are disabled).
//======================================================================
#pragma once

#include <algorithm>
#include <cmath>

#include "admm_l1.hpp"
#include "operators.hpp"
#include "prox.hpp"
#include "smoothing_newton.hpp"
#include "ssn_l1.hpp"
#include "types.hpp"

namespace smop {

struct AsResult {
    Vec  x;
    Vec  xi;
    Vec  grad;
    int    as_iter      = 0;
    int    iter_smoothing = 0;
    int    iter_ssn     = 0;
    int    iter_admm    = 0;
    double eta          = 0.0;
    double obj          = 0.0;
    std::vector<Index> reducedn;   // active-set size after each round
};

// control_addpram as a function of n (same table as the reference code)
inline double default_control_addpram(Index n, Index m)
{
    double c;
    if      (n < 5000)  c = 0.4;
    else if (n < 1e4)   c = 0.1;
    else if (n < 5e4)   c = 0.04;
    else if (n < 1e5)   c = 0.01;
    else if (n < 1e6)   c = 0.001;
    else                c = 0.001;
    if (static_cast<double>(m) / static_cast<double>(n) > 0.05) c *= 3.0;
    return c;
}

inline AsResult adaptive_sieving(const LinearOperator& A, const Vec& b,
                                 double lambda, const LassoOptions& opt,
                                 const Vec& x0, const Vec& xi0,
                                 const Vec& grad0)
{
    AsResult res;
    const double t0 = wall_time_seconds();
    const Index n = A.cols();
    const Index m = A.rows();
    const int    maxiter_as = std::max(1, opt.maxiter_as);
    const double stoptolas  = opt.stoptolas;
    const double control_addpram =
        (opt.control_addpram > 0) ? opt.control_addpram
                                  : default_control_addpram(n, m);
    const Index add_size_upper_bound =
        std::max<Index>(1, static_cast<Index>(control_addpram * n));

    // ---- warm start ----
    Vec x = (x0.size() == n) ? x0 : Vec::Zero(n);
    Vec xi;
    if (xi0.size() == m) xi = xi0; else xi = b - A.apply(x);
    Vec grad;
    if (grad0.size() == n) grad = grad0;
    else                   grad = A.applyT(A.apply(x) - b);

    // initial KKT residual
    Vec eta_vec;
    double eta = kkt_residual(x, grad, lambda, &eta_vec);

    // initial active set: support + the (up to 10) most violated indices
    std::vector<Index> idx = nonzero_support(x);
    {
        std::vector<Index> viol;
        const double viol_tol = std::max(1e-12, 1e-10 * grad.lpNorm<Eigen::Infinity>());
        for (Index i = 0; i < n; ++i)
            if (std::abs(eta_vec(i)) > viol_tol) viol.push_back(i);
        if (viol.size() > 10) {
            std::partial_sort(viol.begin(), viol.begin() + 10, viol.end(),
                              [&](Index a, Index b) {
                                  return std::abs(eta_vec(a)) > std::abs(eta_vec(b));
                              });
            viol.resize(10);
            // partial_sort leaves the top-10 by |ev| in arbitrary index
            // order; set_union below requires sorted input, otherwise the
            // active set gets duplicates / wrong order and the fill-back of
            // the reduced solution silently corrupts the full solution.
            std::sort(viol.begin(), viol.end());
        }
        std::vector<Index> uni(idx.size() + viol.size());
        auto it = std::set_union(idx.begin(), idx.end(), viol.begin(), viol.end(),
                                 uni.begin());
        uni.resize(static_cast<size_t>(it - uni.begin()));
        idx = std::move(uni);
    }

    double obj = 0.0;
    int as_iter = 0;
    bool run_once = true;
    bool smoothing_broken = false; // after a failed smoothing round, use ADMM
    bool ssn_broken        = false; // after a failed SSNAL round, use ADMM
    int    stall = 0;              // consecutive rounds without active-set growth

    while ((eta > stoptolas && as_iter < maxiter_as) || run_once) {
        run_once = false;
        as_iter++;
        res.reducedn.push_back(static_cast<Index>(idx.size()));

        SubsetOperator Ared(A, idx);
        const Index nr = static_cast<Index>(idx.size());
        Vec xr(nr);
        for (Index j = 0; j < nr; ++j) xr(j) = x(idx[static_cast<size_t>(j)]);

        const bool large =
            (nr > static_cast<Index>(0.5 * m) && m < 300) ||
            (std::min(nr, m) > 5000 && nr < m) ||
            (nr >= 3000);

        // Dispatcher over the three inner solvers.  A column-reduced
        // subproblem with more rows than columns (m > nr) is served best
        // by smoothing Newton (its normal system is nr x nr); the
        // high-dimensional case (nr >= m) -- including large active sets
        // on any m, where the SSN linear systems are cheap dense LDLT
        // solves below m=1500 or preconditioned CG above -- is exactly
        // what SSNAL was designed for.  ADMM is strictly the last-resort
        // fallback: it runs only when both smoothing Newton and SSNAL
        // fail (or are disabled), never as a first choice on big active
        // sets (where its sigma schedule converges far too slowly).
        bool solved = false;
        const bool tall = (m > nr);

        if (tall && opt.use_smoothing && !smoothing_broken) {
            // cap the smoothing budget: on large active sets the smoothed
            // normal system is an expensive dense LDLT, and stalling there
            // for minutes wastes the whole AS round; a 30 s budget either
            // converges or quickly falls through to SSN / ADMM.
            LassoOptions sot = opt;
            sot.time_limit = std::min(opt.time_limit, 30.0);
            SmoothingResult sm = smoothing_newton(Ared, b, lambda, sot, xr);
            res.iter_smoothing += sm.iter;
            if (opt.verbose >= 2)
                std::printf("  [as %2d nr=%4lld] SMOOTH iter=%4d t=%.2fs "
                            "breakyes=%d rel=%.2e\n",
                            as_iter, (long long)nr, sm.iter, sm.time,
                            sm.breakyes, sm.rel_eta);
            if (sm.breakyes == 0 || !sm.x.allFinite()) {
                // remember the failure; restart from the *incoming* warm
                // start, not the diverged smoothing output
                smoothing_broken = true;
            } else {
                xr = std::move(sm.x);
                solved = true;
            }
        }

        if (!solved && opt.use_ssn && !ssn_broken &&
            (!tall || large || smoothing_broken)) {
            SsnResult ss = ssn_l1(Ared, b, lambda, opt, xr);
            res.iter_ssn += ss.iter;
            res.iter_smoothing += ss.itersub;
            if (opt.verbose >= 2)
                std::printf("  [as %2d nr=%4lld] SSN   iter=%4d sub=%4d t=%.2fs "
                            "eta=%.2e\n",
                            as_iter, (long long)nr, ss.iter, ss.itersub,
                            ss.time, ss.eta);
            if (ss.status != 0 || !ss.x.allFinite()) {
                ssn_broken = true;
            } else {
                xr = std::move(ss.x);
                solved = true;
            }
        }

        // ADMM: strictly the last-resort fallback -- only reachable when
        // smoothing Newton and SSNAL both failed or are disabled
        if (!solved) {
            AdmmResult ad = admm_l1(Ared, b, lambda, opt, xr);
            res.iter_admm += ad.iter;
            xr = std::move(ad.x);
            if (opt.verbose >= 2)
                std::printf("  [as %2d nr=%4lld] ADMM iter=%4d t=%.2fs eta=%.2e\n",
                            as_iter, (long long)nr, ad.iter, ad.time, ad.eta);
        }

        // fill back the reduced solution
        for (Index j = 0; j < nr; ++j) x(idx[static_cast<size_t>(j)]) = xr(j);

        Vec Ax = A.apply(x);
        xi = b - Ax;
        grad = -A.applyT(xi);

        // recompute the KKT residual on the full problem
        eta = kkt_residual(x, grad, lambda, &eta_vec);
        obj = 0.5 * Ax.squaredNorm() - Ax.dot(b) + 0.5 * b.squaredNorm()
              + lambda * x.lpNorm<1>();

        bool idx_changed = false;
        if (eta > stoptolas) {
            // find violated coordinates
            // collect only coordinates genuinely far from KKT.  The raw
            // reference threshold 1e-10*||grad||_inf is stricter than the
            // inner-solver tolerance on ill-conditioned real data, so it
            // misclassifies converged in-set coordinates as violations and
            // freezes the active set (endless repeated ADMM solves).
            const double viol_tol =
                std::max(1e-12, 10.0 * stoptolas * (1.0 + grad.lpNorm<Eigen::Infinity>()));
            std::vector<Index> viol;
            viol.reserve(n / 10 + 1);
            for (Index i = 0; i < n; ++i)
                if (std::abs(eta_vec(i)) > viol_tol) viol.push_back(i);

            // reset the active set if it has grown too large
            if ((as_iter > 10 && as_iter % 5 == 0) ||
                static_cast<Index>(idx.size()) > static_cast<Index>(0.3 * n)) {
                idx = nonzero_support(x);
            }

            // merge; cap the number of newly added indices
            std::vector<Index> uni(idx.size() + viol.size());
            auto it = std::set_union(idx.begin(), idx.end(), viol.begin(), viol.end(),
                                     uni.begin());
            uni.resize(static_cast<size_t>(it - uni.begin()));
            const Index newly = static_cast<Index>(uni.size() - idx.size());
            if (newly > add_size_upper_bound) {
                // keep the |add| most violated among the new ones
                std::vector<Index> newones(uni.begin() + static_cast<long long>(idx.size()), uni.end());
                std::partial_sort(newones.begin(),
                                  newones.begin() + static_cast<long long>(add_size_upper_bound),
                                  newones.end(),
                                  [&](Index a, Index b) {
                                      return std::abs(eta_vec(a)) > std::abs(eta_vec(b));
                                  });
                newones.resize(static_cast<size_t>(add_size_upper_bound));
                idx.insert(idx.end(), newones.begin(), newones.end());
                std::sort(idx.begin(), idx.end());
                idx_changed = true;
            } else {
                idx = std::move(uni);
                idx_changed = (newly > 0);
            }

            // stagnation guard: no new indices entered the active set and the
            // KKT residual is not improving -> the subproblem tolerance is at
            // the limit of what the inner solver can reach on this ill-posed
            // data; accept the current solution instead of looping until the
            // AS iteration cap (avoids minutes of repeated identical ADMM
            // solves on real-data problems)
            if (!idx_changed) {
                stall++;
                if (stall >= 3) break;
            } else {
                stall = 0;
            }
        } else {
            stall = 0;
        }

        if (wall_time_seconds() - t0 > opt.time_limit) break;
    }

    res.x        = std::move(x);
    res.xi       = std::move(xi);
    res.grad     = std::move(grad);
    res.as_iter  = as_iter;
    res.eta      = eta;
    res.obj      = obj;
    return res;
}

} // namespace smop
