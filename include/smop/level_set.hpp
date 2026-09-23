//======================================================================
// level_set.hpp -- outer level-set method for the BMOP
//
//     min_x  ||x||_1   s.t.   ||A x - b||_2 <= delta
//
// Port of the reference `level_set_lasso.m` (LevelsetLasso folder):
// find the root mu* of the univariate nonsmooth equation
//     phi(mu) = psi(mu) - delta = 0,
//     psi(mu) = ||A x(mu) - b||_2,  x(mu) = argmin 1/2||Ax-b||^2 + mu||x||_1.
//
// The bracket [mu0, muinf] (psi(mu0) <= delta <= psi(muinf)) is shrunk
// each round; the root-finding step is chosen among
//   * secant (default, superlinear once phi is smooth),
//   * Newton (analytic phi' from the active set, optional),
//   * bisection (fallback whenever the fast step is invalid).
// The subproblem is solved by the adaptive-sieving solver (or directly by
// the Lasso solvers when use_as = false), with warm start and tightened
// tolerances as the bracket narrows, exactly as in the reference code.
//
// Robustness features (all inherited from the reference MATLAB code):
//   * every fast step is validated against the bracket and reverted to
//     bisection on violation,
//   * stagnation detection shrinks the bisection fraction mucont,
//   * when the bracket becomes inconsistent, historical (mu, psi) points
//     are used to relocate mu0 / muinf,
//   * NaN/Inf guards, iteration caps and a wall-clock time limit.
//======================================================================
#pragma once

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

#include "adaptive_sieving.hpp"
#include "admm_l1.hpp"
#include "linsolve.hpp"
#include "operators.hpp"
#include "prox.hpp"
#include "smoothing_newton.hpp"
#include "types.hpp"

namespace smop {

//----------------------------------------------------------------------
// bisection fraction mucont (port of solvers/updatemucont.m)
//----------------------------------------------------------------------
inline double update_mucont(double ratio_lower, double ratio_upper,
                            bool exist_initialmu, double ratio_value)
{
    if (exist_initialmu) {
        if (ratio_lower == 0.0) {
            if (ratio_upper > 10)  return 0.05;
            if (ratio_upper > 5)   return 0.2;
            if (ratio_upper > 2)   return 0.3;
            if (ratio_upper > 1.6) return 0.5;
            if (ratio_upper > 1.2) return 0.6;
            if (ratio_upper > 1.06) return 0.7;
            if (ratio_upper > 1.05) return 0.8;
            return 0.9;
        }
        const double s = ratio_upper + ratio_lower;
        if (s > 5) return 0.3;
        if (s > 3) return 0.4;
        if (s > 2) return 0.5;
        if (s > 1) return 0.6;
        if (s > 0) return 0.7;
        return 0.8;
    }
    if (ratio_value > 5) return 0.2;
    if (ratio_value > 3) return 0.4;
    if (ratio_value > 2) return 0.4;
    return 0.5;
}

//----------------------------------------------------------------------
// full-dimensional Lasso solve (used when adaptive sieving is disabled)
//----------------------------------------------------------------------
inline LassoResult solve_lasso_full(const LinearOperator& A, const Vec& b,
                                    double lambda, const LassoOptions& opt,
                                    const Vec& x0)
{
    LassoResult res;
    Vec x = (x0.size() == A.cols()) ? x0 : Vec::Zero(A.cols());
    const bool large = (A.cols() >= 3000 || !opt.use_smoothing);
    if (large) {
        AdmmResult ad = admm_l1(A, b, lambda, opt, x);
        res.x = std::move(ad.x);
        res.xi = std::move(ad.xi);
        res.grad = std::move(ad.grad);
        res.info.iter = ad.iter;
        res.info.eta = ad.eta;
        res.info.obj = ad.obj;
        res.info.time = ad.time;
        res.info.status = (ad.termcode == 0) ? 0 : ad.termcode;
        res.info.breakyes = (ad.termcode == 0) ? 1 : 0;
    } else {
        SmoothingResult sm = smoothing_newton(A, b, lambda, opt, x);
        // gate the smoothing output by the *true* KKT residual, not just the
        // internal break flag: a smoothed problem may declare convergence
        // while the non-smooth KKT residual is still large (robustness).
        Vec sm_grad = A.applyT(A.apply(sm.x) - b);
        double sm_eta = 0.0;
        if (sm.x.allFinite() && sm_grad.allFinite()) {
            sm_eta = kkt_residual(sm.x, sm_grad, lambda);
        }
        const double eta_gate = std::max(opt.stoptol, 1e-5);
        if (sm.breakyes == 0 || !sm.x.allFinite() || sm_eta > eta_gate) {
            // restart ADMM from the *incoming* warm start: the diverged
            // smoothing output must not feed the fallback (robustness)
            AdmmResult ad = admm_l1(A, b, lambda, opt, x);
            res.x = std::move(ad.x);
            res.xi = std::move(ad.xi);
            res.grad = std::move(ad.grad);
            res.info.iter = ad.iter;
            res.info.eta = ad.eta;
            res.info.obj = ad.obj;
            res.info.time = ad.time;
            res.info.status = (ad.termcode == 0) ? 0 : ad.termcode;
            res.info.breakyes = (ad.termcode == 0) ? 1 : 0;
        } else {
            Vec Ax = A.apply(sm.x);
            res.x = std::move(sm.x);
            res.xi = b - Ax;
            res.grad = A.applyT(Ax - b);
            res.info.iter = sm.iter;
            res.info.eta = sm.rel_eta;
            res.info.etaorg = sm.eta_org;
            res.info.obj = sm.obj;
            res.info.time = sm.time;
            res.info.status = 0;
            res.info.breakyes = sm.breakyes;
        }
    }
    return res;
}

//----------------------------------------------------------------------
// main entry: solve the BMOP by the level-set method
//----------------------------------------------------------------------
inline SmopResult solve_bmop(const LinearOperator& A, const Vec& b,
                             double delta, const SmopOptions& opt,
                             const Vec& x0 = Vec())
{
    SmopResult out;
    SmopInfo&  info = out.info;
    const double t0 = wall_time_seconds();
    const Index n = A.cols();
    const double stoptol = opt.stoptol;
    const int maxiter_levelset = std::max(1, opt.maxiter_levelset);
    const double normb = b.norm();

    // trivial case: x = 0 is already feasible
    if (normb <= delta) {
        out.x = Vec::Zero(n);
        out.xi = b;
        out.mu = 0.0;
        info.psi = normb;
        info.eta = std::abs(normb - delta) / std::max(1.0, delta);
        info.status = 0;
        info.msg = "x = 0 is feasible";
        return out;
    }

    const double mumax = A.applyT(b).lpNorm<Eigen::Infinity>();
    double mu0 = (opt.mu0 > 0) ? opt.mu0 : 0.0;
    double muinf = (opt.muinf > 0) ? opt.muinf : mumax;
    if (muinf <= mu0) muinf = mumax;
    double mudiff = muinf - mu0;

    double ratio_lower = 0.0, ratio_upper = normb / delta;
    double psi_lower = -100.0;                // (psi(mu0)-delta); -100 = undefined
    double psi_upper = normb - delta;         // psi(muinf) - delta

    double mucont = update_mucont(ratio_lower, ratio_upper, false, ratio_upper);
    double mu;
    if (opt.initial_mu > 0) {
        mu = opt.initial_mu;
    } else {
        mu = std::max(2.0 * delta * std::sqrt(2.0 * std::log(static_cast<double>(n))),
                      mucont * mudiff + mu0);
        if (mu >= muinf) mu = mucont * mudiff + mu0;
    }

    // warm start
    Vec x = (x0.size() == n) ? x0 : Vec::Zero(n);
    Vec xi, grad;
    double psi = 0.0;
    if (x.squaredNorm() > 0.0) {
        Vec Ax = A.apply(x);
        xi = b - Ax;
        psi = xi.norm();
        grad = -A.applyT(xi);
    } else {
        xi = b;
        psi = normb;
        grad = -A.applyT(b);
    }

    if (opt.verbose >= 1)
        std::printf("  [level-set] n=%lld mumax=%.4e mu0=%.4e muinf=%.4e "
                    "mu_init=%.4e\n",
                    (long long)n, mumax, mu0, muinf, mu);

    double abseta = std::abs(psi - delta);
    double eta = abseta / std::max(1.0, delta);

    // history
    std::vector<std::pair<double, double>> run_mu_psi; // (mu, psi)
    std::vector<double> run_ratio;

    int iter = 0, iter_bisection = 0, iter_ns = 0;
    int iter_smoothing = 0, iter_admm = 0;
    int as_iter_last = -1;
    double last_mu = mu, last_h = psi - delta; // secant second point
    bool mustusebisection = false;

    const bool use_as  = opt.use_as;
    const bool use_sec = opt.use_secant && !opt.use_newton; // mutually exclusive
    const bool use_new = opt.use_newton;
    const double t_limit = opt.time_limit;
    int status = 0;

    // fast-step validator: true => revert to bisection
    auto validate_fast_step = [&](double mu_new, double phi_prime) {
        if (!std::isfinite(mu_new) || !std::isfinite(phi_prime) || phi_prime <= 0.0)
            return true;
        if (mu_new < 0.0 || mu_new < mu0 * 0.99) return true;
        if (mu_new > 1.01 * muinf && eta > 1e-6) return true;
        if (as_iter_last <= 0) return true;
        if (mu_new < 0.35 * (mu0 + muinf) && eta > 5e-3) return true;
        if (mu0 < 1e-10 && muinf <= 2e-3 && eta < 2e-2 && mu_new < 0.4 * muinf && eta > 1e-2)
            return true;
        if (eta > 0.04 && eta < 0.2 && mu_new < 0.035 && mu_new < 0.37 * mudiff + mu0)
            return true;
        return false;
    };

    while (eta > stoptol && iter < maxiter_levelset) {
        iter++;

        // ---- tighten the subproblem tolerances as the bracket narrows ----
        // (same schedule as the reference: one tolerance for both the
        // adaptive-sieving gate and the inner Lasso solvers; a stricter
        // smoothing stoptol than stoptolas makes the inner Newton stall on
        // ill-conditioned real-data subproblems and burn time in ADMM)
        LassoOptions sub = opt.lasso;
        sub.verbose = opt.verbose;   // propagate outer verbosity to the AS loop
        if (mudiff > 1e-3) {
            sub.stoptolas = 1e-6;
            sub.stoptol   = 1e-6;
        } else {
            sub.stoptolas = std::max(std::min(1e-8, stoptol * 1e-2), 1e-10);
            sub.stoptol   = std::max(std::min(1e-8, stoptol * 1e-2), 1e-10);
        }
        // the subproblem must never outlive the outer time budget
        {
            const double remain = opt.time_limit - (wall_time_seconds() - t0);
            const double cap = (remain > 0) ? std::max(1.0, remain) : 1.0;
            if (sub.time_limit <= 0.0 || sub.time_limit > cap) sub.time_limit = cap;
        }

        // ---- update the bisection fraction ----
        double ratio_value = (psi > delta) ? ratio_upper : ratio_lower;
        if (eta > std::max(stoptol, 5e-4)) {
            mucont = update_mucont(ratio_lower, ratio_upper, false, ratio_value);
            if (iter > 4 && run_ratio.size() >= 4) {
                double mn = run_ratio[run_ratio.size() - 4], mx = mn;
                for (size_t k = run_ratio.size() - 4; k < run_ratio.size(); ++k) {
                    mn = std::min(mn, run_ratio[k]);
                    mx = std::max(mx, run_ratio[k]);
                }
                if (mx > 0 && mn / mx > 0.97 && mn > 1.0) {
                    mucont = 0.5;
                } else if (mx > 0 && mn / mx > 0.97 && mn < 1.0) {
                    mucont = (ratio_value < 0.9) ? 0.4 : 0.5;
                }
            }
        } else {
            mucont = 0.5;
        }

        // ---- pick the next mu ----
        const bool fast_eligible =
            (eta < 1e-2 || mudiff / (muinf + mu0) < 1e-2 || ratio_value < 1.2);
        bool did_fast = false;

        if (fast_eligible && (use_new || (use_sec && iter > 2))) {
            if (use_new && !mustusebisection) {
                // Newton step: phi'(mu) = mu * betaK / psi,
                // betaK = uK^T (A_K^T A_K)^+ uK,  u = A^T xi / mu
                const double h = psi - delta;
                Vec atxi = A.applyT(xi);
                Vec u = atxi / mu;
                std::vector<Index> K = nonzero_support(x, 1e-12);
                if (!K.empty() && psi > 0.0) {
                    SubsetOperator AK(A, K);
                    Vec uK(static_cast<Index>(K.size()));
                    for (size_t j = 0; j < K.size(); ++j) uK(Index(j)) = u[K[j]];
                    Vec z = normal_solve(AK, uK, 1e-10);
                    const double betaK = uK.dot(z);
                    const double grad_n = mu * betaK / psi;
                    const double mu_new = mu - h / grad_n;
                    if (!validate_fast_step(mu_new, grad_n)) {
                        mu = mu_new;
                        did_fast = true;
                        iter_ns++;
                    }
                }
            }
            if (!did_fast && use_sec) {
                if (!mustusebisection) {
                    // secant step on (mu, psi) vs (last_mu, last_h)
                    const double h = psi - delta;
                    const double denom = mu - last_mu;
                    if (std::abs(denom) > 1e-300) {
                        const double grad_sec = (h - last_h) / denom;
                        const double mu_new = mu - h / grad_sec;
                        if (!validate_fast_step(mu_new, grad_sec)) {
                            last_mu = mu;
                            last_h = h;
                            mu = mu_new;
                            did_fast = true;
                            iter_ns++;
                        }
                    }
                } else {
                    mustusebisection = false;
                }
            }
        }
        if (!did_fast) {
            last_mu = mu;
            last_h = psi - delta;
            mu = mu0 + mucont * mudiff;
            iter_bisection++;
        }

        // ---- solve the subproblem at mu ----
        LassoResult subres;
        if (use_as) {
            AsResult as = adaptive_sieving(A, b, mu, sub, x, xi, grad);
            iter_smoothing += as.iter_smoothing;
            iter_admm += as.iter_admm;
            as_iter_last = as.as_iter;
            for (Index r : as.reducedn) info.reducedn.push_back(r);
            x = std::move(as.x);
            xi = std::move(as.xi);
            grad = std::move(as.grad);
        } else {
            LassoResult lr = solve_lasso_full(A, b, mu, sub, x);
            if (lr.info.breakyes > 0) iter_smoothing += lr.info.iter;
            else                      iter_admm += lr.info.iter;
            as_iter_last = -1;
            x = std::move(lr.x);
            xi = std::move(lr.xi);
            grad = std::move(lr.grad);
        }

        // ---- update the bracket ----
        psi = xi.norm();
        if (psi > delta) {
            muinf = mu;
            psi_upper = psi - delta;
            ratio_upper = psi / delta;
        } else {
            mu0 = mu;
            psi_lower = psi - delta;
            ratio_lower = psi / delta;
        }
        mudiff = muinf - mu0;
        abseta = std::abs(psi - delta);
        eta = abseta / std::max(1.0, delta);

        run_mu_psi.emplace_back(mu, psi);
        run_ratio.push_back((psi > delta) ? ratio_upper : ratio_lower);
        info.mupath.push_back(mu);
        info.psipath.push_back(psi);

        // ---- bracket repair using the history ----
        const bool change_lower = (psi_lower > 0.0);
        const bool change_upper = (psi_upper < 0.0);
        if ((psi_lower * psi_upper > 0.0 || mudiff < 1e-13) && eta > stoptol) {
            if (change_lower) {
                double besth = -1e300, bestmu = 0.0;
                bool found = false;
                for (const auto& p : run_mu_psi) {
                    const double h = p.second - delta;
                    if (h < 0.0 && h > besth) { besth = h; bestmu = p.first; found = true; }
                }
                if (found) {
                    mu0 = bestmu;
                    psi_lower = besth;
                    ratio_lower = (besth + delta) / delta;
                } else {
                    mu0 = 0.0;
                    psi_lower = -100.0;
                    ratio_lower = 0.0;
                }
            }
            if (change_upper) {
                double besth = 1e300, bestmu = 0.0;
                bool found = false;
                for (const auto& p : run_mu_psi) {
                    const double h = p.second - delta;
                    if (h > 0.0 && h < besth) { besth = h; bestmu = p.first; found = true; }
                }
                if (found) {
                    muinf = bestmu;
                    psi_upper = besth;
                    ratio_upper = (besth + delta) / delta;
                } else {
                    muinf = mumax;
                    psi_upper = normb - delta;
                    ratio_upper = normb / delta;
                }
            }
            mudiff = muinf - mu0;
        }

        if (opt.verbose >= 1) {
            std::printf("  iter %3d | mu=%9.3e psi=%9.3e eta=%8.1e | [mu0=%9.3e muinf=%9.3e] | %s\n",
                        iter, mu, psi, eta, mu0, muinf,
                        (did_fast ? (use_new ? "Newton" : "Secant") : "Bisection"));
        }

        // ---- stall detection: the bracket keeps shrinking but psi has
        // stopped moving (the subproblem hits its numerical floor), so
        // further bisection rounds are pure waste; report it instead of
        // spinning up to maxiter (occurs on deltas below the reachable
        // residual, e.g. 1% noise radius on pyrim/triazines)
        if (eta > stoptol && run_mu_psi.size() >= 25) {
            const double ps_now  = run_mu_psi.back().second;
            const double ps_prev = run_mu_psi[run_mu_psi.size() - 25].second;
            const double rel = std::abs(ps_now - ps_prev) /
                               std::max(1e-300, std::abs(ps_now));
            if (rel < 1e-10 && mu < 1e-8 * std::max(1.0, mumax)) {
                status = 3;
                info.msg = "stalled: psi(mu) plateau above delta (delta unreachable for this data)";
                break;
            }
        }

        if (wall_time_seconds() - t0 > t_limit) {
            status = 2;
            info.msg = "time limit reached";
            break;
        }
        if (!std::isfinite(eta) || !std::isfinite(psi)) {
            status = 3;
            info.msg = "non-finite iterate";
            break;
        }
    }

    if (iter >= maxiter_levelset && eta > stoptol) {
        status = 1;
        info.msg = "max level-set iterations reached";
    }
    if (status == 0 && eta <= stoptol) info.msg = "converged";

    out.x  = std::move(x);
    out.xi = std::move(xi);
    out.y  = std::move(grad);
    out.mu = mu;
    info.iter = iter;
    info.iter_bisection = iter_bisection;
    info.iter_newton_or_secant = iter_ns;
    info.iter_smoothing = iter_smoothing;
    info.iter_admm = iter_admm;
    info.eta = eta;
    info.abseta = abseta;
    info.psi = psi;
    info.mu = mu;
    info.time = wall_time_seconds() - t0;
    info.status = status;
    return out;
}

} // namespace smop
