//======================================================================
// smoothing_newton.hpp -- smoothing-Newton solver for the Lasso subproblem
//
//     min_x  (1/2)||A x - b||_2^2 + lambda ||x||_1
//
// Port of the reference MATLAB routine `smoothingNewton.m` (LevelsetLasso
// folder of the SMOP code): the non-smooth |t| is replaced by a quadratic
// smoothing s_eps(t) and the smoothing parameter eps is driven to zero
// together with x by a semismooth Newton iteration on the merit function
//
//     phi(eps, x) = eps^2 + ||G(eps, x)||^2,
//     G = x - f_eps(x - grad) + g_eps(-x + grad) + kappa |eps| x,
//
// where f_eps, g_eps are the smoothed KKT terms.  The Newton system
//     ((tmp - u) I + diag(u) A^T A) dx = rhs,   u = fx + gx
// is solved by (diagonally preconditioned) CG; the line search is the
// Armijo backtracking of the reference code.
//
// Robustness features:
//   * NaN / non-finite guards on every iteration,
//   * CG with preconditioner, iteration cap and time limit,
//   * fallback to the previous iterate when the line search fails,
//   * `breakyes == 0` signals non-convergence (the caller may fall back
//     to the ADMM solver).
//======================================================================
#pragma once

#include <Eigen/LU>

#include <cmath>
#include <limits>

#include "linsolve.hpp"
#include "operators.hpp"
#include "prox.hpp"
#include "types.hpp"

namespace smop {

//------------------------------------------------------------------------
// s_eps(t): quadratic smoothing of |t|, plus its derivatives.
//   s = |t|          for |t| >= eps/2   (fx = sign(t), fe = 0)
//   s = (t+eps/2)^2/(2eps)              (|t| < eps/2)
//   s = 0            for t <= -eps/2
//------------------------------------------------------------------------
inline void smoothing_fun(double eps, const Vec& t, Vec& s, Vec& fe, Vec& fx)
{
    const Index n = t.size();
    s.resize(n);
    fe.resize(n);
    fx.resize(n);
    eps = std::abs(eps);
    const double half = 0.5 * eps;
    for (Index i = 0; i < n; ++i) {
        const double ti = t(i);
        if (ti >= half) {
            s(i) = ti; fe(i) = 0.0; fx(i) = 1.0;
        } else if (ti <= -half) {
            s(i) = 0.0; fe(i) = 0.0; fx(i) = 0.0;
        } else {
            const double tmp = ti / eps;
            const double tp = ti + half;
            s(i)  = tp * tp / (2.0 * eps);
            fe(i) = 0.125 - 0.5 * tmp * tmp;
            fx(i) = 0.5 + tmp;
        }
    }
}

//------------------------------------------------------------------------
// Merit function phi(eps, x) and its building blocks.
//   xinput = x - grad
//------------------------------------------------------------------------
inline void find_phi(const Vec& xinput, double eps, const Vec& x,
                     double lambda, double kappa,
                     double& phi, Vec& G, Vec& fe, Vec& fx, Vec& ge, Vec& gx)
{
    Vec f = xinput;
    f.array() -= lambda;
    Vec g = -xinput;
    g.array() -= lambda;
    Vec ff, gg;
    smoothing_fun(eps, f, ff, fe, fx);
    smoothing_fun(eps, g, gg, ge, gx);
    G = x - ff + gg;
    G.array() += kappa * std::abs(eps) * x.array();
    phi = eps * eps + G.squaredNorm();
}

//------------------------------------------------------------------------
struct SmoothingResult {
    Vec  x;
    double obj     = 0.0;
    int    iter    = 0;
    int    breakyes = 0;   // 1/2 converged; 0 not converged
    double eta_org = 0.0;
    double rel_eta = 0.0;
    double phi     = 0.0;
    double time    = 0.0;
    int    cg_iters = 0;
};

//------------------------------------------------------------------------
inline SmoothingResult smoothing_newton(const LinearOperator& A,
                                        const Vec& b, double lambda,
                                        const LassoOptions& opt,
                                        const Vec& x_in)
{
    SmoothingResult res;
    const double t0 = wall_time_seconds();
    const Index n = A.cols();
    const double stoptol = opt.stoptol;
    const int    maxiter = std::max(1, opt.maxiter);
    const double eps_hat = (opt.eps_hat > 0) ? opt.eps_hat : 1.0;
    const double kappa   = opt.kappa;
    const double rho     = (opt.rho > 0 && opt.rho < 1) ? opt.rho : 0.7;
    const double sigma   = opt.armijo_sigma;
    const double eta_hat = opt.eta_hat;
    const double r       = 0.25 / std::max(1.0, eps_hat);
    const double delta   = std::sqrt(2.0) * std::max(eta_hat, r * eps_hat);

    Vec x = (x_in.size() == n) ? x_in : Vec::Zero(n);
    Vec Ax = A.apply(x);
    Vec grad = A.applyT(Ax - b);


    // initial (relative) KKT residual
    Vec evec;
    double eta_org = 0.0, rel_eta = 0.0;
    {
        Vec pg = proj_inf(x - grad, lambda);
        Vec ev = grad + pg;
        eta_org = ev.norm();
        rel_eta = eta_org / (1.0 + grad.norm() + x.norm());
    }

    double epsilon = eps_hat, epsilon0 = epsilon;
    Vec x0 = x;
    double phi = 0.0; Vec G, fe, fx, ge, gx;
    find_phi(x - grad, epsilon, x, lambda, kappa, phi, G, fe, fx, ge, gx);

    int iter = 0, breakyes = 0;
    int cg_iters = 0;
    int no_progress = 0;   // consecutive iterations without merit decrease

    while (iter < maxiter) {
        if (epsilon < 1e-13 || !std::isfinite(phi) ||
            !G.allFinite() || !fe.allFinite() || !gx.allFinite()) break;

        iter++;
        const double theta   = r * std::min(1.0, phi);
        const double delta_eps = -epsilon + eps_hat * theta;
        Vec rhs = -G - delta_eps * (kappa * ((epsilon >= 0) ? 1.0 : -1.0) * x - fe + ge);

        Vec u = fx + gx;
        const double tmp = 1.0 + kappa * std::abs(epsilon);
        Vec dx;
        if (u.maxCoeff() <= 0.0) {
            dx = rhs / tmp;
        } else {
            // plain CG with a zero start, exactly as in the reference
            // smoothingNewton.m: no preconditioner, maxit = max(100, 4n),
            // tol = 1e-8.  (A preconditioned / dense-direct solve changes the
            // iteration path and can converge to a different -- non-sparse --
            // solution of the ill-posed subproblem, which then makes the
            // adaptive-sieving active set grow and stalls the whole solve.)
            int cgflag = 0;
            Vec atav(n);   // reused across CG iterations
            auto matvec = [&](const Vec& v, Vec& y) {
                atav = A.applyT(A.apply(v));
                y.resize(n);
                for (Index i = 0; i < n; ++i)
                    y(i) = (tmp - u(i)) * v(i) + u(i) * atav(i);
            };
            Vec dx0;
            cg_iters += cg_precond(matvec, Vec(), rhs, dx0, 1e-8,
                                   std::max(100, 4 * static_cast<int>(n)),
                                   std::max(opt.time_limit - (wall_time_seconds() - t0), 1.0),
                                   &cgflag);
            dx = std::move(dx0);
        }

        // ---- Armijo line search (backtracking, rho^k) ----
        const int maxitline = 30;
        const double phi0 = phi;
        Vec Adx = A.apply(dx);
        Vec ATAdx = A.applyT(Adx);
        double alp = 1.0;
        bool accept = false;
        Vec xnew, xinnew, G2, fe2, fx2, ge2, gx2;
        double phi_new = phi;
        for (int k_inner = 0; k_inner < maxitline; ++k_inner) {
            alp = std::pow(rho, k_inner);
            xnew = x0 + alp * dx;
            const double epsnew = epsilon0 + alp * delta_eps;
            Vec graddelta = grad + alp * ATAdx;
            xinnew = xnew - graddelta;
            find_phi(xinnew, epsnew, xnew, lambda, kappa,
                     phi_new, G2, fe2, fx2, ge2, gx2);
            if (phi_new <= (1.0 - 2.0 * sigma * (1.0 - delta) * alp) * phi0) {
                accept = true;
                break;
            }
            if (!std::isfinite(phi_new)) break;
        }
        if (!accept) {
            // smallest step of the backtracking: keep the last trial point
            alp = std::pow(rho, maxitline - 1);
            xnew = x0 + alp * dx;
            epsilon = epsilon0 + alp * delta_eps;
            Vec graddelta = grad + alp * ATAdx;
            find_phi(xnew - graddelta, epsilon, xnew, lambda, kappa,
                     phi_new, G2, fe2, fx2, ge2, gx2);
            // refuse a strongly worsening point (robustness guard)
            if (phi_new > 2.0 * phi0 && alp < 1e-6) {
                breakyes = 0;
                x = x0; // stay at the previous accepted point
                break;
            }
        } else {
            epsilon = epsilon0 + alp * delta_eps;
        }

        // ---- update ----
        x  = xnew;
        Ax.noalias() += alp * Adx;
        grad = A.applyT(Ax - b);
        G = G2; fe = fe2; fx = fx2; ge = ge2; gx = gx2;
        phi = phi_new;

        // robustness: refuse a diverged iterate (the reference has no norm
        // guard; a smoothed sub-problem can otherwise blow up and poison the
        // downstream fallback paths)
        const double xnorm_scale = std::max(1.0, x_in.norm());
        if (!x.allFinite() || x.norm() > 1e10 * xnorm_scale) {
            breakyes = 0;
            x = x0;
            break;
        }

        Vec pg = proj_inf(x - grad, lambda);
        Vec ev = grad + pg;
        eta_org = ev.norm();
        rel_eta = eta_org / (1.0 + grad.norm() + x.norm());

        if (rel_eta < stoptol) { breakyes = 1; break; }
        if (std::sqrt(phi) < stoptol * 1e-2 || epsilon < 1e-13) { breakyes = 2; break; }

        // stagnation guard: a smoothed sub-problem that stops making merit
        // progress will never converge within the budget -- break out early
        // so the caller falls back to ADMM instead of burning minutes in CG
        if (phi_new >= phi0 * 0.999) {
            no_progress++;
            if (no_progress >= 8) { breakyes = 0; break; }
        } else {
            no_progress = 0;
        }

        x0 = x;
        epsilon0 = epsilon;

        if (wall_time_seconds() - t0 > opt.time_limit) { breakyes = 0; break; }
    }

    // final objective
    double obj = 0.5 * (Ax - b).squaredNorm() + lambda * x.lpNorm<1>();

    res.x        = std::move(x);
    res.obj      = obj;
    res.iter     = iter;
    res.breakyes = breakyes;
    res.eta_org  = eta_org;
    res.rel_eta  = rel_eta;
    res.phi      = phi;
    res.time     = wall_time_seconds() - t0;
    res.cg_iters = cg_iters;
    return res;
}

} // namespace smop
