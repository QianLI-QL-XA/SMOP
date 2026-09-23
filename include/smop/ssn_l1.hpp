//======================================================================
// ssn_l1.hpp -- SSNAL: semismooth Newton augmented Lagrangian method
// for the Lasso subproblem
//
//     min_x  (1/2)||A x - b||_2^2 + lambda ||x||_1
//
// Port of the reference MATLAB code (ClassicLasso folder):
//   Classic_Lasso_SSNAL.m / _main.m / Classic_Lasso_SSNCG.m /
//   Classic_Lasso_linsys_solver.m / Classic_Lasso_SSNAL_Wrapper.m
// (Xudong Li, Defeng Sun, Kim-Chuan Toh, SIAM J. Optim. 28 (2018)).
//
// Structure (dual view):
//   (D) max { -1/2 ||xi||^2 + <b,xi> | A^T xi + y = 0, ||y||_inf <= lambda }
//   ALM outer loop:          solve the xi/y saddle point of L_sigma, then
//                            x <- -sigma * (yinput - y), update sigma by the
//                            primal/dual winning counters.
//   SSN inner loop (SSNCG):  the xi-subproblem is solved by a semismooth
//                            Newton iteration on the projection operator
//                            proj_inf; the Newton direction solves
//                            (I + sigma A_pp A_pp^T) dxi = -GradLxi over the
//                            free coordinates pp = { i : |y_i| < lambda },
//                            with a backtracking/bisection line search.
//   Linear systems:          dense LDLT when m <= 1500, otherwise a reduced
//                            (I/sigma + A_pp^T A_pp) solve when the free set
//                            is small, otherwise preconditioned CG.
//
// The reference code is designed for the high-dimensional setting
// (m << n); the level-set outer loop feeds it column-reduced active sets
// where nr >= m, and the AS dispatcher prefers smoothing Newton when
// m > nr instead (see adaptive_sieving.hpp).
//======================================================================
#pragma once

#ifdef SMOP_USE_CUDA
#include "gpu_operator.hpp"
#endif

#include <algorithm>
#include <cmath>

#include "admm_l1.hpp"      // feasratio_trigger
#include "linsolve.hpp"
#include "operators.hpp"
#include "prox.hpp"
#include "types.hpp"

namespace smop {

struct SsnResult {
    Vec  x;         // primal solution
    Vec  xi;        // dual slack (m)
    Vec  y;         // dual variable (n)
    Vec  grad;      // A^T (A x - b)
    double obj      = 0.0;
    double eta      = 0.0;     // relative KKT residual
    double primfeas = 0.0;
    double dualfeas = 0.0;
    int    iter     = 0;       // ALM outer iterations
    int    itersub  = 0;       // total SSN inner iterations
    int    cg_iters = 0;
    int    status   = 0;       // 0 ok, 1 max iter, 2 time limit, 3 failure
    int    breakyes = 0;       // 1 converged
    double time     = 0.0;
};

//----------------------------------------------------------------------
// Materialise the columns of a column-subset operator into a dense
// m x sp matrix.  When the subset wraps a sparse matrix this is a direct
// per-column sparse copy (O(nnz of the selected columns)); the generic
// fallback builds the columns by apply() of unit vectors.  The unit-vector
// path scans *every* selected column for each of the sp unit vectors, i.e.
// O(sp^2 * nnz_per_col) -- on triazines-size active sets (~7000 columns)
// that is billions of wasted operations per Newton direction, so the
// sparse copy matters (SSNAL linsys was the dominant cost there).
//----------------------------------------------------------------------
inline void subset_dense_columns(const LinearOperator& A, Mat& C)
{
    const Index m = A.rows();
    const Index sp = A.cols();
    C.setZero(m, sp);
    if (const auto* sub = dynamic_cast<const SubsetOperator*>(&A)) {
        if (const SpMat* spm = sub->sparse_matrix()) {
            const std::vector<Index>& cols = sub->columns();
            for (Index j = 0; j < sp; ++j)
                for (SpMat::InnerIterator it(*spm, cols[static_cast<size_t>(j)]); it; ++it)
                    C(it.row(), j) = it.value();
            return;
        }
        if (const Mat* dm = sub->dense_matrix()) {
            const std::vector<Index>& cols = sub->columns();
            for (Index j = 0; j < sp; ++j) C.col(j) = dm->col(cols[static_cast<size_t>(j)]);
            return;
        }
    }
    Vec e(sp);
    for (Index j = 0; j < sp; ++j) { e.setZero(); e(j) = 1.0; C.col(j) = A.apply(e); }
}

//----------------------------------------------------------------------
// Linear solver for the Newton direction:
//   solve (I + sigma A_pp A_pp^T) dxi = rhs   (free set pp = {rr==0})
// Dense direct LDLT when m is small; reduced (I/sigma + A_pp^T A_pp)
// solve when the free set is small; PCG otherwise.  Mirror of
// Classic_Lasso_linsys_solver.m with the sparse-path branches dropped
// (our operators are dense subset views).
//----------------------------------------------------------------------
inline Vec ssn_linsys_solve(const LinearOperator& A, const Vec& rr, double sigma,
                            const Vec& rhs, double time_limit,
                            int* solve_ok, int* cg_iters)
{
    const Index m = A.rows();
    std::vector<Index> pp;
    for (Index i = 0; i < rr.size(); ++i)
        if (rr(i) == 0.0) pp.push_back(i);
    const Index sp = static_cast<Index>(pp.size());
    *solve_ok = 1;
    if (sp < 1) return rhs;            // fully constrained: dxi = rhs

    SubsetOperator App(A, pp);

    // dense direct: M = I + sigma * App App^T (m x m)
    if (m <= 1500) {
        Mat C;
        subset_dense_columns(App, C);
        Mat M = Mat::Identity(m, m) + sigma * (C * C.transpose());
        Eigen::LDLT<Mat> ldlt(M);
        return ldlt.solve(rhs);
    }

    // reduced direct: M = I/sigma + App^T App (sp x sp); dxi = rhs - App tmp
    if (sp <= static_cast<Index>(0.7 * m) && sp <= 10000) {
        Mat C;
        subset_dense_columns(App, C);
        Mat G;
#ifdef SMOP_USE_CUDA
        // normal matrix on the GPU (cuBLAS gemm) when a device is present
        if (!smop_gpu_xtx(C, G)) G = C.transpose() * C;
#else
        G = C.transpose() * C;
#endif
        Mat M = G + (1.0 / sigma) * Mat::Identity(sp, sp);
        Eigen::LDLT<Mat> ldlt(M);
        Vec tmp = ldlt.solve(C.transpose() * rhs);
        return rhs - C * tmp;
    }

    // preconditioned CG on the m-dimensional system
    Vec row2 = App.rowNorm2();
    Vec diag(m);
    for (Index i = 0; i < m; ++i) diag(i) = 1.0 + sigma * row2(i);
    Vec dxi(rhs);
    std::vector<Index> rel(static_cast<size_t>(sp));
    for (Index j = 0; j < sp; ++j) rel[static_cast<size_t>(j)] = j;
    auto matvec = [&](const Vec& v, Vec& y) {
#ifdef SMOP_USE_CUDA
        // column-subset products on the GPU dense matrix when available
        Vec av;
        if (App.subset_gemv(rel, true, v, av)) {
            Vec u;
            if (App.subset_gemv(rel, false, av, u)) { y = v + sigma * u; return; }
        }
#endif
        y = v + sigma * App.apply(App.applyT(v));
    };
    int flag = 0;
    *cg_iters += cg_precond(matvec, diag, rhs, dxi, 1e-6,
                            std::max(100, 2 * static_cast<int>(m)),
                            time_limit, &flag);
    *solve_ok = (flag <= 0) ? 1 : -1;
    return dxi;
}

//----------------------------------------------------------------------
// SSN inner iteration (port of Classic_Lasso_SSNCG.m).  Updates y, xi,
// Atxi, ytmp; returns the inner iteration count and a break code
// (-1 good termination; 1/11 stagnation; 12 time limit).
//----------------------------------------------------------------------
struct SsnSubOut {
    Vec  y, xi, Atxi, ytmp, rr;
    int  itersub  = 0;
    int  breakyes = 0;
    int  cg_iters = 0;
    double Ly     = 0.0;
};

inline SsnSubOut ssn_subproblem(const LinearOperator& A, const Vec& b, double ld,
                                const Vec& x0, const Vec& Atxi0, Vec xi0,
                                const Vec& y0, double sigma,
                                const LassoOptions& opt, double t0)
{
    SsnSubOut o;
    const double stoptol = opt.stoptol;
    const int    maxitersub = 50;

    // initial projection and multiplier residual
    Vec yinput = -Atxi0 - x0 / sigma;
    proj_inf(yinput, ld, o.y, o.rr);
    o.ytmp = yinput - o.y;
    o.xi = xi0;
    o.Atxi = Atxi0;

    double Ly0 = b.dot(o.xi) - 0.5 * o.xi.squaredNorm()
               - 0.5 * sigma * o.ytmp.squaredNorm();
    o.Ly = Ly0;

    std::vector<double> hist_priminf;
    std::vector<double> hist_dualinf;
    std::vector<int>    hist_solveok;
    hist_priminf.reserve(maxitersub);
    hist_dualinf.reserve(maxitersub);
    hist_solveok.reserve(maxitersub);

    int itersub;
    for (itersub = 1; itersub <= maxitersub; ++itersub) {
        Vec Rdz = o.Atxi + o.y;
        double normRd = Rdz.norm();
        Vec msigAytmp = -sigma * A.apply(o.ytmp);
        Vec GradLxi = -(o.xi - b + msigAytmp);
        double normGradLxi = GradLxi.norm();
        double dualinf_sub = normRd / (1.0 + o.y.norm());
        double priminf_sub = normGradLxi;
        double tolsubconst = (std::max(priminf_sub, dualinf_sub) < stoptol) ? 0.9 : 0.05;
        double tolsub = std::max(std::min(1.0, 0.5 * dualinf_sub),
                                 tolsubconst * stoptol);

        if (normGradLxi < tolsub && itersub > 1) { o.breakyes = -1; break; }

        hist_priminf.push_back(priminf_sub);
        hist_dualinf.push_back(dualinf_sub);

        // stagnation guards (simplified core rules of the reference)
        if (itersub > 4) {
            const int k = static_cast<int>(hist_priminf.size());
            double mn = 1e300, mx = 0.0;
            for (int j = std::max(0, k - 4); j < k; ++j) {
                mn = std::min(mn, hist_priminf[j]);
                mx = std::max(mx, hist_priminf[j]);
            }
            bool allbad = true;
            for (int j = std::max(0, k - 4); j < k; ++j)
                if (hist_solveok[j] > -1) allbad = false;
            if (allbad && mx > 0.0 && mn / mx > 0.9 && mx < 5.0 * stoptol) {
                o.breakyes = 1; break;
            }
            if (itersub >= 10) {
                int cnt = 0; double sum = 0.0;
                for (int j = 2; j < itersub; ++j)
                    if (hist_dualinf[j] > hist_dualinf[j - 1]) { cnt++; sum += hist_dualinf[j] / hist_dualinf[j - 1]; }
                if (cnt >= 3 && sum / cnt > 1.25) { o.breakyes = 6; break; }
            }
        }

        // Newton direction: rhs = GradLxi, solve the reduced linear system
        Vec rhs = GradLxi;
        double tolpsqmr = std::min(5e-3, 0.1 * rhs.norm());
        double const2 = 1.0;
        if (itersub > 1 && hist_priminf.size() >= 2 &&
            (hist_priminf[itersub - 1] > 0.5 * hist_priminf[0] ||
             (hist_dualinf[itersub - 1] > 1.1 * hist_dualinf[itersub - 2])))
            const2 = 0.5;
        tolpsqmr *= const2;

        int solve_ok = 1, cg = 0;
        Vec dxi = ssn_linsys_solve(A, o.rr, sigma, rhs,
                                   std::max(opt.time_limit - (wall_time_seconds() - t0), 1.0),
                                   &solve_ok, &cg);
        o.cg_iters += cg;
        hist_solveok.push_back(solve_ok);
        Vec Atdxi = A.applyT(dxi);

        // line search (port of findstep in Classic_Lasso_SSNCG.m)
        double tmp1 = dxi.dot(b - o.xi);
        double tmp2 = dxi.squaredNorm();
        double g0 = tmp1 + sigma * Atdxi.dot(o.ytmp);
        double alp = 0.0; int iterstep = 0;
        if (g0 > 0.0) {
            const double c1 = 1e-4, c2 = 0.9;
            const int maxit = std::max(1, (int)std::ceil(std::log(1.0 / 1e-5) / std::log(2.0)));
            const int stepop = (itersub <= 3 && dualinf_sub > 1e-4) ? 1 : 2;
            double LB = 0.0, UB = 1.0, gLB = g0, gUB = g0;
            Vec xi_new, y_new, ytmp_new;
            for (iterstep = 1; iterstep <= maxit; ++iterstep) {
                alp = (iterstep == 1) ? 1.0 : 0.5 * (LB + UB);
                xi_new = o.xi + alp * dxi;
                Vec yinput2 = o.ytmp + o.y - alp * Atdxi;
                proj_inf(yinput2, ld, y_new, o.rr);
                ytmp_new = yinput2 - y_new;
                double galp = tmp1 - alp * tmp2 + sigma * Atdxi.dot(ytmp_new);
                if (iterstep == 1) {
                    gLB = g0; gUB = galp;
                    if ((g0 > 0.0 && galp > 0.0) || (g0 < 0.0 && galp < 0.0)) {
                        o.Atxi = o.Atxi + alp * Atdxi;
                        o.Ly = b.dot(xi_new) - 0.5 * xi_new.squaredNorm()
                             - 0.5 * sigma * ytmp_new.squaredNorm();
                        break;
                    }
                }
                if (std::abs(galp) < c2 * std::abs(g0)) {
                    double Lynew = b.dot(xi_new) - 0.5 * xi_new.squaredNorm()
                                 - 0.5 * sigma * ytmp_new.squaredNorm();
                    if (Lynew - Ly0 - c1 * alp * g0 > -1e-8 / std::max(1.0, std::abs(Ly0)) &&
                        (stepop == 1 || (stepop == 2 && std::abs(galp) < 1e-5))) {
                        o.Atxi = o.Atxi + alp * Atdxi;
                        o.Ly = Lynew;
                        break;
                    }
                }
                if (galp * gUB < 0.0)      { LB = alp; gLB = galp; }
                else if (galp * gLB < 0.0) { UB = alp; gUB = galp; }
            }
            if (iterstep > maxit) o.Atxi = o.Atxi + alp * Atdxi;
            o.xi = xi_new;
            o.y = y_new;
            o.ytmp = ytmp_new;
            if (alp < 1e-10) { o.breakyes = 11; break; }
        } else {
            // no ascent direction: keep current point (reference returns here)
            break;
        }

        if (wall_time_seconds() - t0 > opt.time_limit) { o.breakyes = 12; break; }
        if (!o.xi.allFinite() || !o.y.allFinite())    { o.breakyes = 3;  break; }
    }
    o.itersub = itersub - 1;
    return o;
}

//----------------------------------------------------------------------
// ALM outer loop (port of Classic_Lasso_SSNAL_main.m) with the
// rescaling machinery omitted (bscale = cscale = 1 throughout, which is
// the default reference path).
//----------------------------------------------------------------------
inline SsnResult ssn_l1(const LinearOperator& A, const Vec& b, double lambda,
                        const LassoOptions& opt, const Vec& x_in)
{
    SsnResult res;
    const double t0 = wall_time_seconds();
    const Index m = A.rows(), n = A.cols();
    const int    maxiter = std::max(1, opt.maxiter);
    const double stoptol = opt.stoptol;

    Vec x = (x_in.size() == n) ? x_in : Vec::Zero(n);
    Vec xi = Vec::Zero(m);
    Vec y  = Vec::Zero(n);
    Vec Ax = A.apply(x);
    Vec Atxi = A.applyT(xi);

    // sigma = max(1/sqrt(Lip), min(1, sigmaLip, lambda)) with Lip=1 => 1
    double sigma = 1.0;
    const double sigmamax = 1e7, sigmamin = 1e-4;
    int prim_win = 0, dual_win = 0;
    double primfeas = 0.0, dualfeas = 0.0, eta = 1.0;

    int iter;
    for (iter = 1; iter <= maxiter; ++iter) {
        SsnSubOut sub = ssn_subproblem(A, b, lambda, x, Atxi, xi, y, sigma, opt, t0);
        res.itersub += sub.itersub;
        res.cg_iters += sub.cg_iters;
        xi = std::move(sub.xi);
        Atxi = std::move(sub.Atxi);
        y = std::move(sub.y);
        x = -sigma * sub.ytmp;
        Ax = A.apply(x);

        Vec Rd = Atxi + y;
        dualfeas = Rd.norm() / (1.0 + y.norm());
        Vec Rp = Ax - b + xi;
        primfeas = Rp.norm() / (1.0 + b.norm());

        // termination check on the *original* residuals (bscale=cscale=1)
        if (std::max(primfeas, dualfeas) < 500.0 * std::max(1e-6, stoptol)) {
            Vec grad = A.applyT(Ax - b);
            eta = kkt_residual(x, grad, lambda);
            if (eta < stoptol) { res.status = 0; res.breakyes = 1; break; }
        }

        // adaptive sigma by the primal/dual winning counters -- checked
        // every outer iteration.  On real data the sparse 3/6/12/... x1.25
        // schedule leaves sigma stuck and the ALM crawls on large active
        // sets (triazines delta=0.1 needs the fast schedule to converge in
        // ~143s; the trigger schedule runs out of time); 1.5x per round
        // tracks the primal/dual winner much faster and does not change
        // the converged solution (mpg delta=0.1 is identical under both).
        if (primfeas < dualfeas) prim_win++; else dual_win++;
        if (prim_win > std::max(1, static_cast<int>(1.2 * dual_win))) {
            prim_win = 0;
            sigma = std::min(sigmamax, sigma * 1.5);
        } else if (dual_win > std::max(1, static_cast<int>(1.2 * prim_win))) {
            dual_win = 0;
            sigma = std::max(sigmamin, sigma / 1.5);
        }

        if (opt.verbose >= 2)
            std::printf("  [ssn %2d] sub=%2d sigma=%.2e prim=%.2e dual=%.2e eta=%.2e\n",
                        iter, sub.itersub, sigma, primfeas, dualfeas, eta);

        if (wall_time_seconds() - t0 > opt.time_limit) { res.status = 2; break; }
        if (!x.allFinite())                            { res.status = 3; break; }
    }
    if (iter > maxiter) res.status = 1;

    Vec grad = A.applyT(Ax - b);
    res.x        = std::move(x);
    res.xi       = std::move(xi);
    res.y        = std::move(y);
    res.grad     = std::move(grad);
    res.obj      = 0.5 * (Ax - b).squaredNorm() + lambda * res.x.lpNorm<1>();
    res.eta      = eta;
    res.primfeas = primfeas;
    res.dualfeas = dualfeas;
    res.iter     = iter - 1;
    res.time     = wall_time_seconds() - t0;
    return res;
}

} // namespace smop
