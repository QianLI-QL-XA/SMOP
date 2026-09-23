//======================================================================
// linsolve.hpp -- iterative linear-system solvers (header-only).
//
//   cg_precond    : preconditioned CG for SPD systems given as an
//                   operator `matvec` and a diagonal preconditioner.
//   cg_smoothing  : CG for  M = (tmp - u).*I + diag(u) * A^T A
//                   (the smoothing-Newton linear system).
//   cg_aat        : CG for  M = I + sigma * A A^T
//                   (the ADMM xi-subproblem), diagonal preconditioner
//                   1 / (1 + sigma * rowNorm2).
//   normal_solve  : (A_K^T A_K + reg*I) z = u_K  (regularized normal
//                   equations, used by the Newton step for phi').
//
// All solvers are guarded against non-finite values, zero denominators,
// time limits and iteration caps, and always return a status code.
//======================================================================
#pragma once

#ifdef SMOP_USE_CUDA
#include "gpu_operator.hpp"
#endif

#include <Eigen/Cholesky>

#include <cmath>
#include <functional>

#include "operators.hpp"
#include "types.hpp"

namespace smop {

//------------------------------------------------------------------------
// Preconditioned CG.  Returns the number of iterations performed.
//------------------------------------------------------------------------
inline int cg_precond(
    const std::function<void(const Vec&, Vec&)>& matvec,
    const Vec& diag_precond,   // M_ii (must be positive); empty => none
    const Vec& rhs, Vec& x,
    double tol, int maxit, double time_limit,
    int* flag_out = nullptr)
{
    const Index n = rhs.size();
    if (n == 0) { if (flag_out) *flag_out = 0; return 0; }

    const bool use_pre = (diag_precond.size() == n);
    const double t0 = wall_time_seconds();
    Vec r = rhs;
    if (x.size() != n) x.setZero(n);
    if (x.squaredNorm() > 0.0) {
        Vec ax(n);
        matvec(x, ax);
        r = rhs - ax;
    }
    const double rhsnorm = rhs.norm();
    if (flag_out) *flag_out = 0;
    if (rhsnorm == 0.0) { x.setZero(); return 0; }

    Vec p(n), z(n);
    double rz = 0.0;
    if (use_pre) {
        for (Index i = 0; i < n; ++i) z(i) = r(i) / std::max(diag_precond(i), 1e-14);
        rz = r.dot(z);
    } else {
        z = r;
        rz = r.squaredNorm();
    }
    p = z;

    const double stop_tol = tol * tol * rhsnorm * rhsnorm;

    Vec ap(n);   // reused across iterations: avoids one n-vector allocation
                 // per CG step (ADMM runs 1e4 outer x ~100 CG iterations)
    for (int it = 1; it <= maxit; ++it) {
        if (!std::isfinite(rz) || rz <= 0.0) { if (flag_out) *flag_out = -1; break; }
        matvec(p, ap);
        const double pap = p.dot(ap);
        if (!std::isfinite(pap) || pap <= 0.0) { if (flag_out) *flag_out = -2; break; }
        const double alpha = rz / pap;
        x.noalias() += alpha * p;
        r.noalias() -= alpha * ap;
        if (r.squaredNorm() <= stop_tol) { if (flag_out) *flag_out = 0; return it; }
        if (use_pre) {
            for (Index i = 0; i < n; ++i) z(i) = r(i) / std::max(diag_precond(i), 1e-14);
            const double rznew = r.dot(z);
            p = z + (rznew / rz) * p;
            rz = rznew;
        } else {
            const double rznew = r.squaredNorm();
            p = r + (rznew / rz) * p;
            rz = rznew;
        }
        if (wall_time_seconds() - t0 > time_limit) { if (flag_out) *flag_out = 3; return it; }
    }
    if (flag_out) *flag_out = 1; // max iterations
    return maxit;
}

//------------------------------------------------------------------------
// CG for the smoothing-Newton system:
//   M = diag(tmp - u) + diag(u) * A^T A,   u = fx + gx >= 0.
// Preconditioner: max(tmp - u + u * colNorm2, eps).
//------------------------------------------------------------------------
inline int cg_smoothing(const LinearOperator& A, const Vec& u,
                        double tmp, const Vec& rhs, Vec& x,
                        double tol, int maxit, double time_limit,
                        int* flag_out = nullptr)
{
    const Index n = A.cols();
    const Vec col2 = A.columnNorm2();
    Vec diag(n);
    for (Index i = 0; i < n; ++i)
        diag(i) = std::max(tmp - u(i) + u(i) * col2(i), 1e-14);

    Vec atav(n);   // reused across matvec calls (one n-vector allocation per
                   // CG solve instead of per iteration)
    auto matvec = [&](const Vec& v, Vec& y) {
        // y = (tmp - u).*v + u .* (A^T (A v))
        atav = A.applyT(A.apply(v));
        y.resize(n);
        for (Index i = 0; i < n; ++i)
            y(i) = (tmp - u(i)) * v(i) + u(i) * atav(i);
    };
    return cg_precond(matvec, diag, rhs, x, tol, maxit, time_limit, flag_out);
}

//------------------------------------------------------------------------
// CG for (I + sigma A A^T) x = rhs. Preconditioner: 1/(1 + sigma*rowNorm2).
//------------------------------------------------------------------------
inline int cg_aat(const LinearOperator& A, double sigma,
                  const Vec& rhs, Vec& x,
                  double tol, int maxit, double time_limit,
                  int* flag_out = nullptr)
{
    const Index m = A.rows();
    const Vec row2 = A.rowNorm2();
    Vec diag(m);
    for (Index i = 0; i < m; ++i)
        diag(i) = 1.0 + sigma * row2(i);

    auto matvec = [&](const Vec& v, Vec& y) {
        y = v + sigma * A.apply(A.applyT(v));
    };
    return cg_precond(matvec, diag, rhs, x, tol, maxit, time_limit, flag_out);
}

//------------------------------------------------------------------------
// Solve (A_K^T A_K + reg*I) z = u_K for the reduced operator A_K = A[:,K].
// reg is bumped adaptively when the normal matrix is (numerically)
// singular; the matrix is r x r with r = |K| (typically small).
//------------------------------------------------------------------------
inline Vec normal_solve(const LinearOperator& Ared, const Vec& uK, double reg)
{
    const Index r = Ared.cols();
    Vec z(r);
    if (r == 0) return z;
    Mat C(Ared.rows(), r);
    Vec e(r);
    for (Index j = 0; j < r; ++j) {
        e.setZero();
        e(j) = 1.0;
        C.col(j) = Ared.apply(e);
    }
    Mat G;
#ifdef SMOP_USE_CUDA
    if (!smop_gpu_xtx(C, G)) G = C.transpose() * C;
#else
    G = C.transpose() * C;
#endif
    G.diagonal().array() += reg;
    Eigen::LDLT<Mat> ldlt(G);
    if (ldlt.info() != Eigen::Success) {
        G.diagonal().array() += 1e-6;
        Eigen::LDLT<Mat> ldlt2(G);
        z = ldlt2.solve(uK);
    } else {
        z = ldlt.solve(uK);
    }
    return z;
}

//------------------------------------------------------------------------
// Gram matrix G = A^T A (r x r) built by probing the operator with the
// canonical basis; used by the dense direct path of smoothing Newton.
//------------------------------------------------------------------------
inline Mat gram_matrix(const LinearOperator& Ared)
{
    const Index r = Ared.cols();
    const Index m = Ared.rows();
    Mat C(m, r);
    Vec e(r);
    for (Index j = 0; j < r; ++j) {
        e.setZero();
        e(j) = 1.0;
        C.col(j) = Ared.apply(e);
    }
#ifdef SMOP_USE_CUDA
    Mat G;
    if (smop_gpu_xtx(C, G)) return G;
#endif
    return C.transpose() * C;
}

} // namespace smop
