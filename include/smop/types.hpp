//======================================================================
// smop: Level-Set Method for Sparse Optimization with Least-Squares
//       Constraints (BMOP: min ||x||_1 s.t. ||Ax - b||_2 <= delta)
//
// Reference implementation of the algorithms in:
//   Q. Li, D. F. Sun, Y. Yuan, "An Efficient Sieving-Based Secant Method
//   for Sparse Optimization Problems with Least-Squares Constraints",
//   SIAM J. Optim. (DOI 10.1137/23M1594443)
//   X. Li, D. F. Sun, K.-C. Toh, "A Highly Efficient Semismooth Newton
//   Augmented Lagrangian Method for Solving Lasso Problems",
//   SIAM J. Optim. 28 (2018) 1842-1866  (SSNAL)
//
// types.hpp -- basic types, options and result structs (header-only).
//======================================================================
#pragma once

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

namespace smop {

using Index    = Eigen::Index;
using Vec      = Eigen::VectorXd;
using Mat      = Eigen::MatrixXd;
using SpMat    = Eigen::SparseMatrix<double, Eigen::ColMajor>;
using IntVec   = Eigen::VectorXi;

//------------------------------------------------------------------------
// Options for a single Lasso subproblem
//     min_x  (1/2)||Ax - b||_2^2 + lambda * ||x||_1
//------------------------------------------------------------------------
struct LassoOptions {
    // stopping tolerance (relative KKT residual eta)
    double stoptol   = 1e-6;
    // max outer (smoothing-Newton) iterations; a diverging smoothed
    // sub-problem burns minutes for nothing, so cap it and fall back to ADMM
    int    maxiter   = 200;
    int    maxiter_admm = 10000;
    // smoothing-Newton parameters (MATLAB defaults from smoothingNewton.m)
    double eps_hat   = 1.0;    // initial smoothing parameter (MATLAB default)
    double kappa     = 1.0;    // regularization for the smoothing term (default 1)
    double rho       = 0.7;    // line-search backtracking factor
    double armijo_sigma = 5e-7; // Armijo parameter (sigma in the reference)
    double eta_hat   = 1e-4;
    // max inner AS (adaptive sieving) iterations
    int    maxiter_as = 20;
    // AS tolerance (usually tightened by the outer level-set loop)
    double stoptolas = 1e-6;
    // fraction of n added per AS round; 0 -> automatic (by n)
    double control_addpram = 0.0;
    // subproblem solvers
    bool   use_smoothing = true; // smoothing Newton (primary when m > nr)
    bool   use_ssn       = true; // SSNAL (primary when nr >= m, high dim)
    bool   use_admm      = true; // ADMM (fallback when the other two fail)
    // time limit (seconds) for one subproblem solve
    double time_limit   = 3600.0;
    // verbosity: 0 silent, 1 outer summary, 2 detailed
    int    verbose      = 0;
    // warm-start gradient (A^T(Ax-b)) at current x; empty => compute
    Vec    grad0;
};

//------------------------------------------------------------------------
// Options for the outer level-set BMOP solver
//     min_x  ||x||_1   s.t.   ||Ax - b||_2 <= delta
//------------------------------------------------------------------------
struct SmopOptions {
    // outer stopping tolerance: eta = |psi(mu) - delta| / max(1, delta)
    double stoptol        = 1e-6;
    int    maxiter_levelset = 200;
    // root-finding scheme
    bool   use_secant     = true;  // secant method (fast, recommended)
    bool   use_newton     = false; // Newton step (analytic phi')
    // adaptive sieving on/off
    bool   use_as         = true;
    // initial bracket [mu0, muinf]; mu0=0 => auto; muinf<=0 => ||A^T b||_inf
    double mu0            = 0.0;
    double muinf          = -1.0;
    // user-provided initial mu; 0 => auto (Gaussian-width heuristic)
    double initial_mu     = 0.0;
    // time limit for the whole solve (seconds)
    double time_limit     = 3600.0;
    // verbosity: 0 silent, 1 outer summary, 2 detailed
    int    verbose        = 0;
    // subproblem options (forwarded to the AS / Lasso solver)
    LassoOptions lasso;
};

//------------------------------------------------------------------------
// Info collected by one call of the (sub) solvers
//------------------------------------------------------------------------
struct LassoInfo {
    int    iter     = 0;    // solver iterations (smoothing or ADMM)
    double eta      = 0.0;  // relative KKT residual
    double etaorg   = 0.0;  // absolute KKT residual norm
    double obj      = 0.0;  // primal objective
    double time     = 0.0;  // seconds
    int    status   = 0;    // 0 ok, 1 max iter, 2 time limit, 3 failure
    int    breakyes = 0;    // solver-specific break code (0 => not converged)
    std::string msg;
};

struct LassoResult {
    Vec x;      // primal solution (full dimension when sieving is off)
    Vec xi;     // residual b - A x
    Vec grad;   // A^T (A x - b)
    LassoInfo info;
};

struct SmopInfo {
    int    iter = 0;            // outer level-set iterations
    int    iter_bisection = 0;
    int    iter_newton_or_secant = 0;
    int    iter_smoothing = 0;  // total inner smoothing-Newton iterations
    int    iter_ssn       = 0;  // total inner SSNAL iterations
    int    iter_admm      = 0;  // total inner ADMM iterations
    double eta   = 0.0;
    double abseta = 0.0;
    double psi   = 0.0;         // ||A x - b|| at the returned x
    double mu    = 0.0;         // final regularization parameter
    double time  = 0.0;
    int    status = 0;          // 0 ok, 1 max iter, 2 time limit, 3 error
    std::string msg;
    std::vector<Index>    reducedn;   // active-set sizes per AS round
    std::vector<double>   mupath;     // mu trajectory
    std::vector<double>   psipath;    // psi(mu) trajectory
};

struct SmopResult {
    Vec x;      // solution of the BMOP
    Vec xi;     // b - A x
    Vec y;      // dual multiplier (truncated), y = -A^T xi / (1 + ...)
    double mu   = 0.0;       // final mu
    SmopInfo info;
};

//------------------------------------------------------------------------
// small helpers
//------------------------------------------------------------------------
inline double wall_time_seconds()
{
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

// number of entries capturing a fraction r of the L1 energy (descending)
inline int findnnz(const Vec& x, double r, double tiny = 1e-16)
{
    const Index n = x.size();
    int k = 0;
    std::vector<double> vals;
    vals.reserve(n);
    double s1 = 0.0, mx = 0.0;
    for (Index i = 0; i < n; ++i) {
        const double a = std::abs(x(i));
        if (a > 0.0) { vals.push_back(a); s1 += a; mx = std::max(mx, a); }
    }
    if (vals.empty() || std::min(s1, mx) <= tiny) return 0;
    std::sort(vals.begin(), vals.end(), std::greater<double>());
    double acc = 0.0;
    for (size_t j = 0; j < vals.size(); ++j) {
        acc += vals[j];
        if (acc > r * s1) { k = static_cast<int>(j) + 1; break; }
    }
    return k;
}

} // namespace smop
