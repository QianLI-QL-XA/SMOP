//======================================================================
// smop.hpp -- unified public API of the smop package
//
//   smop :: Level-Set Method for Sparse Optimization with Least-Squares
//           Constraints  (BMOP: min ||x||_1 s.t. ||Ax - b||_2 <= delta)
//
// Main entry points:
//   solve_bmop(A, b, delta, opts)   -- BMOP by the level-set method
//   solve_lasso(A, b, lambda, opts) -- single Lasso subproblem
// where A may be any LinearOperator (DenseOperator / SparseOperator or a
// user-defined subclass).
//
// This header is self-contained (header-only C++17); the only external
// dependency is Eigen.  Bindings for Python / R / MATLAB live in the
// python/, R/ and matlab/ folders of the package.
//======================================================================
#pragma once

#include "adaptive_sieving.hpp"
#include "admm_l1.hpp"
#include "level_set.hpp"
#include "linsolve.hpp"
#include "operators.hpp"
#include "prox.hpp"
#include "smoothing_newton.hpp"
#include "types.hpp"

// optional CUDA acceleration (GpuSparseOperator / GpuDenseOperator and
// the smop_gpu_* helpers); only pulled in when building with -DSMOP_USE_CUDA
#ifdef SMOP_USE_CUDA
#include "gpu_operator.hpp"
#endif

namespace smop {

//----------------------------------------------------------------------
// convenience overloads (dense / sparse matrices)
//----------------------------------------------------------------------
inline SmopResult solve_bmop(const Mat& A, const Vec& b, double delta,
                             const SmopOptions& opt = SmopOptions(),
                             const Vec& x0 = Vec())
{
    DenseOperator op(A);
    return solve_bmop(op, b, delta, opt, x0);
}

inline SmopResult solve_bmop(const SpMat& A, const Vec& b, double delta,
                             const SmopOptions& opt = SmopOptions(),
                             const Vec& x0 = Vec())
{
    SparseOperator op(A);
    return solve_bmop(op, b, delta, opt, x0);
}

//----------------------------------------------------------------------
// single Lasso subproblem
//     min_x  (1/2)||A x - b||_2^2 + lambda ||x||_1
// (used directly by the interfaces; internally also used by the
//  level-set outer loop)
//----------------------------------------------------------------------
inline LassoResult solve_lasso(const LinearOperator& A, const Vec& b,
                               double lambda, const LassoOptions& opt,
                               const Vec& x0 = Vec())
{
    return solve_lasso_full(A, b, lambda, opt, x0);
}

inline LassoResult solve_lasso(const Mat& A, const Vec& b, double lambda,
                               const LassoOptions& opt = LassoOptions(),
                               const Vec& x0 = Vec())
{
    DenseOperator op(A);
    return solve_lasso_full(op, b, lambda, opt, x0);
}

inline LassoResult solve_lasso(const SpMat& A, const Vec& b, double lambda,
                               const LassoOptions& opt = LassoOptions(),
                               const Vec& x0 = Vec())
{
    SparseOperator op(A);
    return solve_lasso_full(op, b, lambda, opt, x0);
}

//----------------------------------------------------------------------
// version string
//----------------------------------------------------------------------
inline const char* version() { return "0.1.0"; }

} // namespace smop
