//======================================================================
// gpu_operator.hpp -- CUDA-accelerated sparse operator (optional).
//
// Interface + PIMPL: all CUDA code (and all Eigen objects it touches)
// lives in smop/src/gpu_operator.cu, compiled by nvcc into one object.
// User code is compiled with a plain C++ compiler and never sees nvcc.
//
// IMPORTANT design point (why GpuSparseOperator does NOT derive from
// SparseOperator): Eigen SparseMatrix objects constructed inside the
// nvcc TU and destroyed inside a g++ TU are not reliably managed across
// that boundary under -O3 (allocator/typeinfo mismatch -> heap errors).
// Instead this class derives from LinearOperator, keeps its own CPU
// copy of the matrix inside the .cu TU, and exposes it to
// SubsetOperator through LinearOperator::sparse_matrix(), a virtual
// hook that works across translation units.
//
// Build recipe:
//   nvcc -arch=sm_XXX -c smop/src/gpu_operator.cu          -> .o
//   g++  ... example.cpp smop/src/gpu_operator.o -lcudart  -> exe
//
// Runtime: without a CUDA device, the class falls back to a CPU
// SparseOperator copy internally (constructing one per operator).
// The reduced subproblems stay on CPU (too small for GPU transfers).
// Memory: CSR + CSC device copies -> ~2 x (nnz x 16 bytes).
//======================================================================
#pragma once

#include <memory>

#include "operators.hpp"
#include "types.hpp"

namespace smop {

class GpuSparseOperator : public LinearOperator {
public:
    explicit GpuSparseOperator(const SpMat& A);
    ~GpuSparseOperator() override;
    GpuSparseOperator(const GpuSparseOperator&) = delete;
    GpuSparseOperator& operator=(const GpuSparseOperator&) = delete;

    Index rows() const override { return rows_; }
    Index cols() const override { return cols_; }
    Vec apply(const Vec& x) const override;
    Vec applyT(const Vec& y) const override;
    Vec columnNorm2() const override;
    Vec rowNorm2() const override;
    const SpMat* sparse_matrix() const override { return cpu_.get(); }

    // true when the GPU path is actually active for this operator
    bool gpu_active() const;

private:
    struct Impl;               // CUDA buffers, defined in gpu_operator.cu
    Impl* impl_ = nullptr;
    int rows_ = 0, cols_ = 0;
    // CPU copy of the matrix (Eigen object fully owned by the .cu TU);
    // used for sparse_matrix() views, columnNorm2/rowNorm2 and the
    // no-GPU fallback.  Managed here so its lifetime never crosses TUs.
    std::unique_ptr<SpMat> cpu_;
};

//----------------------------------------------------------------------
// Dense (Eigen MatrixXd) operator accelerated with cuBLAS gemv.
// Same TU-safety design as GpuSparseOperator: derives from
// LinearOperator, owns its CPU copy inside the .cu TU, and exposes it
// through LinearOperator::dense_matrix().
//   apply  (y = A x)  -> cublasDgemv N
//   applyT (r = A^T y) -> cublasDgemv T
// The dense matrices of the UCI set (space_ga 3107x5005, abalone
// 4177x6435, bodyfat 252x116280, mpg 392x3432) get a large speedup on
// the full-matrix products of the outer sieve.
//----------------------------------------------------------------------
class GpuDenseOperator : public LinearOperator {
public:
    explicit GpuDenseOperator(const Mat& A);
    ~GpuDenseOperator() override;
    GpuDenseOperator(const GpuDenseOperator&) = delete;
    GpuDenseOperator& operator=(const GpuDenseOperator&) = delete;

    Index rows() const override { return rows_; }
    Index cols() const override { return cols_; }
    Vec apply(const Vec& x) const override;
    Vec applyT(const Vec& y) const override;
    Vec columnNorm2() const override;
    Vec rowNorm2() const override;
    const Mat* dense_matrix() const override { return cpu_.get(); }

    // GPU column-subset products on the uploaded dense matrix (used by
    // the SSN PCG matvec / reduced normal-matrix path)
    bool subset_gemv(const std::vector<Index>& cols, bool trans,
                     const Vec& x, Vec& y) const override;

    bool gpu_active() const;

private:
    struct Impl;               // CUDA buffer + cuBLAS handle, in .cu
    Impl* impl_ = nullptr;
    int rows_ = 0, cols_ = 0;
    std::unique_ptr<Mat> cpu_;
};

// runtime check: is there a usable CUDA device?
bool smop_gpu_available();

//----------------------------------------------------------------------
// GPU dense cross-product  G = X^T X  (X: m x k col-major, G: k x k).
// Uses cuBLAS gemm; returns true on success, false when no CUDA device
// or allocation failed (caller then falls back to the CPU product).
// C is the dense column materialisation (subset_dense_columns output)
// of the reduced operator, i.e. exactly the normal matrix the SSN /
// smoothing Newton subproblems need.
//----------------------------------------------------------------------
bool smop_gpu_xtx(const Mat& X, Mat& G);

} // namespace smop
