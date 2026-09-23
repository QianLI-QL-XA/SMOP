//======================================================================
// operators.hpp -- matrix-vector operator abstraction (header-only).
//
// The whole package talks to the data only through LinearOperator:
//   y = A x,   z = A^T y.
// Concrete implementations:
//   DenseOperator   -- wraps an Eigen::MatrixXd
//   SparseOperator  -- wraps an Eigen::SparseMatrix<double>
//   SubsetOperator  -- column-subset view A[:, idx] (no data copy), used
//                      by the adaptive-sieving inner loop
//   (User code may supply its own LinearOperator subclass, e.g. for
//    implicit / function-evaluated operators.)
//
// Performance: the sparse apply / applyT / norm paths are OpenMP-parallel
// over the columns when the matrix is large enough (gate in
// smop_parallel_large).  Column-wise parallelism is race-free for the
// gather-style products (applyT, columnNorm2); the scatter-style products
// (apply, rowNorm2) accumulate into a per-thread vector reduced under a
// critical section, which preserves exact serial semantics.
//======================================================================
#pragma once

#include <Eigen/SparseCore>

#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "types.hpp"

namespace smop {

// thread-count gate for the OpenMP parallel paths: parallelize only when
// the matrix is large enough that the fork/join overhead pays off
inline bool smop_parallel_large(long long nnz)
{
#ifdef _OPENMP
    return nnz >= 200000;
#else
    (void)nnz;
    return false;
#endif
}

class LinearOperator {
public:
    virtual ~LinearOperator() = default;
    virtual Vec  apply (const Vec& x) const = 0;   // y = A x
    virtual Vec  applyT(const Vec& y) const = 0;   // z = A^T y
    virtual Index rows() const = 0;                // m
    virtual Index cols() const = 0;                // n
    // squared 2-norms of the columns / rows (used as CG preconditioners)
    virtual Vec columnNorm2() const = 0;
    virtual Vec rowNorm2() const = 0;

    Vec applyAAT(const Vec& y) const { return apply(applyT(y)); }
    Vec applyATA(const Vec& x) const { return applyT(apply(x)); }

    // underlying CPU sparse matrix, if the operator keeps one (used by
    // SubsetOperator to build column views without dynamic_cast across
    // translation units; GpuSparseOperator overrides this to expose its
    // CPU copy -- the Eigen object then lives entirely in the CUDA TU)
    virtual const SpMat* sparse_matrix() const { return nullptr; }
    // same hook for dense matrices (GpuDenseOperator overrides it)
    virtual const Mat* dense_matrix() const { return nullptr; }
    // GPU column-subset products on the underlying dense GPU matrix:
    //   y = A[:,cols] x        (trans == false)
    //   y = A[:,cols]^T x      (trans == true)
    // returns true when executed on the GPU (caller skips the CPU path).
    virtual bool subset_gemv(const std::vector<Index>& cols, bool trans,
                             const Vec& x, Vec& y) const { return false; }
};

//----------------------------------------------------------------------
class DenseOperator : public LinearOperator {
public:
    explicit DenseOperator(const Mat& A) : A_(A) {}
    explicit DenseOperator(Mat&& A) : A_(std::move(A)) {}

    Vec apply(const Vec& x) const override { return A_ * x; }
    Vec applyT(const Vec& y) const override { return A_.transpose() * y; }
    Index rows() const override { return A_.rows(); }
    Index cols() const override { return A_.cols(); }
    Vec columnNorm2() const override { return A_.colwise().squaredNorm(); }
    Vec rowNorm2() const override { return A_.rowwise().squaredNorm(); }
    const Mat& matrix() const { return A_; }

private:
    Mat A_;
};

//----------------------------------------------------------------------
class SparseOperator : public LinearOperator {
public:
    explicit SparseOperator(const SpMat& A) : A_(A) {}

    Vec apply(const Vec& x) const override
    {
        const Index m = A_.rows();
        const long long nnz = static_cast<long long>(A_.nonZeros());
        if (!smop_parallel_large(nnz)) {
            Vec y = Vec::Zero(m);
            for (int k = 0; k < A_.outerSize(); ++k)
                for (SpMat::InnerIterator it(A_, k); it; ++it)
                    y(it.row()) += it.value() * x(it.col());
            return y;
        }
        Vec y = Vec::Zero(m);
#ifdef _OPENMP
#pragma omp parallel
        {
            Vec yl = Vec::Zero(m);
#pragma omp for schedule(static)
            for (int k = 0; k < A_.outerSize(); ++k)
                for (SpMat::InnerIterator it(A_, k); it; ++it)
                    yl(it.row()) += it.value() * x(it.col());
#pragma omp critical
            y += yl;
        }
#endif
        return y;
    }
    Vec applyT(const Vec& y) const override
    {
        const Index n = A_.cols();
        const long long nnz = static_cast<long long>(A_.nonZeros());
        Vec r = Vec::Zero(n);
        if (!smop_parallel_large(nnz)) {
            for (int k = 0; k < A_.outerSize(); ++k)
                for (SpMat::InnerIterator it(A_, k); it; ++it)
                    r(it.col()) += it.value() * y(it.row());
            return r;
        }
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
        for (int k = 0; k < A_.outerSize(); ++k)
            for (SpMat::InnerIterator it(A_, k); it; ++it)
                r(it.col()) += it.value() * y(it.row());
#endif
        return r;
    }
    Index rows() const override { return A_.rows(); }
    Index cols() const override { return A_.cols(); }
    Vec columnNorm2() const override
    {
        const Index n = A_.cols();
        const long long nnz = static_cast<long long>(A_.nonZeros());
        Vec c = Vec::Zero(n);
        if (!smop_parallel_large(nnz)) {
            for (int k = 0; k < A_.outerSize(); ++k)
                for (SpMat::InnerIterator it(A_, k); it; ++it)
                    c(it.col()) += it.value() * it.value();
            return c;
        }
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
        for (int k = 0; k < A_.outerSize(); ++k)
            for (SpMat::InnerIterator it(A_, k); it; ++it)
                c(it.col()) += it.value() * it.value();
#endif
        return c;
    }
    Vec rowNorm2() const override
    {
        const Index m = A_.rows();
        const long long nnz = static_cast<long long>(A_.nonZeros());
        Vec r = Vec::Zero(m);
        if (!smop_parallel_large(nnz)) {
            for (int k = 0; k < A_.outerSize(); ++k)
                for (SpMat::InnerIterator it(A_, k); it; ++it)
                    r(it.row()) += it.value() * it.value();
            return r;
        }
#ifdef _OPENMP
#pragma omp parallel
        {
            Vec rl = Vec::Zero(m);
#pragma omp for schedule(static)
            for (int k = 0; k < A_.outerSize(); ++k)
                for (SpMat::InnerIterator it(A_, k); it; ++it)
                    rl(it.row()) += it.value() * it.value();
#pragma omp critical
            r += rl;
        }
#endif
        return r;
    }
    const SpMat& matrix() const { return A_; }

private:
    SpMat A_;
};

//----------------------------------------------------------------------
// View of A[:, cols]  --  no data is copied; used by adaptive sieving.
//   apply : y = A[:,cols] x          (scatter x into a full vector)
//   applyT: y = (A^T w)[cols]        (gather)
//----------------------------------------------------------------------
class SubsetOperator : public LinearOperator {
public:
    SubsetOperator(const LinearOperator& base, std::vector<Index> cols)
        : base_(&base), cols_(std::move(cols))
    {
        // resolve the column map onto the *base matrix*: a nested subset
        // (SubsetOperator over a SubsetOperator, as in SSNAL's free-set
        // operator built on the AS reduced operator) must compose its
        // index map onto the outermost one and inherit the underlying
        // sparse/dense matrix directly, otherwise every apply/linsys
        // would fall back to the generic scatter path
        if (const auto* ss = dynamic_cast<const SubsetOperator*>(base_)) {
            const auto& bc = ss->columns();
            for (auto& c : cols_) c = bc[static_cast<size_t>(c)];
            sparse_ = ss->sparse_matrix();
            dense_  = ss->dense_matrix();
        } else if (const auto* sp = dynamic_cast<const SparseOperator*>(base_)) {
            sparse_ = &sp->matrix();
        } else if (const auto* dp = dynamic_cast<const DenseOperator*>(base_)) {
            dense_ = &dp->matrix();
        }
        // GPU operator: expose its CPU matrix via the virtual hook (works
        // across TU boundaries where dynamic_cast typeinfo may be split)
        if (!sparse_ && !dense_ && base_ && base_->sparse_matrix()) {
            sparse_ = base_->sparse_matrix();
        }
        if (!sparse_ && !dense_ && base_ && base_->dense_matrix()) {
            dense_ = base_->dense_matrix();
        }
        // keep the column index list sorted (gather/scatter are cheap)
        std::sort(cols_.begin(), cols_.end());
    }

    Vec apply(const Vec& x) const override
    {
        if (sparse_) {
            const Index m = base_->rows();
            Vec y = Vec::Zero(m);
            const long long nnz = static_cast<long long>(cols_.size());
            if (!smop_parallel_large(nnz)) {
                for (size_t j = 0; j < cols_.size(); ++j) {
                    const Index c = cols_[j];
                    for (SpMat::InnerIterator it(*sparse_, c); it; ++it)
                        y(it.row()) += it.value() * x(Index(j));
                }
                return y;
            }
#ifdef _OPENMP
#pragma omp parallel
            {
                Vec yl = Vec::Zero(m);
#pragma omp for schedule(static)
                for (int j = 0; j < static_cast<int>(cols_.size()); ++j) {
                    const Index c = cols_[static_cast<size_t>(j)];
                    for (SpMat::InnerIterator it(*sparse_, c); it; ++it)
                        yl(it.row()) += it.value() * x(Index(j));
                }
#pragma omp critical
                y += yl;
            }
#endif
            return y;
        }
        if (dense_) {
            Vec y = Vec::Zero(base_->rows());
            for (size_t j = 0; j < cols_.size(); ++j)
                y.noalias() += dense_->col(cols_[j]) * x(Index(j));
            return y;
        }
        const Index n = base_->cols();
        Vec xfull = Vec::Zero(n);
        for (size_t j = 0; j < cols_.size(); ++j) xfull(cols_[j]) = x(Index(j));
        return base_->apply(xfull);
    }
    Vec applyT(const Vec& y) const override
    {
        if (sparse_) {
            Vec r(cols_.size());
            if (!smop_parallel_large(static_cast<long long>(cols_.size()))) {
                for (size_t j = 0; j < cols_.size(); ++j) {
                    const Index c = cols_[j];
                    double s = 0.0;
                    for (SpMat::InnerIterator it(*sparse_, c); it; ++it)
                        s += it.value() * y(it.row());
                    r(Index(j)) = s;
                }
                return r;
            }
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
            for (int j = 0; j < static_cast<int>(cols_.size()); ++j) {
                const Index c = cols_[static_cast<size_t>(j)];
                double s = 0.0;
                for (SpMat::InnerIterator it(*sparse_, c); it; ++it)
                    s += it.value() * y(it.row());
                r(Index(j)) = s;
            }
#endif
            return r;
        }
        if (dense_) {
            Vec r(cols_.size());
            for (size_t j = 0; j < cols_.size(); ++j)
                r(Index(j)) = dense_->col(cols_[j]).dot(y);
            return r;
        }
        Vec at = base_->applyT(y);
        Vec r(cols_.size());
        for (size_t j = 0; j < cols_.size(); ++j) r(Index(j)) = at(cols_[j]);
        return r;
    }
    Index rows() const override { return base_->rows(); }
    Index cols() const override { return static_cast<Index>(cols_.size()); }
    Vec columnNorm2() const override
    {
        if (sparse_) {
            Vec r(cols_.size());
            if (!smop_parallel_large(static_cast<long long>(cols_.size()))) {
                for (size_t j = 0; j < cols_.size(); ++j) {
                    const Index c = cols_[j];
                    double s = 0.0;
                    for (SpMat::InnerIterator it(*sparse_, c); it; ++it)
                        s += it.value() * it.value();
                    r(Index(j)) = s;
                }
                return r;
            }
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
            for (int j = 0; j < static_cast<int>(cols_.size()); ++j) {
                const Index c = cols_[static_cast<size_t>(j)];
                double s = 0.0;
                for (SpMat::InnerIterator it(*sparse_, c); it; ++it)
                    s += it.value() * it.value();
                r(Index(j)) = s;
            }
#endif
            return r;
        }
        Vec c2 = base_->columnNorm2();
        Vec r(cols_.size());
        for (size_t j = 0; j < cols_.size(); ++j) r(Index(j)) = c2(cols_[j]);
        return r;
    }
    Vec rowNorm2() const override { return base_->rowNorm2(); }

    const std::vector<Index>& columns() const { return cols_; }
    // underlying sparse matrix when the base is a SparseOperator (nullptr
    // otherwise); lets linear-system builders materialise dense column
    // blocks by direct sparse-column copy instead of nr scatter applies
    const SpMat* sparse_matrix() const { return sparse_; }
    const Mat*   dense_matrix()  const { return dense_; }
    const LinearOperator* base() const { return base_; }

    // forward the GPU column-subset hook to the base, composing this
    // subset's column map so the GPU operator sees columns of the base
    // matrix (a nested subset therefore reaches the outermost matrix)
    bool subset_gemv(const std::vector<Index>& cols, bool trans,
                     const Vec& x, Vec& y) const override
    {
        if (!base_) return false;
        std::vector<Index> bc(cols.size());
        for (size_t i = 0; i < cols.size(); ++i)
            bc[i] = cols_[static_cast<size_t>(cols[i])];
        return base_->subset_gemv(bc, trans, x, y);
    }

private:
    const LinearOperator* base_;
    const SpMat*          sparse_ = nullptr;
    const Mat*            dense_  = nullptr;
    std::vector<Index> cols_;
};

} // namespace smop
