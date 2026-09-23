//======================================================================
// gpu_operator.cu -- CUDA implementation of GpuSparseOperator and
// GpuDenseOperator.  Compiled with nvcc (see gpu_operator.hpp).
//
// Kernels:
//   * smop_csr_spmv_kernel : block-per-row CSR SpMV for y = A x (one
//     block per row, 256 threads, tree reduction inside the block) --
//     good for the long rows of the log1p / E2006 matrices.
//   * smop_csc_spmv_kernel : column-per-thread CSC SpMV for y = A^T y
//     (each column is an independent dot product).
//   * smop_dense_subset_gemv_kernel : column-subset products on the
//     dense GPU matrix (SSN PCG matvec / reduced normal equations).
//
// Dense operators use cuBLAS gemv/gemm for the full products; the
// sparse ones keep the hand-written kernels (no cuSPARSE dependency).
//
// Memory: sparse operators upload CSR + CSC (~2 x nnz x 16 B); dense
// operators upload the whole m x n matrix once.
//
// All Eigen objects (CPU matrix copies) are owned here, inside the
// nvcc TU, so no Eigen object's lifetime crosses a compiler boundary.
//======================================================================

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <Eigen/SparseCore>
#include <memory>
#include <vector>

#include "smop/gpu_operator.hpp"

namespace smop {

//----------------------------------------------------------------------
// block-per-row CSR SpMV:  y = A x
//----------------------------------------------------------------------
__global__ void smop_csr_spmv_kernel(const int* rowptr, const int* colidx,
                                     const double* vals, const double* x,
                                     double* y, int rows)
{
    const int i = blockIdx.x;
    if (i >= rows) return;
    const int start = rowptr[i], end = rowptr[i + 1];
    double s = 0.0;
    for (int k = start + threadIdx.x; k < end; k += blockDim.x)
        s += vals[k] * x[colidx[k]];
    __shared__ double sh[256];
    sh[threadIdx.x] = s;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) sh[threadIdx.x] += sh[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) y[i] = sh[0];
}

//----------------------------------------------------------------------
// column-per-thread CSC SpMV:  y = A^T y
//----------------------------------------------------------------------
__global__ void smop_csc_spmv_kernel(const int* colptr, const int* rowidx,
                                     const double* vals, const double* y,
                                     double* x, int cols)
{
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= cols) return;
    double s = 0.0;
    for (int k = colptr[j]; k < colptr[j + 1]; ++k)
        s += vals[k] * y[rowidx[k]];
    x[j] = s;
}

struct GpuSparseOperator::Impl {
    int* d_rowptr_ = nullptr;
    int* d_colidx_ = nullptr;
    double* d_vals_ = nullptr;
    int* d_cscptr_ = nullptr;
    int* d_cscidx_ = nullptr;
    double* d_cscval_ = nullptr;
    double* d_x_ = nullptr;
    double* d_y_ = nullptr;
    ~Impl();
};

GpuSparseOperator::Impl::~Impl()
{
    if (d_rowptr_) cudaFree(d_rowptr_);
    if (d_colidx_) cudaFree(d_colidx_);
    if (d_vals_)   cudaFree(d_vals_);
    if (d_cscptr_) cudaFree(d_cscptr_);
    if (d_cscidx_) cudaFree(d_cscidx_);
    if (d_cscval_) cudaFree(d_cscval_);
    if (d_x_)      cudaFree(d_x_);
    if (d_y_)      cudaFree(d_y_);
}

GpuSparseOperator::GpuSparseOperator(const SpMat& A)
    : rows_(static_cast<int>(A.rows())), cols_(static_cast<int>(A.cols()))
{
    // CPU copy: element-by-element (avoids Eigen's sparse copy ctor in
    // this TU -- plain data movement is exact and cheap for our build).
    cpu_ = std::make_unique<SpMat>(A.rows(), A.cols());
    cpu_->reserve(static_cast<Index>(A.nonZeros()));
    for (int k = 0; k < A.outerSize(); ++k)
        for (SpMat::InnerIterator it(A, k); it; ++it)
            cpu_->insert(it.row(), it.col()) = it.value();
    cpu_->makeCompressed();

    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev < 1) return; // CPU fallback

    const int nnz = static_cast<int>(A.nonZeros());
    Impl* im = new Impl();
    // --- CSR layout: counting-sort conversion from CSC (see header). --
    std::vector<int> rowptr(rows_ + 1, 0);
    for (int k = 0; k < nnz; ++k) ++rowptr[A.innerIndexPtr()[k] + 1];
    for (int i = 1; i <= rows_; ++i) rowptr[i] += rowptr[i - 1];
    std::vector<int> colidx(nnz);
    std::vector<double> vals(nnz);
    std::vector<int> cur(rowptr.begin(), rowptr.end() - 1);   // write cursor per row
    for (int j = 0; j < cols_; ++j) {
        for (int k = A.outerIndexPtr()[j]; k < A.outerIndexPtr()[j + 1]; ++k) {
            const int r = A.innerIndexPtr()[k];
            const int pos = cur[static_cast<size_t>(r)]++;
            colidx[static_cast<size_t>(pos)] = j;
            vals[static_cast<size_t>(pos)] = A.valuePtr()[k];
        }
    }
    // --- CSC layout: Eigen's native storage of A. ----
    cudaMalloc(&im->d_rowptr_, sizeof(int) * (rows_ + 1));
    cudaMalloc(&im->d_colidx_, sizeof(int) * nnz);
    cudaMalloc(&im->d_vals_,   sizeof(double) * nnz);
    cudaMalloc(&im->d_cscptr_, sizeof(int) * (cols_ + 1));
    cudaMalloc(&im->d_cscidx_, sizeof(int) * nnz);
    cudaMalloc(&im->d_cscval_, sizeof(double) * nnz);
    cudaMalloc(&im->d_x_,      sizeof(double) * static_cast<size_t>(cols_));
    cudaMalloc(&im->d_y_,      sizeof(double) * static_cast<size_t>(rows_));
    cudaMemcpy(im->d_rowptr_, rowptr.data(), sizeof(int) * (rows_ + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(im->d_colidx_, colidx.data(), sizeof(int) * nnz, cudaMemcpyHostToDevice);
    cudaMemcpy(im->d_vals_,   vals.data(),   sizeof(double) * nnz, cudaMemcpyHostToDevice);
    cudaMemcpy(im->d_cscptr_, A.outerIndexPtr(), sizeof(int) * (cols_ + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(im->d_cscidx_, A.innerIndexPtr(), sizeof(int) * nnz, cudaMemcpyHostToDevice);
    cudaMemcpy(im->d_cscval_, A.valuePtr(), sizeof(double) * nnz, cudaMemcpyHostToDevice);
    impl_ = im;
}

GpuSparseOperator::~GpuSparseOperator() { delete impl_; }

Vec GpuSparseOperator::apply(const Vec& x) const
{
    Vec y = Vec::Zero(rows_);
    if (!impl_) {
        // CPU fallback: plain CSC SpMV over the CPU copy
        for (int j = 0; j < cols_; ++j)
            for (SpMat::InnerIterator it(*cpu_, j); it; ++it)
                y[it.row()] += it.value() * x[j];
        return y;
    }
    cudaMemcpy(impl_->d_x_, x.data(), sizeof(double) * static_cast<size_t>(cols_),
               cudaMemcpyHostToDevice);
    smop_csr_spmv_kernel<<<rows_, 256>>>(impl_->d_rowptr_, impl_->d_colidx_,
                                         impl_->d_vals_, impl_->d_x_,
                                         impl_->d_y_, rows_);
    cudaMemcpy(y.data(), impl_->d_y_, sizeof(double) * static_cast<size_t>(rows_),
               cudaMemcpyDeviceToHost);
    return y;
}

Vec GpuSparseOperator::applyT(const Vec& y) const
{
    Vec r = Vec::Zero(cols_);
    if (!impl_) {
        // CPU fallback: plain CSC SpMV over the CPU copy
        for (int j = 0; j < cols_; ++j)
            for (SpMat::InnerIterator it(*cpu_, j); it; ++it)
                r[j] += it.value() * y[it.row()];
        return r;
    }
    cudaMemcpy(impl_->d_y_, y.data(), sizeof(double) * static_cast<size_t>(rows_),
               cudaMemcpyHostToDevice);
    const int threads = 256;
    const int blocks = (cols_ + threads - 1) / threads;
    smop_csc_spmv_kernel<<<blocks, threads>>>(impl_->d_cscptr_, impl_->d_cscidx_,
                                              impl_->d_cscval_, impl_->d_y_,
                                              impl_->d_x_, cols_);
    cudaMemcpy(r.data(), impl_->d_x_, sizeof(double) * static_cast<size_t>(cols_),
               cudaMemcpyDeviceToHost);
    return r;
}

Vec GpuSparseOperator::columnNorm2() const
{
    Vec cn = Vec::Zero(cols_);
    for (int j = 0; j < cols_; ++j) {
        double s = 0.0;
        for (SpMat::InnerIterator it(*cpu_, j); it; ++it) {
            const double v = it.value();
            s += v * v;
        }
        cn[j] = s;
    }
    return cn;
}

Vec GpuSparseOperator::rowNorm2() const
{
    Vec rn = Vec::Zero(rows_);
    for (int j = 0; j < cols_; ++j)
        for (SpMat::InnerIterator it(*cpu_, j); it; ++it) {
            const double v = it.value();
            rn[it.row()] += v * v;
        }
    return rn;
}

bool GpuSparseOperator::gpu_active() const { return impl_ != nullptr; }

//======================================================================
// GpuDenseOperator -- cuBLAS gemv for dense Eigen MatrixXd operators.
//======================================================================
struct GpuDenseOperator::Impl {
    double* d_A_ = nullptr;    // m x n col-major (Eigen layout)
    double* d_x_ = nullptr;    // n
    double* d_y_ = nullptr;    // m
    cublasHandle_t h_ = nullptr;
    ~Impl();
};

GpuDenseOperator::Impl::~Impl()
{
    if (d_A_) cudaFree(d_A_);
    if (d_x_) cudaFree(d_x_);
    if (d_y_) cudaFree(d_y_);
    if (h_)   cublasDestroy(h_);
}

GpuDenseOperator::GpuDenseOperator(const Mat& A)
    : rows_(static_cast<int>(A.rows())), cols_(static_cast<int>(A.cols()))
{
    // CPU copy: plain memcpy (MatrixXd is contiguous col-major)
    cpu_ = std::make_unique<Mat>(A.rows(), A.cols());
    if (A.size() > 0) {
        std::memcpy(cpu_->data(), A.data(),
                    sizeof(double) * static_cast<size_t>(A.size()));
    }

    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev < 1) return; // CPU fallback
    Impl* im = new Impl();
    if (cublasCreate(&im->h_) != CUBLAS_STATUS_SUCCESS) { delete im; return; }
    const size_t nel = static_cast<size_t>(rows_) * static_cast<size_t>(cols_);
    cudaMalloc(&im->d_A_, sizeof(double) * nel);
    cudaMalloc(&im->d_x_, sizeof(double) * static_cast<size_t>(cols_));
    cudaMalloc(&im->d_y_, sizeof(double) * static_cast<size_t>(rows_));
    cudaMemcpy(im->d_A_, A.data(), sizeof(double) * nel, cudaMemcpyHostToDevice);
    impl_ = im;
}

GpuDenseOperator::~GpuDenseOperator() { delete impl_; }

Vec GpuDenseOperator::apply(const Vec& x) const
{
    Vec y = Vec::Zero(rows_);
    if (!impl_) {
        // CPU fallback (plain loops over the contiguous col-major copy)
        const double* A = cpu_->data();
        const int m = rows_, n = cols_;
        for (int j = 0; j < n; ++j) {
            const double xj = x[j];
            const double* col = A + static_cast<size_t>(j) * m;
            for (int i = 0; i < m; ++i) y[i] += col[i] * xj;
        }
        return y;
    }
    cudaMemcpy(impl_->d_x_, x.data(), sizeof(double) * static_cast<size_t>(cols_),
               cudaMemcpyHostToDevice);
    const double alpha = 1.0, beta = 0.0;
    cublasDgemv(impl_->h_, CUBLAS_OP_N, rows_, cols_, &alpha, impl_->d_A_, rows_,
                impl_->d_x_, 1, &beta, impl_->d_y_, 1);
    cudaMemcpy(y.data(), impl_->d_y_, sizeof(double) * static_cast<size_t>(rows_),
               cudaMemcpyDeviceToHost);
    return y;
}

Vec GpuDenseOperator::applyT(const Vec& y) const
{
    Vec r = Vec::Zero(cols_);
    if (!impl_) {
        const double* A = cpu_->data();
        const int m = rows_, n = cols_;
        for (int j = 0; j < n; ++j) {
            double s = 0.0;
            const double* col = A + static_cast<size_t>(j) * m;
            for (int i = 0; i < m; ++i) s += col[i] * y[i];
            r[j] = s;
        }
        return r;
    }
    cudaMemcpy(impl_->d_y_, y.data(), sizeof(double) * static_cast<size_t>(rows_),
               cudaMemcpyHostToDevice);
    const double alpha = 1.0, beta = 0.0;
    cublasDgemv(impl_->h_, CUBLAS_OP_T, rows_, cols_, &alpha, impl_->d_A_, rows_,
                impl_->d_y_, 1, &beta, impl_->d_x_, 1);
    cudaMemcpy(r.data(), impl_->d_x_, sizeof(double) * static_cast<size_t>(cols_),
               cudaMemcpyDeviceToHost);
    return r;
}

Vec GpuDenseOperator::columnNorm2() const
{
    Vec cn = Vec::Zero(cols_);
    const double* A = cpu_->data();
    const int m = rows_;
    for (int j = 0; j < cols_; ++j) {
        double s = 0.0;
        const double* col = A + static_cast<size_t>(j) * m;
        for (int i = 0; i < m; ++i) s += col[i] * col[i];
        cn[j] = s;
    }
    return cn;
}

Vec GpuDenseOperator::rowNorm2() const
{
    Vec rn = Vec::Zero(rows_);
    const double* A = cpu_->data();
    const int m = rows_, n = cols_;
    for (int j = 0; j < n; ++j) {
        const double* col = A + static_cast<size_t>(j) * m;
        for (int i = 0; i < m; ++i) rn[i] += col[i] * col[i];
    }
    return rn;
}

bool GpuDenseOperator::gpu_active() const { return impl_ != nullptr; }

//----------------------------------------------------------------------
// column-subset products on the GPU dense matrix:
//   trans=false: y[i] = sum_j A[i, cols[j]] x[j]   (one thread per row)
//   trans=true : y[j] = sum_i A[i, cols[j]] x[i]   (one thread per col)
//----------------------------------------------------------------------
__global__ void smop_dense_subset_gemv_kernel(const double* A, const int* cols,
                                              const double* x, double* y,
                                              int m, int k, bool trans)
{
    if (!trans) {
        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= m) return;
        double s = 0.0;
        for (int j = 0; j < k; ++j)
            s += A[static_cast<size_t>(cols[j]) * m + i] * x[j];
        y[i] = s;
    } else {
        const int j = blockIdx.x * blockDim.x + threadIdx.x;
        if (j >= k) return;
        double s = 0.0;
        const double* col = A + static_cast<size_t>(cols[j]) * m;
        for (int i = 0; i < m; ++i) s += col[i] * x[i];
        y[j] = s;
    }
}

bool GpuDenseOperator::subset_gemv(const std::vector<Index>& cols, bool trans,
                                   const Vec& x, Vec& y) const
{
    if (!impl_ || cols.empty()) return false;
    const int k = static_cast<int>(cols.size());
    std::vector<int> ci(static_cast<size_t>(k));
    for (int j = 0; j < k; ++j) ci[static_cast<size_t>(j)] = static_cast<int>(cols[static_cast<size_t>(j)]);
    const int xlen = trans ? rows_ : k;
    const int ylen = trans ? k : rows_;
    int* d_cols = nullptr; double* d_x = nullptr, * d_y = nullptr;
    const bool ok = (cudaMalloc(&d_cols, sizeof(int) * static_cast<size_t>(k)) == cudaSuccess) &&
                    (cudaMalloc(&d_x, sizeof(double) * static_cast<size_t>(xlen)) == cudaSuccess) &&
                    (cudaMalloc(&d_y, sizeof(double) * static_cast<size_t>(ylen)) == cudaSuccess);
    if (!ok) {
        if (d_cols) cudaFree(d_cols);
        if (d_x) cudaFree(d_x);
        if (d_y) cudaFree(d_y);
        return false;
    }
    cudaMemcpy(d_cols, ci.data(), sizeof(int) * static_cast<size_t>(k), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x.data(), sizeof(double) * static_cast<size_t>(xlen), cudaMemcpyHostToDevice);
    const int threads = 256;
    const int nthreads = trans ? k : rows_;
    const int blocks = (nthreads + threads - 1) / threads;
    smop_dense_subset_gemv_kernel<<<blocks, threads>>>(impl_->d_A_, d_cols, d_x, d_y,
                                                       rows_, k, trans);
    y = Vec::Zero(ylen);
    cudaMemcpy(y.data(), d_y, sizeof(double) * static_cast<size_t>(ylen), cudaMemcpyDeviceToHost);
    cudaFree(d_cols);
    cudaFree(d_x);
    cudaFree(d_y);
    return true;
}

//----------------------------------------------------------------------
// G = X^T X on the GPU (cuBLAS gemm).  Returns true on success; false
// falls back to the CPU product.  X is the m x k column materialisation
// of the reduced operator -- the normal matrix of SSN / smoothing.
//----------------------------------------------------------------------
bool smop_gpu_xtx(const Mat& X, Mat& G)
{
    const int m = static_cast<int>(X.rows()), k = static_cast<int>(X.cols());
    if (m <= 0 || k <= 0) return false;
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev < 1) return false;
    cublasHandle_t h = nullptr;
    if (cublasCreate(&h) != CUBLAS_STATUS_SUCCESS) return false;
    double* d_X = nullptr; double* d_G = nullptr;
    if (cudaMalloc(&d_X, sizeof(double) * static_cast<size_t>(m) * static_cast<size_t>(k)) != cudaSuccess ||
        cudaMalloc(&d_G, sizeof(double) * static_cast<size_t>(k) * static_cast<size_t>(k)) != cudaSuccess) {
        if (d_X) cudaFree(d_X);
        if (d_G) cudaFree(d_G);
        cublasDestroy(h);
        return false;
    }
    cudaMemcpy(d_X, X.data(), sizeof(double) * static_cast<size_t>(m) * static_cast<size_t>(k),
               cudaMemcpyHostToDevice);
    G = Mat::Zero(k, k);
    const double alpha = 1.0, beta = 0.0;
    cublasDgemm(h, CUBLAS_OP_T, CUBLAS_OP_N, k, k, m, &alpha, d_X, m, d_X, m,
                &beta, d_G, k);
    cudaMemcpy(G.data(), d_G, sizeof(double) * static_cast<size_t>(k) * static_cast<size_t>(k),
               cudaMemcpyDeviceToHost);
    cudaFree(d_X);
    cudaFree(d_G);
    cublasDestroy(h);
    return true;
}

bool smop_gpu_available()
{
    int ndev = 0;
    return (cudaGetDeviceCount(&ndev) == cudaSuccess && ndev > 0);
}

} // namespace smop