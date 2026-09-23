# GPU acceleration support (optional)

SMOP ships an optional CUDA path: `GpuSparseOperator` / `GpuDenseOperator`
(`include/smop/gpu_operator.hpp` + `src/gpu_operator.cu`), covering three
classes of matrix computation:

- **Sparse full-dimensional SpMV** (`A x` / `Aᵀ y`, hand-written CSR/CSC
  kernels) — outer level-set sieving;
- **Dense full-dimensional gemv** (cuBLAS `cublasDgemv`) — top-level operator
  for dense UCI data;
- **SSN/smoothing subproblems** (added in this round):
  - `GpuDenseOperator::subset_gemv`: column-sliced products (the `App`/`Appᵀ`
    of PCG matvec), a hand-written kernel that picks columns of the GPU dense
    matrix by index;
  - `smop_gpu_xtx`: reduced Gram matrix `G = CᵀC` (cuBLAS `cublasDgemm`).

`ssn_l1.hpp` (`G` construction and PCG matvec in the reduced branch) and
`linsolve.hpp` (`normal_solve` / `gram_matrix`) prefer GPU under
`SMOP_USE_CUDA` and fall back to CPU automatically on failure.

## Evaluation (RTX 5060 / WSL Ubuntu, measured 2026-09)

### δ=0.1 (sparse solutions) — after GPU-izing the subproblems

| Dataset | Shape | CPU (OpenMP, native) | GPU (SMOP_GPU=1) | Solution match |
| --- | --- | ---: | ---: | --- |
| triazines | sparse 77.6M nnz | 14.0 s | 13.09 s | yes (1.8652) |
| bodyfat | dense 252×116280 | 0.88 s | 0.72 s | yes (1.6786) |
| mpg | dense 392×3432 | 2.08 s | 2.18 s | yes (738.5933) |

- Small matrices (mpg): GPU overhead slightly exceeds the gain, which is
  expected; for mid/large matrices the reduced branch (triazines) gets a
  stable ~7% gain from the `G=CᵀC` GPU gemm.
- space_ga / abalone at δ=0.1: in the first few rounds each SSN round dropped
  from ~13–19 s (CPU) to 5.5–14 s (GPU, about 2.4×), but the whole run still
  times out (611 s / 613 s, feas 0.13–0.35 not converged) — this is a method
  boundary, not a GPU issue: their δ=0.1 solutions need 2000–3000+ nonzero
  columns, so the active set explodes.

### δ=0.5 (sparse-solution scenario, user-specified regime)

| Dataset | CPU | GPU | Solution match |
| --- | ---: | ---: | --- |
| bodyfat | 1.07 s | 0.16 s | yes |
| mpg | 2.08 s | 0.47 s | yes |
| space_ga | 0.11 s | 0.11 s | yes (converged) |
| abalone | 7.57 s | 7.6 s | yes (converged) |

Dense full-dimensional gemv pays off at δ=0.5 (sparser solution, small active
set, many top-level calls); δ=0.1 is dominated by SSN subproblems.

## Build

```bash
# 1) compile the CUDA object (host code must share the instruction set with the
#    caller, otherwise cross-TU memory errors occur)
nvcc -std=c++17 -O3 -arch=sm_120 -Xcompiler -march=native \
     -I <smop>/include -I /usr/include/eigen3 \
     -I /usr/local/cuda-13.3/targets/x86_64-linux/include \
     -c <smop>/src/gpu_operator.cu -o gpu_operator.o

# 2) compile the caller and link (same -march=native, needs SMOP_USE_CUDA)
g++ -std=c++17 -O3 -march=native -fopenmp -DSMOP_USE_CUDA \
    -I <smop>/include -I /usr/include/eigen3 \
    example.cpp gpu_operator.o \
    -L/usr/local/cuda-13.3/targets/x86_64-linux/lib -lcudart -lcublas \
    -Xlinker -rpath -Xlinker /usr/local/cuda-13.3/lib64 \
    -Xlinker -rpath -Xlinker /usr/local/cuda-13.3/targets/x86_64-linux/lib \
    -o example
```

cuBLAS headers/libraries live in
`/usr/local/cuda-13.3/targets/x86_64-linux/{include,lib}`
(`apt install libcublas-13-3 libcublas-dev-13-3`). The CPU build (without
`SMOP_USE_CUDA`) has zero CUDA dependencies and behaves exactly the same.

## Runtime

```bash
SMOP_GPU=1 SMOP_NAMES=... ./example
```

`example_all_datasets` enables the GPU branch only when compiled with
`SMOP_USE_CUDA` and run with `SMOP_GPU=1` (sparse nnz≥1,000,000 or dense
size≥1,000,000). With no CUDA device both GPU operators fall back to CPU
automatically (they keep an internal Eigen copy); `smop_gpu_xtx` /
`subset_gemv` return false and the call site falls back to CPU.

## Implementation notes (pitfalls logged)

1. **Cross-TU compilation**: GPU operators do not inherit the CPU operators.
   When an Eigen `SparseMatrix` is constructed in the nvcc TU and destroyed in
   the g++ TU, `-O3` triggers `free(): invalid pointer` due to allocator/
   alignment mismatch; instead we use a `LinearOperator` base + internal
   `std::unique_ptr` (the Eigen object's lifetime stays entirely inside the
   .cu file) and expose the CPU matrix to `SubsetOperator` via virtual
   `sparse_matrix()` / `dense_matrix()` (no `dynamic_cast` across TUs).
2. **Eigen sparse transpose miscompiles under nvcc**: a Sparse2Sparse
   assignment of `At = A.transpose()` executed on the host produces wrong CSR
   indices; replaced by a hand-written counting-sort CSC→CSR conversion.
3. **`-march=native` must be used on both sides** — nvcc
   (`-Xcompiler -march=native`) and g++ — otherwise cross-TU alignment
   assumptions diverge (crashes under O3, works under O1/ASAN — a key clue).
4. **cuBLAS arguments**: `cublasDgemv` / `cublasDgemm` alpha/beta must be
   pointers (`const double alpha=1.0, beta=0.0;` and take their addresses).
5. **Device memory**: the sparse operator keeps CSR + CSC copies ≈ 2 × nnz ×
   16 B (log1p 96.7M nnz ≈ 3.1 GB); the dense operator keeps one full matrix
   (space_ga 3107×5005 ≈ 124 MB).
6. **Column-list uploads in subset_gemv**: each PCG matvec uploads `cols`
   (int×sp) plus the input vector; for sp≤10000 the overhead is negligible.
   This is the trade-off for "GPU touches only subset columns" — the CPU
   column-slice path (`SubsetOperator`) is already fast, and the GPU version
   mainly serves dense large active sets.
