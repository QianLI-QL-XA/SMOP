# smop — Verification Report

Date: 2026-09-23 (performance-tuning round). Solver: sieving-based secant level-set method with the

three-tier dispatch (smoothing Newton / SSNAL / ADMM), C++17 header-only core,

compiled with g++ 15.2 `-O3 -march=native -fopenmp` (WSL Ubuntu).

Reference: Q. Li, D. Sun, Y. Yuan, *An Efficient Sieving-Based Secant Method for*

*Sparse Optimization Problems with Least-Squares Constraints*, SIAM J. Optim.

**34**(2):2038–2066 (2024), [doi:10.1137/23M1594443](https://doi.org/10.1137/23M1594443).

## 1. UCI real-data benchmark (δ = 0.1 / 0.5 · ‖b‖)

11 data files from the UCI repository — 10 unique datasets (pyrim\_scaled\_expanded5 / pyrim\_scale\_expanded5 are the same data, identical MD5) (`C:\...\Group lasso\UCIdata\*.mat`,

exported to a sparse-aware binary format by `tests/export_ucidata.py`).



| Dataset           | m×n              | type   | δ=0.1 before  | δ=0.1 now       | speedup | δ=0.5 now |
| ----------------- | ---------------- | ------ | ------------- | --------------- | ------- | --------- |
| pyrim             | 74×201,376       | sparse | 7.6 s         | **0.36 s**      | 21×     | 0.10 s    |
| triazines         | 186×635,376      | sparse | 143 s         | **15.2 s**      | 9.4×    | 0.72 s    |
| bodyfat           | 252×116,280      | dense  | 0.75 s        | 0.73 s          | \~1×    | 0.24 s    |
| mpg               | 392×3,432        | dense  | 4.2 s         | **2.14 s**      | 2.0×    | 0.34 s    |
| space\_ga         | 3,107×5,005      | dense  | 868 s timeout | 320 s timeout ¹ | —       | 0.16 s    |
| abalone           | 4,177×6,435      | dense  | 318 s timeout | 307 s timeout ¹ | —       | 2.65 s    |
| E2006.test        | 3,308×150,358    | sparse | 0.11 s        | 0.14 s          | \~1×    | 0.06 s    |
| E2006.train       | 16,087×150,360   | sparse | 215 s         | **147 s**       | 1.5×    | 0.27 s    |
| log1p.E2006.test  | 3,308×4,272,226  | sparse | 2.2 s         | 1.89 s          | 1.2×    | 1.00 s    |
| log1p.E2006.train | 16,087×4,272,227 | sparse | 55.9 s        | **47.5 s**      | 1.2×    | 1.84 s    |

¹ `space_ga` / `abalone` at δ=0.1 are method-boundary cases: the optimal support

requires 2100–2400 non-zero columns, so the sparsity premise of sieving does not

hold and convergence is slow within the time budget; both converge quickly at δ=0.5.

**Solution consistency:** ‖x‖₁ reproduces the pre-optimization values exactly

(pyrim 0.8603, triazines 1.8652, mpg 738.5933, E2006.train 1663.3650,

log1p.train 1.4967) — the speedups do not change the convergence point.

## 2. δ-gradient test (δ = 0.5 / 0.1 / 0.01 / 0.001 · ‖b‖)



| Dataset     | δ=0.5    | δ=0.1         | δ=0.01                   | δ=0.001                |
| ----------- | -------- | ------------- | ------------------------ | ---------------------- |
| pyrim       | 0.10 s ✓ | 0.36 s ✓      | 4.5 s stall ² (7.3e-3)   | 5.2 s stall ² (5.9e-2) |
| triazines   | 0.72 s ✓ | 15.2 s ✓      | 74 s stall ² (1.3e-1)    | —                      |
| bodyfat     | 0.24 s ✓ | 0.73 s ✓      | 2.0 s ✓                  | 1.5 s ✓                |
| mpg         | 0.34 s ✓ | 2.14 s ✓      | 9.1 s ✓                  | 4.9 s ✓                |
| space\_ga   | 0.16 s ✓ | 320 s timeout | 383 s timeout (feas 7.9) | —                      |
| abalone     | 2.65 s ✓ | 307 s timeout | 558 s timeout (feas 19)  | —                      |
| E2006.test  | 0.06 s ✓ | 0.14 s ✓      | 316 s timeout (feas 3.1) | —                      |
| E2006.train | 0.27 s ✓ | 147 s ✓       | 301 s timeout (feas 8.6) | —                      |
| log1p.test  | 1.00 s ✓ | 1.89 s ✓      | 600 s timeout (feas 13)  | —                      |
| log1p.train | 1.84 s ✓ | 47.5 s ✓      | timeout (same class)     | —                      |

² **Stall (status=3)**: ψ(μ) freezes above δ as μ→0 — δ is below the dataset's

achievable residual (≈6.5e-2·‖b‖ for pyrim), so the outer bisection has no root

to find. The solver now detects and reports this cleanly ("delta unreachable")

instead of spinning through 200 iterations.

**Pattern:** difficulty grows monotonically as δ shrinks — δ=0.5 all pass

(≤ 2.65 s); δ=0.1 eight pass (two ill-conditioned timeouts); δ=0.01 only

low-noise datasets converge; δ=0.001 still fine on small datasets (bodyfat, mpg).

## 3. Subproblem dispatch



| Subproblem shape                              | Primary solver       | Rationale                                                                                 |
| --------------------------------------------- | -------------------- | ----------------------------------------------------------------------------------------- |
| `m > nr` (tall)                               | **smoothing Newton** | nr×nr normal equations; 30 s budget; fall through on failure/timeout                      |
| `nr ≥ m` or large active set (incl. m ≥ 3000) | **SSNAL**            | dense LDLT (m ≤ 1500) / PCG (m > 1500); nested linsys pass-through gives 8–13 s per round |
| both fail / disabled                          | **ADMM**             | strict last resort only (never preferred for large active sets)                           |

## 4. GPU acceleration (optional, off by default)

`SMOP_GPU=1` with `-DSMOP_USE_CUDA` (cuBLAS gemv/gemm + hand-written CSR/CSC

SpMV and column-subset kernels; see `docs/GPU.md`). Solutions are bitwise

consistent with the CPU path.



| Dataset                           | CPU    | GPU         | change                  |
| --------------------------------- | ------ | ----------- | ----------------------- |
| triazines δ=0.1 (sparse, reduced) | 14.0 s | **13.09 s** | −7%                     |
| bodyfat δ=0.1 (dense)             | 0.88 s | **0.72 s**  | −18%                    |
| mpg δ=0.1 (dense small)           | 2.08 s | 2.18 s      | +5% (transfer overhead) |
| space\_ga δ=0.1 per-SSN-round     | \~13 s | **5.45 s**  | \~2.4×                  |
| abalone δ=0.1 per-SSN-round       | \~19 s | **14.2 s**  | \~1.3×                  |

Full-matrix SpMV alone (E2006.train 151.6→147.6 s, log1p.train 45.9→49.2 s) is

neutral because the current dispatch is SSN-dominated and subproblems run on the

CPU; the dense/SSN GPU path is where the wins are.

## 5. Reproducibility



```
\# export all .mat files -> tests/ucidata/\*.bin

python tests/export\_ucidata.py

\# compile & run everything in WSL (Linux), -fopenmp required

wsl -d Ubuntu -u root -- bash -lc "g++ -std=c++17 -O3 -march=native -fopenmp \\

&#x20; -I /mnt/c/.../smop/include -I /usr/include/eigen3 \\

&#x20; /mnt/c/.../smop/examples/example\_all\_datasets.cpp \\

&#x20; -o /root/smop-build/example\_all && cd /root/smop-build && ./example\_all"

\# per-dataset subset / time budget

\# SMOP\_NAMES=mpg\_scale\_expanded7 SMOP\_TLIMIT=600 ./example\_all
```

Raw logs: `tests/ucidata_logs/*.log`. Regression suite: `tests/test_cpp.cpp`

(T1–T5), `tests/pysmop_ref.py`, `matlab/test_smop_octave.m`.