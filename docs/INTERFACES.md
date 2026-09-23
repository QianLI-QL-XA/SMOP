# SMOP multi-language interface guide

SMOP: sieving-based secant level-set method for sparse optimization problems
(`min ||x||_1  s.t.  ||Ax-b||_2 <= delta`) and for the Lasso
(`min 1/2||Ax-b||^2 + lambda||x||_1`). The C++17 core is header-only (Eigen)
and ships with **C++ / Python / R / MATLAB (Octave)** interfaces. The underlying
algorithm follows `Level_set_method.pdf` and the MATLAB implementation in
`ClassicLasso/`.

```
smop/
├── include/smop/            # C++17 header-only core (10 headers + gpu_operator.hpp)
│   ├── types.hpp            # base types Vec/Mat/SpMat/options/results
│   ├── operators.hpp        # LinearOperator/SparseOperator/DenseOperator/SubsetOperator
│   ├── prox.hpp             # proximal operators
│   ├── linsolve.hpp         # linear solvers (reduced normal equations / PCG, optional GPU)
│   ├── smoothing_newton.hpp # smoothing Newton (preferred for tall subproblems)
│   ├── ssn_l1.hpp           # SSNAL (preferred for large active sets; dense LDLT / PCG)
│   ├── admm_l1.hpp          # ADMM (strict last resort when smoothing/SSN fail)
│   ├── adaptive_sieving.hpp # adaptive sieving + dispatch decisions
│   ├── level_set.hpp        # outer level-set loop (secant/Newton/bisection)
│   ├── smop.hpp             # entry point: solve_bmop / solve_lasso
│   └── gpu_operator.hpp     # optional CUDA operators (disabled by default)
├── src/gpu_operator.cu      # CUDA implementation (compiled separately with nvcc)
├── python/                  # pybind11 interface (pip install ./python)
├── R/                       # Rcpp interface (R CMD INSTALL --preclean .)
├── matlab/                  # MEX interface (build_smop or mkoctfile --mex)
├── examples/                # examples for each interface
└── tests/                   # C++ regression + UCI data export/tests
```

## 1. C++ core

```cpp
#include "smop/smop.hpp"
// A: m×n dense Eigen::MatrixXd or a LinearOperator (wrap sparse matrices with SparseOperator)
smop::SmopOptions opt;
opt.verbose = 1;
smop::SmopResult r = smop::solve_bmop(A, b, delta, opt);
// r.x solution vector; r.info.status==0 converged; r.info.eta outer residual; r.info.reducedn active-set size
smop::LassoResult lr = smop::solve_lasso(A, b, lambda, lopt);
```

Compile (Linux/WSL, needs Eigen):
```bash
g++ -std=c++17 -O3 -march=native -fopenmp -I <smop>/include -I /usr/include/eigen3 \
    your.cpp -o your_app
```

## 2. Python interface

```bash
pip install ./python            # from smop/python/ (needs pybind11, numpy, Eigen)
python -c "import smop; print(smop.version())"    # -> 0.1.0
```

```python
import numpy as np, smop

A = np.random.standard_normal((40, 150))   # m×n dense float64
b = np.random.standard_normal(40)
delta = 0.5 * np.linalg.norm(b)

res = smop.solve_bmop(A, b, delta,
                      stoptol=1e-6, maxiter=200, use_secant=True,
                      use_as=True, verbose=1, time_limit=300)
x = res["x"]                      # solution vector
print(res["status"], res["msg"])  # 0 converged
print(np.linalg.norm(A @ x - b), delta)   # constraint satisfaction

lr = smop.solve_lasso(A, b, 0.05) # Lasso
print(lr["obj"], lr["eta"])
```

`solve_bmop` keywords: `stoptol, maxiter, use_secant, use_newton, use_as,
mu0, muinf, initial_mu, time_limit, verbose, x0`;
`solve_lasso`: `stoptol, maxiter, use_smoothing, use_admm, eps_hat, kappa, rho,
armijo_sigma, eta_hat, maxiter_as, time_limit, x0`.
Returned dict contains `x, xi, y, mu, psi, eta, iter, iter_bisection,
iter_newton_or_secant, iter_smoothing, iter_admm, status, msg, mupath,
psipath, reducedn, time`.

## 3. R interface

```bash
R CMD INSTALL --preclean .        # from smop/R/ (needs Rcpp; Eigen ships in third_party/)
```

```r
library(smop)
set.seed(0)
m <- 40; n <- 150
A <- matrix(rnorm(m * n), m, n)
xtrue <- rep(0, n); xtrue[sample(n, 8)] <- rnorm(8)
b <- A %*% xtrue + 0.02 * rnorm(m)
delta <- 1.02 * norm(A %*% xtrue - b, "2")

res <- solve_bmop(A, b, delta, options = list(verbose = 0, stoptol = 1e-6))
cat(res$status, res$msg)                 # 0 converged
cat(norm(A %*% res$x - b, "2"), delta)   # constraint satisfaction
lr <- solve_lasso(A, b, 0.05)            # Lasso
smop_version()                           # "0.1.0"
```

Conventions: the length of `b` must equal the number of rows of `A`; if you pass
column-vector data (`b` length = `ncol(A)`), the interface transposes it
automatically. `options` is a named list with the same fields as the Python
keywords.

## 4. MATLAB / Octave interface

MATLAB (needs a C++17 mex compiler):
```matlab
cd smop/matlab
build_smop          % one-time compile of smop_mex + smop_lasso_mex
res = smop(A, b, delta);                    % A:m×n dense, b:m×1
res = smop(A, b, delta, struct('verbose',1,'stoptol',1e-6));
fprintf('%d %s\n', res.status, res.msg);
fprintf('%g %g\n', norm(A*res.x-b), delta);
lr = smop_lasso(A, b, 0.05);              % Lasso subproblem
fprintf('obj %g  kkt %g  status %d\n', lr.obj, lr.eta, lr.status);
```

Octave (alternative verification channel when MATLAB is unavailable):
```bash
cd smop/matlab
mkoctfile --mex -I../include -I../third_party/eigen-3.4.0 smop_mex.cpp smop_lasso_mex.cpp
octave --no-gui test_smop_octave.m
```

`smop` returns a struct with fields: `x, xi, y, mu, psi, eta, iter,
iter_bisection, iter_newton_or_secant, iter_smoothing, iter_admm, status, msg,
mupath, psipath, reducedn`; `smop_lasso` returns: `x, xi, grad, obj, iter, eta,
status, msg, time`.

## 5. Interface consistency verification (2026-09-23, all passed)

Same random problem (m=40, n=150, 8-sparse true solution, seed=0,
delta=1.02·‖A·x_true−b‖):

| Interface | solve_bmop status | ‖Ax−b‖ vs δ | ‖x‖₁ (true 7.63) | iter | solve_lasso KKT |
| --- | --- | --- | --- | --- | --- |
| C++ core (T1–T5) | 0 converged | ≤δ | consistent | — | <1e-6 |
| Python | 0 converged | 1.29e-01 ≈ 1.29e-01 | 7.586 | 7 | obj 3.44e-01 |
| R | 0 converged | 1.292e-01 ≈ 1.292e-01 | 7.586 | 7 | 6.7e-07 |
| MATLAB/Octave | 0 converged | 1.630e-01 ≈ 1.630e-01 | 8.648 | 7 | 6.6e-07 |

(Python and R produce identical solutions for the same seed; Octave/MATLAB
differ slightly because their randn streams differ, but constraint
satisfaction and convergence are consistent.)

### Full UCI regression (C++ core, 11 datasets × δ=0.1/0.5)

- δ=0.5: 11/11 converged (0.08–2.81 s);
- δ=0.1: 9/11 converged, `‖x‖₁` matches the historical baseline item by item
  (pyrim 0.8603, triazines 1.8652, mpg 738.5933, E2006.train 1663.3650,
  log1p.train 1.4967 …);
- space_ga / abalone δ=0.1 time out — this is a method boundary (their δ=0.1
  solutions need 2000–3000+ nonzero columns);
- GPU build (SMOP_USE_CUDA) produces exactly the same solutions as CPU (see
  `docs/GPU.md`).

## 6. Frequently asked questions

- **Compiling directly on Windows is blocked by Smart App Control**: Windows may
  flag our binaries with SAC; the only stable build channel is WSL Ubuntu
  (g++ 15.2 + `-fopenmp`), see `build_wsl.ps1`.
- **Python cannot find the module**: make sure you run `pip install .` from
  `smop/python/` (src layout); Eigen lives in `smop/third_party/eigen-3.4.0/`.
- **R package install reports wrong include path**: Makevars uses
  `-I../../include` (relative to `R/src/`), already fixed; reinstall with
  `R CMD INSTALL --preclean .`.
- **R interface recompilation**: if you regenerate `src/RcppExports.cpp` with
  `Rcpp::compileAttributes()`, delete the legacy registration block at the end
  (the old `RcppExport SEXP smop_version_cpp(void); ...` declarations and
  `{"smop_solve_bmop", ...}` entries), otherwise they collide with the
  `std::string`-returning functions and compilation fails; keep only the new
  `_smop_*` registrations.
- **Octave mkoctfile must use `--mex`**: the default .oct format produces
  binaries that cannot be loaded.
- **GPU acceleration**: disabled by default; when needed see `docs/GPU.md` for
  build and run instructions.
