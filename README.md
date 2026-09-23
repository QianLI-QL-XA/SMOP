# smop

**Level-set method for sparse optimization with a least-squares constraint.**



```
min  ‖x‖₁    s.t.    ‖Ax − b‖₂ ≤ δ
```

`smop` implements the *sieving-based secant level-set method* of

**Qian Li, Defeng Sun, Yancheng Yuan**, *"An Efficient Sieving-Based Secant Method*

*for Sparse Optimization Problems with Least-Squares Constraints"*,

SIAM Journal on Optimization **34**(2), 2038–2066 (2024)

[DOI: 10.1137/23M1594443](https://doi.org/10.1137/23M1594443).

The numerical core is **header-only C++17** (Eigen) with interfaces for

**C++ / Python / R / MATLAB (Octave)**. The method combines an outer

level-set (secant / Newton / bisection) root-finding on the Pareto frontier of

`μ ↦ ‖x(μ)‖₁` vs `ψ(μ) = ‖Ax(μ) − b‖`, adaptive sieving to keep subproblems

small, and a three-tier subproblem dispatch: **smoothing Newton** →

**SSNAL** (semi-smooth Newton augmented Lagrangian) → **ADMM** (strict

fallback). An optional **CUDA** path offloads dense matrix–vector products and

SSN normal-matrix builds to the GPU.

## Features



* Header-only C++17 core (10 headers + one optional CUDA translation unit), Eigen-based, no other dependencies.

* OpenMP-parallel operator application on large sparse matrices.

* Adaptive sieving (port of `AdaptSieving_lasso_solve.m` from the reference code) with automatic subproblem-size control.

* Subproblem dispatch mirroring the paper: smoothing Newton for tall subproblems, SSNAL for large active sets, ADMM only when both fail.

* First-class interfaces: Python (pybind11), R (Rcpp), MATLAB (MEX, verified under Octave 11 too).

* Optional CUDA acceleration (`SMOP_GPU=1`): cuBLAS gemv/gemm + hand-written SpMV and column-subset kernels (see `docs/GPU.md`).

## Requirements



| Interface | Requirements                                                                               |
| --------- | ------------------------------------------------------------------------------------------ |
| C++       | C++17 compiler (g++ / clang++ / MSVC), [Eigen 3.4.x](https://eigen.tuxfamily.org/) headers |
| Python    | Python ≥ 3.8, numpy, pybind11 ≥ 2.11, Eigen                                                |
| R         | R ≥ 3.6, Rcpp ≥ 1.0, Eigen                                                                 |
| MATLAB    | MATLAB with a C++17 MEX compiler; Octave ≥ 8 with `mkoctfile` also works                   |

## Getting Eigen (all platforms)



```
cd smop/third\_party

curl -L -o eigen.tar.gz https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz

tar xzf eigen.tar.gz        # -> third\_party/eigen-3.4.0/
```

You may instead install Eigen from the system package manager and pass its

include path explicitly (see platform sections below).



***

## Platform guides

### Linux (Ubuntu / Debian)



```
\# 1. toolchain

sudo apt-get update && sudo apt-get install -y g++ libeigen3-dev

\# 2. C++ (either CMake or direct g++)

cmake -S . -B build && cmake --build build --config Release

./build/test\_cpp                          # regression suite (T1–T5)

\# or directly:

g++ -std=c++17 -O3 -march=native -fopenmp \\

&#x20;   -I include -I /usr/include/eigen3 \\

&#x20;   examples/example\_cpp.cpp -o example\_cpp

./example\_cpp

\# 3. Python

pip install numpy pybind11

cd python && pip install .

python -c "import smop; print(smop.version())"      # 0.1.0

\# 4. R (needs r-cran-rcpp / Rcpp)

cd R && R CMD INSTALL --preclean .

Rscript -e 'library(smop); print(smop\_version())'

\# 5. MATLAB / Octave

cd matlab

mkoctfile --mex -I../include -I../third\_party/eigen-3.4.0 smop\_mex.cpp smop\_lasso\_mex.cpp

octave --no-gui test\_smop\_octave.m
```

### macOS



```
\# 1. toolchain + Eigen (Homebrew; Intel Mac uses /usr/local/include/eigen3)

brew install gcc libomp eigen

\#   - M-series:  Eigen at /opt/homebrew/include/eigen3

\#   - Intel:     Eigen at /usr/local/include/eigen3

\# 2. C++

cmake -S . -B build && cmake --build build --config Release

./build/test\_cpp

\# or directly (add -I /opt/homebrew/include/eigen3 on Apple Silicon):

g++ -std=c++17 -O3 -fopenmp -I include -I /opt/homebrew/include/eigen3 \\

&#x20;   examples/example\_cpp.cpp -o example\_cpp

./example\_cpp

\# (clang++ also works; needs \`brew install libomp\` for -fopenmp, or drop -fopenmp)

\# 3. Python

pip install numpy pybind11

cd python && pip install .

python -c "import smop; print(smop.version())"

\# 4. R

cd R && R CMD INSTALL --preclean .

Rscript -e 'library(smop); print(smop\_version())'

\# 5. MATLAB / Octave

cd matlab

mkoctfile --mex -I../include -I../third\_party/eigen-3.4.0 smop\_mex.cpp smop\_lasso\_mex.cpp

octave --no-gui test\_smop\_octave.m

\# in MATLAB itself:  cd matlab; build\_smop   (needs a C++17 mex compiler)
```

### Windows

**Recommended path: WSL (Ubuntu)**. Windows-native builds are possible but newly

built executables may be blocked by Smart App Control (SAC); the WSL toolchain

is the stable, fully verified channel.



```
\# 1. install WSL + Ubuntu (one time)

wsl --install -d Ubuntu

wsl -d Ubuntu -u root -- bash -lc "apt-get update && apt-get install -y g++ libeigen3-dev"

\# 2. C++ (build & run entirely inside WSL)

wsl -d Ubuntu -u root -- bash -lc "g++ -std=c++17 -O3 -march=native -fopenmp \\

&#x20; -I /mnt/c/Users/\<you>/.../smop/include -I /usr/include/eigen3 \\

&#x20; /mnt/c/Users/\<you>/.../smop/examples/example\_cpp.cpp -o /root/example\_cpp && /root/example\_cpp"

\# 3. Python / R / MATLAB: same commands as the Linux section, inside WSL.
```

**Alternative: native MSYS2 + VS Code** (Windows-only; see SAC caveat below).



* Install MSYS2 with the UCRT64 toolchain, then set the VS Code `C/C++` extension

  IntelliSense to `C:/msys64/ucrt64/bin/g++.exe` (the repo ships

  `.vscode/c_cpp_properties.json` and `.vscode/tasks.json`; `Ctrl+Shift+B`

  compiles the test suite).

* Build:  `g++ -std=c++17 -O2 -I include -I <eigen3 path> examples/example_cpp.cpp -o example_cpp`

* **Caveat**: on machines with Smart App Control enabled, compiling object files

  works but *running* freshly built `.exe` may be blocked. Use the WSL path, or

  validate numerically with `tests/pysmop_ref.py` / the Python interface.

* Python / R / MATLAB on native Windows follow the same commands as Linux

  (Python: `pip install ./python`; R: needs Rtools; MATLAB: `build_smop` with a

  MinGW-w64 or MSVC C++17 mex compiler).



***

## Quick start

### C++



```
\#include "smop/smop.hpp"

Eigen::MatrixXd A = ...;            // m × n

Eigen::VectorXd b = ...;            // m

smop::SmopResult r = smop::solve\_bmop(A, b, delta);

// r.x : solution;  r.info.status == 0 -> converged

smop::LassoResult lr = smop::solve\_lasso(A, b, lambda);
```

### Python



```
import numpy as np, smop

A = np.random.standard\_normal((40, 150))

b = np.random.standard\_normal(40)

delta = 0.5 \* np.linalg.norm(b)

res = smop.solve\_bmop(A, b, delta, verbose=1)

x = res\["x"]

print(res\["status"], res\["msg"])

print(np.linalg.norm(A @ x - b), delta)          # constraint satisfaction
```

### R



```
library(smop)

res <- solve\_bmop(A, b, delta, options = list(verbose = 0))

cat(res\$status, res\$msg)

lr <- solve\_lasso(A, b, 0.05)
```

### MATLAB



```
res = smop(A, b, delta);

fprintf('%d %s\n', res.status, res.msg);

fprintf('%g %g\n', norm(A\*res.x-b), delta);
```

Full option lists, result schemas, and per-language details:

[docs/INTERFACES.md](docs/INTERFACES.md).



***

## Actual numerical results

### 4.1 UCI benchmark (11 data files, 10 unique datasets)

11 `.mat` files from the UCI repository (`pyrim_scaled_expanded5` and

`pyrim_scale_expanded5` are the same data, identical MD5). Time is wall-clock

for the full solve, δ = scale · ‖b‖. Machine: WSL Ubuntu, g++ 15.2

`-O3 -march=native -fopenmp`.



| Dataset           | m×n              | type   | δ=0.1     | δ=0.5  | converged ‖x‖₁ (δ=0.1) |
| ----------------- | ---------------- | ------ | --------- | ------ | ---------------------- |
| pyrim             | 74×201,376       | sparse | 0.36 s    | 0.10 s | 0.8603                 |
| triazines         | 186×635,376      | sparse | 15.2 s    | 0.72 s | 1.8652                 |
| bodyfat           | 252×116,280      | dense  | 0.73 s    | 0.24 s | —                      |
| mpg               | 392×3,432        | dense  | 2.14 s    | 0.34 s | 738.5933               |
| space\_ga         | 3,107×5,005      | dense  | timeout ¹ | 0.16 s | —                      |
| abalone           | 4,177×6,435      | dense  | timeout ¹ | 2.65 s | —                      |
| E2006.test        | 3,308×150,358    | sparse | 0.14 s    | 0.06 s | —                      |
| E2006.train       | 16,087×150,360   | sparse | 147 s     | 0.27 s | 1663.3650              |
| log1p.E2006.test  | 3,308×4,272,226  | sparse | 1.89 s    | 1.00 s | —                      |
| log1p.E2006.train | 16,087×4,272,227 | sparse | 47.5 s    | 1.84 s | 1.4967                 |

¹ `space_ga` / `abalone` at δ=0.1 are method-boundary cases: the optimal support

needs 2100–2400 non-zero columns, so the sparsity premise of sieving does not

hold; both converge quickly at δ=0.5.

### 4.2 δ-gradient test (δ = 0.5 / 0.1 / 0.01 / 0.001 · ‖b‖)

Difficulty grows monotonically as δ shrinks — δ=0.5 all 10 pass (≤ 2.65 s);

δ=0.1 eight pass; δ=0.01 only low-noise datasets (bodyfat, mpg) converge and

pyrim/triazines report "delta unreachable" (δ below the achievable residual);

δ=0.001 still fine on small datasets (bodyfat 1.5 s, mpg 4.9 s).

### 4.3 GPU (optional, RTX 5060 / WSL)

Solutions bitwise-consistent with CPU. bodyfat δ=0.1: 0.88 s → 0.72 s (−18%);

triazines δ=0.1: 14.0 s → 13.09 s (−7%); per-SSN-round on large dense active

sets: \~2.4× (space\_ga 13 s → 5.45 s). Full-matrix SpMV alone is neutral — the

wins come from the dense/SSN GPU path (`SMOP_GPU=1`, see `docs/GPU.md`).

Full tables, stalls, and the dispatch rationale:

[docs/VERIFICATION.md](docs/VERIFICATION.md).



***

## Repository layout



```
smop/

├── include/smop/          C++ core (header-only)

├── src/gpu\_operator.cu    optional CUDA kernels (nvcc TU)

├── python/                pybind11 package

├── R/                     Rcpp package

├── matlab/                MEX interface (smop / smop\_lasso)

├── examples/              per-language runnable examples

├── tests/                 C++ regression, UCI export + results

└── docs/                  GPU.md, INTERFACES.md, VERIFICATION.md
```

## Citing



```
@article{LiSunYuan2024,

&#x20; author  = {Li, Qian and Sun, Defeng and Yuan, Yancheng},

&#x20; title   = {An Efficient Sieving-Based Secant Method for Sparse Optimization

&#x20;            Problems with Least-Squares Constraints},

&#x20; journal = {SIAM Journal on Optimization},

&#x20; volume  = {34},

&#x20; number  = {2},

&#x20; pages   = {2038--2066},

&#x20; year    = {2024},

&#x20; doi     = {10.1137/23M1594443}

}
```

## License

[MIT](LICENSE) © 2026 Qian Li.