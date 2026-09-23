# smop 多语言接口使用说明

smop：sieving-based secant level-set method 求解稀疏优化问题（`min ||x||_1  s.t.  ||Ax-b||_2 <= delta`）
与 Lasso（`min 1/2||Ax-b||^2 + lambda||x||_1`）。C++17 核心（header-only + Eigen），
提供 **C++ / Python / R / MATLAB(Octave)** 四种接口。底层算法参考
`Level_set_method.pdf` 与 `ClassicLasso/` 的 MATLAB 实现。

```
smop/
├── include/smop/            # C++17 header-only 核心（10 头文件 + gpu_operator.hpp）
│   ├── types.hpp            # 基础类型 Vec/Mat/SpMat/选项/结果
│   ├── operators.hpp        # LinearOperator/SparseOperator/DenseOperator/SubsetOperator
│   ├── prox.hpp             # 邻近算子
│   ├── linsolve.hpp         # 线性系统（reduced 正规方程 / PCG，GPU 可选）
│   ├── smoothing_newton.hpp # smoothing Newton（瘦高子问题首选）
│   ├── ssn_l1.hpp           # SSNAL（大活跃集首选，稠密 LDLT / PCG）
│   ├── admm_l1.hpp          # ADMM（smoothing/SSN 均失败时的严格兜底）
│   ├── adaptive_sieving.hpp # adaptive sieving + 调度决策
│   ├── level_set.hpp        # 外层 level-set（secant/Newton/bisection）
│   ├── smop.hpp             # 总入口：solve_bmop / solve_lasso
│   └── gpu_operator.hpp     # 可选 CUDA 算子（默认关闭）
├── src/gpu_operator.cu      # CUDA 实现（nvcc 单独编译）
├── python/                  # pybind11 接口（pip install ./python）
├── R/                       # Rcpp 接口（R CMD INSTALL --preclean .）
├── matlab/                  # MEX 接口（build_smop 或 mkoctfile --mex）
├── examples/                # 各接口示例
└── tests/                   # C++ 回归 + UCI 数据导出/测试
```

## 1. C++ 核心

```cpp
#include "smop/smop.hpp"
// A: m×n 稠密 Eigen::MatrixXd 或 LinearOperator（稀疏用 SparseOperator 包装）
smop::SmopOptions opt;
opt.verbose = 1;
smop::SmopResult r = smop::solve_bmop(A, b, delta, opt);
// r.x 解向量；r.info.status==0 收敛；r.info.eta 外层残差；r.info.reducedn 活跃集规模
smop::LassoResult lr = smop::solve_lasso(A, b, lambda, lopt);
```

编译（Linux/WSL，需 Eigen）：
```bash
g++ -std=c++17 -O3 -march=native -fopenmp -I <smop>/include -I /usr/include/eigen3 \
    your.cpp -o your_app
```

## 2. Python 接口

```bash
pip install ./python            # 从 smop/python/ 目录（需 pybind11、numpy、Eigen）
python -c "import smop; print(smop.version())"    # -> 0.1.0
```

```python
import numpy as np, smop

A = np.random.standard_normal((40, 150))   # m×n 稠密 float64
b = np.random.standard_normal(40)
delta = 0.5 * np.linalg.norm(b)

res = smop.solve_bmop(A, b, delta,
                      stoptol=1e-6, maxiter=200, use_secant=True,
                      use_as=True, verbose=1, time_limit=300)
x = res["x"]                      # 解向量
print(res["status"], res["msg"])  # 0 converged
print(np.linalg.norm(A @ x - b), delta)   # 约束满足性

lr = smop.solve_lasso(A, b, 0.05) # Lasso
print(lr["obj"], lr["eta"])
```

`solve_bmop` 可用关键字：`stoptol, maxiter, use_secant, use_newton, use_as,
mu0, muinf, initial_mu, time_limit, verbose, x0`；
`solve_lasso`：`stoptol, maxiter, use_smoothing, use_admm, eps_hat, kappa, rho,
armijo_sigma, eta_hat, maxiter_as, time_limit, x0`。
返回 dict 含 `x, xi, y, mu, psi, eta, iter, iter_bisection, iter_newton_or_secant,
iter_smoothing, iter_admm, status, msg, mupath, psipath, reducedn, time`。

## 3. R 接口

```bash
R CMD INSTALL --preclean .        # 从 smop/R/ 目录（需 Rcpp；Eigen 在 third_party/）
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
cat(norm(A %*% res$x - b, "2"), delta)   # 约束满足性
lr <- solve_lasso(A, b, 0.05)            # Lasso
smop_version()                           # "0.1.0"
```

约定：`b` 长度须等于 `A` 的行数；若传入的是列向量数据（`b` 长度 = `ncol(A)`），
接口自动转置。`options` 为命名列表，字段同 Python 关键字。

## 4. MATLAB / Octave 接口

MATLAB（需 C++17 mex 编译器）：
```matlab
cd smop/matlab
build_smop          % 一次性编译 smop_mex + smop_lasso_mex
res = smop(A, b, delta);                    % A:m×n 稠密, b:m×1
res = smop(A, b, delta, struct('verbose',1,'stoptol',1e-6));
fprintf('%d %s\n', res.status, res.msg);
fprintf('%g %g\n', norm(A*res.x-b), delta);
lr = smop_lasso(A, b, 0.05);              % Lasso 子问题
fprintf('obj %g  kkt %g  status %d\n', lr.obj, lr.eta, lr.status);
```

Octave（无 MATLAB 时的替代验证通道）：
```bash
cd smop/matlab
mkoctfile --mex -I../include -I../third_party/eigen-3.4.0 smop_mex.cpp smop_lasso_mex.cpp
octave --no-gui test_smop_octave.m
```

`smop` 返回 struct 字段：`x, xi, y, mu, psi, eta, iter, iter_bisection,
iter_newton_or_secant, iter_smoothing, iter_admm, status, msg, mupath,
psipath, reducedn`；`smop_lasso` 返回：`x, xi, grad, obj, iter, eta, status, msg, time`。

## 5. 接口一致性验证（2026-09-23，全部通过）

同一随机问题（m=40, n=150, 8 稀疏真解, seed=0, delta=1.02·‖A·x_true−b‖）：

| 接口 | solve_bmop status | ‖Ax−b‖ vs δ | ‖x‖₁ (真 7.63) | iter | solve_lasso KKT |
| --- | --- | --- | --- | --- | --- |
| C++ 核心（T1–T5） | 0 converged | ≤δ | 一致 | — | <1e-6 |
| Python | 0 converged | 1.29e-01 ≈ 1.29e-01 | 7.586 | 7 | obj 3.44e-01 |
| R | 0 converged | 1.292e-01 ≈ 1.292e-01 | 7.586 | 7 | 6.7e-07 |
| MATLAB/Octave | 0 converged | 1.630e-01 ≈ 1.630e-01 | 8.648 | 7 | 6.6e-07 |

（Python/R 同 seed 解完全一致；Octave 与 MATLAB 的 randn 流不同故数值略异，
但约束满足、收敛性一致。）

### 全 UCI 数据回归（C++ 核心，11 数据集 × δ=0.1/0.5）

- δ=0.5：11/11 全部收敛（0.08–2.81s）；
- δ=0.1：9/11 收敛，`‖x‖₁` 与历史基线逐项一致
  （pyrim 0.8603、triazines 1.8652、mpg 738.5933、E2006.train 1663.3650、
  log1p.train 1.4967 …）；
- space_ga / abalone δ=0.1 超时属方法边界（解需 2000–3000+ 非零列）；
- GPU 版（SMOP_USE_CUDA）解与 CPU 完全一致（见 `docs/GPU.md`）。

## 6. 常见问题

- **Windows 直接编译被 Smart App Control 拦截**：本项目产物在 Windows 上可能被
  SAC 拦截，唯一稳定通道是 WSL Ubuntu（g++ 15.2 + `-fopenmp`），见 `build_wsl.ps1`。
- **Python 找不到模块**：确认从 `smop/python/` 运行 `pip install .`（src 布局），
  Eigen 在 `smop/third_party/eigen-3.4.0/`。
- **R 包安装报 include 路径错**：Makevars 用 `-I../../include`
  （相对 `R/src/`），已修正；重装时 `R CMD INSTALL --preclean .`。
- **R 接口重编译注意**：若用 `Rcpp::compileAttributes()` 重新生成
  `src/RcppExports.cpp`，需删除生成文件末尾的旧式注册段
  （`RcppExport SEXP smop_version_cpp(void); ...` 及
  `{"smop_solve_bmop", ...}` 等旧名条目），否则与 `std::string` 返回的函数
  重名冲突导致编译失败；只保留 `_smop_*` 新风格注册。
- **Octave 用 mkoctfile 必须加 `--mex`**：默认按 .oct 格式编译会得到无法加载
  的产物。
- **GPU 加速**：默认关闭；需要时见 `docs/GPU.md` 的编译与运行说明。
