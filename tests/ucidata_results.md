# SMOP 全 UCI 数据集测试结果

日期：2026-09-23（性能优化轮）；求解器：sieving-based secant level-set 方法（smoothing Newton / SSNAL / ADMM 三路调度，C++17 header-only，WSL g++ 15.2 -O3 -fopenmp 编译）

数据源：`C:\Users\qianl\OneDrive\codes\Group lasso\UCIdata\*.mat`（导出为 `tests\ucidata\*.bin`，统一稠密 / 稀疏 CSC 格式）

## 汇总表（δ=0.1・‖b‖ 与 δ=0.5・‖b‖，本轮优化后）



| 数据集               | m×n           | 类型 | δ=0.1 优化前 | δ=0.1 本轮  | 提速   | δ=0.5 本轮 |
| ----------------- | ------------- | -- | --------- | --------- | ---- | -------- |
| pyrim             | 74×201376     | 稀疏 | 7.6s      | **0.36s** | 21×  | 0.10s    |
| triazines         | 186×635376    | 稀疏 | 143s      | **15.2s** | 9.4× | 0.72s    |
| bodyfat           | 252×116280    | 稠密 | 0.75s     | 0.73s     | \~1× | 0.24s    |
| mpg               | 392×3432      | 稠密 | 4.2s      | **2.14s** | 2.0× | 0.34s    |
| space\_ga         | 3107×5005     | 稠密 | 868s 超时   | 320s 超时 ¹ | —    | 0.16s    |
| abalone           | 4177×6435     | 稠密 | 318s 超时   | 307s 超时 ¹ | —    | 2.65s    |
| E2006.test        | 3308×150358   | 稀疏 | 0.11s     | 0.14s     | \~1× | 0.06s    |
| E2006.train       | 16087×150360  | 稀疏 | 215s      | **147s**  | 1.5× | 0.27s    |
| log1p.E2006.test  | 3308×4272226  | 稀疏 | 2.2s      | 1.89s     | 1.2× | 1.00s    |
| log1p.E2006.train | 16087×4272227 | 稀疏 | 55.9s     | **47.5s** | 1.2× | 1.84s    |

¹ space\_ga /abalone 的 δ=0.1 为病态问题（最优解需 2100–2400 个非零列，稀疏 sieving 前提不成立），限时内未收敛属方法适用范围外；δ=0.5 全部快速收敛。

## 关键结论



1. **δ=0.5：10/10 全部快速收敛**（0.06–2.65s），解与历次完全一致（|x|₁、nnz 逐项相同）。

2. **δ=0.1：8/10 收敛**；本次性能优化把收敛数据集总耗时从约 429s 降到约 217s（约 2 倍），其中 pyrim/triazines 各提速 10–21 倍。

3. **解一致性验证**：pyrim（0.8603）、triazines（1.8652）、mpg（738.59）、E2006.train（1663.37）、log1p.train（1.4967）的 ‖x‖₁ 与优化前完全一致 —— 加速没有改变收敛点。

4. E2006.train δ=0.1 仍需 27 轮（B=20 二分 + N/S=7）—— 外层二分受参考 mucont 表约束（ratio\_upper>10 → 0.05），实验放宽到 0.10 无效（轮数不变），保持参考一致。

## δ 梯度测试（2026-09-23：δ = 0.5 / 0.1 / 0.01 / 0.001 ·‖b‖）

| 数据集 | δ=0.5 | δ=0.1 | δ=0.01 | δ=0.001 |
|---|---|---|---|---|
| pyrim | 0.10s ✓ | 0.36s ✓ | 4.5s 停滞¹ (7.3e-3) | 5.2s 停滞¹ (5.9e-2) |
| triazines | 0.72s ✓ | 15.2s ✓ | 74s 停滞¹ (1.3e-1) | — |
| bodyfat | 0.24s ✓ | 0.73s ✓ | 2.0s ✓ | 1.5s ✓ |
| mpg | 0.34s ✓ | 2.14s ✓ | 9.1s ✓ | 4.9s ✓ |
| space_ga | 0.16s ✓ | 320s 超时 | 383s 超时 (feas 7.9) | — |
| abalone | 2.65s ✓ | 307s 超时 | 558s 超时 (feas 19) | — |
| E2006.test | 0.06s ✓ | 0.14s ✓ | 316s 超时 (feas 3.1) | — |
| E2006.train | 0.27s ✓ | 147s ✓ | 301s 超时 (feas 8.6) | — |
| log1p.test | 1.00s ✓ | 1.89s ✓ | 600s 超时 (feas 13) | — |
| log1p.train | 1.84s ✓ | 47.5s ✓ | 超时（同类） | — |

¹ **停滞（status=3，新增检测）**：psi(μ) 在 μ→0 极限处冻结在 delta 之上——δ 低于该数据可达残差（约 6.5e-2·‖b‖，pyrim），外层 bisection 无根可找。原实现空转 200 轮（pyrim 6.6s / triazines 91s），现 45 / 41 轮即干净报告（4.5s / 74s），并给出"delta unreachable"诊断。

**规律**：δ 越小难度单调上升——δ=0.5 全过（≤2.65s）；δ=0.1 八过（两病态超时）；δ=0.01 仅低噪声数据（bodyfat/mpg）收敛，pyrim/triazines 停滞（δ 不可达），大活跃集数据集受时间预算限制（bisection 每轮子问题成本高）；δ=0.001 小数据集仍可收敛（bodyfat/mpg）。

## 求解器调度（2026-09-23 调度调整：ADMM 严格兜底）



| 子问题形态                  | 首选                   | 说明                                                |
| ---------------------- | -------------------- | ------------------------------------------------- |
| m > nr（瘦高）             | **smoothing Newton** | nr×nr 法方程，30s 预算上限，失败 / 超时 → 下一档                  |
| nr ≥ m 或大活跃集（含 m≥3000） | **SSNAL**            | 稠密 LDLT（m≤1500）/ PCG（m>1500），linsys 嵌套透传后每轮 8–13s |
| 以上均失败 / 被禁用            | **ADMM**             | 仅最后兜底，不再作为大活跃集首选                                  |

原 huge（large && m≥3000 → 直通 ADMM）分支已删除。效果：space\_ga δ=0.1 每轮子问题 170s→13s（SSN 接管，active set 稳定～2200–2400）；600s 预算下 feas 0.31→2.4e-2（bisection 收敛慢系解本身需 2385 非零列的病态问题，非调度缺陷）。

## 本轮代码优化（2026-09-23 性能轮）



| 优化                         | 内容                                                                                                                                     | 效果                                                                                                                       |
| -------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| **SSNAL linsys 嵌套透传（决定性）** | SubsetOperator 嵌套（SSN 自由集算子包 AS 缩减算子）时透传底层稀疏 / 稠密矩阵并做列号映射；`subset_dense_columns` 直接稀疏拷列构建稠密 C 块（O (nnz)），替代原单位向量 apply 全扫（O (sp²・nnz)） | triazines δ=0.1 345s 超时 → **15.2s 收敛**；pyrim 7.6s→0.36s；mpg 4.2s→2.1s；**这是本轮发现的最大性能缺陷**（7000 活跃列时每 Newton 方向约 59 亿次无效操作） |
| OpenMP 并行                  | SparseOperator 与 SubsetOperator 的 apply/applyT/columnNorm2/rowNorm2 按列并行（局部向量 + critical 归约保持语义），阈值 20 万 nnz 门控；编译 -fopenmp            | log1p 系列全维 viol / 乘 96.7M nnz 并行化；E2006.train 215s→147s                                                                  |
| SSN sigma 每轮自适应 ×1.5       | 保留历史优化（mpg δ=0.1 解不变，triazines 大活跃集收敛所需）                                                                                               | —                                                                                                                        |
| **CG buffer 复用（第二轮）** | cg_precond 的 `ap`、cg_smoothing/smoothing Newton 的 `atav` 移出迭代循环复用（原每 CG 迭代分配 n 向量） | 全量验证：解逐项一致；log1p.test -14%、log1p.train -7%、triazines -8%、E2006.test -20%；δ=0.1 收敛数据集总耗时 217s→207s |

## 全量验证（第二轮，δ=0.1/0.5，CG buffer 复用后）

| 数据集 | δ=0.1 优化前 | δ=0.1 本轮 | δ=0.5 优化前 | δ=0.5 本轮 |
|---|---|---|---|---|
| pyrim | 0.34s | 0.33s | 0.12s | 0.09s |
| triazines | 14.0s | **12.8s** | 0.74s | 0.77s |
| bodyfat | 0.73s | 0.77s | 0.31s | 0.23s |
| mpg | 2.08s | 2.10s | 0.41s | 0.39s |
| space_ga | 320s 超时 | 337s 超时 | 0.16s | 0.15s |
| abalone | 307s 超时 | 398s 超时 | 2.65s | 2.25s |
| E2006.test | 0.15s | **0.12s** | 0.07s | 0.06s |
| E2006.train | 150s | **145s** | 0.27s | 0.29s |
| log1p.test | 2.21s | **1.91s** | 1.03s | 0.95s |
| log1p.train | 47.8s | **44.5s** | 1.88s | 1.80s |

所有收敛解的 |x|₁ 与 nnz 与优化前逐项一致。δ=0.5 全部 10/10 收敛。

## 复现



```
\# 导出（全部 .mat → tests/ucidata/\*.bin）

python tests/export\_ucidata.py

\# WSL 编译 + 跑全部（-fopenmp 必需）

wsl -d Ubuntu -u root -- bash -lc "g++ -std=c++17 -O3 -march=native -fopenmp \\

&#x20; -I /mnt/c/Users/qianl/OneDrive/codes/SMOP/smop/include -I /usr/include/eigen3 \\

&#x20; /mnt/c/Users/qianl/OneDrive/codes/SMOP/smop/examples/example\_all\_datasets.cpp \\

&#x20; -o /root/smop-build/example\_all && cd /root/smop-build && ./example\_all"

\# 单数据集子集/时间上限

\# SMOP\_NAMES=mpg\_scale\_expanded7 SMOP\_TLIMIT=600 ./example\_all
```

原始日志：`tests\ucidata_logs\*.log`（本轮：`opt2_full.log`）。
---

## GPU 加速评估（可选功能，默认关闭）

`GpuSparseOperator`（`include/smop/gpu_operator.hpp` + `src/gpu_operator.cu`）把大稀疏矩阵的全矩阵 SpMV 放到 GPU（RTX 5060 / WSL 实测）。native 跨 TU 修复后与 CPU 版基本持平：

| 数据集 | nnz | CPU（OpenMP, native） | GPU 版（SMOP_GPU=1） |
| --- | ---: | ---: | ---: |
| E2006.train | 19.97M | 151.6s | 147.6s |
| log1p.E2006.train | 96.7M | 45.9s | 49.2s |
| triazines | 77.6M | 14.0s | 14.4s |
| pyrim | 8.05M | 0.34s | 0.46s |

解（‖x‖₁、nnz）与 CPU 逐项一致。**结论**：当前调度以 SSN 为主、子问题（列切片）在 CPU，GPU 全维 SpMV 只服务少量顶层调用——无显著收益；要让 GPU 真正提速需子问题整体 GPU 化。故 GPU 分支默认关闭（`SMOP_GPU=1` 启用，编译需 `-DSMOP_USE_CUDA` + nvcc 编译 `src/gpu_operator.cu`）。详见 `docs/GPU.md`。

### GPU 子问题加速（2026-09-23 第二轮：稠密 + SSN 子问题 GPU 化）

新增 `GpuDenseOperator`（cuBLAS gemv）、`subset_gemv`（列切片 kernel）与
`smop_gpu_xtx`（cuBLAS gemm 算 `G=CᵀC`），并接入 `ssn_l1.hpp`（reduced 分支
G 构建、PCG matvec）与 `linsolve.hpp`（normal_solve / gram_matrix）。验证
（解逐项与 CPU 一致）：

| 数据集 | CPU | GPU（SMOP_GPU=1） | 变化 |
| --- | ---: | ---: | --- |
| triazines δ=0.1（稀疏，reduced 分支） | 14.0s | **13.09s** | -7% |
| bodyfat δ=0.1（稠密） | 0.88s | **0.72s** | -18% |
| mpg δ=0.1（稠密小矩阵） | 2.08s | 2.18s | +5%（传输开销） |
| space_ga δ=0.5 | 0.11s | 0.11s | 持平（收敛） |
| abalone δ=0.5 | 7.57s | 7.6s | 持平（收敛） |
| space_ga δ=0.1 每轮 SSN | ~13s | **5.45s** | ~2.4× |
| abalone δ=0.1 每轮 SSN | ~19s | **14.2s** | ~1.3× |

- space_ga/abalone δ=0.1 整体仍超时（611s/613s，feas 0.13–0.35）——方法边界
  （解需 2000–3000+ 非零列），非 GPU 缺陷；δ=0.5 稀疏解场景全部收敛。
- 修复了 `gpu_operator.cu` 被误删的稀疏 kernel / 两个算子实现（整文件重写为
  完整版：CSR/CSC kernel + GpuSparseOperator + GpuDenseOperator + subset_gemv +
  smop_gpu_xtx + smop_gpu_available）。CPU 回归 T1–T5 全 PASS，CPU 版零 CUDA 依赖。