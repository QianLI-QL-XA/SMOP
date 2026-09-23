# GPU 加速支持（可选）

smop 提供可选的 CUDA 加速路径：`GpuSparseOperator` / `GpuDenseOperator`
（`include/smop/gpu_operator.hpp` + `src/gpu_operator.cu`），覆盖三类矩阵计算：

- **稀疏全维 SpMV**（`A x` / `Aᵀ y`，手写 CSR/CSC kernel）——level-set 外层 sieve；
- **稠密全维 gemv**（cuBLAS `cublasDgemv`）——稠密 UCI 数据的顶层算子；
- **SSN/smoothing 子问题**（本轮新增）：
  - `GpuDenseOperator::subset_gemv`：列切片乘积（PCG matvec 的 `App`/`Appᵀ`），
    手写 kernel 按列索引取 GPU 稠密矩阵的列；
  - `smop_gpu_xtx`：reduced 正规矩阵 `G = CᵀC`（cuBLAS `cublasDgemm`）。

`ssn_l1.hpp`（reduced 分支的 `G` 构建与 PCG matvec）与 `linsolve.hpp`
（`normal_solve` / `gram_matrix`）在 `SMOP_USE_CUDA` 下优先走 GPU，失败自动回退 CPU。

## 评估结论（RTX 5060 / WSL Ubuntu 实测，2026-09）

### δ=0.1（稀疏解）——GPU 化子问题后

| 数据集 | 形态 | CPU（OpenMP, native） | GPU（SMOP_GPU=1） | 解一致 |
| --- | --- | ---: | ---: | --- |
| triazines | 稀疏 77.6M nnz | 14.0 s | 13.09 s | 是（1.8652） |
| bodyfat | 稠密 252×116280 | 0.88 s | 0.72 s | 是（1.6786） |
| mpg | 稠密 392×3432 | 2.08 s | 2.18 s | 是（738.5933） |

- 小矩阵（mpg）GPU 开销略大于收益，属正常；中大矩阵 reduced 分支（triazines）
  的 `G=CᵀC` 走 GPU gemm 有稳定 ~7% 收益。
- space_ga / abalone δ=0.1：GPU 前几轮每轮 SSN 从 CPU ~13–19 s 降至 5.5–14 s
  （约 2.4×），但整体仍超时（611 s / 613 s，feas 0.13–0.35 未收敛）——这是方法
  边界而非 GPU 问题：这两个数据集的 δ=0.1 解需要 2000–3000+ 非零列，活跃集爆炸。

### δ=0.5（稀疏解场景，用户指定口径）

| 数据集 | CPU | GPU | 解一致 |
| --- | ---: | ---: | --- |
| bodyfat | 1.07 s | 0.16 s | 是 |
| mpg | 2.08 s | 0.47 s | 是 |
| space_ga | 0.11 s | 0.11 s | 是（收敛） |
| abalone | 7.57 s | 7.6 s | 是（收敛） |

稠密全维 gemv 在 δ=0.5（解较稀疏、活跃集小、顶层调用多）收益显著；
δ=0.1 仍由 SSN 子问题主导。

## 编译

```bash
# 1) 编译 CUDA 目标（host 代码必须与调用方同指令集，否则跨 TU 内存错误）
nvcc -std=c++17 -O3 -arch=sm_120 -Xcompiler -march=native \
     -I <smop>/include -I /usr/include/eigen3 \
     -I /usr/local/cuda-13.3/targets/x86_64-linux/include \
     -c <smop>/src/gpu_operator.cu -o gpu_operator.o

# 2) 编译调用方并链接（同样 -march=native，需 SMOP_USE_CUDA 宏）
g++ -std=c++17 -O3 -march=native -fopenmp -DSMOP_USE_CUDA \
    -I <smop>/include -I /usr/include/eigen3 \
    example.cpp gpu_operator.o \
    -L/usr/local/cuda-13.3/targets/x86_64-linux/lib -lcudart -lcublas \
    -Xlinker -rpath -Xlinker /usr/local/cuda-13.3/lib64 \
    -Xlinker -rpath -Xlinker /usr/local/cuda-13.3/targets/x86_64-linux/lib \
    -o example
```

cuBLAS 头文件/库在 `/usr/local/cuda-13.3/targets/x86_64-linux/{include,lib}`
（`apt install libcublas-13-3 libcublas-dev-13-3`）。CPU 版（不带 `SMOP_USE_CUDA`）
零 CUDA 依赖，行为完全不变。

## 运行时

```bash
SMOP_GPU=1 SMOP_NAMES=... ./example
```

`example_all_datasets` 仅在 `SMOP_USE_CUDA` 编译且 `SMOP_GPU=1` 时启用 GPU 分支
（稀疏 nnz≥1,000,000 或稠密 size≥1,000,000）。无 CUDA 设备时两个 GPU 算子自动
回退到 CPU（内部持有 Eigen 副本），`smop_gpu_xtx` / `subset_gemv` 返回 false 由
调用点回退 CPU。

## 实现要点（踩坑记录）

1. **跨 TU 编译**：GPU 算子不继承对应 CPU 算子。Eigen `SparseMatrix` 在 nvcc TU
   内构造、g++ TU 内析构时，`-O3` 下会因分配器/对齐不一致产生 `free(): invalid
   pointer`；改用 `LinearOperator` 基类 + 内部 `std::unique_ptr`（Eigen 对象生命
   周期完全在 .cu 内），并通过虚函数 `sparse_matrix()` / `dense_matrix()` 把 CPU
   矩阵暴露给 `SubsetOperator`（跨 TU 不用 `dynamic_cast`）。
2. **Eigen 稀疏转置在 nvcc 下被错误编译**：`At = A.transpose()` 的 Sparse2Sparse
   赋值在 host 执行产生错误 CSR 索引；改为手动 counting-sort CSC→CSR 转换。
3. **`-march=native` 必须 nvcc（`-Xcompiler -march=native`）与 g++ 两侧同时使用**，
   否则跨 TU 对齐假设不一致（O3 下崩溃、O1/ASAN 下正常——这是关键线索）。
4. **cuBLAS 参数**：`cublasDgemv` / `cublasDgemm` 的 alpha/beta 必须传指针
   （`const double alpha=1.0, beta=0.0;` 取地址）。
5. **显存**：稀疏算子 CSR + CSC 两份设备副本 ≈ 2 × nnz × 16 B（log1p 96.7M nnz ≈
   3.1 GB）；稠密算子整矩阵一份（space_ga 3107×5005 ≈ 124 MB）。
6. **每轮 subset_gemv 的列列表上传**：PCG 内每次 matvec 上传 `cols`（int×sp）与
   输入向量，sp≤10000 时开销可忽略；这是换取"GPU 只算子集列"的取舍——CPU 基线
   的列切片路径（SubsetOperator）已足够快，GPU 版主要服务稠密大活跃集。
