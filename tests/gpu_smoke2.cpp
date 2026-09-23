#include <cstdio>
#include <fstream>
#include <vector>
#include "smop/smop.hpp"
#include "smop/gpu_operator.hpp"

static bool load_bin(const std::string& path, smop::SpMat& A, smop::Vec& b)
{
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    int m = 0, n = 0, flag = 0; long long nnz = 0;
    f.read(reinterpret_cast<char*>(&m), 4);
    f.read(reinterpret_cast<char*>(&n), 4);
    f.read(reinterpret_cast<char*>(&flag), 4);
    f.read(reinterpret_cast<char*>(&nnz), 8);
    if (m <= 0 || n <= 0 || flag != 1) return false;
    std::vector<int> cp(static_cast<size_t>(n) + 1);
    std::vector<int> ri(static_cast<size_t>(nnz));
    std::vector<double> va(static_cast<size_t>(nnz));
    std::vector<double> bv(static_cast<size_t>(m));
    f.read(reinterpret_cast<char*>(cp.data()), sizeof(int) * cp.size());
    f.read(reinterpret_cast<char*>(ri.data()), sizeof(int) * ri.size());
    f.read(reinterpret_cast<char*>(va.data()), sizeof(double) * va.size());
    f.read(reinterpret_cast<char*>(bv.data()), sizeof(double) * bv.size());
    if (!f) return false;
    A.resize(m, n);
    Eigen::VectorXi rsv(n);
    for (int j = 0; j < n; ++j) rsv(j) = cp[static_cast<size_t>(j) + 1] - cp[static_cast<size_t>(j)];
    A.reserve(rsv);
    for (int j = 0; j < n; ++j)
        for (int k = cp[j]; k < cp[j + 1]; ++k)
            A.insert(ri[static_cast<size_t>(k)], j) = va[static_cast<size_t>(k)];
    A.makeCompressed();
    b = Eigen::Map<smop::Vec>(bv.data(), m);
    return true;
}

int main()
{
    smop::SpMat A;
    smop::Vec b;
    if (!load_bin("/mnt/c/Users/qianl/OneDrive/codes/SMOP/smop/tests/ucidata/pyrim_scaled_expanded5.bin", A, b)) {
        printf("load failed\n"); return 1;
    }
    printf("loaded m=%lld n=%lld nnz=%lld\n", (long long)A.rows(),
           (long long)A.cols(), (long long)A.nonZeros());
    smop::GpuSparseOperator gA(A);
    printf("gpu_active=%d\n", (int)gA.gpu_active());
    smop::Vec x = smop::Vec::Random(A.cols());
    double acc = 0;
    for (int it = 0; it < 200; ++it) {
        smop::Vec y = gA.apply(x);
        smop::Vec z = gA.applyT(y);
        acc += z(0);
        if (it % 50 == 0) printf("it=%d ok\n", it);
    }
    printf("done acc=%.3f\n", acc);
    return 0;
}
