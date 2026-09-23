#include <cstdio>
#include "smop/smop.hpp"
#include "smop/gpu_operator.hpp"

int main()
{
    smop::SpMat A(4, 4);
    A.insert(0, 0) = 1.0; A.insert(1, 1) = 2.0; A.insert(2, 2) = 3.0;
    A.insert(3, 3) = 4.0; A.insert(0, 2) = 0.5; A.insert(3, 1) = -1.0;
    A.makeCompressed();
    printf("gpu_available=%d\n", (int)smop::smop_gpu_available());

    {
        smop::GpuSparseOperator gA(A);
        printf("gpu_active=%d rows=%lld cols=%lld\n", (int)gA.gpu_active(),
               (long long)gA.rows(), (long long)gA.cols());
        smop::Vec x(4); x << 1, 1, 1, 1;
        smop::Vec y = gA.apply(x);
        smop::Vec z = gA.applyT(y);
        printf("y=[%g %g %g %g]\n", y(0), y(1), y(2), y(3));
        printf("z=[%g %g %g %g]\n", z(0), z(1), z(2), z(3));
    }
    printf("destructor ok\n");

    // a second, larger matrix (column-heavy like pyrim)
    {
        const int m = 74, n = 5000;
        smop::SpMat B(m, n);
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < 40; ++i)
                B.insert((i + j) % m, j) = 1.0;
        B.makeCompressed();
        smop::GpuSparseOperator gB(B);
        smop::Vec x = smop::Vec::Random(n);
        smop::Vec y = gB.apply(x);
        printf("B gpu_active=%d y.norm=%.6f\n", (int)gB.gpu_active(), y.norm());
    }
    printf("all ok\n");
    return 0;
}
