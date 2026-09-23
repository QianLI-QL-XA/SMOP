//======================================================================
// example_cpp.cpp -- minimal C++ usage of the smop core
// Build (from the package root):
//   g++ -std=c++17 -O2 -I third_party/eigen-3.4.0 -I include examples/example_cpp.cpp -o example_cpp
//======================================================================
#include <cstdio>
#include <random>

#include "smop/smop.hpp"

int main()
{
    const smop::Index m = 40, n = 150;
    std::mt19937 rng(0);
    std::normal_distribution<double> nd(0.0, 1.0);

    // random problem with a sparse true solution
    smop::Mat A(m, n);
    for (smop::Index i = 0; i < m; ++i)
        for (smop::Index j = 0; j < n; ++j) A(i, j) = nd(rng);

    smop::Vec xtrue = smop::Vec::Zero(n);
    for (int k = 0; k < 8; ++k) xtrue(rng() % n) = nd(rng);
    smop::Vec b = A * xtrue + 0.02 * smop::Vec::Random(m);
    const double delta = 1.02 * (A * xtrue - b).norm();

    // solve
    smop::SmopResult r = smop::solve_bmop(A, b, delta);
    const double psi = (A * r.x - b).norm();

    std::printf("smop version %s\n", smop::version());
    std::printf("iter=%d (B=%d S/N=%d)  mu=%.3e  ||Ax-b||=%.6e  delta=%.6e\n",
                r.info.iter, r.info.iter_bisection, r.info.iter_newton_or_secant,
                r.mu, psi, delta);
    std::printf("status=%d  msg=%s  time=%.2fs\n", r.info.status,
                r.info.msg.c_str(), r.info.time);
    std::printf("sparsity: ||x||_0=%d (true=%d)\n",
                int((r.x.array().abs() > 1e-8).count()), int((xtrue.array().abs() > 0).count()));
    return 0;
}
