//======================================================================
// test_cpp.cpp -- smoke + correctness tests for the smop core
//
// Build (from the package root):
//   g++ -std=c++17 -O2 -I third_party/eigen -I include tests/test_cpp.cpp -o test_cpp
// Run: ./test_cpp
//
// Checks:
//   T1  BMOP on a small random problem: ||Ax-b|| ~= delta (feasibility),
//       support / sparsity of the returned solution,
//   T2  same problem with use_as = false (direct subproblem solvers),
//   T3  same problem with use_newton = true,
//   T4  sparse-matrix overload on a medium problem (sieving reduces the
//       subproblem dimension),
//   T5  single Lasso solve agrees with the KKT residual criterion.
//======================================================================
#include <cstdio>
#include <cmath>
#include <numeric>
#include <random>

#include "smop/smop.hpp"

using namespace smop;

static int g_fail = 0;

#define CHECK(cond, msg)                                                     \
    do {                                                                     \
        if (cond) { std::printf("  [PASS] %s\n", msg); }                     \
        else     { std::printf("  [FAIL] %s\n", msg); ++g_fail; }            \
    } while (0)

static Vec dense_sparse_solution(Index n, int k, std::mt19937& rng)
{
    std::normal_distribution<double> nd(0.0, 1.0);
    std::uniform_int_distribution<Index> ui(0, n - 1);
    Vec x = Vec::Zero(n);
    int added = 0;
    while (added < k) {
        Index j = ui(rng);
        if (x(j) == 0.0) { x(j) = nd(rng); ++added; }
    }
    return x;
}

static void run_problem(const Mat& A, const Vec& b, double delta,
                        const SmopOptions& opt, const char* tag,
                        double tol_feas = 2e-3)
{
    std::printf("== %s ==\n", tag);
    SmopResult r = solve_bmop(A, b, delta, opt);
    const double psi = (A * r.x - b).norm();
    const double feas = std::abs(psi - delta) / std::max(1.0, delta);
    std::printf("  iter=%d (B=%d N/S=%d)  mu=%.3e  psi=%.6e  delta=%.6e  feas=%.2e  "
                "nnz=%d  nnz999=%d  time=%.2fs  status=%d\n",
                r.info.iter, r.info.iter_bisection, r.info.iter_newton_or_secant,
                r.mu, psi, delta, feas, int((r.x.array().abs() > 0).count()),
                findnnz(r.x, 0.999), r.info.time, r.info.status);
    CHECK(r.info.status == 0 || r.info.status == 1, "solver returned (converged or maxiter)");
    CHECK(feas <= tol_feas, "||Ax-b|| matches delta within tolerance");
    CHECK(r.x.allFinite(), "solution is finite");
    if (opt.verbose) {
        std::printf("  reducedn(max=%d avg=%.1f)  mupath=%zu\n",
                    r.info.reducedn.empty() ? 0 : int(*std::max_element(r.info.reducedn.begin(), r.info.reducedn.end())),
                    r.info.reducedn.empty() ? 0.0 :
                        double(std::accumulate(r.info.reducedn.begin(), r.info.reducedn.end(), Index(0))) / r.info.reducedn.size(),
                    r.info.mupath.size());
    }
}

int main()
{
    std::printf("smop C++ test suite (version %s)\n\n", smop::version());
    std::mt19937 rng(2026);

    // ---- problem: n=120, m=40, true support 8 ----
    const Index n = 120, m = 40;
    Mat A = Mat::Zero(m, n);
    for (Index i = 0; i < m; ++i)
        for (Index j = 0; j < n; ++j) A(i, j) = std::normal_distribution<double>(0, 1)(rng);
    Vec xtrue = dense_sparse_solution(n, 8, rng);
    Vec b = A * xtrue + 0.02 * Vec::Random(m);
    const double delta = (A * xtrue - b).norm() * 1.02;

    SmopOptions opt;
    opt.verbose = 0;

    run_problem(A, b, delta, opt, "T1 BMOP default (secant + AS + smoothing)");

    SmopOptions opt2 = opt;
    opt2.use_as = false;
    run_problem(A, b, delta, opt2, "T2 BMOP without adaptive sieving");

    SmopOptions opt3 = opt;
    opt3.use_newton = true;   // Newton branch (use_secant is disabled inside)
    opt3.use_secant = false;
    run_problem(A, b, delta, opt3, "T3 BMOP with Newton steps", 5e-3);

    // ---- T4: sparse medium problem, check AS reduces the subproblems ----
    {
        const Index n2 = 2000, m2 = 150;
        SpMat Asp(m2, n2);
        {
            std::uniform_int_distribution<Index> ui(0, n2 - 1);
            std::vector<Eigen::Triplet<double>> trips;
            trips.reserve(static_cast<size_t>(m2 * 30));
            for (Index i = 0; i < m2; ++i)
                for (int t = 0; t < 30; ++t) {
                    Index j = ui(rng);
                    trips.emplace_back(i, j, std::normal_distribution<double>(0, 1)(rng));
                }
            Asp.setFromTriplets(trips.begin(), trips.end());
        }
        Vec x2 = dense_sparse_solution(n2, 12, rng);
        Vec b2 = Asp * x2 + 0.01 * Vec::Random(m2);
        const double delta2 = (Asp * x2 - b2).norm() * 1.02;
        SmopOptions o4;
        o4.verbose = 0;
        o4.stoptol = 1e-5;
        SmopResult r4 = solve_bmop(Asp, b2, delta2, o4);
        const double psi4 = (Asp * r4.x - b2).norm();
        const double feas4 = std::abs(psi4 - delta2) / std::max(1.0, delta2);
        std::printf("== T4 sparse BMOP ==\n");
        std::printf("  iter=%d  feas=%.2e  nnz999=%d  time=%.2fs\n",
                    r4.info.iter, feas4, findnnz(r4.x, 0.999), r4.info.time);
        if (!r4.info.reducedn.empty()) {
            Index mx = *std::max_element(r4.info.reducedn.begin(), r4.info.reducedn.end());
            std::printf("  AS reduced dims: max=%d (n=%d)\n", int(mx), int(n2));
            CHECK(mx < n2, "adaptive sieving kept subproblems below n");
        }
        CHECK(feas4 <= 1e-3, "sparse BMOP feasibility");
        CHECK(r4.info.iter <= 60, "sparse BMOP converged in reasonable iterations");
    }

    // ---- T5: single Lasso ----
    {
        const Index nl = 100, ml = 30;
        Mat Al = Mat::Zero(ml, nl);
        for (Index i = 0; i < ml; ++i)
            for (Index j = 0; j < nl; ++j) Al(i, j) = std::normal_distribution<double>(0, 1)(rng);
        Vec xl = dense_sparse_solution(nl, 6, rng);
        Vec bl = Al * xl + 0.01 * Vec::Random(ml);
        const double lambda = 0.05;
        LassoOptions lo;
        lo.stoptol = 1e-7;
        LassoResult lr = solve_lasso(Al, bl, lambda, lo);
        Vec gl = Al.transpose() * (Al * lr.x - bl);
        double eta = kkt_residual(lr.x, gl, lambda);
        std::printf("== T5 single Lasso ==\n");
        std::printf("  iter=%d  obj=%.6e  eta=%.2e  nnz=%d\n",
                    lr.info.iter, lr.info.obj, eta, int((lr.x.array().abs()>1e-8).count()));
        CHECK(eta < 1e-5, "lasso KKT residual small");
        CHECK(lr.info.obj > 0, "lasso objective positive");
    }

    std::printf("\n%s: %d failure(s)\n", g_fail ? "FAILED" : "ALL TESTS PASSED", g_fail);
    return g_fail ? 1 : 0;
}
