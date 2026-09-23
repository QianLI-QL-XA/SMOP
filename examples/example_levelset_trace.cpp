//======================================================================
// example_levelset_trace.cpp -- watch the outer level-set iterations
//
// Build & run:
//     powershell -File build_levelset.ps1
// or in VS Code:  Ctrl+Shift+P -> Tasks: Run Build Task -> "levelset trace"
//
// Prints, for every outer iteration of the BMOP
//     min ||x||_1  s.t.  ||A x - b||_2 <= delta
// the chosen mu, the residual psi(mu) = ||A x(mu) - b||, the relative
// feasibility error eta, the current bracket [mu0, muinf] and the step
// kind (Bisection / Secant / Newton), then a summary row per problem
// and a strategy comparison table (secant vs Newton vs bisection).
//======================================================================
#include <cstdio>
#include <cmath>
#include <random>
#include <algorithm>
#include <numeric>

#include "smop/smop.hpp"

using namespace smop;

//----------------------------------------------------------------------
// random test problem: dense A, k-sparse ground truth, noisy b
//----------------------------------------------------------------------
static void make_problem(Mat& A, Vec& xtrue, Vec& b, int n, int m, int k,
                         double noise, std::mt19937& rng)
{
    std::normal_distribution<double> nd(0.0, 1.0);
    std::uniform_int_distribution<int> ui(0, n - 1);
    A = Mat::Zero(m, n);
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j) A(i, j) = nd(rng);
    xtrue = Vec::Zero(n);
    int added = 0;
    while (added < k) {
        int j = ui(rng);
        if (xtrue(j) == 0.0) { xtrue(j) = nd(rng); ++added; }
    }
    b = A * xtrue;
    for (int i = 0; i < m; ++i) b(i) += noise * nd(rng);
}

//----------------------------------------------------------------------
// one solve + a printed outer-iteration trace (when print_trace)
//----------------------------------------------------------------------
static void run_case(const char* name, int n, int m, int k, double noise,
                     double c, const SmopOptions& base, bool print_trace,
                     std::mt19937& rng)
{
    Mat A; Vec xtrue, b;
    make_problem(A, xtrue, b, n, m, k, noise, rng);
    const double delta = c * b.norm();

    SmopOptions opt = base;
    opt.verbose = print_trace ? 1 : 0;

    std::printf("== %s (n=%d m=%d delta=%.4f) ==\n", name, n, m, delta);
    SmopResult r = solve_bmop(A, b, delta, opt);

    const double psi = (A * r.x - b).norm();
    const double feas = std::abs(psi - delta) / std::max(1.0, delta);
    const double obj = r.x.cwiseAbs().sum();
    const double obj_true = xtrue.cwiseAbs().sum();
    const int nnz999 = findnnz(r.x, 0.999);
    const int k_true = int((xtrue.array().abs() > 1e-10).count());
    const int max_as = r.info.reducedn.empty()
                           ? 0
                           : int(*std::max_element(r.info.reducedn.begin(),
                                                   r.info.reducedn.end()));
    std::printf("  -> iter=%d (B=%d N/S=%d)  feas=%.2e  |x|1=%.4f (true %.4f)"
                "  energy999=%d (k*=%d)  maxASred=%d  time=%.2fs  status=%d %s\n\n",
                r.info.iter, r.info.iter_bisection, r.info.iter_newton_or_secant,
                feas, obj, obj_true, nnz999, k_true, max_as, r.info.time,
                r.info.status, r.info.msg.c_str());
}

int main()
{
    std::printf("smop level-set trace (version %s)\n\n", smop::version());
    std::mt19937 rng(2026);

    SmopOptions base;
    base.stoptol = 1e-6;
    base.time_limit = 600.0;

    // ---- scale x radius matrix (secant + AS + smoothing) ----
    run_case("S1 small  c=0.1", 150, 40,  8, 0.02, 0.1,  base, true,  rng);
    run_case("S2 small  c=0.01",150, 40,  8, 0.02, 0.01, base, true,  rng);
    run_case("S3 small  c=0.3", 150, 40,  8, 0.02, 0.3,  base, false, rng);
    run_case("S4 mid    c=0.1", 400, 80, 15, 0.02, 0.1,  base, false, rng);

    // ---- outer strategy comparison on the same problem ----
    Mat A; Vec xtrue, b;
    make_problem(A, xtrue, b, 150, 40, 8, 0.02, rng);
    const double delta = 0.1 * b.norm();
    std::printf("== strategy comparison (same n=150 m=40 problem, delta=%.4f) ==\n",
                delta);
    std::printf("  %-14s %6s %4s %5s %12s %8s\n",
                "strategy", "iter", "B", "N/S", "feas", "time");
    const char* tags[3] = { "secant+AS", "Newton+AS", "bisection" };
    SmopOptions opts[3] = { base, base, base };
    opts[1].use_secant = false; opts[1].use_newton = true;
    opts[2].use_secant = false; opts[2].use_newton = false;
    for (int i = 0; i < 3; ++i) {
        SmopResult r = solve_bmop(A, b, delta, opts[i]);
        const double psi = (A * r.x - b).norm();
        const double feas = std::abs(psi - delta) / std::max(1.0, delta);
        std::printf("  %-14s %6d %4d %5d %12.2e %8.2f\n",
                    tags[i], r.info.iter, r.info.iter_bisection,
                    r.info.iter_newton_or_secant, feas, r.info.time);
    }
    return 0;
}
