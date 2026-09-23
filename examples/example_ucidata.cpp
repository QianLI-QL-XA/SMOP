//======================================================================
// example_ucidata.cpp -- run the BMOP level-set solver on real UCI data
//
// Data files are produced from the .mat sources by
//     python tests/export_ucidata.py
// into tests/ucidata/<name>.bin (raw doubles: m, n, A row-major, b).
//
// Build & run:
//     powershell -File build_ucidata.ps1
// or in VS Code: Tasks: Run Build Task -> "smop: build & run ucidata"
//
// For every dataset and noise radius c (delta = c * ||b||_2) it prints
// the outer level-set iterations and a summary row
//   feas, |x|_1, nnz (support size), time.
//======================================================================
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include "smop/smop.hpp"

using namespace smop;

struct Data {
    Mat A;
    Vec b;
    int m = 0, n = 0;
    double normb = 0.0;
};

static bool load_bin(const std::string& path, Data& d)
{
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    int m = 0, n = 0;
    f.read(reinterpret_cast<char*>(&m), 4);
    f.read(reinterpret_cast<char*>(&n), 4);
    if (m <= 0 || n <= 0 || (long long)m * n > 500000000LL) {
        std::printf("  [skip] %s: bad dims m=%d n=%d (too large for dense)\n",
                    path.c_str(), m, n);
        return false;
    }
    std::vector<double> buf(static_cast<size_t>(m) * n);
    f.read(reinterpret_cast<char*>(buf.data()), sizeof(double) * buf.size());
    std::vector<double> bv(static_cast<size_t>(m));
    f.read(reinterpret_cast<char*>(bv.data()), sizeof(double) * bv.size());
    if (!f) return false;
    d.m = m; d.n = n;
    d.A = Eigen::Map<Mat>(buf.data(), m, n);
    d.b = Eigen::Map<Vec>(bv.data(), m);
    d.normb = d.b.norm();
    return true;
}

int main(int argc, char** argv)
{
    std::setvbuf(stdout, nullptr, _IONBF, 0);   // unbuffered: progress visible in logs
    std::printf("smop real-data BMOP test (version %s)\n\n", smop::version());
    const std::string dir = std::string(argv[0]);
    const std::string base =
        dir.substr(0, dir.find_last_of("\\/") + 1) + "../tests/ucidata/";

    const std::vector<std::string> names = (argc > 1)
        ? std::vector<std::string>(argv + 1, argv + argc)
        : std::vector<std::string>{
              "mpg_scale_expanded7", "pyrim_scaled_expanded5",
              "triazines_scaled_expanded4", "space_ga_scale_expanded9",
              "E2006.train"};
    // noise-radius scales; override with env SMOP_SCALES = "0.1,0.5"
    std::vector<double> scales = { 0.1, 0.5 };
    if (const char* s = std::getenv("SMOP_SCALES")) {
        scales.clear();
        std::string v(s);
        for (size_t i = 0, j = 0; i <= v.size(); ++i) {
            if (i == v.size() || v[i] == ',') {
                scales.push_back(std::stod(v.substr(j, i - j)));
                j = i + 1;
            }
        }
    }
    const bool verbose = std::getenv("SMOP_VERBOSE") != nullptr;

    SmopOptions opt;
    opt.stoptol = 1e-6;
    opt.time_limit = 1800.0;

    for (const auto& nm : names) {
        Data d;
        if (!load_bin(base + nm + ".bin", d)) {
            std::printf("== %s ==  (not exported yet: run "
                        "python tests/export_ucidata.py %s.mat)\n\n",
                        nm.c_str(), nm.c_str());
            std::fflush(stdout);
            continue;
        }
        std::printf("== %s ==  (m=%d n=%d  ||b||=%.4f)\n", nm.c_str(),
                    d.m, d.n, d.normb);
        std::fflush(stdout);
        for (double c : scales) {
            const double delta = c * d.normb;
            SmopOptions o = opt;
            o.verbose = verbose ? 2 : 0;   // 2: also show the AS inner rounds
            std::printf("  -- delta = %.3f * ||b|| = %.4f --\n", c, delta);
            std::fflush(stdout);
            SmopResult r = solve_bmop(d.A, d.b, delta, o);
            const double psi = (d.A * r.x - d.b).norm();
            const double feas = std::abs(psi - delta) / std::max(1.0, delta);
            const double obj = r.x.cwiseAbs().sum();
            const int nnz = int((r.x.array().abs() > 1e-8).count());
            const int max_as = r.info.reducedn.empty()
                                   ? 0
                                   : int(*std::max_element(
                                         r.info.reducedn.begin(),
                                         r.info.reducedn.end()));
            std::printf("  -> iter=%d (B=%d N/S=%d)  feas=%.2e  |x|1=%.4f  "
                        "nnz(1e-8)=%d  maxASred=%d  time=%.2fs  status=%d %s\n\n",
                        r.info.iter, r.info.iter_bisection,
                        r.info.iter_newton_or_secant, feas, obj, nnz, max_as,
                        r.info.time, r.info.status, r.info.msg.c_str());
            std::fflush(stdout);
        }
    }
    return 0;
}
