//======================================================================
// example_compare.cpp -- outer-scheme comparison on UCI datasets
//
// For each dataset and delta scale, solves the same problem with three
// outer root-finding schemes and prints one CSV line per (dataset, mode):
//   mode 0 = bisection only            (use_secant=false, use_newton=false)
//   mode 1 = secant + adaptive sieving (default: our algorithm)
//   mode 2 = bisection + Newton        (use_secant=false, use_newton=true)
//
// Env:
//   SMOP_NAMES  = comma list  dataset names (default: all 10)
//   SMOP_SCALE  = delta scale (default 0.4)
//   SMOP_TLIMIT = per (dataset,mode) time budget (default 600)
//
// Usage:
//   g++ -std=c++17 -O3 -march=native -fopenmp \
//       -I include -I /usr/include/eigen3 examples/example_compare.cpp -o compare
//   SMOP_NAMES=pyrim_scaled_expanded5,triazines_scaled_expanded4 SMOP_SCALE=0.4 ./compare
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
    Mat  Ad;
    SpMat As;
    Vec  b;
    int  m = 0, n = 0;
    bool sparse = false;
    double normb = 0.0;
};

static bool load_bin(const std::string& path, Data& d)
{
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    int m = 0, n = 0, flag = 0; long long nnz = 0;
    f.read(reinterpret_cast<char*>(&m), 4);
    f.read(reinterpret_cast<char*>(&n), 4);
    f.read(reinterpret_cast<char*>(&flag), 4);
    f.read(reinterpret_cast<char*>(&nnz), 8);
    if (m <= 0 || n <= 0 || (flag != 0 && flag != 1)) return false;
    d.m = m; d.n = n; d.sparse = (flag == 1);
    if (flag == 0) {
        std::vector<double> buf(static_cast<size_t>(m) * n);
        f.read(reinterpret_cast<char*>(buf.data()), sizeof(double) * buf.size());
        std::vector<double> bv(static_cast<size_t>(m));
        f.read(reinterpret_cast<char*>(bv.data()), sizeof(double) * bv.size());
        if (!f) return false;
        d.Ad = Eigen::Map<Mat>(buf.data(), m, n);
        d.b  = Eigen::Map<Vec>(bv.data(), m);
    } else {
        if (nnz < 0 || nnz > 400000000LL) return false;
        std::vector<int> cp(static_cast<size_t>(n) + 1);
        std::vector<int> ri(static_cast<size_t>(nnz));
        std::vector<double> va(static_cast<size_t>(nnz));
        std::vector<double> bv(static_cast<size_t>(m));
        f.read(reinterpret_cast<char*>(cp.data()), sizeof(int) * cp.size());
        f.read(reinterpret_cast<char*>(ri.data()), sizeof(int) * ri.size());
        f.read(reinterpret_cast<char*>(va.data()), sizeof(double) * va.size());
        f.read(reinterpret_cast<char*>(bv.data()), sizeof(double) * bv.size());
        if (!f) return false;
        SpMat As(m, n);
        Eigen::VectorXi rsv(n);
        for (int j = 0; j < n; ++j) rsv(j) = cp[static_cast<size_t>(j) + 1] - cp[static_cast<size_t>(j)];
        As.reserve(rsv);
        for (int j = 0; j < n; ++j)
            for (int k = cp[j]; k < cp[j + 1]; ++k)
                As.insert(ri[static_cast<size_t>(k)], j) = va[static_cast<size_t>(k)];
        As.makeCompressed();
        d.As = std::move(As);
        d.b  = Eigen::Map<Vec>(bv.data(), m);
    }
    d.normb = d.b.norm();
    return true;
}

static std::vector<std::string> split(const char* s)
{
    std::vector<std::string> out;
    if (!s) return out;
    std::string v(s);
    for (size_t i = 0, j = 0; i <= v.size(); ++i) {
        if (i == v.size() || v[i] == ',') {
            std::string t = v.substr(j, i - j);
            if (!t.empty()) out.push_back(t);
            j = i + 1;
        }
    }
    return out;
}

int main(int argc, char** argv)
{
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    const std::string dir = std::string(argv[0]);
    const std::string data_dir =
        dir.substr(0, dir.find_last_of("\\/") + 1) + "../tests/ucidata/";

    const double scale = std::getenv("SMOP_SCALE") ? std::atof(std::getenv("SMOP_SCALE")) : 0.4;
    const double tlimit = std::getenv("SMOP_TLIMIT") ? std::atof(std::getenv("SMOP_TLIMIT")) : 600.0;

    std::vector<std::string> names;
    if (argc > 1) names.assign(argv + 1, argv + argc);
    else if (const char* s = std::getenv("SMOP_NAMES")) names = split(s);
    else names = {
        "pyrim_scaled_expanded5", "triazines_scaled_expanded4",
        "bodyfat_scale_expanded7", "mpg_scale_expanded7",
        "space_ga_scale_expanded9", "abalone_scale_expanded7",
        "E2006.test", "E2006.train", "log1p.E2006.test", "log1p.E2006.train"};

    std::printf("dataset, mode, delta, time_s, iter, iterB, iterNS, feas, x1, nnz, status\n");

    for (const auto& nm : names) {
        Data d;
        const std::string p = data_dir + nm + ".bin";
        if (!load_bin(p, d)) { std::printf("%s, ERR\n", nm.c_str()); continue; }
        const double delta = scale * d.normb;

        for (int mode = 0; mode < 3; ++mode) {
            SmopOptions o;
            o.stoptol = 1e-6;
            o.time_limit = tlimit;
            o.verbose = 0;
            o.use_secant = (mode == 1);
            o.use_newton = (mode == 2);
            double t0 = wall_time_seconds();
            SmopResult r = d.sparse ? solve_bmop(d.As, d.b, delta, o)
                                    : solve_bmop(d.Ad, d.b, delta, o);
            const double psi = d.sparse ? (d.As * r.x - d.b).norm()
                                        : (d.Ad * r.x - d.b).norm();
            const double feas = std::abs(psi - delta) / std::max(1.0, delta);
            const double obj = r.x.cwiseAbs().sum();
            const int nnz = int((r.x.array().abs() > 1e-8).count());
            std::printf("%s, %d, %.3f, %.3f, %d, %d, %d, %.2e, %.4f, %d, %d\n",
                        nm.c_str(), mode, delta, wall_time_seconds() - t0,
                        r.info.iter, r.info.iter_bisection,
                        r.info.iter_newton_or_secant, feas, obj, nnz,
                        r.info.status);
        }
    }
    return 0;
}
