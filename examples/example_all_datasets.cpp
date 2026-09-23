//======================================================================
// example_all_datasets.cpp -- batch BMOP level-set test on every UCI
// dataset exported by tests/export_ucidata.py
//
// Binary format (new unified):
//   <int32 m> <int32 n> <int32 sparse_flag> <int64 nnz>
//   sparse_flag == 0 : A dense row-major [m*n], b [m]
//   sparse_flag == 1 : CSC colptr[n+1] i4, rowidx[nnz] i4, vals[nnz] f8, b [m]
//
// Env:
//   SMOP_SCALES = "0.1,0.5"   noise-radius scales (default)
//   SMOP_TLIMIT = seconds     per-dataset time budget (default 1800)
//   SMOP_NAMES  = comma list  subset of dataset names (default: all)
//   SMOP_VERBOSE=1            show AS inner rounds (verbose=2)
//======================================================================
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include "smop/smop.hpp"
#include "smop/gpu_operator.hpp"   // optional CUDA path (nvcc build only)

using namespace smop;

struct Data {
    Mat  Ad;       // dense matrix (sparse==false)
    SpMat As;      // sparse matrix (sparse==true)
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
        Eigen::VectorXi rsv(n);          // per-column nnz (not cumulative!)
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
    std::printf("smop real-data BMOP batch test (version %s)\n\n", smop::version());

    const std::string dir = std::string(argv[0]);
    const std::string data_dir =
        dir.substr(0, dir.find_last_of("\\/") + 1) + "../tests/ucidata/";

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
    const double tlimit = std::getenv("SMOP_TLIMIT")
        ? std::atof(std::getenv("SMOP_TLIMIT")) : 1800.0;
    const bool verbose = std::getenv("SMOP_VERBOSE") != nullptr;

    std::vector<std::string> names;
    if (argc > 1) names.assign(argv + 1, argv + argc);
    else if (const char* s = std::getenv("SMOP_NAMES")) names = split(s);
    else names = {
        "pyrim_scaled_expanded5", "triazines_scaled_expanded4",
        "bodyfat_scale_expanded7", "mpg_scale_expanded7",
        "space_ga_scale_expanded9", "abalone_scale_expanded7",
        "E2006.test", "E2006.train", "log1p.E2006.test", "log1p.E2006.train"};

    std::printf("data dir : %s\n", data_dir.c_str());
    std::printf("scales   :");
    for (double c : scales) std::printf(" %.3f", c);
    std::printf("\ntime budget per dataset: %.0fs\n\n", tlimit);

    int nfail = 0;
    for (const auto& nm : names) {
        Data d;
        const std::string p = data_dir + nm + ".bin";
        if (!load_bin(p, d)) {
            std::printf("== %s ==  (missing or unreadable bin)\n\n", nm.c_str());
            nfail++;
            continue;
        }
        std::printf("== %s ==  (m=%d n=%d %s nnz=%lld ||b||=%.4f)\n",
                    nm.c_str(), d.m, d.n, d.sparse ? "sparse" : "dense",
                    d.sparse ? (long long)d.As.nonZeros() : 0LL, d.normb);
        std::fflush(stdout);

        for (double c : scales) {
            const double delta = c * d.normb;
            SmopOptions o;
            o.stoptol = 1e-6;
            o.time_limit = tlimit;
            o.verbose = verbose ? 2 : 0;
            std::printf("  -- delta = %.3f * ||b|| = %.4f --\n", c, delta);
            std::fflush(stdout);
            double t0 = wall_time_seconds();
            // large sparse problems: optional GPU SpMV.  Off by default:
            // the solver spends most time in CPU subproblems (SSN/ADMM on
            // column subsets) where the GPU has no win, and the TU-safe
            // host build would slow the CPU path.  Compile with
            // -DSMOP_USE_CUDA and set SMOP_GPU=1 to enable.
#ifdef SMOP_USE_CUDA
            const bool use_gpu = std::getenv("SMOP_GPU") != nullptr;
            SmopResult r;
            if (use_gpu && smop_gpu_available()) {
                if (d.sparse && d.As.nonZeros() >= 1000000LL) {
                    GpuSparseOperator gA(d.As);
                    r = solve_bmop(gA, d.b, delta, o);
                } else if (!d.sparse &&
                           static_cast<long long>(d.Ad.size()) >= 1000000LL) {
                    GpuDenseOperator gA(d.Ad);
                    r = solve_bmop(gA, d.b, delta, o);
                } else {
                    r = d.sparse ? solve_bmop(d.As, d.b, delta, o)
                                 : solve_bmop(d.Ad, d.b, delta, o);
                }
            } else {
                r = d.sparse ? solve_bmop(d.As, d.b, delta, o)
                             : solve_bmop(d.Ad, d.b, delta, o);
            }
#else
            SmopResult r;
            r = d.sparse ? solve_bmop(d.As, d.b, delta, o)
                         : solve_bmop(d.Ad, d.b, delta, o);
#endif
            const double psi = d.sparse ? (d.As * r.x - d.b).norm()
                                        : (d.Ad * r.x - d.b).norm();
            const double feas = std::abs(psi - delta) / std::max(1.0, delta);
            const double obj = r.x.cwiseAbs().sum();
            const int nnz = int((r.x.array().abs() > 1e-8).count());
            const int max_as = r.info.reducedn.empty()
                ? 0 : int(*std::max_element(r.info.reducedn.begin(),
                                            r.info.reducedn.end()));
            std::printf("  -> iter=%d (B=%d N/S=%d)  feas=%.2e  |x|1=%.4f  "
                        "nnz(1e-8)=%d  maxASred=%d  time=%.2fs  status=%d %s\n\n",
                        r.info.iter, r.info.iter_bisection,
                        r.info.iter_newton_or_secant, feas, obj, nnz, max_as,
                        wall_time_seconds() - t0, r.info.status,
                        r.info.msg.c_str());
            std::fflush(stdout);
        }
    }
    std::printf("DONE (unreadable=%d)\n", nfail);
    return 0;
}
