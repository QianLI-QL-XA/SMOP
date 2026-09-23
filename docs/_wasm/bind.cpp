//======================================================================
// wasm_bind.cpp -- SMOP online demo binding for Emscripten/WASM
//   exports solve_bmop / solve_lasso as JSON strings
//   inputs are plain JS arrays (converted on the C++ side via val)
//======================================================================
#include <emscripten/bind.h>
#include <emscripten/val.h>
#include "smop/smop.hpp"
#include <sstream>
#include <iomanip>
#include <string>

using namespace emscripten;
using smop::Index;

static std::string esc(const std::string& s)
{
    std::string o;
    o.reserve(s.size());
    for (char c : s) {
        if (c == '"' || c == '\\') o += '\\';
        o += c;
    }
    return o;
}

std::string wasm_solve_bmop(const val& Aflat, int rows, int cols,
                            const val& b, double delta,
                            double stoptol, double time_limit, int verbose, bool use_as)
{
    smop::Mat A(rows, cols);
    for (int j = 0; j < cols; ++j)
        for (int i = 0; i < rows; ++i)
            A(i, j) = Aflat[(size_t)j * rows + i].as<double>();
    smop::Vec B(rows);
    for (int i = 0; i < rows; ++i) B(i) = b[i].as<double>();

    smop::SmopOptions opt;
    opt.stoptol    = stoptol;
    opt.time_limit = time_limit;
    opt.verbose    = verbose;
    opt.use_as     = use_as;
    smop::SmopResult r = smop::solve_bmop(A, B, delta, opt);
    std::ostringstream o;
    o << std::setprecision(12);
    o << "{\"status\":" << r.info.status
      << ",\"msg\":\"" << esc(r.info.msg) << "\""
      << ",\"iter\":" << r.info.iter
      << ",\"iter_bisection\":" << r.info.iter_bisection
      << ",\"iter_newton_or_secant\":" << r.info.iter_newton_or_secant
      << ",\"iter_smoothing\":" << r.info.iter_smoothing
      << ",\"iter_ssn\":" << r.info.iter_ssn
      << ",\"iter_admm\":" << r.info.iter_admm
      << ",\"eta\":" << r.info.eta
      << ",\"psi\":" << r.info.psi
      << ",\"mu\":" << r.info.mu
      << ",\"time\":" << r.info.time
      << ",\"obj\":" << r.x.cwiseAbs().sum()
      << ",\"nnz\":" << (r.x.cwiseAbs().array() > 1e-10).count()
      << ",\"x\":[";
    for (Index i = 0; i < r.x.size(); ++i) { if (i) o << ","; o << r.x(i); }
    o << "]}";
    return o.str();
}

std::string wasm_solve_lasso(const val& Aflat, int rows, int cols,
                             const val& b, double lambda,
                             double stoptol, double time_limit, int verbose)
{
    smop::Mat A(rows, cols);
    for (int j = 0; j < cols; ++j)
        for (int i = 0; i < rows; ++i)
            A(i, j) = Aflat[(size_t)j * rows + i].as<double>();
    smop::Vec B(rows);
    for (int i = 0; i < rows; ++i) B(i) = b[i].as<double>();

    smop::LassoOptions opt;
    opt.stoptol    = stoptol;
    opt.time_limit = time_limit;
    opt.verbose    = verbose;
    smop::LassoResult r = smop::solve_lasso(A, B, lambda, opt);
    std::ostringstream o;
    o << std::setprecision(12);
    o << "{\"status\":" << r.info.status
      << ",\"msg\":\"" << esc(r.info.msg) << "\""
      << ",\"iter\":" << r.info.iter
      << ",\"eta\":" << r.info.eta
      << ",\"time\":" << r.info.time
      << ",\"obj\":" << r.info.obj
      << ",\"nnz\":" << (r.x.cwiseAbs().array() > 1e-10).count()
      << ",\"x\":[";
    for (Index i = 0; i < r.x.size(); ++i) { if (i) o << ","; o << r.x(i); }
    o << "]}";
    return o.str();
}

EMSCRIPTEN_BINDINGS(smop_wasm) {
    function("solve_bmop", &wasm_solve_bmop);
    function("solve_lasso", &wasm_solve_lasso);
}
