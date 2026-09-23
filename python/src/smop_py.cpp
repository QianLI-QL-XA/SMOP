//======================================================================
// smop_py.cpp -- pybind11 module exposing the smop core to Python
//
//   import smop
//   res = smop.solve_bmop(A, b, delta, **options)
//   res = smop.solve_lasso(A, b, lambda, **options)
//
//   A : 2-D numpy array (float64), dense.  x0 : optional 1-D array.
//   Result keys for solve_bmop:
//     x, xi, mu, psi, eta, abseta, iter, iter_bisection,
//     iter_newton_or_secant, iter_smoothing, iter_admm, status, msg,
//     mupath, psipath, reducedn, time
//   Result keys for solve_lasso:
//     x, xi, grad, obj, iter, eta, status, msg, time
//======================================================================
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <string>
#include <vector>

#include "smop/smop.hpp"

namespace py = pybind11;

namespace {

//------------------------------------------------------------------------
// options parsing
//------------------------------------------------------------------------
smop::SmopOptions smop_options(const py::kwargs& kw)
{
    smop::SmopOptions o;
    auto get = [&](const char* key, auto& field, bool is_int) {
        if (kw.contains(key)) {
            if (is_int) field = kw[key].cast<int>();
            else        field = kw[key].cast<double>();
        }
    };
    auto getb = [&](const char* key, bool& field) {
        if (kw.contains(key)) field = kw[key].cast<bool>();
    };
    get("stoptol", o.stoptol, false);
    get("maxiter", o.maxiter_levelset, true);
    getb("use_secant", o.use_secant);
    getb("use_newton", o.use_newton);
    getb("use_as", o.use_as);
    get("mu0", o.mu0, false);
    get("muinf", o.muinf, false);
    get("initial_mu", o.initial_mu, false);
    get("time_limit", o.time_limit, false);
    get("verbose", o.verbose, true);
    return o;
}

smop::LassoOptions lasso_options(const py::kwargs& kw)
{
    smop::LassoOptions o;
    auto get = [&](const char* key, auto& field, bool is_int) {
        if (kw.contains(key)) {
            if (is_int) field = kw[key].cast<int>();
            else        field = kw[key].cast<double>();
        }
    };
    auto getb = [&](const char* key, bool& field) {
        if (kw.contains(key)) field = kw[key].cast<bool>();
    };
    get("stoptol", o.stoptol, false);
    get("maxiter", o.maxiter, true);
    getb("use_smoothing", o.use_smoothing);
    getb("use_admm", o.use_admm);
    get("eps_hat", o.eps_hat, false);
    get("kappa", o.kappa, false);
    get("rho", o.rho, false);
    get("armijo_sigma", o.armijo_sigma, false);
    get("eta_hat", o.eta_hat, false);
    get("maxiter_as", o.maxiter_as, true);
    get("time_limit", o.time_limit, false);
    return o;
}

//------------------------------------------------------------------------
// array conversion
//------------------------------------------------------------------------
smop::Mat to_mat(const py::array_t<double, py::array::c_style | py::array::forcecast>& a)
{
    auto buf = a.request();
    if (buf.ndim != 2) throw std::runtime_error("A must be a 2-D array");
    smop::Mat M(buf.shape[0], buf.shape[1]);
    const double* p = static_cast<const double*>(buf.ptr);
    for (smop::Index i = 0; i < M.rows(); ++i)
        for (smop::Index j = 0; j < M.cols(); ++j)
            M(i, j) = p[i * buf.strides[0] / sizeof(double) + j * buf.strides[1] / sizeof(double)];
    return M;
}

smop::Vec to_vec(const py::array_t<double, py::array::c_style | py::array::forcecast>& a)
{
    auto buf = a.request();
    if (buf.ndim != 1) throw std::runtime_error("vector must be 1-D");
    smop::Vec v(buf.shape[0]);
    const double* p = static_cast<const double*>(buf.ptr);
    for (smop::Index i = 0; i < v.size(); ++i)
        v(i) = p[i * buf.strides[0] / sizeof(double)];
    return v;
}

py::array_t<double> to_np(const smop::Vec& v)
{
    py::array_t<double> out(v.size());
    std::memcpy(out.mutable_data(), v.data(), v.size() * sizeof(double));
    return out;
}

//------------------------------------------------------------------------
// result dicts
//------------------------------------------------------------------------
py::dict lasso_result_dict(const smop::LassoResult& r)
{
    py::dict d;
    d["x"] = to_np(r.x);
    d["xi"] = to_np(r.xi);
    d["grad"] = to_np(r.grad);
    d["obj"] = r.info.obj;
    d["iter"] = r.info.iter;
    d["eta"] = r.info.eta;
    d["etaorg"] = r.info.etaorg;
    d["status"] = r.info.status;
    d["msg"] = r.info.msg;
    d["time"] = r.info.time;
    return d;
}

py::dict smop_result_dict(const smop::SmopResult& r)
{
    py::dict d;
    d["x"] = to_np(r.x);
    d["xi"] = to_np(r.xi);
    d["y"] = to_np(r.y);
    d["mu"] = r.mu;
    d["psi"] = r.info.psi;
    d["eta"] = r.info.eta;
    d["abseta"] = r.info.abseta;
    d["iter"] = r.info.iter;
    d["iter_bisection"] = r.info.iter_bisection;
    d["iter_newton_or_secant"] = r.info.iter_newton_or_secant;
    d["iter_smoothing"] = r.info.iter_smoothing;
    d["iter_admm"] = r.info.iter_admm;
    d["status"] = r.info.status;
    d["msg"] = r.info.msg;
    d["time"] = r.info.time;
    d["mupath"] = r.info.mupath;
    d["psipath"] = r.info.psipath;
    d["reducedn"] = r.info.reducedn;
    return d;
}

} // namespace

PYBIND11_MODULE(_core, m)
{
    m.doc() = "smop: level-set method for sparse optimization (BMOP) - C++ core";
    m.def("version", &smop::version);

    m.def("solve_bmop",
          [](py::array_t<double, py::array::c_style | py::array::forcecast> A,
             py::array_t<double, py::array::c_style | py::array::forcecast> b,
             double delta, py::kwargs kw) {
              smop::Mat Am = to_mat(A);
              smop::Vec bv = to_vec(b);
              smop::SmopOptions o = smop_options(kw);
              smop::Vec x0;
              if (kw.contains("x0")) x0 = to_vec(kw["x0"].cast<py::array_t<double>>());
              smop::SmopResult r = smop::solve_bmop(Am, bv, delta, o, x0);
              return smop_result_dict(r);
          },
          py::arg("A"), py::arg("b"), py::arg("delta"),
          "Solve the BMOP min ||x||_1 s.t. ||Ax-b||_2 <= delta by the level-set method.\n"
          "Optional keyword options: stoptol, maxiter, use_secant, use_newton, use_as,\n"
          "mu0, muinf, initial_mu, time_limit, verbose, x0.");

    m.def("solve_lasso",
          [](py::array_t<double, py::array::c_style | py::array::forcecast> A,
             py::array_t<double, py::array::c_style | py::array::forcecast> b,
             double lambda, py::kwargs kw) {
              smop::Mat Am = to_mat(A);
              smop::Vec bv = to_vec(b);
              smop::LassoOptions o = lasso_options(kw);
              smop::Vec x0;
              if (kw.contains("x0")) x0 = to_vec(kw["x0"].cast<py::array_t<double>>());
              smop::LassoResult r = smop::solve_lasso(Am, bv, lambda, o, x0);
              return lasso_result_dict(r);
          },
          py::arg("A"), py::arg("b"), py::arg("lambda"),
          "Solve the Lasso min 1/2||Ax-b||^2 + lambda||x||_1 by smoothing Newton\n"
          "(ADMM fallback / direct ADMM on large problems).");
}
