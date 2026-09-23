//======================================================================
// smop_r.cpp -- Rcpp bindings for the smop core (R interface)
//
// Package structure:
//   R/DESCRIPTION, R/NAMESPACE, R/R/smop.R
//   R/src/smop_r.cpp  (this file)
//
// Build with:  R CMD INSTALL --preclean .
// (requires Rtools on Windows; Rcpp is the only R dependency)
//======================================================================
#include <Rcpp.h>

#include "smop/smop.hpp"

// [[Rcpp::plugins(cpp17)]]
// [[Rcpp::depends(Rcpp)]]

namespace {

inline smop::Mat rmat_to_eigen(const Rcpp::NumericMatrix& M)
{
    smop::Mat A(M.nrow(), M.ncol());
    for (smop::Index i = 0; i < A.rows(); ++i)
        for (smop::Index j = 0; j < A.cols(); ++j)
            A(i, j) = M(i, j);
    return A;
}

inline smop::Vec rvec_to_eigen(const Rcpp::NumericVector& v)
{
    smop::Vec x(v.size());
    for (smop::Index i = 0; i < x.size(); ++i) x(i) = v[i];
    return x;
}

inline Rcpp::NumericVector eigen_to_r(const smop::Vec& v)
{
    Rcpp::NumericVector out(v.size());
    for (smop::Index i = 0; i < v.size(); ++i) out[i] = v(i);
    return out;
}

// options from a named list; unknown elements are ignored
inline smop::SmopOptions smop_opts(const Rcpp::List& opt)
{
    smop::SmopOptions o;
    auto d = [&](const char* key, double& f) {
        if (opt.containsElementNamed(key)) f = Rcpp::as<double>(opt[key]);
    };
    auto i_ = [&](const char* key, int& f) {
        if (opt.containsElementNamed(key)) f = Rcpp::as<int>(opt[key]);
    };
    auto b_ = [&](const char* key, bool& f) {
        if (opt.containsElementNamed(key)) f = Rcpp::as<bool>(opt[key]);
    };
    d("stoptol", o.stoptol);
    i_("maxiter", o.maxiter_levelset);
    b_("use_secant", o.use_secant);
    b_("use_newton", o.use_newton);
    b_("use_as", o.use_as);
    d("mu0", o.mu0);
    d("muinf", o.muinf);
    d("initial_mu", o.initial_mu);
    d("time_limit", o.time_limit);
    i_("verbose", o.verbose);
    return o;
}

inline smop::LassoOptions lasso_opts(const Rcpp::List& opt)
{
    smop::LassoOptions o;
    auto d = [&](const char* key, double& f) {
        if (opt.containsElementNamed(key)) f = Rcpp::as<double>(opt[key]);
    };
    auto i_ = [&](const char* key, int& f) {
        if (opt.containsElementNamed(key)) f = Rcpp::as<int>(opt[key]);
    };
    auto b_ = [&](const char* key, bool& f) {
        if (opt.containsElementNamed(key)) f = Rcpp::as<bool>(opt[key]);
    };
    d("stoptol", o.stoptol);
    i_("maxiter", o.maxiter);
    b_("use_smoothing", o.use_smoothing);
    b_("use_admm", o.use_admm);
    d("eps_hat", o.eps_hat);
    d("kappa", o.kappa);
    d("rho", o.rho);
    d("armijo_sigma", o.armijo_sigma);
    d("eta_hat", o.eta_hat);
    i_("maxiter_as", o.maxiter_as);
    d("time_limit", o.time_limit);
    return o;
}

} // namespace

// [[Rcpp::export]]
Rcpp::List smop_solve_bmop(Rcpp::NumericMatrix A, Rcpp::NumericVector b,
                           double delta, Rcpp::Nullable<Rcpp::List> options = R_NilValue,
                           Rcpp::Nullable<Rcpp::NumericVector> x0 = R_NilValue)
{
    smop::Mat Am = rmat_to_eigen(A);
    smop::Vec bv = rvec_to_eigen(b);
    smop::SmopOptions o;
    smop::Vec x0v;
    if (options.isNotNull()) o = smop_opts(Rcpp::as<Rcpp::List>(options));
    if (x0.isNotNull())      x0v = rvec_to_eigen(Rcpp::as<Rcpp::NumericVector>(x0));

    smop::SmopResult r = smop::solve_bmop(Am, bv, delta, o, x0v);
    return Rcpp::List::create(
        Rcpp::Named("x")   = eigen_to_r(r.x),
        Rcpp::Named("xi")  = eigen_to_r(r.xi),
        Rcpp::Named("y")   = eigen_to_r(r.y),
        Rcpp::Named("mu")  = r.mu,
        Rcpp::Named("psi") = r.info.psi,
        Rcpp::Named("eta") = r.info.eta,
        Rcpp::Named("iter")= r.info.iter,
        Rcpp::Named("iter_bisection")  = r.info.iter_bisection,
        Rcpp::Named("iter_newton_or_secant") = r.info.iter_newton_or_secant,
        Rcpp::Named("iter_smoothing")  = r.info.iter_smoothing,
        Rcpp::Named("iter_admm")       = r.info.iter_admm,
        Rcpp::Named("status")= r.info.status,
        Rcpp::Named("msg")  = r.info.msg,
        Rcpp::Named("time") = r.info.time,
        Rcpp::Named("mupath")  = Rcpp::wrap(r.info.mupath),
        Rcpp::Named("psipath") = Rcpp::wrap(r.info.psipath),
        Rcpp::Named("reducedn")= Rcpp::wrap(r.info.reducedn));
}

// [[Rcpp::export]]
Rcpp::List smop_solve_lasso(Rcpp::NumericMatrix A, Rcpp::NumericVector b,
                            double lambda, Rcpp::Nullable<Rcpp::List> options = R_NilValue,
                            Rcpp::Nullable<Rcpp::NumericVector> x0 = R_NilValue)
{
    smop::Mat Am = rmat_to_eigen(A);
    smop::Vec bv = rvec_to_eigen(b);
    smop::LassoOptions o;
    smop::Vec x0v;
    if (options.isNotNull()) o = lasso_opts(Rcpp::as<Rcpp::List>(options));
    if (x0.isNotNull())      x0v = rvec_to_eigen(Rcpp::as<Rcpp::NumericVector>(x0));

    smop::LassoResult r = smop::solve_lasso(Am, bv, lambda, o, x0v);
    return Rcpp::List::create(
        Rcpp::Named("x")    = eigen_to_r(r.x),
        Rcpp::Named("xi")   = eigen_to_r(r.xi),
        Rcpp::Named("grad") = eigen_to_r(r.grad),
        Rcpp::Named("obj")  = r.info.obj,
        Rcpp::Named("iter") = r.info.iter,
        Rcpp::Named("eta")  = r.info.eta,
        Rcpp::Named("status")= r.info.status,
        Rcpp::Named("msg")  = r.info.msg,
        Rcpp::Named("time") = r.info.time);
}

// [[Rcpp::export]]
std::string smop_version_cpp() { return smop::version(); }
