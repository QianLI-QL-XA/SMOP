//======================================================================
// smop_mex.cpp -- MEX interface for the smop core (MATLAB)
//
//   res = smop_mex(A, b, delta)            % default options
//   res = smop_mex(A, b, delta, opts)      % opts: struct with optional
//                                          %   fields stoptol, maxiter,
//                                          %   use_secant, use_newton,
//                                          %   use_as, mu0, muinf, verbose,
//                                          %   time_limit
//   res is a struct with fields:
//     x, xi, y, mu, psi, eta, iter, iter_bisection, iter_newton_or_secant,
//     iter_smoothing, iter_admm, status, msg, mupath, psipath, reducedn
//
// Build (from the matlab/ folder):
//   build.m
//======================================================================
#include "mex.h"

#include "smop/smop.hpp"

namespace {

void badarg(const char* msg)
{
    mexErrMsgIdAndTxt("smop:badarg", "%s", msg);
}

mxArray* vec_to_mx(const smop::Vec& v)
{
    mxArray* out = mxCreateDoubleMatrix(static_cast<mwSize>(v.size()), 1, mxREAL);
    double* p = mxGetPr(out);
    for (smop::Index i = 0; i < v.size(); ++i) p[i] = v(i);
    return out;
}

mxArray* intvec_to_mx(const std::vector<smop::Index>& v)
{
    mxArray* out = mxCreateDoubleMatrix(static_cast<mwSize>(v.size()), 1, mxREAL);
    double* p = mxGetPr(out);
    for (size_t i = 0; i < v.size(); ++i) p[i] = static_cast<double>(v[i]);
    return out;
}

mxArray* dblvec_to_mx(const std::vector<double>& v)
{
    mxArray* out = mxCreateDoubleMatrix(static_cast<mwSize>(v.size()), 1, mxREAL);
    double* p = mxGetPr(out);
    for (size_t i = 0; i < v.size(); ++i) p[i] = v[i];
    return out;
}

double sget(const mxArray* s, const char* f, double def)
{
    mxArray* v = mxGetField(s, 0, f);
    if (!v || !mxIsDouble(v) || mxIsEmpty(v) || mxGetNumberOfElements(v) != 1) return def;
    return mxGetScalar(v);
}

bool bget(const mxArray* s, const char* f, bool def)
{
    mxArray* v = mxGetField(s, 0, f);
    if (!v || mxGetNumberOfElements(v) != 1) return def;
    return mxGetScalar(v) != 0.0;
}

} // namespace

void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    if (nrhs < 3 || nrhs > 4)
        badarg("usage: res = smop_mex(A, b, delta [, opts])");
    (void)nlhs;

    // ---- A ----
    if (!mxIsDouble(prhs[0]) || mxIsSparse(prhs[0]) ||
        mxGetNumberOfDimensions(prhs[0]) != 2)
        badarg("A must be a dense double matrix");
    const mwSize m = mxGetM(prhs[0]);
    const mwSize n = mxGetN(prhs[0]);
    const double* Ap = mxGetPr(prhs[0]);
    smop::Mat Am(static_cast<smop::Index>(m), static_cast<smop::Index>(n));
    for (mwSize j = 0; j < n; ++j)
        for (mwSize i = 0; i < m; ++i)
            Am(static_cast<smop::Index>(i), static_cast<smop::Index>(j)) =
                Ap[j * m + i];

    // ---- b ----
    if (!mxIsDouble(prhs[1]) || mxGetNumberOfElements(prhs[1]) != m)
        badarg("b must be a double vector of length size(A,1)");
    const double* bp = mxGetPr(prhs[1]);
    smop::Vec bv(static_cast<smop::Index>(m));
    for (mwSize i = 0; i < m; ++i) bv(static_cast<smop::Index>(i)) = bp[i];

    // ---- delta ----
    if (!mxIsDouble(prhs[2]) || mxGetNumberOfElements(prhs[2]) != 1)
        badarg("delta must be a scalar");
    const double delta = mxGetScalar(prhs[2]);

    // ---- options ----
    smop::SmopOptions opt;
    if (nrhs == 4) {
        if (!mxIsStruct(prhs[3])) badarg("opts must be a struct");
        const mxArray* s = prhs[3];
        opt.stoptol          = sget(s, "stoptol", opt.stoptol);
        opt.maxiter_levelset = static_cast<int>(sget(s, "maxiter", opt.maxiter_levelset));
        opt.use_secant       = bget(s, "use_secant", opt.use_secant);
        opt.use_newton       = bget(s, "use_newton", opt.use_newton);
        opt.use_as           = bget(s, "use_as", opt.use_as);
        opt.mu0              = sget(s, "mu0", opt.mu0);
        opt.muinf            = sget(s, "muinf", opt.muinf);
        opt.initial_mu       = sget(s, "initial_mu", opt.initial_mu);
        opt.verbose          = static_cast<int>(sget(s, "verbose", opt.verbose));
        opt.time_limit       = sget(s, "time_limit", opt.time_limit);
    }

    // ---- solve ----
    smop::SmopResult r = smop::solve_bmop(Am, bv, delta, opt);

    // ---- output struct ----
    static const char* names[] = {
        "x", "xi", "y", "mu", "psi", "eta", "iter", "iter_bisection",
        "iter_newton_or_secant", "iter_smoothing", "iter_admm", "status",
        "msg", "mupath", "psipath", "reducedn"};
    const int nf = static_cast<int>(sizeof(names) / sizeof(names[0]));
    plhs[0] = mxCreateStructMatrix(1, 1, nf, names);

    mxSetField(plhs[0], 0, "x",  vec_to_mx(r.x));
    mxSetField(plhs[0], 0, "xi", vec_to_mx(r.xi));
    mxSetField(plhs[0], 0, "y",  vec_to_mx(r.y));
    mxSetField(plhs[0], 0, "mu", mxCreateDoubleScalar(r.mu));
    mxSetField(plhs[0], 0, "psi", mxCreateDoubleScalar(r.info.psi));
    mxSetField(plhs[0], 0, "eta", mxCreateDoubleScalar(r.info.eta));
    mxSetField(plhs[0], 0, "iter", mxCreateDoubleScalar(r.info.iter));
    mxSetField(plhs[0], 0, "iter_bisection", mxCreateDoubleScalar(r.info.iter_bisection));
    mxSetField(plhs[0], 0, "iter_newton_or_secant",
               mxCreateDoubleScalar(r.info.iter_newton_or_secant));
    mxSetField(plhs[0], 0, "iter_smoothing", mxCreateDoubleScalar(r.info.iter_smoothing));
    mxSetField(plhs[0], 0, "iter_admm", mxCreateDoubleScalar(r.info.iter_admm));
    mxSetField(plhs[0], 0, "status", mxCreateDoubleScalar(r.info.status));
    mxSetField(plhs[0], 0, "msg", mxCreateString(r.info.msg.c_str()));
    mxSetField(plhs[0], 0, "mupath",  dblvec_to_mx(r.info.mupath));
    mxSetField(plhs[0], 0, "psipath", dblvec_to_mx(r.info.psipath));
    mxSetField(plhs[0], 0, "reducedn", intvec_to_mx(r.info.reducedn));
}
