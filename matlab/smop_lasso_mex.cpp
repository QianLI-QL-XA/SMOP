//======================================================================
// smop_lasso_mex.cpp -- MEX interface for the Lasso solver (MATLAB)
//
//   res = smop_lasso(A, b, lambda)            % default options
//   res = smop_lasso(A, b, lambda, opts)      % opts: struct with optional
//                                             %   fields stoptol, maxiter,
//                                             %   use_smoothing, use_admm,
//                                             %   eps_hat, kappa, rho,
//                                             %   armijo_sigma, eta_hat,
//                                             %   maxiter_as, time_limit
//   res is a struct with fields:
//     x, xi, grad, obj, iter, eta, status, msg, time
//
// Build (from the matlab/ folder):
//   build_smop   (compiles smop_mex and smop_lasso_mex)
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
        badarg("usage: res = smop_lasso(A, b, lambda [, opts])");
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

    // ---- lambda ----
    if (!mxIsDouble(prhs[2]) || mxGetNumberOfElements(prhs[2]) != 1)
        badarg("lambda must be a scalar");
    const double lambda = mxGetScalar(prhs[2]);

    // ---- options ----
    smop::LassoOptions opt;
    if (nrhs == 4) {
        if (!mxIsStruct(prhs[3])) badarg("opts must be a struct");
        const mxArray* s = prhs[3];
        opt.stoptol      = sget(s, "stoptol", opt.stoptol);
        opt.maxiter      = static_cast<int>(sget(s, "maxiter", opt.maxiter));
        opt.use_smoothing = bget(s, "use_smoothing", opt.use_smoothing);
        opt.use_ssn      = bget(s, "use_ssn", opt.use_ssn);
        opt.use_admm     = bget(s, "use_admm", opt.use_admm);
        opt.eps_hat      = sget(s, "eps_hat", opt.eps_hat);
        opt.kappa        = sget(s, "kappa", opt.kappa);
        opt.rho          = sget(s, "rho", opt.rho);
        opt.armijo_sigma = sget(s, "armijo_sigma", opt.armijo_sigma);
        opt.eta_hat      = sget(s, "eta_hat", opt.eta_hat);
        opt.maxiter_as   = static_cast<int>(sget(s, "maxiter_as", opt.maxiter_as));
        opt.time_limit   = sget(s, "time_limit", opt.time_limit);
    }

    // ---- solve ----
    smop::LassoResult r = smop::solve_lasso(Am, bv, lambda, opt);

    // ---- output struct ----
    static const char* names[] = {
        "x", "xi", "grad", "obj", "iter", "eta", "status", "msg", "time"};
    const int nf = static_cast<int>(sizeof(names) / sizeof(names[0]));
    plhs[0] = mxCreateStructMatrix(1, 1, nf, names);

    mxSetField(plhs[0], 0, "x",     vec_to_mx(r.x));
    mxSetField(plhs[0], 0, "xi",    vec_to_mx(r.xi));
    mxSetField(plhs[0], 0, "grad",  vec_to_mx(r.grad));
    mxSetField(plhs[0], 0, "obj",   mxCreateDoubleScalar(r.info.obj));
    mxSetField(plhs[0], 0, "iter",  mxCreateDoubleScalar(r.info.iter));
    mxSetField(plhs[0], 0, "eta",   mxCreateDoubleScalar(r.info.eta));
    mxSetField(plhs[0], 0, "status", mxCreateDoubleScalar(r.info.status));
    mxSetField(plhs[0], 0, "msg",   mxCreateString(r.info.msg.c_str()));
    mxSetField(plhs[0], 0, "time",  mxCreateDoubleScalar(r.info.time));
}
