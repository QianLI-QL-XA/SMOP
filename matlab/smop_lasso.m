function res = smop_lasso(A, b, lambda, opts)
%SMOP_LASSO  Solve min 1/2||Ax-b||_2^2 + lambda*||x||_1 (Lasso).
%
%   res = smop_lasso(A, b, lambda)
%   res = smop_lasso(A, b, lambda, opts)
%
%   A       : m x n dense double matrix
%   b       : m x 1 vector
%   lambda  : scalar regularization weight
%   opts    : (optional) struct with fields
%               stoptol      (1e-6)  relative KKT tolerance
%               maxiter      (200)   max outer iterations
%               use_smoothing (true) smoothing-Newton subproblem
%               use_ssn      (true)  SSNAL subproblem
%               use_admm     (true)  ADMM fallback
%               eps_hat, kappa, rho, armijo_sigma, eta_hat, maxiter_as,
%               time_limit   (3600)
%
%   res is a struct with fields
%     x, xi, grad, obj, iter, eta, status, msg, time
%
% Example:
%   A = randn(40,150); b = randn(40,1);
%   res = smop_lasso(A, b, 0.05);
%   fprintf('obj = %g, kkt = %g\n', res.obj, res.eta);
%
% Build the MEX file first (once) with: build_smop

if nargin < 3, error('smop:arg', 'usage: smop_lasso(A, b, lambda [, opts])'); end
if nargin < 4, opts = struct(); end
res = smop_lasso_mex(A, b, lambda, opts);
end
