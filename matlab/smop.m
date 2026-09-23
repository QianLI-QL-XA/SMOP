function res = smop(A, b, delta, opts)
%BMOP  Solve min ||x||_1  s.t.  ||A x - b||_2 <= delta (level-set method).
%
%   res = BMOP(A, b, delta)
%   res = BMOP(A, b, delta, opts)
%
%   A      : m x n dense double matrix
%   b      : m x 1 vector
%   delta  : scalar constraint radius
%   opts   : (optional) struct with fields
%              stoptol   (1e-6)  outer tolerance
%              maxiter   (200)   max level-set iterations
%              use_secant (true) secant root-finding steps
%              use_newton (false) analytic-Newton steps
%              use_as     (true) adaptive sieving for subproblems
%              mu0, muinf, initial_mu : bracket / initial point
%              verbose   (0)
%              time_limit (3600)
%
%   res is a struct with fields
%     x, xi, y, mu, psi, eta, iter, iter_bisection,
%     iter_newton_or_secant, iter_smoothing, iter_admm, status, msg,
%     mupath, psipath, reducedn
%
% Example:
%   rng(1); A = randn(40,150); xt = zeros(150,1);
%   xt(randperm(150,8)) = randn(8,1); b = A*xt + 0.02*randn(40,1);
%   delta = 1.02*norm(A*xt - b);
%   res = smop(A, b, delta); fprintf('||Ax-b|| = %g, delta = %g\n', norm(A*res.x-b), delta);
%
% See also: smop_lasso, build_smop
%
% Build the MEX file first (once) with: build_smop

if nargin < 3, error('smop:arg', 'usage: smop(A, b, delta [, opts])'); end
if nargin < 4, opts = struct(); end
res = smop_mex(A, b, delta, opts);
end
