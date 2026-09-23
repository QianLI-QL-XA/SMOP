% test_smop_octave.m -- validate the smop mex interface under Octave
addpath('/mnt/c/Users/qianl/OneDrive/codes/SMOP/smop/matlab');
rng(0);
m = 40; n = 150;
A = randn(m, n);
xtrue = zeros(n, 1);
idx = randperm(n, 8); xtrue(idx) = randn(8, 1);
b = A * xtrue + 0.02 * randn(m, 1);
delta = 1.02 * norm(A * xtrue - b);

res = smop(A, b, delta);
fprintf('status = %d (%s)\n', res.status, res.msg);
fprintf('||Ax-b|| = %.6e  delta = %.6e\n', norm(A * res.x - b), delta);
fprintf('||x||_1 = %.6e  (true %.6e)\n', sum(abs(res.x)), sum(abs(xtrue)));
fprintf('iter = %d (B=%d S/N=%d)\n', res.iter, res.iter_bisection, ...
        res.iter_newton_or_secant);
fprintf('support = %d (true 8)\n', sum(abs(res.x) > 1e-8));
% options struct path
res2 = smop(A, b, delta, struct('verbose', 0, 'stoptol', 1e-6));
fprintf('status2 = %d\n', res2.status);
% Lasso via smop_lasso_mex
lr = smop_lasso(A, b, 0.05);
fprintf('lasso obj = %.6e  kkt = %.2e  status = %d\n', lr.obj, lr.eta, lr.status);
fprintf('OCTAVE_TEST_DONE\n');
