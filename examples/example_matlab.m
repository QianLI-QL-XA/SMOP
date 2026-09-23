% example_matlab.m -- run after build_smop (from the matlab/ folder)
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

% plot the mu-path
if ~isempty(res.mupath)
    figure; semilogy(res.mupath, res.psipath, '-o');
    xlabel('\mu'); ylabel('\psi(\mu) = ||A x(\mu) - b||');
    hold on; yline(delta, 'r--', '\delta'); grid on;
end
