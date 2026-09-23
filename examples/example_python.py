# example_python.py -- run: python examples/example_python.py
# (requires: pip install ./python   from the package root)
import numpy as np

import smop

rng = np.random.default_rng(0)
m, n = 40, 150
A = rng.standard_normal((m, n))
xtrue = np.zeros(n)
xtrue[rng.choice(n, 8, replace=False)] = rng.standard_normal(8)
b = A @ xtrue + 0.02 * rng.standard_normal(m)
delta = 1.02 * np.linalg.norm(A @ xtrue - b)

res = smop.solve_bmop(A, b, delta, verbose=1)
print()
print("status :", res["status"], res["msg"])
print("||Ax-b|| = %.6e  delta = %.6e" % (np.linalg.norm(A @ res["x"] - b), delta))
print("||x||_1  = %.6e  (true %.6e)" % (np.abs(res["x"]).sum(), np.abs(xtrue).sum()))
print("iter    :", res["iter"], " bisection:", res["iter_bisection"],
      " secant/newton:", res["iter_newton_or_secant"])
print("support :", int((np.abs(res["x"]) > 1e-8).sum()), "(true 8)")

# single Lasso subproblem as well
lr = smop.solve_lasso(A, b, 0.05)
print("lasso   : obj=%.6e  kkt_eta=%.2e" %
      (lr["obj"], np.linalg.norm(A.T @ (A @ lr["x"] - b)) * 1e0))
