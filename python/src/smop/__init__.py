#======================================================================
# __init__.py -- public Python interface of smop
#======================================================================
import numpy as np

from . import _core
from ._core import solve_bmop, solve_lasso, version

__all__ = ["solve_bmop", "solve_lasso", "version", "__version__"]
__version__ = version()


def demo(n=150, m=40, seed=2026, delta_scale=1.02):
    """Build a random sparse problem with a known solution and solve it.

    Returns (res, xtrue, A, b, delta).
    """
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((m, n))
    xtrue = np.zeros(n)
    idx = rng.choice(n, size=8, replace=False)
    xtrue[idx] = rng.standard_normal(8)
    b = A @ xtrue + 0.02 * rng.standard_normal(m)
    delta = np.linalg.norm(A @ xtrue - b) * delta_scale
    res = solve_bmop(A, b, delta)
    return res, xtrue, A, b, delta
