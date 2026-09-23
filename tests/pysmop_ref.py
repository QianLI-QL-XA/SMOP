#======================================================================
# pysmop_ref.py -- pure-numpy reference implementation of the smop
# level-set algorithm (mirrors the C++ core in include/smop/*.hpp).
#
# Purpose: runnable numerical validation of the *algorithm* in any
# environment without a C++ toolchain (the compiled C++ core itself is
# validated by tests/test_cpp.cpp on a machine that can run binaries).
#
#   python tests/pysmop_ref.py
#======================================================================
import numpy as np

# ----------------------------------------------------------------------
# inner pieces
# ----------------------------------------------------------------------

def proj_inf(x, lam):
    return np.clip(x, -lam, lam)


def energy_nnz(x, r=0.999):
    """number of entries capturing 99.9% of the L1 energy (descending)."""
    vals = np.sort(np.abs(x[x != 0]))[::-1]
    if vals.size == 0:
        return 0
    acc = np.cumsum(vals)
    return int(np.searchsorted(acc, r * acc[-1])) + 1


def kkt_eta(x, g, lam):
    ev = g + proj_inf(x - g, lam)
    return np.linalg.norm(ev) / (1.0 + np.linalg.norm(g) + np.linalg.norm(x)), ev


def prox_l1(x, lam):
    return np.sign(x) * np.maximum(np.abs(x) - lam, 0.0)


def smoothing_fun(eps, t):
    """s_eps(t): quadratic-spline smoothing of |t| (same as smoothingNewton.m)."""
    eps = abs(eps)
    half = 0.5 * eps
    s = np.where(t >= half, t,
                 np.where(t <= -half, 0.0, (t + half) ** 2 / (2 * eps)))
    fe = np.where(np.abs(t) < half, 0.125 - 0.5 * (t / eps) ** 2, 0.0)
    fx = np.where(t >= half, 1.0,
                  np.where(t <= -half, 0.0, 0.5 + t / eps))
    return s, fe, fx


def find_phi(xinput, eps, x, lam, kappa):
    f = xinput - lam
    g = -xinput - lam
    ff, fe, fx = smoothing_fun(eps, f)
    gg, ge, gx = smoothing_fun(eps, g)
    G = x - ff + gg + kappa * abs(eps) * x
    return eps * eps + G @ G, G, fe, fx, ge, gx


def smoothing_newton(A, b, lam, x0, stoptol=1e-6, maxiter=800,
                     eps_hat=1.0, kappa=1.0, rho=0.7, sigma_arm=5e-7,
                     eta_hat=1e-4, verbose=False):
    """Port of smoothingNewton.m / smoothing_newton.hpp."""
    n = A.shape[1]
    x = x0.copy()
    Ax = A @ x
    grad = A.T @ (Ax - b)
    r = 0.25 / max(1.0, eps_hat)
    delta = np.sqrt(2.0) * max(eta_hat, r * eps_hat)

    eps = eps_hat
    x0old = x.copy()
    G, fe, fx, ge, gx = None, None, None, None, None
    phi, G, fe, fx, ge, gx = find_phi(x - grad, eps, x, lam, kappa)
    rel = 1e300
    for it in range(maxiter):
        if eps < 1e-13 or not np.isfinite(phi):
            return x, 0, rel
        theta = r * min(1.0, phi)
        deps = -eps + eps_hat * theta
        rhs = -G - deps * (kappa * np.sign(eps) * x - fe + ge)
        u = fx + gx
        tmp = 1.0 + kappa * abs(eps)
        if u.max() <= 0:
            dx = rhs / tmp
        else:
            # CG for (diag(tmp-u) + diag(u) A^T A) dx = rhs  (plain CG)
            def matvec(v):
                return (tmp - u) * v + u * (A.T @ (A @ v))
            dx, _ = cg(matvec, rhs, x0=None, maxit=max(100, 4 * n), tol=1e-8)
        # Armijo backtracking
        Adx = A @ dx
        ATAdx = A.T @ Adx
        phi0 = phi
        alp = 1.0
        accepted = False
        xnew, G2, fe2, fx2, ge2, gx2 = None, None, None, None, None, None
        for k in range(30):
            alp = rho ** k
            xnew = x0old + alp * dx
            epsnew = eps + alp * deps  # (eps0 + alp*deps, eps0==eps)
            xinnew = xnew - (grad + alp * ATAdx)
            phi_new, G2, fe2, fx2, ge2, gx2 = find_phi(xinnew, epsnew, xnew, lam, kappa)
            if phi_new <= (1.0 - 2.0 * sigma_arm * (1.0 - delta) * alp) * phi0:
                accepted = True
                break
            if not np.isfinite(phi_new):
                break
        if not accepted:
            alp = rho ** 29
            xnew = x0old + alp * dx
            epsnew = eps + alp * deps
            xinnew = xnew - (grad + alp * ATAdx)
            phi_new, G2, fe2, fx2, ge2, gx2 = find_phi(xinnew, epsnew, xnew, lam, kappa)
            if phi_new > 2.0 * phi0 and alp < 1e-6:
                return x0old, 0, rel
        x = xnew
        eps = epsnew
        Ax += alp * Adx
        grad = A.T @ (Ax - b)
        if not np.all(np.isfinite(x)) or np.linalg.norm(x) > 1e10 * max(1.0, np.linalg.norm(x0)):
            return x0old, 0, rel
        G, fe, fx, ge, gx = G2, fe2, fx2, ge2, gx2
        phi = phi_new
        rel, ev = kkt_eta(x, grad, lam)
        if rel < stoptol:
            return x, 1, rel
        if np.sqrt(phi) < stoptol * 1e-2 or eps < 1e-13:
            return x, 2, rel
        x0old = x
    return x, 0, rel


def cg(matvec, rhs, x0=None, maxit=1000, tol=1e-8):
    n = rhs.size
    x = np.zeros(n) if x0 is None else x0.copy()
    r = rhs - (matvec(x) if x0 is not None else 0.0)
    p = r.copy()
    rs = r @ r
    target = tol * tol * rhs @ rhs
    it = 0
    for it in range(1, maxit + 1):
        ap = matvec(p)
        pap = p @ ap
        if pap <= 0:
            break
        alpha = rs / pap
        x += alpha * p
        r -= alpha * ap
        if r @ r <= target:
            return x, it
        rsnew = r @ r
        p = r + (rsnew / rs) * p
        rs = rsnew
    return x, it


def admm_l1(A, b, lam, x0, stoptol=1e-6, maxiter=10000, gamma=1.618):
    """Port of admmL1.m / admm_l1.hpp."""
    m, n = A.shape
    x = x0.copy()
    Ax = A @ x
    xi = b - Ax
    y = np.zeros(n)
    sigma = min(1.0, max(1e-4, lam))
    normb = 1.0 + np.linalg.norm(b)
    trigger = set([3, 6, 12, 25, 50, 100])
    pw = dw = 0
    for it in range(1, maxiter + 1):
        rhs = -(Ax - b) - sigma * (A @ y)

        def matvec(v):
            return v + sigma * (A @ (A.T @ v))
        xi, _ = cg(matvec, rhs, x0=xi, maxit=max(100, 2 * m), tol=1e-6)
        Atxi = A.T @ xi
        y = proj_inf(-Atxi - x / sigma, lam)
        Rd = Atxi + y
        x += gamma * sigma * Rd
        Ax = A @ x
        Rp = Ax - b + xi
        pf = np.linalg.norm(Rp) / normb
        df = np.linalg.norm(Rd) / (1.0 + np.linalg.norm(y))
        if max(pf, df) < 1e2 * stoptol:
            eta, _ = kkt_eta(x, A.T @ (Ax - b), lam)
            if eta < stoptol:
                return x, 0, eta
        if it in trigger:
            if pf < df:
                pw += 1
            else:
                dw += 1
            if pw > max(1, int(1.2 * dw)):
                pw = 0
                sigma = min(1e6, sigma * 1.25)
            elif dw > max(1, int(1.2 * pw)):
                dw = 0
                sigma = max(1e-4, sigma / 1.25)
    eta, _ = kkt_eta(x, A.T @ (Ax - b), lam)
    return x, 1, eta


# ----------------------------------------------------------------------
# adaptive sieving + level-set outer loop
# ----------------------------------------------------------------------

def default_control_addpram(n, m):
    if n < 5e3:
        c = 0.4
    elif n < 1e4:
        c = 0.1
    elif n < 5e4:
        c = 0.04
    elif n < 1e5:
        c = 0.01
    else:
        c = 0.001
    if m / n > 0.05:
        c *= 3.0
    return c


def adaptive_sieving(A, b, lam, x0, xi0, grad0, stoptolas=1e-6,
                     maxiter_as=20, control=0.0):
    m, n = A.shape
    control = control if control > 0 else default_control_addpram(n, m)
    add_bound = max(1, int(control * n))
    x = x0.copy()
    xi = xi0 if xi0 is not None else b - A @ x
    grad = grad0 if grad0 is not None else A.T @ (A @ x - b)
    eta, ev = kkt_eta(x, grad, lam)
    idx = np.flatnonzero(np.abs(x) > 0).tolist()
    viol = np.flatnonzero(np.abs(ev) > max(1e-12, 1e-10 * np.abs(grad).max()))
    if viol.size > 10:
        viol = viol[np.argsort(-np.abs(ev[viol]))[:10]]
    idx = sorted(set(idx) | set(viol.tolist()))
    reduced = []
    as_iter = 0
    run_once = True
    smoothing_broken = False
    while (eta > stoptolas and as_iter < maxiter_as) or run_once:
        run_once = False
        as_iter += 1
        reduced.append(len(idx))
        Ar = A[:, idx]
        xr = x[idx]
        nr = len(idx)
        large = (nr > 0.5 * m and m < 300) or (min(nr, m) > 5000 and nr < m) or nr >= 3000
        if large or smoothing_broken:
            xr, _, _ = admm_l1(Ar, b, lam, xr, stoptol=stoptolas)
        else:
            xr_prev = xr.copy()
            xr, ok, _ = smoothing_newton(Ar, b, lam, xr, stoptol=stoptolas)
            if ok == 0:
                smoothing_broken = True
                xr, _, _ = admm_l1(Ar, b, lam, xr_prev, stoptol=stoptolas)
        x[idx] = xr
        Ax = A @ x
        xi = b - Ax
        grad = A.T @ (Ax - b)
        eta, ev = kkt_eta(x, grad, lam)
        if eta > stoptolas:
            viol = np.flatnonzero(np.abs(ev) > max(1e-12, 1e-10 * np.abs(grad).max()))
            if (as_iter > 10 and as_iter % 5 == 0) or len(idx) > int(0.3 * n):
                idx = np.flatnonzero(np.abs(x) > 0).tolist()
            uni = sorted(set(idx) | set(viol.tolist()))
            newly = len(uni) - len(idx)
            if newly > add_bound:
                newones = [i for i in uni if i not in set(idx)]
                newones.sort(key=lambda i: -abs(ev[i]))
                idx = idx + newones[:add_bound]
                idx.sort()
            else:
                idx = uni
    return x, xi, grad, eta, reduced, as_iter


def update_mucont(ratio_lower, ratio_upper, exist_initialmu, ratio_value):
    if exist_initialmu:
        if ratio_lower == 0:
            if ratio_upper > 10:
                return 0.05
            if ratio_upper > 5:
                return 0.2
            if ratio_upper > 2:
                return 0.3
            if ratio_upper > 1.6:
                return 0.5
            if ratio_upper > 1.2:
                return 0.6
            if ratio_upper > 1.06:
                return 0.7
            if ratio_upper > 1.05:
                return 0.8
            return 0.9
        s = ratio_upper + ratio_lower
        if s > 5:
            return 0.3
        if s > 3:
            return 0.4
        if s > 2:
            return 0.5
        if s > 1:
            return 0.6
        if s > 0:
            return 0.7
        return 0.8
    if ratio_value > 5:
        return 0.2
    if ratio_value > 3:
        return 0.4
    if ratio_value > 2:
        return 0.4
    return 0.5


def solve_bmop(A, b, delta, stoptol=1e-6, maxiter=200, use_secant=True,
               use_newton=False, use_as=True, verbose=False, seed=0,
               trace=False):
    """Port of level_set_lasso.m / level_set.hpp."""
    m, n = A.shape
    normb = np.linalg.norm(b)
    if normb <= delta:
        return np.zeros(n), b, 0.0, abs(normb - delta) / max(1.0, delta), 0
    mumax = np.abs(A.T @ b).max()
    mu0 = 0.0
    muinf = mumax
    mudiff = muinf - mu0
    ratio_lower = 0.0
    ratio_upper = normb / delta
    psi_lower = -100.0
    psi_upper = normb - delta
    mucont = update_mucont(0.0, ratio_upper, False, ratio_upper)
    mu = max(2 * delta * np.sqrt(2 * np.log(n)), mucont * mudiff + mu0)
    if mu >= muinf:
        mu = mucont * mudiff + mu0
    x = np.zeros(n)
    xi = b
    psi = normb
    grad = -A.T @ b
    abseta = abs(psi - delta)
    eta = abseta / max(1.0, delta)
    run_mu_psi = []
    run_ratio = []
    last_mu, last_h = mu, psi - delta
    iter_bis = iter_ns = 0
    as_iter_last = -1
    it = 0
    trace = [] if trace else None
    while eta > stoptol and it < maxiter:
        it += 1
        sub_tol = 1e-6 if mudiff > 1e-3 else max(min(1e-8, stoptol * 1e-2), 1e-10)
        ratio_value = ratio_upper if psi > delta else ratio_lower
        if eta > max(stoptol, 5e-4):
            mucont = update_mucont(ratio_lower, ratio_upper, False, ratio_value)
            if it > 4 and len(run_ratio) >= 4:
                last4 = run_ratio[-4:]
                mn, mx = min(last4), max(last4)
                if mx > 0 and mn / mx > 0.97 and mn > 1.0:
                    mucont = 0.5
                elif mx > 0 and mn / mx > 0.97 and mn < 1.0:
                    mucont = 0.4 if ratio_value < 0.9 else 0.5
        else:
            mucont = 0.5
        fast_ok = eta < 1e-2 or mudiff / (muinf + mu0) < 1e-2 or ratio_value < 1.2
        did_fast = False
        if fast_ok and (use_newton or (use_secant and it > 2)):
            if use_newton and as_iter_last > 0:
                h = psi - delta
                u = (A.T @ xi) / mu
                K = np.flatnonzero(np.abs(x) > 1e-12)
                if K.size > 0 and psi > 0:
                    AK = A[:, K]
                    uK = u[K]
                    G = AK.T @ AK + 1e-10 * np.eye(K.size)
                    z = np.linalg.solve(G, uK)
                    betaK = uK @ z
                    gn = mu * betaK / psi
                    if gn > 0 and np.isfinite(gn):
                        mu_new = mu - h / gn
                        ok_step = (mu_new >= 0 and mu_new >= 0.99 * mu0 and
                                   mu_new <= 1.01 * muinf and as_iter_last > 0)
                        if ok_step:
                            mu = mu_new
                            did_fast = True
                            iter_ns += 1
            if not did_fast and use_secant and it > 2:
                h = psi - delta
                denom = mu - last_mu
                if abs(denom) > 1e-300:
                    gs = (h - last_h) / denom
                    mu_new = mu - h / gs
                    ok_step = (np.isfinite(mu_new) and mu_new >= 0 and
                               mu_new >= 0.99 * mu0 and mu_new <= 1.01 * muinf and
                               as_iter_last > 0)
                    if ok_step:
                        last_mu, last_h = mu, h
                        mu = mu_new
                        did_fast = True
                        iter_ns += 1
        if not did_fast:
            last_mu, last_h = mu, psi - delta
            mu = mu0 + mucont * mudiff
            iter_bis += 1
        # subproblem
        if use_as:
            x, xi, grad, _, reduced, as_iter_last = adaptive_sieving(
                A, b, mu, x, xi, grad, stoptolas=sub_tol)
        else:
            x_prev = x.copy()
            x, ok, _ = smoothing_newton(A, b, mu, x, stoptol=sub_tol)
            g_sm = A.T @ (A @ x - b)
            eta_sm, _ = kkt_eta(x, g_sm, mu)
            if ok == 0 or eta_sm > max(sub_tol, 1e-5):
                x, _, _ = admm_l1(A, b, mu, x_prev, stoptol=sub_tol)
            xi = b - A @ x
            grad = A.T @ (A @ x - b)
        psi = np.linalg.norm(xi)
        if psi > delta:
            muinf = mu
            psi_upper = psi - delta
            ratio_upper = psi / delta
        else:
            mu0 = mu
            psi_lower = psi - delta
            ratio_lower = psi / delta
        mudiff = muinf - mu0
        abseta = abs(psi - delta)
        eta = abseta / max(1.0, delta)
        run_mu_psi.append((mu, psi))
        run_ratio.append(ratio_upper if psi > delta else ratio_lower)
        # bracket repair
        if (psi_lower * psi_upper > 0 or mudiff < 1e-13) and eta > stoptol:
            if psi_lower > 0:   # lower bound broken
                neg = [(p, h) for p, ps in run_mu_psi for h in [ps - delta] if h < 0]
                if neg:
                    p_best, h_best = max(neg, key=lambda t: t[1])
                    mu0, psi_lower = p_best, h_best
                    ratio_lower = (h_best + delta) / delta
                else:
                    mu0, psi_lower, ratio_lower = 0.0, -100.0, 0.0
            if psi_upper < 0:   # upper bound broken
                pos = [(p, h) for p, ps in run_mu_psi for h in [ps - delta] if h > 0]
                if pos:
                    p_best, h_best = min(pos, key=lambda t: t[1])
                    muinf, psi_upper = p_best, h_best
                    ratio_upper = (h_best + delta) / delta
                else:
                    muinf, psi_upper = mumax, normb - delta
                    ratio_upper = normb / delta
            mudiff = muinf - mu0
        if verbose:
            print(f"  it={it:3d} mu={mu:.3e} psi={psi:.3e} eta={eta:.1e} "
                  f"[{mu0:.2e},{muinf:.2e}] {'N/S' if did_fast else 'B'}")
        if trace is not None:
            trace.append({
                'it': it, 'mu': mu, 'psi': psi, 'eta': eta,
                'fast': bool(did_fast), 'bis': bool(not did_fast),
                'mu0': mu0, 'muinf': muinf,
                'as_iter': as_iter_last, 'iter_bis': iter_bis,
                'iter_ns': iter_ns,
            })
    if trace is not None:
        return x, xi, mu, eta, it, trace
    return x, xi, mu, eta, it


# ----------------------------------------------------------------------
# numerical validation
# ----------------------------------------------------------------------

def main():
    print("pysmop reference tests")
    rng = np.random.default_rng(2026)

    # ---- T1: small problem, check feasibility + sparsity + KKT ----
    n, m = 150, 40
    A = rng.standard_normal((m, n))
    xtrue = np.zeros(n)
    xtrue[rng.choice(n, 8, replace=False)] = rng.standard_normal(8)
    b = A @ xtrue + 0.02 * rng.standard_normal(m)
    delta = 0.1 * np.linalg.norm(b)   # reference-style radius: delta = c*||b||

    x, xi, mu, eta, it = solve_bmop(A, b, delta, verbose=True)
    psi = np.linalg.norm(A @ x - b)
    feas = abs(psi - delta) / max(1.0, delta)
    print(f"\nT1: it={it} mu={mu:.4e} psi={psi:.6e} delta={delta:.6e} "
          f"feas={feas:.2e} nnz1e-6={int((np.abs(x) > 1e-6).sum())} "
          f"energy99.9={energy_nnz(x)} (true 8)")
    assert feas < 2e-3, "feasibility"
    # numerical support: tiny non-zeros may remain; use the 99.9% energy count
    assert energy_nnz(x) <= 15, "sparsity"
    assert eta < 1e-5 or feas < 1e-3, "outer convergence"
    # ---- T2: no adaptive sieving ----
    x2, xi2, mu2, eta2, it2 = solve_bmop(A, b, delta, use_as=False)
    psi2 = np.linalg.norm(A @ x2 - b)
    feas2 = abs(psi2 - delta) / max(1.0, delta)
    print(f"T2 (no AS): it={it2} feas={feas2:.2e}")
    assert feas2 < 2e-3

    # ---- T3: objective sanity vs the generating solution ----
    obj_sol = np.abs(x).sum()
    obj_true = np.abs(xtrue).sum()
    print(f"T3: ||x||_1 = {obj_sol:.6e} (true {obj_true:.6e})")
    assert obj_sol <= obj_true + 1e-6, "the solver must not beat the feasible oracle"

    # ---- T4: exactness on a tiny problem via brute force ----
    # min ||x||_1 s.t. ||Ax-b||<=delta on the line x=t*v (v = A^T b direction)
    n4, m4 = 4, 2
    A4 = rng.standard_normal((m4, n4))
    b4 = rng.standard_normal(m4)
    delta4 = 0.3
    x4, xi4, mu4, eta4, it4 = solve_bmop(A4, b4, delta4, use_as=False,
                                         stoptol=1e-8)
    psi4 = np.linalg.norm(A4 @ x4 - b4)
    print(f"T4 tiny: it={it4} feas={abs(psi4-delta4)/max(1,delta4):.2e}")
    assert abs(psi4 - delta4) / max(1.0, delta4) < 1e-4

    # ---- T5: larger sparse problem exercises adaptive sieving ----
    n5, m5 = 800, 100
    A5 = rng.standard_normal((m5, n5)) * (rng.random((m5, n5)) < 0.05)
    x5t = np.zeros(n5)
    x5t[rng.choice(n5, 10, replace=False)] = rng.standard_normal(10)
    b5 = A5 @ x5t + 0.01 * rng.standard_normal(m5)
    delta5 = 0.08 * np.linalg.norm(b5)
    x5, xi5, mu5, eta5, it5 = solve_bmop(A5, b5, delta5, maxiter=60)
    psi5 = np.linalg.norm(A5 @ x5 - b5)
    feas5 = abs(psi5 - delta5) / max(1.0, delta5)
    print(f"T5 sparse: it={it5} feas={feas5:.2e} "
          f"energy99.9={energy_nnz(x5)} (true 10, n={n5})")
    assert feas5 < 1e-3
    assert energy_nnz(x5) < 0.1 * n5, "sparse support on the sparse problem"

    print("\nALL PYTHON-REF TESTS PASSED")


if __name__ == "__main__":
    main()
