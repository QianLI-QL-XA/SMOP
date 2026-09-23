# -*- coding: utf-8 -*-
"""Level-set method (BMOP) end-to-end study.

Runs the reference implementation (pysmop_ref) over a matrix of problem
scales / noise-radii and over the three outer-loop variants (secant,
Newton, bisection), then prints a comparison table and saves every trace
to study_results/ for plotting.

    python tests/level_set_study.py
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pysmop_ref as m


def make_problem(rng, n, mm, k, noise):
    A = rng.standard_normal((mm, n))
    xtrue = np.zeros(n)
    xtrue[rng.choice(n, k, replace=False)] = rng.standard_normal(k)
    b = A @ xtrue + noise * rng.standard_normal(mm)
    return A, b, xtrue


def run_case(name, A, b, delta, xtrue, **kw):
    t0 = time.time()
    x, xi, mu, eta, it, trace = m.solve_bmop(A, b, delta, trace=True, **kw)
    psi = np.linalg.norm(xi)
    feas = abs(psi - delta) / max(1.0, delta)
    obj = float(np.abs(x).sum())
    obj_true = float(np.abs(xtrue).sum())
    en = m.energy_nnz(x)
    k_true = int((np.abs(xtrue) > 1e-10).sum())
    n_bis = sum(1 for t in trace if t["bis"])
    n_ns = sum(1 for t in trace if t["fast"])
    max_as = max((t["as_iter"] for t in trace), default=-1)
    return dict(name=name, n=A.shape[1], m=A.shape[0], delta=float(delta),
                it=it, bis=n_bis, ns=n_ns, feas=feas, obj=obj,
                obj_true=obj_true, energy=en, k_true=k_true,
                max_as=max_as, time=time.time() - t0, mu=float(mu),
                trace=trace)


def main():
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "study_results")
    os.makedirs(out_dir, exist_ok=True)

    rng = np.random.default_rng(777)
    results = []

    # ---- scale x radius matrix (default: secant + AS + smoothing) ----
    cases = [
        ("S1 small c=0.1", 150, 40, 8, 0.02, 0.1),
        ("S2 small c=0.01", 150, 40, 8, 0.02, 0.01),
        ("S3 small c=0.3", 150, 40, 8, 0.02, 0.3),
        ("S4 mid c=0.1", 400, 80, 15, 0.02, 0.1),
        ("S5 big c=0.1", 800, 100, 20, 0.02, 0.1),
    ]
    for name, n, mm, k, noise, c in cases:
        A, b, xtrue = make_problem(rng, n, mm, k, noise)
        delta = c * np.linalg.norm(b)
        print(f"[run] {name} (n={n}, m={mm}, delta={delta:.4f})", flush=True)
        results.append(run_case(name, A, b, delta, xtrue))

    # ---- outer-loop variant comparison on the small problem ----
    A, b, xtrue = make_problem(rng, 150, 40, 8, 0.02)
    delta = 0.1 * np.linalg.norm(b)
    for label, kw in [
        ("C1 secant+AS", {}),
        ("C2 newton+AS", {"use_secant": False, "use_newton": True}),
        ("C3 bisection", {"use_secant": False, "use_newton": False}),
    ]:
        print(f"[run] {label}", flush=True)
        results.append(run_case(label, A, b, delta, xtrue, **kw))

    # ---- save ----
    with open(os.path.join(out_dir, "study.json"), "w", encoding="utf-8") as f:
        json.dump([{k: v for k, v in r.items() if k != "trace"} for r in results],
                  f, ensure_ascii=False, indent=1)
    for r in results:
        with open(os.path.join(out_dir, f"trace_{r['name'].replace(' ','_')}.json"),
                  "w", encoding="utf-8") as f:
            json.dump(r["trace"], f)

    # ---- summary table ----
    print("\n" + "=" * 108)
    hdr = f"{'case':<20}{'n':>5}{'m':>4}{'delta':>10}{'it':>4}{'B':>4}{'N/S':>5}" \
          f"{'feas':>11}{'|x|1':>9}{'true|1|':>9}{'en99':>6}{'k*':>5}{'maxAS':>7}{'sec':>8}"
    print(hdr)
    print("-" * 108)
    for r in results:
        print(f"{r['name']:<20}{r['n']:>5}{r['m']:>4}{r['delta']:>10.4f}"
              f"{r['it']:>4}{r['bis']:>4}{r['ns']:>5}{r['feas']:>11.2e}"
              f"{r['obj']:>9.4f}{r['obj_true']:>9.4f}{r['energy']:>6}"
              f"{r['k_true']:>5}{r['max_as']:>7}{r['time']:>8.1f}")
    print("=" * 108)
    print("results saved to tests/study_results/")


if __name__ == "__main__":
    main()
