#!/usr/bin/env python3
# Copyright 2026 Ashwin Kumar Baskaran
# SPDX-License-Identifier: MIT
"""
Classical baselines for a MAX-3-SAT instance, to compare QAOA against.

For each .cnf or .wcnf file you give it, it finds the true optimum by brute
force, then runs three heuristics many times and reports how often each one
reaches that optimum:

  1. Greedy hill-climbing: flip whichever bit helps most, stop when nothing
     does. Run from lots of random starts.
  2. Simulated annealing: a standard cooling schedule, so it can get out of
     local optima.
  3. Random sampling: the floor. It gets the same number of evaluations
     greedy used on average.

Weighted files are run twice, once counting clauses and once counting weight.
Brute force is 2^n, so keep it to instances with about 20 variables or fewer.

    python src/classical_baselines.py instances/Fifteen.cnf

Results print to the screen and go to classical_baseline_summary.csv.
"""
import os, sys, csv, time, math, random, argparse
from collections import defaultdict

# Reading .cnf / .wcnf files
def parse_cnf(path):
    """Return (n_vars, clauses, weights, is_weighted).
    clauses: list of [lit, lit, lit] (DIMACS ints)
    weights: list of float, all 1.0 for CNF
    """
    clauses, weights, n_vars = [], [], 0
    is_weighted = False
    with open(path, encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s[0] in ("c", "%"):
                continue
            if s[0] == "p":
                parts = s.split()
                is_weighted = parts[1].lower() == "wcnf"
                n_vars = int(parts[2])
                continue
            body = s.partition(" c ")[0] if " c " in s else s
            tokens = body.split()
            if not tokens:
                continue
            if is_weighted:
                w = float(tokens[0])
                lits = [int(x) for x in tokens[1:] if x != "0"]
            else:
                w = 1.0
                lits = [int(x) for x in tokens if x != "0"]
            if lits:
                clauses.append(lits)
                weights.append(w)
    return n_vars, clauses, weights, is_weighted

def sat_clause(cl, bits):
    return any((l > 0 and bits[abs(l)-1] == 1) or
               (l < 0 and bits[abs(l)-1] == 0) for l in cl)

def score_unw(clauses, bits):
    return sum(1 for cl in clauses if sat_clause(cl, bits))

def score_w(clauses, weights, bits):
    return sum(weights[j] for j, cl in enumerate(clauses)
               if sat_clause(cl, bits))


# Heuristic 1: greedy hill-climbing (one run)
def greedy_hillclimb(clauses, weights, n_vars, init_bits, weighted=False):
    """Flip the single bit giving largest score improvement; stop at LO.

    Returns: (best_score, best_bits, n_evals, n_iters).
    Each iteration evaluates n_vars neighbours = n_vars+1 evals counting
    current.
    """
    bits = init_bits[:]
    score_fn = (lambda b: score_w(clauses, weights, b)) if weighted \
               else (lambda b: score_unw(clauses, b))
    best_score = score_fn(bits)
    n_evals  = 1
    n_iters  = 0
    improved = True
    while improved:
        improved = False
        n_iters += 1
        best_delta = 0.0
        best_flip  = -1
        for i in range(n_vars):
            bits[i] ^= 1
            ns = score_fn(bits)
            n_evals += 1
            d = ns - best_score
            if d > best_delta:
                best_delta = d
                best_flip  = i
                best_ns    = ns
            bits[i] ^= 1
        if best_flip >= 0:
            bits[best_flip] ^= 1
            best_score = best_ns
            improved = True
    return best_score, bits, n_evals, n_iters


# Heuristic 2: simulated annealing (one run)
def simulated_annealing(clauses, weights, n_vars, init_bits,
                         T0=2.0, T_min=0.01, alpha=0.97, max_iters=2000,
                         weighted=False, rng=None):
    rng = rng or random.Random()
    bits = init_bits[:]
    score_fn = (lambda b: score_w(clauses, weights, b)) if weighted \
               else (lambda b: score_unw(clauses, b))
    cur = score_fn(bits); n_evals = 1
    best_score = cur; best_bits = bits[:]
    T = T0
    it = 0
    while T > T_min and it < max_iters:
        i = rng.randrange(n_vars)
        bits[i] ^= 1
        ns = score_fn(bits); n_evals += 1
        d  = ns - cur
        if d > 0 or rng.random() < math.exp(d / T):
            cur = ns
            if cur > best_score:
                best_score = cur; best_bits = bits[:]
        else:
            bits[i] ^= 1   # revert
        T *= alpha
        it += 1
    return best_score, best_bits, n_evals, it


# Heuristic 3: random sampling
def random_sampling(clauses, weights, n_vars, n_samples, weighted=False,
                     rng=None):
    rng = rng or random.Random()
    score_fn = (lambda b: score_w(clauses, weights, b)) if weighted \
               else (lambda b: score_unw(clauses, b))
    best_score = 0; best_bits = [0] * n_vars
    for _ in range(n_samples):
        bits = [rng.randrange(2) for _ in range(n_vars)]
        s = score_fn(bits)
        if s > best_score:
            best_score = s; best_bits = bits
    return best_score, best_bits, n_samples, 1


# Brute force, for the true optimum
def brute_force(clauses, weights, n_vars, weighted=False):
    best_unw, best_w = 0, 0.0
    n_unw_opt, n_w_opt = 0, 0
    for mask in range(1 << n_vars):
        bits = [(mask >> i) & 1 for i in range(n_vars)]
        unw = score_unw(clauses, bits)
        wsc = score_w(clauses, weights, bits)
        if unw > best_unw:
            best_unw = unw; n_unw_opt = 1
        elif unw == best_unw:
            n_unw_opt += 1
        if wsc > best_w:
            best_w = wsc; n_w_opt = 1
        elif abs(wsc - best_w) < 1e-9:
            n_w_opt += 1
    return best_unw, best_w, n_unw_opt, n_w_opt


# Runs every method on one instance and compares them
def benchmark(label, cnf_path, weighted_mode=False, K=100, seed=42):
    print(f"\n  {'='*70}")
    print(f"  Benchmark: {label}  (file={cnf_path}, weighted={weighted_mode})")
    print(f"  {'='*70}")
    n_vars, clauses, weights, is_w = parse_cnf(cnf_path)
    n_cl = len(clauses)
    total_w = sum(weights)
    opt_unw, opt_w, n_unw_opt, n_w_opt = brute_force(clauses, weights,
                                                      n_vars, weighted_mode)
    opt = opt_w if weighted_mode else opt_unw
    opt_count = n_w_opt if weighted_mode else n_unw_opt
    label_unit = "weight" if weighted_mode else "clauses"

    print(f"    n_vars={n_vars}  n_clauses={n_cl}  "
          f"total_weight={total_w:.0f}")
    print(f"    Brute-force OPT (this mode): {opt:g} {label_unit}  "
          f"({opt_count}/{1<<n_vars} optima)")
    print(f"    Optima rate: {100*opt_count/(1<<n_vars):.1f}%")

    # All-zeros, all-ones reference
    zeros = [0]*n_vars; ones = [1]*n_vars
    score_fn = (lambda b: score_w(clauses, weights, b)) if weighted_mode \
               else (lambda b: score_unw(clauses, b))
    print(f"\n    All-zeros : {score_fn(zeros):g} / {opt:g}  "
          f"(alpha={score_fn(zeros)/opt:.4f})")
    print(f"    All-ones  : {score_fn(ones):g} / {opt:g}  "
          f"(alpha={score_fn(ones)/opt:.4f})")

    results = []
    rng = random.Random(seed)

    # Greedy hill-climbing from K random starts
    hits, alphas, evals, times = 0, [], [], []
    t0 = time.time()
    for _ in range(K):
        init = [rng.randrange(2) for _ in range(n_vars)]
        st = time.time()
        s, _, nev, _ = greedy_hillclimb(clauses, weights, n_vars, init,
                                         weighted=weighted_mode)
        times.append(time.time() - st)
        alphas.append(s / opt)
        evals.append(nev)
        if abs(s - opt) < 1e-9:
            hits += 1
    tot = time.time() - t0
    results.append({
        "method": "Greedy hill-climb",
        "K": K, "success_rate": hits/K,
        "mean_alpha": sum(alphas)/K, "min_alpha": min(alphas),
        "max_alpha": max(alphas),
        "mean_evals": sum(evals)/K, "max_evals": max(evals),
        "mean_time_ms": 1000*sum(times)/K, "total_time_s": tot,
    })

    # Simulated annealing, K runs
    hits, alphas, evals, times = 0, [], [], []
    t0 = time.time()
    for k in range(K):
        init = [rng.randrange(2) for _ in range(n_vars)]
        st = time.time()
        s, _, nev, _ = simulated_annealing(clauses, weights, n_vars, init,
                                            weighted=weighted_mode,
                                            rng=random.Random(seed*1000 + k))
        times.append(time.time() - st)
        alphas.append(s / opt)
        evals.append(nev)
        if abs(s - opt) < 1e-9:
            hits += 1
    tot = time.time() - t0
    results.append({
        "method": "Simulated annealing",
        "K": K, "success_rate": hits/K,
        "mean_alpha": sum(alphas)/K, "min_alpha": min(alphas),
        "max_alpha": max(alphas),
        "mean_evals": sum(evals)/K, "max_evals": max(evals),
        "mean_time_ms": 1000*sum(times)/K, "total_time_s": tot,
    })

    # Random sampling, one run
    # Same number of evaluations greedy used, so the comparison is fair
    avg_greedy_evals = int(results[0]["mean_evals"])
    t0 = time.time()
    s, _, _, _ = random_sampling(clauses, weights, n_vars, avg_greedy_evals,
                                  weighted=weighted_mode,
                                  rng=random.Random(seed))
    tot = time.time() - t0
    results.append({
        "method": f"Random sampling ({avg_greedy_evals} samples)",
        "K": 1, "success_rate": (1.0 if abs(s - opt) < 1e-9 else 0.0),
        "mean_alpha": s/opt, "min_alpha": s/opt, "max_alpha": s/opt,
        "mean_evals": avg_greedy_evals, "max_evals": avg_greedy_evals,
        "mean_time_ms": 1000*tot, "total_time_s": tot,
    })

    # Print results table
    print(f"\n    {'Method':<35} {'K':>4} {'succ%':>6} "
          f"{'meanA':>7} {'minA':>6} {'meanEv':>7} {'ms/run':>7}")
    print(f"    {'-'*35} {'-'*4} {'-'*6} {'-'*7} {'-'*6} {'-'*7} {'-'*7}")
    for r in results:
        print(f"    {r['method']:<35} {r['K']:>4} "
              f"{100*r['success_rate']:>5.1f}% "
              f"{r['mean_alpha']:>7.4f} {r['min_alpha']:>6.3f} "
              f"{r['mean_evals']:>7.0f} {r['mean_time_ms']:>7.2f}")
    return label, results, opt


# Command line
if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Classical baselines for one or more .cnf / .wcnf files.")
    ap.add_argument("cnf_files", nargs="+", help="DIMACS .cnf or .wcnf file(s)")
    ap.add_argument("--runs", type=int, default=1000,
                    help="runs per heuristic (default 1000)")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed (default 42)")
    ap.add_argument("--out", default="classical_baseline_summary.csv",
                    help="CSV to write (default classical_baseline_summary.csv)")
    args = ap.parse_args()

    all_results = []
    for path in args.cnf_files:
        is_weighted = parse_cnf(path)[3]
        name = os.path.basename(path)
        modes = [False, True] if is_weighted else [False]
        for weighted_mode in modes:
            label = f"{name} ({'weighted' if weighted_mode else 'unweighted'})"
            lab, res, opt = benchmark(label, path, weighted_mode,
                                      K=args.runs, seed=args.seed)
            for r in res:
                r["instance"] = lab
                r["opt"]      = opt
            all_results.extend(res)

    cols = ["instance","method","K","opt","success_rate",
            "mean_alpha","min_alpha","max_alpha",
            "mean_evals","max_evals","mean_time_ms","total_time_s"]
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in all_results:
            w.writerow({k: r.get(k, "") for k in cols})
    print(f"\n  Summary written -> {args.out}")
