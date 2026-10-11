"""Noisy diagnosis: does learning from failure survive diagnoses that are
sometimes wrong? (PROJECT_STATUS.md SS27)

SS23 showed failure memory helps when failures are delayed and *diagnostic*,
but its diagnosis was an oracle: when it fired it always named the true wrong
turn. Here, with probability eps it names a random other ancestor of the dead
node instead - possibly a node on the gold path, so trusting it prunes the
only solution. Same environment and trees as SS23 (eps=0 reproduces it).

Policies, each on the same trees and budget:
  <X>             no diagnosis used
  <X>+hard        prune the named subtree on the first diagnosis (SS23's +diag)
  <X>+vote2       prune only once two separate failures name the same node
  Posterior*+soft Bayesian likelihood-ratio down-weighting instead of pruning,
                  with the true eps (calibrated) or a fixed guess eps_hat=0.2
for X in DFS, BestFirst-mean, Posterior-localblame (SS23's best posterior
variant) and Posterior (exact).

Pre-registered hypotheses:
  N1  hard pruning degrades as eps grows and falls below no-diagnosis at
      some break-even eps*.
  N2  the calibrated soft update is never significantly worse than
      no-diagnosis, and beats hard pruning at high eps.
  N3  vote2 moves the break-even eps* higher than hard pruning.
Paired per-tree tests (trees independent), Holm over all tests.
"""

from __future__ import annotations

import csv
import os
import time

import numpy as np
import matplotlib.pyplot as plt

from fags.delayed_failure import DelayedFailureTree, dfs, best_first_mean, posterior_search
from fags.stats import cluster_paired_test, holm

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "v2")
os.makedirs(RESULTS_DIR, exist_ok=True)

TREES = int(os.environ.get("ND_TREES", 300))
EPS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]
RHOS = [0.5, 1.0]
DELAYS = [3, 5]
D_PRIMES = [0.5, 1.0]
H, B = 8, 3
BUDGETS = [100, 400]
REPORT = 100
FIXED_GUESS = 0.2


def policies(d, eps):
    soft_cal = eps if eps > 0 else None  # eps=0: the calibrated soft update is the hard kill
    return {
        "DFS": lambda t, b: dfs(t, b),
        "DFS+hard": lambda t, b: dfs(t, b, True, "hard"),
        "DFS+vote2": lambda t, b: dfs(t, b, True, "vote2"),
        "BestFirst-mean": lambda t, b: best_first_mean(t, b),
        "BestFirst-mean+hard": lambda t, b: best_first_mean(t, b, True, "hard"),
        "BestFirst-mean+vote2": lambda t, b: best_first_mean(t, b, True, "vote2"),
        "PostLB": lambda t, b: posterior_search(t, b, d, local_blame=True),
        "PostLB+hard": lambda t, b: posterior_search(t, b, d, True, local_blame=True),
        "PostLB+soft": lambda t, b: posterior_search(t, b, d, True, local_blame=True, diag_error_hat=soft_cal),
        "PostLB+soft(0.2)": lambda t, b: posterior_search(t, b, d, True, local_blame=True, diag_error_hat=FIXED_GUESS),
        "Post": lambda t, b: posterior_search(t, b, d),
        "Post+hard": lambda t, b: posterior_search(t, b, d, True),
        "Post+soft": lambda t, b: posterior_search(t, b, d, True, diag_error_hat=soft_cal),
    }


TESTS = []
for base in ("DFS", "BestFirst-mean", "PostLB"):
    TESTS += [(f"{base}+hard", base), (f"{base}+vote2" if base != "PostLB" else "PostLB+soft", base)]
TESTS += [("PostLB+soft", "PostLB+hard"), ("PostLB+soft(0.2)", "PostLB"), ("PostLB+soft(0.2)", "PostLB+hard"),
          ("Post+soft", "Post"), ("Post+soft", "Post+hard"),
          ("DFS+vote2", "DFS+hard"), ("BestFirst-mean+vote2", "BestFirst-mean+hard")]


def main():
    t0 = time.time()
    rows, tests = [], []
    succ_at: dict[tuple, dict[str, np.ndarray]] = {}
    for d in D_PRIMES:
        for K in DELAYS:
            for rho in RHOS:
                for eps in EPS:
                    pol = policies(d, eps)
                    costs = {n: [] for n in pol}
                    for i in range(TREES):
                        seed = 7919 * i + 104729 * K + int(1000 * d)  # same trees as SS23
                        tree = DelayedFailureTree(seed, H, B, K, d, diagnosis_rate=rho, diagnosis_error=eps)
                        for n, fn in pol.items():
                            c = fn(tree, max(BUDGETS))
                            costs[n].append(np.inf if c is None else c)
                    succ = {n: {b: (np.asarray(cs) <= b).astype(int) for b in BUDGETS} for n, cs in costs.items()}
                    succ_at[(d, K, rho, eps)] = {n: succ[n][REPORT] for n in pol}
                    for n in pol:
                        rows.append({"d_prime": d, "max_delay": K, "rho": rho, "eps": eps, "method": n,
                                     **{f"success@{b}": float(succ[n][b].mean()) for b in BUDGETS}})
                    for a, b in TESTS:
                        r = cluster_paired_test(succ[b][REPORT], succ[a][REPORT], list(range(TREES)), n_perm=4000, n_boot=400)
                        tests.append({"d_prime": d, "max_delay": K, "rho": rho, "eps": eps, "comparison": f"{a} - {b}", **r})
                    print(f"[{time.time() - t0:6.0f}s] d'={d} K={K} rho={rho} eps={eps}  " + "  ".join(
                        f"{n}={100 * succ[n][REPORT].mean():.0f}" for n in pol), flush=True)

    for t, p in zip(tests, holm([t["p"] for t in tests])):
        t["p_holm"] = p
    for name, data in (("noisy_diagnosis.csv", rows), ("noisy_diagnosis_tests.csv", tests)):
        with open(os.path.join(RESULTS_DIR, name), "w", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, fieldnames=list(data[0].keys())); wr.writeheader(); wr.writerows(data)

    # Figure: gain over no-diagnosis vs eps, for hard / vote2 / soft, rho=0.5.
    fig, axes = plt.subplots(len(D_PRIMES), len(DELAYS), figsize=(12, 8), sharey=True, squeeze=False)
    lines = [("DFS+hard", "DFS", "#d1242f", "-"), ("DFS+vote2", "DFS", "#d1242f", "--"),
             ("BestFirst-mean+hard", "BestFirst-mean", "#bf8700", "-"), ("BestFirst-mean+vote2", "BestFirst-mean", "#bf8700", "--"),
             ("PostLB+hard", "PostLB", "#1f6feb", "-"), ("PostLB+soft", "PostLB", "#1f6feb", ":"),
             ("PostLB+soft(0.2)", "PostLB", "#2da44e", ":")]
    for r_i, d in enumerate(D_PRIMES):
        for c_i, K in enumerate(DELAYS):
            ax = axes[r_i, c_i]
            for a, b, color, ls in lines:
                ys = [100 * (succ_at[(d, K, 0.5, e)][a].mean() - succ_at[(d, K, 0.5, e)][b].mean()) for e in EPS]
                ax.plot(EPS, ys, color=color, ls=ls, marker="o", ms=3, label=f"{a} vs {b}")
            ax.axhline(0, color="black", lw=0.7)
            ax.set_title(f"d'={d}, K={K}, rho=0.5", fontsize=10)
            ax.set_xlabel("diagnosis error rate eps")
            ax.grid(alpha=0.3)
        axes[r_i, 0].set_ylabel(f"gain over no diagnosis @ {REPORT} (pp)")
    axes[0, 0].legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "noisy_diagnosis.png"), dpi=150)
    plt.close()

    names = list(policies(1.0, 0.1))
    with open(os.path.join(RESULTS_DIR, "noisy_diagnosis_summary.txt"), "w", encoding="utf-8") as f:
        f.write(f"NOISY DIAGNOSIS ({TREES} trees/cell, H={H}, b={B}); success @ {REPORT} expansions (%)\n\n")
        for d in D_PRIMES:
            for K in DELAYS:
                for rho in RHOS:
                    f.write(f"d'={d} K={K} rho={rho}\n  eps  " + " ".join(f"{n[:13]:>13}" for n in names) + "\n")
                    for eps in EPS:
                        f.write(f"  {eps:<4} " + " ".join(f"{100 * succ_at[(d, K, rho, eps)][n].mean():13.1f}" for n in names) + "\n")
                    f.write("\n")
        f.write("BREAK-EVEN: smallest eps at which <X>+hard / +vote2 / soft is significantly WORSE than no diagnosis (Holm)\n")
        for d in D_PRIMES:
            for K in DELAYS:
                for rho in RHOS:
                    parts = []
                    for a, b in [("DFS+hard", "DFS"), ("DFS+vote2", "DFS"), ("BestFirst-mean+hard", "BestFirst-mean"),
                                 ("BestFirst-mean+vote2", "BestFirst-mean"), ("PostLB+hard", "PostLB"), ("PostLB+soft", "PostLB"),
                                 ("PostLB+soft(0.2)", "PostLB")]:
                        worse = [t["eps"] for t in tests if t["comparison"] == f"{a} - {b}" and t["d_prime"] == d
                                 and t["max_delay"] == K and t["rho"] == rho and t["p_holm"] < 0.05 and t["diff"] < 0]
                        parts.append(f"{a}: {min(worse) if worse else 'never'}")
                    f.write(f"  d'={d} K={K} rho={rho}: " + "; ".join(parts) + "\n")
        f.write("\nSIGNIFICANT TESTS (Holm p<0.05)\n")
        for t in tests:
            if t["p_holm"] < 0.05:
                f.write(f"  d'={t['d_prime']} K={t['max_delay']} rho={t['rho']} eps={t['eps']:<4} {t['comparison']:<38} "
                        f"diff={100*t['diff']:+6.1f}pp [{100*t['ci_lo']:+.1f},{100*t['ci_hi']:+.1f}]\n")
        f.write(f"\n{len(tests)} tests; runtime {time.time() - t0:.0f}s\n")


if __name__ == "__main__":
    main()
