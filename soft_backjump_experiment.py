"""Soft backjumping: defer, don't delete (PROJECT_STATUS.md SS30).

SS27 found that pruning on every diagnosis (hard) is harmful once ~5-20% of
diagnoses are wrong; a two-vote rule and a Bayesian soft update fix that, but
the Bayesian update rides on the posterior ordering, which is weak with a
poor verifier (d'=0.5). This tests a third option that keeps depth-first /
best-first ordering: on a diagnosis, *defer* the named subtree behind
everything else instead of deleting it.

  defer       defer on every diagnosis
  vote_defer  defer on the first diagnosis of a node, prune on the second

Same environment, trees, grid and budget as SS27 (noisy_diagnosis_experiment.py).

Pre-registered hypotheses:
  S1  eps = 0: defer costs exactly what hard pruning costs (checked on 1200
      runs before this experiment; re-checked here).
  S2  defer is never significantly worse than using no diagnosis.
  S3  vote_defer is at least as good as both vote2 and defer (no significant
      losses to either).
  S4  in the weak-verifier regime (d'=0.5) the best defer variant beats the
      Bayesian PostLB+soft from SS27.
Per-tree paired tests, Holm over all tests.
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
TREES = int(os.environ.get("SB_TREES", 300))
EPS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]
RHOS = [0.5, 1.0]
DELAYS = [3, 5]
D_PRIMES = [0.5, 1.0]
H, B = 8, 3
REPORT = 100
MODES = ["hard", "vote2", "defer", "vote_defer"]


def policies(d, eps):
    pol = {}
    for name, fn in (("DFS", dfs), ("BestFirst-mean", best_first_mean)):
        pol[name] = (lambda f: lambda t: f(t, REPORT))(fn)
        for m in MODES:
            pol[f"{name}+{m}"] = (lambda f, m: lambda t: f(t, REPORT, True, m))(fn, m)
    pol["PostLB+soft"] = lambda t: posterior_search(t, REPORT, d, True, local_blame=True,
                                                    diag_error_hat=eps if eps > 0 else None)
    return pol


def tests_for():
    out = []
    for base in ("DFS", "BestFirst-mean"):
        out += [(f"{base}+defer", base, "S2"), (f"{base}+vote_defer", f"{base}+vote2", "S3"),
                (f"{base}+vote_defer", f"{base}+defer", "S3"), (f"{base}+defer", f"{base}+hard", "vs hard"),
                (f"{base}+vote_defer", base, "vs none")]
    out += [("DFS+vote_defer", "PostLB+soft", "S4"), ("DFS+defer", "PostLB+soft", "S4")]
    return out


def main():
    t0 = time.time()
    rows, tests, s1_violations = [], [], 0
    succ_at = {}
    for d in D_PRIMES:
        for K in DELAYS:
            for rho in RHOS:
                for eps in EPS:
                    pol = policies(d, eps)
                    res = {n: [] for n in pol}
                    for i in range(TREES):
                        seed = 7919 * i + 104729 * K + int(1000 * d)  # same trees as SS23/SS27
                        tree = DelayedFailureTree(seed, H, B, K, d, diagnosis_rate=rho, diagnosis_error=eps)
                        out = {n: fn(tree) for n, fn in pol.items()}
                        for n, c in out.items():
                            res[n].append(int(c is not None))
                        if eps == 0.0:
                            s1_violations += out["DFS+hard"] != out["DFS+defer"]
                            s1_violations += out["BestFirst-mean+hard"] != out["BestFirst-mean+defer"]
                    succ = {n: np.asarray(v) for n, v in res.items()}
                    succ_at[(d, K, rho, eps)] = succ
                    for n in pol:
                        rows.append({"d_prime": d, "max_delay": K, "rho": rho, "eps": eps, "method": n,
                                     f"success@{REPORT}": float(succ[n].mean())})
                    for a, b, h in tests_for():
                        r = cluster_paired_test(succ[b], succ[a], list(range(TREES)), n_perm=4000, n_boot=400)
                        tests.append({"hyp": h, "d_prime": d, "max_delay": K, "rho": rho, "eps": eps,
                                      "comparison": f"{a} - {b}", **r})
                    print(f"[{time.time() - t0:6.0f}s] d'={d} K={K} rho={rho} eps={eps}  " + "  ".join(
                        f"{n}={100 * succ[n].mean():.0f}" for n in pol), flush=True)

    for t, p in zip(tests, holm([t["p"] for t in tests])):
        t["p_holm"] = p
    for name, data in (("soft_backjump.csv", rows), ("soft_backjump_tests.csv", tests)):
        with open(os.path.join(RESULTS_DIR, name), "w", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, fieldnames=list(data[0].keys())); wr.writeheader(); wr.writerows(data)

    fig, axes = plt.subplots(len(D_PRIMES), len(DELAYS), figsize=(12, 8), sharey=True, squeeze=False)
    styles = {"DFS": ("#6e7781", "-"), "DFS+hard": ("#d1242f", "-"), "DFS+vote2": ("#d1242f", "--"),
              "DFS+defer": ("#1f6feb", "-"), "DFS+vote_defer": ("#1f6feb", "--"), "PostLB+soft": ("#2da44e", ":")}
    for r_i, d in enumerate(D_PRIMES):
        for c_i, K in enumerate(DELAYS):
            ax = axes[r_i, c_i]
            for n, (color, ls) in styles.items():
                ax.plot(EPS, [100 * succ_at[(d, K, 0.5, e)][n].mean() for e in EPS], color=color, ls=ls, marker="o", ms=3, label=n)
            ax.set_title(f"d'={d}, K={K}, rho=0.5", fontsize=10)
            ax.set_xlabel("diagnosis error rate eps")
            ax.grid(alpha=0.3)
        axes[r_i, 0].set_ylabel(f"success within {REPORT} expansions (%)")
    axes[0, 0].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "soft_backjump.png"), dpi=150)
    plt.close()

    with open(os.path.join(RESULTS_DIR, "soft_backjump_summary.txt"), "w", encoding="utf-8") as f:
        f.write(f"SOFT BACKJUMPING ({TREES} trees/cell; success @ {REPORT} expansions, %)\n")
        f.write(f"S1 check (eps=0, defer cost == hard cost): {s1_violations} differences\n\n")
        names = list(policies(1.0, 0.1))
        for d in D_PRIMES:
            for K in DELAYS:
                for rho in RHOS:
                    f.write(f"d'={d} K={K} rho={rho}\n  eps  " + " ".join(f"{n[:14]:>14}" for n in names) + "\n")
                    for eps in EPS:
                        f.write(f"  {eps:<4} " + " ".join(f"{100 * succ_at[(d, K, rho, eps)][n].mean():14.1f}" for n in names) + "\n")
                    f.write("\n")
        f.write("TALLY over cells with eps > 0 (Holm p<0.05)\n")
        for a, b, h in tests_for():
            g = [t for t in tests if t["comparison"] == f"{a} - {b}" and t["eps"] > 0]
            w = sum(t["p_holm"] < 0.05 and t["diff"] > 0 for t in g)
            l = sum(t["p_holm"] < 0.05 and t["diff"] < 0 for t in g)
            f.write(f"  [{h:<7}] {a:<26} - {b:<22} wins {w:2d}  losses {l:2d}  of {len(g)}\n")
        f.write(f"\n{len(tests)} tests; runtime {time.time() - t0:.0f}s\n")
    print(open(os.path.join(RESULTS_DIR, "soft_backjump_summary.txt"), encoding="utf-8").read()[-2500:])


if __name__ == "__main__":
    main()
