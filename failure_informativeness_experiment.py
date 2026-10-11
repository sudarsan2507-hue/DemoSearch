"""When does learning from failure pay off? (PROJECT_STATUS.md SS23)

Sweeps failure delay K, failure diagnosticity rho and verifier
discriminability d' in the environment of fags/delayed_failure.py, running
every policy on the SAME trees with the SAME expansion budget.

Pre-registered hypotheses (from the factorized-prior argument in
fags/delayed_failure.py):
  H1  rho = 0: using diagnosis changes nothing (sanity: identical runs).
  H2  the gain from using diagnosis grows with both rho and K.
  H3  local blame (renormalising among the dead node's siblings) falls
      behind exact posterior elimination as K grows.
  H4  every systematic search with memory beats memoryless restarts.
Trees are independent, so paired tests are per tree; Holm over all tests.
"""

from __future__ import annotations

import csv
import os
import time

import numpy as np
import matplotlib.pyplot as plt

from fags.delayed_failure import (
    DelayedFailureTree, dfs, fags_style, best_first_mean, random_restarts, beam, posterior_search,
)
from fags.stats import cluster_paired_test, holm

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "v2")
os.makedirs(RESULTS_DIR, exist_ok=True)

TREES = int(os.environ.get("FI_TREES", 300))
DELAYS = [0, 1, 2, 3, 4, 5]
RHOS = [0.0, 0.5, 1.0]
D_PRIMES = [0.5, 1.0, 2.0]
H, B = 8, 3
BUDGETS = [25, 50, 100, 200, 400]
REPORT = 100
BEAM_WIDTHS = [1, 2, 3, 5, 8, 13]


def policies(d):
    return {
        "Restarts": lambda t, b: random_restarts(t, b, d, seed=t.seed),
        "DFS": lambda t, b: dfs(t, b),
        "DFS+diag": lambda t, b: dfs(t, b, use_diagnosis=True),
        "FAGS-style": lambda t, b: fags_style(t, b),
        "FAGS-style+diag": lambda t, b: fags_style(t, b, use_diagnosis=True),
        "BestFirst-mean": lambda t, b: best_first_mean(t, b),
        "BestFirst-mean+diag": lambda t, b: best_first_mean(t, b, use_diagnosis=True),
        "Posterior": lambda t, b: posterior_search(t, b, d),
        "Posterior+diag": lambda t, b: posterior_search(t, b, d, use_diagnosis=True),
        "Posterior-localblame": lambda t, b: posterior_search(t, b, d, local_blame=True),
    }


TESTS = [  # (a, b): report a - b
    ("DFS+diag", "DFS"), ("FAGS-style+diag", "FAGS-style"), ("BestFirst-mean+diag", "BestFirst-mean"),
    ("Posterior+diag", "Posterior"), ("Posterior", "Posterior-localblame"),
    ("FAGS-style", "Restarts"), ("FAGS-style", "BestFirst-mean"), ("BestFirst-mean", "Posterior"),
]


def main():
    t0 = time.time()
    rows, tests, beam_rows = [], [], []
    succ_at: dict[tuple, dict[str, np.ndarray]] = {}
    for d in D_PRIMES:
        for rho in RHOS:
            for K in DELAYS:
                pol = policies(d)
                costs = {n: [] for n in pol}
                bok = {w: [] for w in BEAM_WIDTHS}; bcost = {w: [] for w in BEAM_WIDTHS}
                bfm_at_beam = {w: [] for w in BEAM_WIDTHS}
                for i in range(TREES):
                    seed = 7919 * i + 104729 * K + int(1000 * d)  # same trees across rho
                    tree = DelayedFailureTree(seed, H, B, K, d, diagnosis_rate=rho)
                    for n, fn in pol.items():
                        c = fn(tree, max(BUDGETS))
                        costs[n].append(np.inf if c is None else c)
                    if rho == 0.0:
                        for w in BEAM_WIDTHS:
                            ok, c = beam(tree, w, d)
                            bok[w].append(int(ok)); bcost[w].append(c)
                            bfm_at_beam[w].append(int(best_first_mean(tree, c) is not None))
                succ = {n: {b: (np.asarray(cs) <= b).astype(int) for b in BUDGETS} for n, cs in costs.items()}
                succ_at[(d, rho, K)] = {n: succ[n][REPORT] for n in pol}
                for n, cs in costs.items():
                    row = {"d_prime": d, "rho": rho, "max_delay": K, "method": n,
                           "median_cost": float(np.median(cs))}
                    row.update({f"success@{b}": float(succ[n][b].mean()) for b in BUDGETS})
                    rows.append(row)
                cl = list(range(TREES))
                for a, b in TESTS:
                    r = cluster_paired_test(succ[b][REPORT], succ[a][REPORT], cl, n_perm=4000, n_boot=400)
                    tests.append({"d_prime": d, "rho": rho, "max_delay": K, "comparison": f"{a} - {b}", **r})
                if rho == 0.0:
                    for w in BEAM_WIDTHS:
                        r = cluster_paired_test(bok[w], bfm_at_beam[w], cl, n_perm=4000, n_boot=400)
                        beam_rows.append({"d_prime": d, "max_delay": K, "width": w, "mean_cost": float(np.mean(bcost[w])),
                                          "beam_success": float(np.mean(bok[w])),
                                          "bestfirst_mean_same_cost": float(np.mean(bfm_at_beam[w]))})
                        tests.append({"d_prime": d, "rho": rho, "max_delay": K,
                                      "comparison": f"BestFirst-mean - Beam-{w} (same cost)", **r})
                print(f"[{time.time() - t0:6.0f}s] d'={d} rho={rho} K={K}  " + "  ".join(
                    f"{n}={100 * succ[n][REPORT].mean():.0f}" for n in pol), flush=True)

    for t, p in zip(tests, holm([t["p"] for t in tests])):
        t["p_holm"] = p

    def dump(name, data):
        with open(os.path.join(RESULTS_DIR, name), "w", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, fieldnames=list(data[0].keys())); wr.writeheader(); wr.writerows(data)
    dump("failure_informativeness.csv", rows)
    dump("failure_informativeness_beam.csv", beam_rows)
    dump("failure_informativeness_tests.csv", tests)

    # Figure: gain from using diagnosis (Posterior+diag - Posterior and
    # DFS+diag - DFS) vs delay, one line per rho, panels per d'.
    fig, axes = plt.subplots(2, len(D_PRIMES), figsize=(15, 8), sharey="row")
    for col, d in enumerate(D_PRIMES):
        for row_i, (a, b) in enumerate((("Posterior+diag", "Posterior"), ("DFS+diag", "DFS"))):
            ax = axes[row_i, col]
            for rho, color in zip(RHOS, ("#6e7781", "#1f6feb", "#d1242f")):
                ys = [100 * (succ_at[(d, rho, K)][a].mean() - succ_at[(d, rho, K)][b].mean()) for K in DELAYS]
                ax.plot(DELAYS, ys, marker="o", color=color, label=f"rho={rho}")
            ax.axhline(0, color="black", lw=0.6)
            ax.set_title(f"{a} minus {b}, d'={d}", fontsize=10)
            ax.set_xlabel("max failure delay K")
            ax.grid(alpha=0.3)
        axes[0, col].legend(fontsize=8)
    axes[0, 0].set_ylabel(f"success gain @ {REPORT} expansions (pp)")
    axes[1, 0].set_ylabel(f"success gain @ {REPORT} expansions (pp)")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "failure_informativeness.png"), dpi=150)
    plt.close()

    names = list(policies(1.0))
    with open(os.path.join(RESULTS_DIR, "failure_informativeness_summary.txt"), "w", encoding="utf-8") as f:
        f.write(f"FAILURE INFORMATIVENESS SWEEP ({TREES} trees/cell, H={H}, b={B}); success @ {REPORT} expansions (%)\n\n")
        for d in D_PRIMES:
            for rho in RHOS:
                f.write(f"d'={d} rho={rho}\n  K  " + " ".join(f"{n[:14]:>14}" for n in names) + "\n")
                for K in DELAYS:
                    f.write(f"  {K}  " + " ".join(f"{100 * succ_at[(d, rho, K)][n].mean():14.1f}" for n in names) + "\n")
                f.write("\n")
        f.write("SIGNIFICANT TESTS (Holm p<0.05)\n")
        for t in tests:
            if t["p_holm"] < 0.05:
                f.write(f"  d'={t['d_prime']} rho={t['rho']} K={t['max_delay']}  {t['comparison']:<40} "
                        f"diff={100*t['diff']:+6.1f}pp [{100*t['ci_lo']:+.1f},{100*t['ci_hi']:+.1f}]\n")
        f.write(f"\n{len(tests)} tests total; runtime {time.time() - t0:.0f}s\n")


if __name__ == "__main__":
    main()
