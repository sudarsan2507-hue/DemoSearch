"""Real-task test of the failure-informativeness theory: Sudoku with forward
checking (PROJECT_STATUS.md SS28).

Unlike SS23/SS27, nothing about failures is injected: the verifier is the
least-constraining-value heuristic, failures are domain wipe-outs, the
diagnosis is the conflict set (sound, as in FC-CBJ), and failure delay is
varied through a natural, standard design choice - variable ordering:
MRV (fail-first: mistakes surface quickly) vs static row-major order
(mistakes surface many assignments later). Every policy is the unchanged
implementation from fags/delayed_failure.py, run on the same search tree.

Puzzles: 200 from sapientinc/sudoku-extreme test split, stratified by its
backtrack rating (4 bins x 50), plus Norvig's top95.

Pre-registered predictions (from SS23):
  R1  FAGS-style revival (no diagnosis) is not better than DFS or
      BestFirst-mean - non-diagnostic failures cannot re-rank the frontier.
  R2  Using the conflict-set diagnosis never hurts any systematic policy.
  R3  The diagnosis gain is larger under static ordering (longer natural
      failure delay) than under MRV.
  R4  Memoryless restarts are far worse than any systematic search.
Puzzles are independent, so paired tests are per puzzle; Holm over all.
"""

from __future__ import annotations

import csv
import math
import os
import random
import time

import numpy as np

from fags.sudoku_env import SudokuTree
from fags.delayed_failure import dfs, fags_style, best_first_mean, random_restarts
from fags.stats import cluster_bootstrap_ci, cluster_paired_test, holm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "sudoku")
RESULTS_DIR = os.path.join(ROOT, "results", "v2")
os.makedirs(RESULTS_DIR, exist_ok=True)

PER_BIN = int(os.environ.get("SUDOKU_PER_BIN", 50))
RATING_BINS = [(0, 5), (6, 20), (21, 50), (51, 10_000)]
BUDGET = int(os.environ.get("SUDOKU_BUDGET", 20000))
BUDGETS = [500, 2000, 5000, BUDGET]
ORDERINGS = ["mrv", "static"]
D_HAT = 0.5  # softmax temperature for restarts (scores are -#values eliminated)

POLICIES = {
    "Restarts": lambda t: random_restarts(t, BUDGET, D_HAT, seed=t.seed),
    "DFS": lambda t: dfs(t, BUDGET),
    "DFS+CBJ": lambda t: dfs(t, BUDGET, True),
    "FAGS-style": lambda t: fags_style(t, BUDGET),
    "FAGS-style+CBJ": lambda t: fags_style(t, BUDGET, True),
    "BestFirst-mean": lambda t: best_first_mean(t, BUDGET),
    "BestFirst-mean+CBJ": lambda t: best_first_mean(t, BUDGET, True),
}
DIAG_PAIRS = [("DFS+CBJ", "DFS"), ("FAGS-style+CBJ", "FAGS-style"), ("BestFirst-mean+CBJ", "BestFirst-mean")]
OTHER_PAIRS = [("FAGS-style", "DFS"), ("FAGS-style", "BestFirst-mean"), ("DFS", "Restarts")]


def load_puzzles():
    rows = list(csv.DictReader(open(os.path.join(DATA, "extreme_test.csv"), encoding="utf-8")))
    rng = random.Random(2026)
    puzzles = []
    for lo, hi in RATING_BINS:
        pool = [r for r in rows if lo <= int(r["rating"]) <= hi]
        for j, r in enumerate(rng.sample(pool, PER_BIN)):
            puzzles.append({"id": f"extreme:r{lo}-{hi}:{j}", "set": f"extreme r{lo}-{hi}",
                            "question": r["question"], "answer": r["answer"]})
    with open(os.path.join(DATA, "top95.txt"), encoding="utf-8") as f:
        for i, line in enumerate(l.strip() for l in f if l.strip()):
            puzzles.append({"id": f"top95:{i}", "set": "top95", "question": line, "answer": None})
    return puzzles


def main():
    t0 = time.time()
    puzzles = load_puzzles()
    cost = {(o, p): [] for o in ORDERINGS for p in POLICIES}
    jumps = {o: [] for o in ORDERINGS}
    rows = []
    for k, pz in enumerate(puzzles):
        for order in ORDERINGS:
            cache: dict = {}
            for name, fn in POLICIES.items():
                tree = SudokuTree(pz["question"], seed=k, cache=cache, ordering=order)
                c = fn(tree)
                cost[(order, name)].append(math.inf if c is None else c)
                if name == "DFS+CBJ":
                    jumps[order].append(float(np.mean(tree.jump_distances)) if tree.jump_distances else 0.0)
                if c is not None and pz["answer"]:
                    goal = next(p for p in cache if p != "base" and tree.is_goal(p))
                    assert "".join(map(str, tree.solution(goal))) == pz["answer"], (pz["id"], order, name)
                rows.append({"puzzle": pz["id"], "set": pz["set"], "ordering": order, "method": name,
                             "cost": "" if c is None else c})
            del cache
        if k % 25 == 0:
            print(f"[{time.time() - t0:6.0f}s] puzzle {k + 1}/{len(puzzles)}", flush=True)

    n = len(puzzles)
    clusters = list(range(n))
    summary, tests = [], []
    for order in ORDERINGS:
        summary.append(f"-- ordering = {order} ({n} puzzles, budget {BUDGET}) --")
        for name in POLICIES:
            cs = np.asarray(cost[(order, name)])
            parts = []
            for b in BUDGETS:
                m, lo, hi = cluster_bootstrap_ci((cs <= b).astype(float), clusters)
                parts.append(f"@{b}={100 * m:5.1f}%")
            med = np.median(cs)
            summary.append(f"  {name:<19} " + " ".join(parts) + f"   median cost={'>' + str(BUDGET) if math.isinf(med) else int(med)}")
        summary.append(f"  mean CBJ jump distance (assignments skipped per backjump): {np.mean(jumps[order]):.2f}")
        for a, b in DIAG_PAIRS + OTHER_PAIRS:
            ca, cb = np.asarray(cost[(order, a)]), np.asarray(cost[(order, b)])
            # success at the full budget, and log-cost with unsolved runs censored at the budget
            r1 = cluster_paired_test((cb <= BUDGET).astype(float), (ca <= BUDGET).astype(float), clusters)
            la, lb = np.log(np.minimum(ca, BUDGET)), np.log(np.minimum(cb, BUDGET))
            r2 = cluster_paired_test(lb, la, clusters)
            tests.append({"ordering": order, "comparison": f"{a} - {b}", "metric": f"success@{BUDGET}", **r1})
            tests.append({"ordering": order, "comparison": f"{a} - {b}", "metric": "log_cost (censored)", **r2})
    # R3: is the diagnosis gain bigger under static ordering? Per-puzzle
    # log-cost reduction from CBJ, compared across orderings (paired by puzzle).
    for a, b in DIAG_PAIRS:
        gain = {o: np.log(np.minimum(np.asarray(cost[(o, b)]), BUDGET)) - np.log(np.minimum(np.asarray(cost[(o, a)]), BUDGET))
                for o in ORDERINGS}
        r = cluster_paired_test(gain["mrv"], gain["static"], clusters)
        tests.append({"ordering": "static vs mrv", "comparison": f"gain of {a} over {b}", "metric": "log_cost reduction", **r})
        summary.append(f"  R3 {a}: mean log-cost reduction mrv={gain['mrv'].mean():.3f} (x{math.exp(gain['mrv'].mean()):.2f}), "
                       f"static={gain['static'].mean():.3f} (x{math.exp(gain['static'].mean()):.2f})")

    for t, p in zip(tests, holm([t["p"] for t in tests])):
        t["p_holm"] = p
    for name, data in (("sudoku_runs.csv", rows), ("sudoku_tests.csv", tests)):
        with open(os.path.join(RESULTS_DIR, name), "w", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, fieldnames=list(data[0].keys())); wr.writeheader(); wr.writerows(data)
    with open(os.path.join(RESULTS_DIR, "sudoku_summary.txt"), "w", encoding="utf-8") as f:
        f.write("SUDOKU (real task): FC search, LCV verifier, conflict-set diagnosis\n\n")
        f.write("\n".join(summary) + "\n\nTESTS (per-puzzle paired sign-flip, Holm)\n")
        for t in tests:
            f.write(f"  {t['ordering']:<14} {t['comparison']:<38} {t['metric']:<20} diff={t['diff']:+.3f} "
                    f"[{t['ci_lo']:+.3f},{t['ci_hi']:+.3f}] p_holm={t['p_holm']:.3g}\n")
        f.write(f"\nruntime {time.time() - t0:.0f}s\n")
    print(open(os.path.join(RESULTS_DIR, "sudoku_summary.txt"), encoding="utf-8").read())


if __name__ == "__main__":
    main()
