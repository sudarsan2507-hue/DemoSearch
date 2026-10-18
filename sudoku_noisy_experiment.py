"""Soft backjumping on a real task: Sudoku with noisy conflict-set
diagnoses (PROJECT_STATUS.md SS31).

SS30 found deferring a diagnosed subtree (instead of deleting it) is the most
robust way to act on noisy diagnoses, but only on synthetic trees. Here the
search trees, verifier and failures are real Sudoku (SS28); with probability
eps the culprit handed to the search is replaced by a random other ancestor
of the failure, which may lie on the solution path. eps = 0 is not rerun:
SS28 has those numbers and defer == hard there by construction (SS30, S1).

Pre-registered:
  T1  hard pruning ends up significantly worse than using no diagnosis at
      some eps (it can delete the solution).
  T2  defer is never significantly worse than using no diagnosis.
  T3  defer beats hard pruning at every eps > 0.
  T4  (exploratory) defer vs vote2.
Same 295 puzzles as SS28; per-puzzle paired tests, Holm over all tests.
"""

from __future__ import annotations

import csv
import math
import os
import time

import numpy as np

from fags.sudoku_env import SudokuTree
from fags.delayed_failure import dfs, best_first_mean
from fags.stats import cluster_paired_test, holm
from sudoku_experiment import load_puzzles, ORDERINGS, RESULTS_DIR

BUDGET = int(os.environ.get("SUDOKU_BUDGET", 20000))
EPS = [0.1, 0.3, 0.5]
MODES = ["hard", "vote2", "defer"]
SEARCHES = {"DFS": dfs, "BestFirst-mean": best_first_mean}


def main():
    t0 = time.time()
    puzzles = load_puzzles()
    if os.environ.get("SUDOKU_LIMIT"):
        puzzles = puzzles[: int(os.environ["SUDOKU_LIMIT"])]
    cost: dict[tuple, list] = {}
    for k, pz in enumerate(puzzles):
        for order in ORDERINGS:
            cache: dict = {}
            for sname, fn in SEARCHES.items():
                c = fn(SudokuTree(pz["question"], seed=k, cache=cache, ordering=order), BUDGET)
                cost.setdefault((order, sname, "none", 0.0), []).append(math.inf if c is None else c)
                for eps in EPS:
                    for mode in MODES:
                        tree = SudokuTree(pz["question"], seed=k, cache=cache, ordering=order, diagnosis_error=eps)
                        c = fn(tree, BUDGET, True, mode)
                        cost.setdefault((order, sname, mode, eps), []).append(math.inf if c is None else c)
            del cache
        if k % 25 == 0:
            print(f"[{time.time() - t0:6.0f}s] puzzle {k + 1}/{len(puzzles)}", flush=True)

    n = len(puzzles)
    cl = list(range(n))
    rows, tests = [], []
    for (order, sname, mode, eps), cs in cost.items():
        cs = np.asarray(cs)
        rows.append({"ordering": order, "search": sname, "mode": mode, "eps": eps,
                     f"success@{BUDGET}": float((cs <= BUDGET).mean()), "median_cost": float(np.median(cs))})
    for order in ORDERINGS:
        for sname in SEARCHES:
            none = (np.asarray(cost[(order, sname, "none", 0.0)]) <= BUDGET).astype(float)
            for eps in EPS:
                s = {m: (np.asarray(cost[(order, sname, m, eps)]) <= BUDGET).astype(float) for m in MODES}
                for a, b, h in (("hard", None, "T1"), ("defer", None, "T2"), ("defer", "hard", "T3"), ("defer", "vote2", "T4")):
                    base = none if b is None else s[b]
                    r = cluster_paired_test(base, s[a], cl, n_perm=4000, n_boot=400)
                    tests.append({"hyp": h, "ordering": order, "search": sname, "eps": eps,
                                  "comparison": f"{a} - {b or 'none'}", **r})
    for t, p in zip(tests, holm([t["p"] for t in tests])):
        t["p_holm"] = p
    for name, data in (("sudoku_noisy.csv", rows), ("sudoku_noisy_tests.csv", tests)):
        with open(os.path.join(RESULTS_DIR, name), "w", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, fieldnames=list(data[0].keys())); wr.writeheader(); wr.writerows(data)
    with open(os.path.join(RESULTS_DIR, "sudoku_noisy_summary.txt"), "w", encoding="utf-8") as f:
        f.write(f"SUDOKU WITH NOISY CONFLICT-SET DIAGNOSES ({n} puzzles, budget {BUDGET}); success %\n\n")
        for order in ORDERINGS:
            for sname in SEARCHES:
                none = np.mean(np.asarray(cost[(order, sname, "none", 0.0)]) <= BUDGET)
                f.write(f"{order:<6} {sname:<15} no diagnosis {100 * none:5.1f}%\n")
                for eps in EPS:
                    f.write(f"         eps={eps:<4} " + "  ".join(
                        f"{m}={100 * np.mean(np.asarray(cost[(order, sname, m, eps)]) <= BUDGET):5.1f}%" for m in MODES) + "\n")
        f.write("\nTESTS (success, per-puzzle sign-flip, Holm)\n")
        for t in tests:
            f.write(f"  {t['hyp']} {t['ordering']:<6} {t['search']:<15} eps={t['eps']:<4} {t['comparison']:<14} "
                    f"diff={100 * t['diff']:+6.1f}pp  p_holm={t['p_holm']:.4g}\n")
        f.write(f"\nruntime {time.time() - t0:.0f}s\n")
    print(open(os.path.join(RESULTS_DIR, "sudoku_noisy_summary.txt"), encoding="utf-8").read()[:2500])


if __name__ == "__main__":
    main()
