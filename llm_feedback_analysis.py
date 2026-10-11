"""Analysis for llm_feedback_experiment.py (PROJECT_STATUS.md SS29).

The submitted program after k calls is the first one that passes the
visible tests within the first k attempts, else the k-th (or last) attempt.
Success = that program passes the hidden EvalPlus tests. Only problems
whose first attempt failed a visible test can differ between conditions,
so paired tests are run on that subset (clustered per problem, Holm), and
overall pass rates are also reported over all problems.

Usage: python llm_feedback_analysis.py [model_tag ...]   (default: every runs file found)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from fags.stats import cluster_bootstrap_ci, cluster_paired_test, holm
from llm_feedback_experiment import CONDITIONS, MAX_ATTEMPTS

OUT_DIR = Path(__file__).resolve().parent / "results" / "v2"
PAIRS = [  # (a, b): a - b ; hypothesis label
    ("binary", "restart", "L1"), ("diagnostic", "restart", "L2"), ("diagnostic", "binary", "L2"),
    ("noisy_hard", "diagnostic", "L3"), ("noisy_hard", "restart", "L3"),
    ("noisy_soft", "noisy_hard", "L4"), ("noisy_soft", "restart", "L4"), ("self_diag", "binary", "L5"),
]


def submitted(attempts: list[dict], k: int) -> dict:
    for a in attempts[:k]:
        if a["visible_pass"]:
            return a
    return attempts[:k][-1]


def analyse(tag: str) -> list[str]:
    first = {r["task_id"]: r for r in map(json.loads, (OUT_DIR / f"llm_feedback_first_{tag}.jsonl").open(encoding="utf-8"))}
    runs: dict[int, dict[str, dict]] = {}
    for r in map(json.loads, (OUT_DIR / f"llm_feedback_runs_{tag}.jsonl").open(encoding="utf-8")):
        runs.setdefault(r["task_id"], {})[r["condition"]] = r
    complete = sorted(t for t, d in runs.items() if all(c in d for c in CONDITIONS))
    n_all = len(first)
    n_fail = sum(not r["visible_pass"] for r in first.values())
    out = [f"== {tag} ==",
           f"problems with a first attempt: {n_all}; first-attempt hidden pass {100 * np.mean([r['hidden_pass'] for r in first.values()]):.1f}%; "
           f"first attempt failed a visible test: {n_fail}; repair runs complete for {len(complete)} of them"]
    if not complete:
        return out
    succ = {c: {k: np.array([int(submitted(runs[t][c]["attempts"], k)["hidden_pass"]) for t in complete])
                for k in range(1, MAX_ATTEMPTS + 1)} for c in CONDITIONS}
    vis = {c: np.array([int(submitted(runs[t][c]["attempts"], MAX_ATTEMPTS)["visible_pass"]) for t in complete]) for c in CONDITIONS}
    calls = {c: np.mean([len(runs[t][c]["attempts"]) for t in complete]) for c in CONDITIONS}
    base_ok = sum(int(r["hidden_pass"]) for t, r in first.items() if r["visible_pass"])

    out.append(f"\nOn the {len(complete)} repaired problems (hidden-test success of the submitted program):")
    out.append(f"  {'condition':<11} " + " ".join(f"@{k} calls" for k in range(1, MAX_ATTEMPTS + 1))
               + "   mean calls  visible-pass  false-confidence  | overall (all problems)")
    clusters = complete
    for c in CONDITIONS:
        m, lo, hi = cluster_bootstrap_ci(succ[c][MAX_ATTEMPTS], clusters)
        fc = np.mean((vis[c] == 1) & (succ[c][MAX_ATTEMPTS] == 0))
        overall = (base_ok + succ[c][MAX_ATTEMPTS].sum()) / (n_all - n_fail + len(complete))
        out.append(f"  {c:<11} " + " ".join(f"{100 * succ[c][k].mean():6.1f}% " for k in range(1, MAX_ATTEMPTS + 1))
                   + f"  [{100 * lo:4.1f},{100 * hi:4.1f}]  {calls[c]:5.2f}      {100 * vis[c].mean():5.1f}%         {100 * fc:5.1f}%"
                   + f"        | {100 * overall:5.1f}%")
    nb = np.mean([runs[t]["noisy_hard"]["corrupted_reports"] for t in complete])
    out.append(f"  mean corrupted test reports per noisy_hard run: {nb:.2f}")
    tests = []
    for a, b, h in PAIRS:
        r = cluster_paired_test(succ[b][MAX_ATTEMPTS], succ[a][MAX_ATTEMPTS], clusters)
        tests.append((h, a, b, r))
    adj = holm([r["p"] for *_, r in tests])
    out.append("\nPaired tests on success@%d (per-problem sign-flip, Holm over %d):" % (MAX_ATTEMPTS, len(tests)))
    for (h, a, b, r), p in zip(tests, adj):
        out.append(f"  {h}  {a:<11} - {b:<11} diff={100 * r['diff']:+6.1f}pp [{100 * r['ci_lo']:+6.1f},{100 * r['ci_hi']:+6.1f}]  "
                   f"p={r['p']:.4f}  p_holm={p:.4f}")
    return out


def main():
    tags = sys.argv[1:] or sorted(p.stem.removeprefix("llm_feedback_runs_") for p in OUT_DIR.glob("llm_feedback_runs_*.jsonl"))
    lines = ["LLM FAILURE-FEEDBACK EXPERIMENT (MBPP+, hidden EvalPlus tests)", ""]
    for t in tags:
        lines += analyse(t) + [""]
    text = "\n".join(lines)
    (OUT_DIR / "llm_feedback_summary.txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
