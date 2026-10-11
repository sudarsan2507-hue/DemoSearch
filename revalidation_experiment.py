"""Re-validation of the core FAGS / beam / best-first / MCTS conclusions on the
fixed generator (PROJECT_STATUS.md SS21).

Every result in SS4-18 was produced while the generator made every 3+ hop
gold path contradict itself, so the gold path was unreachable in ~75-98% of
queries. This reruns the comparisons those sections rest on, under the SAME
goal-test protocol they used (success = the search touches answer_node), so
any change is attributable to the fix alone. Methodology upgrades over the
originals:
  * 3 independent graph seeds instead of 1
  * cluster-level paired permutation tests (cluster = graph seed x gold
    path), since queries cycle over gold paths (Small: 40 paths/1000 queries)
  * Holm correction across every test reported
  * one legacy-generator run per size to confirm the old numbers reproduce
"""

from __future__ import annotations

import csv
import os
import sys
import time

import numpy as np

from fags.graph_generator import generate_dataset
from fags.verifier import Verifier
from fags.memory import create_memory
from fags.baseline_search import baseline_search
from fags.failure_search import failure_search
from fags.beam_search import beam_search
from fags.best_first_search import best_first_search
from fags.mcts_search import mcts_search
from fags.stats import cluster_bootstrap_ci, cluster_paired_test, holm
from budget_matched_control_experiment import random_restart_baseline

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "v2")
os.makedirs(RESULTS_DIR, exist_ok=True)

SIZES = {"Small": 20, "Medium": 100, "Large": 1000}
SEEDS = [101, 202, 303]
QUERIES_PER_SEED = int(os.environ.get("REVAL_QUERIES", 500))
MAX_DEPTH = 6
BEAM_WIDTHS = [2, 3, 5, 8]
MATCHED_WIDTHS = [3, 8]  # best-first / MCTS / RRB get these beams' per-query budgets


def rrb_by_calls(graph, query, verifier, target_calls: int, max_depth: int = MAX_DEPTH, max_restarts: int = 200):
    """Random-restart greedy, budget-matched on verifier calls (edges scored)
    rather than nodes_visited. Returns (success, calls_spent)."""
    calls = 0
    for _ in range(max_restarts):
        r = baseline_search(graph, query, verifier, max_depth=max_depth)
        calls += r.edges_explored
        if r.success:
            return True, calls, r
        if calls >= target_calls:
            break
    return False, calls, r


def on_gold(res, q) -> bool:
    """Did a successful search reach the answer along the gold path itself?"""
    return res.success and list(res.path) == list(q.gold_path)


def run_condition(size_label: str, n_nodes: int, seed: int, legacy: bool) -> dict[str, list]:
    graph, queries = generate_dataset(n_nodes, QUERIES_PER_SEED, seed=seed, legacy_evidence_bug=legacy)
    v = Verifier(noise_std=0.08, seed=seed)
    out: dict[str, list] = {"cluster": [(seed, q.gold_path_id) for q in queries]}

    def record(name, res, q, success=None, calls=None):
        out.setdefault(name, []).append(int(res.success if success is None else success))
        out.setdefault(name + ":nodes", []).append(res.nodes_visited)
        out.setdefault(name + ":calls", []).append(res.edges_explored if calls is None else calls)
        out.setdefault(name + ":ongold", []).append(int(on_gold(res, q) and (success is None or success)))

    def record_rrb_calls(name, q, target):
        ok, calls, last = rrb_by_calls(graph, q, v, target)
        record(name, last, q, success=ok, calls=calls)

    BIG = 10 ** 9
    for q in queries:
        record("Baseline", baseline_search(graph, q, v, max_depth=MAX_DEPTH), q)
        fr = failure_search(graph, q, v, create_memory("threshold", threshold=0.10),
                            max_depth=MAX_DEPTH, max_backtracks=3, enable_re_verification=True)
        record("FAGS", fr, q)
        record("RRB@FAGS", random_restart_baseline(graph, q, v, max(fr.nodes_visited, 1), MAX_DEPTH), q)
        if legacy:
            record("Beam-5", beam_search(graph, q, v, beam_width=5, max_depth=MAX_DEPTH), q)
            continue
        record_rrb_calls("RRB@FAGS/calls", q, max(fr.edges_explored, 1))
        for w in BEAM_WIDTHS:
            br = beam_search(graph, q, v, beam_width=w, max_depth=MAX_DEPTH)
            record(f"Beam-{w}", br, q)
            if w in MATCHED_WIDTHS:
                # Legacy accounting (SS15-16): match on nodes_visited.
                budget = max(br.nodes_visited, 2)
                record(f"BestFirst@{w}", best_first_search(graph, q, v, node_budget=budget, max_depth=MAX_DEPTH), q)
                record(f"MCTS@{w}", mcts_search(graph, q, v, node_budget=budget, max_depth=MAX_DEPTH), q)
                record(f"RRB@Beam-{w}", random_restart_baseline(graph, q, v, budget, MAX_DEPTH), q)
                # Like-for-like accounting: match on verifier calls.
                calls = max(br.edges_explored, 1)
                record(f"BestFirst@{w}/calls", best_first_search(graph, q, v, node_budget=BIG, max_depth=MAX_DEPTH, score_budget=calls), q)
                record(f"MCTS@{w}/calls", mcts_search(graph, q, v, node_budget=BIG, max_depth=MAX_DEPTH, score_budget=calls), q)
                record_rrb_calls(f"RRB@Beam-{w}/calls", q, calls)
    return out


def merge(dst: dict, src: dict):
    for k, vals in src.items():
        dst.setdefault(k, []).extend(vals)


def main():
    t0 = time.time()
    rows, tests = [], []
    for size_label, n in SIZES.items():
        legacy = run_condition(size_label, n, SEEDS[0], legacy=True)
        fixed: dict[str, list] = {}
        for s in SEEDS:
            merge(fixed, run_condition(size_label, n, s, legacy=False))
            print(f"[{time.time() - t0:7.0f}s] {size_label} seed={s} done", flush=True)

        for gen_label, data in (("legacy (seed 101)", legacy), ("fixed (3 seeds)", fixed)):
            for name in [k for k in data if ":" not in k and k != "cluster"]:
                m, lo, hi = cluster_bootstrap_ci(data[name], data["cluster"])
                rows.append({
                    "graph": size_label, "generator": gen_label, "method": name,
                    "acc": m, "ci_lo": lo, "ci_hi": hi,
                    "mean_nodes": float(np.mean(data[name + ":nodes"])),
                    "mean_verifier_calls": float(np.mean(data[name + ":calls"])),
                    "success_on_gold_path": float(np.mean(data[name + ":ongold"])),
                    "n_queries": len(data[name]), "n_clusters": len(set(data["cluster"])),
                })

        c = fixed["cluster"]
        pairs = [("FAGS", "Baseline"), ("FAGS", "RRB@FAGS"), ("FAGS", "RRB@FAGS/calls")]
        for w in MATCHED_WIDTHS:
            for suffix in ("", "/calls"):
                pairs += [(f"Beam-{w}", f"RRB@Beam-{w}{suffix}"), (f"Beam-{w}", f"BestFirst@{w}{suffix}"),
                          (f"Beam-{w}", f"MCTS@{w}{suffix}")]
        pairs += [("Beam-3", "FAGS"), ("Beam-8", "FAGS")]
        for a, b in pairs:
            r = cluster_paired_test(fixed[b], fixed[a], c)  # diff = a - b
            tests.append({"graph": size_label, "comparison": f"{a} - {b}", **r})

    adj = holm([t["p"] for t in tests])
    for t, p in zip(tests, adj):
        t["p_holm"] = p

    with open(os.path.join(RESULTS_DIR, "revalidation_accuracy.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    with open(os.path.join(RESULTS_DIR, "revalidation_tests.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(tests[0].keys()))
        w.writeheader(); w.writerows(tests)

    with open(os.path.join(RESULTS_DIR, "revalidation_summary.txt"), "w", encoding="utf-8") as f:
        f.write("RE-VALIDATION ON THE FIXED GENERATOR (goal-test protocol, as in SS4-18)\n")
        f.write(f"{len(SEEDS)} graph seeds x {QUERIES_PER_SEED} queries per size; CIs and tests clustered on (seed, gold path)\n\n")
        for size_label in SIZES:
            f.write(f"-- {size_label} --\n")
            for r in rows:
                if r["graph"] == size_label:
                    f.write(f"  {r['generator']:<18} {r['method']:<14} acc={100*r['acc']:5.1f}% "
                            f"[{100*r['ci_lo']:5.1f},{100*r['ci_hi']:5.1f}]  nodes={r['mean_nodes']:6.1f}  "
                            f"calls={r['mean_verifier_calls']:6.1f}  "
                            f"via-gold-path={100*r['success_on_gold_path']:5.1f}%\n")
            for t in tests:
                if t["graph"] == size_label:
                    f.write(f"  TEST {t['comparison']:<24} diff={100*t['diff']:+6.1f}pp "
                            f"[{100*t['ci_lo']:+6.1f},{100*t['ci_hi']:+6.1f}]  p={t['p']:.4f}  p_holm={t['p_holm']:.4f}\n")
            f.write("\n")
        f.write(f"runtime {time.time() - t0:.0f}s\n")
    print(open(os.path.join(RESULTS_DIR, "revalidation_summary.txt"), encoding="utf-8").read())


if __name__ == "__main__":
    sys.exit(main())
