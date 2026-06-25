"""Budget-Matched Random-Restart Control, repeated with a stronger verifier.

budget_matched_control_experiment.py showed that FAGS only beats a dumb,
budget-matched random-restart baseline on the Small graph when using the
weak rule-based Verifier; Medium/Large were ties or losses. Open question:
is that because the rule-based verifier's signal is too noisy for FAGS's
targeted memory to exploit? This script reruns the identical control with
HybridVerifier (rule-based + BAAI/bge-small-en-v1.5 embeddings), which has
materially better discriminative power, to see if a stronger verifier
changes the verdict.

Single graph size (500 nodes / 500 queries), matching the scale used by
the repo's other embedding-based experiments (hybrid_sweep_experiment.py,
bge_scale_experiment.py) since real model inference is far slower than the
synthetic rule-based scoring.
"""

from __future__ import annotations

import os

os.environ["HF_HUB_OFFLINE"] = "1"

import csv
import time
import matplotlib.pyplot as plt
import numpy as np

from fags import FailureType, KnowledgeGraph, Query, SearchResult
from fags.graph_generator import generate_dataset
from fags.verifier import HybridVerifier
from fags.memory import create_memory
from fags.baseline_search import baseline_search
from fags.failure_search import failure_search
from fags.evaluation import evaluate_results

RESULTS_DIR = r"d:\Projects\DemoSearch\results"
os.makedirs(RESULTS_DIR, exist_ok=True)

MAX_DEPTH = 6
MAX_BACKTRACKS = 3
MAX_RESTARTS = 40


def random_restart_baseline(
    graph: KnowledgeGraph,
    query: Query,
    verifier,
    target_budget: int,
    max_depth: int = MAX_DEPTH,
    max_restarts: int = MAX_RESTARTS,
) -> SearchResult:
    """Same control as budget_matched_control_experiment.py: re-run plain
    greedy baseline_search until ~target_budget nodes are spent, OR-ing
    success across restarts."""
    t0 = time.perf_counter()
    total_nodes = 0
    total_edges = 0
    restarts = 0
    success = False
    last_res = None

    while restarts < max_restarts:
        res = baseline_search(graph, query, verifier, max_depth=max_depth)
        total_nodes += res.nodes_visited
        total_edges += res.edges_explored
        restarts += 1
        last_res = res
        if res.success:
            success = True
            break
        if total_nodes >= target_budget:
            break

    elapsed = time.perf_counter() - t0
    return SearchResult(
        query_id=query.id,
        success=success,
        path=last_res.path if last_res else [query.start_node],
        nodes_visited=total_nodes,
        search_depth=last_res.search_depth if last_res else 0,
        runtime=elapsed,
        failure_type=FailureType.NONE if success else FailureType.BUDGET_EXHAUSTED,
        backtracks=restarts - 1,
        edges_explored=total_edges,
    )


def main():
    num_nodes = 500
    query_count = 500
    seed = 42

    print(f"Generating Medium KG ({num_nodes} nodes) and {query_count} queries...")
    graph, queries = generate_dataset(num_nodes=num_nodes, num_queries=query_count, seed=seed)

    print("Loading HybridVerifier (BAAI/bge-small-en-v1.5 + rule-based)...")
    verifier = HybridVerifier(model_name="BAAI/bge-small-en-v1.5", alpha=0.5, noise_std=0.30, seed=seed)
    fags_memory = create_memory("threshold", threshold=0.10)

    baseline_results: list[SearchResult] = []
    fags_results: list[SearchResult] = []
    rrb_results: list[SearchResult] = []

    for i, q in enumerate(queries):
        if i % 100 == 0:
            print(f"  query {i}/{query_count}...")

        fags_res = failure_search(
            graph=graph, query=q, verifier=verifier, memory=fags_memory,
            max_depth=MAX_DEPTH, max_backtracks=MAX_BACKTRACKS, enable_re_verification=True,
        )
        fags_results.append(fags_res)

        base_res = baseline_search(graph, q, verifier, max_depth=MAX_DEPTH)
        baseline_results.append(base_res)

        rrb_res = random_restart_baseline(
            graph, q, verifier, target_budget=max(fags_res.nodes_visited, 1), max_depth=MAX_DEPTH,
        )
        rrb_results.append(rrb_res)

    vs_rrb = evaluate_results(rrb_results, fags_results, "FAGS vs Budget-Matched RRB (HybridVerifier)")

    acc_baseline = float(np.mean([1 if r.success else 0 for r in baseline_results]))
    acc_rrb = float(np.mean([1 if r.success else 0 for r in rrb_results]))
    acc_fags = float(np.mean([1 if r.success else 0 for r in fags_results]))
    mean_nodes_fags = float(np.mean([r.nodes_visited for r in fags_results]))
    mean_nodes_rrb = float(np.mean([r.nodes_visited for r in rrb_results]))

    csv_path = os.path.join(RESULTS_DIR, "budget_matched_control_embedding.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Verifier", "Baseline Acc (1x cost)", "Random-Restart Baseline Acc (budget-matched)",
            "FAGS Acc", "FAGS Mean Nodes", "RRB Mean Nodes", "FAGS - RRB Acc Gain", "p-value (FAGS vs RRB)",
        ])
        writer.writerow([
            "HybridVerifier (rule+BGE)", f"{acc_baseline:.2%}", f"{acc_rrb:.2%}",
            f"{acc_fags:.2%}", f"{mean_nodes_fags:.2f}", f"{mean_nodes_rrb:.2f}",
            f"{vs_rrb['accuracy_gain']:.2%}", f"{vs_rrb['p_value_accuracy']:.5e}",
        ])
    print(f"\nTable written to {csv_path}")

    plt.figure(figsize=(6, 6))
    labels = ["Baseline\n(1x cost)", "Random-Restart\nBaseline\n(budget-matched)", "FAGS\n(Threshold t=0.10)"]
    accs = [acc_baseline * 100, acc_rrb * 100, acc_fags * 100]
    plt.bar(labels, accs, color=["black", "darkorange", "crimson"])
    plt.ylabel("Accuracy (%)")
    plt.title("FAGS vs Budget-Matched Control (HybridVerifier)")
    plt.grid(axis="y", linestyle=":", alpha=0.6)
    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, "budget_matched_control_embedding.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Plot written to {plot_path}")

    summary_path = os.path.join(RESULTS_DIR, "budget_matched_control_embedding_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as rf:
        rf.write("=" * 50 + "\n")
        rf.write("BUDGET-MATCHED CONTROL WITH HYBRIDVERIFIER - SUMMARY\n")
        rf.write("=" * 50 + "\n\n")
        rf.write(
            "Repeats budget_matched_control_experiment.py's control (does FAGS\n"
            "beat a dumb baseline given the SAME node-visit budget?) but swaps the\n"
            "weak rule-based Verifier for HybridVerifier (rule-based + BGE\n"
            "embeddings, alpha=0.5), to test whether a stronger verifier signal\n"
            "changes the verdict. 500-node graph, 500 queries, seed=42.\n\n"
        )
        rf.write(f"Baseline (1x cost) accuracy:              {acc_baseline:.2%}\n")
        rf.write(f"Random-Restart Baseline (budget-matched): {acc_rrb:.2%} (mean nodes={mean_nodes_rrb:.2f})\n")
        rf.write(f"FAGS accuracy:                            {acc_fags:.2%} (mean nodes={mean_nodes_fags:.2f})\n")
        rf.write(f"FAGS - RRB accuracy gain:                 {vs_rrb['accuracy_gain']:+.2%}\n")
        rf.write(f"p-value (FAGS vs RRB, paired t-test):     {vs_rrb['p_value_accuracy']:.5e}\n\n")

        rf.write("CONCLUSION\n----------\n")
        if vs_rrb["p_value_accuracy"] < 0.05 and vs_rrb["accuracy_gain"] > 0:
            rf.write(
                "With a stronger verifier, FAGS beats the budget-matched control with\n"
                "statistical significance: a better verifier signal does let targeted\n"
                "failure-memory revival add value beyond raw extra compute.\n"
            )
        else:
            rf.write(
                "Even with a stronger verifier, FAGS does NOT clear the budget-matched\n"
                "random-restart control with statistical significance. The verifier's\n"
                "discriminative power alone is not what was limiting FAGS - the\n"
                "underlying recovery mechanism itself does not reliably beat spending\n"
                "the same compute on dumb retries.\n"
            )

    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
