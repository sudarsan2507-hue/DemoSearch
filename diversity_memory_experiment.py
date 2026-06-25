"""Diversity Memory experiment.

Tests the hypothesis behind fags/memory.py's new DiversityMemory: FAGS's
existing memory strategies (Top1/Top2/Threshold) always revive the
highest-scoring rejected candidate, but distractor edges are deliberately
confusable with the gold relation - so the highest-scoring reject is often
just another guess from the same confusable cluster as the winner, not a
genuinely different hypothesis. DiversityMemory instead skips candidates
that are the same as, confusable with, or highly coherent with the winning
relation, forcing revival onto a structurally different relation family.

The real bar (set by budget_matched_control_experiment.py /
budget_matched_control_embedding_experiment.py): does this beat a
budget-matched random-restart control, where Top1Memory/ThresholdMemory
did not? Same 3 graph sizes / 1000 queries / seed=101 as the canonical
run, for direct comparability with prior results.
"""

from __future__ import annotations

import os
import csv
import time
import matplotlib.pyplot as plt
import numpy as np

from fags import FailureType, KnowledgeGraph, Query, SearchResult
from fags.graph_generator import generate_dataset
from fags.verifier import Verifier
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
    verifier: Verifier,
    target_budget: int,
    max_depth: int = MAX_DEPTH,
    max_restarts: int = MAX_RESTARTS,
) -> SearchResult:
    """Same control as budget_matched_control_experiment.py."""
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


def run_size(size_label: str, num_nodes: int, query_count: int, seed: int) -> dict:
    print(f"\n--- {size_label} graph ({num_nodes} nodes), {query_count} queries ---")
    graph, queries = generate_dataset(num_nodes=num_nodes, num_queries=query_count, seed=seed)
    verifier = Verifier(noise_std=0.08, seed=seed)

    top1_memory = create_memory("top1")
    diversity_memory = create_memory("diversity", coherence_threshold=0.5)

    baseline_results, fags_top1_results, fags_div_results, rrb_results = [], [], [], []

    for q in queries:
        baseline_results.append(baseline_search(graph, q, verifier, max_depth=MAX_DEPTH))

        fags_top1_results.append(failure_search(
            graph=graph, query=q, verifier=verifier, memory=top1_memory,
            max_depth=MAX_DEPTH, max_backtracks=MAX_BACKTRACKS, enable_re_verification=True,
        ))

        div_res = failure_search(
            graph=graph, query=q, verifier=verifier, memory=diversity_memory,
            max_depth=MAX_DEPTH, max_backtracks=MAX_BACKTRACKS, enable_re_verification=True,
        )
        fags_div_results.append(div_res)

        rrb_results.append(random_restart_baseline(
            graph, q, verifier, target_budget=max(div_res.nodes_visited, 1), max_depth=MAX_DEPTH,
        ))

    vs_baseline = evaluate_results(baseline_results, fags_div_results, f"{size_label}: Diversity vs 1x Baseline")
    vs_top1 = evaluate_results(fags_top1_results, fags_div_results, f"{size_label}: Diversity vs Top1")
    vs_rrb = evaluate_results(rrb_results, fags_div_results, f"{size_label}: Diversity vs Budget-Matched RRB")

    return {
        "size_label": size_label,
        "acc_baseline": float(np.mean([1 if r.success else 0 for r in baseline_results])),
        "acc_top1": float(np.mean([1 if r.success else 0 for r in fags_top1_results])),
        "acc_diversity": float(np.mean([1 if r.success else 0 for r in fags_div_results])),
        "acc_rrb": float(np.mean([1 if r.success else 0 for r in rrb_results])),
        "mean_nodes_diversity": float(np.mean([r.nodes_visited for r in fags_div_results])),
        "mean_nodes_rrb": float(np.mean([r.nodes_visited for r in rrb_results])),
        "diversity_vs_baseline_gain": vs_baseline["accuracy_gain"],
        "diversity_vs_baseline_p": vs_baseline["p_value_accuracy"],
        "diversity_vs_top1_gain": vs_top1["accuracy_gain"],
        "diversity_vs_top1_p": vs_top1["p_value_accuracy"],
        "diversity_vs_rrb_gain": vs_rrb["accuracy_gain"],
        "diversity_vs_rrb_p": vs_rrb["p_value_accuracy"],
    }


def main():
    sizes = {"Small": 20, "Medium": 100, "Large": 1000}
    query_count = 1000
    seed = 101

    records = [run_size(label, n, query_count, seed) for label, n in sizes.items()]

    csv_path = os.path.join(RESULTS_DIR, "diversity_memory_table.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Graph Size", "Baseline Acc", "FAGS-Top1 Acc", "FAGS-Diversity Acc", "RRB Acc (budget-matched)",
            "Diversity vs Baseline Gain", "Diversity vs Baseline p", "Diversity vs Top1 Gain", "Diversity vs Top1 p",
            "Diversity vs RRB Gain", "Diversity vs RRB p",
        ])
        for r in records:
            writer.writerow([
                r["size_label"], f"{r['acc_baseline']:.2%}", f"{r['acc_top1']:.2%}", f"{r['acc_diversity']:.2%}",
                f"{r['acc_rrb']:.2%}", f"{r['diversity_vs_baseline_gain']:+.2%}", f"{r['diversity_vs_baseline_p']:.3e}",
                f"{r['diversity_vs_top1_gain']:+.2%}", f"{r['diversity_vs_top1_p']:.3e}",
                f"{r['diversity_vs_rrb_gain']:+.2%}", f"{r['diversity_vs_rrb_p']:.3e}",
            ])
    print(f"\nTable written to {csv_path}")

    labels = [r["size_label"] for r in records]
    x = np.arange(len(labels))
    width = 0.2

    plt.figure(figsize=(9, 6))
    plt.bar(x - 1.5 * width, [r["acc_baseline"] * 100 for r in records], width, label="Baseline (1x)", color="black")
    plt.bar(x - 0.5 * width, [r["acc_top1"] * 100 for r in records], width, label="FAGS-Top1 (8-17x)", color="gray")
    plt.bar(x + 0.5 * width, [r["acc_rrb"] * 100 for r in records], width, label="Random-Restart (budget-matched)", color="darkorange")
    plt.bar(x + 1.5 * width, [r["acc_diversity"] * 100 for r in records], width, label="FAGS-Diversity (budget-matched cost)", color="crimson")
    plt.xticks(x, labels)
    plt.ylabel("Accuracy (%)")
    plt.title("Diversity Memory vs Baseline, Top1, and Budget-Matched Control")
    plt.legend()
    plt.grid(axis="y", linestyle=":", alpha=0.6)
    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, "diversity_memory.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Plot written to {plot_path}")

    summary_path = os.path.join(RESULTS_DIR, "diversity_memory_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as rf:
        rf.write("=" * 50 + "\n")
        rf.write("DIVERSITY MEMORY EXPERIMENT - SUMMARY\n")
        rf.write("=" * 50 + "\n\n")
        rf.write(
            "DiversityMemory revives the highest-scoring rejected candidate that is\n"
            "NOT the same as / confusable with / highly coherent with the winning\n"
            "relation, instead of always taking the single highest score (Top1Memory).\n"
            "The bar that matters: does it beat the budget-matched random-restart\n"
            "control, where Top1Memory/ThresholdMemory did not (see\n"
            "budget_matched_control_experiment.py)?\n\n"
        )

        wins_vs_rrb = 0
        for r in records:
            rf.write(f"-- {r['size_label']} graph --\n")
            rf.write(f"  Baseline (1x):                       {r['acc_baseline']:.2%}\n")
            rf.write(f"  FAGS-Top1 (8-17x cost):               {r['acc_top1']:.2%}\n")
            rf.write(f"  FAGS-Diversity ({r['mean_nodes_diversity']:.1f} nodes):           {r['acc_diversity']:.2%}\n")
            rf.write(f"  Random-Restart (matched to Diversity, {r['mean_nodes_rrb']:.1f} nodes): {r['acc_rrb']:.2%}\n")
            rf.write(f"  Diversity vs Baseline: {r['diversity_vs_baseline_gain']:+.2%} (p={r['diversity_vs_baseline_p']:.3e})\n")
            rf.write(f"  Diversity vs Top1:     {r['diversity_vs_top1_gain']:+.2%} (p={r['diversity_vs_top1_p']:.3e})\n")
            rf.write(f"  Diversity vs RRB:      {r['diversity_vs_rrb_gain']:+.2%} (p={r['diversity_vs_rrb_p']:.3e})\n\n")
            if r["diversity_vs_rrb_gain"] > 0 and r["diversity_vs_rrb_p"] < 0.05:
                wins_vs_rrb += 1

        rf.write("CONCLUSION\n----------\n")
        rf.write(f"DiversityMemory beat the budget-matched random-restart control with\n")
        rf.write(f"significance on {wins_vs_rrb}/{len(records)} graph sizes.\n")
        if wins_vs_rrb == len(records):
            rf.write(
                "Consistent win across all sizes: avoiding confusable-cluster reverts\n"
                "is a real fix for the gap found in budget_matched_control_experiment.py.\n"
            )
        elif wins_vs_rrb > 0:
            rf.write(
                "Partial win: diversity-aware revival helps on some graph sizes but not\n"
                "all - an improvement over Top1Memory's complete inability to clear this\n"
                "bar, but not a full fix.\n"
            )
        else:
            rf.write(
                "No graph size cleared the bar. Even explicitly steering revival away\n"
                "from confusable/coherent relations doesn't make targeted memory beat\n"
                "diversified random exploration at matched cost.\n"
            )

    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
