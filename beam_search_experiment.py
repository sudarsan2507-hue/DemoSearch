"""Beam Search experiment.

The final, structurally different test: every FAGS variant tried so far
(Top1/Top2/Threshold/Diversity memory, shield, certificate, RBSC, RTC-lite,
failure-pattern avoidance) is "walk one path, detect failure, pick one
candidate to revive" - and none of them reliably beat a budget-matched
random-restart control (see budget_matched_control_experiment.py /
diversity_memory_experiment.py). Beam search (fags/beam_search.py) never
commits to one path: it keeps the K best live hypotheses at every hop, so
the question becomes whether CORRELATED diversity (beam, derived from real
partial-path scores) beats UNCORRELATED diversity (independent noisy
greedy re-rolls) at matched search cost - and whether either beats plain
FAGS.

Sweeps beam_width across all 3 graph sizes (same 1000 queries / seed=101 as
the canonical comparisons, for direct comparability).
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
from fags.beam_search import beam_search
from fags.evaluation import evaluate_results

RESULTS_DIR = r"d:\Projects\DemoSearch\results"
os.makedirs(RESULTS_DIR, exist_ok=True)

MAX_DEPTH = 6
MAX_BACKTRACKS = 3
MAX_RESTARTS = 60
BEAM_WIDTHS = [1, 2, 3, 5, 8]


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

    baseline_results = [baseline_search(graph, q, verifier, max_depth=MAX_DEPTH) for q in queries]

    fags_memory = create_memory("top1")
    fags_results = [
        failure_search(graph=graph, query=q, verifier=verifier, memory=fags_memory,
                        max_depth=MAX_DEPTH, max_backtracks=MAX_BACKTRACKS, enable_re_verification=True)
        for q in queries
    ]

    acc_baseline = float(np.mean([1 if r.success else 0 for r in baseline_results]))
    acc_fags = float(np.mean([1 if r.success else 0 for r in fags_results]))

    beam_records = []
    for bw in BEAM_WIDTHS:
        print(f"  Running Beam Search (width={bw})...")
        beam_results = [beam_search(graph, q, verifier, beam_width=bw, max_depth=MAX_DEPTH) for q in queries]
        rrb_results = [
            random_restart_baseline(graph, q, verifier, target_budget=max(br.nodes_visited, 1), max_depth=MAX_DEPTH)
            for q, br in zip(queries, beam_results)
        ]

        vs_baseline = evaluate_results(baseline_results, beam_results, f"Beam(w={bw}) vs Baseline")
        vs_rrb = evaluate_results(rrb_results, beam_results, f"Beam(w={bw}) vs RRB")

        beam_records.append({
            "beam_width": bw,
            "acc_beam": float(np.mean([1 if r.success else 0 for r in beam_results])),
            "mean_nodes_beam": float(np.mean([r.nodes_visited for r in beam_results])),
            "acc_rrb": float(np.mean([1 if r.success else 0 for r in rrb_results])),
            "mean_nodes_rrb": float(np.mean([r.nodes_visited for r in rrb_results])),
            "gain_vs_baseline": vs_baseline["accuracy_gain"],
            "p_vs_baseline": vs_baseline["p_value_accuracy"],
            "gain_vs_rrb": vs_rrb["accuracy_gain"],
            "p_vs_rrb": vs_rrb["p_value_accuracy"],
        })

    return {
        "size_label": size_label,
        "acc_baseline": acc_baseline,
        "acc_fags": acc_fags,
        "beam_records": beam_records,
    }


def main():
    sizes = {"Small": 20, "Medium": 100, "Large": 1000}
    query_count = 1000
    seed = 101

    size_results = [run_size(label, n, query_count, seed) for label, n in sizes.items()]

    # ── CSV ──
    csv_path = os.path.join(RESULTS_DIR, "beam_search_table.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Graph Size", "Beam Width", "Baseline Acc", "FAGS-Top1 Acc", "Beam Acc", "Beam Mean Nodes",
            "RRB Acc (budget-matched)", "RRB Mean Nodes", "Beam vs Baseline Gain", "Beam vs Baseline p",
            "Beam vs RRB Gain", "Beam vs RRB p",
        ])
        for sr in size_results:
            for br in sr["beam_records"]:
                writer.writerow([
                    sr["size_label"], br["beam_width"], f"{sr['acc_baseline']:.2%}", f"{sr['acc_fags']:.2%}",
                    f"{br['acc_beam']:.2%}", f"{br['mean_nodes_beam']:.2f}", f"{br['acc_rrb']:.2%}",
                    f"{br['mean_nodes_rrb']:.2f}", f"{br['gain_vs_baseline']:+.2%}", f"{br['p_vs_baseline']:.3e}",
                    f"{br['gain_vs_rrb']:+.2%}", f"{br['p_vs_rrb']:.3e}",
                ])
    print(f"\nTable written to {csv_path}")

    # ── Plot: one panel per graph size, accuracy vs nodes visited ──
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), sharey=False)
    for ax, sr in zip(axes, size_results):
        beam_x = [br["mean_nodes_beam"] for br in sr["beam_records"]]
        beam_y = [br["acc_beam"] * 100 for br in sr["beam_records"]]
        rrb_x = [br["mean_nodes_rrb"] for br in sr["beam_records"]]
        rrb_y = [br["acc_rrb"] * 100 for br in sr["beam_records"]]

        ax.plot(beam_x, beam_y, marker="o", color="crimson", label="Beam Search")
        ax.plot(rrb_x, rrb_y, marker="^", color="darkorange", label="Random-Restart (matched cost)")
        ax.axhline(sr["acc_baseline"] * 100, color="black", linestyle="--", label="Baseline (1x)")
        ax.axhline(sr["acc_fags"] * 100, color="gray", linestyle=":", label="FAGS-Top1")
        for br in sr["beam_records"]:
            ax.annotate(f"w={br['beam_width']}", (br["mean_nodes_beam"], br["acc_beam"] * 100),
                        textcoords="offset points", xytext=(4, 4), fontsize=8)
        ax.set_title(sr["size_label"])
        ax.set_xlabel("Mean Nodes Visited (search cost)")
        ax.set_ylabel("Accuracy (%)")
        ax.grid(True, linestyle=":", alpha=0.5)

    axes[0].legend(fontsize=8, loc="lower right")
    plt.suptitle("Beam Search: Accuracy vs Search Cost, by Beam Width")
    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, "beam_search_sweep.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Plot written to {plot_path}")

    # ── Summary ──
    summary_path = os.path.join(RESULTS_DIR, "beam_search_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as rf:
        rf.write("=" * 50 + "\n")
        rf.write("BEAM SEARCH EXPERIMENT - SUMMARY\n")
        rf.write("=" * 50 + "\n\n")
        rf.write(
            "Beam search keeps the K best live hypotheses concurrently at every hop,\n"
            "instead of FAGS's commit-then-recover design. Tests whether correlated\n"
            "diversity (real partial-path scores) beats uncorrelated diversity (random\n"
            "restarts) at matched search cost, across beam widths "
            f"{BEAM_WIDTHS}.\n\n"
        )

        total_wins_vs_rrb = 0
        total_configs = 0
        for sr in size_results:
            rf.write(f"-- {sr['size_label']} graph (Baseline={sr['acc_baseline']:.2%}, "
                      f"FAGS-Top1={sr['acc_fags']:.2%}) --\n")
            for br in sr["beam_records"]:
                total_configs += 1
                win = br["gain_vs_rrb"] > 0 and br["p_vs_rrb"] < 0.05
                total_wins_vs_rrb += int(win)
                rf.write(
                    f"  width={br['beam_width']}: Beam={br['acc_beam']:.2%} "
                    f"({br['mean_nodes_beam']:.1f} nodes) vs RRB={br['acc_rrb']:.2%} "
                    f"({br['mean_nodes_rrb']:.1f} nodes) | "
                    f"gain={br['gain_vs_rrb']:+.2%} p={br['p_vs_rrb']:.2e} "
                    f"{'<-- BEAM WINS' if win else ''}\n"
                )
            rf.write("\n")

        rf.write("CONCLUSION\n----------\n")
        rf.write(f"Beam Search beat the budget-matched random-restart control with\n")
        rf.write(f"significance in {total_wins_vs_rrb}/{total_configs} (graph size x beam width) configurations.\n")
        if total_wins_vs_rrb >= total_configs * 0.5:
            rf.write(
                "Majority win: correlated diversity (beam) beats uncorrelated diversity\n"
                "(random restarts) at matched cost more often than not - a structurally\n"
                "different search algorithm succeeds where every revival-selection\n"
                "heuristic on top of FAGS's architecture failed.\n"
            )
        elif total_wins_vs_rrb > 0:
            rf.write(
                "Partial win: beam search beats the random-restart control in some\n"
                "configurations but not consistently - better than every FAGS variant\n"
                "tried, which never won outside the Small graph, but not a clean result.\n"
            )
        else:
            rf.write(
                "No configuration cleared the bar. Even abandoning the commit-then-\n"
                "recover architecture entirely doesn't make targeted/correlated search\n"
                "beat diversified random exploration at matched cost in this graph\n"
                "topology - the verifier's signal quality, not the search architecture,\n"
                "is the binding constraint.\n"
            )

    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
