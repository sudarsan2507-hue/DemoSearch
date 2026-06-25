"""Beam Search with a stronger verifier (HybridVerifier).

beam_search_experiment.py showed beam search strictly dominates FAGS-Top1
in the cost/accuracy tradeoff using the weak rule-based Verifier. Open
question: does a stronger verifier (rule-based + BAAI/bge-small-en-v1.5
embeddings) help beam search the way it HURT FAGS in
budget_matched_control_embedding_experiment.py? Prediction: it should help,
since beam search's mechanism (keep top-K under verifier scoring) directly
depends on the verifier's ranking quality to avoid pruning the gold
hypothesis out of the beam - unlike FAGS, whose bottleneck was the
commit-then-recover architecture itself, not verifier quality.

Single graph size (500 nodes / 500 queries, seed=42), matching the scale of
the repo's other embedding-based experiments since real model inference is
far slower than the synthetic rule-based scorer, and beam search multiplies
verifier calls by ~width per hop.
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
from fags.beam_search import beam_search
from fags.evaluation import evaluate_results

RESULTS_DIR = r"d:\Projects\DemoSearch\results"
os.makedirs(RESULTS_DIR, exist_ok=True)

NUM_NODES = 500
QUERY_COUNT = 500
SEED = 42
MAX_DEPTH = 6
MAX_BACKTRACKS = 5  # matches hybrid_sweep_experiment.py's convention for this verifier/scale
MAX_RESTARTS = 60
BEAM_WIDTHS = [1, 2, 3, 5]


def random_restart_baseline(
    graph: KnowledgeGraph,
    query: Query,
    verifier,
    target_budget: int,
    max_depth: int = MAX_DEPTH,
    max_restarts: int = MAX_RESTARTS,
) -> SearchResult:
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
    print(f"Generating graph ({NUM_NODES} nodes, {QUERY_COUNT} queries, seed={SEED})...")
    graph, queries = generate_dataset(num_nodes=NUM_NODES, num_queries=QUERY_COUNT, seed=SEED)

    print("Loading HybridVerifier (BAAI/bge-small-en-v1.5 + rule-based)...")
    verifier = HybridVerifier(model_name="BAAI/bge-small-en-v1.5", alpha=0.5, noise_std=0.30, seed=SEED)

    print("Running plain Baseline...")
    baseline_results = [baseline_search(graph, q, verifier, max_depth=MAX_DEPTH) for q in queries]

    print("Running plain FAGS (Top1)...")
    fags_memory = create_memory("top1")
    fags_results = [
        failure_search(graph=graph, query=q, verifier=verifier, memory=fags_memory,
                        max_depth=MAX_DEPTH, max_backtracks=MAX_BACKTRACKS, enable_re_verification=True)
        for q in queries
    ]

    acc_baseline = float(np.mean([1 if r.success else 0 for r in baseline_results]))
    acc_fags = float(np.mean([1 if r.success else 0 for r in fags_results]))
    mean_nodes_fags = float(np.mean([r.nodes_visited for r in fags_results]))

    beam_records = []
    for bw in BEAM_WIDTHS:
        print(f"Running Beam Search (width={bw})...")
        beam_results = [beam_search(graph, q, verifier, beam_width=bw, max_depth=MAX_DEPTH) for q in queries]

        print(f"Running budget-matched Random-Restart control for width={bw}...")
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

    csv_path = os.path.join(RESULTS_DIR, "beam_search_embedding_table.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Beam Width", "Baseline Acc", "FAGS-Top1 Acc", "FAGS-Top1 Nodes", "Beam Acc", "Beam Nodes",
            "RRB Acc (budget-matched)", "RRB Nodes", "Beam vs Baseline Gain", "Beam vs Baseline p",
            "Beam vs RRB Gain", "Beam vs RRB p",
        ])
        for br in beam_records:
            writer.writerow([
                br["beam_width"], f"{acc_baseline:.2%}", f"{acc_fags:.2%}", f"{mean_nodes_fags:.2f}",
                f"{br['acc_beam']:.2%}", f"{br['mean_nodes_beam']:.2f}", f"{br['acc_rrb']:.2%}",
                f"{br['mean_nodes_rrb']:.2f}", f"{br['gain_vs_baseline']:+.2%}", f"{br['p_vs_baseline']:.3e}",
                f"{br['gain_vs_rrb']:+.2%}", f"{br['p_vs_rrb']:.3e}",
            ])
    print(f"\nTable written to {csv_path}")

    plt.figure(figsize=(8, 6))
    beam_x = [br["mean_nodes_beam"] for br in beam_records]
    beam_y = [br["acc_beam"] * 100 for br in beam_records]
    rrb_x = [br["mean_nodes_rrb"] for br in beam_records]
    rrb_y = [br["acc_rrb"] * 100 for br in beam_records]

    plt.plot(beam_x, beam_y, marker="o", color="crimson", label="Beam Search (HybridVerifier)")
    plt.plot(rrb_x, rrb_y, marker="^", color="darkorange", label="Random-Restart (matched cost)")
    plt.axhline(acc_baseline * 100, color="black", linestyle="--", label="Baseline (1x)")
    plt.axhline(acc_fags * 100, color="gray", linestyle=":", label=f"FAGS-Top1 ({mean_nodes_fags:.1f} nodes)")
    for br in beam_records:
        plt.annotate(f"w={br['beam_width']}", (br["mean_nodes_beam"], br["acc_beam"] * 100),
                      textcoords="offset points", xytext=(4, 4), fontsize=9)
    plt.xlabel("Mean Nodes Visited (search cost)")
    plt.ylabel("Accuracy (%)")
    plt.title("Beam Search vs Budget-Matched Control (HybridVerifier)")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, "beam_search_embedding.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Plot written to {plot_path}")

    summary_path = os.path.join(RESULTS_DIR, "beam_search_embedding_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as rf:
        rf.write("=" * 50 + "\n")
        rf.write("BEAM SEARCH WITH HYBRIDVERIFIER - SUMMARY\n")
        rf.write("=" * 50 + "\n\n")
        rf.write(
            "Repeats beam_search_experiment.py's comparison but swaps the weak\n"
            "rule-based Verifier for HybridVerifier (rule-based + BGE embeddings,\n"
            "alpha=0.5), to test whether a stronger verifier helps beam search the\n"
            "way it HURT FAGS in budget_matched_control_embedding_experiment.py.\n"
            f"{NUM_NODES}-node graph, {QUERY_COUNT} queries, seed={SEED}.\n\n"
        )
        rf.write(f"Baseline accuracy:          {acc_baseline:.2%}\n")
        rf.write(f"FAGS-Top1 accuracy:         {acc_fags:.2%} ({mean_nodes_fags:.2f} nodes)\n\n")

        wins_vs_rrb = 0
        beats_fags_at_lower_cost = 0
        for br in beam_records:
            win = br["gain_vs_rrb"] > 0 and br["p_vs_rrb"] < 0.05
            wins_vs_rrb += int(win)
            dominates_fags = br["acc_beam"] > acc_fags and br["mean_nodes_beam"] < mean_nodes_fags
            beats_fags_at_lower_cost += int(dominates_fags)
            rf.write(
                f"width={br['beam_width']}: Beam={br['acc_beam']:.2%} ({br['mean_nodes_beam']:.1f} nodes) "
                f"vs RRB={br['acc_rrb']:.2%} ({br['mean_nodes_rrb']:.1f} nodes) | "
                f"gain={br['gain_vs_rrb']:+.2%} p={br['p_vs_rrb']:.2e} "
                f"{'<-- BEAM WINS' if win else ''}"
                f"{' | DOMINATES FAGS (higher acc, lower cost)' if dominates_fags else ''}\n"
            )

        rf.write("\nCONCLUSION\n----------\n")
        rf.write(f"Beam Search beat the budget-matched RRB control with significance in "
                  f"{wins_vs_rrb}/{len(beam_records)} widths.\n")
        rf.write(f"Beam Search strictly dominated FAGS-Top1 (higher accuracy AND lower cost) in "
                  f"{beats_fags_at_lower_cost}/{len(beam_records)} widths.\n\n")
        if wins_vs_rrb == len(beam_records):
            rf.write(
                "Confirms the prediction: with a stronger verifier, beam search's\n"
                "advantage over the budget-matched control holds (or strengthens) just\n"
                "as cleanly as with the weak verifier - opposite of what happened to\n"
                "FAGS, whose advantage reversed under a stronger verifier. Beam search's\n"
                "mechanism scales with verifier quality; FAGS's commit-then-recover\n"
                "design does not.\n"
            )
        elif wins_vs_rrb > 0:
            rf.write(
                "Partial confirmation: beam search still beats the control on some\n"
                "widths with a stronger verifier, but not as cleanly as with the weak\n"
                "verifier - unlike FAGS, which got WORSE across the board, beam search's\n"
                "advantage is dampened rather than reversed.\n"
            )
        else:
            rf.write(
                "Does not confirm the prediction: with a stronger verifier, beam search\n"
                "no longer beats the budget-matched control - contrary to expectation,\n"
                "the verifier-quality dependence of beam search's advantage looks more\n"
                "like FAGS's than predicted.\n"
            )

    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
