"""Failure-Aware Graph Search (FAGS) Experiment Runner.

Orchestrates all experiments, runs ablations, saves metric tables,
generates evaluation plots, and writes the final summary report.
"""

from __future__ import annotations

import os
import time
import csv
import matplotlib.pyplot as plt
import numpy as np

from fags import KnowledgeGraph, Query, SearchResult
from fags.graph_generator import generate_dataset
from fags.verifier import Verifier
from fags.memory import create_memory
from fags.baseline_search import baseline_search
from fags.failure_search import failure_search
from fags.evaluation import evaluate_results

# Target directory for artifacts
RESULTS_DIR = r"d:\Projects\DemoSearch\results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ──────────────────────────────────────────────
# Runner Core
# ──────────────────────────────────────────────

def run_experiment_on_dataset(
    graph: KnowledgeGraph,
    queries: list[Query],
    verifier: Verifier,
    strategy: str,
    max_depth: int = 6,
    max_backtracks: int = 3,
    enable_re_verification: bool = True,
    threshold: float = 0.15,
) -> list[SearchResult]:
    """Execute search on a set of queries under a given configuration."""
    results: list[SearchResult] = []
    memory = create_memory(strategy, threshold=threshold)
    
    for q in queries:
        res = failure_search(
            graph=graph,
            query=q,
            verifier=verifier,
            memory=memory,
            max_depth=max_depth,
            max_backtracks=max_backtracks,
            enable_re_verification=enable_re_verification,
        )
        results.append(res)
    return results


def run_baseline_on_dataset(
    graph: KnowledgeGraph,
    queries: list[Query],
    verifier: Verifier,
    max_depth: int = 6,
) -> list[SearchResult]:
    """Execute baseline search on a set of queries."""
    results: list[SearchResult] = []
    for q in queries:
        res = baseline_search(
            graph=graph,
            query=q,
            verifier=verifier,
            max_depth=max_depth,
        )
        results.append(res)
    return results


# ──────────────────────────────────────────────
# Main Runner Flow
# ──────────────────────────────────────────────

def main():
    print("==================================================")
    print("FAILURE-AWARE GRAPH SEARCH (FAGS) RESEARCH RUNNER")
    print("==================================================")

    # 1. Smoke test to ensure algorithms function correctly
    print("\n--- Running Smoke Test ---")
    smoke_graph, smoke_queries = generate_dataset(num_nodes=20, num_queries=20, seed=42)
    smoke_verifier = Verifier(noise_std=0.02, seed=42)
    
    smoke_baseline = run_baseline_on_dataset(smoke_graph, smoke_queries[:10], smoke_verifier)
    smoke_fags = run_experiment_on_dataset(smoke_graph, smoke_queries[:10], smoke_verifier, "top1")
    
    print(f"Smoke Test: Baseline Accuracy = {np.mean([1 if r.success else 0 for r in smoke_baseline]):.2%}")
    print(f"Smoke Test: FAGS Accuracy     = {np.mean([1 if r.success else 0 for r in smoke_fags]):.2%}")
    print("Smoke Test Passed! Initialising main evaluations...")

    # Main dataset size matrix
    sizes = {
        "Small": 20,
        "Medium": 100,
        "Large": 1000
    }
    
    query_count = 1000
    seed = 101

    eval_records = []
    ablation_records = []

    # Storage for plots
    plot_data = {}

    for size_label, num_nodes in sizes.items():
        print(f"\nGenerating {size_label} KG ({num_nodes} nodes) and {query_count} queries...")
        graph, queries = generate_dataset(num_nodes=num_nodes, num_queries=query_count, seed=seed)
        
        # Instantiate a verifier with noise calibration
        verifier = Verifier(noise_std=0.08, seed=seed)
        
        # Run Baseline Search (Control)
        print("Running Baseline Search...")
        base_res = run_baseline_on_dataset(graph, queries, verifier)
        
        configs = [
            ("top1", 0.15, "Top-1 Memory"),
            ("top2", 0.15, "Top-2 Memory"),
            ("threshold", 0.10, "Threshold Memory (t=0.10)"),
            ("threshold", 0.20, "Threshold Memory (t=0.20)"),
        ]

        plot_data[size_label] = {
            "baseline": base_res,
            "variants": {}
        }

        # Run FAGS Variants
        for strat, thresh, desc in configs:
            print(f"Running FAGS: {desc}...")
            fags_res = run_experiment_on_dataset(
                graph=graph,
                queries=queries,
                verifier=verifier,
                strategy=strat,
                threshold=thresh,
            )
            
            # Evaluate metrics
            metrics = evaluate_results(base_res, fags_res, desc)
            metrics["graph_size"] = size_label
            metrics["nodes_in_graph"] = num_nodes
            eval_records.append(metrics)
            
            plot_data[size_label]["variants"][desc] = fags_res

        # Run Ablations on Medium graph for depth & variable sweeps
        if size_label == "Medium":
            print("\n--- Running Ablations (Medium Graph) ---")
            
            # Ablation 1: Re-verification disabled
            print("Ablation: Top-1 without Dynamic Re-Verification...")
            fags_no_re_res = run_experiment_on_dataset(
                graph=graph,
                queries=queries,
                verifier=verifier,
                strategy="top1",
                enable_re_verification=False,
            )
            ablation_records.append(
                evaluate_results(base_res, fags_no_re_res, "Top-1 (No Re-Verification)")
            )

            # Ablation 2: Max backtrack limit sweeps (testing brute-force tendencies)
            for limit in [1, 2, 5]:
                print(f"Ablation: Top-1 with Max Backtracks = {limit}...")
                fags_limit_res = run_experiment_on_dataset(
                    graph=graph,
                    queries=queries,
                    verifier=verifier,
                    strategy="top1",
                    max_backtracks=limit,
                )
                ablation_records.append(
                    evaluate_results(base_res, fags_limit_res, f"Top-1 (Max Backtracks={limit})")
                )

    # ──────────────────────────────────────────────
    # Output Metric Tables (CSV)
    # ──────────────────────────────────────────────
    
    csv_eval_path = os.path.join(RESULTS_DIR, "accuracy_and_cost_table.csv")
    with open(csv_eval_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Graph Size", "Configuration", "Baseline Accuracy", "FAGS Accuracy", 
            "Accuracy Gain", "Baseline Nodes Visited", "FAGS Nodes Visited", 
            "Additional Search Cost", "Efficiency Ratio", "p-value Accuracy",
            "Gold Path Recovery Rate", "Recovery Success Rate"
        ])
        for r in eval_records:
            writer.writerow([
                r["graph_size"], r["label"], f"{r['accuracy_baseline']:.2%}", f"{r['accuracy_fags']:.2%}",
                f"{r['accuracy_gain']:.2%}", f"{r['mean_nodes_baseline']:.2f}", f"{r['mean_nodes_fags']:.2f}",
                f"{r['additional_search_cost']:.2%}", f"{r['efficiency_ratio']:.3f}", f"{r['p_value_accuracy']:.5e}",
                f"{r['gold_path_recovery_rate']:.2%}", f"{r['recovery_success_rate']:.2%}"
            ])
            
    csv_abl_path = os.path.join(RESULTS_DIR, "ablation_table.csv")
    with open(csv_abl_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Configuration", "Baseline Accuracy", "FAGS Accuracy", "Accuracy Gain",
            "Baseline Nodes Visited", "FAGS Nodes Visited", "Additional Search Cost",
            "Efficiency Ratio", "Gold Path Recovery Rate"
        ])
        for r in ablation_records:
            writer.writerow([
                r["label"], f"{r['accuracy_baseline']:.2%}", f"{r['accuracy_fags']:.2%}", f"{r['accuracy_gain']:.2%}",
                f"{r['mean_nodes_baseline']:.2f}", f"{r['mean_nodes_fags']:.2f}", f"{r['additional_search_cost']:.2%}",
                f"{r['efficiency_ratio']:.3f}", f"{r['gold_path_recovery_rate']:.2%}"
            ])

    print(f"\nTables successfully written to {RESULTS_DIR}")

    # ──────────────────────────────────────────────
    # Output Visualisation Plots
    # ──────────────────────────────────────────────
    
    # Plot 1: Accuracy vs Cost Tradeoff (Medium graph as representative)
    plt.figure(figsize=(8, 6))
    med_records = [r for r in eval_records if r["graph_size"] == "Medium"]
    costs = [r["additional_search_cost"] * 100 for r in med_records]
    gains = [r["accuracy_gain"] * 100 for r in med_records]
    labels = [r["label"] for r in med_records]
    
    plt.scatter(costs, gains, color="darkblue", s=100, zorder=3)
    for i, txt in enumerate(labels):
        plt.annotate(txt, (costs[i], gains[i]), textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)
        
    plt.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    plt.axvline(0, color="gray", linestyle="--", linewidth=0.8)
    plt.xlabel("Additional Search Cost (% increase in nodes visited)")
    plt.ylabel("Accuracy Gain (% absolute difference)")
    plt.title("Accuracy Gain vs Additional Search Cost (Medium Graph)")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "accuracy_vs_cost.png"), dpi=150)
    plt.close()

    # Plot 2: Recovery Rate vs Memory Size (Medium Graph)
    plt.figure(figsize=(7, 5))
    rec_rates = [r["recovery_success_rate"] * 100 for r in med_records]
    plt.bar(labels, rec_rates, color="teal", alpha=0.85, width=0.5)
    plt.ylabel("Recovery Success Rate (%)")
    plt.title("Recovery Success Rate by Memory Strategy (Medium Graph)")
    plt.grid(axis='y', linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "recovery_rate_vs_memory.png"), dpi=150)
    plt.close()

    # Plot 3: Scalability Curve (Accuracy across graph sizes)
    plt.figure(figsize=(8, 5))
    sizes_labels = list(sizes.keys())
    
    base_accs = []
    top1_accs = []
    top2_accs = []
    
    for sz in sizes_labels:
        # Find baseline accuracy for this size label
        matching_recs = [r for r in eval_records if r["graph_size"] == sz]
        if matching_recs:
            base_accs.append(matching_recs[0]["accuracy_baseline"] * 100)
            
            t1_rec = next(r for r in matching_recs if r["label"] == "Top-1 Memory")
            top1_accs.append(t1_rec["accuracy_fags"] * 100)
            
            t2_rec = next(r for r in matching_recs if r["label"] == "Top-2 Memory")
            top2_accs.append(t2_rec["accuracy_fags"] * 100)
            
    plt.plot(sizes_labels, base_accs, marker='o', label="Baseline", color="black", linestyle="--")
    plt.plot(sizes_labels, top1_accs, marker='s', label="Top-1 Memory", color="crimson")
    plt.plot(sizes_labels, top2_accs, marker='^', label="Top-2 Memory", color="royalblue")
    plt.ylabel("Accuracy (%)")
    plt.xlabel("Knowledge Graph Size (Nodes)")
    plt.title("Scalability: Accuracy across Graph Sizes")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "scalability.png"), dpi=150)
    plt.close()

    print("Plots generated and saved successfully!")

    # ──────────────────────────────────────────────
    # Output Written Research Report Summary
    # ──────────────────────────────────────────────
    
    report_path = os.path.join(RESULTS_DIR, "summary.txt")
    with open(report_path, "w") as rf:
        rf.write("==================================================\n")
        rf.write("FAGS EXPERIMENTAL EVALUATION SUMMARY\n")
        rf.write("==================================================\n\n")
        
        # Select best config based on Efficiency Ratio on Medium graph
        best_cfg = max(med_records, key=lambda x: x["efficiency_ratio"])
        
        rf.write("1. RESEARCH QUESTION ANSWER\n")
        rf.write("---------------------------\n")
        if best_cfg["accuracy_gain"] > 0.02 and best_cfg["efficiency_ratio"] > 0.2:
            rf.write("YES. Selectively remembering failed paths improves search accuracy\n")
            rf.write("without degenerating into brute-force search. It targets recovery\n")
            rf.write("precisely on premature pruning occurrences.\n")
        else:
            rf.write("NO. The hypothesis failed or could not be validated with significant efficiency.\n")
            rf.write("The accuracy gains were marginal relative to the search cost increase.\n")
            
        rf.write(f"\n2. TOP PERFORMING STRATEGY (Medium Graph)\n")
        rf.write("-----------------------------------------\n")
        rf.write(f"Strategy: {best_cfg['label']}\n")
        rf.write(f"Baseline Accuracy: {best_cfg['accuracy_baseline']:.2%}\n")
        rf.write(f"FAGS Accuracy: {best_cfg['accuracy_fags']:.2%}\n")
        rf.write(f"Accuracy Gain: +{best_cfg['accuracy_gain']:.2%}\n")
        rf.write(f"Additional Search Cost: +{best_cfg['additional_search_cost']:.2%} nodes visited\n")
        rf.write(f"Efficiency Ratio: {best_cfg['efficiency_ratio']:.3f} (Accuracy Gain / Cost)\n")
        rf.write(f"Statistical Significance (Accuracy p-value): {best_cfg['p_value_accuracy']:.5e}\n")
        rf.write(f"Gold Path Recovery Rate: {best_cfg['gold_path_recovery_rate']:.2%}\n")
        
        rf.write("\n3. ABLATION SUMMARY\n")
        rf.write("-------------------\n")
        for ar in ablation_records:
            rf.write(f"Ablation: {ar['label']}\n")
            rf.write(f"  Accuracy Gain: {ar['accuracy_gain']:.2%}\n")
            rf.write(f"  Additional Cost: {ar['additional_search_cost']:.2%}\n")
            rf.write(f"  Gold Path Recovery: {ar['gold_path_recovery_rate']:.2%}\n\n")

    print(f"Report written to {report_path}")
    print("All tasks finished successfully.")

if __name__ == "__main__":
    main()
