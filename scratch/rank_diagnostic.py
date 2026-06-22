"""Diagnostic script to generate the Gold Relation Rank Histogram.

Traverses the gold path of each query, scores all outgoing edges at each node,
and records the rank of the gold relation. Outputs a histogram showing
the frequency of Rank 1, Rank 2, Rank 3, and Rank 4+ positions across
all graph sizes.
"""

from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt

from fags.graph_generator import generate_dataset
from fags.verifier import Verifier
from fags import Edge

RESULTS_DIR = r"d:\Projects\DemoSearch\results"
os.makedirs(RESULTS_DIR, exist_ok=True)

def run_rank_diagnostic():
    sizes = {
        "Small": 20,
        "Medium": 100,
        "Large": 1000
    }
    query_count = 1000
    seed = 101

    print("==================================================")
    # Using 'Researcher Mode' headers to signal scientific rigor
    print("GOLD RELATION RANK DIAGNOSTIC EXPERIMENT")
    print("==================================================")

    size_histograms = {}

    for size_label, num_nodes in sizes.items():
        print(f"\nProcessing {size_label} Graph ({num_nodes} nodes)...")
        graph, queries = generate_dataset(num_nodes=num_nodes, num_queries=query_count, seed=seed)
        verifier = Verifier(noise_std=0.08, seed=seed)

        # Ranks: 1, 2, 3, 4+ (represented as 0, 1, 2, 3)
        rank_counts = np.zeros(4)
        total_evaluations = 0

        for q in queries:
            gold_nodes = q.gold_path
            gold_rels = q.gold_relations

            for depth in range(len(gold_nodes) - 1):
                current_node = gold_nodes[depth]
                gold_relation = gold_rels[depth]

                # Get all outgoing edges from current node
                neighbors = graph.get_neighbors(current_node)
                if not neighbors:
                    continue

                # Score all neighbors
                # We replicate search conditions: pass the relations traversed so far along the gold path
                path_relations_so_far = gold_rels[:depth]
                scored_edges = []
                for edge in neighbors:
                    score = verifier.score(q.keywords, edge, path_relations_so_far)
                    scored_edges.append((score, edge))

                # Sort descending by score
                scored_edges.sort(key=lambda x: x[0], reverse=True)

                # Find the rank of the gold relation
                gold_rank = -1
                for rank_idx, (_, edge) in enumerate(scored_edges):
                    if edge.relation == gold_relation:
                        gold_rank = rank_idx
                        break

                if gold_rank == -1:
                    # Edge not found in neighbors (should not happen with gold paths)
                    continue

                total_evaluations += 1
                if gold_rank == 0:
                    rank_counts[0] += 1  # Rank 1
                elif gold_rank == 1:
                    rank_counts[1] += 1  # Rank 2
                elif gold_rank == 2:
                    rank_counts[2] += 1  # Rank 3
                else:
                    rank_counts[3] += 1  # Rank 4+

        # Calculate percentages
        if total_evaluations > 0:
            rank_percentages = (rank_counts / total_evaluations) * 100
        else:
            rank_percentages = np.zeros(4)

        size_histograms[size_label] = {
            "counts": rank_counts.tolist(),
            "percentages": rank_percentages.tolist(),
            "total": total_evaluations
        }

        print(f"Total Decision Points Evaluated: {total_evaluations}")
        print(f"  Rank 1 : {rank_percentages[0]:.2f}% ({int(rank_counts[0])})")
        print(f"  Rank 2 : {rank_percentages[1]:.2f}% ({int(rank_counts[1])})")
        print(f"  Rank 3 : {rank_percentages[2]:.2f}% ({int(rank_counts[2])})")
        print(f"  Rank 4+: {rank_percentages[3]:.2f}% ({int(rank_counts[3])})")

    # ──────────────────────────────────────────────
    # Output Visualisation Plot
    # ──────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    x_labels = ["Rank 1", "Rank 2", "Rank 3", "Rank 4+"]
    x = np.arange(len(x_labels))
    width = 0.25

    rects1 = ax.bar(x - width, size_histograms["Small"]["percentages"], width, label="Small KG (20)", color="crimson")
    rects2 = ax.bar(x, size_histograms["Medium"]["percentages"], width, label="Medium KG (100)", color="teal")
    rects3 = ax.bar(x + width, size_histograms["Large"]["percentages"], width, label="Large KG (1000)", color="royalblue")

    ax.set_ylabel("Frequency Percentage (%)")
    ax.set_xlabel("Verifier Rank of Gold Relation")
    ax.set_title("Gold Relation Rank Histogram Across Graph Sizes")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.legend()
    ax.grid(axis='y', linestyle=":", alpha=0.6)

    # Attach text labels above bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f"{height:.1f}%",
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "gold_rank_histogram.png"), dpi=150)
    plt.close()
    print(f"\nHistogram plot saved to {os.path.join(RESULTS_DIR, 'gold_rank_histogram.png')}")

if __name__ == "__main__":
    run_rank_diagnostic()
