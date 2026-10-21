"""Gold Relation Rank Diagnostic, re-measured with HybridVerifier.

scratch/rank_diagnostic.py established the rule-based Verifier's ceiling:
across Small/Medium/Large graphs it ranks the correct (gold) relation #1
only ~57-60% of the time, #2 ~18-19%, #3 ~9-10%, #4+ ~12-14% - identified
in PROJECT_STATUS.md SS4 as the real bottleneck behind every search
algorithm's ceiling (FAGS, beam search, and MCTS all inherit whatever the
verifier can and can't distinguish). Every search-algorithm comparison
done since has used either the rule-based Verifier or HybridVerifier for
accuracy numbers, but this rank histogram itself was never re-measured
against HybridVerifier specifically - this closes that gap directly,
quantifying how much (if any) the embedding signal raises the ceiling
search algorithms are bumping into.

Same diagnostic logic as scratch/rank_diagnostic.py (traverse each query's
gold path, score every outgoing candidate at each hop, record the gold
relation's rank), swapping in HybridVerifier (rule + BAAI/bge-small-en-v1.5
embeddings, alpha=0.5) in place of the rule-based Verifier. Query count
reduced to 500 (from 1000) per graph size to keep runtime reasonable given
real model inference, matching the scale convention of this repo's other
embedding-based experiments.

IMPORTANT: noise_std=0.08, matching the rule-based reference exactly. The
repo's other embedding experiments (hybrid_sweep_experiment.py,
beam_search_embedding_experiment.py) use noise_std=0.30 for HybridVerifier,
but that value was chosen to make HybridVerifier *fail sometimes* so FAGS/
beam-search comparisons stay interesting - it's a difficulty knob, not a
reflection of the verifier's true discriminative power. A first pass at
noise_std=0.30 showed a ~27-point rank-1 DROP for HybridVerifier; a quick
matched-noise check (noise_std=0.08 for both) showed the gap nearly vanish
(56.66% vs 56.00%), confirming the drop was almost entirely a noise-level
artifact, not a real signal-quality difference. Use noise_std=0.08 here so
this diagnostic measures signal quality, not difficulty-tuning.
"""

from __future__ import annotations

import os

os.environ["HF_HUB_OFFLINE"] = "1"

import csv
import numpy as np
import matplotlib.pyplot as plt

from fags.graph_generator import generate_dataset
from fags.verifier import HybridVerifier

RESULTS_DIR = r"d:\Projects\DemoSearch\results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Reference numbers from the rule-based Verifier's histogram (post
# reproducibility-fix regeneration, see PROJECT_STATUS.md SS4 / commit
# "Regenerate canonical results under the fixed deterministic generator").
RULE_BASED_REFERENCE = {
    "Small": [59.76, 18.59, 9.41, 12.24],
    "Medium": [57.38, 19.27, 9.51, 13.84],
    "Large": [60.19, 17.87, 9.06, 12.88],
}


def run_rank_diagnostic():
    sizes = {"Small": 20, "Medium": 100, "Large": 1000}
    query_count = 500
    seed = 101

    print("=" * 50)
    print("GOLD RELATION RANK DIAGNOSTIC - HYBRIDVERIFIER")
    print("=" * 50)

    size_histograms = {}

    for size_label, num_nodes in sizes.items():
        print(f"\nProcessing {size_label} Graph ({num_nodes} nodes)...")
        graph, queries = generate_dataset(num_nodes=num_nodes, num_queries=query_count, seed=seed)
        if size_label == "Small":
            print("Loading HybridVerifier (BAAI/bge-small-en-v1.5 + rule-based)...")
        verifier = HybridVerifier(model_name="BAAI/bge-small-en-v1.5", alpha=0.5, noise_std=0.08, seed=seed)

        rank_counts = np.zeros(4)
        total_evaluations = 0

        for q in queries:
            gold_nodes = q.gold_path
            gold_rels = q.gold_relations

            for depth in range(len(gold_nodes) - 1):
                current_node = gold_nodes[depth]
                gold_relation = gold_rels[depth]

                neighbors = graph.get_neighbors(current_node)
                if not neighbors:
                    continue

                path_relations_so_far = gold_rels[:depth]
                scored_edges = []
                for edge in neighbors:
                    score = verifier.score(q.keywords, edge, path_relations_so_far)
                    scored_edges.append((score, edge))

                scored_edges.sort(key=lambda x: x[0], reverse=True)

                gold_rank = -1
                for rank_idx, (_, edge) in enumerate(scored_edges):
                    if edge.relation == gold_relation:
                        gold_rank = rank_idx
                        break

                if gold_rank == -1:
                    continue

                total_evaluations += 1
                if gold_rank == 0:
                    rank_counts[0] += 1
                elif gold_rank == 1:
                    rank_counts[1] += 1
                elif gold_rank == 2:
                    rank_counts[2] += 1
                else:
                    rank_counts[3] += 1

        rank_percentages = (rank_counts / total_evaluations * 100) if total_evaluations > 0 else np.zeros(4)
        size_histograms[size_label] = {
            "counts": rank_counts.tolist(),
            "percentages": rank_percentages.tolist(),
            "total": total_evaluations,
        }

        print(f"Total Decision Points Evaluated: {total_evaluations}")
        print(f"  Rank 1 : {rank_percentages[0]:.2f}% ({int(rank_counts[0])})")
        print(f"  Rank 2 : {rank_percentages[1]:.2f}% ({int(rank_counts[1])})")
        print(f"  Rank 3 : {rank_percentages[2]:.2f}% ({int(rank_counts[2])})")
        print(f"  Rank 4+: {rank_percentages[3]:.2f}% ({int(rank_counts[3])})")

    # ── CSV ──
    csv_path = os.path.join(RESULTS_DIR, "gold_rank_histogram_hybrid_table.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Graph Size", "Verifier", "Rank 1 %", "Rank 2 %", "Rank 3 %", "Rank 4+ %", "Total Decision Points"])
        for size_label in sizes:
            p = size_histograms[size_label]["percentages"]
            writer.writerow([size_label, "HybridVerifier", f"{p[0]:.2f}", f"{p[1]:.2f}", f"{p[2]:.2f}", f"{p[3]:.2f}", size_histograms[size_label]["total"]])
            rp = RULE_BASED_REFERENCE[size_label]
            writer.writerow([size_label, "Rule-based (reference)", f"{rp[0]:.2f}", f"{rp[1]:.2f}", f"{rp[2]:.2f}", f"{rp[3]:.2f}", ""])
    print(f"\nTable written to {csv_path}")

    # ── Plot: HybridVerifier vs rule-based reference, grouped by graph size ──
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5), sharey=True)
    x_labels = ["Rank 1", "Rank 2", "Rank 3", "Rank 4+"]
    x = np.arange(len(x_labels))
    width = 0.35

    for ax, size_label in zip(axes, sizes):
        hybrid_p = size_histograms[size_label]["percentages"]
        rule_p = RULE_BASED_REFERENCE[size_label]
        ax.bar(x - width / 2, rule_p, width, label="Rule-based (reference)", color="gray")
        ax.bar(x + width / 2, hybrid_p, width, label="HybridVerifier", color="crimson")
        ax.set_title(size_label)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels)
        ax.set_ylabel("Frequency (%)")
        ax.grid(axis="y", linestyle=":", alpha=0.6)
    axes[0].legend(fontsize=9)
    plt.suptitle("Gold Relation Rank: HybridVerifier vs Rule-based Reference")
    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, "gold_rank_histogram_hybrid.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Plot written to {plot_path}")

    # ── Summary ──
    summary_path = os.path.join(RESULTS_DIR, "gold_rank_histogram_hybrid_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as rf:
        rf.write("=" * 50 + "\n")
        rf.write("GOLD RELATION RANK DIAGNOSTIC - HYBRIDVERIFIER vs RULE-BASED\n")
        rf.write("=" * 50 + "\n\n")
        rf.write(
            "Re-measures the SS4 gold-rank ceiling (rule-based Verifier: ~57-60%\n"
            "rank-1) using HybridVerifier (rule + BAAI/bge-small-en-v1.5 embeddings,\n"
            "alpha=0.5), to quantify how much the embedding signal raises the\n"
            "discriminative ceiling every search algorithm in this project inherits.\n\n"
        )
        for size_label in sizes:
            hybrid_p = size_histograms[size_label]["percentages"]
            rule_p = RULE_BASED_REFERENCE[size_label]
            rf.write(f"-- {size_label} graph --\n")
            rf.write(f"  Rule-based:     Rank1={rule_p[0]:.2f}%  Rank2={rule_p[1]:.2f}%  Rank3={rule_p[2]:.2f}%  Rank4+={rule_p[3]:.2f}%\n")
            rf.write(f"  HybridVerifier: Rank1={hybrid_p[0]:.2f}%  Rank2={hybrid_p[1]:.2f}%  Rank3={hybrid_p[2]:.2f}%  Rank4+={hybrid_p[3]:.2f}%\n")
            rf.write(f"  Rank-1 change: {hybrid_p[0] - rule_p[0]:+.2f} percentage points\n\n")

        avg_change = np.mean([size_histograms[s]["percentages"][0] - RULE_BASED_REFERENCE[s][0] for s in sizes])
        rf.write("CONCLUSION\n----------\n")
        if avg_change > 3:
            rf.write(
                f"HybridVerifier raises the rank-1 ceiling by an average of {avg_change:+.2f}\n"
                "percentage points across graph sizes - a real, meaningful improvement in\n"
                "discriminative power, not just noise. This is consistent with HybridVerifier\n"
                "producing higher raw accuracy in every comparison it's been used in\n"
                "(SS7, SS11) - the rank ceiling itself moves, not just downstream search\n"
                "outcomes.\n"
            )
        elif avg_change > 0:
            rf.write(
                f"HybridVerifier raises the rank-1 ceiling only marginally (avg "
                f"{avg_change:+.2f} points) - most of its accuracy benefit in other\n"
                "experiments likely comes from a cleaner SCORE GAP at a given rank\n"
                "(less noisy ordering among top candidates) rather than from ranking the\n"
                "gold relation #1 more often outright.\n"
            )
        else:
            rf.write(
                f"HybridVerifier does NOT raise the rank-1 ceiling (avg {avg_change:+.2f}\n"
                "points) despite producing higher downstream search accuracy in SS7/SS11.\n"
                "This would mean its benefit comes entirely from elsewhere (e.g. a cleaner\n"
                "score gap reducing noise-driven misranking at the search level) rather\n"
                "than from genuinely ranking gold relations higher.\n"
            )

    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    run_rank_diagnostic()
