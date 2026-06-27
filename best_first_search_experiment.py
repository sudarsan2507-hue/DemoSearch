"""Global Best-First Search vs Beam Search experiment.

Four independent refinements on top of beam search's pruning RULE (hard
diversity cap, soft diversity penalty, sum aggregation, FPG composition)
all failed to beat plain beam search. This instead changes the pruning
PARADIGM: fags/best_first_search.py keeps one global priority queue across
all depths and always expands whichever hypothesis has the best score next,
until a total node-visit budget is spent - rather than keeping a fixed K
hypotheses alive at every depth uniformly.

Per-query budget-matched comparison (same discipline as
budget_matched_control_experiment.py): for each query, run beam search at a
given width, take its EXACT nodes_visited for that query, then run
best-first search with node_budget set to that exact value - so the two
algorithms get precisely the same per-query search cost, and any accuracy
difference is attributable to the pruning paradigm alone.

Same 3 graph sizes / 1000 queries / seed=101 as the canonical comparisons,
swept across beam widths {2, 3, 5, 8}.
"""

from __future__ import annotations

import os
import csv
import numpy as np
import matplotlib.pyplot as plt

from fags.graph_generator import generate_dataset
from fags.verifier import Verifier
from fags.beam_search import beam_search
from fags.best_first_search import best_first_search
from fags.evaluation import evaluate_results

RESULTS_DIR = r"d:\Projects\DemoSearch\results"
os.makedirs(RESULTS_DIR, exist_ok=True)

MAX_DEPTH = 6
BEAM_WIDTHS = [2, 3, 5, 8]


def run_size(size_label: str, num_nodes: int, query_count: int, seed: int) -> list[dict]:
    print(f"\n--- {size_label} graph ({num_nodes} nodes), {query_count} queries ---")
    graph, queries = generate_dataset(num_nodes=num_nodes, num_queries=query_count, seed=seed)
    verifier = Verifier(noise_std=0.08, seed=seed)

    records = []
    for bw in BEAM_WIDTHS:
        print(f"  beam_width={bw}...")
        beam_results = [beam_search(graph, q, verifier, beam_width=bw, max_depth=MAX_DEPTH) for q in queries]
        bf_results = [
            best_first_search(graph, q, verifier, node_budget=max(br.nodes_visited, 1), max_depth=MAX_DEPTH)
            for q, br in zip(queries, beam_results)
        ]

        acc_beam = float(np.mean([1 if r.success else 0 for r in beam_results]))
        acc_bf = float(np.mean([1 if r.success else 0 for r in bf_results]))
        nodes_beam = float(np.mean([r.nodes_visited for r in beam_results]))
        nodes_bf = float(np.mean([r.nodes_visited for r in bf_results]))

        vs_beam = evaluate_results(beam_results, bf_results, f"{size_label} w={bw}: best-first vs beam")

        records.append({
            "size_label": size_label, "beam_width": bw,
            "acc_beam": acc_beam, "nodes_beam": nodes_beam,
            "acc_bf": acc_bf, "nodes_bf": nodes_bf,
            "gain": vs_beam["accuracy_gain"], "p": vs_beam["p_value_accuracy"],
        })
    return records


def main():
    sizes = {"Small": 20, "Medium": 100, "Large": 1000}
    query_count = 1000
    seed = 101

    all_records = []
    for label, n in sizes.items():
        all_records.extend(run_size(label, n, query_count, seed))

    csv_path = os.path.join(RESULTS_DIR, "best_first_search_table.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Graph Size", "Beam Width", "Beam Acc", "Beam Nodes", "Best-First Acc", "Best-First Nodes",
            "Best-First - Beam Gain", "p-value",
        ])
        for r in all_records:
            writer.writerow([
                r["size_label"], r["beam_width"], f"{r['acc_beam']:.2%}", f"{r['nodes_beam']:.2f}",
                f"{r['acc_bf']:.2%}", f"{r['nodes_bf']:.2f}", f"{r['gain']:+.2%}", f"{r['p']:.3e}",
            ])
    print(f"\nTable written to {csv_path}")

    sizes_labels = list(sizes.keys())
    fig, axes = plt.subplots(1, len(sizes_labels), figsize=(15, 5), sharey=False)
    for ax, size_label in zip(axes, sizes_labels):
        rows = [r for r in all_records if r["size_label"] == size_label]
        rows.sort(key=lambda r: r["beam_width"])
        ax.plot([r["beam_width"] for r in rows], [r["acc_beam"] * 100 for r in rows], marker="o", color="crimson", label="Beam Search")
        ax.plot([r["beam_width"] for r in rows], [r["acc_bf"] * 100 for r in rows], marker="s", color="navy", label="Best-First (budget-matched)")
        ax.set_title(size_label)
        ax.set_xlabel("Beam Width (best-first uses matched node budget)")
        ax.set_ylabel("Accuracy (%)")
        ax.grid(True, linestyle=":", alpha=0.5)
    axes[0].legend(fontsize=9)
    plt.suptitle("Global Best-First Search vs Beam Search (per-query budget-matched)")
    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, "best_first_search.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Plot written to {plot_path}")

    summary_path = os.path.join(RESULTS_DIR, "best_first_search_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as rf:
        rf.write("=" * 50 + "\n")
        rf.write("GLOBAL BEST-FIRST SEARCH vs BEAM SEARCH - SUMMARY\n")
        rf.write("=" * 50 + "\n\n")
        rf.write(
            "Per-query budget-matched comparison: for each query, best-first search\n"
            "gets the EXACT node budget beam search used for that query, so any\n"
            "accuracy difference is attributable to the pruning paradigm (global\n"
            "frontier vs fixed-width-per-hop) alone, not search cost.\n\n"
        )

        wins, losses = 0, 0
        for size_label in sizes_labels:
            rf.write(f"-- {size_label} graph --\n")
            for r in [x for x in all_records if x["size_label"] == size_label]:
                tag = ""
                if r["gain"] > 0 and r["p"] < 0.05:
                    tag, wins = "  <-- BEST-FIRST WINS", wins + 1
                elif r["gain"] < 0 and r["p"] < 0.05:
                    tag, losses = "  <-- BEAM WINS", losses + 1
                rf.write(
                    f"  width={r['beam_width']}: Beam={r['acc_beam']:.2%} ({r['nodes_beam']:.1f} nodes) "
                    f"vs Best-First={r['acc_bf']:.2%} ({r['nodes_bf']:.1f} nodes) "
                    f"gain={r['gain']:+.2%} p={r['p']:.3e}{tag}\n"
                )
            rf.write("\n")

        rf.write(f"Best-first wins: {wins}/{len(all_records)}\n")
        rf.write(f"Beam wins: {losses}/{len(all_records)}\n\n")

        rf.write("CONCLUSION\n----------\n")
        if wins > losses and wins > 0:
            rf.write(
                "Global best-first search beats beam search at matched cost more often\n"
                "than it loses - concentrating budget on the best-scoring frontier\n"
                "instead of spreading it evenly across a fixed width is a real\n"
                "improvement, the first pruning-paradigm change (rather than a rule\n"
                "tweak within beam search) to clear this bar.\n"
            )
        elif losses > wins:
            rf.write(
                "Beam search beats global best-first search at matched cost more often\n"
                "than it loses. Keeping a fixed, diverse width alive at every depth\n"
                "apparently matters more than concentrating budget on the single\n"
                "best-looking frontier node - best-first search's greedy concentration\n"
                "risks tunnel vision on a locally-strong but ultimately wrong\n"
                "hypothesis, burning the whole budget on it before the queue\n"
                "naturally surfaces an alternative.\n"
            )
        else:
            rf.write(
                "No consistent difference - global best-first search and beam search\n"
                "are statistically equivalent at matched cost; the pruning paradigm\n"
                "doesn't matter as much as just having the same budget to spend.\n"
            )

    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
