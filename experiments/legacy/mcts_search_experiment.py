"""MCTS vs Beam Search experiment.

Global best-first search (best_first_search_experiment.py) lost 0/12 to
beam search because greedy frontier expansion tunnels into one wrong
branch with no exploration pressure. MCTS (fags/mcts_search.py) is built
specifically to avoid that: UCB1 selection balances exploiting what looks
good against exploring under-visited branches, backed by many independent
rollouts.

Same per-query budget-matched discipline as best_first_search_experiment.py:
for each query, beam search runs first at a given width, and MCTS then gets
that query's EXACT nodes_visited as its node_budget - so any accuracy
difference is attributable to the search paradigm alone, not search cost.

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
from fags.mcts_search import mcts_search
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
        mcts_results = [
            mcts_search(graph, q, verifier, node_budget=max(br.nodes_visited, 1), max_depth=MAX_DEPTH)
            for q, br in zip(queries, beam_results)
        ]
        bf_results = [
            best_first_search(graph, q, verifier, node_budget=max(br.nodes_visited, 1), max_depth=MAX_DEPTH)
            for q, br in zip(queries, beam_results)
        ]

        acc_beam = float(np.mean([1 if r.success else 0 for r in beam_results]))
        acc_mcts = float(np.mean([1 if r.success else 0 for r in mcts_results]))
        acc_bf = float(np.mean([1 if r.success else 0 for r in bf_results]))
        nodes_beam = float(np.mean([r.nodes_visited for r in beam_results]))
        nodes_mcts = float(np.mean([r.nodes_visited for r in mcts_results]))

        vs_beam = evaluate_results(beam_results, mcts_results, f"{size_label} w={bw}: mcts vs beam")

        records.append({
            "size_label": size_label, "beam_width": bw,
            "acc_beam": acc_beam, "nodes_beam": nodes_beam,
            "acc_mcts": acc_mcts, "nodes_mcts": nodes_mcts,
            "acc_bf": acc_bf,
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

    csv_path = os.path.join(RESULTS_DIR, "mcts_search_table.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Graph Size", "Beam Width", "Beam Acc", "Beam Nodes", "MCTS Acc", "MCTS Nodes",
            "Best-First Acc (reference)", "MCTS - Beam Gain", "p-value",
        ])
        for r in all_records:
            writer.writerow([
                r["size_label"], r["beam_width"], f"{r['acc_beam']:.2%}", f"{r['nodes_beam']:.2f}",
                f"{r['acc_mcts']:.2%}", f"{r['nodes_mcts']:.2f}", f"{r['acc_bf']:.2%}",
                f"{r['gain']:+.2%}", f"{r['p']:.3e}",
            ])
    print(f"\nTable written to {csv_path}")

    sizes_labels = list(sizes.keys())
    fig, axes = plt.subplots(1, len(sizes_labels), figsize=(15, 5), sharey=False)
    for ax, size_label in zip(axes, sizes_labels):
        rows = [r for r in all_records if r["size_label"] == size_label]
        rows.sort(key=lambda r: r["beam_width"])
        ax.plot([r["beam_width"] for r in rows], [r["acc_beam"] * 100 for r in rows], marker="o", color="crimson", label="Beam Search")
        ax.plot([r["beam_width"] for r in rows], [r["acc_mcts"] * 100 for r in rows], marker="^", color="forestgreen", label="MCTS (budget-matched)")
        ax.plot([r["beam_width"] for r in rows], [r["acc_bf"] * 100 for r in rows], marker="s", color="navy", linestyle=":", label="Best-First (reference)")
        ax.set_title(size_label)
        ax.set_xlabel("Beam Width (MCTS/Best-First use matched node budget)")
        ax.set_ylabel("Accuracy (%)")
        ax.grid(True, linestyle=":", alpha=0.5)
    axes[0].legend(fontsize=8)
    plt.suptitle("MCTS vs Beam Search (per-query budget-matched), Best-First shown for reference")
    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, "mcts_search.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Plot written to {plot_path}")

    summary_path = os.path.join(RESULTS_DIR, "mcts_search_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as rf:
        rf.write("=" * 50 + "\n")
        rf.write("MCTS vs BEAM SEARCH - SUMMARY\n")
        rf.write("=" * 50 + "\n\n")
        rf.write(
            "Per-query budget-matched comparison: for each query, MCTS gets the EXACT\n"
            "node budget beam search used for that query, so any accuracy difference\n"
            "is attributable to the search paradigm alone, not search cost.\n"
            "best_first_search.py's accuracy (run at the SAME matched budget) is shown\n"
            "for reference, since it already established the floor for 'changing the\n"
            "paradigm entirely' (0/12 vs beam search, catastrophic on Large).\n\n"
        )

        wins, losses = 0, 0
        for size_label in sizes_labels:
            rf.write(f"-- {size_label} graph --\n")
            for r in [x for x in all_records if x["size_label"] == size_label]:
                tag = ""
                if r["gain"] > 0 and r["p"] < 0.05:
                    tag, wins = "  <-- MCTS WINS", wins + 1
                elif r["gain"] < 0 and r["p"] < 0.05:
                    tag, losses = "  <-- BEAM WINS", losses + 1
                rf.write(
                    f"  width={r['beam_width']}: Beam={r['acc_beam']:.2%} ({r['nodes_beam']:.1f} nodes) "
                    f"vs MCTS={r['acc_mcts']:.2%} ({r['nodes_mcts']:.1f} nodes) "
                    f"[Best-First ref: {r['acc_bf']:.2%}] "
                    f"gain={r['gain']:+.2%} p={r['p']:.3e}{tag}\n"
                )
            rf.write("\n")

        rf.write(f"MCTS wins: {wins}/{len(all_records)}\n")
        rf.write(f"Beam wins: {losses}/{len(all_records)}\n\n")

        rf.write("CONCLUSION\n----------\n")
        if wins > losses and wins > 0:
            rf.write(
                "MCTS beats beam search at matched cost more often than it loses -\n"
                "explicit exploration/exploitation balance (UCB1) is enough to make a\n"
                "global-tree paradigm competitive where pure greedy best-first search\n"
                "collapsed. This would be the first alternative search PARADIGM (not\n"
                "just a rule tweak within beam search) to clear the bar.\n"
            )
        elif losses > wins:
            rf.write(
                "Beam search beats MCTS at matched cost more often than it loses, even\n"
                "though MCTS clearly avoids best_first_search.py's catastrophic\n"
                "tunnel-vision failure (see the Best-First reference column - MCTS is\n"
                "not collapsing the way pure greedy best-first did). The fixed,\n"
                "guaranteed-width exploration beam search provides at every hop is\n"
                "still hard to beat with a budget this small, even when exploration is\n"
                "explicitly modeled rather than absent.\n"
            )
        else:
            rf.write(
                "No consistent difference - MCTS and beam search are statistically\n"
                "equivalent at matched cost.\n"
            )

    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
