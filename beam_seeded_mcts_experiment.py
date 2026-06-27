"""Beam-Seeded MCTS vs plain Beam Search vs pure MCTS.

mcts_search_experiment.py found pure MCTS losing to beam search at small
budgets (5-40 nodes) because its early simulations are mostly spent
rediscovering diversity beam search gets for free; mcts_large_budget_experiment.py
found MCTS only catches up once given ~5-10x more budget (200+ nodes).
fags/beam_seeded_mcts_search.py tries to close that gap by spending the
first `seed_depth` hops on plain beam-search-style expansion (guaranteed
breadth) before handing off to MCTS/UCB1 for the remaining hops - the
hypothesis is that this should let MCTS's value-learning pay off at much
smaller budgets than 200+ nodes, since the diversity-discovery cost is
sidestepped by construction.

Same per-query budget-matched discipline as mcts_search_experiment.py: for
each query, beam search runs first at a given width, and both pure MCTS
and the hybrid then get that query's EXACT nodes_visited as their
node_budget. Widths {2,3,5,8,15,25,40} - the small-to-moderate range where
pure MCTS previously lost - across all 3 graph sizes, 500 queries,
seed=101.
"""

from __future__ import annotations

import os
import csv
import numpy as np
import matplotlib.pyplot as plt

from fags.graph_generator import generate_dataset
from fags.verifier import Verifier
from fags.beam_search import beam_search
from fags.mcts_search import mcts_search
from fags.beam_seeded_mcts_search import beam_seeded_mcts_search
from fags.evaluation import evaluate_results

RESULTS_DIR = r"d:\Projects\DemoSearch\results"
os.makedirs(RESULTS_DIR, exist_ok=True)

MAX_DEPTH = 6
QUERY_COUNT = 500
SEED = 101
BEAM_WIDTHS = [2, 3, 5, 8, 15, 25, 40]
SEED_DEPTH = 2
HYBRID_BEAM_WIDTH = 5


def run_size(size_label: str, num_nodes: int) -> list[dict]:
    print(f"\n--- {size_label} graph ({num_nodes} nodes), {QUERY_COUNT} queries ---")
    graph, queries = generate_dataset(num_nodes=num_nodes, num_queries=QUERY_COUNT, seed=SEED)
    verifier = Verifier(noise_std=0.08, seed=SEED)

    records = []
    for bw in BEAM_WIDTHS:
        print(f"  beam_width={bw}...")
        beam_results = [beam_search(graph, q, verifier, beam_width=bw, max_depth=MAX_DEPTH) for q in queries]
        mcts_results = [
            mcts_search(graph, q, verifier, node_budget=max(br.nodes_visited, 1), max_depth=MAX_DEPTH)
            for q, br in zip(queries, beam_results)
        ]
        hybrid_results = [
            beam_seeded_mcts_search(
                graph, q, verifier, node_budget=max(br.nodes_visited, 1), max_depth=MAX_DEPTH,
                seed_depth=SEED_DEPTH, beam_width=HYBRID_BEAM_WIDTH,
            )
            for q, br in zip(queries, beam_results)
        ]

        acc_beam = float(np.mean([1 if r.success else 0 for r in beam_results]))
        acc_mcts = float(np.mean([1 if r.success else 0 for r in mcts_results]))
        acc_hybrid = float(np.mean([1 if r.success else 0 for r in hybrid_results]))
        nodes_beam = float(np.mean([r.nodes_visited for r in beam_results]))
        nodes_hybrid = float(np.mean([r.nodes_visited for r in hybrid_results]))

        vs_beam = evaluate_results(beam_results, hybrid_results, f"{size_label} w={bw}: hybrid vs beam")
        vs_mcts = evaluate_results(mcts_results, hybrid_results, f"{size_label} w={bw}: hybrid vs mcts")

        records.append({
            "size_label": size_label, "beam_width": bw,
            "acc_beam": acc_beam, "nodes_beam": nodes_beam,
            "acc_mcts": acc_mcts,
            "acc_hybrid": acc_hybrid, "nodes_hybrid": nodes_hybrid,
            "gain_vs_beam": vs_beam["accuracy_gain"], "p_vs_beam": vs_beam["p_value_accuracy"],
            "gain_vs_mcts": vs_mcts["accuracy_gain"], "p_vs_mcts": vs_mcts["p_value_accuracy"],
        })
    return records


def main():
    sizes = {"Small": 20, "Medium": 100, "Large": 1000}
    all_records = []
    for label, n in sizes.items():
        all_records.extend(run_size(label, n))

    csv_path = os.path.join(RESULTS_DIR, "beam_seeded_mcts_table.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Graph Size", "Beam Width", "Beam Acc", "Beam Nodes", "MCTS Acc", "Hybrid Acc", "Hybrid Nodes",
            "Hybrid vs Beam Gain", "Hybrid vs Beam p", "Hybrid vs MCTS Gain", "Hybrid vs MCTS p",
        ])
        for r in all_records:
            writer.writerow([
                r["size_label"], r["beam_width"], f"{r['acc_beam']:.2%}", f"{r['nodes_beam']:.2f}",
                f"{r['acc_mcts']:.2%}", f"{r['acc_hybrid']:.2%}", f"{r['nodes_hybrid']:.2f}",
                f"{r['gain_vs_beam']:+.2%}", f"{r['p_vs_beam']:.3e}",
                f"{r['gain_vs_mcts']:+.2%}", f"{r['p_vs_mcts']:.3e}",
            ])
    print(f"\nTable written to {csv_path}")

    sizes_labels = list(sizes.keys())
    fig, axes = plt.subplots(1, len(sizes_labels), figsize=(15, 5))
    for ax, size_label in zip(axes, sizes_labels):
        rows = [r for r in all_records if r["size_label"] == size_label]
        rows.sort(key=lambda r: r["beam_width"])
        ax.plot([r["beam_width"] for r in rows], [r["acc_beam"] * 100 for r in rows], marker="o", color="crimson", label="Beam Search")
        ax.plot([r["beam_width"] for r in rows], [r["acc_mcts"] * 100 for r in rows], marker="^", color="forestgreen", label="Pure MCTS")
        ax.plot([r["beam_width"] for r in rows], [r["acc_hybrid"] * 100 for r in rows], marker="s", color="purple", label="Beam-Seeded MCTS")
        ax.set_title(size_label)
        ax.set_xlabel("Beam Width (MCTS/Hybrid use matched node budget)")
        ax.set_ylabel("Accuracy (%)")
        ax.grid(True, linestyle=":", alpha=0.5)
    axes[0].legend(fontsize=8)
    plt.suptitle("Beam-Seeded MCTS vs Beam Search vs Pure MCTS (per-query budget-matched)")
    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, "beam_seeded_mcts.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Plot written to {plot_path}")

    summary_path = os.path.join(RESULTS_DIR, "beam_seeded_mcts_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as rf:
        rf.write("=" * 50 + "\n")
        rf.write("BEAM-SEEDED MCTS vs BEAM SEARCH vs PURE MCTS - SUMMARY\n")
        rf.write("=" * 50 + "\n\n")
        rf.write(
            "Per-query budget-matched comparison: MCTS and the hybrid both get each\n"
            f"query's EXACT node budget from beam search at a given width. Hybrid uses\n"
            f"seed_depth={SEED_DEPTH}, beam_width={HYBRID_BEAM_WIDTH} for its beam-seeding phase.\n"
            "Tests whether beam-seeding lets MCTS's advantage show up at budgets well\n"
            "below the ~200-300 node crossover found for pure MCTS.\n\n"
        )

        wins_vs_beam, losses_vs_beam = 0, 0
        wins_vs_mcts, losses_vs_mcts = 0, 0
        for size_label in sizes_labels:
            rf.write(f"-- {size_label} graph --\n")
            for r in [x for x in all_records if x["size_label"] == size_label]:
                tag_beam = ""
                if r["gain_vs_beam"] > 0 and r["p_vs_beam"] < 0.05:
                    tag_beam, wins_vs_beam = " <-- HYBRID BEATS BEAM", wins_vs_beam + 1
                elif r["gain_vs_beam"] < 0 and r["p_vs_beam"] < 0.05:
                    tag_beam, losses_vs_beam = " <-- BEAM WINS", losses_vs_beam + 1
                tag_mcts = ""
                if r["gain_vs_mcts"] > 0 and r["p_vs_mcts"] < 0.05:
                    tag_mcts, wins_vs_mcts = " <-- HYBRID BEATS MCTS", wins_vs_mcts + 1
                elif r["gain_vs_mcts"] < 0 and r["p_vs_mcts"] < 0.05:
                    tag_mcts, losses_vs_mcts = " <-- MCTS WINS", losses_vs_mcts + 1
                rf.write(
                    f"  width={r['beam_width']}: Beam={r['acc_beam']:.2%} ({r['nodes_beam']:.1f} nodes) | "
                    f"MCTS={r['acc_mcts']:.2%} | Hybrid={r['acc_hybrid']:.2%} ({r['nodes_hybrid']:.1f} nodes)\n"
                    f"    vs Beam: gain={r['gain_vs_beam']:+.2%} p={r['p_vs_beam']:.2e}{tag_beam}\n"
                    f"    vs MCTS: gain={r['gain_vs_mcts']:+.2%} p={r['p_vs_mcts']:.2e}{tag_mcts}\n"
                )
            rf.write("\n")

        rf.write(f"Hybrid beats Beam: {wins_vs_beam}/{len(all_records)} | Beam beats Hybrid: {losses_vs_beam}/{len(all_records)}\n")
        rf.write(f"Hybrid beats MCTS: {wins_vs_mcts}/{len(all_records)} | MCTS beats Hybrid: {losses_vs_mcts}/{len(all_records)}\n\n")

        rf.write("CONCLUSION\n----------\n")
        if wins_vs_beam > 0 and losses_vs_beam == 0:
            rf.write(
                "The hybrid beats plain beam search at small/moderate budgets where pure\n"
                "MCTS previously lost outright - beam-seeding successfully shifts the\n"
                "crossover point lower. This is the first composition attempt in this\n"
                "whole project to turn a loss into a win rather than just replicate an\n"
                "existing pattern or wash out to a null.\n"
            )
        elif wins_vs_beam > losses_vs_beam:
            rf.write(
                "The hybrid beats beam search more often than it loses, but not\n"
                "uniformly - a partial win, better than pure MCTS's complete loss at\n"
                "these budgets but not a clean across-the-board result.\n"
            )
        else:
            rf.write(
                "The hybrid does not beat beam search at these budgets either. Compare\n"
                "against the 'vs MCTS' rows: if the hybrid is statistically indistinguishable\n"
                "from pure MCTS, beam-seeding didn't change MCTS's fundamental budget\n"
                "requirement, just gave it a (possibly redundant) head start that gets\n"
                "overtaken by the same dynamics that made pure MCTS need ~200+ nodes in\n"
                "the first place.\n"
            )

    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
