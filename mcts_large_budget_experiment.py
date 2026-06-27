"""Does MCTS catch up to beam search at larger node budgets?

mcts_search_experiment.py tested budgets of 5-40 nodes (matched to beam
widths 2-8) and found MCTS losing 10/12 configurations, though it clearly
fixed best_first_search.py's tunnel-vision collapse. The standing
hypothesis was that MCTS needs far more simulations than a 5-40 node
budget gives it to pay off - most of that budget gets spent on
selection/expansion bookkeeping near the root rather than informative
rollouts. This tests that directly by sweeping both algorithms across a
much wider, independent range of budgets (up to 1280 nodes - on the Large
graph that's a sizeable fraction of the whole reachable graph) and plotting
accuracy vs mean nodes visited for both, to see whether the curves
converge, cross, or stay parallel.

Not per-query budget-matched this time (that discipline matters for tight
paired significance tests at a few comparable budget points; for a coarse
"where do these curves sit relative to each other" question, independent
sweeps plotted on a shared cost axis are simpler and sufficient). Medium
and Large graphs only - Small (20 nodes) has no room for "large" budgets to
mean anything. 500 queries (not 1000) to keep runtime reasonable given
MCTS's added simulation overhead at the high end of this range.
"""

from __future__ import annotations

import os
import csv
import time
import numpy as np
import matplotlib.pyplot as plt

from fags.graph_generator import generate_dataset
from fags.verifier import Verifier
from fags.beam_search import beam_search
from fags.mcts_search import mcts_search

RESULTS_DIR = r"d:\Projects\DemoSearch\results"
os.makedirs(RESULTS_DIR, exist_ok=True)

MAX_DEPTH = 6
QUERY_COUNT = 500
SEED = 101
BEAM_WIDTHS = [2, 3, 5, 8, 15, 25, 40, 60]
MCTS_BUDGETS = [10, 20, 40, 80, 160, 320, 640, 1280]


def run_size(size_label: str, num_nodes: int) -> dict:
    print(f"\n--- {size_label} graph ({num_nodes} nodes), {QUERY_COUNT} queries ---")
    graph, queries = generate_dataset(num_nodes=num_nodes, num_queries=QUERY_COUNT, seed=SEED)
    verifier = Verifier(noise_std=0.08, seed=SEED)

    beam_points = []
    for bw in BEAM_WIDTHS:
        t0 = time.time()
        results = [beam_search(graph, q, verifier, beam_width=bw, max_depth=MAX_DEPTH) for q in queries]
        acc = float(np.mean([1 if r.success else 0 for r in results]))
        nodes = float(np.mean([r.nodes_visited for r in results]))
        print(f"  beam width={bw}: acc={acc:.2%} nodes={nodes:.1f} ({time.time()-t0:.1f}s)")
        beam_points.append({"param": bw, "acc": acc, "nodes": nodes})

    mcts_points = []
    for budget in MCTS_BUDGETS:
        t0 = time.time()
        results = [mcts_search(graph, q, verifier, node_budget=budget, max_depth=MAX_DEPTH) for q in queries]
        acc = float(np.mean([1 if r.success else 0 for r in results]))
        nodes = float(np.mean([r.nodes_visited for r in results]))
        print(f"  MCTS budget={budget}: acc={acc:.2%} nodes={nodes:.1f} ({time.time()-t0:.1f}s)")
        mcts_points.append({"param": budget, "acc": acc, "nodes": nodes})

    return {"size_label": size_label, "beam_points": beam_points, "mcts_points": mcts_points}


def main():
    sizes = {"Medium": 100, "Large": 1000}
    size_results = [run_size(label, n) for label, n in sizes.items()]

    csv_path = os.path.join(RESULTS_DIR, "mcts_large_budget_table.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Graph Size", "Algorithm", "Param (width or budget)", "Accuracy", "Mean Nodes"])
        for sr in size_results:
            for p in sr["beam_points"]:
                writer.writerow([sr["size_label"], "Beam", p["param"], f"{p['acc']:.2%}", f"{p['nodes']:.2f}"])
            for p in sr["mcts_points"]:
                writer.writerow([sr["size_label"], "MCTS", p["param"], f"{p['acc']:.2%}", f"{p['nodes']:.2f}"])
    print(f"\nTable written to {csv_path}")

    fig, axes = plt.subplots(1, len(size_results), figsize=(13, 5.5))
    if len(size_results) == 1:
        axes = [axes]
    for ax, sr in zip(axes, size_results):
        bx = [p["nodes"] for p in sr["beam_points"]]
        by = [p["acc"] * 100 for p in sr["beam_points"]]
        mx = [p["nodes"] for p in sr["mcts_points"]]
        my = [p["acc"] * 100 for p in sr["mcts_points"]]
        ax.plot(bx, by, marker="o", color="crimson", label="Beam Search")
        ax.plot(mx, my, marker="^", color="forestgreen", label="MCTS")
        ax.set_xscale("log")
        ax.set_title(sr["size_label"])
        ax.set_xlabel("Mean Nodes Visited (log scale)")
        ax.set_ylabel("Accuracy (%)")
        ax.grid(True, linestyle=":", alpha=0.5)
    axes[0].legend(fontsize=9)
    plt.suptitle("Beam Search vs MCTS Across a Wide Budget Range")
    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, "mcts_large_budget.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Plot written to {plot_path}")

    summary_path = os.path.join(RESULTS_DIR, "mcts_large_budget_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as rf:
        rf.write("=" * 50 + "\n")
        rf.write("MCTS vs BEAM SEARCH ACROSS A WIDE BUDGET RANGE - SUMMARY\n")
        rf.write("=" * 50 + "\n\n")
        rf.write(
            "Independent sweeps (not per-query budget-matched) plotted on a shared\n"
            "mean-nodes-visited axis, to see whether MCTS's accuracy curve catches up\n"
            "to or crosses beam search's as budget grows, given the standing theory\n"
            "that MCTS needs far more than the 5-40 node budgets tested previously to\n"
            "pay off.\n\n"
        )
        for sr in size_results:
            rf.write(f"-- {sr['size_label']} graph --\n")
            rf.write("  Beam Search:\n")
            for p in sr["beam_points"]:
                rf.write(f"    width={p['param']}: {p['acc']:.2%} @ {p['nodes']:.1f} nodes\n")
            rf.write("  MCTS:\n")
            for p in sr["mcts_points"]:
                rf.write(f"    budget={p['param']}: {p['acc']:.2%} @ {p['nodes']:.1f} nodes\n")

            # Find the gap at roughly-matched cost points (nearest neighbor by nodes)
            rf.write("  Approximate gap at similar cost (nearest-neighbor matching, descriptive only):\n")
            for p in sr["mcts_points"]:
                nearest_beam = min(sr["beam_points"], key=lambda b: abs(b["nodes"] - p["nodes"]))
                gap = p["acc"] - nearest_beam["acc"]
                rf.write(
                    f"    MCTS@{p['nodes']:.0f} nodes ({p['acc']:.2%}) vs "
                    f"Beam@{nearest_beam['nodes']:.0f} nodes ({nearest_beam['acc']:.2%}): gap={gap:+.2%}\n"
                )
            rf.write("\n")

        rf.write("CONCLUSION\n----------\n")
        any_crossed = False
        for sr in size_results:
            gaps = []
            for p in sr["mcts_points"]:
                nearest_beam = min(sr["beam_points"], key=lambda b: abs(b["nodes"] - p["nodes"]))
                gaps.append(p["acc"] - nearest_beam["acc"])
            if max(gaps) > 0:
                any_crossed = True
                rf.write(f"{sr['size_label']}: MCTS closes the gap and surpasses beam search at high budgets "
                          f"(best gap {max(gaps):+.2%}).\n")
            else:
                rf.write(f"{sr['size_label']}: MCTS never catches up even at the highest budget tested "
                          f"(best gap {max(gaps):+.2%}).\n")
        rf.write("\n")
        if any_crossed:
            rf.write(
                "MCTS's disadvantage shrinks (or reverses) with enough budget - the\n"
                "5-40 node range tested previously was too small a budget regime for it,\n"
                "not a fundamental loss to beam search's paradigm.\n"
            )
        else:
            rf.write(
                "MCTS's gap to beam search persists across two orders of magnitude of\n"
                "budget (up to 1280 nodes) - not just an artifact of testing too small a\n"
                "budget range. Beam search's fixed-width guarantee remains ahead\n"
                "regardless of how much budget either algorithm gets.\n"
            )

    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
