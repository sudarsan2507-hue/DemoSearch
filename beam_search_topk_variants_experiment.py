"""Top-K ranking-rule variants for beam search.

diverse_beam_search_experiment.py and beam_search_fpg_experiment.py both left
the ranking rule itself untouched (cumulative mean per-hop verifier score) -
they only constrained or nudged who gets picked under that same rule. This
tests two genuinely different rules, added to fags/beam_search.py:

  - score_aggregation="sum": ranks hypotheses by total accumulated score
    instead of the mean (the convention classic NLP beam search uses -
    log-prob sum), which inherently rewards longer paths with more
    accumulated evidence rather than treating any-length paths as
    comparable on their average.
  - diversity_penalty_weight>0: a SOFT version of the hard
    max_children_per_parent cap from diverse_beam_search_experiment.py -
    greedy iterative selection where a crowded parent's later candidates
    sink gradually (proportional to the penalty) instead of being banned
    outright once a quota is hit.

Cost-neutral comparison (same beam width) vs the established plain-beam
baseline (mean aggregation, no diversity penalty) - same 3 graph sizes /
1000 queries / seed=101 as every other canonical comparison in this repo.
"""

from __future__ import annotations

import os
import csv
import matplotlib.pyplot as plt
import numpy as np

from fags.graph_generator import generate_dataset
from fags.verifier import Verifier
from fags.beam_search import beam_search
from fags.evaluation import evaluate_results

RESULTS_DIR = r"d:\Projects\DemoSearch\results"
os.makedirs(RESULTS_DIR, exist_ok=True)

MAX_DEPTH = 6
BEAM_WIDTHS = [5, 8]
DIVERSITY_PENALTIES = [0.0, 0.05, 0.1, 0.2]
AGGREGATIONS = ["mean", "sum"]


def run_size(size_label: str, num_nodes: int, query_count: int, seed: int) -> list[dict]:
    print(f"\n--- {size_label} graph ({num_nodes} nodes), {query_count} queries ---")
    graph, queries = generate_dataset(num_nodes=num_nodes, num_queries=query_count, seed=seed)
    verifier = Verifier(noise_std=0.08, seed=seed)

    records = []
    for bw in BEAM_WIDTHS:
        baseline_results = [
            beam_search(graph, q, verifier, beam_width=bw, max_depth=MAX_DEPTH,
                        score_aggregation="mean", diversity_penalty_weight=0.0)
            for q in queries
        ]
        acc_baseline = float(np.mean([1 if r.success else 0 for r in baseline_results]))
        nodes_baseline = float(np.mean([r.nodes_visited for r in baseline_results]))

        for agg in AGGREGATIONS:
            for dpw in DIVERSITY_PENALTIES:
                if agg == "mean" and dpw == 0.0:
                    results = baseline_results  # this IS the baseline config
                else:
                    print(f"  width={bw}, aggregation={agg}, diversity_penalty={dpw}...")
                    results = [
                        beam_search(graph, q, verifier, beam_width=bw, max_depth=MAX_DEPTH,
                                    score_aggregation=agg, diversity_penalty_weight=dpw)
                        for q in queries
                    ]
                acc = float(np.mean([1 if r.success else 0 for r in results]))
                nodes = float(np.mean([r.nodes_visited for r in results]))
                vs_baseline = evaluate_results(baseline_results, results, f"{size_label} w={bw} {agg}/{dpw} vs baseline")

                records.append({
                    "size_label": size_label, "beam_width": bw, "aggregation": agg, "diversity_penalty": dpw,
                    "acc": acc, "nodes": nodes, "acc_baseline": acc_baseline, "nodes_baseline": nodes_baseline,
                    "gain_vs_baseline": vs_baseline["accuracy_gain"], "p_vs_baseline": vs_baseline["p_value_accuracy"],
                })
    return records


def main():
    sizes = {"Small": 20, "Medium": 100, "Large": 1000}
    query_count = 1000
    seed = 101

    all_records = []
    for label, n in sizes.items():
        all_records.extend(run_size(label, n, query_count, seed))

    csv_path = os.path.join(RESULTS_DIR, "beam_search_topk_variants_table.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Graph Size", "Beam Width", "Aggregation", "Diversity Penalty", "Accuracy", "Mean Nodes",
            "Baseline Accuracy (mean, dpw=0)", "Gain vs Baseline", "p-value",
        ])
        for r in all_records:
            writer.writerow([
                r["size_label"], r["beam_width"], r["aggregation"], r["diversity_penalty"],
                f"{r['acc']:.2%}", f"{r['nodes']:.2f}", f"{r['acc_baseline']:.2%}",
                f"{r['gain_vs_baseline']:+.2%}", f"{r['p_vs_baseline']:.3e}",
            ])
    print(f"\nTable written to {csv_path}")

    sizes_labels = list(sizes.keys())
    fig, axes = plt.subplots(1, len(sizes_labels), figsize=(16, 5.5), sharey=False)
    for ax, size_label in zip(axes, sizes_labels):
        for bw in BEAM_WIDTHS:
            for agg in AGGREGATIONS:
                rows = [r for r in all_records if r["size_label"] == size_label and r["beam_width"] == bw and r["aggregation"] == agg]
                rows.sort(key=lambda r: r["diversity_penalty"])
                xs = [r["diversity_penalty"] for r in rows]
                ys = [r["acc"] * 100 for r in rows]
                ax.plot(xs, ys, marker="o", label=f"w={bw}, {agg}")
        ax.set_title(size_label)
        ax.set_xlabel("Diversity Penalty Weight")
        ax.set_ylabel("Accuracy (%)")
        ax.grid(True, linestyle=":", alpha=0.5)
    axes[0].legend(fontsize=8)
    plt.suptitle("Beam Search Top-K Variants: Sum vs Mean Aggregation, Soft Diversity Penalty")
    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, "beam_search_topk_variants.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Plot written to {plot_path}")

    summary_path = os.path.join(RESULTS_DIR, "beam_search_topk_variants_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as rf:
        rf.write("=" * 50 + "\n")
        rf.write("BEAM SEARCH TOP-K RANKING VARIANTS - SUMMARY\n")
        rf.write("=" * 50 + "\n\n")
        rf.write(
            "Tests two different ranking rules against the established plain-beam\n"
            "baseline (mean aggregation, no diversity penalty), at the SAME beam\n"
            "width (cost-neutral): score_aggregation='sum' vs 'mean', and a soft\n"
            "diversity_penalty_weight vs the hard max_children_per_parent cap\n"
            "tested previously.\n\n"
        )

        non_baseline = [r for r in all_records if not (r["aggregation"] == "mean" and r["diversity_penalty"] == 0.0)]

        rf.write("SUM vs MEAN (diversity_penalty=0, isolating aggregation only)\n")
        rf.write("-" * 60 + "\n")
        sum_wins, sum_losses = 0, 0
        for r in [x for x in non_baseline if x["aggregation"] == "sum" and x["diversity_penalty"] == 0.0]:
            tag = ""
            if r["gain_vs_baseline"] > 0 and r["p_vs_baseline"] < 0.05:
                tag, sum_wins = "  <-- SUM WINS", sum_wins + 1
            elif r["gain_vs_baseline"] < 0 and r["p_vs_baseline"] < 0.05:
                tag, sum_losses = "  <-- SUM LOSES", sum_losses + 1
            rf.write(f"  {r['size_label']} w={r['beam_width']}: {r['acc']:.2%} vs baseline {r['acc_baseline']:.2%} "
                      f"gain={r['gain_vs_baseline']:+.2%} p={r['p_vs_baseline']:.3e}{tag}\n")
        rf.write(f"  Sum aggregation: {sum_wins} significant wins, {sum_losses} significant losses (of 6 size x width configs)\n\n")

        rf.write("SOFT DIVERSITY PENALTY (mean aggregation, varying weight)\n")
        rf.write("-" * 60 + "\n")
        div_wins, div_losses = 0, 0
        for r in [x for x in non_baseline if x["aggregation"] == "mean" and x["diversity_penalty"] > 0.0]:
            tag = ""
            if r["gain_vs_baseline"] > 0 and r["p_vs_baseline"] < 0.05:
                tag, div_wins = "  <-- WINS", div_wins + 1
            elif r["gain_vs_baseline"] < 0 and r["p_vs_baseline"] < 0.05:
                tag, div_losses = "  <-- LOSES", div_losses + 1
            rf.write(f"  {r['size_label']} w={r['beam_width']} penalty={r['diversity_penalty']}: {r['acc']:.2%} vs baseline "
                      f"{r['acc_baseline']:.2%} gain={r['gain_vs_baseline']:+.2%} p={r['p_vs_baseline']:.3e}{tag}\n")
        rf.write(f"  Soft diversity penalty: {div_wins} significant wins, {div_losses} significant losses (of 18 configs)\n\n")

        rf.write("SUM + SOFT DIVERSITY PENALTY COMBINED\n")
        rf.write("-" * 60 + "\n")
        combo_wins, combo_losses = 0, 0
        for r in [x for x in non_baseline if x["aggregation"] == "sum" and x["diversity_penalty"] > 0.0]:
            tag = ""
            if r["gain_vs_baseline"] > 0 and r["p_vs_baseline"] < 0.05:
                tag, combo_wins = "  <-- WINS", combo_wins + 1
            elif r["gain_vs_baseline"] < 0 and r["p_vs_baseline"] < 0.05:
                tag, combo_losses = "  <-- LOSES", combo_losses + 1
            rf.write(f"  {r['size_label']} w={r['beam_width']} penalty={r['diversity_penalty']}: {r['acc']:.2%} vs baseline "
                      f"{r['acc_baseline']:.2%} gain={r['gain_vs_baseline']:+.2%} p={r['p_vs_baseline']:.3e}{tag}\n")
        rf.write(f"  Sum+diversity combined: {combo_wins} significant wins, {combo_losses} significant losses (of 18 configs)\n\n")

        rf.write("CONCLUSION\n----------\n")
        rf.write(
            "Report each axis separately rather than pooling win/loss counts - a\n"
            "pooled tally hides which idea (if any) actually did something:\n"
            f"  - Sum aggregation alone (the genuinely new idea): {sum_wins} wins, {sum_losses} losses (of 6).\n"
            f"  - Soft diversity penalty (mean aggregation): {div_wins} wins, {div_losses} losses (of 18).\n"
            f"  - Sum + soft diversity combined: {combo_wins} wins, {combo_losses} losses (of 18).\n\n"
        )
        if sum_wins == 0 and sum_losses == 0:
            rf.write(
                "Sum aggregation alone is a clean null - ranking by total accumulated\n"
                "score instead of average doesn't measurably change accuracy. Any\n"
                "wins/losses elsewhere come from the diversity-penalty axis, which (if\n"
                "present) should be cross-checked against diverse_beam_search_experiment.py's\n"
                "result before treating it as new: a soft penalty replicating the same\n"
                "hard-cap pattern (helps Small/Medium, hurts Large) is not a new finding.\n"
            )
        else:
            rf.write("Sum aggregation alone shows a real effect - inspect the per-size breakdown above.\n")

    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
