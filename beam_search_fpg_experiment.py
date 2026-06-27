"""Beam Search + Failure Pattern Graph experiment.

failure_pattern_graph_experiment.py found that composing the learned FPG
penalty with FAGS actively hurt it (significantly worse at penalty>=0.2):
suppressing "risky-looking" transitions also suppressed alternatives FAGS's
reactive memory needed for successful revival. Beam search doesn't have
that failure mode - it never relies on reviving one specific suppressed
candidate, since multiple hypotheses are already explored in parallel. So
the FPG's learned signal might instead help here, nudging which near-equal-
scoring hypotheses survive each prune, rather than conflicting with a
recovery mechanism that doesn't exist in this architecture.

Reuses fags.failure_pattern_graph.train_failure_pattern_graph and
PatternAwareVerifier unchanged - PatternAwareVerifier already implements
the same .score() interface beam_search() consumes, so composing them
needs no new core code.

Trains the FPG on one graph (seed=101), evaluates on a different held-out
graph (seed=202) - same train/test discipline as
failure_pattern_graph_experiment.py. Cost-neutral comparison (same beam
width) at widths {3, 5, 8} x penalty weights {0.0, 0.05, 0.1, 0.15, 0.2, 0.3}
- the lower end of this range is well below what hurt FAGS, to see whether
beam search tolerates (or even benefits from) penalties that broke FAGS.
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
from fags.failure_pattern_graph import train_failure_pattern_graph, PatternAwareVerifier

RESULTS_DIR = r"d:\Projects\DemoSearch\results"
os.makedirs(RESULTS_DIR, exist_ok=True)

NUM_NODES = 100
QUERY_COUNT = 1000
TRAIN_SEED = 101
TEST_SEED = 202
MAX_DEPTH = 6
BEAM_WIDTHS = [3, 5, 8]
PENALTY_WEIGHTS = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3]


def main():
    print(f"Generating TRAIN graph ({NUM_NODES} nodes, seed={TRAIN_SEED})...")
    train_graph, train_queries = generate_dataset(num_nodes=NUM_NODES, num_queries=QUERY_COUNT, seed=TRAIN_SEED)
    train_verifier = Verifier(noise_std=0.08, seed=TRAIN_SEED)

    print("Training Failure Pattern Graph on baseline-search outcomes...")
    fpg = train_failure_pattern_graph(train_graph, train_queries, train_verifier, max_depth=MAX_DEPTH)

    print(f"Generating held-out TEST graph ({NUM_NODES} nodes, seed={TEST_SEED})...")
    test_graph, test_queries = generate_dataset(num_nodes=NUM_NODES, num_queries=QUERY_COUNT, seed=TEST_SEED)
    test_verifier = Verifier(noise_std=0.08, seed=TEST_SEED)

    records = []
    for bw in BEAM_WIDTHS:
        print(f"\n--- beam_width={bw} ---")
        plain_results = [beam_search(test_graph, q, test_verifier, beam_width=bw, max_depth=MAX_DEPTH) for q in test_queries]
        acc_plain = float(np.mean([1 if r.success else 0 for r in plain_results]))
        nodes_plain = float(np.mean([r.nodes_visited for r in plain_results]))

        for pw in PENALTY_WEIGHTS:
            print(f"  penalty={pw}...")
            if pw == 0.0:
                results = plain_results
            else:
                wrapped_verifier = PatternAwareVerifier(test_verifier, fpg, penalty_weight=pw)
                results = [beam_search(test_graph, q, wrapped_verifier, beam_width=bw, max_depth=MAX_DEPTH) for q in test_queries]

            acc = float(np.mean([1 if r.success else 0 for r in results]))
            nodes = float(np.mean([r.nodes_visited for r in results]))
            vs_plain = evaluate_results(plain_results, results, f"w={bw} penalty={pw} vs plain")

            records.append({
                "beam_width": bw, "penalty": pw, "acc": acc, "nodes": nodes,
                "acc_plain": acc_plain, "nodes_plain": nodes_plain,
                "gain_vs_plain": vs_plain["accuracy_gain"], "p_vs_plain": vs_plain["p_value_accuracy"],
            })

    csv_path = os.path.join(RESULTS_DIR, "beam_search_fpg_table.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Beam Width", "Penalty Weight", "Accuracy", "Mean Nodes", "Plain Beam Accuracy",
            "Gain vs Plain Beam", "p-value",
        ])
        for r in records:
            writer.writerow([
                r["beam_width"], r["penalty"], f"{r['acc']:.2%}", f"{r['nodes']:.2f}",
                f"{r['acc_plain']:.2%}", f"{r['gain_vs_plain']:+.2%}", f"{r['p_vs_plain']:.3e}",
            ])
    print(f"\nTable written to {csv_path}")

    plt.figure(figsize=(8, 6))
    for bw in BEAM_WIDTHS:
        rows = [r for r in records if r["beam_width"] == bw]
        plt.plot([r["penalty"] for r in rows], [r["acc"] * 100 for r in rows], marker="o", label=f"width={bw}")
    plt.xlabel("Failure-Pattern Penalty Weight")
    plt.ylabel("Accuracy (%)")
    plt.title("Beam Search + Failure Pattern Graph: Accuracy vs Penalty Weight")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, "beam_search_fpg_sweep.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Plot written to {plot_path}")

    summary_path = os.path.join(RESULTS_DIR, "beam_search_fpg_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as rf:
        rf.write("=" * 50 + "\n")
        rf.write("BEAM SEARCH + FAILURE PATTERN GRAPH - SUMMARY\n")
        rf.write("=" * 50 + "\n\n")
        rf.write(
            "Composes the learned Failure Pattern Graph (trained on a held-out\n"
            "graph, seed=101) with beam search (tested on a different graph,\n"
            "seed=202), via the same PatternAwareVerifier wrapper that hurt FAGS in\n"
            "failure_pattern_graph_experiment.py. Cost-neutral comparison (same\n"
            "width) vs plain beam search.\n\n"
        )

        nonzero = [r for r in records if r["penalty"] > 0]
        sig_wins = sum(1 for r in nonzero if r["gain_vs_plain"] > 0 and r["p_vs_plain"] < 0.05)
        sig_losses = sum(1 for r in nonzero if r["gain_vs_plain"] < 0 and r["p_vs_plain"] < 0.05)

        for bw in BEAM_WIDTHS:
            rf.write(f"-- width={bw} (plain beam: {next(r['acc_plain'] for r in records if r['beam_width']==bw):.2%}) --\n")
            for r in [x for x in records if x["beam_width"] == bw and x["penalty"] > 0]:
                tag = ""
                if r["gain_vs_plain"] > 0 and r["p_vs_plain"] < 0.05:
                    tag = "  <-- FPG WINS"
                elif r["gain_vs_plain"] < 0 and r["p_vs_plain"] < 0.05:
                    tag = "  <-- FPG LOSES"
                rf.write(f"  penalty={r['penalty']}: {r['acc']:.2%} gain={r['gain_vs_plain']:+.2%} p={r['p_vs_plain']:.3e}{tag}\n")
            rf.write("\n")

        rf.write(f"Significant wins for FPG composition: {sig_wins}/{len(nonzero)}\n")
        rf.write(f"Significant losses for FPG composition: {sig_losses}/{len(nonzero)}\n\n")

        rf.write("CONCLUSION\n----------\n")
        if sig_wins > sig_losses and sig_wins > 0:
            rf.write(
                "Composing the learned failure-pattern signal with beam search helps\n"
                "more often than it hurts - unlike FAGS, where the same signal was\n"
                "actively harmful, beam search's lack of a single-candidate revival\n"
                "step lets the avoidance signal add value instead of conflicting with\n"
                "the search mechanism.\n"
            )
        elif sig_losses > sig_wins:
            rf.write(
                "Composing the learned failure-pattern signal with beam search hurts\n"
                "more often than it helps, similar in direction to the FAGS result\n"
                "(though beam search's architecture doesn't structurally conflict with\n"
                "it the way FAGS's revival did) - the avoidance signal isn't reliable\n"
                "enough to improve on the verifier's own ranking once it's already\n"
                "competing across multiple live hypotheses.\n"
            )
        else:
            rf.write(
                "No consistent effect - the failure-pattern penalty is statistically a\n"
                "wash when composed with beam search, neither helping nor hurting\n"
                "reliably across widths/penalty weights tested.\n"
            )

    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
