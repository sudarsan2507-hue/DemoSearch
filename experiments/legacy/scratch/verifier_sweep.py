"""Verifier Quality Sweep Diagnostic Script.

Simulates a verifier of exact target Rank-1 accuracy levels:
[50%, 60%, 70%, 80%, 90%, 95%]
by artificially overriding decision ranks with probability Q.

Runs Baseline and FAGS (Top-2) to measure accuracy gain,
gold path recovery rate, search cost, and efficiency ratios.
"""

from __future__ import annotations

import os
import csv
import random
import numpy as np
import matplotlib.pyplot as plt

from fags.graph_generator import generate_dataset
from fags.verifier import Verifier
from fags.memory import create_memory
from fags.baseline_search import baseline_search
from fags.failure_search import failure_search
from fags.evaluation import evaluate_results
from fags import KnowledgeGraph, Query, Edge

RESULTS_DIR = r"d:\Projects\DemoSearch\results"
os.makedirs(RESULTS_DIR, exist_ok=True)


class SimulatedVerifier(Verifier):
    """A verifier wrapper that artificially overrides scores to guarantee
    that the correct (gold) relation is Rank-1 with exactly probability Q.
    """

    def __init__(self, target_rank1: float, seed: int = 42) -> None:
        super().__init__(noise_std=0.0, seed=seed)
        self.target_rank1 = target_rank1
        self.sim_rng = random.Random(seed)

    def score_candidates(
        self,
        keywords: list[str],
        candidates: list[Edge],
        gold_relation: str | None,
        path_relations: list[str],
        failed_relations: list[str] | None = None,
        is_re_verification: bool = False,
    ) -> list[tuple[float, Edge]]:
        """Scores all candidates and forces gold relation to Rank-1 with probability Q."""
        # 1. Base rule-based scoring
        scored = []
        for edge in candidates:
            if is_re_verification:
                s = self.re_score(keywords, edge, path_relations, failed_relations)
            else:
                s = self.score(keywords, edge, path_relations)
            scored.append((s, edge))
        
        # Sort descending by score
        scored.sort(key=lambda x: x[0], reverse=True)

        if not gold_relation or not any(e.relation == gold_relation for _, e in scored):
            return scored

        # 2. Artificially enforce Rank-1 accuracy Q
        is_gold_rank1 = self.sim_rng.random() < self.target_rank1

        # Find current index of gold relation
        gold_idx = next(i for i, (_, e) in enumerate(scored) if e.relation == gold_relation)

        if is_gold_rank1:
            # Move gold relation to Rank-1
            if gold_idx > 0:
                gold_val = scored.pop(gold_idx)
                # Boost score slightly above the current winner to place it first
                boosted_score = min(1.0, scored[0][0] + 0.10)
                scored.insert(0, (boosted_score, gold_val[1]))
        else:
            # Enforce gold relation is NOT Rank-1
            if gold_idx == 0 and len(scored) > 1:
                # Demote gold relation to Rank-2 (or worse) by lowering its score below the second place
                gold_val = scored.pop(0)
                demoted_score = max(0.0, scored[0][0] - 0.10)
                scored.insert(1, (demoted_score, gold_val[1]))

        return scored


# ══════════════════════════════════════════════
# Search adaptations using SimulatedVerifier
# ══════════════════════════════════════════════

def run_simulated_baseline(
    graph: KnowledgeGraph,
    query: Query,
    verifier: SimulatedVerifier,
    max_depth: int = 10,
) -> SearchResult:
    """Execute baseline search using simulated verifier overrides."""
    import time
    from fags import FailureType, SearchResult
    
    t0 = time.perf_counter()
    current = query.start_node
    path = [current]
    visited_nodes = {current}
    visited_edges = set()
    path_relations = []
    edges_explored = 0
    low_score_streak = 0
    failure_type = FailureType.NONE

    gold_nodes = query.gold_path
    gold_relations = query.gold_relations

    for depth in range(max_depth):
        neighbors = graph.get_neighbors(current)
        candidates = [
            e for e in neighbors
            if e.target not in visited_nodes
            and (current, e.relation, e.target) not in visited_edges
        ]

        if not candidates:
            failure_type = FailureType.DEAD_END
            break

        # Determine if the gold relation is available at this step
        gold_relation = None
        if depth < len(gold_nodes) - 1 and current == gold_nodes[depth]:
            gold_relation = gold_relations[depth]

        # Score with override simulated verifier
        scored = verifier.score_candidates(
            query.keywords, candidates, gold_relation, path_relations, is_re_verification=False
        )
        edges_explored += len(candidates)

        best_score, best_edge = scored[0]

        if best_score < 0.15:
            low_score_streak += 1
        else:
            low_score_streak = 0

        if low_score_streak >= 2:
            failure_type = FailureType.PATH_MISALIGNMENT
            break

        visited_edges.add((current, best_edge.relation, best_edge.target))
        current = best_edge.target
        path.append(current)
        visited_nodes.add(current)
        path_relations.append(best_edge.relation)

        if current == query.answer_node:
            elapsed = time.perf_counter() - t0
            return SearchResult(
                query_id=query.id, success=True, path=path,
                nodes_visited=len(visited_nodes), search_depth=len(path)-1,
                runtime=elapsed, failure_type=FailureType.NONE,
                edges_explored=edges_explored, visited_node_set=visited_nodes
            )

    if failure_type == FailureType.NONE:
        failure_type = FailureType.BUDGET_EXHAUSTED

    elapsed = time.perf_counter() - t0
    return SearchResult(
        query_id=query.id, success=False, path=path,
        nodes_visited=len(visited_nodes), search_depth=len(path)-1,
        runtime=elapsed, failure_type=failure_type,
        edges_explored=edges_explored, visited_node_set=visited_nodes
    )


def run_simulated_fags(
    graph: KnowledgeGraph,
    query: Query,
    verifier: SimulatedVerifier,
    memory: FailureMemory,
    max_depth: int = 10,
    max_backtracks: int = 3,
    enable_re_verification: bool = True,
) -> SearchResult:
    """Execute FAGS using simulated verifier overrides."""
    import time
    from fags import FailureType, SearchResult
    
    t0 = time.perf_counter()
    memory.clear()

    current = query.start_node
    path = [current]
    path_relations = []
    visited_nodes = {current}
    visited_edges = set()
    local_alternatives = {}
    failed_relations = []
    
    edges_explored = 0
    backtracks = 0
    recovery_attempts = 0
    recovery_successes = 0
    gold_path_pruned = False
    gold_path_recovered = False

    gold_nodes = query.gold_path
    gold_relations = query.gold_relations
    low_score_streak = 0
    failure_type = FailureType.NONE

    while True:
        depth = len(path) - 1

        if depth >= max_depth:
            failure_type = FailureType.BUDGET_EXHAUSTED
        else:
            neighbors = graph.get_neighbors(current)
            candidates = [
                e for e in neighbors
                if e.target not in visited_nodes
                and (current, e.relation, e.target) not in visited_edges
            ]

            if depth not in local_alternatives:
                gold_relation = None
                if depth < len(gold_nodes) - 1 and current == gold_nodes[depth]:
                    gold_relation = gold_relations[depth]

                scored = verifier.score_candidates(
                    query.keywords, candidates, gold_relation, path_relations, is_re_verification=False
                )
                local_alternatives[depth] = scored
                edges_explored += len(candidates)

            level_candidates = local_alternatives[depth]

            if not level_candidates:
                failure_type = FailureType.DEAD_END

        if failure_type != FailureType.NONE:
            # Recovery Pipeline
            parent_depth = len(path) - 2
            if parent_depth >= 0 and parent_depth in local_alternatives and local_alternatives[parent_depth]:
                path.pop()
                if path_relations:
                    failed_relations.append(path_relations.pop())
                current = path[-1]
                low_score_streak = 0
                failure_type = FailureType.NONE
                continue

            recovery_success = False
            recovery_attempts += 1

            while backtracks < max_backtracks and not memory.is_empty():
                best_entry = memory.peek_best()
                
                if best_entry.target_id in visited_nodes:
                    memory.pop_best()
                    continue

                if enable_re_verification:
                    # Score candidate under re-verification conditions
                    gold_relation = None
                    for gd in range(len(gold_nodes) - 1):
                        if best_entry.node_id == gold_nodes[gd] and best_entry.relation == gold_relations[gd]:
                            gold_relation = gold_relations[gd]

                    cand_edge = Edge(source=best_entry.node_id, target=best_entry.target_id, relation=best_entry.relation)
                    re_scored = verifier.score_candidates(
                        query.keywords, [cand_edge], gold_relation, best_entry.path_so_far, failed_relations, is_re_verification=True
                    )
                    if re_scored[0][0] < 0.20:
                        memory.pop_best()
                        continue

                entry = memory.pop_best()
                backtracks += 1
                
                restored_depth = entry.depth + 1
                if restored_depth < len(gold_nodes) and entry.target_id == gold_nodes[restored_depth]:
                    gold_path_recovered = True

                current = entry.target_id
                path = entry.path_so_far + [current]
                visited_nodes.add(current)
                visited_edges.add((entry.node_id, entry.relation, entry.target_id))
                
                path_relations = []
                for i in range(len(path) - 1):
                    for edge in graph.get_neighbors(path[i]):
                        if edge.target == path[i+1]:
                            path_relations.append(edge.relation)
                            break
                
                for d in list(local_alternatives.keys()):
                    if d >= restored_depth:
                        del local_alternatives[d]
                
                low_score_streak = 0
                failure_type = FailureType.NONE
                recovery_success = True
                break

            if not recovery_success:
                break
            continue

        best_score, best_edge = level_candidates[0]

        if depth < len(gold_nodes) - 1 and current == gold_nodes[depth]:
            if best_edge.relation != gold_relations[depth]:
                gold_path_pruned = True

        rejected = level_candidates[1:]
        if rejected:
            memory.store(current, rejected, best_score, depth, list(path))

        if best_score < 0.15:
            low_score_streak += 1
        else:
            low_score_streak = 0

        if low_score_streak >= 2:
            failure_type = FailureType.PATH_MISALIGNMENT
            failed_relations.append(best_edge.relation)
            local_alternatives[depth].pop(0)
            continue

        visited_edges.add((current, best_edge.relation, best_edge.target))
        local_alternatives[depth].pop(0)
        current = best_edge.target
        path.append(current)
        visited_nodes.add(current)
        path_relations.append(best_edge.relation)

        if current == query.answer_node:
            elapsed = time.perf_counter() - t0
            return SearchResult(
                query_id=query.id, success=True, path=path,
                nodes_visited=len(visited_nodes), search_depth=len(path)-1,
                runtime=elapsed, failure_type=FailureType.NONE,
                backtracks=backtracks, recovery_attempts=recovery_attempts,
                recovery_successes=recovery_successes,
                gold_path_pruned=gold_path_pruned, gold_path_recovered=gold_path_recovered,
                memory_size_at_end=memory.size, edges_explored=edges_explored,
                visited_node_set=visited_nodes
            )

    elapsed = time.perf_counter() - t0
    return SearchResult(
        query_id=query.id, success=False, path=path,
        nodes_visited=len(visited_nodes), search_depth=len(path)-1,
        runtime=elapsed, failure_type=failure_type,
        backtracks=backtracks, recovery_attempts=recovery_attempts,
        recovery_successes=recovery_successes,
        gold_path_pruned=gold_path_pruned, gold_path_recovered=gold_path_recovered,
        memory_size_at_end=memory.size, edges_explored=edges_explored,
        visited_node_set=visited_nodes
    )


# ══════════════════════════════════════════════
# Main Sweep
# ══════════════════════════════════════════════

def main():
    print("==================================================")
    print("VERIFIER QUALITY SWEEP DIAGNOSTIC (ARTIFICIAL)")
    print("==================================================")

    num_nodes = 100
    query_count = 1000
    seed = 101

    graph, queries = generate_dataset(num_nodes=num_nodes, num_queries=query_count, seed=seed)
    targets = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95]
    results = []

    for target in targets:
        print(f"\nEvaluating at Verifier Quality Level: {target:.0%}")
        
        sim_verifier = SimulatedVerifier(target_rank1=target, seed=seed)
        
        # Run baseline
        base_res = []
        for q in queries:
            base_res.append(run_simulated_baseline(graph, q, sim_verifier))
            
        # Run FAGS
        fags_res = []
        memory = create_memory("top2")
        for q in queries:
            fags_res.append(run_simulated_fags(graph, q, sim_verifier, memory))

        metrics = evaluate_results(base_res, fags_res, f"Verifier_{target:.0%}")
        metrics["target_acc"] = target
        results.append(metrics)

    # ──────────────────────────────────────────────
    # Output CSV Table
    # ──────────────────────────────────────────────
    csv_path = os.path.join(RESULTS_DIR, "verifier_quality_sweep.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Verifier Level", "Baseline Acc", "FAGS Acc", "Acc Gain",
            "Baseline Nodes", "FAGS Nodes", "Additional Search Cost",
            "Efficiency Ratio", "Gold Path Recovery Rate", "Recovery Success Rate"
        ])
        for r in results:
            writer.writerow([
                f"{r['target_acc']:.0%}", f"{r['accuracy_baseline']:.2%}", f"{r['accuracy_fags']:.2%}",
                f"{r['accuracy_gain']:.2%}", f"{r['mean_nodes_baseline']:.2f}", f"{r['mean_nodes_fags']:.2f}",
                f"{r['additional_search_cost']:.2%}", f"{r['efficiency_ratio']:.3f}", f"{r['gold_path_recovery_rate']:.2%}",
                f"{r['recovery_success_rate']:.2%}"
            ])

    # Display clean table
    print("\n--- RESULTS SUMMARY ---")
    print(f"{'Level':<8} | {'Base Acc':<8} | {'FAGS Acc':<8} | {'Gain':<6} | {'Add. Cost':<10} | {'Eff. Ratio':<10} | {'Gold Recov':<10}")
    print("-" * 75)
    for r in results:
        level_str = f"{r['target_acc']:.0%}"
        base_str = f"{r['accuracy_baseline']:.2%}"
        fags_str = f"{r['accuracy_fags']:.2%}"
        gain_str = f"{r['accuracy_gain']:+.2%}"
        cost_str = f"{r['additional_search_cost']:.2%}"
        ratio_str = f"{r['efficiency_ratio']:.3f}"
        gold_str = f"{r['gold_path_recovery_rate']:.2%}"
        
        print(f"{level_str:<8} | {base_str:<8} | {fags_str:<8} | {gain_str:<6} | {cost_str:<10} | {ratio_str:<10} | {gold_str:<10}")

    # ──────────────────────────────────────────────
    # Plot Generation
    # ──────────────────────────────────────────────
    levels_pct = [r["target_acc"] * 100 for r in results]
    base_accs = [r["accuracy_baseline"] * 100 for r in results]
    fags_accs = [r["accuracy_fags"] * 100 for r in results]
    costs = [r["additional_search_cost"] * 100 for r in results]
    gains = [r["accuracy_gain"] * 100 for r in results]
    ratios = [r["efficiency_ratio"] for r in results]
    gold_rec = [r["gold_path_recovery_rate"] * 100 for r in results]

    # Plot 1: Accuracies vs Verifier Level
    plt.figure(figsize=(8, 5))
    plt.plot(levels_pct, base_accs, marker='o', color='black', linestyle='--', label='Baseline Accuracy')
    plt.plot(levels_pct, fags_accs, marker='s', color='crimson', label='FAGS Accuracy')
    plt.xlabel('Verifier Quality Level (Target Rank-1 %)')
    plt.ylabel('Search Accuracy (%)')
    plt.title('Baseline vs FAGS Accuracy across Verifier Quality')
    plt.legend(loc='upper left')
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "verifier_sweep_accuracies.png"), dpi=150)
    plt.close()

    # Plot 2: Cost and Efficiency
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    ax1.plot(levels_pct, costs, marker='o', color='teal', label='Additional Search Cost (%)')
    ax1.set_ylabel('Additional Search Cost (%)')
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc='upper right')
    ax1.set_title("Search Cost and Efficiency vs Verifier Quality")

    ax2.plot(levels_pct, ratios, marker='s', color='darkorange', label='Efficiency Ratio')
    ax2.set_xlabel('Verifier Quality Level (Target Rank-1 %)')
    ax2.set_ylabel('Efficiency Ratio (Gain / Cost)')
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "verifier_sweep_metrics.png"), dpi=150)
    plt.close()

    # Plot 3: Gold Path Recovery Rate vs Verifier Level
    plt.figure(figsize=(7, 5))
    plt.plot(levels_pct, gold_rec, marker='^', color='purple', label='Gold Path Recovery Rate (%)')
    plt.xlabel('Verifier Quality Level (Target Rank-1 %)')
    plt.ylabel('Gold Path Recovery Rate (%)')
    plt.title('Gold Path Recovery Rate vs Verifier Quality')
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "verifier_sweep_gold_recovery.png"), dpi=150)
    plt.close()

    print(f"\nPlots saved to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
