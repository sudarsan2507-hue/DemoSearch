"""Diagnostic script to audit the life-cycle of gold paths in the recovery pipeline.

For every query where the gold path is pruned (i.e. not Rank-1 at some step):
  1. Did the gold relation enter Failure Memory?
  2. Was it popped from memory during search?
  3. Was it discarded due to the Visited Set (Branch Memory collision)?
  4. Was it discarded due to Re-Verification score thresholds?
  5. Was it revived successfully?
  6. Did it fail after revival (e.g. pruned again, hit dead end, etc.)?
  7. Did it successfully reach the answer?
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from fags.graph_generator import generate_dataset
from fags.verifier import Verifier
from fags.memory import create_memory
from fags import Edge, FailureType, KnowledgeGraph, Query

# Relevance floor
_RELEVANCE_FLOOR = 0.15

RESULTS_DIR = r"d:\Projects\DemoSearch\results"
os.makedirs(RESULTS_DIR, exist_ok=True)

@dataclass
class AuditStats:
    queries_evaluated: int = 0
    gold_pruned_in_search: int = 0
    gold_stored_in_memory: int = 0
    gold_popped_from_memory: int = 0
    gold_dropped_by_visited: int = 0
    gold_dropped_by_reverification: int = 0
    gold_revived_successfully: int = 0
    gold_failed_after_revival: int = 0
    gold_reached_answer: int = 0


def audit_recovery_pipeline(
    graph: KnowledgeGraph,
    queries: list[Query],
    verifier: Verifier,
    strategy: str = "top2",
    max_depth: int = 10,
    max_backtracks: int = 5,
    enable_re_verification: bool = True,
) -> AuditStats:
    """Run search on queries and audit exactly where gold path entries drop out."""
    stats = AuditStats()
    stats.queries_evaluated = len(queries)

    # Initialize memory strategy
    memory = create_memory(strategy)

    for q in queries:
        memory.clear()
        
        # Determine if the gold path is ever pruned during standard traversal
        # We find this by scoring the gold relation at each node along the gold path
        gold_nodes = q.gold_path
        gold_rels = q.gold_relations
        is_gold_pruned = False
        
        for depth in range(len(gold_nodes) - 1):
            curr_n = gold_nodes[depth]
            gold_r = gold_rels[depth]
            neighbors = graph.get_neighbors(curr_n)
            scored = []
            for edge in neighbors:
                score = verifier.score(q.keywords, edge, gold_rels[:depth])
                scored.append((score, edge))
            scored.sort(key=lambda x: x[0], reverse=True)
            
            # Gold is pruned if it is not Rank-1
            if scored and scored[0][1].relation != gold_r:
                is_gold_pruned = True
                break

        if not is_gold_pruned:
            # Skip queries where the baseline easily finds the answer without pruning
            continue
            
        stats.gold_pruned_in_search += 1

        # Track lifecycle of this query's gold relations inside memory
        # Store metadata about gold relations that get put into memory: (source, relation, target)
        stored_gold_entries: set[tuple[str, str, str]] = set()

        # Run modified search loop to trace the audit pathway
        current = q.start_node
        path = [current]
        path_relations: list[str] = []
        visited_nodes = {current}
        visited_edges = set()
        local_alternatives = {}
        failed_relations = []
        
        backtracks_count = 0
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

                if not depth in local_alternatives:
                    scored = []
                    for edge in candidates:
                        s = verifier.score(q.keywords, edge, path_relations)
                        scored.append((s, edge))
                    scored.sort(key=lambda x: x[0], reverse=True)
                    local_alternatives[depth] = scored

                level_candidates = local_alternatives[depth]

                if not level_candidates:
                    failure_type = FailureType.DEAD_END

            # If failure occurs, enter audit recovery
            if failure_type != FailureType.NONE:
                # ── Recovery Triggered ──
                # Check local backtracking first
                parent_depth = len(path) - 2
                if parent_depth >= 0 and parent_depth in local_alternatives and local_alternatives[parent_depth]:
                    failed_node = path.pop()
                    if path_relations:
                        failed_relations.append(path_relations.pop())
                    current = path[-1]
                    low_score_streak = 0
                    failure_type = FailureType.NONE
                    continue

                # Consult memory
                recovery_success = False
                while backtracks_count < max_backtracks and not memory.is_empty():
                    # We peek first to audit
                    best_entry = memory.peek_best()
                    is_gold_entry = (best_entry.node_id, best_entry.relation, best_entry.target_id) in stored_gold_entries
                    
                    if is_gold_entry:
                        stats.gold_popped_from_memory += 1

                    # 1. Visited Node Collision check
                    if best_entry.target_id in visited_nodes:
                        if is_gold_entry:
                            stats.gold_dropped_by_visited += 1
                        memory.pop_best()
                        continue

                    # 2. Dynamic Re-Verification check
                    if enable_re_verification:
                        re_score = verifier.re_score(
                            q.keywords,
                            Edge(source=best_entry.node_id, target=best_entry.target_id, relation=best_entry.relation),
                            best_entry.path_so_far,
                            failed_relations
                        )
                        if re_score < 0.20:
                            if is_gold_entry:
                                stats.gold_dropped_by_reverification += 1
                            memory.pop_best()
                            continue

                    # Pop for real as we are reviving it
                    entry = memory.pop_best()
                    backtracks_count += 1

                    if is_gold_entry:
                        stats.gold_revived_successfully += 1

                    # Revive path
                    current = entry.target_id
                    path = entry.path_so_far + [current]
                    visited_nodes.add(current)
                    visited_edges.add((entry.node_id, entry.relation, entry.target_id))
                    
                    # Reconstruct path relations
                    path_relations = []
                    for i in range(len(path) - 1):
                        for edge in graph.get_neighbors(path[i]):
                            if edge.target == path[i+1]:
                                path_relations.append(edge.relation)
                                break
                    
                    for d in list(local_alternatives.keys()):
                        if d >= entry.depth + 1:
                            del local_alternatives[d]

                    low_score_streak = 0
                    failure_type = FailureType.NONE
                    recovery_success = True
                    break

                if not recovery_success:
                    # Final Search Failure
                    break
                continue

            # Process normal step
            best_score, best_edge = level_candidates[0]
            
            # Audit Memory Storing
            # Check if any of the rejected candidates are gold path steps
            rejected = level_candidates[1:]
            for score, edge in rejected:
                # Find if this edge matches gold path transition
                for gd in range(len(gold_nodes) - 1):
                    if current == gold_nodes[gd] and edge.relation == gold_rels[gd] and edge.target == gold_nodes[gd+1]:
                        # A gold relation is being sent to memory
                        stored_gold_entries.add((current, edge.relation, edge.target))
                        stats.gold_stored_in_memory += 1

            if rejected:
                memory.store(
                    current_node=current,
                    candidates=rejected,
                    winner_score=best_score,
                    depth=depth,
                    path_so_far=list(path),
                )

            # Check streak
            if best_score < _RELEVANCE_FLOOR:
                current_low_streak = low_score_streak + 1
            else:
                current_low_streak = 0

            if current_low_streak >= 2:
                failure_type = FailureType.PATH_MISALIGNMENT
                failed_relations.append(best_edge.relation)
                local_alternatives[depth].pop(0)
                continue

            # Move forward
            low_score_streak = current_low_streak
            visited_edges.add((current, best_edge.relation, best_edge.target))
            local_alternatives[depth].pop(0)
            
            current = best_edge.target
            path.append(current)
            visited_nodes.add(current)
            path_relations.append(best_edge.relation)

            # Check contradiction
            node_obj = graph.get_node(current)
            has_contradiction = False
            if node_obj and node_obj.evidence:
                for k, v in node_obj.evidence.items():
                    if k in q.gold_path: # simplified trace matching
                        has_contradiction = True
                        break
            if has_contradiction:
                failure_type = FailureType.CONTRADICTION
                failed_relations.append(best_edge.relation)
                continue

            # Check answer
            if current == q.answer_node:
                # Search succeeded
                # Did we reach the answer via a revived gold path?
                # Check if we successfully revived the gold relation during this search
                if stats.gold_revived_successfully > 0:
                    stats.gold_reached_answer += 1
                break

        # If search terminated without reaching answer, and we did revive a gold relation, it failed post-revival
        if current != q.answer_node and stats.gold_revived_successfully > 0:
            stats.gold_failed_after_revival += 1

    return stats


def main():
    print("==================================================")
    print("GOLD PATH RECOVERY PIPELINE AUDIT")
    print("==================================================")

    sizes = ["Small", "Medium", "Large"]
    node_counts = [20, 100, 1000]
    seed = 101

    for size_label, num_nodes in zip(sizes, node_counts):
        print(f"\nRunning Audit on {size_label} Graph ({num_nodes} nodes)...")
        graph, queries = generate_dataset(num_nodes=num_nodes, num_queries=1000, seed=seed)
        verifier = Verifier(noise_std=0.08, seed=seed)

        stats_top2 = audit_recovery_pipeline(
            graph=graph,
            queries=queries,
            verifier=verifier,
            strategy="top2",
            max_backtracks=5
        )

        print(f"Audit Results ({size_label}):")
        print(f"  1. Queries with Gold Path Pruned      : {stats_top2.gold_pruned_in_search}")
        print(f"  2. Gold Paths Stored in Memory        : {stats_top2.gold_stored_in_memory}")
        print(f"  3. Gold Paths Popped from Memory      : {stats_top2.gold_popped_from_memory}")
        print(f"  4. Dropped by Visited Set Collision   : {stats_top2.gold_dropped_by_visited}")
        print(f"  5. Dropped by Re-Verification Floor   : {stats_top2.gold_dropped_by_reverification}")
        print(f"  6. Revived Successfully               : {stats_top2.gold_revived_successfully}")
        print(f"  7. Failed After Revival               : {stats_top2.gold_failed_after_revival}")
        print(f"  8. Reached Answer Node                : {stats_top2.gold_reached_answer}")


if __name__ == "__main__":
    main()
