"""Diagnostic script to profile gold paths AFTER they are successfully revived.

Measures:
  1. Average hops survived after revival before another failure or answer.
  2. Distribution of hops survived (0, 1, 2, 3+ hops).
  3. Verifier score comparison (Before revival vs After revival).
  4. Root causes of post-revival failures:
     - A) Next-hop verifier mistake (correct relation outranked)
     - B) Visited-set collision (next node already visited)
     - C) Search budget exhaustion (depth limit hit)
     - D) Local dead end (no outgoing edges)
  5. Calculation of expected recovery rate if the verifier were perfect after revival.
"""

from __future__ import annotations

import os
import numpy as np
from dataclasses import dataclass, field

from fags.graph_generator import generate_dataset
from fags.verifier import Verifier
from fags.memory import create_memory
from fags import Edge, FailureType, KnowledgeGraph, Query

# Relevance floor
_RELEVANCE_FLOOR = 0.15

RESULTS_DIR = r"d:\Projects\DemoSearch\results"
os.makedirs(RESULTS_DIR, exist_ok=True)

@dataclass
class PostRevivalTracker:
    # Scores tracking
    scores_before_revival: list[float] = field(default_factory=list)
    scores_after_revival: list[float] = field(default_factory=list)
    
    # Hops survived tracking
    hops_survived: list[int] = field(default_factory=list)
    
    # Failure category counts
    fail_next_hop_mistake: int = 0
    fail_visited_collision: int = 0
    fail_budget_exhaustion: int = 0
    fail_dead_end: int = 0
    
    # Successful outcomes
    reached_answer: int = 0
    total_revivals: int = 0


def audit_post_revival(
    graph: KnowledgeGraph,
    queries: list[Query],
    verifier: Verifier,
    strategy: str = "top2",
    max_depth: int = 10,
    max_backtracks: int = 5,
    enable_re_verification: bool = True,
) -> PostRevivalTracker:
    tracker = PostRevivalTracker()
    memory = create_memory(strategy)

    for q in queries:
        memory.clear()
        gold_nodes = q.gold_path
        gold_rels = q.gold_relations
        
        # Verify if gold path is pruned (so it's a recovery query candidate)
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
            if scored and scored[0][1].relation != gold_r:
                is_gold_pruned = True
                break
                
        if not is_gold_pruned:
            continue

        # Keep track of gold transitions in memory
        stored_gold_entries: dict[tuple[str, str, str], tuple[float, int]] = {}

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

            if failure_type != FailureType.NONE:
                # Consult memory
                revived_node = None
                revived_entry = None
                is_gold_revived = False

                while backtracks_count < max_backtracks and not memory.is_empty():
                    best_entry = memory.peek_best()
                    key = (best_entry.node_id, best_entry.relation, best_entry.target_id)
                    is_gold = key in stored_gold_entries

                    # Visited Node Collision
                    if best_entry.target_id in visited_nodes:
                        memory.pop_best()
                        continue

                    # Re-Verification check
                    re_score = verifier.re_score(
                        q.keywords,
                        Edge(source=best_entry.node_id, target=best_entry.target_id, relation=best_entry.relation),
                        best_entry.path_so_far,
                        failed_relations
                    )
                    if enable_re_verification and re_score < 0.20:
                        memory.pop_best()
                        continue

                    # Perform Revival
                    entry = memory.pop_best()
                    backtracks_count += 1
                    
                    if is_gold:
                        is_gold_revived = True
                        tracker.total_revivals += 1
                        orig_score, _ = stored_gold_entries[key]
                        tracker.scores_before_revival.append(orig_score)
                        tracker.scores_after_revival.append(re_score)
                    
                    revived_node = entry.target_id
                    revived_entry = entry
                    break

                if revived_node is None:
                    # Final search failure
                    break

                # Resume from revived node
                current = revived_node
                path = revived_entry.path_so_far + [current]
                visited_nodes.add(current)
                visited_edges.add((revived_entry.node_id, revived_entry.relation, revived_entry.target_id))
                
                path_relations = []
                for i in range(len(path) - 1):
                    for edge in graph.get_neighbors(path[i]):
                        if edge.target == path[i+1]:
                            path_relations.append(edge.relation)
                            break

                for d in list(local_alternatives.keys()):
                    if d >= revived_entry.depth + 1:
                        del local_alternatives[d]

                low_score_streak = 0
                failure_type = FailureType.NONE

                # If this was a gold revival, we trace its survival path until another failure or solution
                if is_gold_revived:
                    # Determine where we are on the gold path
                    # Find matching index in gold path
                    gold_idx = -1
                    for idx, nid in enumerate(gold_nodes):
                        if nid == current:
                            gold_idx = idx
                            break

                    # Trace step-by-step from revived point
                    hops_count = 0
                    post_curr = current
                    post_path = list(path)
                    post_relations = list(path_relations)
                    post_visited = set(visited_nodes)
                    post_edges = set(visited_edges)
                    post_low_streak = 0
                    post_failed_relations = list(failed_relations)
                    post_local_alts = {}

                    while True:
                        post_depth = len(post_path) - 1

                        if post_depth >= max_depth:
                            tracker.fail_budget_exhaustion += 1
                            break

                        post_neighbors = graph.get_neighbors(post_curr)
                        post_candidates = [
                            e for e in post_neighbors
                            if e.target not in post_visited
                            and (post_curr, e.relation, e.target) not in post_edges
                        ]

                        if not post_candidates:
                            tracker.fail_dead_end += 1
                            break

                        scored_post = []
                        for edge in post_candidates:
                            s = verifier.score(q.keywords, edge, post_relations)
                            scored_post.append((s, edge))
                        scored_post.sort(key=lambda x: x[0], reverse=True)

                        best_post_score, best_post_edge = scored_post[0]

                        # Check if this matches gold path
                        gold_hop_idx = gold_idx + hops_count
                        is_gold_step_correct = False
                        if gold_hop_idx < len(gold_rels):
                            target_gold_rel = gold_rels[gold_hop_idx]
                            target_gold_node = gold_nodes[gold_hop_idx + 1]
                            if best_post_edge.relation == target_gold_rel:
                                is_gold_step_correct = True

                        if not is_gold_step_correct:
                            # Search branched off the gold path (verifier mistake)
                            tracker.fail_next_hop_mistake += 1
                            break

                        # Visited collision check for next node
                        if best_post_edge.target in post_visited:
                            tracker.fail_visited_collision += 1
                            break

                        # Move forward
                        hops_count += 1
                        post_curr = best_post_edge.target
                        post_path.append(post_curr)
                        post_visited.add(post_curr)
                        post_edges.add((best_post_edge.source, best_post_edge.relation, best_post_edge.target))
                        post_relations.append(best_post_edge.relation)

                        if post_curr == q.answer_node:
                            tracker.reached_answer += 1
                            break

                    tracker.hops_survived.append(hops_count)
                continue

            # Standard path progression
            best_score, best_edge = level_candidates[0]
            rejected = level_candidates[1:]
            
            for score, edge in rejected:
                for gd in range(len(gold_nodes) - 1):
                    if current == gold_nodes[gd] and edge.relation == gold_rels[gd] and edge.target == gold_nodes[gd+1]:
                        stored_gold_entries[(current, edge.relation, edge.target)] = (score, depth)

            if rejected:
                memory.store(
                    current_node=current,
                    candidates=rejected,
                    winner_score=best_score,
                    depth=depth,
                    path_so_far=list(path),
                )

            if best_score < _RELEVANCE_FLOOR:
                current_low_streak = low_score_streak + 1
            else:
                current_low_streak = 0

            if current_low_streak >= 2:
                failure_type = FailureType.PATH_MISALIGNMENT
                failed_relations.append(best_edge.relation)
                local_alternatives[depth].pop(0)
                continue

            low_score_streak = current_low_streak
            visited_edges.add((current, best_edge.relation, best_edge.target))
            local_alternatives[depth].pop(0)
            current = best_edge.target
            path.append(current)
            visited_nodes.add(current)
            path_relations.append(best_edge.relation)

            # Contradiction check
            node_obj = graph.get_node(current)
            has_contra = False
            if node_obj and node_obj.evidence:
                for k, v in node_obj.evidence.items():
                    if k in q.gold_path:
                        has_contra = True
            if has_contra:
                failure_type = FailureType.CONTRADICTION
                failed_relations.append(best_edge.relation)
                continue

            if current == q.answer_node:
                break

    return tracker


def main():
    print("==================================================")
    print("POST-REVIVAL GOLD PATH ANALYSIS")
    print("==================================================")

    sizes = ["Small", "Medium", "Large"]
    node_counts = [20, 100, 1000]
    seed = 101

    for size_label, num_nodes in zip(sizes, node_counts):
        print(f"\nEvaluating {size_label} Graph ({num_nodes} nodes)...")
        graph, queries = generate_dataset(num_nodes=num_nodes, num_queries=1000, seed=seed)
        verifier = Verifier(noise_std=0.08, seed=seed)

        tracker = audit_post_revival(
            graph=graph,
            queries=queries,
            verifier=verifier,
            strategy="top2",
            max_backtracks=5
        )

        total = tracker.total_revivals
        if total == 0:
            print("No gold paths were successfully revived in this graph configuration.")
            continue

        hops = np.array(tracker.hops_survived)
        avg_hops = np.mean(hops)

        # Dist
        h_0 = np.sum(hops == 0)
        h_1 = np.sum(hops == 1)
        h_2 = np.sum(hops == 2)
        h_3 = np.sum(hops >= 3)

        print(f"Total Successful Revivals of Gold Paths: {total}")
        print(f"  1. Average Hops Survived: {avg_hops:.2f}")
        print(f"  2. Hops Survival Distribution:")
        print(f"     0 hops  : {h_0 / total:.2%}")
        print(f"     1 hop   : {h_1 / total:.2%}")
        print(f"     2 hops  : {h_2 / total:.2%}")
        print(f"     3+ hops : {h_3 / total:.2%}")

        print(f"  3. Verifier Scores (Mean):")
        print(f"     Before Revival: {np.mean(tracker.scores_before_revival):.4f}")
        print(f"     After Revival : {np.mean(tracker.scores_after_revival):.4f}")
        
        # Failure classification
        f_mistake = tracker.fail_next_hop_mistake
        f_visited = tracker.fail_visited_collision
        f_budget = tracker.fail_budget_exhaustion
        f_dead = tracker.fail_dead_end
        f_correct = tracker.reached_answer
        
        f_total = f_mistake + f_visited + f_budget + f_dead + f_correct
        if f_total == 0:
            f_total = 1

        print(f"  4. Failure Root Cause Distribution:")
        print(f"     A) Next-hop Verifier Mistake : {f_mistake / f_total:.2%}")
        print(f"     B) Visited-set Collision     : {f_visited / f_total:.2%}")
        print(f"     C) Search Budget Exhaustion  : {f_budget / f_total:.2%}")
        print(f"     D) Local Dead End            : {f_dead / f_total:.2%}")
        print(f"     E) Reached Answer            : {f_correct / f_total:.2%}")

        # Perfect verifier calculation
        # If the verifier were perfect after revival, then any next-hop verifier mistake would NOT happen.
        # The path would successfully run to the answer unless blocked by budget or visited set.
        # So the success rate would be: Reached Answer + Next-hop Verifier Mistake (since those mistakes are solved)
        perfect_expected_success = (f_correct + f_mistake) / f_total
        print(f"  5. Expected Success Rate if Verifier was Perfect After Revival: {perfect_expected_success:.2%}")


if __name__ == "__main__":
    main()
