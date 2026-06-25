"""Failure-Aware Graph Search (FAGS) implementation.

Includes the full recovery pipeline:
  1. Main search loop (using Branch Memory to prevent cycles)
  2. Local Backtracking (trying alternative relations at the current node first)
  3. Failure Memory Storage (Top-1, Top-2, or Threshold strategies)
  4. Failure Memory Revival (popping the best alternative path)
  5. Dynamic Re-Verification (re-scoring alternatives using failed relations)
"""

from __future__ import annotations

import time
from typing import Sequence, Any

from fags import (
    Edge, FailureType, KnowledgeGraph, Query, SearchResult, MemoryEntry,
)
from fags.verifier import Verifier, RELATION_COHERENCE
from fags.memory import FailureMemory

# Relevance floor: two consecutive edges below this trigger path-misalignment
_RELEVANCE_FLOOR = 0.15


def failure_search(
    graph: KnowledgeGraph,
    query: Query,
    verifier: Verifier,
    memory: FailureMemory,
    max_depth: int = 10,
    max_backtracks: int = 3,
    enable_re_verification: bool = True,
    shield_depth: int = 0,
    use_certificate: bool = True,
    certificate_bonus: float = 0.10,
    rbsc_mode: str = "none",  # "none" | "linear" | "nonlinear"
) -> SearchResult:
    """Run Failure-Aware Graph Search (FAGS).

    Parameters
    ----------
    graph : The synthetic knowledge graph.
    query : The query containing start/target nodes and keywords.
    verifier : The relation scorer (with imperfect noise).
    memory : FailureMemory instance (Top1, Top2, or Threshold).
    max_depth : Max path length.
    max_backtracks : Hard cap on recoveries to prevent brute-force search.
    enable_re_verification : If True, re-score candidates dynamically based on failed paths.
    rbsc_mode : Reason-Based Stabilization Certificate mode.
    """
    t0 = time.perf_counter()
    memory.clear()

    # Current path trackers
    current = query.start_node
    path: list[str] = [current]
    path_relations: list[str] = []
    
    # Branch Memory: global set of visited nodes and edges to prevent cycles
    visited_nodes: set[str] = {current}
    visited_edges: set[tuple[str, str, str]] = set()
    
    # Evidence tracker for contradiction detection
    collected_evidence: dict[str, str] = {}
    start_node = graph.get_node(current)
    if start_node and start_node.evidence:
        collected_evidence.update(start_node.evidence)

    # Local backtracking stack of candidates at each level of the current path
    # Maps: depth -> list of (score, edge) untried at that level
    local_alternatives: dict[int, list[tuple[float, Edge]]] = {}

    # Tracking list of failed relations encountered across search branches
    failed_relations: list[str] = []

    # Diagnostics
    edges_explored = 0
    backtracks = 0
    recovery_attempts = 0
    recovery_successes = 0
    
    # Gold path tracking
    gold_path_pruned = False
    gold_path_recovered = False

    shield_hops_remaining = 0
    active_certificate = None
    current_revival_hops = 0
    is_post_revival = False
    hops_survived_post_revival = []
    
    # RBSC Tracking
    revived_margins = []

    # Check if gold path is pruned at start
    gold_nodes = query.gold_path
    gold_relations = query.gold_relations

    low_score_streak = 0
    failure_type = FailureType.NONE

    while True:
        # Determine current depth
        depth = len(path) - 1

        if depth >= max_depth:
            failure_type = FailureType.BUDGET_EXHAUSTED
            # Trigger recovery
            current, path, path_relations, low_score_streak, failure_type, backtracks, recovery_attempts, recovery_successes, gold_path_recovered, revived_rel = _handle_recovery(
                graph, query, verifier, memory, path, path_relations, visited_nodes, visited_edges,
                local_alternatives, failed_relations, max_backtracks, enable_re_verification,
                backtracks, recovery_attempts, recovery_successes, gold_path_recovered, gold_nodes, rbsc_mode
            )
            if current_revival_hops > 0:
                hops_survived_post_revival.append(current_revival_hops)
            current_revival_hops = 0
            if revived_rel is not None:
                active_certificate = revived_rel
                shield_hops_remaining = shield_depth
                is_post_revival = True
                if rbsc_mode != "none" and isinstance(revived_rel, tuple):
                    revived_margins.append(revived_rel[1])
            else:
                is_post_revival = False
            if failure_type != FailureType.NONE:
                # Recovery failed to find a path, search terminates
                break
            continue

        # 1. Query candidate edges
        neighbors = graph.get_neighbors(current)
        candidates: list[Edge] = [
            e for e in neighbors
            if e.target not in visited_nodes
            and (current, e.relation, e.target) not in visited_edges
        ]

        # 2. Score candidates if not already done for this level
        if depth not in local_alternatives:
            scored: list[tuple[float, Edge]] = []
            for edge in candidates:
                s = verifier.score(query, edge, path_relations)
                
                # Certificate Bonus
                if use_certificate and active_certificate is not None:
                    if rbsc_mode != "none" and isinstance(active_certificate, tuple):
                        cert_rel, cert_margin = active_certificate
                        coherence = RELATION_COHERENCE.get((edge.relation, cert_rel), 0.3)
                        if coherence > 0.5:
                            if rbsc_mode == "linear":
                                multiplier = max(0.0, 1.0 - cert_margin)
                            elif rbsc_mode == "nonlinear":
                                multiplier = 1.0 / (1.0 + 5.0 * cert_margin)
                            else:
                                multiplier = 1.0
                            s = min(1.0, s + certificate_bonus * multiplier)
                    else:
                        coherence = RELATION_COHERENCE.get((edge.relation, active_certificate), 0.3)
                        if coherence > 0.5:
                            s = min(1.0, s + certificate_bonus)
                        
                scored.append((s, edge))
                edges_explored += 1
            scored.sort(key=lambda x: x[0], reverse=True)
            local_alternatives[depth] = scored

        level_candidates = local_alternatives[depth]

        if not level_candidates:
            failure_type = FailureType.DEAD_END
            current, path, path_relations, low_score_streak, failure_type, backtracks, recovery_attempts, recovery_successes, gold_path_recovered, revived_rel = _handle_recovery(
                graph, query, verifier, memory, path, path_relations, visited_nodes, visited_edges,
                local_alternatives, failed_relations, max_backtracks, enable_re_verification,
                backtracks, recovery_attempts, recovery_successes, gold_path_recovered, gold_nodes, rbsc_mode
            )
            if current_revival_hops > 0:
                hops_survived_post_revival.append(current_revival_hops)
            current_revival_hops = 0
            if revived_rel is not None:
                active_certificate = revived_rel
                shield_hops_remaining = shield_depth
                is_post_revival = True
                if rbsc_mode != "none" and isinstance(revived_rel, tuple):
                    revived_margins.append(revived_rel[1])
            else:
                is_post_revival = False
            if failure_type != FailureType.NONE:
                break
            continue

        # Get best candidate at this level
        best_score, best_edge = level_candidates[0]

        # Check if the correct gold relation was outranked at this step
        if depth < len(gold_nodes) - 1 and current == gold_nodes[depth]:
            gold_rel = gold_relations[depth]
            if best_edge.relation != gold_rel:
                gold_path_pruned = True

        # Keep track of rejected alternatives for failure memory
        rejected = level_candidates[1:]
        if rejected:
            memory.store(
                current_node=current,
                candidates=rejected,
                winner_score=best_score,
                depth=depth,
                path_so_far=list(path),
                winner_relation=best_edge.relation,
            )

        # 3. Path misalignment check
        current_floor = 0.05 if shield_hops_remaining > 0 else _RELEVANCE_FLOOR
        if best_score < current_floor:
            current_low_streak = low_score_streak + 1
        else:
            current_low_streak = 0

        if current_low_streak >= 2:
            failure_type = FailureType.PATH_MISALIGNMENT
            failed_relations.append(best_edge.relation)
            # Remove this candidate from future tries at this level
            local_alternatives[depth].pop(0)
            
            current, path, path_relations, low_score_streak, failure_type, backtracks, recovery_attempts, recovery_successes, gold_path_recovered, revived_rel = _handle_recovery(
                graph, query, verifier, memory, path, path_relations, visited_nodes, visited_edges,
                local_alternatives, failed_relations, max_backtracks, enable_re_verification,
                backtracks, recovery_attempts, recovery_successes, gold_path_recovered, gold_nodes, rbsc_mode
            )
            if current_revival_hops > 0:
                hops_survived_post_revival.append(current_revival_hops)
            current_revival_hops = 0
            if revived_rel is not None:
                active_certificate = revived_rel
                shield_hops_remaining = shield_depth
                is_post_revival = True
                if rbsc_mode != "none" and isinstance(revived_rel, tuple):
                    revived_margins.append(revived_rel[1])
            else:
                is_post_revival = False
            if failure_type != FailureType.NONE:
                break
            continue

        # 4. Traverse
        low_score_streak = current_low_streak
        visited_edges.add((current, best_edge.relation, best_edge.target))
        # Consume the candidate
        local_alternatives[depth].pop(0)
        
        if active_certificate is not None:
            shield_hops_remaining = max(0, shield_hops_remaining - 1)
            if shield_hops_remaining == 0:
                active_certificate = None
                
        if is_post_revival:
            current_revival_hops += 1
                
        current = best_edge.target
        path.append(current)
        visited_nodes.add(current)
        path_relations.append(best_edge.relation)

        # 5. Check contradiction
        node_obj = graph.get_node(current)
        has_contradiction = False
        if node_obj and node_obj.evidence:
            for k, v in node_obj.evidence.items():
                if k in collected_evidence and collected_evidence[k] != v:
                    has_contradiction = True
                    break
                collected_evidence[k] = v

        if has_contradiction:
            failure_type = FailureType.CONTRADICTION
            failed_relations.append(best_edge.relation)
            current, path, path_relations, low_score_streak, failure_type, backtracks, recovery_attempts, recovery_successes, gold_path_recovered, revived_rel = _handle_recovery(
                graph, query, verifier, memory, path, path_relations, visited_nodes, visited_edges,
                local_alternatives, failed_relations, max_backtracks, enable_re_verification,
                backtracks, recovery_attempts, recovery_successes, gold_path_recovered, gold_nodes, rbsc_mode
            )
            if current_revival_hops > 0:
                hops_survived_post_revival.append(current_revival_hops)
            current_revival_hops = 0
            if revived_rel is not None:
                active_certificate = revived_rel
                shield_hops_remaining = shield_depth
                is_post_revival = True
                if rbsc_mode != "none" and isinstance(revived_rel, tuple):
                    revived_margins.append(revived_rel[1])
            else:
                is_post_revival = False
            if failure_type != FailureType.NONE:
                break
            continue

        # 6. Check answer node
        if current == query.answer_node:
            if current_revival_hops > 0:
                hops_survived_post_revival.append(current_revival_hops)
            current_revival_hops = 0
            elapsed = time.perf_counter() - t0
            
            # Record successful recovery margins
            succ_margins = list(revived_margins)
            
            return SearchResult(
                query_id=query.id,
                success=True,
                path=path,
                nodes_visited=len(visited_nodes),
                search_depth=len(path) - 1,
                runtime=elapsed,
                failure_type=FailureType.NONE,
                backtracks=backtracks,
                recovery_attempts=recovery_attempts,
                recovery_successes=recovery_successes,
                gold_path_pruned=gold_path_pruned,
                gold_path_recovered=gold_path_recovered,
                memory_size_at_end=memory.size,
                edges_explored=edges_explored,
                visited_node_set=visited_nodes,
                hops_survived_post_revival=hops_survived_post_revival,
                successful_recovery_margins=succ_margins,
                failed_recovery_margins=[],
                path_relations=list(path_relations),
            )

    elapsed = time.perf_counter() - t0
    # Record failed recovery margins
    fail_margins = list(revived_margins)
    
    return SearchResult(
        query_id=query.id,
        success=False,
        path=path,
        nodes_visited=len(visited_nodes),
        search_depth=len(path) - 1,
        runtime=elapsed,
        failure_type=failure_type,
        backtracks=backtracks,
        recovery_attempts=recovery_attempts,
        recovery_successes=recovery_successes,
        gold_path_pruned=gold_path_pruned,
        gold_path_recovered=gold_path_recovered,
        memory_size_at_end=memory.size,
        edges_explored=edges_explored,
        visited_node_set=visited_nodes,
        successful_recovery_margins=[],
        failed_recovery_margins=fail_margins,
        path_relations=list(path_relations),
    )


def _handle_recovery(
    graph: KnowledgeGraph,
    query: Query,
    verifier: Verifier,
    memory: FailureMemory,
    path: list[str],
    path_relations: list[str],
    visited_nodes: set[str],
    visited_edges: set[tuple[str, str, str]],
    local_alternatives: dict[int, list[tuple[float, Edge]]],
    failed_relations: list[str],
    max_backtracks: int,
    enable_re_verification: bool,
    backtracks: int,
    recovery_attempts: int,
    recovery_successes: int,
    gold_path_recovered: bool,
    gold_nodes: list[str],
    rbsc_mode: str = "none",
) -> tuple[str, list[str], list[str], int, FailureType, int, int, int, bool, Any]:
    """Handle recovery pipeline: local backtracking first, then failure memory revival."""
    
    # ── 1. Local Backtracking (First Resort) ──
    # Check parent level (depth-1) for remaining untried options
    parent_depth = len(path) - 2
    if parent_depth >= 0 and parent_depth in local_alternatives and local_alternatives[parent_depth]:
        # Perform local backtrack step
        # Pop the current failed leaf node from path
        failed_node = path.pop()
        # Pop the corresponding relation
        if path_relations:
            failed_relations.append(path_relations.pop())

        # The parent becomes the active node again
        active_node = path[-1]
        
        # Reset low-score streak for parent context
        return active_node, path, path_relations, 0, FailureType.NONE, backtracks, recovery_attempts, recovery_successes, gold_path_recovered, None

    # ── 2. Failure Memory Revival (Second Resort) ──
    recovery_attempts += 1
    
    while backtracks < max_backtracks and not memory.is_empty():
        entry = memory.pop_best()
        
        # Check Branch Memory: skip if the target node is already in visited_nodes
        if entry.target_id in visited_nodes:
            continue

        # Dynamic Re-Verification
        if enable_re_verification:
            re_score = verifier.re_score(
                query,
                Edge(source=entry.node_id, target=entry.target_id, relation=entry.relation),
                entry.path_so_far,
                failed_relations
            )
            # If the re-score drops below viability (e.g. 0.20), discard
            if re_score < 0.20:
                continue

        # Path Revival
        backtracks += 1
        
        # Verify if revival returns us to the gold path
        restored_depth = entry.depth + 1
        if restored_depth < len(gold_nodes) and entry.target_id == gold_nodes[restored_depth]:
            gold_path_recovered = True

        new_path = entry.path_so_far + [entry.target_id]
        
        # Reconstruct path relations up to the revived point
        new_relations = []
        for i in range(len(new_path) - 1):
            s, t = new_path[i], new_path[i+1]
            # Find edge relation matching the path traversal
            for edge in graph.get_neighbors(s):
                if edge.target == t:
                    new_relations.append(edge.relation)
                    break
        
        # Update branch state
        visited_nodes.add(entry.target_id)
        visited_edges.add((entry.node_id, entry.relation, entry.target_id))
        
        # Clear out search stacks beyond this branch point
        for d in list(local_alternatives.keys()):
            if d >= restored_depth:
                del local_alternatives[d]
        
        # Check if answer found immediately upon revival
        if entry.target_id == query.answer_node:
            recovery_successes += 1
        
        revived_val = (entry.relation, entry.rejection_margin) if rbsc_mode != "none" else entry.relation
        return entry.target_id, new_path, new_relations, 0, FailureType.NONE, backtracks, recovery_attempts, recovery_successes, gold_path_recovered, revived_val

    # Return the terminal failure status if no recovery is possible
    return path[-1], path, path_relations, 0, FailureType.DEAD_END, backtracks, recovery_attempts, recovery_successes, gold_path_recovered, None
