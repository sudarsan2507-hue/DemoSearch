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
from dataclasses import dataclass, field
from typing import Any

from fags import (
    Edge, FailureType, KnowledgeGraph, Query, SearchResult,
)
from fags.verifier import Verifier, RELATION_COHERENCE
from fags.memory import FailureMemory

# Relevance floor: two consecutive edges below this trigger path-misalignment
_RELEVANCE_FLOOR = 0.15


@dataclass
class _SearchState:
    """Everything one FAGS run tracks while it walks the graph."""
    current: str
    path: list[str]
    path_relations: list[str] = field(default_factory=list)
    # Branch Memory: global set of visited nodes and edges to prevent cycles
    visited_nodes: set[str] = field(default_factory=set)
    visited_edges: set[tuple[str, str, str]] = field(default_factory=set)
    # Evidence tracker for contradiction detection
    collected_evidence: dict[str, str] = field(default_factory=dict)
    # Local backtracking stack of candidates at each level of the current path
    # Maps: depth -> list of (score, edge) untried at that level
    local_alternatives: dict[int, list[tuple[float, Edge]]] = field(default_factory=dict)
    # Tracking list of failed relations encountered across search branches
    failed_relations: list[str] = field(default_factory=list)
    # Diagnostics
    edges_explored: int = 0
    backtracks: int = 0
    recovery_attempts: int = 0
    recovery_successes: int = 0
    # Gold path tracking
    gold_path_pruned: bool = False
    gold_path_recovered: bool = False
    shield_hops_remaining: int = 0
    active_certificate: Any = None
    current_revival_hops: int = 0
    is_post_revival: bool = False
    hops_survived_post_revival: list[int] = field(default_factory=list)
    # RBSC Tracking
    revived_margins: list[float] = field(default_factory=list)
    low_score_streak: int = 0
    failure_type: FailureType = FailureType.NONE

    def close_revival_window(self) -> None:
        if self.current_revival_hops > 0:
            self.hops_survived_post_revival.append(self.current_revival_hops)
        self.current_revival_hops = 0


@dataclass
class _SearchConfig:
    max_backtracks: int
    enable_re_verification: bool
    shield_depth: int
    use_certificate: bool
    certificate_bonus: float
    rbsc_mode: str


def _recover(st: _SearchState, graph: KnowledgeGraph, query: Query, verifier: Verifier,
             memory: FailureMemory, cfg: _SearchConfig) -> bool:
    """Run the recovery pipeline. Returns True when the search must stop."""
    (st.current, st.path, st.path_relations, st.low_score_streak, st.failure_type, st.backtracks,
     st.recovery_attempts, st.recovery_successes, st.gold_path_recovered, revived_rel) = _handle_recovery(
        graph, query, verifier, memory, st.path, st.path_relations, st.visited_nodes, st.visited_edges,
        st.local_alternatives, st.failed_relations, cfg.max_backtracks, cfg.enable_re_verification,
        st.backtracks, st.recovery_attempts, st.recovery_successes, st.gold_path_recovered, query.gold_path,
        cfg.rbsc_mode
    )
    st.close_revival_window()
    if revived_rel is not None:
        st.active_certificate = revived_rel
        st.shield_hops_remaining = cfg.shield_depth
        st.is_post_revival = True
        if cfg.rbsc_mode != "none" and isinstance(revived_rel, tuple):
            st.revived_margins.append(revived_rel[1])
    else:
        st.is_post_revival = False
    # Recovery failed to find a path when a failure type is left set
    return st.failure_type != FailureType.NONE


def _certificate_boost(s: float, edge: Edge, active_certificate, cfg: _SearchConfig) -> float:
    """Certificate Bonus for relations coherent with the revived one."""
    if cfg.rbsc_mode != "none" and isinstance(active_certificate, tuple):
        cert_rel, cert_margin = active_certificate
        coherence = RELATION_COHERENCE.get((edge.relation, cert_rel), 0.3)
        if coherence > 0.5:
            if cfg.rbsc_mode == "linear":
                multiplier = max(0.0, 1.0 - cert_margin)
            elif cfg.rbsc_mode == "nonlinear":
                multiplier = 1.0 / (1.0 + 5.0 * cert_margin)
            else:
                multiplier = 1.0
            s = min(1.0, s + cfg.certificate_bonus * multiplier)
        return s
    coherence = RELATION_COHERENCE.get((edge.relation, active_certificate), 0.3)
    if coherence > 0.5:
        s = min(1.0, s + cfg.certificate_bonus)
    return s


def _score_level(st: _SearchState, graph: KnowledgeGraph, query: Query, verifier: Verifier,
                 cfg: _SearchConfig) -> list[tuple[float, Edge]]:
    """Score the unvisited candidate edges at the current node, best first."""
    candidates: list[Edge] = [
        e for e in graph.get_neighbors(st.current)
        if e.target not in st.visited_nodes
        and (st.current, e.relation, e.target) not in st.visited_edges
    ]
    scored: list[tuple[float, Edge]] = []
    for edge in candidates:
        s = verifier.score(query, edge, st.path_relations)
        if cfg.use_certificate and st.active_certificate is not None:
            s = _certificate_boost(s, edge, st.active_certificate, cfg)
        scored.append((s, edge))
        st.edges_explored += 1
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored


def _note_gold_pruning(st: _SearchState, query: Query, depth: int, best_edge: Edge) -> None:
    """Check if the correct gold relation was outranked at this step."""
    gold_nodes = query.gold_path
    if depth < len(gold_nodes) - 1 and st.current == gold_nodes[depth]:
        if best_edge.relation != query.gold_relations[depth]:
            st.gold_path_pruned = True


def _traverse(st: _SearchState, depth: int, best_edge: Edge) -> None:
    st.visited_edges.add((st.current, best_edge.relation, best_edge.target))
    # Consume the candidate
    st.local_alternatives[depth].pop(0)
    if st.active_certificate is not None:
        st.shield_hops_remaining = max(0, st.shield_hops_remaining - 1)
        if st.shield_hops_remaining == 0:
            st.active_certificate = None
    if st.is_post_revival:
        st.current_revival_hops += 1
    st.current = best_edge.target
    st.path.append(st.current)
    st.visited_nodes.add(st.current)
    st.path_relations.append(best_edge.relation)


def _contradicts(node_obj, collected_evidence: dict[str, str]) -> bool:
    """Record the node's evidence; True as soon as a value conflicts with what was seen."""
    if node_obj and node_obj.evidence:
        for k, v in node_obj.evidence.items():
            if k in collected_evidence and collected_evidence[k] != v:
                return True
            collected_evidence[k] = v
    return False


def _result(st: _SearchState, query: Query, memory: FailureMemory, t0: float, success: bool) -> SearchResult:
    elapsed = time.perf_counter() - t0
    # Successful runs record their recovery margins as successful, failed runs as failed
    margins = list(st.revived_margins)
    extra = {"hops_survived_post_revival": st.hops_survived_post_revival} if success else {}
    return SearchResult(
        query_id=query.id,
        success=success,
        path=st.path,
        nodes_visited=len(st.visited_nodes),
        search_depth=len(st.path) - 1,
        runtime=elapsed,
        failure_type=FailureType.NONE if success else st.failure_type,
        backtracks=st.backtracks,
        recovery_attempts=st.recovery_attempts,
        recovery_successes=st.recovery_successes,
        gold_path_pruned=st.gold_path_pruned,
        gold_path_recovered=st.gold_path_recovered,
        memory_size_at_end=memory.size,
        edges_explored=st.edges_explored,
        visited_node_set=st.visited_nodes,
        successful_recovery_margins=margins if success else [],
        failed_recovery_margins=[] if success else margins,
        path_relations=list(st.path_relations),
        **extra,
    )


_CONTINUE, _STOP, _FOUND = "continue", "stop", "found"


def _fail_and_recover(st: _SearchState, failure: FailureType, graph: KnowledgeGraph, query: Query,
                      verifier: Verifier, memory: FailureMemory, cfg: _SearchConfig) -> str:
    st.failure_type = failure
    return _STOP if _recover(st, graph, query, verifier, memory, cfg) else _CONTINUE


def _search_step(st: _SearchState, graph: KnowledgeGraph, query: Query, verifier: Verifier,
                 memory: FailureMemory, cfg: _SearchConfig, max_depth: int) -> str:
    """One iteration of the FAGS loop."""
    depth = len(st.path) - 1
    if depth >= max_depth:
        return _fail_and_recover(st, FailureType.BUDGET_EXHAUSTED, graph, query, verifier, memory, cfg)

    # 1-2. Query and score candidate edges if not already done for this level
    if depth not in st.local_alternatives:
        st.local_alternatives[depth] = _score_level(st, graph, query, verifier, cfg)
    level_candidates = st.local_alternatives[depth]
    if not level_candidates:
        return _fail_and_recover(st, FailureType.DEAD_END, graph, query, verifier, memory, cfg)

    # Get best candidate at this level
    best_score, best_edge = level_candidates[0]
    _note_gold_pruning(st, query, depth, best_edge)

    # Keep track of rejected alternatives for failure memory
    rejected = level_candidates[1:]
    if rejected:
        memory.store(
            current_node=st.current,
            candidates=rejected,
            winner_score=best_score,
            depth=depth,
            path_so_far=list(st.path),
            winner_relation=best_edge.relation,
        )

    # 3. Path misalignment check
    current_floor = 0.05 if st.shield_hops_remaining > 0 else _RELEVANCE_FLOOR
    current_low_streak = st.low_score_streak + 1 if best_score < current_floor else 0
    if current_low_streak >= 2:
        st.failed_relations.append(best_edge.relation)
        # Remove this candidate from future tries at this level
        st.local_alternatives[depth].pop(0)
        return _fail_and_recover(st, FailureType.PATH_MISALIGNMENT, graph, query, verifier, memory, cfg)

    # 4. Traverse
    st.low_score_streak = current_low_streak
    _traverse(st, depth, best_edge)

    # 5. Check contradiction
    if _contradicts(graph.get_node(st.current), st.collected_evidence):
        st.failed_relations.append(best_edge.relation)
        return _fail_and_recover(st, FailureType.CONTRADICTION, graph, query, verifier, memory, cfg)

    # 6. Check answer node
    if st.current == query.answer_node:
        st.close_revival_window()
        return _FOUND
    return _CONTINUE


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
    cfg = _SearchConfig(max_backtracks, enable_re_verification, shield_depth,
                        use_certificate, certificate_bonus, rbsc_mode)

    st = _SearchState(current=query.start_node, path=[query.start_node], visited_nodes={query.start_node})
    start_node = graph.get_node(st.current)
    if start_node and start_node.evidence:
        st.collected_evidence.update(start_node.evidence)

    while True:
        outcome = _search_step(st, graph, query, verifier, memory, cfg, max_depth)
        if outcome == _FOUND:
            return _result(st, query, memory, t0, success=True)
        if outcome == _STOP:
            break

    return _result(st, query, memory, t0, success=False)


def _revivable(entry, query: Query, verifier: Verifier, visited_nodes: set[str],
               failed_relations: list[str], enable_re_verification: bool) -> bool:
    """Branch Memory and Dynamic Re-Verification checks for one memory entry."""
    # Check Branch Memory: skip if the target node is already in visited_nodes
    if entry.target_id in visited_nodes:
        return False
    if not enable_re_verification:
        return True
    re_score = verifier.re_score(
        query,
        Edge(source=entry.node_id, target=entry.target_id, relation=entry.relation),
        entry.path_so_far,
        failed_relations
    )
    # If the re-score drops below viability (e.g. 0.20), discard
    return re_score >= 0.20


def _relations_along(graph: KnowledgeGraph, new_path: list[str]) -> list[str]:
    """Reconstruct path relations up to the revived point."""
    new_relations = []
    for i in range(len(new_path) - 1):
        s, t = new_path[i], new_path[i+1]
        # Find edge relation matching the path traversal
        for edge in graph.get_neighbors(s):
            if edge.target == t:
                new_relations.append(edge.relation)
                break
    return new_relations


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
        path.pop()
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
        if not _revivable(entry, query, verifier, visited_nodes, failed_relations, enable_re_verification):
            continue

        # Path Revival
        backtracks += 1

        # Verify if revival returns us to the gold path
        restored_depth = entry.depth + 1
        if restored_depth < len(gold_nodes) and entry.target_id == gold_nodes[restored_depth]:
            gold_path_recovered = True

        new_path = entry.path_so_far + [entry.target_id]
        new_relations = _relations_along(graph, new_path)

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
