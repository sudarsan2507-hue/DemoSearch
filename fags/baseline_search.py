"""Baseline graph search — greedy, no failure memory, no recovery.

The search picks the highest-scoring candidate edge at each step and
terminates on the first failure.  This is the control condition against
which Failure-Aware Search is compared.
"""

from __future__ import annotations

import time

from fags import (
    Edge, FailureType, KnowledgeGraph, Query, SearchResult,
)
from fags.verifier import Verifier

# Relevance floor: two consecutive edges below this trigger path-misalignment
_RELEVANCE_FLOOR = 0.15


def _scored_candidates(graph: KnowledgeGraph, query: Query, verifier: Verifier, current: str,
                       visited_nodes: set[str], visited_edges: set[tuple[str, str, str]],
                       path_relations: list[str]) -> list[tuple[float, Edge]]:
    """Unvisited candidate edges from current, scored and sorted best first."""
    candidates: list[Edge] = [
        e for e in graph.get_neighbors(current)
        if e.target not in visited_nodes
        and (current, e.relation, e.target) not in visited_edges
    ]
    scored: list[tuple[float, Edge]] = [(verifier.score(query, edge, path_relations), edge) for edge in candidates]
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored


def _contradicts(node_obj, collected_evidence: dict[str, str]) -> bool:
    """Record the node's evidence; True as soon as a value conflicts with what was seen."""
    if node_obj and node_obj.evidence:
        for k, v in node_obj.evidence.items():
            if k in collected_evidence and collected_evidence[k] != v:
                return True
            collected_evidence[k] = v
    return False


def baseline_search(
    graph: KnowledgeGraph,
    query: Query,
    verifier: Verifier,
    max_depth: int = 10,
) -> SearchResult:
    """Run a standard greedy search (no memory, no recovery).

    Returns a SearchResult with success/failure, path, and metrics.
    """
    t0 = time.perf_counter()

    current = query.start_node
    path: list[str] = [current]
    visited_nodes: set[str] = {current}
    visited_edges: set[tuple[str, str, str]] = set()
    path_relations: list[str] = []
    edges_explored = 0
    low_score_streak = 0       # for path-misalignment detection
    collected_evidence: dict[str, str] = {}  # for contradiction detection

    # Collect evidence from start node
    start_node = graph.get_node(current)
    if start_node and start_node.evidence:
        collected_evidence.update(start_node.evidence)

    failure_type = FailureType.NONE

    for depth in range(max_depth):
        # 1-2. Get and score candidate edges
        scored = _scored_candidates(graph, query, verifier, current, visited_nodes, visited_edges, path_relations)
        if not scored:
            failure_type = FailureType.DEAD_END
            break
        edges_explored += len(scored)

        # 3. Pick best
        best_score, best_edge = scored[0]

        # 4. Track low-score streak for misalignment
        if best_score < _RELEVANCE_FLOOR:
            low_score_streak += 1
        else:
            low_score_streak = 0

        if low_score_streak >= 2:
            failure_type = FailureType.PATH_MISALIGNMENT
            break

        # 5. Traverse
        visited_edges.add((current, best_edge.relation, best_edge.target))
        current = best_edge.target
        path.append(current)
        visited_nodes.add(current)
        path_relations.append(best_edge.relation)

        # 6. Collect evidence & check contradiction
        if _contradicts(graph.get_node(current), collected_evidence):
            failure_type = FailureType.CONTRADICTION
            break

        # 7. Check answer
        if current == query.answer_node:
            elapsed = time.perf_counter() - t0
            return SearchResult(
                query_id=query.id,
                success=True,
                path=path,
                nodes_visited=len(visited_nodes),
                search_depth=len(path) - 1,
                runtime=elapsed,
                failure_type=FailureType.NONE,
                edges_explored=edges_explored,
                visited_node_set=visited_nodes,
                path_relations=list(path_relations),
            )

    # Budget exhausted (loop finished without answer or prior failure)
    if failure_type == FailureType.NONE:
        failure_type = FailureType.BUDGET_EXHAUSTED

    elapsed = time.perf_counter() - t0
    return SearchResult(
        query_id=query.id,
        success=False,
        path=path,
        nodes_visited=len(visited_nodes),
        search_depth=len(path) - 1,
        runtime=elapsed,
        failure_type=failure_type,
        edges_explored=edges_explored,
        visited_node_set=visited_nodes,
        path_relations=list(path_relations),
    )
