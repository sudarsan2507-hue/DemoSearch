"""Global best-first search — generalizes beam search's fixed-width-per-hop
pruning into a budget-capped global frontier.

Beam search keeps exactly K hypotheses alive at *every* depth, uniformly,
regardless of how promising any of them are - width=8 spends the same
per-hop budget whether the top-8 hypotheses are all strong or only 2 are.
Best-first search instead keeps one global priority queue across all
depths and always expands whichever hypothesis currently has the best
cumulative score, until a total node-visit budget is spent - concentrating
compute on whatever looks best rather than spreading it evenly across a
fixed width. Four independent refinements *on top of* beam search's
pruning rule (hard diversity cap, soft diversity penalty, sum aggregation,
learned failure-pattern composition) all failed to beat it; this changes
the pruning paradigm itself instead of tuning a rule within it.

Same per-hypothesis constraints as beam_search.py: a transition into
evidence-contradicting territory is dropped (not scored), and there's no
PATH_MISALIGNMENT early-exit since a wandering hypothesis just won't be the
frontier's best score and will sit unexpanded until the budget runs out.
"""

from __future__ import annotations

import heapq
import time
from dataclasses import dataclass

from fags import Edge, FailureType, KnowledgeGraph, Query, SearchResult
from fags.verifier import Verifier


@dataclass
class _Node:
    path: list[str]
    path_relations: list[str]
    visited_nodes: set[str]
    visited_edges: set[tuple[str, str, str]]
    collected_evidence: dict[str, str]
    cumulative_score: float
    n_hops: int = 0


def _has_contradiction(node: _Node, target_evidence: dict) -> bool:
    if not target_evidence:
        return False
    for k, v in target_evidence.items():
        if k in node.collected_evidence and node.collected_evidence[k] != v:
            return True
    return False


def best_first_search(
    graph: KnowledgeGraph,
    query: Query,
    verifier: Verifier,
    node_budget: int = 20,
    max_depth: int = 6,
) -> SearchResult:
    """Expand the single best-scoring frontier hypothesis at a time, globally
    across all depths, until node_budget distinct nodes have been visited."""
    t0 = time.perf_counter()
    start = query.start_node

    start_node_obj = graph.get_node(start)
    start_evidence = dict(start_node_obj.evidence) if start_node_obj and start_node_obj.evidence else {}

    all_visited_nodes: set[str] = {start}
    edges_explored = 0

    if start == query.answer_node:
        elapsed = time.perf_counter() - t0
        return SearchResult(
            query_id=query.id, success=True, path=[start], nodes_visited=1, search_depth=0,
            runtime=elapsed, failure_type=FailureType.NONE, edges_explored=0,
            visited_node_set=all_visited_nodes, path_relations=[],
        )

    root = _Node(
        path=[start], path_relations=[], visited_nodes={start}, visited_edges=set(),
        collected_evidence=start_evidence, cumulative_score=0.0, n_hops=0,
    )
    counter = 0
    frontier: list[tuple[float, int, _Node]] = [(0.0, counter, root)]
    best_seen = root

    while frontier and len(all_visited_nodes) < node_budget:
        neg_score, _, node = heapq.heappop(frontier)
        if node.n_hops >= max_depth:
            continue

        current = node.path[-1]
        for edge in graph.get_neighbors(current):
            if edge.target in node.visited_nodes:
                continue
            if (current, edge.relation, edge.target) in node.visited_edges:
                continue
            target_node = graph.get_node(edge.target)
            target_evidence = target_node.evidence if target_node else {}
            if _has_contradiction(node, target_evidence):
                continue

            s = verifier.score(query, edge, node.path_relations)
            edges_explored += 1
            new_cumulative = (node.cumulative_score * node.n_hops + s) / (node.n_hops + 1)
            new_path = node.path + [edge.target]
            all_visited_nodes.add(edge.target)

            if edge.target == query.answer_node:
                elapsed = time.perf_counter() - t0
                return SearchResult(
                    query_id=query.id, success=True, path=new_path,
                    nodes_visited=len(all_visited_nodes), search_depth=len(new_path) - 1,
                    runtime=elapsed, failure_type=FailureType.NONE, edges_explored=edges_explored,
                    visited_node_set=all_visited_nodes, path_relations=node.path_relations + [edge.relation],
                )

            merged_evidence = dict(node.collected_evidence)
            if target_node and target_node.evidence:
                merged_evidence.update(target_node.evidence)

            child = _Node(
                path=new_path,
                path_relations=node.path_relations + [edge.relation],
                visited_nodes=node.visited_nodes | {edge.target},
                visited_edges=node.visited_edges | {(current, edge.relation, edge.target)},
                collected_evidence=merged_evidence,
                cumulative_score=new_cumulative,
                n_hops=node.n_hops + 1,
            )
            counter += 1
            heapq.heappush(frontier, (-new_cumulative, counter, child))
            if new_cumulative > best_seen.cumulative_score:
                best_seen = child

            if len(all_visited_nodes) >= node_budget:
                break

    elapsed = time.perf_counter() - t0
    failure_type = FailureType.BUDGET_EXHAUSTED if best_seen.n_hops > 0 else FailureType.DEAD_END
    return SearchResult(
        query_id=query.id,
        success=False,
        path=best_seen.path,
        nodes_visited=len(all_visited_nodes),
        search_depth=best_seen.n_hops,
        runtime=elapsed,
        failure_type=failure_type,
        edges_explored=edges_explored,
        visited_node_set=all_visited_nodes,
        path_relations=best_seen.path_relations,
    )
