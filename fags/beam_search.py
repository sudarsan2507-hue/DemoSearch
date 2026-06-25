"""Beam search — a structurally different alternative to FAGS.

Every mechanism tested so far (Top1/Top2/Threshold/Diversity memory,
shield, certificate, RBSC, RTC-lite, learned failure-pattern avoidance) sits
on top of the same architecture: walk one path greedily, and when it fails,
decide which single rejected candidate to revive and resume from. Five
independent experiments found that architecture isn't reliably better than
spending the same search budget on diversified random restarts - the
problem isn't *which* candidate gets revived, it's the commit-then-recover
design itself.

Beam search never fully commits to one path. It keeps the K highest-scoring
live hypotheses at every hop, expands all of them in parallel, and prunes
back down to K by score - so a wrong early guess doesn't require detecting
failure and backtracking; the better alternative was already being explored
the whole time. Beam width K controls the cost/accuracy tradeoff, playing
the same role FAGS's backtrack budget does.

A candidate transition into a node whose evidence contradicts evidence
already collected on that hypothesis's path is dropped from consideration
(not scored) - same hard constraint baseline_search/failure_search enforce
via FailureType.CONTRADICTION, just applied per-hypothesis instead of
ending the whole search. The PATH_MISALIGNMENT early-exit heuristic
(two consecutive low scores) is not needed here: a hypothesis that wanders
off-track simply loses to better-scoring siblings at the next prune, which
is what that heuristic was a single-path substitute for.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from fags import Edge, FailureType, KnowledgeGraph, Query, SearchResult
from fags.verifier import Verifier


@dataclass
class _Hypothesis:
    path: list[str]
    path_relations: list[str]
    visited_nodes: set[str]
    visited_edges: set[tuple[str, str, str]]
    collected_evidence: dict[str, str]
    cumulative_score: float  # running mean of per-hop verifier scores
    n_hops: int = 0


def _has_contradiction(hyp: _Hypothesis, target_evidence: dict) -> bool:
    if not target_evidence:
        return False
    for k, v in target_evidence.items():
        if k in hyp.collected_evidence and hyp.collected_evidence[k] != v:
            return True
    return False


def beam_search(
    graph: KnowledgeGraph,
    query: Query,
    verifier: Verifier,
    beam_width: int = 5,
    max_depth: int = 6,
    max_children_per_parent: Optional[int] = None,
) -> SearchResult:
    """Run beam search: K live hypotheses expanded and pruned in parallel.

    max_children_per_parent : if set, caps how many of the new beam's slots
        a single parent hypothesis may fill. Plain top-K pruning can let one
        strong-but-wrong early branch supply most/all of the next beam,
        crowding out genuinely different hypotheses from weaker-scoring
        parents; capping forces a spread across distinct lineages instead.
        The cap is relaxed (to ceil(beam_width / live_parent_count)) when
        there are fewer live parents than that would allow, so a single
        start node can still grow the beam out to full width on the first
        hop instead of being stuck at size 1 forever. None (default)
        reproduces the original unconstrained top-K behaviour.
    """
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

    beam = [_Hypothesis(
        path=[start], path_relations=[], visited_nodes={start}, visited_edges=set(),
        collected_evidence=start_evidence, cumulative_score=0.0, n_hops=0,
    )]

    for _ in range(max_depth):
        pool: list[tuple[float, _Hypothesis, Edge]] = []

        for hyp in beam:
            current = hyp.path[-1]
            for edge in graph.get_neighbors(current):
                if edge.target in hyp.visited_nodes:
                    continue
                if (current, edge.relation, edge.target) in hyp.visited_edges:
                    continue
                target_node = graph.get_node(edge.target)
                target_evidence = target_node.evidence if target_node else {}
                if _has_contradiction(hyp, target_evidence):
                    continue  # invalid transition, never scored

                s = verifier.score(query, edge, hyp.path_relations)
                edges_explored += 1
                new_cumulative = (hyp.cumulative_score * hyp.n_hops + s) / (hyp.n_hops + 1)
                pool.append((new_cumulative, hyp, edge))

        if not pool:
            break  # every live hypothesis dead-ended

        pool.sort(key=lambda x: x[0], reverse=True)
        new_beam: list[_Hypothesis] = []
        children_count: dict[int, int] = {}

        effective_cap = None
        if max_children_per_parent is not None:
            live_parent_count = len({id(p) for _, p, _ in pool})
            adaptive_floor = -(-beam_width // live_parent_count)  # ceil division
            effective_cap = max(max_children_per_parent, adaptive_floor)

        for new_cumulative, parent, edge in pool:
            if len(new_beam) >= beam_width:
                break
            if effective_cap is not None:
                pid = id(parent)
                if children_count.get(pid, 0) >= effective_cap:
                    continue
                children_count[pid] = children_count.get(pid, 0) + 1

            new_path = parent.path + [edge.target]
            all_visited_nodes.add(edge.target)

            if edge.target == query.answer_node:
                elapsed = time.perf_counter() - t0
                return SearchResult(
                    query_id=query.id, success=True, path=new_path,
                    nodes_visited=len(all_visited_nodes), search_depth=len(new_path) - 1,
                    runtime=elapsed, failure_type=FailureType.NONE, edges_explored=edges_explored,
                    visited_node_set=all_visited_nodes, path_relations=parent.path_relations + [edge.relation],
                )

            target_node = graph.get_node(edge.target)
            merged_evidence = dict(parent.collected_evidence)
            if target_node and target_node.evidence:
                merged_evidence.update(target_node.evidence)

            new_beam.append(_Hypothesis(
                path=new_path,
                path_relations=parent.path_relations + [edge.relation],
                visited_nodes=parent.visited_nodes | {edge.target},
                visited_edges=parent.visited_edges | {(parent.path[-1], edge.relation, edge.target)},
                collected_evidence=merged_evidence,
                cumulative_score=new_cumulative,
                n_hops=parent.n_hops + 1,
            ))

        beam = new_beam
        if not beam:
            break

    elapsed = time.perf_counter() - t0
    if beam:
        best = max(beam, key=lambda h: h.cumulative_score)
        failure_type = FailureType.BUDGET_EXHAUSTED
    else:
        best = None
        failure_type = FailureType.DEAD_END

    return SearchResult(
        query_id=query.id,
        success=False,
        path=best.path if best else [start],
        nodes_visited=len(all_visited_nodes),
        search_depth=(len(best.path) - 1) if best else 0,
        runtime=elapsed,
        failure_type=failure_type,
        edges_explored=edges_explored,
        visited_node_set=all_visited_nodes,
        path_relations=best.path_relations if best else [],
    )
