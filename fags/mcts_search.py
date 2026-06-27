"""Monte Carlo Tree Search — an exploration/exploitation-balanced alternative
to both beam search and global best-first search.

Global best-first search (fags/best_first_search.py) lost decisively to
beam search (0/12 configurations) because greedily expanding only the
single best-looking frontier node lets the search tunnel into one wrong
branch with no pressure to ever reconsider once early scores favor it.
MCTS is built specifically to avoid that failure mode: UCB1 selection
explicitly balances "exploit what looks good" against "make sure
under-visited branches still get a chance", and many independent rollouts
feed back into value estimates the tree actually trusts, rather than
committing budget to one running candidate.

Each tree node represents one (graph node, path-so-far) state, carrying its
own visited_nodes/visited_edges (cycle avoidance) and collected_evidence
(contradiction avoidance) inherited and extended from its parent - the same
per-hypothesis bookkeeping beam_search.py and best_first_search.py use.

One simulation = selection (descend via UCB1 through fully-expanded nodes)
+ expansion (add one new child via its highest-scoring untried edge) +
rollout (greedy-by-verifier-score playout from the new child, not added to
the tree, used only to estimate a 0/1 reward) + backpropagation (update
visit counts/values up to the root). Budget is a node-visit cap counting
every distinct graph node touched anywhere - tree or rollout - so it is
directly comparable to beam_search's/best_first_search's nodes_visited cost.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Optional

from fags import Edge, FailureType, KnowledgeGraph, Query, SearchResult
from fags.verifier import Verifier


@dataclass
class _TreeNode:
    graph_node: str
    path_relations: list[str]
    visited_nodes: set[str]
    visited_edges: set[tuple[str, str, str]]
    collected_evidence: dict[str, str]
    depth: int
    parent: Optional["_TreeNode"] = None
    children: dict[str, "_TreeNode"] = field(default_factory=dict)
    untried: Optional[list[tuple[float, Edge]]] = None
    visits: int = 0
    value_sum: float = 0.0
    is_terminal: bool = False

    @property
    def path(self) -> list[str]:
        nodes = []
        n: Optional[_TreeNode] = self
        while n is not None:
            nodes.append(n.graph_node)
            n = n.parent
        return list(reversed(nodes))


def _has_contradiction(collected_evidence: dict, target_evidence: dict) -> bool:
    if not target_evidence:
        return False
    for k, v in target_evidence.items():
        if k in collected_evidence and collected_evidence[k] != v:
            return True
    return False


def _score_candidates(
    graph: KnowledgeGraph, query: Query, verifier: Verifier,
    current: str, path_relations: list[str], visited_nodes: set, visited_edges: set,
    collected_evidence: dict, edges_explored_ref: list[int],
) -> list[tuple[float, Edge]]:
    cands = []
    for edge in graph.get_neighbors(current):
        if edge.target in visited_nodes:
            continue
        if (current, edge.relation, edge.target) in visited_edges:
            continue
        target_node = graph.get_node(edge.target)
        target_evidence = target_node.evidence if target_node else {}
        if _has_contradiction(collected_evidence, target_evidence):
            continue
        s = verifier.score(query, edge, path_relations)
        edges_explored_ref[0] += 1
        cands.append((s, edge))
    cands.sort(key=lambda x: x[0], reverse=True)
    return cands


def _rollout(
    graph: KnowledgeGraph, query: Query, verifier: Verifier, start: _TreeNode,
    all_visited_nodes: set, edges_explored_ref: list[int], max_depth: int,
):
    """Greedy-by-score playout from `start`, never added to the tree.
    Returns (reward, (full_path, full_path_relations)) on success, else (0.0, None)."""
    current = start.graph_node
    path = list(start.path)
    path_relations = list(start.path_relations)
    visited_nodes = set(start.visited_nodes)
    visited_edges = set(start.visited_edges)
    collected_evidence = dict(start.collected_evidence)
    depth = start.depth

    while depth < max_depth:
        cands = _score_candidates(
            graph, query, verifier, current, path_relations, visited_nodes, visited_edges,
            collected_evidence, edges_explored_ref,
        )
        if not cands:
            return 0.0, None
        _, best_edge = cands[0]

        all_visited_nodes.add(best_edge.target)
        path = path + [best_edge.target]
        path_relations = path_relations + [best_edge.relation]
        visited_edges = visited_edges | {(current, best_edge.relation, best_edge.target)}
        visited_nodes = visited_nodes | {best_edge.target}

        if best_edge.target == query.answer_node:
            return 1.0, (path, path_relations)

        target_node = graph.get_node(best_edge.target)
        if target_node and target_node.evidence:
            collected_evidence.update(target_node.evidence)

        current = best_edge.target
        depth += 1

    return 0.0, None


def mcts_search(
    graph: KnowledgeGraph,
    query: Query,
    verifier: Verifier,
    node_budget: int = 20,
    max_depth: int = 6,
    exploration_constant: float = 1.0,
) -> SearchResult:
    t0 = time.perf_counter()
    start = query.start_node

    start_node_obj = graph.get_node(start)
    start_evidence = dict(start_node_obj.evidence) if start_node_obj and start_node_obj.evidence else {}

    all_visited_nodes: set[str] = {start}
    edges_explored_ref = [0]

    if start == query.answer_node:
        elapsed = time.perf_counter() - t0
        return SearchResult(
            query_id=query.id, success=True, path=[start], nodes_visited=1, search_depth=0,
            runtime=elapsed, failure_type=FailureType.NONE, edges_explored=0,
            visited_node_set=all_visited_nodes, path_relations=[],
        )

    root = _TreeNode(
        graph_node=start, path_relations=[], visited_nodes={start}, visited_edges=set(),
        collected_evidence=start_evidence, depth=0,
    )

    def _ucb(child: _TreeNode, parent_visits: int) -> float:
        if child.visits == 0:
            return float("inf")
        exploitation = child.value_sum / child.visits
        exploration = exploration_constant * math.sqrt(math.log(parent_visits) / child.visits)
        return exploitation + exploration

    # Safety cap: a node_budget that's never reached (e.g. a genuine dead-end
    # start with zero candidates) would otherwise spin forever re-selecting
    # the same terminal root. A second case needs its own guard: once
    # node_budget exceeds the graph's actual reachable size, all_visited_nodes
    # can never reach it no matter how exhaustively the tree is searched -
    # stale_count detects "no new node added for a while" (the reachable
    # space is fully explored) and bails, independent of how large
    # node_budget is. max_simulations is a fixed, generous backstop, not
    # scaled by node_budget (scaling it was the actual source of the
    # 128,000+-simulation risk this comment used to warn about).
    max_simulations = 20000
    stale_limit = 300
    stale_count = 0
    simulations = 0

    while len(all_visited_nodes) < node_budget and simulations < max_simulations and stale_count < stale_limit:
        simulations += 1
        _prev_visited_count = len(all_visited_nodes)

        # --- Selection ---
        node = root
        while True:
            if node.untried is None:
                node.untried = _score_candidates(
                    graph, query, verifier, node.graph_node, node.path_relations,
                    node.visited_nodes, node.visited_edges, node.collected_evidence, edges_explored_ref,
                )
            if node.depth >= max_depth or (not node.untried and not node.children):
                node.is_terminal = True
                break
            if node.untried:
                break  # expand here
            node = max(node.children.values(), key=lambda c: _ucb(c, node.visits))

        # --- Expansion + rollout ---
        if node.is_terminal:
            reward = 0.0
            leaf = node
        else:
            score, edge = node.untried.pop(0)
            child_id = edge.target
            all_visited_nodes.add(child_id)

            if child_id == query.answer_node:
                elapsed = time.perf_counter() - t0
                return SearchResult(
                    query_id=query.id, success=True, path=node.path + [child_id],
                    nodes_visited=len(all_visited_nodes), search_depth=node.depth + 1,
                    runtime=elapsed, failure_type=FailureType.NONE, edges_explored=edges_explored_ref[0],
                    visited_node_set=all_visited_nodes, path_relations=node.path_relations + [edge.relation],
                )

            target_node = graph.get_node(child_id)
            merged_evidence = dict(node.collected_evidence)
            if target_node and target_node.evidence:
                merged_evidence.update(target_node.evidence)

            child = _TreeNode(
                graph_node=child_id,
                path_relations=node.path_relations + [edge.relation],
                visited_nodes=node.visited_nodes | {child_id},
                visited_edges=node.visited_edges | {(node.graph_node, edge.relation, child_id)},
                collected_evidence=merged_evidence,
                depth=node.depth + 1,
                parent=node,
            )
            node.children[child_id] = child

            reward, success_info = _rollout(graph, query, verifier, child, all_visited_nodes, edges_explored_ref, max_depth)
            if success_info is not None:
                full_path, full_relations = success_info
                elapsed = time.perf_counter() - t0
                return SearchResult(
                    query_id=query.id, success=True, path=full_path,
                    nodes_visited=len(all_visited_nodes), search_depth=len(full_path) - 1,
                    runtime=elapsed, failure_type=FailureType.NONE, edges_explored=edges_explored_ref[0],
                    visited_node_set=all_visited_nodes, path_relations=full_relations,
                )
            leaf = child

        # --- Backpropagation ---
        n: Optional[_TreeNode] = leaf
        while n is not None:
            n.visits += 1
            n.value_sum += reward
            n = n.parent

        if len(all_visited_nodes) == _prev_visited_count:
            stale_count += 1
        else:
            stale_count = 0

    # Budget exhausted without success: report the tree's most-trusted path
    # (most-visited child at each level, the standard MCTS "robust child" rule).
    elapsed = time.perf_counter() - t0
    node = root
    while node.children:
        node = max(node.children.values(), key=lambda c: c.visits)
    failure_type = FailureType.BUDGET_EXHAUSTED if node.depth > 0 else FailureType.DEAD_END
    return SearchResult(
        query_id=query.id,
        success=False,
        path=node.path,
        nodes_visited=len(all_visited_nodes),
        search_depth=node.depth,
        runtime=elapsed,
        failure_type=failure_type,
        edges_explored=edges_explored_ref[0],
        visited_node_set=all_visited_nodes,
        path_relations=node.path_relations,
    )
