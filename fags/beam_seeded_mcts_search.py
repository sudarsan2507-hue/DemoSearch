"""Beam-Seeded MCTS — fuses beam search's structural diversity guarantee
with MCTS's value-learning refinement.

mcts_search_experiment.py and mcts_large_budget_experiment.py established
two regimes: plain beam search wins at small/moderate budgets (5-200
nodes), MCTS wins at large ones (200+ nodes) - but needs that much budget
because its early simulations are mostly spent rediscovering diversity
beam search gets for free (a guaranteed K distinct hypotheses every hop,
vs. MCTS's single root that has to grow that breadth through trial).

This spends the FIRST `seed_depth` hops running plain beam-search-style
expansion - global top-K pruning by cumulative score, exactly
fags/beam_search.py's algorithm - directly on the MCTS tree structure,
producing up to `beam_width` diverse surviving hypotheses ("tips") at
depth `seed_depth`. Every node visited during seeding is sealed
(`untried = []`) once its round's children are decided, so the beam
phase's pruning decisions are final - MCTS never re-litigates a branch
beam already discarded, keeping the two phases' responsibilities cleanly
separated. Seeding is capped by DEPTH rather than node count: max_depth is
a hard hop limit every algorithm in this codebase shares, so a node-budget-
only stop condition would let seeding consume the whole path length
whenever node_budget is generous, leaving MCTS with zero hops of room to
refine in (this was caught empirically while building this module - see
_beam_seed's docstring).

The REMAINING budget then runs the standard MCTS loop (selection via
UCB1, expansion, rollout, backpropagation - identical to
fags/mcts_search.py) starting from those tips instead of a blank single
node, so UCB1 only ever has to choose how to invest additional budget
among a set that's already structurally diverse, rather than discover
that diversity itself.

Hypothesis under test: this should let MCTS's value-learning pay off at
budgets well below the ~200-300 node crossover point found for pure MCTS,
since the "wasted early simulations" problem is sidestepped by construction.
"""

from __future__ import annotations

import math
import time
from typing import Optional

from fags import Edge, FailureType, KnowledgeGraph, Query, SearchResult
from fags.verifier import Verifier
from fags.mcts_search import _TreeNode, _score_candidates, _rollout


def _beam_seed(
    graph: KnowledgeGraph, query: Query, verifier: Verifier, root: _TreeNode,
    node_budget: int, seed_depth: int, beam_width: int, max_depth: int,
    all_visited_nodes: set, edges_explored_ref: list[int], t0: float,
):
    """Run plain beam-search-style expansion from `root`, building real
    parent/child links directly on the MCTS tree and sealing every visited
    node's `untried` to [] once its round's children are decided.

    Stops at `seed_depth` hops, NOT a node-count budget - max_depth is a
    hard hop cap shared by every algorithm in this codebase (you can't
    extend a path past it), so a node-count-only stop condition would let
    seeding run all the way to max_depth whenever node_budget is generous,
    leaving the MCTS refinement phase with zero hops of room to operate in
    (confirmed empirically: at seed_budget~24/beam_width=5/max_depth=6,
    67% of non-immediately-successful seedings ended with every tip already
    at max_depth). Capping by depth instead guarantees max_depth - seed_depth
    hops always remain for MCTS, regardless of how much budget is available.
    node_budget is still enforced as an overall safety cap.

    Returns (tips, success_result): `tips` is the list of surviving
    frontier _TreeNode objects (empty if every hypothesis died out before
    reaching seed_depth); `success_result` is a completed SearchResult if
    the answer was found during seeding, else None.
    """
    # _TreeNode has no cumulative_score field (that's beam_search.py's
    # _Hypothesis) - tracked here in a side-dict instead of extending the
    # shared MCTS node type for a value only the seeding phase needs.
    cumulative_scores: dict[int, float] = {id(root): 0.0}
    current_level = [root]

    while (
        current_level
        and current_level[0].depth < seed_depth
        and len(all_visited_nodes) < node_budget
    ):
        pool: list[tuple[float, _TreeNode, Edge]] = []
        for node in current_level:
            if node.depth >= max_depth:
                node.untried = []
                continue
            node.untried = _score_candidates(
                graph, query, verifier, node.graph_node, node.path_relations,
                node.visited_nodes, node.visited_edges, node.collected_evidence, edges_explored_ref,
            )
            for score, edge in node.untried:
                pool.append((score, node, edge))

        if not pool:
            break  # every live hypothesis dead-ended or hit max_depth

        pool.sort(key=lambda x: x[0], reverse=True)
        next_level: list[_TreeNode] = []

        for score, parent, edge in pool[:beam_width]:
            child_id = edge.target
            all_visited_nodes.add(child_id)

            if child_id == query.answer_node:
                elapsed = time.perf_counter() - t0
                result = SearchResult(
                    query_id=query.id, success=True, path=parent.path + [child_id],
                    nodes_visited=len(all_visited_nodes), search_depth=parent.depth + 1,
                    runtime=elapsed, failure_type=FailureType.NONE, edges_explored=edges_explored_ref[0],
                    visited_node_set=all_visited_nodes, path_relations=parent.path_relations + [edge.relation],
                )
                return [], result

            target_node = graph.get_node(child_id)
            merged_evidence = dict(parent.collected_evidence)
            if target_node and target_node.evidence:
                merged_evidence.update(target_node.evidence)

            parent_cumulative = cumulative_scores[id(parent)]
            new_cumulative = (parent_cumulative * parent.depth + score) / (parent.depth + 1)
            child = _TreeNode(
                graph_node=child_id,
                path_relations=parent.path_relations + [edge.relation],
                visited_nodes=parent.visited_nodes | {child_id},
                visited_edges=parent.visited_edges | {(parent.graph_node, edge.relation, child_id)},
                collected_evidence=merged_evidence,
                depth=parent.depth + 1,
                parent=parent,
            )
            cumulative_scores[id(child)] = new_cumulative
            parent.children[child_id] = child
            next_level.append(child)

            if len(all_visited_nodes) >= node_budget:
                break

        # Seal every node this round touched - beam's pruning decision here
        # (who got children, who didn't) is final; MCTS never reconsiders it.
        for node in current_level:
            node.untried = []

        current_level = next_level

    return current_level, None


def beam_seeded_mcts_search(
    graph: KnowledgeGraph,
    query: Query,
    verifier: Verifier,
    node_budget: int = 100,
    max_depth: int = 6,
    seed_depth: int = 2,
    beam_width: int = 5,
    exploration_constant: float = 1.0,
) -> SearchResult:
    """Beam-seed the tree for the first `seed_depth` hops, then run standard
    MCTS (UCB1 selection + rollout + backprop) with the remaining budget and
    remaining `max_depth - seed_depth` hops, starting from the seeded tips
    instead of a blank root."""
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

    tips, success_result = _beam_seed(
        graph, query, verifier, root, node_budget, seed_depth, beam_width, max_depth,
        all_visited_nodes, edges_explored_ref, t0,
    )
    if success_result is not None:
        return success_result

    def _report_failure(failure_type_if_no_progress: FailureType) -> SearchResult:
        elapsed = time.perf_counter() - t0
        node = root
        while node.children:
            node = max(node.children.values(), key=lambda c: (c.visits, c.value_sum))
        failure_type = FailureType.BUDGET_EXHAUSTED if node.depth > 0 else failure_type_if_no_progress
        return SearchResult(
            query_id=query.id, success=False, path=node.path, nodes_visited=len(all_visited_nodes),
            search_depth=node.depth, runtime=elapsed, failure_type=failure_type,
            edges_explored=edges_explored_ref[0], visited_node_set=all_visited_nodes,
            path_relations=node.path_relations,
        )

    if not tips:
        return _report_failure(FailureType.DEAD_END)

    # --- Main MCTS refinement phase, starting from the seeded tips ---
    def _ucb(child: _TreeNode, parent_visits: int) -> float:
        if child.visits == 0:
            return float("inf")
        exploitation = child.value_sum / child.visits
        exploration = exploration_constant * math.sqrt(math.log(parent_visits) / child.visits)
        return exploitation + exploration

    max_simulations = 20000
    stale_limit = 300
    stale_count = 0
    simulations = 0

    while len(all_visited_nodes) < node_budget and simulations < max_simulations and stale_count < stale_limit:
        simulations += 1
        prev_visited_count = len(all_visited_nodes)

        # --- Selection --- (sealed seeding-phase nodes just have a single
        # child to "select", so this walks straight down to whichever
        # seeded tip is currently most attractive before doing real UCB1).
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
                break
            node = max(node.children.values(), key=lambda c: _ucb(c, node.visits))

        # --- Expansion + rollout (identical to mcts_search.py) ---
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

        if len(all_visited_nodes) == prev_visited_count:
            stale_count += 1
        else:
            stale_count = 0

    return _report_failure(FailureType.BUDGET_EXHAUSTED)
