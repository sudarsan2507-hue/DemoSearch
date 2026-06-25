"""Failure Pattern Graph (FPG) — learns which relation transitions tend to be
the proximate cause of search failure, across many training queries, and lets
that learned prior steer future searches *away* from those transitions before
they're ever attempted (avoidance), rather than only recovering after the
fact (FAGS's reactive memory-revival).

Training signal: run plain greedy baseline_search over a set of training
queries. For each finished path, the LAST relation transition taken before a
DEAD_END / CONTRADICTION / PATH_MISALIGNMENT failure is blamed as the
"mistake" transition; every other transition (including all transitions on
successful paths) is treated as a neutral/good observation. Transitions are
keyed by (previous_relation_or_START, relation) — a bigram over the relation
vocabulary, which is shared across every generated graph, so a graph trained
on one seed transfers to graphs generated with a different seed.
"""

from __future__ import annotations

from typing import Any, Sequence

from fags import Edge, FailureType, KnowledgeGraph, Query
from fags.baseline_search import baseline_search

START = "START"

# Failure types whose final hop is a genuine "mistake" worth blaming.
# BUDGET_EXHAUSTED just means the search ran out of depth mid-path - the
# transitions on it aren't obviously bad, so they're left unattributed.
_BLAMABLE_FAILURES = {
    FailureType.DEAD_END,
    FailureType.CONTRADICTION,
    FailureType.PATH_MISALIGNMENT,
}


class FailurePatternGraph:
    """Beta-smoothed failure rate per (prev_relation, relation) transition."""

    def __init__(self, prior_alpha: float = 1.0, prior_beta: float = 1.0) -> None:
        self.attempts: dict[tuple[str, str], int] = {}
        self.failures: dict[tuple[str, str], int] = {}
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta

    def observe(self, prev_rel: str, rel: str, was_failure_edge: bool) -> None:
        key = (prev_rel, rel)
        self.attempts[key] = self.attempts.get(key, 0) + 1
        if was_failure_edge:
            self.failures[key] = self.failures.get(key, 0) + 1

    def failure_rate(self, prev_rel: str, rel: str) -> float:
        key = (prev_rel, rel)
        f = self.failures.get(key, 0)
        n = self.attempts.get(key, 0)
        return (f + self.prior_alpha) / (n + self.prior_alpha + self.prior_beta)

    def top_failure_patterns(
        self, min_attempts: int = 5, n: int = 15
    ) -> list[tuple[tuple[str, str], float, int]]:
        """Highest-failure-rate transitions with at least min_attempts observations."""
        rows = [
            (key, self.failure_rate(*key), count)
            for key, count in self.attempts.items()
            if count >= min_attempts
        ]
        rows.sort(key=lambda r: r[1], reverse=True)
        return rows[:n]


def _attribute_outcome(path_relations: Sequence[str], failure_type: FailureType, fpg: FailurePatternGraph) -> None:
    """Record one finished path's transitions into the FPG."""
    if not path_relations:
        return

    is_blamable_failure = failure_type in _BLAMABLE_FAILURES
    for i, rel in enumerate(path_relations):
        prev_rel = path_relations[i - 1] if i > 0 else START
        is_last = i == len(path_relations) - 1
        was_failure_edge = is_blamable_failure and is_last
        fpg.observe(prev_rel, rel, was_failure_edge)


def train_failure_pattern_graph(
    graph: KnowledgeGraph,
    queries: list[Query],
    verifier: Any,
    max_depth: int = 6,
    prior_alpha: float = 1.0,
    prior_beta: float = 1.0,
) -> FailurePatternGraph:
    """Run plain greedy baseline_search over `queries` and learn an FPG from
    which transitions precede failure."""
    fpg = FailurePatternGraph(prior_alpha=prior_alpha, prior_beta=prior_beta)
    for q in queries:
        res = baseline_search(graph, q, verifier, max_depth=max_depth)
        _attribute_outcome(res.path_relations, res.failure_type, fpg)
    return fpg


class PatternAwareVerifier:
    """Wraps any verifier, subtracting a learned failure-pattern penalty from
    its score so candidates matching a known failure pattern are downranked
    *before* being attempted - independent of and composable with whatever
    search loop (baseline_search or failure_search) consumes it."""

    def __init__(self, base_verifier: Any, fpg: FailurePatternGraph, penalty_weight: float = 0.3) -> None:
        self.base_verifier = base_verifier
        self.fpg = fpg
        self.penalty_weight = penalty_weight

    def _penalty(self, edge: Edge, path_relations: Sequence[str] | None) -> float:
        prev_rel = path_relations[-1] if path_relations else START
        return self.penalty_weight * self.fpg.failure_rate(prev_rel, edge.relation)

    def score(self, query: Any, edge: Edge, path_relations: Sequence[str] | None = None) -> float:
        base = self.base_verifier.score(query, edge, path_relations)
        return round(max(0.0, min(1.0, base - self._penalty(edge, path_relations))), 4)

    def re_score(
        self,
        query: Any,
        edge: Edge,
        path_relations: Sequence[str] | None = None,
        failed_relations: Sequence[str] | None = None,
    ) -> float:
        base = self.base_verifier.re_score(query, edge, path_relations, failed_relations)
        return round(max(0.0, min(1.0, base - self._penalty(edge, path_relations))), 4)
