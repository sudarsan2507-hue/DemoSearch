"""Commit-to-an-answer evaluation protocol.

The legacy protocol counts a search as successful the moment it *touches*
query.answer_node, i.e. the search is handed a perfect goal test and never
has to decide what its answer is. Here the agent is told the hop count H
(as in MetaQA's k-hop splits) and must return one endpoint; it is scored by
whether that endpoint is in the query's answer set A(q) = every node
reachable from the start by following the gold relation sequence (Hits@1
against the answer set, the standard KGQA metric). The single labelled
answer_node is only one member of A(q), so labelled-answer accuracy is
capped at mean 1/|A(q)| for any agent that only sees the question.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional, Sequence

from fags import Edge, KnowledgeGraph, Query
from fags.verifier import RELATION_KEYWORDS, Verifier


def answer_set(graph: KnowledgeGraph, query: Query) -> set[str]:
    """Endpoints of simple paths from start_node following gold_relations exactly."""
    frontier = [(query.start_node, (query.start_node,))]
    for rel in query.gold_relations:
        frontier = [
            (e.target, path + (e.target,))
            for node, path in frontier
            for e in graph.get_neighbors(node)
            if e.relation == rel and e.target not in path
        ]
    return {n for n, _ in frontier}


@dataclass
class CommitResult:
    answer: Optional[str]
    path: list[str]
    path_relations: list[str]
    verifier_calls: int


def _contradicts(evidence: dict, target_evidence: dict) -> bool:
    return any(k in evidence and evidence[k] != v for k, v in (target_evidence or {}).items())


def beam_commit(
    graph: KnowledgeGraph,
    query: Query,
    verifier,
    beam_width: int,
    hops: int,
) -> CommitResult:
    """Beam search to exactly `hops` hops, then commit to the hypothesis with
    the best mean per-hop score. beam_width=1 is greedy. No goal test."""
    start_ev = dict(graph.get_node(query.start_node).evidence or {})
    beam = [(0.0, [query.start_node], [], start_ev)]
    calls = 0
    for depth in range(hops):
        pool = []
        for score, path, rels, ev in beam:
            for e in graph.get_neighbors(path[-1]):
                if e.target in path:
                    continue
                tnode = graph.get_node(e.target)
                if _contradicts(ev, tnode.evidence if tnode else {}):
                    continue
                s = verifier.score(query, e, rels)
                calls += 1
                pool.append(((score * depth + s) / (depth + 1), path, rels, ev, e))
        if not pool:
            return CommitResult(None, beam[0][1], beam[0][2], calls)
        pool.sort(key=lambda x: x[0], reverse=True)
        beam = []
        for new_score, path, rels, ev, e in pool[:beam_width]:
            tnode = graph.get_node(e.target)
            new_ev = dict(ev)
            new_ev.update(tnode.evidence or {})
            beam.append((new_score, path + [e.target], rels + [e.relation], new_ev))
    best = max(beam, key=lambda h: h[0])
    return CommitResult(best[1][-1], best[1], best[2], calls)


# ── Verifier ladder for the error decomposition ─────────────────────────

class OraclePlanVerifier:
    """Knows the exact gold relation for the current hop. Scores 1 for it,
    0 otherwise, with a tiny random tie-break. Removes all verifier error,
    leaving only search/graph error."""

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def score(self, query: Query, edge: Edge, path_relations: Sequence[str] | None = None) -> float:
        t = len(path_relations or [])
        want = query.gold_relations[t] if t < len(query.gold_relations) else None
        return (1.0 if edge.relation == want else 0.0) + self.rng.uniform(0, 1e-3)


class _KW:
    def __init__(self, keywords):
        self.keywords = keywords


class HopAwareVerifier(Verifier):
    """The rule-based Verifier, but at hop t it matches only the question
    keywords belonging to hop t's relation (an oracle question decomposition;
    keyword dropout and score noise still apply). The gap between this and
    the ordinary Verifier is the cost of scoring every hop against the whole
    question's keyword bag."""

    def score(self, query, edge, path_relations=None):
        t = len(path_relations or [])
        kws = query.keywords
        if t < len(query.gold_relations):
            hop_kws = set(RELATION_KEYWORDS.get(query.gold_relations[t], []))
            kws = [k for k in query.keywords if k in hop_kws] or query.keywords
        return super().score(_KW(kws), edge, path_relations)
