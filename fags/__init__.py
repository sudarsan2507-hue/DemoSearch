"""Failure-Aware Graph Search (FAGS) — Shared data structures."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ──────────────────────────────────────────────
# Failure taxonomy
# ──────────────────────────────────────────────

class FailureType(Enum):
    """The four failure types defined in the research spec."""
    NONE = "none"
    DEAD_END = "dead_end"
    PATH_MISALIGNMENT = "path_misalignment"
    CONTRADICTION = "contradiction"
    BUDGET_EXHAUSTED = "budget_exhausted"


# ──────────────────────────────────────────────
# Graph primitives
# ──────────────────────────────────────────────

@dataclass
class Node:
    id: str
    label: str
    node_type: str  # "entity" | "answer" | "distractor" | "dead_end"
    domain: str = ""
    properties: dict = field(default_factory=dict)
    # Key-value evidence used for contradiction detection.
    # Example: {"governing_party": "Labor"}
    evidence: dict = field(default_factory=dict)


@dataclass
class Edge:
    source: str
    target: str
    relation: str
    properties: dict = field(default_factory=dict)


class KnowledgeGraph:
    """In-memory directed knowledge graph (adjacency list)."""

    def __init__(self) -> None:
        self.nodes: dict[str, Node] = {}
        self.edges: dict[str, list[Edge]] = {}  # source_id → [Edge, …]

    def add_node(self, node: Node) -> None:
        self.nodes[node.id] = node
        if node.id not in self.edges:
            self.edges[node.id] = []

    def add_edge(self, edge: Edge) -> None:
        if edge.source not in self.edges:
            self.edges[edge.source] = []
        # Avoid exact duplicate edges
        for existing in self.edges[edge.source]:
            if existing.target == edge.target and existing.relation == edge.relation:
                return
        self.edges[edge.source].append(edge)

    def get_neighbors(self, node_id: str) -> list[Edge]:
        return self.edges.get(node_id, [])

    def get_node(self, node_id: str) -> Optional[Node]:
        return self.nodes.get(node_id)

    def node_count(self) -> int:
        return len(self.nodes)

    def edge_count(self) -> int:
        return sum(len(el) for el in self.edges.values())


# ──────────────────────────────────────────────
# Query
# ──────────────────────────────────────────────

@dataclass
class Query:
    id: str
    question: str
    start_node: str
    answer_node: str
    gold_path: list[str]          # ordered node IDs
    gold_relations: list[str]     # relation label per hop
    keywords: list[str]           # keywords the verifier matches against
    difficulty: str = "medium"    # "easy" | "medium" | "hard"
    hop_count: int = 0


# ──────────────────────────────────────────────
# Search result
# ──────────────────────────────────────────────

@dataclass
class SearchResult:
    query_id: str
    success: bool
    path: list[str]
    nodes_visited: int
    search_depth: int
    runtime: float                           # seconds
    failure_type: FailureType = FailureType.NONE
    backtracks: int = 0
    recovery_attempts: int = 0
    recovery_successes: int = 0
    gold_path_pruned: bool = False           # was gold relation rejected at any branch?
    gold_path_recovered: bool = False        # did recovery lead back onto gold path?
    memory_size_at_end: int = 0
    edges_explored: int = 0
    visited_node_set: set = field(default_factory=set)
    hops_survived_post_revival: list[int] = field(default_factory=list)
    trajectory_attempts: int = 0
    trajectory_matches: int = 0
    trajectory_utilities: int = 0
    successful_recovery_margins: list[float] = field(default_factory=list)
    failed_recovery_margins: list[float] = field(default_factory=list)
    path_relations: list[str] = field(default_factory=list)


# ──────────────────────────────────────────────
# Failure memory entry
# ──────────────────────────────────────────────

@dataclass
class MemoryEntry:
    node_id: str            # branch-point node
    relation: str           # the rejected relation
    target_id: str          # where this relation leads
    verifier_score: float
    depth: int
    timestamp: int          # step counter when stored
    path_so_far: list[str] = field(default_factory=list)
    rejection_margin: float = 0.0
