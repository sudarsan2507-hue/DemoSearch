"""Failure memory — stores rejected-but-promising paths for later revival.

Three strategies are implemented:
  • Top1Memory   — store only the single highest-scoring reject per branch
  • Top2Memory   — store the two highest-scoring rejects per branch
  • ThresholdMemory — store rejects scoring ≥ winner_score − threshold

All strategies share a common interface and use a max-heap (by verifier
score, ties broken by shallowest depth) for retrieval.
"""

from __future__ import annotations

import heapq
from abc import ABC, abstractmethod
from typing import Optional, Sequence

from fags import Edge, MemoryEntry


# ══════════════════════════════════════════════
# Abstract base
# ══════════════════════════════════════════════

class FailureMemory(ABC):
    """Abstract failure memory with store / pop / inspect interface."""

    def __init__(self) -> None:
        # Max-heap via negated scores.
        # Heap items: (-score, depth, timestamp, MemoryEntry)
        self._heap: list[tuple[float, int, int, MemoryEntry]] = []
        self._counter = 0  # monotonic timestamp

    @abstractmethod
    def store(
        self,
        current_node: str,
        candidates: Sequence[tuple[float, Edge]],
        winner_score: float,
        depth: int,
        path_so_far: list[str],
    ) -> None:
        """Store rejected candidates according to strategy rules.

        Parameters
        ----------
        current_node : branch-point node ID
        candidates   : (score, Edge) pairs for *rejected* candidates only
                        (winner already removed)
        winner_score : score of the chosen (winner) relation
        depth        : current search depth
        path_so_far  : node IDs from start to current_node (inclusive)
        """

    def pop_best(self) -> Optional[MemoryEntry]:
        """Remove and return the highest-priority stored entry, or None."""
        while self._heap:
            _, _, _, entry = heapq.heappop(self._heap)
            return entry
        return None

    def peek_best(self) -> Optional[MemoryEntry]:
        """Return highest-priority entry without removing it."""
        if self._heap:
            return self._heap[0][3]
        return None

    def is_empty(self) -> bool:
        return len(self._heap) == 0

    @property
    def size(self) -> int:
        return len(self._heap)

    def clear(self) -> None:
        self._heap.clear()
        self._counter = 0

    # ── helper ────────────────────────────────

    def _push(self, entry: MemoryEntry) -> None:
        """Push an entry onto the internal heap."""
        # Negate score for max-heap; prefer shallower depth on tie.
        heapq.heappush(
            self._heap,
            (-entry.verifier_score, entry.depth, entry.timestamp, entry),
        )


# ══════════════════════════════════════════════
# Top-1 Memory
# ══════════════════════════════════════════════

class Top1Memory(FailureMemory):
    """Store only the single highest-scoring rejected candidate per branch."""

    def store(
        self,
        current_node: str,
        candidates: Sequence[tuple[float, Edge]],
        winner_score: float,
        depth: int,
        path_so_far: list[str],
    ) -> None:
        if not candidates:
            return
        # Pick the best rejected candidate
        best_score, best_edge = max(candidates, key=lambda x: x[0])
        self._counter += 1
        margin = max(0.0, winner_score - best_score)
        self._push(MemoryEntry(
            node_id=current_node,
            relation=best_edge.relation,
            target_id=best_edge.target,
            verifier_score=best_score,
            depth=depth,
            timestamp=self._counter,
            path_so_far=list(path_so_far),
            rejection_margin=margin,
        ))


# ══════════════════════════════════════════════
# Top-2 Memory
# ══════════════════════════════════════════════

class Top2Memory(FailureMemory):
    """Store the two highest-scoring rejected candidates per branch."""

    def store(
        self,
        current_node: str,
        candidates: Sequence[tuple[float, Edge]],
        winner_score: float,
        depth: int,
        path_so_far: list[str],
    ) -> None:
        if not candidates:
            return
        sorted_cands = sorted(candidates, key=lambda x: x[0], reverse=True)
        for score, edge in sorted_cands[:2]:
            self._counter += 1
            margin = max(0.0, winner_score - score)
            self._push(MemoryEntry(
                node_id=current_node,
                relation=edge.relation,
                target_id=edge.target,
                verifier_score=score,
                depth=depth,
                timestamp=self._counter,
                path_so_far=list(path_so_far),
                rejection_margin=margin,
            ))


# ══════════════════════════════════════════════
# Threshold Memory
# ══════════════════════════════════════════════

class ThresholdMemory(FailureMemory):
    """Store rejected candidates scoring ≥ winner_score − threshold."""

    def __init__(self, threshold: float = 0.15) -> None:
        super().__init__()
        self.threshold = threshold

    def store(
        self,
        current_node: str,
        candidates: Sequence[tuple[float, Edge]],
        winner_score: float,
        depth: int,
        path_so_far: list[str],
    ) -> None:
        if not candidates:
            return
        floor = winner_score - self.threshold
        for score, edge in candidates:
            if score >= floor:
                self._counter += 1
                margin = max(0.0, winner_score - score)
                self._push(MemoryEntry(
                    node_id=current_node,
                    relation=edge.relation,
                    target_id=edge.target,
                    verifier_score=score,
                    depth=depth,
                    timestamp=self._counter,
                    path_so_far=list(path_so_far),
                    rejection_margin=margin,
                ))


# ══════════════════════════════════════════════
# Factory helper
# ══════════════════════════════════════════════

def create_memory(strategy: str, **kwargs) -> FailureMemory:
    """Create a FailureMemory instance by strategy name.

    Parameters
    ----------
    strategy : "top1" | "top2" | "threshold"
    **kwargs : forwarded to constructor (e.g. threshold=0.15)
    """
    strategy = strategy.lower()
    if strategy == "top1":
        return Top1Memory()
    elif strategy == "top2":
        return Top2Memory()
    elif strategy == "threshold":
        return ThresholdMemory(**kwargs)
    else:
        raise ValueError(f"Unknown memory strategy: {strategy!r}")
