"""Sudoku as a real search environment with natural, diagnostic failures.

Implements the same tree interface as fags/delayed_failure.DelayedFailureTree
(children / is_goal / diagnose / ancestor_candidates / seed), so every policy
in that module runs on it unchanged. Nothing about failures is injected:

  * Search: one node per partial assignment. The variable to branch on is
    chosen by MRV (fewest remaining values; ties -> lowest cell index), so the
    tree is fixed and every policy searches the same one. Forward checking
    removes an assigned value from its 20 peers.
  * Verifier: least-constraining-value. A child's score is minus the number
    of peer domains its value removes something from - a real, imperfect
    heuristic, not a simulated one.
  * Failure: a child whose assignment empties some unassigned cell's domain
    is a dead end, discovered when it is expanded. Mistakes typically surface
    several assignments after the move that caused them, so the failure
    delay arises naturally and is measured (jump_distances).
  * Diagnosis: the conflict set. A wiped-out cell X blames the assignments
    that removed its values. When every child of a node assigning cell Y is
    dead, Y's conflict set is the union of the children's conflict sets
    (minus Y's own assignment) plus the assignments that had already removed
    values from Y's domain; the deepest assignment in it is the culprit, and
    every node below it is dead too (FC-CBJ, Prosser 1993). diagnose()
    returns that culprit when it lies strictly above the exhausted node, i.e.
    exactly when the failure points somewhere chronological backtracking
    would not go. The diagnosis is sound: pruning on it never loses a
    solution (checked in tests).
"""

from __future__ import annotations

from typing import Optional

ROWS = [[r * 9 + c for c in range(9)] for r in range(9)]
COLS = [[r * 9 + c for r in range(9)] for c in range(9)]
BOXES = [[(br * 3 + r) * 9 + bc * 3 + c for r in range(3) for c in range(3)] for br in range(3) for bc in range(3)]
PEERS: list[tuple[int, ...]] = []
for _i in range(81):
    _p = set(ROWS[_i // 9]) | set(COLS[_i % 9]) | set(BOXES[(_i // 27) * 3 + (_i % 9) // 3])
    _p.discard(_i)
    PEERS.append(tuple(sorted(_p)))
FULL = 0x1FF  # bits 0..8 = digits 1..9


def _bits(mask: int):
    v = 0
    while mask:
        if mask & 1:
            yield v
        mask >>= 1
        v += 1


class _Node:
    __slots__ = ("assign", "dom", "assigned", "wiped", "kids", "var")

    def __init__(self, assign, dom, assigned, wiped):
        self.assign = assign      # tuple of (cell, digit_bit) in assignment order (search assignments only)
        self.dom = dom            # tuple of 81 bitmasks
        self.assigned = assigned  # int bitmask of assigned cells (givens + search)
        self.wiped = wiped        # cell whose domain this assignment emptied, or None
        self.kids = None
        self.var = None


def parse(puzzle: str) -> list[int]:
    return [0 if ch in ".0" else int(ch) for ch in puzzle.strip()[:81]]


class SudokuTree:
    """One puzzle. `cache` may be shared between policy runs on the same
    puzzle (node generation is deterministic); diagnosis bookkeeping is per
    instance, so create one SudokuTree per policy run."""

    def __init__(self, puzzle: str, seed: int = 0, cache: Optional[dict] = None, ordering: str = "mrv",
                 diagnosis_error: float = 0.0):
        """ordering: 'mrv' (fewest remaining values first - failures surface
        quickly) or 'static' (first unassigned cell in row-major order -
        mistakes typically surface many assignments later). This is the
        environment's natural failure-delay knob. A shared cache must only
        be reused with the same ordering."""
        self.seed = seed
        self.ordering = ordering
        # diagnosis_error: probability that a returned culprit is replaced by a
        # uniformly random *other* ancestor of the failure (possibly one on the
        # solution path). The environment's own conflict-set bookkeeping stays
        # exact; only what the search is told is corrupted.
        self.eps = diagnosis_error
        self.grid = parse(puzzle)
        self._cache = cache if cache is not None else {}
        if () not in self._cache:
            dom = [FULL] * 81
            assigned = 0
            for i, d in enumerate(self.grid):
                if d:
                    dom[i] = 1 << (d - 1)
                    assigned |= 1 << i
            for i, d in enumerate(self.grid):
                if d:
                    for p in PEERS[i]:
                        if not assigned >> p & 1:
                            dom[p] &= ~(1 << (d - 1))
            self._cache[()] = _Node((), tuple(dom), assigned, None)
            self._cache["base"] = tuple(dom)  # domains after the givens: removals here are nobody's fault
        self._base = self._cache["base"]
        # per-run diagnosis state
        self._dead: dict[tuple, frozenset] = {}
        self.jump_distances: list[int] = []

    # ── tree interface ───────────────────────────────────────────────────

    def node(self, path: tuple) -> _Node:
        return self._cache[path]

    def is_goal(self, path: tuple) -> bool:
        n = self._cache[path]
        return n.wiped is None and n.assigned == (1 << 81) - 1

    def children(self, path: tuple) -> list[tuple[tuple, float]]:
        n = self._cache[path]
        if n.kids is not None:
            return n.kids
        kids: list[tuple[tuple, float]] = []
        if n.wiped is None and n.assigned != (1 << 81) - 1:
            best, best_cnt = -1, 10
            for i in range(81):
                if not n.assigned >> i & 1:
                    if self.ordering == "static":
                        best = i
                        break
                    cnt = bin(n.dom[i]).count("1")  # MRV
                    if cnt < best_cnt:
                        best, best_cnt = i, cnt
            n.var = best
            for k, v in enumerate(_bits(n.dom[best])):
                bit = 1 << v
                dom = list(n.dom)
                dom[best] = bit
                assigned = n.assigned | (1 << best)
                touched, wiped = 0, None
                for p in PEERS[best]:
                    if not assigned >> p & 1 and dom[p] & bit:
                        dom[p] &= ~bit
                        touched += 1
                        if dom[p] == 0 and wiped is None:
                            wiped = p
                cp = path + (k,)
                self._cache[cp] = _Node(n.assign + ((best, bit),), tuple(dom), assigned, wiped)
                kids.append((cp, -float(touched)))
        n.kids = kids
        return kids

    def ancestor_candidates(self, dead: tuple) -> int:
        return max(1, len(dead) - 1)

    # ── diagnosis ─────────────────────────────────────────────────────────

    def _removers(self, assign, cell: int, missing: int) -> set[int]:
        """Depths of the first assignments that removed each bit in `missing`
        from `cell`'s domain (bits already absent after the givens excluded)."""
        need = missing & self._base[cell]
        peers = set(PEERS[cell])
        out = set()
        for depth, (c, bit) in enumerate(assign):
            if need == 0:
                break
            if c in peers and need & bit:
                out.add(depth)
                need &= ~bit
        return out

    def diagnose(self, dead: tuple) -> Optional[tuple]:
        n = self._cache[dead]
        if n.wiped is not None:
            cs = self._removers(n.assign, n.wiped, self._base[n.wiped])
        else:
            # Expanded, no children, not wiped: an unassigned cell with an
            # empty domain can't occur under FC (it would have been wiped), so
            # this is the goal or an exhausted node; nothing to diagnose.
            return None
        self._dead[dead] = frozenset(cs)
        node, culprit = dead, None
        while node:
            parent = node[:-1]
            pn = self._cache[parent]
            if pn.kids is None or any(cp not in self._dead for cp, _ in pn.kids):
                break
            own_depth = len(parent)  # depth of the children's assignment
            conf = set()
            for cp, _ in pn.kids:
                conf |= self._dead[cp]
            conf.discard(own_depth)
            conf |= self._removers(pn.assign, pn.var, self._base[pn.var] & ~pn.dom[pn.var])
            j = max(conf) if conf else -1
            target = parent[: j + 1]  # node created by the assignment at depth j
            self._dead[parent] = frozenset(conf)
            if len(target) < len(parent):
                self._dead[target] = frozenset(conf)
                culprit = target
                self.jump_distances.append(len(parent) - len(target))
                node = target
            else:
                node = parent
        return culprit

    def solution(self, path: tuple) -> list[int]:
        n = self._cache[path]
        return [next(_bits(d)) + 1 for d in n.dom]
