"""Delayed / diagnostic failure environment and failure-aware search policies.

Why this exists: FAGS's premise is that a failure tells you something about
where you went wrong. The KG benchmark can't test that premise because how
informative its failures are is not controllable. Here it is, along two axes:

  * delay K      - how many plausible-looking steps separate a wrong move
                   from the dead end that reveals it;
  * diagnosis rho - the probability that a dead end also names its cause
                   (the wrong turn that led into it), as a contradiction
                   naming the conflicting evidence or a failing test naming
                   the bad line would.

Environment (a lazily generated tree, deterministic per tree_seed):
  * The goal is the gold node at depth H. Every gold node at depth d < H has
    b children: one gold child and b-1 wrong children.
  * A wrong child roots a decoy subtree with life L ~ U{0..K} (truncated at
    depth H). A decoy with life L > 0 has b children of life L-1, one of
    which is a "lure" scored like a correct move, so the search keeps seeing
    plausible continuations for L steps before a dead end.
  * Verifier scores: correct and lure children ~ N(d', 1), others ~ N(0, 1).
  * Cost = node expansions (check a node + score its children = one step).

Key theoretical point (the reason the policies below are organised the way
they are): with a factorized prior - P(child correct | parent gold) =
softmax(d' * scores), independent across nodes - the exact posterior that an
*unexplored* node lies on the gold path equals its prior marginal no matter
which other nodes have dead-ended. A non-diagnostic failure therefore only
ELIMINATES the dead node; it carries no information that should reorder the
remaining frontier. Every systematic search already eliminates, so "failure
memory" can only add value through diagnosis (rho > 0). Policies that
re-rank after a failure without diagnosis - renormalising the posterior among
the dead node's siblings, i.e. blaming the most recent decision, as FAGS's
local backtracking does - are heuristics, not Bayesian updates.
"""

from __future__ import annotations

import heapq
import math
import random
from dataclasses import dataclass
from typing import Optional


# ── Environment ──────────────────────────────────────────────────────────

@dataclass
class _Info:
    kind: str                       # "gold" | "decoy"
    life: int                       # decoy: remaining plausible steps; gold: -1
    culprit: Optional[tuple] = None  # decoy: the wrong turn this subtree hangs off
    children: Optional[list[tuple[tuple, float]]] = None


class DelayedFailureTree:
    def __init__(self, tree_seed: int, depth: int = 8, branching: int = 3,
                 max_delay: int = 2, d_prime: float = 1.0, diagnosis_rate: float = 0.0,
                 diagnosis_error: float = 0.0):
        self.seed = tree_seed
        self.H, self.b, self.K, self.d = depth, branching, max_delay, d_prime
        self.rho = diagnosis_rate
        self.eps = diagnosis_error
        self._info: dict[tuple, _Info] = {(): _Info("gold", -1)}

    def _rng(self, path: tuple, tag: str = "") -> random.Random:
        return random.Random(f"{self.seed}|{tag}|{path}")

    def is_goal(self, path: tuple) -> bool:
        return len(path) == self.H and self._info[path].kind == "gold"

    def children(self, path: tuple) -> list[tuple[tuple, float]]:
        """Children with verifier scores (cached: every policy sees the same tree)."""
        node = self._info[path]
        if node.children is not None:
            return node.children
        rng = self._rng(path)
        d = len(path)
        kids: list[tuple[tuple, float]] = []
        if d < self.H and (node.kind == "gold" or node.life > 0):
            special = rng.randrange(self.b)  # the gold child, or the decoy's lure
            for i in range(self.b):
                cp = path + (i,)
                if node.kind == "gold":
                    if i == special:
                        self._info[cp] = _Info("gold", -1)
                    else:
                        self._info[cp] = _Info("decoy", min(rng.randint(0, self.K), self.H - (d + 1)), culprit=cp)
                else:
                    self._info[cp] = _Info("decoy", node.life - 1, culprit=node.culprit)
                kids.append((cp, rng.gauss(self.d if i == special else 0.0, 1.0)))
        node.children = kids
        return kids

    def diagnose(self, dead: tuple) -> Optional[tuple]:
        """Called when `dead` is expanded and has no children. With
        probability rho a diagnosis fires; it names the wrong turn that caused
        this failure, except with probability eps (diagnosis_error) it names a
        uniformly random *other* ancestor of the dead node instead - possibly
        one on the gold path, in which case acting on it prunes the solution.
        The error draw uses its own stream, so eps=0 reproduces the
        noiseless environment exactly."""
        info = self._info[dead]
        if info.kind != "decoy" or self.rho <= 0:
            return None
        if self._rng(dead, "diag").random() >= self.rho:
            return None
        if self.eps > 0:
            err = self._rng(dead, "diagerr")
            if err.random() < self.eps:
                others = [dead[:i] for i in range(1, len(dead) + 1) if dead[:i] != info.culprit]
                if others:
                    return err.choice(others)
        return info.culprit

    def ancestor_candidates(self, dead: tuple) -> int:
        """How many wrong nodes a diagnosis error can name (agents may use this)."""
        return max(1, len(dead) - 1)


def _under(path: tuple, pruned: set) -> bool:
    return any(path[:i] in pruned for i in range(1, len(path) + 1))


def _act_on_diagnosis(culprit: tuple, counts: dict, pruned: set, mode: str) -> None:
    """mode 'hard': prune the named subtree on the first diagnosis.
    mode 'vote2': prune only once two separate failures name the same node."""
    counts[culprit] = counts.get(culprit, 0) + 1
    if mode == "hard" or (mode == "vote2" and counts[culprit] >= 2):
        pruned.add(culprit)


# ── Policies. Each returns expansions-to-goal, or None if out of budget. ─

def dfs(tree: DelayedFailureTree, budget: int, use_diagnosis: bool = False,
        diag_mode: str = "hard") -> Optional[int]:
    """Chronological backtracking, children in score order. With diagnosis
    this becomes conflict-directed backjumping: jump straight past the named
    culprit instead of exhausting the subtree under it."""
    stack, pruned, counts, cost = [()], set(), {}, 0
    while stack and cost < budget:
        p = stack.pop()
        if pruned and _under(p, pruned):
            continue
        cost += 1
        if tree.is_goal(p):
            return cost
        kids = tree.children(p)
        if kids:
            stack.extend(cp for cp, _ in sorted(kids, key=lambda x: x[1]))
        elif use_diagnosis and (c := tree.diagnose(p)) is not None:
            _act_on_diagnosis(c, counts, pruned, diag_mode)
    return None


def fags_style(tree: DelayedFailureTree, budget: int, use_diagnosis: bool = False,
               diag_mode: str = "hard") -> Optional[int]:
    """FAGS's recovery policy: greedy descent; on a dead end, first the failed
    node's next-best untried sibling (local backtracking), otherwise revive
    the highest raw-score rejected alternative anywhere (failure memory)."""
    memory: list[tuple[float, int, tuple]] = []
    untried: dict[tuple, list[tuple]] = {}
    expanded, pruned, counts = set(), set(), {}
    counter, cost = 0, 0
    current: Optional[tuple] = ()
    while cost < budget and current is not None:
        cost += 1
        expanded.add(current)
        if tree.is_goal(current):
            return cost
        kids = sorted(tree.children(current), key=lambda x: -x[1])
        if kids:
            untried[current] = [cp for cp, _ in kids[1:]]
            for cp, s in kids[1:]:
                counter += 1
                heapq.heappush(memory, (-s, counter, cp))
            current = kids[0][0]
            continue
        dead, current = current, None
        if use_diagnosis and (c := tree.diagnose(dead)) is not None:
            _act_on_diagnosis(c, counts, pruned, diag_mode)
        ok = lambda cp: cp not in expanded and not (pruned and _under(cp, pruned))
        current = next((cp for cp in untried.get(dead[:-1], []) if ok(cp)), None)
        while current is None and memory:
            _, _, cp = heapq.heappop(memory)
            if ok(cp):
                current = cp
    return None


def best_first_mean(tree: DelayedFailureTree, budget: int, use_diagnosis: bool = False,
                    diag_mode: str = "hard") -> Optional[int]:
    """Global best-first on mean raw path score (the repo's best_first_search rule)."""
    frontier, pruned, counts, counter, cost = [(0.0, 0, (), 0.0)], set(), {}, 0, 0
    while frontier and cost < budget:
        _, _, p, total = heapq.heappop(frontier)
        if pruned and _under(p, pruned):
            continue
        cost += 1
        if tree.is_goal(p):
            return cost
        kids = tree.children(p)
        for cp, s in kids:
            counter += 1
            heapq.heappush(frontier, (-((total + s) / len(cp)), counter, cp, total + s))
        if not kids and use_diagnosis and (c := tree.diagnose(p)) is not None:
            _act_on_diagnosis(c, counts, pruned, diag_mode)
    return None


def random_restarts(tree: DelayedFailureTree, budget: int, d_hat: float, seed: int) -> Optional[int]:
    """Memoryless control: sample a descent with P(child) = softmax(d_hat *
    score), restart from the root on every dead end, paying for repeats."""
    rng, cost = random.Random(seed), 0
    while cost < budget:
        p: tuple = ()
        while cost < budget:
            cost += 1
            if tree.is_goal(p):
                return cost
            kids = tree.children(p)
            if not kids:
                break
            m = max(s for _, s in kids)
            p = rng.choices([cp for cp, _ in kids], weights=[math.exp(d_hat * (s - m)) for _, s in kids])[0]
    return None


def beam(tree: DelayedFailureTree, width: int, d_hat: float) -> tuple[bool, int]:
    """Level-synchronous beam on summed log-softmax; no backtracking.
    Returns (success, expansions)."""
    level, cost = [(0.0, ())], 0
    while level:
        pool = []
        for lp, p in level:
            cost += 1
            if tree.is_goal(p):
                return True, cost
            kids = tree.children(p)
            if kids:
                m = max(s for _, s in kids)
                logz = math.log(sum(math.exp(d_hat * (s - m)) for _, s in kids)) + d_hat * m
                pool += [(lp + d_hat * s - logz, cp) for cp, s in kids]
        pool.sort(key=lambda x: -x[0])
        level = pool[:width]
    return False, cost


class _PNode:
    __slots__ = ("path", "parent", "logw", "kids", "z", "alive", "expanded")

    def __init__(self, path, parent, logw):
        self.path, self.parent, self.logw = path, parent, logw
        self.kids: list[_PNode] = []
        self.z = 0.0
        self.alive = True
        self.expanded = False


def posterior_search(tree: DelayedFailureTree, budget: int, d_hat: float,
                     use_diagnosis: bool = False, local_blame: bool = False,
                     diag_error_hat: Optional[float] = None) -> Optional[int]:
    """Expand the unexpanded node with the highest posterior probability of
    lying on the gold path under the factorized prior (Bayes-exact when
    local_blame=False: dead nodes are eliminated, nothing is re-ranked).

    local_blame=True renormalises a dead node's siblings so the parent keeps
    its full mass - i.e. assumes the most recent decision was the mistake.
    Included as the principled-looking but non-Bayesian variant FAGS's local
    backtracking corresponds to.

    Diagnosis handling: with diag_error_hat=None the named culprit is killed
    outright (trusts every diagnosis). With diag_error_hat = e (the agent's
    belief about the diagnosis error rate) it is a soft Bayesian update
    instead: a node is named either as the true culprit (prob 1-e) or by
    error (prob e/m, m = the number of other ancestors the error could have
    picked). A node on the gold path can only be named by error, so the
    named node's subtree mass is multiplied by the likelihood ratio
        LR = (e/m) / ((1-e) + e/m)
    rather than set to 0. e=0 recovers the hard kill. Repeated diagnoses of
    the same node multiply."""
    root = _PNode((), None, 0.0)
    nodes = {(): root}
    cost = 0

    def kill(n: _PNode):
        while n is not None:
            n.alive = False
            par = n.parent
            if par is None:
                return
            if local_blame:
                par.z = sum(math.exp(k.logw) for k in par.kids if k.alive)
            if any(k.alive for k in par.kids):
                return
            n = par

    while cost < budget and root.alive:
        heap, counter, target = [(0.0, 0, root)], 0, None
        while heap:
            neg, _, n = heapq.heappop(heap)
            if not n.expanded:
                target = n
                break
            logz = math.log(n.z)
            for k in n.kids:
                if k.alive:
                    counter += 1
                    heapq.heappush(heap, (neg - (k.logw - logz), counter, k))
        if target is None:
            return None
        cost += 1
        target.expanded = True
        if tree.is_goal(target.path):
            return cost
        kids = tree.children(target.path)
        if kids:
            m = max(s for _, s in kids)
            for cp, s in kids:
                k = _PNode(cp, target, d_hat * (s - m))
                target.kids.append(k)
                nodes[cp] = k
                target.z += math.exp(k.logw)
            continue
        kill(target)
        if use_diagnosis and (c := tree.diagnose(target.path)) is not None and c in nodes and nodes[c].alive:
            e = diag_error_hat
            if not e:
                kill(nodes[c])
            else:
                m = tree.ancestor_candidates(target.path)
                lr = (e / m) / ((1 - e) + e / m)
                node = nodes[c]
                node.logw += math.log(lr)
                if local_blame:
                    par = node.parent
                    par.z = sum(math.exp(k.logw) for k in par.kids if k.alive)
    return None
