"""MetaQA (Zhang et al., 2018) loader and relation-path search with a real
embedding verifier (BAAI/bge-small-en-v1.5).

Search space: directed relation paths from the question's topic entity. A
state is (relation sequence, entity set reached); expanding applies one of
the 18 directed relations (9 KB relations x forward/inverse). Applying a
relation that no entity in the set has yields the empty set - a dead end,
which every policy here observes immediately and discards. After k hops
(k is given by the split, as MetaQA systems assume) the agent commits to
the entity set it reached.

Two verifiers, both real BGE embeddings, no oracle information:
  * relation-level ("hop-blind"): score(r) = cos(q, text(r)) - the same
    for every hop, exactly like the synthetic Verifier's keyword bag.
  * path-level ("hop-aware"): score(prefix) = cos(q, text(r1 then ... rt)),
    i.e. the question is matched against the whole partial path, so the
    verifier knows where in the plan it is.
"""

from __future__ import annotations

import os
import re
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

REL_TEXT = {
    "directed_by": ("the director of the movie", "the movies directed by this person"),
    "written_by": ("the writer of the movie", "the movies written by this person"),
    "starred_actors": ("the actors starring in the movie", "the movies starring this actor"),
    "release_year": ("the release year of the movie", "the movies released in this year"),
    "in_language": ("the language of the movie", "the movies in this language"),
    "has_tags": ("the tags and topics of the movie", "the movies about this topic"),
    "has_genre": ("the genre of the movie", "the movies of this genre"),
    "has_imdb_votes": ("the imdb votes of the movie", "the movies with these imdb votes"),
    "has_imdb_rating": ("the imdb rating of the movie", "the movies with this imdb rating"),
}
DIRECTED = [(r, inv) for r in REL_TEXT for inv in (False, True)]


def rel_text(dr) -> str:
    r, inv = dr
    return REL_TEXT[r][1 if inv else 0]


def path_text(path) -> str:
    return ", then ".join(rel_text(dr) for dr in path)


class KB:
    def __init__(self, kb_path: str):
        self.adj: dict[tuple, dict[str, set]] = defaultdict(lambda: defaultdict(set))
        with open(kb_path, encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\n").split("|")
                if len(parts) != 3:
                    continue
                h, r, t = parts
                self.adj[(r, False)][h].add(t)
                self.adj[(r, True)][t].add(h)

    def step(self, entities: frozenset, dr) -> frozenset:
        table = self.adj[dr]
        out = set()
        for e in entities:
            out |= table.get(e, frozenset())
        return frozenset(out)


@dataclass
class Question:
    text: str          # topic entity replaced by "it"
    topic: str
    answers: frozenset
    hops: int


def load_questions(path: str, hops: int, limit: int | None, seed: int = 0) -> list[Question]:
    qs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            q, a = line.rstrip("\n").split("\t")
            m = re.search(r"\[(.+?)\]", q)
            qs.append(Question(q.replace(m.group(0), "it"), m.group(1), frozenset(a.split("|")), hops))
    if limit is not None and limit < len(qs):
        idx = np.random.default_rng(seed).choice(len(qs), size=limit, replace=False)
        qs = [qs[i] for i in sorted(idx)]
    return qs


class Embedder:
    """Caches BGE embeddings of every relation-path prefix up to max_hops."""

    def __init__(self, max_hops: int, model_name: str = "BAAI/bge-small-en-v1.5", cache_path: str | None = None):
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        import torch
        torch.set_num_threads(int(os.environ.get("TORCH_THREADS", 2)))
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        paths = [()]
        all_paths = []
        for _ in range(max_hops):
            paths = [p + (dr,) for p in paths for dr in DIRECTED]
            all_paths += paths
        texts = [path_text(p) for p in all_paths]
        if cache_path and os.path.exists(cache_path):
            cached = np.load(cache_path, allow_pickle=False)
            assert list(cached["texts"]) == texts, "stale path-embedding cache"
            embs = cached["embs"]
        else:
            embs = self.model.encode(texts, batch_size=256, normalize_embeddings=True, convert_to_numpy=True)
            if cache_path:
                np.savez(cache_path, texts=np.array(texts), embs=embs)
        self.path_emb = {p: e for p, e in zip(all_paths, embs)}
        rel = self.model.encode([rel_text(dr) for dr in DIRECTED], normalize_embeddings=True, convert_to_numpy=True)
        self.rel_emb = {dr: e for dr, e in zip(DIRECTED, rel)}

    def encode_questions(self, texts: list[str]) -> np.ndarray:
        # BGE's recommended retrieval prefix for short queries.
        return self.model.encode(["Represent this sentence for searching relevant passages: " + t for t in texts],
                                 batch_size=128, normalize_embeddings=True, convert_to_numpy=True)


@dataclass
class Outcome:
    answer: frozenset
    path: tuple
    calls: int
    touched_gold: bool  # legacy goal-test: did any visited set contain a gold answer?


def beam_paths(kb: KB, q: Question, q_emb: np.ndarray, emb: Embedder, width: int, mode: str) -> Outcome:
    """Beam over relation paths to exactly q.hops hops; commit to the best
    (mean prefix score). mode: 'relation' (hop-blind) or 'path' (hop-aware)."""
    start = frozenset([q.topic])
    beam = [((), start, 0.0)]
    calls, touched = 0, False
    for depth in range(q.hops):
        pool = []
        for path, ents, total in beam:
            for dr in DIRECTED:
                nxt = kb.step(ents, dr) - {q.topic}
                calls += 1
                if not nxt:
                    continue  # dead end: relation inapplicable
                if nxt & q.answers:
                    touched = True
                p = path + (dr,)
                s = float(q_emb @ (emb.rel_emb[dr] if mode == "relation" else emb.path_emb[p]))
                pool.append((p, nxt, total + s))
        if not pool:
            return Outcome(frozenset(), beam[0][0], calls, touched)
        pool.sort(key=lambda x: -x[2] / (depth + 1))
        beam = pool[:width]
    best = max(beam, key=lambda h: h[2])
    return Outcome(best[1], best[0], calls, touched)


def all_k_paths(kb: KB, q: Question):
    """Every executable k-hop relation path and its answer set."""
    frontier = [((), frozenset([q.topic]))]
    for _ in range(q.hops):
        frontier = [(p + (dr,), n) for p, ents in frontier for dr in DIRECTED
                    if (n := kb.step(ents, dr) - {q.topic})]
    return frontier


def hits1(pred: frozenset, gold: frozenset) -> float:
    """Expected Hits@1 when one predicted entity is returned uniformly at random."""
    return len(pred & gold) / len(pred) if pred else 0.0


def f1(pred: frozenset, gold: frozenset) -> float:
    if not pred:
        return 0.0
    tp = len(pred & gold)
    return 0.0 if tp == 0 else 2 * tp / (len(pred) + len(gold))
