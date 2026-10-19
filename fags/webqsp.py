"""WebQSP (RoG release: each test question ships with its own Freebase
subgraph) and relation-path search with BGE verifiers - the second KGQA
dataset for PROJECT_STATUS.md SS32, mirroring fags/metaqa.py.

Differences from MetaQA: relations are thousands of Freebase relation ids
(turned into text by splitting on '.' and '_'), each question has its own
graph, and the hop count is NOT given - the agent searches paths of up to 2
hops and itself decides whether to commit to a 1-hop or a 2-hop path by
score (mean per-hop score for the relation-level verifier, path score for
the path-level one).
"""

from __future__ import annotations

import glob
import os
import re
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

MAX_HOPS = 2


def rel_words(rel: str) -> str:
    parts = rel.split(".")
    text = " ".join(parts[-2:]) if len(parts) >= 2 else rel
    return re.sub(r"[_]+", " ", text).strip()


def drel_text(dr) -> str:
    rel, inv = dr
    return f"{rel_words(rel)} (reverse)" if inv else rel_words(rel)


def path_text(path) -> str:
    return ", then ".join(drel_text(dr) for dr in path)


@dataclass
class WQuestion:
    qid: str
    text: str
    topics: frozenset
    answers: frozenset
    adj: dict = field(repr=False)  # (rel, inv) -> {entity: set(entities)}

    def out_relations(self, ents) -> set:
        return {dr for dr, table in self.adj.items() if any(e in table for e in ents)}

    def step(self, ents, dr) -> frozenset:
        table = self.adj.get(dr, {})
        out = set()
        for e in ents:
            out |= table.get(e, set())
        return frozenset(out)


def load_questions(data_dir: str, limit: int | None, seed: int = 0) -> list[WQuestion]:
    import pandas as pd
    df = pd.concat(pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(data_dir, "test-*.parquet"))))
    idx = np.arange(len(df))
    if limit is not None and limit < len(df):
        idx = np.sort(np.random.default_rng(seed).choice(len(df), size=limit, replace=False))
    out = []
    for i in idx:
        r = df.iloc[int(i)]
        adj: dict = defaultdict(lambda: defaultdict(set))
        nodes = set()
        for h, rel, t in r.graph:
            adj[(rel, False)][h].add(t)
            adj[(rel, True)][t].add(h)
            nodes.update((h, t))
        topics = frozenset(e for e in r.q_entity if e in nodes)
        answers = frozenset(r.a_entity) if len(r.a_entity) else frozenset(r.answer)
        if topics and answers:
            out.append(WQuestion(r.id, r.question, topics, answers, {k: dict(v) for k, v in adj.items()}))
    return out


class TextEmbedder:
    """BGE embeddings with an in-memory cache keyed by text."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        import torch
        torch.set_num_threads(int(os.environ.get("TORCH_THREADS", 2)))
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self.cache: dict[str, np.ndarray] = {}

    def embed(self, texts: list[str]) -> list[np.ndarray]:
        missing = list({t for t in texts if t not in self.cache})
        if missing:
            for t, e in zip(missing, self.model.encode(missing, batch_size=256, normalize_embeddings=True, convert_to_numpy=True)):
                self.cache[t] = e
        return [self.cache[t] for t in texts]

    def query(self, text: str) -> np.ndarray:
        return self.model.encode("Represent this sentence for searching relevant passages: " + text,
                                 normalize_embeddings=True, convert_to_numpy=True)


@dataclass
class WOutcome:
    answer: frozenset
    path: tuple
    calls: int
    touched_gold: bool


def beam_paths(q: WQuestion, q_emb, emb: TextEmbedder, width: int, mode: str) -> WOutcome:
    """Beam over relation paths of up to MAX_HOPS hops; commit to the best
    complete hypothesis of any length (1 or 2 hops). mode: 'relation'
    (hop-blind: each hop scored alone, ranked by mean) or 'path' (the whole
    partial path is scored against the question)."""
    beam = [((), q.topics, 0.0)]
    finished = []
    calls, touched = 0, False
    for depth in range(MAX_HOPS):
        cands = []
        for path, ents, total in beam:
            for dr in q.out_relations(ents):
                nxt = q.step(ents, dr) - q.topics
                calls += 1
                if not nxt:
                    continue
                touched |= bool(nxt & q.answers)
                cands.append((path + (dr,), nxt, total))
        if not cands:
            break
        texts = [drel_text(c[0][-1]) if mode == "relation" else path_text(c[0]) for c in cands]
        sims = [float(q_emb @ e) for e in emb.embed(texts)]
        scored = []
        for (path, nxt, total), s in zip(cands, sims):
            new_total = total + s if mode == "relation" else s
            rank = new_total / len(path) if mode == "relation" else new_total
            scored.append((rank, path, nxt, new_total))
        scored.sort(key=lambda x: -x[0])
        finished += [(rank, path, nxt) for rank, path, nxt, _ in scored[:width]]
        beam = [(path, nxt, tot) for _, path, nxt, tot in scored[:width]]
    if not finished:
        return WOutcome(frozenset(), (), calls, touched)
    best = max(finished, key=lambda x: x[0])
    return WOutcome(best[2], best[1], calls, touched)


def all_paths(q: WQuestion, cap: int = 4000):
    """Every executable path of 1..MAX_HOPS hops with its answer set (capped)."""
    out, frontier = [], [((), q.topics)]
    for _ in range(MAX_HOPS):
        nxt_frontier = []
        for path, ents in frontier:
            for dr in q.out_relations(ents):
                n = q.step(ents, dr) - q.topics
                if n:
                    out.append((path + (dr,), n))
                    nxt_frontier.append((path + (dr,), n))
                    if len(out) >= cap:
                        return out
        frontier = nxt_frontier
    return out
