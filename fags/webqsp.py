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
