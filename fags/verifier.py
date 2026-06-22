from __future__ import annotations
from typing import Sequence, Any
import os
os.environ["HF_HUB_OFFLINE"] = "1"
"""Rule-based verifier for scoring candidate relations against a question.

Scoring = weighted sum of:
  1. Keyword overlap   (weight 0.50) — question keywords ∩ relation keywords
  2. Relation relevance (weight 0.30) — bidirectional keyword affinity
  3. Path coherence     (weight 0.20) — relation-to-relation semantic grouping

Gaussian noise (configurable σ) ensures the verifier is imperfect,
creating opportunities for failure-aware recovery.

Dynamic re-verification re-scores a revived path using knowledge of
which relations have already failed, penalising similar relations
and boosting temporal opposites.
"""


import random


from fags import Edge

# ──────────────────────────────────────────────
# Relation → keyword mapping
# ──────────────────────────────────────────────

RELATION_KEYWORDS: dict[str, list[str]] = {
    "CAPITAL_OF":          ["capital", "city", "administrative"],
    "LARGEST_CITY":        ["largest", "city", "biggest", "major"],
    "CURRENT_LEADER":      ["current", "leader", "president", "prime", "minister", "head"],
    "FORMER_LEADER":       ["former", "leader", "previous", "past", "president", "ex"],
    "CURRENT_PM":          ["current", "prime", "minister", "pm", "head", "government"],
    "FORMER_PM":           ["former", "prime", "minister", "pm", "previous", "past"],
    "MEMBER_OF":           ["member", "belongs", "party", "organization"],
    "BELONGS_TO":          ["belongs", "member", "part", "affiliated", "party"],
    "ALLY_OF":             ["ally", "partner", "allied", "alliance"],
    "BORDERS":             ["border", "adjacent", "neighbor"],
    "LOCATED_IN":          ["located", "situated", "where", "place", "region"],
    "PART_OF":             ["part", "component", "division", "belongs"],
    "FOUNDED_BY":          ["founded", "created", "established", "founder"],
    "FOUNDED_IN":          ["founded", "established", "year", "created"],
    "CURRENCY_OF":         ["currency", "money", "monetary", "unit"],
    "LANGUAGE_OF":         ["language", "speaks", "tongue", "official"],
    "CONTINENT_OF":        ["continent", "region", "world"],
    "POPULATION_OF":       ["population", "people", "inhabitants"],
    "GOVERNS":             ["governs", "rules", "controls", "government", "governing"],
    "CURRENT_GOVERNMENT":  ["current", "government", "governing", "party", "rules"],
    "EXPORTS":             ["exports", "trade", "sells", "products"],
    "IMPORTS":             ["imports", "buys", "trade", "purchases"],
    "DISCOVERED_BY":       ["discovered", "found", "explorer"],
    "INVENTED_BY":         ["invented", "created", "inventor"],
    "BORN_IN":             ["born", "birthplace", "origin", "native"],
    "DIED_IN":             ["died", "death", "passed"],
    "EDUCATED_AT":         ["educated", "university", "school", "studied"],
    "WORKS_FOR":           ["works", "employed", "job", "company"],
    "SPOUSE_OF":           ["spouse", "married", "wife", "husband"],
    "CHILD_OF":            ["child", "son", "daughter", "parent"],
    "PARENT_OF":           ["parent", "father", "mother"],
    "WROTE":               ["wrote", "author", "book", "publication"],
    "DIRECTED":            ["directed", "director", "film", "movie"],
    "STARRED_IN":          ["starred", "actor", "actress", "film", "movie"],
    "PRODUCED_BY":         ["produced", "producer", "production"],
    "AWARD_WON":           ["award", "prize", "won", "received"],
    "HAS_PROPERTY":        ["property", "characteristic", "feature"],
    "RELATED_TO":          ["related", "connected", "associated"],
    "SUCCESSOR_OF":        ["successor", "followed", "replaced", "next"],
    "PREDECESSOR_OF":      ["predecessor", "before", "preceded", "previous"],
}

# ──────────────────────────────────────────────
# Confusable relation pairs — shared keywords
# ──────────────────────────────────────────────

CONFUSABLE_PAIRS: set[tuple[str, str]] = {
    ("CAPITAL_OF", "LARGEST_CITY"),
    ("CURRENT_LEADER", "FORMER_LEADER"),
    ("CURRENT_PM", "FORMER_PM"),
    ("CURRENT_GOVERNMENT", "FORMER_LEADER"),
    ("MEMBER_OF", "ALLY_OF"),
    ("MEMBER_OF", "BELONGS_TO"),
    ("EXPORTS", "IMPORTS"),
    ("DISCOVERED_BY", "INVENTED_BY"),
    ("BORN_IN", "DIED_IN"),
    ("LOCATED_IN", "PART_OF"),
    ("SUCCESSOR_OF", "PREDECESSOR_OF"),
    ("PARENT_OF", "CHILD_OF"),
    ("FOUNDED_BY", "INVENTED_BY"),
    ("GOVERNS", "CURRENT_GOVERNMENT"),
}

# ──────────────────────────────────────────────
# Relation coherence groups (path coherence)
# ──────────────────────────────────────────────

_COHERENT_GROUPS: list[list[str]] = [
    ["CAPITAL_OF", "LARGEST_CITY", "LOCATED_IN", "BORDERS", "CONTINENT_OF", "PART_OF"],
    ["CURRENT_LEADER", "FORMER_LEADER", "CURRENT_PM", "FORMER_PM",
     "GOVERNS", "CURRENT_GOVERNMENT", "MEMBER_OF", "BELONGS_TO"],
    ["BORN_IN", "DIED_IN", "EDUCATED_AT", "WORKS_FOR", "SPOUSE_OF",
     "CHILD_OF", "PARENT_OF"],
    ["DISCOVERED_BY", "INVENTED_BY", "AWARD_WON", "WROTE"],
    ["DIRECTED", "STARRED_IN", "PRODUCED_BY"],
    ["EXPORTS", "IMPORTS", "CURRENCY_OF"],
    ["POPULATION_OF", "LANGUAGE_OF"],
    ["FOUNDED_BY", "FOUNDED_IN"],
]

RELATION_COHERENCE: dict[tuple[str, str], float] = {}
for _group in _COHERENT_GROUPS:
    for _r1 in _group:
        for _r2 in _group:
            if _r1 != _r2:
                RELATION_COHERENCE[(_r1, _r2)] = 0.8
_DEFAULT_COHERENCE = 0.3


# ──────────────────────────────────────────────
# Temporal-opposite pairs (for re-verification boost)
# ──────────────────────────────────────────────

_TEMPORAL_OPPOSITES: set[frozenset] = {
    frozenset({"CURRENT_LEADER", "FORMER_LEADER"}),
    frozenset({"CURRENT_PM", "FORMER_PM"}),
    frozenset({"CURRENT_GOVERNMENT", "FORMER_LEADER"}),
    frozenset({"SUCCESSOR_OF", "PREDECESSOR_OF"}),
}


def _is_confusable(r1: str, r2: str) -> bool:
    return (r1, r2) in CONFUSABLE_PAIRS or (r2, r1) in CONFUSABLE_PAIRS


def _is_temporal_opposite(r1: str, r2: str) -> bool:
    return frozenset({r1, r2}) in _TEMPORAL_OPPOSITES


# ══════════════════════════════════════════════
# Verifier
# ══════════════════════════════════════════════


# ──────────────────────────────────────────────
# Relation descriptions (for EmbeddingVerifier)
# ──────────────────────────────────────────────

RELATION_DESCRIPTIONS: dict[str, str] = {
    "CAPITAL_OF":          "The capital city or administrative center of a country, state, or region.",
    "LARGEST_CITY":        "The largest, biggest, or most major city in a country or region.",
    "CURRENT_LEADER":      "The current president, leader, or head of state of a country or group.",
    "FORMER_LEADER":       "The former leader, previous president, or past head of state of a country.",
    "CURRENT_PM":          "The current prime minister or head of government of a country.",
    "FORMER_PM":           "The former prime minister or previous head of government of a country.",
    "MEMBER_OF":           "A person or organization belongs to or is a member of a group or institution.",
    "BELONGS_TO":          "A person, entity, or organization is affiliated with or belongs to a political party or group.",
    "ALLY_OF":             "An ally, diplomatic partner, or allied organization of a country or group.",
    "BORDERS":             "A country, state, or region shares a border with or is adjacent to another.",
    "LOCATED_IN":          "An entity, city, or feature is located or situated in a specific place or region.",
    "PART_OF":             "An entity is a part, component, division, or territory of a larger whole.",
    "FOUNDED_BY":          "The founder, creator, or establishers of an organization, company, or city.",
    "FOUNDED_IN":          "The year or time when an organization, company, or city was founded or established.",
    "CURRENCY_OF":         "The official currency or monetary unit used in a country or region.",
    "LANGUAGE_OF":         "The official or primary language spoken in a country or region.",
    "CONTINENT_OF":        "The continent or major geographical region where a place is located.",
    "POPULATION_OF":       "The population, number of people, or inhabitants living in a country or city.",
    "GOVERNS":             "A leader, government, or governing party rules, controls, or governs a country or region.",
    "CURRENT_GOVERNMENT":  "The current governing party or administration of a country.",
    "EXPORTS":             "Products, goods, or commodities that a country trades and sells to other countries.",
    "IMPORTS":             "Products, goods, or commodities that a country trades and buys from other countries.",
    "DISCOVERED_BY":       "The scientist, researcher, or explorer who discovered or found something.",
    "INVENTED_BY":         "The inventor or creator who invented or developed a new technology or device.",
    "BORN_IN":             "The specific city, birthplace, or country where a person was born.",
    "DIED_IN":             "The specific location, city, or place where a person died.",
    "EDUCATED_AT":         "The school, university, or college where a person studied and was educated.",
    "WORKS_FOR":           "The company, employer, or organization where a person is employed and works.",
    "SPOUSE_OF":           "The husband, wife, or marriage partner of a person.",
    "CHILD_OF":            "The son or daughter of a person.",
    "PARENT_OF":           "The parent, mother, or father of a person.",
    "WROTE":               "The author or writer who wrote a book, article, or publication.",
    "DIRECTED":            "The director who directed a movie, film, or theatrical show.",
    "STARRED_IN":          "An actor or actress who starred or acted in a movie, film, or show.",
    "PRODUCED_BY":         "The producer or production company that produced a movie, film, or show.",
    "AWARD_WON":           "An award, prize, or honor won or received by a person or organization.",
    "HAS_PROPERTY":        "A characteristic, property, feature, or attribute of an entity.",
    "RELATED_TO":          "An entity is related, connected, or associated with another.",
    "SUCCESSOR_OF":        "A leader or entity who succeeded, followed, or replaced another in a role.",
    "PREDECESSOR_OF":      "A leader or entity who preceded, came before, or was the predecessor of another."
}

class Verifier:
    """Scores candidate edges for relevance to a question.

    The verifier is intentionally imperfect (via Gaussian noise) so that
    the correct relation is sometimes outscored by a distractor.  This
    creates the recovery opportunities that FAGS is designed to exploit.
    """

    def __init__(
        self,
        noise_std: float = 0.05,
        seed: int = 42,
        keyword_weight: float = 0.50,
        relevance_weight: float = 0.30,
        coherence_weight: float = 0.20,
    ) -> None:
        self.noise_std = noise_std
        self.rng = random.Random(seed)
        self.kw = keyword_weight
        self.rw = relevance_weight
        self.cw = coherence_weight

    # ── public API ────────────────────────────

    def score(
        self,
        query: Any,
        edge: Edge,
        path_relations: Sequence[str] | None = None,
    ) -> float:
        """Score a candidate edge.  Returns float in [0, 1]."""
        if hasattr(query, "keywords"):
            keywords = query.keywords
        elif isinstance(query, str):
            keywords = query.split()
        else:
            keywords = query
            
        rel = edge.relation
        s_kw = self._keyword_overlap(keywords, rel)
        s_rl = self._relation_relevance(keywords, rel)
        s_co = self._path_coherence(rel, path_relations or [])

        raw = self.kw * s_kw + self.rw * s_rl + self.cw * s_co
        noisy = raw + self.rng.gauss(0, self.noise_std)
        return round(max(0.0, min(1.0, noisy)), 4)

    def re_score(
        self,
        query: Any,
        edge: Edge,
        path_relations: Sequence[str] | None = None,
        failed_relations: Sequence[str] | None = None,
    ) -> float:
        """Re-score with knowledge of which relations already failed.

        • Penalise relations confusable with / identical to failed ones.
        • Boost temporal opposites of failed relations.
        """
        base = self.score(query, edge, path_relations)
        if not failed_relations:
            return base

        rel = edge.relation
        penalty = 0.0
        boost = 0.0

        for fr in failed_relations:
            if rel == fr:
                penalty += 0.30
            elif _is_confusable(rel, fr):
                penalty += 0.15
            elif RELATION_COHERENCE.get((rel, fr), 0) > 0.5:
                penalty += 0.05

            if _is_temporal_opposite(rel, fr):
                boost += 0.10

        return round(max(0.0, min(1.0, base - penalty + boost)), 4)

    # ── scoring components ────────────────────

    @staticmethod
    def _keyword_overlap(keywords: Sequence[str], relation: str) -> float:
        """Fraction of question keywords matched by relation keywords."""
        rel_kws = RELATION_KEYWORDS.get(relation, [])
        if not keywords or not rel_kws:
            return 0.0
        hits = 0
        for qk in keywords:
            ql = qk.lower()
            for rk in rel_kws:
                if ql in rk or rk in ql:
                    hits += 1
                    break
        return hits / len(keywords)

    @staticmethod
    def _relation_relevance(keywords: Sequence[str], relation: str) -> float:
        """Bidirectional keyword affinity between question and relation."""
        rel_kws = RELATION_KEYWORDS.get(relation, [])
        if not keywords or not rel_kws:
            return 0.1
        matches = 0
        for rk in rel_kws:
            for qk in keywords:
                if rk.lower() == qk.lower():
                    matches += 2
                elif rk.lower() in qk.lower() or qk.lower() in rk.lower():
                    matches += 1
        return min(1.0, matches / max(1, len(keywords) + len(rel_kws)))

    @staticmethod
    def _path_coherence(relation: str, path_relations: Sequence[str]) -> float:
        """Mean coherence score with relations already on the path."""
        if not path_relations:
            return 0.5
        total = sum(
            RELATION_COHERENCE.get((relation, pr), _DEFAULT_COHERENCE)
            for pr in path_relations
        )
        return total / len(path_relations)



# ══════════════════════════════════════════════


# ══════════════════════════════════════════════
# EmbeddingVerifier
# ══════════════════════════════════════════════

class EmbeddingVerifier:
    """Scores candidate edges using cosine similarity of text embeddings.

    Uses BAAI/bge-small-en-v1.5 to encode queries and natural language relation
    descriptions. Pre-computes relation embeddings to ensure fast evaluation.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        noise_std: float = 0.05,
        seed: int = 42,
    ) -> None:
        from sentence_transformers import SentenceTransformer
        import numpy as np
        
        self.noise_std = noise_std
        self.rng = np.random.default_rng(seed)
        
        # Load model locally
        self.model = SentenceTransformer(model_name)
        
        # Pre-compute relation embeddings
        self.relation_embeddings = {}
        for rel, desc in RELATION_DESCRIPTIONS.items():
            emb = self.model.encode(desc, convert_to_numpy=True, normalize_embeddings=True)
            self.relation_embeddings[rel] = emb
            
        # Cache for last query embedding to speed up score calls
        self._last_query_text = None
        self._last_query_emb = None

    def score(
        self,
        query: Any,
        edge: Edge,
        path_relations: Sequence[str] | None = None,
    ) -> float:
        """Score a candidate edge using Cosine Similarity to relation description.

        Returns float in [0, 1].
        """
        import numpy as np
        
        # Extract query text
        if hasattr(query, "keywords") and not isinstance(query, str):
            query_text = " ".join(query.keywords)
        elif hasattr(query, "question"):
            query_text = query.question
        elif isinstance(query, str):
            query_text = query
        else:
            query_text = " ".join(query)
            
        rel = edge.relation
        if rel not in self.relation_embeddings:
            return 0.0
            
        # Encode query (with caching)
        if self._last_query_text == query_text:
            q_emb = self._last_query_emb
        else:
            q_emb = self.model.encode(query_text, convert_to_numpy=True, normalize_embeddings=True)
            self._last_query_text = query_text
            self._last_query_emb = q_emb
            
        r_emb = self.relation_embeddings[rel]
        
        # Cosine similarity
        cos_sim = float(np.dot(q_emb, r_emb))
        
        # Stretch range [0.3, 0.85] -> [0.0, 1.0] to widen the score gap under noise
        score_val = max(0.0, min(1.0, (cos_sim - 0.3) / (0.85 - 0.3)))
        
        # Apply noise
        noisy = score_val + self.rng.normal(0, self.noise_std)
        return round(max(0.0, min(1.0, noisy)), 4)

    def re_score(
        self,
        query: Any,
        edge: Edge,
        path_relations: Sequence[str] | None = None,
        failed_relations: Sequence[str] | None = None,
    ) -> float:
        """Re-score with knowledge of failed relations. Same penalty rules as Verifier."""
        base = self.score(query, edge, path_relations)
        if not failed_relations:
            return base

        rel = edge.relation
        penalty = 0.0
        boost = 0.0

        for fr in failed_relations:
            if rel == fr:
                penalty += 0.30
            elif _is_confusable(rel, fr):
                penalty += 0.15
            elif RELATION_COHERENCE.get((rel, fr), 0) > 0.5:
                penalty += 0.05

            if _is_temporal_opposite(rel, fr):
                boost += 0.10

        return round(max(0.0, min(1.0, base - penalty + boost)), 4)


# ══════════════════════════════════════════════
# HybridVerifier
# ══════════════════════════════════════════════

class HybridVerifier:
    """Combines Rule-based Verifier and EmbeddingVerifier using a linear weight.

    final_score = alpha * rule_score + (1 - alpha) * bge_score
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        alpha: float = 0.5,
        noise_std: float = 0.05,
        seed: int = 42,
    ) -> None:
        import numpy as np
        
        self.alpha = alpha
        self.noise_std = noise_std
        self.rng = np.random.default_rng(seed)
        
        # Instantiate sub-verifiers with 0.0 noise so we can combine clean scores
        self.bge_verifier = EmbeddingVerifier(model_name=model_name, noise_std=0.0, seed=seed)
        self.rule_verifier = Verifier(noise_std=0.0, seed=seed)

    def score(
        self,
        query: Any,
        edge: Edge,
        path_relations: Sequence[str] | None = None,
    ) -> float:
        """Score candidate edge by combining clean rule and BGE scores + adding noise."""
        s_bge = self.bge_verifier.score(query, edge, path_relations)
        s_rule = self.rule_verifier.score(query, edge, path_relations)
        
        # Linear combination
        raw = self.alpha * s_rule + (1.0 - self.alpha) * s_bge
        
        # Apply noise
        noisy = raw + self.rng.normal(0, self.noise_std)
        return round(max(0.0, min(1.0, noisy)), 4)

    def re_score(
        self,
        query: Any,
        edge: Edge,
        path_relations: Sequence[str] | None = None,
        failed_relations: Sequence[str] | None = None,
    ) -> float:
        """Re-score with failed relations penalties."""
        base = self.score(query, edge, path_relations)
        if not failed_relations:
            return base

        rel = edge.relation
        penalty = 0.0
        boost = 0.0

        for fr in failed_relations:
            if rel == fr:
                penalty += 0.30
            elif _is_confusable(rel, fr):
                penalty += 0.15
            elif RELATION_COHERENCE.get((rel, fr), 0) > 0.5:
                penalty += 0.05

            if _is_temporal_opposite(rel, fr):
                boost += 0.10

        return round(max(0.0, min(1.0, base - penalty + boost)), 4)
