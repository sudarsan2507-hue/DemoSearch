import re

def patch():
    with open(r"d:\Projects\DemoSearch\fags\verifier.py", "r", encoding="utf-8") as f:
        content = f.read()

    # 1. We will insert RELATION_DESCRIPTIONS below RELATION_COHERENCE
    desc_code = """
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
"""
    content = content.replace("RELATION_COHERENCE = {}", "RELATION_COHERENCE = {}" + desc_code)

    # 2. Modify score signature in Verifier class to handle Query objects
    old_score_sig = """    def score(
        self,
        keywords: Sequence[str],
        edge: Edge,
        path_relations: Sequence[str] | None = None,
    ) -> float:
        \"\"\"Score a candidate edge.  Returns float in [0, 1].\"\"\"
        rel = edge.relation
        s_kw = self._keyword_overlap(keywords, rel)"""

    new_score_sig = """    def score(
        self,
        query: Any,
        edge: Edge,
        path_relations: Sequence[str] | None = None,
    ) -> float:
        \"\"\"Score a candidate edge.  Returns float in [0, 1].\"\"\"
        if hasattr(query, "keywords"):
            keywords = query.keywords
        elif isinstance(query, str):
            keywords = query.split()
        else:
            keywords = query
            
        rel = edge.relation
        s_kw = self._keyword_overlap(keywords, rel)"""
    content = content.replace(old_score_sig, new_score_sig)

    # 3. Modify re_score signature in Verifier class
    old_rescore_sig = """    def re_score(
        self,
        keywords: Sequence[str],
        edge: Edge,
        path_relations: Sequence[str] | None = None,
        failed_relations: Sequence[str] | None = None,
    ) -> float:
        \"\"\"Re-score with knowledge of which relations already failed.

        • Penalise relations confusable with / identical to failed ones.
        • Boost temporal opposites of failed relations.
        \"\"\"
        base = self.score(keywords, edge, path_relations)"""

    new_rescore_sig = """    def re_score(
        self,
        query: Any,
        edge: Edge,
        path_relations: Sequence[str] | None = None,
        failed_relations: Sequence[str] | None = None,
    ) -> float:
        \"\"\"Re-score with knowledge of which relations already failed.

        • Penalise relations confusable with / identical to failed ones.
        • Boost temporal opposites of failed relations.
        \"\"\"
        base = self.score(query, edge, path_relations)"""
    content = content.replace(old_rescore_sig, new_rescore_sig)

    # 4. Implement EmbeddingVerifier at the end of the file
    embedding_verifier_code = """


# ══════════════════════════════════════════════
# EmbeddingVerifier
# ══════════════════════════════════════════════

class EmbeddingVerifier:
    \"\"\"Scores candidate edges using cosine similarity of text embeddings.

    Uses BAAI/bge-small-en-v1.5 to encode queries and natural language relation
    descriptions. Pre-computes relation embeddings to ensure fast evaluation.
    \"\"\"

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

    def score(
        self,
        query: Any,
        edge: Edge,
        path_relations: Sequence[str] | None = None,
    ) -> float:
        \"\"\"Score a candidate edge using Cosine Similarity to relation description.

        Returns float in [0, 1].
        \"\"\"
        import numpy as np
        
        # Extract query text
        if hasattr(query, "question"):
            query_text = query.question
        elif isinstance(query, str):
            query_text = query
        else:
            query_text = " ".join(query)
            
        rel = edge.relation
        if rel not in self.relation_embeddings:
            return 0.0
            
        # Encode query
        q_emb = self.model.encode(query_text, convert_to_numpy=True, normalize_embeddings=True)
        r_emb = self.relation_embeddings[rel]
        
        # Cosine similarity (since normalized, it is just dot product)
        cos_sim = float(np.dot(q_emb, r_emb))
        
        # Map cosine similarity from [-1, 1] -> [0, 1]
        score_val = (cos_sim + 1.0) / 2.0
        
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
        \"\"\"Re-score with knowledge of failed relations. Same penalty rules as Verifier.\"\"\"
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
"""
    
    # Write import Any to top of file
    content = "from typing import Sequence, Any\n" + content.replace("from typing import Sequence", "")
    content += embedding_verifier_code

    with open(r"d:\Projects\DemoSearch\fags\verifier.py", "w", encoding="utf-8") as f:
        f.write(content)
    print("Patched verifier.py successfully.")

if __name__ == "__main__":
    patch()
