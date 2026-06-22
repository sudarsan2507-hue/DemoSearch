"""Synthetic knowledge-graph and query generator.

Generates graphs with controlled structure:
  • Gold paths   — correct multi-hop chains (2-5 hops)
  • Distractors  — edges using confusable relations at gold-path branch points
  • Dead ends    — leaf nodes with no useful outgoing edges
  • Contradictions — nodes with evidence conflicting with gold-path evidence
  • Ambiguous paths — branch points where two relations score nearly equally
  • Random connectivity — background edges for realism

Query generation produces questions with keywords derived from the gold
path relations.  Difficulty is controlled by distractor density and the
presence of confusable relations at the first branch point.
"""

from __future__ import annotations

import random
from typing import Sequence

from fags import Edge, KnowledgeGraph, Node, Query
from fags.verifier import RELATION_KEYWORDS, CONFUSABLE_PAIRS

# ──────────────────────────────────────────────
# Relation pools by semantic domain
# ──────────────────────────────────────────────

DOMAIN_RELATIONS: dict[str, list[str]] = {
    "geography": [
        "CAPITAL_OF", "LARGEST_CITY", "BORDERS", "LOCATED_IN",
        "PART_OF", "CONTINENT_OF",
    ],
    "politics": [
        "CURRENT_LEADER", "FORMER_LEADER", "CURRENT_PM", "FORMER_PM",
        "GOVERNS", "CURRENT_GOVERNMENT", "MEMBER_OF", "BELONGS_TO",
    ],
    "biography": [
        "BORN_IN", "DIED_IN", "EDUCATED_AT", "WORKS_FOR",
        "SPOUSE_OF", "CHILD_OF", "PARENT_OF",
    ],
    "science": [
        "DISCOVERED_BY", "INVENTED_BY", "AWARD_WON", "WROTE",
    ],
    "culture": [
        "DIRECTED", "STARRED_IN", "PRODUCED_BY", "LANGUAGE_OF",
        "CURRENCY_OF", "FOUNDED_BY", "FOUNDED_IN",
    ],
    "economics": [
        "EXPORTS", "IMPORTS", "POPULATION_OF",
    ],
    "generic": [
        "RELATED_TO", "HAS_PROPERTY", "SUCCESSOR_OF", "PREDECESSOR_OF",
    ],
}

ALL_RELATIONS: list[str] = [
    rel for rels in DOMAIN_RELATIONS.values() for rel in rels
]

# Flat lookup: relation → domain
_REL_DOMAIN: dict[str, str] = {}
for _dom, _rels in DOMAIN_RELATIONS.items():
    for _r in _rels:
        _REL_DOMAIN[_r] = _dom

# Build confusable lookup: relation → list of confusable partners
_CONFUSABLE_MAP: dict[str, list[str]] = {}
for _a, _b in CONFUSABLE_PAIRS:
    _CONFUSABLE_MAP.setdefault(_a, []).append(_b)
    _CONFUSABLE_MAP.setdefault(_b, []).append(_a)


# ══════════════════════════════════════════════
# Graph generator
# ══════════════════════════════════════════════

class GraphGenerator:
    """Generates a synthetic knowledge graph and associated queries."""

    def __init__(self, num_nodes: int, seed: int = 42) -> None:
        self.num_nodes = num_nodes
        self.rng = random.Random(seed)
        self.graph = KnowledgeGraph()
        self._gold_paths: list[dict] = []   # internal gold-path records
        self._domains = list(DOMAIN_RELATIONS.keys())

    # ── public API ────────────────────────────

    def generate(self) -> KnowledgeGraph:
        """Build the knowledge graph (nodes + edges)."""
        self._create_nodes()
        self._create_gold_paths()
        self._add_distractors()
        self._add_dead_ends()
        self._add_contradictions()
        self._add_random_edges()
        return self.graph

    def generate_queries(self, num_queries: int = 1000) -> list[Query]:
        """Generate *num_queries* queries from the gold paths."""
        if not self._gold_paths:
            raise RuntimeError("Call generate() before generate_queries().")
        queries: list[Query] = []
        for i in range(num_queries):
            gp = self._gold_paths[i % len(self._gold_paths)]
            q = self._query_from_gold_path(gp, query_index=i)
            queries.append(q)
        return queries

    # ── node creation ─────────────────────────

    def _create_nodes(self) -> None:
        domains = self._domains
        for i in range(self.num_nodes):
            domain = domains[i % len(domains)]
            node = Node(
                id=f"n_{i}",
                label=f"Entity_{i}",
                node_type="entity",
                domain=domain,
                properties={"index": i},
                evidence={},
            )
            self.graph.add_node(node)

    # ── gold path creation ────────────────────

    def _create_gold_paths(self) -> None:
        """Create gold paths (correct reasoning chains).

        Number of gold paths scales with graph size to ensure sufficient
        query diversity.
        """
        node_ids = list(self.graph.nodes.keys())
        num_paths = max(30, self.num_nodes * 2)

        for _ in range(num_paths):
            hop_count = self.rng.randint(2, min(5, self.num_nodes // 3 + 1))
            # Sample a sequence of distinct nodes
            if len(node_ids) < hop_count + 1:
                hop_count = len(node_ids) - 1
            path_nodes = self.rng.sample(node_ids, hop_count + 1)

            # Pick relations for each hop from a consistent domain
            domain = self.rng.choice(self._domains)
            domain_rels = DOMAIN_RELATIONS[domain]
            if len(domain_rels) < hop_count:
                # Supplement with generic relations
                domain_rels = domain_rels + DOMAIN_RELATIONS["generic"]
            relations = [self.rng.choice(domain_rels) for _ in range(hop_count)]

            # Create edges
            for j in range(hop_count):
                src, tgt = path_nodes[j], path_nodes[j + 1]
                self.graph.add_edge(Edge(source=src, target=tgt, relation=relations[j]))

            # Mark answer node
            ans_node = self.graph.get_node(path_nodes[-1])
            if ans_node:
                ans_node.node_type = "answer"
                # Add evidence on the answer node
                evidence_key = f"answer_for_{path_nodes[0]}"
                ans_node.evidence[evidence_key] = "correct"

            # Add evidence along intermediate nodes
            for j, nid in enumerate(path_nodes[1:-1], start=1):
                n = self.graph.get_node(nid)
                if n:
                    n.evidence[f"path_{path_nodes[0]}"] = f"step_{j}"

            self._gold_paths.append({
                "nodes": path_nodes,
                "relations": relations,
                "domain": domain,
                "hop_count": hop_count,
            })

    # ── distractor edges ──────────────────────

    def _add_distractors(self) -> None:
        """At each node along each gold path, add confusable distractor edges."""
        node_ids = list(self.graph.nodes.keys())

        for gp in self._gold_paths:
            path_nodes = set(gp["nodes"])
            for j, nid in enumerate(gp["nodes"][:-1]):
                gold_rel = gp["relations"][j]

                # Add 1–3 distractor edges from this node
                num_distractors = self.rng.randint(1, 3)
                for _ in range(num_distractors):
                    # Choose a distractor target NOT on the gold path
                    candidates = [n for n in node_ids if n not in path_nodes and n != nid]
                    if not candidates:
                        continue
                    dist_target = self.rng.choice(candidates)

                    # Choose a confusable relation if available, else random
                    confusables = _CONFUSABLE_MAP.get(gold_rel, [])
                    if confusables and self.rng.random() < 0.6:
                        dist_rel = self.rng.choice(confusables)
                    else:
                        dist_rel = self.rng.choice(ALL_RELATIONS)

                    self.graph.add_edge(Edge(
                        source=nid, target=dist_target, relation=dist_rel,
                    ))

                    # Mark distractor node
                    dn = self.graph.get_node(dist_target)
                    if dn and dn.node_type == "entity":
                        dn.node_type = "distractor"

    # ── dead-end nodes ────────────────────────

    def _add_dead_ends(self) -> None:
        """Create leaf nodes reachable from some gold-path nodes but with
        no outgoing edges (= guaranteed dead ends for the search)."""
        num_dead = max(2, self.num_nodes // 10)
        node_ids = list(self.graph.nodes.keys())

        for i in range(num_dead):
            de_id = f"dead_{i}"
            self.graph.add_node(Node(
                id=de_id, label=f"DeadEnd_{i}", node_type="dead_end",
                domain="generic",
            ))
            # Connect from a random existing node
            src = self.rng.choice(node_ids)
            rel = self.rng.choice(ALL_RELATIONS)
            self.graph.add_edge(Edge(source=src, target=de_id, relation=rel))

    # ── contradiction edges ───────────────────

    def _add_contradictions(self) -> None:
        """Add nodes that contradict evidence on gold-path answer nodes.

        E.g. if gold says governing_party=Labor, a contradiction node
        says governing_party=Liberal.
        """
        num_contradictions = max(2, self.num_nodes // 15)
        node_ids = list(self.graph.nodes.keys())

        for i in range(num_contradictions):
            c_id = f"contra_{i}"
            self.graph.add_node(Node(
                id=c_id, label=f"Contradiction_{i}", node_type="distractor",
                domain="generic",
            ))
            # Pick a gold path and contradict its evidence
            gp = self._gold_paths[i % len(self._gold_paths)]
            ans_id = gp["nodes"][-1]
            ans = self.graph.get_node(ans_id)
            if ans and ans.evidence:
                # Copy evidence keys but with conflicting values
                contra_node = self.graph.get_node(c_id)
                if contra_node:
                    for k, v in ans.evidence.items():
                        contra_node.evidence[k] = f"CONTRADICTS_{v}"

            # Connect from a mid-path node
            mid_idx = len(gp["nodes"]) // 2
            mid_node = gp["nodes"][mid_idx]
            rel = self.rng.choice(ALL_RELATIONS)
            self.graph.add_edge(Edge(source=mid_node, target=c_id, relation=rel))

    # ── random background edges ───────────────

    def _add_random_edges(self) -> None:
        """Add random edges for connectivity and realism."""
        node_ids = list(self.graph.nodes.keys())
        # Also include dead-end and contradiction nodes
        all_ids = list(self.graph.nodes.keys())
        num_random = max(self.num_nodes // 2, 5)

        for _ in range(num_random):
            src = self.rng.choice(all_ids)
            tgt = self.rng.choice(all_ids)
            if src == tgt:
                continue
            rel = self.rng.choice(ALL_RELATIONS)
            self.graph.add_edge(Edge(source=src, target=tgt, relation=rel))

    # ── query generation ──────────────────────

    def _query_from_gold_path(self, gp: dict, query_index: int) -> Query:
        """Build a Query from a gold-path record."""
        path_nodes = gp["nodes"]
        relations = gp["relations"]
        hop_count = gp["hop_count"]

        # Collect keywords from gold-path relations
        all_keywords: list[str] = []
        for rel in relations:
            kws = RELATION_KEYWORDS.get(rel, [])
            all_keywords.extend(kws)
        # Deduplicate while preserving order
        seen: set[str] = set()
        keywords: list[str] = []
        for kw in all_keywords:
            if kw not in seen:
                seen.add(kw)
                keywords.append(kw)

        # Optionally drop some keywords to vary difficulty
        if len(keywords) > 3:
            keep = self.rng.randint(max(2, len(keywords) // 2), len(keywords))
            keywords = self.rng.sample(keywords, keep)

        # Determine difficulty
        first_node = path_nodes[0]
        neighbors = self.graph.get_neighbors(first_node)
        confusable_count = 0
        gold_first_rel = relations[0]
        for edge in neighbors:
            if edge.relation != gold_first_rel:
                if edge.relation in _CONFUSABLE_MAP.get(gold_first_rel, []):
                    confusable_count += 1

        if confusable_count == 0 and hop_count <= 2:
            difficulty = "easy"
        elif confusable_count >= 2 or hop_count >= 4:
            difficulty = "hard"
        else:
            difficulty = "medium"

        # Build a synthetic question string
        rel_phrases = " → ".join(relations)
        question = (
            f"Starting from {self.graph.get_node(path_nodes[0]).label}, "
            f"follow [{rel_phrases}] to reach the answer."
        )

        return Query(
            id=f"q_{query_index}",
            question=question,
            start_node=path_nodes[0],
            answer_node=path_nodes[-1],
            gold_path=list(path_nodes),
            gold_relations=list(relations),
            keywords=keywords,
            difficulty=difficulty,
            hop_count=hop_count,
        )


# ══════════════════════════════════════════════
# Convenience function
# ══════════════════════════════════════════════

def generate_dataset(
    num_nodes: int,
    num_queries: int = 1000,
    seed: int = 42,
) -> tuple[KnowledgeGraph, list[Query]]:
    """One-call graph + query generation."""
    gen = GraphGenerator(num_nodes=num_nodes, seed=seed)
    graph = gen.generate()
    queries = gen.generate_queries(num_queries)
    return graph, queries
