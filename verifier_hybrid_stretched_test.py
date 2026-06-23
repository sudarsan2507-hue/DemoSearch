import sys
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')

from fags.graph_generator import generate_dataset
from fags.verifier import Verifier, EmbeddingVerifier, RELATION_COHERENCE, _DEFAULT_COHERENCE

class HybridStretchedVerifier(EmbeddingVerifier):
    def score(
        self,
        query: Any,
        edge: Edge,
        path_relations: Sequence[str] | None = None,
    ) -> float:
        import numpy as np
        
        # 1. Cosine similarity component (weight 0.80)
        if hasattr(query, "question"):
            query_text = query.question
        elif isinstance(query, str):
            query_text = query
        else:
            query_text = " ".join(query)
            
        rel = edge.relation
        if rel not in self.relation_embeddings:
            cos_sim_mapped = 0.0
        else:
            # Encode query (with caching from superclass)
            if self._last_query_text == query_text:
                q_emb = self._last_query_emb
            else:
                q_emb = self.model.encode(query_text, convert_to_numpy=True, normalize_embeddings=True)
                self._last_query_text = query_text
                self._last_query_emb = q_emb
                
            r_emb = self.relation_embeddings[rel]
            cos_sim = float(np.dot(q_emb, r_emb))
            
            # Stretch range [0.3, 0.85] -> [0.0, 1.0]
            cos_sim_mapped = max(0.0, min(1.0, (cos_sim - 0.3) / (0.85 - 0.3)))
            
        # 2. Path coherence component (weight 0.20)
        if not path_relations:
            s_co = 0.5
        else:
            total = sum(
                RELATION_COHERENCE.get((rel, pr), _DEFAULT_COHERENCE)
                for pr in path_relations
            )
            s_co = total / len(path_relations)
            
        # Weighted raw score
        raw = 0.80 * cos_sim_mapped + 0.20 * s_co
        
        # Apply noise
        noisy = raw + self.rng.normal(0, self.noise_std)
        return round(max(0.0, min(1.0, noisy)), 4)

def evaluate_verifier_accuracy(graph, queries, verifier, label="Verifier", use_question=True):
    total_hops = 0
    top1_correct = 0
    top3_correct = 0
    rr_sum = 0.0

    for q in queries:
        gold_nodes = q.gold_path
        gold_relations = q.gold_relations
        
        q_val = q if use_question else q.keywords
            
        for i in range(len(gold_relations)):
            curr_node = gold_nodes[i]
            gold_rel = gold_relations[i]
            
            neighbors = graph.get_neighbors(curr_node)
            if not neighbors:
                continue
                
            scored_candidates = []
            path_relations_so_far = gold_relations[:i]
            for edge in neighbors:
                s = verifier.score(q_val, edge, path_relations_so_far)
                scored_candidates.append((s, edge.relation))
            
            scored_candidates.sort(key=lambda x: x[0], reverse=True)
            
            gold_rank = -1
            for rank, (score, rel) in enumerate(scored_candidates, start=1):
                if rel == gold_rel:
                    gold_rank = rank
                    break
            
            if gold_rank != -1:
                total_hops += 1
                if gold_rank == 1:
                    top1_correct += 1
                if gold_rank <= 3:
                    top3_correct += 1
                rr_sum += 1.0 / gold_rank

    top1_acc = top1_correct / total_hops if total_hops > 0 else 0.0
    top3_acc = top3_correct / total_hops if total_hops > 0 else 0.0
    mrr = rr_sum / total_hops if total_hops > 0 else 0.0

    print(f"  {label:<40} | Top-1: {top1_acc:.2%} | Top-3: {top3_acc:.2%} | MRR: {mrr:.4f}")

def main():
    graph, queries = generate_dataset(num_nodes=500, num_queries=500, seed=42)
    
    print("==================================================")
    print("HYBRID STRETCHED VERIFIER TESTS (noise=0.30)")
    print("==================================================")
    
    from fags.verifier import Verifier
    rule_v = Verifier(noise_std=0.30, seed=42)
    evaluate_verifier_accuracy(graph, queries, rule_v, "Rule-based Verifier", use_question=True)
    
    # BGE Stretched (no coherence)
    from verifier_scaling_test import ScaledVerifier
    bge_stretched_kw = ScaledVerifier(scale_mode="stretched", noise_std=0.30, seed=42)
    evaluate_verifier_accuracy(graph, queries, bge_stretched_kw, "BGE Stretched (keywords only, no coherence)", use_question=False)
    
    # BGE Stretched Hybrid
    bge_hybrid_kw = HybridStretchedVerifier(noise_std=0.30, seed=42)
    evaluate_verifier_accuracy(graph, queries, bge_hybrid_kw, "BGE Stretched Hybrid (keywords + coherence)", use_question=False)
    evaluate_verifier_accuracy(graph, queries, bge_hybrid_kw, "BGE Stretched Hybrid (question + coherence)", use_question=True)

if __name__ == "__main__":
    from typing import Sequence, Any
    from fags import Edge
    main()
