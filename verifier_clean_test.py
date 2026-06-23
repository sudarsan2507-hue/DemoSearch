import sys
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')

from fags.graph_generator import generate_dataset
from fags.verifier import Verifier, EmbeddingVerifier

def evaluate_verifier_accuracy(graph, queries, verifier, label="Verifier", use_question=True):
    total_hops = 0
    top1_correct = 0
    top3_correct = 0
    rr_sum = 0.0

    for q in queries:
        gold_nodes = q.gold_path
        gold_relations = q.gold_relations
        
        if use_question:
            q_text = q
        else:
            q_text = q.keywords
            
        for i in range(len(gold_relations)):
            curr_node = gold_nodes[i]
            gold_rel = gold_relations[i]
            
            neighbors = graph.get_neighbors(curr_node)
            if not neighbors:
                continue
                
            scored_candidates = []
            for edge in neighbors:
                s = verifier.score(q_text, edge)
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

    print(f"\nResults for {label}:")
    print(f"  Top-1 Accuracy:    {top1_acc:.2%}")
    print(f"  Top-3 Accuracy:    {top3_acc:.2%}")
    print(f"  Mean Reciprocal Rank (MRR): {mrr:.4f}")

def main():
    graph, queries = generate_dataset(num_nodes=500, num_queries=500, seed=42)
    
    # 0.0 Noise
    print("Loading Rule-based Verifier (noise=0.0)...")
    rule_verifier = Verifier(noise_std=0.0, seed=42)
    
    print("Loading BGE Embedding Verifier (noise=0.0)...")
    bge_verifier = EmbeddingVerifier(model_name="BAAI/bge-small-en-v1.5", noise_std=0.0, seed=42)
    
    evaluate_verifier_accuracy(graph, queries, rule_verifier, "Rule-based Verifier (Clean, using Query object)", use_question=True)
    evaluate_verifier_accuracy(graph, queries, bge_verifier, "BGE Verifier (Clean, using question text)", use_question=True)
    evaluate_verifier_accuracy(graph, queries, bge_verifier, "BGE Verifier (Clean, using keywords)", use_question=False)

if __name__ == "__main__":
    main()
