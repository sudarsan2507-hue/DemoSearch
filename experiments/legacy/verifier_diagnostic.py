import random
import numpy as np

from fags import Edge, Query
from fags.graph_generator import generate_dataset
from fags.verifier import Verifier, EmbeddingVerifier, RELATION_DESCRIPTIONS

def evaluate_verifier_accuracy(graph, queries, verifier, label="Verifier"):
    total_hops = 0
    top1_correct = 0
    top3_correct = 0
    rr_sum = 0.0

    for q in queries:
        gold_nodes = q.gold_path
        gold_relations = q.gold_relations
        
        # Traverse along the gold path
        for i in range(len(gold_relations)):
            curr_node = gold_nodes[i]
            gold_rel = gold_relations[i]
            
            # Get neighbors from graph
            neighbors = graph.get_neighbors(curr_node)
            if not neighbors:
                continue
                
            # Score all neighbor relations
            scored_candidates = []
            for edge in neighbors:
                s = verifier.score(q, edge)
                scored_candidates.append((s, edge.relation))
            
            # Sort candidates by score descending
            scored_candidates.sort(key=lambda x: x[0], reverse=True)
            
            # Find the rank of the gold relation (handling duplicates by taking first rank)
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
    print(f"  Total evaluations: {total_hops}")
    print(f"  Top-1 Accuracy:    {top1_acc:.2%}")
    print(f"  Top-3 Accuracy:    {top3_acc:.2%}")
    print(f"  Mean Reciprocal Rank (MRR): {mrr:.4f}")
    
    return top1_acc, top3_acc, mrr

def log_sample_queries(graph, queries, verifier, count=5):
    # Select random queries
    sampled_queries = random.sample(queries, min(count, len(queries)))
    
    print("\n==================================================")
    print("VERIFIER SAMPLE QUERY INSPECTION")
    print("==================================================")
    
    for i, q in enumerate(sampled_queries, start=1):
        print(f"\nSample {i}:")
        print(f"  Query Question: {q.question}")
        print(f"  Keywords:       {q.keywords}")
        
        # Get start node and evaluate neighbors
        curr_node = q.start_node
        neighbors = graph.get_neighbors(curr_node)
        
        # Score neighbors
        scored_candidates = []
        for edge in neighbors:
            s = verifier.score(q, edge)
            scored_candidates.append((s, edge.relation))
            
        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        
        print(f"  Top relations:")
        for rank, (score, rel) in enumerate(scored_candidates[:5], start=1):
            desc = RELATION_DESCRIPTIONS.get(rel, "N/A")
            print(f"    {rank}. {rel:<20} Score: {score:.4f} | Description: '{desc}'")

def main():
    random.seed(42)
    
    print("Generating KG and Queries...")
    graph, queries = generate_dataset(num_nodes=500, num_queries=500, seed=42)
    
    # Instantiate verifiers
    print("Loading Rule-based Verifier...")
    rule_verifier = Verifier(noise_std=0.30, seed=42)
    
    print("Loading BGE Embedding Verifier (may take a moment to load model)...")
    bge_verifier = EmbeddingVerifier(model_name="BAAI/bge-small-en-v1.5", noise_std=0.30, seed=42)
    
    # 1. Run Quantitative Diagnostic
    evaluate_verifier_accuracy(graph, queries, rule_verifier, "Rule-based Verifier (noise=0.30)")
    evaluate_verifier_accuracy(graph, queries, bge_verifier, "BGE Embedding Verifier (noise=0.30)")
    
    # 2. Run Qualitative Diagnostic (Top-5 Samples)
    log_sample_queries(graph, queries, bge_verifier, count=3)

if __name__ == "__main__":
    main()
