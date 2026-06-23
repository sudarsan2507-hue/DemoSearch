import os
import numpy as np

# Force offline mode for HF model loading
os.environ["HF_HUB_OFFLINE"] = "1"

from fags import KnowledgeGraph, Query, SearchResult
from fags.graph_generator import generate_dataset
from fags.verifier import EmbeddingVerifier
from fags.memory import create_memory
from fags.baseline_search import baseline_search
from fags.failure_search import failure_search

def run_search_experiment(
    graph: KnowledgeGraph,
    queries: list[Query],
    verifier,
    search_mode: str,
) -> list[SearchResult]:
    results = []
    for q in queries:
        if search_mode == "baseline":
            res = baseline_search(
                graph=graph,
                query=q,
                verifier=verifier,
            )
        elif search_mode == "fags":
            memory = create_memory("threshold", threshold=0.15)
            res = failure_search(
                graph=graph,
                query=q,
                verifier=verifier,
                memory=memory,
                shield_depth=0,
                use_certificate=True,
                certificate_bonus=0.10,
                max_backtracks=5,
            )
        results.append(res)
    return results

def evaluate_isolated_verifier(graph, queries, verifier):
    total_hops = 0
    top1_correct = 0
    
    for q in queries:
        gold_nodes = q.gold_path
        gold_relations = q.gold_relations
        
        for i in range(len(gold_relations)):
            curr_node = gold_nodes[i]
            gold_rel = gold_relations[i]
            
            neighbors = graph.get_neighbors(curr_node)
            if not neighbors:
                continue
                
            scored_candidates = []
            for edge in neighbors:
                s = verifier.score(q, edge)
                scored_candidates.append((s, edge.relation))
            
            scored_candidates.sort(key=lambda x: x[0], reverse=True)
            
            if scored_candidates and scored_candidates[0][1] == gold_rel:
                top1_correct += 1
            total_hops += 1

    return top1_correct / total_hops if total_hops > 0 else 0.0

def main():
    print("==================================================")
    print("BGE EMBEDDING VERIFIER SCALING SWEEP")
    print("==================================================")

    num_nodes = 500
    query_count = 500
    seed = 42

    print(f"Generating Medium KG ({num_nodes} nodes) and {query_count} queries...")
    graph, queries = generate_dataset(num_nodes=num_nodes, num_queries=query_count, seed=seed)

    models = [
        ("BGE-Small", "BAAI/bge-small-en-v1.5"),
        ("BGE-Base", "BAAI/bge-base-en-v1.5"),
        ("BGE-Large", "BAAI/bge-large-en-v1.5"),
    ]
    
    eval_records = []

    for label, model_name in models:
        print(f"\nLoading weights and pre-encoding for {label} ({model_name})...")
        try:
            verifier = EmbeddingVerifier(model_name=model_name, noise_std=0.30, seed=seed)
        except Exception as e:
            print(f"  Error loading model {label}: {e}")
            continue
            
        # 1. Run Isolated Verifier Accuracy
        print(f"  Evaluating isolated Top-1 ranking accuracy...")
        top1_acc = evaluate_isolated_verifier(graph, queries, verifier)
        
        # 2. Run Baseline Search
        print(f"  Running Baseline greedy search...")
        base_res = run_search_experiment(graph, queries, verifier, "baseline")
        base_acc = np.mean([1 if r.success else 0 for r in base_res])
        
        # 3. Run FAGS + Certificate Search
        print(f"  Running FAGS + Certificate search...")
        fags_res = run_search_experiment(graph, queries, verifier, "fags")
        fags_acc = np.mean([1 if r.success else 0 for r in fags_res])
        
        # 4. Calculate Search Metrics
        all_hops = []
        for r in fags_res:
            if hasattr(r, "hops_survived_post_revival"):
                all_hops.extend(r.hops_survived_post_revival)
        avg_hops = np.mean(all_hops) if all_hops else 0.0

        queries_with_recovery = sum(1 for r in fags_res if getattr(r, "recovery_attempts", 0) > 0)
        successful_recoveries = sum(1 for r in fags_res if getattr(r, "recovery_attempts", 0) > 0 and r.success)
        recovery_rate = successful_recoveries / queries_with_recovery if queries_with_recovery > 0 else 0.0

        avg_cost_base = np.mean([r.edges_explored for r in base_res])
        avg_cost_fags = np.mean([r.edges_explored for r in fags_res])

        print(f"  Top-1 Verifier Acc: {top1_acc:.2%}")
        print(f"  [Baseline] Accuracy: {base_acc:.2%}")
        print(f"  [FAGS]     Accuracy: {fags_acc:.2%} | Recovery: {recovery_rate:.2%} | Survival: {avg_hops:.2f} hops")

        eval_records.append({
            "label": label,
            "top1_acc": top1_acc,
            "base_acc": base_acc,
            "fags_acc": fags_acc,
            "recovery_rate": recovery_rate,
            "avg_survival": avg_hops,
            "cost_base": avg_cost_base,
            "cost_fags": avg_cost_fags,
        })

    print("\n==================================================")
    print("FINAL BGE SCALING SWEEP COMPARISON")
    print("==================================================")
    print(f"{'Model Size':<12} | {'Top-1 Verifier':<14} | {'Base Acc':<10} | {'FAGS Acc':<10} | {'Recovery Rate':<13} | {'Avg Survival':<12} | {'Cost (FAGS)':<11}")
    print("-" * 95)
    for r in eval_records:
        print(f"{r['label']:<12} | {r['top1_acc']:<14.1%} | {r['base_acc']:<10.1%} | {r['fags_acc']:<10.1%} | {r['recovery_rate']:<13.1%} | {r['avg_survival']:<12.2f} | {r['cost_fags']:<11.1f}")

if __name__ == "__main__":
    main()
