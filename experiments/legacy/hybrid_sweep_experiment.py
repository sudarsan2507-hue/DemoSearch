import os
import numpy as np

from fags import KnowledgeGraph, Query, SearchResult
from fags.graph_generator import generate_dataset
from fags.verifier import HybridVerifier
from fags.memory import create_memory
from fags.baseline_search import baseline_search
from fags.failure_search import failure_search

def run_experiment(
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

def main():
    print("==================================================")
    print("HYBRID VERIFIER ALPHA SWEEP")
    print("==================================================")

    num_nodes = 500
    query_count = 500
    seed = 42

    print(f"Generating Medium KG ({num_nodes} nodes) and {query_count} queries...")
    graph, queries = generate_dataset(num_nodes=num_nodes, num_queries=query_count, seed=seed)

    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
    
    eval_records = []

    for alpha in alphas:
        print(f"\nRunning with alpha = {alpha:.2f}...")
        # Create Hybrid Verifier
        verifier = HybridVerifier(model_name="BAAI/bge-small-en-v1.5", alpha=alpha, noise_std=0.30, seed=seed)
        
        # Run baseline
        base_res = run_experiment(graph, queries, verifier, "baseline")
        base_acc = np.mean([1 if r.success else 0 for r in base_res])
        
        # Run FAGS
        fags_res = run_experiment(graph, queries, verifier, "fags")
        fags_acc = np.mean([1 if r.success else 0 for r in fags_res])
        
        # Calculate FAGS Hops Survived Post-Revival
        all_hops = []
        for r in fags_res:
            if hasattr(r, "hops_survived_post_revival"):
                all_hops.extend(r.hops_survived_post_revival)
        avg_hops = np.mean(all_hops) if all_hops else 0.0

        # FAGS Recovery Rate
        queries_with_recovery = sum(1 for r in fags_res if getattr(r, "recovery_attempts", 0) > 0)
        successful_recoveries = sum(1 for r in fags_res if getattr(r, "recovery_attempts", 0) > 0 and r.success)
        recovery_rate = successful_recoveries / queries_with_recovery if queries_with_recovery > 0 else 0.0

        avg_cost_base = np.mean([r.edges_explored for r in base_res])
        avg_cost_fags = np.mean([r.edges_explored for r in fags_res])

        print(f"  [Baseline] Accuracy: {base_acc:.2%}")
        print(f"  [FAGS]     Accuracy: {fags_acc:.2%} | Recovery: {recovery_rate:.2%} | Survival: {avg_hops:.2f} hops")

        eval_records.append({
            "alpha": alpha,
            "base_acc": base_acc,
            "fags_acc": fags_acc,
            "recovery_rate": recovery_rate,
            "avg_survival": avg_hops,
            "cost_base": avg_cost_base,
            "cost_fags": avg_cost_fags,
        })

    print("\n==================================================")
    print("FINAL HYBRID SWEEP RESULTS COMPARISON")
    print("==================================================")
    print(f"{'Alpha (Rule wt)':<15} | {'Base Acc':<10} | {'FAGS Acc':<10} | {'Recovery Rate':<13} | {'Avg Survival':<12} | {'Cost (FAGS)':<11}")
    print("-" * 85)
    for r in eval_records:
        print(f"alpha = {r['alpha']:<8.2f} | {r['base_acc']:<10.1%} | {r['fags_acc']:<10.1%} | {r['recovery_rate']:<13.1%} | {r['avg_survival']:<12.2f} | {r['cost_fags']:<11.1f}")

if __name__ == "__main__":
    main()
