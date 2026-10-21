import os
import csv
import numpy as np

from fags import KnowledgeGraph, Query, SearchResult
from fags.graph_generator import generate_dataset
from fags.verifier import Verifier, EmbeddingVerifier
from fags.memory import create_memory
from fags.baseline_search import baseline_search
from fags.failure_search import failure_search
from fags.evaluation import evaluate_results

RESULTS_DIR = r"d:\Projects\DemoSearch\results"
os.makedirs(RESULTS_DIR, exist_ok=True)

def run_experiment(
    graph: KnowledgeGraph,
    queries: list[Query],
    verifier,
    search_mode: str, # "baseline" | "fags"
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
            # Clean V2 FAGS: ThresholdMemory + 1-Hop Certificate
            memory = create_memory("threshold", threshold=0.15)
            res = failure_search(
                graph=graph,
                query=q,
                verifier=verifier,
                memory=memory,
                shield_depth=0, # No naive shield
                use_certificate=True,
                certificate_bonus=0.10,
                max_backtracks=5,
            )
        results.append(res)
    return results

def main():
    print("==================================================")
    print("VERIFIER SEARCH COMPARISON EXPERIMENT")
    print("==================================================")

    num_nodes = 500
    query_count = 500
    seed = 42

    print(f"Generating Medium KG ({num_nodes} nodes) and {query_count} queries...")
    graph, queries = generate_dataset(num_nodes=num_nodes, num_queries=query_count, seed=seed)

    # Instantiate verifiers with 0.30 noise
    print("Loading Rule-based Verifier...")
    rule_verifier = Verifier(noise_std=0.30, seed=seed)
    
    print("Loading BGE Embedding Verifier (may take a moment to load model)...")
    bge_verifier = EmbeddingVerifier(model_name="BAAI/bge-small-en-v1.5", noise_std=0.30, seed=seed)

    configs = [
        ("Rule Verifier - Baseline", rule_verifier, "baseline"),
        ("Rule Verifier - FAGS + Cert", rule_verifier, "fags"),
        ("BGE Verifier - Baseline", bge_verifier, "baseline"),
        ("BGE Verifier - FAGS + Cert", bge_verifier, "fags"),
    ]

    eval_records = []

    for label, verifier, search_mode in configs:
        print(f"\nRunning {label}...")
        res = run_experiment(graph, queries, verifier, search_mode)
        
        acc = np.mean([1 if r.success else 0 for r in res])
        
        # Calculate Average Hops Survived Post Revival (for FAGS runs)
        all_hops = []
        for r in res:
            if hasattr(r, "hops_survived_post_revival"):
                all_hops.extend(r.hops_survived_post_revival)
        avg_hops = np.mean(all_hops) if all_hops else 0.0

        # Recovery Rate
        queries_with_recovery = sum(1 for r in res if getattr(r, "recovery_attempts", 0) > 0)
        successful_recoveries = sum(1 for r in res if getattr(r, "recovery_attempts", 0) > 0 and r.success)
        recovery_rate = successful_recoveries / queries_with_recovery if queries_with_recovery > 0 else 0.0

        avg_cost = np.mean([r.edges_explored for r in res])

        print(f"  Accuracy: {acc:.2%}")
        print(f"  Recovery Rate: {recovery_rate:.2%}")
        print(f"  Avg Survival: {avg_hops:.2f} hops")
        print(f"  Cost: {avg_cost:.1f} edges explored")

        eval_records.append({
            "label": label,
            "accuracy": acc,
            "recovery_rate": recovery_rate,
            "avg_survival": avg_hops,
            "cost": avg_cost,
        })

    print("\n==================================================")
    print("FINAL SEARCH PERFORMANCE COMPARISON")
    print("==================================================")
    print(f"{'Configuration':<28} | {'Avg Survival':<12} | {'Recovery Rate':<13} | {'Accuracy':<8} | {'Cost':<6}")
    print("-" * 75)
    for r in eval_records:
        surv = f"{r['avg_survival']:.2f}" if r["avg_survival"] > 0 else "N/A"
        rec = f"{r['recovery_rate']:.1%}" if r["recovery_rate"] > 0 else "N/A"
        print(f"{r['label']:<28} | {surv:<12} | {rec:<13} | {r['accuracy']:<8.1%} | {r['cost']:<6.1f}")

if __name__ == "__main__":
    main()
