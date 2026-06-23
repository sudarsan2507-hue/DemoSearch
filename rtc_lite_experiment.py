import os
import csv
import matplotlib.pyplot as plt
import numpy as np

from fags import KnowledgeGraph, Query, SearchResult
from fags.graph_generator import generate_dataset
from fags.verifier import Verifier
from fags.memory import create_memory
from fags.baseline_search import baseline_search
from fags.failure_search import failure_search
from fags.evaluation import evaluate_results

RESULTS_DIR = r"d:\Projects\DemoSearch\results"
os.makedirs(RESULTS_DIR, exist_ok=True)

def run_experiment_on_dataset(
    graph: KnowledgeGraph,
    queries: list[Query],
    verifier: Verifier,
    use_certificate: bool,
    use_rtc_lite: bool,
    certificate_bonus: float = 0.10,
) -> list[SearchResult]:
    results = []
    for q in queries:
        memory = create_memory("top1", threshold=0.15)
        res = failure_search(
            graph=graph,
            query=q,
            verifier=verifier,
            memory=memory,
            shield_depth=0, # No naive shield
            use_certificate=use_certificate,
            use_rtc_lite=use_rtc_lite,
            certificate_bonus=certificate_bonus,
            max_backtracks=5,
        )
        results.append(res)
    return results

def main():
    print("==================================================")
    print("KILL TEST: RTC-LITE VS CERTIFICATE")
    print("==================================================")

    num_nodes = 500
    query_count = 500
    seed = 42

    print(f"Generating Medium KG ({num_nodes} nodes) and {query_count} queries...")
    graph, queries = generate_dataset(num_nodes=num_nodes, num_queries=query_count, seed=seed)

    verifier = Verifier(noise_std=0.30, seed=seed)

    configs = [
        ("Certificate (1-Hop)", True, False),
        ("RTC-Lite (2-Hop)", True, True),
    ]

    eval_records = []

    for label, use_cert, use_rtc in configs:
        print(f"\nRunning {label}...")
        res = run_experiment_on_dataset(
            graph=graph, queries=queries, verifier=verifier, 
            use_certificate=use_cert, use_rtc_lite=use_rtc, certificate_bonus=0.10
        )
        
        acc = np.mean([1 if r.success else 0 for r in res])
        
        # Calculate Average Hops Survived Post Revival
        all_hops = []
        for r in res:
            all_hops.extend(r.hops_survived_post_revival)
        avg_hops = np.mean(all_hops) if all_hops else 0.0

        # Calculate Trajectory Metrics
        total_attempts = sum(r.trajectory_attempts for r in res)
        total_matches = sum(r.trajectory_matches for r in res)
        total_utilities = sum(r.trajectory_utilities for r in res)

        traj_match_rate = total_matches / total_attempts if total_attempts > 0 else 0.0
        traj_utility_rate = total_utilities / total_matches if total_matches > 0 else 0.0

        # Recovery Rate (Successes / Queries where a revival happened)
        # simplified as just overall accuracy for now, but to match exactly:
        queries_with_recovery = sum(1 for r in res if r.recovery_attempts > 0)
        successful_recoveries = sum(1 for r in res if r.recovery_attempts > 0 and r.success)
        recovery_rate = successful_recoveries / queries_with_recovery if queries_with_recovery > 0 else 0.0

        avg_cost = np.mean([r.edges_explored for r in res])

        print(f"  Accuracy: {acc:.2%}")
        print(f"  Recovery Rate: {recovery_rate:.2%}")
        print(f"  Avg Survival: {avg_hops:.2f} hops")
        print(f"  Cost: {avg_cost:.1f} edges explored")
        if use_rtc:
            print(f"  Trajectory Match Rate: {traj_match_rate:.2%} ({total_matches}/{total_attempts})")
            print(f"  Trajectory Utility Rate: {traj_utility_rate:.2%} ({total_utilities}/{total_matches})")

        eval_records.append({
            "label": label,
            "accuracy": acc,
            "recovery_rate": recovery_rate,
            "avg_survival": avg_hops,
            "cost": avg_cost,
            "match_rate": traj_match_rate,
            "utility_rate": traj_utility_rate,
        })

    print("\n==================================================")
    print("FINAL METRICS (KILL TEST)")
    print("==================================================")
    print(f"{'Method':<20} | {'Avg Survival':<12} | {'Recovery Rate':<13} | {'Accuracy':<8} | {'Cost':<6} | {'Match Rate':<10} | {'Utility Rate':<12}")
    print("-" * 105)
    for r in eval_records:
        mr = f"{r['match_rate']:.1%}" if r["match_rate"] > 0 else "N/A"
        ur = f"{r['utility_rate']:.1%}" if r["utility_rate"] > 0 else "N/A"
        print(f"{r['label']:<20} | {r['avg_survival']:<12.2f} | {r['recovery_rate']:<13.1%} | {r['accuracy']:<8.1%} | {r['cost']:<6.1f} | {mr:<10} | {ur:<12}")

if __name__ == "__main__":
    main()
