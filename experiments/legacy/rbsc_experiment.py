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
    rbsc_mode: str,
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
            rbsc_mode=rbsc_mode,
            certificate_bonus=certificate_bonus,
            max_backtracks=5,
        )
        results.append(res)
    return results

def main():
    print("==================================================")
    print("RBSC EXPERIMENT: CERTIFICATE VS RBSC LINEAR VS NONLINEAR")
    print("==================================================")

    num_nodes = 500
    query_count = 500
    seed = 42

    print(f"Generating Medium KG ({num_nodes} nodes) and {query_count} queries...")
    graph, queries = generate_dataset(num_nodes=num_nodes, num_queries=query_count, seed=seed)

    verifier = Verifier(noise_std=0.30, seed=seed)

    configs = [
        ("Certificate Baseline", True, "none"),
        ("RBSC Linear", True, "linear"),
        ("RBSC Nonlinear", True, "nonlinear"),
    ]

    eval_records = []

    for label, use_cert, rbsc_mode in configs:
        print(f"\nRunning {label}...")
        res = run_experiment_on_dataset(
            graph=graph, queries=queries, verifier=verifier, 
            use_certificate=use_cert, rbsc_mode=rbsc_mode, certificate_bonus=0.10
        )
        
        acc = np.mean([1 if r.success else 0 for r in res])
        
        # Calculate Average Hops Survived Post Revival
        all_hops = []
        for r in res:
            all_hops.extend(r.hops_survived_post_revival)
        avg_hops = np.mean(all_hops) if all_hops else 0.0

        # Recovery Rate
        queries_with_recovery = sum(1 for r in res if r.recovery_attempts > 0)
        successful_recoveries = sum(1 for r in res if r.recovery_attempts > 0 and r.success)
        recovery_rate = successful_recoveries / queries_with_recovery if queries_with_recovery > 0 else 0.0

        avg_cost = np.mean([r.edges_explored for r in res])

        # Track Margins
        all_succ_margins = []
        all_fail_margins = []
        for r in res:
            all_succ_margins.extend(r.successful_recovery_margins)
            all_fail_margins.extend(r.failed_recovery_margins)

        avg_succ_margin = np.mean(all_succ_margins) if all_succ_margins else 0.0
        avg_fail_margin = np.mean(all_fail_margins) if all_fail_margins else 0.0

        print(f"  Accuracy: {acc:.2%}")
        print(f"  Recovery Rate: {recovery_rate:.2%}")
        print(f"  Avg Survival: {avg_hops:.2f} hops")
        print(f"  Cost: {avg_cost:.1f} edges explored")
        if rbsc_mode != "none":
            print(f"  Avg Succ Margin: {avg_succ_margin:.4f}")
            print(f"  Avg Fail Margin: {avg_fail_margin:.4f}")

        eval_records.append({
            "label": label,
            "accuracy": acc,
            "recovery_rate": recovery_rate,
            "avg_survival": avg_hops,
            "cost": avg_cost,
            "succ_margin": avg_succ_margin,
            "fail_margin": avg_fail_margin,
        })

    print("\n==================================================")
    print("FINAL METRICS (RBSC)")
    print("==================================================")
    print(f"{'Method':<22} | {'Avg Survival':<12} | {'Recovery Rate':<13} | {'Accuracy':<8} | {'Cost':<6} | {'Avg Succ Margin':<15} | {'Avg Fail Margin':<15}")
    print("-" * 110)
    for r in eval_records:
        sm = f"{r['succ_margin']:.4f}" if r["succ_margin"] > 0 else "N/A"
        fm = f"{r['fail_margin']:.4f}" if r["fail_margin"] > 0 else "N/A"
        print(f"{r['label']:<22} | {r['avg_survival']:<12.2f} | {r['recovery_rate']:<13.1%} | {r['accuracy']:<8.1%} | {r['cost']:<6.1f} | {sm:<15} | {fm:<15}")

if __name__ == "__main__":
    main()
