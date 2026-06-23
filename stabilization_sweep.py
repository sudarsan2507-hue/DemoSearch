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
    strategy: str,
    shield_depth: int,
    use_certificate: bool,
    certificate_bonus: float,
) -> list[SearchResult]:
    results = []
    
    for q in queries:
        memory = create_memory(strategy, threshold=0.15)
        res = failure_search(
            graph=graph,
            query=q,
            verifier=verifier,
            memory=memory,
            shield_depth=shield_depth,
            use_certificate=use_certificate,
            certificate_bonus=certificate_bonus,
        )
        results.append(res)
    return results

def run_baseline_on_dataset(
    graph: KnowledgeGraph,
    queries: list[Query],
    verifier: Verifier,
) -> list[SearchResult]:
    results = []
    for q in queries:
        res = baseline_search(graph=graph, query=q, verifier=verifier)
        results.append(res)
    return results

def export_sweep(filename: str, records: list[dict], title: str):
    path = os.path.join(RESULTS_DIR, filename)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Configuration", "Baseline Accuracy", "FAGS Accuracy", 
            "Accuracy Gain", "Additional Search Cost", "Efficiency Ratio", 
            "Gold Path Recovery Rate", "Avg Hops Survived Post Revival"
        ])
        for r in records:
            writer.writerow([
                r["label"], f"{r['accuracy_baseline']:.2%}", f"{r['accuracy_fags']:.2%}",
                f"{r['accuracy_gain']:.2%}", f"{r['additional_search_cost']:.2%}", 
                f"{r['efficiency_ratio']:.3f}", f"{r['gold_path_recovery_rate']:.2%}",
                f"{r['avg_hops_survived']:.2f}"
            ])
    print(f"Exported {title} to {filename}")

def plot_sweep(filename: str, records: list[dict], title: str, xlabel: str):
    labels = [r["label"] for r in records]
    fags_accs = [r["accuracy_fags"] * 100 for r in records]
    avg_hops = [r["avg_hops_survived"] for r in records]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()
    
    x = np.arange(len(labels))
    width = 0.35
    
    ax1.bar(x - width/2, fags_accs, width, color="crimson", label="Accuracy")
    ax2.bar(x + width/2, avg_hops, width, color="darkorange", label="Avg Hops")
    
    ax1.set_ylabel("Accuracy (%)", color="crimson")
    ax2.set_ylabel("Avg Hops Survived", color="darkorange")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=15, ha="right")
    ax1.tick_params(axis='y', labelcolor="crimson")
    ax2.tick_params(axis='y', labelcolor="darkorange")
    
    plt.title(title)
    fig.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, filename), dpi=150)
    plt.close()

def execute_batch(graph, queries, verifier, base_res, configs, label_prefix=""):
    records = []
    for label, params in configs:
        full_label = f"{label_prefix}{label}"
        print(f"\nRunning {full_label}...")
        res = run_experiment_on_dataset(
            graph=graph, queries=queries, verifier=verifier, strategy="top1",
            shield_depth=params["shield_depth"],
            use_certificate=params["use_certificate"],
            certificate_bonus=params["certificate_bonus"]
        )
        
        acc = np.mean([1 if r.success else 0 for r in res])
        print(f"  Accuracy: {acc:.2%}")
        
        metrics = evaluate_results(base_res, res, full_label)
        
        all_hops = []
        for r in res:
            all_hops.extend(r.hops_survived_post_revival)
            
        avg_hops = np.mean(all_hops) if all_hops else 0.0
        metrics["avg_hops_survived"] = avg_hops
        records.append(metrics)
    return records

def main():
    print("==================================================")
    print("FAGS RECOVERY STABILIZATION SWEEPS")
    print("==================================================")

    num_nodes = 500
    query_count = 500
    seed = 42

    print(f"Generating Medium KG ({num_nodes} nodes) and {query_count} queries...")
    graph, queries = generate_dataset(num_nodes=num_nodes, num_queries=query_count, seed=seed)

    # Verifier with high noise (0.30) to trigger revivals and decision decay
    verifier = Verifier(noise_std=0.30, seed=seed)

    print("Running Baseline Search...")
    base_res = run_baseline_on_dataset(graph, queries, verifier)
    base_acc = np.mean([1 if r.success else 0 for r in base_res])
    print(f"  Baseline Accuracy: {base_acc:.2%}")

    # 1. K Sweep
    k_configs = [
        ("K=0", {"shield_depth": 0, "use_certificate": True, "certificate_bonus": 0.10}),
        ("K=1", {"shield_depth": 1, "use_certificate": True, "certificate_bonus": 0.10}),
        ("K=2", {"shield_depth": 2, "use_certificate": True, "certificate_bonus": 0.10}),
        ("K=3", {"shield_depth": 3, "use_certificate": True, "certificate_bonus": 0.10}),
        ("K=5", {"shield_depth": 5, "use_certificate": True, "certificate_bonus": 0.10}),
    ]
    print("\n--- Starting K Sweep ---")
    k_records = execute_batch(graph, queries, verifier, base_res, k_configs)
    export_sweep("k_sweep.csv", k_records, "K Sweep")
    plot_sweep("k_sweep.png", k_records, "Accuracy and Path Survival vs Shield Depth", "Shield Depth (K)")

    # 2. Bonus Sweep
    b_configs = [
        ("Bonus=0.00", {"shield_depth": 3, "use_certificate": True, "certificate_bonus": 0.00}),
        ("Bonus=0.05", {"shield_depth": 3, "use_certificate": True, "certificate_bonus": 0.05}),
        ("Bonus=0.10", {"shield_depth": 3, "use_certificate": True, "certificate_bonus": 0.10}),
        ("Bonus=0.15", {"shield_depth": 3, "use_certificate": True, "certificate_bonus": 0.15}),
        ("Bonus=0.20", {"shield_depth": 3, "use_certificate": True, "certificate_bonus": 0.20}),
    ]
    print("\n--- Starting Bonus Sweep ---")
    b_records = execute_batch(graph, queries, verifier, base_res, b_configs)
    export_sweep("bonus_sweep.csv", b_records, "Bonus Sweep")
    plot_sweep("bonus_sweep.png", b_records, "Accuracy and Path Survival vs Certificate Bonus", "Bonus")

    # 3. Ablation Study
    a_configs = [
        ("No Shield/Cert", {"shield_depth": 0, "use_certificate": False, "certificate_bonus": 0.00}),
        ("Shield Only",    {"shield_depth": 3, "use_certificate": False, "certificate_bonus": 0.00}),
        ("Cert Only",      {"shield_depth": 0, "use_certificate": True,  "certificate_bonus": 0.10}),
        ("Shield+Cert",    {"shield_depth": 3, "use_certificate": True,  "certificate_bonus": 0.10}),
    ]
    print("\n--- Starting Ablation Study ---")
    a_records = execute_batch(graph, queries, verifier, base_res, a_configs)
    export_sweep("ablation.csv", a_records, "Ablation Study")
    plot_sweep("ablation.png", a_records, "Ablation Study: Shield vs Certificate", "Configuration")

    print("\nAll experiments finished successfully.")

if __name__ == "__main__":
    main()
