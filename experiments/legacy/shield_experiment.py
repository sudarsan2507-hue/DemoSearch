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
        res = baseline_search(
            graph=graph,
            query=q,
            verifier=verifier,
        )
        results.append(res)
    return results

def main():
    print("==================================================")
    print("FAGS RECOVERY SHIELD EXPERIMENT")
    print("==================================================")

    # Use a medium graph
    num_nodes = 500
    query_count = 500
    seed = 42

    print(f"Generating Medium KG ({num_nodes} nodes) and {query_count} queries...")
    graph, queries = generate_dataset(num_nodes=num_nodes, num_queries=query_count, seed=seed)

    # Verifier with high noise to trigger frequent memory revivals
    verifier = Verifier(noise_std=0.30, seed=seed)

    print("Running Baseline Search...")
    base_res = run_baseline_on_dataset(graph, queries, verifier)
    base_acc = np.mean([1 if r.success else 0 for r in base_res])
    print(f"  Baseline Accuracy: {base_acc:.2%}")

    configs = [
        ("FAGS (No Shield)", 0),
        ("FAGS (Shield K=1)", 1),
        ("FAGS (Shield K=2)", 2),
        ("FAGS (Shield K=3)", 3),
    ]

    eval_records = []

    for label, k in configs:
        print(f"\nRunning {label}...")
        res = run_experiment_on_dataset(
            graph=graph, queries=queries, verifier=verifier, strategy="top1", shield_depth=k
        )
        
        acc = np.mean([1 if r.success else 0 for r in res])
        print(f"  Accuracy: {acc:.2%}")
        
        metrics = evaluate_results(base_res, res, label)
        
        # Calculate Average Hops Survived Post Revival
        # SearchResult has hops_survived_post_revival: list[int]
        all_hops = []
        for r in res:
            all_hops.extend(r.hops_survived_post_revival)
            
        avg_hops = np.mean(all_hops) if all_hops else 0.0
        metrics["avg_hops_survived"] = avg_hops
        metrics["shield_k"] = k
        
        eval_records.append(metrics)

    # ──────────────────────────────────────────────
    # Output Metric Table (CSV)
    # ──────────────────────────────────────────────
    
    csv_eval_path = os.path.join(RESULTS_DIR, "shield_experiment_table.csv")
    with open(csv_eval_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Configuration", "Baseline Accuracy", "FAGS Accuracy", 
            "Accuracy Gain", "Additional Search Cost", "Efficiency Ratio", 
            "Gold Path Recovery Rate", "Avg Hops Survived Post Revival"
        ])
        for r in eval_records:
            writer.writerow([
                r["label"], f"{r['accuracy_baseline']:.2%}", f"{r['accuracy_fags']:.2%}",
                f"{r['accuracy_gain']:.2%}", f"{r['additional_search_cost']:.2%}", 
                f"{r['efficiency_ratio']:.3f}", f"{r['gold_path_recovery_rate']:.2%}",
                f"{r['avg_hops_survived']:.2f}"
            ])
            
    print(f"\nTable successfully written to {csv_eval_path}")

    # ──────────────────────────────────────────────
    # Output Visualisation Plots
    # ──────────────────────────────────────────────
    
    labels = [r["label"] for r in eval_records]
    fags_accs = [r["accuracy_fags"] * 100 for r in eval_records]
    recovery_rates = [r["gold_path_recovery_rate"] * 100 for r in eval_records]
    avg_hops = [r["avg_hops_survived"] for r in eval_records]

    # Plot 1: Accuracies and Recovery Rate
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()
    
    x = np.arange(len(labels))
    width = 0.35
    
    ax1.bar(x - width/2, fags_accs, width, color="crimson", label="FAGS Accuracy")
    ax2.bar(x + width/2, recovery_rates, width, color="teal", label="Gold Recovery Rate")
    
    ax1.set_ylabel("Accuracy (%)", color="crimson")
    ax2.set_ylabel("Gold Path Recovery Rate (%)", color="teal")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.tick_params(axis='y', labelcolor="crimson")
    ax2.tick_params(axis='y', labelcolor="teal")
    
    plt.title("Search Accuracy and Recovery Rate vs Shield Depth")
    fig.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "shield_accuracy.png"), dpi=150)
    plt.close()

    # Plot 2: Average Hops Survived
    plt.figure(figsize=(8, 5))
    plt.plot(labels, avg_hops, marker='o', color="darkorange", linestyle="-", linewidth=2, markersize=8)
    plt.ylabel("Average Hops Survived Post Revival")
    plt.title("Path Survival Duration vs Shield Depth")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "shield_hops_survived.png"), dpi=150)
    plt.close()

    print("Plots generated and saved successfully!")

    # ──────────────────────────────────────────────
    # Output Written Research Report Summary
    # ──────────────────────────────────────────────
    
    report_path = os.path.join(RESULTS_DIR, "shield_experiment_summary.txt")
    with open(report_path, "w") as rf:
        rf.write("==================================================\n")
        rf.write("FAGS RECOVERY SHIELD EXPERIMENT SUMMARY\n")
        rf.write("==================================================\n\n")
        
        for r in eval_records:
            rf.write(f"Configuration: {r['label']}\n")
            rf.write(f"  Accuracy Gain: +{r['accuracy_gain']:.2%}\n")
            rf.write(f"  Recovery Rate: {r['gold_path_recovery_rate']:.2%}\n")
            rf.write(f"  Efficiency Ratio: {r['efficiency_ratio']:.3f}\n")
            rf.write(f"  Avg Hops Survived Post Revival: {r['avg_hops_survived']:.2f}\n\n")

    print(f"Report written to {report_path}")
    print("All tasks finished successfully.")

if __name__ == "__main__":
    main()
