"""Metrics evaluation, analysis, and statistical significance testing.

Computes metrics per configuration, including:
  • Accuracy (with 95% binomial confidence intervals)
  • Mean / std dev of nodes visited, search depth, and runtime
  • Recovery success rate
  • Additional search cost (relative change in nodes visited vs baseline)
  • Efficiency Ratio (accuracy gain vs additional search cost)
  • Gold Path Recovery Rate (our custom metric targeting pruned path revival)
  
Performs paired t-tests between search variants and baseline to verify
statistical significance of accuracy and node visit changes.
"""

from __future__ import annotations

import math
from typing import Sequence
import numpy as np
from scipy import stats

from fags import SearchResult


def evaluate_results(
    baseline_results: list[SearchResult],
    fags_results: list[SearchResult],
    label: str,
) -> dict:
    """Analyze search results for a FAGS configuration vs a baseline.

    Parameters
    ----------
    baseline_results : Result set from baseline search.
    fags_results : Result set from failure search under comparison.
    label : String description of FAGS config (e.g. 'Top-1').
    """
    n = len(baseline_results)
    assert len(fags_results) == n, "Result sets must be of identical size"

    # Extraction arrays
    b_accs = np.array([1 if r.success else 0 for r in baseline_results])
    f_accs = np.array([1 if r.success else 0 for r in fags_results])

    b_nodes = np.array([r.nodes_visited for r in baseline_results])
    f_nodes = np.array([r.nodes_visited for r in fags_results])

    b_depths = np.array([r.search_depth for r in baseline_results])
    f_depths = np.array([r.search_depth for r in fags_results])

    b_runtimes = np.array([r.runtime for r in baseline_results])
    f_runtimes = np.array([r.runtime for r in fags_results])

    # Core statistics
    acc_baseline = float(np.mean(b_accs))
    acc_fags = float(np.mean(f_accs))
    acc_gain = acc_fags - acc_baseline

    mean_nodes_b = float(np.mean(b_nodes))
    mean_nodes_f = float(np.mean(f_nodes))
    
    # Search cost & Efficiency Ratio
    additional_search_cost = (
        (mean_nodes_f - mean_nodes_b) / mean_nodes_b if mean_nodes_b > 0 else 0.0
    )
    efficiency_ratio = acc_gain / additional_search_cost if additional_search_cost > 0 else 0.0

    # 95% Confidence Intervals for Binomial Accuracy
    def binom_ci(acc, size):
        if size == 0:
            return 0.0, 0.0
        se = math.sqrt((acc * (1 - acc)) / size)
        margin = 1.96 * se
        return max(0.0, acc - margin), min(1.0, acc + margin)

    fags_ci_low, fags_ci_high = binom_ci(acc_fags, n)
    base_ci_low, base_ci_high = binom_ci(acc_baseline, n)

    # Paired t-tests
    t_stat_acc, p_val_acc = stats.ttest_rel(f_accs, b_accs)
    t_stat_cost, p_val_cost = stats.ttest_rel(f_nodes, b_nodes)

    # Recovery metrics
    total_recoveries_attempted = sum(r.backtracks for r in fags_results)
    queries_triggering_recovery = sum(1 for r in fags_results if r.backtracks > 0)
    
    recovered_correct_queries = sum(
        1 for b, f in zip(baseline_results, fags_results)
        if not b.success and f.success and f.backtracks > 0
    )
    recovery_success_rate = (
        recovered_correct_queries / queries_triggering_recovery
        if queries_triggering_recovery > 0 else 0.0
    )

    # Gold path recovery metrics
    # denominator: baseline search pruned gold path & failed
    pruned_and_failed_baseline = sum(
        1 for f in fags_results if f.gold_path_pruned
    )
    # numerator: among those pruned & failed in baseline, how many were successfully recovered by FAGS
    gold_path_recovered = sum(
        1 for f in fags_results if f.gold_path_pruned and f.success and f.gold_path_recovered
    )
    gold_path_recovery_rate = (
        gold_path_recovered / pruned_and_failed_baseline
        if pruned_and_failed_baseline > 0 else 0.0
    )

    return {
        "label": label,
        "n": n,
        "accuracy_baseline": acc_baseline,
        "accuracy_baseline_ci": (base_ci_low, base_ci_high),
        "accuracy_fags": acc_fags,
        "accuracy_fags_ci": (fags_ci_low, fags_ci_high),
        "accuracy_gain": acc_gain,
        "mean_nodes_baseline": mean_nodes_b,
        "mean_nodes_fags": mean_nodes_f,
        "additional_search_cost": additional_search_cost,
        "efficiency_ratio": efficiency_ratio,
        "p_value_accuracy": p_val_acc,
        "p_value_cost": p_val_cost,
        "queries_triggering_recovery": queries_triggering_recovery,
        "total_recoveries_attempted": total_recoveries_attempted,
        "recovery_success_rate": recovery_success_rate,
        "gold_path_recovery_rate": gold_path_recovery_rate,
        "gold_pruned_count": pruned_and_failed_baseline,
        "gold_recovered_count": gold_path_recovered,
        "mean_depth_baseline": float(np.mean(b_depths)),
        "mean_depth_fags": float(np.mean(f_depths)),
        "mean_runtime_baseline": float(np.mean(b_runtimes)),
        "mean_runtime_fags": float(np.mean(f_runtimes)),
    }
