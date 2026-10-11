"""Cluster-aware statistics for paired accuracy comparisons.

Queries generated from the same gold path (and evaluated on the same graph)
are not independent, so per-query paired t-tests overstate significance -
badly on the Small graph, where 1000 queries cycle over only 40 gold paths.
Everything here resamples / sign-flips whole clusters instead of queries.
"""

from __future__ import annotations

from typing import Hashable, Sequence

import numpy as np


def _cluster_sums(values: np.ndarray, clusters: Sequence[Hashable]) -> np.ndarray:
    index: dict[Hashable, int] = {}
    ids = np.fromiter((index.setdefault(c, len(index)) for c in clusters), dtype=np.int64, count=len(values))
    return np.bincount(ids, weights=values, minlength=len(index)), np.bincount(ids, minlength=len(index))


def cluster_bootstrap_ci(
    x: Sequence[float],
    clusters: Sequence[Hashable],
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Mean of x with a cluster-bootstrap (1 - alpha) CI. Returns (mean, lo, hi)."""
    x = np.asarray(x, dtype=float)
    sums, counts = _cluster_sums(x, clusters)
    rng = np.random.default_rng(seed)
    k = len(sums)
    draws = rng.integers(0, k, size=(n_boot, k))
    boot = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    return float(x.mean()), float(np.quantile(boot, alpha / 2)), float(np.quantile(boot, 1 - alpha / 2))


def cluster_paired_test(
    a: Sequence[float],
    b: Sequence[float],
    clusters: Sequence[Hashable],
    n_perm: int = 10000,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict:
    """Paired comparison of b vs a (per-query outcomes on the same queries).

    Returns the mean difference (b - a), a cluster-bootstrap 95% CI for it,
    and a two-sided p-value from a cluster-level sign-flip permutation test
    (under H0 the sign of each cluster's summed difference is exchangeable).
    """
    d = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
    sums, counts = _cluster_sums(d, clusters)
    n = counts.sum()
    observed = sums.sum() / n

    rng = np.random.default_rng(seed)
    signs = rng.choice((-1.0, 1.0), size=(n_perm, len(sums)))
    perm = (signs * sums).sum(axis=1) / n
    p = (np.sum(np.abs(perm) >= abs(observed) - 1e-12) + 1) / (n_perm + 1)
    # A permutation p-value cannot go below 1/(n_perm+1), which Holm then
    # multiplies by the number of tests - with hundreds of tests nothing can
    # survive. When the permutation p is at its resolution floor, use the
    # normal approximation to the same sign-flip statistic instead (its
    # variance under H0 is sum of squared cluster sums; accurate with many
    # clusters).
    if p <= 10 / (n_perm + 1):
        var = float(np.sum(sums ** 2))
        if var > 0:
            from scipy.stats import norm
            p = float(2 * norm.sf(abs(sums.sum()) / np.sqrt(var)))

    k = len(sums)
    draws = rng.integers(0, k, size=(n_boot, k))
    boot = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    return {
        "diff": float(observed),
        "ci_lo": float(np.quantile(boot, 0.025)),
        "ci_hi": float(np.quantile(boot, 0.975)),
        "p": float(p),
        "n_clusters": int(k),
        "n": int(n),
    }


def holm(pvalues: Sequence[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values (same order as input)."""
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj.tolist()
