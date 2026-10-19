"""Second KGQA dataset: WebQSP with real BGE verifiers (PROJECT_STATUS.md SS32).

Re-tests SS22/SS24 on a harder, real-question dataset where the hop count is
not given (the agent picks a 1- or 2-hop path itself):
  W1  search saturates: beam width beyond ~3 adds little (beam-32 is the
      near-exhaustive reference; exhaustive path-level scoring is too
      expensive here - thousands of 2-hop paths per question).
  W2  hop-aware (path-level) scoring beats hop-blind (relation-level).
  W3  the verifier, not search, is the ceiling: large oracle gap.
  W4  the legacy goal-test protocol inflates accuracy.
Metrics: expected Hits@1 and F1 vs the gold answer set. Paired tests
clustered on the oracle relation path, Holm.
"""

from __future__ import annotations

import csv
import os
import time

import numpy as np

from fags.webqsp import load_questions, TextEmbedder, beam_paths, all_paths
from fags.metaqa import hits1, f1
from fags.stats import cluster_bootstrap_ci, cluster_paired_test, holm

ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(ROOT, "results", "v2")
N = int(os.environ.get("WEBQSP_N", 500))
WIDTHS = [1, 3, 8, 32]


def name(mode, w):
    return f"{mode}-{'greedy' if w == 1 else f'beam-{w}'}"


def main():
    t0 = time.time()
    qs = load_questions(os.path.join(ROOT, "data", "webqsp"), N, seed=7)
    emb = TextEmbedder()
    print(f"[{time.time() - t0:5.0f}s] {len(qs)} questions loaded", flush=True)
    res: dict[str, dict[str, list]] = {}
    clusters = []

    def add(n, h, f, calls, touched):
        d = res.setdefault(n, {"hits": [], "f1": [], "calls": [], "touched": []})
        d["hits"].append(h); d["f1"].append(f); d["calls"].append(calls); d["touched"].append(int(touched))

    for i, q in enumerate(qs):
        qe = emb.query(q.text)
        paths = all_paths(q)
        if paths:
            op, oset = max(paths, key=lambda pn: (f1(pn[1], q.answers), -len(pn[1])))
        else:
            op, oset = (), frozenset()
        clusters.append(op)
        add("oracle", hits1(oset, q.answers), f1(oset, q.answers), 0, True)
        for mode in ("rel", "path"):
            for w in WIDTHS:
                o = beam_paths(q, qe, emb, w, "relation" if mode == "rel" else "path")
                add(name(mode, w), hits1(o.answer, q.answers), f1(o.answer, q.answers), o.calls, o.touched_gold)
        if i % 50 == 0:
            print(f"[{time.time() - t0:5.0f}s] {i + 1}/{len(qs)} (embedding cache {len(emb.cache)})", flush=True)

    rows, tests, lines = [], [], [f"WEBQSP ({len(qs)} questions, {len(set(clusters))} oracle path types)"]
    for n, d in res.items():
        m, lo, hi = cluster_bootstrap_ci(d["hits"], clusters)
        rows.append({"method": n, "hits1": m, "ci_lo": lo, "ci_hi": hi, "f1": float(np.mean(d["f1"])),
                     "mean_calls": float(np.mean(d["calls"])), "goal_test_rate": float(np.mean(d["touched"]))})
        lines.append(f"  {n:<14} Hits@1={100 * m:5.1f}% [{100 * lo:5.1f},{100 * hi:5.1f}]  F1={100 * np.mean(d['f1']):5.1f}%  "
                     f"calls={np.mean(d['calls']):8.1f}  legacy-goal-test={100 * np.mean(d['touched']):5.1f}%")
    comps = [(name("path", w), name("rel", w), "W2") for w in WIDTHS]
    comps += [(name("path", 3), name("path", 1), "W1"), (name("path", 8), name("path", 3), "W1"),
              (name("path", 32), name("path", 8), "W1"), ("oracle", name("path", 32), "W3")]
    for a, b, h in comps:
        r = cluster_paired_test(res[b]["hits"], res[a]["hits"], clusters)
        tests.append({"hyp": h, "comparison": f"{a} - {b}", **r})
    for t, p in zip(tests, holm([t["p"] for t in tests])):
        t["p_holm"] = p
    for fname, data in (("webqsp_results.csv", rows), ("webqsp_tests.csv", tests)):
        with open(os.path.join(RESULTS_DIR, fname), "w", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, fieldnames=list(data[0].keys())); wr.writeheader(); wr.writerows(data)
    with open(os.path.join(RESULTS_DIR, "webqsp_summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n\nTESTS (Hits@1 diff, cluster sign-flip, Holm)\n")
        for t in tests:
            f.write(f"  {t['hyp']} {t['comparison']:<30} diff={100 * t['diff']:+6.1f}pp "
                    f"[{100 * t['ci_lo']:+6.1f},{100 * t['ci_hi']:+6.1f}] p_holm={t['p_holm']:.4g}\n")
        f.write(f"\nruntime {time.time() - t0:.0f}s\n")
    print(open(os.path.join(RESULTS_DIR, "webqsp_summary.txt"), encoding="utf-8").read())


if __name__ == "__main__":
    main()
