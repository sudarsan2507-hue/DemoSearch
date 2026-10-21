"""External validity: do the synthetic findings hold on MetaQA with a real
embedding verifier? (PROJECT_STATUS.md SS24)

For 2-hop and 3-hop MetaQA test questions (random sample), compares:
  rel-*       relation-level BGE verifier (hop-blind, like the synthetic one)
  path-*      path-level BGE verifier (scores the whole partial path)
  *-greedy / *-beam-w   search over relation paths, commit after k hops
  path-exhaustive       score every executable k-hop path (no search loss)
  oracle                best executable k-hop path by F1 (the ceiling for
                        any relation-path agent)
Metrics: expected Hits@1 (one predicted entity at random) and F1 against
the gold answer set; plus the legacy goal-test rate (did the search touch a
gold answer anywhere) to show how much that protocol inflates. Clustered on
the oracle relation path (MetaQA questions are template-generated per path
type), Holm-corrected.
"""

from __future__ import annotations

import csv
import os
import time

import numpy as np

from fags.metaqa import KB, Embedder, load_questions, beam_paths, all_k_paths, hits1, f1
from fags.stats import cluster_bootstrap_ci, cluster_paired_test, holm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "metaqa")
RESULTS_DIR = os.path.join(ROOT, "results", "v2")
os.makedirs(RESULTS_DIR, exist_ok=True)

N = int(os.environ.get("METAQA_N", 1000))
WIDTHS = [1, 3, 8, 18]


def main():
    t0 = time.time()
    kb = KB(os.path.join(DATA, "kb_kb.txt"))
    emb = Embedder(max_hops=3, cache_path=os.path.join(DATA, "path_embeddings_bge_small.npz"))
    print(f"[{time.time() - t0:5.0f}s] KB + path embeddings ready", flush=True)

    rows, tests, lines = [], [], []
    for hops in (2, 3):
        qs = load_questions(os.path.join(DATA, f"{hops}-hop_test.txt"), hops, N, seed=hops)
        q_embs = emb.encode_questions([q.text for q in qs])
        res: dict[str, dict[str, list]] = {}
        clusters = []

        def add(name, h, f, calls, touched):
            d = res.setdefault(name, {"hits": [], "f1": [], "calls": [], "touched": []})
            d["hits"].append(h); d["f1"].append(f); d["calls"].append(calls); d["touched"].append(touched)

        for qi, (q, qe) in enumerate(zip(qs, q_embs)):
            paths = all_k_paths(kb, q)
            if paths:
                oracle_path, oracle_set = max(paths, key=lambda pn: (f1(pn[1], q.answers), -len(pn[1])))
                ex_path, ex_set = max(paths, key=lambda pn: float(qe @ emb.path_emb[pn[0]]))
            else:
                oracle_path, oracle_set, ex_set = (), frozenset(), frozenset()
            clusters.append(oracle_path)
            add("oracle", hits1(oracle_set, q.answers), f1(oracle_set, q.answers), 0, True)
            add("path-exhaustive", hits1(ex_set, q.answers), f1(ex_set, q.answers), len(paths), True)
            for mode in ("rel", "path"):
                for w in WIDTHS:
                    o = beam_paths(kb, q, qe, emb, w, "relation" if mode == "rel" else "path")
                    name = f"{mode}-{'greedy' if w == 1 else f'beam-{w}'}"
                    add(name, hits1(o.answer, q.answers), f1(o.answer, q.answers), o.calls, int(o.touched_gold))
            if qi % 200 == 0:
                print(f"[{time.time() - t0:5.0f}s] {hops}-hop {qi}/{len(qs)}", flush=True)

        lines.append(f"-- {hops}-hop ({len(qs)} questions, {len(set(clusters))} oracle path types) --")
        for name, d in res.items():
            m, lo, hi = cluster_bootstrap_ci(d["hits"], clusters)
            rows.append({"hops": hops, "method": name, "hits1": m, "ci_lo": lo, "ci_hi": hi,
                         "f1": float(np.mean(d["f1"])), "mean_calls": float(np.mean(d["calls"])),
                         "goal_test_rate": float(np.mean(d["touched"]))})
            lines.append(f"  {name:<16} Hits@1={100*m:5.1f}% [{100*lo:5.1f},{100*hi:5.1f}]  F1={100*np.mean(d['f1']):5.1f}%  "
                         f"calls={np.mean(d['calls']):7.1f}  legacy-goal-test={100*np.mean(d['touched']):5.1f}%")
        comps = [(f"path-{'greedy' if w == 1 else f'beam-{w}'}", f"rel-{'greedy' if w == 1 else f'beam-{w}'}") for w in WIDTHS]
        comps += [("path-beam-3", "path-greedy"), ("path-beam-18", "path-beam-3"),
                  ("path-exhaustive", "path-beam-3"), ("oracle", "path-exhaustive")]
        for a, b in comps:
            r = cluster_paired_test(res[b]["hits"], res[a]["hits"], clusters)
            tests.append({"hops": hops, "comparison": f"{a} - {b}", **r})

    for t, p in zip(tests, holm([t["p"] for t in tests])):
        t["p_holm"] = p
    for name, data in (("metaqa_results.csv", rows), ("metaqa_tests.csv", tests)):
        with open(os.path.join(RESULTS_DIR, name), "w", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, fieldnames=list(data[0].keys())); wr.writeheader(); wr.writerows(data)
    with open(os.path.join(RESULTS_DIR, "metaqa_summary.txt"), "w", encoding="utf-8") as f:
        f.write("METAQA (vanilla, test split sample) - relation-path search with BGE verifiers\n\n")
        f.write("\n".join(lines) + "\n\nTESTS (Hits@1 diff, cluster sign-flip, Holm)\n")
        for t in tests:
            f.write(f"  {t['hops']}-hop {t['comparison']:<34} diff={100*t['diff']:+6.1f}pp "
                    f"[{100*t['ci_lo']:+6.1f},{100*t['ci_hi']:+6.1f}] p_holm={t['p_holm']:.4g}\n")
        f.write(f"\nruntime {time.time() - t0:.0f}s\n")
    print(open(os.path.join(RESULTS_DIR, "metaqa_summary.txt"), encoding="utf-8").read())


if __name__ == "__main__":
    main()
