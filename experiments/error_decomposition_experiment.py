"""Where does the error come from? Commit protocol + verifier ladder
(PROJECT_STATUS.md SS22).

Under the commit protocol (fags/committed.py), each query's error is split
into additive components by swapping in progressively stronger verifiers
while holding the search algorithm fixed:

  search loss          = acc(oracle plan, beam-64) - acc(oracle plan, algo)
  verifier-noise loss  = acc(oracle plan, algo)    - acc(hop-aware, algo)
  hop-blindness loss   = acc(hop-aware, algo)      - acc(hop-blind, algo)

"hop-blind" is the repo's ordinary rule-based Verifier (noise 0.08), which
scores every hop against the whole question's keyword bag. Also reported:
labelled-answer accuracy against its Bayes ceiling mean(1/|A(q)|), and
the legacy goal-test accuracy of the same beam for comparison.
"""

from __future__ import annotations

import csv
import os
import time

import numpy as np

from fags.graph_generator import generate_dataset
from fags.verifier import Verifier
from fags.beam_search import beam_search
from fags.committed import answer_set, beam_commit, OraclePlanVerifier, HopAwareVerifier
from fags.stats import cluster_bootstrap_ci, cluster_paired_test, holm

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "v2")
os.makedirs(RESULTS_DIR, exist_ok=True)

SIZES = {"Small": 20, "Medium": 100, "Large": 1000}
SEEDS = [101, 202, 303]
QUERIES = int(os.environ.get("DECOMP_QUERIES", 500))
WIDTHS = [1, 3, 8, 64]
VERIFIERS = {
    "hop-blind": lambda s: Verifier(noise_std=0.08, seed=s),
    "hop-aware": lambda s: HopAwareVerifier(noise_std=0.08, seed=s),
    "oracle-plan": lambda s: OraclePlanVerifier(seed=s),
}


def main():
    t0 = time.time()
    rows, tests, summary = [], [], []
    for size, n in SIZES.items():
        hits: dict[tuple, list] = {}
        labelled: dict[tuple, list] = {}
        calls: dict[tuple, list] = {}
        legacy, clusters, bayes, aset_sizes = [], [], [], []
        for seed in SEEDS:
            graph, queries = generate_dataset(n, QUERIES, seed=seed)
            sets = [answer_set(graph, q) for q in queries]
            clusters += [(seed, q.gold_path_id) for q in queries]
            bayes += [1 / len(a) for a in sets]
            aset_sizes += [len(a) for a in sets]
            lv = Verifier(noise_std=0.08, seed=seed)
            legacy += [int(beam_search(graph, q, lv, beam_width=8, max_depth=6).success) for q in queries]
            for vname, mk in VERIFIERS.items():
                for w in WIDTHS:
                    v = mk(seed)
                    for q, A in zip(queries, sets):
                        r = beam_commit(graph, q, v, w, q.hop_count)
                        hits.setdefault((vname, w), []).append(int(r.answer in A))
                        labelled.setdefault((vname, w), []).append(int(r.answer == q.answer_node))
                        calls.setdefault((vname, w), []).append(r.verifier_calls)
            print(f"[{time.time() - t0:6.0f}s] {size} seed {seed}", flush=True)

        for (vname, w), xs in hits.items():
            m, lo, hi = cluster_bootstrap_ci(xs, clusters)
            rows.append({"graph": size, "verifier": vname, "beam_width": w, "hits1_answer_set": m,
                         "ci_lo": lo, "ci_hi": hi, "labelled_acc": float(np.mean(labelled[(vname, w)])),
                         "mean_verifier_calls": float(np.mean(calls[(vname, w)]))})

        ceiling = hits[("oracle-plan", 64)]
        summary.append(f"-- {size} -- mean |A(q)|={np.mean(aset_sizes):.2f}, "
                       f"labelled-answer Bayes ceiling={100*np.mean(bayes):.1f}%, "
                       f"legacy goal-test beam-8 acc={100*np.mean(legacy):.1f}%")
        for w in WIDTHS:
            hb, ha, op = (np.mean(hits[(k, w)]) for k in ("hop-blind", "hop-aware", "oracle-plan"))
            summary.append(
                f"  width {w:>2}: hop-blind {100*hb:5.1f}% | losses -> search {100*(np.mean(ceiling)-op):5.1f}pp, "
                f"verifier-noise {100*(op-ha):5.1f}pp, hop-blindness {100*(ha-hb):5.1f}pp "
                f"(ceiling {100*np.mean(ceiling):.1f}%)  [labelled acc {100*np.mean(labelled[('hop-blind', w)]):.1f}%]")
            for a, b in (("hop-aware", "hop-blind"), ("oracle-plan", "hop-aware")):
                r = cluster_paired_test(hits[(b, w)], hits[(a, w)], clusters)
                tests.append({"graph": size, "width": w, "comparison": f"{a} - {b}", **r})

    for t, p in zip(tests, holm([t["p"] for t in tests])):
        t["p_holm"] = p

    with open(os.path.join(RESULTS_DIR, "error_decomposition.csv"), "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys())); wr.writeheader(); wr.writerows(rows)
    with open(os.path.join(RESULTS_DIR, "error_decomposition_tests.csv"), "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(tests[0].keys())); wr.writeheader(); wr.writerows(tests)
    with open(os.path.join(RESULTS_DIR, "error_decomposition_summary.txt"), "w", encoding="utf-8") as f:
        f.write("ERROR DECOMPOSITION UNDER THE COMMIT PROTOCOL (Hits@1 vs answer set)\n")
        f.write(f"{len(SEEDS)} seeds x {QUERIES} queries per size, fixed generator\n\n")
        f.write("\n".join(summary) + "\n\nTESTS (cluster sign-flip, Holm-adjusted)\n")
        for t in tests:
            f.write(f"  {t['graph']:<6} w={t['width']:<2} {t['comparison']:<26} diff={100*t['diff']:+6.1f}pp "
                    f"[{100*t['ci_lo']:+6.1f},{100*t['ci_hi']:+6.1f}] p_holm={t['p_holm']:.4f}\n")
        f.write(f"\nruntime {time.time() - t0:.0f}s\n")
    print(open(os.path.join(RESULTS_DIR, "error_decomposition_summary.txt"), encoding="utf-8").read())


if __name__ == "__main__":
    main()
