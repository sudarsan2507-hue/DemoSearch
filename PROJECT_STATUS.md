# DemoSearch / FAGS — Project Status

_Analysis compiled from code (`fags/`, top-level scripts), `results/*.csv`, and `results/*.png`. Initial pass 2026-06-22; updated 2026-06-23 with the budget-matched control experiment (§6-7) and again after regenerating all canonical results under the fixed deterministic generator (§4 numbers, §8)._

## 1. What this project is

A research codebase testing one hypothesis:

> **Can a graph-search agent that "remembers" the paths it wrongly rejected, and revives the best one on dead-end/failure, beat a plain greedy graph search — without degenerating into brute-force exploration?**

The setup is a synthetic multi-hop knowledge-graph QA task (`fags/graph_generator.py`): each graph has gold answer paths (2–5 hops), distractor edges that look plausible at branch points, dead ends, and contradiction nodes. A **Verifier** scores candidate relations against a question; search picks the best-scoring edge at each hop.

Two search algorithms are compared on the same graphs/queries:
- **Baseline search** (`fags/baseline_search.py`) — greedy, no memory, stops on first failure.
- **FAGS — Failure-Aware Graph Search** (`fags/failure_search.py`) — same greedy search, but when it hits a dead end / contradiction / misalignment it can backtrack locally, and otherwise pops the best previously-rejected branch from a **failure memory** and resumes from there.

## 2. Core architecture (`fags/` package)

| Module | Role |
|---|---|
| `__init__.py` | Shared types: `Node`, `Edge`, `KnowledgeGraph`, `Query`, `SearchResult`, `MemoryEntry`, `FailureType` (dead_end / path_misalignment / contradiction / budget_exhausted) |
| `graph_generator.py` | Builds synthetic KGs + queries with gold paths, distractors, dead ends, contradictions, ambiguous branch points |
| `verifier.py` | Scores candidate relations against a query. Three implementations exist (see §3) |
| `memory.py` | Failure memory strategies: **Top‑1** (keep best reject per branch), **Top‑2**, **Threshold** (keep rejects within a margin of the winner). Max-heap retrieval by score |
| `baseline_search.py` | Control condition |
| `failure_search.py` | FAGS itself: local backtracking → memory revival → optional dynamic re-verification → optional "certificate" scoring bonus → optional RBSC/RTC variants (see §3) |
| `evaluation.py` | Accuracy w/ 95% CI, nodes visited, "additional search cost" vs baseline, **Efficiency Ratio** (accuracy gain ÷ cost), **Gold Path Recovery Rate**, paired t-tests |

## 3. Evolution of the verifier and FAGS mechanisms

The project iterated through several rounds of patches (`patch_*.py` — these are one-shot scripts that string-replaced code into `fags/failure_search.py` / `fags/verifier.py`, now superseded by the current source):

1. **Rule-based `Verifier`** — weighted keyword overlap (0.50) + relation relevance (0.30) + path coherence (0.20), plus configurable Gaussian noise to simulate an imperfect verifier.
2. **`EmbeddingVerifier`** — real semantic scoring via `sentence-transformers`, model `BAAI/bge-small-en-v1.5` (BGE), run offline (`HF_HUB_OFFLINE=1`).
3. **`HybridVerifier`** — combines the BGE embedding verifier with the rule-based one.
4. FAGS itself grew extra knobs on top of plain memory-revival:
   - **Dynamic re-verification** — re-score alternatives at a revived branch knowing which relation already failed there.
   - **Shield (`shield_depth`)** — after a revival, suppress further backtracking for N hops so the search doesn't immediately re-thrash.
   - **Certificate bonus (`use_certificate`, `certificate_bonus`)** — once revived onto a relation, give a coherence-based score bonus to semantically-related relations for the next hops.
   - **RBSC (Reason-Based Stabilization Certificate, `rbsc_mode`: none/linear/nonlinear)** — scales the certificate bonus down by how marginal the revival was (a barely-better revival gets less of a confidence boost).
   - **RTC-lite (`use_rtc_lite`)** — a lighter-weight variant of the same idea, tested in `rtc_lite_experiment.py`.

Top-level scripts (`bge_scale_experiment.py`, `bge_vs_minilm_experiment.py`, `hybrid_sweep_experiment.py`, `rbsc_experiment.py`, `shield_experiment.py`, `stabilization_sweep.py`, `rtc_lite_experiment.py`, the `verifier_*` and `debug_hops.py` scripts) are each a standalone exploration of one of these knobs against the rule-based or embedding verifiers. They are not wired into `main.py`'s pipeline — they were run ad hoc to probe specific questions.

## 4. What `main.py` actually produces (the canonical experiment)

`main.py` runs the full matrix: 3 graph sizes (Small=20 / Medium=100 / Large=1000 nodes) × 1000 queries × 4 memory configs (Top-1, Top-2, Threshold t=0.10, Threshold t=0.20), plus ablations on the Medium graph (re-verification on/off, max-backtracks ∈ {1,2,3,5}). Outputs land in `results/`.

### Headline result (`results/summary.txt`)

> **NO** — the hypothesis is not validated with significant efficiency. Accuracy gains are real and statistically significant, but trivial relative to the search-cost explosion.

| Graph | Best config | Baseline Acc | FAGS Acc | Gain | Extra nodes visited | Efficiency Ratio |
|---|---|---|---|---|---|---|
| Medium | Threshold (t=0.20) | 4.30% | 10.80% | **+6.50%** | **+941%** | 0.007 |

- p-value for the accuracy gain is tiny (5.6e‑12) — the gain is real, not noise.
- **Gold Path Recovery Rate is ~0%** — FAGS almost never actually finds its way back onto the true gold path; the accuracy gain comes mostly from get­ting *lucky* on alternate routes, at the cost of 8–17× more nodes visited (`results/accuracy_and_cost_table.csv`, `results/accuracy_vs_cost.png`, `results/scalability.png`).
- Ablations (`results/ablation_table.csv`): turning off dynamic re-verification *increases* the accuracy gain (+7.00%) but balloons cost to +1717%. Capping backtracks (1/2/5) barely changes the picture — FAGS is consistently expensive no matter how it's throttled.

### Verifier-quality sweep (`results/verifier_sweep_summary.txt`, `verifier_quality_sweep.csv`, `verifier_sweep_table.csv`, `results/sweep_efficiency.png`)

- FAGS's efficiency is **non-monotonic** in verifier quality: it peaks around **60–80% verifier accuracy** (Efficiency Ratio ~0.10–0.14) and *drops* as the verifier gets better (95% verifier accuracy → ratio ~0.10, lower than the 70% case despite higher raw accuracy gain).
- Below ~50% verifier accuracy, FAGS collapses into near-exhaustive search (constant backtracking).
- Conclusion in the file: FAGS efficiency "is generally low" across the sweep — it's never clearly worth its cost in this graph topology.

### Shield / Certificate ablation (`results/shield_experiment_summary.txt`, `shield_experiment_table.csv`, `results/shield_accuracy.png`, `results/shield_hops_survived.png`, `k_sweep.csv`, `bonus_sweep.csv`)

- Shield and Certificate mechanisms were added to try to make revival "stick" (survive more hops post-revival) instead of immediately re-failing.
- Result: differences between No-Shield / Shield-only / Cert-only / Shield+Cert are small and noisy (accuracy gains 3.8–5.8%, recovery rate 0–0.45%, efficiency ratio 0.004–0.006). **Cert Only** nudges accuracy gain highest (+5.80%) with the best hops-survived (1.88); the others are clustered close together with no clear winner.
- `k_sweep.csv` (shield depth K=0..5) and `bonus_sweep.csv` (certificate bonus 0.00–0.20) both show flat, noisy curves with no clear optimum — these knobs aren't moving the needle.

### Structural diagnostic (`results/gold_rank_histogram.png`)

- Across Small/Medium/Large graphs, the verifier ranks the **correct (gold) relation #1 about 58–60%** of the time, #2 about 18%, #3 about 9%, and 4th-or-worse about 13%. This is consistent across graph sizes — the verifier's discriminative power, not graph size, is the bottleneck. It explains why FAGS's "remember and revive" approach has a ceiling: when the gold relation is buried at rank 4+, simply trying the 2nd-best (Top‑1 memory) isn't enough to recover it.

## 5. Net takeaway (pre-control-experiment)

- FAGS reliably produces a **statistically significant but small accuracy gain** (~4–12 points absolute) over greedy baseline, on every graph size and verifier quality tested.
- That gain consistently costs **roughly 8–17× more node visits** (up to 1700%+ in some ablations), and the custom "Gold Path Recovery Rate" metric stays near 0%, meaning the gain isn't really "intelligent recovery onto the right path" — it looks more like FAGS doing a wider, costlier search that occasionally stumbles onto a correct alternate route.
- Add-on mechanisms tried so far to fix this (dynamic re-verification, shield depth, certificate bonus, RBSC linear/nonlinear, RTC-lite, embedding-based verifiers BGE/MiniLM, hybrid verifier) have **not yet found a configuration where Efficiency Ratio is convincingly good** — best observed ratios are ~0.10–0.14 in a narrow 60–80% verifier-accuracy band, otherwise ~0.003–0.03.
- Current written conclusion (`results/summary.txt`, `verifier_sweep_summary.txt`): the core hypothesis is **not validated** as efficient; FAGS recovers some accuracy but does so by approaching brute-force exploration rather than targeted recovery.

All of the above compares FAGS (8–17× the node budget) against a single-shot 1× baseline — not an apples-to-apples comparison. §7 closes that gap.

## 6. Follow-up: Budget-Matched Random-Restart Control (`budget_matched_control_experiment.py`)

**Question:** if a dumb baseline gets the *same* node-visit budget FAGS actually spends per query — via random restarts, since `Verifier`'s Gaussian noise makes each `baseline_search` call stochastic — does FAGS's targeted failure-memory revival still beat it? (Previously the comparison let FAGS spend 8–17× more compute for free.)

**Method:** for every query, run FAGS once, record its `nodes_visited`, then re-run plain greedy `baseline_search` repeatedly (each call independently noisy) until the cumulative node count matches FAGS's spend for that exact query, OR-ing success across restarts. Same 3 graph sizes / 1000 queries / seed=101 as the canonical run; FAGS uses the repo's headline-best config (Threshold Memory, t=0.10).

**Result** (`results/budget_matched_control.csv`, `.png`, `_summary.txt`):

| Graph | Baseline (1×) | Random-Restart Baseline (budget-matched) | FAGS | FAGS − RRB | p-value | Verdict |
|---|---|---|---|---|---|---|
| Small | 4.40% | 8.90% | 16.20% | **+7.30%** | 2.1e-09 | FAGS wins (significant) |
| Medium | 4.30% | 8.40% | 8.70% | +0.30% | 0.78 | no significant difference |
| Large | 10.30% | 15.40% | 13.80% | −1.60% | 0.20 | no significant difference |

**Reading:** FAGS only clears the budget-matched dumb control on the Small graph. On Medium it's a statistical tie; on Large the dumb random-restart control scores numerically *higher* (not significant). So most of FAGS's previously-reported gains over the 1× baseline are explained by **spending more search budget**, not by intelligently navigating back to the gold path — this lines up with the ~0% Gold Path Recovery Rate seen everywhere else in this repo (§4). The control experiment is the more direct answer to the original research question than any of the FAGS-side knob-tuning (shield/certificate/RBSC/RTC-lite) done previously.

**Side finding — reproducibility bug fixed:** while building this control, re-running the *same* script with the *same* `seed=` gave different numbers each process run. Root cause: `fags/graph_generator.py` built `_CONFUSABLE_MAP` by iterating the module-level `CONFUSABLE_PAIRS` **set**, whose iteration order depends on Python's per-process string-hash randomization — so distractor-edge generation (and every downstream number in this repo) was never actually deterministic across processes despite the explicit `seed=` parameters everywhere. Fixed by sorting the iteration (`for _a, _b in sorted(CONFUSABLE_PAIRS):`); verified byte-identical CSVs across repeated runs and across different `PYTHONHASHSEED` values after the fix. This doesn't change any of the *qualitative* conclusions already drawn (both pre- and post-fix runs told the same story), but every existing CSV in `results/` was generated under an unrecorded, effectively-random hash seed — treat exact decimal values as having ±1 percentage point of run-to-run noise; the qualitative pattern (small but real gain, huge cost, ~0% gold recovery) is stable.

## 7. Net takeaway (updated)

- The original FAGS-vs-1×-baseline comparison (§4) overstates FAGS: once a dumb baseline gets the same node-visit budget (§6), FAGS only wins decisively on the Small graph, is a statistical tie on Medium, and loses (not significantly) on Large.
- Combined with the ~0% Gold Path Recovery Rate seen across every experiment, the evidence now points to **FAGS's accuracy gains being mostly an artifact of spending 8–17× more search budget**, not of the failure-memory mechanism doing intelligent targeted recovery.
- None of the add-on knobs tried (dynamic re-verification, shield depth, certificate bonus, RBSC, RTC-lite, better embedding verifiers) changed this picture — they tune the cost/accuracy tradeoff slightly but don't address the underlying issue.
- **Revised recommendation:** the verifier's discriminative power (§4, gold-rank histogram: only ~58–60% rank-1 accuracy) is the real ceiling. Further work should target improving the verifier signal itself, or accept that FAGS-style failure memory is not worth its search-cost overhead in this graph topology.

## 8. State of the repo / housekeeping notes

- `patch_*.py` at the project root are one-off code-mutation scripts (string find/replace against `fags/failure_search.py` and `fags/verifier.py`) used during development to add features (certificate params, RBSC, RTC-lite, verifier descriptions). They already did their job — the resulting code is in `fags/`. They're historical, not part of the run pipeline.
- Many `verifier_*.py` and `*_experiment.py` / `*_sweep.py` scripts at the root are one-off probes, not integrated into `main.py`; each hardcodes its own small experiment matrix.
- `scratch/` holds ad hoc audit/diagnostic scripts (`post_revival_audit.py`, `rank_diagnostic.py`, `recovery_audit.py`, `verifier_sweep.py` duplicate).
- The repo is now under git (see `COMMITS.md` for the commit-by-commit breakdown), pushed to `https://github.com/sudarsan2507-hue/DemoSearch.git`.
- All numbers in this doc as of 2026-06-23 reflect a full regeneration of `main.py`, `shield_experiment.py`, `stabilization_sweep.py`, `verifier_sweep.py`, and `scratch/rank_diagnostic.py` under the fixed deterministic generator (§6). `verifier_sweep.py` also had a latent, unrelated bug fixed (`ControlledVerifier.score()` wasn't extracting `query.keywords`, so it crashed against the current `fags/verifier.py`).
