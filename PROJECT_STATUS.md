# DemoSearch / FAGS — Project Status

_Analysis compiled from code (`fags/`, top-level scripts), `results/*.csv`, and `results/*.png`. Initial pass 2026-06-22; updated 2026-06-23 with the budget-matched control experiment (§6), the regenerated canonical results (§4 numbers, §20), the HybridVerifier follow-up control (§7), the learned failure-pattern-graph experiment (§8), the diversity-aware memory experiment (§9), the headline beam search result (§10), beam search's own verifier-quality stress test (§11), and the diverse-beam-pruning experiment (§12); updated again 2026-06-27 with the beam search + Failure Pattern Graph composition (§13), the top-K ranking-rule variants (§14), and global best-first search as an alternative pruning paradigm (§15); updated again 2026-06-28 with MCTS as a second alternative paradigm (§16), MCTS's crossover at large budgets (§17), and a beam-seeded MCTS hybrid built to close the small-budget gap (§18)._

> **⚠ 2026-10-11 correction — read §21–25 first.** Every number in §4–18 was produced by a generator bug that made every 3+ hop gold path contradict itself (the gold path was unreachable in 76–98% of queries), under a protocol that hands the search a perfect goal test, with inconsistent cost accounting across algorithms and per-query tests on pseudo-replicated queries. §21 re-validates the core claims on the fixed generator (several reverse), §22 replaces the goal-test protocol with commit-to-an-answer and decomposes the error, §23 tests *when* learning from failure can help at all (with a theoretical result), and §24 checks external validity on MetaQA. §4–18 are kept unchanged below as the historical record; reproduce them with `generate_dataset(..., legacy_evidence_bug=True)`.

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

All of the above compares FAGS (8–17× the node budget) against a single-shot 1× baseline — not an apples-to-apples comparison. §6 closes that gap.

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

## 7. Does a stronger verifier rescue FAGS? (`budget_matched_control_embedding_experiment.py`)

**Question:** §6 found FAGS only beats the budget-matched control on the Small graph with the weak rule-based `Verifier`. Is that because the verifier signal is too noisy for targeted memory to exploit? Re-ran the identical budget-matched control with `HybridVerifier` (rule-based + `BAAI/bge-small-en-v1.5` embeddings, alpha=0.5) on a 500-node graph / 500 queries (seed=42, matching the scale of the repo's other embedding experiments since real model inference is much slower than the synthetic scorer).

**Result** (`results/budget_matched_control_embedding.csv`, `.png`, `_summary.txt`):

| Verifier | Baseline (1×) | Random-Restart Baseline (budget-matched) | FAGS | FAGS − RRB | p-value |
|---|---|---|---|---|---|
| HybridVerifier (rule+BGE) | 3.60% | **14.80%** | 6.80% | **−8.00%** | 5.7e-05 |

**Reading:** the opposite of "rescued" — with a *stronger* verifier the budget-matched dumb control beats FAGS by a wide, statistically significant margin (14.80% vs 6.80%). So the verifier's discriminative power was not the bottleneck holding FAGS back; the failure-memory/revival mechanism itself doesn't reliably turn a search budget into accuracy as well as plain randomized retries do. A plausible reason: FAGS's memory revival is *targeted* — it keeps retrying specific previously-rejected branches — whereas random restarts explore a more diverse set of paths per unit of budget, and diversity seems to matter more than targeting here.

## 8. Does a learned cross-query "failure pattern graph" help? (`failure_pattern_graph_experiment.py`)

**User's proposal:** instead of (or alongside) FAGS's reactive within-query memory, learn a cross-query failure-pattern signal — which relation transitions tend to precede a dead-end/contradiction/misalignment — from past searches, and use it to steer *away* from those transitions before they're attempted (avoidance), not just recover from them after the fact.

**Implementation** (`fags/failure_pattern_graph.py`):
- `FailurePatternGraph`: Beta-smoothed failure rate per `(prev_relation, relation)` transition bigram.
- `train_failure_pattern_graph(...)`: runs plain greedy `baseline_search` over a training set; for each finished path, blames the *last* transition before a DEAD_END/CONTRADICTION/PATH_MISALIGNMENT as the "mistake," and treats every other transition (including all of a successful path's) as neutral/good.
- `PatternAwareVerifier`: wraps any verifier, subtracting `penalty_weight × learned_failure_rate(prev_rel, candidate_rel)` from its score — composable with both `baseline_search` (→ "Pattern-Aware Greedy", same cost as baseline) and `failure_search` (→ "FAGS+FPG").
- Required adding a `path_relations` field to `SearchResult` (additive, all four return sites in `baseline_search.py`/`failure_search.py` updated) so a finished path's relation sequence is recoverable for training.

**Method:** trained the FPG on one graph (100 nodes, 1000 queries, seed=101), evaluated on a **different, held-out** graph (seed=202) — genuine transfer test, not memorization. Swept `penalty_weight` ∈ {0.0, 0.1, 0.2, 0.3, 0.5}.

**Result** (`results/failure_pattern_graph_table.csv`, `.png`, `_summary.txt`, `_patterns.txt`):
- Training signal was real and substantial: 2,176 transition observations across 425 distinct `(prev_rel, rel)` pairs, 951 attributed failure-edges. Top learned failure-prone patterns are dominated by **same-relation-twice repeats** (e.g. `CURRENT_PM→CURRENT_PM` at 92% failure rate, `WROTE→WROTE` at 83%) — almost none of these are in the hand-authored `CONFUSABLE_PAIRS`, and most have the *default* `RELATION_COHERENCE` value, meaning the FPG learned a genuinely new signal, not a rediscovery of existing hand-coded knowledge.
- **Pattern-Aware Greedy vs Baseline** (same 1× search cost): no penalty weight beats baseline with significance (best: +0.10%, p=0.88). The new signal exists but doesn't translate into better single-shot greedy decisions.
- **FAGS+FPG vs plain FAGS**: monotonically *worse* as penalty weight increases — significantly worse at penalty=0.2 (−2.30%, p=0.030), 0.3 (−2.40%, p=0.026), and 0.5 (−2.90%, p=0.0063). Penalizing "risky-looking" transitions apparently also suppresses some of the legitimate alternatives FAGS's memory needs for successful revival.

**Reading:** the idea was implemented properly (real signal, proper train/test split, not a sparsity artifact) and **didn't pan out** — worse, it actively conflicts with FAGS's existing recovery mechanism. The same-relation-repeat pattern is a real structural insight about this graph generator's distractor placement, but turning it into a flat score penalty removes useful options more often than it removes bad ones.

## 9. Diversity-aware memory: does avoiding confusable reverts close the gap? (`diversity_memory_experiment.py`)

**Hypothesis (from §6's reading):** FAGS's memory always revives the highest-scoring rejected candidate, but distractors are deliberately confusable with the gold relation — so "second-best" is usually just another guess from the same confusable cluster as the winner, not a genuinely different hypothesis. Added `DiversityMemory` to `fags/memory.py`: it revives the highest-scoring reject that is **not** the same relation as, confusable with, or highly-coherent with (≥0.5) the winning relation, falling back to the best score only if every reject is too similar. This needed an additive `winner_relation` parameter threaded through `FailureMemory.store()` (all 4 strategies updated; `failure_search.py` now passes it).

**Verified the mechanism actually fires:** instrumented a run over 1000 Medium-graph queries — a genuinely diverse alternative was available and chosen in 27,748 of 28,001 `store()` calls (99.1%); only 253 times did every reject fall in the winner's cluster, forcing a same-cluster fallback. This is not a mechanism that rarely triggers.

**Result** (`results/diversity_memory_table.csv`, `.png`, `_summary.txt`), same 3 sizes / 1000 queries / seed=101 as §6's control:

| Graph | Baseline | FAGS-Top1 | FAGS-Diversity | RRB (matched to Diversity) | Diversity vs Top1 | Diversity vs RRB |
|---|---|---|---|---|---|---|
| Small | 5.00% | 16.90% | 16.40% | 8.70% | −0.50% (p=0.61) | **+7.70%** (p=7.3e-10) |
| Medium | 3.60% | 7.60% | 8.90% | 8.80% | +1.30% (p=0.18) | +0.10% (p=0.93) |
| Large | 9.10% | 13.80% | 12.00% | 14.80% | −1.80% (p=0.07) | **−2.80%** (p=0.020) |

**Reading:** despite firing on ~99% of opportunities, DiversityMemory is statistically indistinguishable from plain Top1Memory on every graph size (p > 0.05 throughout) — picking a *structurally different* relation doesn't make it more likely to be the *correct* one; it just substitutes one kind of guess for another. Against the budget-matched control specifically: the Small-graph win merely matches what plain FAGS already achieved there (§6); Medium remains the same statistical tie as before; and Large gets *worse* — what was a non-significant loss for plain FAGS becomes a significant one for DiversityMemory. The targeted-revival design's weakness isn't *which* specific candidate it revives — it's that revival itself, however chosen, isn't a reliable source of accuracy at this verifier quality.

## 10. Beam search: the structurally different algorithm that actually works (`fags/beam_search.py`, `beam_search_experiment.py`)

**The pivot:** every mechanism tested in §3-9 sits on the same architecture — walk one path greedily, detect failure, pick one rejected candidate to revive, repeat. Five independent experiments showed that architecture isn't reliably better than spending the same search budget on diversified random restarts, no matter which candidate gets revived or how that choice is informed. `fags/beam_search.py` abandons the architecture itself: it keeps the **K best live hypotheses concurrently** at every hop (classic beam search — expand all live hypotheses, score every resulting candidate with the same `Verifier`, keep the global top-K), so a wrong early guess never needs to be *detected* and *recovered from* — the better alternative was already being explored the whole time. A candidate transition into evidence-contradicting territory is dropped per-hypothesis (same hard constraint as `FailureType.CONTRADICTION`, just scoped to one hypothesis instead of ending the whole search); the single-path `PATH_MISALIGNMENT` early-exit isn't needed since a wandering hypothesis just loses the next prune.

**Method:** swept beam width K ∈ {1, 2, 3, 5, 8} across all 3 graph sizes (1000 queries, seed=101 — same setup as every other comparison in this doc), against plain Baseline, plain FAGS-Top1, and a budget-matched random-restart control (RRB) sized to each beam run's actual node cost per query.

**Result** (`results/beam_search_table.csv`, `.png`, `_summary.txt`) — beam search doesn't just beat the random-restart control, it **strictly dominates FAGS-Top1 in the cost/accuracy tradeoff**:

| Graph | Beam config | Beam Acc | Beam Nodes | FAGS-Top1 Acc | FAGS-Top1 Nodes |
|---|---|---|---|---|---|
| Medium | width=3 | **22.70%** | 15.05 | 8.90% | 28.45 |
| Medium | width=8 | **37.40%** | 27.85 | 8.90% | 28.45 |
| Large | width=2 | **18.80%** | 11.37 | 13.80% | 34.68 |
| Large | width=5 | **24.10%** | 25.28 | 13.80% | 34.68 |

Beam width=3 on Medium gets 2.5× FAGS's accuracy at **half** FAGS's search cost. Beam width=2 on Large beats FAGS's accuracy at **a third** of FAGS's cost. Against the budget-matched random-restart control specifically, beam search won with statistical significance in **14 of 15** (graph size × beam width) configurations — the lone exception (Large, width=1) was a non-significant tie.

**Reading:** the problem was never which candidate FAGS revives — it was the commit-then-recover architecture itself. Holding multiple real, score-derived hypotheses concurrently (correlated diversity) beats both targeted single-path revival *and* uncorrelated random-restart diversity, decisively and consistently. This is the first mechanism tried in this entire investigation that clears the budget-matched bar on more than one graph size.

## 11. Does a stronger verifier help beam search the way it hurt FAGS? (`beam_search_embedding_experiment.py`)

**Question:** §7 found a stronger verifier (HybridVerifier) made FAGS's loss to the budget-matched control *worse* (14.80% vs 6.80%, p=5.7e-05) — the bottleneck was architectural, not verifier quality. Beam search's mechanism is different: it depends on the verifier's ranking quality to avoid pruning the gold hypothesis out of the top-K. Prediction: a stronger verifier should help, not hurt. Repeated `beam_search_experiment.py`'s comparison with `HybridVerifier` (rule+BGE) on a 500-node graph / 500 queries (seed=42), sweeping width ∈ {1, 2, 3, 5}.

**Result** (`results/beam_search_embedding_table.csv`, `.png`, `_summary.txt`):

| Width | Beam Acc | Beam Nodes | RRB Acc (matched) | vs RRB | FAGS-Top1 (ref) |
|---|---|---|---|---|---|
| 1 | 3.80% | 6.7 | 5.20% | −1.40% (p=0.25) | 7.00% @ 48.1 nodes |
| 2 | 7.00% | 12.4 | 7.60% | −0.60% (p=0.71) | |
| 3 | 8.60% | 17.8 | 9.20% | −0.60% (p=0.73) | dominates FAGS (higher acc, 1/3 the cost) |
| 5 | **14.40%** | 27.7 | 10.40% | **+4.00%** (p=0.033) | dominates FAGS (2× acc, ~half the cost) |

**Reading:** partial confirmation. At low widths beam search is a statistical *tie* with the budget-matched control (never a significant loss, unlike FAGS, which lost decisively under the same stronger verifier) — and at width=5 it pulls clearly ahead. Across every width except w=1, beam search beats plain FAGS-Top1 outright at a fraction of FAGS's cost (48.1 nodes). So the prediction holds directionally: a stronger verifier doesn't reverse beam search's advantage the way it reversed FAGS's — it just shifts where the advantage shows up, requiring more width before it clearly separates from the dumb control. Beam search's advantage scales with verifier quality; FAGS's commit-then-recover design actively gets worse with it.

## 12. Diverse beam pruning: can the beam itself be improved? (`max_children_per_parent` in `fags/beam_search.py`, `diverse_beam_search_experiment.py`)

**Hypothesis:** plain beam search's top-K pruning is purely score-based, so one strong-but-wrong early branch can supply most/all of the next beam, crowding out hypotheses from weaker-scoring (but possibly correct) parents. Added `max_children_per_parent` to `beam_search()`: caps how many slots a single parent can fill, forcing a spread across distinct lineages. The cap is relaxed automatically when there are fewer live parents than needed to fill the beam otherwise — without this, a single start node would permanently cap the beam at size 1 from the very first hop (caught via smoke-testing before the full run).

**Method:** cost-neutral comparison (same beam width, so ~same search cost) — capped vs plain/uncapped beam search, at widths {5, 8} × caps {1, 2, 3}, across all 3 graph sizes (1000 queries, seed=101).

**Result** (`results/diverse_beam_search_table.csv`, `.png`, `_summary.txt`) — the aggregate tally (5 significant wins / 4 significant losses out of 18 configs) is misleading; it hides a clean **size-dependent split**:

| Graph | Capping vs plain beam (same cost) |
|---|---|
| Small | **Helps** — 3/6 configs significant wins, 0 losses |
| Medium | **Helps** — 2/6 configs significant wins, 0 losses |
| Large | **Hurts** — 0/6 wins, 4/6 significant losses (up to −5.40%, p=4.6e-07) |

**Reading:** Small graphs have few genuinely good candidates per branch to begin with, so capping barely removes anything valuable while still spreading slots across more lineages — a cheap win. Large graphs have many more genuinely good candidates per branch; forcing a spread throws away legitimately strong options just because they happen to share a parent, which costs more than the diversity buys. This is the *same failure mode* `DiversityMemory` hit for FAGS (§9) — forcing structural diversity doesn't reliably correlate with correctness — just resurfacing in a different mechanism, and specifically on the graph size where it matters most. **Net: diversity-capping is not a safe default.** It should be tuned per graph scale (or just left off) rather than applied uniformly; plain/uncapped beam search remains the better choice at the Large-graph regime where beam search's win over FAGS was largest.

## 13. Composing beam search with the learned Failure Pattern Graph (`beam_search_fpg_experiment.py`)

**Hypothesis:** §8 found composing the FPG penalty with FAGS actively hurt it — suppressing "risky-looking" transitions also suppressed alternatives FAGS's reactive memory needed for revival. Beam search has no such revival step (multiple hypotheses are already explored in parallel), so the same learned signal might help here instead of conflicting with the mechanism. Reused `PatternAwareVerifier` (§8) unchanged, wrapping the verifier `beam_search()` consumes — no new core code needed, since both already share the same `.score()` interface.

**Method:** trained the FPG on one graph (seed=101), evaluated on a different held-out graph (seed=202) — same train/test discipline as §8. Cost-neutral comparison (same beam width) at widths {3, 5, 8} × penalty weights {0.05, 0.1, 0.15, 0.2, 0.3} — including the weights that significantly hurt FAGS, to see whether beam search tolerates them.

**Result** (`results/beam_search_fpg_table.csv`, `.png`, `_summary.txt`): a clean null across all 15 configurations — **0 significant wins, 0 significant losses**. Accuracy wobbles within noise at every width/penalty combination (e.g. width=8 ranges 43.0%–45.5% across penalty 0–0.3, none of it significant).

**Reading:** the hypothesis is half-confirmed. Beam search never gets *hurt* by the FPG penalty the way FAGS did — even at penalty=0.3, where FAGS lost −2.90% (p=0.0063), beam search shows no significant change at any width. That's consistent with the "no revival step to conflict with" theory. But it also doesn't *help*: the learned avoidance signal doesn't add anything once candidates are already competing across multiple live hypotheses and pruned by raw verifier score — beam search's own selection pressure is apparently already capturing what the FPG signal would add. Net: composing FPG with beam search is safe but pointless here, a genuinely different (neutral) outcome from the same composition with FAGS (actively harmful) or DiversityMemory (size-dependent).

## 14. Changing the top-K ranking rule itself (`score_aggregation`, `diversity_penalty_weight` in `fags/beam_search.py`, `beam_search_topk_variants_experiment.py`)

**Hypothesis:** §12 and §13 only constrained or nudged the *existing* ranking rule (cumulative mean per-hop score) — they never changed the rule itself. Added two genuinely different rules to `beam_search()`: `score_aggregation="sum"` (rank by total accumulated score, the convention classic NLP beam search uses, instead of the mean) and `diversity_penalty_weight` (a **soft** version of §12's hard `max_children_per_parent` cap — greedy iterative selection where a crowded parent's later candidates sink gradually instead of being banned outright once a quota is hit).

**Method:** cost-neutral comparison (same beam width) vs the established plain-beam baseline (mean, no penalty), at widths {5, 8} × diversity penalties {0, 0.05, 0.1, 0.2} × aggregation {mean, sum} — same 3 graph sizes / 1000 queries / seed=101.

**Result** (`results/beam_search_topk_variants_table.csv`, `.png`, `_summary.txt`) — reported per-axis, since a pooled win/loss tally would hide which idea (if any) actually did something:

| Axis | Significant wins | Significant losses |
|---|---|---|
| Sum aggregation alone (the genuinely new idea) | 0/6 | 0/6 |
| Soft diversity penalty (mean aggregation) | 3/18 | 2/18 |
| Sum + soft diversity combined | 0/18 | 0/18 |

**Reading:** sum aggregation — the one idea here that wasn't already tested in some form — is a **clean null**: ranking by total accumulated score instead of average doesn't measurably change accuracy anywhere. The soft diversity penalty's 3 wins / 2 losses **just replicate §12's hard-cap pattern** (helps Small/Medium, hurts Large) with a softer mechanism — not a new finding, the same failure mode resurfacing. Combining sum with the diversity penalty washes the effect out entirely. This is now the fourth and fifth independent refinement attempt (after diversity-capping and FPG composition) that fails to find a configuration reliably beating plain beam search's simplest rule — increasingly strong evidence the simple version (mean aggregation, plain top-K, no diversity adjustment) is at or near a local optimum for this verifier's signal, not that the right tweak hasn't been found yet.

## 15. Global best-first search: a different paradigm, decisively worse (`fags/best_first_search.py`, `best_first_search_experiment.py`)

**Hypothesis:** all four refinements in §12-14 changed a *rule* inside beam search's fixed-width-per-hop pruning. The remaining open angle was changing the *paradigm*: `fags/best_first_search.py` keeps one global priority queue across all depths and always expands whichever hypothesis has the best cumulative score next, until a total node-visit budget is spent — rather than keeping exactly K hypotheses alive at every depth uniformly. In principle this should let the search concentrate compute on whatever looks most promising instead of spending it evenly.

**Method:** per-query budget-matched comparison (same discipline as §6): for each query, beam search runs first at a given width, and best-first search then gets that query's *exact* `nodes_visited` as its budget — so any accuracy difference is attributable purely to the pruning paradigm, not search cost. Widths {2, 3, 5, 8} × all 3 graph sizes × 1000 queries × seed=101.

**Result** (`results/best_first_search_table.csv`, `.png`, `_summary.txt`): the most decisive result of the whole investigation, in the *negative* direction — **beam search wins all 12/12 configurations**, with extreme significance (p as low as 1e-57). The gap is large and gets worse with more budget, not better:

| Graph | Width | Beam Acc | Best-First Acc |
|---|---|---|---|
| Medium | 8 | 37.30% | 15.80% |
| Large | 2 | 17.20% | **0.90%** |
| Large | 8 | 26.10% | 3.10% |

**Reading:** this is not a marginal loss like the diversity/FPG experiments — best-first search is *catastrophically* worse, especially on Large (near-0% at low budgets). The mechanism is clear in hindsight: with this verifier's signal (~58-60% rank-1 accuracy, distractors deliberately confusable with the gold relation), greedily expanding only the single best-looking frontier node lets the search tunnel deep into one wrong branch that happens to look locally strong at every step, burning the *entire* budget there before the queue is ever forced to surface an alternative. Beam search's fixed width isn't just "a budget knob" — it's a structural guarantee that several distinct initial branches get explored *no matter how good any one of them looks*, which turns out to be the actual source of its robustness on a graph this noisy and confusable. Concentrating budget on the best-looking option, rather than guaranteeing breadth, is actively dangerous here.

## 16. MCTS: fixes best-first's collapse, still falls short of beam search (`fags/mcts_search.py`, `mcts_search_experiment.py`)

**Hypothesis:** §15's global best-first search failed because pure greedy frontier expansion has no pressure to ever reconsider a branch once it looks best — it tunnels in and burns the whole budget there. Monte Carlo Tree Search is built specifically to prevent that: UCB1 selection explicitly balances exploiting what looks good against exploring under-visited branches, with many independent rollouts feeding value estimates the tree actually trusts. `fags/mcts_search.py` implements one simulation as selection (UCB1 descent) → expansion (add the best-scoring untried child) → rollout (greedy-by-score playout, not added to the tree) → backpropagation, budget-capped by total distinct nodes visited (tree + rollout combined) for direct comparability.

**Method:** same per-query budget-matched discipline as §15 — MCTS gets each query's *exact* `nodes_visited` from beam search at a given width as its `node_budget`. Widths {2, 3, 5, 8} × all 3 graph sizes × 1000 queries × seed=101, with §15's best-first numbers shown alongside for reference.

**Result** (`results/mcts_search_table.csv`, `.png`, `_summary.txt`): **MCTS loses to beam search in 10/12 configurations** (the other 2 are non-significant ties, p=0.084 and p=0.063, both numerically favoring beam) — but it sits clearly *between* best-first and beam search, not anywhere near best-first's collapse:

| Graph | Width | Beam | MCTS | Best-First (§15, reference) |
|---|---|---|---|---|
| Medium | 8 | 37.00% | 28.30% | 13.60% |
| Large | 2 | 17.20% | 13.00% | 1.00% |
| Large | 8 | 27.40% | 17.40% | 4.10% |

**Reading:** the hypothesis about *why* best-first failed is confirmed — adding explicit exploration pressure via UCB1 completely fixes the tunnel-vision collapse (MCTS beats best-first by 10-15x on Large at low budgets, instead of being stuck near 0%). But at these small node budgets (5-40 nodes) it still falls short of beam search. §17 tests whether that's fundamental or just the wrong budget regime for MCTS.

## 17. Does MCTS catch up at much larger budgets? (`mcts_large_budget_experiment.py`)

**Hypothesis:** §16 noted MCTS typically needs far more than single-digit-to-low-double-digit simulations to pay off — the 5-40 node range tested might simply be too small a budget regime for it, not a fundamental loss. Swept both algorithms independently across a much wider, shared cost range (beam widths up to 60, MCTS budgets up to 1280) on Medium and Large graphs (500 queries, seed=101; Small skipped as too tiny for "large" budgets to mean anything), plotting accuracy vs mean nodes visited for both.

**A measurement pitfall caught before reporting:** the first pass swept beam only up to width=60 (~186 mean nodes on Large), while MCTS at budget=1280 reached ~372 mean nodes — more than double. Comparing MCTS's high-budget point against beam's *capped* maximum would have overstated the gap (an early read showed +56% on Large, which was exactly that artifact). Beam search was re-run at widths up to 350 to reach a genuinely comparable ~374-node cost before drawing any conclusion.

**Result, with the fair comparison** (`results/mcts_large_budget_table.csv`, `.png`, `_summary.txt`):

| Graph | Mean Nodes | MCTS Acc | Beam Acc | Gap |
|---|---|---|---|---|
| Medium | ~48 (near graph exhaustion) | 76.00% | 70.00% | **+6.00%** |
| Large | ~213 | 52.00% | 52.60% | −0.60% (tied) |
| Large | ~315 | 78.20% | 73.20% | **+5.00%** |
| Large | ~373 | 98.20% | 82.80% | **+15.40%** |

**Reading:** the standing theory holds, with a corrected magnitude. MCTS genuinely catches up to and then pulls ahead of beam search — but only once given a budget roughly **5-10× larger** than anything tested in §16 (200+ nodes, vs. the 5-40 node range where FAGS-style mechanisms were compared). Below that crossover, beam search wins; above it, MCTS's exploration/exploitation balance starts to compound and it pulls steadily further ahead as budget grows (the gap nearly triples from +5% to +15% between 315 and 373 nodes on Large, suggesting it would keep widening past the highest budget tested). This nuances §16's "undefeated" framing: beam search's fixed-width design isn't beaten by anything at the cost scale that's actually comparable to FAGS's revival mechanism, but it is **not the best algorithm at every budget** — MCTS is the better choice once you're willing to spend several hundred nodes per query rather than a few dozen.

## 18. Beam-Seeded MCTS: a true hybrid, built to close the small-budget gap (`fags/beam_seeded_mcts_search.py`, `beam_seeded_mcts_experiment.py`)

**Hypothesis:** §17 showed MCTS only beats beam search once given 5-10× more budget — likely because its early simulations are spent rediscovering the diversity beam search gets for free. `beam_seeded_mcts_search()` tries to remove that cost directly: spend the first `seed_depth` hops running plain beam-search-style expansion (global top-K pruning, building real parent/child links on the tree, sealing each round's losers since beam's pruning decision is meant to be final), producing up to `beam_width` diverse tips — then hand the *remaining* budget to standard MCTS (UCB1 selection, rollout, backpropagation) starting from those tips instead of a blank root.

**A real bug caught before the experiment even ran:** the first version capped seeding by *node count* (a fraction of the total budget). Since `max_depth` is a hard hop limit every algorithm in this codebase shares, a generous node budget let seeding run all the way to `max_depth` on its own — leaving the MCTS phase with *zero hops of room* to refine anything. Measured directly: at a representative budget, **67% of non-immediately-successful seedings ended with every tip already at max_depth**, silently wasting most of the remaining budget (confirmed by a smoke test where accuracy at budget=80 was *worse* than at budget=40 — a clear signal something was broken, since more budget should never hurt). Fixed by capping the seeding phase by **depth** (`seed_depth`, default 2) instead of node count, guaranteeing `max_depth - seed_depth` hops always remain for MCTS regardless of how much budget is available.

**Method (post-fix):** same per-query budget-matched discipline as §16 — beam search runs first at a given width, and both pure MCTS and the hybrid get that query's *exact* `nodes_visited` as their budget. Widths {2, 3, 5, 8, 15, 25, 40} (the range where pure MCTS lost outright) × all 3 graph sizes × 500 queries × seed=101. Hybrid config: `seed_depth=2`, `beam_width=5`.

**Result** (`results/beam_seeded_mcts_table.csv`, `.png`, `_summary.txt`): **the hybrid never beats plain beam search — 0 wins out of 21 (graph size × width) configurations, 18 significant losses, 3 ties.** Against pure MCTS specifically it's a wash leaning slightly negative: 4 wins, 7 losses, 10 ties — at the smallest budgets (width 2-3) the hybrid is actually *worse* than pure MCTS (e.g. Large width=2: Hybrid 4.60% vs MCTS 12.00%, p=2.5e-06), likely because the fixed `beam_width=5` seeding phase consumes a disproportionate share of a tiny budget before MCTS gets anything to work with; at moderate widths (Medium width=15, Large width=8/25/40) it sometimes edges past pure MCTS but never past beam search.

**Reading:** the depth-based fix was necessary and correct (it's a real, generally-applicable bug for anyone composing a depth-capped search with a budget-capped one), but fixing it only made the hybrid *function correctly* — it didn't make it *win*. Beam-seeding does not shift MCTS's effective crossover point lower; the diversity beam search "gives away for free" isn't actually what was costing pure MCTS at these budgets, or whatever it does cost is recovered elsewhere in a way this composition doesn't capture. This is the **eighth** independent composition/improvement attempt across the whole project (FPG+FAGS harmful, diversity-memory null, hard diversity-cap mixed, FPG+beam null, sum-aggregation null, soft diversity-penalty replicates the hard-cap pattern, best-first catastrophic, and now this) — and not one has beaten the relevant baseline in its budget regime. That consistency is itself informative: at small-to-moderate budgets, plain beam search isn't just "the best option found so far," it's resisted every structurally-different attempt to improve on it.

## 19. Net takeaway (updated)

- The original FAGS-vs-1×-baseline comparison (§4) overstates FAGS: once a dumb baseline gets the same node-visit budget (§6), FAGS only wins decisively on the Small graph, is a statistical tie on Medium, and loses (not significantly) on Large — and loses *significantly* once the verifier is upgraded (§7).
- Combined with the ~0% Gold Path Recovery Rate seen across every experiment, the evidence points to **FAGS's accuracy gains being mostly an artifact of spending 8–17× more search budget**, not of the failure-memory mechanism doing intelligent targeted recovery — and a better verifier makes this worse for FAGS, not better.
- None of the add-on knobs or mechanism redesigns tried on top of FAGS's architecture (dynamic re-verification, shield depth, certificate bonus, RBSC, RTC-lite, better embedding verifiers, a learned cross-query failure-pattern penalty, diversity-aware revival) changed this picture.
- **The underlying research question has a clear positive answer once the architecture changes:** beam search (§10) strictly dominates FAGS in the cost/accuracy tradeoff across both graph sizes that matter (Medium, Large), beats the budget-matched random-restart control on 14/15 configurations with the weak verifier, and — unlike FAGS — never loses to that control under a stronger verifier either (§11), pulling clearly ahead again at wider beams.
- **At the small-to-moderate budget scale comparable to FAGS's revival mechanism (5-40 nodes), seven independent refinement/replacement attempts all failed to beat plain beam search**: hard diversity-capping, FPG composition, sum aggregation, and soft diversity penalty (§12-14, rule-level tweaks); global best-first search (§15, catastrophic paradigm replacement); MCTS (§16); and beam-seeded MCTS (§18, a true hybrid built specifically to close this gap, which did not).
- **But beam search is not the best choice at every budget** (§17): MCTS catches up to and then decisively passes beam search once given roughly 5-10× more budget (200+ nodes per query) — the gap reaches +15% in MCTS's favor at ~373 nodes and was still widening at the highest budget tested. Algorithm choice should depend on the budget regime: beam search for cheap/moderate search, MCTS if willing to spend substantially more per query.
- **Revised recommendation:** stop iterating on FAGS's commit-then-recover design — seven independent experiments now confirm it's not salvageable at any budget scale tested. For the search algorithm itself: use plain beam search at low-to-moderate budgets (where it remains undefeated by eight separate attempts including a purpose-built hybrid); switch to MCTS if the budget can be pushed to several hundred nodes per query. The search-algorithm side of this investigation is fairly exhausted at this point; the highest-value remaining direction is the verifier signal itself (the gold-rank ceiling from §4 was never re-measured against beam search or MCTS specifically).

## 20. State of the repo / housekeeping notes

- `patch_*.py` at the project root are one-off code-mutation scripts (string find/replace against `fags/failure_search.py` and `fags/verifier.py`) used during development to add features (certificate params, RBSC, RTC-lite, verifier descriptions). They already did their job — the resulting code is in `fags/`. They're historical, not part of the run pipeline.
- Many `verifier_*.py` and `*_experiment.py` / `*_sweep.py` scripts at the root are one-off probes, not integrated into `main.py`; each hardcodes its own small experiment matrix.
- `scratch/` holds ad hoc audit/diagnostic scripts (`post_revival_audit.py`, `rank_diagnostic.py`, `recovery_audit.py`, `verifier_sweep.py` duplicate).
- The repo is now under git (see `COMMITS.md` for the commit-by-commit breakdown), pushed to `https://github.com/sudarsan2507-hue/DemoSearch.git`.
- All numbers in this doc as of 2026-06-23 reflect a full regeneration of `main.py`, `shield_experiment.py`, `stabilization_sweep.py`, `verifier_sweep.py`, and `scratch/rank_diagnostic.py` under the fixed deterministic generator (§6). `verifier_sweep.py` also had a latent, unrelated bug fixed (`ControlledVerifier.score()` wasn't extracting `query.keywords`, so it crashed against the current `fags/verifier.py`).

## 21. Re-validation on the fixed generator (`revalidation_experiment.py`, `results/v2/revalidation_*`)

**The bug.** `_create_gold_paths` wrote evidence `path_{start} = step_j` on every intermediate gold node — one key, a different value per step — so walking any 3+ hop gold path tripped the contradiction check that baseline, FAGS, beam, best-first and MCTS all enforce. By hop count: 2 hops 9% blocked, 3/4/5 hops **100%** blocked. A verifier that picks the exact gold edge every time scored 2.5% / 11.5% / 24.3% (Small/Medium/Large). Every success in §4–18 was a detour through background edges, which is why Gold Path Recovery was ~0% everywhere. Fixed by giving each step its own key (`gold_{path_id}_step_{j}`); the old behaviour is kept behind `legacy_evidence_bug=True` and reproduces the old numbers (e.g. Medium baseline 3.6%, FAGS 10.4%, Beam-5 30.4% vs §4/§10's 4.3% / 10.8% / ~30%).

**Second problem: cost accounting.** `beam_search` scores every neighbour of every live hypothesis but counts only the K nodes it keeps; `best_first_search` counts every scored child. Matching the two on `nodes_visited` (as §15–16 did) gave best-first ~20× fewer verifier calls (Large, width 8: 32 vs 629). `best_first_search` / `mcts_search` now accept `score_budget` (verifier calls), and every matched comparison is run both ways.

**Method.** Same goal-test protocol as §4–18 (so differences are attributable to the fix), 3 graph seeds × 500 queries per size, CIs and paired tests clustered on (seed, gold path) via `fags/stats.py` (cluster sign-flip permutation, cluster bootstrap), Holm over all 51 tests.

| Claim (section) | Fixed generator | Verdict |
|---|---|---|
| FAGS ≈ budget-matched restarts (§6) | FAGS − RRB(calls-matched): **+34.3 / +22.3 / +3.2pp** (Small/Med/Large), all Holm-sig. Node-matched: +25.2 / +6.3 / −8.6pp | **Reversed**: FAGS beats restarts per verifier call on every size |
| Beam strictly dominates FAGS (§10) | Beam-3 vs FAGS: −1.1 / −1.5 / +13.9pp at ~1.7× FAGS's calls (153 vs 91 Small) | **Weakened**: beam wins only with more calls, except on Large |
| Best-first is catastrophic (§15) | Node-matched: Beam-8 +46–55pp. Call-matched: **best-first wins by 2–51pp** | **Accounting artifact** — but confounded: best-first goal-tests every *scored* child, beam only kept ones (see §22) |
| MCTS < beam at small budgets (§16) | Beam ahead by 10–30pp under both accountings | **Holds** |
| Accuracies | Small baseline 3.8% → 43.4%; Beam-8 Small 97.8% | Old absolute numbers meaningless |

The best-first reversal is itself the lesson: under the goal-test protocol the *ranking* of algorithms depends on bookkeeping (what counts as cost, when the goal test fires). That protocol cannot rank search algorithms reliably, which motivates §22.

## 22. Commit-to-an-answer protocol and error decomposition (`fags/committed.py`, `error_decomposition_experiment.py`, `results/v2/error_decomposition_*`)

**Protocol.** The agent is told the hop count H (as MetaQA systems are), runs to exactly H hops with no goal test, and commits to one endpoint. Scored by Hits@1 against the answer set A(q) = every node reachable by following the gold relation sequence (the labelled `answer_node` is one member; mean |A(q)| = 1.9–2.5, so labelled-answer accuracy is capped at a Bayes ceiling of 60–67% for any agent that sees only the question).

**Decomposition.** Hold the search fixed, swap verifiers: oracle-plan (knows each hop's gold relation) → hop-aware (rule verifier matching only hop t's keywords: an oracle *decomposition*, keyword dropout and noise kept) → hop-blind (the repo's `Verifier`, whole keyword bag every hop). 3 seeds × 500 queries × 3 sizes, Holm over 24 tests.

| Graph | Width | Hop-blind Hits@1 | Search loss | Verifier-noise loss | **Hop-blindness loss** | Legacy goal-test (beam-8) |
|---|---|---|---|---|---|---|
| Small | 1 / 3 / 8 / 64 | 32.5 / 39.6 / 39.5 / 40.0% | 30.8 / 1.6 / 0 / 0pp | 1.4 / 5.8 / 4.8 / 4.4pp | 35.3 / 53.0 / 55.7 / 55.6pp | 97.7% |
| Medium | 1 / 3 / 8 / 64 | 22.0 / 28.5 / 27.8 / 28.5% | 34.4 / 2.5 / 0 / 0pp | 7.1 / 6.9 / 7.2 / 7.9pp | 36.5 / 62.1 / 65.0 / 63.5pp | 70.5% |
| Large | 1 / 3 / 8 / 64 | 20.9 / 26.2 / 26.7 / 26.9% | 33.5 / 1.9 / 0 / 0pp | 6.7 / 8.1 / 6.4 / 7.1pp | 38.9 / 63.8 / 66.9 / 66.1pp | 50.9% |

Every hop-aware vs hop-blind difference is Holm-significant (+35 to +67pp). Readings:
1. **Search is solved at width 3.** Under an honest protocol, widening the beam past 3 changes nothing; the whole FAGS/beam/best-first/MCTS contest of §4–18 was being fought over a quantity (search loss ≤2.5pp at w≥3) that barely matters.
2. **The dominant error is hop-blindness** (53–67pp): the verifier scores each hop against the whole question, so it cannot tell which hop it is on. Raw verifier noise costs only 4–8pp — which is why upgrading the verifier's *scores* (BGE, hybrid, §7/§11) never moved anything.
3. **The goal-test protocol inflates accuracy 1.9–2.5×** (Small 97.7% vs 39.5% for the same beam), because touching the answer node anywhere is far easier than committing to it.

Caveat: the hop-aware verifier uses an oracle decomposition of the question. §24 tests a realistic, non-oracle hop-aware verifier on real data.

## 23. When can learning from failure help? (`fags/delayed_failure.py`, `failure_informativeness_experiment.py`, `results/v2/failure_informativeness*`)

**Theory (the result that reframes the project).** Model the search as inference over which root-to-leaf path is the gold one, with a factorized prior P(child correct | parent on gold path) = q(child), Σ q = 1 over siblings (e.g. softmax of verifier scores). Let D be the set of nodes observed dead (subtree contains no goal). *Proposition:* for any unexpanded node n whose ancestors are alive, P(n ∈ π | D) = P(n ∈ π) / P(π ∩ D = ∅). *Proof:* nodes on a path through n are exactly n's ancestors, n, and n's descendants; ancestors are alive and descendants are unobserved, so {n ∈ π} ⊆ {π ∩ D = ∅}. ∎ The denominator is shared by every frontier node, so **non-diagnostic failures cannot change the ranking of unexplored alternatives — they can only eliminate the dead node.** Every systematic search already eliminates, so "failure memory" adds value only if failures are *diagnostic* (they point at their cause — a contradiction naming the conflicting evidence, a failing test naming the line), or if the prior is non-factorized (failures correlated across branches, the §8 FPG idea). FAGS's revive-the-best-rejected-score policy, and any "re-rank after failure" policy, is not justified by Bayesian evidence when failures are non-diagnostic. The KG benchmark's dead ends and contradictions are non-diagnostic, which explains §6–9's nulls.

**Environment.** Trees of depth 8, branching 3; a wrong move roots a decoy subtree whose "lure" children look correct for L ~ U{0..K} steps before a dead end (K = failure delay); with probability ρ a dead end names the wrong turn that caused it (diagnosis); verifier scores N(d′,1) for correct/lure vs N(0,1). Policies (same trees, same expansion budget): memoryless restarts, DFS, FAGS-style (local backtracking + raw-score memory), best-first on mean score, exact posterior search (elimination only), posterior with *local blame* (renormalize the dead node's siblings = blame the latest decision), each systematic one with and without using diagnosis (DFS+diagnosis is conflict-directed backjumping). 300 trees per cell × K∈{0..5} × ρ∈{0,.5,1} × d′∈{.5,1,2}; per-tree paired tests, Holm over 540.

**Pre-registered hypotheses and outcomes** (success within 100 expansions):
- **H1** (ρ=0 ⇒ diagnosis changes nothing): **holds** exactly (all differences 0).
- **H2** (diagnosis gain grows with ρ and K): **holds** — mean gain 0pp at K≤1, +1.2pp at K=2, **+17.6 / +34.5 / +40.6pp at K=3/4/5**; 58 significant wins, 0 losses across the four search families. Even ρ=0.5 captures nearly all of it (decoy subtrees produce many failures; one diagnosis suffices).
- **H3** (local blame falls behind exact elimination as K grows): **refuted** — local blame wins 12 cells, loses 3. Exact posterior ordering expands the *most probable* node, which is breadth-heavy and expensive at depth 8; local blame keeps the parent's mass and so pushes depth. Most-probable-first ≠ cheapest-to-goal; the gap is about ordering cost, not inference quality.
- **H4** (memory beats memoryless restarts): **holds** in 54/54 cells (e.g. d′=1, K=3: restarts 8% vs DFS 92%).
- FAGS-style revival without diagnosis never beats plain best-first-mean (0 wins, 3 losses); best-first-mean vs beam at equal cost: no reliable difference (2 significant wins, 6 losses — all 6 against width-1 beam — out of 108).

**Reading.** FAGS's intuition is right in a specific, testable regime — **failures that are delayed and diagnostic** — and provably idle outside it. Scoring-based revival ("revive the best-scoring reject") is never the ingredient that helps; using the *content* of the failure is.

Limitation: the diagnosis signal is an oracle that names the true culprit with probability ρ; a noisy diagnosis (sometimes naming the wrong ancestor) is the obvious next variant. The posterior policies assume a factorized prior, which the lure mechanism violates; an exact posterior under the true generative model could extract some information from non-diagnostic failures.

## 24. External validity: MetaQA with a real embedding verifier (`fags/metaqa.py`, `metaqa_experiment.py`, `results/v2/metaqa_*`)

**Setup.** MetaQA vanilla (Zhang et al., 2018; KB of 134k movie triples; data from the `camazlucas/MetaQA` mirror into `data/metaqa/`, git-ignored), random 1000 test questions each from 2-hop and 3-hop. Search over directed relation paths (18 relations incl. inverses) from the topic entity, commit after k hops, score expected Hits@1 and F1 against the gold answer set. Verifiers are real `BAAI/bge-small-en-v1.5` embeddings, no oracle information: *relation-level* (cos(question, relation text), the same at every hop — hop-blind) and *path-level* (cos(question, text of the whole partial path) — hop-aware). Oracle = best executable k-hop path by F1. Clustered on oracle path type (24 / 31 clusters — conservative; most question templates share a path type), Holm over 16 tests.

| Split | rel-greedy | rel-beam-3/8/18 | path-greedy | path-beam-3/8/18 | path-exhaustive | oracle | legacy goal-test (beams) |
|---|---|---|---|---|---|---|---|
| 2-hop | 87.4% | 87.2% | 91.2% | 91.5% | 91.4% | 100% | 100% |
| 3-hop | 29.5% | 35.9 / 35.7 / 35.7% | 35.8% | **45.5%** | 49.3% | 100% | 98–100% |

Readings, against §22's synthetic findings:
1. **Search saturates at width 3 — replicates.** Widths 3, 8, 18 give identical answers on both splits; exhaustive scoring of every executable path adds only −0.1pp (2-hop) / +3.8pp (3-hop, n.s.).
2. **Hop-aware scoring beats hop-blind — replicates in direction, smaller and not significant under template clustering:** +4.3pp (2-hop), +9.6–9.8pp (3-hop), CIs overlapping 0. The effect is smaller than synthetic because MetaQA relations are type-constrained: a relation at the wrong hop usually yields an empty set and is pruned, which a synthetic graph never does.
3. **The verifier, not the search, is the ceiling:** oracle − path-exhaustive = **+50.7pp** on 3-hop (Holm p = 0.007). Even scoring every possible path, a frozen embedding verifier picks the right one under half the time.
4. **Goal-test inflation replicates and is extreme:** 98–100% "success" vs 36–46% committed Hits@1 on 3-hop.
5. MetaQA's failures (empty entity sets) are immediate (K = 0) and non-diagnostic, so by §23 failure memory has nothing to offer there; FAGS was not run on MetaQA.

## 25. Net takeaway (supersedes §5 and §19)

- The headline numbers of §4–18 were artifacts of a generator bug (gold paths unreachable), a protocol that leaks the goal, inconsistent cost accounting, and pseudo-replicated statistics. On the fixed generator, **FAGS does beat budget-matched random restarts per verifier call on all three sizes** (§21) — the original "it's just extra budget" conclusion was itself an artifact.
- Under an honest commit-to-an-answer protocol, **search is a solved sub-problem at beam width ~3** on both the synthetic benchmark and MetaQA; the remaining error is the verifier — overwhelmingly its **hop-blindness** in the synthetic task (53–67pp), and ranking quality on MetaQA 3-hop (51pp gap to oracle).
- **Learning from failure provably cannot re-rank the frontier when failures are non-diagnostic** (§23 proposition); it helps — by up to +41pp at equal budget — exactly when failures are delayed *and* name their cause. Revival by raw score is never the helpful ingredient. That benefit survives noisy diagnoses only when they are treated as evidence (Bayesian down-weighting or two-vote agreement): pruning on every diagnosis becomes harmful at 5–20% diagnosis error (§27). The same pattern holds on a real task with nothing injected (Sudoku, §28): score-based revival ≈ systematic search, conflict-set diagnosis always helps, and helps significantly more when the search ordering lets failures surface late. The most robust way to act on a diagnosis is to *defer* the named branch rather than delete it (soft backjumping, §30): same cost as backjumping when diagnoses are right, never loses the solution when they are wrong, and never worse than ignoring diagnoses in 80/80 tested conditions. The pattern carries over to LLM agents repairing code (§29): on three models, failure feedback that says why raises repair success by 34–48 points over a fresh attempt, corrupted diagnoses erase most of that, and filtering diagnoses by agreement between two checkers recovers much of it.
- Research directions that follow: (a) hop-/plan-aware verifiers (question decomposition, path-level scoring, LLM verifiers) evaluated under the commit protocol; (b) environments whose failures carry diagnostic content (contradictions naming evidence, unit-test failures, type errors) and learned diagnosis with noise; (c) cost-aware ordering (most-probable-first ≠ cheapest-to-goal, H3).

## 26. Housekeeping for the v2 work

- New modules: `fags/stats.py` (cluster tests, Holm), `fags/committed.py`, `fags/delayed_failure.py`, `fags/metaqa.py`. New field `Query.gold_path_id`; new params `legacy_evidence_bug`, `score_budget`.
- New scripts: `revalidation_experiment.py`, `error_decomposition_experiment.py`, `failure_informativeness_experiment.py`, `metaqa_experiment.py`; all outputs under `results/v2/`.
- `results/gold_rank_histogram_hybrid_*` (untracked) was generated at noise 0.30 although the script now says 0.08, and it measures relation rank, which §22 shows is not the binding quantity; treat it as superseded.
- Wall-clock (4-core machine shared with other jobs): re-validation 52 min, decomposition 63 min, failure sweep 20 min, MetaQA 11 min (+10 min one-off path-embedding cache).

## 27. Noisy diagnosis: does learning from failure survive wrong diagnoses? (`noisy_diagnosis_experiment.py`, `results/v2/noisy_diagnosis*`)

**Why.** §23's diagnosis was an oracle — when it fired it always named the true wrong turn. Real diagnoses (contradiction messages, failing tests, type errors) are sometimes wrong, and a wrong one can name a node *on the solution path*; acting on it by pruning destroys the only solution.

**Setup.** Same environment and same trees as §23 (verified: ε = 0 reproduces §23 exactly, e.g. Posterior+diag at d′=1, K=5: 0.910 vs 0.910). When a diagnosis fires, with probability ε it names a uniformly random *other* ancestor of the dead node. Grid: ε ∈ {0, .05, .1, .2, .3, .5} × ρ ∈ {.5, 1} × K ∈ {3, 5} × d′ ∈ {.5, 1}, 300 trees per cell, success within 100 expansions, per-tree paired tests, Holm over 624 tests. Ways of acting on a diagnosis:
- **hard** — prune the named subtree on first diagnosis (§23's +diag);
- **vote2** — prune only once two separate failures name the same node;
- **soft** — Bayesian: a node on the gold path can only be named by error, so multiply the named node's subtree mass by LR = (ε̂/m) / ((1−ε̂) + ε̂/m) (m = other ancestors the error could have picked) instead of zeroing it; run calibrated (ε̂ = ε) and with a fixed guess ε̂ = 0.2. Implemented in `posterior_search(..., diag_error_hat=...)`.

Sanity check: at ε = 0.5, ρ = 1, K = 5, hard-pruning DFS leaves **73/100 trees unsolvable even with a 20,000-expansion budget**; the soft update leaves 0/100.

**Pre-registered hypotheses — all three hold** (counts over the 40 cells with ε > 0):

| Comparison | Sig. wins | Sig. losses |
|---|---|---|
| X+hard vs no diagnosis (DFS / BestFirst-mean / PostLB) | 11 / 8 / 13 | **17 / 21 / 15** |
| X+vote2 vs no diagnosis (DFS / BestFirst-mean) | 25 / 20 | 6 / 8 |
| vote2 vs hard (DFS / BestFirst-mean) | **40 / 40** | 0 / 0 |
| PostLB+soft (calibrated) vs no diagnosis | **38** | **0** |
| PostLB+soft (fixed ε̂ = 0.2) vs no diagnosis | 39 | 0 |
| PostLB+soft vs PostLB+hard (calibrated / fixed guess) | 40 / 40 | 0 / 0 |
| Post (exact) +soft vs no diagnosis / vs hard | 36 / 35 | 0 / 0 |

- **N1 (hard pruning has a break-even error rate):** holds. Hard pruning becomes significantly *worse than ignoring diagnoses* at ε* ≈ **0.05–0.2 when K = 3** and ≈ 0.3–0.5 (or never, at ρ = .5) when K = 5 — the more valuable diagnosis is (longer delays), the more error it tolerates. E.g. d′=1, K=3, ρ=.5: DFS 92.3% vs DFS+hard 81.0% at ε = .1 and 38.0% at ε = .5.
- **N2 (soft is never worse than no diagnosis, and beats hard at high ε):** holds — 0 significant losses in 80 soft-vs-no-diagnosis tests, and it beats hard in every ε > 0 cell. It is robust to not knowing ε: the fixed guess ε̂ = 0.2 performs the same as calibrated (e.g. 77.7% vs 73.0% at ε = .5, d′=.5, K=3).
- **N3 (voting raises the break-even):** holds — vote2 beats hard in all 80 tests; its ε* moves from 0.05–0.2 to 0.2–0.5 or never.

**Reading.** The §23 conclusion survives realistic, imperfect diagnosis — but only if the agent treats a diagnosis as *evidence*, not as a command. Acting on every diagnosis (hard pruning, i.e. textbook backjumping) is actively harmful once ~5–20% of diagnoses are wrong; requiring agreement (vote2) or doing the Bayesian down-weighting keeps the benefit while removing the catastrophic failure mode. In absolute terms the best policy depends on the regime: PostLB+soft is best at d′ = 1 (≥ 92% up to ε = .3), while DFS+vote2 is best at d′ = 0.5, K = 5, where the posterior ordering's breadth bias (§23 H3) holds PostLB back (DFS+vote2 79% vs PostLB+soft 34% at ε = .3). A soft (likelihood-weighted) variant of DFS/best-first — combining depth-first ordering with Bayesian diagnosis handling — is the obvious next policy.

**Limitation (§27).** Diagnosis errors here are uniform over ancestors; real diagnosers likely err *systematically* (e.g. blaming the most recent step), which an agent could learn. ε̂ is fixed per run; learning it online from diagnoses that later prove wrong is untested.

## 28. Real task: Sudoku with natural, diagnostic failures (`fags/sudoku_env.py`, `sudoku_experiment.py`, `results/v2/sudoku_*`)

**Why.** §23 and §27 inject delay and diagnosis into a synthetic tree. This tests the same predictions where nothing is injected. No LLM API key or local generative model was available, so an LLM code-generation task was not possible; Sudoku is the classic domain whose failures naturally carry diagnostic content.

**Environment.** Forward-checking search over partial assignments, with a fixed branching variable per node so every policy searches the same tree:
- **Verifier:** the least-constraining-value heuristic. A child's score is minus the number of peer domains its value removes something from — a real, imperfect heuristic.
- **Failure:** a domain wipe-out, discovered when the dead child is expanded.
- **Diagnosis:** the conflict set (FC-CBJ, Prosser 1993). An exhausted node's conflict set is the union of its children's conflict sets, plus the assignments that had already removed values from its own variable. The deepest assignment in it is the culprit, and its whole subtree is provably dead. The diagnosis is sound, so the "+CBJ" hard pruning never loses a solution.
- **Failure delay:** varied by a standard, natural design choice. **MRV** ordering (fail-first) surfaces mistakes quickly; **static** row-major ordering surfaces them many assignments later.

Every policy is the unchanged code from `fags/delayed_failure.py`, plugged in through the same tree interface. The posterior policies are omitted: their per-step sweep is impractical on trees 50–60 levels deep.

**Verification.** On 40 hard puzzles, every solution matched the dataset's answer, and DFS+CBJ solved every puzzle DFS solved and was never more expensive (expected for backjumping). The experiment script also asserts every returned solution against the reference answer.

**Data.** 200 puzzles from `sapientinc/sudoku-extreme` (test split, 422,786 puzzles), stratified 4 × 50 by its backtrack rating (0–5, 6–20, 21–50, 51+), plus Norvig's top95. Budget 20,000 expansions. Puzzles are independent, so paired tests are per puzzle, on success@20000 and on log-cost (unsolved runs censored at the budget, which understates gains), Holm over 27 tests. Runtime 33 min.

| Ordering | Restarts | DFS | DFS+CBJ | FAGS-style | FAGS-style+CBJ | BestFirst-mean | BestFirst-mean+CBJ |
|---|---|---|---|---|---|---|---|
| MRV (success@20000) | 45.1% | 88.8% | 89.8% | 91.5% | 91.9% | 88.1% | 90.5% |
| static (success@20000) | 2.7% | 37.3% | **47.1%** | 36.6% | **44.1%** | 37.6% | **46.1%** |
| CBJ speed-up, MRV / static | | | ×1.06 / **×1.22** | | ×1.02 / **×1.13** | | ×1.06 / **×1.20** |

**Pre-registered predictions — all four hold:**
- **R1 (revival by raw score is no better than systematic search):** holds. FAGS-style vs DFS and vs BestFirst-mean: no significant difference under either ordering (Holm p ≥ 0.06). Under MRV, FAGS-style is numerically slightly ahead (+2.7pp vs DFS, +3.4pp vs BestFirst-mean); under static ordering it is numerically slightly behind.
- **R2 (diagnosis never hurts):** holds. The conflict set significantly lowers cost for all three policies under both orderings (Holm p ≤ 2e-12). Under static ordering it also raises success by +9.8 / +7.5 / +8.5pp (DFS / FAGS-style / BestFirst-mean, all Holm p ≤ 3e-5).
- **R3 (the gain grows with failure delay):** holds. The per-puzzle cost reduction from diagnosis is significantly larger under static ordering than under MRV for all three policies (+0.10 to +0.14 in log-cost, Holm p ≤ 2e-10).
- **R4 (memory beats memoryless restarts):** holds. DFS beats restarts by +43.7pp (MRV) and +34.6pp (static).

**Reading.** On a real task with real heuristics and real failure explanations, the §23 picture reappears. Reviving the best-scoring rejected alternative adds nothing reliable over plain systematic search. Using what a failure says about its cause helps, and helps more when the search design lets mistakes surface late. This also explains a known result in the constraint-satisfaction literature, that backjumping adds little on top of forward checking with MRV (Chen & van Beek, 2001): fail-first ordering keeps failure delay short, which is exactly the regime where §23 predicts diagnosis is worth little.

**Limitations.** The diagnosis here is exact (sound), so §27's noisy-diagnosis results are not tested on a real task. The measured backjump distance differs only modestly between orderings (2.03 vs 2.55 assignments skipped), so "delay" is manipulated through the ordering rather than measured directly. The verifier is a fixed heuristic, not a learned or LLM verifier. The posterior policies were not run.

## 29. Real LLM agents: code repair with failure feedback (`fags/llm.py`, `fags/codegen.py`, `llm_feedback_experiment.py`, `llm_feedback_analysis.py`, `results/v2/llm_*`)

**Why.** §23–28 test the theory on synthetic search and Sudoku. This asks the same question of LLM agents, where "learning from failure" (Reflexion-style retry loops) is common practice: does feedback after a failure help only when it says *why*, and what happens when the "why" is sometimes wrong?

**Task and protocol.** MBPP+ (EvalPlus; 373 usable problems after excluding 5 whose reference solution fails here). The agent sees the task and one example assert. EvalPlus's extended inputs are split per problem with a fixed seed: the 3 base inputs plus a random half of the rest are **visible** (the agent's code runs on them and failures can be reported back), the other half is **hidden** and used only for scoring. A first design used only the 3 base asserts as the visible suite; almost no first attempt failed them (14/300 for gpt-oss-20b), so there was nothing to repair. That design was discarded before any repair run. All conditions share the same first attempt; on a visible failure the agent gets up to 2 repairs (3 LLM calls in total), stopping early once the visible inputs pass. Success = the submitted program passes every hidden input. Temperature 0.8, fixed seeds, every LLM response cached on disk. Models via Groq: `qwen/qwen3.8-27b`, `openai/gpt-oss-120b`, `openai/gpt-oss-20b`.

**Conditions** (what the agent is told after a visible failure):
- **restart:** nothing; a fresh independent attempt (memoryless control).
- **binary:** "your code failed at least one test" (a non-diagnostic failure, i.e. FAGS's premise).
- **diagnostic:** up to 3 failing visible inputs, each with the call, the expected value and what the code returned.
- **noisy_hard:** the same report, but each expected value is replaced by a plausible wrong one with probability 0.5, presented as fact.
- **noisy_soft:** the identical corrupted reports, presented as unreliable ("about half of the expected values are wrong; check each against the task").
- Follow-up, qwen only: **noisy_vote2** (two independent unreliable checkers; a value is shown as fact only when both agree, otherwise both candidates are shown; a wrong value is stated as fact in 6% of reports vs 48% for noisy_hard) and **noisy_verify** (noisy_soft plus an explicit procedure: work out the correct output from the task first, judge each report, then fix).

Analysis is on the problems whose first attempt failed a visible input (qwen 79, gpt-oss-120b 73, gpt-oss-20b 75), the only ones where conditions can differ; paired per-problem sign-flip tests, Holm within each model.

**Results** (success on hidden tests within 3 calls, % of initially failing problems):

| Condition | qwen3.8-27b | gpt-oss-120b | gpt-oss-20b |
|---|---|---|---|
| restart | 22.8 | 20.5 | 20.0 |
| binary | 19.0 | 38.4 | 22.7 |
| **diagnostic** | **57.0** | **68.5** | **56.0** |
| noisy_hard | 32.9 | 26.0 | 24.0 |
| noisy_soft | 39.2 | 45.2 | 29.3 |
| noisy_vote2 | 49.4 | — | — |
| noisy_verify | 40.5 | — | — |

**Pre-registered hypotheses:**

| | Hypothesis | qwen3.8-27b | gpt-oss-120b | gpt-oss-20b |
|---|---|---|---|---|
| L1 | binary is not better than restart | **holds** (−3.8pp, n.s.) | **fails**: +17.8pp, Holm p = 0.013 | **holds** (+2.7pp, n.s.) |
| L2 | diagnostic beats restart and binary | **holds**: +34.2 / +38.0pp | **holds**: +47.9 / +30.1pp | **holds**: +36.0 / +33.3pp (all Holm p < 0.0001) |
| L3 | noisy_hard is worse than diagnostic and not better than restart | **holds**: −24.1pp; vs restart n.s. | **holds**: −42.5pp; vs restart n.s. | **holds**: −32.0pp; vs restart n.s. |
| L4 | noisy_soft beats noisy_hard and is not worse than restart | partial: +6.3pp (n.s.); +16.5 vs restart (Holm p = 0.054) | **holds**: +19.2pp (p = 0.008); +24.7 vs restart (p = 0.0005) | partial: +5.3pp (n.s.); +9.3 vs restart (n.s.) |
| L6 | noisy_vote2 beats noisy_hard (qwen) | **holds**: +16.5pp, Holm p = 0.018; +26.6 vs restart (p = 0.001) | | |
| L7 | noisy_verify beats noisy_hard (qwen) | not supported: +7.6pp, n.s. (+17.7 vs restart, p = 0.048) | | |

**Reading.**
1. **The core claim replicates on three LLMs.** Telling the agent *why* it failed raises repair success by 34–48 points over a fresh attempt, on every model, with Holm p < 0.0001. Corrupting half of those diagnoses erases most of the gain (−24 to −43 points), leaving it statistically indistinguishable from a restart.
2. **A non-diagnostic failure is worth little, with one model-dependent exception.** "It failed" adds nothing for qwen3.8-27b and gpt-oss-20b, but +17.8 points for gpt-oss-120b. The theory's proposition concerns ranking *unexplored* options; an LLM retry is not a pure frontier choice. A strong model can treat "my first approach was wrong" as information about its own approach (elimination in its own idea space), which the proposition allows. Even there it reaches only about half the diagnostic gain (38.4% vs 68.5%).
3. **Handling unreliable diagnoses.** A one-line warning helps clearly only for the strongest model (gpt-oss-120b, +19 points); weaker models barely use it. A *structural* filter works regardless: the two-checker vote (the LLM analogue of §27's vote2) recovers +16.5 points over corrupted-as-fact on qwen, reaching 49.4% against 57.0% for clean diagnoses. Asking the model to verify each report itself helps less and not significantly. As in §27, cleaning the evidence before it reaches the agent beats asking the agent to be careful.

**Limitations.** One benchmark (MBPP+), three open-weight models through one provider, a fixed corruption rate (0.5) and repair budget (3 calls); vote2/verify were run on qwen only. The corruption is injected (plausible wrong expected values), not a naturally noisy diagnoser. The repair loop is an LLM conversation, not tree search, so §30's deferral has no direct analogue here and was not tested.

## 30. Soft backjumping: defer, don't delete (`fags/delayed_failure.py` modes `defer` / `vote_defer`, `soft_backjump_experiment.py`, `results/v2/soft_backjump*`)

**Why.** §27 showed hard pruning on diagnoses becomes harmful once ~5–20% of them are wrong. The Bayesian soft update fixed that but inherits the posterior ordering's weakness with a poor verifier (d′ = 0.5). This tests a policy that keeps depth-first / best-first ordering: on a diagnosis, **move the named subtree behind everything else instead of deleting it** ("soft backjumping"). A correct diagnosis names a dead subtree, so the goal is always found before it would be revisited — the cost is provably identical to hard pruning. A wrong diagnosis only delays the solution; it can never remove it. Also tested: `vote_defer` (defer on the first diagnosis of a node, prune on the second).

**Setup.** §27's exact trees, grid (ε × ρ × K × d′, 300 trees per cell) and budget (100 expansions); per-tree paired tests, Holm over 576. Before the run: on 1,200 runs with ε = 0, defer and hard pruning had identical costs; at ε = 0.5 with a 20,000-expansion budget, hard-pruning DFS left 73/100 trees unsolvable, defer 0/100.

**Pre-registered hypotheses** (tallies over the 40 cells with ε > 0):

| Hypothesis | Comparison | Sig. wins | Sig. losses | Verdict |
|---|---|---|---|---|
| S1 | defer vs hard at ε = 0 (cost) | — | — | **holds exactly** (0 differences in 1,200 + 2,400 runs) |
| S2 | DFS+defer / BestFirst-mean+defer vs no diagnosis | 36 / 34 | **0 / 0** | **holds** |
| S3 | vote_defer vs vote2 (DFS / BF) | 0 / 1 | 0 / 0 | tie |
| S3 | vote_defer vs defer (DFS / BF) | 0 / 0 | **23 / 22** | **refuted** — adding the vote hurts |
| S4 | DFS+defer vs PostLB+soft | 11 | **0** | **holds** |
| — | defer vs hard (DFS / BF) | **40 / 40** | 0 / 0 | |

Example (d′ = 0.5, K = 5, ρ = 0.5, success @ 100): at ε = 0.3, DFS+defer **83.0%** vs DFS+vote2 79.0%, DFS+hard 36.0%, PostLB+soft 34.0%, no diagnosis 30.0%; at ε = 0.5, 59.0% vs 50.7% / 20.3% / 16.7% / 30.0%. With the stronger verifier (d′ = 1, K = 3, ρ = 0.5), DFS+defer stays ≥ 93% up to ε = 0.5, where hard pruning falls to 38%.

**Reading.** Soft backjumping is the best way of acting on failure diagnoses found in this project: identical to classic backjumping when diagnoses are right, complete when they are wrong, never significantly worse than ignoring diagnoses in any of 80 tested cells, better than hard pruning in all 80, and better than the Bayesian update wherever they differ. It needs no estimate of the diagnosis error rate. The vote refinement, which helped hard pruning in §27, *hurts* here: once deferral already makes wrong diagnoses cheap, waiting for a second one only throws away the value of correct first diagnoses. DFS+defer vs vote2 was not among the pre-registered tests; numerically defer is ≥ vote2 in every reported cell.

**Limitations.** Same synthetic environment as §23/§27 (uniform diagnosis errors). Sudoku's own diagnoses are exact (§28), where defer ≡ hard by S1; §31 tests deferral on Sudoku with noisy diagnoses. The LLM analogue of "defer" is not obvious, since an LLM's context cannot simply postpone a hypothesis.
