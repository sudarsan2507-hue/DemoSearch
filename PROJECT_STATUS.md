# DemoSearch / FAGS — Project Status

_Analysis compiled from code (`fags/`, top-level scripts), `results/*.csv`, and `results/*.png`. Initial pass 2026-06-22; updated 2026-06-23 with the budget-matched control experiment (§6), the regenerated canonical results (§4 numbers, §20), the HybridVerifier follow-up control (§7), the learned failure-pattern-graph experiment (§8), the diversity-aware memory experiment (§9), the headline beam search result (§10), beam search's own verifier-quality stress test (§11), and the diverse-beam-pruning experiment (§12); updated again 2026-06-27 with the beam search + Failure Pattern Graph composition (§13), the top-K ranking-rule variants (§14), and global best-first search as an alternative pruning paradigm (§15); updated again 2026-06-28 with MCTS as a second alternative paradigm (§16), MCTS's crossover at large budgets (§17), and a beam-seeded MCTS hybrid built to close the small-budget gap (§18)._

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
