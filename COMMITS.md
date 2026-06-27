# Commit Log / Plan

Tracker for how this project's history is being split into commits. Whenever an
algorithm/mechanism changes (new search variant, verifier type, sweep, etc.), it
gets its own row here before/when it's committed. Pace target: ~5 commits/day.
`[x]` = pushed, `[ ]` = planned but not yet committed.

## Day 1 — 2026-06-22

- [x] 1. Project scaffold (`requirements.txt`, `.gitignore`)
- [x] 2. FAGS core library (`fags/` package: graph generator, verifier, memory, baseline + failure search, evaluation)
- [x] 3. Main experiment runner + canonical results (`main.py`, accuracy/ablation tables, scalability + cost plots, `summary.txt`)
- [x] 4. Diagnostic scratch scripts + gold-rank histogram (`scratch/rank_diagnostic.py`, `recovery_audit.py`, `post_revival_audit.py`)
- [x] 5. Verifier-quality sweep experiment (`verifier_sweep.py`, sweep CSVs/plots, `verifier_sweep_summary.txt`)

## Day 2 — 2026-06-23

- [x] 6. Shield mechanism + experiment (`patch_fags.py`, `patch_fags2.py`, `patch_fags3.py`, `debug_hops.py`, `shield_experiment.py`, shield results)
- [x] 7. Stabilization (K / certificate-bonus) sweep (`patch_fags_params.py`, `stabilization_sweep.py`, `k_sweep.*`, `bonus_sweep.*`, `ablation.*`)
- [x] 8. RTC-lite experiment (`patch_rtc.py`, `rtc_lite_experiment.py`)
- [x] 9. RBSC experiment (`patch_rbsc.py`, `rbsc_experiment.py`)
- [x] 10. Verifier overhaul: embeddings/hybrid + test scripts (`patch_verifier*.py`, `patch_searches.py`, `verifier_diagnostic*.py`, `verifier_clean_test.py`, `verifier_hybrid*.py`, `verifier_scaling_test.py`, `verifier_experiment.py`, `hybrid_sweep_experiment.py`, `bge_scale_experiment.py`, `bge_vs_minilm_experiment.py`)

## Day 3 — 2026-06-23

- [x] 11. Analysis doc (`PROJECT_STATUS.md`)

## Day 4 — 2026-06-23

- [x] 12. Fix graph-generator reproducibility bug (`fags/graph_generator.py`: sort `CONFUSABLE_PAIRS` iteration so `_CONFUSABLE_MAP` — and therefore every seeded experiment — is deterministic across processes/hash seeds)
- [x] 13. Add budget-matched random-restart control experiment (`budget_matched_control_experiment.py`, `results/budget_matched_control.*`) — tests whether FAGS beats a dumb baseline given the same node-visit budget; answer: only on the Small graph
- [x] 14. Update `PROJECT_STATUS.md` with the control-experiment finding and revised net takeaway

## Day 5 — 2026-06-23

- [x] 15. Regenerate canonical results under the fixed deterministic generator (`main.py`, `shield_experiment.py`, `stabilization_sweep.py`, `verifier_sweep.py`, `scratch/rank_diagnostic.py`); fixed an unrelated latent bug in `verifier_sweep.py`'s `ControlledVerifier.score()`
- [x] 16. Update `PROJECT_STATUS.md` with regenerated numbers and housekeeping notes

## Day 6 — 2026-06-23

- [x] 17. Add HybridVerifier budget-matched control (`budget_matched_control_embedding_experiment.py`, `results/budget_matched_control_embedding.*`) — tests whether a stronger (rule+BGE) verifier rescues FAGS; answer: no, FAGS loses to the dumb control significantly worse than with the weak verifier
- [x] 18. Update `PROJECT_STATUS.md` with the HybridVerifier finding and final revised recommendation

## Day 7 — 2026-06-23

- [x] 19. Add `path_relations` field to `SearchResult` (`fags/__init__.py`, `fags/baseline_search.py`, `fags/failure_search.py`) so a finished path's relation sequence is recoverable; fixes a copy-paste gap where the field was only populated on the success-path return, not the failure-path one
- [x] 20. Add Failure Pattern Graph mechanism (`fags/failure_pattern_graph.py`: `FailurePatternGraph`, `train_failure_pattern_graph`, `PatternAwareVerifier`) — the user's proposed cross-query learned-avoidance idea
- [x] 21. Add Failure Pattern Graph experiment (`failure_pattern_graph_experiment.py`, `results/failure_pattern_graph_*`) — tests it with a proper train/test split; finding: real learned signal, but doesn't beat baseline at equal cost, and significantly hurts FAGS when composed with it
- [x] 22. Update `PROJECT_STATUS.md` with the Failure Pattern Graph finding and final revised takeaway

## Day 8 — 2026-06-23

- [x] 23. Add `DiversityMemory` to `fags/memory.py` — revives the highest-scoring rejected candidate that is NOT the same as/confusable with/highly coherent with the winning relation (the user's proposed fix for §6's finding that targeted revival loses to dumb random restarts); threaded an additive `winner_relation` param through `FailureMemory.store()` and `failure_search.py`'s call site
- [x] 24. Add Diversity Memory experiment (`diversity_memory_experiment.py`, `results/diversity_memory_*`) — tests it against the budget-matched control across all 3 graph sizes; finding: mechanism fires on ~99% of opportunities but is statistically indistinguishable from plain Top1Memory everywhere, and the Large-graph loss vs the control becomes significant where it wasn't before
- [x] 25. Update `PROJECT_STATUS.md` with the Diversity Memory finding and final revised takeaway

## Day 9 — 2026-06-23

- [x] 26. Add beam search (`fags/beam_search.py`) — a structurally different algorithm replacing FAGS's commit-then-recover design with K live hypotheses expanded and pruned concurrently at every hop
- [x] 27. Add beam search experiment (`beam_search_experiment.py`, `results/beam_search_*`) — sweeps beam width across all 3 graph sizes vs Baseline/FAGS-Top1/budget-matched RRB; **headline finding: beam search strictly dominates FAGS in the cost/accuracy tradeoff and beats the budget-matched control in 14/15 configurations** — the first mechanism in this whole investigation to clear that bar
- [x] 28. Update `PROJECT_STATUS.md` with the beam search finding as the new headline result

## Day 10 — 2026-06-23

- [x] 29. Add beam search HybridVerifier experiment (`beam_search_embedding_experiment.py`, `results/beam_search_embedding_*`) — tests whether a stronger verifier helps beam search the way it hurt FAGS; finding: never loses to the budget-matched control (vs FAGS's decisive loss under the same verifier), and pulls significantly ahead at width=5
- [x] 30. Update `PROJECT_STATUS.md` with the beam search verifier-quality stress test

## Day 11 — 2026-06-23

- [x] 31. Add `max_children_per_parent` diversity cap to `fags/beam_search.py` — caps how many of the new beam's slots one parent hypothesis can fill, with an adaptive floor so a single start node can still grow the beam to full width on the first hop
- [x] 32. Add diverse beam search experiment (`diverse_beam_search_experiment.py`, `results/diverse_beam_search_*`) — cost-neutral comparison vs plain beam search at widths {5,8} x caps {1,2,3} across all 3 graph sizes; finding: size-dependent split, helps on Small/Medium, significantly hurts on Large (same "diversity != correctness" failure mode as DiversityMemory for FAGS)
- [x] 33. Update `PROJECT_STATUS.md` with the diverse beam pruning finding; revises production recommendation to plain (uncapped) beam search

## Day 13 — 2026-06-27

- [x] 34. Add beam search + Failure Pattern Graph experiment (`beam_search_fpg_experiment.py`, `results/beam_search_fpg_*`) — composes the existing `PatternAwareVerifier` (no new core code) with beam search across widths {3,5,8} x penalties {0.05-0.3}; finding: clean null (0 significant wins, 0 significant losses) - never hurts beam the way it hurt FAGS, but doesn't help either
- [x] 35. Update `PROJECT_STATUS.md` with the beam+FPG null result

## Day 14 — 2026-06-27

- [x] 36. Add `score_aggregation` (sum vs mean) and `diversity_penalty_weight` (soft per-parent penalty) to `fags/beam_search.py` — two genuinely different top-K ranking rules, vs §12's hard cap which only constrained the existing rule
- [x] 37. Add beam search top-K ranking variants experiment (`beam_search_topk_variants_experiment.py`, `results/beam_search_topk_variants_*`) — widths {5,8} x diversity penalties {0,0.05,0.1,0.2} x aggregation {mean,sum} across all 3 graph sizes; finding: sum aggregation is a clean null (0/6), soft diversity penalty just replicates §12's hard-cap pattern (3 wins/2 losses, same Small/Medium-helps-Large-hurts split), combined washes out to 0/18 — fourth/fifth refinement attempt to fail to beat plain beam search
- [x] 38. Update `PROJECT_STATUS.md` with the top-K ranking-rule variants finding

## Day 15 — 2026-06-27

- [x] 39. Add global best-first search (`fags/best_first_search.py`) — alternative pruning paradigm to beam search: one global priority queue across all depths, always expand the best-scoring frontier hypothesis, budget-capped instead of width-capped
- [x] 40. Add best-first vs beam search experiment (`best_first_search_experiment.py`, `results/best_first_search_*`) — per-query budget-matched comparison (exact same node budget) across widths {2,3,5,8} x all 3 graph sizes; **finding: beam search wins all 12/12 configurations with extreme significance (p as low as 1e-57), best-first collapses to near-0% on Large at low budgets** — the most decisive result of the whole investigation, confirming fixed-width beam's guaranteed breadth is load-bearing, not arbitrary
- [x] 41. Update `PROJECT_STATUS.md` with the best-first search finding

## Day 16 — 2026-06-28

- [x] 42. Add Monte Carlo Tree Search (`fags/mcts_search.py`) — UCB1 selection + greedy rollout + backpropagation, budget-capped by total nodes visited; built to test whether explicit exploration/exploitation balance fixes best-first search's tunnel-vision collapse from §15
- [x] 43. Add MCTS vs beam search experiment (`mcts_search_experiment.py`, `results/mcts_search_*`) — per-query budget-matched comparison across widths {2,3,5,8} x all 3 graph sizes, with best-first shown for reference; **finding: MCTS loses to beam search 10/12 (2 non-significant ties) but completely fixes best-first's collapse (10-15x better than best-first on Large at low budgets)** — confirms the exploration-vs-exploitation theory but still doesn't beat beam search
- [x] 44. Update `PROJECT_STATUS.md` with the MCTS finding

## Day 17 — 2026-06-28

- [x] 45. Fix MCTS safety-net scaling bug (`fags/mcts_search.py`) before testing larger node budgets — max_simulations was scaling UP with node_budget, backwards; could spin 128,000+ wasted simulations once budget exceeds a graph's reachable size. Replaced with a stale-count detector (bails once the reachable space stops growing, independent of budget size) plus a fixed backstop
- [x] 46. Add MCTS-at-large-budgets experiment (`mcts_large_budget_experiment.py`, `results/mcts_large_budget_*`) — sweeps both algorithms up to 1280-node budgets on Medium/Large graphs; caught and corrected a measurement pitfall (beam was only swept to width=60/~186 nodes while MCTS reached ~372, which would have overstated the gap as +56% instead of the true +15.40%) by extending beam's sweep to matched cost before reporting; **finding: MCTS does catch up to and pass beam search, but only at ~5-10x larger budgets (200+ nodes) than anything compared in §16 (5-40 nodes)**
- [x] 47. Update `PROJECT_STATUS.md` with the corrected MCTS-crossover finding — revises the "beam search undefeated" framing to "undefeated at small/moderate budgets; MCTS wins at large ones"

## Day 18+ — as new work happens

- [ ] (new rows added here as algorithms/experiments change)
