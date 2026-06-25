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

## Day 9+ — as new work happens

- [ ] (new rows added here as algorithms/experiments change)
