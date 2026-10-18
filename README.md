# DemoSearch (FAGS)

A research codebase testing one hypothesis: **can a graph-search agent that "remembers" the paths it wrongly rejected, and revives the best one on a dead end, beat plain greedy search — without degenerating into brute-force exploration?**

The testbed is a synthetic multi-hop knowledge-graph QA task: each graph has gold answer paths (2–5 hops), distractor edges that look plausible at branch points, dead ends, and contradiction nodes. A **Verifier** scores candidate relations against a question, and a search algorithm picks the best-scoring edge at each hop.

## What's in here

- **Baseline search** — greedy, no memory, stops on first failure
- **FAGS (Failure-Aware Graph Search)** — same greedy search, but backtracks locally on a dead end/contradiction and can resume from the best previously-rejected branch in a failure memory
- A large set of follow-up experiments comparing FAGS against alternative search paradigms: beam search, diverse/diversity-aware beam pruning, best-first search, and MCTS (including a beam-seeded MCTS hybrid), plus budget-matched controls and embedding-model ablations (BGE vs MiniLM)

See `PROJECT_STATUS.md` for the full experiment log and results write-up. **Start at §21–25**: results before §21 were produced under a generator bug that made most gold paths unreachable and are kept only as a historical record.

### v2 experiments (fixed generator, cluster-level statistics)

```bash
python revalidation_experiment.py         # §21 core claims re-tested on the fixed generator
python error_decomposition_experiment.py  # §22 commit protocol + search/verifier/hop-blindness decomposition
python failure_informativeness_experiment.py  # §23 when failure memory can help (delay x diagnosis x verifier)
python metaqa_experiment.py               # §24 MetaQA with BGE verifiers (needs data/metaqa/, see §24)
python noisy_diagnosis_experiment.py      # §27 failure diagnoses that are sometimes wrong
python sudoku_experiment.py               # §28 real task: Sudoku with conflict-set diagnosis (needs data/sudoku/)
python llm_feedback_experiment.py qwen/qwen3.8-27b   # §29 LLM code repair with failure feedback (needs GROQ_API_KEY in .env, data/mbppplus/)
python llm_feedback_analysis.py           # §29 analysis of the LLM runs
python soft_backjump_experiment.py        # §30 soft backjumping: defer instead of delete
```

Outputs go to `results/v2/`.

## Tech Stack

Python, NumPy, SciPy, Matplotlib, Tabulate

## How to Run

```bash
pip install -r requirements.txt
python main.py
```

Individual experiments are standalone scripts, e.g.:
```bash
python beam_search_experiment.py
python mcts_search_experiment.py
```

Results (CSVs and plots) are written to `results/`.
