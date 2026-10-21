# DemoSearch (FAGS)

**When does learning from failure help a search agent?**

This started as FAGS (Failure-Aware Graph Search): a knowledge-graph search agent that remembers branches it rejected and revives the best one at a dead end. It grew into a study of failure feedback across synthetic search, Sudoku, two knowledge-graph QA datasets and LLM code repair. Every number below is in [`PROJECT_STATUS.md`](PROJECT_STATUS.md); **start at §21** (earlier sections predate a generator bug fix).

| Evaluated on | Scale |
|---|---|
| Synthetic knowledge graphs and search trees | 3 graph sizes × 3 seeds; up to 300 trees per condition |
| Sudoku (real puzzles, real heuristic, real failure explanations) | 295 hard puzzles |
| Knowledge-graph QA with BGE embedding verifiers | MetaQA (2,000 questions), WebQSP (500 questions) |
| LLM code repair (MBPP+ with held-out tests) | 3 models: qwen3.8-27b, gpt-oss-120b, gpt-oss-20b |

All comparisons are paired, with cluster-level tests and Holm correction; hypotheses were written down before each run.

## Key results

1. **A failure only helps if it says *why*.** A bare "dead end" can't change which option to try next (proved in §23), so reviving the best-scoring rejected branch, FAGS's core idea, never beat plain systematic search. A diagnosis that names the cause adds up to +70 points, and the later mistakes surface, the more it adds.
2. **The same holds for LLMs repairing code.** Telling the model *why* a test failed gives 2.5–3.3× the repair success of a fresh retry on all three models. Wrong diagnoses erase the gain; requiring two checkers to agree recovers much of it, while asking the model to double-check does not (§29, §33a).
3. **Defer, don't delete.** Classic backjumping prunes whatever a diagnosis blames, so one wrong diagnosis can delete the solution. Deferring the blamed branch instead ("soft backjumping", §30) costs nothing when diagnoses are right and stays above ignoring them even when half are wrong (§31).
4. **In knowledge-graph QA the verifier is the bottleneck, not the search.** Searching wider than ~3 hypotheses changes nothing, the best possible path beats what the search picks by 50–70 points, and scoring by whether the search ever *touched* the answer inflated accuracy 2–3× (§22, §24, §32).

<table>
  <tr>
    <td width="50%"><img src="docs/figures/diagnosis_vs_delay.png" alt="Diagnosis gain vs failure delay"><br><sub><b>Fig. 1</b> · Diagnosis gain vs how late mistakes surface, for three verifier strengths (A) and four search policies (B).</sub></td>
    <td width="50%"><img src="docs/figures/llm_feedback.png" alt="LLM code repair by feedback type"><br><sub><b>Fig. 2</b> · LLM code repair by type of failure feedback, per model, with 95% confidence intervals.</sub></td>
  </tr>
  <tr>
    <td width="50%"><img src="docs/figures/sudoku_soft_backjumping.png" alt="Soft backjumping on Sudoku with noisy diagnoses"><br><sub><b>Fig. 3</b> · Acting on wrong diagnoses in Sudoku: deferring vs two pruning rules, under two variable orderings.</sub></td>
    <td width="50%"><img src="docs/figures/kgqa_verifier_ceiling.png" alt="Verifier ceiling in knowledge-graph QA"><br><sub><b>Fig. 4</b> · Knowledge-graph QA: the gap to the best path (A) and why wider search doesn't close it (B).</sub></td>
  </tr>
</table>

## Setup

```bash
pip install -r requirements.txt
```

LLM experiments need a Groq key: copy `.env.example` to `.env` and set `GROQ_API_KEY` (`.env` is git-ignored). Datasets go in `data/` (also git-ignored):

| Folder | Source (HuggingFace unless noted) |
|---|---|
| `data/metaqa/` | `camazlucas/MetaQA` |
| `data/sudoku/` | `sapientinc/sudoku-extreme` + norvig.com/top95.txt |
| `data/mbppplus/` | `evalplus/mbppplus` |
| `data/webqsp/` | `rmanluo/RoG-webqsp` |

## Run

Each script writes to `results/v2/`; § numbers refer to `PROJECT_STATUS.md`.

```bash
python revalidation_experiment.py             # §21 original claims, re-tested
python error_decomposition_experiment.py      # §22 honest scoring + error breakdown
python failure_informativeness_experiment.py  # §23 failure delay x diagnosis
python metaqa_experiment.py                   # §24 MetaQA
python noisy_diagnosis_experiment.py          # §27 wrong diagnoses
python sudoku_experiment.py                   # §28 Sudoku
python llm_feedback_experiment.py qwen/qwen3.8-27b && python llm_feedback_analysis.py   # §29 LLM repair
python soft_backjump_experiment.py            # §30 soft backjumping
python sudoku_noisy_experiment.py             # §31 Sudoku + wrong diagnoses (~2 h)
python webqsp_experiment.py                   # §32 WebQSP
python make_readme_figures.py                 # rebuild the figures above
```

The §33 LLM follow-ups add conditions via `LLM_EXTRA_CONDITIONS` (e.g. `noisy_vote2,noisy_verify` or `llm_diag,llm_diag_vote2`). LLM runs resume after quota limits, and every response is cached, so reruns are free.

## Code

| Path | What |
|---|---|
| `fags/graph_generator.py`, `verifier.py` | Synthetic knowledge graphs and verifiers |
| `fags/failure_search.py`, `beam_search.py`, `best_first_search.py`, `mcts_search.py` | Search algorithms (FAGS = `failure_search.py`) |
| `fags/delayed_failure.py` | Diagnostic-failure trees and recovery policies, incl. soft backjumping |
| `fags/sudoku_env.py` | Sudoku with conflict-set diagnosis |
| `fags/metaqa.py`, `webqsp.py` | Relation-path search with BGE verifiers |
| `fags/codegen.py`, `llm.py` | MBPP+ test harness and cached Groq client |
| `fags/committed.py`, `stats.py` | Commit-to-an-answer scoring; clustered tests with Holm correction |

Python · NumPy · SciPy · pandas · Matplotlib · sentence-transformers (BGE) · Groq
