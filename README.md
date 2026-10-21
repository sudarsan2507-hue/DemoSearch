# DemoSearch (FAGS)

**When does learning from failure help a search agent?**

This started as FAGS (Failure-Aware Graph Search): a knowledge-graph search agent that remembers branches it rejected and revives the best one at a dead end. It grew into a study of failure feedback across synthetic search, Sudoku, two knowledge-graph QA datasets and LLM code repair. Every number below is in [`PROJECT_STATUS.md`](PROJECT_STATUS.md); **start at §21** (earlier sections predate a generator bug fix).

## Key results

**1. A failure only helps if it says *why*, and then it helps most when mistakes surface late.** A bare "dead end" can't change which option to try next (proved in §23), so reviving the best-scoring rejected branch, FAGS's core idea, never beat plain systematic search. A diagnosis that names the cause adds +15 to +70 points once mistakes take 5 steps to surface.

![Diagnosis gain vs failure delay](docs/figures/diagnosis_vs_delay.png)

**2. The same holds for LLMs repairing code.** Telling the model *why* a test failed raises repair success 2.5–3.3× over a fresh retry on all three models; telling it only *that* it failed helps much less (and not at all for two of the three). Wrong diagnoses erase the gain, and requiring two checkers to agree recovers much of it (§29, §33a).

![LLM code repair by feedback type](docs/figures/llm_feedback.png)

**3. Defer, don't delete.** When some diagnoses are wrong, classic backjumping (prune what the diagnosis blames) can delete the solution. Deferring the blamed branch instead ("soft backjumping", §30) costs nothing when diagnoses are right and stays above ignoring them even when half are wrong (§31).

![Soft backjumping on Sudoku with noisy diagnoses](docs/figures/sudoku_soft_backjumping.png)

**4. In knowledge-graph QA the verifier is the bottleneck, not the search.** Beam width beyond ~3 changes nothing, the best possible path beats what the search picks by 50–70 points, and scoring a search by whether it ever *touched* the answer inflates accuracy 2–3× (§22, §24, §32).

![Verifier ceiling in knowledge-graph QA](docs/figures/kgqa_verifier_ceiling.png)

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
