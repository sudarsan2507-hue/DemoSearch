"""Real LLM agent task: does failure feedback help code repair only when it
is diagnostic, and does noisy diagnosis need to be treated as evidence?
(PROJECT_STATUS.md SS29)

Task: MBPP+ (EvalPlus). The agent sees the task and one example assert.
EvalPlus's extended inputs are split per problem (seeded): the 3 base
inputs plus a random half of the rest are VISIBLE (the agent's code is run
on them and failures can be reported back); the other half is HIDDEN and
used only for scoring. (A first design used only the 3 base asserts as the
visible suite; first attempts almost never failed them - 14/300 for
gpt-oss-20b - so there was nothing to repair.) The agent may repair up to
MAX_ATTEMPTS-1 times while a visible input fails. Success = passing every
hidden input. All conditions share the same first attempt, so they differ
only in what the agent is told after a visible failure, at an equal budget
of LLM calls:

  restart      fresh independent sample, no information (memoryless control)
  binary       "your code failed a test" only - a non-diagnostic failure
               (FAGS-style: knows it failed, not why)
  diagnostic   the failing visible tests: call, expected value, actual result
  noisy_hard   same report, but each expected value is wrong with prob 0.5;
               presented as fact
  noisy_soft   the identical corrupted reports, presented as unreliable
               evidence ("about half are wrong; check against the task")
  self_diag    binary + the model must first explain the likely bug itself
               (a natural, noisy diagnoser)

Pre-registered hypotheses (from SS23/SS27):
  L1  binary is not better than restart (non-diagnostic failure).
  L2  diagnostic beats restart and binary.
  L3  noisy_hard is worse than diagnostic, and not better than restart.
  L4  noisy_soft beats noisy_hard and is not worse than restart.
  L5  (exploratory) self_diag is not better than binary.
self_diag is implemented but not in CONDITIONS: it was dropped to fit the
free-tier token quota (200k tokens/day for gpt-oss models).

Resumable: run again after a daily quota stop; finished work is skipped and
LLM calls are cached on disk.

Usage: python -m experiments.llm_feedback_experiment [model]   (default qwen/qwen3.8-27b)
"""

from __future__ import annotations

import ast
import json
import random
import re
import sys
import time
from pathlib import Path

import os

from fags.codegen import load_problems, task_prompt, extract_code, run_split, describe_failures, failure_reports
from fags.llm import GroqLLM, QuotaExhausted

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "results" / "v2"
MAX_ATTEMPTS = 3
TEMPERATURE = 0.8
CORRUPT_PROB = 0.5
EXCLUDED_PATH = OUT_DIR / "mbppplus_excluded.json"  # written by the reference-solution check
CONDITIONS = ["restart", "binary", "diagnostic", "noisy_hard", "noisy_soft"]
# Follow-up conditions (SS29b), run only when listed in LLM_EXTRA_CONDITIONS so
# that runs already in progress on other models are unaffected:
#   noisy_vote2   two independent unreliable checkers; a report is stated as
#                 fact only when both agree, otherwise both values are shown
#                 (the LLM analogue of SS27's two-vote rule). Checker A is
#                 identical to noisy_hard's checker.
#   noisy_verify  noisy_soft's warning plus an explicit procedure: derive the
#                 correct output from the task first, judge each report, then fix.
EXTRA_CONDITIONS = [c for c in os.environ.get("LLM_EXTRA_CONDITIONS", "").split(",") if c]
FIX = "Write a corrected version. Reply with the complete Python code in a single ```python code block."


def _line(call: str, expected: str, actual: str) -> str:
    did = ("returned `" + actual + "`") if not str(actual).startswith(("raised", "error", "timed")) else actual
    return f"- `{call}` should return `{expected}`, but your code {did}."


def vote2_message(results: list[dict], rng_key: str) -> tuple[str, int]:
    a = failure_reports(results, CORRUPT_PROB, random.Random(rng_key))
    b = failure_reports(results, CORRUPT_PROB, random.Random(rng_key + "|B"))
    lines, wrong_as_fact = [], 0
    for ra, rb in zip(a, b):
        if ra["call"] is None:
            lines.append(f"- `{ra['test']}` failed: {ra['actual']}")
        elif ra["shown_expected"] == rb["shown_expected"]:
            lines.append(_line(ra["call"], ra["shown_expected"], ra["actual"]))
            wrong_as_fact += ra["shown_expected"] != ra["true_expected"]
        else:
            lines.append(f"- `{ra['call']}` failed (your code returned `{ra['actual']}`). The two checkers disagree "
                         f"on the expected result: `{ra['shown_expected']}` or `{rb['shown_expected']}` - at most one is right.")
    return ("Two independent automated checkers tested your code. Each is unreliable on its own (about half of "
            "the expected values it reports are wrong); where they agree the value is shown as expected, where "
            "they disagree both values are shown.\n" + "\n".join(lines) + f"\n\n{FIX}"), wrong_as_fact


def feedback_message(cond: str, results: list[dict], rng_key: str) -> tuple[str, int]:
    if cond == "binary":
        return "Your code failed at least one test. " + FIX, 0
    if cond == "self_diag":
        return ("Your code failed at least one test (no further details are available). First explain briefly "
                "what is most likely wrong, then write a corrected version in a single ```python code block."), 0
    if cond == "diagnostic":
        report, _ = describe_failures(results)
        return f"Your code failed these tests:\n{report}\n\n{FIX}", 0
    report, n_bad = describe_failures(results, CORRUPT_PROB, random.Random(rng_key))
    if cond == "noisy_hard":
        return f"Your code failed these tests:\n{report}\n\n{FIX}", n_bad
    if cond == "noisy_soft":
        return ("An automated checker reported the problems below. The checker is unreliable: about half of the "
                "expected values it reports are wrong. Treat them as hints only - check each against the task "
                "description and the example test, and ignore any that contradict them.\n"
                f"{report}\n\n{FIX}"), n_bad
    if cond == "noisy_verify":
        return ("An automated checker reported the problems below. The checker is unreliable: about half of the "
                "expected values it reports are wrong.\n"
                f"{report}\n\n"
                "Before changing any code: for each listed call, work out from the task description what the correct "
                "result is, and state whether the checker's expected value is right or wrong. Then write a corrected "
                "version based on your own conclusions, in a single ```python code block."), n_bad
    if cond == "noisy_vote2":
        return vote2_message(results, rng_key)
    raise ValueError(cond)


def _same_value(a: str, b: str) -> bool:
    try:
        return ast.literal_eval(a) == ast.literal_eval(b)
    except Exception:
        return a.strip() == b.strip()


def llm_diagnoses(llm, p, code: str, results: list[dict], seed: int) -> dict[int, str]:
    """A separate reviewer call (same model) predicts the correct result of each
    reported failing call from the task alone - it never sees the true
    expected values, so its mistakes are natural, not injected.
    Returns {report index: predicted value as text}."""
    reps = [r for r in failure_reports(results) if r["call"] is not None]
    if not reps:
        return {}
    lines = "\n".join(f"{i + 1}. `{r['call']}` returned `{r['actual']}`" for i, r in enumerate(reps))
    prompt = (f"You are reviewing a Python solution that fails some tests.\nTask: {p.text}\n"
              f"Example test from the task: {p.visible[0]}\n\nCode:\n```python\n{code}\n```\n\n"
              "For each call below, the code's actual result is shown. Work out from the task what the correct "
              "result should be.\n" + lines +
              "\n\nReply with one line per call in the form `<number>: <correct result as a Python literal>` "
              "and nothing else.")
    text = llm.chat([{"role": "user", "content": prompt}], seed=seed)
    out = {}
    for line in re.sub(r"<think>.*?</think>", "", text, flags=re.S).splitlines():
        m = re.match(r"\s*`?(\d+)[:.)]\s*`?(.+?)`?\s*$", line)
        if m and 1 <= int(m.group(1)) <= len(reps):
            out[int(m.group(1)) - 1] = m.group(2).strip()
    return out


def llm_diag_message(llm, p, code: str, results: list[dict], seed: int, votes: int) -> tuple[str, dict]:
    """llm_diag (votes=1): the reviewer's predictions reported as fact.
    llm_diag_vote2 (votes=2): two independent reviewers; a value is reported as
    fact only when both agree, otherwise both candidates are shown."""
    reps = [r for r in failure_reports(results) if r["call"] is not None]
    preds = [llm_diagnoses(llm, p, code, results, seed + 50 * v) for v in range(votes)]
    stats = {"diag_reports": 0, "diag_correct": 0, "wrong_as_fact": 0, "diag_calls": votes}
    lines = [f"- `{r['test']}` failed: {r['actual']}" for r in failure_reports(results) if r["call"] is None]
    for i, r in enumerate(reps):
        cand = [pr.get(i) for pr in preds]
        stats["diag_reports"] += 1
        stats["diag_correct"] += bool(cand[0]) and _same_value(cand[0], r["true_expected"])
        if votes == 1 or (all(cand) and _same_value(cand[0], cand[1])):
            if cand[0]:
                lines.append(_line(r["call"], cand[0], r["actual"]))
                stats["wrong_as_fact"] += not _same_value(cand[0], r["true_expected"])
            else:
                lines.append(f"- `{r['call']}` gave a wrong result (your code returned `{r['actual']}`).")
        else:
            shown = " or ".join(f"`{c}`" for c in cand if c) or "unknown"
            lines.append(f"- `{r['call']}` failed (your code returned `{r['actual']}`). Two reviewers disagree on "
                         f"the expected result: {shown}.")
    head = ("Your code failed these tests:\n" if votes == 1 else
            "Two independent reviewers checked your code; where they agree the expected value is shown, where "
            "they disagree both are shown.\n")
    return head + "\n".join(lines) + f"\n\n{FIX}", stats


def attempt_record(p, text: str) -> dict:
    code = extract_code(text)
    vis = run_split(p, code, "visible")
    hid = run_split(p, code, "hidden")
    return {"code": code, "visible_pass": all(r["passed"] for r in vis), "visible": vis,
            "hidden_pass": all(r["passed"] for r in hid)}


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "qwen/qwen3.8-27b"
    tag = model.split("/")[-1]
    first_path = OUT_DIR / f"llm_feedback_first_{tag}_v2.jsonl"
    runs_path = OUT_DIR / f"llm_feedback_runs_{tag}_v2.jsonl"
    llm = GroqLLM(model, temperature=TEMPERATURE, max_tokens=1536)
    excluded = set(json.loads(EXCLUDED_PATH.read_text()))
    problems = [p for p in load_problems() if p.hidden_idx and p.task_id not in excluded]
    t0 = time.time()

    first = {}
    if first_path.exists():
        for line in first_path.open(encoding="utf-8"):
            r = json.loads(line); first[r["task_id"]] = r
    done = set()
    if runs_path.exists():
        for line in runs_path.open(encoding="utf-8"):
            r = json.loads(line); done.add((r["task_id"], r["condition"]))

    try:
        # Phase 1: the shared first attempt for every problem.
        for p in problems:
            if p.task_id in first:
                continue
            text = llm.chat([{"role": "user", "content": task_prompt(p)}], seed=0)
            rec = {"task_id": p.task_id, "response": text, **attempt_record(p, text)}
            first[p.task_id] = rec
            with first_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
            if len(first) % 50 == 0:
                print(f"[{time.time() - t0:6.0f}s] first attempts {len(first)}/{len(problems)} "
                      f"(visible-fail so far {sum(not r['visible_pass'] for r in first.values())}), calls={llm.calls}", flush=True)

        # Phase 2: repair conditions, problem by problem (all 6 together).
        failing = [p for p in problems if not first[p.task_id]["visible_pass"]]
        print(f"first attempt: {len(problems) - len(failing)}/{len(problems)} pass visible; "
              f"hidden pass {sum(first[p.task_id]['hidden_pass'] for p in problems)}; repairing {len(failing)}", flush=True)
        for k, p in enumerate(failing):
            for ci, cond in enumerate(CONDITIONS + EXTRA_CONDITIONS):
                if (p.task_id, cond) in done:
                    continue
                f0 = first[p.task_id]
                history = [{"role": "user", "content": task_prompt(p)}, {"role": "assistant", "content": f0["response"]}]
                attempts = [{k2: f0[k2] for k2 in ("code", "visible_pass", "hidden_pass")}]
                last_vis = f0["visible"]
                corrupted = 0
                diag_stats = {"diag_reports": 0, "diag_correct": 0, "wrong_as_fact": 0, "diag_calls": 0}
                for a in range(1, MAX_ATTEMPTS):
                    if attempts[-1]["visible_pass"]:
                        break
                    seed = 1000 * (ci + 1) + a
                    if cond == "restart":
                        text = llm.chat([{"role": "user", "content": task_prompt(p)}], seed=seed)
                    elif cond in ("llm_diag", "llm_diag_vote2"):
                        msg, st = llm_diag_message(llm, p, attempts[-1]["code"], last_vis, seed + 7,
                                                   votes=1 if cond == "llm_diag" else 2)
                        for key in diag_stats:
                            diag_stats[key] += st[key]
                        history.append({"role": "user", "content": msg})
                        text = llm.chat(history, seed=seed)
                        history.append({"role": "assistant", "content": text})
                    else:
                        msg, n_bad = feedback_message(cond, last_vis, f"{p.task_id}|{a}")
                        corrupted += n_bad
                        history.append({"role": "user", "content": msg})
                        text = llm.chat(history, seed=seed)
                        history.append({"role": "assistant", "content": text})
                    rec = attempt_record(p, text)
                    last_vis = rec["visible"]
                    attempts.append({k2: rec[k2] for k2 in ("code", "visible_pass", "hidden_pass")})
                out = {"task_id": p.task_id, "condition": cond, "model": model, "attempts": attempts,
                       "corrupted_reports": corrupted}
                if cond.startswith("llm_diag"):
                    out.update(diag_stats)
                with runs_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(out) + "\n")
                done.add((p.task_id, cond))
            print(f"[{time.time() - t0:6.0f}s] repaired {k + 1}/{len(failing)} problems, calls={llm.calls} "
                  f"cache_hits={llm.cache_hits}", flush=True)
        print("ALL DONE", flush=True)
    except QuotaExhausted as e:
        print(f"QUOTA EXHAUSTED after {llm.calls} calls - re-run later to resume. ({e})", flush=True)


if __name__ == "__main__":
    main()
