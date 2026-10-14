"""MBPP+ code-generation harness for the LLM failure-feedback experiment.

* The agent sees the task text and the first visible assert (standard MBPP
  prompting). Its code is run against all 3 visible asserts; feedback, if
  any, comes only from those.
* Success is judged on EvalPlus's hidden test script (many extra inputs),
  which the agent never sees - so diagnostic feedback cannot simply leak
  the answer key.
* Model-written code runs in a separate `python -E` subprocess with a
  timeout and a minimal environment, in a fresh temporary directory. This isolates the interpreter
  and working directory but is NOT a security sandbox.
"""

from __future__ import annotations

import ast
import json
import os
import random
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "mbppplus" / "test.parquet"
TIMEOUT = 15


@dataclass
class Problem:
    task_id: int
    text: str
    visible: list[str]          # MBPP's 3 base asserts (the first is shown in the prompt)
    hidden: str                 # EvalPlus test script (defines inputs/results/assertion, then loops)
    imports: str
    prefix: str = ""            # hidden script without its final loop
    check: str = ""             # the loop body line, e.g. "assertion(f(*inp), exp, 0)"
    entry: str = ""             # function name under test
    visible_idx: tuple = ()     # EvalPlus input indices whose results the agent may see
    hidden_idx: tuple = ()      # EvalPlus input indices used only for scoring


_LOOP = re.compile(r"^for i, \(inp, exp\) in enumerate\(zip\(inputs, results\)\):\s*$", re.M)
_CHECK = re.compile(r"assertion\((\w+)\(\*inp\)")


def _split(task_id: int, script: str):
    """Split an EvalPlus script into (prefix, check line, entry, visible, hidden).
    The 3 base inputs plus a seeded random half of the rest are visible."""
    m = _LOOP.search(script)
    if m is None:  # scripts using ref_func have no `results`; same loop shape otherwise
        m = re.search(r"^for i, inp in enumerate\(inputs\):\s*$", script, re.M)
    prefix, tail = script[: m.start()], script[m.end():]
    check = tail.strip().splitlines()[0].strip()
    entry = _CHECK.search(check).group(1)
    ns: dict = {}
    exec(compile(prefix, "<evalplus-prefix>", "exec"), ns)  # dataset-provided test data, not model code
    n = len(ns["inputs"])
    rest = list(range(3, n))
    random.Random(task_id).shuffle(rest)
    half = len(rest) // 2
    return prefix, check, entry, tuple(range(min(3, n))) + tuple(sorted(rest[:half])), tuple(sorted(rest[half:]))


def load_problems() -> list[Problem]:
    df = pd.read_parquet(DATA)
    out = []
    for r in df.itertuples():
        p = Problem(int(r.task_id), r.prompt, list(r.test_list), r.test, "\n".join(r.test_imports))
        try:
            p.prefix, p.check, p.entry, p.visible_idx, p.hidden_idx = _split(p.task_id, r.test)
        except Exception:
            pass  # unsplittable script: excluded by the experiment (no hidden_idx)
        out.append(p)
    return out


def task_prompt(p: Problem) -> str:
    return (f"You are an expert Python programmer, and here is your task: {p.text}\n"
            f"Your code should pass this test:\n\n{p.visible[0]}\n\n"
            "Reply with the complete Python code (imports included) in a single ```python code block.")


def extract_code(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, flags=re.S)
    if blocks:
        return max(blocks, key=len).strip()
    return text.strip()


# ── execution ────────────────────────────────────────────────────────────

_VISIBLE_RUNNER = r'''
import ast, json, sys
spec = json.load(open(sys.argv[1], encoding="utf-8"))
ns = {}
def short(x):
    s = repr(x)
    return s if len(s) <= 300 else s[:300] + "..."
out = []
try:
    exec(spec["imports"] + "\n" + spec["code"], ns)
except BaseException as e:
    print(json.dumps([{"test": t, "passed": False, "call": None, "expected": None,
                       "actual": "error while loading the code: " + type(e).__name__ + ": " + str(e)[:200]}
                      for t in spec["tests"]]))
    sys.exit(0)
for t in spec["tests"]:
    rec = {"test": t, "passed": False, "call": None, "expected": None, "actual": None}
    try:
        node = ast.parse(t).body[0]
        cmp = node.test if isinstance(node, ast.Assert) else None
        if isinstance(cmp, ast.Compare) and len(cmp.ops) == 1 and isinstance(cmp.ops[0], ast.Eq):
            rec["call"] = ast.unparse(cmp.left)
            exp = eval(compile(ast.Expression(cmp.comparators[0]), "<t>", "eval"), ns)
            rec["expected"] = short(exp)
            try:
                act = eval(compile(ast.Expression(cmp.left), "<t>", "eval"), ns)
                rec["actual"] = short(act)
                rec["passed"] = bool(act == exp)
            except BaseException as e:
                rec["actual"] = "raised " + type(e).__name__ + ": " + str(e)[:200]
        else:
            try:
                exec(t, ns)
                rec["passed"] = True
            except BaseException as e:
                rec["actual"] = "raised " + type(e).__name__ + ": " + str(e)[:200]
    except BaseException as e:
        rec["actual"] = "raised " + type(e).__name__ + ": " + str(e)[:200]
    out.append(rec)
print(json.dumps(out))
'''


def _run(script: str, args: list[str], workdir: str) -> subprocess.CompletedProcess:
    # -E ignores PYTHON* environment variables. Not -I: that also drops the
    # user site-packages, where numpy (needed by the hidden tests) lives here.
    # The script directory is a fresh temp dir holding only our files.
    env = {k: os.environ[k] for k in ("PATH", "SYSTEMROOT", "APPDATA", "LOCALAPPDATA", "USERPROFILE", "TEMP", "TMP")
           if k in os.environ}
    return subprocess.run([sys.executable, "-E", script, *args], cwd=workdir, capture_output=True,
                          text=True, timeout=TIMEOUT, env=env)


def run_visible(p: Problem, code: str) -> list[dict]:
    with tempfile.TemporaryDirectory() as d:
        runner = Path(d, "runner.py"); runner.write_text(_VISIBLE_RUNNER, encoding="utf-8")
        spec = Path(d, "spec.json"); spec.write_text(json.dumps({"code": code, "tests": p.visible, "imports": p.imports}), encoding="utf-8")
        try:
            r = _run(str(runner), [str(spec)], d)
            return json.loads(r.stdout.strip().splitlines()[-1])
        except subprocess.TimeoutExpired:
            return [{"test": t, "passed": False, "call": None, "expected": None,
                     "actual": f"timed out after {TIMEOUT}s"} for t in p.visible]
        except Exception as e:  # crashed interpreter, unparsable output
            return [{"test": t, "passed": False, "call": None, "expected": None,
                     "actual": f"crashed: {type(e).__name__}"} for t in p.visible]


_SPLIT_RUNNER = r'''
import json, sys
spec = json.load(open(sys.argv[1], encoding="utf-8"))
def short(x):
    s = repr(x)
    return s if len(s) <= 300 else s[:300] + "..."
ns = {}
exec(spec["imports"] + "\n" + spec["prefix"], ns)
try:
    exec(spec["code"], ns)
    load_err = None
except BaseException as e:
    load_err = "error while loading the code: " + type(e).__name__ + ": " + str(e)[:200]
out = []
for i in spec["idx"]:
    inp = ns["inputs"][i]
    ns["inp"] = inp
    exp = ns["results"][i] if "results" in ns else ns["ref_func"](*inp)
    ns["exp"] = exp
    args = ", ".join(short(a) for a in inp)
    rec = {"test": spec["entry"] + "(" + args + ")", "call": spec["entry"] + "(" + args + ")",
           "expected": short(exp), "actual": None, "passed": False}
    if load_err:
        rec["actual"] = load_err
    else:
        try:
            act = eval(spec["entry"] + "(*inp)", ns)
            rec["actual"] = short(act)
            ns["__act"] = act
            exec(spec["check"].replace(spec["entry"] + "(*inp)", "__act", 1), ns)
            rec["passed"] = True
        except AssertionError:
            pass
        except BaseException as e:
            rec["actual"] = "raised " + type(e).__name__ + ": " + str(e)[:200]
    out.append(rec)
print(json.dumps(out))
'''


def run_split(p: Problem, code: str, which: str) -> list[dict]:
    """Run the agent's code on the visible or hidden EvalPlus split, one
    record per input (call, expected, actual, passed)."""
    idx = list(p.visible_idx if which == "visible" else p.hidden_idx)
    with tempfile.TemporaryDirectory() as d:
        runner = Path(d, "runner.py"); runner.write_text(_SPLIT_RUNNER, encoding="utf-8")
        spec = Path(d, "spec.json")
        spec.write_text(json.dumps({"code": code, "imports": p.imports, "prefix": p.prefix, "check": p.check,
                                    "entry": p.entry, "idx": idx}), encoding="utf-8")
        try:
            r = _run(str(runner), [str(spec)], d)
            return json.loads(r.stdout.strip().splitlines()[-1])
        except subprocess.TimeoutExpired:
            return [{"test": f"input #{i}", "call": None, "expected": None, "passed": False,
                     "actual": f"timed out after {TIMEOUT}s"} for i in idx]
        except Exception as e:
            return [{"test": f"input #{i}", "call": None, "expected": None, "passed": False,
                     "actual": f"crashed: {type(e).__name__}"} for i in idx]


def run_hidden(p: Problem, code: str) -> bool:
    with tempfile.TemporaryDirectory() as d:
        f = Path(d, "hidden.py")
        f.write_text(p.imports + "\n" + code + "\n\n" + p.hidden, encoding="utf-8")
        try:
            return _run(str(f), [], d).returncode == 0
        except subprocess.TimeoutExpired:
            return False


# ── feedback messages ────────────────────────────────────────────────────

def _perturb(expected_repr: str, others: list[str], rng: random.Random) -> str:
    """A plausible but wrong expected value: another visible test's expected
    value when it differs, else a type-preserving perturbation."""
    alts = [o for o in others if o and o != expected_repr]
    if alts:
        return rng.choice(alts)
    try:
        v = ast.literal_eval(expected_repr)
    except Exception:
        return "None"
    if isinstance(v, bool):
        return repr(not v)
    if isinstance(v, int):
        return repr(v + rng.choice([-2, -1, 1, 2]))
    if isinstance(v, float):
        return repr(round(v * 1.5 + 1, 4))
    if isinstance(v, str):
        return repr(v[::-1] if len(v) > 1 and v[::-1] != v else v + "a")
    if isinstance(v, (list, tuple)):
        w = list(v)[::-1] if len(v) > 1 and list(v)[::-1] != list(v) else list(v) + list(v)[:1] or [0]
        return repr(type(v)(w))
    if isinstance(v, (set, dict)):
        return repr(type(v)())
    return "None"


def failure_reports(results: list[dict], corrupt_prob: float = 0.0, rng: random.Random | None = None,
                    max_reports: int = 3) -> list[dict]:
    """Structured version of describe_failures: one dict per reported failing
    test with the expected value the checker shows (possibly corrupted).
    Draws from rng in exactly the same order as describe_failures, so the same
    rng seed yields the same corruptions in both."""
    rng = rng or random.Random(0)
    all_expected = [r["expected"] for r in results]
    out = []
    for r in [r for r in results if not r["passed"]][:max_reports]:
        item = {"call": r["call"], "test": r["test"], "actual": r["actual"], "true_expected": r["expected"],
                "shown_expected": r["expected"], "corrupted": False}
        if r["call"] is not None and r["expected"] is not None and corrupt_prob > 0 and rng.random() < corrupt_prob:
            item["shown_expected"] = _perturb(r["expected"], all_expected, rng)
            item["corrupted"] = True
        out.append(item)
    return out


def describe_failures(results: list[dict], corrupt_prob: float = 0.0, rng: random.Random | None = None,
                      max_reports: int = 3) -> tuple[str, int]:
    """Human-readable report of (at most max_reports) failing visible tests.
    With corrupt_prob > 0 each reported test's expected value is replaced by a
    wrong one with that probability. Returns (text, number_of_corrupted_reports)."""
    rng = rng or random.Random(0)
    lines, corrupted = [], 0
    all_expected = [r["expected"] for r in results]
    for r in [r for r in results if not r["passed"]][:max_reports]:
        if r["call"] is None or r["expected"] is None:
            lines.append(f"- `{r['test']}` failed: {r['actual']}")
            continue
        exp = r["expected"]
        if corrupt_prob > 0 and rng.random() < corrupt_prob:
            exp = _perturb(exp, all_expected, rng)
            corrupted += 1
        lines.append(f"- `{r['call']}` should return `{exp}`, but your code {('returned `' + r['actual'] + '`') if not str(r['actual']).startswith(('raised', 'error', 'timed')) else r['actual']}.")
    return "\n".join(lines), corrupted
