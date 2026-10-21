"""Builds the four README figures in docs/figures/ from the result files in
results/v2/ (no numbers are typed in by hand except where noted)."""

from __future__ import annotations

import csv
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from llm_feedback_analysis import submitted

ROOT = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(ROOT, "results", "v2")
OUT = os.path.join(ROOT, "docs", "figures")
os.makedirs(OUT, exist_ok=True)

# Reference palette (dataviz skill), light mode; slots 1-3 validated together.
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID, AXIS, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 10, "text.color": INK,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.facecolor": SURF, "figure.facecolor": SURF, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "legend.frameon": False, "savefig.dpi": 160, "savefig.bbox": "tight",
})


def title(ax, main, sub):
    ax.set_title(main, loc="left", fontsize=12, fontweight="bold", color=INK, pad=22)
    ax.text(0, 1.02, sub, transform=ax.transAxes, fontsize=9, color=INK2, va="bottom")


# 1. Why the failure was made matters: diagnosis gain vs failure delay.
def fig_delay():
    succ = defaultdict(dict)
    for r in csv.DictReader(open(os.path.join(RES, "failure_informativeness.csv"), encoding="utf-8")):
        if r["rho"] == "1.0":
            succ[(float(r["d_prime"]), int(r["max_delay"]))][r["method"]] = float(r["success@100"])
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for d, color in zip((0.5, 1.0, 2.0), (S1, S2, S3)):
        ks = range(6)
        gain = [100 * (succ[(d, k)]["DFS+diag"] - succ[(d, k)]["DFS"]) for k in ks]
        ax.plot(ks, gain, color=color, lw=2, marker="o", ms=4)
        ax.annotate(f"verifier d′={d:g}  (+{gain[-1]:.0f})", (5, gain[-1]), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=9, color=INK2)
    ax.set_xlabel("steps between a mistake and the dead end it causes")
    ax.set_ylabel("success gain (points)")
    ax.set_xlim(-0.2, 7.6)
    ax.set_xticks(range(6))
    title(ax, "Knowing why a failure happened pays off when mistakes surface late",
          "Synthetic search trees: DFS with vs without failure diagnosis, 300 trees per point (§23)")
    fig.savefig(os.path.join(OUT, "diagnosis_vs_delay.png"))
    plt.close(fig)


# 2. LLM code repair: what the model is told after a failing test.
def fig_llm():
    conds = [("restart", "Fresh retry"), ("binary", "\"It failed\""), ("diagnostic", "Exact diagnosis"),
             ("noisy_hard", "Diagnosis, 50% wrong"), ("noisy_vote2", "Two checkers must agree")]
    models = [("qwen3.8-27b", "qwen3.8-27b"), ("gpt-oss-120b", "gpt-oss-120b"), ("gpt-oss-20b", "gpt-oss-20b")]
    vals = {}
    for tag, _ in models:
        runs = defaultdict(dict)
        for line in open(os.path.join(RES, f"llm_feedback_runs_{tag}_v2.jsonl"), encoding="utf-8"):
            r = json.loads(line)
            runs[r["task_id"]][r["condition"]] = r
        done = [t for t, d in runs.items() if all(c in d for c, _ in conds)]
        vals[tag] = [100 * np.mean([submitted(runs[t][c]["attempts"], 3)["hidden_pass"] for t in done]) for c, _ in conds]
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    y = np.arange(len(conds))
    h = 0.24
    for i, ((tag, label), color) in enumerate(zip(models, (S1, S2, S3))):
        bars = ax.barh(y + (i - 1) * (h + 0.02), vals[tag], height=h, color=color, label=label)
        for b, v in zip(bars, vals[tag]):
            ax.text(v + 0.8, b.get_y() + b.get_height() / 2, f"{v:.0f}", va="center", fontsize=8, color=INK2)
    ax.set_yticks(y, [c[1] for c in conds], color=INK)
    ax.invert_yaxis()
    ax.set_xlim(0, 80)
    ax.set_xlabel("% of initially failing problems solved (hidden tests, 3 LLM calls)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right", fontsize=9)
    title(ax, "LLMs repair code when told why it failed — not just that it failed",
          "MBPP+ code repair, 73–79 initially failing problems per model (§29, §33a)")
    fig.savefig(os.path.join(OUT, "llm_feedback.png"))
    plt.close(fig)


# 3. Soft backjumping on real Sudoku with noisy diagnoses.
def fig_sudoku():
    rows = list(csv.DictReader(open(os.path.join(RES, "sudoku_noisy.csv"), encoding="utf-8")))
    key = [k for k in rows[0] if k.startswith("success@")][0]
    get = lambda mode, eps: 100 * float(next(r[key] for r in rows if r["ordering"] == "mrv" and r["search"] == "DFS"
                                              and r["mode"] == mode and float(r["eps"]) == eps))
    eps = [0.1, 0.3, 0.5]
    none = get("none", 0.0)
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.axhline(none, color=MUTED, lw=1.5, ls=(0, (4, 3)))
    ax.text(0.1, none - 2, f"ignore diagnoses ({none:.0f}%)", ha="left", va="top", fontsize=9, color=MUTED)
    for mode, label, color in (("defer", "Defer (soft backjumping)", S1), ("vote2", "Prune if two agree", S2),
                               ("hard", "Prune immediately", S3)):
        ys = [get(mode, e) for e in eps]
        ax.plot(eps, ys, color=color, lw=2, marker="o", ms=4)
        ax.annotate(f"{label}  {ys[-1]:.0f}%", (eps[-1], ys[-1]), xytext=(8, 0), textcoords="offset points",
                    va="center", fontsize=9, color=INK2)
    ax.set_xticks(eps, ["10%", "30%", "50%"])
    ax.set_xlim(0.07, 0.78)
    ax.set_ylim(0, 100)
    ax.set_xlabel("share of failure diagnoses that are wrong")
    ax.set_ylabel("puzzles solved (%)")
    title(ax, "Defer, don't delete: robust use of unreliable diagnoses",
          "295 hard Sudoku puzzles, DFS with fail-first ordering, 20,000-step budget (§31)")
    fig.savefig(os.path.join(OUT, "sudoku_soft_backjumping.png"))
    plt.close(fig)


# 4. KGQA: the verifier is the ceiling, and goal-test scoring hides it.
def fig_kgqa():
    # Synthetic (Medium graph): best committed search = hop-blind beam-8; legacy beam-8 goal-test
    # rate and oracle-plan ceiling from error_decomposition_summary.txt (§22).
    syn_search = syn_goal = None
    for r in csv.DictReader(open(os.path.join(RES, "error_decomposition.csv"), encoding="utf-8")):
        if r["graph"] == "Medium" and r["beam_width"] == "8":
            if r["verifier"] == "hop-blind":
                syn_search = 100 * float(r["hits1_answer_set"])
    for line in open(os.path.join(RES, "error_decomposition_summary.txt"), encoding="utf-8"):
        if line.startswith("-- Medium"):
            syn_goal = float(line.split("legacy goal-test beam-8 acc=")[1].rstrip("%\n"))
    mq = {r["method"]: r for r in csv.DictReader(open(os.path.join(RES, "metaqa_results.csv"), encoding="utf-8")) if r["hops"] == "3"}
    wq = {r["method"]: r for r in csv.DictReader(open(os.path.join(RES, "webqsp_results.csv"), encoding="utf-8"))}
    groups = ["Synthetic KG", "MetaQA 3-hop", "WebQSP"]
    search = [syn_search, 100 * float(mq["path-beam-8"]["hits1"]), 100 * float(wq["rel-beam-32"]["hits1"])]
    oracle = [100.0, 100 * float(mq["oracle"]["hits1"]), 100 * float(wq["oracle"]["hits1"])]
    goal = [syn_goal, 100 * float(mq["path-beam-8"]["goal_test_rate"]), 100 * float(wq["rel-beam-32"]["goal_test_rate"])]
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    x = np.arange(len(groups))
    w = 0.25
    for i, (vals, label, color) in enumerate(((search, "Search's committed answer", S1),
                                               (oracle, "Best possible path", S2),
                                               (goal, "Old 'touched the answer' score", S3))):
        bars = ax.bar(x + (i - 1) * (w + 0.02), vals, width=w, color=color, label=label)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.0f}", ha="center", fontsize=8, color=INK2)
    ax.set_xticks(x, groups, color=INK)
    ax.set_ylim(0, 112)
    ax.set_ylabel("accuracy (%)")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=8.5)
    title(ax, "In knowledge-graph QA the verifier, not the search, is the bottleneck",
          "Same search at its best width; MetaQA 3-hop and WebQSP use real BGE embedding verifiers (§22, §24, §32)")
    fig.savefig(os.path.join(OUT, "kgqa_verifier_ceiling.png"))
    plt.close(fig)


if __name__ == "__main__":
    fig_delay()
    fig_llm()
    fig_sudoku()
    fig_kgqa()
    print("wrote", sorted(os.listdir(OUT)))
