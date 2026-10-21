"""Builds the four README figures in docs/figures/ from the result files in
results/v2/. Each figure is a portrait, two-part summary of one finding:
a headline view plus a supporting view. Rerun after any experiment rerun."""

from __future__ import annotations

import csv
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fags.stats import cluster_bootstrap_ci
from llm_feedback_analysis import submitted

ROOT = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(ROOT, "results", "v2")
OUT = os.path.join(ROOT, "docs", "figures")
os.makedirs(OUT, exist_ok=True)

# Reference palette (dataviz skill), light mode. Slots 1-4 validated as a set
# for adjacent-pair use; the lighter slots always carry direct labels.
S1, S2, S3, S4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, MUTED, GRID, AXIS, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
NEUTRAL = "#b9b7ae"  # baseline conditions
plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 9.5, "text.color": INK,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.facecolor": SURF, "figure.facecolor": SURF, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "legend.frameon": False, "savefig.dpi": 170, "savefig.bbox": "tight",
})
PORTRAIT = (6.2, 8.4)


def header(fig, title, subtitle):
    fig.text(0.02, 0.985, title, fontsize=13, fontweight="bold", color=INK, va="top")
    fig.text(0.02, 0.952, subtitle, fontsize=9.5, color=INK2, va="top")


def panel_title(ax, letter, text):
    ax.set_title(f"{letter}  {text}", loc="left", fontsize=10.5, fontweight="bold", color=INK, pad=8)


def footnote(fig, text):
    fig.text(0.02, 0.012, text, fontsize=8, color=MUTED, va="bottom", wrap=True)


def end_label(ax, x, y, text, color=INK2, dy=0):
    ax.annotate(text, (x, y), xytext=(7, dy), textcoords="offset points", va="center", fontsize=8.5, color=color)


# 1. Failure delay: diagnosis gain grows with how late mistakes surface.
def fig_delay():
    succ = defaultdict(dict)
    for r in csv.DictReader(open(os.path.join(RES, "failure_informativeness.csv"), encoding="utf-8")):
        if r["rho"] == "1.0":
            succ[(float(r["d_prime"]), int(r["max_delay"]))][r["method"]] = float(r["success@100"])
    ks = list(range(6))
    fig, (a1, a2) = plt.subplots(2, 1, figsize=PORTRAIT, gridspec_kw={"hspace": 0.45})
    header(fig, "Failure diagnosis pays off when mistakes surface late",
           "Synthetic search trees: success within 100 expansions, with vs without diagnosis (§23)")
    for d, color in zip((0.5, 1.0, 2.0), (S1, S2, S3)):
        g = [100 * (succ[(d, k)]["DFS+diag"] - succ[(d, k)]["DFS"]) for k in ks]
        a1.plot(ks, g, color=color, lw=2, marker="o", ms=4)
        end_label(a1, ks[-1], g[-1], f"d′ = {d:g}   +{g[-1]:.0f}")
    panel_title(a1, "A", "Gain for depth-first search, by verifier strength d′")
    a1.set_ylabel("success gain (points)")
    a1.text(0.02, 0.93, "weaker verifier → bigger gain", transform=a1.transAxes, fontsize=8.5, color=INK2)
    pol = [("DFS", "DFS", S1, 4), ("FAGS-style", "FAGS-style revival", S2, -5), ("BestFirst-mean", "Best-first", S3, -4),
           ("Posterior", "Bayesian posterior", S4, 0)]
    for base, label, color, dy in pol:
        g = [100 * (succ[(1.0, k)][f"{base}+diag"] - succ[(1.0, k)][base]) for k in ks]
        a2.plot(ks, g, color=color, lw=2, marker="o", ms=4)
        end_label(a2, ks[-1], g[-1], f"{label}   +{g[-1]:.0f}", dy=dy)
    panel_title(a2, "B", "Same effect for every search policy (d′ = 1)")
    a2.set_ylabel("success gain (points)")
    for ax in (a1, a2):
        ax.set_xlim(-0.2, 7.9)
        ax.set_xticks(ks)
        ax.set_xlabel("steps between a mistake and the dead end it causes")
    footnote(fig, "300 trees per point; every failure names its cause (ρ = 1). Gains at delay ≤ 1 are exactly 0: "
                  "an immediate failure already pinpoints the mistake.")
    fig.savefig(os.path.join(OUT, "diagnosis_vs_delay.png"))
    plt.close(fig)


# 2. LLM code repair: small multiples, one panel per model, with 95% CIs.
def fig_llm():
    conds = [("restart", "Fresh retry", NEUTRAL), ("binary", '"It failed"', NEUTRAL),
             ("diagnostic", "Exact diagnosis", S1),
             ("noisy_hard", "50%-wrong diagnosis, as fact", S2), ("noisy_soft", "…flagged as unreliable", S2),
             ("noisy_verify", "…model told to verify", S2), ("noisy_vote2", "…two checkers must agree", S2)]
    models = ["qwen3.8-27b", "gpt-oss-120b", "gpt-oss-20b"]
    fig, axes = plt.subplots(3, 1, figsize=(6.2, 9.4), sharex=True, gridspec_kw={"hspace": 0.35})
    header(fig, "LLMs repair code when told why it failed",
           "MBPP+ problems whose first attempt failed; success on hidden tests within 3 LLM calls (§29, §33a)")
    for ax, tag in zip(axes, models):
        runs = defaultdict(dict)
        for line in open(os.path.join(RES, f"llm_feedback_runs_{tag}_v2.jsonl"), encoding="utf-8"):
            r = json.loads(line)
            runs[r["task_id"]][r["condition"]] = r
        done = sorted(t for t, d in runs.items() if all(c in d for c, _, _ in conds))
        y = np.arange(len(conds))
        for i, (c, label, color) in enumerate(conds):
            xs = [int(submitted(runs[t][c]["attempts"], 3)["hidden_pass"]) for t in done]
            m, lo, hi = (100 * v for v in cluster_bootstrap_ci(xs, done, seed=1))
            ax.barh(i, m, height=0.62, color=color)
            ax.errorbar(m, i, xerr=[[m - lo], [hi - m]], fmt="none", ecolor=INK2, elinewidth=1, capsize=2.5)
            ax.text(hi + 1.2, i, f"{m:.0f}%", va="center", fontsize=8.5, color=INK2)
        ax.set_yticks(y, [label for _, label, _ in conds], color=INK, fontsize=9)
        ax.invert_yaxis()
        ax.grid(axis="y", visible=False)
        ax.set_xlim(0, 92)
        panel_title(ax, "", f"{tag}   (n = {len(done)} problems)")
    axes[-1].set_xlabel("% of initially failing problems solved   (bars: mean, whiskers: 95% bootstrap CI)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (NEUTRAL, S1, S2)]
    fig.legend(handles, ["no diagnosis", "true diagnosis", "unreliable diagnosis (50% wrong)"],
               loc="upper left", bbox_to_anchor=(0.02, 0.935), ncol=3, fontsize=8.5)
    footnote(fig, "Visible tests give the feedback; success is judged only on held-out EvalPlus inputs the model never sees.")
    fig.savefig(os.path.join(OUT, "llm_feedback.png"))
    plt.close(fig)


# 3. Soft backjumping on real Sudoku with noisy diagnoses, both orderings.
def fig_sudoku():
    rows = list(csv.DictReader(open(os.path.join(RES, "sudoku_noisy.csv"), encoding="utf-8")))
    key = [k for k in rows[0] if k.startswith("success@")][0]

    def get(order, mode, eps):
        return 100 * float(next(r[key] for r in rows if r["ordering"] == order and r["search"] == "DFS"
                                 and r["mode"] == mode and float(r["eps"]) == eps))
    eps = [0.1, 0.3, 0.5]
    fig, axes = plt.subplots(2, 1, figsize=PORTRAIT, gridspec_kw={"hspace": 0.45})
    header(fig, "Defer, don't delete: acting on unreliable diagnoses",
           "295 hard Sudoku puzzles, depth-first search, 20,000-step budget (§31)")
    for ax, order, letter, text in ((axes[0], "mrv", "A", "Fail-first ordering (mistakes surface quickly)"),
                                    (axes[1], "static", "B", "Row-by-row ordering (mistakes surface late)")):
        none = get(order, "none", 0.0)
        ax.axhline(none, color=MUTED, lw=1.4, ls=(0, (4, 3)))
        ax.text(0.1, none - 2.5, f"ignore diagnoses  {none:.0f}%", ha="left", va="top", fontsize=8.5, color=MUTED)
        nudge = {"mrv": {"defer": 0, "vote2": 0, "hard": 0}, "static": {"defer": 7, "vote2": 6, "hard": -6}}[order]
        for mode, label, color in (("defer", "Defer (soft backjumping)", S1), ("vote2", "Prune if two agree", S2),
                                   ("hard", "Prune immediately", S3)):
            ys = [get(order, mode, e) for e in eps]
            ax.plot(eps, ys, color=color, lw=2, marker="o", ms=4)
            end_label(ax, eps[-1], ys[-1], f"{label}  {ys[-1]:.0f}%", dy=nudge[mode])
        panel_title(ax, letter, text)
        ax.set_xticks(eps, ["10%", "30%", "50%"])
        ax.set_xlim(0.07, 0.8)
        ax.set_ylim(0, 100)
        ax.set_ylabel("puzzles solved (%)")
        ax.set_xlabel("share of failure diagnoses that are wrong")
    footnote(fig, "A wrong diagnosis blames a random earlier move, possibly one on the solution path. Defer beat both "
                  "pruning rules in all 12 tested conditions and was never worse than ignoring diagnoses.")
    fig.savefig(os.path.join(OUT, "sudoku_soft_backjumping.png"))
    plt.close(fig)


# 4. KGQA: the verifier is the ceiling, and search width doesn't matter.
def fig_kgqa():
    syn = {r["beam_width"]: 100 * float(r["hits1_answer_set"])
           for r in csv.DictReader(open(os.path.join(RES, "error_decomposition.csv"), encoding="utf-8"))
           if r["graph"] == "Medium" and r["verifier"] == "hop-blind"}
    syn_goal = None
    for line in open(os.path.join(RES, "error_decomposition_summary.txt"), encoding="utf-8"):
        if line.startswith("-- Medium"):
            syn_goal = float(line.split("legacy goal-test beam-8 acc=")[1].rstrip("%\n"))
    mq = {r["method"]: r for r in csv.DictReader(open(os.path.join(RES, "metaqa_results.csv"), encoding="utf-8")) if r["hops"] == "3"}
    wq = {r["method"]: r for r in csv.DictReader(open(os.path.join(RES, "webqsp_results.csv"), encoding="utf-8"))}
    groups = ["Synthetic KG", "MetaQA 3-hop", "WebQSP"]
    search = [syn["8"], 100 * float(mq["path-beam-8"]["hits1"]), 100 * float(wq["rel-beam-32"]["hits1"])]
    oracle = [100.0, 100 * float(mq["oracle"]["hits1"]), 100 * float(wq["oracle"]["hits1"])]
    goal = [syn_goal, 100 * float(mq["path-beam-8"]["goal_test_rate"]), 100 * float(wq["rel-beam-32"]["goal_test_rate"])]

    fig, (a1, a2) = plt.subplots(2, 1, figsize=PORTRAIT, gridspec_kw={"hspace": 0.55})
    header(fig, "In knowledge-graph QA the verifier is the bottleneck",
           "Same relation-path search scored three ways, on three datasets (§22, §24, §32)")
    x = np.arange(len(groups))
    w = 0.25
    for i, (vals, label, color) in enumerate(((search, "Search's committed answer", S1),
                                               (oracle, "Best possible path", S2),
                                               (goal, "Old 'touched the answer' score", S3))):
        bars = a1.bar(x + (i - 1) * (w + 0.02), vals, width=w, color=color, label=label)
        for b, v in zip(bars, vals):
            a1.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.0f}", ha="center", fontsize=8, color=INK2)
    a1.set_xticks(x, groups, color=INK)
    a1.set_ylim(0, 115)
    a1.set_ylabel("accuracy (%)")
    a1.grid(axis="x", visible=False)
    a1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3, fontsize=8)
    panel_title(a1, "A", "Large gap to the best path; the old score hid it")

    series = [("Synthetic KG", [(1, syn["1"]), (3, syn["3"]), (8, syn["8"]), (64, syn["64"])], S1),
              ("MetaQA 3-hop", [(1, 100 * float(mq["path-greedy"]["hits1"])), (3, 100 * float(mq["path-beam-3"]["hits1"])),
                                (8, 100 * float(mq["path-beam-8"]["hits1"])), (18, 100 * float(mq["path-beam-18"]["hits1"]))], S2),
              ("WebQSP", [(1, 100 * float(wq["rel-greedy"]["hits1"])), (3, 100 * float(wq["rel-beam-3"]["hits1"])),
                          (8, 100 * float(wq["rel-beam-8"]["hits1"])), (32, 100 * float(wq["rel-beam-32"]["hits1"]))], S3)]
    for (name, pts, color), dy in zip(series, (-6, 0, 6)):
        xs, ys = zip(*pts)
        a2.plot(xs, ys, color=color, lw=2, marker="o", ms=4)
        end_label(a2, xs[-1], ys[-1], f"{name}  {ys[-1]:.0f}%", dy=dy)
    a2.set_xscale("log")
    a2.set_xticks([1, 3, 8, 18, 32, 64], ["1", "3", "8", "18", "32", "64"])
    a2.set_xlim(0.8, 260)
    a2.set_ylim(0, 60)
    a2.set_xlabel("beam width (number of hypotheses kept; 1 = greedy)")
    a2.set_ylabel("accuracy (%)")
    panel_title(a2, "B", "Searching wider doesn't help past ~3 hypotheses")
    footnote(fig, "Synthetic: rule-based verifier, Medium graph, 1,500 queries. MetaQA (1,000 questions) and WebQSP "
                  "(500 questions): BGE embedding verifiers. Accuracy = committed answer in the gold answer set.")
    fig.savefig(os.path.join(OUT, "kgqa_verifier_ceiling.png"))
    plt.close(fig)


if __name__ == "__main__":
    fig_delay()
    fig_llm()
    fig_sudoku()
    fig_kgqa()
    print("wrote", sorted(os.listdir(OUT)))
