"""Builds the four README figures in docs/figures/ from the result files in
results/v2/, in a conventional paper style: serif type, boxed axes, panel
labels (a)/(b), boxed legends, distinct markers per series, no titles inside
the image (the captions live in the README). Rerun after any experiment rerun."""

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
from experiments.llm_feedback_analysis import submitted

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results", "v2")
OUT = os.path.join(ROOT, "docs", "figures")
os.makedirs(OUT, exist_ok=True)

# Colour-blind-safe categorical colours (validated as a set); every series also
# gets its own marker and the legend names it, so colour is never needed alone.
C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
GRAY = "#9a9a9a"
MARKERS = ["o", "s", "^", "D"]
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "mathtext.fontset": "stix",
    "font.size": 10, "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 8.5,
    "axes.linewidth": 0.8, "axes.edgecolor": "black", "axes.facecolor": "white", "figure.facecolor": "white",
    "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
    "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "axes.grid": True, "grid.color": "#d9d9d9", "grid.linestyle": ":", "grid.linewidth": 0.6, "axes.axisbelow": True,
    "legend.frameon": True, "legend.edgecolor": "black", "legend.fancybox": False, "legend.framealpha": 1.0,
    "lines.linewidth": 1.4, "lines.markersize": 5, "savefig.dpi": 200, "savefig.bbox": "tight",
})
PORTRAIT = (4.6, 6.6)


def label(ax, letter):
    ax.text(-0.14, 1.04, f"({letter})", transform=ax.transAxes, fontsize=11, fontweight="bold", va="bottom")


# Figure 1. Diagnosis gain vs failure delay.
def fig_delay():
    succ = defaultdict(dict)
    for r in csv.DictReader(open(os.path.join(RES, "failure_informativeness.csv"), encoding="utf-8")):
        if r["rho"] == "1.0":
            succ[(float(r["d_prime"]), int(r["max_delay"]))][r["method"]] = float(r["success@100"])
    ks = list(range(6))
    fig, (a1, a2) = plt.subplots(2, 1, figsize=PORTRAIT, gridspec_kw={"hspace": 0.38})
    for (d, color), m in zip(zip((0.5, 1.0, 2.0), (C1, C2, C3)), MARKERS):
        a1.plot(ks, [100 * (succ[(d, k)]["DFS+diag"] - succ[(d, k)]["DFS"]) for k in ks],
                color=color, marker=m, label=f"$d'={d:g}$")
    a1.legend(title="Verifier strength", loc="upper left", title_fontsize=8.5)
    pol = [("DFS", "Depth-first"), ("FAGS-style", "FAGS-style revival"), ("BestFirst-mean", "Best-first"),
           ("Posterior", "Bayesian posterior")]
    for (base, name), color, m in zip(pol, (C1, C2, C3, C4), MARKERS):
        a2.plot(ks, [100 * (succ[(1.0, k)][f"{base}+diag"] - succ[(1.0, k)][base]) for k in ks],
                color=color, marker=m, label=name)
    a2.legend(title="Search policy ($d'=1$)", loc="upper left", title_fontsize=8.5)
    for ax, letter in ((a1, "a"), (a2, "b")):
        ax.set_xticks(ks)
        ax.set_xlim(-0.3, 5.3)
        ax.set_ylim(-4, 75)
        ax.set_xlabel("Failure delay $K$ (steps from mistake to dead end)")
        ax.set_ylabel("Success gain from diagnosis (pp)")
        label(ax, letter)
    fig.savefig(os.path.join(OUT, "diagnosis_vs_delay.png"))
    plt.close(fig)


# Figure 2. LLM code repair by feedback type, one panel per model.
def fig_llm():
    conds = [("restart", "Fresh retry", GRAY), ("binary", "Failure signal only", GRAY),
             ("diagnostic", "Exact diagnosis", C1),
             ("noisy_hard", "Noisy diagnosis (as fact)", C2), ("noisy_soft", "Noisy, flagged unreliable", C2),
             ("noisy_verify", "Noisy, model verifies", C2), ("noisy_vote2", "Noisy, two checkers agree", C2)]
    models = ["qwen3.8-27b", "gpt-oss-120b", "gpt-oss-20b"]
    fig, axes = plt.subplots(3, 1, figsize=(4.6, 8.2), sharex=True, gridspec_kw={"hspace": 0.32})
    for ax, tag, letter in zip(axes, models, "abc"):
        runs = defaultdict(dict)
        for line in open(os.path.join(RES, f"llm_feedback_runs_{tag}_v2.jsonl"), encoding="utf-8"):
            r = json.loads(line)
            runs[r["task_id"]][r["condition"]] = r
        done = sorted(t for t, d in runs.items() if all(c in d for c, _, _ in conds))
        for i, (c, _, color) in enumerate(conds):
            xs = [int(submitted(runs[t][c]["attempts"], 3)["hidden_pass"]) for t in done]
            m, lo, hi = (100 * v for v in cluster_bootstrap_ci(xs, done, seed=1))
            ax.barh(i, m, height=0.65, color=color, edgecolor="black", linewidth=0.5)
            ax.errorbar(m, i, xerr=[[m - lo], [hi - m]], fmt="none", ecolor="black", elinewidth=0.8, capsize=2)
        ax.set_yticks(range(len(conds)), [name for _, name, _ in conds])
        ax.invert_yaxis()
        ax.grid(axis="y", visible=False)
        ax.set_xlim(0, 85)
        ax.tick_params(axis="y", length=0)
        ax.text(0.98, 0.05, f"{tag}\n$n={len(done)}$", transform=ax.transAxes, ha="right", va="bottom", fontsize=8.5,
                bbox={"boxstyle": "square,pad=0.3", "facecolor": "white", "edgecolor": "black", "linewidth": 0.6})
        label(ax, letter)
    axes[-1].set_xlabel("Problems solved on held-out tests (%)")
    fig.savefig(os.path.join(OUT, "llm_feedback.png"))
    plt.close(fig)


# Figure 3. Acting on noisy diagnoses in Sudoku, two variable orderings.
def fig_sudoku():
    rows = list(csv.DictReader(open(os.path.join(RES, "sudoku_noisy.csv"), encoding="utf-8")))
    key = [k for k in rows[0] if k.startswith("success@")][0]

    def get(order, mode, eps):
        return 100 * float(next(r[key] for r in rows if r["ordering"] == order and r["search"] == "DFS"
                                 and r["mode"] == mode and float(r["eps"]) == eps))
    eps = [0.1, 0.3, 0.5]
    fig, axes = plt.subplots(2, 1, figsize=PORTRAIT, gridspec_kw={"hspace": 0.38})
    for ax, order, letter, name in ((axes[0], "mrv", "a", "MRV ordering"), (axes[1], "static", "b", "Static ordering")):
        ax.axhline(get(order, "none", 0.0), color="black", lw=1.0, ls="--", label="No diagnosis")
        for (mode, lab), color, m in zip((("defer", "Defer (soft backjumping)"), ("vote2", "Prune after two votes"),
                                           ("hard", "Prune immediately")), (C1, C2, C3), MARKERS):
            ax.plot([100 * e for e in eps], [get(order, mode, e) for e in eps], color=color, marker=m, label=lab)
        ax.set_xticks([10, 30, 50])
        ax.set_xlim(5, 55)
        ax.set_ylim(0, 100)
        ax.set_xlabel(r"Diagnosis error rate $\epsilon$ (%)")
        ax.set_ylabel("Puzzles solved (%)")
        ax.text(0.02, 0.04 if order == "mrv" else 0.95, name, transform=ax.transAxes, ha="left",
                va="bottom" if order == "mrv" else "top", fontsize=9, style="italic")
        label(ax, letter)
    axes[1].legend(loc="upper right")
    fig.savefig(os.path.join(OUT, "sudoku_soft_backjumping.png"))
    plt.close(fig)


# Figure 4. Knowledge-graph QA: verifier ceiling and beam-width sweep.
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
    groups = ["Synthetic", "MetaQA (3-hop)", "WebQSP"]
    search = [syn["8"], 100 * float(mq["path-beam-8"]["hits1"]), 100 * float(wq["rel-beam-32"]["hits1"])]
    oracle = [100.0, 100 * float(mq["oracle"]["hits1"]), 100 * float(wq["oracle"]["hits1"])]
    goal = [syn_goal, 100 * float(mq["path-beam-8"]["goal_test_rate"]), 100 * float(wq["rel-beam-32"]["goal_test_rate"])]

    fig, (a1, a2) = plt.subplots(2, 1, figsize=PORTRAIT, gridspec_kw={"hspace": 0.38})
    x = np.arange(len(groups))
    w = 0.26
    for i, (vals, name, color, hatch) in enumerate(((search, "Committed answer", C1, ""),
                                                     (oracle, "Oracle path", C2, "//"),
                                                     (goal, "Goal-test score", C3, ".."))):
        a1.bar(x + (i - 1) * w, vals, width=w, color=color, edgecolor="black", linewidth=0.5, hatch=hatch, label=name)
    a1.set_xticks(x, groups)
    a1.set_ylim(0, 115)
    a1.set_ylabel("Accuracy (%)")
    a1.grid(axis="x", visible=False)
    a1.tick_params(axis="x", length=0)
    a1.legend(loc="upper center", ncol=3, fontsize=7.8, handlelength=1.4, columnspacing=0.9)
    label(a1, "a")

    series = [("Synthetic", [(1, syn["1"]), (3, syn["3"]), (8, syn["8"]), (64, syn["64"])], C1),
              ("MetaQA (3-hop)", [(1, 100 * float(mq["path-greedy"]["hits1"])), (3, 100 * float(mq["path-beam-3"]["hits1"])),
                                  (8, 100 * float(mq["path-beam-8"]["hits1"])), (18, 100 * float(mq["path-beam-18"]["hits1"]))], C2),
              ("WebQSP", [(1, 100 * float(wq["rel-greedy"]["hits1"])), (3, 100 * float(wq["rel-beam-3"]["hits1"])),
                          (8, 100 * float(wq["rel-beam-8"]["hits1"])), (32, 100 * float(wq["rel-beam-32"]["hits1"]))], C3)]
    for (name, pts, color), m in zip(series, MARKERS):
        xs, ys = zip(*pts)
        a2.plot(xs, ys, color=color, marker=m, label=name)
    a2.set_xscale("log")
    a2.set_xticks([1, 3, 8, 18, 32, 64], ["1", "3", "8", "18", "32", "64"])
    a2.minorticks_off()
    a2.set_xlim(0.8, 80)
    a2.set_ylim(0, 60)
    a2.set_xlabel("Beam width")
    a2.set_ylabel("Accuracy (%)")
    a2.legend(loc="lower right")
    label(a2, "b")
    fig.savefig(os.path.join(OUT, "kgqa_verifier_ceiling.png"))
    plt.close(fig)


if __name__ == "__main__":
    fig_delay()
    fig_llm()
    fig_sudoku()
    fig_kgqa()
    print("wrote", sorted(os.listdir(OUT)))
