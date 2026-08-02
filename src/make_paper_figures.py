"""
The three figures the paper needs, from the committed CSVs.

  fig_inversion   reported success against attributable harm, per attack
  fig_triple      Qwen-3B under one precision change across three attack types
  fig_regimes     cold versus in-context refusal, five families, both attacks
"""
import csv
import os
from datadir import path as _dp, PAPER
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.titlesize": 8.5,
    "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5, "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 200, "savefig.bbox": "tight", "pdf.fonttype": 42,
})
INK, GREY, RED, BLUE = "#1a1a1a", "#b8c4d0", "#b6493f", "#41668c"
FAMILIES = ["3b", "qwen3b", "8b", "qwen7b", "mistral7b"]
PRETTY = {"3b": "Llama-3B", "8b": "Llama-8B", "qwen3b": "Qwen-3B",
          "qwen7b": "Qwen-7B", "mistral7b": "Mistral-7B"}


def load(p):
    if not os.path.exists(_dp(p)):
        return None
    rows = list(csv.DictReader(open(_dp(p))))
    if not rows:
        return None
    k = "id" if "id" in rows[0] else "idx"
    return {r[k]: r for r in rows}


def cell(ic, jd=None):
    I = load(ic)
    if I is None:
        return None
    J = load(jd) if jd else None
    ids = sorted(set(I) & set(J)) if J else sorted(I)
    if not ids:
        return None
    n = len(ids)
    cold = [int(I[i]["refused_cold"]) for i in ids]
    ctx = [int(I[i]["refused_incontext"]) for i in ids]
    out = dict(n=n, cold=100 * sum(cold) / n, ctx=100 * sum(ctx) / n,
               reported=100 * sum(1 for x in ctx if x == 0) / n)
    if J:
        hz = [int(J[i]["unsafe"]) for i in ids]
        out["attrib"] = 100 * sum(1 for c, x, h in zip(cold, ctx, hz)
                                  if c == 1 and x == 0 and h == 1) / n
    return out


# ---------------------------------------------------------------- fig_inversion
# Matched configurations only, so the bars are the same population as the
# headline test in the paper: a config counts only where both attacks ran.
rep, att = {"CoSafe": [], "Crescendo": []}, {"CoSafe": [], "Crescendo": []}
for m in FAMILIES:
    for q in ["fp16", "fp16cuda", "nf4", "awq", "gptq"]:
        cs = cell(f"incontext_{m}_{q}.csv", f"judged_mt_{m}_{q}.csv")
        cr = cell(f"incontext_cres_{m}_{q}.csv", f"judged_cres_{m}_{q}.csv")
        if cs and cr and "attrib" in cs and "attrib" in cr:
            rep["CoSafe"].append(cs["reported"]); att["CoSafe"].append(cs["attrib"])
            rep["Crescendo"].append(cr["reported"]); att["Crescendo"].append(cr["attrib"])

fig, ax = plt.subplots(figsize=(3.3, 2.5))
x = np.arange(2)
w = 0.36
names = ["CoSafe", "Crescendo"]
r = [np.mean(rep[k]) for k in names]
a = [np.mean(att[k]) for k in names]
ax.bar(x - w / 2, r, w, color=GREY, edgecolor=INK, linewidth=0.7,
       label="reported success")
ax.bar(x + w / 2, a, w, color=RED, edgecolor=INK, linewidth=0.7,
       label="attributable harm")
for xi, v in zip(x - w / 2, r):
    ax.text(xi, v + 1.5, f"{v:.1f}%", ha="center", fontsize=7.5, fontweight="bold")
for xi, v in zip(x + w / 2, a):
    ax.text(xi, v + 1.5, f"{v:.1f}%", ha="center", fontsize=7.5, fontweight="bold")
ax.annotate("", xy=(1.18, 8), xytext=(0.18, 2),
            arrowprops=dict(arrowstyle="->", color=RED, lw=1.2))
ax.text(0.68, 13, "ordering\nreverses", color=RED, fontsize=7,
        ha="center", style="italic")
ax.set_xticks(x)
ax.set_xticklabels([f"{k}\n(n={len(rep[k])})" for k in names])
ax.set_ylabel("rate (%)")
ax.set_ylim(0, max(r) * 1.22)
ax.legend(frameon=False, loc="upper right")
ax.set_title("Reported success inverts the attack ordering")
fig.savefig(os.path.join(PAPER, "fig_inversion.pdf"))
print(f"fig_inversion: CoSafe {r[0]:.1f}/{a[0]:.1f}, Crescendo {r[1]:.1f}/{a[1]:.1f}")

# ------------------------------------------------------------------- fig_triple
# Baseline is fp16cuda where it exists, so the precision change is the only
# thing differing between the bars: same card, same run, FP16 against NF4.
def _base(stem):
    return stem.replace("qwen3b_", "qwen3b_fp16cuda").replace("fp16cuda.csv", "fp16cuda.csv") \
        if load(stem.replace("fp16", "fp16cuda")) else None

rows = []
st_f = load("singleturn_st_qwen3b_fp16cuda.csv") or load("singleturn_st_qwen3b_fp16.csv")
st_n = load("singleturn_st_qwen3b_nf4.csv")
if st_f and st_n:
    ids = sorted(set(st_f) & set(st_n))
    rows.append(("single-turn",
                 100 * sum(1 - int(st_f[i]["refused"]) for i in ids) / len(ids),
                 100 * sum(1 - int(st_n[i]["refused"]) for i in ids) / len(ids)))
for lab, pre in [("CoSafe", "incontext_"), ("Crescendo", "incontext_cres_")]:
    f = cell(f"{pre}qwen3b_fp16cuda.csv") or cell(f"{pre}qwen3b_fp16.csv")
    n = cell(f"{pre}qwen3b_nf4.csv")
    if f and n:
        rows.append((lab, f["reported"], n["reported"]))

fig, ax = plt.subplots(figsize=(3.3, 2.5))
x = np.arange(len(rows))
w = 0.36
f16 = [r[1] for r in rows]
nf4 = [r[2] for r in rows]
ax.bar(x - w / 2, f16, w, color=BLUE, edgecolor=INK, linewidth=0.7, label="FP16")
ax.bar(x + w / 2, nf4, w, color=RED, edgecolor=INK, linewidth=0.7, label="NF4")
for i, (lab, a_, b_) in enumerate(rows):
    d = b_ - a_
    ax.text(i, max(a_, b_) + 3, f"{d:+.1f}pp", ha="center", fontsize=7,
            fontweight="bold" if abs(d) > 5 else "normal",
            color=RED if abs(d) > 5 else INK)
ax.set_xticks(x)
ax.set_xticklabels([r[0] for r in rows])
ax.set_ylabel("reported attack success (%)")
ax.set_ylim(0, 100)
ax.legend(frameon=False, loc="upper left")
ax.set_title("Qwen-3B: one precision change, three attacks")
fig.savefig(os.path.join(PAPER, "fig_triple.pdf"))
print("fig_triple:", [(r[0], round(r[2] - r[1], 1)) for r in rows])

# ------------------------------------------------------------------ fig_regimes
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), sharey=True)
for ax, (name, pre) in zip(axes, [("CoSafe", "incontext_"),
                                  ("Crescendo", "incontext_cres_")]):
    labels, cold, ctx = [], [], []
    for m in FAMILIES:
        c = cell(f"{pre}{m}_fp16.csv")
        if c:
            labels.append(PRETTY[m])
            cold.append(c["cold"])
            ctx.append(c["ctx"])
    x = np.arange(len(labels))
    w = 0.36
    ax.bar(x - w / 2, cold, w, color=GREY, edgecolor=INK, linewidth=0.7,
           label="no conversation")
    ax.bar(x + w / 2, ctx, w, color=BLUE, edgecolor=INK, linewidth=0.7,
           label="after attack")
    for i, (a_, b_) in enumerate(zip(cold, ctx)):
        ax.annotate("", xy=(i + w / 2, b_), xytext=(i - w / 2, a_),
                    arrowprops=dict(arrowstyle="->", lw=0.8,
                                    color=RED if abs(b_ - a_) > 8 else "#999"))
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_title(name)
    ax.set_ylim(0, 105)
axes[0].set_ylabel("refusal rate (%)")
axes[0].legend(frameon=False, loc="lower left")
fig.suptitle("Multi-turn vulnerability is a family property, not a precision one",
             y=1.03, fontsize=9, fontweight="bold")
fig.savefig(os.path.join(PAPER, "fig_regimes.pdf"))
print("fig_regimes: written")
