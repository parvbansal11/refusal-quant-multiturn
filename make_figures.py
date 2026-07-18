"""
Paper figures from the committed in-context CSVs. Writes PDF (vector) + PNG.

fig1_multiturn.pdf   in-context refusal by precision, 3B and 8B. The null.
fig2_cold_vs_ic.pdf  cold vs in-context refusal, 8B. Quant raises cold refusal;
                     context erases the gap.
fig3_coupling.pdf    projection-behaviour AUC by precision and condition. The
                     refusal direction stays behaviourally coupled under quant.

Usage: python make_figures.py
"""
import csv
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.titlesize": 8,
    "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 200, "savefig.bbox": "tight", "pdf.fonttype": 42,
})

FILES = {
    ("3B", "fp16"): "incontext_fp16_full.csv",
    ("3B", "nf4"):  "incontext_int4_full.csv",
    ("8B", "fp16"): "incontext_8b_fp16.csv",
    ("8B", "awq"):  "incontext_8b_awq.csv",
    ("8B", "gptq"): "incontext_8b_gptq.csv",
}
INK = "#1b1b1b"
C = {"fp16": "#4c72b0", "nf4": "#dd8452", "awq": "#55a868", "gptq": "#c44e52"}


def load(path):
    return list(csv.DictReader(open(path)))


DATA = {k: load(v) for k, v in FILES.items()}


def rate(model, quant, field):
    rows = DATA[(model, quant)]
    return 100 * np.mean([int(r[field]) for r in rows])


def wilson(p, n=100, z=1.96):
    p /= 100
    c = z * np.sqrt(p * (1 - p) / n) * 100
    return c


def save(fig, name):
    fig.savefig(f"{name}.pdf")
    fig.savefig(f"{name}.png", dpi=300)
    plt.close(fig)


# fig1: in-context refusal by precision, two panels
fig, axes = plt.subplots(1, 2, figsize=(5.2, 2.1), sharey=True)
for ax, model, quants in ((axes[0], "3B", ["fp16", "nf4"]),
                          (axes[1], "8B", ["fp16", "awq", "gptq"])):
    vals = [rate(model, q, "refused_incontext") for q in quants]
    err = [wilson(v) for v in vals]
    x = np.arange(len(quants))
    ax.bar(x, vals, width=0.6, color=[C[q] for q in quants], edgecolor=INK, linewidth=0.6)
    ax.errorbar(x, vals, yerr=err, fmt="none", ecolor=INK, elinewidth=0.8, capsize=2.5)
    for xi, v in zip(x, vals):
        ax.text(xi, v + max(err) + 2, f"{v:.0f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([q.upper() for q in quants])
    ax.set_title(f"Llama-3.2-3B" if model == "3B" else "Llama-3.1-8B")
    ax.set_ylim(0, 60)
axes[0].set_ylabel("multi-turn refusal (%)")
fig.suptitle("Multi-turn (CoSafe) refusal by precision", y=1.02, fontsize=8.5)
save(fig, "fig1_multiturn")

# fig2: cold vs in-context, 8B, grouped by precision
fig, ax = plt.subplots(figsize=(3.4, 2.3))
quants = ["fp16", "awq", "gptq"]
x = np.arange(len(quants))
w = 0.38
cold = [rate("8B", q, "refused_cold") for q in quants]
ic = [rate("8B", q, "refused_incontext") for q in quants]
ax.bar(x - w / 2, cold, w, label="single turn (cold)", color="#8c8c8c", edgecolor=INK, linewidth=0.6)
ax.bar(x + w / 2, ic, w, label="multi-turn (in-context)", color="#4c72b0", edgecolor=INK, linewidth=0.6)
for xi, v in zip(x - w / 2, cold):
    ax.text(xi, v + 1.5, f"{v:.0f}", ha="center", va="bottom", fontsize=6.5)
for xi, v in zip(x + w / 2, ic):
    ax.text(xi, v + 1.5, f"{v:.0f}", ha="center", va="bottom", fontsize=6.5)
ax.set_xticks(x)
ax.set_xticklabels([q.upper() for q in quants])
ax.set_ylabel("refusal (%)")
ax.set_ylim(0, 48)
ax.set_title("Llama-3.1-8B: quantization caution\nis single-turn only")
ax.legend(frameon=False, loc="upper left")
save(fig, "fig2_cold_vs_ic")

# fig3: coupling AUC by precision and condition
def auc(model, quant, cond):
    from scipy import stats
    rows = DATA[(model, quant)]
    pf = "proj_cold" if cond == "cold" else "proj_incontext"
    rfl = "refused_cold" if cond == "cold" else "refused_incontext"
    s = np.array([float(r[pf]) for r in rows])
    y = np.array([int(r[rfl]) for r in rows])
    u = stats.mannwhitneyu(s[y == 1], s[y == 0]).statistic
    return u / (y.sum() * (len(y) - y.sum()))


fig, ax = plt.subplots(figsize=(3.4, 2.3))
runs = [("3B", "fp16"), ("3B", "nf4"), ("8B", "fp16"), ("8B", "awq"), ("8B", "gptq")]
x = np.arange(len(runs))
ax.bar(x - 0.2, [auc(m, q, "cold") for m, q in runs], 0.38, label="single turn",
       color="#b0b0b0", edgecolor=INK, linewidth=0.5)
ax.bar(x + 0.2, [auc(m, q, "ic") for m, q in runs], 0.38, label="multi-turn",
       color="#4c72b0", edgecolor=INK, linewidth=0.5)
ax.axhline(0.5, color=INK, lw=0.6, ls=":")
ax.set_xticks(x)
ax.set_xticklabels([f"{m}\n{q.upper()}" for m, q in runs], fontsize=6)
ax.set_ylabel("AUC (projection predicts refusal)")
ax.set_ylim(0.5, 1.0)
ax.set_title("Refusal direction stays coupled under quantization")
ax.legend(frameon=False, loc="lower right", ncol=2)
save(fig, "fig3_coupling")

print("wrote fig1_multiturn, fig2_cold_vs_ic, fig3_coupling (.pdf + .png)")
