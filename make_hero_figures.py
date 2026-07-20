"""
Headline figures for the two-column paper, built from the committed CSVs.
  fig_hero.pdf     two panels: multi-turn refusal (the null) and multi-turn
                   harmful-response rate with the safety-prompt positive control.
  fig_twometric.pdf single vs multi across both metrics, both models.
"""
import csv, glob, os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.titlesize": 9.5,
    "axes.labelsize": 9, "legend.fontsize": 8, "xtick.labelsize": 8.5,
    "ytick.labelsize": 8, "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 200, "savefig.bbox": "tight", "pdf.fonttype": 42,
})
C = {"FP16": "#3b6ea5", "NF4": "#d98c3f", "AWQ": "#4f9d6b", "GPTQ": "#b6493f",
     "safety prompt": "#7a5aa8"}
INK = "#1a1a1a"


def harm(tag):
    p = f"judged_{tag}.csv"
    if not os.path.exists(p):
        return None
    rows = list(csv.DictReader(open(p)))
    return 100 * np.mean([int(r["unsafe"]) for r in rows])


def wilson(p, n=100, z=1.96):
    p /= 100.0
    return z * np.sqrt(p * (1 - p) / n) * 100


def bars(ax, labels, vals, errs=None, note=None):
    x = np.arange(len(labels))
    cols = [C.get(l, "#888") for l in labels]
    ax.bar(x, vals, width=0.62, color=cols, edgecolor=INK, linewidth=0.7, zorder=3)
    if errs is not None:
        ax.errorbar(x, vals, yerr=errs, fmt="none", ecolor=INK, elinewidth=0.9,
                    capsize=3, zorder=4)
    top = max([v + (e if errs is not None else 0) for v, e in
               zip(vals, errs if errs is not None else [0]*len(vals))])
    for xi, v in zip(x, vals):
        ax.text(xi, v + top*0.03 + 0.6, f"{v:.0f}", ha="center", va="bottom",
                fontsize=8.5)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.grid(axis="y", color="#dddddd", linewidth=0.6, zorder=0)


# ---- fig_hero: the two-metric null in one glance ----
fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.7))

# Panel A: multi-turn refusal, 8B (verified)
ref = {"FP16": 27, "AWQ": 32, "GPTQ": 31}
labels = list(ref)
vals = [ref[l] for l in labels]
bars(axes[0], labels, vals, [wilson(v) for v in vals])
axes[0].set_ylim(0, 60)
axes[0].set_ylabel("multi-turn refusal (%)")
axes[0].set_title("(a) Refusal is flat across precision\n(Cochran $Q$ $p=0.12$)")

# Panel B: multi-turn harm, 8B + safety-prompt control
hb = {"FP16": harm("mt_8b_fp16"), "AWQ": harm("mt_8b_awq"),
      "GPTQ": harm("mt_8b_gptq"), "safety prompt": harm("mt_8b_fp16_sys")}
hb = {k: (v if v is not None else 0) for k, v in hb.items()}
labels2 = list(hb); vals2 = [hb[l] for l in labels2]
bars(axes[1], labels2, vals2)
axes[1].set_ylim(0, 12)
axes[1].set_ylabel("harmful responses (%)")
axes[1].set_title("(b) Harm is flat too; a safety prompt\ncuts it to 0 (positive control)")
axes[1].axvspan(2.5, 3.5, color="#f0ebf7", zorder=1)
for lab in axes[1].get_xticklabels():
    if lab.get_text() == "safety prompt":
        lab.set_fontsize(7.5)
fig.suptitle("Llama-3.1-8B under the CoSafe multi-turn attack", y=1.13,
             fontsize=10, fontweight="bold")
fig.subplots_adjust(top=0.74, wspace=0.32)
fig.savefig("fig_hero.pdf"); fig.savefig("fig_hero.png", dpi=300)
plt.close(fig)

# ---- fig_twometric: single vs multi, both metrics, both models ----
fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.7))
# left: refusal single(cold)->multi(in-ctx) for 8B precisions (verified)
prec = ["FP16", "AWQ", "GPTQ"]
cold = [23, 31, 34]; ic = [27, 32, 31]
x = np.arange(len(prec)); w = 0.38
axes[0].bar(x - w/2, cold, w, label="single turn", color="#b9c6d6",
            edgecolor=INK, linewidth=0.6, zorder=3)
axes[0].bar(x + w/2, ic, w, label="multi turn", color="#3b6ea5",
            edgecolor=INK, linewidth=0.6, zorder=3)
axes[0].set_xticks(x); axes[0].set_xticklabels(prec)
axes[0].set_ylabel("refusal (%)"); axes[0].set_ylim(0, 45)
axes[0].set_title("(a) Refusal metric (8B)")
axes[0].legend(frameon=False, loc="upper left")
axes[0].grid(axis="y", color="#dddddd", linewidth=0.6, zorder=0)
# right: harm single->multi for 8B precisions
hs = [harm("st_8b_fp16"), harm("st_8b_awq"), harm("st_8b_gptq")]
hm = [harm("mt_8b_fp16"), harm("mt_8b_awq"), harm("mt_8b_gptq")]
axes[1].bar(x - w/2, hs, w, label="single turn", color="#c8ddcf",
            edgecolor=INK, linewidth=0.6, zorder=3)
axes[1].bar(x + w/2, hm, w, label="multi turn", color="#4f9d6b",
            edgecolor=INK, linewidth=0.6, zorder=3)
axes[1].set_xticks(x); axes[1].set_xticklabels(prec)
axes[1].set_ylabel("harmful responses (%)"); axes[1].set_ylim(0, 12)
axes[1].set_title("(b) Harm metric (8B)")
axes[1].legend(frameon=False, loc="upper left")
axes[1].grid(axis="y", color="#dddddd", linewidth=0.6, zorder=0)
fig.suptitle("Quantization moves neither metric, single-turn or multi-turn",
             y=1.10, fontsize=10, fontweight="bold")
fig.subplots_adjust(top=0.80, wspace=0.32)
fig.savefig("fig_twometric.pdf"); fig.savefig("fig_twometric.png", dpi=300)
plt.close(fig)

print("wrote fig_hero and fig_twometric (harm 8B:",
      f"mt fp16={harm('mt_8b_fp16')}, awq={harm('mt_8b_awq')}, gptq={harm('mt_8b_gptq')}, sys={harm('mt_8b_fp16_sys')})")
