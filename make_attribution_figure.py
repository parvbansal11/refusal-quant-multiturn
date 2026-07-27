"""
Decomposition figure: what the reported multi-turn attack success rate is
actually made of. Waterfall from reported ASR down to attack-attributable harm.
"""
import csv
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.titlesize": 9.5,
    "axes.labelsize": 9, "legend.fontsize": 8, "xtick.labelsize": 8.5,
    "ytick.labelsize": 8, "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 200, "savefig.bbox": "tight", "pdf.fonttype": 42,
})
INK = "#1a1a1a"

RUNS = [("3B", "fp16", "incontext_fp16_full.csv", "judged_mt_3b_fp16.csv"),
        ("3B", "nf4",  "incontext_int4_full.csv", "judged_mt_3b_nf4.csv"),
        ("8B", "fp16", "incontext_8b_fp16.csv",   "judged_mt_8b_fp16.csv"),
        ("8B", "awq",  "incontext_8b_awq.csv",    "judged_mt_8b_awq.csv"),
        ("8B", "gptq", "incontext_8b_gptq.csv",   "judged_mt_8b_gptq.csv")]


def parts(ic_path, judged_path):
    ic = {r["id"]: r for r in csv.DictReader(open(ic_path))}
    jd = {r["id"]: r for r in csv.DictReader(open(judged_path))}
    ids = sorted(set(ic) & set(jd))
    rc = np.array([int(ic[i]["refused_cold"]) for i in ids])
    ri = np.array([int(ic[i]["refused_incontext"]) for i in ids])
    hz = np.array([int(jd[i]["unsafe"]) for i in ids])
    return dict(no_harm=100*((ri == 0) & (hz == 0)).mean(),
                pre=100*((rc == 0) & (hz == 1)).mean(),
                attr=100*((rc == 1) & (ri == 0) & (hz == 1)).mean())


agg = [parts(a, b) for _, _, a, b in RUNS]
mean = {k: np.mean([p[k] for p in agg]) for k in ("no_harm", "pre", "attr")}
reported = mean["no_harm"] + mean["pre"] + mean["attr"]

fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.9))

# (a) stacked decomposition of the reported number
ax = axes[0]
bottom = 0
segs = [("no harmful content", mean["no_harm"], "#b8c4d0"),
        ("harm without the attack", mean["pre"], "#e0a95f"),
        ("attack-attributable harm", mean["attr"], "#b6493f")]
for label, val, col in segs:
    ax.bar(0, val, 0.5, bottom=bottom, color=col, edgecolor=INK,
           linewidth=0.7, label=f"{label} ({val:.0f}%)")
    bottom += val
ax.set_xlim(-0.55, 0.55)
ax.set_xticks([0]); ax.set_xticklabels(["reported ASR"])
ax.set_ylabel("share of scenarios (%)")
ax.set_ylim(0, 85)
ax.text(0, reported + 2, f"{reported:.0f}%", ha="center", fontsize=9.5,
        fontweight="bold")
ax.set_title("(a) What reported ASR is made of")
ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.22),
          fontsize=7.2)

# (b) per-run: reported vs attributable
ax = axes[1]
labels = [f"{m}\n{q.upper()}" for m, q, _, _ in RUNS]
rep = [p["no_harm"] + p["pre"] + p["attr"] for p in agg]
att = [p["attr"] for p in agg]
x = np.arange(len(RUNS)); w = 0.4
ax.bar(x - w/2, rep, w, label="reported ASR", color="#b8c4d0",
       edgecolor=INK, linewidth=0.6)
ax.bar(x + w/2, att, w, label="attack-attributable harm", color="#b6493f",
       edgecolor=INK, linewidth=0.6)
for xi, v in zip(x - w/2, rep):
    ax.text(xi, v + 1.5, f"{v:.0f}", ha="center", va="bottom", fontsize=7.5)
for xi, v in zip(x + w/2, att):
    ax.text(xi, v + 1.5, f"{v:.0f}", ha="center", va="bottom", fontsize=7.5)
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7.5)
ax.set_ylim(0, 85); ax.set_ylabel("rate (%)")
ax.set_title("(b) Every model, both metrics")
ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.16),
          fontsize=7.4)

fig.suptitle("Reported multi-turn attack success is 40x its attributable harm",
             y=1.05, fontsize=10, fontweight="bold")
fig.subplots_adjust(top=0.80, bottom=0.30, wspace=0.30)
fig.savefig("fig_attribution.pdf"); fig.savefig("fig_attribution.png", dpi=300)
print(f"wrote fig_attribution (reported {reported:.0f}%, attributable {mean['attr']:.0f}%)")
