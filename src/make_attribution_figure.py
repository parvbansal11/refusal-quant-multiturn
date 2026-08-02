"""
Decomposition figure: what a reported multi-turn attack success rate is made of.

Auto-discovers runs by pairing incontext_<tag>.csv with its harm labels, so new
models, precisions, and attacks appear without editing this file.

  (a) stacked decomposition of reported ASR, per attack
  (b) reported ASR vs attack-attributable harm, per run
"""
import csv, glob, os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.titlesize": 9.5,
    "axes.labelsize": 9, "legend.fontsize": 7.5, "xtick.labelsize": 8,
    "ytick.labelsize": 8, "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 200, "savefig.bbox": "tight", "pdf.fonttype": 42,
})
INK = "#1a1a1a"
LEGACY = {"fp16_full": "judged_mt_3b_fp16.csv", "int4_full": "judged_mt_3b_nf4.csv"}


def judged_for(tag):
    for cand in (f"judged_mt_{tag}.csv", f"judged_{tag}.csv", LEGACY.get(tag, "")):
        if cand and os.path.exists(cand):
            return cand
    return None


def parts(ic_path, judged_path):
    ic = {r["id"]: r for r in csv.DictReader(open(ic_path))}
    jd = {r["id"]: r for r in csv.DictReader(open(judged_path))}
    ids = sorted(set(ic) & set(jd))
    if not ids:
        return None
    rc = np.array([int(ic[i]["refused_cold"]) for i in ids])
    ri = np.array([int(ic[i]["refused_incontext"]) for i in ids])
    hz = np.array([int(jd[i]["unsafe"]) for i in ids])
    return dict(no_harm=100 * ((ri == 0) & (hz == 0)).mean(),
                pre=100 * ((rc == 0) & (hz == 1)).mean(),
                attr=100 * ((rc == 1) & (ri == 0) & (hz == 1)).mean())


def pretty(tag):
    t = tag.replace("cres_", "").replace("_full", "")
    t = t.replace("fp16", "FP16").replace("nf4", "NF4")
    t = t.replace("awq", "AWQ").replace("gptq", "GPTQ").replace("int4", "NF4")
    return t.replace("_", "\n", 1)


runs = []
for ic_path in sorted(glob.glob("incontext_*.csv")):
    tag = os.path.basename(ic_path)[len("incontext_"):-len(".csv")]
    jp = judged_for(tag)
    if not jp:
        continue
    p = parts(ic_path, jp)
    if p:
        p.update(tag=tag, attack="Crescendo" if tag.startswith("cres") else "CoSafe")
        runs.append(p)

if not runs:
    raise SystemExit("No judged runs found. Run judge_harm.py first.")

attacks = sorted({r["attack"] for r in runs})
fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.0),
                         gridspec_kw={"width_ratios": [1, 2.1]})

# (a) stacked decomposition, one bar per attack
ax = axes[0]
segs = [("no harmful content", "no_harm", "#b8c4d0"),
        ("harm without the attack", "pre", "#e0a95f"),
        ("attack-attributable harm", "attr", "#b6493f")]
x = np.arange(len(attacks))
bottom = np.zeros(len(attacks))
for label, key, col in segs:
    vals = np.array([np.mean([r[key] for r in runs if r["attack"] == a]) for a in attacks])
    ax.bar(x, vals, 0.5, bottom=bottom, color=col, edgecolor=INK, linewidth=0.7,
           label=label)
    bottom += vals
for xi, tot in zip(x, bottom):
    ax.text(xi, tot + 1.5, f"{tot:.0f}%", ha="center", fontsize=9, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(attacks)
ax.set_ylabel("share of scenarios (%)")
ax.set_ylim(0, max(bottom) * 1.25 + 5)
ax.set_title("(a) What reported ASR is made of")
ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.14), fontsize=7)

# (b) per-run reported vs attributable
ax = axes[1]
runs_sorted = sorted(runs, key=lambda r: (r["attack"], r["tag"]))
labels = [pretty(r["tag"]) for r in runs_sorted]
rep = [r["no_harm"] + r["pre"] + r["attr"] for r in runs_sorted]
att = [r["attr"] for r in runs_sorted]
x = np.arange(len(runs_sorted)); w = 0.4
ax.bar(x - w/2, rep, w, label="reported ASR", color="#b8c4d0", edgecolor=INK, linewidth=0.6)
ax.bar(x + w/2, att, w, label="attack-attributable harm", color="#b6493f", edgecolor=INK, linewidth=0.6)
for xi, v in zip(x - w/2, rep):
    ax.text(xi, v + 1.2, f"{v:.0f}", ha="center", va="bottom", fontsize=6.5)
for xi, v in zip(x + w/2, att):
    ax.text(xi, v + 1.2, f"{v:.0f}", ha="center", va="bottom", fontsize=6.5)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=6 if len(runs_sorted) > 8 else 7)
ax.set_ylim(0, max(rep) * 1.2 + 5)
ax.set_ylabel("rate (%)")
ax.set_title("(b) Every run, both measures")
ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.14), fontsize=7)

pooled_rep = np.mean([r["no_harm"] + r["pre"] + r["attr"] for r in runs])
pooled_att = np.mean([r["attr"] for r in runs])
fig.suptitle(f"Reported multi-turn attack success is "
             f"{pooled_rep/max(pooled_att,1e-9):.0f}x its attributable harm",
             y=1.04, fontsize=10, fontweight="bold")
fig.subplots_adjust(top=0.80, bottom=0.28, wspace=0.28)
fig.savefig("fig_attribution.pdf"); fig.savefig("fig_attribution.png", dpi=300)
print(f"wrote fig_attribution over {len(runs)} runs "
      f"({', '.join(attacks)}): reported {pooled_rep:.0f}%, attributable {pooled_att:.0f}%")
