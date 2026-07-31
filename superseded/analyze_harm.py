"""
Significance on the Llama Guard harm labels. FP16 vs each quantization, per
model, for single-turn and multi-turn. Reads judged_*.csv from judge_harm.py.

Usage: python analyze_harm.py
"""
import csv, glob, os
import numpy as np
from scipy import stats

GROUPS = {
    ("single", "3B"): [("fp16", "st_3b_fp16"), ("nf4", "st_3b_nf4")],
    ("single", "8B"): [("fp16", "st_8b_fp16"), ("awq", "st_8b_awq"), ("gptq", "st_8b_gptq")],
    ("multi", "3B"):  [("fp16", "mt_3b_fp16"), ("nf4", "mt_3b_nf4")],
    ("multi", "8B"):  [("fp16", "mt_8b_fp16"), ("awq", "mt_8b_awq"), ("gptq", "mt_8b_gptq")],
}


def load(tag):
    p = f"judged_{tag}.csv"
    if not os.path.exists(p):
        return None
    return np.array([int(r["unsafe"]) for r in csv.DictReader(open(p))])


def mcnemar(a, b):
    b01 = int(np.sum((a == 0) & (b == 1)))
    b10 = int(np.sum((a == 1) & (b == 0)))
    n = b01 + b10
    if n == 0:
        return 0.0, 1.0, b10, b01
    chi2 = (abs(b10 - b01) - 1) ** 2 / n
    return chi2, stats.chi2.sf(chi2, 1), b10, b01


for (mode, model), runs in GROUPS.items():
    loaded = [(prec, load(tag), tag) for prec, tag in runs]
    have = [(p, d) for p, d, _ in loaded if d is not None]
    missing = [t for p, d, t in loaded if d is None]
    print(f"\n=== {mode} {model} (harmful-response rate) ===")
    if missing:
        print(f"  missing: {', '.join(missing)}")
    if not have:
        continue
    for p, d in have:
        print(f"  {p:>5}: harmful {d.mean():.0%}  (n={len(d)})")
    base = dict(have).get("fp16")
    if base is None:
        continue
    for p, d in have:
        if p == "fp16":
            continue
        chi2, pv, b10, b01 = mcnemar(base, d)
        arrow = "up" if d.mean() > base.mean() else ("down" if d.mean() < base.mean() else "flat")
        print(f"  fp16 vs {p:>4}: {base.mean():.0%} -> {d.mean():.0%} ({arrow})  "
              f"McNemar chi2={chi2:.2f} p={pv:.3f} (disc {b10}/{b01})")
