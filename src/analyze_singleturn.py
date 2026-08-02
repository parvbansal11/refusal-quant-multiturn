"""
Single-turn significance once the cloud CSVs exist. FP16 vs each quantization,
per model. Skips cleanly if a file is missing, so it runs incrementally as the
cloud runs land.

Expects singleturn_st_<model>_<prec>.csv from singleturn_hardened.py with columns
idx,projection,refused.

Usage: python analyze_singleturn.py
"""
import csv, os
import numpy as np
from scipy import stats

GROUPS = {
    "3B": [("fp16", "singleturn_st_3b_fp16.csv"),
           ("nf4",  "singleturn_st_3b_nf4.csv")],
    "8B": [("fp16", "singleturn_st_8b_fp16.csv"),
           ("awq",  "singleturn_st_8b_awq.csv"),
           ("gptq", "singleturn_st_8b_gptq.csv")],
}

def load(path):
    if not os.path.exists(path):
        return None
    rows = list(csv.DictReader(open(path)))
    return (np.array([int(r["refused"]) for r in rows]),
            np.array([float(r["projection"]) for r in rows]))

def mcnemar(a, b):
    b01 = int(np.sum((a == 0) & (b == 1)))
    b10 = int(np.sum((a == 1) & (b == 0)))
    n = b01 + b10
    if n == 0:
        return 0.0, 1.0, b10, b01
    chi2 = (abs(b10 - b01) - 1) ** 2 / n
    return chi2, stats.chi2.sf(chi2, 1), b10, b01

for model, runs in GROUPS.items():
    loaded = [(prec, load(path), path) for prec, path in runs]
    have = [(prec, d) for prec, d, _ in loaded if d is not None]
    missing = [path for _, d, path in loaded if d is None]
    print(f"\n=== {model} single-turn ===")
    if missing:
        print(f"  missing (run on cloud): {', '.join(missing)}")
    if not have:
        continue
    for prec, (r, p) in have:
        print(f"  {prec:>5}: refusal {r.mean():.0%}  proj {p.mean():+.3f}  n={len(r)}")
    base = dict(have).get("fp16")
    if base is None:
        continue
    rf, pf = base
    for prec, (r, p) in have:
        if prec == "fp16":
            continue
        chi2, pv, b10, b01 = mcnemar(rf, r)
        t, pt = stats.ttest_rel(p, pf)
        print(f"  fp16 vs {prec:>4}: refusal {rf.mean():.0%}->{r.mean():.0%} "
              f"McNemar chi2={chi2:.2f} p={pv:.3f} (disc {b10}/{b01}); "
              f"proj paired t={t:+.2f} p={pt:.3f}")
