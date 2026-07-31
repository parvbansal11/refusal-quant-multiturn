"""
Significance analysis for the committed in-context CSVs. No GPU, no model load.

Regenerates every number in RESULTS.md and the paper tables from the five
incontext_*.csv files:
  - per-run means (cold and in-context, projection and refusal)
  - 3B fp16 vs nf4: McNemar on refusal, paired t on projection
  - 8B fp16/awq/gptq: pairwise McNemar, Cochran's Q, paired t on projection
  - projection-behaviour coupling (point-biserial r, AUC) per run
  - per-category in-context refusal

Usage: python analyze_multiturn.py
"""
import csv
import numpy as np
from scipy import stats

FILES = {
    ("3B", "fp16"): "incontext_fp16_full.csv",
    ("3B", "nf4"):  "incontext_int4_full.csv",
    ("8B", "fp16"): "incontext_8b_fp16.csv",
    ("8B", "awq"):  "incontext_8b_awq.csv",
    ("8B", "gptq"): "incontext_8b_gptq.csv",
}


def load(path):
    out = {}
    for r in csv.DictReader(open(path)):
        out[r["id"]] = {
            "cat": r["category"],
            "pc": float(r["proj_cold"]), "rc": int(r["refused_cold"]),
            "pi": float(r["proj_incontext"]), "ri": int(r["refused_incontext"]),
        }
    return out


DATA = {k: load(v) for k, v in FILES.items()}
IDS = {m: sorted(DATA[(m, "fp16")]) for m in ("3B", "8B")}


def col(model, quant, field):
    d = DATA[(model, quant)]
    return np.array([d[i][field] for i in IDS[model]])


def mcnemar(a, b):
    """Continuity-corrected McNemar chi-square on paired binary refusal."""
    b01 = int(np.sum((a == 0) & (b == 1)))
    b10 = int(np.sum((a == 1) & (b == 0)))
    n = b01 + b10
    if n == 0:
        return 0.0, 1.0, b10, b01
    chi2 = (abs(b10 - b01) - 1) ** 2 / n
    return chi2, stats.chi2.sf(chi2, 1), b10, b01


def cochran_q(mat):
    """Cochran's Q across k paired binary conditions (columns)."""
    k = mat.shape[1]
    T, G = mat.sum(1), mat.sum(0)
    den = k * T.sum() - np.sum(T ** 2)
    if den == 0:
        return 0.0, 1.0
    Q = k * (k - 1) * np.sum((G - G.mean()) ** 2) / den
    return Q, stats.chi2.sf(Q, k - 1)


def coupling(score, label):
    if label.sum() in (0, len(label)):
        return float("nan"), float("nan")
    r = stats.pointbiserialr(label, score).correlation
    u = stats.mannwhitneyu(score[label == 1], score[label == 0]).statistic
    return r, u / (label.sum() * (len(label) - label.sum()))


print("MEANS")
print(f"{'run':>9} | {'cold pj':>7} {'cold rf':>7} | {'ic pj':>7} {'ic rf':>6} {'ASR':>6}")
for (m, q) in FILES:
    pc, rc, pi, ri = (col(m, q, f) for f in ("pc", "rc", "pi", "ri"))
    print(f"{m+'/'+q:>9} | {pc.mean():+7.3f} {rc.mean():7.1%} | "
          f"{pi.mean():+7.3f} {ri.mean():6.1%} {1-ri.mean():6.1%}")

print("\n3B fp16 vs nf4")
for cond, rf, pf in (("cold", "rc", "pc"), ("in-context", "ri", "pi")):
    chi2, p, b10, b01 = mcnemar(col("3B", "fp16", rf), col("3B", "nf4", rf))
    pa, pb = col("3B", "fp16", pf), col("3B", "nf4", pf)
    t, pt = stats.ttest_rel(pb, pa)
    print(f"  {cond:>10}: refusal McNemar chi2={chi2:.3f} p={p:.3f} "
          f"(disc {b10}/{b01}); proj paired t={t:+.2f} p={pt:.3f}")

print("\n8B fp16/awq/gptq")
for cond, rf in (("cold", "rc"), ("in-context", "ri")):
    Q, pq = cochran_q(np.stack([col("8B", q, rf) for q in ("fp16", "awq", "gptq")], 1))
    print(f"  {cond:>10}: Cochran Q={Q:.3f} p={pq:.3f}")
    for qa, qb in (("fp16", "awq"), ("fp16", "gptq"), ("awq", "gptq")):
        chi2, p, b10, b01 = mcnemar(col("8B", qa, rf), col("8B", qb, rf))
        print(f"    {qa} vs {qb}: McNemar chi2={chi2:.3f} p={p:.3f} (disc {b10}/{b01})")

print("\nPROJECTION-BEHAVIOUR COUPLING")
for (m, q) in FILES:
    for cond, rf, pf in (("cold", "rc", "pc"), ("in-context", "ri", "pi")):
        r, a = coupling(col(m, q, pf), col(m, q, rf))
        print(f"  {m+'/'+q:>9} {cond:>10}: r_pb={r:+.3f} AUC={a:.3f}")

print("\nPER-CATEGORY IN-CONTEXT REFUSAL")
cats = sorted({d["cat"] for d in DATA[("3B", "fp16")].values()})
for c in cats:
    line = f"  {c[:34]:>34}: "
    for (m, q) in FILES:
        d = DATA[(m, q)]
        v = np.mean([d[i]["ri"] for i in IDS[m] if d[i]["cat"] == c])
        line += f"{m[0]}{q[0]} {v:4.0%}  "
    print(line)
