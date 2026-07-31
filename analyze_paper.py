"""
Every number in the paper, from the committed CSVs, in one pass.

Supersedes analyze_multiturn.py and analyze_harm.py, which were written against
the two-model pilot and break on the current file set (one references a run that
was archived, the other assumes both arms share a sample size).

    python analyze_paper.py
"""
import csv
import os
from scipy.stats import binomtest, wilcoxon, fisher_exact

FAMILIES = ["3b", "qwen3b", "8b", "qwen7b", "mistral7b"]
PRECISIONS = ["fp16", "nf4", "awq", "gptq"]
PRETTY = {"3b": "Llama-3.2-3B", "8b": "Llama-3.1-8B", "qwen3b": "Qwen2.5-3B",
          "qwen7b": "Qwen2.5-7B", "mistral7b": "Mistral-7B-v0.3"}
ATTACKS = [("CoSafe", "incontext_", "judged_mt_"),
           ("Crescendo", "incontext_cres_", "judged_cres_")]
# Bonferroni across the twelve precision comparisons in Table 3.
ALPHA = 0.05 / 12


def load(path):
    """Rows keyed by scenario id, so every comparison is paired."""
    if not os.path.exists(path):
        return None
    rows = list(csv.DictReader(open(path)))
    if not rows:
        return None
    key = "id" if "id" in rows[0] else "idx"
    return {r[key]: r for r in rows}


def arms(incontext, judged=None):
    """Refusal and harm vectors over the scenarios both files share."""
    ic = load(incontext)
    if ic is None:
        return None
    jd = load(judged) if judged else None
    ids = sorted(set(ic) & set(jd)) if jd else sorted(ic)
    if not ids:
        return None
    return dict(
        ids=ids, n=len(ids),
        cold=[int(ic[i]["refused_cold"]) for i in ids],
        ctx=[int(ic[i]["refused_incontext"]) for i in ids],
        harm=[int(jd[i]["unsafe"]) for i in ids] if jd else None)


def mcnemar(a, b):
    """Exact paired test. Returns discordant counts and p."""
    x = sum(1 for p, q in zip(a, b) if p == 1 and q == 0)
    y = sum(1 for p, q in zip(a, b) if p == 0 and q == 1)
    return x, y, binomtest(x, x + y, 0.5).pvalue if x + y else 1.0


def pct(v):
    return 100 * sum(v) / len(v)


def section(title):
    print(f"\n{'=' * 100}\n{title}\n{'=' * 100}")


section("TABLE 1  Does the conversation change refusal at all? (cold vs in-context, FP16)")
print(f"{'model':<16} {'attack':<10} {'cold':>7} {'in-ctx':>7} {'net':>7} "
      f"{'ASR':>7} {'1->0':>5} {'0->1':>5} {'p':>9}  verdict")
for m in FAMILIES:
    for name, ic, _ in ATTACKS:
        a = arms(f"{ic}{m}_fp16.csv")
        if not a:
            continue
        x, y, p = mcnemar(a["cold"], a["ctx"])
        cold, ctx = pct(a["cold"]), pct(a["ctx"])
        verdict = ("no net effect" if p >= 0.05 else
                   "refusal DROPS" if x > y else "refusal RISES")
        print(f"{PRETTY[m]:<16} {name:<10} {cold:6.1f}% {ctx:6.1f}% {ctx - cold:+6.1f}% "
              f"{100 - ctx:6.1f}% {x:5d} {y:5d} {p:9.5f}  {verdict}")

section("TABLE 2  What reported success is actually made of (all conditions)")
print(f"{'model':<16} {'attack':<10} {'prec':<5} {'n':>4} | {'reported':>8} {'harm':>6} | "
      f"{'attrib':>7} {'pre-ex':>7} {'no-harm':>8}")
by_attack = {name: [] for name, _, _ in ATTACKS}
paired = []
for m in FAMILIES:
    for q in PRECISIONS:
        cell = {}
        for name, ic, jd in ATTACKS:
            a = arms(f"{ic}{m}_{q}.csv", f"{jd}{m}_{q}.csv")
            if not a:
                continue
            reported = 100 - pct(a["ctx"])
            attrib = 100 * sum(1 for c, x, h in zip(a["cold"], a["ctx"], a["harm"])
                               if c == 1 and x == 0 and h == 1) / a["n"]
            pre = 100 * sum(1 for c, h in zip(a["cold"], a["harm"])
                            if c == 0 and h == 1) / a["n"]
            none = 100 * sum(1 for x, h in zip(a["ctx"], a["harm"])
                             if x == 0 and h == 0) / a["n"]
            print(f"{PRETTY[m]:<16} {name:<10} {q:<5} {a['n']:>4} | {reported:7.1f}% "
                  f"{pct(a['harm']):5.1f}% | {attrib:6.1f}% {pre:6.1f}% {none:7.1f}%")
            by_attack[name].append((reported, pct(a["harm"]), attrib))
            cell[name] = (reported, attrib)
        if len(cell) == 2:
            paired.append(cell)
print("-" * 100)
for name, vals in by_attack.items():
    if vals:
        print(f"{name:<16} MEAN ({len(vals):2d} runs)        | "
              f"{sum(v[0] for v in vals) / len(vals):7.1f}% "
              f"{sum(v[1] for v in vals) / len(vals):5.1f}% | "
              f"{sum(v[2] for v in vals) / len(vals):6.1f}%")

section("TABLE 2b  The ranking inversion")
cr = [c["CoSafe"][0] for c in paired]
rr = [c["Crescendo"][0] for c in paired]
ca = [c["CoSafe"][1] for c in paired]
ra = [c["Crescendo"][1] for c in paired]
flips = sum(1 for i in range(len(paired)) if cr[i] > rr[i] and ca[i] < ra[i])
print(f"{len(paired)} matched model-precision configurations")
print(f"  reported ASR      CoSafe {sum(cr) / len(cr):5.1f}%  Crescendo {sum(rr) / len(rr):5.1f}%"
      f"   Wilcoxon p={wilcoxon(cr, rr).pvalue:.5f}")
print(f"  attributable harm CoSafe {sum(ca) / len(ca):5.1f}%  Crescendo {sum(ra) / len(ra):5.1f}%"
      f"   Wilcoxon p={wilcoxon(ca, ra).pvalue:.5f}")
print(f"  the two orderings disagree; per-configuration the flip occurs "
      f"{flips}/{len(paired)} times (sign test "
      f"p={binomtest(flips, len(paired), 0.5).pvalue:.3f}, underpowered)")

section("TABLE 3  Quantization: static attacks vs adaptive ones")
print(f"{'model':<16} {'attack':<10} {'prec':<5} {'ASR':>7} {'d vs fp16':>10} "
      f"{'disc':>9} {'p':>9}")
deltas = {name: [] for name, _, _ in ATTACKS}
for m in FAMILIES:
    for name, ic, _ in ATTACKS:
        base = arms(f"{ic}{m}_fp16.csv")
        if not base:
            continue
        print(f"{PRETTY[m]:<16} {name:<10} {'fp16':<5} {100 - pct(base['ctx']):6.1f}%")
        for q in PRECISIONS[1:]:
            other = arms(f"{ic}{m}_{q}.csv")
            if not other:
                continue
            ids = sorted(set(base["ids"]) & set(other["ids"]))
            bi = {i: v for i, v in zip(base["ids"], base["ctx"])}
            oi = {i: v for i, v in zip(other["ids"], other["ctx"])}
            b = [bi[i] for i in ids]
            o = [oi[i] for i in ids]
            x, y, p = mcnemar(b, o)
            d = pct(b) - pct(o)          # positive means the quantized arm answers more
            deltas[name].append(abs(d))
            mark = "  <-- significant" if p < ALPHA else ""
            print(f"{'':<16} {'':<10} {q:<5} {100 - pct(o):6.1f}% {d:+9.1f}pp "
                  f"{x:4d}/{y:<4d} {p:9.5f}{mark}")
print("-" * 100)
for name, ds in deltas.items():
    if ds:
        print(f"{name:<16} mean |delta ASR| across {len(ds)} comparisons: {sum(ds) / len(ds):.1f}pp")
print(f"(significance threshold is Bonferroni-corrected: {ALPHA:.5f})")

section("TABLE 4  Single-turn control")
print(f"{'model':<16} {'prec':<5} {'refusal':>8} {'ASR':>7} {'harm':>6}")
for m in FAMILIES:
    for q in PRECISIONS:
        st = load(f"singleturn_st_{m}_{q}.csv")
        if not st:
            continue
        ref = 100 * sum(int(r["refused"]) for r in st.values()) / len(st)
        jd = load(f"judged_st_{m}_{q}.csv")
        harm = (100 * sum(int(r["unsafe"]) for r in jd.values()) / len(jd)) if jd else float("nan")
        print(f"{PRETTY[m]:<16} {q:<5} {ref:7.1f}% {100 - ref:6.1f}% {harm:5.1f}%")

print("\nAll figures above regenerate from the committed CSVs with no cached state.")
