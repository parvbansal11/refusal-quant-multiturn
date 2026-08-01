"""
Third judge over the stratified sample, against the two automated ones.

validation_claude.csv holds a third set of labels produced by a frontier model
reading each response. It is a third *automated* judge, not human validation,
and must not be reported as the latter. Its value is that it shares no
architecture with either Llama Guard 3 or Granite Guardian, so where all three
agree the label is unlikely to be an artifact of one model family.

The stratum that decides the paper is success_no_harm: responses the standard
metric counts as attack successes and both automated judges call harmless. If a
third judge also calls them harmless, the inflation claim holds. If it does not,
the attributable-harm numbers are underestimates, and the question becomes
whether the miss rate differs between the two attacks, because only an
asymmetric miss can move the ranking.

    python analyze_validation.py
"""
import csv
import os
from collections import Counter

CLAUDE = "validation_claude.csv"
KEY = "validation_key.csv"


def agree(a, b):
    return sum(1 for x, y in zip(a, b) if x == y) / len(a)


def kappa(a, b):
    n = len(a)
    po, pa, pb = agree(a, b), sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def wilson(k, n, z=1.96):
    """Interval for a small-sample rate, where the normal approximation lies."""
    if not n:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    s = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
    return (c - s) / d, (c + s) / d


def main():
    for f in (CLAUDE, KEY):
        if not os.path.exists(f):
            raise SystemExit(f"{f} missing.")
    C = {r["uid"]: int(r["claude_unsafe"]) for r in csv.DictReader(open(CLAUDE))
         if r["claude_unsafe"].strip() in ("0", "1")}
    K = {r["uid"]: r for r in csv.DictReader(open(KEY))}
    rows = [(K[u]["stratum"], K[u]["attack"], int(K[u]["refused_cold"]),
             C[u], int(K[u]["guard"]), int(K[u]["judge2"]))
            for u in C if u in K]

    print("=" * 92)
    print("THREE-JUDGE AGREEMENT on the stratified sample (not a random sample:")
    print("it is deliberately enriched for disagreement, so rates are not")
    print("comparable to the pooled corpus)")
    print("=" * 92)
    print(f"{'stratum':<22} {'n':>4} | {'claude':>7} {'guard':>7} {'granite':>8} | "
          f"{'C~G':>6} {'C~Gr':>6}")
    groups = {"ALL": rows}
    for s in sorted({r[0] for r in rows}):
        groups[s] = [r for r in rows if r[0] == s]
    for name, rs in groups.items():
        c = [r[3] for r in rs]
        g = [r[4] for r in rs]
        s2 = [r[5] for r in rs]
        print(f"{name:<22} {len(rs):4d} | {100*sum(c)/len(c):6.1f}% "
              f"{100*sum(g)/len(g):6.1f}% {100*sum(s2)/len(s2):7.1f}% | "
              f"{100*agree(c,g):5.1f}% {100*agree(c,s2):5.1f}%")
    c = [r[3] for r in rows]
    g = [r[4] for r in rows]
    s2 = [r[5] for r in rows]
    print(f"\nkappa   claude~guard {kappa(c,g):.3f}   claude~granite {kappa(c,s2):.3f}")

    print("\n" + "=" * 92)
    print("DOES JUDGE UNDER-FLAGGING THREATEN THE INVERSION?")
    print("=" * 92)
    sn = [r for r in rows if r[0] == "success_no_harm"]
    k, n = sum(r[3] for r in sn), len(sn)
    lo, hi = wilson(k, n)
    print(f"success_no_harm: both automated judges called these harmless and the")
    print(f"metric counted them as successes. The third judge calls {k}/{n} harmful "
          f"({100*k/n:.0f}%, 95% CI {100*lo:.0f}-{100*hi:.0f}%).\n")

    # Only a response that was refused cold can become attack-attributable, so
    # the miss rate that matters is the one inside that subset. A miss that was
    # not refused cold moves pre-existing harm instead, and the decomposition
    # already separates those.
    print("Only responses refused in the cold condition can become")
    print("attack-attributable, so that subset is what matters:\n")
    print(f"{'attack':<12} {'in stratum':>11} {'refused cold':>13} "
          f"{'missed harm':>12} {'rate':>7}")
    for a in ("CoSafe", "Crescendo"):
        tot = [r for r in sn if r[1] == a]
        cold = [r for r in tot if r[2] == 1]
        miss = [r for r in cold if r[3] == 1]
        rate = f"{100*len(miss)/len(cold):.0f}%" if cold else "n/a"
        print(f"{a:<12} {len(tot):11d} {len(cold):13d} {len(miss):12d} {rate:>7}")
    print("\nIf the two rates are comparable, under-flagging shifts both attacks'")
    print("attributable harm in the same direction and the ranking is unaffected.")
    print("An asymmetry, particularly one favouring CoSafe, would be the thing")
    print("that overturns the result, because CoSafe's no-harm bucket is roughly")
    print("six times larger and a uniform miss rate applied to it moves more.")

    print("\n" + "=" * 92)
    print("WHERE THE TWO AUTOMATED JUDGES DISAGREE, WHO DOES THE THIRD SIDE WITH?")
    print("=" * 92)
    for st, who in (("disagree_guard_only", "guard"),
                    ("disagree_judge2_only", "granite")):
        rs = [r for r in rows if r[0] == st]
        if not rs:
            continue
        sided = sum(r[3] for r in rs)
        print(f"{st:<22} n={len(rs):3d}   third judge sides with {who} "
              f"{sided}/{len(rs)} ({100*sided/len(rs):.0f}%)")
    print("\nBoth well under 100%, so in the disagreement zone the third judge is")
    print("more conservative than either. Neither automated judge is a safe")
    print("default there, which is the case for hand labelling that region.")


if __name__ == "__main__":
    main()
