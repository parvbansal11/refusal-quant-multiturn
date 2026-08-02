"""
Blind sample for hand labelling.

Automated judges disagreeing with each other is a reason to look at the text,
not a reason to pick one. This pulls a stratified sample across the cases the
paper actually rests on and writes it without any verdict attached, so the
labels come back uncontaminated. The key is written separately and is not needed
until scoring.

Strata, forty each:

  disagree_guard_only   Llama Guard says harmful, second judge says safe
  disagree_judge2_only  the reverse
  success_no_harm       counted as attack success, both judges say no harm.
                        This is the stratum the central claim lives in.
  attributable          refused cold, answered in context, both judges harmful
  pre_existing          answered cold as well, so the attack added nothing

    python export_validation_sample.py granite
    -> validation_sample.csv   (blind, hand-label the human_unsafe column)
    -> validation_key.csv      (do not open until labelling is finished)
"""
import csv
import os
from datadir import path as _dp
import random
import sys

FAMILIES = ["3b", "qwen3b", "8b", "qwen7b", "mistral7b"]
PRECISIONS = ["fp16", "nf4", "awq", "gptq"]
ATTACKS = [("CoSafe", "incontext_", "mt_"), ("Crescendo", "incontext_cres_", "cres_")]
PER_STRATUM = 40
SEED = 0


def load(path):
    if not os.path.exists(_dp(path)):
        return None
    rows = list(csv.DictReader(open(_dp(path))))
    if not rows:
        return None
    key = "id" if "id" in rows[0] else "idx"
    return {r[key]: r for r in rows}


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "granite"
    strata = {k: [] for k in ["disagree_guard_only", "disagree_judge2_only",
                              "success_no_harm", "attributable", "pre_existing"]}

    for m in FAMILIES:
        for q in PRECISIONS:
            for attack, ic, kind in ATTACKS:
                I = load(f"{ic}{m}_{q}.csv")
                G = load(f"judged_{kind}{m}_{q}.csv")
                S = load(f"judged2_{tag}_{kind}{m}_{q}.csv")
                if not I or not G or not S:
                    continue
                for i in sorted(set(I) & set(G) & set(S)):
                    g, s = int(G[i]["unsafe"]), int(S[i]["unsafe"])
                    cold = int(I[i]["refused_cold"])
                    ctx = int(I[i]["refused_incontext"])
                    rec = dict(uid=f"{m}_{q}_{kind}{i}", model=m, precision=q,
                               attack=attack, request=G[i].get("final_user", ""),
                               response=G[i]["response"], guard=g, judge2=s,
                               refused_cold=cold, refused_incontext=ctx)
                    if g == 1 and s == 0:
                        strata["disagree_guard_only"].append(rec)
                    elif g == 0 and s == 1:
                        strata["disagree_judge2_only"].append(rec)
                    if ctx == 0 and g == 0 and s == 0:
                        strata["success_no_harm"].append(rec)
                    if cold == 1 and ctx == 0 and g == 1 and s == 1:
                        strata["attributable"].append(rec)
                    if cold == 0 and g == 1 and s == 1:
                        strata["pre_existing"].append(rec)

    rng = random.Random(SEED)
    picked = []
    for name, pool in strata.items():
        take = rng.sample(pool, min(PER_STRATUM, len(pool)))
        for r in take:
            r["stratum"] = name
        picked += take
        print(f"{name:<22} pool {len(pool):5d}  sampled {len(take)}")

    if not picked:
        raise SystemExit(f"nothing to sample. Run judge_harm2.py {tag} first.")
    rng.shuffle(picked)

    # Blind file: request, response, and an empty column. No verdicts, no
    # stratum, and shuffled, so nothing signals what the answer should be.
    with open(_dp("validation_sample.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["uid", "request", "response", "human_unsafe"])
        w.writeheader()
        for r in picked:
            w.writerow({"uid": r["uid"], "request": r["request"],
                        "response": r["response"], "human_unsafe": ""})

    with open(_dp("validation_key.csv"), "w", newline="") as f:
        cols = ["uid", "stratum", "model", "precision", "attack", "guard",
                "judge2", "refused_cold", "refused_incontext"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in picked:
            w.writerow({k: r[k] for k in cols})

    print(f"\n{len(picked)} rows -> validation_sample.csv")
    print("Label human_unsafe as 1 (harmful) or 0 (not), then run score_validation.py")


if __name__ == "__main__":
    main()
