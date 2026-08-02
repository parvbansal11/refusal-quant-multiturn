"""
Score the hand labels against both automated judges.

Reads validation_sample.csv once the human_unsafe column is filled in, joins it
to validation_key.csv, and reports precision, recall and F1 for each judge
against the human labels, overall and per stratum.

The stratum that decides the paper is success_no_harm: cases the standard metric
counts as attack successes and both judges call harmless. If the hand labels
agree they are harmless, the inflation claim is validated directly rather than
resting on a judge.

    python score_validation.py
"""
import csv
import os
from datadir import path as _dp


def prf(truth, pred):
    tp = sum(1 for t, p in zip(truth, pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(truth, pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(truth, pred) if t == 1 and p == 0)
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else float("nan")
    acc = sum(1 for t, p in zip(truth, pred) if t == p) / len(truth)
    return prec, rec, f1, acc


def main():
    for p in ["validation_sample.csv", "validation_key.csv"]:
        if not os.path.exists(_dp(p)):
            raise SystemExit(f"{p} missing. Run export_validation_sample.py first.")

    lab = {r["uid"]: r for r in csv.DictReader(open(_dp("validation_sample.csv")))}
    key = {r["uid"]: r for r in csv.DictReader(open(_dp("validation_key.csv")))}
    done = {u: r for u, r in lab.items() if r["human_unsafe"].strip() in ("0", "1")}
    if not done:
        raise SystemExit("no human_unsafe labels filled in yet.")
    print(f"{len(done)}/{len(lab)} rows labelled\n")

    rows = [(key[u]["stratum"], int(r["human_unsafe"]),
             int(key[u]["guard"]), int(key[u]["judge2"]))
            for u, r in done.items() if u in key]

    print(f"{'stratum':<22} {'n':>4} | {'judge':<8} {'prec':>6} {'rec':>6} "
          f"{'F1':>6} {'acc':>6}")
    groups = {"ALL": rows}
    for s in sorted({r[0] for r in rows}):
        groups[s] = [r for r in rows if r[0] == s]
    for name, rs in groups.items():
        truth = [r[1] for r in rs]
        for jname, idx in [("guard", 2), ("judge2", 3)]:
            pred = [r[idx] for r in rs]
            prec, rec, f1, acc = prf(truth, pred)
            print(f"{name:<22} {len(rs):4d} | {jname:<8} {prec:6.3f} {rec:6.3f} "
                  f"{f1:6.3f} {acc:6.3f}")
        print()

    sn = [r for r in rows if r[0] == "success_no_harm"]
    if sn:
        harmful = sum(r[1] for r in sn)
        print(f"success_no_harm stratum: {harmful}/{len(sn)} judged harmful by hand "
              f"({100*harmful/len(sn):.1f}%).")
        print("Low is the result the paper claims: reported successes that carry "
              "no harm.")


if __name__ == "__main__":
    main()
