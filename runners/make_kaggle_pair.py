"""Generate kaggle_pair.ipynb: a focused re-run of the PAIR attack.

The first PAIR run was invalid because the attacker model (Llama-3.2-3B) refused
on 61% of rungs and those refusals were scored as attack prompts. The fix, now in
src/pair_attack.py, is a Mistral-7B attacker (18.5% refusal), a refusal check on
the attacker's own output, and one resampled retry per rung. This notebook runs
the corrected attack on Qwen2.5-3B at FP16 and NF4, on one T4, with the attacker
loaded in 4-bit so it shares the card with the target.

It removes the stale PAIR artifacts first, so the corrected outputs replace them
rather than being skipped, then labels them with both harm judges.

    python runners/make_kaggle_pair.py   ->  runners/kaggle_pair.ipynb
"""
import json
import os

setup = r'''# Setup: clone latest, pin environment, load HF token, show GPU.
import os, subprocess, time
T0 = time.time()
WORK, REPO = "/kaggle/working", "/kaggle/working/refusal-quant-multiturn"
CACHE = "/kaggle/temp/hf"
os.environ["HF_HUB_CACHE"] = f"{CACHE}/hub"
os.environ["HF_HUB_DISABLE_XET"] = "1"

def sh(cmd, timeout=None):
    print(f"\n### {cmd}", flush=True)
    try:
        return subprocess.run(cmd, shell=True, timeout=timeout,
                              cwd=REPO if os.path.isdir(REPO) else WORK).returncode
    except subprocess.TimeoutExpired:
        print(f"TIMEOUT after {timeout/60:.0f} min: {cmd}", flush=True)
        return 124

import socket
try:
    socket.getaddrinfo("github.com", 443); print("network OK")
except socket.gaierror:
    raise SystemExit("No internet. Session options > Internet -> On.")

if not os.path.isdir(REPO):
    sh(f"git clone -q https://github.com/parvbansal11/refusal-quant-multiturn {REPO}")
else:
    sh("git fetch -q origin && git reset -q --hard origin/main")
sh("git log --oneline -1")

sh("pip install -q --extra-index-url https://download.pytorch.org/whl/cu128 -r runners/runpod-environment-lock.txt")
sh("pip install -q scipy bitsandbytes")

# The secret service occasionally errors transiently at startup, so retry a few
# times. The attacker (Mistral) and Llama Guard are gated, so a missing token
# means the run cannot proceed; fail early rather than after loading models.
tok = None
for attempt in range(5):
    try:
        from kaggle_secrets import UserSecretsClient
        tok = UserSecretsClient().get_secret("HF_TOKEN")
        break
    except Exception as e:
        print(f"HF_TOKEN attempt {attempt+1} failed: {e}", flush=True)
        time.sleep(10)
if tok:
    os.environ["HF_TOKEN"] = tok
    print("HF_TOKEN loaded")
else:
    raise SystemExit(
        "HF_TOKEN not available after retries. Attach it under Add-ons > Secrets "
        "(name HF_TOKEN) and re-run. The gated attacker and Llama Guard need it.")

sh("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader")
print(f"setup done in {(time.time()-T0)/60:.1f} min")
'''

run = r'''# Re-run PAIR on Qwen2.5-3B, FP16 and NF4, corrected attacker.
N = 100
os.environ["ATTACKER_QUANT"] = "nf4"   # 4-bit Mistral attacker fits beside the target on one T4

os.makedirs(os.path.join(REPO, "bundles"), exist_ok=True)  # gitignored, absent in a fresh clone

# The stale PAIR files are committed; remove them so the corrected run is not
# skipped and the old labels are not reused.
for group, names in [
    ("incontext", ["incontext_pair_qwen3b_fp16cuda.csv", "incontext_pair_qwen3b_nf4.csv",
                    "incontext_pair_qwen3b_fp16cuda_meta.json", "incontext_pair_qwen3b_nf4_meta.json"]),
    ("completions", ["completions_pair_qwen3b_fp16cuda.csv", "completions_pair_qwen3b_nf4.csv"]),
    ("judged", ["judged_pair_qwen3b_fp16cuda.csv", "judged_pair_qwen3b_nf4.csv",
                "judged2_granite_pair_qwen3b_fp16cuda.csv", "judged2_granite_pair_qwen3b_nf4.csv"]),
]:
    for n in names:
        p = os.path.join(REPO, "data", group, n)
        if os.path.exists(p):
            os.remove(p); print("removed", n)

# The refusal-direction bundle is gitignored, so it is absent from the clone.
# pair_attack still needs the direction vector (not just the layer index), so
# rebuild it for Qwen-3B here. This needs no HF token (Qwen and Alpaca are
# ungated) and takes a few minutes. Then --layer 24 selects the causal layer
# the committed 32-prompt ablation identified.
LAYER = 24
sh("MODEL=qwen3b python src/refusal_direction.py --n 128 --holdout 32", timeout=0.75*3600)

sh(f"MODEL=qwen3b QUANT=fp16 ATTACKER_QUANT=nf4 python src/pair_attack.py "
   f"--n {N} --layer {LAYER} --tag pair_qwen3b_fp16cuda", timeout=2.5*3600)
sh(f"MODEL=qwen3b QUANT=nf4 ATTACKER_QUANT=nf4 python src/pair_attack.py "
   f"--n {N} --layer {LAYER} --tag pair_qwen3b_nf4", timeout=2.5*3600)
'''

judge = r'''# Label the new PAIR completions with both judges, then summarise.
sh("python src/judge_harm.py", timeout=1.5*3600)
sh("python src/judge_harm2.py granite", timeout=1.5*3600)

# Quick read-out: reported ASR, attacker-declined rate, harm by each judge.
import csv, glob
for arm in ["pair_qwen3b_fp16cuda", "pair_qwen3b_nf4"]:
    ic = os.path.join(REPO, "data", "incontext", f"incontext_{arm}.csv")
    if not os.path.exists(ic):
        print(arm, "MISSING"); continue
    rows = list(csv.DictReader(open(ic)))
    asr = 100*sum(1 for r in rows if int(r["refused_incontext"]) == 0)/len(rows)
    dec = sum(int(r.get("attacker_declined", 0)) for r in rows)
    q = sum(int(r["queries"]) for r in rows)
    line = f"{arm}: reported ASR {asr:.1f}%  attacker-declined {dec}/{q} rungs"
    for jd, tag in [("judged", "guard"), ("judged", "granite")]:
        jp = os.path.join(REPO, "data", "judged",
                          f"{'judged2_granite_' if tag=='granite' else 'judged_'}{arm}.csv")
        if os.path.exists(jp):
            jr = list(csv.DictReader(open(jp)))
            harm = 100*sum(int(r["unsafe"]) for r in jr)/len(jr)
            line += f"  {tag}-harm {harm:.1f}%"
    print(line)
print(f"\nDONE in {(time.time()-T0)/3600:.1f}h. Download data/ from the Output tab.")
'''

cells = [setup, run, judge]
nb = {
    "cells": [{"cell_type": "code", "metadata": {}, "source": c.splitlines(keepends=True),
               "outputs": [], "execution_count": None} for c in cells],
    "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3",
                                "language": "python"},
                 "language_info": {"name": "python"}, "accelerator": "GPU"},
    "nbformat": 4, "nbformat_minor": 5,
}
out = os.path.join(os.path.dirname(__file__), "kaggle_pair.ipynb")
with open(out, "w") as f:
    json.dump(nb, f, indent=1)
print("wrote", out)
