"""Generate kaggle_fp16cuda.ipynb: the FP16-on-CUDA replication plus PAIR.

Why this run exists. Every FP16 arm in the repo was produced on Apple Silicon
and every quantized arm on CUDA, and run_runpod.sh's FRESH=1 path, which was
meant to redo FP16 on the pod, never executed (superseded_mps/ is empty and
skip_if kept the MPS files). So the quantization comparisons currently vary
precision and compute backend together. That is a problem specifically here,
because the mechanism the paper proposes is that adaptive attacks amplify small
perturbations, and a backend change is a small perturbation.

The fix is to re-run FP16 on the same hardware as the quantized arms, under the
tag fp16cuda so nothing is overwritten. That yields two comparisons:

    fp16cuda vs nf4     precision, hardware held fixed        the clean test
    fp16 vs fp16cuda    hardware, precision held fixed        the confound size

The second is the control the study was missing. If it is flat, the published
quantization numbers stand as they are.

PAIR runs last, both precisions in the same session, so the third attack is
never subject to the same problem.

    python make_kaggle_fp16cuda.py    ->  kaggle_fp16cuda.ipynb
"""
import json

setup = r'''# Setup: clone, pin environment, load HF token, show GPU.
import os, subprocess, sys, time, shutil
T0 = time.time()
WORK, REPO = "/kaggle/working", "/kaggle/working/refusal-quant-multiturn"
CACHE = "/kaggle/temp/hf"
os.environ["HF_HUB_CACHE"] = f"{CACHE}/hub"
os.environ["HF_HUB_DISABLE_XET"] = "1"

def sh(cmd, cwd=None, timeout=None):
    print(f"\n### {cmd}", flush=True)
    try:
        return subprocess.run(cmd, shell=True, timeout=timeout,
                              cwd=cwd or (REPO if os.path.isdir(REPO) else WORK)).returncode
    except subprocess.TimeoutExpired:
        print(f"TIMEOUT after {timeout / 60:.0f} min, moving on: {cmd}", flush=True)
        return 124

import socket
try:
    socket.getaddrinfo("github.com", 443)
    print("network OK")
except socket.gaierror:
    raise SystemExit(
        "\n" + "=" * 68 +
        "\nNO INTERNET ACCESS IN THIS NOTEBOOK.\n\n"
        "Fix: right panel > Session options > Internet -> On.\n"
        "If the toggle is greyed out, Kaggle needs a verified phone number:\n"
        "  kaggle.com/settings > Phone Verification.\n\n"
        "Also confirm Persistence is set to 'Files only' so results survive.\n"
        + "=" * 68)

if not os.path.isdir(REPO):
    sh(f"git clone -q https://github.com/parvbansal11/refusal-quant-multiturn {REPO}", cwd=WORK)
else:
    sh("git fetch -q origin && git reset -q --hard origin/main")
sh("git log --oneline -1")

sh("pip install -q --extra-index-url https://download.pytorch.org/whl/cu128 -r runners/runpod-environment-lock.txt")
sh("pip install -q scipy bitsandbytes")

try:
    from kaggle_secrets import UserSecretsClient
    os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
    print("HF_TOKEN loaded")
except Exception as e:
    print("WARNING: no HF_TOKEN secret ->", e)

sh("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader")
sh("python -c \"import torch,transformers;print('torch',torch.__version__,'tf',transformers.__version__)\"")
sh(f"df -h {WORK} /kaggle/temp")
print(f"\nsetup done in {(time.time()-T0)/60:.1f} min")
'''

runner = r'''# FP16 replication on CUDA, then PAIR. Time-aware and resumable: every step
# skips output that already exists, and nothing new starts near the limit, so
# Save Version still commits whatever finished.

N = 200
BUDGET_H = 10.5
DEADLINE = T0 + BUDGET_H * 3600

# Ordered by what the paper most needs. Qwen-3B first: it carries the triple
# (single-turn / CoSafe / Crescendo under one precision change), which is the
# cleanest evidence in the paper and therefore the thing most worth de-confounding.
FAMILIES = ["qwen3b", "3b", "8b", "qwen7b"]
# Qwen-3B is the only family that needs both PAIR arms; the dose-response claim
# rests on one model measured across three attacks, not on breadth.
PAIR_FAMILIES = ["qwen3b"]

def left_h():
    return (DEADLINE - time.time()) / 3600

def have(name):
    return os.path.exists(os.path.join(REPO, name))

def step(cmd, out, need_h=0.6):
    if have(out):
        print(f"skip (exists): {out}"); return True
    if left_h() < need_h:
        print(f"STOP: {left_h():.1f}h left, need ~{need_h}h for {out}"); return False
    sh(cmd, timeout=min(need_h * 3, max(left_h() - 0.2, 0.1)) * 3600)
    return True

def purge_cache():
    hub = os.path.join(CACHE, "hub")
    if os.path.isdir(hub):
        shutil.rmtree(hub, ignore_errors=True)

def causal_layer():
    import torch
    try:
        return torch.load(os.path.join(REPO, "bundles/refusal_direction_fp16.pt"),
                          map_location="cpu").get("ablation_best_layer")
    except Exception:
        return None

sh(f"python src/cosafe_to_scenarios.py --per-category {N//4} --out scenarios.json")

for M in FAMILIES:
    if left_h() < 1.0:
        print(f"\nSTOP: {left_h():.1f}h left, not starting {M}"); break
    print(f"\n{'#'*30} {M} fp16 on CUDA {'#'*30}", flush=True)
    TAG = f"{M}_fp16cuda"

    # The direction and its causal layer belong to the base model. Rebuild them
    # here rather than trusting a bundle built on other hardware.
    rm = os.path.join(REPO, "bundles/refusal_direction_fp16.pt")
    if os.path.exists(rm):
        os.remove(rm)
    sh(f"MODEL={M} python src/refusal_direction.py --n 128 --holdout 32",
       timeout=min(1.5, max(left_h() - 0.2, 0.1)) * 3600)
    sh(f"MODEL={M} python src/ablation.py",
       timeout=min(1.5, max(left_h() - 0.2, 0.1)) * 3600)
    L = causal_layer()
    if L is None:
        print(f"!! SKIP {M}: no causal layer, extraction or ablation failed above")
        purge_cache(); continue
    print(f"{M} causal layer {L}", flush=True)

    E = f"MODEL={M} QUANT=fp16 python"
    step(f"{E} cosafe_incontext.py --n {N} --tag {TAG}", f"incontext_{TAG}.csv", 1.0)
    step(f"{E} dump_completions.py --mode multi --n {N} --tag mt_{TAG}",
         f"completions_mt_{TAG}.csv", 0.8)
    step(f"{E} crescendo_attack.py --n {N} --tag cres_{TAG}",
         f"incontext_cres_{TAG}.csv", 2.0)
    step(f"{E} singleturn_hardened.py --n {N} --tag st_{TAG}",
         f"singleturn_st_{TAG}.csv", 0.8)
    step(f"{E} dump_completions.py --mode single --n {N} --tag st_{TAG}",
         f"completions_st_{TAG}.csv", 0.5)

    # PAIR, both precisions, same session and same card, so the third attack is
    # never confounded the way the first two were.
    if M in PAIR_FAMILIES:
        for Q in ["fp16", "nf4"]:
            PT = f"pair_{M}_{'fp16cuda' if Q == 'fp16' else Q}"
            step(f"MODEL={M} QUANT={Q} python src/pair_attack.py --n 100 --tag {PT}",
                 f"incontext_{PT}.csv", 2.5)

    purge_cache()

print(f"\ngeneration done, {left_h():.1f}h left", flush=True)
'''

judge = r'''# Label everything new with both judges, in one place, before the session ends.
# Judging before the clock runs out matters more than finishing generation: a
# session that dies with completions but no labels has produced nothing usable.
sh("python src/judge_harm.py", timeout=min(4.0, max(left_h() - 0.5, 0.1)) * 3600)
sh("python judge_harm2.py granite", timeout=min(3.0, max(left_h() - 0.3, 0.1)) * 3600)

sh("python src/analyze_paper.py")
sh("python src/analyze_agreement.py granite")

# Keep only the artefacts, not the weights, inside the 20GB output quota.
sh(f"du -sh {WORK} && ls {REPO}/*.csv | wc -l")
print(f"\nALL DONE in {(time.time()-T0)/3600:.1f}h")
'''

cells = [("code", setup), ("code", runner), ("code", judge)]
nb = {
    "cells": [{"cell_type": t, "metadata": {}, "source": s.splitlines(keepends=True),
               "outputs": [], "execution_count": None} for t, s in cells],
    "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3",
                                "language": "python"},
                 "language_info": {"name": "python"},
                 "accelerator": "GPU"},
    "nbformat": 4, "nbformat_minor": 5,
}
with open("kaggle_fp16cuda.ipynb", "w") as f:
    json.dump(nb, f, indent=1)
print("wrote kaggle_fp16cuda.ipynb")
