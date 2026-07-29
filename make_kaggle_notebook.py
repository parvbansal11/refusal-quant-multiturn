"""Generate kaggle_study.ipynb: a time-aware, resumable quantized-run notebook."""
import json

setup = r'''# Setup: clone, pin environment, load HF token, show GPU.
import os, subprocess, sys, time, shutil
T0 = time.time()
WORK, REPO = "/kaggle/working", "/kaggle/working/refusal-quant-multiturn"
# Weights go to scratch, not /kaggle/working: the working directory is the 20GB
# persisted output quota, and a single 8B checkpoint in fp16 fills most of it.
CACHE = "/kaggle/temp/hf"
os.environ["HF_HOME"] = CACHE
os.environ["HF_HUB_DISABLE_XET"] = "1"

def sh(cmd, cwd=None):
    print(f"\n### {cmd}", flush=True)
    return subprocess.run(cmd, shell=True, cwd=cwd or (REPO if os.path.isdir(REPO) else WORK)).returncode

# Preflight: without network access nothing below can work, and the run would
# otherwise emit hundreds of misleading "file not found" errors.
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
    sh("git pull -q")

sh("pip install -q --extra-index-url https://download.pytorch.org/whl/cu128 -r runpod-environment-lock.txt")
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

runner = r'''# Quantized runs only (CUDA-only work). FP16 runs on the local machine in
# parallel, so nothing here duplicates it.
#
# Time-aware: stops starting new work near the session limit and exits cleanly,
# so Save Version still commits everything finished so far.
# Resumable: every step skips output that already exists.

N = 200
BUDGET_H = 10.5          # stay under Kaggle's GPU session limit
DEADLINE = T0 + BUDGET_H * 3600

# priority order: the 8B four-way sweep is what the paper rests on
CHUNKS = [
    ("8b",        ["nf4", "awq", "gptq"]),
    ("3b",        ["nf4"]),
    ("qwen7b",    ["nf4", "awq", "gptq"]),
    ("qwen3b",    ["nf4"]),
    ("mistral7b", ["nf4"]),
]

def left_h():
    return (DEADLINE - time.time()) / 3600

def have(name):
    return os.path.exists(os.path.join(REPO, name))

def step(cmd, out, need_h=0.6):
    """Run cmd unless its output exists or too little time remains."""
    if have(out):
        print(f"skip (exists): {out}"); return True
    if left_h() < need_h:
        print(f"STOP: {left_h():.1f}h left, need ~{need_h}h for {out}"); return False
    sh(cmd)
    return True

def purge_cache():
    """Drop downloaded weights between model families. Five families at 4-16GB
    each overrun any disk here, and every checkpoint is used exactly once."""
    hub = os.path.join(CACHE, "hub")
    if os.path.isdir(hub):
        shutil.rmtree(hub, ignore_errors=True)

def causal_layer():
    """The layer ablation.py proved mediates refusal, or None if it did not run.
    Downstream scripts refuse without it, so check once instead of watching
    every attack fail in turn."""
    import torch
    try:
        return torch.load(os.path.join(REPO, "refusal_direction_fp16.pt"),
                          map_location="cpu").get("ablation_best_layer")
    except Exception:
        return None

sh(f"python cosafe_to_scenarios.py --per-category {N//4} --out scenarios.json")

for model, precs in CHUNKS:
    if left_h() < 1.0:
        print(f"\nout of budget ({left_h():.1f}h left), stopping before {model}")
        break
    print(f"\n{'='*64}\n{model}  precisions={precs}  |  {left_h():.1f}h left\n{'='*64}", flush=True)

    purge_cache()

    # direction + causal layer are per base model; rebuild once per model
    bundle = os.path.join(REPO, "refusal_direction_fp16.pt")
    if os.path.exists(bundle):
        os.remove(bundle)
    sh(f"MODEL={model} python refusal_direction.py")
    sh(f"MODEL={model} python ablation.py --quick")

    layer = causal_layer()
    if layer is None:
        print(f"SKIP {model}: no causal layer (direction or ablation failed above).")
        continue
    print(f"{model}: causal layer {layer}", flush=True)

    for q in precs:
        tag = f"{model}_{q}"
        env = f"MODEL={model} QUANT={q} python"
        print(f"\n--- {tag}  ({left_h():.1f}h left) ---", flush=True)
        ok = True
        ok &= step(f"{env} cosafe_incontext.py --n {N} --tag {tag}",               f"incontext_{tag}.csv")
        ok &= step(f"{env} dump_completions.py --mode multi --n {N} --tag mt_{tag}", f"completions_mt_{tag}.csv")
        ok &= step(f"{env} crescendo_attack.py --n {N} --tag cres_{tag}",          f"incontext_cres_{tag}.csv", need_h=1.0)
        ok &= step(f"{env} singleturn_hardened.py --n {N} --tag st_{tag}",         f"singleturn_st_{tag}.csv", need_h=0.4)
        ok &= step(f"{env} dump_completions.py --mode single --n {N} --tag st_{tag}", f"completions_st_{tag}.csv", need_h=0.4)
        if not ok:
            break

print(f"\nruns finished, {left_h():.1f}h left")
'''

judge = r'''# Score every completion with Llama Guard 3, then run the analyses.
# Safe to re-run: judge_harm.py skips files already judged.
if left_h() > 0.8:
    sh("python judge_harm.py")
else:
    print(f"skipping judge, only {left_h():.1f}h left (run it in the next session)")

for a in ["analyze_attack_attribution.py", "analyze_refusal_vs_harm.py",
          "analyze_multiturn.py", "analyze_harm.py", "analyze_singleturn.py"]:
    sh(f"python {a}")
'''

collect = r'''# Copy results to /kaggle/working root so Save Version captures them,
# and print what was produced.
import glob
os.makedirs(f"{WORK}/results", exist_ok=True)
n = 0
for pat in ["incontext_*.csv", "singleturn_st_*.csv", "completions_*.csv",
            "judged_*.csv", "*_meta.json"]:
    for f in glob.glob(os.path.join(REPO, pat)):
        shutil.copy(f, f"{WORK}/results/"); n += 1
print(f"copied {n} files to {WORK}/results")
print(f"total elapsed: {(time.time()-T0)/3600:.1f}h")
sh(f"ls -1 {WORK}/results | head -40", cwd=WORK)
'''

cells = [
    ("markdown", "# Quantized multi-turn safety study\n\n"
                 "Runs the CUDA-only precisions (NF4, AWQ, GPTQ) across model families and "
                 "both multi-turn attacks. FP16 runs locally in parallel.\n\n"
                 "**Before running:** Internet **On**, Persistence **Files only**, "
                 "and add a read-only `HF_TOKEN` under Add-ons > Secrets.\n\n"
                 "Time-aware and resumable: it stops cleanly near the session limit, and "
                 "re-running Save Version continues where it left off."),
    ("code", setup),
    ("markdown", "## Runs\nPriority order, 8B four-way sweep first."),
    ("code", runner),
    ("markdown", "## Harm judging and analysis"),
    ("code", judge),
    ("markdown", "## Collect outputs"),
    ("code", collect),
]

nb = {
    "cells": [
        ({"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}
         if kind == "markdown" else
         {"cell_type": "code", "execution_count": None, "metadata": {},
          "outputs": [], "source": src.splitlines(keepends=True)})
        for kind, src in cells
    ],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
        "accelerator": "GPU",
    },
    "nbformat": 4, "nbformat_minor": 5,
}
with open("kaggle_study.ipynb", "w") as f:
    json.dump(nb, f, indent=1)
print("wrote kaggle_study.ipynb")
