"""
Quantized runs for a Kaggle notebook (CUDA). Paste into a single cell.

Only the quantized precisions need CUDA, so this covers exactly the work that
cannot run on Apple Silicon: NF4, AWQ, and GPTQ across both multi-turn attacks
plus the single-turn control. FP16 runs stay local (run_local_fp16.sh).

Kaggle notes:
  - Turn Internet ON in the notebook settings (needed for model downloads).
  - Add your Hugging Face token as a Kaggle Secret named HF_TOKEN
    (Add-ons > Secrets), so gated Llama checkpoints resolve.
  - Sessions are time-limited and can be interrupted, so this is chunked and
    resumable: every step skips work whose output already exists in
    /kaggle/working, and that directory survives a committed notebook version.
  - Download the CSVs from /kaggle/working, or save them as a Kaggle Dataset,
    before the session ends.

Set WHICH below to pick a chunk that comfortably fits one session.
"""
import os, subprocess, sys, textwrap

WORK = "/kaggle/working"
REPO = f"{WORK}/refusal-quant-multiturn"
N = int(os.environ.get("N", "200"))

# One model family per chunk, all its precisions together, so each session
# downloads a base model once and reuses it. Sized for a 96GB card.
# Run B first: the 8B four-way sweep is the comparison the paper rests on.
CHUNKS = {
    "A": [("3b", "fp16"), ("3b", "nf4")],
    "B": [("8b", "fp16"), ("8b", "nf4"), ("8b", "awq"), ("8b", "gptq")],
    "C": [("qwen3b", "fp16"), ("qwen3b", "nf4")],
    "D": [("qwen7b", "fp16"), ("qwen7b", "nf4"), ("qwen7b", "awq"), ("qwen7b", "gptq")],
    "E": [("mistral7b", "fp16"), ("mistral7b", "nf4")],
    "JUDGE": [],          # run judge_harm.py over whatever completions exist
}
WHICH = os.environ.get("WHICH", "B")


def sh(cmd, check=False):
    print(f"\n### {cmd}", flush=True)
    r = subprocess.run(cmd, shell=True, cwd=REPO if os.path.isdir(REPO) else WORK)
    if check and r.returncode:
        sys.exit(f"failed: {cmd}")
    return r.returncode


def setup():
    if not os.path.isdir(REPO):
        sh(f"git clone https://github.com/parvbansal11/refusal-quant-multiturn {REPO}", check=True)
    else:
        sh("git pull -q")
    # pinned environment that produced the committed results
    sh("pip install -q --extra-index-url https://download.pytorch.org/whl/cu128 "
       "-r runpod-environment-lock.txt")
    sh("pip install -q scipy bitsandbytes")
    # Hugging Face token from Kaggle Secrets
    try:
        from kaggle_secrets import UserSecretsClient
        os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
        print("HF_TOKEN loaded from Kaggle Secrets")
    except Exception as e:
        print(f"WARNING: no HF_TOKEN secret ({e}); gated models will fail")
    os.environ.setdefault("HF_HOME", f"{WORK}/hf")
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    sh("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader")


def exists(name):
    return os.path.exists(os.path.join(REPO, name))


def run_cell(model, quant):
    tag = f"{model}_{quant}"
    envq = f"QUANT={quant} " if quant != "fp16" else ""
    base = f"MODEL={model} {envq}python"

    # direction and causal layer are per base model; regenerate if absent
    if not exists("refusal_direction_fp16.pt"):
        sh(f"MODEL={model} python refusal_direction.py")
        sh(f"MODEL={model} python ablation.py --quick")

    if not exists(f"incontext_{tag}.csv"):
        sh(f"{base} cosafe_incontext.py --n {N} --tag {tag}")
    if not exists(f"completions_mt_{tag}.csv"):
        sh(f"{base} dump_completions.py --mode multi --tag mt_{tag}")
    if not exists(f"incontext_cres_{tag}.csv"):
        sh(f"{base} crescendo_attack.py --n {N} --tag cres_{tag}")
    if not exists(f"singleturn_st_{tag}.csv"):
        sh(f"{base} singleturn_hardened.py --n {N} --tag st_{tag}")
    if not exists(f"completions_st_{tag}.csv"):
        sh(f"{base} dump_completions.py --mode single --tag st_{tag}")


def main():
    setup()
    sh(f"python cosafe_to_scenarios.py --per-category {max(N//4,25)} --out scenarios.json")

    if WHICH == "JUDGE":
        sh("python judge_harm.py")
        sh("python analyze_attack_attribution.py")
        sh("python analyze_refusal_vs_harm.py")
    else:
        current = None
        for model, quant in CHUNKS[WHICH]:
            print(f"\n{'='*60}\n{model} / {quant}\n{'='*60}", flush=True)
            # the direction and causal layer are per base model, so rebuild the
            # bundle only when the base model changes, not per precision
            if model != current:
                bundle = os.path.join(REPO, "refusal_direction_fp16.pt")
                if os.path.exists(bundle):
                    os.remove(bundle)
                current = model
            run_cell(model, quant)

    print("\nOutputs in", REPO)
    sh("ls -1 incontext_*.csv singleturn_st_*.csv completions_*.csv judged_*.csv 2>/dev/null | wc -l")
    print("Download these before the session ends, or save as a Kaggle Dataset.")


if __name__ == "__main__":
    main()
