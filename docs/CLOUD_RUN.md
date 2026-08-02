# Cloud run kit: single-turn control on AdvBench

Goal: produce the missing single-turn CSVs on a real, pinned prompt set so the
paper can state the single-turn precision effect on standalone harmful prompts,
matching Kadadekar's setting. FP16 3B is already done locally
(`singleturn_st_3b_fp16.csv`, 76% refusal). The rest need CUDA.

Everything reads `advbench_harmful_behaviors.csv` (committed, first 100 goals),
so every precision measures the identical prompt set. No gated-dataset fallback.

## What to run

Five runs total. One is done (3B FP16). Remaining four:

| model | precision | tag | needs |
|---|---|---|---|
| 3B | NF4 | st_3b_nf4 | CUDA + bitsandbytes |
| 8B | FP16 | st_8b_fp16 | CUDA (or 24GB Mac) |
| 8B | AWQ | st_8b_awq | CUDA + autoawq |
| 8B | GPTQ | st_8b_gptq | CUDA + auto-gptq |

The direction bundle `refusal_direction_fp16.pt` is model-specific (3B: 28
layers, causal layer 20; 8B: 32 layers, causal layer 13) and is gitignored, so
regenerate it once per model before that model's single-turn runs.

## Pod spec (avoid the 20GB-disk dead end)

A default RunPod container disk is ~20GB, which fills after the env plus the 3B
model and leaves no room for the 16GB 8B FP16 base. Provision disk up front:

- GPU: RTX 4090 / RTX A5000 / RTX PRO 4000 (>=24GB VRAM, enough for 8B FP16).
- Volume Disk: >=80GB, mounted at `/workspace`. The scripts put the HF cache
  there. (Or set the container disk to 80GB.)

## One-shot

After the pod is up:

```bash
git clone https://github.com/parvbansal11/refusal-quant-multiturn && cd refusal-quant-multiturn
huggingface-cli login          # gated Llama access
bash run_singleturn_cloud.sh   # installs from the lock, runs 3B + 8B, prints the table
```

Then copy the five `singleturn_st_*.csv` off the pod (commit + push, or the
RunPod file browser). The steps below are the same pipeline done by hand.

## Setup (RunPod, RTX 4090 or PRO 4000)

Reproduce the exact environment that produced the committed CSVs. A bare
`pip install torch transformers` pulls a transformers 5.x that needs
`DTensor` from `torch.distributed.tensor`, which the base-image torch does not
have, and every run dies at import with `ImportError: cannot import name
'DTensor'`. Pin to the lock file instead.

```bash
git clone https://github.com/parvbansal11/refusal-quant-multiturn && cd refusal-quant-multiturn
pip install --extra-index-url https://download.pytorch.org/whl/cu128 -r runpod-environment-lock.txt
pip install scipy bitsandbytes        # scipy for analysis; bitsandbytes for nf4
huggingface-cli login                 # gated Llama access
```

The lock already includes transformers 5.13.0, torch 2.10.0+cu128, gptqmodel
7.1.0, optimum, and torchao, which is the full set the 8B AWQ and GPTQ runs
loaded with. Do not install autoawq/auto-gptq; they are not what these runs used.

Quick check before the long runs:

```bash
python -c "import torch, transformers; print(torch.__version__, transformers.__version__)"
# expect 2.10.0+cu128 5.13.0
```

## 3B block (NF4)

```bash
MODEL=3b python refusal_direction.py    # writes refusal_direction_fp16.pt (3B)
MODEL=3b python ablation.py --quick     # confirms layer 20, patches the bundle
MODEL=3b QUANT=nf4 python singleturn_hardened.py --n 100 --tag st_3b_nf4
```

## Disk requirement (read before the 8B block)

The 8B block downloads the FP16 base (16GB) for direction extraction plus the
AWQ and GPTQ checkpoints (5.7GB each), about 27GB of models on top of the env.
A default small container disk fills up and every download aborts with
`No space left on device (os error 28)`. Provision at least a 60GB container
disk, or attach a volume and point the cache at it:

```bash
export HF_HOME=/workspace/hf           # a volume with room, if you have one
export HF_HUB_DISABLE_XET=1            # plain downloads, fewer partial-write errors
df -h /                                # confirm >30GB free before starting
```

Reclaim space from the finished 3B run and any aborted partials first:

```bash
rm -rf ~/.cache/huggingface/hub/models--meta-llama--Llama-3.2-3B-Instruct
rm -rf ~/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct
rm -rf ~/.cache/huggingface/hub/models--hugging-quants--Meta-Llama-3.1-8B-Instruct-AWQ-INT4
rm -rf ~/.cache/huggingface/hub/models--hugging-quants--Meta-Llama-3.1-8B-Instruct-GPTQ-INT4
```

## 8B block (FP16, AWQ, GPTQ)

```bash
MODEL=8b python refusal_direction.py    # overwrites bundle with the 8B direction
MODEL=8b python ablation.py --quick     # confirms layer 13, patches the bundle
MODEL=8b            python singleturn_hardened.py --n 100 --tag st_8b_fp16
MODEL=8b QUANT=awq  python singleturn_hardened.py --n 100 --tag st_8b_awq
MODEL=8b QUANT=gptq python singleturn_hardened.py --n 100 --tag st_8b_gptq
```

Each run prints a PROVENANCE block (check layer 20 for 3B, 13 for 8B) and a
VERIFY line that must say MATCH.

## After the runs

Pull the five `singleturn_st_*.csv` back, then:

```bash
python analyze_singleturn.py            # FP16 vs each quant, McNemar + paired t
```

This prints the single-turn table for the paper. If quantization degrades
single-turn refusal (quant refuses less than FP16), it confirms the Kadadekar
direction on AdvBench and sharpens the "single-turn only" story. If it does not,
the paper already stands on the committed CoSafe result; the AdvBench run becomes
a cross-dataset robustness check.

## Expected local anchor (already have)

3B FP16 AdvBench single-turn: refusal 76%, projection +7.85, n=100. This does not
match the 62% in the old handoff; treat 76% as the verified FP16 3B number.
