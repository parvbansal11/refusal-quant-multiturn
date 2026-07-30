#!/usr/bin/env bash
# Complete study on one CUDA pod: every family, every precision that has a
# checkpoint, both multi-turn attacks, the single-turn control, harm judging,
# and every analysis. n=200 throughout, including the completions the judge
# scores, so the harm arm and the refusal arm cover the same scenarios.
#
# Needs Ampere or newer. GPTQ's Marlin kernels fail to build on Turing, and
# gptqmodel retries the build instead of reporting it, which costs the session.
#
#   cd /workspace && git clone https://github.com/parvbansal11/refusal-quant-multiturn
#   cd refusal-quant-multiturn && hf auth login     # Llama and Llama Guard are gated
#   FRESH=1 nohup bash run_runpod.sh > /workspace/study.log 2>&1 &
#   tail -f /workspace/study.log
#
# Resumable: every step skips work whose output already exists, so a dropped
# pod costs only the step in flight. Re-running the same command continues.
#
# FRESH=1 moves the FP16 results produced on Apple Silicon aside first, so
# every number in the final set comes off one device with one direction. The
# displaced files stay in superseded_mps/ as an independent replication.
#
# Split across GPUs by giving each its own clone and family list, sharing one
# weight cache, then judging once after the CSVs are merged into one directory:
#   CUDA_VISIBLE_DEVICES=0 FAMILIES="8b qwen7b"           SKIP_JUDGE=1 bash run_runpod.sh
#   CUDA_VISIBLE_DEVICES=1 FAMILIES="3b qwen3b mistral7b" SKIP_JUDGE=1 bash run_runpod.sh
set -uo pipefail

N=${N:-200}
FAMILIES=${FAMILIES:-"8b qwen7b mistral7b 3b qwen3b"}
PRECS=${PRECS:-"fp16 nf4 awq gptq"}
FRESH=${FRESH:-0}
SKIP_JUDGE=${SKIP_JUDGE:-0}
# 256 per class off AdvBench-520 with a quarter held out, rather than the 128
# the script defaults to: extraction is activations only, so a better direction
# and a real holdout cost minutes. ABLATE=--quick trades the full layer sweep
# for six candidate layers if the pod is short-lived.
DIRN=${DIRN:-256}
ABLATE=${ABLATE:-}

# Weights belong on the volume: the container's own disk is ~20GB and a single
# 8B checkpoint in fp16 fills most of it. Only the hub cache moves, not HF_HOME.
# Redirecting HF_HOME relocates the login token too, and `hf auth login` writes
# it under the default home, so the gated repos would 401 an hour in.
if [ -d /workspace ]; then export HF_HUB_CACHE=${HF_HUB_CACHE:-/workspace/hf/hub}; mkdir -p "$HF_HUB_CACHE"; fi
export HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false

echo "study start $(date)"
echo "families=$FAMILIES  precisions=$PRECS  n=$N  cache=${HF_HUB_CACHE:-default}"
df -h "${HF_HUB_CACHE:-$HOME}" | tail -1

run () { echo; echo "### $*"; eval "$*" || echo "!! FAILED (continuing): $*"; }
skip_if () { [ -f "$1" ] && { echo "skip (exists): $1"; return 0; } || return 1; }

pip install -q --extra-index-url https://download.pytorch.org/whl/cu128 -r runpod-environment-lock.txt
pip install -q scipy bitsandbytes

# Fail here rather than 40 minutes in: the gated repos and the GPU generation
# are both things a wrong pod or a missing token turns into a silent no-op.
python - <<'PY' || exit 1
import sys, torch, transformers
from huggingface_hub import HfApi
print("torch", torch.__version__, "transformers", transformers.__version__)
if not torch.cuda.is_available():
    sys.exit("no CUDA visible")
cc = torch.cuda.get_device_capability(0)
print(f"gpu {torch.cuda.get_device_name(0)}  sm_{cc[0]}{cc[1]}  "
      f"{torch.cuda.get_device_properties(0).total_memory/2**30:.0f}GB")
if cc[0] < 8:
    print("WARNING: pre-Ampere, GPTQ will be skipped")
try:
    print("HF user:", HfApi().whoami()["name"])
except Exception as e:
    sys.exit(f"not logged in to Hugging Face ({e}); run: hf auth login")
PY

GPTQ_OK=$(python -c "import torch;print(1 if torch.cuda.get_device_capability(0)[0]>=8 else 0)")

if [ "$FRESH" = "1" ]; then
  mkdir -p superseded_mps
  for f in incontext_*_fp16.csv incontext_*_fp16_meta.json incontext_cres_*_fp16.csv \
           incontext_cres_*_fp16_meta.json singleturn_st_*_fp16.csv \
           singleturn_st_*_fp16_meta.json completions_*_fp16.csv; do
    [ -e "$f" ] && mv "$f" superseded_mps/
  done
  echo "moved $(ls superseded_mps 2>/dev/null | wc -l) MPS FP16 files to superseded_mps/"
fi

python cosafe_to_scenarios.py --per-category $((N / 4)) --out scenarios.json || true

# Which precisions actually have a checkpoint for this family. nf4 is built at
# load time by bitsandbytes so it works anywhere; awq and gptq need a published
# 4-bit repo, and only some families have one.
avail () {
  MODEL=$1 python - <<'PY'
from refusal_direction import BASE_KEY, _AWQ, _GPTQ
ok = ["fp16", "nf4"]
if BASE_KEY in _AWQ:
    ok.append("awq")
if BASE_KEY in _GPTQ:
    ok.append("gptq")
print(" ".join(ok))
PY
}

for M in $FAMILIES; do
  echo; echo "######################## $M ########################"

  # The direction and its causal layer are properties of the base model, so
  # they are rebuilt once per family and shared by that family's precisions.
  # Removing the bundle first stops a previous family's layer leaking across.
  rm -f refusal_direction_fp16.pt
  run "MODEL=$M python refusal_direction.py --n $DIRN --holdout $((DIRN / 4))"
  run "MODEL=$M python ablation.py $ABLATE"

  LAYER=$(python -c "
import torch
try: print(torch.load('refusal_direction_fp16.pt', map_location='cpu').get('ablation_best_layer',''))
except Exception: print('')
" 2>/dev/null)
  if [ -z "$LAYER" ]; then
    echo "!! SKIP $M: no causal layer, extraction or ablation failed above"
    continue
  fi
  echo "$M causal layer $LAYER"

  for Q in $PRECS; do
    case " $(avail "$M") " in *" $Q "*) ;; *) echo "skip: no $Q checkpoint for $M"; continue ;; esac
    [ "$Q" = "gptq" ] && [ "$GPTQ_OK" != "1" ] && { echo "skip: gptq needs Ampere"; continue; }

    TAG="${M}_${Q}"
    echo; echo "-------- $TAG --------"
    E="MODEL=$M QUANT=$Q python"

    # attack 1: CoSafe coreference
    skip_if "incontext_${TAG}.csv" || \
      run "$E cosafe_incontext.py --n $N --tag $TAG"
    skip_if "completions_mt_${TAG}.csv" || \
      run "$E dump_completions.py --mode multi --n $N --tag mt_${TAG}"

    # attack 2: Crescendo escalation, which writes its own completions
    skip_if "incontext_cres_${TAG}.csv" || \
      run "$E crescendo_attack.py --n $N --tag cres_${TAG}"

    # single-turn control
    skip_if "singleturn_st_${TAG}.csv" || \
      run "$E singleturn_hardened.py --n $N --tag st_${TAG}"
    skip_if "completions_st_${TAG}.csv" || \
      run "$E dump_completions.py --mode single --n $N --tag st_${TAG}"
  done

  df -h "${HF_HUB_CACHE:-$HOME}" | tail -1
done

if [ "$SKIP_JUDGE" = "1" ]; then
  echo; echo "generation done $(date); judging skipped (SKIP_JUDGE=1)"
  exit 0
fi

echo; echo "######## harm judging ########"
run "python judge_harm.py"

echo; echo "######## analysis ########"
for A in analyze_attack_attribution.py analyze_refusal_vs_harm.py \
         analyze_multiturn.py analyze_harm.py analyze_singleturn.py; do
  run "python $A"
done
for F in make_attribution_figure.py make_hero_figures.py make_novelty_figure.py make_figures.py; do
  [ -f "$F" ] && run "python $F"
done

echo; echo "DONE $(date)"
ls -1 incontext_*.csv judged_*.csv 2>/dev/null | wc -l | xargs echo "result files:"
