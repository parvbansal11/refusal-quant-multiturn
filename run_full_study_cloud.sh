#!/usr/bin/env bash
# Full study: 3 model families x precisions x 2 multi-turn attacks, n=200.
#
# Produces the same CSV formats the existing analysis reads, so every
# analyze_*.py script works unchanged on the expanded results.
#
# Run detached on a CUDA pod with a persistent volume:
#   cd /workspace && git clone <repo> && cd refusal-quant-multiturn
#   export HF_HOME=/workspace/hf && hf auth login
#   nohup bash run_full_study_cloud.sh > /workspace/study.log 2>&1 &
#   tail -f /workspace/study.log
#
# Resumable: every step skips work whose output already exists, so a dropped
# pod costs only the step in flight.
set -uo pipefail

N=${N:-200}
if [ -d /workspace ]; then export HF_HOME=/workspace/hf; mkdir -p "$HF_HOME"; fi
export HF_HUB_DISABLE_XET=1
echo "cache=${HF_HOME:-default}  n=$N"
df -h "${HF_HOME:-$HOME}" | tail -1

pip install -q --extra-index-url https://download.pytorch.org/whl/cu128 -r runpod-environment-lock.txt
pip install -q scipy bitsandbytes
python - <<'PY'
import torch, transformers
from huggingface_hub import HfApi
print("torch", torch.__version__, "transformers", transformers.__version__)
print("HF user:", HfApi().whoami()["name"])
PY

run () { echo; echo "### $*"; eval "$*" || echo "!! FAILED (continuing): $*"; }

python cosafe_to_scenarios.py --per-category 50 --out scenarios.json || true

# family: MODEL key -> precisions to sweep
# nf4 works everywhere via bitsandbytes; awq/gptq only where checkpoints exist.
run_family () {
  local M=$1; shift
  local PRECS=("$@")

  # direction + causal layer must be re-derived per base model
  run "MODEL=$M python refusal_direction.py"
  run "MODEL=$M python ablation.py --quick"

  for Q in "${PRECS[@]}"; do
    local ENVQ=""; [ "$Q" != "fp16" ] && ENVQ="QUANT=$Q"
    local TAG="${M}_${Q}"

    # single-turn control (AdvBench)
    run "MODEL=$M $ENVQ python singleturn_hardened.py --n $N --tag st_${TAG}"

    # attack 1: CoSafe coreference
    run "MODEL=$M $ENVQ python cosafe_incontext.py --n $N --tag ${TAG}"
    run "MODEL=$M $ENVQ python dump_completions.py --mode multi --tag mt_${TAG}"

    # attack 2: Crescendo escalation (writes its own completions)
    run "MODEL=$M $ENVQ python crescendo_attack.py --n $N --tag cres_${TAG}"

    # single-turn completions for the harm judge
    run "MODEL=$M $ENVQ python dump_completions.py --mode single --tag st_${TAG}"
  done
}

# ---- Llama family ----
run_family 3b  fp16 nf4
run_family 8b  fp16 nf4 awq gptq

# ---- Qwen family ----
run_family qwen3b fp16 nf4
run_family qwen7b fp16 nf4 awq gptq

# ---- Mistral family ----
run_family mistral7b fp16 nf4

# ---- judge everything, then analyse ----
run "python judge_harm.py"
run "python analyze_multiturn.py"
run "python analyze_singleturn.py"
run "python analyze_harm.py"
run "python analyze_attack_attribution.py"
run "python analyze_refusal_vs_harm.py"

echo
echo "DONE. Commit incontext_*.csv singleturn_st_*.csv completions_*.csv judged_*.csv off the pod."
