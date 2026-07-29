#!/usr/bin/env bash
# Full FP16 study on Apple Silicon (MPS). No CUDA needed.
#
# Covers everything that does not require quantized kernels: both multi-turn
# attacks, three model families, and a larger sample. NF4/AWQ/GPTQ cannot run
# on MPS, so those precisions stay on the cloud budget (run_full_study_cloud.sh).
#
#   nohup bash run_local_fp16.sh > local_study.log 2>&1 &
#   tail -f local_study.log
#
# Resumable: each step skips work whose output already exists.
set -uo pipefail
N=${N:-200}
export PYTORCH_ENABLE_MPS_FALLBACK=1
PY=./venv/bin/python

echo "FP16 local study, n=$N, started $(date)"

run () { echo; echo "### $*"; eval "$*" || echo "!! FAILED (continuing): $*"; }
skip_if () { [ -f "$1" ] && { echo "skip (exists): $1"; return 0; } || return 1; }

$PY cosafe_to_scenarios.py --per-category $((N/4)) --out scenarios.json || true

for M in 3b qwen3b 8b qwen7b mistral7b; do
  echo; echo "######################## $M ########################"

  # direction + ablation-confirmed causal layer, per base model
  run "MODEL=$M $PY refusal_direction.py"
  run "MODEL=$M $PY ablation.py --quick"

  # attack 1: CoSafe coreference
  skip_if "incontext_${M}_fp16.csv" || \
    run "MODEL=$M $PY cosafe_incontext.py --n $N --tag ${M}_fp16"
  skip_if "completions_mt_${M}_fp16.csv" || \
    run "MODEL=$M $PY dump_completions.py --mode multi --tag mt_${M}_fp16"

  # attack 2: Crescendo escalation (emits its own completions)
  skip_if "incontext_cres_${M}_fp16.csv" || \
    run "MODEL=$M $PY crescendo_attack.py --n $N --tag cres_${M}_fp16"

  # single-turn control
  skip_if "singleturn_st_${M}_fp16.csv" || \
    run "MODEL=$M $PY singleturn_hardened.py --n $N --tag st_${M}_fp16"
  skip_if "completions_st_${M}_fp16.csv" || \
    run "MODEL=$M $PY dump_completions.py --mode single --tag st_${M}_fp16"
done

echo; echo "######## harm judging + analysis ########"
run "$PY judge_harm.py"
run "$PY analyze_attack_attribution.py"
run "$PY analyze_refusal_vs_harm.py"
run "$PY analyze_multiturn.py"
run "$PY analyze_harm.py"

echo; echo "DONE $(date)"
