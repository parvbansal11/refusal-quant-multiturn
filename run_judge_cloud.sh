#!/usr/bin/env bash
# Strict-judge rescore on a fresh CUDA pod.
# Generates real completions for every precision (single-turn AdvBench and
# multi-turn CoSafe), then scores them with Llama Guard 3 for harmful content.
#
# Prereqs: repo cloned into /workspace, HF login done (gated access to the
# target models AND meta-llama/Llama-Guard-3-8B), run under nohup:
#
#   cd /workspace/refusal-quant-multiturn
#   export HF_HOME=/workspace/hf
#   nohup bash run_judge_cloud.sh > /workspace/judge.log 2>&1 &
#   tail -f /workspace/judge.log
set -euo pipefail

if [ -d /workspace ]; then export HF_HOME=/workspace/hf; mkdir -p "$HF_HOME"; fi
export HF_HUB_DISABLE_XET=1
echo "cache: ${HF_HOME:-default}"; df -h "${HF_HOME:-$HOME}" | tail -1

pip install -q --extra-index-url https://download.pytorch.org/whl/cu128 -r runpod-environment-lock.txt
pip install -q scipy bitsandbytes
python -c "import torch,transformers;print('torch',torch.__version__,'transformers',transformers.__version__)"
python - <<'PY'
from huggingface_hub import HfApi
print("HF logged in as", HfApi().whoami()["name"])
PY

run () { echo; echo "### $*"; eval "$*"; }

# CoSafe scenarios (safe categories, 25 per category)
run "python cosafe_to_scenarios.py --per-category 25 --out scenarios.json"

# --- single-turn completions (AdvBench) ---
run "MODEL=3b            python dump_completions.py --mode single --tag st_3b_fp16"
run "MODEL=3b QUANT=nf4  python dump_completions.py --mode single --tag st_3b_nf4"
run "MODEL=8b            python dump_completions.py --mode single --tag st_8b_fp16"
run "MODEL=8b QUANT=awq  python dump_completions.py --mode single --tag st_8b_awq"
run "MODEL=8b QUANT=gptq python dump_completions.py --mode single --tag st_8b_gptq"

# --- multi-turn completions (CoSafe attack) ---
run "MODEL=3b            python dump_completions.py --mode multi --tag mt_3b_fp16"
run "MODEL=3b QUANT=nf4  python dump_completions.py --mode multi --tag mt_3b_nf4"
run "MODEL=8b            python dump_completions.py --mode multi --tag mt_8b_fp16"
run "MODEL=8b QUANT=awq  python dump_completions.py --mode multi --tag mt_8b_awq"
run "MODEL=8b QUANT=gptq python dump_completions.py --mode multi --tag mt_8b_gptq"

# --- judge everything with Llama Guard 3, then significance ---
run "python judge_harm.py"
run "python analyze_harm.py"
echo; echo "DONE. judged_*.csv hold per-item harm labels. Commit them off the pod."
