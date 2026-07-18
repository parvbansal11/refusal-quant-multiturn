#!/usr/bin/env bash
# One-shot single-turn pipeline for a fresh CUDA pod.
# Runs the 3B and 8B blocks and prints the significance table.
# Prereqs: repo cloned, cwd = repo root, HF login done (gated Llama access).
#
#   bash run_singleturn_cloud.sh
#
# Put the HF cache on a roomy volume. The 8B FP16 base is 16GB and AWQ/GPTQ are
# 5.7GB each, so a 20GB container disk is not enough. Mount a >=80GB volume at
# /workspace (this script points the cache there automatically) or resize the
# container disk to 80GB.
set -euo pipefail

if [ -d /workspace ]; then
  export HF_HOME=/workspace/hf
  mkdir -p "$HF_HOME"
fi
export HF_HUB_DISABLE_XET=1   # plain downloads, avoids the partial-write aborts

echo "cache: ${HF_HOME:-$HOME/.cache/huggingface}"
df -h "${HF_HOME:-$HOME}" | tail -1

pip install -q --extra-index-url https://download.pytorch.org/whl/cu128 -r runpod-environment-lock.txt
pip install -q scipy bitsandbytes
python -c "import torch,transformers;print('torch',torch.__version__,'transformers',transformers.__version__)"

python - <<'PY'
from huggingface_hub import HfApi
try:
    print("HF logged in as", HfApi().whoami()["name"])
except Exception:
    raise SystemExit("Not logged in. Run: huggingface-cli login  (gated Llama access needed)")
PY

run () { echo; echo "### $*"; eval "$*"; }

# 3B block (singleturn_st_3b_fp16.csv is already committed; nf4 is regenerated)
run "MODEL=3b python refusal_direction.py"
run "MODEL=3b python ablation.py --quick"                 # expect layer 20
run "MODEL=3b QUANT=nf4 python singleturn_hardened.py --n 100 --tag st_3b_nf4"

# 8B block
run "MODEL=8b python refusal_direction.py"
run "MODEL=8b python ablation.py --quick"                 # expect layer 13
run "MODEL=8b            python singleturn_hardened.py --n 100 --tag st_8b_fp16"
run "MODEL=8b QUANT=awq  python singleturn_hardened.py --n 100 --tag st_8b_awq"
run "MODEL=8b QUANT=gptq python singleturn_hardened.py --n 100 --tag st_8b_gptq"

run "python analyze_singleturn.py"
echo; echo "DONE. Results in singleturn_st_*.csv. Commit or copy them off the pod."
