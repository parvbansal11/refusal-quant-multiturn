#!/usr/bin/env bash
# Ablation at the full 32 prompts and 11 candidate layers, per model family.
#
# The paper currently carries a limitation: layer selection was run with
# --quick, which is 16 prompts over 6 candidate layers. At 16 prompts the
# resolution is 6.25% per prompt, and Llama-8B picked layer 10 on one run and 12
# on another. This closes that.
#
# refusal_direction.py and ablation.py both read and write one shared bundle,
# so a run for the next model overwrites the last one's directions. Each model's
# bundle is copied out to refusal_direction_fp16_<model>.pt before moving on,
# and the pre-existing bundle is restored at the end.
#
#   nohup bash run_ablation_full.sh > ablation_full.log 2>&1 &
set -uo pipefail
export PYTORCH_ENABLE_MPS_FALLBACK=1
PY=./venv/bin/python
BUNDLE=bundles/refusal_direction_fp16.pt
MODELS=${MODELS:-"8b 3b qwen3b qwen7b mistral7b"}

[ -f "$BUNDLE" ] && cp "$BUNDLE" "$BUNDLE.prerun"
echo "full ablation sweep, models: $MODELS, started $(date)"

for M in $MODELS; do
  echo; echo "######################## $M ########################"
  OUT="bundles/refusal_direction_fp16_${M}.pt"
  if [ -f "$OUT" ]; then echo "skip (exists): $OUT"; continue; fi

  MODEL=$M $PY src/refusal_direction.py || { echo "!! direction failed: $M"; continue; }
  MODEL=$M $PY src/ablation.py || { echo "!! ablation failed: $M"; continue; }
  cp "$BUNDLE" "$OUT"
  $PY - "$OUT" <<'EOF'
import sys, torch
b = torch.load(sys.argv[1], map_location="cpu")
print(f"  {sys.argv[1]}  dprime_peak={b['best_layer']}  "
      f"ablation_layer={b['ablation_best_layer']}  "
      f"baseline={b['ablation_baseline']:.3f}")
print(f"  per-layer refusal after ablation: {b['ablation_results']}")
EOF
done

# put back whatever bundle was there before, so nothing downstream shifts under
# a run that was not asked for
[ -f "$BUNDLE.prerun" ] && mv "$BUNDLE.prerun" "$BUNDLE"
echo; echo "DONE $(date)"
