# Refusal Direction Under Quantization and Multi-Turn Pressure

Interpretability study of how INT4 quantization affects a language model's
internal refusal representation across a multi-turn conversation.

**Model:** Llama-3.2-3B-Instruct (FP16 and INT4-AWQ)
**Author:** Parv Bansal, AIMS-DTU Research Internship 2026

## Research question

Recent work (Kadadekar, 2026, arXiv 2606.10154) found that INT4 quantization
causes a large drop in refusal *behaviour* while the refusal *direction* stays
geometrically intact (cosine > 0.97), and standard probes could not explain the
gap. Separately, STAR (2026, arXiv 2603.15684) showed that in full-precision
models the projection onto the refusal direction decays across conversation
turns. No work measures this trajectory under quantization.

**Q1.** Is the quantized model's refusal loss *static* (present from turn 0) or
*dynamic* (compounding across a multi-turn conversation)?

**Q2.** Does the projection onto the refusal direction decay differently, in
starting level or per-turn slope, under INT4 versus FP16?

This extends Kadadekar's stated open question (multi-turn) rather than combining
papers. It is a diagnostic study, not a new attack.

## Method

- Extract the refusal direction via difference-in-means (Arditi et al.).
- Select the functional layer by ablation (remove the direction, measure the
  drop in refusal). This gives the causal bottleneck, not just the most
  separable layer.
- Measure projection onto that direction across multi-turn escalation, FP16 vs
  INT4, with a length-matched neutral control.
- Analysis separates intercept (turn-0 level) from slope (per-turn decay) to
  distinguish the static and dynamic hypotheses.

## Files

| File | Purpose |
| :-- | :-- |
| `refusal_direction.py` | Extract the refusal direction (FP16, MPS), validate on a held-out split, save directions + raw activations. |
| `ablation.py` | Ablate each candidate layer's direction and find the causal refusal bottleneck. |

## Progress

**Day 1 (FP16, local, Apple Silicon):**
- Refusal direction extracted from Llama-3.2-3B; held-out separation accuracy 100%.
- Ablation sweep identifies **layer 20** as the causal refusal bottleneck:
  ablating its direction drops refusal from 81% to 0%, while a random-direction
  control stays at baseline.
- Raw activations saved, so layer analysis and the INT4 comparison need no
  re-extraction.

**Next:** multi-turn projection decay (FP16), then the INT4-AWQ comparison
(cloud, since AWQ kernels do not run on Apple Metal).

## Setup

```bash
python3 -m venv venv && source venv/bin/activate
pip install torch transformers accelerate datasets huggingface_hub numpy
# requires Hugging Face access to meta-llama/Llama-3.2-3B-Instruct
python refusal_direction.py        # extract + validate direction
python ablation.py --quick         # find the bottleneck layer
```

## References

- Arditi et al. Refusal in LLMs is mediated by a single direction. 2024.
- Kadadekar. Quality Is Not a Safety Proxy Under Quantization. arXiv:2606.10154, 2026.
- STAR. State-Dependent Safety Failures in Multi-Turn Language Model Interaction. arXiv:2603.15684, 2026.
- Chhabra & Khalili. Refusal in Compressed Models via Mechanistic Interpretability. arXiv:2504.04215, 2025.
