# Refusal Direction Under Quantization and Multi-Turn Pressure

Interpretability study of how INT4 quantization affects a language model's
internal refusal representation across a multi-turn conversation.

**Model:** Llama-3.2-3B-Instruct (FP16 and INT4)
**Author:** Parv Bansal, AIMS-DTU Research Internship 2026

> Private research repository. Contains standard safety benchmark prompts used
> for measuring refusal behaviour. Not for redistribution.

## Research question

Kadadekar (2026, arXiv 2606.10154) found INT4 quantization causes a large drop
in *single-turn* refusal behaviour while the refusal *direction* stays
geometrically intact. STAR (2026, arXiv 2603.15684) showed the projection onto
the refusal direction decays across conversation turns in full-precision models.
This work asks whether quantization changes that multi-turn behaviour: is a
quantized model's refusal loss static or dynamic, and does it show up in the
refusal-direction projection?

Extends Kadadekar's stated multi-turn future work; a diagnostic study, not a
new attack.

## Method

- Extract the refusal direction via difference-in-means (Arditi et al.).
- Select the functional layer by ablation (remove the direction, measure the
  drop in refusal). Causal bottleneck for this model: layer 20.
- Measure projection onto that direction and jailbreak rate on published
  multi-turn attacks (CoSafe, safe categories), FP16 vs INT4, on the same
  scenarios (paired), with proper significance tests.

## Current status (Day 4)

**Primary result: a clean, well-powered null.** On 100 CoSafe multi-turn
attacks, nf4 4-bit quantization does not measurably degrade Llama-3.2-3B's
jailbreak robustness (FP16 and INT4 both 77% ASR, McNemar p = 0.617) or shift
the refusal-direction projection (paired t p = 0.571, dz = 0.057). See
`RESULTS.md` for the full table and stats.

An earlier quick-run (20 scenarios) suggested a small INT4 effect; it did not
survive the full run or the significance test. Documented as a lesson.

**Next:** test AWQ (Kadadekar's degradation was AWQ/GPTQ-specific), to check
whether the null is nf4-specific or robust across quantization methods.

## Files

| File | Purpose |
| :-- | :-- |
| `refusal_direction.py` | Extract the refusal direction; FP16 or 4-bit via QUANT env var. |
| `ablation.py` | Ablate each candidate layer's direction; find the causal bottleneck. |
| `cosafe_to_scenarios.py` | Convert CoSafe (safe categories) to scenarios.json. |
| `cosafe_incontext.py` | Jailbreak rate + projection on CoSafe final turns, cold vs in-context. |
| `scenario_projection.py` | Decay + recovery tracker across multi-turn scenarios. |
| `multiturn_projection.py` | Projection vs benign-context depth. |
| `analyze_results.py` | Paired significance tests (McNemar + paired t) on the CSVs. |

## Setup

```bash
python3 -m venv venv && source venv/bin/activate
pip install torch transformers accelerate datasets bitsandbytes huggingface_hub
# FP16 work runs on Mac (MPS); 4-bit requires CUDA.
python refusal_direction.py            # extract direction (FP16)
python ablation.py --quick             # find the bottleneck layer
python cosafe_to_scenarios.py          # build scenarios.json
python cosafe_incontext.py --n 100 --tag fp16_full
QUANT=nf4 python cosafe_incontext.py --n 100 --tag int4_full   # CUDA only
python analyze_results.py              # significance tests
```

## References

- Arditi et al. Refusal in LLMs is mediated by a single direction. 2024.
- Kadadekar. Quality Is Not a Safety Proxy Under Quantization. arXiv:2606.10154, 2026.
- STAR. State-Dependent Safety Failures in Multi-Turn LM Interaction. arXiv:2603.15684, 2026.
- Chhabra & Khalili. Refusal in Compressed Models via Mechanistic Interpretability. arXiv:2504.04215, 2025.
- Yu et al. CoSafe: Evaluating Multi-Turn Dialogue Coreference Safety. EMNLP 2024.
