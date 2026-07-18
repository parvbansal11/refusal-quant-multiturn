# Abstract (AAAI-27, ~155 words)

Quantization lets large language models run on edge hardware, where real use is
multi-turn. Prior work measured quantization's effect on refusal at a single
turn only and reported degradation. We give the first measurement across a
multi-turn jailbreak. We extract the refusal direction by difference-in-means,
confirm its causal layer by ablation, then measure refusal and its projection
under CoSafe coreference attacks for Llama-3.2-3B (FP16, NF4) and Llama-3.1-8B
(FP16, AWQ, GPTQ). Multi-turn refusal does not vary with precision: 3B holds at
23% (McNemar p=0.62), 8B at 27 to 32% (Cochran Q p=0.12). Precision differences
that appear at a single turn, where 8B quantization raises refusal by 8 to 11
points (Cochran Q p<0.01), do not survive the attack. The refusal direction
stays behaviorally coupled at every precision (AUC 0.79 to 0.93), so the
convergence is behavioral saturation, not representational loss. Single-turn
safety evaluation does not predict how quantized models rank under conversational
attack.
