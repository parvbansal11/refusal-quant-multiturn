# Abstract (AAAI-27, ~165 words)

Quantization lets large language models run on edge hardware, where real use is
multi-turn. Prior work measured quantization's effect on refusal at a single turn
only and reported degradation. We test that claim across single and multi-turn
settings for Llama-3.2-3B (FP16, NF4) and Llama-3.1-8B (FP16, AWQ, GPTQ), reading
both refusal behavior and the projection onto the refusal direction whose causal
layer we confirm by ablation. We do not find degradation. On standalone AdvBench
prompts, refusal is flat or higher under quantization for both models and all
three methods, with no significant comparison. Under CoSafe multi-turn attacks,
precision does not change refusal (3B 23% vs 23%, McNemar p=0.62; 8B 27 to 32%,
Cochran Q p=0.12); the one significant single-turn effect, 8B refusing more after
quantization, vanishes under attack. Quantization does shift the refusal-direction
projection significantly (p<0.001) without changing behavior, so its effect is
representational, not behavioral. Single-turn safety evaluation does not predict
how quantized models rank under conversational attack.
