# Abstract (AAAI-27)

Quantization lets large language models run on edge hardware, where real use is
multi-turn. Prior work measured quantization's effect on refusal at a single turn
only and reported degradation. We give the first measurement across a multi-turn
jailbreak, testing the claim on Llama-3.2-3B (FP16, NF4) and Llama-3.1-8B (FP16,
AWQ, GPTQ) with both a refusal classifier and a Llama Guard 3 harm judge. We do
not find degradation. On standalone AdvBench prompts, refusal is flat or higher
under quantization for both models and all three methods, with no significant
comparison. Under CoSafe multi-turn attacks, precision changes neither refusal
(3B 23% vs 23%, McNemar p=0.62; 8B 27 to 32%, Cochran Q p=0.12) nor
harmful-response rate; the one significant single-turn effect, 8B refusing more
after quantization, vanishes under attack. A safety-prompt positive control
confirms the harm metric can detect a real intervention, so the null is a true
absence of effect. Along the way we find refusal-based attack success overstates
harm by an order of magnitude (73% non-refusal versus 6% harmful). Single-turn
safety evaluation does not predict how quantized models rank under conversational
attack.
