# Results Log

Model: Llama-3.2-3B-Instruct. Refusal direction: layer 20 (ablation-confirmed
causal bottleneck; ablating it drops refusal 81% -> 0%, random-direction control
stays at baseline). Held-out direction validation: 100% separation accuracy.

## Day 3: FP16 vs INT4 (nf4) on CoSafe multi-turn attacks

Measurement: CoSafe coreference multi-turn attacks (safe categories only:
non-violent unethical, financial crime, privacy, misinformation). For each
scenario, projection onto the layer-20 refusal direction and refusal at the
final (unsafe) turn, cold vs in-context. Greedy decoding. Layer-20 direction
(FP16) used as the common reference frame for both precisions.

Run: `--quick`, 20 scenarios. Preliminary / directional only.

| Metric (final turn)        | FP16   | INT4 (nf4) |
|----------------------------|--------|------------|
| Cold projection            | +0.464 | +0.332     |
| In-context projection      | -2.026 | -1.823     |
| Projection shift (ic-cold) | -2.489 | -2.155     |
| Cold refusal               | 20.0%  | 20.0%      |
| In-context refusal         | 15.0%  | 10.0%      |
| Jailbreak rate (ASR)       | 85.0%  | 90.0%      |

### Read
- Directional signal matches hypothesis: INT4 is jailbroken more (90% vs 85%)
  and refuses less (10% vs 15%) than FP16 on identical multi-turn attacks.
- NOT significant: 20 scenarios only, so a 5-point gap is ~one scenario flipping.
- Open tension: INT4 refuses LESS behaviorally but its in-context projection is
  LESS negative (-1.823 vs -2.026). Behavior and mechanism point slightly
  differently. Investigate whether this is real or a measurement artifact.

### Next
- Full run: all 100 scenarios, both precisions, for statistical weight.
- Add a significance test on the jailbreak-rate difference.
- Investigate the behavior-vs-projection tension.
- Later: AWQ (canonical, matches Kadadekar) as a second quantization method.

### Notes
- FP16 is robust to benign context and to CoSafe (in-context refusal stayed
  ~15%, no erosion) — the flat FP16 baseline is what makes any INT4 movement
  legible.
- Cloud (RunPod RTX 4090). AWQ/nf4 kernels do not run on Apple Metal, so INT4
  runs require CUDA; FP16 dev + extraction + ablation run locally on Mac.
