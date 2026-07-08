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

---

## Day 4: Full run (100 scenarios) + significance tests

Same measurement, all 100 CoSafe scenarios, both precisions. Then paired
significance tests (McNemar for behaviour, paired t-test for projection).

| Metric (final turn)        | FP16   | INT4 (nf4) |
|----------------------------|--------|------------|
| Cold projection            | +1.460 | +1.331     |
| In-context projection      | -0.863 | -0.837     |
| Projection shift (ic-cold) | -2.322 | -2.168     |
| In-context refusal         | 23.0%  | 23.0%      |
| Jailbreak rate (ASR)       | 77.0%  | 77.0%      |

### Significance (n = 100, paired)
- Behaviour (in-context refusal): 23% vs 23%. McNemar chi2 = 0.250, p = 0.617.
  Discordant pairs 2 vs 2 (symmetric noise). NULL.
- Projection (in-context): mean paired diff (INT4 - FP16) = +0.026,
  paired t = 0.567, p = 0.571, effect size dz = 0.057. NOT significant.

### Conclusion
**Clean, well-powered null.** nf4 4-bit quantization does not measurably degrade
Llama-3.2-3B multi-turn jailbreak robustness on CoSafe, at either the
behavioural or the representational level.

Note: the Day-3 quick-run gap (85% vs 90% ASR) was small-sample noise; it
vanished at n = 100. The small projection gap that looked "consistent" across
runs also failed its paired test. Both were noise. This is why the full run +
significance test mattered.

### Where this leaves the project
- The specific hypothesis ("nf4 worsens multi-turn safety") did not hold for
  this model. A null is a legitimate result but a modest paper on its own.
- Strongest next test: **AWQ** (Kadadekar's degradation was AWQ/GPTQ-specific,
  not nf4). If AWQ also nulls -> robust cross-method null. If AWQ shows an
  effect -> the finding is quantization-method-dependent.
- Alternative framings: (a) document the preserved-multi-turn-robustness null
  as a tension with Kadadekar's single-turn degradation; (b) revisit model
  size / attack strength (scope risk for timeline).
