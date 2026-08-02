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

---

## Day 5: 8B three-way + single-turn contrast (all from committed CSVs)

Recomputed every number from the five committed incontext_*.csv via
`analyze_multiturn.py`. Figures via `make_figures.py`.

### Multi-turn (in-context) refusal
| model | fp16 | nf4 | awq | gptq | test |
|---|---|---|---|---|---|
| 3B | 23% | 23% | - | - | McNemar chi2=0.25 p=0.617; proj paired-t p=0.572 |
| 8B | 27% | - | 32% | 31% | Cochran Q=4.2 p=0.122; pairwise p=0.13/0.22 |

Null across precision, both models. Confirms the Day-4 3B null and extends it to
the 8B three-way.

### Single-turn (cold) refusal: an 8B precision effect that multi-turn erases
| condition | fp16 | awq | gptq | test |
|---|---|---|---|---|
| 8B cold | 23% | 31% | 34% | McNemar p=0.043 / 0.010; Cochran Q p=0.008 |
| 8B in-context | 27% | 32% | 31% | Cochran Q p=0.122 (n.s.) |
| 3B cold | 25% | 23% (nf4) | - | McNemar p=0.688 (n.s.) |

On the cold final turn, 8B quantization RAISES refusal (concentrated in privacy
and misinformation). The gap is significant at a single turn and not significant
in context. Direction stays behaviourally coupled at every precision (AUC
0.79-0.93, `analyze_multiturn.py`), so the multi-turn convergence is behavioural
saturation, not representational loss.

### Data-integrity note
The single-turn degradation table from the earlier handoff (3B 62%->41%, 8B
62%->41%/48%) has no backing CSV anywhere in git history or on disk, and the 3B
fp16 anchor does not reproduce: standalone AdvBench single-turn gives 76%
(`singleturn_st_3b_fp16.csv`), not 62%. Treat 76% as the verified 3B fp16 number.
Quantized single-turn on AdvBench is queued for cloud (`CLOUD_RUN.md`,
`singleturn_hardened.py`, `analyze_singleturn.py`).

### Framing for the paper
Single-turn safety evaluation does not predict the multi-turn ordering of
quantized models: the 8B single-turn precision gap vanishes under CoSafe attack,
where all precisions converge near 30% refusal / 70% ASR. Draft in `paper/`.

---

## Day 6: single-turn AdvBench, all precisions (cloud, RunPod)

Ran `run_singleturn_cloud.sh` on a fresh pod (repo cloned into /workspace, run
under nohup so terminal disconnects stop killing it). Prompts: first 100
`advbench_harmful_behaviors.csv`. Ablation-confirmed layers: 3B layer 20, 8B
layer 12 (NOT 13; candidate set is [10,12,14,16,20,28], 12 gives the full refusal
collapse; the handoff's "13" was wrong and every 8B run used layer 12).

| Single-turn AdvBench | FP16 | quant | McNemar | proj paired-t |
|---|---|---|---|---|
| 3B fp16 vs nf4 | 76% | 82% | p=0.149 (n.s.) | t=-4.39 p<0.001 |
| 8B fp16 vs awq | 95% | 98% | p=0.248 (n.s.) | t=-24.4 p<0.001 |
| 8B fp16 vs gptq | 95% | 95% | p=0.617 (n.s.) | t=-7.84 p<0.001 |

### Read
- No behavioral degradation from quantization on standalone harmful prompts, for
  either model or any method. Point estimates flat or higher under quant. This is
  a well-powered non-replication of the single-turn INT4 degradation claim.
- Projection shifts ARE significant every time (representation moves, behavior
  does not). Mirror image of Kadadekar (intact geometry, degraded behavior).
- Caveat: 8B AdvBench refusal is near ceiling (95%) under the substring
  classifier, which limits power to detect a behavioral drop. Projection result
  is classifier-independent and does show a precision effect. A stricter judge is
  the obvious robustness follow-up.

### Complete behavioral picture (every condition tested)
- ST AdvBench: 3B 76->82 (n.s.), 8B 95->98/95 (n.s.). No degradation.
- ST CoSafe cold: 3B 25->23 (n.s.), 8B 23->31/34 (SIG, quant refuses MORE).
- MT CoSafe: 3B 23=23 (n.s.), 8B 27->32/31 (n.s.).
Only significant behavioral precision effect in the whole study: 8B CoSafe cold,
and it is an increase, erased by the attack. Paper reframed to non-replication +
representation/behavior dissociation (paper/ updated).

---

## Day 7: strict-judge rescore (Llama Guard 3 8B harm labels)

Regenerated real completions for all 10 conditions (dump_completions.py), scored
each response with Llama Guard 3 for actual harmful content (judge_harm.py). This
replaces the lenient substring refusal metric with a real harm classifier.
Metric: harmful-response rate (fraction judged unsafe), lower = safer.

| condition | fp16 | quant | McNemar |
|---|---|---|---|
| single 3B | 1% | nf4 1% | p=1.000 (flat) |
| single 8B | 4% | awq 2% / gptq 3% | p=0.480 / 1.000 (quant lower) |
| multi 3B | 4% | nf4 3% | p=1.000 (quant lower) |
| multi 8B | 6% | awq 4% / gptq 5% | p=0.480 / 1.000 (quant lower) |

### Read
- Quantization does NOT increase harmful output in any condition. Every quantized
  run is <= its fp16 baseline; all comparisons n.s. The null now holds under TWO
  independent metrics (substring refusal AND Llama Guard harm). Kills the
  ceiling-confound objection against the refusal-based null.
- BIG gap: substring metric said ~70% multi-turn "attack success" (non-refusal)
  for 8B, but Llama Guard says only 6% of those responses are actually harmful.
  The CoSafe attack mostly produces non-refusing but non-harmful output. Refusal-
  based ASR massively overstates real harm. This is a metric/methodology finding.
- Caveat (floor effect): harm is rare (1-6%), so low power to detect a small
  quant effect. Partly because CoSafe safe-categories are mild and Llama Guard is
  tuned for severe hazards (weapons, violent crime) so it may under-flag these.
  Not proof of safety on severe harms; it is "on mild prompts, actual harmful
  output is rare regardless of precision."
