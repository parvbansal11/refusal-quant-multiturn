# Redraft plan: from AAAI null result to ICML measurement-validity paper

The AAAI submission argued "quantization does not change multi-turn robustness,
and refusal overstates harm." Both claims survive, but neither is the paper any
more. Two findings that did not exist a week ago now carry it, and the
quantization axis becomes a supporting result rather than the subject.

## Proposed title

**Refusal Is Not Harm: Multi-Turn Jailbreak Benchmarks Rank Attacks Backwards**

Alternative if reviewers object to "backwards" as informal:
*Reported Attack Success Inverts the Ordering of Multi-Turn Jailbreaks*

## Abstract (draft)

Multi-turn jailbreak evaluations almost universally score an attack as
successful when the model fails to refuse. We label 6,000 model responses with
an independent harm judge across five instruction-tuned models, four numeric
precisions, and two multi-turn attacks, and show that this convention does not
merely overstate attack success: it reverses the ordering of attacks. A
coreference attack reports 82.9% success against 23.8% for an escalation attack
(Wilcoxon p<0.001), while causing 1.0% attributable harm against 7.0%
(p=0.019), measured on the same matched configurations. The gap arises because
refusal-based scoring credits an attack for every answer the model would have
given anyway; decomposing reported success against a matched no-attack
condition shows 77% of coreference "successes" contain no harmful content at
all. We further find that multi-turn vulnerability is a property of the model
family rather than of quantization: Qwen2.5 refuses 99% of single-turn harmful
prompts yet drops to 7.5% under multi-turn attack, Llama-3.x is unmoved by
coreference and actively hardens against escalation (+23.0pp refusal, p<1e-4),
and Mistral-7B refuses so rarely that its reported success is genuine. Finally,
quantization leaves static attacks unchanged (mean 1.4pp, 7/7 null) but
significantly alters adaptive ones (mean 8.6pp, three configurations surviving
correction), because an attack that conditions on model output is a different
attack against a different model. We release all completions, harm labels, and
a single script that regenerates every number.

## What changes, section by section

### Introduction
Lead with the inversion, not the null. The current intro builds to "quantization
doesn't matter," which is now Section 6. New arc: benchmarks score refusal,
refusal is not harm, and once you measure harm the attack ordering flips. State
the three contributions explicitly.

### Related Work
Mostly survives. Add framing against StrongREJECT (Souly et al.) — they showed
refusal-based scoring inflates single-turn success; we show it *inverts*
multi-turn *rankings*, which is a stronger and different failure. Keep the
corrected Kadadekar and Chhabra attributions from the AAAI version.

### Method
Add the attribution decomposition as a named contribution rather than an
analysis step:

> reported = no-harm + pre-existing harm + attack-attributable harm

where attack-attributable requires refused-cold AND not-refused-in-context AND
judged-harmful. The cold condition is the control that makes the decomposition
possible, and it is what the field omits.

Also document the two corrected defects, briefly and without drama, because
they affect reproducibility of earlier numbers:
- AdvBench went gated; the fallback list repeated 34 prompts four times, so
  held-out validation was not held out. Now 256 unique prompts, quarter held out.
- Ablation layer selection at 16 prompts is unstable (8B picked 10 and 12 on
  different runs). Affects projection values only; refusal and harm are
  generation-based and unaffected. State this rather than hide it.

### Results 1 — the inversion (new, leads)
Table 2 and 2b from `analyze_paper.py`. Headline: 82.9% vs 23.8% reported,
1.0% vs 7.0% attributable, both significant, opposite directions.
Be precise: this is an aggregate rank inversion. Per-configuration the flip
occurs 8/11 times, sign test p=0.23, underpowered. Say so.

### Results 2 — family regimes (new)
Table 1. Three regimes:
- **Llama-3.x**: CoSafe no net effect (3B +1.0pp p=0.87; 8B +0.5pp p=1.00) while
  reporting 75.5%/73.0% ASR. Crescendo *hardens* them (3B +23.0pp p<1e-4;
  8B +4.0pp p=0.039).
- **Qwen2.5**: genuinely vulnerable to both. 7B CoSafe -14.5pp, Crescendo
  -31.5pp with 63 flips against 0. Single-turn refusal 99% — invisible to
  single-turn benchmarks.
- **Mistral-7B-v0.3**: single-turn refusal 18.5%, harm 57%. Neither attack has
  a detectable effect (p=0.07, 0.20) yet both report 80-85%. Its Crescendo
  harm is 73%, of which 60.5pp is pre-existing.

Mistral is the control that proves the metric is not broken in general — it
works when the model does not refuse. It fails specifically on safety-tuned
models, which are the ones anyone benchmarks.

### Results 3 — quantization (demoted from spine to support)
Table 3. Static attacks 7/7 null, mean 1.4pp. Adaptive attacks mean 8.6pp with
Qwen-3B NF4 (+16.5pp, p<1e-6), Qwen-7B NF4 (+11.5pp, p=1e-4) and Qwen-7B GPTQ
(+16.5pp, p<1e-4) surviving Bonferroni across twelve comparisons.

The Qwen-3B triple is the cleanest evidence and deserves its own figure: one
model, one precision change, three attack types. Single-turn +0.5pp (p=1.00),
CoSafe +1.0pp (p=0.77), Crescendo +16.5pp (p<1e-6, 37 flips vs 4). The only
thing differing between rows two and three is whether later turns depend on the
model's own earlier outputs.

Mechanism: Crescendo's third turn quotes the model's own second-turn output, so
quantization perturbs the attack trajectory and compounds. Direction is not
monotone (3B gets safer, -4.0pp), which is consistent with trajectory divergence
rather than safety degradation. Frame as measurement validity: adaptive attacks
cannot compare model variants without controlling for trajectory divergence.

### Discussion
The recommendation is concrete: report the cold-condition baseline alongside
ASR, or report attributable harm. Either makes the inversion visible. This is
cheap — the cold condition is one extra generation per scenario.

### Limitations (expand, do not minimise)
- Llama Guard 3 is one judge; CoSafe categories are mild relative to its
  severe-hazard taxonomy and may be under-flagged. The Mistral row (73% harm)
  shows the judge is not simply insensitive.
- CoSafe multi-turn and single-turn completions are n=100 for FP16 arms,
  n=200 elsewhere; comparisons are paired on shared ids.
- Mistral quantized and 8B GPTQ were not collected (compute exhausted).
- `incontext_qwen7b_gptq.csv` lost to a storage fault, so Qwen-7B GPTQ has
  Crescendo and single-turn but no CoSafe arm.
- Ablation layer instability, as above.

## Figures to rebuild

1. **Hero**: the inversion. Two bars per attack, reported vs attributable.
2. **Qwen-3B triple**: three attack types, one precision change.
3. **Regime figure**: cold vs in-context refusal, five families, both attacks.
4. Retire `fig3_coupling.pdf` and `fig_twometric.pdf` — superseded.

## Venue

Mentor said ICML. The measurement-validity framing fits ICML or a
NeurIPS Datasets & Benchmarks track well, since the contribution is that a
widely used evaluation convention produces the wrong ordering. If reviewers
want a positive result rather than a critique, Results 2 (family regimes) can
lead instead, with the inversion as the mechanism explaining it.
