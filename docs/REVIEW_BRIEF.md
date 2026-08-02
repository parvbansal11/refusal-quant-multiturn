# Internship review: what to say and how to defend it

## The one-sentence version

I set out to test whether 4-bit quantization weakens multi-turn jailbreak
resistance, found it does not, and in chasing why the null was so clean
discovered that the metric the field uses to score these attacks ranks them in
the wrong order.

If you say nothing else, say that. It shows a null result was the beginning of
the work rather than the end of it.

## The three results, in the order to present them

**1. Reported attack success inverts the attack ranking.**
Two multi-turn attacks, same models, same prompts. CoSafe (coreference) reports
82.9% success; Crescendo (escalation) reports 23.8%. By actual harm the attack
caused, CoSafe is 1.0% and Crescendo is 7.0%. Both differences significant
(Wilcoxon p<0.001 and p=0.019), pointing opposite ways. If you rank attacks by
the standard metric you get the reverse of ranking them by harm.

**2. Multi-turn vulnerability is a model-family property that single-turn
benchmarks cannot see.**
Qwen2.5-7B refuses 99% of single-turn harmful prompts, which looks maximally
safe, and drops to 7.5% under multi-turn attack. Llama-3.x is unaffected by
coreference and actually *hardens* against escalation (3B: refusal rises 23
points, p<1e-4). Mistral-7B barely refuses anything (18.5%), so its 80-85%
reported success is real rather than an artifact.

**3. Quantization only matters for attacks that adapt.**
Static attacks: 7 comparisons, 0 significant, mean 1.4 points. Adaptive
attacks: mean 8.6 points, three configurations significant after Bonferroni.
Cleanest case is Qwen-3B: same model, same NF4 quantization, single-turn +0.5pp
(p=1.00), CoSafe +1.0pp (p=0.77), Crescendo +16.5pp (p<1e-6). The mechanism is
that Crescendo's third turn quotes the model's own prior output, so quantization
perturbs the attack itself and the divergence compounds.

## The technical spine (be ready to be quizzed here)

**Refusal direction.** Difference-in-means between activations on harmful and
harmless prompts (Arditi et al.), taken per layer. Validated on a held-out split
by d-prime and midpoint accuracy.

**Why d-prime alone is not enough, and what ablation adds.** The d-prime peak
lands on the final layer, which is a norm-growth artifact of logit preparation
rather than a refusal representation. So the causal layer is chosen by ablation:
project the direction out of the residual stream at every layer during
generation and see which candidate collapses refusal. For Llama-3.1-8B that is
layer 12 — refusal drops from 87.5% to 0.0%. The control is ablating a random
unit direction, which leaves refusal at 87.5%. That control is what makes it a
causal claim rather than a correlational one.

**Attribution decomposition — the actual novelty.**
reported success = no-harm + pre-existing harm + attack-attributable harm.
A scenario only counts as attack-attributable if the model refused the request
cold, did not refuse it in context, and the response was judged harmful. The
cold condition is the control the field omits, and adding it is one extra
generation per scenario.

**Harm judging.** Llama Guard 3 8B over all 6,000 completions in one pass on one
device, so no verdict is confounded by hardware.

**Statistics.** Everything paired within scenario id. McNemar exact for binary
refusal, Wilcoxon signed-rank for the attack comparison, Bonferroni across the
twelve precision comparisons. Where a test is underpowered I say so — the
per-configuration inversion is 8/11, sign test p=0.23, and I report it as
underpowered rather than as support.

## Two bugs I found and fixed — lead with these, do not hide them

These are the strongest things you can say about your own rigour.

**1. The held-out validation was not held out.**
`walledai/AdvBench` became gated on Hugging Face. The code fell back to an
embedded list of 34 prompts repeated four times. At n=128 that meant every one
of the 32 "held-out" prompts also appeared in training, so the reported
held-out d-prime and 100% accuracy were train-set numbers. I found it while
debugging something unrelated, switched to the 520-prompt AdvBench copy
committed in the repo, and re-ran with 256 unique prompts and a genuine quarter
held out. The numbers came out similar, but they now mean what they claim.

**2. Ablation crashed silently on multi-GPU.**
The direction was pinned to the device of the first parameter, so any model
sharded across two GPUs threw a cross-device error. The layer guard then
correctly refused to run downstream attacks — the right behaviour, but firing on
a fault that had nothing to do with layer choice. Fixed by making the hook
follow the hidden states.

**A limitation I am still carrying:** ablation layer selection at 16 prompts is
unstable. Llama-8B picked layer 10 in one run and 12 in another. Both collapse
refusal to near zero, so both are genuine bottlenecks, and with 16 prompts the
resolution is 6.25% per prompt. It affects projection values only — refusal and
harm are generation-based. The fix is running the full 32-prompt ablation, which
is cheap. Say this before someone asks.

## The last two days, factually

Five model families, four precisions, two attacks, n=200, across three
compute environments: Apple Silicon for FP16, a Kaggle T4 for part of the
quantized sweep, an A100 for the rest and for GPTQ.

**What went wrong and what it taught me:**

- **GPTQ's Marlin kernels require Ampere.** On a T4 the library retries the
  build instead of reporting the failure. That cost about four and a half hours.
  Now the runner checks compute capability once and skips GPTQ below 8.0.
- **Weight cache filled the output quota.** Moved it to scratch and purged
  between families.
- **Redirecting `HF_HOME` moved the login token with the cache**, so gated
  repos started 401-ing an hour into a run. Correct variable is `HF_HUB_CACHE`.
- **A transient I/O error on a network volume** killed one CoSafe arm.
- **The Mac run died to memory pressure** with two 7B models resident.

The generalisable lesson: I now judge before generating, so a session that dies
later still commits labels; and every step is resumable and idempotent, so a
lost session costs the step in flight rather than the run.

**Cost discipline:** the whole cloud sweep came in around $12, by putting FP16
on hardware I already had, using free Kaggle quota for the 6,000-judgment pass,
and buying an A100 only for the work that genuinely needed Ampere.

## Questions you will probably get, and answers

**"Isn't a null result a failure?"**
The null is solid and well-powered, and it is a non-replication of a published
claim, which has value on its own. But the useful part is what the null forced:
if quantization changes nothing, why is reported ASR 83%? That question produced
the inversion finding.

**"Why should I trust Llama Guard?"**
It is one judge, which is a real limitation. The evidence it is not simply
insensitive is Mistral: 73% harm on the same pipeline where Llama scores 3-7%.
A second judge, or human validation on a stratified sample, is the obvious next
step.

**"Is the inversion just because the two attacks use different prompts?"**
Partly, and that is exactly why the cold condition matters. CoSafe's final turns
are often innocuous in isolation, which is why its cold refusal is 14-40% while
Crescendo's is 71-99%. The decomposition conditions on that, so the comparison
is between what each attack adds over its own baseline rather than between raw
rates.

**"Why does quantization affect one attack and not the other?"**
Because Crescendo's later turns are a function of the model's own outputs.
Perturb the weights and you perturb the attack. CoSafe's turns are fixed scripts,
so quantization can only affect the final response. The Qwen-3B triple isolates
this: one model, one precision change, three attack types, and only the
self-referential one moves.

**"What would you do with three more months?"**
In order: a second harm judge plus human validation on a stratified sample;
extend the adaptive-attack finding to a genuinely agentic attacker such as PAIR
or TAP, where the divergence mechanism should be stronger; close the two data
gaps (Mistral quantized, 8B GPTQ); and test whether the cold-condition control
changes published rankings on existing multi-turn benchmarks.

## Framing your learning arc

Say it as three stages, because it is true and it reads as growth:

1. **Ran the experiment I was given.** Reproduced the multi-turn null.
2. **Stopped trusting the inputs.** Found the handoff's headline single-turn
   numbers had no backing data, found the gated-dataset fallback corrupting
   validation, found the multi-GPU crash. Learned that verifying provenance is
   part of the work, not overhead.
3. **Turned the null into a question.** Asked what the 83% was actually
   measuring, added the harm judge and the cold-condition control, and got a
   result that changes what other people should do.

## What not to do in the meeting

- Do not present the inversion as certain across every configuration. It is an
  aggregate result; the per-configuration test is underpowered and you should
  say so before you are asked.
- Do not overstate the quantization finding. Three of twelve comparisons survive
  correction. The contrast with 7/7 null on static attacks is what makes it
  interpretable, not the raw count.
- Do not hide the missing cells. Mistral quantized and 8B GPTQ are absent because
  compute ran out. Name them.

## Numbers worth memorising

| | value |
|---|---|
| Reported ASR, CoSafe vs Crescendo | 82.9% vs 23.8%, p<0.001 |
| Attributable harm, same configs | 1.0% vs 7.0%, p=0.019 |
| CoSafe successes containing no harm | 77% |
| Qwen2.5-7B single-turn vs multi-turn refusal | 99% to 7.5% |
| Llama-3.2-3B Crescendo refusal change | +23.0pp, p<1e-4 |
| Quantization, static vs adaptive | 1.4pp (0/7 sig) vs 8.6pp (3 sig) |
| Qwen-3B NF4 Crescendo | +16.5pp, p<1e-6, 37 flips vs 4 |
| Mistral-7B single-turn refusal / harm | 18.5% / 57% |
| Total judgments | 6,000 |
| Causal layer, Llama-3.1-8B | 12 (refusal 87.5% to 0.0%; random direction 87.5%) |
