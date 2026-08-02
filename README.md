# Refusal Is Not Harm

**Multi-turn jailbreak benchmarks rank attacks backwards.**

Parv Bansal · AIMS-DTU · Research Internship 2026

> Research repository. Contains standard safety-benchmark prompts and model
> completions used to measure refusal and harm. Not for redistribution.

---

## What this is

The project set out to test whether 4-bit quantization weakens a model's
resistance to multi-turn jailbreaks. It does not. But the standard way the field
scores these attacks — counting a non-refusal as a success — turned out to be
measuring the wrong thing, and fixing it produced the actual result:

**The metric ranks two common attacks in the opposite order from the harm they
cause.** A coreference attack (CoSafe) reports 81.9% success while causing 1.1%
attributable harm; an escalation attack (Crescendo) reports 20.2% while causing
6.0%. Across fourteen matched model–precision configurations the reversal in
attributable harm is +4.9 percentage points, with a bootstrap 95% interval of
[+1.9, +8.2] that excludes zero under every one of four harm-labelling rules and
both harm judges.

Three findings, in order of importance:

1. **Reported success inverts the attack ordering.** 94% of coreference
   "successes" contain no harmful content; the metric credits attacks for answers
   the model would have given anyway.
2. **Multi-turn vulnerability is a model-family property, invisible single-turn.**
   Qwen2.5 refuses 99% of single-turn harmful prompts but 7.5% under attack;
   Llama-3.x hardens against escalation; Mistral-7B barely refuses at all.
3. **Quantization matters only for adaptive attacks** (mean 8.1pp vs 1.6pp for
   static), because an attack that quotes the model's own output is a different
   attack against a different model. Separated from a compute-backend confound by
   a matched FP16 replication (mean 0.5pp).

The full write-up is in [`paper/main.tex`](paper/main.tex).

## Method in one paragraph

Five instruction-tuned models (Llama-3.2-3B, Llama-3.1-8B, Qwen2.5-3B/7B,
Mistral-7B), four precisions (FP16, NF4, AWQ, GPTQ), two multi-turn attacks plus
a single-turn control. Every response is scored two ways: a substring refusal
classifier, and harm by **two independent judges** (Llama Guard 3 8B and Granite
Guardian 3.1 2B) over all 6,000 completions. The key addition is a **cold
control** — the same final request with no conversation — which lets reported
success decompose into *no harm + pre-existing harm + attack-attributable harm*.
All comparisons are paired within scenario; effect sizes carry bootstrap
intervals.

## Repository layout

```
data/          all completions and harm labels, grouped by kind
  completions/   raw model responses           (completions_*.csv)
  judged/        harm labels, both judges       (judged_*.csv, judged2_*.csv)
  incontext/     refusal + projection per arm   (incontext_*.csv)
  singleturn/    single-turn control            (singleturn_*.csv)
  validation/    stratified human-check sample  (validation_*.csv)
  advbench_harmful_behaviors.csv                (attack prompts)
src/           all Python; datadir.py resolves data paths for every script
  refusal_direction.py, ablation.py            direction + causal layer
  cosafe_incontext.py, crescendo_attack.py, pair_attack.py, singleturn_*.py
  judge_harm.py, judge_harm2.py                the two harm judges
  analyze_paper.py, analyze_agreement.py, analyze_validation.py
  make_paper_figures.py
runners/       shell + Kaggle notebooks for cloud sweeps
paper/         main.tex, refs.bib, figures
docs/          review brief, results notes, internship summary
archive/       superseded runs and earlier drafts
```

## Reproduce every number

The paper's tables and figures regenerate from the committed CSVs with no GPU:

```bash
python src/analyze_paper.py        # Tables 1–4, the decomposition, the inversion
python src/analyze_agreement.py granite   # two-judge agreement + robustness table
python src/analyze_validation.py   # third-judge check on the stratified sample
python src/make_paper_figures.py   # writes the three figures into paper/
```

Regenerating the data itself (needs a GPU; quantized arms need CUDA) is driven by
the scripts in `runners/`. Paths are resolved through `src/datadir.py`, so scripts
run from the repository root regardless of where the data lives.

## Status

Data collection is complete across both judges and two compute backends. Open
items: human validation on the 200-row stratified sample, and a clean re-run of
the fully-adaptive PAIR attack (a first attempt was dominated by the attacker
model's own refusals). Both are documented as limitations in the paper.
