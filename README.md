# Does 4-Bit Quantization Change Multi-Turn Jailbreak Robustness?

A two-metric evaluation of NF4, AWQ, and GPTQ on Llama-3.2-3B and Llama-3.1-8B.

**Author:** Parv Bansal, AIMS-DTU Research Internship 2026

> Private research repository. Contains standard safety benchmark prompts used to
> measure refusal behaviour. Not for redistribution.

## Summary

Prior work measured quantization's effect on refusal at a single turn only and
reported degradation. This study gives the first measurement across a multi-turn
jailbreak, reading two signals: whether the model refused (substring classifier)
and whether the response is actually harmful (Llama Guard 3).

Quantization does not degrade safety.

- **Multi-turn (CoSafe).** Precision changes neither refusal (3B 23% vs 23%,
  McNemar p=0.62; 8B 27 to 32%, Cochran Q p=0.12) nor harmful-response rate.
- **Single-turn (AdvBench).** Refusal is flat or higher under quantization for
  both models and all three methods, no significant comparison.
- **Positive control.** A safety system prompt cuts 8B multi-turn harm from 6% to
  0% (p=0.041), so the harm metric can detect a real intervention. The
  quantization null is a measured absence of effect, not a lack of power.
- **Evaluation caution.** Refusal-based attack success overstates real harm by an
  order of magnitude (73% non-refusal versus 6% harmful).

Full results log: [RESULTS.md](RESULTS.md). Paper: [paper/](paper/).

## Repository layout

The repo is flat. Files group by role as follows.

### Paper (`paper/`)
| File | Purpose |
| :-- | :-- |
| `main.tex` | AAAI-27 submission (anonymous, `aaai2027` style) |
| `quant_multiturn_2col.tex` | Two-column named version |
| `quant_multiturn.tex` | Single-column named version |
| `refs.bib`, `abstract.md` | Bibliography and abstract text |
| `fig_hero.pdf`, `fig_twometric.pdf`, `fig3_coupling.pdf` | Figures used by `main.tex` |

### Analysis (no GPU, reproduces every number in the paper)
| Script | Output |
| :-- | :-- |
| `analyze_multiturn.py` | Multi-turn refusal and projection stats (McNemar, Cochran Q, paired t, AUC) |
| `analyze_singleturn.py` | Single-turn AdvBench refusal stats |
| `analyze_harm.py` | Llama Guard harmful-response rates and significance |
| `analyze_sysprompt.py` | Safety-prompt positive control |

### Figures
| Script | Purpose |
| :-- | :-- |
| `make_hero_figures.py` | Hero and two-metric figures |
| `make_figures.py` | Multi-turn, cold-vs-in-context, coupling figures |
| `outline_figures.sh` | Convert figure fonts to outlines (AAAI font compliance) |

### Experiment scripts (GPU / cloud)
| Script | Purpose |
| :-- | :-- |
| `refusal_direction.py` | Extract the refusal direction (Arditi difference-in-means) |
| `ablation.py` | Confirm the causal layer by ablation (3B layer 20, 8B layer 12) |
| `cosafe_to_scenarios.py` | Build CoSafe multi-turn scenarios (safe categories) |
| `cosafe_incontext.py` | Multi-turn measurement (cold vs in-context) |
| `singleturn_hardened.py` | Single-turn measurement on pinned AdvBench prompts |
| `dump_completions.py` | Generate model completions for judging |
| `judge_harm.py` | Score completions with Llama Guard 3 |
| `run_singleturn_cloud.sh`, `run_judge_cloud.sh`, `run_sysprompt_cloud.sh` | One-shot cloud runners |

### Data
| Pattern | Contents |
| :-- | :-- |
| `incontext_*.csv` | Multi-turn: projection and refusal, cold and in-context |
| `singleturn_st_*.csv` | Single-turn AdvBench: projection and refusal |
| `completions_*.csv` | Raw model completions (`st_` single-turn, `mt_` multi-turn) |
| `judged_*.csv` | Llama Guard harm labels per completion |
| `advbench_harmful_behaviors.csv` | AdvBench prompt set |
| `runpod-environment-lock.txt` | Exact package versions used for the cloud runs |

## Setup

```bash
python3 -m venv venv && source venv/bin/activate
pip install numpy scipy matplotlib          # analysis and figures only
```

The analysis scripts run on the committed CSVs with no GPU and no model access.

## Reproduce the results

```bash
python analyze_multiturn.py     # multi-turn null (3B and 8B)
python analyze_singleturn.py    # single-turn AdvBench non-replication
python analyze_harm.py          # Llama Guard harm rates
python analyze_sysprompt.py     # safety-prompt positive control
python make_figures.py && python make_hero_figures.py
```

## Regenerate the data (cloud)

The measurements need CUDA for the quantized models. See
[CLOUD_RUN.md](CLOUD_RUN.md) for the RunPod setup and the exact commands. Reproduce
the environment with `pip install -r runpod-environment-lock.txt`.

## Models, attacks, and metrics

- **Models.** Llama-3.2-3B-Instruct (FP16, NF4), Llama-3.1-8B-Instruct (FP16,
  AWQ-INT4, GPTQ-INT4, `hugging-quants` checkpoints).
- **Attacks.** CoSafe coreference (multi-turn, safe categories), AdvBench
  (single-turn).
- **Metrics.** Substring refusal classifier, Llama Guard 3 harmful-response rate,
  projection onto the refusal direction.

## References

- Arditi et al. Refusal in Language Models Is Mediated by a Single Direction. NeurIPS 2024.
- Kadadekar. Quality Is Not a Safety Proxy Under Quantization. arXiv:2606.10154, 2026.
- Yu et al. CoSafe: Evaluating LLM Safety in Multi-Turn Dialogue Coreference. EMNLP 2024.
- Inan et al. Llama Guard: LLM-based Input-Output Safeguard. arXiv:2312.06674, 2023.
- Zou et al. Universal and Transferable Adversarial Attacks on Aligned Language Models. arXiv:2307.15043, 2023.
