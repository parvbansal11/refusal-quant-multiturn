"""
Sharpened measurement: does the model get jailbroken by each CoSafe multi-turn
attack, and what is the projection onto the refusal direction at the final turn?

For each scenario we measure the FINAL (unsafe) turn two ways:
  COLD        - the final turn alone, no history (reference; note that for
                coreference attacks this turn is ambiguous out of context).
  IN-CONTEXT  - the full CoSafe conversation (cooperative history + final turn),
                the actual attack.

The crux comparison is FP16 in-context vs INT4 in-context (same attack, only
precision differs): does INT4 get jailbroken more (lower in-context refusal
rate) and show a lower projection at the final turn? That extends Kadadekar's
single-turn quantization result to the multi-turn setting.

Completions are classified for refusal only, never printed or saved.

Usage:
    python cosafe_incontext.py --quick               # 20 scenarios
    python cosafe_incontext.py --scenarios scenarios.json --n 100
    python cosafe_incontext.py --tag int4            # label the INT4 run
"""
import argparse, csv, json, os, torch
from refusal_direction import load_model, get_device
from ablation import is_refusal

BUNDLE = "refusal_direction_fp16.pt"

_captured = {}
def _cap_hook(module, inp, out):
    h = out[0] if isinstance(out, tuple) else out
    if h.shape[1] > 1:
        _captured["act"] = h[0, -1, :].detach().float()

def build_ids(tok, messages, device):
    text = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    return tok(text, return_tensors="pt").to(device)

@torch.no_grad()
def generate_reply(model, tok, messages, device, max_new_tokens=48):
    ids = build_ids(tok, messages, device)
    gen = model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    return tok.decode(gen[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)

@torch.no_grad()
def projection(model, tok, messages, direction, device):
    _captured.clear()
    ids = build_ids(tok, messages, device)
    model(**ids)
    act = _captured["act"]
    return (act @ direction.to(act.dtype).to(act.device)).item()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", default="scenarios.json")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--tag", default="fp16")
    args = ap.parse_args()
    n = 20 if args.quick else args.n

    device = get_device()
    tok, model = load_model(device)
    bundle = torch.load(BUNDLE, map_location="cpu")
    layer = bundle.get("ablation_best_layer", bundle["best_layer"])
    direction = bundle["directions"][layer]
    print(f"Using layer {layer} refusal direction.")
    model.model.layers[layer].register_forward_hook(_cap_hook)

    scenarios = json.load(open(args.scenarios))[:n]
    print(f"{len(scenarios)} CoSafe scenarios.\n")

    rows = []
    for si, scen in enumerate(scenarios):
        msgs = scen["messages"]
        final = {"role": "user", "content": msgs[-1]["content"]}
        # COLD: final turn alone
        pc = projection(model, tok, [final], direction, device)
        rc = int(is_refusal(generate_reply(model, tok, [final], device)))
        # IN-CONTEXT: full conversation
        pi = projection(model, tok, msgs, direction, device)
        ri = int(is_refusal(generate_reply(model, tok, msgs, device)))
        rows.append((scen.get("id", si), scen.get("category", "?"), pc, rc, pi, ri))
        if (si + 1) % 10 == 0:
            print(f"  {si + 1}/{len(scenarios)} done")

    def mean(xs): return sum(xs) / len(xs)
    proj_cold = mean([r[2] for r in rows]); ref_cold = mean([r[3] for r in rows])
    proj_ic = mean([r[4] for r in rows]);   ref_ic = mean([r[5] for r in rows])

    print("\n=== Final-turn measurement ===")
    print(f"  COLD  (final turn alone):  proj {proj_cold:+.3f}   refusal {ref_cold:.1%}")
    print(f"  IN-CONTEXT (full attack):  proj {proj_ic:+.3f}   refusal {ref_ic:.1%}")
    print(f"\n  In-context refusal rate = {ref_ic:.1%}  ->  jailbreak rate (ASR) = {1-ref_ic:.1%}")
    print(f"  Projection shift (in-context - cold): {proj_ic - proj_cold:+.3f}")
    print("\n  For the paper, compare THIS run (fp16) against the INT4 run:")
    print("   - higher jailbreak rate under INT4 = quantization worsens multi-turn safety")
    print("   - lower in-context projection under INT4 = the refusal-direction signature")

    out = f"incontext_{args.tag}.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "category", "proj_cold", "refused_cold", "proj_incontext", "refused_incontext"])
        w.writerows(rows)
    print(f"\n  Saved -> {out}. Rerun with --tag int4 on the quantized model.")

if __name__ == "__main__":
    main()
