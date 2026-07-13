"""
Per-turn projection logging (Thread 1: the convergence mechanism).

Runs each CoSafe conversation turn by turn and records the projection onto the
refusal direction at EVERY user turn (not just the final one). Comparing FP16 /
AWQ / GPTQ per-turn lets us test whether the precisions CONVERGE as the
conversation deepens -- the proposed mechanism for why the single-turn
quantization gap vanishes multi-turn.

Uses CoSafe's provided assistant turns as context (preserves the attack), same
hardened provenance/verify as the other scripts.

Usage:
    MODEL=8b            python perturn.py --n 100 --tag pt_8b_fp16
    MODEL=8b QUANT=awq  python perturn.py --n 100 --tag pt_8b_awq
    MODEL=8b QUANT=gptq python perturn.py --n 100 --tag pt_8b_gptq
"""
import argparse, csv, json, os, torch
from refusal_direction import load_model, get_device, MODEL_ID
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
def generate_reply(model, tok, messages, device, max_new_tokens=40):
    ids = build_ids(tok, messages, device)
    gen = model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    return tok.decode(gen[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)

@torch.no_grad()
def projection(model, tok, messages, direction, device):
    _captured.clear()
    model(**build_ids(tok, messages, device))
    act = _captured["act"]
    return (act @ direction.to(act.dtype).to(act.device)).item()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", default="scenarios.json")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--tag", default="pt_fp16")
    ap.add_argument("--layer", type=int, default=None)
    args = ap.parse_args()

    quant = os.environ.get("QUANT", "fp16").lower()
    bundle = torch.load(BUNDLE, map_location="cpu")
    if args.layer is not None:
        layer, src = args.layer, "--layer"
    elif "ablation_best_layer" in bundle:
        layer, src = bundle["ablation_best_layer"], "bundle.ablation_best_layer"
    else:
        raise SystemExit("REFUSING TO RUN: no ablation_best_layer and no --layer.")
    direction = bundle["directions"][layer]

    device = get_device()
    tok, model = load_model(device)
    nlayers = len(model.model.layers)
    print("=" * 56)
    print(f"PROVENANCE  MODEL={MODEL_ID}  QUANT={quant}")
    print(f"  layer={layer} (src {src}, {nlayers} layers)  dir_norm={direction.norm():.4f}")
    print(f"  tag={args.tag}  n={args.n}")
    print("=" * 56)
    model.model.layers[layer].register_forward_hook(_cap_hook)

    scenarios = json.load(open(args.scenarios))[:args.n]
    rows = []  # (scenario_id, turn_index, projection, refused)
    for si, scen in enumerate(scenarios):
        history, ut = [], 0
        for m in scen["messages"]:
            if m["role"] == "user":
                ut += 1
                probe = history + [{"role": "user", "content": m["content"]}]
                p = projection(model, tok, probe, direction, device)
                r = int(is_refusal(generate_reply(model, tok, probe, device)))
                rows.append((scen.get("id", si), ut, p, r))
                history.append({"role": "user", "content": m["content"]})
            else:
                history.append({"role": "assistant", "content": m["content"]})
        if (si + 1) % 25 == 0:
            print(f"  {si+1}/{len(scenarios)}")

    # per-turn aggregate
    turns = sorted(set(r[1] for r in rows))
    print("\n=== PER-TURN (this precision) ===")
    print("  turn : mean_proj  refusal")
    agg = {}
    for t in turns:
        ps = [r[2] for r in rows if r[1] == t]; rs = [r[3] for r in rows if r[1] == t]
        agg[t] = (sum(ps)/len(ps), sum(rs)/len(rs))
        print(f"    {t}  : {agg[t][0]:+.3f}    {agg[t][1]:.1%}")

    out = f"perturn_{args.tag}.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["scenario_id","turn","projection","refused"]); w.writerows(rows)
    # verify
    rr = list(csv.DictReader(open(out)))
    ok = len(rr) == len(rows)
    print(f"  VERIFY rows saved {len(rr)} == {len(rows)} -> {'MATCH' if ok else 'MISMATCH!!'}")
    if not ok: raise SystemExit("row count mismatch")
    json.dump({"tag":args.tag,"model":MODEL_ID,"quant":quant,"layer":layer,
               "per_turn":{str(t):agg[t] for t in turns}},
              open(f"perturn_{args.tag}_meta.json","w"), indent=2)
    print(f"  Saved {out} + meta.")

if __name__ == "__main__":
    main()
