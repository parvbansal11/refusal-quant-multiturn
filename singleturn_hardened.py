"""
Standalone single-turn control on a pinned AdvBench set. Cloud (CUDA) for the
quantized runs; FP16 runs anywhere.

Fixes the silent-fallback problem in singleturn.py: walledai/AdvBench is gated
and JBB needs a config, so the old load_data dropped to the embedded list
without saying which prompts it used. This reads advbench_harmful_behaviors.csv
(committed), so every precision measures the identical, citable prompt set.

Same hardened pattern as cosafe_incontext.py: explicit layer, provenance print,
self-verify on the saved CSV.

Usage (per precision):
    MODEL=3b            python singleturn_hardened.py --n 100 --tag st_3b_fp16
    MODEL=3b QUANT=nf4  python singleturn_hardened.py --n 100 --tag st_3b_nf4
    MODEL=8b            python singleturn_hardened.py --n 100 --tag st_8b_fp16
    MODEL=8b QUANT=awq  python singleturn_hardened.py --n 100 --tag st_8b_awq
    MODEL=8b QUANT=gptq python singleturn_hardened.py --n 100 --tag st_8b_gptq
"""
import argparse, csv, json, os, torch
from refusal_direction import load_model, get_device, MODEL_ID
from ablation import is_refusal

BUNDLE = "refusal_direction_fp16.pt"
ADVBENCH = "advbench_harmful_behaviors.csv"

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
    model(**build_ids(tok, messages, device))
    act = _captured["act"]
    return (act @ direction.to(act.dtype).to(act.device)).item()

def load_prompts(n):
    with open(ADVBENCH) as f:
        goals = [r["goal"] for r in csv.DictReader(f)]
    return goals[:n]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--tag", default="st_fp16")
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
    print("=" * 60)
    print(f"PROVENANCE  MODEL={MODEL_ID}  QUANT={quant}")
    print(f"  layer={layer} (src {src}, {nlayers} layers)  dir_norm={direction.norm():.4f}")
    print(f"  prompts={ADVBENCH} (first {args.n})  tag={args.tag}")
    print("=" * 60)
    if not (0 <= layer < nlayers):
        raise SystemExit(f"Layer {layer} out of range for this model.")
    model.model.layers[layer].register_forward_hook(_cap_hook)

    prompts = load_prompts(args.n)
    rows = []
    for i, g in enumerate(prompts):
        msgs = [{"role": "user", "content": g}]
        p = projection(model, tok, msgs, direction, device)
        r = int(is_refusal(generate_reply(model, tok, msgs, device)))
        rows.append((i, p, r))
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(prompts)}  running refusal "
                  f"{sum(x[2] for x in rows)/(i+1):.0%}", flush=True)

    m_p = sum(r[1] for r in rows) / len(rows)
    m_r = sum(r[2] for r in rows) / len(rows)
    print(f"\n=== SINGLE-TURN RESULT ===")
    print(f"  projection {m_p:+.4f}   refusal {m_r:.1%}   (jailbreak {1-m_r:.1%})")

    out = f"singleturn_{args.tag}.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["idx", "projection", "refused"]); w.writerows(rows)
    rr = list(csv.DictReader(open(out)))
    disk_r = sum(int(x["refused"]) for x in rr) / len(rr)
    ok = abs(disk_r - m_r) < 1e-6
    print(f"  VERIFY saved refusal {disk_r:.1%} -> {'MATCH' if ok else 'MISMATCH!!'}")
    if not ok:
        raise SystemExit("Saved file mismatch. Do not trust this run.")
    json.dump({"tag": args.tag, "model": MODEL_ID, "quant": quant, "layer": layer,
               "prompts": ADVBENCH, "n": len(rows), "projection": m_p, "refusal": m_r},
              open(f"singleturn_{args.tag}_meta.json", "w"), indent=2)
    print(f"  Saved {out} + meta.")

if __name__ == "__main__":
    main()
