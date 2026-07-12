"""
Single-turn control measurement (the contrast for the multi-turn null).

Presents each harmful prompt as ONE turn (no conversation) and measures refusal
+ projection onto the refusal direction, FP16 vs AWQ vs GPTQ. If single-turn
refusal degrades under quantization (AWQ/GPTQ refuse less than FP16) while the
multi-turn result did not, that isolates the effect as single-turn-specific.

Uses the same hardened provenance + self-verify as cosafe_incontext.py.

Usage:
    MODEL=8b            python singleturn.py --n 100 --tag st_8b_fp16
    MODEL=8b QUANT=awq  python singleturn.py --n 100 --tag st_8b_awq
    MODEL=8b QUANT=gptq python singleturn.py --n 100 --tag st_8b_gptq
"""
import argparse, csv, json, os, torch
from refusal_direction import load_model, get_device, load_data, MODEL_ID
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
    build = build_ids(tok, messages, device)
    model(**build)
    act = _captured["act"]
    return (act @ direction.to(act.dtype).to(act.device)).item()

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
    print("=" * 56)
    print(f"PROVENANCE  MODEL={MODEL_ID}  QUANT={quant}")
    print(f"  layer={layer} (src {src}, {nlayers} layers)  dir_norm={direction.norm():.4f}")
    print(f"  tag={args.tag}  n={args.n}")
    print("=" * 56)
    model.model.layers[layer].register_forward_hook(_cap_hook)

    harmful, _ = load_data(args.n)
    harmful = harmful[:args.n]
    rows = []
    for i, prompt in enumerate(harmful):
        msgs = [{"role": "user", "content": prompt}]
        p = projection(model, tok, msgs, direction, device)
        r = int(is_refusal(generate_reply(model, tok, msgs, device)))
        rows.append((i, p, r))
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(harmful)}")

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
    if not ok: raise SystemExit("Saved file mismatch.")
    json.dump({"tag": args.tag, "model": MODEL_ID, "quant": quant, "layer": layer,
               "n": len(rows), "projection": m_p, "refusal": m_r},
              open(f"singleturn_{args.tag}_meta.json", "w"), indent=2)
    print(f"  Saved {out} + meta.")

if __name__ == "__main__":
    main()
