"""
Sharpened + provenance-hardened measurement.

For each CoSafe scenario, measure the FINAL (unsafe) turn two ways:
  COLD       - final turn alone.
  IN-CONTEXT - full conversation (the attack).
Records projection onto the refusal direction and refusal at that turn.

HARDENING (prevents the layer/data mismatch that corrupted earlier runs):
  - Prints full provenance: MODEL, QUANT, bundle, layer, direction norm.
  - REFUSES to run unless the layer is explicit (ablation_best_layer in the
    bundle, or --layer). No silent fallback to the d'-argmax layer.
  - After saving, re-reads the CSV and asserts the on-disk means equal the
    in-memory means, and writes a <tag>_meta.json provenance sidecar.

Usage:
    MODEL=8b            python cosafe_incontext.py --n 100 --tag 8b_fp16
    MODEL=8b QUANT=awq  python cosafe_incontext.py --n 100 --tag 8b_awq
"""
import argparse, csv, json, os, torch
from refusal_direction import load_model, get_device, MODEL_ID

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

# import the refusal detector after defining nothing else that clashes
from ablation import is_refusal

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", default="scenarios.json")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--tag", default="fp16")
    ap.add_argument("--layer", type=int, default=None,
                    help="force the measurement layer (overrides bundle)")
    args = ap.parse_args()
    n = 20 if args.quick else args.n

    quant = os.environ.get("QUANT", "fp16").lower()
    bundle = torch.load(BUNDLE, map_location="cpu")

    # --- Explicit layer, NO silent fallback ---
    if args.layer is not None:
        layer = args.layer
        src = "--layer flag"
    elif "ablation_best_layer" in bundle:
        layer = bundle["ablation_best_layer"]
        src = "bundle.ablation_best_layer"
    else:
        raise SystemExit(
            "REFUSING TO RUN: no ablation_best_layer in the bundle and no --layer "
            "given. Run ablation and patch the layer first, or pass --layer N. "
            "(This is the guard that prevents the wrong-layer data mismatch.)")

    direction = bundle["directions"][layer]
    dnorm = direction.norm().item()

    device = get_device()
    tok, model = load_model(device)
    nlayers = len(model.model.layers)

    print("=" * 60)
    print("PROVENANCE")
    print(f"  MODEL_ID        : {MODEL_ID}")
    print(f"  QUANT           : {quant}")
    print(f"  bundle          : {BUNDLE}")
    print(f"  layer used       : {layer}  (source: {src}; model has {nlayers} layers)")
    print(f"  direction norm  : {dnorm:.4f}  (should be ~1.0)")
    print(f"  tag / n         : {args.tag} / {n}")
    print("=" * 60)
    if not (0 <= layer < nlayers):
        raise SystemExit(f"Layer {layer} out of range for this model.")

    model.model.layers[layer].register_forward_hook(_cap_hook)
    scenarios = json.load(open(args.scenarios))[:n]

    rows = []
    for si, scen in enumerate(scenarios):
        msgs = scen["messages"]
        final = {"role": "user", "content": msgs[-1]["content"]}
        pc = projection(model, tok, [final], direction, device)
        rc = int(is_refusal(generate_reply(model, tok, [final], device)))
        pi = projection(model, tok, msgs, direction, device)
        ri = int(is_refusal(generate_reply(model, tok, msgs, device)))
        rows.append((scen.get("id", si), scen.get("category", "?"), pc, rc, pi, ri))
        if (si + 1) % 20 == 0:
            print(f"  {si + 1}/{len(scenarios)} done")

    def mean(j): return sum(r[j] for r in rows) / len(rows)
    m_pc, m_rc, m_pi, m_ri = mean(2), mean(3), mean(4), mean(5)

    print("\n=== RESULT ===")
    print(f"  COLD       proj {m_pc:+.4f}   refusal {m_rc:.1%}")
    print(f"  IN-CONTEXT proj {m_pi:+.4f}   refusal {m_ri:.1%}")
    print(f"  jailbreak (ASR) {1-m_ri:.1%}   projection shift {m_pi-m_pc:+.4f}")

    out = f"incontext_{args.tag}.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id","category","proj_cold","refused_cold","proj_incontext","refused_incontext"])
        w.writerows(rows)

    # --- Self-verify: re-read the file and confirm it matches memory ---
    rr = list(csv.DictReader(open(out)))
    disk_pi = sum(float(r["proj_incontext"]) for r in rr) / len(rr)
    disk_ri = sum(int(r["refused_incontext"]) for r in rr) / len(rr)
    ok = abs(disk_pi - m_pi) < 1e-6 and abs(disk_ri - m_ri) < 1e-6
    print(f"\n  VERIFY (from saved {out}): in-ctx proj {disk_pi:+.4f}, refusal {disk_ri:.1%}"
          f"  -> {'MATCH' if ok else 'MISMATCH!!'}")
    if not ok:
        raise SystemExit("Saved file does not match computed values. Do NOT trust this run.")

    meta = {"tag": args.tag, "model_id": MODEL_ID, "quant": quant, "layer": layer,
            "layer_source": src, "direction_norm": dnorm, "n": len(rows),
            "in_context_proj": m_pi, "in_context_refusal": m_ri,
            "cold_proj": m_pc, "cold_refusal": m_rc}
    json.dump(meta, open(f"incontext_{args.tag}_meta.json", "w"), indent=2)
    print(f"  Saved {out} + incontext_{args.tag}_meta.json  (provenance recorded).")

if __name__ == "__main__":
    main()
