"""
Generate and save the actual model completions, so a judge model can score them.
The refusal scripts only stored a binary keyword flag; this stores the text.

single mode: 100 AdvBench prompts, one turn each.
multi mode:  CoSafe scenarios, full conversation (the attack), final response.

Usage (same MODEL/QUANT env as the rest of the repo):
    MODEL=8b QUANT=awq python dump_completions.py --mode single --tag st_8b_awq
    MODEL=8b QUANT=awq python dump_completions.py --mode multi  --tag mt_8b_awq
"""
import argparse, csv, json, os, torch
from refusal_direction import load_model, get_device


def build_ids(tok, messages, device):
    text = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    return tok(text, return_tensors="pt").to(device)


@torch.no_grad()
def generate(model, tok, messages, device, max_new=128):
    ids = build_ids(tok, messages, device)
    out = model.generate(**ids, max_new_tokens=max_new, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["single", "multi"], required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--advbench", default="advbench_harmful_behaviors.csv")
    ap.add_argument("--scenarios", default="scenarios.json")
    ap.add_argument("--max_new", type=int, default=128)
    args = ap.parse_args()

    out = f"completions_{args.tag}.csv"
    n_expected = args.n
    if args.mode == "multi":
        n_expected = min(args.n, len(json.load(open(args.scenarios))))
    else:
        n_expected = min(args.n, sum(1 for _ in csv.DictReader(open(args.advbench))))
    if os.path.exists(out):
        have = sum(1 for _ in csv.DictReader(open(out)))
        if have >= n_expected:
            print(f"skip {out} (already has {have} rows)")
            return

    device = get_device()
    tok, model = load_model(device)
    rows = []

    if args.mode == "single":
        goals = [r["goal"] for r in csv.DictReader(open(args.advbench))][:args.n]
        for i, g in enumerate(goals):
            resp = generate(model, tok, [{"role": "user", "content": g}], device, args.max_new)
            rows.append({"id": i, "prompt": g, "response": resp})
            if (i + 1) % 25 == 0:
                print(f"  {i+1}/{len(goals)}", flush=True)
        cols = ["id", "prompt", "response"]
    else:
        scen = json.load(open(args.scenarios))[:args.n]
        for si, s in enumerate(scen):
            msgs = s["messages"]
            resp = generate(model, tok, msgs, device, args.max_new)
            rows.append({"id": s.get("id", si), "category": s.get("category", "?"),
                         "final_user": msgs[-1]["content"],
                         "conversation": json.dumps(msgs, ensure_ascii=False),
                         "response": resp})
            if (si + 1) % 25 == 0:
                print(f"  {si+1}/{len(scen)}", flush=True)
        cols = ["id", "category", "final_user", "conversation", "response"]

    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"saved {out} ({len(rows)} rows, mode={args.mode})")


if __name__ == "__main__":
    main()
