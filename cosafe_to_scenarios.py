"""Convert CoSafe (multi-turn coreference) into scenarios.json for the harness.
Downloads only SAFE categories; severe ones (child abuse, self-harm, terrorism,
weapons, violence, sexual, animal abuse) are excluded by omission."""
import json, urllib.request, urllib.parse, argparse

BASE = "https://raw.githubusercontent.com/ErxinYu/CoSafe-Dataset/main/CoSafe%20datasets/"
SAFE_CATEGORIES = [
    "non_violent_unethical_behavior",
    "financial_crime,property_crime,theft",
    "privacy_violation",
    "misinformation_regarding_ethics,laws_and_safety",
]

def fetch_category(cat):
    url = BASE + urllib.parse.quote(cat + ".json")
    raw = urllib.request.urlopen(url).read().decode("utf-8")
    convos = []
    for line in raw.strip().split("\n"):
        line = line.strip()
        if line:
            convos.append(json.loads(line))
    return convos

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-category", type=int, default=25)
    ap.add_argument("--out", default="scenarios.json")
    args = ap.parse_args()

    scenarios = []
    for cat in SAFE_CATEGORIES:
        convos = fetch_category(cat)
        kept = 0
        for i, msgs in enumerate(convos):
            # keep well-formed conversations that end on a user (unsafe) turn
            if (isinstance(msgs, list) and len(msgs) >= 3
                    and msgs[-1].get("role") == "user"
                    and all("role" in m and "content" in m for m in msgs)):
                scenarios.append({
                    "id": f"cosafe_{cat.split(',')[0]}_{i}",
                    "category": cat,
                    "messages": msgs,
                })
                kept += 1
                if kept >= args.per_category:
                    break
        print(f"  {cat}: kept {kept}")
    json.dump(scenarios, open(args.out, "w"), indent=2)
    print(f"\nWrote {len(scenarios)} scenarios -> {args.out}")
    # quick stats
    ut = [sum(1 for m in s["messages"] if m["role"] == "user") for s in scenarios]
    print(f"user turns per scenario: min {min(ut)}, max {max(ut)}, avg {sum(ut)/len(ut):.1f}")

if __name__ == "__main__":
    main()
