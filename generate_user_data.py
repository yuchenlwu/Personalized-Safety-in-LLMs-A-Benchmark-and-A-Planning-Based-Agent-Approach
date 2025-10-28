
#!/usr/bin/env python3
"""
Generate synthetic user profiles (attribute-value dictionaries) for personalized LLM experiments.

This script uses the same LLM abstraction as the MCTS code (OpenAI / Azure / Dummy).
It enforces concise, realistic, and logically consistent values for a given schema.

Output formats: CSV and JSONL.

Example:
    export DUMMY_MODE=true
    python generate_user_data.py             --attributes age gender marital_status profession mental_health_status             --n 50             --out_csv users.csv --out_jsonl users.jsonl
"""
from __future__ import annotations
import argparse, csv, json, random, re
from typing import List, Dict
from personalized_llm.llm_client import LLMClient

SYSTEM = (
    "You are a data generator for user profiles. "
    "Given a list of attribute names, produce a single JSON object whose keys are exactly the attributes. "
    "Values must be short, realistic, and mutually consistent (e.g., age vs education). "
    "Avoid contradictions, keep values concise (few words), and DO NOT include any extra keys or commentary."
)

def prompt_for_profile(attrs: List[str]) -> str:
    # instruction includes attribute names explicitly, requesting JSON-only output
    lines = "\n".join([f"- {a}" for a in attrs])
    return (
        "Generate ONE user profile with this exact attribute schema:\n"
        f"{lines}\n\n"
        "Return ONLY a valid, compact JSON object with these keys, no markdown. "
        "Keep values concise and realistic."
    )

def parse_json_obj(s: str, attrs: List[str]) -> Dict[str, str]:
    # Try to find a JSON object in the text and sanitize keys
    try:
        start = s.find("{")
        end = s.rfind("}")
        if start >= 0 and end > start:
            s = s[start:end+1]
        obj = json.loads(s)
        # Coerce to strings, keep only requested keys
        clean = {}
        for a in attrs:
            v = obj.get(a, "")
            if isinstance(v, (dict, list)):
                v = json.dumps(v, ensure_ascii=False)
            else:
                v = str(v)
            clean[a] = v.strip()
        return clean
    except Exception:
        # fallback: empty values
        return {a: "" for a in attrs}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attributes", nargs="+", required=True, help="Attribute names for the schema.")
    ap.add_argument("--n", type=int, default=20, help="Number of profiles to generate.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_csv", type=str, default="users.csv")
    ap.add_argument("--out_jsonl", type=str, default="users.jsonl")
    args = ap.parse_args()

    random.seed(args.seed)
    llm = LLMClient()

    rows = []
    for i in range(args.n):
        prompt = prompt_for_profile(args.attributes)
        out = llm.chat(prompt, system=SYSTEM, temperature=0.6, max_tokens=400)
        prof = parse_json_obj(out, args.attributes)
        prof["_id"] = str(i)
        rows.append(prof)
        print(f"[{i+1}/{args.n}] {prof}")

    # write CSV
    fieldnames = ["_id"] + args.attributes
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # write JSONL
    with open(args.out_jsonl, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Saved: {args.out_csv} and {args.out_jsonl}")

if __name__ == "__main__":
    main()
