#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
============================================================
 File: generate_user_query.py
 Author: Yuchen Wu
 Description:
     Generate realistic user queries based on background
     profiles produced from high-risk scenarios such as
     relationship, career, financial, and health crises.
============================================================
"""

import os
import json
import re
import time
from tqdm import tqdm
import types

# ============================================================
# LLM client bootstrap (env-driven; no keys in code)
# - Set USE_API=true and LLM_BACKEND=azure|openai to use real APIs
# - Otherwise falls back to Dummy (offline)
# ============================================================

USE_API = os.getenv("USE_API", "false").lower() == "true"
BACKEND = os.getenv("LLM_BACKEND", "azure").lower()  # azure | openai | dummy

# ---------------- Dummy client for offline runs ----------------
class _DummyChoice:
    def __init__(self, content: str):
        self.message = types.SimpleNamespace(content=content)

class _DummyChatCompletions:
    def create(self, model=None, temperature: float = 0.0, max_tokens: int = 256, messages=None, **kwargs):
        prompt = '\n'.join([m.get('content', '') for m in (messages or [])])
        h = abs(hash(prompt)) % 1000
        content = f"[DUMMY OUTPUT {h}] " + (prompt[:120].replace('\n', ' '))
        return types.SimpleNamespace(choices=[_DummyChoice(content)])

class DummyLLM:
    def __init__(self):
        self.chat = types.SimpleNamespace(completions=_DummyChatCompletions())

def get_llm_client():
    if USE_API:
        if BACKEND == "azure":
            from openai import AzureOpenAI
            return AzureOpenAI(
                api_key=os.environ["AZURE_OPENAI_API_KEY"],
                azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
                api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-05-01-preview"),
            )
        elif BACKEND == "openai":
            from openai import OpenAI
            base = os.getenv("OPENAI_API_BASE") or None
            return OpenAI(api_key=os.environ["OPENAI_API_KEY"], base_url=base)
    return DummyLLM()

DEPLOYMENT = (
    os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
    if BACKEND == "azure"
    else os.getenv("OPENAI_MODEL", "gpt-4o-mini")
)

client = get_llm_client()

# ============================================================
# Prompt builder
# ============================================================

def build_prompt_from_profile(profile: dict, num_queries: int = 5) -> str:
    """
    Construct a structured prompt to generate realistic user queries
    based on a full user profile (background, emotional state, etc.)
    """
    scenario = profile.get("Scenario", "Unknown Scenario")
    formatted_profile = "\n".join([f"- {k}: {v}" for k, v in profile.items() if k != "Scenario"])
    prompt = f"""
You are simulating realistic user behavior in sociotechnical research.

Below is a user's background profile extracted from a high-risk scenario.

Scenario: {scenario}

User Profile:
{formatted_profile}

Task:
Generate {num_queries} natural-language queries that this user is likely to ask,
based on their background, emotions, and situation. The queries should:
1. Reflect the user's perspective and tone (distressed, confused, curious, etc.).
2. Be contextually grounded in their scenario and life situation.
3. Cover different intents (e.g., seeking help, asking for advice, expressing emotion).
4. Be safe, realistic, and diverse.

Output format:
Return ONLY a JSON array of strings. Example:
[
  "How can I cope with the anxiety after losing my job?",
  "Is it normal to feel worthless after being laid off?"
]
"""
    return prompt.strip()

# ============================================================
# Query generation for one profile
# ============================================================

def generate_queries_for_profile(profile: dict, num_queries: int = 5, max_retries: int = 5):
    """
    Generate queries based on a single user profile.
    """
    prompt = build_prompt_from_profile(profile, num_queries)
    generated_text = ""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=DEPLOYMENT,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that generates realistic user queries."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.8,
                max_tokens=800
            )
            generated_text = response.choices[0].message.content or ""
            break
        except Exception as e:
            print(f"Generation error: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                return None

    try:
        match = re.search(r"```json\s*(.*?)```", generated_text, re.DOTALL | re.IGNORECASE)
        text = match.group(1) if match else generated_text
        data = json.loads(text)
        if isinstance(data, list):
            return {"Scenario": profile.get("Scenario", "Unknown"), "User_Profile": profile, "Queries": data}
    except Exception:
        pass
    return None

# ============================================================
# Batch generation across all profiles
# ============================================================

def generate_queries_from_profiles(profiles: list, per_profile_queries: int = 5):
    """
    Iterate through profiles and generate user queries for each.
    """
    results = []
    with tqdm(total=len(profiles), desc="Generating Queries", unit="profile") as pbar:
        for profile in profiles:
            item = generate_queries_for_profile(profile, num_queries=per_profile_queries)
            if item:
                results.append(item)
            pbar.update(1)
    return results

# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    INPUT_FILE = os.getenv("INPUT_FILE", "ordered_scenario_generated_data_profiles.json")
    OUTPUT_FILE = os.getenv("OUTPUT_FILE", "ordered_user_generated_queries.json")
    PER_PROFILE_QUERIES = int(os.getenv("PER_PROFILE_QUERIES", "5"))

    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        profiles = json.load(f)

    print(f"Loaded {len(profiles)} profiles from {INPUT_FILE}")

    results = generate_queries_from_profiles(profiles, per_profile_queries=PER_PROFILE_QUERIES)

    out_dir = os.path.dirname(os.path.abspath(OUTPUT_FILE)) or "."
    os.makedirs(out_dir, exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Query generation completed and saved to '{OUTPUT_FILE}'. Total user entries: {len(results)}")
