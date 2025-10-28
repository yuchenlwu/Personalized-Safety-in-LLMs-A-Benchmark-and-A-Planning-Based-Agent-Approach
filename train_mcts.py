
#!/usr/bin/env python3
"""
Train-time MCTS to discover good attribute-question paths.

Usage:
    python train_mcts.py --query "I feel overwhelmed and may hurt myself."             --attributes age mood therapy_history social_support substance_use

Notes:
    - Set OPENAI_* environment variables to use real models.
    - Set DUMMY_MODE=true to run without API calls (for debugging).
    - Outputs a CSV log (mcts_search_log.csv by default) with each discovered path.
"""
from __future__ import annotations
import argparse
from personalized_llm.config import INTERACTION_BUDGET, SEARCH_LOG_FILE, SEED
from personalized_llm.env import AttributeSpace, QueryState
from personalized_llm.llm_client import LLMClient
from personalized_llm.mcts import MCTS
from personalized_llm.utils import append_search_log

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--query", type=str, required=True, help="High-risk user query to condition on.")
    p.add_argument("--attributes", nargs="+", required=True, help="Candidate attribute names to ask.")
    p.add_argument("--episodes", type=int, default=5, help="How many MCTS episodes to run.")
    p.add_argument("--tag", type=str, default="", help="Optional tag to include in the log.")
    return p.parse_args()

def main():
    args = parse_args()
    space = AttributeSpace(attributes=args.attributes)
    llm = LLMClient()

    for ep in range(args.episodes):
        init_state = QueryState()
        mcts = MCTS(space, user_query=args.query, llm=llm)
        path, reward = mcts.search(init_state)
        append_search_log({
            "episode": ep,
            "query": args.query,
            "attributes": " | ".join(args.attributes),
            "budget": INTERACTION_BUDGET,
            "path": " -> ".join(path),
            "final_reward": f"{reward:.4f}",
            "tag": args.tag,
            "seed": SEED,
        }, log_file=SEARCH_LOG_FILE)
        print(f"[EP {ep}] Path: {path} | Reward: {reward:.3f}")

if __name__ == "__main__":
    main()
