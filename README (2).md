
# Personalized LLM — MCTS-Guided Attribute Selection

This repo contains a **runnable** training scaffold for discovering optimal
attribute-question paths using **Monte Carlo Tree Search (MCTS)**. It is designed
for *personalized safety* in high-risk scenarios by learning **which background
attributes to ask first** under a limited interaction budget.

## Features
- Clean Python package with modular components.
- LLM-abstraction supports OpenAI **and** Azure OpenAI.
- **Dummy mode** for offline development without API keys.
- CSV logging of discovered paths for later policy-learning.

## Install
```bash
pip install -r requirements.txt
```

## Environment
Set these variables if you want real API calls:
```bash
# Choose: "openai" or "azure"
export OPENAI_API_TYPE=openai
export OPENAI_API_KEY=sk-...           # required for real calls
export OPENAI_API_BASE=                # optional for OpenAI; required for Azure
export OPENAI_API_VERSION=2024-05-01-preview  # Azure
export OPENAI_DEPLOYMENT=my-deploy     # Azure
export OPENAI_MODEL=gpt-4o-mini        # OpenAI
```
Or run **dummy mode** (no external calls, deterministic text):
```bash
export DUMMY_MODE=true
```

## Run
```bash
python train_mcts.py       --query "I feel overwhelmed and may hurt myself."       --attributes age mood therapy_history social_support substance_use       --episodes 3
```
Output is appended to `mcts_search_log.csv`.

## Next steps
- Train a policy network on `mcts_search_log.csv` to imitate MCTS decisions.
- Replace the judge prompt with your exact evaluation rubric.
- Plug in your real user-simulator or dataset instead of the LLM simulator.
