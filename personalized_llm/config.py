
"""
Configuration helpers for environment variables and defaults.
This file centralizes API config, retry settings, and search hyperparameters.
"""
from __future__ import annotations
import os

# === API & runtime settings ===
OPENAI_API_TYPE = os.getenv("OPENAI_API_TYPE", "openai")  # "openai" or "azure"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "")
OPENAI_API_VERSION = os.getenv("OPENAI_API_VERSION", "2024-05-01-preview")
OPENAI_DEPLOYMENT = os.getenv("OPENAI_DEPLOYMENT", "")   # Azure: model deployment name
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  # OpenAI: model name

# Retry policy
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "4"))
RETRY_BACKOFF_SEC = float(os.getenv("RETRY_BACKOFF_SEC", "2.0"))

# Search / training
MCTS_MAX_DEPTH = int(os.getenv("MCTS_MAX_DEPTH", "7"))
MCTS_NUM_ROLLOUTS = int(os.getenv("MCTS_NUM_ROLLOUTS", "200"))
MCTS_UCT_C = float(os.getenv("MCTS_UCT_C", "1.3"))
INTERACTION_BUDGET = int(os.getenv("INTERACTION_BUDGET", "4"))  # how many attributes to ask per episode

# Logging
SEARCH_LOG_FILE = os.getenv("SEARCH_LOG_FILE", "mcts_search_log.csv")

# Misc
SEED = int(os.getenv("SEED", "42"))
DUMMY_MODE = os.getenv("DUMMY_MODE", "false").lower() == "true"
