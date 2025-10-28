
"""Utilities for logging and CSV writing."""
from __future__ import annotations
import csv
from typing import Dict, Any
from . import config

def append_search_log(row: Dict[str, Any], log_file: str = None):
    path = log_file or config.SEARCH_LOG_FILE
    header = list(row.keys())
    try:
        exists = False
        try:
            with open(path, "r", newline="", encoding="utf-8") as f:
                exists = True
        except FileNotFoundError:
            exists = False
        with open(path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=header)
            if not exists:
                w.writeheader()
            w.writerow(row)
    except Exception as e:
        print(f"[WARN] Failed to write log: {e}")
