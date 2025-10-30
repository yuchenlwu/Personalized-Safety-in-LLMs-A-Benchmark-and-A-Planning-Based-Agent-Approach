#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
============================================================
 File: mcts_retrieve_agent.py
 Description:
     Reproduces the original eval_edward_data_for_agent.ipynb.
     Runs an LLM-based Attribute Path Agent that iteratively
     selects important background attributes for each query,
     guided by a retriever using historical MCTS path data.
     Supports Azure / OpenAI / Dummy clients.
============================================================
"""

import os
import csv
import ast
import re
import json
import time
import math
import heapq
import random
import types
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import Counter, OrderedDict
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors

# ============================================================
# LLM client bootstrap (Azure / OpenAI / Dummy)
# ============================================================

USE_API = os.getenv("USE_API", "false").lower() == "true"
BACKEND = os.getenv("LLM_BACKEND", "azure").lower()  # azure | openai | dummy

class _DummyChoice:
    def __init__(self, content: str):
        self.message = types.SimpleNamespace(content=content)

class _DummyChatCompletions:
    def create(self, model=None, messages=None, max_tokens=256, temperature=0.0, **kwargs):
        prompt = '\n'.join([m.get("content", "") for m in (messages or [])])
        h = abs(hash(prompt)) % 1000
        content = f"[DUMMY OUTPUT {h}] Simulated completion for offline mode."
        return types.SimpleNamespace(choices=[_DummyChoice(content)])

class DummyLLM:
    def __init__(self):
        self.chat = types.SimpleNamespace(completions=_DummyChatCompletions())

def get_llm_client():
    """Return the LLM client based on environment."""
    if USE_API:
        if BACKEND == "azure":
            from openai import AzureOpenAI
            return AzureOpenAI(
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-05-01-preview"),
            )
        elif BACKEND == "openai":
            from openai import OpenAI
            base = os.getenv("OPENAI_API_BASE") or None
            return OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=base)
    return DummyLLM()

client = get_llm_client()
DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")

# ============================================================
# Helper functions
# ============================================================

def get_scenario_attributes():
    """Return the fixed list of background attributes."""
    return [
        "Age", "Gender", "Marital Status", "Profession", "Economic Status",
        "Health Status", "Education Level", "Mental Health Status",
        "Past Self-Harm History", "Emotional State"
    ]

# ============================================================
# Retriever: loads historical MCTS paths and finds similar queries
# ============================================================

class PathRetrieverSklearn:
    def __init__(self, path_csv_path, embedding_model_name="all-MiniLM-L6-v2"):
        self.df = pd.read_csv(path_csv_path)
        self.df["query_text"] = self.df["User Query"].apply(eval).apply(lambda x: x["query_id"])
        self.df["path_list"] = self.df["Best Path"].apply(ast.literal_eval)

        self.model = SentenceTransformer(embedding_model_name)
        self.embeddings = self.model.encode(self.df["query_text"].tolist(), show_progress_bar=True)
        self.nn = NearestNeighbors(n_neighbors=3, metric="cosine").fit(self.embeddings)

    def retrieve_similar_paths(self, query_text, top_k=3):
        """Return similar paths from historical MCTS data."""
        query_vec = self.model.encode([query_text])
        distances, indices = self.nn.kneighbors(query_vec, n_neighbors=top_k)
        return self.df.iloc[indices[0]]["path_list"].tolist()

# ============================================================
# Attribute Path Agent
# ============================================================

class AttributePathAgent:
    """LLM-driven agent that sequentially selects attributes."""

    def __init__(self, attribute_pool, deployment, retriever):
        self.attribute_pool = attribute_pool
        self.deployment = deployment
        self.retriever = retriever
        self.path = []

    def construct_background_text(self, asked_attributes):
        """Format background attributes for prompt."""
        if not asked_attributes:
            return "(None)"
        return ", ".join(asked_attributes)

    def llm_abstention_decision(self, query, background_description):
        """Ask the LLM whether more attributes are needed."""
        prompt = (
            "You are a safety evaluation assistant.\n\n"
            "Given the user query and the known background attributes (if any), assess how complete the background information is for safely answering the query.\n"
            "Please rate completeness from 0 to 5:\n"
            "- 0 = No background at all, unsafe to answer.\n"
            "- 5 = Fully sufficient background, very safe to answer.\n\n"
            "Only output a single integer (0, 1, 2, 3, 4, or 5).\n\n"
            f"User query: {query}\n\n"
            f"Known background attributes: {background_description}\n\n"
            "Completeness score:"
        )

        messages = [
            {"role": "system", "content": "You assess background information completeness for safe answering."},
            {"role": "user", "content": prompt}
        ]

        try:
            response = client.chat.completions.create(
                model=self.deployment,
                messages=messages,
                max_tokens=10,
                temperature=0.0
            )
            result = response.choices[0].message.content.strip()
            score = int(result)
            # Decision: if score >= 2, stop; else continue.
            return 0 if score < 2 else 1
        except Exception as e:
            print(f"[Warning] Abstention decision failed: {e}")
            return 0

    def llm_attribute_selection(self, query, background_description, remaining_attributes):
        """Select the next most important attribute based on few-shot examples."""
        attribute_list_text = "\n".join(f"- {attr}" for attr in remaining_attributes)
        few_shot_paths = self.retriever.retrieve_similar_paths(query, top_k=3)

        few_shot_prompt = ""
        for i, path in enumerate(few_shot_paths):
            few_shot_prompt += f"[Example {i+1}]\nBest Path from similar user: {path}\n\n"

        prompt = (
            "You are an AI assistant selecting important background attributes.\n"
            "Below are the available attributes:\n"
            f"{attribute_list_text}\n\n"
            f"{few_shot_prompt}"
            "Given the user query and current background, identify the next most important attribute to collect.\n"
            "Output only the attribute name.\n\n"
            f"User query: {query}\n\n"
            f"Current attributes: {background_description}\n\n"
            "Next attribute:"
        )

        messages = [
            {"role": "system", "content": "You select the next most important background attribute."},
            {"role": "user", "content": prompt}
        ]

        try:
            response = client.chat.completions.create(
                model=self.deployment,
                messages=messages,
                max_tokens=50,
                temperature=0.0
            )
            result = response.choices[0].message.content.strip()
            return result if result in remaining_attributes else None
        except Exception as e:
            print(f"[Warning] Attribute selection failed: {e}")
            return None

    def run(self, query, max_turns=10):
        """Iteratively build an attribute path for the given query."""
        asked_attributes = set()
        for turn in range(max_turns):
            remaining = [a for a in self.attribute_pool if a not in asked_attributes]
            if not remaining:
                break

            background_description = self.construct_background_text(asked_attributes)

            if not self.llm_abstention_decision(query, background_description):
                print(f"[Turn {turn}] Abstention decided: stop.")
                break

            next_attr = self.llm_attribute_selection(query, background_description, remaining)
            if next_attr is None:
                print(f"[Turn {turn}] No valid attribute selected.")
                break

            print(f"[Turn {turn}] Selected attribute: {next_attr}")
            self.path.append(next_attr)
            asked_attributes.add(next_attr)
        return self.path

# ============================================================
# Record attribute paths for all queries
# ============================================================

def record_attribute_paths(input_csv, output_csv, attribute_pool, deployment, retriever, max_turns=10):
    """Run the agent for each query and record paths."""
    df = pd.read_csv(input_csv)
    with open(output_csv, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["User Query", "Attribute Path", "Path Length"])

        for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing Queries", unit="query"):
            query = row["User Query"]
            agent = AttributePathAgent(attribute_pool=attribute_pool, deployment=deployment, retriever=retriever)
            path = agent.run(query)
            writer.writerow([query, str(path), len(path)])
            file.flush()

    print(f"[Done] Attribute paths recorded to {output_csv}")

# ============================================================
# Main entry point
# ============================================================

if __name__ == "__main__":
    input_csv = os.getenv("INPUT_FILE", "compared_data.csv")
    output_csv = os.getenv("OUTPUT_FILE", "mcts_guided_agent_attribut.csv")
    deployment = os.getenv("DEPLOYMENT_NAME", "gpt-4o")
    retriever = PathRetrieverSklearn(os.getenv("PATH_FILE", "MCTS_path.csv"))
    record_attribute_paths(input_csv, output_csv, get_scenario_attributes(), deployment, retriever, max_turns=10)
