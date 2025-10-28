
"""
LLM Client abstraction.
- Supports OpenAI (api_type='openai') and Azure OpenAI (api_type='azure').
- Has a 'dummy' mode that returns deterministic strings for offline dev/testing.
"""
from __future__ import annotations
import time
import random
from typing import Optional
from . import config

try:
    from openai import OpenAI, AzureOpenAI
except Exception:
    OpenAI = None
    AzureOpenAI = None

class LLMClient:
    def __init__(self):
        self.api_type = config.OPENAI_API_TYPE
        self.dummy = config.DUMMY_MODE or (OpenAI is None and AzureOpenAI is None)
        if not self.dummy:
            if self.api_type == "azure":
                self.client = AzureOpenAI(
                    api_key=config.OPENAI_API_KEY,
                    api_version=config.OPENAI_API_VERSION,
                    azure_endpoint=config.OPENAI_API_BASE,
                )
            else:
                self.client = OpenAI(api_key=config.OPENAI_API_KEY, base_url=config.OPENAI_API_BASE or None)

    def chat(self, prompt: str, system: Optional[str] = None, temperature: float = 0.3, max_tokens: int = 512) -> str:
        """Send a chat completion request with retries."""
        if self.dummy:
            # Deterministic canned response for testing
            random.seed(hash(prompt) % (2**32))
            return f"[DUMMY LLM OUTPUT] {prompt[:60]}..."

        for attempt in range(config.MAX_RETRIES):
            try:
                if self.api_type == "azure":
                    resp = self.client.chat.completions.create(
                        model=config.OPENAI_DEPLOYMENT,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        messages=[
                            {"role": "system", "content": system or "You are a helpful assistant."},
                            {"role": "user", "content": prompt},
                        ],
                    )
                    return resp.choices[0].message.content or ""
                else:
                    resp = self.client.chat.completions.create(
                        model=config.OPENAI_MODEL,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        messages=[
                            {"role": "system", "content": system or "You are a helpful assistant."},
                            {"role": "user", "content": prompt},
                        ],
                    )
                    return resp.choices[0].message.content or ""
            except Exception as e:
                if attempt == config.MAX_RETRIES - 1:
                    return f"LLM_CALL_FAILED: {e}"
                time.sleep(config.RETRY_BACKOFF_SEC * (attempt + 1))
