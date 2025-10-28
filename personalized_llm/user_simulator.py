
    """
    Simulate asking a single attribute and receiving a plausible value.
    In production, this would involve a real user; for training, we use the LLM to simulate.
    """
    from __future__ import annotations
    from .llm_client import LLMClient
    from .env import QueryState

    SIMULATOR_SYS = (
        "You are a user profile simulator. Given an attribute name and current known "
        "background info, return a short, realistic value for that attribute. "
        "Keep it concise and consistent with the known info."
    )

    def simulate_attribute_value(attr: str, state: QueryState, llm: LLMClient) -> str:
        user_prompt = f"""Known data:
{state.as_prompt_fragment()}

Attribute to provide: {attr}
Return only the value (a short phrase)."""
        out = llm.chat(user_prompt, system=SIMULATOR_SYS, temperature=0.4, max_tokens=64)
        return out.strip()
