"""OpenAI API client for generating reviews and (later) judging them.

Centralized so the same call path is used for trace generation, eval
baselines, and LLM-as-judge scoring.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

load_dotenv()

_client: OpenAI | None = None


def get_client() -> OpenAI:
    """Lazily construct the OpenAI client. One per process."""
    global _client
    if _client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set in environment")
        _client = OpenAI(api_key=api_key)
    return _client


@dataclass
class CompletionResult:
    """One completion call's result + metadata."""
    content: str
    prompt_tokens: int
    completion_tokens: int
    model: str


# Rough USD costs per 1M tokens (approximate, check OpenAI pricing for exact)
COSTS_PER_M = {
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
}


def estimate_cost(result: CompletionResult) -> float:
    """Rough USD cost estimate from token usage. Approximate, not billing."""
    for known_model in COSTS_PER_M:
        if result.model.startswith(known_model):
            rates = COSTS_PER_M[known_model]
            return (
                result.prompt_tokens * rates["input"] / 1_000_000
                + result.completion_tokens * rates["output"] / 1_000_000
            )
    return 0.0


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=30),
    reraise=True,
)
def complete(
    messages: list[dict[str, str]],
    model: str = "gpt-4o",
    temperature: float = 0.3,
    max_tokens: int = 2000,
) -> CompletionResult:
    """Call the OpenAI chat completions API with retries on transient errors."""
    client = get_client()
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return CompletionResult(
        content=response.choices[0].message.content or "",
        prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
        completion_tokens=response.usage.completion_tokens if response.usage else 0,
        model=response.model,
    )