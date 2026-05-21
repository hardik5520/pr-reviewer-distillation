"""Prompt templates for code review generation and evaluation.

The single source of truth for what a good review looks like lives in
REVIEW_RUBRIC.md at the repo root. This module loads it and embeds it
into prompts sent to GPT 4o (or any model implementing the OpenAI chat
completions API).
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

# Resolve the repo root regardless of where the caller runs from.
# src/distill/prompts.py -> src/distill -> src -> repo_root
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_RUBRIC_PATH = _REPO_ROOT / "REVIEW_RUBRIC.md"


@lru_cache(maxsize=1)
def load_rubric() -> str:
    """Read REVIEW_RUBRIC.md from the repo root. Cached after first call."""
    if not _RUBRIC_PATH.exists():
        raise FileNotFoundError(
            f"REVIEW_RUBRIC.md not found at {_RUBRIC_PATH}. "
            "The rubric must exist before generating traces."
        )
    return _RUBRIC_PATH.read_text(encoding="utf-8")


SYSTEM_PROMPT_TEMPLATE = """You are an expert code reviewer. Your job is \
to review a pull request diff and write a concrete, useful review that \
follows the rubric below.

Read the rubric carefully. Your review must follow its format and tone \
exactly. Prioritize real defects over nits. Tie every comment to a \
specific file and line in the diff. Do not invent issues that are not in \
the code. If the diff has no real issues, say so directly.

# RUBRIC

{rubric}
"""


USER_PROMPT_TEMPLATE = """Review the following pull request.

# Title
{title}

# Description
{body}

# Diff
<diff>
{diff}
</diff>

Write your review now, following the rubric's required structure \
(Summary, Issues, Strengths) exactly.
"""


def build_review_prompt(
    diff: str,
    title: str = "",
    body: str = "",
) -> list[dict[str, str]]:
    """Build the messages list to send to a chat completion API.

    Args:
        diff: Unified diff text for the PR. The actual code change.
        title: PR title for context. Optional.
        body: PR description for context. Optional.

    Returns:
        A list of role/content dicts ready to pass to OpenAI's chat API.
    """
    rubric = load_rubric()
    system = SYSTEM_PROMPT_TEMPLATE.format(rubric=rubric)
    user = USER_PROMPT_TEMPLATE.format(
        title=title or "(no title)",
        body=body or "(no description)",
        diff=diff,
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]