"""Filters for deciding which PRs make good training data."""
from typing import Any

BOT_AUTHORS = {
    "dependabot[bot]",
    "renovate[bot]",
    "pre-commit-ci[bot]",
    "github-actions[bot]",
    "snyk-bot",
}

SKIP_TITLE_PREFIXES_LOWER = (
    "docs:", "doc:", "chore:", "style:", "ci:", "build:",
    "bump ", "release ", "version ",
)

# Gitmoji and emoji prefixes that signal non-substantive PRs
SKIP_TITLE_PREFIXES_RAW = (
    "⬆", "⬇",        # bumps
    "📝", "🔖",       # docs, release tags
    "👥",            # contributor / people list
    "🔧",            # tooling config
    "👷",            # CI / workflow
    "🔒",            # security / dependency pins
    "💄",            # style / cosmetic
    "🔥",            # remove / delete
    "➖",            # drop dependency
    "✅",            # tests only
)

# Substring patterns that almost always mean non-substantive
SKIP_TITLE_CONTAINS = (
    "update people",
    "update sponsors",
    "update contributors",
    "update github topic",
    "pin github actions",
    "pin dependencies",
    "remove deprecated",
    "remove previously deprecated",
    "drop support for",
)

# Files that are not code
DOCS_EXTENSIONS = (".md", ".rst", ".txt", ".adoc")
DATA_AND_CONFIG_EXTENSIONS = (
    ".yml", ".yaml", ".json", ".toml",
    ".cfg", ".ini", ".lock",
)
NON_CODE_EXTENSIONS = DOCS_EXTENSIONS + DATA_AND_CONFIG_EXTENSIONS


def is_bot(author: str) -> bool:
    return author in BOT_AUTHORS or author.endswith("[bot]")


def has_skip_prefix(title: str) -> bool:
    lower = title.lower()
    if any(lower.startswith(p) for p in SKIP_TITLE_PREFIXES_LOWER):
        return True
    if any(title.startswith(p) for p in SKIP_TITLE_PREFIXES_RAW):
        return True
    return False


def has_skip_keyword(title: str) -> bool:
    """Title contains a phrase that almost always means non-substantive."""
    lower = title.lower()
    return any(kw in lower for kw in SKIP_TITLE_CONTAINS)


def get_changed_files(diff: str) -> list[str]:
    files = []
    for line in diff.splitlines():
        if line.startswith("diff --git "):
            parts = line.split(" b/", 1)
            if len(parts) == 2:
                files.append(parts[1])
    return files


def is_non_code_only(diff: str) -> bool:
    """True if every changed file is docs, config, or data. No actual code."""
    files = get_changed_files(diff)
    if not files:
        return False
    return all(f.endswith(NON_CODE_EXTENSIONS) for f in files)


def should_keep_summary(pr_summary: dict[str, Any]) -> tuple[bool, str]:
    if not pr_summary.get("merged_at"):
        return False, "not merged"
    if is_bot(pr_summary["user"]["login"]):
        return False, "bot author"
    if has_skip_prefix(pr_summary["title"]):
        return False, "skip title prefix"
    if has_skip_keyword(pr_summary["title"]):
        return False, "skip title keyword"
    return True, "ok"


def should_keep_full(pr: dict[str, Any], diff: str) -> tuple[bool, str]:
    total_changes = pr["additions"] + pr["deletions"]
    if total_changes < 10:
        return False, "too small"
    if total_changes > 1000:
        return False, "too large"
    if len(diff) < 200:
        return False, "diff too short"
    if len(diff) > 50_000:
        return False, "diff too long"
    if pr["changed_files"] > 30:
        return False, "too many files"
    if is_non_code_only(diff):
        return False, "no code files"
    return True, "ok"