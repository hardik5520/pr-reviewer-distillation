"""GitHub API client for fetching pull request data."""
from __future__ import annotations

import os
from typing import Any, Iterator

import httpx
from dotenv import load_dotenv

load_dotenv()

GITHUB_API = "https://api.github.com"


def _headers() -> dict[str, str]:
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        raise RuntimeError("GITHUB_TOKEN not set in environment")
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def get_pr(owner: str, repo: str, number: int) -> dict[str, Any]:
    """Fetch a single PR by number. Returns the full PR JSON."""
    url = f"{GITHUB_API}/repos/{owner}/{repo}/pulls/{number}"
    response = httpx.get(url, headers=_headers(), timeout=30.0)
    response.raise_for_status()
    return response.json()


def list_prs(
    owner: str,
    repo: str,
    state: str = "closed",
    limit: int = 100,
) -> Iterator[dict[str, Any]]:
    """Yield PRs from a repo, newest first.

    Stops after `limit` PRs total. Handles pagination automatically.
    Note: the list endpoint returns less detail than `get_pr`. To get
    full info (diff stats, etc) you still need a follow-up `get_pr` call.
    """
    per_page = min(100, limit)
    fetched = 0
    page = 1

    while fetched < limit:
        url = f"{GITHUB_API}/repos/{owner}/{repo}/pulls"
        params = {
            "state": state,
            "per_page": per_page,
            "page": page,
            "sort": "created",
            "direction": "desc",
        }
        response = httpx.get(
            url, headers=_headers(), params=params, timeout=30.0
        )
        response.raise_for_status()
        batch = response.json()

        if not batch:
            return  # no more PRs in this repo

        for pr in batch:
            if fetched >= limit:
                return
            yield pr
            fetched += 1

        page += 1

def get_pr_diff(owner: str, repo: str, number: int) -> str:
    """Fetch the raw unified diff for a PR as a string.

    Uses the same URL as get_pr but a different Accept header so GitHub
    returns the diff text instead of JSON. Can be very large for big PRs.
    """
    url = f"{GITHUB_API}/repos/{owner}/{repo}/pulls/{number}"
    headers = _headers()
    headers["Accept"] = "application/vnd.github.v3.diff"
    response = httpx.get(url, headers=headers, timeout=30.0)
    response.raise_for_status()
    return response.text