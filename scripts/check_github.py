"""Sanity check: verify GitHub API access works.

Fetches one well-known PR and prints basic info.
Run: uv run python scripts/check_github.py
"""
import os
import httpx
from dotenv import load_dotenv

load_dotenv()

token = os.getenv("GITHUB_TOKEN")
if not token:
    raise SystemExit("GITHUB_TOKEN not set in .env")

# Stable, known PR to test against
owner, repo, pr_number = "fastapi", "fastapi", 1

url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}"
headers = {
    "Authorization": f"Bearer {token}",
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
}

response = httpx.get(url, headers=headers, timeout=30.0)
response.raise_for_status()
pr = response.json()

print(f"OK. Fetched PR #{pr['number']} from {owner}/{repo}")
print(f"  Title:   {pr['title']}")
print(f"  Author:  {pr['user']['login']}")
print(f"  State:   {pr['state']}")
print(f"  Created: {pr['created_at']}")

remaining = response.headers.get("X-RateLimit-Remaining")
limit = response.headers.get("X-RateLimit-Limit")
print(f"\nRate limit: {remaining}/{limit} requests remaining this hour")