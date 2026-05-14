"""List recent closed PRs from a repo to verify the client works.

Run: uv run python scripts/list_prs.py
"""
from distill.github_client import list_prs

OWNER = "fastapi"
REPO = "fastapi"
LIMIT = 10

print(f"Fetching last {LIMIT} closed PRs from {OWNER}/{REPO}...\n")

for i, pr in enumerate(list_prs(OWNER, REPO, state="closed", limit=LIMIT), start=1):
    merged = "merged" if pr.get("merged_at") else "closed"
    print(f"{i:2}. #{pr['number']:<6} [{merged:6}] {pr['title']}")
    print(f"       by {pr['user']['login']} on {pr['created_at'][:10]}")