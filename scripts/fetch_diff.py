"""Fetch the diff for one known PR and inspect what it looks like.

Run: uv run python scripts/fetch_diff.py
"""
from distill.github_client import get_pr, get_pr_diff

OWNER = "fastapi"
REPO = "fastapi"
PR_NUMBER = 15487  # pick any recent-ish merged PR

pr = get_pr(OWNER, REPO, PR_NUMBER)
diff = get_pr_diff(OWNER, REPO, PR_NUMBER)

print(f"PR #{PR_NUMBER}: {pr['title']}")
print(f"  Files changed:  {pr['changed_files']}")
print(f"  Lines added:    +{pr['additions']}")
print(f"  Lines deleted:  -{pr['deletions']}")
print(f"  Diff size:      {len(diff):,} chars over {diff.count(chr(10)):,} lines")
print()
print("First 40 lines of the diff:")
print("-" * 70)
print("\n".join(diff.splitlines()[:40]))
print("-" * 70)