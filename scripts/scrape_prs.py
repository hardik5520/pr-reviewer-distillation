"""Scrape merged PRs from a repo into data/raw/{owner}__{repo}.jsonl

Run: uv run python scripts/scrape_prs.py
"""
import json
from pathlib import Path

from tqdm import tqdm

from distill.github_client import get_pr, get_pr_diff, list_prs

OWNER = "fastapi"
REPO = "fastapi"
TARGET = 20       # how many merged PRs we want to keep
SCAN_LIMIT = 100  # how many closed PRs to scan to find them

out_dir = Path("data/raw")
out_dir.mkdir(parents=True, exist_ok=True)
out_file = out_dir / f"{OWNER}__{REPO}.jsonl"

print(f"Target: {TARGET} merged PRs from {OWNER}/{REPO}")
print(f"Output: {out_file}\n")

kept = 0
progress = tqdm(total=TARGET, desc="Saved")

with open(out_file, "w") as f:
    for pr_summary in list_prs(OWNER, REPO, state="closed", limit=SCAN_LIMIT):
        # filter 1: only merged PRs (closed != merged on GitHub)
        if not pr_summary.get("merged_at"):
            continue

        number = pr_summary["number"]

        # fetch full details and the raw diff
        pr = get_pr(OWNER, REPO, number)
        diff = get_pr_diff(OWNER, REPO, number)

        record = {
            "owner": OWNER,
            "repo": REPO,
            "number": number,
            "title": pr["title"],
            "author": pr["user"]["login"],
            "merged_at": pr["merged_at"],
            "additions": pr["additions"],
            "deletions": pr["deletions"],
            "changed_files": pr["changed_files"],
            "body": pr.get("body") or "",
            "diff": diff,
        }
        f.write(json.dumps(record) + "\n")
        kept += 1
        progress.update(1)

        if kept >= TARGET:
            break

progress.close()
print(f"\nDone. Saved {kept} merged PRs to {out_file}")