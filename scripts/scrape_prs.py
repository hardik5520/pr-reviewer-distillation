"""Scrape merged + filtered PRs from multiple repos into data/raw/.

Run: uv run python scripts/scrape_prs.py
"""
import json
from collections import Counter
from pathlib import Path

from tqdm import tqdm

from distill.filters import should_keep_full, should_keep_summary
from distill.github_client import get_pr, get_pr_diff, list_prs

# Repos to scrape. Edit this list freely.
REPOS = [
    ("fastapi", "fastapi"),
    ("pydantic", "pydantic"),
    ("pallets", "flask"),
    ("encode", "httpx"),
    ("fastapi", "sqlmodel"),
]

TARGET_PER_REPO = 20
SCAN_LIMIT = 500

out_dir = Path("data/raw")
out_dir.mkdir(parents=True, exist_ok=True)

overall_kept = 0
overall_scanned = 0
overall_reasons: Counter = Counter()

for owner, repo in REPOS:
    out_file = out_dir / f"{owner}__{repo}.jsonl"

    # skip if already scraped, makes the script idempotent
    if out_file.exists():
        existing = sum(1 for _ in open(out_file))
        print(f"\n[skip] {owner}/{repo} already scraped ({existing} records)")
        overall_kept += existing
        continue

    print(f"\n[{owner}/{repo}] target {TARGET_PER_REPO}, scanning up to {SCAN_LIMIT}")

    kept = 0
    scanned = 0
    reasons: Counter = Counter()
    progress = tqdm(total=TARGET_PER_REPO, desc=f"  {repo}", leave=False)

    with open(out_file, "w") as f:
        for pr_summary in list_prs(owner, repo, state="closed", limit=SCAN_LIMIT):
            scanned += 1

            ok, reason = should_keep_summary(pr_summary)
            if not ok:
                reasons[reason] += 1
                continue

            number = pr_summary["number"]
            pr = get_pr(owner, repo, number)
            diff = get_pr_diff(owner, repo, number)

            ok, reason = should_keep_full(pr, diff)
            if not ok:
                reasons[reason] += 1
                continue

            record = {
                "owner": owner,
                "repo": repo,
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
            reasons["ok"] += 1
            progress.update(1)

            if kept >= TARGET_PER_REPO:
                break

    progress.close()
    print(f"  Saved {kept}/{TARGET_PER_REPO}, scanned {scanned}")
    overall_kept += kept
    overall_scanned += scanned
    for reason, count in reasons.items():
        overall_reasons[reason] += count

print(f"\n=== Summary ===")
print(f"Repos:    {len(REPOS)}")
print(f"Kept:     {overall_kept}")
print(f"Scanned:  {overall_scanned}")
print(f"\nAggregate filter breakdown:")
for reason, count in overall_reasons.most_common():
    print(f"  {count:4d}  {reason}")