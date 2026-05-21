"""Generate (diff, review) training pairs by running GPT 4o on each scraped PR.

Reads data/raw/*.jsonl, calls GPT 4o for each PR using the prompt template,
saves the result to data/traces/{owner}__{repo}.jsonl.

Idempotent. Re-running skips PRs that already have traces, so a crashed
or interrupted run picks up cleanly.

Run: uv run python scripts/generate_traces.py
"""
import json
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm

from distill.openai_client import complete, estimate_cost
from distill.prompts import build_review_prompt

MODEL = "gpt-4o"
TEMPERATURE = 0.3

raw_dir = Path("data/raw")
traces_dir = Path("data/traces")
traces_dir.mkdir(parents=True, exist_ok=True)

raw_files = sorted(raw_dir.glob("*.jsonl"))
if not raw_files:
    raise SystemExit("No files in data/raw/. Run scrape_prs.py first.")

total_cost = 0.0
total_generated = 0

for raw_file in raw_files:
    out_file = traces_dir / raw_file.name

    # Find already-done PR numbers for idempotency
    done_numbers: set[int] = set()
    if out_file.exists():
        with open(out_file) as f:
            for line in f:
                done_numbers.add(json.loads(line)["number"])

    # Load PRs and skip ones already done
    with open(raw_file) as f:
        prs = [json.loads(line) for line in f]

    pending = [p for p in prs if p["number"] not in done_numbers]
    if not pending:
        print(f"[{raw_file.name}] all {len(prs)} PRs already have traces, skipping")
        continue

    print(f"[{raw_file.name}] generating {len(pending)} traces ({len(done_numbers)} already done)")

    # Append mode + flush every line so a crash never loses progress
    with open(out_file, "a") as f:
        for pr in tqdm(pending, desc=f"  {raw_file.stem}", leave=False):
            messages = build_review_prompt(
                diff=pr["diff"],
                title=pr.get("title", ""),
                body=pr.get("body", ""),
            )
            result = complete(messages, model=MODEL, temperature=TEMPERATURE)
            cost = estimate_cost(result)
            total_cost += cost

            trace = {
                **pr,  # carry over all original PR fields
                "review": result.content,
                "review_model": result.model,
                "review_prompt_tokens": result.prompt_tokens,
                "review_completion_tokens": result.completion_tokens,
                "review_cost_usd": cost,
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }
            f.write(json.dumps(trace) + "\n")
            f.flush()
            total_generated += 1

print(f"\n=== Done ===")
print(f"Generated: {total_generated}")
print(f"Estimated total cost: ${total_cost:.3f}")