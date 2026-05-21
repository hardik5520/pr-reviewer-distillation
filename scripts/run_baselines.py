"""Generate baseline reviews on the golden set from GPT 4o and GPT 4o mini.

These are the reference points the fine tuned model will be compared against
later. Saved to eval/baselines/{model}.jsonl, one record per golden PR.

Idempotent: PRs already done are skipped.

Run: uv run python scripts/run_baselines.py
"""
import json
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm

from distill.openai_client import complete, estimate_cost
from distill.prompts import build_review_prompt

# Models to baseline. Add more later (claude, gemini, etc.) by extending this list.
MODELS = [
    "gpt-4o",
    "gpt-4o-mini",
]
TEMPERATURE = 0.3

golden_file = Path("eval/golden/golden.jsonl")
baselines_dir = Path("eval/baselines")
baselines_dir.mkdir(parents=True, exist_ok=True)

if not golden_file.exists():
    raise SystemExit("eval/golden/golden.jsonl not found. Label some PRs first.")

# Load all golden PRs
with open(golden_file) as f:
    golden_prs = [json.loads(line) for line in f]

print(f"Loaded {len(golden_prs)} golden PRs")
print(f"Running baselines for {len(MODELS)} models: {', '.join(MODELS)}\n")

total_cost = 0.0
total_generated = 0

for model in MODELS:
    out_file = baselines_dir / f"{model.replace('/', '_')}.jsonl"

    # idempotency: skip PRs already done for this model
    done_keys: set[tuple] = set()
    if out_file.exists():
        with open(out_file) as f:
            for line in f:
                rec = json.loads(line)
                done_keys.add((rec["owner"], rec["repo"], rec["number"]))

    pending = [
        p for p in golden_prs
        if (p["owner"], p["repo"], p["number"]) not in done_keys
    ]
    if not pending:
        print(f"[{model}] all {len(golden_prs)} PRs already done, skipping")
        continue

    print(f"[{model}] generating {len(pending)} baselines ({len(done_keys)} already done)")

    with open(out_file, "a") as f:
        for pr in tqdm(pending, desc=f"  {model}", leave=False):
            messages = build_review_prompt(
                diff=pr["diff"],
                title=pr.get("title", ""),
                body=pr.get("body", ""),
            )
            result = complete(messages, model=model, temperature=TEMPERATURE)
            cost = estimate_cost(result)
            total_cost += cost

            record = {
                "owner": pr["owner"],
                "repo": pr["repo"],
                "number": pr["number"],
                "title": pr["title"],
                "model": result.model,
                "review": result.content,
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
                "cost_usd": cost,
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }
            f.write(json.dumps(record) + "\n")
            f.flush()
            total_generated += 1

print(f"\n=== Done ===")
print(f"Generated: {total_generated} baseline reviews")
print(f"Estimated cost: ${total_cost:.3f}")