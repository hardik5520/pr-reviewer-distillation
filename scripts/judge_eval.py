"""LLM-as-judge eval: score each model's reviews against the golden set.

For each of the 13 golden PRs, asks GPT-4o (acting as a neutral judge) to
score the 3 candidate reviews against the user's hand-labeled review.

Inputs:
  - eval/golden/golden.jsonl                  (hand-labeled "ideal" reviews)
  - eval/baselines/gpt-4o.jsonl               (teacher model)
  - eval/baselines/gpt-4o-mini.jsonl          (cheap baseline)
  - eval/baselines/pr-reviewer-7b.jsonl       (your trained model)

Output:
  - eval/results/judge_scores.jsonl           (per-PR per-model scores + reasoning)
  - eval/results/summary.json                 (aggregate scores)

Runs locally (calls OpenAI API). Cost: ~$0.50 for 13 PRs × 3 models.

Run: uv run python scripts/judge_eval.py
"""
import json
from pathlib import Path

from distill.openai_client import complete, estimate_cost

JUDGE_MODEL = "gpt-4o"
GOLDEN_PATH = Path("eval/golden/golden.jsonl")
BASELINES_DIR = Path("eval/baselines")
RESULTS_DIR = Path("eval/results")

MODELS_TO_JUDGE = [
    ("gpt-4o", BASELINES_DIR / "gpt-4o.jsonl"),
    ("gpt-4o-mini", BASELINES_DIR / "gpt-4o-mini.jsonl"),
    ("pr-reviewer-7b", BASELINES_DIR / "pr-reviewer-7b.jsonl"),                  # v1, base model
    ("pr-reviewer-7b-instruct", BASELINES_DIR / "pr-reviewer-7b-instruct.jsonl"),  # v2, instruct model
]


JUDGE_SYSTEM = """You are an expert code review evaluator. Your job is to \
compare a CANDIDATE code review against a REFERENCE code review written by \
a human expert, and score how well the candidate matches the reference's \
substance and judgment.

You will return a JSON object with the following structure:
{
  "issue_coverage": <int 1-5>,
  "issue_coverage_reasoning": "<one sentence>",
  "false_positives": <int 1-5>,
  "false_positives_reasoning": "<one sentence>",
  "severity_calibration": <int 1-5>,
  "severity_calibration_reasoning": "<one sentence>",
  "actionability": <int 1-5>,
  "actionability_reasoning": "<one sentence>",
  "overall": <int 1-5>,
  "overall_reasoning": "<two sentences>"
}

Scoring rubric (1-5 for each dimension):
  5 = matches or exceeds reference
  4 = slightly weaker but very close
  3 = partially matches; missing some important things
  2 = mostly misses the point
  1 = totally off

DIMENSIONS:

issue_coverage: Did the candidate identify the same real issues the reference \
identified? Missing critical issues = low score. Catching extras the reference \
missed = neutral to positive (only if they're real).

false_positives: Did the candidate invent issues that aren't real? Lots of \
made-up nitpicks, hallucinated bugs, or strawman concerns = low score. Clean \
focused review = high score.

severity_calibration: Does the candidate match the reference's sense of what's \
important? Treating a minor naming issue as critical = low. Burying a real bug \
in a long list of nits = low. Same priorities as reference = high.

actionability: Can a developer act on this review? Concrete file/line/fix \
suggestions = high. Vague handwaving like "consider improving error handling" \
without specifics = low.

overall: Holistic judgment. Would a senior engineer be roughly as happy \
receiving this review as the reference review?

Return ONLY the JSON object. No preamble, no markdown fences.
"""


def build_judge_prompt(pr: dict, reference_review: str, candidate_review: str) -> list[dict]:
    user_msg = (
        f"# Pull Request\n\n"
        f"Repo: {pr['owner']}/{pr['repo']}\n"
        f"Title: {pr['title']}\n\n"
        f"# REFERENCE REVIEW (human expert)\n\n"
        f"{reference_review}\n\n"
        f"---\n\n"
        f"# CANDIDATE REVIEW (to be scored)\n\n"
        f"{candidate_review}\n\n"
        f"---\n\n"
        f"Score the candidate review against the reference. Return JSON only."
    )
    return [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": user_msg},
    ]


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in open(path)]


def index_by_pr(records: list[dict]) -> dict[str, dict]:
    return {f"{r['owner']}/{r['repo']}#{r['number']}": r for r in records}


def main():
    print(f"Loading golden set from {GOLDEN_PATH}...")
    golden = load_jsonl(GOLDEN_PATH)
    golden_idx = index_by_pr(golden)
    print(f"  {len(golden)} golden PRs")

    candidates = {}
    for model_name, path in MODELS_TO_JUDGE:
        if not path.exists():
            raise SystemExit(f"Missing {path}")
        records = load_jsonl(path)
        candidates[model_name] = index_by_pr(records)
        print(f"  Loaded {len(records)} reviews from {model_name}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    scores_path = RESULTS_DIR / "judge_scores_v2.jsonl"
    summary_path = RESULTS_DIR / "summary_v2.json"

    all_scores = []
    total_cost = 0.0

    for pr_key, golden_record in golden_idx.items():
        print(f"\n=== {pr_key} ===")
        reference = golden_record["golden_review"]

        for model_name, _ in MODELS_TO_JUDGE:
            if pr_key not in candidates[model_name]:
                print(f"  [{model_name}] SKIP (no review found)")
                continue

            candidate = candidates[model_name][pr_key]["review"]
            messages = build_judge_prompt(golden_record, reference, candidate)

            print(f"  [{model_name}] judging...", end=" ", flush=True)
            
            result = complete(
                messages=messages,
                model=JUDGE_MODEL,
                temperature=0.0,
            )

            # Strip markdown fences if GPT-4o wrapped the JSON
            raw = result.content.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()

            try:
                score = json.loads(raw)
            except json.JSONDecodeError as e:
                print(f"FAILED to parse JSON: {e}")
                print(f"  Raw response: {result.content[:200]}")
                continue

            total_cost += estimate_cost(result)

            record = {
                "pr": pr_key,
                "model": model_name,
                **score,
            }
            all_scores.append(record)
            print(f"overall={score['overall']}/5")

    # Write per-PR results
    with open(scores_path, "w") as f:
        for record in all_scores:
            f.write(json.dumps(record) + "\n")

    # Aggregate
    by_model = {}
    for score in all_scores:
        m = score["model"]
        if m not in by_model:
            by_model[m] = {
                "issue_coverage": [],
                "false_positives": [],
                "severity_calibration": [],
                "actionability": [],
                "overall": [],
            }
        for dim in by_model[m]:
            by_model[m][dim].append(score[dim])

    summary = {}
    for model_name, dims in by_model.items():
        summary[model_name] = {
            dim: round(sum(vals) / len(vals), 2) for dim, vals in dims.items()
        }

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Print to terminal
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"{'Model':<20} {'Coverage':<10} {'FalsePos':<10} {'Severity':<10} {'Action':<10} {'Overall':<10}")
    print("-" * 70)
    for model_name, scores in summary.items():
        print(
            f"{model_name:<20} "
            f"{scores['issue_coverage']:<10} "
            f"{scores['false_positives']:<10} "
            f"{scores['severity_calibration']:<10} "
            f"{scores['actionability']:<10} "
            f"{scores['overall']:<10}"
        )
    print(f"\nTotal cost: ${total_cost:.3f}")
    print(f"Detailed scores: {scores_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()