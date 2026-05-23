"""LLM-as-judge eval using Claude instead of GPT-4o.

Same logic as judge_eval.py but uses Anthropic's Claude as the judge to
check whether the GPT-4o judge results are stable across different
judges, or judge-dependent.

Output: eval/results/judge_scores_claude.jsonl and summary_claude.json

Run: uv run python scripts/judge_eval_claude.py
"""
import json
import os
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

JUDGE_MODEL = "claude-sonnet-4-6"
GOLDEN_PATH = Path("eval/golden/golden.jsonl")
BASELINES_DIR = Path("eval/baselines")
RESULTS_DIR = Path("eval/results")

# MODELS_TO_JUDGE = [
#     ("gpt-4o", BASELINES_DIR / "gpt-4o.jsonl"),
#     ("gpt-4o-mini", BASELINES_DIR / "gpt-4o-mini.jsonl"),
#     ("pr-reviewer-7b", BASELINES_DIR / "pr-reviewer-7b.jsonl"),
# ]
MODELS_TO_JUDGE = [
    ("gpt-4o", BASELINES_DIR / "gpt-4o.jsonl"),
    ("gpt-4o-mini", BASELINES_DIR / "gpt-4o-mini.jsonl"),
    ("pr-reviewer-7b", BASELINES_DIR / "pr-reviewer-7b.jsonl"),                  # v1, base model
    ("pr-reviewer-7b-instruct", BASELINES_DIR / "pr-reviewer-7b-instruct.jsonl"),  # v2, instruct model
]

# Claude pricing (Opus 4.7 approx): $15/M input, $75/M output
COST_PER_M = {"input": 3.0, "output": 15.0}


JUDGE_SYSTEM = """You are an expert code review evaluator. Your job is to \
compare a CANDIDATE code review against a REFERENCE code review written by \
a human expert, and score how well the candidate matches the reference's \
substance and judgment.

Return a JSON object with this exact structure:
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

Scoring rubric (1-5 per dimension):
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
made-up nitpicks or hallucinated bugs = low score. Clean focused review = high.

severity_calibration: Does the candidate match the reference's sense of what's \
important? Severity inflation/deflation = low. Same priorities = high.

actionability: Can a developer act on this review? Concrete file/line/fix \
suggestions = high. Vague handwaving = low.

overall: Holistic judgment. Would a senior engineer be roughly as happy \
receiving this review as the reference review?

Return ONLY the JSON object. No preamble, no markdown fences, no explanatory \
text before or after. Even if the candidate review appears empty, garbled, or \
nonsensical, return a valid JSON object with all scores set to 1 and reasoning \
fields explaining the issue. Never refuse to score."""


def build_user_msg(pr: dict, reference: str, candidate: str) -> str:
    return (
        f"# Pull Request\n\n"
        f"Repo: {pr['owner']}/{pr['repo']}\n"
        f"Title: {pr['title']}\n\n"
        f"# REFERENCE REVIEW (human expert)\n\n"
        f"{reference}\n\n"
        f"---\n\n"
        f"# CANDIDATE REVIEW (to be scored)\n\n"
        f"{candidate}\n\n"
        f"---\n\n"
        f"Score the candidate review against the reference. Return JSON only."
    )


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in open(path)]


def index_by_pr(records: list[dict]) -> dict[str, dict]:
    return {f"{r['owner']}/{r['repo']}#{r['number']}": r for r in records}


def main():
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise SystemExit("ANTHROPIC_API_KEY not set in .env")
    client = Anthropic(api_key=api_key)

    print(f"Loading golden set from {GOLDEN_PATH}...")
    golden = load_jsonl(GOLDEN_PATH)
    golden_idx = index_by_pr(golden)
    print(f"  {len(golden)} golden PRs")

    candidates = {}
    for model_name, path in MODELS_TO_JUDGE:
        if not path.exists():
            raise SystemExit(f"Missing {path}")
        candidates[model_name] = index_by_pr(load_jsonl(path))
        print(f"  Loaded {len(candidates[model_name])} from {model_name}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    scores_path = RESULTS_DIR / "judge_scores_claude_v2.jsonl"
    summary_path = RESULTS_DIR / "summary_claude_v2.json"

    all_scores = []
    total_cost = 0.0

    for pr_key, golden_record in golden_idx.items():
        print(f"\n=== {pr_key} ===")
        reference = golden_record["golden_review"]

        for model_name, _ in MODELS_TO_JUDGE:
            if pr_key not in candidates[model_name]:
                print(f"  [{model_name}] SKIP")
                continue

            candidate = candidates[model_name][pr_key]["review"]
            user_msg = build_user_msg(golden_record, reference, candidate)

            print(f"  [{model_name}] judging...", end=" ", flush=True)

            response = client.messages.create(
                model=JUDGE_MODEL,
                max_tokens=1024,
                system=JUDGE_SYSTEM,
                messages=[{"role": "user", "content": user_msg}],
            )

            raw = response.content[0].text.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()

            try:
                score = json.loads(raw)
            except json.JSONDecodeError as e:
                print(f"FAILED to parse: {e}")
                print(f"  Claude returned: {raw[:300]!r}")
                # Assign worst-case scores so the model isn't credited for unparseable output
                score = {
                    "issue_coverage": 1,
                    "issue_coverage_reasoning": "parse failure",
                    "false_positives": 1,
                    "false_positives_reasoning": "parse failure",
                    "severity_calibration": 1,
                    "severity_calibration_reasoning": "parse failure",
                    "actionability": 1,
                    "actionability_reasoning": "parse failure",
                    "overall": 1,
                    "overall_reasoning": "Judge returned non-JSON response, assigned 1/5",
                }

            total_cost += (
                response.usage.input_tokens * COST_PER_M["input"] / 1_000_000
                + response.usage.output_tokens * COST_PER_M["output"] / 1_000_000
            )

            record = {"pr": pr_key, "model": model_name, **score}
            all_scores.append(record)
            print(f"overall={score['overall']}/5")

    with open(scores_path, "w") as f:
        for record in all_scores:
            f.write(json.dumps(record) + "\n")

    by_model = {}
    for score in all_scores:
        m = score["model"]
        if m not in by_model:
            by_model[m] = {
                "issue_coverage": [], "false_positives": [],
                "severity_calibration": [], "actionability": [], "overall": [],
            }
        for dim in by_model[m]:
            by_model[m][dim].append(score[dim])

    summary = {
        m: {dim: round(sum(vals) / len(vals), 2) for dim, vals in dims.items()}
        for m, dims in by_model.items()
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print("CLAUDE-AS-JUDGE RESULTS")
    print("=" * 70)
    print(f"{'Model':<20} {'Coverage':<10} {'FalsePos':<10} {'Severity':<10} {'Action':<10} {'Overall':<10}")
    print("-" * 70)
    for model_name, scores in summary.items():
        print(
            f"{model_name:<20} "
            f"{scores['issue_coverage']:<10} {scores['false_positives']:<10} "
            f"{scores['severity_calibration']:<10} {scores['actionability']:<10} "
            f"{scores['overall']:<10}"
        )
    print(f"\nTotal cost: ${total_cost:.3f}")
    print(f"Detailed: {scores_path}")


if __name__ == "__main__":
    main()