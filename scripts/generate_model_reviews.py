"""Generate reviews from the trained model on the 13 golden PRs.

Loads Qwen 2.5 Coder 7B (4-bit base from Modal volume) + your trained
LoRA adapter, runs inference on each golden PR, saves results to
eval/baselines/pr-reviewer-7b.jsonl in the same format as the GPT-4o
baselines.

After this, we have three review files to compare against golden:
  - eval/baselines/gpt-4o.jsonl       (teacher model)
  - eval/baselines/gpt-4o-mini.jsonl  (cheap baseline)
  - eval/baselines/pr-reviewer-7b.jsonl  (your trained student) ← NEW

Run: uv run modal run scripts/generate_model_reviews.py
"""
import modal

# Same pinned versions as training. Don't change.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        "accelerate==1.9.0",
        "datasets==3.6.0",
        "hf-transfer==0.1.9",
        "huggingface_hub==0.34.2",
        "peft==0.16.0",
        "transformers==4.54.0",
        "trl==0.19.1",
        "unsloth[cu128-torch270]==2025.7.8",
        "unsloth_zoo==2025.7.10",
    )
    .env({"HF_HOME": "/model_cache"})
)

volume = modal.Volume.from_name("pr-reviewer-models", create_if_missing=True)
app = modal.App("pr-reviewer-eval-gen", image=image)


# Config
# MODEL_NAME = "unsloth/Qwen2.5-Coder-7B"
MODEL_NAME = "unsloth/Qwen2.5-Coder-7B-Instruct"
# ADAPTER_PATH = "/model_cache/checkpoints/pr-reviewer-7b/final"
ADAPTER_PATH = "/model_cache/checkpoints/pr-reviewer-7b-instruct/final"
MAX_SEQ_LENGTH = 16384
MAX_NEW_TOKENS = 1024     # max length of generated review
TEMPERATURE = 0.3         # same as we used for GPT-4o trace generation


@app.function(
    gpu="L40S",
    timeout=60 * 30,
    volumes={"/model_cache": volume},
)
def generate_reviews(golden_records: list[dict]) -> list[dict]:
    """Load model + adapter, run inference on each golden PR."""
    import json
    from pathlib import Path

    # unsloth MUST be first
    import unsloth  # noqa: F401
    from unsloth import FastLanguageModel

    # Load rubric (we need it for the system prompt)
    rubric_path = Path("/model_cache/data/REVIEW_RUBRIC.md")
    rubric = rubric_path.read_text()

    SYSTEM_PROMPT = (
        "You are an expert code reviewer. Your job is to review a pull "
        "request diff and write a concrete, useful review that follows "
        "the rubric below.\n\n# RUBRIC\n\n" + rubric
    )

    print(f"Loading {MODEL_NAME} + adapter from {ADAPTER_PATH}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=ADAPTER_PATH,    # loads base + adapter together
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )

    # Switch to inference mode (2x faster)
    FastLanguageModel.for_inference(model)

    results = []
    for i, pr in enumerate(golden_records, 1):
        print(f"\n[{i}/{len(golden_records)}] {pr['owner']}/{pr['repo']}#{pr['number']}")

        user_msg = (
            f"Review the following pull request.\n\n"
            f"# Title\n{pr['title']}\n\n"
            f"# Description\n{pr.get('body') or '(no description)'}\n\n"
            f"# Diff\n<diff>\n{pr['diff']}\n</diff>\n\n"
            f"Write your review now, following the rubric's required "
            f"structure (Summary, Issues, Strengths) exactly."
        )

        prompt = (
            f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{user_msg}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            do_sample=True,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id,
        )

        # Strip the prompt out, keep only what the model generated
        generated_tokens = outputs[0][inputs.input_ids.shape[1]:]
        review = tokenizer.decode(generated_tokens, skip_special_tokens=True)

        # Clean up trailing <|im_end|> if not stripped
        review = review.split("<|im_end|>")[0].strip()

        print(f"  Generated {len(review):,} chars")

        results.append({
            "owner": pr["owner"],
            "repo": pr["repo"],
            "number": pr["number"],
            "title": pr["title"],
            "review": review,
            "model": "pr-reviewer-7b",
        })

    return results


@app.local_entrypoint()
def main():
    """Read golden set locally, send PRs to Modal, save results locally."""
    import json
    from pathlib import Path

    golden_path = Path("eval/golden/golden.jsonl")
    if not golden_path.exists():
        raise SystemExit(f"Missing {golden_path}")

    golden_records = [json.loads(line) for line in open(golden_path)]
    print(f"Loaded {len(golden_records)} golden PRs")

    print("Running inference on Modal...")
    results = generate_reviews.remote(golden_records)

    # out_path = Path("eval/baselines/pr-reviewer-7b.jsonl")
    out_path = Path("eval/baselines/pr-reviewer-7b-instruct.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for record in results:
            f.write(json.dumps(record) + "\n")

    print(f"\nSaved {len(results)} reviews to {out_path}")