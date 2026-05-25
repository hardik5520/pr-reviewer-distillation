# pr-reviewer-distillation

A 7B code review model distilled from GPT-4o, deployed as a live OpenAI-compatible API.

- **Model**: [Hardik55/pr-reviewer-7b-instruct-awq](https://huggingface.co/Hardik55/pr-reviewer-7b-instruct-awq) (AWQ 4-bit, ~4 GB)
- **Live API**: `https://hardik5520--pr-reviewer-vllm-serve.modal.run` (OpenAI-compatible)
- **Base**: Qwen 2.5 Coder 7B Instruct + QLoRA on 73 distillation traces
- **Total compute cost**: ~$8 (training + quantization + eval)

## Try it

    from openai import OpenAI

    client = OpenAI(
        base_url="https://hardik5520--pr-reviewer-vllm-serve.modal.run/v1",
        api_key="not-needed",
    )

    response = client.chat.completions.create(
        model="pr-reviewer-7b-instruct-awq",
        messages=[
            {"role": "system", "content": "You are an expert code reviewer. Follow this rubric: Summary, Issues, Strengths."},
            {"role": "user", "content": "Review this diff:\n<your diff here>"},
        ],
        max_tokens=512,
    )
    print(response.choices[0].message.content)

First request triggers a ~60 sec cold start (Modal scales to zero between requests).

## What this is

A full-stack distillation experiment: scrape real PRs → generate teacher reviews with GPT-4o → fine-tune a 7B student → AWQ quantize → serve via vLLM on Modal. End-to-end production pipeline, not a notebook demo.

## Eval results

Scored against 13 hand-labeled "golden" PR reviews, judged by GPT-4o.

| Model | Coverage | False Positives | Severity | Actionability | Overall |
|---|---|---|---|---|---|
| GPT-4o (teacher) | 2.62 | 3.46 | 2.38 | 3.46 | **2.62** |
| GPT-4o-mini | 2.77 | 3.38 | 2.62 | 3.38 | **2.77** |
| pr-reviewer-7b (v1, base) | 2.08 | 4.46 | 2.00 | 1.62 | **2.00** |
| pr-reviewer-7b (v2, Instruct) | 2.31 | 2.77 | 2.08 | 3.00 | **2.38** |

**Key finding**: switching from base Qwen to the Instruct variant of the same model improved overall score 19% and actionability 85%, with no other changes. With only 73 training examples, fine-tuning a base model can't teach instruction-following AND review style simultaneously.

Detailed scores: [eval/results/](eval/results/)

## Pipeline overview

    GitHub PRs (5 repos)
           ↓ scripts/scrape_prs.py
        92 PRs (raw diff + metadata)
           ↓ scripts/generate_traces.py
        92 GPT-4o reviews ($1)
           ↓ scripts/make_splits.py
        73 train / 9 val / 10 test
           ↓ scripts/label_golden.py
        13 hand-labeled golden reviews (eval anchor)
           ↓ scripts/train.py (Modal, L40S, QLoRA)
        LoRA adapter (~300 MB, $0.40)
           ↓ scripts/merge_and_push.py (Modal, L40S)
        Merged model → AWQ 4-bit → Hugging Face ($1)
           ↓ scripts/serve_vllm.py (Modal, L4)
        Live OpenAI-compatible API

## Stack

- **Training**: Unsloth + QLoRA on Modal (L40S GPU)
- **Quantization**: autoawq
- **Serving**: vLLM on Modal (L4 GPU, scales to zero)
- **Distribution**: Hugging Face Hub
- **Eval**: GPT-4o as judge with structured rubric

## Repo layout

    data/                # scraped PRs, traces, splits (gitignored)
    eval/
      golden/            # 13 hand-labeled reference reviews (committed)
      baselines/         # model outputs on golden set (committed)
      results/           # judge scores and summaries (committed)
    src/distill/         # reusable library code
    scripts/             # CLI entry points
    REVIEW_RUBRIC.md     # rubric used for trace generation and inference

## Reproducing

Requires: Python 3.11, Modal account, OpenAI API key, Hugging Face account.

    # Setup
    uv sync
    cp .env.example .env  # fill in OPENAI_API_KEY, HF_TOKEN

    # Data
    uv run python scripts/scrape_prs.py
    uv run python scripts/generate_traces.py
    uv run python scripts/make_splits.py

    # (Optional, takes hours) Hand-label your own golden set:
    uv run python scripts/label_golden.py

    # Train + eval
    uv run python scripts/run_baselines.py
    uv run modal run scripts/upload_training_data.py
    uv run modal run --detach scripts/train.py
    uv run modal run scripts/generate_model_reviews.py
    uv run python scripts/judge_eval.py

    # Production
    uv run modal run --detach scripts/merge_and_push.py
    uv run modal deploy scripts/serve_vllm.py

Total wall-clock: ~3 hours. Total cost: under $10.

## Writeup

Engineering story: [link to Medium post 1, coming today]
Eval analysis: [link to Medium post 2, coming tomorrow]