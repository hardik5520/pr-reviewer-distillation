# pr-reviewer-distillation

Distilling a GPT-4o powered PR review agent into a fine-tuned 7B 
open source model. Goal: match or beat GPT-4o on code review for 
roughly $40 of one-time compute.

## Status

Work in progress. Currently: data pipeline.

## Approach

1. Harvest (diff, review) traces from existing PR Review Agent.
2. SFT a Qwen 2.5 Coder 7B base on the traces with QLoRA.
3. DPO pass using human reviews as preference signal.
4. AWQ 4-bit quantize, serve with vLLM.
5. Evaluate against GPT-4o on a hand-labeled golden set.

## Stack

Unsloth, vLLM, Modal, FastAPI, Phoenix.

## Repo layout

- `data/` scraped PRs, generated traces, train/eval splits (gitignored)
- `scripts/` scraper, trace generator, training, eval CLIs
- `src/distill/` library code
- `eval/` eval harness and golden set
- `notebooks/` exploration

## Reproducing

(coming soon)