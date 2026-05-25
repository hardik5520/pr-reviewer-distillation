"""Merge LoRA adapter into base model, quantize to AWQ, push to Hugging Face.

Split into two functions so Unsloth's transformers-patching from step 1
doesn't leak into AWQ in step 2.

Run: uv run modal run --detach scripts/merge_and_push.py
"""

import modal
import os
from pathlib import Path

# Image 1: training stack (Unsloth) for merging the adapter
merge_image = (
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
    .env({"HF_HOME": "/model_cache", "HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

# Image 2: minimal AWQ + HF stack, no Unsloth
awq_image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        "transformers==4.47.1", # last version before vLLM-specific patches
        "accelerate==1.9.0",
        "huggingface_hub==0.34.2",
        "hf-transfer==0.1.9",
        "autoawq==0.2.7.post3",
    )
    .env({"HF_HOME": "/model_cache", "HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

volume = modal.Volume.from_name("pr-reviewer-models", create_if_missing=True)
app = modal.App("pr-reviewer-merge-push")

# Config
ADAPTER_PATH = "/model_cache/checkpoints/pr-reviewer-7b-instruct/final"
MERGED_PATH = "/model_cache/merged/pr-reviewer-7b-instruct-merged"
AWQ_PATH = "/model_cache/awq/pr-reviewer-7b-instruct-awq"
HF_REPO_ID = "Hardik55/pr-reviewer-7b-instruct-awq"


@app.function(
    image=merge_image,
    gpu="L40S",
    timeout=60 * 30,
    volumes={"/model_cache": volume},
)
def merge_adapter():
    """Load adapter, merge into base, save merged 16-bit model."""
    print("=" * 60)
    print("Step 1/3: Merging adapter into base model")
    print("=" * 60)

    import unsloth  # noqa: F401
    from unsloth import FastLanguageModel

    print(f"Loading adapter from {ADAPTER_PATH}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=ADAPTER_PATH,
        max_seq_length=8192,
        dtype=None,
        load_in_4bit=False,
    )

    print(f"Saving merged model to {MERGED_PATH}...")
    Path(MERGED_PATH).mkdir(parents=True, exist_ok=True)
    model.save_pretrained_merged(
        MERGED_PATH,
        tokenizer,
        save_method="merged_16bit",
    )
    volume.commit()
    print("Merged model saved (~14 GB).")
    return "merge_ok"


@app.function(
    image=awq_image,
    gpu="L40S",
    timeout=60 * 60,
    volumes={"/model_cache": volume},
    secrets=[modal.Secret.from_dotenv()],
)
def quantize_and_push():
    """Load merged model, AWQ-quantize, upload to HuggingFace."""
    import torch
    from awq import AutoAWQForCausalLM
    from transformers import AutoTokenizer
    from huggingface_hub import HfApi, login

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise SystemExit("HF_TOKEN not found")

    print("=" * 60)
    print("Step 2/3: Quantizing merged model to AWQ 4-bit")
    print("=" * 60)

    print(f"Loading merged model from {MERGED_PATH}...")
    awq_model = AutoAWQForCausalLM.from_pretrained(
        MERGED_PATH,
        safetensors=True,
        device_map="auto",
    )
    awq_tokenizer = AutoTokenizer.from_pretrained(MERGED_PATH, trust_remote_code=True)

    quant_config = {
        "zero_point": True,
        "q_group_size": 128,
        "w_bit": 4,
        "version": "GEMM",
    }

    print("Running AWQ quantization (this takes 5-10 minutes)...")
    awq_model.quantize(awq_tokenizer, quant_config=quant_config)

    print(f"Saving AWQ model to {AWQ_PATH}...")
    Path(AWQ_PATH).mkdir(parents=True, exist_ok=True)
    awq_model.save_quantized(AWQ_PATH)
    awq_tokenizer.save_pretrained(AWQ_PATH)
    volume.commit()

    del awq_model
    torch.cuda.empty_cache()
    print("AWQ model saved (~4 GB).")

    # Step 3: push
    print()
    print("=" * 60)
    print("Step 3/3: Pushing to HuggingFace")
    print("=" * 60)

    login(token=hf_token)

    model_card = """---
license: apache-2.0
base_model: unsloth/Qwen2.5-Coder-7B-Instruct
tags:
- code-review
- distillation
- qwen
- awq
- 4-bit
language:
- en
---

# pr-reviewer-7b-instruct-awq

A 7B code review model distilled from GPT-4o, fine-tuned on 73 PRs from popular
Python repositories (fastapi, pydantic, flask, httpx, sqlmodel).

## Model Details

- **Base**: [unsloth/Qwen2.5-Coder-7B-Instruct](https://huggingface.co/unsloth/Qwen2.5-Coder-7B-Instruct)
- **Fine-tuning**: QLoRA (rank=32, alpha=32, dropout=0.05, all linear layers)
- **Quantization**: AWQ 4-bit (~4 GB)
- **Training data**: 73 PRs, GPT-4o-generated reviews
- **Eval**: 13 hand-labeled PRs, GPT-4o-as-judge

## Eval Results (GPT-4o judge, scale 1-5)

| Model | Coverage | FalsePos | Severity | Action | Overall |
|---|---|---|---|---|---|
| GPT-4o (teacher) | 2.62 | 3.46 | 2.38 | 3.46 | **2.62** |
| GPT-4o-mini | 2.77 | 3.38 | 2.62 | 3.38 | **2.77** |
| v1 (base Qwen) | 2.08 | 4.46 | 2.00 | 1.62 | **2.00** |
| v2 (Instruct, this) | 2.31 | 2.77 | 2.08 | 3.00 | **2.38** |

## Usage with vLLM

```bash
vllm serve Hardik55/pr-reviewer-7b-instruct-awq --quantization awq
```

## Lessons Learned

v1 vs v2 documents a real distillation lesson: with only 73 examples, fine-tuning a
base model cannot teach instruction-following AND review style simultaneously.
Switching to the Instruct base variant improved overall score 19% (actionability 85%)
with no other changes.

## Limitations

- 73 training examples is small; model often hedges
- Distillation ceiling: cannot exceed teacher quality
- AWQ calibrated on default wikitext-2 (not domain-specific)

## Source

https://github.com/Hardik55/pr-reviewer-distillation
"""

    readme_path = Path(AWQ_PATH) / "README.md"
    readme_path.write_text(model_card)

    api = HfApi()
    print(f"Uploading {AWQ_PATH} to hf.co/{HF_REPO_ID}...")
    api.upload_folder(
        folder_path=AWQ_PATH,
        repo_id=HF_REPO_ID,
        repo_type="model",
        commit_message="Initial upload of pr-reviewer-7b-instruct-awq",
    )

    print()
    print("=" * 60)
    print(f"DONE. https://huggingface.co/{HF_REPO_ID}")
    print("=" * 60)
    return "push_ok"


@app.local_entrypoint()
def main():
    # print("Step 1: merge adapter...")
    # merge_adapter.remote()
    # print("Step 1 complete.")
    # print()
    print("Step 2-3: quantize + push (separate container)...")
    quantize_and_push.remote()
    print("Done.")