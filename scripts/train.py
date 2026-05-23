"""Real training run: fine-tune Qwen 2.5 Coder 7B on the PR review dataset.

Reads train.jsonl from the Modal volume, formats each example using the
review rubric prompt template, trains with QLoRA on an L40S GPU, and
saves the trained adapter back to the volume.

Run: uv run modal run --detach scripts/train.py
"""
import modal

# Same pinned versions as hello_finetune.py. Known to work.
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
app = modal.App("pr-reviewer-train", image=image)


# ===== Training configuration =====
# MODEL_NAME = "unsloth/Qwen2.5-Coder-7B"
MODEL_NAME = "unsloth/Qwen2.5-Coder-7B-Instruct"
MAX_SEQ_LENGTH = 16384
LORA_R = 32
LORA_ALPHA = 32
LEARNING_RATE = 2e-4
NUM_EPOCHS = 3
BATCH_SIZE = 2
GRADIENT_ACCUMULATION = 4
SAVE_STEPS = 25
LOGGING_STEPS = 5


@app.function(
    gpu="L40S",
    timeout=60 * 120,   # 2 hours max
    volumes={"/model_cache": volume},
)
def train():
    """Fine-tune Qwen 2.5 Coder 7B on data/splits/train.jsonl."""
    import json
    from pathlib import Path

    # unsloth MUST be imported first so it patches transformers/peft/trl
    import unsloth  # noqa: F401
    from unsloth import FastLanguageModel
    from datasets import Dataset
    from trl import SFTTrainer
    from transformers import TrainingArguments

    # ===== Load the rubric (embedded in every prompt) =====
    print("Loading rubric...")
    rubric_path = Path("/model_cache/data/REVIEW_RUBRIC.md")
    if not rubric_path.exists():
        # fallback: bundle rubric into the upload script if missing
        raise SystemExit(
            f"Rubric not found at {rubric_path}. "
            "Add REVIEW_RUBRIC.md to upload_training_data.py."
        )
    rubric = rubric_path.read_text()

    # ===== Load training data from the Modal volume =====
    print("Loading training data...")
    train_path = Path("/model_cache/data/splits/train.jsonl")
    val_path = Path("/model_cache/data/splits/val.jsonl")

    train_records = [json.loads(line) for line in open(train_path)]
    val_records = [json.loads(line) for line in open(val_path)]
    print(f"Loaded {len(train_records)} train, {len(val_records)} val")

    # ===== Format each PR into a chat-formatted training example =====
    print("Formatting examples...")
    SYSTEM_PROMPT = (
        "You are an expert code reviewer. Your job is to review a pull "
        "request diff and write a concrete, useful review that follows "
        "the rubric below.\n\n# RUBRIC\n\n" + rubric
    )

    def format_example(pr: dict) -> dict:
        user_msg = (
            f"Review the following pull request.\n\n"
            f"# Title\n{pr['title']}\n\n"
            f"# Description\n{pr.get('body') or '(no description)'}\n\n"
            f"# Diff\n<diff>\n{pr['diff']}\n</diff>\n\n"
            f"Write your review now, following the rubric's required "
            f"structure (Summary, Issues, Strengths) exactly."
        )
        return {
            "text": (
                f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
                f"<|im_start|>user\n{user_msg}<|im_end|>\n"
                f"<|im_start|>assistant\n{pr['review']}<|im_end|>"
            )
        }

    train_dataset = Dataset.from_list([format_example(p) for p in train_records])
    val_dataset = Dataset.from_list([format_example(p) for p in val_records])

    # ===== Load model and attach LoRA =====
    print(f"Loading {MODEL_NAME}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )

    print("Adding LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=LORA_ALPHA,
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    # ===== Train =====
    print("Starting training...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        args=TrainingArguments(
            # output_dir="/model_cache/checkpoints/pr-reviewer-7b",
            output_dir="/model_cache/checkpoints/pr-reviewer-7b-instruct",
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION,
            num_train_epochs=NUM_EPOCHS,
            learning_rate=LEARNING_RATE,
            lr_scheduler_type="cosine",
            warmup_ratio=0.05,
            weight_decay=0.01,
            optim="adamw_8bit",
            logging_steps=LOGGING_STEPS,
            save_strategy="steps",
            save_steps=SAVE_STEPS,
            save_total_limit=3,
            eval_strategy="steps",
            eval_steps=SAVE_STEPS,
            bf16=True,
            seed=42,
            report_to="none",
        ),
    )
    trainer.train()

    # ===== Save the final adapter =====
    print("Saving final adapter...")
    # final_path = "/model_cache/checkpoints/pr-reviewer-7b/final"
    final_path = "/model_cache/checkpoints/pr-reviewer-7b-instruct/final"
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    volume.commit()

    print(f"Done. Adapter saved to {final_path}")
    return "ok"


@app.local_entrypoint()
def main():
    print("Submitting training job to Modal...")
    result = train.remote()
    print(f"Result: {result}")