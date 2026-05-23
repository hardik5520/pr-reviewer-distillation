"""Smoke test for the fine-tuning pipeline on Modal.

Uses Modal's known-good pinned versions from their official Unsloth example.
The unsloth[cu128-torch270] extra brings the right torch with it, avoiding
the torchao/torch version mismatch loop we hit before.

Run: uv run modal run --detach scripts/hello_finetune.py
"""
import modal

# Known-good versions from Modal's official Unsloth example.
# Do not bump these without testing the whole stack.
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
app = modal.App("pr-reviewer-hello", image=image)


@app.function(
    gpu="T4",
    timeout=60 * 30,
    volumes={"/model_cache": volume},
)
def train_tiny():
    """Fine-tune Qwen 2.5 0.5B on 5 examples. Smoke test only."""
    # unsloth MUST be imported first so it patches transformers/peft/trl
    import unsloth  # noqa: F401
    from unsloth import FastLanguageModel
    from datasets import Dataset
    from trl import SFTTrainer
    from transformers import TrainingArguments

    print("Loading Qwen 2.5 0.5B...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Qwen2.5-0.5B",
        max_seq_length=4096,
        dtype=None,
        load_in_4bit=True,
    )

    print("Adding LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
    )

    print("Preparing 5 toy examples...")
    examples = [
        {
            "text": (
                f"<|im_start|>system\nYou are a code reviewer.<|im_end|>\n"
                f"<|im_start|>user\nReview this diff: example diff {i}<|im_end|>\n"
                f"<|im_start|>assistant\nReview {i}: looks fine.<|im_end|>"
            )
        }
        for i in range(5)
    ]
    dataset = Dataset.from_list(examples)

    print("Training...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=4096,
        args=TrainingArguments(
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            max_steps=5,
            learning_rate=2e-4,
            output_dir="/model_cache/hello-checkpoint",
            logging_steps=1,
            save_strategy="no",
            report_to="none",
        ),
    )
    trainer.train()

    print("Saving adapter to Modal volume...")
    model.save_pretrained("/model_cache/hello-adapter")
    tokenizer.save_pretrained("/model_cache/hello-adapter")
    volume.commit()

    print("Done. Hello-fine-tune complete.")
    return "ok"


@app.local_entrypoint()
def main():
    print("Submitting hello-fine-tune job to Modal...")
    result = train_tiny.remote()
    print(f"Result: {result}")