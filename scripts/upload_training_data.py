"""Upload local training data files to the Modal volume.

The Modal training container can't read your laptop's filesystem. We
copy data/splits/*.jsonl into the persistent Modal volume so the
training function can load it from /model_cache/data/splits/.

Run once before training. Re-run if your splits change.

Run: uv run modal run scripts/upload_training_data.py
"""
import modal

app = modal.App("pr-reviewer-upload")
volume = modal.Volume.from_name("pr-reviewer-models", create_if_missing=True)


@app.function(
    volumes={"/model_cache": volume},
    timeout=60 * 10,
)
def upload(splits: dict[str, bytes]):
    from pathlib import Path

    splits_dir = Path("/model_cache/data/splits")
    splits_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path("/model_cache/data")

    for name, content in splits.items():
        if name.startswith("../"):
            # rubric goes one level up from splits/
            out_path = data_dir / name.replace("../", "")
        else:
            out_path = splits_dir / name
        out_path.write_bytes(content)
        print(f"Wrote {out_path} ({len(content):,} bytes)")

    volume.commit()
    print("Upload complete.")


@app.local_entrypoint()
def main():
    from pathlib import Path

    splits_dir = Path("data/splits")
    files = ["train.jsonl", "val.jsonl", "test.jsonl"]

    splits = {}
    for name in files:
        path = splits_dir / name
        if not path.exists():
            raise SystemExit(f"Missing {path}. Run make_splits.py first.")
        splits[name] = path.read_bytes()
        print(f"Read {path} ({len(splits[name]):,} bytes)")

    # Also include the rubric
    rubric_path = Path("REVIEW_RUBRIC.md")
    if not rubric_path.exists():
        raise SystemExit("REVIEW_RUBRIC.md missing in repo root")
    splits["../REVIEW_RUBRIC.md"] = rubric_path.read_bytes()
    print(f"Read {rubric_path} ({len(splits['../REVIEW_RUBRIC.md']):,} bytes)")

    print("\nUploading to Modal volume...")
    upload.remote(splits)