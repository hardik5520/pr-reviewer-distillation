"""Split traces into train/val/test sets, stratified by repo.

Reads data/traces/*.jsonl, splits 80/10/10, writes:
- data/splits/train.jsonl
- data/splits/val.jsonl
- data/splits/test.jsonl

Deterministic: same input always produces the same split (fixed seed).
Stratified: each split contains roughly the same proportion from each repo,
so no single repo dominates training and is absent from test.

Run: uv run python scripts/make_splits.py
"""
import json
import random
from collections import defaultdict
from pathlib import Path

SEED = 42
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
# TEST_RATIO = remainder (0.10)

traces_dir = Path("data/traces")
splits_dir = Path("data/splits")
splits_dir.mkdir(parents=True, exist_ok=True)

# Load all traces, grouped by repo for stratification
by_repo: dict[str, list[dict]] = defaultdict(list)
total = 0
for path in sorted(traces_dir.glob("*.jsonl")):
    with open(path) as f:
        for line in f:
            trace = json.loads(line)
            repo_key = f"{trace['owner']}/{trace['repo']}"
            by_repo[repo_key].append(trace)
            total += 1

if total == 0:
    raise SystemExit("No traces in data/traces/. Run generate_traces.py first.")

print(f"Loaded {total} traces from {len(by_repo)} repos\n")
print("Stratified split per repo:")

rng = random.Random(SEED)
train: list[dict] = []
val: list[dict] = []
test: list[dict] = []

for repo, traces in sorted(by_repo.items()):
    shuffled = traces.copy()
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)
    # remainder goes to test (handles rounding cleanly)

    repo_train = shuffled[:n_train]
    repo_val = shuffled[n_train : n_train + n_val]
    repo_test = shuffled[n_train + n_val :]

    train.extend(repo_train)
    val.extend(repo_val)
    test.extend(repo_test)

    print(
        f"  {repo:25} {n:3}  →  "
        f"train {len(repo_train):2}, val {len(repo_val):2}, test {len(repo_test):2}"
    )

# Sort each split by a stable key so file contents are deterministic
def sort_key(t: dict) -> tuple:
    return (t["owner"], t["repo"], t["number"])

train.sort(key=sort_key)
val.sort(key=sort_key)
test.sort(key=sort_key)

# Write the three split files
for name, records in [("train", train), ("val", val), ("test", test)]:
    out_path = splits_dir / f"{name}.jsonl"
    with open(out_path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

print(f"\n=== Split summary ===")
print(f"Train: {len(train):3}  ({100*len(train)/total:.0f}%)")
print(f"Val:   {len(val):3}  ({100*len(val)/total:.0f}%)")
print(f"Test:  {len(test):3}  ({100*len(test)/total:.0f}%)")
print(f"Total: {len(train) + len(val) + len(test)}")
print(f"\nWrote to {splits_dir}/")