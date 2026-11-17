# %%
# Setup: Functions required for fine-tuned model inference

import json
import re
from pathlib import Path

from mlx_lm import generate
from mlx_lm.sample_utils import make_sampler

# 1. Locate project directory
current_dir = Path.cwd()

PROJECT_ROOT = (
    current_dir.parent
    if current_dir.name in ("noteboks", "notebooks")
    else current_dir
)

# 2. Reload the original system prompt
train_path = (
    PROJECT_ROOT / "data" / "processed"
    / "semeval" / "train.jsonl"
)

with open(train_path, encoding="utf-8") as f:
    first_record = json.loads(f.readline())

SYSTEM_PROMPT = first_record["messages"][0]["content"]


# 3. Build non-thinking prompt
def build_review_prompt(review):

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": review}
    ]

    prompt = ft_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )

    if not prompt.rstrip().endswith("</think>"):
        assert prompt.endswith("<|im_start|>assistant\n")
        prompt += "<think>\n\n</think>\n\n"

    return prompt


# 4. Parse model output
def parse_model_output(raw_response):

    cleaned = re.sub(
        r"^\s*<think>.*?</think>\s*",
        "",
        raw_response,
        flags=re.DOTALL
    ).strip()

    predictions = json.loads(cleaned)

    assert isinstance(predictions, list)

    for item in predictions:
        assert isinstance(item, dict)
        assert set(item.keys()) == {"term", "sentiment"}
        assert isinstance(item["term"], str)
        assert item["sentiment"] in {
            "positive", "negative", "neutral", "conflict"
        }

    return predictions


print("Inference setup completed!")

# %%
# Cell 1: Prepare MLX fine-tuning datasets

import json
from pathlib import Path

current_dir = Path.cwd()

PROJECT_ROOT = (
    current_dir.parent
    if current_dir.name in ("noteboks", "notebooks")
    else current_dir
)

SOURCE_DIR = PROJECT_ROOT / "data" / "processed" / "semeval"
MLX_DIR = PROJECT_ROOT / "data" / "processed" / "mlx_sft_v1"

MLX_DIR.mkdir(parents=True, exist_ok=True)

THINK_PREFIX = "<think>\n\n</think>\n\n"

file_mapping = {
    "train.jsonl": "train.jsonl",
    "val.jsonl": "valid.jsonl"
}

for source_name, target_name in file_mapping.items():

    source = SOURCE_DIR / source_name
    destination = MLX_DIR / target_name

    count = 0

    with open(source, encoding="utf-8") as fin, \
         open(destination, "w", encoding="utf-8") as fout:

        for line in fin:
            record = json.loads(line)
            messages = record["messages"]

            assert [m["role"] for m in messages] == [
                "system", "user", "assistant"
            ]

            # Original target must be a valid JSON array
            target = messages[2]["content"]
            assert isinstance(json.loads(target), list)

            mlx_record = {
                "messages": [
                    messages[0],
                    messages[1],
                    {
                        "role": "assistant",
                        "content": THINK_PREFIX + target
                    }
                ]
            }

            fout.write(json.dumps(mlx_record) + "\n")
            count += 1

    print(f"{target_name}: {count} samples")

print("\nMLX data directory:", MLX_DIR)

# %%
# Cell 2: Verify training and inference alignment

from mlx_lm import load

MODEL_ID = "Qwen/Qwen3-1.7B-MLX-4bit"

model, tokenizer = load(MODEL_ID)

with open(MLX_DIR / "train.jsonl", encoding="utf-8") as f:
    sample = json.loads(f.readline())

messages = sample["messages"]

rendered = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=False
)

print("TRAINING SAMPLE END:")
print(repr(rendered[-300:]))

print("\nASSISTANT TARGET:")
print(messages[-1]["content"])

# The rendered training sample should include our non-thinking prefix
assert THINK_PREFIX in rendered
assert rendered.endswith("<|im_end|>\n")

print("\nTraining format check: PASSED")

# %%
# Cell 3: Verify prompt masking

import json
from mlx_lm.tuner.datasets import ChatDataset

with open(MLX_DIR / "train.jsonl", encoding="utf-8") as f:
    sample = json.loads(f.readline())

dataset = ChatDataset(
    [sample],
    tokenizer,
    mask_prompt=True
)

token_ids, offset = dataset.process(sample)

completion_text = tokenizer.decode(token_ids[offset:])

print("Total tokens:", len(token_ids))
print("Prompt tokens (masked):", offset)
print("Completion tokens:", len(token_ids) - offset)

print("\nTRAINING TARGET:")
print(repr(completion_text[:300]))

assert 0 < offset < len(token_ids)
assert "[]" in completion_text

print("\nPrompt masking check: PASSED")

# %%
# Load the smoke-test fine-tuned adapter

from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

ADAPTER_DIR = PROJECT_ROOT / "outputs" / "adapters" / "smoke_v1"

assert (ADAPTER_DIR / "adapters.safetensors").exists()

ft_model, ft_tokenizer = load(
    MODEL_ID,
    adapter_path=str(ADAPTER_DIR)
)

print("Fine-tuned model loaded successfully!")


# %%
# Test fine-tuned model on validation examples

# One example without aspects
empty_example = next(
    r for r in val_data
    if len(json.loads(r["messages"][2]["content"])) == 0
)

# One example containing multiple aspects
multi_example = next(
    r for r in val_data
    if len(json.loads(r["messages"][2]["content"])) >= 2
)

for record in [empty_example, multi_example]:

    review = record["messages"][1]["content"]
    gold = json.loads(record["messages"][2]["content"])

    # Reuse our established non-thinking prompt
    prompt = build_review_prompt(review)

    response = generate(
        ft_model,
        ft_tokenizer,
        prompt=prompt,
        max_tokens=256,
        sampler=make_sampler(temp=0.0),
        verbose=False
    )

    print("\n" + "=" * 60)
    print("INPUT:", review)
    print("GOLD:", gold)
    print("RAW PREDICTION:", response)

    try:
        prediction = parse_model_output(response)
        print("PARSED:", prediction)
    except (ValueError, TypeError, AssertionError) as e:
        print("Parsing failed:", e)

# %%
# Reload validation data for fine-tuned model testing

import json
from pathlib import Path

# Locate ReviewLens project root
current_dir = Path.cwd()

PROJECT_ROOT = (
    current_dir.parent
    if current_dir.name in ("noteboks", "notebooks")
    else current_dir
)

VAL_PATH = PROJECT_ROOT / "data" / "processed" / "semeval" / "val.jsonl"

with open(VAL_PATH, "r", encoding="utf-8") as f:
    val_data = [json.loads(line) for line in f]

print("Validation samples loaded:", len(val_data))

# %%
# Smoke adapter evaluation on the same 30 validation samples

import json
from collections import Counter
from mlx_lm import generate
from mlx_lm.sample_utils import make_sampler

smoke_results = []

for i, record in enumerate(val_data[:30], start=1):

    review = record["messages"][1]["content"]
    gold = json.loads(record["messages"][2]["content"])

    prompt = build_review_prompt(review)

    raw = generate(
        ft_model,
        ft_tokenizer,
        prompt=prompt,
        max_tokens=256,
        sampler=make_sampler(temp=0.0),
        verbose=False
    )

    try:
        pred = parse_model_output(raw)
        success = True
    except (ValueError, AssertionError, TypeError):
        pred = []
        success = False

    smoke_results.append({
        "sentence_id": record["sentence_id"],
        "gold": gold,
        "prediction": pred,
        "parse_success": success,
        "raw_response": raw
    })

    if i % 10 == 0:
        print(f"Completed {i}/30")


def aspect_pairs(items):
    return Counter(
        (" ".join(a["term"].lower().split()), a["sentiment"])
        for a in items
    )


tp = fp = fn = 0

for r in smoke_results:
    gold = aspect_pairs(r["gold"])
    pred = aspect_pairs(r["prediction"])

    tp += sum((gold & pred).values())
    fp += sum((pred - gold).values())
    fn += sum((gold - pred).values())

precision = tp / (tp + fp) if tp + fp else 0
recall = tp / (tp + fn) if tp + fn else 0
f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0

empty_predictions = sum(
    len(r["prediction"]) == 0 for r in smoke_results
)

no_aspect = [r for r in smoke_results if len(r["gold"]) == 0]

print("\n=== Smoke Adapter Evaluation ===")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1: {f1:.4f}")

print(f"Empty predictions: {empty_predictions}/30")
print(
    "No-aspect accuracy:",
    sum(len(r["prediction"]) == 0 for r in no_aspect),
    "/",
    len(no_aspect)
)

print(
    "JSON parse rate:",
    sum(r["parse_success"] for r in smoke_results),
    "/30"
)

# %%
# Diagnose training labels and completion masking

import json
from pathlib import Path
from collections import Counter
from mlx_lm.tuner.datasets import ChatDataset

cwd = Path.cwd()
root = cwd.parent if cwd.name in ("noteboks", "notebooks") else cwd

train_file = root / "data/processed/mlx_sft_v1/train.jsonl"

with open(train_file, encoding="utf-8") as f:
    records = [json.loads(line) for line in f]

def get_target(record):
    content = record["messages"][-1]["content"]
    assert content.startswith("<think>\n\n</think>\n\n")
    return json.loads(content.split("</think>", 1)[1].strip())

# 1. Check target distribution
counts = Counter(
    "empty" if len(get_target(r)) == 0 else "nonempty"
    for r in records
)

print("=== Training target distribution ===")
print(counts)
print("Empty ratio:", round(counts["empty"] / len(records), 4))

# 2. Select one nonempty training example
sample = next(r for r in records if len(get_target(r)) >= 2)

# 3. Check the rendered prompt and completion
prompt_text = ft_tokenizer.apply_chat_template(
    sample["messages"][:-1],
    tokenize=False,
    add_generation_prompt=True
)

full_text = ft_tokenizer.apply_chat_template(
    sample["messages"],
    tokenize=False,
    add_generation_prompt=False
)

print("\n=== Template alignment ===")
print("Prompt is prefix of full text:", full_text.startswith(prompt_text))
assert full_text.startswith(prompt_text)

# 4. Inspect what MLX actually trains on
dataset = ChatDataset([sample], ft_tokenizer, mask_prompt=True)
token_ids, offset = dataset.process(sample)

completion = ft_tokenizer.decode(token_ids[offset:])

print("\n=== Actual training completion ===")
print(repr(completion[:500]))
print("Prompt tokens:", offset)
print("Completion tokens:", len(token_ids) - offset)

# %%
# Verify exact training-inference prompt alignment

sample = next(
    r for r in records
    if len(get_target(r)) >= 2
)

review = sample["messages"][1]["content"]

training_prefix = ft_tokenizer.apply_chat_template(
    sample["messages"][:-1],
    tokenize=False,
    add_generation_prompt=True
)

inference_prefix = build_review_prompt(review)

expected_prefix = (
    training_prefix
    + "<think>\n\n</think>\n\n"
)

print(
    "Training-inference alignment:",
    inference_prefix == expected_prefix
)

print("\nInference prompt ending:")
print(repr(inference_prefix[-100:]))

# %%
# Cell 4: Audit training sequence lengths

import json
from pathlib import Path
from mlx_lm import load

cwd = Path.cwd()
PROJECT_ROOT = cwd.parent if cwd.name in ("noteboks", "notebooks") else cwd

DATA_DIR = PROJECT_ROOT / "data/processed/mlx_sft_v1"

_, audit_tokenizer = load("Qwen/Qwen3-1.7B-MLX-4bit")

for split in ["train", "valid"]:

    path = DATA_DIR / f"{split}.jsonl"
    lengths = []

    with open(path, encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)

            tokens = audit_tokenizer.apply_chat_template(
                record["messages"],
                tokenize=True,
                add_generation_prompt=False
            )

            lengths.append(len(tokens))

    lengths.sort()

    print(f"\n=== {split.upper()} ===")
    print("Samples:", len(lengths))
    print("Median length:", lengths[len(lengths) // 2])
    print("95th percentile:", lengths[int(len(lengths) * 0.95)])
    print("Maximum length:", max(lengths))
    print("Over 1024 tokens:", sum(n > 1024 for n in lengths))
# %%
