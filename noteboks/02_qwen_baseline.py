# %%
# Cell 1: Project setup and load task instructions

import json
import sys
from pathlib import Path

from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

# Locate project root
current_dir = Path.cwd()

if current_dir.name in ("noteboks", "notebooks"):
    PROJECT_ROOT = current_dir.parent
else:
    PROJECT_ROOT = current_dir

DATA_DIR = PROJECT_ROOT / "data" / "processed" / "semeval"
train_path = DATA_DIR / "train.jsonl"

assert train_path.exists(), f"Missing file: {train_path}"

# Reuse exactly the same system prompt as our SFT dataset
with open(train_path, encoding="utf-8") as f:
    first_record = json.loads(f.readline())

SYSTEM_PROMPT = first_record["messages"][0]["content"]

print("Python:", sys.executable)
print("Training file:", train_path)
print("System prompt loaded successfully!")

# %%
# Cell 2: Load pretrained Qwen3 model

MODEL_ID = "Qwen/Qwen3-1.7B-MLX-4bit"

print("Loading:", MODEL_ID)

model, tokenizer = load(MODEL_ID)

print("Model loaded successfully!")
print("Tokenizer type:", type(tokenizer).__name__)

# %%
# Cell 3: Baseline inference

review = (
    "The screen is beautiful, "
    "but the battery life is terrible."
)

messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": review}
]

# Format the conversation for Qwen
prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False
)

# Generate output without fine-tuning
response = generate(
    model,
    tokenizer,
    prompt=prompt,
    max_tokens=192,
    sampler=make_sampler(temp=0.0),
    verbose=False
)

print("INPUT:")
print(review)

print("\nMODEL OUTPUT:")
print(response)

# Check whether it is valid JSON
try:
    predictions = json.loads(response.strip())
    assert isinstance(predictions, list)

    print("\nJSON parsing: SUCCESS")
    print("Extracted aspects:", len(predictions))

except (json.JSONDecodeError, AssertionError) as e:
    print("\nJSON parsing: FAILED")
    print("Reason:", e)

# %%
# Cell 4: Clean and validate model output

import re
import json

def parse_model_output(raw_response):
    # Remove a leading Qwen thinking block if present
    cleaned = re.sub(
        r"^\s*<think>.*?</think>\s*",
        "",
        raw_response,
        flags=re.DOTALL
    ).strip()

    # Parse JSON
    predictions = json.loads(cleaned)

    # Validate structure
    assert isinstance(predictions, list)

    for item in predictions:
        assert isinstance(item, dict)
        assert set(item.keys()) == {"term", "sentiment"}
        assert isinstance(item["term"], str)
        assert item["sentiment"] in {
            "positive", "negative", "neutral", "conflict"
        }

    return predictions


try:
    predictions = parse_model_output(response)

    print("JSON parsing: SUCCESS")
    print(json.dumps(predictions, indent=2))

except (ValueError, AssertionError, TypeError) as e:
    print("JSON parsing: FAILED")
    print("Reason:", e)

# %%
# Cell 5: Load validation data

import json
import random
from collections import Counter

val_path = DATA_DIR / "val.jsonl"

with open(val_path, "r", encoding="utf-8") as f:
    val_data = [json.loads(line) for line in f]

print("Validation samples:", len(val_data))

# Group samples by number of annotated aspects
groups = {
    "no_aspect": [],
    "one_aspect": [],
    "multiple_aspects": []
}

for record in val_data:
    target = json.loads(record["messages"][2]["content"])
    n = len(target)

    if n == 0:
        groups["no_aspect"].append(record)
    elif n == 1:
        groups["one_aspect"].append(record)
    else:
        groups["multiple_aspects"].append(record)

# Fixed random sample for reproducibility
rng = random.Random(42)

sample_data = []

for group in groups.values():
    sample_data.extend(
        rng.sample(group, min(4, len(group)))
    )

rng.shuffle(sample_data)

print("Selected samples:", len(sample_data))
print("Group sizes:", {k: len(v) for k, v in groups.items()})

# %%
# Cell 6: Run baseline inference

from mlx_lm import generate
from mlx_lm.sample_utils import make_sampler

baseline_results = []

for i, record in enumerate(sample_data, start=1):

    review = record["messages"][1]["content"]

    gold = json.loads(
        record["messages"][2]["content"]
    )

    prompt = build_review_prompt(review)

    response = generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=256,
        sampler=make_sampler(temp=0.0),
        verbose=False
    )

    try:
        prediction = parse_model_output(response)
        parse_success = True

    except (ValueError, AssertionError, TypeError):
        prediction = []
        parse_success = False

    baseline_results.append({
        "sentence_id": record["sentence_id"],
        "text": review,
        "gold": gold,
        "prediction": prediction,
        "parse_success": parse_success,
        "raw_response": response
    })

    print(
        f"[{i}/{len(sample_data)}] "
        f"ID={record['sentence_id']} | "
        f"JSON valid={parse_success}"
    )

print("\nInference completed!")

# %%
# Cell 7: Evaluate aspect-sentiment extraction

def normalize_term(term):
    return " ".join(term.lower().split())


def to_counter(aspects):
    return Counter(
        (
            normalize_term(a["term"]),
            a["sentiment"]
        )
        for a in aspects
    )


def evaluate_results(results):

    tp = 0
    fp = 0
    fn = 0

    for result in results:

        gold = to_counter(result["gold"])
        pred = to_counter(result["prediction"])

        matched = gold & pred

        tp += sum(matched.values())
        fp += sum((pred - gold).values())
        fn += sum((gold - pred).values())

    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0
    )

    parse_rate = sum(
        r["parse_success"] for r in results
    ) / len(results)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "parse_success_rate": parse_rate,
        "tp": tp,
        "fp": fp,
        "fn": fn
    }


metrics = evaluate_results(baseline_results)

print("=== Baseline Evaluation ===")

for name, value in metrics.items():
    if name in {"tp", "fp", "fn"}:
        print(f"{name}: {value}")
    else:
        print(f"{name}: {value:.4f}")

# %%
# Cell 8: Diagnose JSON parsing failures

failed_results = [
    r for r in baseline_results
    if not r["parse_success"]
]

print("Total samples:", len(baseline_results))
print("Parsing failures:", len(failed_results))

for i, result in enumerate(failed_results, 1):

    print("\n" + "=" * 65)
    print(f"FAILURE {i}")
    print("Sentence ID:", result["sentence_id"])

    print("\nINPUT:")
    print(result["text"])

    print("\nGOLD:")
    print(result["gold"])

    print("\nRAW MODEL OUTPUT:")
    print(repr(result["raw_response"][:800]))

    print("\nOUTPUT END:")
    print(repr(result["raw_response"][-150:]))

    try:
        parse_model_output(result["raw_response"])
    except (ValueError, AssertionError, TypeError) as e:
        print("\nERROR TYPE:", type(e).__name__)
        print("ERROR:", str(e))

# %%
# Cell 9: Fix Qwen thinking mode

def build_review_prompt(review):

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": review}
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )

    # Current Qwen MLX template may ignore enable_thinking
    # Explicitly close an empty thinking block
    if not prompt.rstrip().endswith("</think>"):
        assert prompt.endswith("<|im_start|>assistant\n"), (
            "Unexpected chat template format"
        )
        prompt += "<think>\n\n</think>\n\n"

    return prompt

# %%
# Cell 10: Test the corrected prompt

review = next(
    r["text"]
    for r in baseline_results
    if r["sentence_id"] == "2888"
)

prompt = build_review_prompt(review)

print("PROMPT END:")
print(repr(prompt[-100:]))

response_fixed = generate(
    model,
    tokenizer,
    prompt=prompt,
    max_tokens=512,
    sampler=make_sampler(temp=0.0),
    verbose=False
)

print("\nRAW OUTPUT:")
print(response_fixed)

try:
    prediction_fixed = parse_model_output(response_fixed)
    print("\nJSON parsing: SUCCESS")
    print("Prediction:", prediction_fixed)
except (ValueError, AssertionError, TypeError) as e:
    print("\nJSON parsing: FAILED")
    print("Reason:", e)

# %%
# Analyze over-extraction and exact match

from collections import Counter

def is_exact_match(result):
    return (
        to_counter(result["gold"])
        == to_counter(result["prediction"])
    )

# 1. No-aspect accuracy
no_aspect_results = [
    r for r in baseline_results
    if len(r["gold"]) == 0
]

correct_empty = sum(
    len(r["prediction"]) == 0
    for r in no_aspect_results
)

print("=== No-Aspect Performance ===")
print("No-aspect samples:", len(no_aspect_results))
print("Correct empty predictions:", correct_empty)
print(
    "No-aspect accuracy:",
    round(correct_empty / len(no_aspect_results), 4)
)

# 2. Exact sentence-level match
exact_matches = sum(
    is_exact_match(r)
    for r in baseline_results
)

print("\n=== Exact Match ===")
print("Exact matches:", exact_matches)
print("Total samples:", len(baseline_results))
print("Exact match rate:", exact_matches / len(baseline_results))

# 3. Inspect false positives
print("\n=== Examples of Over-extraction ===")

errors = []

for r in baseline_results:
    gold = to_counter(r["gold"])
    pred = to_counter(r["prediction"])

    extra = pred - gold
    missing = gold - pred

    if extra or missing:
        errors.append({
            "text": r["text"],
            "gold": r["gold"],
            "prediction": r["prediction"],
            "extra": list(extra.elements()),
            "missing": list(missing.elements())
        })

# Sort by number of extra predictions
errors.sort(
    key=lambda x: len(x["extra"]),
    reverse=True
)

for error in errors[:4]:
    print("\n" + "=" * 60)
    print("TEXT:", error["text"])
    print("GOLD:", error["gold"])
    print("PREDICTED:", error["prediction"])
    print("EXTRA:", error["extra"])
    print("MISSING:", error["missing"])

# %%
# Full validation baseline with checkpoints

import json
from pathlib import Path
from mlx_lm import generate
from mlx_lm.sample_utils import make_sampler

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "baseline"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RESULT_FILE = OUTPUT_DIR / "qwen3_baseline_v1.jsonl"

# First test 30 samples; later change to len(val_data)
N_TO_RUN = len(val_data)

# Load completed predictions
saved = {}

if RESULT_FILE.exists():
    with open(RESULT_FILE, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                result = json.loads(line)
                saved[str(result["sentence_id"])] = result

sampler = make_sampler(temp=0.0)

for i, record in enumerate(val_data[:N_TO_RUN], start=1):

    sid = str(record["sentence_id"])

    if sid in saved:
        continue

    review = record["messages"][1]["content"]
    gold = json.loads(record["messages"][2]["content"])

    prompt = build_review_prompt(review)

    raw = generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=256,
        sampler=sampler,
        verbose=False
    )

    try:
        prediction = parse_model_output(raw)
        parse_success = True
    except (ValueError, TypeError, AssertionError):
        prediction = []
        parse_success = False

    result = {
        "sentence_id": sid,
        "text": review,
        "gold": gold,
        "prediction": prediction,
        "parse_success": parse_success,
        "raw_response": raw
    }

    with open(RESULT_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")

    saved[sid] = result

    if i % 10 == 0:
        print(f"Completed {i}/{N_TO_RUN}")

# Evaluate completed subset
results = [
    saved[str(r["sentence_id"])]
    for r in val_data[:N_TO_RUN]
]

metrics = evaluate_results(results)

print("\n=== Validation Baseline ===")

for key, value in metrics.items():
    print(f"{key}: {value:.4f}")

no_aspect = [r for r in results if len(r["gold"]) == 0]

no_aspect_acc = (
    sum(len(r["prediction"]) == 0 for r in no_aspect)
    / len(no_aspect)
    if no_aspect else 0
)

exact_match = sum(
    to_counter(r["gold"]) == to_counter(r["prediction"])
    for r in results
) / len(results)

print(f"No-Aspect Accuracy: {no_aspect_acc:.4f}")
print(f"Exact Match: {exact_match:.4f}")
print(f"\nResults saved to: {RESULT_FILE}")

# %%
# Check final baseline coverage

print("Evaluated samples:", len(results))
print("Unique sentence IDs:", len({
    r["sentence_id"] for r in results
}))

no_aspect = [
    r for r in results
    if len(r["gold"]) == 0
]

print("No-aspect samples:", len(no_aspect))

print(
    "Total gold pairs:",
    sum(len(r["gold"]) for r in results)
)

print(
    "Total predicted pairs:",
    sum(len(r["prediction"]) for r in results)
)

assert len(results) == len(val_data)
assert len({r["sentence_id"] for r in results}) == len(val_data)

print("Full validation coverage: PASSED")

# %%
# Cell: Save final baseline summary

import json

assert len(results) == 457

metrics = evaluate_results(results)

no_aspect_results = [
    r for r in results if len(r["gold"]) == 0
]

summary = {
    "model": "Qwen/Qwen3-1.7B-MLX-4bit",
    "dataset": "SemEval validation",
    "n_samples": len(results),
    **metrics,
    "no_aspect_accuracy": (
        sum(len(r["prediction"]) == 0 for r in no_aspect_results)
        / len(no_aspect_results)
    ),
    "exact_match_rate": (
        sum(
            to_counter(r["gold"]) == to_counter(r["prediction"])
            for r in results
        ) / len(results)
    )
}

summary_path = OUTPUT_DIR / "baseline_summary_v1.json"

with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print(json.dumps(summary, indent=2))
print("\nSaved:", summary_path)

# %%
