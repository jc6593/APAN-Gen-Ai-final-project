
# %%
# ============================================================
# ReviewLens - M2: QLoRA Checkpoint Evaluation
#
# Experiment: reviewlens_epoch1
# Compare 7 checkpoints on the same validation samples
#
# Metrics:
#   - Precision / Recall / Micro-F1
#   - Negative Aspect Recall
#   - Empty Predictions
#   - False Empty Predictions
#   - No-Aspect Accuracy
#   - JSON Parse Rate
# ============================================================

import json
import re
from pathlib import Path
from collections import Counter

import mlx.core as mx
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler


# %%
# ============================================================
# Cell 1: Project configuration
# ============================================================

# This script is designed to run from the Terminal
ROOT = Path(__file__).resolve().parents[1]

MODEL_ID = "Qwen/Qwen3-1.7B-MLX-4bit"

DATA_DIR = ROOT / "data" / "processed" / "semeval"

BASELINE_FILE = (
    ROOT / "outputs" / "baseline"
    / "qwen3_baseline_v1.jsonl"
)
# Experiment 2: Full validation evaluation

ADAPTER_DIR = (
    ROOT / "outputs" / "adapters"
    / "reviewlens_layers16"
)

OUT_DIR = (
    ROOT / "outputs" / "checkpoint_eval"
    / "layers16"
)

OUT_DIR.mkdir(parents=True, exist_ok=True)

STEPS = [800, 1200]

N = 457

print("=== ReviewLens Checkpoint Evaluation ===")
print("Project root:", ROOT)
print("Model:", MODEL_ID)
print("Adapter directory:", ADAPTER_DIR)
print("Checkpoints:", STEPS)


# %%
# ============================================================
# Cell 2: Load datasets
# ============================================================

def read_jsonl(path):
    path = Path(path)

    assert path.exists(), f"File not found: {path}"

    with open(path, "r", encoding="utf-8") as f:
        return [
            json.loads(line)
            for line in f
            if line.strip()
        ]


train = read_jsonl(DATA_DIR / "train.jsonl")
validation = read_jsonl(DATA_DIR / "val.jsonl")

assert 0 < N <= len(validation)

# Use exactly the same samples for every checkpoint
samples = validation[:N]

# Reuse the original SFT system prompt
SYSTEM_PROMPT = train[0]["messages"][0]["content"]

print("\n=== Dataset Information ===")
print("Training samples:", len(train))
print("Validation samples:", len(validation))
print("Evaluation samples:", len(samples))


# %%
# ============================================================
# Cell 3: Load Qwen with LoRA adapter
# ============================================================

assert ADAPTER_DIR.exists(), ADAPTER_DIR

model, tokenizer = load(
    MODEL_ID,
    adapter_path=str(ADAPTER_DIR)
)

model.eval()

# Deterministic decoding
sampler = make_sampler(temp=0.0)

print("\nModel loaded successfully!")


# %%
# ============================================================
# Cell 4: Prompt construction
# ============================================================

def build_prompt(text):

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": text
        }
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )

    # Explicitly disable thinking for this MLX template
    if not prompt.rstrip().endswith("</think>"):

        assert prompt.endswith(
            "<|im_start|>assistant\n"
        )

        prompt += "<think>\n\n</think>\n\n"

    return prompt


# %%
# ============================================================
# Cell 5: Parse model predictions
# ============================================================

VALID_SENTIMENTS = {
    "positive",
    "negative",
    "neutral",
    "conflict"
}


def parse_prediction(raw):

    # Remove complete thinking block if present
    cleaned = re.sub(
        r"^\s*<think>.*?</think>\s*",
        "",
        raw,
        flags=re.DOTALL
    ).strip()

    prediction = json.loads(cleaned)

    assert isinstance(prediction, list)

    for item in prediction:

        assert isinstance(item, dict)

        assert set(item.keys()) == {
            "term", "sentiment"
        }

        assert isinstance(item["term"], str)

        assert item["sentiment"] in VALID_SENTIMENTS

    return prediction


# %%
# ============================================================
# Cell 6: Evaluation functions
# ============================================================

def pairs(items):

    return Counter(
        (
            " ".join(a["term"].lower().split()),
            a["sentiment"]
        )
        for a in items
    )


def evaluate(records):

    assert len(records) > 0

    tp = 0
    fp = 0
    fn = 0

    negative_gold = 0
    negative_tp = 0

    for r in records:

        gold = pairs(r["gold"])
        pred = pairs(r["prediction"])

        matched = gold & pred

        tp += sum(matched.values())
        fp += sum((pred - gold).values())
        fn += sum((gold - pred).values())

        # Recall specifically for negative aspects
        negative_gold += sum(
            count
            for (term, sentiment), count in gold.items()
            if sentiment == "negative"
        )

        negative_tp += sum(
            count
            for (term, sentiment), count in matched.items()
            if sentiment == "negative"
        )

    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0
    )

    negative_recall = (
        negative_tp / negative_gold
        if negative_gold else 0
    )

    # Samples without annotated aspects
    no_aspect = [
        r for r in records
        if len(r["gold"]) == 0
    ]

    # Samples containing annotated aspects
    has_aspect = [
        r for r in records
        if len(r["gold"]) > 0
    ]

    # Correct empty prediction requires successful parsing
    correct_empty = sum(
        r["parse_success"] and not r["prediction"]
        for r in no_aspect
    )

    # Incorrect empty prediction on an annotated sentence
    false_empty = sum(
        r["parse_success"] and not r["prediction"]
        for r in has_aspect
    )

    empty_predictions = sum(
        not r["prediction"]
        for r in records
    )

    parse_successes = sum(
        r["parse_success"]
        for r in records
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "negative_recall": negative_recall,
        "empty_predictions": empty_predictions,
        "false_empty": false_empty,
        "correct_empty": correct_empty,
        "no_aspect_accuracy": (
            correct_empty / len(no_aspect)
            if no_aspect else 0
        ),
        "parse_rate": parse_successes / len(records),
        "parse_failures": len(records) - parse_successes,
        "tp": tp,
        "fp": fp,
        "fn": fn
    }


# %%
# ============================================================
# Cell 7: Load original Qwen baseline
# ============================================================

baseline_data = read_jsonl(BASELINE_FILE)

baseline_by_id = {
    str(r["sentence_id"]): r
    for r in baseline_data
}

baseline_records = [
    baseline_by_id[str(r["sentence_id"])]
    for r in samples
]

assert len(baseline_records) == N

summary = {
    "Baseline": evaluate(baseline_records)
}

print("\nBaseline loaded successfully!")


# %%
# ============================================================
# Cell 8: Evaluate all QLoRA checkpoints
# ============================================================

for step in STEPS:

    checkpoint = (
        ADAPTER_DIR
        / f"{step:07d}_adapters.safetensors"
    )

    assert checkpoint.exists(), (
        f"Missing checkpoint: {checkpoint}"
    )

    print("\n" + "=" * 60)
    print(f"Evaluating checkpoint {step}")
    print("=" * 60)

    # Replace LoRA weights with selected checkpoint
    model.load_weights(
        str(checkpoint),
        strict=False
    )

    mx.eval(model.parameters())
    model.eval()

    results = []

    for i, record in enumerate(samples, start=1):

        review = record["messages"][1]["content"]

        gold = json.loads(
            record["messages"][2]["content"]
        )

        prompt = build_prompt(review)

        raw = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=256,
            sampler=sampler,
            verbose=False
        )

        try:
            prediction = parse_prediction(raw)
            parse_success = True

        except (ValueError, TypeError, AssertionError):
            prediction = []
            parse_success = False

        results.append({
            "sentence_id": record["sentence_id"],
            "text": review,
            "gold": gold,
            "prediction": prediction,
            "parse_success": parse_success,
            "raw_response": raw
        })

        if i % 20 == 0 or i == N:
            print(f"  {i}/{N} completed")

    # --------------------------------------------------------
    # Save predictions for this checkpoint
    # --------------------------------------------------------

    output_file = (
        OUT_DIR / f"step_{step}_val{N}.jsonl"
    )

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as f:

        for row in results:
            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False
                ) + "\n"
            )

    # Evaluate and update summary
    summary[f"Step {step}"] = evaluate(results)

    print(f"Checkpoint {step} completed.")
    print("Results saved:", output_file)


# %%
# ============================================================
# Cell 9: Display comparison
# ============================================================

print("\n=== CHECKPOINT COMPARISON ===")

header = (
    f"{'Model':<12} "
    f"{'Precision':>9} "
    f"{'Recall':>8} "
    f"{'F1':>8} "
    f"{'NegRecall':>10} "
    f"{'Empty':>7} "
    f"{'FalseEmpty':>10} "
    f"{'NoAspect':>9} "
    f"{'JSON':>7}"
)

print(header)
print("-" * len(header))

for name, metrics in summary.items():

    print(
        f"{name:<12} "
        f"{metrics['precision']:>9.4f} "
        f"{metrics['recall']:>8.4f} "
        f"{metrics['f1']:>8.4f} "
        f"{metrics['negative_recall']:>10.4f} "
        f"{metrics['empty_predictions']:>7} "
        f"{metrics['false_empty']:>10} "
        f"{metrics['no_aspect_accuracy']:>9.4f} "
        f"{metrics['parse_rate']:>7.4f}"
    )


# %%
# ============================================================
# Cell 10: Save comparison summary
# ============================================================

summary_file = (
    OUT_DIR / f"comparison_val{N}.json"
)

with open(
    summary_file,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        summary,
        f,
        indent=2,
        ensure_ascii=False
    )

print("\nComparison saved successfully!")
print("Summary file:", summary_file)
print("All checkpoints evaluated:", len(STEPS))
