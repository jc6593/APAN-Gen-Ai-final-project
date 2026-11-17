
# %%
# ReviewLens - Final Held-out Test
# Frozen model: 16-layer QLoRA, Step 1200

import json
import re
import gc
from pathlib import Path
from collections import Counter

import mlx.core as mx
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "Qwen/Qwen3-1.7B-MLX-4bit"

DATA_DIR = ROOT / "data/processed/semeval"
ADAPTER_DIR = ROOT / "outputs/adapters/reviewlens_layers16"
CHECKPOINT = ADAPTER_DIR / "0001200_adapters.safetensors"

OUT_DIR = ROOT / "outputs/final_test"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


train = read_jsonl(DATA_DIR / "train.jsonl")
test_data = read_jsonl(DATA_DIR / "test.jsonl")

SYSTEM_PROMPT = train[0]["messages"][0]["content"]

assert len(test_data) == 800
assert CHECKPOINT.exists()

print("Test samples:", len(test_data))
print("Selected checkpoint:", CHECKPOINT)


# %%
# Prompt and parsing - same rules as validation

def build_prompt(tokenizer, text):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": text}
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )

    if not prompt.rstrip().endswith("</think>"):
        assert prompt.endswith("<|im_start|>assistant\n")
        prompt += "<think>\n\n</think>\n\n"

    return prompt


def parse_prediction(raw):
    cleaned = re.sub(
        r"^\s*<think>.*?</think>\s*",
        "", raw, flags=re.DOTALL
    ).strip()

    pred = json.loads(cleaned)
    assert isinstance(pred, list)

    for item in pred:
        assert isinstance(item, dict)
        assert set(item) == {"term", "sentiment"}
        assert isinstance(item["term"], str)
        assert item["sentiment"] in {
            "positive", "negative", "neutral", "conflict"
        }

    return pred


def pairs(items):
    return Counter(
        (" ".join(a["term"].lower().split()), a["sentiment"])
        for a in items
    )


# %%
# Evaluation metrics

def evaluate(records):
    tp = fp = fn = 0
    negative_gold = negative_tp = 0

    for row in records:
        gold = pairs(row["gold"])
        pred = pairs(row["prediction"])
        matched = gold & pred

        tp += sum(matched.values())
        fp += sum((pred - gold).values())
        fn += sum((gold - pred).values())

        negative_gold += sum(
            n for (_, s), n in gold.items() if s == "negative"
        )
        negative_tp += sum(
            n for (_, s), n in matched.items() if s == "negative"
        )

    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0
    )

    no_aspect = [r for r in records if not r["gold"]]
    has_aspect = [r for r in records if r["gold"]]

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "negative_recall": (
            negative_tp / negative_gold if negative_gold else 0
        ),
        "correct_empty": sum(
            r["parse_success"] and not r["prediction"]
            for r in no_aspect
        ),
        "false_empty": sum(
            r["parse_success"] and not r["prediction"]
            for r in has_aspect
        ),
        "no_aspect_accuracy": (
            sum(r["parse_success"] and not r["prediction"]
                for r in no_aspect) / len(no_aspect)
            if no_aspect else 0
        ),
        "parse_rate": (
            sum(r["parse_success"] for r in records) / len(records)
        ),
        "tp": tp,
        "fp": fp,
        "fn": fn
    }


# %%
# Run inference and save each result immediately

def run_test(name, adapter_path=None, checkpoint=None):
    path = OUT_DIR / f"{name}_test800.jsonl"

    saved = {}
    if path.exists():
        for row in read_jsonl(path):
            saved[str(row["sentence_id"])] = row

    if len(saved) == len(test_data):
        print(f"{name}: loaded completed predictions")
    else:
        if adapter_path:
            model, tokenizer = load(
                MODEL_ID, adapter_path=str(adapter_path)
            )
            model.load_weights(str(checkpoint), strict=False)
            mx.eval(model.parameters())
        else:
            model, tokenizer = load(MODEL_ID)

        model.eval()
        sampler = make_sampler(temp=0.0)

        for i, record in enumerate(test_data, 1):
            sid = str(record["sentence_id"])

            if sid in saved:
                continue

            text = record["messages"][1]["content"]
            gold = json.loads(record["messages"][2]["content"])

            raw = generate(
                model,
                tokenizer,
                prompt=build_prompt(tokenizer, text),
                max_tokens=256,
                sampler=sampler,
                verbose=False
            )

            try:
                pred = parse_prediction(raw)
                success = True
            except (ValueError, AssertionError, TypeError):
                pred = []
                success = False

            result = {
                "sentence_id": sid,
                "text": text,
                "gold": gold,
                "prediction": pred,
                "parse_success": success,
                "raw_response": raw
            }

            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")

            saved[sid] = result

            if i % 100 == 0 or i == len(test_data):
                print(f"{name}: {i}/{len(test_data)}")

        del model
        gc.collect()

    assert set(saved) == {
        str(r["sentence_id"]) for r in test_data
    }

    return [
        saved[str(r["sentence_id"])] for r in test_data
    ]


# %%
# Final comparison: do not change settings based on test scores

print("\n=== BASELINE TEST ===")
baseline = run_test("baseline")

print("\n=== FINETUNED TEST ===")
finetuned = run_test(
    "layers16_step1200",
    adapter_path=ADAPTER_DIR,
    checkpoint=CHECKPOINT
)

summary = {
    "Baseline": evaluate(baseline),
    "16L Step 1200": evaluate(finetuned)
}

print("\n=== FINAL HELD-OUT TEST ===")
for name, metrics in summary.items():
    print(f"\n{name}")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}" if isinstance(value, float)
              else f"  {key}: {value}")

with open(OUT_DIR / "final_test_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print("\nFinal test results saved successfully!")
