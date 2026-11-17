
# %%
# ReviewLens - Checkpoint error analysis
# No additional inference or model loading required

import json
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]

def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

validation = read_jsonl(
    ROOT / "data/processed/semeval/val.jsonl"
)

gold = {
    str(r["sentence_id"]): json.loads(
        r["messages"][2]["content"]
    )
    for r in validation
}

texts = {
    str(r["sentence_id"]): r["messages"][1]["content"]
    for r in validation
}

files = {
    "Baseline": ROOT / "outputs/baseline/qwen3_baseline_v1.jsonl",
    "Step 50": ROOT / "outputs/checkpoint_eval/step_50_val457.jsonl",
    "Step 100": ROOT / "outputs/checkpoint_eval/step_100_val457.jsonl",
    "Step 150": ROOT / "outputs/checkpoint_eval/step_150_val457.jsonl",
    "Step 200": ROOT / "outputs/checkpoint_eval/step_200_val457.jsonl",
}

results = {}

for name, path in files.items():
    rows = read_jsonl(path)
    results[name] = {
        str(r["sentence_id"]): r for r in rows
    }
    assert len(rows) == len(gold)
    assert set(results[name]) == set(gold)

def pairs(items):
    return Counter(
        (" ".join(a["term"].lower().split()), a["sentiment"])
        for a in items
    )

def analyze(name):
    rows = results[name]
    no_aspect_ids = [
        sid for sid in gold if len(gold[sid]) == 0
    ]
    nonempty_ids = [
        sid for sid in gold if len(gold[sid]) > 0
    ]

    parse_failures = sum(
        not r["parse_success"] for r in rows.values()
    )

    correct_empty = sum(
        rows[sid]["parse_success"]
        and not rows[sid]["prediction"]
        for sid in no_aspect_ids
    )

    false_empty = sum(
        rows[sid]["parse_success"]
        and not rows[sid]["prediction"]
        for sid in nonempty_ids
    )

    negative_gold = 0
    negative_tp = 0

    for sid in gold:
        g = pairs(gold[sid])
        p = pairs(rows[sid]["prediction"])

        negative_gold += sum(
            n for (term, sentiment), n in g.items()
            if sentiment == "negative"
        )

        negative_tp += sum(
            n for (term, sentiment), n in (g & p).items()
            if sentiment == "negative"
        )

    return {
        "negative_recall": negative_tp / negative_gold,
        "correct_empty": correct_empty,
        "false_empty": false_empty,
        "parse_failures": parse_failures
    }

print("=== PRODUCT-RELEVANT METRICS ===")

for name in files:
    m = analyze(name)
    print(
        f"{name:10} | "
        f"Negative Recall: {m['negative_recall']:.4f} | "
        f"Correct Empty: {m['correct_empty']} | "
        f"False Empty: {m['false_empty']} | "
        f"Parse Failures: {m['parse_failures']}"
    )

# %%
# Inspect examples where Step 50 finds negative feedback
# that Step 200 misses

def matched(sid, name):
    return (
        pairs(gold[sid])
        & pairs(results[name][sid]["prediction"])
    )

print("\n=== NEGATIVE ASPECTS FOUND BY STEP 50 BUT MISSED BY STEP 200 ===")

shown = 0

for sid in gold:
    extra_correct = matched(sid, "Step 50") - matched(sid, "Step 200")

    if not any(
        sentiment == "negative"
        for term, sentiment in extra_correct
    ):
        continue

    print("\nSentence:", sid)
    print("Text:", texts[sid][:500])
    print("Gold:", gold[sid])
    print("Step 50:", results["Step 50"][sid]["prediction"])
    print("Step 200:", results["Step 200"][sid]["prediction"])

    shown += 1
    if shown == 3:
        break
