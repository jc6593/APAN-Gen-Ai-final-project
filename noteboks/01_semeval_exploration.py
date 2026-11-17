# %%
# Cell 0: Project configuration

from pathlib import Path

# Compatible with both VS Code Interactive and Terminal
current_dir = Path.cwd()

if current_dir.name in ("noteboks", "notebooks"):
    PROJECT_ROOT = current_dir.parent
else:
    PROJECT_ROOT = current_dir

DATA_DIR = PROJECT_ROOT / "data" / "raw" / "semeval"
DATA_DIR.mkdir(parents=True, exist_ok=True)

print("Project root:", PROJECT_ROOT)
print("Data directory:", DATA_DIR)

# %%
# Cell 1: Import libraries
from huggingface_hub import hf_hub_download
import xml.etree.ElementTree as ET
from collections import Counter

# %%
# Cell 2: Download datasets
repo_id = "alexcadillon/SemEval2014Task4"

train_path = hf_hub_download(
    repo_id=repo_id,
    repo_type="dataset",
    filename=(
        "SemEval'14-ABSA-TrainData_v2 & AnnotationGuidelines/"
        "Laptop_Train_v2.xml"
    ),
    local_dir=str(DATA_DIR)
)

test_path = hf_hub_download(
    repo_id=repo_id,
    repo_type="dataset",
    filename="ABSA_Gold_TestData/Laptops_Test_Gold.xml",
    local_dir=str(DATA_DIR)
)

print("Train file:", train_path)
print("Test file:", test_path)

# %%
# Cell 3: Check data paths

from pathlib import Path

print("Current directory:", Path.cwd())
print("Training file:", train_path)
print("Test file:", test_path)

print("Train exists:", Path(train_path).exists())
print("Test exists:", Path(test_path).exists())

# %%
# Cell 4: Parse SemEval XML

import xml.etree.ElementTree as ET

def parse_semeval(filepath):
    tree = ET.parse(filepath)
    root = tree.getroot()

    records = []

    for sentence in root.findall(".//sentence"):
        sentence_id = sentence.get("id")
        text = sentence.findtext("text", default="")
        aspects = []

        for aspect in sentence.findall("./aspectTerms/aspectTerm"):
            aspects.append({
                "term": aspect.get("term"),
                "sentiment": aspect.get("polarity"),
                "start": int(aspect.get("from")),
                "end": int(aspect.get("to"))
            })

        records.append({
            "sentence_id": sentence_id,
            "text": text,
            "aspects": aspects
        })

    return records

train_data = parse_semeval(train_path)
test_data = parse_semeval(test_path)

print("Train sentences:", len(train_data))
print("Test sentences:", len(test_data))
print("\nFirst example:")
print(train_data[0])

# %%
# Cell 5: Explore annotations

from collections import Counter

sentiment_counts = Counter(
    aspect["sentiment"]
    for record in train_data
    for aspect in record["aspects"]
)

aspect_counts = Counter(
    len(record["aspects"])
    for record in train_data
)

mixed_example = next(
    (
        record for record in train_data
        if len({
            aspect["sentiment"]
            for aspect in record["aspects"]
        }) >= 2
    ),
    None
)

print("=== Sentiment distribution ===")
print(sentiment_counts)

print("\n=== Aspects per sentence ===")
print(sorted(aspect_counts.items()))

print("\n=== Mixed sentiment example ===")
print(mixed_example)

# %%
# Cell 6: Validate aspect annotations

total_aspects = 0
invalid_spans = []
mismatched_terms = []

for record in train_data:
    text = record["text"]

    for aspect in record["aspects"]:
        total_aspects += 1

        start = aspect["start"]
        end = aspect["end"]

        # Check whether character positions are valid
        if start < 0 or end > len(text) or start >= end:
            invalid_spans.append({
                "sentence_id": record["sentence_id"],
                "aspect": aspect,
                "text": text
            })
            continue

        # Check whether extracted text matches the label
        extracted = text[start:end]

        if extracted != aspect["term"]:
            mismatched_terms.append({
                "sentence_id": record["sentence_id"],
                "expected": aspect["term"],
                "extracted": extracted
            })

print("Total aspect annotations:", total_aspects)
print("Invalid spans:", len(invalid_spans))
print("Mismatched aspect terms:", len(mismatched_terms))

print("\nFirst mismatched example:")
print(mismatched_terms[:3])

# %%
# Cell 7: Inspect special cases

# No aspect annotations
no_aspect = next(
    (r for r in train_data if len(r["aspects"]) == 0),
    None
)

# Positive and negative aspects in the same sentence
mixed_pos_neg = next(
    (
        r for r in train_data
        if {"positive", "negative"}.issubset(
            {a["sentiment"] for a in r["aspects"]}
        )
    ),
    None
)

# Conflict sentiment
conflict_example = next(
    (
        r for r in train_data
        if any(a["sentiment"] == "conflict"
               for a in r["aspects"])
    ),
    None
)

# Sentence with the most aspect annotations
max_aspects = max(
    train_data,
    key=lambda r: len(r["aspects"])
)

print("=== No aspect example ===")
print(no_aspect)

print("\n=== Positive + Negative example ===")
print(mixed_pos_neg)

print("\n=== Conflict example ===")
print(conflict_example)

print("\n=== Most annotated sentence ===")
print(max_aspects)

# %%
# Cell 8: Convert SemEval annotations to SFT format

import json

SYSTEM_PROMPT = """
You are an aspect-based sentiment extraction model.

Given a product review sentence:
1. Extract explicitly mentioned product aspect terms.
2. Assign sentiment to each aspect.
3. Use only: positive, negative, neutral, conflict.
4. Copy aspect terms exactly from the original sentence.
5. Return ONLY a JSON array.
6. If no explicit aspect terms are present, return [].

Each item must contain:
- term
- sentiment

Do not add explanations or extra text.
""".strip()


def convert_to_sft(record):

    # Sort aspects by their position in the original sentence
    sorted_aspects = sorted(
        record["aspects"],
        key=lambda a: (a["start"], a["end"])
    )

    target = [
        {
            "term": aspect["term"],
            "sentiment": aspect["sentiment"]
        }
        for aspect in sorted_aspects
    ]

    return {
        "sentence_id": record["sentence_id"],
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": record["text"]
            },
            {
                "role": "assistant",
                "content": json.dumps(
                    target,
                    ensure_ascii=False
                )
            }
        ]
    }


train_sft = [
    convert_to_sft(record)
    for record in train_data
]

print("Converted samples:", len(train_sft))

# %%
# Cell 9: Inspect converted SFT examples

sample_ids = ["812", "38", "1054"]

for sentence_id in sample_ids:

    example = next(
        record for record in train_sft
        if record["sentence_id"] == sentence_id
    )

    print("=" * 65)
    print("Sentence ID:", sentence_id)

    print("\nINPUT:")
    print(example["messages"][1]["content"])

    print("\nTARGET OUTPUT:")
    print(
        json.dumps(
            json.loads(example["messages"][2]["content"]),
            indent=2,
            ensure_ascii=False
        )
    )

    print()

# %%
# Cell 10: Validate SFT data

test_sft = [
    convert_to_sft(record)
    for record in test_data
]

VALID_SENTIMENTS = {
    "positive", "negative", "neutral", "conflict"
}

def validate_sft(records):
    for record in records:
        text = record["messages"][1]["content"]
        target = json.loads(
            record["messages"][2]["content"]
        )

        assert isinstance(target, list)

        for aspect in target:
            assert set(aspect.keys()) == {
                "term", "sentiment"
            }
            assert aspect["sentiment"] in VALID_SENTIMENTS
            assert aspect["term"] in text

    print(f"Validated {len(records)} samples")


validate_sft(train_sft)
validate_sft(test_sft)

# %%
# Cell 11: Train-validation split

import random

SEED = 42
VAL_RATIO = 0.15
rng = random.Random(SEED)

# Stratify by presence of annotated aspects
groups = {
    "no_aspect": [],
    "has_aspect": []
}

for record in train_sft:
    target = json.loads(
        record["messages"][2]["content"]
    )

    if len(target) == 0:
        groups["no_aspect"].append(record)
    else:
        groups["has_aspect"].append(record)

train_split = []
val_split = []

for group in groups.values():
    rng.shuffle(group)

    n_val = round(len(group) * VAL_RATIO)

    val_split.extend(group[:n_val])
    train_split.extend(group[n_val:])

rng.shuffle(train_split)
rng.shuffle(val_split)

print("Train:", len(train_split))
print("Validation:", len(val_split))
print("Test:", len(test_sft))

# Check sentence ID overlap
train_ids = {r["sentence_id"] for r in train_split}
val_ids = {r["sentence_id"] for r in val_split}

print(
    "Overlapping sentence IDs:",
    len(train_ids & val_ids)
)

# Check duplicate text across splits
def text_key(record):
    text = record["messages"][1]["content"]
    return " ".join(text.lower().split())

train_texts = {text_key(r) for r in train_split}
val_texts = {text_key(r) for r in val_split}
test_texts = {text_key(r) for r in test_sft}

print(
    "Train/Val duplicate texts:",
    len(train_texts & val_texts)
)

print(
    "Train/Test duplicate texts:",
    len(train_texts & test_texts)
)

# %%
# Cell 12: Prevent duplicate-text leakage

import random
import json
from collections import defaultdict

SEED = 42
VAL_RATIO = 0.15

def text_key(record):
    text = record["messages"][1]["content"]
    return " ".join(text.lower().split())

# 1. Keep the official test set unchanged
test_texts = {text_key(r) for r in test_sft}

# Remove training samples whose text appears in the test set
train_pool = [
    r for r in train_sft
    if text_key(r) not in test_texts
]

print("Original training pool:", len(train_sft))
print("Removed due to test overlap:",
      len(train_sft) - len(train_pool))


# 2. Group identical texts together
text_groups = defaultdict(list)

for record in train_pool:
    text_groups[text_key(record)].append(record)


# 3. Stratify groups by whether they contain aspects
groups = {
    "no_aspect": [],
    "has_aspect": []
}

for records in text_groups.values():

    has_aspect = any(
        len(json.loads(r["messages"][2]["content"])) > 0
        for r in records
    )

    category = "has_aspect" if has_aspect else "no_aspect"
    groups[category].append(records)


# 4. Split whole groups, never individual duplicate rows
rng = random.Random(SEED)

train_split = []
val_split = []

for category, grouped_records in groups.items():

    rng.shuffle(grouped_records)

    total = sum(len(g) for g in grouped_records)
    target_val = round(total * VAL_RATIO)

    val_count = 0

    for group in grouped_records:

        if val_count < target_val:
            val_split.extend(group)
            val_count += len(group)
        else:
            train_split.extend(group)

rng.shuffle(train_split)
rng.shuffle(val_split)


# 5. Verify that all splits are disjoint by text
train_texts = {text_key(r) for r in train_split}
val_texts = {text_key(r) for r in val_split}

print("\n=== Final dataset sizes ===")
print("Train:", len(train_split))
print("Validation:", len(val_split))
print("Test:", len(test_sft))

print("\n=== Leakage checks ===")
print("Train / Val overlap:",
      len(train_texts & val_texts))

print("Train / Test overlap:",
      len(train_texts & test_texts))

print("Val / Test overlap:",
      len(val_texts & test_texts))

assert not (train_texts & val_texts)
assert not (train_texts & test_texts)
assert not (val_texts & test_texts)

print("\nAll leakage checks passed!")


# %%
# Cell 13: Check conflicting annotations for duplicate texts

def target_signature(record):
    target = json.loads(record["messages"][2]["content"])
    return tuple(sorted(
        (a["term"], a["sentiment"])
        for a in target
    ))

conflicting_groups = []

for text, records in text_groups.items():

    if len(records) <= 1:
        continue

    targets = {target_signature(r) for r in records}

    if len(targets) > 1:
        conflicting_groups.append((text, records))

print("Duplicate text groups with conflicting labels:",
      len(conflicting_groups))

for text, records in conflicting_groups[:3]:
    print("\nTEXT:", text)

    for r in records:
        print(
            r["sentence_id"],
            target_signature(r)
        )

# %%
# Cell 14: Save final SFT datasets

import json
from pathlib import Path

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "semeval"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

datasets = {
    "train": train_split,
    "val": val_split,
    "test": test_sft
}

def save_jsonl(records, filepath):
    with open(filepath, "w", encoding="utf-8") as f:
        for record in records:
            f.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )

for name, records in datasets.items():
    filepath = OUTPUT_DIR / f"{name}.jsonl"
    save_jsonl(records, filepath)
    print(f"Saved {name}: {len(records)} samples")


# Verify files by reading them back
print("\n=== File verification ===")

for name, records in datasets.items():
    filepath = OUTPUT_DIR / f"{name}.jsonl"

    with open(filepath, "r", encoding="utf-8") as f:
        loaded = [json.loads(line) for line in f]

    assert len(loaded) == len(records)
    assert loaded == records

    print(f"{name}.jsonl: OK")

print("\nAll datasets saved successfully!")
# %%
