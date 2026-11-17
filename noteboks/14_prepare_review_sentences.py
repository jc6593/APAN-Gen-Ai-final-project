
# %%
# ============================================================
# ReviewLens M3 - Sentence Segmentation
#
# Input: 189 cleaned Amazon reviews
# Output: Sentence-level records with source traceability
# ============================================================

import json
from pathlib import Path
from collections import Counter
from statistics import median

import pysbd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data/processed/amazon"

SOURCE = DATA_DIR / "acer_aspire5_reviews_clean.jsonl"
OUTPUT = DATA_DIR / "acer_aspire5_sentences.jsonl"


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [
            json.loads(line)
            for line in f
            if line.strip()
        ]


reviews = read_jsonl(SOURCE)

print("Reviews loaded:", len(reviews))
assert len(reviews) == 189


# %%
# Cell 2: Segment sentences and preserve offsets

segmenter = pysbd.Segmenter(
    language="en",
    clean=False
)


def segment_with_offsets(text):
    """
    Split text and locate each sentence in the exact
    original clean_text.

    start_char is inclusive; end_char is exclusive.
    """

    raw_segments = segmenter.segment(text)

    spans = []
    cursor = 0

    for segment in raw_segments:

        sentence = segment.strip()

        if not sentence:
            continue

        start = text.find(sentence, cursor)

        if start < 0:
            raise ValueError(
                f"Cannot align sentence: {sentence[:100]!r}"
            )

        # Do not silently lose meaningful text
        if text[cursor:start].strip():
            raise ValueError(
                f"Uncovered text: {text[cursor:start]!r}"
            )

        end = start + len(sentence)

        assert text[start:end] == sentence

        spans.append({
            "sentence_text": sentence,
            "start_char": start,
            "end_char": end
        })

        cursor = end

    if text[cursor:].strip():
        raise ValueError(
            f"Uncovered trailing text: {text[cursor:]!r}"
        )

    if not spans:
        raise ValueError("No sentences produced")

    return spans


# %%
# Cell 3: Build sentence-level dataset

sentence_records = []
sentences_per_review = []

for review in reviews:

    text = review["clean_text"]

    try:
        spans = segment_with_offsets(text)
    except ValueError as exc:
        raise ValueError(
            f"Segmentation failed for "
            f"{review['review_id']}: {exc}"
        ) from exc

    sentences_per_review.append(len(spans))

    for idx, span in enumerate(spans, start=1):

        sentence_records.append({
            "sentence_id": (
                f"{review['review_id']}_S{idx:03d}"
            ),
            "review_id": review["review_id"],
            "sentence_index": idx,

            "asin": review["asin"],
            "parent_asin": review["parent_asin"],
            "product_model": review["product_model"],
            "rating": review["rating"],

            "sentence_text": span["sentence_text"],
            "start_char": span["start_char"],
            "end_char": span["end_char"]
        })


# %%
# Cell 4: Validate segmentation

assert len(sentence_records) >= len(reviews)

sentence_ids = [
    s["sentence_id"] for s in sentence_records
]

assert len(sentence_ids) == len(set(sentence_ids))

reviews_by_id = {
    r["review_id"]: r for r in reviews
}

for s in sentence_records:

    original = reviews_by_id[s["review_id"]]["clean_text"]

    assert (
        original[s["start_char"]:s["end_char"]]
        == s["sentence_text"]
    )

assert {
    s["review_id"] for s in sentence_records
} == set(reviews_by_id)


print("\n=== SENTENCE DATASET ===")
print("Original reviews:", len(reviews))
print("Total sentences:", len(sentence_records))
print("Median sentences per review:",
      median(sentences_per_review))
print("Maximum sentences per review:",
      max(sentences_per_review))

lengths = [len(s["sentence_text"]) for s in sentence_records]

print("Median sentence characters:", median(lengths))
print("Maximum sentence characters:", max(lengths))
print("Sentences over 900 characters:",
      sum(n > 900 for n in lengths))

print("\nAll source-offset checks PASSED!")


# %%
# Cell 5: Inspect a negative review

negative_review = next(
    r for r in reviews if r["rating"] == 1.0
)

print("\n=== EXAMPLE: ONE-STAR REVIEW ===")
print("Review ID:", negative_review["review_id"])

examples = [
    s for s in sentence_records
    if s["review_id"] == negative_review["review_id"]
]

for s in examples[:12]:
    print(
        f"\n{s['sentence_id']} "
        f"[{s['start_char']}:{s['end_char']}]"
    )
    print(s["sentence_text"])


# %%
# Cell 6: Save sentence records

with open(OUTPUT, "w", encoding="utf-8") as f:
    for record in sentence_records:
        f.write(
            json.dumps(record, ensure_ascii=False) + "\n"
        )

print("\nSaved:", OUTPUT)
print("Sentence preparation completed!")
