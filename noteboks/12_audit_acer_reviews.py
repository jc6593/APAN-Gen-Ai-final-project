
# %%
# ReviewLens M3 - Acer Aspire 3 Review Audit

import json
import re
from pathlib import Path
from collections import Counter
from statistics import median

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data/processed/amazon"

SOURCE_FILE = DATA_DIR / "target_reviews_probe_2m.jsonl"

TARGET_ASIN = "B0C2ZMJW53"

def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [
            json.loads(line)
            for line in f
            if line.strip()
        ]

all_reviews = read_jsonl(SOURCE_FILE)

acer_reviews = [
    r for r in all_reviews
    if r.get("parent_asin") == TARGET_ASIN
]

print("=== ACER DATA OVERVIEW ===")
print("Usable Acer reviews:", len(acer_reviews))

assert len(acer_reviews) > 0


# %%
# Cell 2: Check product variants

asin_counts = Counter(
    r.get("asin") or "MISSING"
    for r in acer_reviews
)

print("\n=== PRODUCT VARIANTS ===")
print("Distinct ASINs:", len(asin_counts))

for asin, count in asin_counts.most_common():
    print(f"{asin}: {count} reviews")


# %%
# Cell 3: Rating distribution

rating_counts = Counter(
    r.get("rating")
    for r in acer_reviews
)

print("\n=== RATING DISTRIBUTION ===")

for rating in sorted(
    rating_counts,
    key=lambda x: float(x or 0)
):
    print(f"{rating} stars: {rating_counts[rating]}")

print(
    "Verified purchases:",
    sum(r.get("verified_purchase") is True
        for r in acer_reviews)
)


# %%
# Cell 4: Duplicate review detection

def normalize_text(text):
    return re.sub(
        r"\s+", " ", str(text).strip().lower()
    )

text_counts = Counter(
    normalize_text(r["text"])
    for r in acer_reviews
)

duplicate_groups = {
    text: count
    for text, count in text_counts.items()
    if count > 1
}

print("\n=== DUPLICATE CHECK ===")
print("Duplicate text groups:", len(duplicate_groups))
print(
    "Extra duplicate records:",
    sum(n - 1 for n in duplicate_groups.values())
)


# %%
# Cell 5: Review length analysis

lengths = [len(r["text"]) for r in acer_reviews]

print("\n=== TEXT LENGTH ===")
print("Minimum characters:", min(lengths))
print("Median characters:", median(lengths))
print("Maximum characters:", max(lengths))


# %%
# Cell 6: Preview reviews at different ratings

print("\n=== REVIEW EXAMPLES BY RATING ===")

for rating in [1, 2, 3, 4, 5]:

    examples = [
        r for r in acer_reviews
        if r.get("rating") == float(rating)
    ]

    print(f"\n--- {rating} STAR ---")
    print("Available:", len(examples))

    for record in examples[:2]:
        print("Review ID:", record["review_id"])
        print("ASIN:", record.get("asin"))
        print("Text:", record["text"][:400])
        print()


# %%
# Cell 7: Save product-specific candidate data

OUTPUT_FILE = (
    DATA_DIR / "acer_aspire3_candidate_reviews.jsonl"
)

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for record in acer_reviews:
        f.write(
            json.dumps(record, ensure_ascii=False) + "\n"
        )

print("\nSaved Acer candidate reviews:", OUTPUT_FILE)
print("Audit completed!")
